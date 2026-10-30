"""Time-indexed reference: action rows stamped with the instant they are for.

Row i of a chunk inferred from an observation at `t_obs` means "be here at
`t_obs + (i + 1) / fps`". The controller samples the buffer at the instant a
target is due at the arm instead of popping one row per tick, so skipping stale
rows, blending a seam, and reporting staleness in seconds all fall out of the
time axis instead of being reconstructed from tick counts.

Two landing paths:

- Crossfade: the old plan still has rows after the last sample. Everything
  sampled so far stays; after it the new chunk wins, ramped in from the old
  plan over `blend_s` with a smoothstep. An old plan that ends inside the
  window is held at its last row.
- Rebase: the plan has run out (or this is the first chunk), so the arm is
  holding the last value it was commanded, sampled at `last_sampled`. The new
  chunk resumes from there: rows before `max(last_sampled, t_obs + 1/fps)` are
  motion the arm already made, and the rest are shifted so that point lands on
  the next sample. The shift is lazy, since the next sample time is unknown
  when the chunk lands.

Single-task asyncio access only: no lock.
"""

from __future__ import annotations

import math

import numpy as np

from vla.config_util import VLAError


class ReferenceBufferError(VLAError, ValueError):
    """Raised when `ReferenceBuffer.land()` receives a malformed chunk."""


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class ReferenceBuffer:
    """Rows on a time axis.

    `interp_cols` leading columns are linear in time and crossfaded; the rest
    (the gripper) are zero-order held and switch outright at the seam. None
    means every column interpolates. `consume=True` is delta-EE: rows are
    per-tick deltas, so `take()` runs each at most once and nothing is
    interpolated or blended.
    """

    def __init__(
        self,
        fps: float,
        *,
        blend_s: float = 0.0,
        interp_cols: int | None = None,
        consume: bool = False,
    ) -> None:
        self._period = 1.0 / fps
        # Row times and tick times are the same instants computed two ways;
        # compare with slack so rounding noise never decides a row.
        self._tol = self._period * 1e-6
        self._blend_s = 0.0 if consume else blend_s
        self._interp_cols = 0 if consume else interp_cols
        self._consume = consume
        self.times = np.empty(0)
        self.rows: np.ndarray | None = None
        self.obs_t = np.empty(0)
        self.last_sampled = -math.inf
        # Set by a rebase: the time on the new chunk's axis that the next
        # sample pins to. None once pinned.
        self._pin: float | None = None
        self.rebases = 0
        self.overlap_s = 0.0
        self.skipped_rows = 0
        self.sampled_obs_t: float | None = None
        # Set when exhausted() reports the plan ran out, so the next landing
        # rebases even if the last sample fell a hair before the last row:
        # comparing times[-1] with last_sampled alone would send that landing
        # down the crossfade path and stamp it in the past.
        self._ran_out = False

    # -- queries ---------------------------------------------------------------

    def exhausted(self, t: float) -> bool:
        """Nothing left to command at `t`: the caller waits for a landing.

        Records that the plan ran out, which is what decides the next landing's
        path, so call it with the tick's sample time, never wall-clock now.
        The controller calls this every tick before sampling.
        """
        if not len(self.times):
            return True
        if self._pin is not None or self._consume:
            # Pending rows always start at the next sample; consumed rows are
            # removed, so any row left is still to run.
            return False
        if t > self.times[-1]:
            self._ran_out = True
            return True
        return False

    def horizon(self) -> float:
        """Time of the last row; +inf while a rebase is unpinned."""
        if not len(self.times):
            return -math.inf
        if self._pin is not None:
            return math.inf
        return float(self.times[-1])

    # -- landing ---------------------------------------------------------------

    def land(self, chunk: np.ndarray, t_obs: float) -> None:
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 2 or chunk.shape[0] == 0:
            raise ReferenceBufferError(
                "a chunk must be a non-empty 2D numpy array shaped (rows, action_dim), got "
                f"{type(chunk).__name__}"
                + (f" with shape {chunk.shape}" if isinstance(chunk, np.ndarray) else "")
            )
        if self.rows is not None and self.rows.shape[1] != chunk.shape[1]:
            raise ReferenceBufferError(
                f"chunk has {chunk.shape[1]} columns, the buffer holds {self.rows.shape[1]}"
            )
        times = t_obs + (np.arange(chunk.shape[0]) + 1) * self._period
        rows = chunk.astype(np.float64)  # a copy: the caller's array is never aliased
        obs = np.full(chunk.shape[0], float(t_obs))
        if (
            self._ran_out
            or self._pin is not None
            or not len(self.times)
            or self.times[-1] <= self.last_sampled
        ):
            self._rebase(times, rows, obs, t_obs)
        else:
            self._crossfade(times, rows, obs)

    def _rebase(self, times, rows, obs, t_obs: float) -> None:
        was_pinned = self._pin is not None
        self._ran_out = False
        freeze = max(self.last_sampled, t_obs + self._period)
        if freeze > times[-1]:
            # Every row is motion the arm already made along the old plan.
            self._set(np.empty(0), None, np.empty(0))
            self._pin = None
            return
        if self._consume:
            keep = times >= freeze
            times, rows, obs = times[keep], rows[keep], obs[keep]
        else:
            # Start exactly at the freeze point, so the arm resumes from the
            # chunk's value there rather than a fraction of a row later.
            keep = times > freeze
            first = self._at(times, rows, freeze)
            times = np.concatenate([[freeze], times[keep]])
            rows = np.vstack([first, rows[keep]])
            obs = np.concatenate([[obs[0]], obs[keep]])
        self._set(times, rows, obs)
        self._pin = freeze
        if not was_pinned:
            self.rebases += 1
        self.overlap_s = 0.0

    def _crossfade(self, times, rows, obs) -> None:
        seam = self.last_sampled
        new = times > seam
        if not new.any():
            # Entirely behind what is already commanded, while the old plan
            # still has rows ahead: nothing in it is usable.
            return
        times, rows, obs = times[new], rows[new], obs[new]
        n = self._n_interp(rows)
        if self._blend_s > 0 and n:
            w = _smoothstep((times - seam) / self._blend_s)[:, None]
            # np.interp holds the old plan's last row past its end.
            old = np.column_stack(
                [np.interp(times, self.times, self.rows[:, c]) for c in range(n)]
            )
            rows[:, :n] = w * rows[:, :n] + (1.0 - w) * old
        self.overlap_s = min(self._blend_s, max(0.0, float(self.times[-1]) - seam))
        keep = self.times < times[0]
        old_rows = self.rows[keep]  # fancy index: a copy
        # Held columns (the gripper) take the new value from the seam, even on
        # old rows kept between the seam and the first new row.
        old_rows[self.times[keep] > seam, n:] = rows[0, n:]
        self._set(
            np.concatenate([self.times[keep], times]),
            np.vstack([old_rows, rows]),
            np.concatenate([self.obs_t[keep], obs]),
        )

    # -- sampling --------------------------------------------------------------

    def sample(self, t: float) -> np.ndarray | None:
        """The reference at `t`, or None when exhausted. Times must not decrease."""
        if self.exhausted(t):
            return None
        self._apply_pin(t)
        i = self._index(t)
        out = self._at(self.times, self.rows, t)
        self.sampled_obs_t = float(self.obs_t[i])
        # No later sample can need anything before row i.
        self._set(self.times[i:], self.rows[i:], self.obs_t[i:])
        self.last_sampled = t
        return out

    def take(self, t: float) -> np.ndarray | None:
        """`consume` mode: the latest unconsumed row due by `t`, at most once.

        None means hold (the next row is not due yet) or exhausted, so callers
        check `exhausted()` first. Older due rows are
        skipped (counted), never run late in a burst.
        """
        if self.exhausted(t):
            return None
        self._apply_pin(t)
        due = int(np.searchsorted(self.times, t + self._tol, side="right"))
        self.last_sampled = t
        if due == 0:
            return None
        self.skipped_rows += due - 1
        row = self.rows[due - 1].copy()
        self.sampled_obs_t = float(self.obs_t[due - 1])
        self._set(self.times[due:], self.rows[due:], self.obs_t[due:])
        return row

    # -- helpers ---------------------------------------------------------------

    def _apply_pin(self, t: float) -> None:
        if self._pin is not None:
            self.times = self.times + (t - self._pin)
            self._pin = None

    def _index(self, t: float) -> int:
        return max(0, int(np.searchsorted(self.times, t, side="right")) - 1)

    def _n_interp(self, rows: np.ndarray) -> int:
        return rows.shape[1] if self._interp_cols is None else self._interp_cols

    def _at(self, times: np.ndarray, rows: np.ndarray, t: float) -> np.ndarray:
        """Linear in time for interpolated columns, held for the rest."""
        i = max(0, int(np.searchsorted(times, t, side="right")) - 1)
        out = rows[i].copy()
        for c in range(self._n_interp(rows)):
            out[c] = np.interp(t, times, rows[:, c])
        return out

    def _set(self, times, rows, obs) -> None:
        self.times, self.obs_t = times, obs
        if rows is not None:
            self.rows = rows
        elif self.rows is not None:
            self.rows = self.rows[:0]
