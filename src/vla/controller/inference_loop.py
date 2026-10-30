"""When to infer: one request in flight, fired on the buffer's remaining time.

Replaces both schedulers. Every queue-depth knob they had (queue_threshold,
actions_per_chunk, merge modes) was a stand-in for a time axis the buffer now
has. The trigger reads straight off it: fire when the plan has less left than
the next inference is expected to take, plus a blend window and a margin.

Two regimes:

- overlap: fire on that trigger, so a chunk lands while the old plan still has
  `blend_s` to crossfade against.
- sequential: when `2 * expected_latency + blend_s + margin_s >= chunk_s`.
  With one request in flight, a chunk observed at o lands at o + L, and the
  next can only be observed then, landing at o + 2L. Past that bound the plan
  runs out between landings anyway: overlap would starve every chunk, and
  past L ~ chunk_s it would observe mid-chunk and land rows the arm already
  passed. So fire only once the plan has run out (the controller's `wait()`),
  observe at rest, and let the rebase run the whole chunk. This is what
  `mode: sequential` did.

The remaining time is measured from the buffer's last sample, not the clock:
each tick samples a lead (up to 2/fps) ahead of now.

`expected_latency` is the max of the last 5 observation-to-landing intervals:
an underestimate starves and an overestimate only fires early. Until two
samples exist it is the chunk duration, and the regime stays overlap, which
fires right after each landing.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Awaitable, Callable

import numpy as np

from vla.config_util import VLAError

from .reference_buffer import ReferenceBuffer

InferFn = Callable[[], Awaitable[tuple[np.ndarray, float]]]

_EXPECTED_WINDOW = 5
_MIN_SAMPLES = 2


class InferenceError(VLAError, RuntimeError):
    """An inference failed, timed out, or returned an unusable chunk."""


class InferenceLoop:
    def __init__(
        self,
        infer: InferFn,
        buffer: ReferenceBuffer,
        *,
        chunk_s: float,
        blend_s: float,
        margin_s: float,
        timeout_s: float,
        clock: Callable[[], float],
    ) -> None:
        self._infer = infer
        self._buffer = buffer
        self._chunk_s = chunk_s
        self._blend_s = blend_s
        self._margin_s = margin_s
        self._timeout_s = timeout_s
        self._clock = clock
        # Observation-to-landing seconds, most recent 50 (status reports the mean).
        self.latencies: deque[float] = deque(maxlen=50)
        self._inflight: asyncio.Task[None] | None = None
        self._pending_error: Exception | None = None

    @property
    def expected_latency(self) -> float:
        if len(self.latencies) < _MIN_SAMPLES:
            return self._chunk_s
        return max(list(self.latencies)[-_EXPECTED_WINDOW:])

    @property
    def regime(self) -> str:
        if (
            len(self.latencies) >= _MIN_SAMPLES
            and 2 * self.expected_latency + self._blend_s + self._margin_s >= self._chunk_s
        ):
            return "sequential"
        return "overlap"

    def poll(self) -> None:
        """Called once per tick: raise a background failure, maybe fire."""
        self._raise_pending_error()
        if self._inflight is not None or self.regime != "overlap":
            return
        lead = self.expected_latency + self._blend_s + self._margin_s
        if self._buffer.horizon() - self._buffer.last_sampled < lead:
            self._fire()

    async def wait(self) -> None:
        """Block until a chunk lands, firing one if none is in flight."""
        self._raise_pending_error()
        if self._inflight is None:
            self._fire()
        try:
            # Not shielded: a timeout or a cancelled caller abandons the request.
            await asyncio.wait_for(self._inflight, self._timeout_s)
        except asyncio.TimeoutError:
            raise InferenceError(
                f"inference did not land within inference_timeout_s={self._timeout_s}"
            ) from None
        self._raise_pending_error()

    async def close(self) -> None:
        task, self._inflight = self._inflight, None
        self._pending_error = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            # Whatever it concluded is discarded by the shutdown this is part of.
            pass

    def _fire(self) -> None:
        # Kept on self: asyncio holds only a weak reference to a bare task.
        self._inflight = asyncio.create_task(self._infer_and_land())

    async def _infer_and_land(self) -> None:
        try:
            chunk, t_obs = await self._infer()
            self._buffer.land(chunk, t_obs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Raised from the next poll()/wait(): an exception left in a task
            # would only log "never retrieved" and vanish.
            self._pending_error = (
                exc if isinstance(exc, InferenceError) else InferenceError(f"inference failed: {exc}")
            )
        else:
            self.latencies.append(self._clock() - t_obs)
        finally:
            self._inflight = None

    def _raise_pending_error(self) -> None:
        if self._pending_error is not None:
            err, self._pending_error = self._pending_error, None
            raise err
