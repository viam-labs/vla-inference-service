"""Queue of postprocessed actions, served one per control tick.

Both schedulers merge each freshly inferred chunk onto the tail and `get()`
from the head; `AsyncScheduler` under `merge="aligned"` instead `replace()`s
the unconsumed rows with the landing chunk.

This used to be a verbatim port of lerobot's `policies/rtc/action_queue.py`,
carrying RTC's replace mode, inference-delay bookkeeping and a parallel
policy-space queue for `prev_chunk_left_over`. None of that had a caller:
`mode: "rtc"` was never implemented. Port the upstream file again when an
`RTCScheduler` actually needs it.
"""

from __future__ import annotations

from threading import Lock

import numpy as np

from vla.config_util import VLAError


class ActionQueueError(VLAError, ValueError):
    """Raised when `ActionQueue.merge()` receives a malformed chunk."""


def _check_chunk(actions) -> None:
    """Anything but a 2D ndarray is refused here rather than detonating
    frames later: a list degrades `get()` to returning lists, a 1D array
    makes `qsize()` report the action dimension as a step count."""
    if not isinstance(actions, np.ndarray) or actions.ndim != 2:
        raise ActionQueueError(
            "actions must be a 2D numpy array shaped (time_steps, action_dim), got "
            f"{type(actions).__name__}"
            + (f" with shape {actions.shape}" if isinstance(actions, np.ndarray) else "")
        )


class ActionQueue:
    def __init__(self) -> None:
        self.queue: np.ndarray | None = None
        self.last_index = 0
        self.lock = Lock()

    def get(self) -> np.ndarray | None:
        """Next action, as a copy so a caller cannot corrupt the stored chunk."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def qsize(self) -> int:
        with self.lock:
            return 0 if self.queue is None else len(self.queue) - self.last_index

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.last_index = 0

    def merge(self, actions: np.ndarray) -> None:
        """Append a `(time_steps, action_dim)` chunk after the unconsumed tail."""
        _check_chunk(actions)
        with self.lock:
            if self.queue is None:
                self.queue = actions.copy()
            else:
                self.queue = np.concatenate([self.queue[self.last_index :], actions])
            self.last_index = 0

    def replace(
        self, actions: np.ndarray, blend_ticks: int = 0, blend_dims: int | None = None
    ) -> None:
        """Replace the unconsumed rows with `actions`, the landing chunk's rows
        from the next tick on.

        Row j of `actions` serves the same tick the j-th unconsumed row would
        have, so the first `b = min(blend_ticks, unconsumed, len(actions))`
        rows can crossfade from old to new, `w_j = (j+1)/(b+1)`. Only the first
        `blend_dims` columns blend (None means all). The rest -- the gripper --
        take the new value, since averaging open and close half-closes it.
        """
        _check_chunk(actions)
        with self.lock:
            new = actions.copy()
            old = None if self.queue is None else self.queue[self.last_index :]
            if old is not None and old.shape[1] != new.shape[1]:
                raise ActionQueueError(
                    f"cannot replace a {old.shape[1]}-column queue with a "
                    f"{new.shape[1]}-column chunk"
                )
            b = 0 if old is None else min(blend_ticks, len(old), len(new))
            if b:
                d = new.shape[1] if blend_dims is None else blend_dims
                w = (np.arange(1, b + 1) / (b + 1))[:, None]
                new[:b, :d] = old[:b, :d] + w * (new[:b, :d] - old[:b, :d])
            self.queue = new
            self.last_index = 0
