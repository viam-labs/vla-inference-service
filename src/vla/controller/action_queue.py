"""Append-only queue of postprocessed actions, served one per control tick.

Both schedulers merge each freshly inferred chunk onto the tail and `get()`
from the head, so a chunk arriving mid-execution extends the queue rather
than replacing it. Rows already consumed are dropped on the next merge.

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
        """Append a `(time_steps, action_dim)` chunk after the unconsumed tail.

        Anything but a 2D ndarray is refused here rather than detonating
        frames later: a list degrades `get()` to returning lists, a 1D array
        makes `qsize()` report the action dimension as a step count.
        """
        if not isinstance(actions, np.ndarray) or actions.ndim != 2:
            raise ActionQueueError(
                "actions must be a 2D numpy array shaped (time_steps, action_dim), got "
                f"{type(actions).__name__}"
                + (f" with shape {actions.shape}" if isinstance(actions, np.ndarray) else "")
            )
        with self.lock:
            if self.queue is None:
                self.queue = actions.copy()
            else:
                self.queue = np.concatenate([self.queue[self.last_index :], actions])
            self.last_index = 0
