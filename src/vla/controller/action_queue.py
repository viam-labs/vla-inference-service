"""Numpy port of lerobot `policies/rtc/action_queue.py::ActionQueue`.

Method names deliberately match upstream so future upstream diffs remain
reviewable against this file. Upstream's only torch usage is `.clone()`,
`torch.cat`, and slicing, so this port is mechanical:
`.clone()` -> `.copy()`, `torch.cat` -> `np.concatenate`.

Two parallel arrays are maintained:
  original_queue  policy-space actions, the source of `prev_chunk_left_over`
                  (returned by `get_left_over`)
  queue           postprocessed actions, what the robot actually executes
                  (returned by `get_processed_left_over`)

Verified against lerobot at git SHA
`ff7cc3de1de830f5f3276918a013d04bdf9ea4be`,
`src/lerobot/policies/rtc/action_queue.py` (all 247 lines), including the
`_check_and_resolve_delays` oddity: on an index-delta mismatch it logs a
warning and returns the *unclamped* `real_delay`, not
`max(0, real_delay)`. That branch is ported verbatim -- see
`tests/controller/test_action_queue_differential.py` for the mechanical
proof this port agrees with upstream, and
`tests/controller/test_action_queue.py::test_index_mismatch_returns_unclamped_delay`
/ `test_index_match_does_not_warn` for the two branches individually.

`real_delay` and `action_index_before_inference` are coerced via
`vla.config_util.as_int`: both arrive as protobuf-Struct doubles in
production (e.g. `2.0`), never plain ints. A fractional value (`2.5`) is a
config/caller bug, not a truncation target, so it raises `ActionQueueError`
rather than being silently floored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from vla.config_util import ConfigError, VLAError, as_int

LOGGER = logging.getLogger(__name__)


class ActionQueueError(VLAError, ValueError):
    """Raised when `ActionQueue.merge()` receives a malformed argument."""


def _as_int_field(value: Any, field_name: str) -> int:
    try:
        return as_int(value, field_name)
    except ConfigError as exc:
        raise ActionQueueError(str(exc)) from exc


@dataclass(frozen=True)
class QueueSettings:
    rtc_enabled: bool = False


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting
       for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity
    """

    def __init__(self, cfg: QueueSettings) -> None:
        self.queue: np.ndarray | None = None  # Processed actions for robot rollout
        self.original_queue: np.ndarray | None = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> np.ndarray | None:
        """Get the next action from the queue.

        Returns a copy to prevent external modifications from corrupting
        the queue's stored chunk.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Number of unconsumed actions in the queue."""
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """True if no actions remain."""
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """Index of the next action to be consumed."""
        with self.lock:
            return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        """Unconsumed *original* (policy-space) actions.

        This is `prev_chunk_left_over` for RTC: the source of the next
        chunk's correction target. Distinct from `get_processed_left_over`,
        which is robot-space -- see
        tests/controller/test_action_queue.py::test_left_over_and_processed_left_over_are_not_interchangeable.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].copy()

    def get_processed_left_over(self) -> np.ndarray | None:
        """Unconsumed *processed* (robot-space) actions -- what the robot
        still has queued for execution."""
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].copy()

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: Any,
        action_index_before_inference: Any | None = None,
    ) -> None:
        """Merge new actions into the queue.

        RTC enabled: replaces the queue, accounting for inference delay.
        RTC disabled: appends to the queue, maintaining continuity.
        """
        delay_int = _as_int_field(real_delay, "real_delay")
        idx_int = (
            None
            if action_index_before_inference is None
            else _as_int_field(action_index_before_inference, "action_index_before_inference")
        )
        with self.lock:
            delay = self._check_and_resolve_delays(delay_int, idx_int)

            if self.cfg.rtc_enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(
        self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay: int
    ) -> None:
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to
        the time spent during inference, when the robot was executing
        previous actions.
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped_delay:].copy()
        self.queue = processed_actions[clamped_delay:].copy()

        LOGGER.debug("original_actions shape: %s", self.original_queue.shape)
        LOGGER.debug("processed_actions shape: %s", self.queue.shape)
        LOGGER.debug("real_delay: %s, clamped_delay: %s", real_delay, clamped_delay)

        self.last_index = 0

    def _append_actions_queue(
        self, original_actions: np.ndarray, processed_actions: np.ndarray
    ) -> None:
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.
        """
        if self.queue is None:
            self.original_queue = original_actions.copy()
            self.queue = processed_actions.copy()
            return

        self.original_queue = np.concatenate([self.original_queue, original_actions.copy()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = np.concatenate([self.queue, processed_actions.copy()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference. On a mismatch, logs a
        warning and returns the *unclamped* `real_delay` -- ported verbatim
        from upstream, oddity included.

        Note on that oddity, found while mutation-testing this port: the
        difference between returning `real_delay` here and returning
        `effective_delay` (as the non-mismatch path does) only matters when
        `real_delay` is negative, and every caller of this method re-clamps
        with its own `max(0, ...)` downstream -- `_replace_actions_queue`
        via `min(real_delay, len(original_actions), len(processed_actions))`,
        and non-RTC mode ignores the returned delay entirely. So this
        specific branch's return value is provably unobservable in this
        class's own state; a test asserting otherwise would be asserting a
        tautology. It is still ported verbatim (rather than "simplified" to
        always return `effective_delay`) because faithfulness to upstream --
        checked mechanically by the differential test, not by hand -- is the
        entire point of this file existing as a port rather than a
        reimplementation.
        """
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                LOGGER.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
