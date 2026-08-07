"""Chunk scheduling: turn action chunks into one action per control tick.

Only the sequential strategy ships today. `RTCScheduler` is a follow-up plan;
the `ActionQueue` underneath already supports both modes (`QueueSettings.
rtc_enabled`).

`ChunkScheduler.next_action` is typed to allow returning `None`, but
`SequentialScheduler` never actually does -- it raises `SchedulerError`
instead when it cannot produce an action. That makes an "action is None"
branch inside the controller's loop unreachable in this phase: it would
exist for `RTCScheduler`, where a background inference thread can genuinely
leave the queue empty for a tick.

`starvation_grace_ticks` itself is *not* unreachable, though -- the
controller (`vla.controller.service`) consumes it independently, as a bound
on consecutive tick *failures* (not empty-queue ticks) when
`safety.stop_on_error` is `False`: more than `starvation_grace_ticks`
failures in a row stops the arm and halts regardless of `stop_on_error`, so
a deployment that opts out of per-failure halting still cannot spin forever
reporting "running" while every tick silently fails. That reuse of the same
config field for a related but distinct purpose is deliberate -- both are
"how much starvation/failure to tolerate before giving up" -- rather than
adding a second, near-duplicate field ahead of `RTCScheduler` actually
existing.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Awaitable, Callable

import numpy as np

from vla.config_util import VLAError

from .action_queue import ActionQueue, ActionQueueError, QueueSettings

LOGGER = logging.getLogger(__name__)

InferFn = Callable[[dict[str, Any] | None], Awaitable[tuple[np.ndarray, np.ndarray]]]


class SchedulerError(VLAError, RuntimeError):
    """Raised when the scheduler cannot produce an action.

    A `RuntimeError` base (matching `SafetyError`/`ObservationError`'s
    convention): the inputs are usually well-typed on their own terms, it's
    the runtime response from the policy that's unusable.
    """


class ChunkScheduler(abc.ABC):
    @abc.abstractmethod
    async def next_action(self) -> np.ndarray | None:
        """Return the action for this tick, or None if none is available."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode-scoped state."""

    @abc.abstractmethod
    def qsize(self) -> int:
        """Actions remaining in the queue."""


class SequentialScheduler(ChunkScheduler):
    """Blocking: when the queue drains, infer and refill.

    Inference latency directly stalls the control loop, which is exactly
    what RTC exists to fix -- but it is simple, and correct behavior here is
    the baseline the RTC path will be compared against.
    """

    def __init__(self, infer: InferFn) -> None:
        self._infer = infer
        self._queue = ActionQueue(QueueSettings(rtc_enabled=False))

    async def next_action(self) -> np.ndarray:
        action = self._queue.get()
        if action is not None:
            return action

        # If `infer` raises, it does so before any queue mutation below, so
        # the queue is left exactly as it was (empty) -- not partially
        # written and not corrupted such that the next call misbehaves.
        processed, raw = await self._infer(None)

        try:
            if processed.shape[0] == 0:
                raise SchedulerError("policy returned an empty action chunk")
        except AttributeError as exc:
            # `.shape` on a non-ndarray (e.g. a policy service returning
            # plain lists after a decode bug) must not leak as a bare
            # AttributeError -- callers only know to catch SchedulerError.
            raise SchedulerError(
                "policy returned a malformed action chunk: expected numpy arrays from "
                f"infer(), got processed={type(processed).__name__!r} "
                f"raw={type(raw).__name__!r}"
            ) from exc

        try:
            self._queue.merge(raw, processed, real_delay=0)
        except ActionQueueError as exc:
            # ActionQueue validates strictly (2D ndarray, matching action
            # dims) and raises its own ActionQueueError -- a caller of this
            # scheduler should never have to also know about that type, so
            # it is never let through "two frames down".
            raise SchedulerError(f"policy returned a malformed action chunk: {exc}") from exc

        action = self._queue.get()
        if action is None:  # pragma: no cover - guarded by the shape check above
            raise SchedulerError("queue empty immediately after merge")
        return action

    def reset(self) -> None:
        self._queue.clear()

    def qsize(self) -> int:
        return self._queue.qsize()
