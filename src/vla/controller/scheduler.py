"""Chunk scheduling: turn action chunks into one action per control tick.

Two strategies ship: `SequentialScheduler` (blocking refill) and
`AsyncScheduler` (overlapped refill). `RTCScheduler` is still a follow-up
plan -- deferred until CUDA latency is measured, since RTC needs `delay <
chunk_length` to function at all (see the design doc's measured-latency
section: on the measured Apple Silicon target, `delay > chunk_length`, so
RTC would discard every chunk on every merge). The `ActionQueue` underneath
already supports both modes (`QueueSettings.rtc_enabled`).

`ChunkScheduler.next_action` is typed to allow returning `None`.
`SequentialScheduler` never actually does -- it raises `SchedulerError`
instead when it cannot produce an action. `AsyncScheduler` is the first
scheduler that genuinely returns `None`: when its queue is empty and a
background inference is already in flight, returning `None` (instead of
blocking) is the entire point of the overlap -- blocking there would freeze
the event loop exactly as badly as `SequentialScheduler` does, for exactly
the latency `AsyncScheduler` exists to hide.

That makes the controller's (`vla.controller.service`) "action is None"
branch in its tick loop reachable for the first time. `starvation_grace_ticks`
is consumed there for two related but distinct purposes that happen to share
one config field:

  - a bound on consecutive tick *failures* (exceptions raised out of
    `next_action`) when `safety.stop_on_error` is `False`: more than
    `starvation_grace_ticks` failures in a row stops the arm and halts
    regardless of `stop_on_error`.
  - a bound on consecutive *empty* ticks (`next_action` returning `None`,
    only possible under `AsyncScheduler`): more than `starvation_grace_ticks`
    such ticks in a row stops the arm and halts unconditionally -- this one
    is never gated by `stop_on_error`, since an empty tick is not a failure
    to skip, it is an absence of anything to do.

Both readings are "how much starvation/failure to tolerate before giving
up," which is why one field serves both, rather than adding a second,
near-duplicate field.
"""

from __future__ import annotations

import abc
import asyncio
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


def _validate_and_merge(queue: ActionQueue, processed: np.ndarray, raw: np.ndarray) -> None:
    """Validate a freshly-inferred chunk and merge it into `queue`.

    Shared by `SequentialScheduler` and `AsyncScheduler` so the two cannot
    drift apart on what counts as a malformed policy response -- both go
    from a raw `infer()` result to a merged queue through this, and only
    this.

    `real_delay=0` is baked in here rather than threaded through as a
    parameter: both callers run `ActionQueue` in append mode
    (`rtc_enabled=False`), where the delay is ignored outright, and giving
    either scheduler a way to pass a computed delay would imply RTC
    semantics that do not apply to either of them.
    """
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
        queue.merge(raw, processed, real_delay=0)
    except ActionQueueError as exc:
        # ActionQueue validates strictly (2D ndarray, matching action
        # dims) and raises its own ActionQueueError -- a caller of the
        # scheduler should never have to also know about that type, so it
        # is never let through "two frames down".
        raise SchedulerError(f"policy returned a malformed action chunk: {exc}") from exc


class ChunkScheduler(abc.ABC):
    @abc.abstractmethod
    async def next_action(self) -> np.ndarray | None:
        """Return the action for this tick, or None if none is available."""

    @abc.abstractmethod
    async def reset(self) -> None:
        """Clear episode-scoped state.

        Async on the ABC itself, not only on `AsyncScheduler`: clearing
        state safely can require awaiting the settlement of in-flight work
        first -- `AsyncScheduler.reset` must cancel its background
        inference and await it before touching the queue, or a
        `queue.clear()` racing a still-running merge would prove nothing.
        `SequentialScheduler.reset` has no such work, but implements the
        same async signature for a uniform, safe interface: a generic
        caller that forgot to `await` a `ChunkScheduler.reset()` call would
        otherwise silently no-op for one scheduler and corrupt state for
        the other, instead of failing the same way (a "coroutine was never
        awaited" warning) for both.
        """

    @abc.abstractmethod
    def qsize(self) -> int:
        """Actions remaining in the queue."""

    async def close(self) -> None:
        """Release background resources.

        Concrete, not abstract: most schedulers (`SequentialScheduler`)
        hold none and get this no-op for free. `AsyncScheduler` overrides
        it to cancel and await its in-flight inference task, so stopping
        the controller never leaves an orphaned background task running
        past the point it reports stopped.
        """
        return None


class SequentialScheduler(ChunkScheduler):
    """Blocking: when the queue drains, infer and refill.

    Inference latency directly stalls the control loop, which is exactly
    what `AsyncScheduler` exists to overlap -- but this is simple, and
    correct behavior here is the baseline the overlapped path is compared
    against.
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
        # Note this is *not* wrapped as SchedulerError: an arbitrary
        # exception from `infer` itself propagates as-is, unlike
        # AsyncScheduler, which must wrap it (see AsyncScheduler's
        # docstring) because it surfaces on a later, unrelated call.
        processed, raw = await self._infer(None)
        _validate_and_merge(self._queue, processed, raw)

        action = self._queue.get()
        if action is None:  # pragma: no cover - guarded by the shape check above
            raise SchedulerError("queue empty immediately after merge")
        return action

    async def reset(self) -> None:
        self._queue.clear()

    def qsize(self) -> int:
        return self._queue.qsize()


class AsyncScheduler(ChunkScheduler):
    """Overlaps execution with inference instead of stalling between chunks.

    `SequentialScheduler` blocks the control loop for the full inference
    latency every time the queue drains. When inference latency approaches
    or exceeds chunk duration -- measured at ~5.3s vs. a 5.0s chunk on
    Apple Silicon -- that is close to a 50% duty cycle, in multi-second
    freezes. RTC cannot rescue this case (it needs `delay < chunk_length`,
    and here `delay > chunk_length`: it would discard the entire chunk on
    every merge, permanent starvation). Plain overlap can: keep serving the
    current chunk's queued actions while the next chunk infers in the
    background, merged in **append** mode (`QueueSettings.rtc_enabled=
    False`) so the new chunk extends the queue instead of replacing it.

    The honest cost is a discontinuity at each chunk boundary -- the last
    action of chunk *k* and the first action of chunk *k+1* come from
    observations however-many seconds apart the inference took, and can
    disagree. The safety layer's delta clamp bounds that to one tick's
    budget, so it hitches rather than lurches. Smoothing that seam is
    exactly RTC's job, which cannot do it here -- this scheduler makes no
    attempt at smoothing; it is the "keep moving, plainly" strategy, not
    the "keep moving, smoothly" one.

    One fact makes an occasionally-stale observation acceptable for this
    policy: `n_obs_steps` is 1, so the policy consumes a single current
    observation with no temporal buffer for a stale frame to corrupt --
    a stale observation is merely stale, not buffer-corrupting.

    Concurrency contract:
      - Exactly one inference in flight at a time, tracked by `_inflight`,
        never started again while it is not `None`.
      - `next_action` never blocks except on the very first call (nothing
        queued, nothing in flight -- there is no chunk to overlap with yet)
        and, symmetrically, whenever the queue has run dry before any
        background inference was ever requested (e.g. `queue_threshold=0`).
      - When the queue is empty and an inference is already in flight,
        `next_action` returns `None` immediately rather than blocking --
        blocking there would freeze the event loop exactly as badly as
        `SequentialScheduler` does, defeating the entire point of overlap.
      - A background failure is never raised from inside the background
        task itself (an unhandled exception there would just log "Task
        exception was never retrieved" and vanish silently) -- it is
        captured and re-raised from the *next* call to `next_action`,
        wrapped as `SchedulerError` if it is not one already, and raised
        exactly once.
    """

    def __init__(self, infer: InferFn, queue_threshold: int) -> None:
        self._infer = infer
        self._queue = ActionQueue(QueueSettings(rtc_enabled=False))
        self._queue_threshold = queue_threshold
        self._inflight: asyncio.Task[None] | None = None
        self._pending_error: Exception | None = None

    async def next_action(self) -> np.ndarray | None:
        self._raise_pending_error()

        action = self._queue.get()
        if action is not None:
            self._maybe_start_background_inference()
            return action

        if self._inflight is not None:
            # Starved, but not stuck: inference is already working on the
            # next chunk. Returning None here (never blocking) is the
            # entire point of this scheduler -- the caller (the
            # controller's tick loop) holds position for the tick and
            # tries again next time.
            return None

        # Nothing queued, nothing in flight: either the very first call, or
        # queue_threshold left the queue run completely dry before a
        # refill was ever requested. Either way there is no chunk to
        # overlap with right now -- block, the only honest option.
        self._start_background_inference()
        await self._inflight
        self._raise_pending_error()

        action = self._queue.get()
        if action is None:  # pragma: no cover - guarded by the shape check in _validate_and_merge
            raise SchedulerError("queue empty immediately after merge")
        return action

    async def reset(self) -> None:
        await self._cancel_inflight()
        self._queue.clear()
        self._pending_error = None

    async def close(self) -> None:
        await self._cancel_inflight()
        self._pending_error = None

    def qsize(self) -> int:
        return self._queue.qsize()

    def _raise_pending_error(self) -> None:
        if self._pending_error is not None:
            err, self._pending_error = self._pending_error, None
            raise err

    def _maybe_start_background_inference(self) -> None:
        # No `await` between this check and `_start_background_inference`'s
        # `asyncio.create_task` call -- the two together are one atomic
        # step from asyncio's cooperative-scheduling point of view (a task
        # only ever switches at an `await`), which is what keeps "exactly
        # one inference in flight" true even under back-to-back
        # `next_action` calls with no scheduling gap between them.
        if self._inflight is None and self._queue.qsize() <= self._queue_threshold:
            self._start_background_inference()

    def _start_background_inference(self) -> None:
        # Keeping the task on `self` is load-bearing, not defensive: a
        # bare `create_task` handle can be garbage-collected before it
        # ever runs, since asyncio only holds a weak reference to it.
        self._inflight = asyncio.create_task(self._infer_and_merge())

    async def _infer_and_merge(self) -> None:
        try:
            processed, raw = await self._infer(None)
            _validate_and_merge(self._queue, processed, raw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Always wrapped, unlike SequentialScheduler: a bare exception
            # here surfaces on a *later*, unrelated `next_action` call, not
            # the one that triggered inference -- a caller only knows to
            # catch SchedulerError from this scheduler, and by the time
            # this is raised there is no other frame left to explain where
            # it came from.
            self._pending_error = (
                exc
                if isinstance(exc, SchedulerError)
                else SchedulerError(f"background inference failed: {exc}")
            )
        finally:
            self._inflight = None

    async def _cancel_inflight(self) -> None:
        task = self._inflight
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            # Whatever this in-flight inference concluded with -- an error,
            # or a completed merge that raced the cancellation attempt --
            # is about to be discarded (`reset`'s `queue.clear()` right
            # after this returns, or `close` shutting the scheduler down
            # outright); it must not escape here over a result this call
            # is itself throwing away.
            pass
        self._inflight = None
