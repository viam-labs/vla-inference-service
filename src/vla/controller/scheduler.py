"""Chunk scheduling: turn action chunks into one action per control tick.

Two strategies ship: `SequentialScheduler` (blocking refill) and
`AsyncScheduler` (overlapped refill). `RTCScheduler` is still a follow-up
plan -- deferred until CUDA latency is measured, since RTC needs `delay <
chunk_length` to function at all (on the measured Apple Silicon target
`delay > chunk_length`, so RTC would discard every chunk on every merge).
The `ActionQueue` underneath already supports both modes.

`ChunkScheduler.next_action` is typed to allow returning `None`.
`SequentialScheduler` never actually does -- it raises `SchedulerError`
instead. `AsyncScheduler` is the first scheduler that genuinely returns
`None`: when its queue is empty and a background inference is already in
flight, returning `None` instead of blocking is the entire point of the
overlap.

That makes the controller's "action is None" branch reachable, where
`starvation_grace_ticks` is consumed for two related purposes:

  - a bound on consecutive tick *failures* (exceptions out of `next_action`)
    when `safety.stop_on_error` is `False`.
  - a bound on consecutive *empty* ticks (`next_action` returning `None`,
    only possible under `AsyncScheduler`) -- never gated by `stop_on_error`,
    since an empty tick is not a failure to skip but an absence of anything
    to do.

Both readings are "how much starvation/failure to tolerate before giving
up," which is why one field serves both.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
import time
from collections import deque
from typing import Any, Awaitable, Callable

import numpy as np

from vla.config_util import VLAError

from .action_queue import ActionQueue, ActionQueueError, QueueSettings

LOGGER = logging.getLogger(__name__)

InferFn = Callable[[dict[str, Any] | None], Awaitable[tuple[np.ndarray, np.ndarray]]]

# AsyncScheduler's starvation-risk warning: how many recent completed
# inferences to average over, and the minimum before the average counts as
# "a stable reading" rather than a possible cold-start outlier.
_LATENCY_WINDOW = 5
_MIN_LATENCY_SAMPLES_BEFORE_WARNING = 2


class SchedulerError(VLAError, RuntimeError):
    """Raised when the scheduler cannot produce an action.

    A `RuntimeError` base (matching `SafetyError`/`ObservationError`): the
    inputs are usually well-typed on their own terms, it's the runtime
    response from the policy that's unusable.
    """


def _validate_and_merge(
    queue: ActionQueue,
    processed: np.ndarray,
    raw: np.ndarray,
    actions_per_chunk: int | None = None,
) -> None:
    """Validate a freshly-inferred chunk and merge it into `queue`.

    Shared by both schedulers so the two cannot drift apart on what counts
    as a malformed policy response. `real_delay=0` is baked in rather than a
    parameter: both callers run `ActionQueue` in append mode, where the
    delay is ignored, and a computed-delay parameter would imply RTC
    semantics that apply to neither.

    `actions_per_chunk` truncates before the merge, not after: the discarded
    tail must never reach the queue, or a later `qsize()` would count
    actions that will never be executed and the refill would fire late.
    Truncation is deliberately *not* a validation failure when the chunk is
    shorter than N -- a policy is free to return fewer rows than the
    operator budgeted for, and slicing past the end is already a no-op.
    """
    try:
        if processed.shape[0] == 0:
            raise SchedulerError("policy returned an empty action chunk")
    except AttributeError as exc:
        # `.shape` on a non-ndarray (e.g. a policy service returning plain
        # lists after a decode bug) must not leak as a bare AttributeError
        # -- callers only know to catch SchedulerError.
        raise SchedulerError(
            "policy returned a malformed action chunk: expected numpy arrays from "
            f"infer(), got processed={type(processed).__name__!r} "
            f"raw={type(raw).__name__!r}"
        ) from exc

    if actions_per_chunk is not None:
        processed = processed[:actions_per_chunk]
        raw = raw[:actions_per_chunk]

    try:
        queue.merge(raw, processed, real_delay=0)
    except ActionQueueError as exc:
        # ActionQueue raises its own type; a caller of the scheduler should
        # never have to also know about it.
        raise SchedulerError(f"policy returned a malformed action chunk: {exc}") from exc


class ChunkScheduler(abc.ABC):
    @abc.abstractmethod
    async def next_action(self) -> np.ndarray | None:
        """Return the action for this tick, or None if none is available."""

    @abc.abstractmethod
    def qsize(self) -> int:
        """Actions remaining in the queue."""

    async def close(self) -> None:
        """Release background resources.

        Concrete, not abstract: `SequentialScheduler` holds none and gets
        this no-op for free. `AsyncScheduler` overrides it to cancel and
        await its in-flight inference, so stopping the controller never
        leaves an orphaned background task running past the point it
        reports stopped.
        """
        return None


class SequentialScheduler(ChunkScheduler):
    """Blocking: when the queue drains, infer and refill.

    Inference latency directly stalls the control loop, which is what
    `AsyncScheduler` exists to overlap -- but this is simple, and correct
    behavior here is the baseline the overlapped path is compared against.
    """

    def __init__(self, infer: InferFn, actions_per_chunk: int | None = None) -> None:
        self._infer = infer
        self._queue = ActionQueue(QueueSettings(rtc_enabled=False))
        self._actions_per_chunk = actions_per_chunk

    async def next_action(self) -> np.ndarray:
        action = self._queue.get()
        if action is not None:
            return action

        # If `infer` raises it does so before any queue mutation below, so
        # the queue is left exactly as it was. Deliberately NOT wrapped as
        # SchedulerError: an arbitrary exception from `infer` propagates
        # as-is, unlike AsyncScheduler, which must wrap it because it
        # surfaces on a later, unrelated call.
        processed, raw = await self._infer(None)
        _validate_and_merge(self._queue, processed, raw, self._actions_per_chunk)

        action = self._queue.get()
        if action is None:  # pragma: no cover - guarded by the shape check above
            raise SchedulerError("queue empty immediately after merge")
        return action

    def qsize(self) -> int:
        return self._queue.qsize()


class AsyncScheduler(ChunkScheduler):
    """Overlaps execution with inference instead of stalling between chunks.

    `SequentialScheduler` blocks the control loop for the full inference
    latency every time the queue drains. When inference latency approaches
    or exceeds chunk duration -- measured at ~5.3s vs. a 5.0s chunk on
    Apple Silicon -- that is close to a 50% duty cycle, in multi-second
    freezes. RTC cannot rescue this case (it needs `delay < chunk_length`;
    here `delay > chunk_length`, so it would discard the entire chunk on
    every merge). Plain overlap can: keep serving the current chunk's queued
    actions while the next chunk infers in the background, merged in
    **append** mode so the new chunk extends the queue instead of replacing
    it.

    The honest cost is a discontinuity at each chunk boundary -- the last
    action of chunk *k* and the first of *k+1* come from observations
    however-many seconds apart the inference took, and can disagree. The
    safety layer's delta clamp bounds that to one tick's budget, so it
    hitches rather than lurches. Smoothing that seam is RTC's job, which
    cannot do it here. `n_obs_steps` is 1, so the policy consumes a single
    current observation with no temporal buffer for a stale frame to
    corrupt -- a stale observation is merely stale, not buffer-corrupting.

    Concurrency contract:
      - Exactly one inference in flight at a time, tracked by `_inflight`,
        never started again while it is not `None`.
      - `next_action` never blocks except on the very first call (nothing
        queued, nothing in flight) and, symmetrically, whenever the queue
        has run dry before any background inference was requested (e.g.
        `queue_threshold=0`).
      - When the queue is empty and an inference is already in flight,
        `next_action` returns `None` immediately rather than blocking.
      - A background failure is never raised from inside the background
        task itself (an unhandled exception there would just log "Task
        exception was never retrieved" and vanish) -- it is captured and
        re-raised from the *next* call to `next_action`, wrapped as
        `SchedulerError` if it is not one already, exactly once.

    `queue_threshold` decides how much of the overlap this realizes: a
    threshold that fires the refill too late leaves the queue draining to
    empty before the next chunk lands, even though a background task did
    fire. That is silent unless something says so, so once `fps` and a
    stable latency reading are known, `next_action` warns once (never every
    tick) if `queue_threshold < ceil(observed_latency * fps)`.
    """

    def __init__(
        self,
        infer: InferFn,
        queue_threshold: int,
        fps: float = 10.0,
        actions_per_chunk: int | None = None,
    ) -> None:
        self._infer = infer
        self._queue = ActionQueue(QueueSettings(rtc_enabled=False))
        self._queue_threshold = queue_threshold
        self._actions_per_chunk = actions_per_chunk
        self._fps = fps
        self._inflight: asyncio.Task[None] | None = None
        self._pending_error: Exception | None = None
        self._latencies: deque[float] = deque(maxlen=_LATENCY_WINDOW)
        self._warned_starvation_risk = False

    async def next_action(self) -> np.ndarray | None:
        self._raise_pending_error()

        action = self._queue.get()
        if action is not None:
            self._maybe_start_background_inference()
            return action

        if self._inflight is not None:
            # Starved, but not stuck: inference is already working on the
            # next chunk. Returning None here (never blocking) is the whole
            # point -- the caller holds position for the tick and tries again.
            return None

        # Nothing queued, nothing in flight: either the very first call, or
        # queue_threshold let the queue run dry before a refill was ever
        # requested. Either way there is nothing to overlap with -- block,
        # the only honest option.
        self._start_background_inference()
        await self._inflight
        self._raise_pending_error()

        action = self._queue.get()
        if action is None:  # pragma: no cover - guarded by _validate_and_merge
            raise SchedulerError("queue empty immediately after merge")
        return action

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
        # No `await` between this check and `create_task` -- the two are one
        # atomic step under asyncio's cooperative scheduling (a task only
        # switches at an `await`), which is what keeps "exactly one inference
        # in flight" true under back-to-back `next_action` calls.
        if self._inflight is None and self._queue.qsize() <= self._queue_threshold:
            self._start_background_inference()

    def _start_background_inference(self) -> None:
        # Keeping the task on `self` is load-bearing, not defensive: asyncio
        # holds only a weak reference, so a bare handle can be
        # garbage-collected before it ever runs.
        self._inflight = asyncio.create_task(self._infer_and_merge())

    async def _infer_and_merge(self) -> None:
        started = time.perf_counter()
        try:
            processed, raw = await self._infer(None)
            _validate_and_merge(self._queue, processed, raw, self._actions_per_chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Always wrapped, unlike SequentialScheduler: this surfaces on a
            # *later*, unrelated `next_action` call, and by then there is no
            # other frame left to explain where it came from.
            self._pending_error = (
                exc
                if isinstance(exc, SchedulerError)
                else SchedulerError(f"background inference failed: {exc}")
            )
        else:
            # Only a *completed* inference is a meaningful latency sample; a
            # failed one may have failed immediately, or stalled unrelatedly.
            self._latencies.append(time.perf_counter() - started)
            self._maybe_warn_starvation_risk()
        finally:
            self._inflight = None

    def _maybe_warn_starvation_risk(self) -> None:
        if self._warned_starvation_risk:
            return
        if len(self._latencies) < _MIN_LATENCY_SAMPLES_BEFORE_WARNING:
            # A single sample could be a cold-start outlier.
            return

        avg_latency = sum(self._latencies) / len(self._latencies)
        required = math.ceil(avg_latency * self._fps)
        if self._queue_threshold >= required:
            return

        self._warned_starvation_risk = True
        LOGGER.warning(
            "queue_threshold=%d is too low for the observed inference latency "
            "(avg %.3fs over %d sample(s) at fps=%.2f); avoiding starvation "
            "needs queue_threshold>=%d. The queue will drain before the next "
            "chunk arrives, and the arm will hold position for ~%d tick(s) per "
            "chunk. Raise queue_threshold (up to n_action_steps - 1), lower "
            "fps, or reduce the policy's number of denoising/diffusion steps "
            "(num_steps) to cut latency.",
            self._queue_threshold,
            avg_latency,
            len(self._latencies),
            self._fps,
            required,
            required - self._queue_threshold,
        )

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
            # Whatever this in-flight inference concluded with -- an error, or
            # a completed merge that raced the cancellation -- is about to be
            # discarded by the shutdown this call is part of; it must not
            # escape over a result the caller is throwing away.
            pass
        self._inflight = None
