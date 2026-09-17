import asyncio
import logging
import time

import numpy as np
import pytest

from vla.controller.action_queue import ActionQueueError
from vla.controller.scheduler import (
    AsyncScheduler,
    ChunkScheduler,
    SchedulerError,
    SequentialScheduler,
)


class RecordingInfer:
    """Stands in for a call to the policy service."""

    def __init__(self, n=4, dim=2, fail_after=None):
        self.calls = 0
        self.n = n
        self.dim = dim
        self.fail_after = fail_after
        self.last_rtc = "unset"

    async def __call__(self, rtc):
        self.calls += 1
        self.last_rtc = rtc
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("inference exploded")
        base = np.full((self.n, self.dim), float(self.calls), dtype=np.float32)
        return base + 100.0, base


# ---------------------------------------------------------------------------
# Plan's baseline suite (Task 15, Step 1).
# ---------------------------------------------------------------------------


async def test_first_tick_triggers_inference():
    infer = RecordingInfer()
    s = SequentialScheduler(infer)
    action = await s.next_action()
    assert infer.calls == 1
    np.testing.assert_allclose(action, [101.0, 101.0])


async def test_queue_drains_before_reinferring():
    infer = RecordingInfer(n=3)
    s = SequentialScheduler(infer)
    for _ in range(3):
        await s.next_action()
    assert infer.calls == 1
    await s.next_action()
    assert infer.calls == 2


async def test_serves_processed_actions_not_raw():
    infer = RecordingInfer()
    s = SequentialScheduler(infer)
    action = await s.next_action()
    assert action[0] == 101.0  # processed = raw + 100


async def test_sequential_mode_sends_no_rtc_payload():
    infer = RecordingInfer()
    s = SequentialScheduler(infer)
    await s.next_action()
    assert infer.last_rtc is None


async def test_qsize_reflects_remaining():
    infer = RecordingInfer(n=4)
    s = SequentialScheduler(infer)
    await s.next_action()
    assert s.qsize() == 3


async def test_qsize_is_zero_before_any_tick():
    # Assert the pre-tick default explicitly, not only the post-tick value.
    s = SequentialScheduler(RecordingInfer(n=4))
    assert s.qsize() == 0


async def test_empty_chunk_is_an_error():
    class EmptyInfer:
        async def __call__(self, rtc):
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    s = SequentialScheduler(EmptyInfer())
    with pytest.raises(SchedulerError, match="empty"):
        await s.next_action()


# ---------------------------------------------------------------------------
# Inference failures must propagate, and must not corrupt the queue.
# ---------------------------------------------------------------------------


async def test_inference_failure_propagates():
    s = SequentialScheduler(RecordingInfer(fail_after=0))
    with pytest.raises(RuntimeError, match="exploded"):
        await s.next_action()


async def test_inference_failure_is_not_swallowed_as_none():
    # A scheduler that caught the exception and returned None instead would
    # violate next_action's documented contract (it never returns None; it
    # raises) and would silently stall the control loop instead of failing
    # loudly.
    s = SequentialScheduler(RecordingInfer(fail_after=0))
    with pytest.raises(RuntimeError):
        result = await s.next_action()
        assert result is not None  # pragma: no cover - only reached on regression


async def test_queue_is_still_usable_after_a_failed_inference():
    # A failed inference happens before ActionQueue.merge() is ever called,
    # so the queue must be left exactly as it was (empty) -- not partially
    # written, not in a state that makes the next call misbehave.
    infer = RecordingInfer(n=3, fail_after=0)
    s = SequentialScheduler(infer)
    with pytest.raises(RuntimeError, match="exploded"):
        await s.next_action()
    assert s.qsize() == 0

    infer.fail_after = None  # let the retry succeed
    action = await s.next_action()
    assert infer.calls == 2
    np.testing.assert_allclose(action, [102.0, 102.0])
    assert s.qsize() == 2


async def test_repeated_inference_failures_each_propagate_independently():
    infer = RecordingInfer(fail_after=0)
    s = SequentialScheduler(infer)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="exploded"):
            await s.next_action()
    assert infer.calls == 3
    assert s.qsize() == 0


# ---------------------------------------------------------------------------
# Malformed policy responses must surface as SchedulerError, not as an
# ActionQueueError raised two frames down inside ActionQueue.merge().
# ---------------------------------------------------------------------------


async def test_mismatched_action_dims_surfaces_as_scheduler_error_not_action_queue_error():
    class MismatchedInfer:
        async def __call__(self, rtc):
            processed = np.zeros((4, 2), dtype=np.float32)
            raw = np.zeros((4, 3), dtype=np.float32)  # wrong action dim
            return processed, raw

    s = SequentialScheduler(MismatchedInfer())
    with pytest.raises(SchedulerError, match="malformed") as exc_info:
        await s.next_action()
    assert not isinstance(exc_info.value, ActionQueueError)


async def test_non_array_response_surfaces_as_scheduler_error_not_attribute_error():
    class ListInfer:
        async def __call__(self, rtc):
            # A policy service returning plain lists (e.g. a decode bug) must
            # not blow up with a bare AttributeError on `.shape`.
            return [[1.0, 2.0]], [[1.0, 2.0]]

    s = SequentialScheduler(ListInfer())
    with pytest.raises(SchedulerError, match="malformed"):
        await s.next_action()


async def test_one_dimensional_response_surfaces_as_scheduler_error():
    class FlatInfer:
        async def __call__(self, rtc):
            flat = np.zeros(4, dtype=np.float32)
            return flat, flat

    s = SequentialScheduler(FlatInfer())
    with pytest.raises(SchedulerError, match="malformed") as exc_info:
        await s.next_action()
    assert not isinstance(exc_info.value, ActionQueueError)


async def test_conforming_arrays_reach_the_queue_unmodified():
    # The scheduler must pass ActionQueue.merge() proper 2D ndarrays of
    # matching action dims when the policy behaves -- confirmed by checking
    # the underlying ActionQueue's own state, not just next_action()'s
    # return value.
    infer = RecordingInfer(n=4, dim=3)
    s = SequentialScheduler(infer)
    await s.next_action()
    assert isinstance(s._queue.queue, np.ndarray)
    assert s._queue.queue.ndim == 2
    assert s._queue.queue.shape == (4, 3)


# ---------------------------------------------------------------------------
# ABC contract: ChunkScheduler must actually enforce its interface.
# ---------------------------------------------------------------------------


def test_chunk_scheduler_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ChunkScheduler()


def test_subclass_omitting_a_method_cannot_be_instantiated():
    class Incomplete(ChunkScheduler):
        async def next_action(self):
            return None


def test_subclass_implementing_every_method_can_be_instantiated():
    class Complete(ChunkScheduler):
        async def next_action(self):
            return None

        def qsize(self) -> int:
            return 0

    Complete()  # must not raise


def test_sequential_scheduler_is_a_chunk_scheduler():
    assert isinstance(SequentialScheduler(RecordingInfer()), ChunkScheduler)


# ---------------------------------------------------------------------------
# AsyncScheduler
# ---------------------------------------------------------------------------


class ControllableInfer:
    """An infer() whose completion the test controls with an event, so tests
    can assert what happens *while* inference is genuinely suspended in
    flight -- not just before and after it settles."""

    def __init__(self, n=4, dim=2):
        self.n = n
        self.dim = dim
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._release = asyncio.Event()
        self._entered = asyncio.Event()

    def release(self) -> None:
        self._release.set()

    async def wait_until_entered(self) -> None:
        await self._entered.wait()
        self._entered.clear()

    async def __call__(self, rtc):
        self.calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self._entered.set()
        try:
            await self._release.wait()
        finally:
            self.concurrent -= 1
        self._release.clear()
        base = np.full((self.n, self.dim), float(self.calls), dtype=np.float32)
        return base + 100.0, base


def test_async_scheduler_is_a_chunk_scheduler():
    assert isinstance(AsyncScheduler(RecordingInfer(), queue_threshold=1), ChunkScheduler)


async def test_async_first_tick_blocks_and_triggers_inference():
    # Mirrors SequentialScheduler's test_first_tick_triggers_inference --
    # the very first call has nothing to overlap with, so it must behave
    # identically to the blocking scheduler here.
    infer = RecordingInfer()
    s = AsyncScheduler(infer, queue_threshold=1)
    action = await s.next_action()
    assert infer.calls == 1
    np.testing.assert_allclose(action, [101.0, 101.0])


async def test_async_serves_processed_actions_not_raw():
    infer = RecordingInfer()
    s = AsyncScheduler(infer, queue_threshold=1)
    action = await s.next_action()
    assert action[0] == 101.0  # processed = raw + 100


async def test_async_sends_no_rtc_payload():
    infer = RecordingInfer()
    s = AsyncScheduler(infer, queue_threshold=1)
    await s.next_action()
    assert infer.last_rtc is None


async def test_async_merges_in_append_mode_not_replace():
    # AsyncScheduler must use rtc_enabled=False (append) -- replace mode
    # would discard, rather than extend, the still-unconsumed tail of the
    # previous chunk on every background merge.
    infer = RecordingInfer(n=3, dim=2)
    s = AsyncScheduler(infer, queue_threshold=1)
    await s.next_action()
    assert s._queue.cfg.rtc_enabled is False


async def test_async_qsize_is_zero_before_any_tick():
    s = AsyncScheduler(RecordingInfer(n=4), queue_threshold=1)
    assert s.qsize() == 0


async def test_async_empty_chunk_is_a_scheduler_error():
    class EmptyInfer:
        async def __call__(self, rtc):
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    s = AsyncScheduler(EmptyInfer(), queue_threshold=1)
    with pytest.raises(SchedulerError, match="empty"):
        await s.next_action()


async def test_async_mismatched_action_dims_surfaces_as_scheduler_error():
    class MismatchedInfer:
        async def __call__(self, rtc):
            processed = np.zeros((4, 2), dtype=np.float32)
            raw = np.zeros((4, 3), dtype=np.float32)
            return processed, raw

    s = AsyncScheduler(MismatchedInfer(), queue_threshold=1)
    with pytest.raises(SchedulerError, match="malformed") as exc_info:
        await s.next_action()
    assert not isinstance(exc_info.value, ActionQueueError)


async def test_async_non_array_response_surfaces_as_scheduler_error():
    class ListInfer:
        async def __call__(self, rtc):
            return [[1.0, 2.0]], [[1.0, 2.0]]

    s = AsyncScheduler(ListInfer(), queue_threshold=1)
    with pytest.raises(SchedulerError, match="malformed"):
        await s.next_action()


async def test_async_first_call_failure_propagates_wrapped_as_scheduler_error():
    # Unlike SequentialScheduler (which lets an arbitrary infer() exception
    # through unwrapped), AsyncScheduler always wraps -- even on the very
    # first, blocking call -- because the same code path is used for
    # background failures that surface on an unrelated later call, and a
    # caller of this scheduler only knows to catch SchedulerError.
    s = AsyncScheduler(RecordingInfer(fail_after=0), queue_threshold=1)
    with pytest.raises(SchedulerError, match="exploded"):
        await s.next_action()


async def test_async_first_call_failure_leaves_queue_usable():
    infer = RecordingInfer(n=3, fail_after=0)
    s = AsyncScheduler(infer, queue_threshold=1)
    with pytest.raises(SchedulerError, match="exploded"):
        await s.next_action()
    assert s.qsize() == 0

    infer.fail_after = None
    action = await s.next_action()
    assert infer.calls == 2
    np.testing.assert_allclose(action, [102.0, 102.0])


# --- overlap: the whole point --------------------------------------------


async def test_overlap_keeps_wall_time_far_below_sequential():
    """Correctness alone would pass against SequentialScheduler too -- the
    point of AsyncScheduler is wall-clock: while one chunk is executing, the
    next is already inferring, so total time should track tick pacing plus
    roughly *one* inference, not `num_chunks` inferences."""
    n = 10
    dim = 2
    infer_delay = 0.04
    tick_pause = 0.008
    threshold = n - 1  # refill as soon as a single action has been consumed
    num_chunks = 6
    total_actions = n * num_chunks

    class TimedInfer:
        def __init__(self):
            self.calls = 0

        async def __call__(self, rtc):
            self.calls += 1
            await asyncio.sleep(infer_delay)
            base = np.full((n, dim), float(self.calls), dtype=np.float32)
            return base + 100.0, base

    async def _drain(scheduler, count):
        started = time.perf_counter()
        for _ in range(count):
            action = await scheduler.next_action()
            assert action is not None
            await asyncio.sleep(tick_pause)
        return time.perf_counter() - started

    async_infer = TimedInfer()
    async_sched = AsyncScheduler(async_infer, queue_threshold=threshold)
    async_elapsed = await _drain(async_sched, total_actions)
    # An aggressive threshold (n - 1) plus real scheduling jitter means the
    # exact fire count can land a little above the naive total_actions / n
    # -- what matters here is overlap (asserted below), not pinning this to
    # an exact count that depends on timing coincidences.
    assert num_chunks <= async_infer.calls <= num_chunks + 2

    seq_infer = TimedInfer()
    seq_sched = SequentialScheduler(seq_infer)
    seq_elapsed = await _drain(seq_sched, total_actions)
    assert seq_infer.calls == num_chunks

    assert async_elapsed < seq_elapsed * 0.85, (
        f"AsyncScheduler ({async_elapsed:.3f}s) is not meaningfully faster than "
        f"SequentialScheduler ({seq_elapsed:.3f}s) for identical inference timing "
        "-- overlap does not appear to be happening"
    )
    # Absolute sanity check independent of the sequential run: far below the
    # naive fully-serialized bound of num_chunks blocking waits.
    naive_bound = num_chunks * infer_delay + total_actions * tick_pause
    assert async_elapsed < naive_bound - 2 * infer_delay


# --- exactly one inference in flight --------------------------------------


async def test_exactly_one_background_inference_at_a_time():
    infer = ControllableInfer(n=3, dim=2)
    s = AsyncScheduler(infer, queue_threshold=2)

    first = asyncio.create_task(s.next_action())
    await infer.wait_until_entered()
    assert infer.concurrent == 1

    # A concurrent call while the first (blocking-path) inference is still
    # in flight must see `_inflight` already set and return None -- not
    # start a second, concurrent inference.
    concurrent_result = await s.next_action()
    assert concurrent_result is None
    assert infer.calls == 1

    infer.release()
    action = await first
    np.testing.assert_allclose(action, [101.0, 101.0])
    assert infer.calls == 1

    # The blocking-fill call above does not itself check the refill
    # threshold -- only a subsequent fast-path call (queue already
    # non-empty) does. This pops the 2nd of 3 actions, leaving qsize=1 <=
    # threshold=2, which fires inference #2 in the background.
    action = await s.next_action()
    np.testing.assert_allclose(action, [101.0, 101.0])
    await infer.wait_until_entered()
    assert infer.calls == 2
    assert infer.concurrent == 1

    # Rapid, repeated next_action() calls while #2 is in flight must all
    # either serve an already-queued action or return None -- never fire a
    # third inference while #2 is unresolved, however many times the
    # threshold condition gets re-checked. Each of these calls only ever
    # takes the fast (no internal await) path, so a spurious extra
    # `create_task` here would not actually start running until the event
    # loop gets a chance later -- checked again below, after that chance.
    for _ in range(5):
        await s.next_action()
    assert infer.calls == 2
    assert infer.max_concurrent == 1

    infer.release()
    await asyncio.sleep(0.01)  # let #2 (and only #2) settle
    # If the guard above had spuriously fired a 3rd inference (overwriting
    # `_inflight` and orphaning #2's task instead of skipping), it would
    # only actually start running once given this chance -- catch it here,
    # not just by its absence immediately after the rapid-call loop.
    assert infer.calls == 2
    assert infer.max_concurrent == 1


# --- None when starved, without blocking ----------------------------------


async def test_starved_returns_none_and_stays_responsive():
    infer = ControllableInfer(n=2, dim=2)
    s = AsyncScheduler(infer, queue_threshold=1)

    first = asyncio.create_task(s.next_action())
    await infer.wait_until_entered()
    infer.release()
    await first  # chunk of 2 merged; action #1 returned, 1 left queued

    action2 = await s.next_action()  # pops the last one; fires #2 in background
    assert action2 is not None
    await infer.wait_until_entered()  # confirm #2 has actually started

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())

    started = time.perf_counter()
    result = await s.next_action()  # queue empty, #2 in flight -> must not block
    elapsed = time.perf_counter() - started

    await asyncio.sleep(0.05)  # let the ticker accumulate while #2 is held open

    ticker_task.cancel()
    try:
        await ticker_task
    except asyncio.CancelledError:
        pass

    assert result is None
    assert elapsed < 0.02, (
        f"next_action() took {elapsed:.4f}s while starved -- it must return "
        "immediately instead of blocking on the in-flight inference"
    )
    assert ticks > 100, (
        f"event loop only ticked {ticks} times while a background inference was "
        "in flight -- something is blocking it"
    )

    infer.release()
    await asyncio.sleep(0.01)  # let #2 settle so it doesn't leak into other tests


# --- background failure surfaces exactly once ------------------------------


async def test_background_failure_surfaces_on_next_call_exactly_once():
    infer = RecordingInfer(n=5, dim=2, fail_after=1)
    s = AsyncScheduler(infer, queue_threshold=3)

    action = await s.next_action()  # call #1 succeeds (blocking)
    np.testing.assert_allclose(action, [101.0, 101.0])

    # The blocking-fill call above does not itself check the refill
    # threshold -- this fast-path call does: pops the 2nd action (qsize=3
    # <= threshold=3), firing #2 (which will fail) in the background.
    action = await s.next_action()
    np.testing.assert_allclose(action, [101.0, 101.0])

    await asyncio.sleep(0.02)  # let the failing background inference settle

    with pytest.raises(SchedulerError, match="exploded"):
        await s.next_action()
    assert infer.calls == 2

    # The error must not be raised again -- it was already delivered once.
    infer.fail_after = None
    action = await s.next_action()  # queue still has 3 actions left from chunk #1
    assert action is not None


async def test_background_empty_chunk_is_not_double_wrapped():
    # A malformed-chunk failure from a *background* inference must go
    # through _validate_and_merge's own SchedulerError untouched -- not get
    # a second "background inference failed: " prefix stapled on top, which
    # is reserved for exceptions _validate_and_merge never saw at all (a
    # bare exception raised by infer() itself).
    class EmptyThenRecording:
        def __init__(self):
            self.calls = 0

        async def __call__(self, rtc):
            self.calls += 1
            if self.calls == 1:
                return np.full((4, 2), 1.0, dtype=np.float32), np.full((4, 2), 1.0, dtype=np.float32)
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    infer = EmptyThenRecording()
    s = AsyncScheduler(infer, queue_threshold=3)

    await s.next_action()  # call #1 succeeds (blocking)
    await s.next_action()  # fast path fires call #2 (empty chunk) in the background
    await asyncio.sleep(0.02)

    with pytest.raises(SchedulerError) as exc_info:
        await s.next_action()
    message = str(exc_info.value)
    assert "empty" in message
    assert "background inference failed" not in message, (
        f"a SchedulerError from _validate_and_merge must not be re-wrapped: {message!r}"
    )


# --- close() ----------------------------------------------------------------


async def test_close_cancels_in_flight_inference():
    infer = ControllableInfer(n=4, dim=2)
    s = AsyncScheduler(infer, queue_threshold=3)

    task = asyncio.create_task(s.next_action())
    await infer.wait_until_entered()

    await s.close()

    assert infer.concurrent == 0
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_close_with_nothing_in_flight_is_safe():
    s = AsyncScheduler(RecordingInfer(), queue_threshold=1)
    await s.close()  # must not raise


async def test_sequential_scheduler_close_is_a_safe_no_op():
    # ChunkScheduler.close() is concrete on the ABC precisely so schedulers
    # that hold no background resources get it for free.
    s = SequentialScheduler(RecordingInfer())
    await s.next_action()
    await s.close()
    assert s.qsize() == 3  # close() must not clear the queue for this scheduler


# ---------------------------------------------------------------------------
# Starvation-risk warning: queue_threshold too low for observed latency.
#
# Real inference latency is only known once inference has actually run --
# `_observe_latency` below drives the same two steps `_infer_and_merge` does
# (record a sample, then run the check) for the precise, timing-independent
# boundary tests, and the warning is exercised once more through the real
# code path (a genuinely slow infer(), actually awaited) to prove the wiring
# from a completed background inference to the check is real.


def _observe_latency(scheduler, seconds):
    """What `_infer_and_merge` does after a completed inference."""
    scheduler._latencies.append(seconds)
    scheduler._maybe_warn_starvation_risk()
# ---------------------------------------------------------------------------

_SCHEDULER_LOGGER = "vla.controller.scheduler"


async def test_no_warning_when_threshold_meets_required_runway(caplog):
    # avg latency 0.3s at fps=4.0 -> 0.3 * 4 = 1.2 -> ceil = 2.
    # threshold=2 meets (is not below) the requirement -- no warning.
    s = AsyncScheduler(RecordingInfer(), queue_threshold=2, fps=4.0)
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        _observe_latency(s, 0.3)
        _observe_latency(s, 0.3)
    assert not caplog.records


async def test_warns_when_threshold_below_required_runway(caplog):
    s = AsyncScheduler(RecordingInfer(), queue_threshold=1, fps=4.0)
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        _observe_latency(s, 0.3)
        _observe_latency(s, 0.3)
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "queue_threshold=1" in message
    assert "0.3" in message
    assert "fps=4" in message
    assert "queue_threshold>=2" in message
    assert "hold position" in message
    assert "num_steps" in message or "denoising" in message


async def test_warning_uses_ceil_not_floor_at_the_boundary(caplog):
    # avg * fps = 1.2 exactly straddles 1 and 2. Floor/int-truncation would
    # compute required=1 (threshold=1 would then satisfy it -- no warning);
    # ceil correctly computes required=2 (threshold=1 does not satisfy it --
    # warning fires). This is the precise off-by-one this test exists to
    # catch: mutating math.ceil to int() or math.floor must fail it.
    s = AsyncScheduler(RecordingInfer(), queue_threshold=1, fps=4.0)
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        _observe_latency(s, 0.3)
        _observe_latency(s, 0.3)
    assert len(caplog.records) == 1


async def test_no_warning_when_threshold_exactly_meets_ceil_boundary(caplog):
    # The flip side of the same boundary: threshold == required (not <) must
    # not warn -- pins the strict "<" comparison against an "<=" mutation.
    s = AsyncScheduler(RecordingInfer(), queue_threshold=2, fps=4.0)
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        _observe_latency(s, 0.3)
        _observe_latency(s, 0.3)
    assert not caplog.records


async def test_warns_only_once_despite_many_observations(caplog):
    s = AsyncScheduler(RecordingInfer(), queue_threshold=1, fps=4.0)
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        for _ in range(10):
            _observe_latency(s, 0.3)
    assert len(caplog.records) == 1


async def test_no_warning_before_minimum_sample_count(caplog):
    # A single sample could be a cold-start outlier -- "a stable reading"
    # requires more than one observation before the check fires at all.
    s = AsyncScheduler(RecordingInfer(), queue_threshold=1, fps=4.0)
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        _observe_latency(s, 0.3)
    assert not caplog.records


async def test_warning_reflects_a_real_completed_background_inference(caplog):
    class SlowInfer:
        def __init__(self):
            self.calls = 0

        async def __call__(self, rtc):
            self.calls += 1
            await asyncio.sleep(0.05)
            base = np.full((1, 2), float(self.calls), dtype=np.float32)
            return base + 100.0, base

    # required = ceil(0.05 * 100) = 5, comfortably above threshold=0.
    s = AsyncScheduler(SlowInfer(), queue_threshold=0, fps=100.0)
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        await s.next_action()  # call #1 completes (blocking) -- 1 sample: gate not open yet
        assert not caplog.records
        await s.next_action()  # fast path: queue (n=1) now empty, fires call #2 in background
        await asyncio.sleep(0.2)  # let call #2 actually complete for real
    assert len(caplog.records) == 1


# ---------------------------------------------------------------------------
# actions_per_chunk: execute only the head of each chunk.
# ---------------------------------------------------------------------------


async def test_sequential_truncates_the_chunk_to_actions_per_chunk():
    infer = RecordingInfer(n=10)
    s = SequentialScheduler(infer, actions_per_chunk=3)
    await s.next_action()
    # 3 queued, 1 consumed. The 7 discarded rows must be gone, not merely
    # unread: a queue that still holds them would refill late and execute
    # actions inferred from a much older observation.
    assert s.qsize() == 2


async def test_truncation_makes_the_policy_be_called_more_often():
    """The entire point: shorter chunks mean re-observing sooner. Draining
    the same number of actions must cost proportionally more inferences."""
    infer = RecordingInfer(n=12)
    s = SequentialScheduler(infer, actions_per_chunk=4)
    for _ in range(12):
        await s.next_action()
    assert infer.calls == 3  # not 1, which is what the untruncated chunk gives


async def test_async_truncates_the_chunk_to_actions_per_chunk():
    infer = RecordingInfer(n=10)
    s = AsyncScheduler(infer, queue_threshold=0, actions_per_chunk=3)
    await s.next_action()
    assert s.qsize() == 2
    await s.close()


async def test_no_actions_per_chunk_executes_the_whole_chunk():
    # The default must not silently shorten an existing deployment's chunk.
    infer = RecordingInfer(n=10)
    s = SequentialScheduler(infer)
    await s.next_action()
    assert s.qsize() == 9


async def test_actions_per_chunk_larger_than_the_chunk_is_not_an_error():
    """A policy may return fewer rows than the operator budgeted for (a
    checkpoint swap, say). Slicing past the end is a no-op, and treating it
    as malformed would refuse a chunk that is perfectly executable."""
    infer = RecordingInfer(n=4)
    s = SequentialScheduler(infer, actions_per_chunk=50)
    await s.next_action()
    assert s.qsize() == 3


async def test_truncation_keeps_the_head_of_the_chunk_not_the_tail():
    """k=0 is the accurate end of the chunk; the tail is what truncation
    exists to throw away. Reversing the slice would still pass a qsize
    assertion while executing exactly the wrong actions."""

    class RampInfer:
        async def __call__(self, rtc):
            base = np.arange(10, dtype=np.float32).reshape(10, 1)
            return base + 100.0, base

    s = SequentialScheduler(RampInfer(), actions_per_chunk=3)
    first = await s.next_action()
    second = await s.next_action()
    np.testing.assert_allclose(first, [100.0])
    np.testing.assert_allclose(second, [101.0])


async def test_truncation_applies_to_the_raw_chunk_too():
    """`raw` feeds `get_left_over` (the RTC prefix). Truncating only the
    processed array would leave the two queues at different lengths."""
    infer = RecordingInfer(n=10)
    s = SequentialScheduler(infer, actions_per_chunk=3)
    await s.next_action()
    # Both arrays hold 3 rows; `get()` advances last_index without trimming.
    assert s._queue.original_queue.shape[0] == s._queue.queue.shape[0] == 3
