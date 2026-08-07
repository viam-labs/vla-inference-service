import numpy as np
import pytest

from vla.controller.action_queue import ActionQueueError
from vla.controller.scheduler import ChunkScheduler, SchedulerError, SequentialScheduler


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


async def test_reset_clears_the_queue():
    infer = RecordingInfer(n=5)
    s = SequentialScheduler(infer)
    await s.next_action()
    s.reset()
    await s.next_action()
    assert infer.calls == 2


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

        def reset(self) -> None:
            pass

        # qsize() deliberately omitted.

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_implementing_every_method_can_be_instantiated():
    class Complete(ChunkScheduler):
        async def next_action(self):
            return None

        def reset(self) -> None:
            pass

        def qsize(self) -> int:
            return 0

    Complete()  # must not raise


def test_sequential_scheduler_is_a_chunk_scheduler():
    assert isinstance(SequentialScheduler(RecordingInfer()), ChunkScheduler)
