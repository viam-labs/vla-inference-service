"""InferenceLoop: one request in flight, fired on the buffer's remaining time."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from vla.controller.inference_loop import InferenceError, InferenceLoop
from vla.controller.reference_buffer import ReferenceBuffer

pytestmark = pytest.mark.asyncio


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _loop(infer, buf, clock, *, timeout_s=5.0):
    # chunk_s 1.0, blend 0.2, margin 0.2: overlap while 2 * expected latency < 0.6.
    return InferenceLoop(
        infer, buf, chunk_s=1.0, blend_s=0.2, margin_s=0.2, timeout_s=timeout_s, clock=clock
    )


def _timed_infer(clock, latency, calls):
    async def infer():
        calls.append(clock())
        t_obs = clock()
        clock.t += latency
        return np.zeros((10, 2)), t_obs

    return infer


async def _settle():
    for _ in range(3):
        await asyncio.sleep(0)


async def test_wait_fires_and_lands_and_records_observation_to_landing():
    clock, calls, buf = Clock(), [], ReferenceBuffer(10.0)
    loop = _loop(_timed_infer(clock, 0.3, calls), buf, clock)
    await loop.wait()
    assert len(calls) == 1
    assert not buf.exhausted(clock())
    assert list(loop.latencies) == [pytest.approx(0.3)]


async def test_prior_is_the_chunk_duration_until_the_first_sample():
    clock, calls, buf = Clock(), [], ReferenceBuffer(10.0)
    loop = _loop(_timed_infer(clock, 0.1, calls), buf, clock)
    assert loop.expected_latency == 1.0
    await loop.wait()
    assert loop.expected_latency == pytest.approx(0.1)


async def test_one_slow_sample_already_selects_sequential():
    clock, calls, buf = Clock(), [], ReferenceBuffer(10.0)
    loop = _loop(_timed_infer(clock, 0.7, calls), buf, clock)
    await loop.wait()
    assert loop.regime == "sequential"


async def test_expected_latency_is_the_max_of_the_last_five():
    clock, buf = Clock(), ReferenceBuffer(10.0)
    latencies = iter([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])

    async def infer():
        t_obs = clock()
        clock.t += next(latencies)
        return np.zeros((10, 2)), t_obs

    loop = _loop(infer, buf, clock)
    for _ in range(6):
        await loop.wait()
    assert loop.expected_latency == pytest.approx(0.1)  # the 0.5 has aged out
    await loop.wait()
    assert loop.expected_latency == pytest.approx(0.2)


async def test_poll_fires_on_the_plan_left_after_the_last_sample():
    clock, calls, buf = Clock(), [], ReferenceBuffer(10.0)
    loop = _loop(_timed_infer(clock, 0.1, calls), buf, clock)
    await loop.wait()
    await loop.wait()
    buf.sample(clock())  # pins the 10-row chunk: horizon = last_sampled + 0.9
    assert loop.regime == "overlap"
    horizon = buf.horizon()
    # Remaining time is measured from the last sample, which runs a lead ahead
    # of the clock; the clock itself is irrelevant here.
    buf.sample(horizon - 0.6)  # 0.6 left >= 0.1 + 0.2 + 0.2
    loop.poll()
    await _settle()
    assert len(calls) == 2
    buf.sample(horizon - 0.4)  # 0.4 left < 0.5
    loop.poll()
    await _settle()
    assert len(calls) == 3


async def test_sequential_regime_never_fires_from_poll():
    clock, calls, buf = Clock(), [], ReferenceBuffer(10.0)
    loop = _loop(_timed_infer(clock, 0.7, calls), buf, clock)  # 2*0.7 + 0.4 >= 1.0
    await loop.wait()
    await loop.wait()
    assert loop.regime == "sequential"
    buf.sample(clock())
    clock.t = buf.horizon()  # nothing left at all
    loop.poll()
    await _settle()
    assert len(calls) == 2


async def test_wait_times_out_on_a_hung_policy():
    never = asyncio.Event()

    async def infer():
        await never.wait()

    loop = _loop(infer, ReferenceBuffer(10.0), Clock(), timeout_s=0.01)
    with pytest.raises(InferenceError, match="did not land"):
        await loop.wait()
    assert loop._inflight is None


async def test_a_failure_surfaces_once_wrapped():
    async def infer():
        raise RuntimeError("boom")

    loop = _loop(infer, ReferenceBuffer(10.0), Clock())
    with pytest.raises(InferenceError, match="boom"):
        await loop.wait()
    loop.poll()  # already raised: does not raise again


async def test_a_background_failure_surfaces_on_the_next_poll():
    clock, calls, buf = Clock(), [], ReferenceBuffer(10.0)
    good = _timed_infer(clock, 0.1, calls)
    fail = False

    async def infer():
        if fail:
            raise RuntimeError("late boom")
        return await good()

    loop = _loop(infer, buf, clock)
    await loop.wait()
    await loop.wait()
    buf.sample(clock())
    fail = True
    buf.sample(buf.horizon())  # nothing left after this sample
    loop.poll()  # fires
    await _settle()
    with pytest.raises(InferenceError, match="late boom"):
        loop.poll()


async def test_close_cancels_the_inflight_request():
    started = asyncio.Event()

    async def infer():
        started.set()
        await asyncio.Event().wait()

    loop = _loop(infer, ReferenceBuffer(10.0), Clock())
    waiter = asyncio.create_task(loop.wait())
    await started.wait()
    task = loop._inflight
    waiter.cancel()
    await loop.close()
    assert task.cancelled()
    assert loop._inflight is None
    with pytest.raises(asyncio.CancelledError):
        await waiter


# The latency window caps at the most recent 50 entries, so avg_latency_s
# never becomes a lifetime average (moved from the controller's own deque).


async def test_latency_window_keeps_only_the_most_recent_50():
    loop = _loop(None, ReferenceBuffer(10.0), Clock())
    for i in range(60):
        loop.latencies.append(float(i))
    assert len(loop.latencies) == 50
    assert loop.latencies[0] == 10.0  # the first 10 (0..9) were pushed out
    assert loop.latencies[-1] == 59.0


async def test_latency_window_does_not_trim_under_50():
    loop = _loop(None, ReferenceBuffer(10.0), Clock())
    for i in range(10):
        loop.latencies.append(float(i))
    assert list(loop.latencies) == [float(i) for i in range(10)]
