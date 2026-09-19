import asyncio
from datetime import timedelta

import pytest
from grpclib.exceptions import GRPCError

from tests.fakes import FakeArm, UnstreamableArm
from vla.controller.trajectory import ArmStream, interpolate


def test_n_equals_1_returns_target_only():
    assert interpolate([0, 0], [4, 30], 1) == [[4, 30]]


def test_n_equals_4_linear_spacing():
    assert interpolate([0, 10], [4, 30], 4) == [[1, 15], [2, 20], [3, 25], [4, 30]]


def test_length_equals_n():
    assert len(interpolate([0, 0, 0], [1, 2, 3], 7)) == 7


def test_last_point_equals_target_exactly():
    # 0.29 -> 0.87 over 3 steps is a case where naive arithmetic drifts:
    # 0.29 + (0.87 - 0.29) * 3 / 3 == 0.8700000000000001, not 0.87. The last
    # point must come from `list(target)`, not accumulation.
    prev = [0.29]
    target = [0.87]
    points = interpolate(prev, target, 3)
    assert points[-1] == list(target)


def test_raises_on_n_less_than_1():
    with pytest.raises(ValueError):
        interpolate([0, 0], [1, 1], 0)


def test_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        interpolate([0, 0], [1, 1, 1], 3)


# ---------------------------------------------------------------------------
# ArmStream
# ---------------------------------------------------------------------------


async def _drain(stream, timeout=2.0):
    """Give the background consumer task a chance to run to completion."""
    deadline = asyncio.get_event_loop().time() + timeout
    while stream._task is not None and not stream._task.done():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("background stream task never finished")
        await asyncio.sleep(0.001)


async def _wait_for_points(arm, n, timeout=2.0):
    """`queue.put_nowait` never yields, so the background consumer task needs
    an explicit chance to run before a just-`send()`'d batch shows up in
    `arm.stream_points`."""
    deadline = asyncio.get_event_loop().time() + timeout
    while len(arm.stream_points) < n:
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(f"arm never received {n} points; got {len(arm.stream_points)}")
        await asyncio.sleep(0.001)


async def test_start_enqueues_nothing():
    arm = FakeArm(positions=[0.0] * 6)
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    await asyncio.sleep(0.01)  # let the background task run if it's going to
    assert arm.stream_points == []
    await stream.close()


async def test_first_send_carries_the_rest_point_and_the_first_ticks_motion():
    arm = FakeArm(positions=[0.0] * 2)
    box = [0.0]
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={}, clock=lambda: box[0])
    await stream.start([0.0, 0.0])
    await stream.send([3.0, 30.0])
    await _wait_for_points(arm, 4)

    n = round(30.0 / 10.0)
    # The property that fixes C1: the rest point and the first tick's motion
    # points must ship in the SAME batch, not two -- whichever point a driver
    # anchors its wall clock to at stream start, this keeps it within one
    # transport hop of `_t0`.
    assert len(arm.stream_batches) == 1
    assert len(arm.stream_batches[0]) == 1 + n

    assert arm.stream_points[0].time == timedelta(0)
    assert arm.stream_points[0].positions == [0.0, 0.0]

    motion = arm.stream_points[1:]
    assert [p.positions for p in motion] == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    times = [p.time.total_seconds() for p in motion]
    assert times[0] < times[1] < times[2]
    # `timedelta` only holds microsecond precision, and each point is rounded
    # to it independently, so the achievable tolerance on a *difference* of
    # two such points is ~1us, not float epsilon.
    assert abs((times[1] - times[0]) - 1 / 30.0) < 1e-6
    assert abs((times[2] - times[1]) - 1 / 30.0) < 1e-6
    assert times[0] >= 1 / 10.0
    await stream.close()


async def test_origin_is_anchored_at_first_send_not_start():
    """Regression: the clock must start at the first send(), not at start().

    A real run blocks on the first inference between start() and the first
    send() -- if _t0 were set in start(), every stamp in the run would carry
    that whole delay as dead time, past due the moment it reaches the driver,
    collapsing the stream back into bursts at fps.
    """
    arm = FakeArm(positions=[0.0] * 2)
    box = [0.0]
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={}, clock=lambda: box[0])
    await stream.start([0.0, 0.0])
    box[0] = 0.6  # simulate a slow first inference elapsing before the first send
    await stream.send([3.0, 30.0])
    await _wait_for_points(arm, 4)

    assert arm.stream_points[0].time == timedelta(0)
    assert arm.stream_points[0].positions == [0.0, 0.0]
    step = 1.0 / 30.0
    first_motion = arm.stream_points[1]
    assert abs(first_motion.time.total_seconds() - (1 / 10.0 + step)) < 1e-6
    await stream.close()


async def test_send_spacing_is_uniform_across_a_tick_boundary():
    """n = round(stream_hz / fps) only tiles the tick if the step is
    1/(n*fps), not 1/stream_hz: at 100/30 (n=3) that would leave gaps of
    10, 10, 13.3ms instead of three even ~11.1ms gaps."""
    arm = FakeArm(positions=[0.0] * 2)
    box = [0.0]
    fps, stream_hz = 30.0, 100.0
    stream = ArmStream(arm, fps=fps, stream_hz=stream_hz, extra={}, clock=lambda: box[0])
    n = round(stream_hz / fps)
    assert n == 3
    step = 1.0 / (n * fps)

    await stream.start([0.0, 0.0])
    await stream.send([3.0, 3.0])
    box[0] += 1.0 / fps  # simulate exactly one tick of real elapsed time
    await stream.send([6.0, 6.0])
    await _wait_for_points(arm, 1 + 2 * n)

    motion_times = [p.time.total_seconds() for p in arm.stream_points[1:]]
    diffs = [b - a for a, b in zip(motion_times, motion_times[1:])]
    assert len(diffs) == 2 * n - 1  # within both ticks and across their boundary
    for d in diffs:
        assert abs(d - step) < 1e-6
    await stream.close()


async def test_send_monotonic_guard_survives_rewound_clock():
    arm = FakeArm(positions=[0.0] * 2)
    box = [0.0]
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={}, clock=lambda: box[0])
    await stream.start([0.0, 0.0])
    await stream.send([3.0, 30.0])
    box[0] = -100.0  # rewind
    await stream.send([6.0, 60.0])
    await _wait_for_points(arm, 7)
    times = [p.time for p in arm.stream_points]
    assert all(a < b for a, b in zip(times, times[1:]))
    await stream.close()


async def test_unstreamable_arm_check_raises_runtime_error_mentioning_arm_write():
    arm = UnstreamableArm(positions=[0.0] * 6)
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([0.0] * 6)  # must not raise: nothing is sent or awaited yet
    await _drain(stream)
    with pytest.raises(RuntimeError, match="arm_write"):
        stream.check()


async def test_check_raises_stream_fault_once_then_closed_error():
    arm = FakeArm(positions=[0.0] * 2)
    arm.fail_stream_after_points = 3
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([0.0, 0.0])
    await stream.send([1.0, 1.0])  # first batch: 1 rest point + 3 motion points >= 3
    await _drain(stream)
    with pytest.raises(GRPCError, match="arm fault"):
        stream.check()
    with pytest.raises(RuntimeError, match="closed the trajectory stream"):
        stream.check()


async def test_close_is_idempotent_and_closes_the_fake():
    arm = FakeArm(positions=[0.0] * 2)
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([0.0, 0.0])
    await stream.close()
    assert arm.stream_closed is True
    await stream.close()  # must not raise


async def test_close_without_start_is_a_noop():
    arm = FakeArm(positions=[0.0] * 2)
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.close()  # must not raise
