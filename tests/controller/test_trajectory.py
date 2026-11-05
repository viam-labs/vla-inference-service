import asyncio
import random
from datetime import timedelta

import pytest
from grpclib.exceptions import GRPCError

from tests.fakes import FakeArm, UnstreamableArm
from vla.controller.trajectory import ArmStream, cubic_tick, hermite, interpolate, monotone_tangents


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
# Cubic segment
# ---------------------------------------------------------------------------


def test_hermite_with_chord_tangents_is_linear():
    prev, target = [0.0, 10.0], [4.0, 30.0]
    d = [t - p for p, t in zip(prev, target)]
    got = hermite(prev, target, 4, d, d)
    want = interpolate(prev, target, 4)
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=1e-12)


def test_hermite_last_point_is_target_exactly():
    assert hermite([0.29], [0.87], 3, [0.0], [0.5])[-1] == [0.87]


def test_hermite_validates_like_interpolate():
    with pytest.raises(ValueError):
        hermite([0.0], [1.0], 0, [0.0], [0.0])
    with pytest.raises(ValueError):
        hermite([0.0], [1.0, 2.0], 2, [0.0], [0.0])


def test_monotone_tangents_zeroes_a_tangent_against_the_chord():
    m0, m1 = monotone_tangents([-1.0], [1.0], [1.0])
    assert m0 == [0.0] and m1 == [1.0]


def test_monotone_tangents_zero_chord_zeroes_both():
    assert monotone_tangents([2.0], [3.0], [0.0]) == ([0.0], [0.0])


def test_monotone_tangents_negligible_chord_is_zero_not_nan():
    assert monotone_tangents([1.0], [5e-324], [5e-324]) == ([0.0], [0.0])


def test_monotone_tangents_rescales_onto_radius_three():
    m0, m1 = monotone_tangents([5.0], [1.0], [1.0])
    assert (m0[0] ** 2 + m1[0] ** 2) == pytest.approx(9.0)
    assert m0[0] / m1[0] == pytest.approx(5.0)


def test_cubic_tick_points_stay_inside_the_chord():
    rng = random.Random(4)
    for _ in range(2000):
        dof = 3
        prev = [rng.uniform(-10, 10) for _ in range(dof)]
        target = [p + rng.choice([0.0, rng.uniform(-5, 5)]) for p in prev]
        tangent = [rng.uniform(-20, 20) for _ in range(dof)]
        points, _ = cubic_tick(prev, target, rng.randint(1, 10), tangent)
        for pt in points:
            for j in range(dof):
                lo, hi = min(prev[j], target[j]), max(prev[j], target[j])
                assert lo - 1e-9 <= pt[j] <= hi + 1e-9


def test_cubic_tick_on_a_ramp_matches_linear_after_the_first_tick():
    knots = [[0.5 * i, -2.0 * i] for i in range(6)]
    tangent = None
    for i, (prev, target) in enumerate(zip(knots, knots[1:])):
        points, tangent = cubic_tick(prev, target, 5, tangent)
        if i == 0:
            continue  # starts from rest: not linear
        for g, w in zip(points, interpolate(prev, target, 5)):
            assert g == pytest.approx(w, abs=1e-9)


def test_cubic_tick_returns_the_limited_end_tangent():
    # Carried tangent 10 against chord 1: alpha=10, beta=1, rescaled onto r=3.
    _, tangent = cubic_tick([0.0], [1.0], 5, [10.0])
    assert tangent[0] == pytest.approx(3.0 / (101 ** 0.5))


def test_cubic_tick_none_tangent_starts_from_rest():
    points, tangent = cubic_tick([0.0], [1.0], 4, None)
    assert points[0][0] < 0.25  # slower than linear's first step
    assert tangent == [1.0]


def test_cubic_tick_with_one_point_is_the_target():
    points, _ = cubic_tick([0.0, 1.0], [2.0, 3.0], 1, [5.0, 5.0])
    assert points == [[2.0, 3.0]]


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
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([0.0, 0.0])
    await stream.send([3.0, 30.0], 0.0)
    await _wait_for_points(arm, 4)

    n = round(30.0 / 10.0)
    # The rest point and the first tick's motion points must ship in the SAME
    # batch, not two: whichever point a driver anchors its wall clock to at
    # stream start, this keeps it within one transport hop of `_t0`.
    assert len(arm.stream_batches) == 1
    assert len(arm.stream_batches[0]) == 1 + n

    assert arm.stream_points[0].time == timedelta(0)
    assert arm.stream_points[0].positions == [0.0, 0.0]

    motion = arm.stream_points[1:]
    assert [p.positions for p in motion] == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    times = [p.time.total_seconds() for p in motion]
    # timedelta holds microseconds, each stamp rounded on its own: ~1us tolerance.
    assert abs((times[1] - times[0]) - 1 / 30.0) < 1e-6
    assert abs((times[2] - times[1]) - 1 / 30.0) < 1e-6
    assert abs(times[0] - (1 / 10.0 + 1 / 30.0)) < 1e-6
    await stream.close()


async def test_origin_is_the_first_sends_nominal_time():
    """t0 is the first send's nominal tick time, not anything before it.

    A real run blocks on the first inference between start() and the first
    send(); an origin taken earlier would carry that delay as dead time in
    every stamp, past due on arrival.
    """
    arm = FakeArm(positions=[0.0] * 2)
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([0.0, 0.0])
    await stream.send([3.0, 30.0], 0.6)  # first tick's nominal time, after a slow inference
    await _wait_for_points(arm, 4)

    assert arm.stream_points[0].time == timedelta(0)
    first_motion = arm.stream_points[1]
    assert abs(first_motion.time.total_seconds() - (1 / 10.0 + 1 / 30.0)) < 1e-6
    await stream.close()


async def test_send_spacing_is_uniform_across_a_tick_boundary():
    """n = round(stream_hz / fps) only tiles the tick if the step is
    1/(n*fps), not 1/stream_hz: at 100/30 (n=3) that would leave gaps of
    10, 10, 13.3ms instead of three even ~11.1ms gaps."""
    arm = FakeArm(positions=[0.0] * 2)
    fps, stream_hz = 30.0, 100.0
    stream = ArmStream(arm, fps=fps, stream_hz=stream_hz, extra={})
    n = round(stream_hz / fps)
    assert n == 3
    step = 1.0 / (n * fps)

    await stream.start([0.0, 0.0])
    await stream.send([3.0, 3.0], 0.0)
    await stream.send([6.0, 6.0], 1.0 / fps)
    await _wait_for_points(arm, 1 + 2 * n)

    motion_times = [p.time.total_seconds() for p in arm.stream_points[1:]]
    diffs = [b - a for a, b in zip(motion_times, motion_times[1:])]
    assert len(diffs) == 2 * n - 1
    for d in diffs:
        assert abs(d - step) < 1e-6
    await stream.close()


async def test_a_late_send_inside_the_lead_keeps_uniform_spacing():
    """The point of nominal stamping: when a send happens no longer changes
    its stamps, so tick jitter never reaches the arm."""
    arm = FakeArm(positions=[0.0] * 2)
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([0.0, 0.0])
    await stream.send([3.0, 3.0], 0.0)
    await asyncio.sleep(0.05)  # deliberate real sleep: the late send under test; this tick's send runs 50 ms late in real time
    await stream.send([6.0, 6.0], 0.1)
    await _wait_for_points(arm, 7)
    times = [p.time.total_seconds() for p in arm.stream_points[1:]]
    for a, b in zip(times, times[1:]):
        assert abs((b - a) - 1 / 30.0) < 1e-6
    await stream.close()


async def test_send_monotonic_guard_survives_a_rewound_nominal_time():
    arm = FakeArm(positions=[0.0] * 2)
    stream = ArmStream(arm, fps=10.0, stream_hz=30.0, extra={})
    await stream.start([0.0, 0.0])
    await stream.send([3.0, 30.0], 0.0)
    await stream.send([6.0, 60.0], -100.0)  # rewind
    await _wait_for_points(arm, 7)
    times = [p.time for p in arm.stream_points]
    assert all(a < b for a, b in zip(times, times[1:]))
    await stream.close()


async def test_cubic_stream_first_send_starts_from_rest_and_carries_the_chord():
    arm = FakeArm(positions=[0.0])
    stream = ArmStream(arm, fps=10.0, stream_hz=50.0, extra={}, interp="cubic")
    await stream.start([0.0])
    await stream.send([1.0], 0.0)
    await _wait_for_points(arm, 6)
    first = arm.stream_points[1].positions[0]
    assert 0.0 < first < 0.2  # linear would be exactly 0.2
    assert arm.stream_points[-1].positions == [1.0]
    await stream.close()


async def test_cubic_stream_contiguous_send_carries_the_tangent():
    arm = FakeArm(positions=[0.0])
    stream = ArmStream(arm, fps=10.0, stream_hz=50.0, extra={}, interp="cubic")
    await stream.start([0.0])
    await stream.send([1.0], 0.0)
    await stream.send([2.0], 0.1)  # constant velocity, contiguous: linear from here
    await _wait_for_points(arm, 11)
    second = [p.positions[0] for p in arm.stream_points[6:]]
    assert second == pytest.approx([1.2, 1.4, 1.6, 1.8, 2.0], abs=1e-9)
    await stream.close()


async def test_cubic_stream_resets_the_tangent_after_a_hold():
    arm = FakeArm(positions=[0.0])
    stream = ArmStream(arm, fps=10.0, stream_hz=50.0, extra={}, interp="cubic")
    await stream.start([0.0])
    await stream.send([1.0], 0.0)
    await stream.send([2.0], 0.5)  # four ticks later: the arm held at 1.0
    await _wait_for_points(arm, 11)
    # From rest: 1.072. A wrongly carried tangent would give linear's 1.2.
    assert arm.stream_points[6].positions[0] == pytest.approx(1.072)
    await stream.close()


async def test_cubic_stream_rewound_nominal_counts_as_contiguous():
    arm = FakeArm(positions=[0.0])
    stream = ArmStream(arm, fps=10.0, stream_hz=50.0, extra={}, interp="cubic")
    await stream.start([0.0])
    await stream.send([1.0], 0.0)
    await stream.send([2.0], -100.0)  # rewind: the max() guard stamps it one step on
    await _wait_for_points(arm, 11)
    second = [p.positions[0] for p in arm.stream_points[6:]]
    assert second == pytest.approx([1.2, 1.4, 1.6, 1.8, 2.0], abs=1e-9)
    await stream.close()


async def test_cubic_stream_stamps_match_linear():
    stamps = {}
    for interp in ("linear", "cubic"):
        arm = FakeArm(positions=[0.0])
        stream = ArmStream(arm, fps=10.0, stream_hz=50.0, extra={}, interp=interp)
        await stream.start([0.0])
        await stream.send([1.0], 0.0)
        await stream.send([3.0], 0.1)
        await _wait_for_points(arm, 11)
        stamps[interp] = [p.time for p in arm.stream_points]
        await stream.close()
    assert stamps["linear"] == stamps["cubic"]


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
    await stream.send([1.0, 1.0], 0.0)  # first batch: 1 rest point + 3 motion points >= 3
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
