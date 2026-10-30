"""Smoothness gates over the real controller loop, run in virtual time.

Scenario tests are plain sync functions on purpose: under the repo's
`asyncio_mode = "auto"`, an `async def` test would run on pytest-asyncio's
real loop and silently stop being virtual. Each drives its own
`VirtualLoop` through `tests.smoothness.run_session` / `run_virtual`.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from tests.smoothness import VirtualLoop, run_virtual
from tests.smoothness import (
    PERIOD,
    STAMP_TOL,
    TAG,
    gap_jitter,
    knot_accel,
    landing_obs_age,
    lead_error,
    long_gaps,
    residual,
    residual_accel,
    run_session,
    seam_jump,
    seeded_jitter,
)


def test_virtual_sleep_takes_no_wall_time_and_advances_the_clock():
    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(3600.0)
        return loop.time()

    t0 = time.perf_counter()
    assert run_virtual(main()) == pytest.approx(3600.0)
    assert time.perf_counter() - t0 < 1.0


def test_concurrent_sleepers_wake_in_order_without_adding_up():
    async def main():
        loop = asyncio.get_running_loop()
        woke = []

        async def sleeper(d):
            await asyncio.sleep(d)
            woke.append((d, loop.time()))

        await asyncio.gather(sleeper(0.3), sleeper(0.1), sleeper(0.2))
        return woke, loop.time()

    woke, end = run_virtual(main())
    assert [d for d, _ in woke] == [0.1, 0.2, 0.3]
    for d, t in woke:
        assert t == pytest.approx(d)
    assert end == pytest.approx(0.3)


def test_a_deadlock_raises_instead_of_hanging():
    async def main():
        await asyncio.get_running_loop().create_future()  # nothing will ever set it

    with pytest.raises(RuntimeError, match="deadlock"):
        run_virtual(main())


def test_wait_for_times_out_in_virtual_time():
    async def main():
        never = asyncio.get_running_loop().create_future()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(never, 5.0)
        return asyncio.get_running_loop().time()

    assert run_virtual(main()) == pytest.approx(5.0)


def _assert_healthy(session):
    assert session.status["state"] == "running", session.status
    assert session.status["last_error"] == "", session.status
    # Knots are only recorded when streaming; setpoints only in setpoint mode.
    assert len(session.arm.knots) + len(session.arm.setpoints) > 50


def test_steady():
    s = run_session(latency_s=0.33)
    _assert_healthy(s)
    assert gap_jitter(s.arm.stream_points) <= STAMP_TOL
    assert long_gaps(s.arm.stream_points) == 0
    print(f"steady knot_accel baseline: {knot_accel(s):.4f} deg")  # for #4; no gate


def test_stream_lead_is_exactly_one_tick():
    # On main the first tick includes the blocking first inference and
    # overruns, so tick 1 runs at the same elapsed time; the monotonic guard
    # then pushes every later stamp a full period further ahead, for the
    # whole run. The arm trails the policy by an extra 100 ms at 10 fps.
    s = run_session(latency_s=0.33)
    _assert_healthy(s)
    assert lead_error(s.arm) <= STAMP_TOL, lead_error(s.arm)


def test_jittery_ticks():
    # A regression gate, not an xfail. On main, the extra period of lead (see
    # above) happens to absorb any sub-period jitter. After change 2 the stamps
    # follow the nominal grid, so jitter cannot reach them by construction.
    # Delays of at most 0.6 periods inside a tick whose other work takes no
    # virtual time never push a tick past its own deadline, so this scenario
    # never re-anchors. Don't change the seed to "add coverage": re-anchoring
    # has its own TickSchedule unit tests.
    s = run_session(latency_s=0.33, jitter=seeded_jitter(seed=7))
    _assert_healthy(s)
    assert gap_jitter(s.arm.stream_points) <= STAMP_TOL
    assert long_gaps(s.arm.stream_points) == 0


def test_slow_inference_lands_fresh_rows():
    latency = 1.23
    # Unblended rows: the TAG decode in landing_obs_age needs them.
    s = run_session(latency_s=latency, blend_s=0.0)
    _assert_healthy(s)
    assert landing_obs_age(s) <= latency + 2 * PERIOD


def test_starvation_holds_are_accounted_for():
    s = run_session(latency_s=5.5, duration_s=30.0)
    _assert_healthy(s)
    held = s.status["starved_ticks"] + s.status["reanchors"]
    assert long_gaps(s.arm.stream_points) <= held
    # (FakeArm itself asserts every stamp is strictly increasing.)


def test_seam_blend_softens_the_switch():
    hard = run_session(latency_s=1.23, offsets=(1.0, -1.0), blend_s=0.0)
    blended = run_session(latency_s=1.23, offsets=(1.0, -1.0), blend_s=0.3)
    _assert_healthy(hard)
    _assert_healthy(blended)
    # Consecutive chunks disagree by 2 deg; a hard switch puts that into one
    # second difference, and a 3-tick smoothstep spreads the 2-degree step over
    # 3 increments of 0.26/0.48/0.26 of it.
    assert seam_jump(blended) <= 0.5 * seam_jump(hard), (seam_jump(blended), seam_jump(hard))


def test_targets_land_on_the_reference_at_their_due_time():
    for arm_write in ("stream", "setpoint"):
        s = run_session(latency_s=0.33, blend_s=0.0, arm_write=arm_write)
        _assert_healthy(s)
        assert s.status["rebases"] == 1, s.status
        _, res = residual(s, after=s.policy.reply_times[1] + 3 * PERIOD)
        # Observations fall on the tick grid here, so samples land on rows and the
        # measured residual is ~2e-7. The bound is the worst case off-grid:
        # linear interpolation of the 4 s, 5-degree sinusoid errs by at most
        # h^2/8 * max|f''| = 0.1^2/8 * 5*(2*pi/4)^2 ~ 0.0154 deg. A target a tick
        # early or late is ~0.8 deg off.
        assert len(res) > 100
        assert np.abs(res).max() <= 0.016, (arm_write, np.abs(res).max())


def test_crossfade_adds_less_acceleration_than_the_motion_itself_has():
    fps, d = 30.0, 0.2
    # A B-tick smoothstep peaks at 6*d/B^2 deg/tick^2: 0.0053 at B = 15. The
    # reference's own peak is 5*(2*pi/4)^2 / fps^2 ~ 0.0137 at 30 fps.
    blend_s = 15 / fps
    kwargs = dict(fps=fps, latency_s=0.3, offsets=(d / 2, -d / 2), duration_s=20.0)
    soft = run_session(blend_s=blend_s, **kwargs)
    hard = run_session(blend_s=0.0, **kwargs)
    _assert_healthy(soft)
    _assert_healthy(hard)
    assert soft.status["regime"] == "overlap", soft.status
    # Past the rebased first chunk and the blend that supersedes it.
    after = soft.policy.reply_times[1] + blend_s + 3 / fps
    assert residual_accel(soft, after=after) <= 0.01
    assert residual_accel(hard, after=hard.policy.reply_times[1] + 3 / fps) >= 0.15


def test_overlap_keeps_up_below_the_regime_threshold():
    # One request in flight: landings 2L apart. 2*2.2 + 0.2 + 0.2 = 4.8 < 5.0.
    s = run_session(latency_s=2.2, duration_s=30.0)
    _assert_healthy(s)
    assert s.status["regime"] == "overlap"
    assert s.status["starved_ticks"] == 0


def test_past_half_the_chunk_switches_to_sequential():
    # 2*3.03 + 0.4 >= 5.0: overlap would starve every chunk here (measured: 49
    # starved ticks under the old single-L threshold), so it runs whole chunks.
    s = run_session(latency_s=3.03, duration_s=30.0)
    _assert_healthy(s)
    assert s.status["regime"] == "sequential"
    assert s.status["overlap_s"] == 0.0


def test_latency_above_the_chunk_runs_whole_chunks_from_their_start():
    s = run_session(latency_s=5.5, duration_s=40.0)
    _assert_healthy(s)
    assert s.status["regime"] == "sequential"
    assert s.status["overlap_s"] == 0.0
    # The fill's own latency sample already says sequential, so every chunk
    # after it is observed at rest and rebased whole.
    assert s.status["rebases"] >= 3
    # Each rebased chunk runs to its end: about CHUNK ticks each, minus the
    # couple of rows already behind the freeze point.
    assert len(s.arm.knots) >= 45 * (s.status["rebases"] - 1)
    # The TAG column names each knot's row. Every hold follows a chunk's last
    # row, and the next chunk resumes at its first rows: observed at rest, its
    # freeze point is row 0. (A position-continuity check across a hold needs a
    # policy that reacts to the observed state; ScriptedPolicy is open-loop in
    # time, so that property is pinned by the ReferenceBuffer unit tests.)
    tags = np.array([p[TAG] for _, p in s.arm.knots])
    stamps = np.array([stamp for _, stamp in s.arm.batch_stamps])
    holds = np.flatnonzero(np.diff(stamps) > 2 * PERIOD)
    assert len(holds) >= 2
    for i in holds:
        assert tags[i] % 1000 >= 47, (i, tags[i])
        assert tags[i + 1] % 1000 <= 2, (i, tags[i + 1])


def test_a_hung_policy_fails_after_inference_timeout_s():
    s = run_session(latency_s=1e6, duration_s=10.0, inference_timeout_s=5.0)
    assert s.status["state"] == "error"
    assert "did not land" in s.status["last_error"]
