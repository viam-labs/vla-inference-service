"""Smoothness gates over the real controller loop, run in virtual time.

Scenario tests are plain sync functions on purpose: under the repo's
`asyncio_mode = "auto"`, an `async def` test would run on pytest-asyncio's
real loop and silently stop being virtual. Each drives its own
`VirtualLoop` through `tests.smoothness.run_session` / `run_virtual`.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.smoothness import VirtualLoop, run_virtual
from tests.smoothness import (
    PERIOD,
    STAMP_TOL,
    gap_jitter,
    knot_accel,
    landing_obs_age,
    lead_error,
    long_gaps,
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
    assert len(session.arm.knots) > 50


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
    s = run_session(latency_s=latency)
    _assert_healthy(s)
    assert landing_obs_age(s) <= latency + 2 * PERIOD


def test_starvation_holds_are_accounted_for():
    s = run_session(latency_s=5.5, duration_s=30.0)
    _assert_healthy(s)
    held = s.status["starved_ticks"] + s.status["reanchors"]
    assert long_gaps(s.arm.stream_points) <= held
    # (FakeArm itself asserts every stamp is strictly increasing.)


def test_seam_blend_softens_the_switch():
    hard = run_session(latency_s=1.23, offsets=(1.0, -1.0))
    blended = run_session(latency_s=1.23, offsets=(1.0, -1.0), blend_ticks=3)
    _assert_healthy(hard)
    _assert_healthy(blended)
    # Consecutive chunks disagree by 2 deg; a hard switch puts that into one
    # second difference, and a 3-tick linear crossfade spreads it over 4 steps
    # (~0.25x). Measured while planning: 10.1 vs 2.58, before the first landing
    # was excluded from seam_jump.
    assert seam_jump(blended) <= 0.5 * seam_jump(hard), (seam_jump(blended), seam_jump(hard))


def test_aligned_starves_past_half_the_chunk():
    # Each landing leaves CHUNK - L fresh rows and the next lands L ticks
    # later, so aligned keeps up only while L <= CHUNK / 2 (25 ticks here).
    under = run_session(latency_s=1.97, duration_s=20.0, starvation_grace_ticks=3)
    _assert_healthy(under)
    assert under.status["starved_ticks"] == 0
    over = run_session(latency_s=3.03, duration_s=20.0, starvation_grace_ticks=3)
    assert over.status["state"] == "error", over.status
    assert "starved" in over.status["last_error"]
