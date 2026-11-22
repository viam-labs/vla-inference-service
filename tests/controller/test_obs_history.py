"""Observation history: a burst of n_obs_steps frames, 1/fps apart, oldest first."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import FakeArm, FakeCamera
from tests.smoothness import run_session, run_virtual
from vla.controller.gripper import make_gripper_adapter
from vla.controller.inference_loop import InferenceLoop
from vla.controller.observation import ObservationBuilder, ObservationError
from vla.controller.reference_buffer import ReferenceBuffer

FPS = 20.0
KEY = "observation.images.top"


class CountingArm(FakeArm):
    """Joint 0 reads 0, 1, 2, ... so frame order is visible in the state."""

    async def get_joint_positions(self, **kwargs):
        self.positions[0] = float(getattr(self, "n", 0))
        self.n = getattr(self, "n", 0) + 1
        return await super().get_joint_positions(**kwargs)


class FailsOnSecondRead(FakeCamera):
    async def get_images(self, *a, **kw):
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls == 2:
            raise RuntimeError("camera dropped")
        return await super().get_images(*a, **kw)


def _builder(clock, camera=None, n=2):
    return ObservationBuilder(
        cameras={KEY: camera or FakeCamera()},
        arm=CountingArm(positions=[0.0] * 6),
        gripper=make_gripper_adapter({"type": "none"}, {}),
        state_joint_indices=[0, 1, 2],
        state_units="degrees",
        image_sizes={KEY: (64, 64)},
        clock=clock,
        n_obs_steps=n,
        fps=FPS,
    )


def _build(camera=None, n=2):
    async def main():
        loop = asyncio.get_running_loop()
        return await _builder(loop.time, camera, n).build()

    async def inline(fn, /, *a, **kw):
        return fn(*a, **kw)

    with patch("vla.controller.observation.to_thread", inline):
        return run_virtual(main())


def test_single_step_has_no_history():
    assert _build(n=1).history == ()


def test_burst_is_spaced_oldest_first_and_newest_is_t_obs():
    obs = _build(n=3)
    assert [f.state[0] for f in obs.history] == [0.0, 1.0, 2.0]
    t = [f.t_obs for f in obs.history]
    np.testing.assert_allclose(np.diff(t), 1.0 / FPS)
    assert obs.t_obs == t[-1] and obs.state[0] == 2.0
    assert obs.duration_s == pytest.approx(2.0 / FPS)


def test_failed_read_mid_burst_is_an_observation_error():
    with pytest.raises(ObservationError, match="camera"):
        _build(FailsOnSecondRead())


def test_capture_time_counts_toward_expected_latency_and_regime():
    def loop(capture_s):
        lp = InferenceLoop(
            None, ReferenceBuffer(10.0, blend_s=0.2, interp_cols=1, consume=False, smooth_rows=0),
            chunk_s=1.0, blend_s=0.2, margin_s=0.2, timeout_s=5.0, clock=lambda: 0.0,
            capture_s=capture_s,
        )
        lp.latencies.append(0.1)
        return lp

    assert loop(0.0).regime == "overlap"
    burst = loop(0.2)
    assert burst.expected_latency == pytest.approx(0.3)
    assert burst.regime == "sequential"  # 2 * 0.3 + 0.4 >= 1.0


def _lists_of(req, n):
    assert len(req["images"][KEY]) == n and len(req["state"]) == n
    assert all(isinstance(p, dict) for p in req["images"][KEY])


def test_end_to_end_request_carries_history():
    s = run_session(latency_s=0.2, duration_s=6.0, n_obs_steps=2, fps=FPS)
    assert s.policy.requests
    for req in s.policy.requests:
        _lists_of(req, 2)


def test_end_to_end_single_step_request_unchanged():
    s = run_session(latency_s=0.2, duration_s=4.0)
    for req in s.policy.requests:
        assert isinstance(req["images"][KEY], dict) and isinstance(req["state"], dict)
