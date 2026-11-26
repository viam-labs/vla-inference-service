"""Tests for action_space="ee-pose" limbs: state read, teleop streaming, clamps, grips."""

from __future__ import annotations

import json

import numpy as np
import pytest
from viam.proto.common import Pose

from tests.fakes import FakeArm, FakeDoCommandGripper, FakeMotion
from vla.controller.config import LimbConfig, SafetyConfig
from vla.controller.ee_pose import Limbs
from vla.controller.gripper import make_gripper_adapter
from vla.controller.pose import state_delta, state_rotation
from vla.controller.observation import pose_state_from_proto
from vla.controller.safety import CartesianSafetyLayer, SafetyError

pytestmark = pytest.mark.asyncio

DOWN = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)


def _limbs(safety=None, positions=(30.0, 70.0)):
    cfgs = [
        LimbConfig("left", "la", "lg", {"type": "do_command", "name": "lgrip", "open_value": 100.0, "closed_value": 0.0}),
        LimbConfig("right", "ra", "rg", {"type": "do_command", "name": "rgrip", "open_value": 100.0, "closed_value": 0.0}),
    ]
    motion = FakeMotion({"lg": Pose(x=300, y=300, z=400, o_x=0.3, o_y=-0.2, o_z=-0.93, theta=25.0), "rg": Pose(x=300, y=-300, z=400, **DOWN)})
    grips = {"lgrip": FakeDoCommandGripper(position=positions[0]), "rgrip": FakeDoCommandGripper(position=positions[1])}
    arms = {"la": FakeArm(positions=[0.0] * 6), "ra": FakeArm(positions=[0.0] * 6)}
    deps = {**grips, **arms}
    limbs = Limbs(
        cfgs,
        motion=motion,
        motion_name="builtin",
        arms=arms,
        grippers=[make_gripper_adapter(c.gripper, deps) for c in cfgs],
        safety=CartesianSafetyLayer(safety or SafetyConfig(max_start_delta_mm=50.0)),
        start_limit_mm=(safety or SafetyConfig(max_start_delta_mm=50.0)).max_start_delta_mm,
        lead_limit_mm=(safety or SafetyConfig(max_start_delta_mm=50.0)).max_lead_mm,
        rpc_timeout_s=1.0,
    )
    return limbs, motion, grips, arms


async def test_read_state_layout():
    limbs, motion, _, _ = _limbs()
    state = await limbs.read_state()
    assert state.shape == (20,)
    np.testing.assert_allclose(state[:9], pose_state_from_proto(motion.poses["lg"]))
    np.testing.assert_allclose(state[9:18], pose_state_from_proto(motion.poses["rg"]))
    # do_command grips normalize as (value - open) / (closed - open): 30 -> 0.7, 70 -> 0.3
    np.testing.assert_allclose(state[18:], [0.7, 0.3])


async def test_start_stops_stale_pipeline_then_starts_each_limb_at_measured_pose():
    limbs, motion, _, _ = _limbs()
    await limbs.start()
    kinds = [next(iter(c)) for c in motion.commands]
    assert kinds == ["teleop_stop", "teleop_start", "teleop_start"]
    starts = [json.loads(c["teleop_start"]) for c in motion.commands[1:]]
    assert [s["componentName"] for s in starts] == ["lg", "rg"]
    assert starts[0]["destination"]["pose"]["x"] == pytest.approx(300.0)


async def test_command_streams_each_limb_and_writes_grips():
    limbs, motion, grips, _ = _limbs()
    await limbs.start()
    state = await limbs.read_state()
    target = state.copy()
    target[2] += 10.0  # left up 10 mm
    target[9 + 0] += 5.0  # right +5 mm x
    target[18:] = [1.0, 0.25]
    clamped = await limbs.command(target, first=True)
    assert clamped is False
    moves = [c for c in motion.commands if "teleop_move" in c]
    assert [m["component_name"] for m in moves] == ["lg", "rg"]
    assert motion.poses["lg"].z == pytest.approx(410.0, abs=1e-6)
    assert motion.poses["rg"].x == pytest.approx(305.0, abs=1e-6)
    assert grips["lgrip"].position == pytest.approx(0.0)  # closure 1 -> closed_value 0
    assert grips["rgrip"].position == pytest.approx(75.0)


async def test_first_command_refuses_a_far_target():
    limbs, motion, _, _ = _limbs()
    await limbs.start()
    target = await limbs.read_state()
    target[0] += 80.0
    with pytest.raises(SafetyError, match="max_start_delta_mm"):
        await limbs.command(target, first=True)
    assert not [c for c in motion.commands if "teleop_move" in c]


async def test_later_commands_clamp_against_last_commanded_not_measured():
    limbs, motion, _, _ = _limbs()
    await limbs.start()
    target = await limbs.read_state()
    await limbs.command(target, first=True)
    motion.poses["lg"] = Pose(x=260, y=300, z=400, **DOWN)  # a lagging arm (within max_lead_mm) must not move the clamp origin
    target = target.copy()
    target[0] += 100.0  # 100 mm in one tick, ceiling 40
    assert await limbs.command(target, first=False) is True
    assert motion.poses["lg"].x == pytest.approx(340.0, abs=1e-6)


async def test_stop_ends_the_pipeline_and_stops_arms():
    limbs, motion, _, arms = _limbs()
    await limbs.start()
    await limbs.stop()
    stops = [c for c in motion.commands if "teleop_stop" in c][1:]  # [0] is start's stale-pipeline stop
    assert stops == [{"teleop_stop": True}]
    assert all(a.stopped for a in arms.values())


async def test_command_rejects_wrong_width():
    limbs, _, _, _ = _limbs()
    await limbs.start()
    with pytest.raises(SafetyError, match="20"):
        await limbs.command(np.zeros(19), first=True)


async def test_pose_round_trips_through_the_wire():
    limbs, motion, _, _ = _limbs()
    await limbs.start()
    state = await limbs.read_state()
    await limbs.command(state, first=True)
    np.testing.assert_allclose(pose_state_from_proto(motion.poses["lg"]), state[:9], atol=1e-6)


async def test_rotation_is_clamped_about_the_same_axis():
    limbs, motion, _, _ = _limbs()
    await limbs.start()
    state = await limbs.read_state()
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    from scipy.spatial.transform import Rotation

    target = state.copy()
    rot = state_rotation(state[:9]) * Rotation.from_rotvec(0.5 * axis)
    target[3:9] = rot.as_matrix()[:2, :].reshape(6)
    assert await limbs.command(target, first=True) is True
    delta = state_delta(state[:9], pose_state_from_proto(motion.poses["lg"]))
    np.testing.assert_allclose(delta[3:6], 0.12 * axis, atol=1e-6)


async def test_runaway_lead_over_a_frozen_arm_is_refused():
    limbs, motion, _, _ = _limbs()
    await limbs.start()
    motion.follow = False
    target = await limbs.read_state()
    await limbs.command(target, first=True)
    with pytest.raises(SafetyError, match="max_lead_mm"):
        for _ in range(10):
            target = target.copy()
            target[0] += 30.0
            await limbs.command(target, first=False)
    assert len([c for c in motion.commands if "teleop_move" in c]) < 2 * 6


async def test_normal_lag_does_not_trip_the_lead_bound():
    limbs, motion, _, _ = _limbs()
    await limbs.start()
    target = await limbs.read_state()
    await limbs.command(target, first=True)
    for _ in range(10):
        target = target.copy()
        target[0] += 30.0
        await limbs.command(target, first=False)


async def test_grip_clip_is_counted_and_reported():
    limbs, _, _, _ = _limbs()
    await limbs.start()
    target = await limbs.read_state()
    target[18:] = [1.4, 0.5]
    assert await limbs.command(target, first=True) is True
    assert limbs._safety.clamp_counts["gripper"] == 1
