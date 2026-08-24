"""Tests for viam-labs:vla:controller's lifecycle and control loop.

Two corrections from the plan draft's Task 17 test listing, both load-bearing
(see the "BLOCKER RESOLVED" callout in the design plan): the arm is commanded
via ``arm.move_to_joint_positions(JointPositions(values=...))`` -- a single
``JointPositions``, no ``MoveOptions`` -- because ``move_through_joint_
positions``/``MoveOptions`` ship in no released viam-sdk (installed 0.80.0
has only ``move_to_joint_positions``, which takes no options). Every test in
the plan asserting on ``MoveOptions``/``options.HasField(...)`` is rewritten
below to assert on the single-``JointPositions`` call instead, and the
velocity-ceiling coverage moved to asserting the derived
``max_joint_delta_degs`` per-tick clamp (verifiable through ``ControllerConfig``
tests) plus the startup log line, rather than a ``MoveOptions`` payload that
has no enforcement path on this SDK.
"""

from __future__ import annotations

import asyncio
import logging
import math

import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig

from tests.fakes import FakeArm, FakeCamera, FakeGripper, StalledArm
from vla.controller.service import VLAController
from vla.policy.fake_backend import FakePolicyBackend
from vla.policy.service import VLAPolicy
from vla.wire import encode_matrix

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class FakePolicyClient:
    """Duck-types the policy service's DoCommand surface (Task 7)."""

    def __init__(
        self,
        action_dim=5,
        n=4,
        state="ready",
        relative=False,
        supports_rtc=True,
        image_feature_keys=("observation.images.top",),
        extra_image_shapes=None,
        error="",
        action_value=0.0,
    ):
        self.action_dim = action_dim
        self.n = n
        self.state = state
        self.relative = relative
        self.supports_rtc = supports_rtc
        self.image_feature_keys = list(image_feature_keys)
        self.extra_image_shapes = extra_image_shapes or {}
        self.error = error
        self.action_value = action_value
        self.infer_calls = 0
        self.fail_infer = False
        self.infer_delay_s = 0.0

    async def do_command(self, command, **kwargs):
        name = command.get("command")
        if name == "status":
            return {"state": self.state, "error": self.error}
        if name == "specs":
            input_features = {
                key: [3.0, 224.0, 224.0] for key in self.image_feature_keys
            }
            input_features.update(self.extra_image_shapes)
            input_features["observation.state"] = [float(self.action_dim)]
            # Numbers are floats on purpose. A real call crosses gRPC, and
            # protobuf Struct stores every number as a double, so the
            # controller never sees ints here. Returning ints would hide
            # that from tests.
            return {
                "policy_type": "fake",
                "action_dim": float(self.action_dim),
                "n_action_steps": float(self.n),
                "input_features": input_features,
                "output_features": {"action": [float(self.action_dim)]},
                "image_feature_keys": self.image_feature_keys,
                "supports_rtc": self.supports_rtc,
                "rtc_enabled": False,
                "relative_actions": self.relative,
                "device": "cpu",
            }
        if name == "infer":
            self.infer_calls += 1
            if self.infer_delay_s:
                await asyncio.sleep(self.infer_delay_s)
            if self.fail_infer:
                raise RuntimeError("inference exploded")
            chunk = np.full((self.n, self.action_dim), self.action_value, dtype=np.float32)
            return {
                "actions": encode_matrix(chunk),
                "raw_actions": encode_matrix(chunk),
                "latency_s": 0.001,
            }
        raise ValueError(name)


def _config(**overrides):
    attrs = {
        "policy_service": "p",
        "arm": "a",
        "cameras": {"observation.images.top": "cam"},
        "state_joint_indices": [0, 1, 2, 3, 4],
        "fps": 50.0,
        "safety": {"max_start_delta_degs": 1000.0},
    }
    attrs.update(overrides)
    s = Struct()
    s.update(attrs)
    return ServiceConfig(
        name="c", api="rdk:service:generic", model="viam-labs:vla:controller", attributes=s
    )


def _deps(policy=None, arm=None, camera=None, **extra):
    deps = {
        "p": policy or FakePolicyClient(),
        "a": arm or FakeArm(positions=[0.0] * 6),
        "cam": camera or FakeCamera(),
    }
    deps.update(extra)
    return deps


def _svc(config=None, deps=None):
    svc = VLAController("c")
    svc.reconfigure(config or _config(), deps or _deps())
    return svc


async def _wait_for_state(svc, *states, timeout=2.0):
    """Poll status until it reaches one of `states`; raise on timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        status = await svc.do_command({"command": "status"})
        if status["state"] in states:
            return status
        await asyncio.sleep(0.01)
    raise AssertionError(f"never reached {states}: last status {status}")


# ---------------------------------------------------------------------------
# validate_config / dependencies
# ---------------------------------------------------------------------------


async def test_validate_returns_dependencies():
    required, optional = VLAController.validate_config(_config())
    assert set(required) == {"p", "a", "cam"}
    assert optional == []


async def test_validate_config_raises_on_bad_config():
    bad = _config(cameras={})
    with pytest.raises(Exception, match="cameras"):
        VLAController.validate_config(bad)


# ---------------------------------------------------------------------------
# reconfigure / cold policy tolerance
# ---------------------------------------------------------------------------


async def test_reconfigure_succeeds_with_cold_policy():
    # First boot is the normal case; reconfigure must not fail on it.
    policy = FakePolicyClient(state="loading")
    svc = _svc(deps=_deps(policy=policy))
    assert (await svc.do_command({"command": "status"}))["state"] == "idle"


async def test_reconfigure_does_not_touch_the_policy_at_all():
    """reconfigure() must never call the policy service -- specs are fetched
    lazily after the first start(), not eagerly during reconfigure."""

    class ExplodingPolicy(FakePolicyClient):
        async def do_command(self, command, **kwargs):
            raise AssertionError("policy must not be contacted during reconfigure()")

    # Must not raise.
    _svc(deps=_deps(policy=ExplodingPolicy()))


async def test_status_starts_idle():
    assert (await _svc().do_command({"command": "status"}))["state"] == "idle"


async def test_reconfigure_logs_derived_velocity_budget(caplog):
    with caplog.at_level(logging.INFO, logger="vla.controller.service"):
        _svc(config=_config(safety={"max_start_delta_degs": 1000.0, "max_vel_degs_per_sec": 40.0}))
    messages = [r.message for r in caplog.records]
    assert any("max_joint_delta_degs" in m and "max_vel_degs_per_sec" in m for m in messages), messages


async def test_reconfigure_logs_budget_even_without_max_vel_configured(caplog):
    with caplog.at_level(logging.INFO, logger="vla.controller.service"):
        _svc()
    messages = [r.message for r in caplog.records]
    assert any("max_joint_delta_degs" in m for m in messages), messages


# ---------------------------------------------------------------------------
# start() acks immediately / waiting_for_policy
# ---------------------------------------------------------------------------


async def test_start_acks_immediately_with_cold_policy():
    policy = FakePolicyClient(state="loading")
    svc = _svc(deps=_deps(policy=policy))
    out = await svc.do_command({"command": "start", "task": "t"})
    assert out["ok"] is True
    assert (await svc.do_command({"command": "status"}))["state"] == "waiting_for_policy"
    await svc.do_command({"command": "stop"})


async def test_start_ack_returns_before_policy_ready_timeout():
    """Acking must not depend on how long policy_ready_timeout_s is -- start()
    itself must return promptly even with an enormous configured timeout."""
    policy = FakePolicyClient(state="loading")
    svc = _svc(
        config=_config(policy_ready_timeout_s=600),
        deps=_deps(policy=policy),
    )
    started = asyncio.get_event_loop().time()
    await svc.do_command({"command": "start", "task": "t"})
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed < 0.5, f"start() blocked for {elapsed:.2f}s waiting on the policy"
    await svc.do_command({"command": "stop"})


async def test_starting_twice_while_running_raises():
    svc = _svc()
    await svc.do_command({"command": "start", "task": "t"})
    with pytest.raises(Exception, match="already running"):
        await svc.do_command({"command": "start", "task": "t"})
    await svc.do_command({"command": "stop"})


async def test_start_without_task_and_no_default_raises():
    svc = _svc()
    with pytest.raises(Exception, match="task"):
        await svc.do_command({"command": "start"})


async def test_start_uses_configured_default_task_when_omitted():
    policy = FakePolicyClient()
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(config=_config(task="pick up the block"), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert policy.infer_calls >= 1


async def test_policy_load_failure_surfaces_as_error():
    policy = FakePolicyClient(state="failed", error="checkpoint not found")
    svc = _svc(deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "checkpoint not found" in status["last_error"]


async def test_policy_timeout_surfaces_as_error_and_stops_arm():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(state="loading")
    svc = _svc(
        config=_config(policy_ready_timeout_s=1),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error", timeout=5.0)
    assert "not ready" in status["last_error"] or "timed out" in status["last_error"].lower()
    assert arm.stopped >= 1


# ---------------------------------------------------------------------------
# The loop itself: runs, commands the arm, paces to fps.
# ---------------------------------------------------------------------------


async def test_loop_runs_and_commands_the_arm():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.3)
    await svc.do_command({"command": "stop"})
    assert policy.infer_calls >= 1
    assert len(arm.moves) >= 1


async def test_arm_commanded_via_move_to_joint_positions_single_call():
    # move_to_joint_positions is the only method the installed SDK's Arm
    # exposes that FakeArm implements; move_through_joint_positions does not
    # exist. A single positions object, not a list wrapped in one. The write
    # is full-width (every arm joint, 6 here), not just the 5 driven ones --
    # see test_non_contiguous_state_joint_indices_map_to_the_correct_arm_joints
    # for why a narrower, purely-positional write is actually wrong.
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0, "max_vel_degs_per_sec": 30.0}),
        deps=_deps(arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1
    positions = arm.moves[0]
    assert hasattr(positions, "values")
    assert len(positions.values) == 6


async def test_non_contiguous_state_joint_indices_map_to_the_correct_arm_joints():
    # Exact repro from code review: state_joint_indices=[1,2,3,4,5] on a
    # 6-joint arm whose base (joint 0) the policy does not drive. A
    # positional write (clamped slot i -> arm joint i) would scramble every
    # joint by one, abandon joint 5 entirely, and blow the delta budget on
    # joint 0 to boot -- this is the exact bug this test guards against.
    arm = FakeArm(positions=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    policy = FakePolicyClient(action_dim=5, action_value=999.0)  # push hard, same direction
    svc = _svc(
        config=_config(
            state_joint_indices=[1, 2, 3, 4, 5],
            safety={"max_start_delta_degs": 1000.0},  # default max_joint_delta_degs=8
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1
    values = list(arm.moves[0].values)
    assert len(values) == 6
    # Joint 0 is never driven by this policy -- it must hold its measured
    # position exactly, never receive a clamped-but-misdirected value.
    assert values[0] == pytest.approx(10.0)
    # Joints 1..5 are each clamped by +8 from their OWN measured position
    # and must land on the joint they were actually computed against.
    assert values[1] == pytest.approx(28.0)
    assert values[2] == pytest.approx(38.0)
    assert values[3] == pytest.approx(48.0)
    assert values[4] == pytest.approx(58.0)
    assert values[5] == pytest.approx(68.0)


async def test_state_joint_indices_exceeding_arm_joint_count_refuses_to_start():
    arm = FakeArm(positions=[0.0] * 3)  # only 3 joints; state_joint_indices needs 0..4
    policy = FakePolicyClient(action_dim=5)
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0}),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "state_joint_indices" in status["last_error"]
    # Not just "some error mentioning state_joint_indices" -- ObservationBuilder
    # (Task 14) independently re-checks this same bound on the very first
    # tick and produces a similarly-worded message, so a bare substring
    # check alone would still pass even with _check_joint_indices deleted
    # entirely (only the *timing* -- before vs. during the first tick -- and
    # exact wording would differ). This phrase is unique to this method.
    assert "arm reports joints 0.." in status["last_error"]
    assert len(arm.moves) == 0
    assert arm.stopped >= 1


async def test_gripper_joint_index_exceeding_arm_joint_count_refuses_to_start():
    arm = FakeArm(positions=[0.0] * 5)  # 5 joints; gripper wants index 5
    policy = FakePolicyClient(action_dim=6)
    svc = _svc(
        config=_config(
            gripper={"type": "arm_joint", "joint_index": 5},
            safety={"max_start_delta_degs": 1000.0},
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "gripper.joint_index" in status["last_error"]
    assert len(arm.moves) == 0
    assert arm.stopped >= 1


async def test_delta_clamp_uses_measured_position_not_last_commanded():
    # With FakeArm (which snaps its measured position to whatever was last
    # commanded), "measured" and "last commanded" are indistinguishable --
    # this is structurally untestable without an arm whose measured position
    # never moves. Across several ticks, the commanded value must stay
    # clamped to +8 from the (never-moving) measured position of 0 every
    # time -- not accumulate (8, 16, 24, ...), which is what a controller-
    # level regression that fed the last commanded value back in as "current"
    # would produce.
    arm = StalledArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_dim=5, action_value=50.0)
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0}),  # default max_joint_delta_degs=8
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 3
    for move in arm.moves:
        for v in list(move.values)[:5]:
            assert v == pytest.approx(8.0), list(move.values)


async def test_action_units_radians_are_converted_before_commanding_the_arm():
    # If action_units="radians" -> degrees conversion were skipped (an
    # identity mutation), a policy emitting pi/2 rad (~90deg) would be
    # commanded as ~1.57 "degrees" -- small enough to never even reach the
    # delta clamp. Converted correctly, it hits and is capped by the default
    # max_joint_delta_degs=8.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_dim=5, action_value=math.pi / 2)
    svc = _svc(
        config=_config(
            action_units="radians", safety={"max_start_delta_degs": 1000.0}
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1
    first_value = list(arm.moves[0].values)[0]
    assert abs(first_value - 8.0) < 0.5, first_value


async def test_check_start_applies_only_to_the_first_tick_not_every_tick():
    # A large jump mid-run must be clamped, never refused -- check_start only
    # gates the very first commanded action.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_dim=5, action_value=1.0)  # within default max_start_delta_degs=15
    svc = _svc(config=_config(safety={}), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.15)  # several ticks/chunks at the initial small value

    policy.action_value = 999.0  # far beyond max_start_delta_degs=15, mid-run
    await asyncio.sleep(0.2)  # enough for a fresh chunk to pick up the new value
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["state"] != "error"
    assert len(arm.moves) >= 2


async def test_arm_joint_gripper_is_commanded_and_clamped():
    # gripper.type="arm_joint" is the spec's recommended default but was
    # otherwise never exercised end-to-end through the controller loop.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_dim=6, action_value=999.0)
    svc = _svc(
        config=_config(
            gripper={"type": "arm_joint", "joint_index": 5},
            safety={"max_start_delta_degs": 1000.0},
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1
    values = list(arm.moves[0].values)
    assert len(values) == 6
    # The gripper joint (index 5) must actually be commanded -- not silently
    # dropped -- and must be clamped by the same degree-based delta clamp as
    # any other joint, not passed through raw (999).
    assert values[5] == pytest.approx(8.0), values


async def test_moved_joint_values_land_within_the_derived_delta_clamp():
    # With max_vel_degs_per_sec=30 at fps=50, max_joint_delta_degs = 0.6deg/tick.
    # A policy emitting large actions must still only move the arm by the
    # clamped amount per tick -- proof the velocity ceiling is enforced via
    # the safety layer, since there is no MoveOptions to carry it instead.
    arm = FakeArm(positions=[0.0] * 5 + [0.0])
    policy = FakePolicyClient(action_dim=5, action_value=999.0)
    svc = _svc(
        config=_config(
            fps=50.0, safety={"max_start_delta_degs": 1000.0, "max_vel_degs_per_sec": 30.0}
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1
    first_move = list(arm.moves[0].values)
    for v in first_move:
        assert abs(v - 0.0) <= 0.6 + 1e-6


async def test_gripper_write_happens_after_arm_move_for_non_arm_joint_gripper():
    arm = FakeArm(positions=[0.0] * 5)
    gripper = FakeGripper()
    policy = FakePolicyClient(action_dim=6, action_value=0.5)
    svc = _svc(
        config=_config(
            gripper={"type": "gripper", "name": "g", "mode": "inputs"},
        ),
        deps=_deps(policy=policy, arm=arm, g=gripper),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1
    # gripper.sent[0] is the pre-flight probe's write-back of its own current
    # value ([0.0], FakeGripper's default), which now happens before the
    # arm's first move -- the real tick's value shows up afterward.
    assert len(gripper.sent) >= 2
    assert gripper.sent[-1] == [0.5]


async def test_gripper_write_order_is_after_arm_move_not_before():
    """Both happening is necessary but not sufficient -- prove the order too:
    a swap (write the gripper, then move the arm) must be caught."""
    events: list[str] = []
    arm = FakeArm(positions=[0.0] * 5)
    gripper = FakeGripper()

    real_move = arm.move_to_joint_positions
    real_go_to_inputs = gripper.go_to_inputs

    async def tracked_move(*a, **k):
        events.append("arm")
        return await real_move(*a, **k)

    async def tracked_go_to_inputs(*a, **k):
        events.append("gripper")
        return await real_go_to_inputs(*a, **k)

    arm.move_to_joint_positions = tracked_move
    gripper.go_to_inputs = tracked_go_to_inputs

    policy = FakePolicyClient(action_dim=6, action_value=0.5)
    svc = _svc(
        config=_config(gripper={"type": "gripper", "name": "g", "mode": "inputs"}),
        deps=_deps(policy=policy, arm=arm, g=gripper),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})

    # events[0] is the pre-flight probe's own gripper write, before any arm
    # move -- expected and fine. What matters is that once the arm starts
    # moving, its own tick's gripper write always comes right after it, not
    # before.
    first_arm_idx = events.index("arm")
    assert events[first_arm_idx + 1] == "gripper", events


async def test_gripper_incompatible_mode_is_caught_before_any_arm_motion():
    # mode="inputs" against a driver lacking go_to_inputs must be discovered
    # by the pre-flight probe, before the arm has ever been commanded --
    # not on the first real tick, which would break the refuse-before-motion
    # discipline every other _run() check maintains.
    arm = FakeArm(positions=[0.0] * 5)
    gripper = FakeGripper(supports_inputs=False)
    policy = FakePolicyClient(action_dim=6)
    svc = _svc(
        config=_config(gripper={"type": "gripper", "name": "g", "mode": "inputs"}),
        deps=_deps(policy=policy, arm=arm, g=gripper),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "go_to_inputs" in status["last_error"] or "does not implement" in status["last_error"]
    assert len(arm.moves) == 0, "the arm must never have been commanded before the gripper probe failed"
    assert arm.stopped >= 1


async def test_extra_configured_camera_not_needed_by_policy_is_ignored():
    """A camera mapped in config but not requested by the policy's
    image_feature_keys must never be read, let alone cause a failure."""
    arm = FakeArm(positions=[0.0] * 5)
    policy = FakePolicyClient(image_feature_keys=("observation.images.top",))
    unused_camera = FakeCamera(fail=True)  # would blow up the tick if ever read
    svc = _svc(
        config=_config(
            cameras={
                "observation.images.top": "cam",
                "observation.images.wrist": "unused_cam",
            }
        ),
        deps=_deps(policy=policy, arm=arm, unused_cam=unused_camera),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["state"] == "running"
    assert unused_camera.reads == 0


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


async def test_stop_halts_the_loop():
    policy = FakePolicyClient()
    svc = _svc(deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    calls = policy.infer_calls
    await asyncio.sleep(0.2)
    assert policy.infer_calls == calls


async def test_stop_transitions_to_stopped_not_idle():
    svc = _svc()
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert (await svc.do_command({"command": "status"}))["state"] == "stopped"


async def test_stop_before_any_start_is_safe():
    svc = _svc()
    await svc.do_command({"command": "stop"})  # must not raise
    assert (await svc.do_command({"command": "status"}))["state"] == "idle"


async def test_restart_after_stop_works():
    policy = FakePolicyClient()
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    calls_before_restart = policy.infer_calls
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert policy.infer_calls > calls_before_restart


# ---------------------------------------------------------------------------
# Error handling: inference failure -- stop_on_error (default True) stops
# the arm and halts.
# ---------------------------------------------------------------------------


async def test_inference_failure_stops_arm_and_records_error():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    policy.fail_infer = True
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "exploded" in status["last_error"]
    assert arm.stopped >= 1


async def test_camera_failure_stops_arm_and_records_error():
    arm = FakeArm(positions=[0.0] * 6)
    camera = FakeCamera(fail=True)
    svc = _svc(deps=_deps(arm=arm, camera=camera))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "camera" in status["last_error"].lower()
    assert arm.stopped >= 1
    assert len(arm.moves) == 0  # the failing tick must never have commanded a move


async def test_camera_failure_never_leaves_the_loop_dead_without_stopping():
    """The mutation this guards against: catching the tick's exception but
    forgetting to call arm.stop() before halting."""
    arm = FakeArm(positions=[0.0] * 6)
    camera = FakeCamera(fail=True)
    svc = _svc(deps=_deps(arm=arm, camera=camera))
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "error")
    assert arm.stopped >= 1


async def test_stop_on_error_false_skips_a_single_tick_and_keeps_running():
    # fps=10 (period 0.1s) so a 0.05s sleep reliably captures exactly the
    # first tick's failure -- before a second tick, let alone escalation via
    # starvation_grace_ticks (default 3), can happen.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    policy.fail_infer = True
    svc = _svc(
        config=_config(
            fps=10.0, safety={"max_start_delta_degs": 1000.0, "stop_on_error": False}
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.05)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "running"
    assert "exploded" in status["last_error"]
    assert arm.stopped == 0
    assert len(arm.moves) == 0
    await svc.do_command({"command": "stop"})


async def test_stop_on_error_false_recovers_once_inference_starts_succeeding():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    policy.fail_infer = True
    svc = _svc(
        config=_config(
            # A generous grace bound: this test is about recovery mid-failure,
            # not about the escalation bound itself (see the dedicated
            # escalation test below).
            starvation_grace_ticks=100,
            safety={"max_start_delta_degs": 1000.0, "stop_on_error": False},
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    policy.fail_infer = False
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1


async def test_stop_on_error_false_escalates_after_starvation_grace_ticks_exceeded():
    # 113 consecutive failures leaving the loop reporting "running" forever
    # is exactly the hole this bound closes: more than starvation_grace_ticks
    # consecutive failures must stop the arm and halt, regardless of
    # stop_on_error.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    policy.fail_infer = True
    svc = _svc(
        config=_config(
            fps=50.0,
            starvation_grace_ticks=2,
            safety={"max_start_delta_degs": 1000.0, "stop_on_error": False},
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error", timeout=2.0)
    assert "consecutive" in status["last_error"].lower()
    assert "starvation_grace_ticks" in status["last_error"]
    assert arm.stopped >= 1
    assert len(arm.moves) == 0


async def test_consecutive_failure_count_resets_on_a_successful_tick():
    # Without a reset, N-1 sporadic failures interspersed with successes
    # would eventually cross the same threshold as N-in-a-row -- which is
    # not what "consecutive" is supposed to mean. n=1 (one action per chunk)
    # makes every tick call infer(), so waiting for infer_calls to advance
    # gives an exact, non-timing-dependent handle on "exactly one tick has
    # happened since the flag was flipped" -- a blind sleep()-based toggle
    # can't guarantee that and risks two failures landing back to back by
    # scheduling luck alone.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01, n=1)
    svc = _svc(
        config=_config(
            fps=20.0,
            starvation_grace_ticks=1,
            safety={"max_start_delta_degs": 1000.0, "stop_on_error": False},
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")

    async def _wait_for_call(target: int) -> None:
        for _ in range(500):
            if policy.infer_calls >= target:
                return
            await asyncio.sleep(0.002)
        raise AssertionError(f"infer_calls never reached {target}")

    for _ in range(6):
        policy.fail_infer = True
        await _wait_for_call(policy.infer_calls + 1)
        policy.fail_infer = False
        await _wait_for_call(policy.infer_calls + 1)

    status = await svc.do_command({"command": "status"})
    assert status["state"] == "running"
    assert arm.stopped == 0  # checked before the explicit stop() below, which legitimately stops it
    await svc.do_command({"command": "stop"})


# ---------------------------------------------------------------------------
# Error handling: arm command failure -- always stops immediately,
# regardless of stop_on_error.
# ---------------------------------------------------------------------------


async def test_arm_failure_stops_immediately():
    arm = FakeArm(positions=[0.0] * 6)
    arm.fail_next_move = True
    svc = _svc(deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert arm.stopped >= 1
    assert status["state"] == "error"


async def test_arm_failure_stops_immediately_even_with_stop_on_error_false():
    arm = FakeArm(positions=[0.0] * 6)
    arm.fail_next_move = True
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0, "stop_on_error": False}),
        deps=_deps(arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert arm.stopped >= 1
    assert status["state"] == "error"


# ---------------------------------------------------------------------------
# Refusals that must happen before any motion.
# ---------------------------------------------------------------------------


async def test_action_dim_mismatch_refuses_to_start():
    policy = FakePolicyClient(action_dim=99)
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0}),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "action_dim" in status["last_error"]
    assert len(arm.moves) == 0
    assert arm.stopped >= 1


async def test_gripper_channel_from_checkpoint_with_gripper_type_none_refuses():
    # 5 joints configured, gripper.type defaults to "none", but the policy's
    # action_dim is 6 -- exactly one channel too many. This is the specific,
    # named case (not the generic action_dim mismatch above).
    policy = FakePolicyClient(action_dim=6)
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0}),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    # Not just "gripper" (the generic action_dim mismatch message also
    # contains that word, in "no gripper") -- the specific named phrasing,
    # so a mutant that deletes this branch and falls through to the generic
    # message is actually caught.
    assert "checkpoint emits a gripper channel" in status["last_error"]
    assert len(arm.moves) == 0
    assert arm.stopped >= 1


async def test_camera_feature_not_mapped_refuses_and_names_it():
    policy = FakePolicyClient(
        image_feature_keys=("observation.images.top", "observation.images.wrist")
    )
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0}),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "observation.images.wrist" in status["last_error"]
    # Not just the key's presence (a bare, un-guarded KeyError on
    # cfg.cameras[key] would also mention the key in its repr) -- the
    # graceful, named phrasing that says *why* it's a problem.
    assert "not" in status["last_error"] and "mapped" in status["last_error"]
    assert len(arm.moves) == 0
    assert arm.stopped >= 1


async def test_first_action_far_from_current_pose_refuses_and_stops():
    # max_start_delta_degs defaults to 15 -- _config()'s own default override
    # (1000.0) must be cleared with safety={} to actually exercise that
    # default. The policy emits 999 for every dimension while the arm sits
    # at 0 -- refuse rather than moving there.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=999.0)
    svc = _svc(config=_config(safety={}), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "start" in status["last_error"].lower() or "pose" in status["last_error"].lower()
    assert len(arm.moves) == 0, "the arm must never have been commanded"
    assert arm.stopped >= 1


async def test_first_action_within_start_delta_is_accepted():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=1.0)  # within default max_start_delta_degs=15
    svc = _svc(config=_config(), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["state"] != "error"
    assert len(arm.moves) >= 1


# ---------------------------------------------------------------------------
# RTC mode refusals.
# ---------------------------------------------------------------------------


async def test_rtc_mode_refuses_relative_action_checkpoint():
    policy = FakePolicyClient(relative=True)
    svc = _svc(config=_config(mode="rtc"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "relative" in status["last_error"].lower()


async def test_rtc_mode_refuses_policy_without_rtc_support():
    policy = FakePolicyClient(supports_rtc=False)
    svc = _svc(config=_config(mode="rtc"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "rtc" in status["last_error"].lower()


async def test_rtc_mode_refuses_even_when_supported_and_not_relative():
    # mode=rtc is unimplemented outright; RTCScheduler is a follow-up plan.
    policy = FakePolicyClient(supports_rtc=True, relative=False)
    svc = _svc(config=_config(mode="rtc"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "not implemented" in status["last_error"].lower()


async def test_sequential_mode_never_touches_rtc_fields():
    # mode=sequential must not even look at supports_rtc/relative_actions.
    policy = FakePolicyClient(supports_rtc=False, relative=True)
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(
        config=_config(mode="sequential"),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["state"] != "error"


async def test_auto_mode_resolves_to_sequential():
    policy = FakePolicyClient()
    svc = _svc(config=_config(mode="auto"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["mode"] == "sequential"


# ---------------------------------------------------------------------------
# reconfigure while running.
# ---------------------------------------------------------------------------


async def test_reconfigure_while_running_stops_and_does_not_resume():
    arm = FakeArm(positions=[0.0] * 6)
    deps = _deps(arm=arm)
    svc = _svc(deps=deps)
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    svc.reconfigure(_config(), deps)
    await asyncio.sleep(0.2)
    assert (await svc.do_command({"command": "status"}))["state"] == "idle"
    assert arm.stopped >= 1


async def test_reconfigure_while_waiting_for_policy_stops_cleanly():
    policy = FakePolicyClient(state="loading")
    arm = FakeArm(positions=[0.0] * 6)
    deps = _deps(policy=policy, arm=arm)
    svc = _svc(deps=deps)
    await svc.do_command({"command": "start", "task": "t"})
    assert (await svc.do_command({"command": "status"}))["state"] == "waiting_for_policy"
    svc.reconfigure(_config(), deps)
    await asyncio.sleep(0.1)
    assert (await svc.do_command({"command": "status"}))["state"] == "idle"


async def test_reconfigure_resets_clamp_counts_and_latencies():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=1.0)
    deps = _deps(arm=arm, policy=policy)
    svc = _svc(deps=deps)
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    status_before = await svc.do_command({"command": "status"})
    assert status_before["avg_latency_s"] >= 0

    svc.reconfigure(_config(), deps)
    status_after = await svc.do_command({"command": "status"})
    assert status_after["avg_latency_s"] == 0.0
    assert status_after["clamp_counts"] == {}
    assert status_after["measured_fps"] == 0.0


async def test_can_start_again_shortly_after_reconfigure_cancelled_previous_run():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    deps = _deps(arm=arm, policy=policy)
    svc = _svc(deps=deps)
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)
    svc.reconfigure(_config(), deps)
    await asyncio.sleep(0.1)  # let the cancellation actually settle
    out = await svc.do_command({"command": "start", "task": "t"})
    assert out["ok"] is True
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})


async def test_reconfigure_stops_the_old_arm_not_the_new_one():
    # Every other reconfigure test reuses one `deps` dict and one arm, which
    # cannot catch a mutation that moved the old-arm capture in reconfigure()
    # to *after* _cfg/_deps are rebound -- at that point "the arm named `a`"
    # already resolves to the new arm, and the stop would silently target
    # the wrong object. Two distinct FakeArm instances make that observable.
    old_arm = FakeArm(positions=[0.0] * 6)
    new_arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    svc = _svc(deps=_deps(policy=policy, arm=old_arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.1)

    new_deps = _deps(policy=policy, arm=new_arm)
    svc.reconfigure(_config(), new_deps)
    await asyncio.sleep(0.1)

    assert old_arm.stopped >= 1
    assert new_arm.stopped == 0


async def test_start_resets_telemetry_from_a_previous_run():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    status_before = await svc.do_command({"command": "status"})
    assert status_before["avg_latency_s"] > 0
    assert status_before["measured_fps"] > 0

    await svc.do_command({"command": "start", "task": "t"})
    status_immediately_after = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status_immediately_after["avg_latency_s"] == 0.0
    assert status_immediately_after["measured_fps"] == 0.0


async def test_last_error_is_cleared_on_start_after_a_previous_failure():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    policy.fail_infer = True
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert status["last_error"] != ""

    policy.fail_infer = False
    await svc.do_command({"command": "start", "task": "t"})
    status_immediately_after = await svc.do_command({"command": "status"})
    assert status_immediately_after["last_error"] == ""
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


async def test_close_stops_the_arm():
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.close()
    assert arm.stopped >= 1


async def test_close_cancels_the_loop():
    policy = FakePolicyClient()
    svc = _svc(deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.close()
    calls = policy.infer_calls
    await asyncio.sleep(0.2)
    assert policy.infer_calls == calls


async def test_close_before_any_start_is_safe():
    svc = _svc()
    await svc.close()  # must not raise


async def test_close_while_waiting_for_policy_stops_the_arm():
    policy = FakePolicyClient(state="loading")
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await svc.close()
    assert arm.stopped >= 1


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


async def test_status_reports_clamp_counts():
    svc = _svc()
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    assert "clamp_counts" in await svc.do_command({"command": "status"})


async def test_status_reports_every_documented_field():
    svc = _svc()
    status = await svc.do_command({"command": "status"})
    for key in (
        "state",
        "mode",
        "queue_size",
        "avg_latency_s",
        "measured_fps",
        "clamp_counts",
        "last_error",
    ):
        assert key in status, f"missing status field: {key}"


async def test_status_queue_size_reflects_the_real_scheduler_not_hardcoded():
    # A loose "0 <= queue_size <= 7" bound would pass even if queue_size were
    # hardcoded to 0 -- deterministic instead: fps=2 (period 0.5s) means the
    # second tick is not due for a long while, so shortly after the first
    # tick completes (one infer, one merge of a 7-step chunk, one get()) the
    # queue must hold exactly 6, not 0 and not 7.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(n=7, action_value=0.01)
    svc = _svc(config=_config(fps=2.0), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)  # first tick has completed; second is not due until t=0.5s
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["queue_size"] == 6


async def test_status_last_error_is_empty_string_before_any_failure():
    svc = _svc()
    assert (await svc.do_command({"command": "status"}))["last_error"] == ""


async def test_status_avg_latency_reflects_recorded_latencies():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=1.0)
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["avg_latency_s"] > 0


async def test_mode_reports_configured_value_before_specs_resolve():
    # mode="rtc" specifically (not "sequential"): with a cold policy that
    # never leaves "loading", specs never resolve, so this stays reporting
    # the *configured* value the whole time. Using "sequential" here would
    # not catch a mutant that hardcoded status()'s mode field to
    # "sequential" -- it would coincidentally match either way.
    policy = FakePolicyClient(state="loading")
    svc = _svc(config=_config(mode="rtc"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    status = await svc.do_command({"command": "status"})
    assert status["mode"] == "rtc"
    await svc.do_command({"command": "stop"})


# ---------------------------------------------------------------------------
# Timing: measured_fps reflects reality; overrun warns; no busy-wait.
# ---------------------------------------------------------------------------


async def test_measured_fps_reflects_the_configured_pace():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    svc = _svc(config=_config(fps=20.0), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.5)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    # Tight enough to catch a 2x pacing error (a mutant doubling `period`
    # would land near 10, not near 20) while still tolerating real scheduling
    # jitter under a test runner.
    assert 12.0 < status["measured_fps"] < 35.0


async def test_measured_fps_reflects_actual_pace_not_the_configured_target():
    # The single most misleading thing measured_fps could report: a
    # hardcoded `self._measured_fps = cfg.fps` would show ~200 here and this
    # must fail. fps=200 (period 5ms) is configured, but every tick is
    # forced to take ~50ms by the policy's own artificial delay, so the real
    # achieved pace is ~20fps -- nowhere near the configured target.
    # n=1: without this, SequentialScheduler only calls infer() (and thus
    # only pays infer_delay_s) once every n_action_steps ticks -- the other
    # ticks drain the cached chunk almost instantly, and the *last* tick
    # before status() happens to be sampled could easily be one of those
    # fast ones, making measured_fps report near the configured target by
    # accident rather than the forced-slow actual pace.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01, n=1)
    policy.infer_delay_s = 0.05
    svc = _svc(config=_config(fps=200.0), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.3)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert 5.0 < status["measured_fps"] < 40.0


async def test_tick_overrun_warns_instead_of_silently_drifting(caplog):
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    policy.infer_delay_s = 0.05  # much longer than the 1/200s budget below
    svc = _svc(config=_config(fps=200.0), deps=_deps(policy=policy, arm=arm))
    with caplog.at_level(logging.WARNING, logger="vla.controller.service"):
        await svc.do_command({"command": "start", "task": "t"})
        await asyncio.sleep(0.2)
        await svc.do_command({"command": "stop"})
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("overran" in m for m in warnings), warnings


# ---------------------------------------------------------------------------
# The loop must not block the event loop -- responsiveness-ticker technique
# from Task 7 (tests/policy/test_service.py).
# ---------------------------------------------------------------------------


async def test_slow_infer_does_not_block_the_event_loop():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    policy.infer_delay_s = 0.3
    svc = _svc(config=_config(fps=5.0), deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0.3)
    ticker_task.cancel()
    try:
        await ticker_task
    except asyncio.CancelledError:
        pass

    await svc.do_command({"command": "stop"})

    assert ticks > 100, (
        f"event loop only ticked {ticks} times in 0.3s while a slow infer was in "
        "flight -- the controller loop is likely blocking the event loop"
    )


# ---------------------------------------------------------------------------
# Observation warn thresholds are wired through config, not hardcoded.
# ---------------------------------------------------------------------------


async def test_stale_frame_warn_threshold_is_configurable_through_the_controller(caplog):
    import datetime

    stale_camera = FakeCamera(
        captured_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
    )
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    svc = _svc(
        config=_config(stale_frame_warn_s=0.05),
        deps=_deps(policy=policy, arm=arm, camera=stale_camera),
    )
    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await svc.do_command({"command": "start", "task": "t"})
        await asyncio.sleep(0.15)
        await svc.do_command({"command": "stop"})
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("stale" in m.lower() for m in warnings), warnings


async def test_high_stale_frame_warn_threshold_suppresses_the_warning(caplog):
    import datetime

    stale_camera = FakeCamera(
        captured_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)
    )
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    svc = _svc(
        config=_config(stale_frame_warn_s=100.0),
        deps=_deps(policy=policy, arm=arm, camera=stale_camera),
    )
    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await svc.do_command({"command": "start", "task": "t"})
        await asyncio.sleep(0.15)
        await svc.do_command({"command": "stop"})
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("stale" in m.lower() for m in warnings), warnings


async def test_duration_warn_threshold_is_configurable_through_the_controller(caplog):
    # duration_warn_s=0.0 means any positive observation-assembly duration
    # warns -- proof the field is actually threaded through to
    # ObservationBuilder, not silently dropped.
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    svc = _svc(
        config=_config(duration_warn_s=0.0),
        deps=_deps(policy=policy, arm=arm),
    )
    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await svc.do_command({"command": "start", "task": "t"})
        await asyncio.sleep(0.1)
        await svc.do_command({"command": "stop"})
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("observation assembly took" in m for m in warnings), warnings


async def test_high_duration_warn_threshold_suppresses_the_warning(caplog):
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(action_value=0.01)
    svc = _svc(
        config=_config(duration_warn_s=100.0),
        deps=_deps(policy=policy, arm=arm),
    )
    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await svc.do_command({"command": "start", "task": "t"})
        await asyncio.sleep(0.1)
        await svc.do_command({"command": "stop"})
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("observation assembly took" in m for m in warnings), warnings


# ---------------------------------------------------------------------------
# _record_latency: the trim to the most recent 50 entries, tested directly
# rather than through the full timed loop -- deterministic, and the exact
# unit responsible for avg_latency_s not becoming a lifetime average.
# ---------------------------------------------------------------------------


async def test_record_latency_trims_to_the_most_recent_50():
    svc = VLAController("c")
    for i in range(60):
        svc._record_latency(float(i))
    assert len(svc._latencies) == 50
    assert svc._latencies[0] == 10.0  # the first 10 (0..9) were pushed out
    assert svc._latencies[-1] == 59.0


async def test_record_latency_does_not_trim_under_50():
    svc = VLAController("c")
    for i in range(10):
        svc._record_latency(float(i))
    assert svc._latencies == [float(i) for i in range(10)]


# ---------------------------------------------------------------------------
# Dependency-key handling: production dependencies arrive keyed by an object
# with a `.name` attribute (a ResourceName), not always a bare string.
# ---------------------------------------------------------------------------


class _NamedKey:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_NamedKey({self.name!r})"


async def test_dependency_keys_with_name_attribute_are_resolved():
    policy = FakePolicyClient()
    arm = FakeArm(positions=[0.0] * 6)
    camera = FakeCamera()
    deps = {
        _NamedKey("p"): policy,
        _NamedKey("a"): arm,
        _NamedKey("cam"): camera,
    }
    svc = _svc(deps=deps)
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["state"] != "error"


# ---------------------------------------------------------------------------
# mode: "async" -- end to end through the real controller loop (Task 22).
# ---------------------------------------------------------------------------


async def test_async_mode_end_to_end_commands_the_arm_and_reports_mode():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(n=5)
    svc = _svc(
        config=_config(mode="async", fps=50.0, queue_threshold=3),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.2)
    status = await svc.do_command({"command": "status"})
    assert status["mode"] == "async"
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1
    assert policy.infer_calls >= 1


# ---------------------------------------------------------------------------
# unused_image_features (Task: smolvla_base inheritance) end-to-end: the
# REAL policy service (vla.policy.service.VLAPolicy) over FakePolicyBackend
# stands in for "p" here, not the duck-typed FakePolicyClient every other
# test in this file uses -- the point is to prove the actual
# PolicyConfig.unused_image_features -> LeRobotBackend/FakePolicyBackend ->
# specs.image_feature_keys pipeline, not a hand-rolled stand-in for it. No
# torch/lerobot required: FakePolicyBackend exists precisely so this is
# exercisable without either.
# ---------------------------------------------------------------------------


async def test_controller_runs_when_policy_declares_more_cameras_than_are_mapped(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("{}")

    declared_cameras = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    policy = VLAPolicy("p")
    policy._backend_factory = lambda: FakePolicyBackend(
        action_dim=5, n_action_steps=4, camera_keys=declared_cameras
    )
    policy_attrs = Struct()
    policy_attrs.update({
        "model_path": str(tmp_path),
        # camera3 is a checkpoint artifact this robot never wires up --
        # exactly the smolvla_base 3-camera inheritance case.
        "unused_image_features": ["observation.images.camera3"],
    })
    policy.reconfigure(
        ServiceConfig(
            name="p",
            api="rdk:service:generic",
            model="viam-labs:vla:policy",
            attributes=policy_attrs,
        ),
        {},
    )
    await policy.await_ready()

    arm = FakeArm(positions=[0.0] * 5)
    svc = _svc(
        config=_config(
            cameras={
                "observation.images.camera1": "cam1",
                "observation.images.camera2": "cam2",
                # camera3 is deliberately unmapped: the controller must
                # never need to know it exists, let alone read/encode it.
            },
            state_joint_indices=[0, 1, 2, 3, 4],
        ),
        deps=_deps(
            policy=policy,
            arm=arm,
            cam1=FakeCamera(),
            cam2=FakeCamera(),
        ),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "running", "error")
    await svc.do_command({"command": "stop"})
    assert status["state"] == "running"
    assert len(arm.moves) >= 1


async def test_async_mode_starvation_escalates_when_inference_stalls():
    """The `action is None` branch in `_loop` is only reachable under
    AsyncScheduler -- this exercises it end to end, not just at the
    scheduler level. Once a background inference stalls indefinitely and
    the queue drains, consecutive empty ticks must escalate past
    `starvation_grace_ticks` and halt -- the same bound that already
    escalates a run of consecutive tick *failures*, now doing the same job
    for a run of consecutive *empty* ticks."""

    class StallingPolicy(FakePolicyClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._hang = asyncio.Event()

        async def do_command(self, command, **kwargs):
            if command.get("command") == "infer":
                self.infer_calls += 1
                if self.infer_calls > 1:
                    # Never set: simulates a policy call that never returns,
                    # so every inference after the first stalls forever.
                    await self._hang.wait()
                chunk = np.full((self.n, self.action_dim), self.action_value, dtype=np.float32)
                return {
                    "actions": encode_matrix(chunk),
                    "raw_actions": encode_matrix(chunk),
                    "latency_s": 0.001,
                }
            return await super().do_command(command, **kwargs)

    arm = FakeArm(positions=[0.0] * 6)
    policy = StallingPolicy(n=3)
    svc = _svc(
        config=_config(
            mode="async",
            fps=100.0,
            queue_threshold=2,
            starvation_grace_ticks=2,
            safety={"max_start_delta_degs": 1000.0},
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error", timeout=5.0)
    assert "starvation_grace_ticks" in status["last_error"]
    assert "starved" in status["last_error"].lower()
    assert arm.stopped >= 1
    # The first chunk's real actions were commanded before starvation took
    # over -- this is not a "nothing ever worked" failure.
    assert len(arm.moves) >= 1


async def test_async_mode_stop_cancels_a_background_inference_not_on_the_call_stack():
    """The background inference is not always on the loop task's own await
    stack -- once AsyncScheduler starts serving already-queued actions
    instead of blocking on it, cancelling the loop task alone does not
    reach it. `_stop()` must still reach it via `scheduler.close()`, or it
    would keep running orphaned past the point the controller reports
    stopped."""

    class ControllablePolicy(FakePolicyClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.second_call_started = asyncio.Event()
            self.hang = asyncio.Event()

        async def do_command(self, command, **kwargs):
            if command.get("command") == "infer":
                self.infer_calls += 1
                if self.infer_calls == 2:
                    self.second_call_started.set()
                    await self.hang.wait()  # never released in this test
                chunk = np.full((self.n, self.action_dim), self.action_value, dtype=np.float32)
                return {
                    "actions": encode_matrix(chunk),
                    "raw_actions": encode_matrix(chunk),
                    "latency_s": 0.001,
                }
            return await super().do_command(command, **kwargs)

    arm = FakeArm(positions=[0.0] * 6)
    policy = ControllablePolicy(n=5)
    svc = _svc(
        config=_config(mode="async", fps=200.0, queue_threshold=4),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.wait_for(policy.second_call_started.wait(), timeout=2.0)
    # The 2nd inference is now hung in the background, independent of
    # whatever tick the loop task itself happens to be suspended on.
    await asyncio.sleep(0.05)
    await svc.do_command({"command": "stop"})

    assert svc._scheduler._inflight is None


# ---------------------------------------------------------------------------
# queue_threshold: derived default vs. explicit override (Task 22 follow-up).
# ---------------------------------------------------------------------------


async def test_async_mode_derives_queue_threshold_from_n_action_steps_when_unset():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(n=7)  # n_action_steps=7 -> derived threshold = 6
    svc = _svc(
        config=_config(mode="async", fps=50.0),  # queue_threshold left unset
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert svc._scheduler._queue_threshold == 6


async def test_async_mode_explicit_queue_threshold_overrides_the_derived_default():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(n=7)  # would derive 6 if left unset
    svc = _svc(
        config=_config(mode="async", fps=50.0, queue_threshold=2),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert svc._scheduler._queue_threshold == 2


async def test_async_mode_explicit_zero_threshold_is_honored_not_derived():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(n=7)
    svc = _svc(
        config=_config(mode="async", fps=50.0, queue_threshold=0),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    await svc.do_command({"command": "stop"})
    assert svc._scheduler._queue_threshold == 0


# ---------------------------------------------------------------------------
# status.starved_ticks: cumulative, distinct from the escalation counter.
# ---------------------------------------------------------------------------


async def test_status_reports_starved_ticks_zero_before_any_starvation():
    svc = _svc()
    assert (await svc.do_command({"command": "status"}))["starved_ticks"] == 0


async def test_status_starved_ticks_increments_while_holding_position():
    class StallingPolicy(FakePolicyClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._hang = asyncio.Event()

        async def do_command(self, command, **kwargs):
            if command.get("command") == "infer":
                self.infer_calls += 1
                if self.infer_calls > 1:
                    await self._hang.wait()  # never set: stalls forever
                chunk = np.full((self.n, self.action_dim), self.action_value, dtype=np.float32)
                return {
                    "actions": encode_matrix(chunk),
                    "raw_actions": encode_matrix(chunk),
                    "latency_s": 0.001,
                }
            return await super().do_command(command, **kwargs)

    arm = FakeArm(positions=[0.0] * 6)
    policy = StallingPolicy(n=3)
    svc = _svc(
        config=_config(
            mode="async",
            fps=100.0,
            queue_threshold=2,
            starvation_grace_ticks=1_000_000,  # never escalate -- only measure the counter
            safety={"max_start_delta_degs": 1000.0},
        ),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["starved_ticks"] > 0
    assert status["state"] == "running"  # not escalated -- the grace bound was never hit


async def test_starved_ticks_does_not_carry_over_across_a_restart():
    # Same reasoning as the existing "restart must not report the previous
    # run's telemetry" guarantee for avg_latency_s/measured_fps/clamp_counts
    # -- a fresh start() must not report yesterday's starved_ticks as if
    # they belonged to a run that has not produced a single tick yet.
    class StallingPolicy(FakePolicyClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._hang = asyncio.Event()

        async def do_command(self, command, **kwargs):
            if command.get("command") == "infer":
                self.infer_calls += 1
                if self.infer_calls > 1:
                    await self._hang.wait()
                chunk = np.full((self.n, self.action_dim), self.action_value, dtype=np.float32)
                return {
                    "actions": encode_matrix(chunk),
                    "raw_actions": encode_matrix(chunk),
                    "latency_s": 0.001,
                }
            return await super().do_command(command, **kwargs)

    arm = FakeArm(positions=[0.0] * 6)
    policy = StallingPolicy(n=3)
    cfg = _config(
        mode="async",
        fps=100.0,
        queue_threshold=2,
        starvation_grace_ticks=1_000_000,
        safety={"max_start_delta_degs": 1000.0},
    )
    svc = _svc(config=cfg, deps=_deps(policy=policy, arm=arm))

    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    first_run_status = await svc.do_command({"command": "status"})
    assert first_run_status["starved_ticks"] > 0
    await svc.do_command({"command": "stop"})

    # Restart the *same* controller instance directly (no reconfigure() in
    # between, which has its own reset) -- this isolates start()'s own
    # reset of starved_ticks specifically.
    policy.infer_calls = 0  # let the 2nd run's first inference succeed again
    await svc.do_command({"command": "start", "task": "t"})
    status = await svc.do_command({"command": "status"})
    assert status["starved_ticks"] == 0
    await svc.do_command({"command": "stop"})


async def test_sequential_mode_never_reports_starved_ticks():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(n=3)
    svc = _svc(deps=_deps(policy=policy, arm=arm))  # default mode="auto" -> sequential
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.1)
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["starved_ticks"] == 0


# ---------------------------------------------------------------------------
# End-to-end: the starvation-risk warning fires through the real controller.
# ---------------------------------------------------------------------------


async def test_async_mode_warns_when_queue_threshold_too_low_for_observed_latency(caplog):
    class SlowPolicy(FakePolicyClient):
        async def do_command(self, command, **kwargs):
            if command.get("command") == "infer":
                self.infer_calls += 1
                await asyncio.sleep(0.03)
                chunk = np.full((self.n, self.action_dim), self.action_value, dtype=np.float32)
                return {
                    "actions": encode_matrix(chunk),
                    "raw_actions": encode_matrix(chunk),
                    "latency_s": 0.001,
                }
            return await super().do_command(command, **kwargs)

    arm = FakeArm(positions=[0.0] * 6)
    # fps=100, latency~0.03s -> required = ceil(0.03*100) = 3, comfortably
    # above the deliberately-too-low threshold=0.
    policy = SlowPolicy(n=5)
    svc = _svc(
        config=_config(mode="async", fps=100.0, queue_threshold=0),
        deps=_deps(policy=policy, arm=arm),
    )
    with caplog.at_level(logging.WARNING, logger="vla.controller.scheduler"):
        await svc.do_command({"command": "start", "task": "t"})
        await _wait_for_state(svc, "running")
        await asyncio.sleep(0.3)
        await svc.do_command({"command": "stop"})

    messages = [r.message for r in caplog.records]
    assert any("queue_threshold=0" in m and "queue_threshold>=" in m for m in messages), messages
