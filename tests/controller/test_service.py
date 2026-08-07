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

import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig

from tests.fakes import FakeArm, FakeCamera, FakeGripper
from vla.controller.service import VLAController
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
    # exist. A single positions object, not a list wrapped in one.
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
    assert len(positions.values) == 5


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
    assert len(gripper.sent) >= 1
    assert gripper.sent[0] == [0.5]


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


async def test_stop_on_error_false_skips_the_tick_and_keeps_running():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    policy.fail_infer = True
    svc = _svc(
        config=_config(safety={"max_start_delta_degs": 1000.0, "stop_on_error": False}),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
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
        config=_config(safety={"max_start_delta_degs": 1000.0, "stop_on_error": False}),
        deps=_deps(policy=policy, arm=arm),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    policy.fail_infer = False
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    assert len(arm.moves) >= 1


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
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient(n=7)  # 7-step chunk
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await _wait_for_state(svc, "running")
    await asyncio.sleep(0.02)  # let exactly the first infer/merge happen
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    # After the very first infer, the queue holds n_action_steps - 1 (one
    # already consumed this tick) or n_action_steps, depending on timing --
    # either way it must reflect the scheduler, not a constant.
    assert 0 <= status["queue_size"] <= 7


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
    policy = FakePolicyClient(state="loading")
    svc = _svc(config=_config(mode="sequential"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    status = await svc.do_command({"command": "status"})
    assert status["mode"] == "sequential"
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
    # Generous bounds: real scheduling jitter under a test runner is
    # substantial, but it must land in the neighborhood of 20, not e.g. 2
    # (10x too slow) or 200 (10x too fast, suggesting no pacing at all).
    assert 5.0 < status["measured_fps"] < 60.0


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
