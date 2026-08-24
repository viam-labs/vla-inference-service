import re

import pytest

from vla.config_util import VLAError
from vla.controller.gripper import (
    GripperConfigError,
    GripperRuntimeError,
    make_gripper_adapter,
)
from tests.fakes import FakeGripper, FakeServo

# ---------------------------------------------------------------------------
# none
# ---------------------------------------------------------------------------


def test_none_adapter_contributes_nothing():
    a = make_gripper_adapter({"type": "none"}, {})
    assert a.in_state is False
    assert a.dependency_name is None


def test_missing_gripper_block_defaults_to_none():
    a = make_gripper_adapter(None, {})
    assert a.in_state is False


# ---------------------------------------------------------------------------
# arm_joint
# ---------------------------------------------------------------------------


def test_arm_joint_adapter_is_carried_by_the_arm():
    a = make_gripper_adapter({"type": "arm_joint", "joint_index": 5}, {})
    assert a.in_state is True
    assert a.dependency_name is None  # no separate resource
    assert a.uses_degrees is True
    assert a.arm_joint_index == 5


def test_arm_joint_requires_index():
    with pytest.raises(GripperConfigError, match="joint_index"):
        make_gripper_adapter({"type": "arm_joint"}, {})


def test_arm_joint_accepts_protobuf_double_index():
    # Struct delivers every number as a double -- 5.0, never plain 5.
    a = make_gripper_adapter({"type": "arm_joint", "joint_index": 5.0}, {})
    assert a.arm_joint_index == 5


def test_arm_joint_rejects_fractional_index():
    with pytest.raises(GripperConfigError, match="joint_index"):
        make_gripper_adapter({"type": "arm_joint", "joint_index": 5.5}, {})


def test_arm_joint_rejects_negative_index():
    with pytest.raises(GripperConfigError, match="joint_index"):
        make_gripper_adapter({"type": "arm_joint", "joint_index": -1}, {})


def test_arm_joint_index_zero_is_valid():
    # 0 is falsy in Python -- a `not raw.get("joint_index")` check would
    # wrongly reject the first joint.
    a = make_gripper_adapter({"type": "arm_joint", "joint_index": 0}, {})
    assert a.arm_joint_index == 0


# ---------------------------------------------------------------------------
# servo: unit convention (normalized 0..1 -> min_deg..max_deg)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_angle",
    [(0.0, 0), (0.5, 45), (1.0, 90)],
)
async def test_servo_write_maps_normalized_range(value, expected_angle):
    servo = FakeServo()
    a = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {"s": servo}
    )
    await a.write(value)
    assert servo.moves == [expected_angle]


@pytest.mark.parametrize(
    "angle, expected_value",
    [(0, 0.0), (45, 0.5), (90, 1.0)],
)
async def test_servo_read_maps_to_normalized_range(angle, expected_value):
    servo = FakeServo(angle=angle)
    a = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {"s": servo}
    )
    assert await a.read() == pytest.approx(expected_value)


async def test_servo_adapter_writes_denormalized_int():
    servo = FakeServo()
    a = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {"s": servo}
    )
    await a.write(0.25)
    assert servo.moves == [22]  # int(round(0.25 * 90)) with 1-degree resolution


async def test_servo_write_clamps_above_range():
    servo = FakeServo()
    a = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {"s": servo}
    )
    await a.write(5.0)
    assert servo.moves == [90]


async def test_servo_write_clamps_below_range():
    servo = FakeServo()
    a = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {"s": servo}
    )
    await a.write(-5.0)
    assert servo.moves == [0]


def test_servo_uses_degrees_is_false():
    # Servo carries a normalized 0..1 value, not a degree value, even though
    # it is denormalized onto min_deg..max_deg internally.
    a = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {}
    )
    assert a.uses_degrees is False


def test_servo_requires_name():
    with pytest.raises(GripperConfigError, match="name"):
        make_gripper_adapter({"type": "servo", "min_deg": 0, "max_deg": 90}, {})


def test_servo_rejects_inverted_range():
    with pytest.raises(GripperConfigError, match="max_deg"):
        make_gripper_adapter(
            {"type": "servo", "name": "s", "min_deg": 90, "max_deg": 0}, {}
        )


def test_servo_min_max_defaults():
    servo = FakeServo(angle=45)
    a = make_gripper_adapter({"type": "servo", "name": "s"}, {"s": servo})
    # Documented defaults: min_deg=0, max_deg=90.
    import asyncio

    assert asyncio.run(a.read()) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# gripper + mode=inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
async def test_gripper_inputs_mode_roundtrip_at_normalized_boundaries(value):
    g = FakeGripper(inputs=[value])
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "inputs"}, {"g": g})
    assert await a.read() == pytest.approx(value)
    await a.write(value)
    assert g.sent == [[value]]


async def test_gripper_inputs_mode_roundtrip():
    g = FakeGripper(inputs=[0.3])
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "inputs"}, {"g": g})
    assert await a.read() == pytest.approx(0.3)
    await a.write(0.8)
    assert g.sent == [[0.8]]


async def test_gripper_inputs_mode_write_clamps_above_range():
    g = FakeGripper()
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "inputs"}, {"g": g})
    await a.write(1.5)
    assert g.sent == [[1.0]]


async def test_gripper_inputs_mode_write_clamps_below_range():
    g = FakeGripper()
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "inputs"}, {"g": g})
    await a.write(-1.5)
    assert g.sent == [[0.0]]


def test_gripper_inputs_uses_degrees_is_false():
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "inputs"}, {})
    assert a.uses_degrees is False


def test_gripper_mode_defaults_to_inputs():
    a = make_gripper_adapter({"type": "gripper", "name": "g"}, {"g": FakeGripper()})
    from vla.controller.gripper import InputsGripper

    assert isinstance(a, InputsGripper)


async def test_gripper_inputs_mode_unimplemented_go_to_inputs_raises_clear_error():
    # go_to_inputs is abstract in the real SDK -- not every driver implements
    # it. Surface a clear error naming the threshold fallback, not a bare
    # NotImplementedError bubbling up from deep inside the SDK.
    g = FakeGripper(supports_inputs=False)
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "inputs"}, {"g": g})
    with pytest.raises(GripperRuntimeError, match="threshold"):
        await a.write(0.5)


async def test_gripper_inputs_mode_unimplemented_error_names_the_driver():
    g = FakeGripper(supports_inputs=False)
    a = make_gripper_adapter({"type": "gripper", "name": "grip1", "mode": "inputs"}, {"grip1": g})
    with pytest.raises(GripperRuntimeError, match="grip1"):
        await a.write(0.5)


def test_gripper_requires_name():
    with pytest.raises(GripperConfigError, match="name"):
        make_gripper_adapter({"type": "gripper", "mode": "inputs"}, {})


def test_gripper_rejects_unknown_mode():
    with pytest.raises(GripperConfigError, match="mode"):
        make_gripper_adapter({"type": "gripper", "name": "g", "mode": "wat"}, {})


# ---------------------------------------------------------------------------
# gripper + mode=threshold
# ---------------------------------------------------------------------------


async def test_gripper_threshold_mode_opens_below_threshold():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g}
    )
    await a.write(0.2)
    assert g.opened == 1
    assert g.grabbed == 0


async def test_gripper_threshold_mode_grabs_above_threshold():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g}
    )
    await a.write(0.9)
    assert g.grabbed == 1


async def test_gripper_threshold_mode_at_exact_threshold_closes():
    # >= threshold closes -- boundary decision, documented and tested.
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g}
    )
    await a.write(0.5)
    assert g.grabbed == 1
    assert g.opened == 0


async def test_threshold_mode_does_not_resend_same_state():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g}
    )
    await a.write(0.9)
    await a.write(0.95)
    assert g.grabbed == 1  # already closed; no redundant command
    assert g.opened == 0


async def test_threshold_mode_does_not_resend_same_open_state():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g}
    )
    await a.write(0.1)
    await a.write(0.2)
    await a.write(0.0)
    assert g.opened == 1
    assert g.grabbed == 0


async def test_threshold_mode_first_write_always_commands_even_when_staying_open():
    # The adapter's internal notion of "current state" starts unknown
    # (`None`), regardless of what the fake happens to report. A mutant
    # that initializes it to `False` instead would make this call a no-op,
    # since should_close (False) would spuriously equal the "already open"
    # default -- and the arm would never be explicitly commanded open on
    # startup even though nothing has told it to be open yet.
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g}
    )
    await a.write(0.1)  # below threshold -> stays open, but must still command
    assert g.opened == 1


async def test_threshold_mode_resends_on_state_change():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g}
    )
    await a.write(0.9)  # close
    await a.write(0.1)  # open
    await a.write(0.9)  # close again
    assert g.grabbed == 2
    assert g.opened == 1


def test_threshold_uses_degrees_is_false():
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {}
    )
    assert a.uses_degrees is False


def test_close_threshold_default_is_half():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "threshold"}, {"g": g})
    import asyncio

    asyncio.run(a.write(0.5))
    assert g.grabbed == 1


# ---------------------------------------------------------------------------
# validate(): close_threshold constraints
# ---------------------------------------------------------------------------


def test_close_threshold_rejected_outside_threshold_mode():
    with pytest.raises(GripperConfigError, match="close_threshold"):
        make_gripper_adapter(
            {"type": "gripper", "name": "g", "mode": "inputs", "close_threshold": 0.5}, {}
        )


def test_close_threshold_rejected_for_servo_type():
    with pytest.raises(GripperConfigError, match="close_threshold"):
        make_gripper_adapter(
            {"type": "servo", "name": "s", "close_threshold": 0.5}, {}
        )


def test_close_threshold_rejected_for_arm_joint_type():
    with pytest.raises(GripperConfigError, match="close_threshold"):
        make_gripper_adapter(
            {"type": "arm_joint", "joint_index": 5, "close_threshold": 0.5}, {}
        )


def test_close_threshold_rejected_for_none_type():
    with pytest.raises(GripperConfigError, match="close_threshold"):
        make_gripper_adapter({"type": "none", "close_threshold": 0.5}, {})


@pytest.mark.parametrize("bad", [-0.1, 1.1, -1.0, 2.0])
def test_close_threshold_rejected_outside_unit_range(bad):
    with pytest.raises(GripperConfigError, match="close_threshold"):
        make_gripper_adapter(
            {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": bad}, {}
        )


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_close_threshold_accepted_at_unit_range_boundaries(value):
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": value}, {}
    )
    assert a is not None  # constructs without raising


# ---------------------------------------------------------------------------
# top-level validation
# ---------------------------------------------------------------------------


def test_unknown_type_errors():
    with pytest.raises(GripperConfigError, match="unknown|one of"):
        make_gripper_adapter({"type": "tentacle"}, {})


def test_gripper_config_error_is_a_vla_error():
    assert issubclass(GripperConfigError, VLAError)
    assert issubclass(GripperConfigError, ValueError)


def test_gripper_runtime_error_is_a_vla_error():
    assert issubclass(GripperRuntimeError, VLAError)
    assert issubclass(GripperRuntimeError, RuntimeError)


# ---------------------------------------------------------------------------
# fixture guards -- these two fakes silently swallowed the exact inputs the
# tests below depend on, so they are pinned here.
# ---------------------------------------------------------------------------


def test_fake_gripper_preserves_an_explicitly_empty_inputs_list():
    """A zero-DOF gripper model reports `[]`, which is a meaningful fixture.

    `list(inputs or [0.0])` collapsed it to `[0.0]`, making the empty-inputs
    refusal in `InputsGripper`/`ThresholdGripper` untestable.
    """
    assert FakeGripper(inputs=[]).inputs == []
    assert FakeGripper().inputs == [0.0]  # default unchanged


# ---------------------------------------------------------------------------
# zero-DOF get_current_inputs() refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        {"type": "gripper", "name": "g", "mode": "inputs"},
        {"type": "gripper", "name": "g", "mode": "threshold"},
    ],
    ids=["inputs", "threshold"],
)
async def test_zero_dof_gripper_read_refuses_instead_of_reporting_zero(block):
    """A zero-DOF gripper model reports no inputs at all, so there is no
    aperture to read. Reporting 0.0 looks like a gripper held fully open."""
    adapter = make_gripper_adapter(block, {"g": FakeGripper(inputs=[])})
    with pytest.raises(GripperRuntimeError, match="no kinematic DOF"):
        await adapter.read()


async def test_zero_dof_refusal_names_the_working_alternative():
    adapter = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "inputs"}, {"g": FakeGripper(inputs=[])}
    )
    with pytest.raises(GripperRuntimeError, match=re.escape('gripper.type="do_command"')):
        await adapter.read()


async def test_nonempty_inputs_still_read_normally():
    adapter = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "inputs"}, {"g": FakeGripper(inputs=[0.25])}
    )
    assert await adapter.read() == pytest.approx(0.25)
