import re

import pytest

from vla.config_util import VLAError
from vla.controller.gripper import (
    GripperConfigError,
    GripperRuntimeError,
    make_gripper_adapter,
)
from tests.fakes import FakeDoCommandGripper, FakeServo

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
# do_command
# ---------------------------------------------------------------------------
def _do_cmd(gripper, **overrides):
    block = {
        "type": "do_command",
        "name": "grip",
        "open_value": 95.0,
        "closed_value": 0.0,
        **overrides,
    }
    return make_gripper_adapter(block, {"grip": gripper})


async def test_fake_do_command_gripper_round_trips():
    g = FakeDoCommandGripper(position=42.0)
    assert await g.do_command({"get": True}) == {"position": 42.0}
    await g.do_command({"set": 7.0})
    assert g.commands == [{"get": True}, {"set": 7.0}]
    assert await g.do_command({"get": True}) == {"position": 7.0}


async def test_fake_do_command_gripper_honors_a_custom_read_key():
    g = FakeDoCommandGripper(position=3.0, read_key="pos")
    assert await g.do_command({"get": True}) == {"pos": 3.0}
    assert await g.do_command({"set": 9.0}) == {"pos": 9.0}


def test_do_command_adapter_wires_its_attributes():
    a = _do_cmd(FakeDoCommandGripper())
    assert a.in_state is True
    assert a.uses_degrees is False
    assert a.dependency_name == "grip"
    assert a.arm_joint_index is None


@pytest.mark.parametrize("missing", ["open_value", "closed_value"])
def test_do_command_requires_both_bounds(missing):
    block = {"type": "do_command", "name": "grip", "open_value": 95.0, "closed_value": 0.0}
    del block[missing]
    with pytest.raises(GripperConfigError, match=missing):
        make_gripper_adapter(block, {"grip": FakeDoCommandGripper()})


def test_do_command_rejects_equal_bounds():
    """Equal endpoints make the read mapping a division by zero."""
    with pytest.raises(GripperConfigError, match="must differ"):
        _do_cmd(FakeDoCommandGripper(), open_value=50.0, closed_value=50.0)


def test_do_command_requires_a_name():
    with pytest.raises(GripperConfigError, match="name"):
        make_gripper_adapter(
            {"type": "do_command", "open_value": 95.0, "closed_value": 0.0}, {}
        )


@pytest.mark.parametrize(
    "raw,expected",
    [(95.0, 0.0), (0.0, 1.0), (47.5, 0.5)],
    ids=["open-rail", "closed-rail", "midpoint"],
)
async def test_do_command_read_normalizes_an_inverted_scale(raw, expected):
    """so-101 counts *up* toward open; this module's 0.0 means fully open."""
    adapter = _do_cmd(FakeDoCommandGripper(position=raw))
    assert await adapter.read() == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,expected", [(840.0, 0.0), (2.0, 1.0), (421.0, 0.5)], ids=["open", "closed", "mid"]
)
async def test_do_command_read_handles_xarm_raw_units(raw, expected):
    adapter = _do_cmd(
        FakeDoCommandGripper(position=raw, read_key="pos"),
        read_key="pos",
        open_value=840.0,
        closed_value=2.0,
    )
    assert await adapter.read() == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,expected", [(98.0, 0.0), (-5.0, 1.0)], ids=["open-rail", "closed-rail"]
)
async def test_do_command_read_clamps_just_past_a_rail(raw, expected):
    """Not defensive: so-101's openPosition is 95 but the servo travels to 100,
    so an ordinary reading of 98 maps to -0.03 unclamped and puts an
    out-of-range value into the state vector the policy sees. Both of these are
    inside the slack band, so they clamp rather than being refused."""
    adapter = _do_cmd(FakeDoCommandGripper(position=raw))
    assert await adapter.read() == expected


async def test_do_command_read_uses_the_configured_read_key():
    adapter = _do_cmd(FakeDoCommandGripper(position=95.0, read_key="pos"), read_key="pos")
    assert await adapter.read() == pytest.approx(0.0)


async def test_do_command_read_errors_when_the_key_is_absent():
    """The fake answers under `some_other_key`; the adapter is configured for
    the default `position`, so the key it wants is missing. The message must
    name both the key the driver actually returned (`some_other_key`) and the
    key the adapter was configured to read (`position`)."""
    adapter = _do_cmd(FakeDoCommandGripper(read_key="some_other_key"))
    with pytest.raises(GripperRuntimeError, match="some_other_key"):
        await adapter.read()
    with pytest.raises(GripperRuntimeError, match="position"):
        await adapter.read()


async def test_do_command_read_errors_on_a_non_mapping_response():
    """`FakeDoCommandGripper` structurally always returns a dict, so it cannot
    reach the `got = res` branch (non-Mapping response). A driver whose
    `DoCommand` returns nothing -- `None` -- is the realistic case."""

    class NoneReturningGripper:
        async def do_command(self, command, *, timeout=None, **kwargs):
            return None

    adapter = _do_cmd(NoneReturningGripper())
    with pytest.raises(GripperRuntimeError, match="got None"):
        await adapter.read()


@pytest.mark.parametrize(
    "raw", ["halfway", True, None], ids=["string", "bool", "json-null"]
)
async def test_do_command_read_errors_on_a_non_numeric_value(raw):
    """`bool` needs excluding explicitly -- `isinstance(True, int)` is True in
    Python, so a driver returning True would otherwise read as fully closed.
    `None` is the likeliest real malformed response (JSON null)."""
    adapter = _do_cmd(FakeDoCommandGripper(position=raw))
    with pytest.raises(GripperRuntimeError, match="non-numeric"):
        await adapter.read()


async def test_do_command_read_emits_the_get_command():
    g = FakeDoCommandGripper(position=95.0)
    await _do_cmd(g).read()
    assert g.commands == [{"get": True}]


@pytest.mark.parametrize(
    "raw",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "-inf"],
)
async def test_do_command_read_refuses_a_non_finite_reading(raw):
    """`_clamp_unit` would turn these into a fabricated rail reading -- nan and
    +inf both land on 0.0, i.e. "confidently fully open" -- so they are refused
    instead. Matches how `config_util.as_float` already treats non-finite
    config values."""
    adapter = _do_cmd(FakeDoCommandGripper(position=raw))
    with pytest.raises(GripperRuntimeError, match="non-finite"):
        await adapter.read()


@pytest.mark.parametrize(
    "raw", [840.0, -100.0], ids=["far-above-open", "far-below-closed"]
)
async def test_do_command_read_refuses_a_grossly_out_of_range_reading(raw):
    """The clamp absorbs calibration slop, not a mis-scaled endpoint pair. A
    driver reporting raw units against a percent config would otherwise report
    a frozen rail forever."""
    adapter = _do_cmd(FakeDoCommandGripper(position=raw))
    with pytest.raises(GripperRuntimeError, match="do not describe this driver"):
        await adapter.read()


async def test_do_command_read_handles_a_non_inverted_scale():
    """No real driver we target counts down toward open, but the named-endpoint
    formulation claims to carry one with no special case. Pin the claim."""
    adapter = _do_cmd(
        FakeDoCommandGripper(position=25.0), open_value=0.0, closed_value=100.0
    )
    assert await adapter.read() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# do_command: write()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected", [(0.0, 95.0), (1.0, 0.0), (0.5, 47.5)], ids=["open", "closed", "mid"]
)
async def test_do_command_write_maps_onto_the_driver_scale(value, expected):
    g = FakeDoCommandGripper()
    await _do_cmd(g).write(value)
    assert g.commands == [{"set": pytest.approx(expected)}]


@pytest.mark.parametrize("value,expected", [(-0.2, 95.0), (1.7, 0.0)])
async def test_do_command_write_clamps_out_of_range_actions(value, expected):
    g = FakeDoCommandGripper()
    await _do_cmd(g).write(value)
    assert g.commands == [{"set": pytest.approx(expected)}]


async def test_do_command_write_merges_write_args():
    g = FakeDoCommandGripper()
    await _do_cmd(g, write_args={"wait": False}).write(1.0)
    assert g.commands == [{"set": pytest.approx(0.0), "wait": False}]


async def test_do_command_write_omits_extras_by_default():
    g = FakeDoCommandGripper()
    await _do_cmd(g).write(1.0)
    assert list(g.commands[0]) == ["set"]


@pytest.mark.parametrize(
    "bad_args",
    [{"set": 12.0}, {"get": True}, {"get": True, "set": 12.0}, {"wait": False, "get": True}],
    ids=["set", "get", "both", "get-alongside-valid"],
)
def test_do_command_rejects_reserved_keys_in_write_args(bad_args):
    """Both halves of the {"get": ...} / {"set": ...} protocol are reserved. A
    `set` entry would replace the computed setpoint; a `get` entry makes a
    driver that checks it first (FakeDoCommandGripper does) treat every write as
    a read. Both leave the gripper silently not tracking the policy.
    """
    with pytest.raises(GripperConfigError, match="protocol keys"):
        _do_cmd(FakeDoCommandGripper(), write_args=bad_args)


@pytest.mark.parametrize(
    "bad", [[("wait", False)], [], 0, False, "", "wait", 42], ids=
    ["pairs-list", "empty-list", "zero", "false", "empty-string", "string", "int"]
)
def test_do_command_rejects_non_mapping_write_args(bad):
    """The falsy cases matter most: an earlier draft used `or {}`, which folded
    them into `{}` silently, so a misspelled write_args block was dropped with
    nothing raised."""
    with pytest.raises(GripperConfigError, match="write_args"):
        _do_cmd(FakeDoCommandGripper(), write_args=bad)


@pytest.mark.parametrize(
    "open_value,closed_value",
    [(95.0, 0.0), (840.0, 2.0), (0.0, 100.0)],
    ids=["so101-percent", "xarm-raw-units", "non-inverted"],
)
async def test_do_command_round_trips_through_the_driver(open_value, closed_value):
    """write() then read() lands back on the value written, on every scale --
    including a non-inverted one, which no real driver uses but the
    endpoint-named formulation claims to carry with no special case."""
    g = FakeDoCommandGripper()
    adapter = _do_cmd(g, open_value=open_value, closed_value=closed_value)
    await adapter.write(0.25)
    assert await adapter.read() == pytest.approx(0.25)
