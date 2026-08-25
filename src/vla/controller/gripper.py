"""Adapters carrying a policy's continuous gripper channel onto Viam components.

A VLA emits one continuous gripper value per tick. Viam offers three
components that can carry it at different fidelity, so the config picks one
explicitly:

  arm_joint  gripper is joint N of the arm; value is a joint angle
             following ``action_units``, like every other joint.
  servo      get_position()/move(angle); both int degrees, so the adapter
             maps normalized 0..1 onto min_deg..max_deg at 1-degree
             resolution.
  gripper + mode=inputs      get_current_inputs()/go_to_inputs(). Both are a
                              *frame-system* interface, not an aperture
                              channel: one value per kinematic DOF, in
                              radians or meters. That only carries a usable
                              aperture when the driver's gripper model is
                              jointed -- most gripper models (built by
                              ``gripper.MakeModel``) are zero-DOF, so this
                              mode does not work against them. Preferred when
                              the driver implements both -- they are abstract
                              in the SDK, so not every driver does.
  gripper + mode=threshold   read via get_current_inputs() (same
                              frame-system/zero-DOF caveat as above), write
                              by thresholding the normalized value to
                              open()/grab(). Binary fallback for drivers that
                              do not implement go_to_inputs.
  none       no gripper channel.

Unit convention: ``arm_joint`` is in degrees (or whatever ``action_units``
is); every other variant is normalized 0.0-1.0, 0 = fully open, matching how
LeRobot datasets typically encode a gripper channel. That convention
describes what each *adapter* hands the controller -- a separate claim from
what the underlying driver call hands the adapter, which for
``get_current_inputs()``/``go_to_inputs()`` is the frame-system DOF vector
described above, not a normalized aperture.
"""

from __future__ import annotations

import abc
from typing import Any, Mapping

from vla.config_util import ConfigError, VLAError
from vla.config_util import as_choice as _as_choice
from vla.config_util import as_float as _as_float
from vla.config_util import as_int as _as_int
from vla.config_util import as_str as _as_str

GRIPPER_TYPES = ("arm_joint", "servo", "gripper", "do_command", "none")
GRIPPER_MODES = ("inputs", "threshold")

_DEFAULT_MIN_DEG = 0.0
_DEFAULT_MAX_DEG = 90.0
_DEFAULT_CLOSE_THRESHOLD = 0.5
_DEFAULT_READ_KEY = "position"


class GripperConfigError(VLAError, ValueError):
    """Raised for an invalid gripper config block."""


# config_util's coercion helpers raise `ConfigError`, not this module's own
# `GripperConfigError` -- wrap every call site so a caller catching
# `GripperConfigError` (or `VLAError`) never has to also know about
# `ConfigError` to cover every rejection path this module can raise.
def as_int(value: Any, field_name: str, *, minimum: int | float | None = None) -> int:
    try:
        return _as_int(value, field_name, minimum=minimum)
    except ConfigError as exc:
        raise GripperConfigError(str(exc)) from exc


def as_float(
    value: Any, field_name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    try:
        return _as_float(value, field_name, minimum=minimum, maximum=maximum)
    except ConfigError as exc:
        raise GripperConfigError(str(exc)) from exc


def as_str(value: Any, field_name: str) -> str:
    try:
        return _as_str(value, field_name)
    except ConfigError as exc:
        raise GripperConfigError(str(exc)) from exc


def as_choice(value: Any, field_name: str, allowed) -> str:
    try:
        return _as_choice(value, field_name, allowed)
    except ConfigError as exc:
        raise GripperConfigError(str(exc)) from exc


class GripperRuntimeError(VLAError, RuntimeError):
    """Raised when a gripper adapter cannot perform a requested operation
    at runtime -- as opposed to a config-time mistake.

    Two cases today: a driver whose ``go_to_inputs`` is unimplemented (it is
    an abstract SDK method, so this is a real possibility, not a
    hypothetical) -- a bare `NotImplementedError` bubbling up from deep
    inside the SDK is not an actionable error message, so this wraps it with
    the fix; and an empty ``get_current_inputs()`` from a zero-DOF gripper
    model, which carries no aperture to read at all.

    The second case is really a *static* property of the driver's kinematic
    model -- it will never spontaneously start working once seen -- but it
    can only be discovered at runtime, on the first ``await``ed read:
    ``make_gripper_adapter`` is synchronous and is handed the resolved
    component, not its DOF count, so nothing short of an actual call can
    tell config time apart from a working driver.
    """


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


async def _read_first_input(gripper: Any, name: str | None) -> float:
    """Read a gripper's aperture out of its frame-system inputs.

    ``get_current_inputs`` is a *frame system* interface, not an aperture
    channel: it returns one value per kinematic DOF, in radians or meters
    (``framesystem.InputEnabled`` in the RDK: "Input units are always in
    meters or radians"). ``gripper.MakeModel`` builds a zero-DOF model, so for
    most drivers the list is empty and carries no aperture at all. This used
    to fall back to 0.0, which reports a gripper permanently held fully open
    -- wrong, and invisible, for the entire life of the session. When a
    driver's model does have DOF, this module assumes index 0 is the
    aperture; a driver whose gripper model has more than one joint would
    have other slots that `values[0]` silently ignores.

    Known limitation: the value returned here is the driver's raw
    frame-system value, in radians or meters -- it is NOT normalized to
    0.0-1.0 the way every other adapter's `read()` is, even though
    `InputsGripper.write()` sends a normalized 0.0-1.0 value into this same
    channel. Fixing that would require configured radian/meter bounds for
    the joint; that was considered and declined because no driver we support
    actually has a jointed gripper model, so the surface would be
    speculative. `do_command` is the variant that normalizes honestly for a
    jointed driver.
    """
    values = await gripper.get_current_inputs()
    if not values:
        raise GripperRuntimeError(
            f"gripper {name!r} reports no kinematic DOF, so get_current_inputs() "
            "carries no aperture value; if the driver exposes a proportional "
            'DoCommand, use gripper.type="do_command"'
        )
    return float(values[0])


class GripperAdapter(abc.ABC):
    """Base for every gripper variant.

    ``in_state`` says whether this adapter's value participates in the
    controller's observation/action vector at all (`False` only for
    ``none``). ``dependency_name`` is the Viam resource name this adapter
    needs as a dependency, or `None` when the channel rides the arm
    (``arm_joint``) or does not exist (``none``).
    """

    in_state: bool = True
    uses_degrees: bool = False
    dependency_name: str | None = None
    arm_joint_index: int | None = None

    async def read(self) -> float:
        raise NotImplementedError

    async def write(self, value: float) -> None:
        raise NotImplementedError


class NoGripper(GripperAdapter):
    in_state = False

    async def read(self) -> float:  # pragma: no cover - never called
        raise GripperRuntimeError("no gripper configured; read() should never be called")

    async def write(self, value: float) -> None:  # pragma: no cover
        raise GripperRuntimeError("no gripper configured; write() should never be called")


class ArmJointGripper(GripperAdapter):
    """The arm already carries this channel; read/write ride the arm call.

    This adapter has no `read`/`write` of its own -- the controller reads
    and writes this joint index as part of the arm's own joint vector.
    """

    uses_degrees = True

    def __init__(self, joint_index: int) -> None:
        self.arm_joint_index = joint_index


class ServoGripper(GripperAdapter):
    def __init__(self, name: str, servo: Any, min_deg: float, max_deg: float) -> None:
        if max_deg <= min_deg:
            raise GripperConfigError(
                f"gripper.max_deg must exceed gripper.min_deg, got "
                f"min_deg={min_deg!r} max_deg={max_deg!r}"
            )
        self.dependency_name = name
        self._servo = servo
        self._min = min_deg
        self._max = max_deg

    async def read(self) -> float:
        deg = float(await self._servo.get_position())
        return (deg - self._min) / (self._max - self._min)

    async def write(self, value: float) -> None:
        clamped = _clamp_unit(value)
        await self._servo.move(int(round(self._min + clamped * (self._max - self._min))))


class InputsGripper(GripperAdapter):
    """The symmetric ``get_current_inputs``/``go_to_inputs`` pair."""

    def __init__(self, name: str, gripper: Any) -> None:
        self.dependency_name = name
        self._gripper = gripper

    async def read(self) -> float:
        return await _read_first_input(self._gripper, self.dependency_name)

    async def write(self, value: float) -> None:
        try:
            await self._gripper.go_to_inputs([_clamp_unit(value)])
        except NotImplementedError as exc:
            raise GripperRuntimeError(
                f"gripper {self.dependency_name!r} does not implement go_to_inputs; "
                'reconfigure gripper.mode to "threshold" for this driver'
            ) from exc


class ThresholdGripper(GripperAdapter):
    """Binary fallback for drivers without ``go_to_inputs``."""

    def __init__(self, name: str, gripper: Any, close_threshold: float) -> None:
        self.dependency_name = name
        self._gripper = gripper
        self._threshold = close_threshold
        # `None`, not `False`: the physical state on startup is unknown, so
        # the first write must always command explicitly rather than being
        # skipped because it happens to match an assumed initial state.
        self._closed: bool | None = None

    async def read(self) -> float:
        return await _read_first_input(self._gripper, self.dependency_name)

    async def write(self, value: float) -> None:
        should_close = float(value) >= self._threshold
        if should_close == self._closed:
            return  # avoid re-commanding the same state every tick
        self._closed = should_close
        if should_close:
            await self._gripper.grab()
        else:
            await self._gripper.open()


class DoCommandGripper(GripperAdapter):
    """Proportional control through ``DoCommand``, for drivers that expose it
    there rather than through the typed API.

    Contract: any gripper whose ``DoCommand`` implements ``{"get": True}`` ->
    ``{read_key: number}`` and ``{"set": number}``. Both
    ``devrel:so101:gripper`` and ``viam:ufactory:gripper`` do, with the same
    request shape -- they differ only in the read response key and the value
    range.

    ``open_value``/``closed_value`` are the driver's own native values at the
    two extremes: percent for so-101 (95 / 0), raw units for xarm (840 / 2).
    Expressing the mapping in terms of the two *named endpoints* rather than a
    min/max ordering is what lets it carry a driver whose scale runs opposite
    to this module's ``0.0 = fully open`` convention -- which both of those
    drivers do, since for them a higher number means more open.
    """

    def __init__(
        self,
        name: str,
        gripper: Any,
        open_value: float,
        closed_value: float,
        read_key: str,
        write_args: Mapping[str, Any],
    ) -> None:
        if open_value == closed_value:
            raise GripperConfigError(
                "gripper.open_value and gripper.closed_value must differ, both are "
                f"{open_value!r}"
            )
        self.dependency_name = name
        self._gripper = gripper
        self._open = open_value
        self._closed = closed_value
        self._read_key = read_key
        self._write_args = dict(write_args)


def make_gripper_adapter(
    raw: Mapping[str, Any] | None, dependencies: Mapping[str, Any]
) -> GripperAdapter:
    """Build the configured gripper adapter.

    ``raw`` is the ``gripper`` block from controller config (already a plain
    dict from ``struct_to_dict``); `None` or omitted means ``{"type":
    "none"}``. ``dependencies`` maps resource name -> resolved Viam
    component, exactly as the resource layer hands them to `reconfigure`.
    """
    raw = raw or {"type": "none"}
    if not isinstance(raw, Mapping):
        raise GripperConfigError(f"gripper must be an object, got {raw!r}")

    kind = raw.get("type", "none")
    if kind not in GRIPPER_TYPES:
        raise GripperConfigError(
            f"gripper.type must be one of {GRIPPER_TYPES}, got {kind!r}"
        )

    if kind != "gripper" and "close_threshold" in raw:
        raise GripperConfigError(
            'close_threshold is only valid with gripper.type="gripper" '
            f'mode="threshold" (got gripper.type={kind!r})'
        )

    if kind == "none":
        return NoGripper()

    if kind == "arm_joint":
        if "joint_index" not in raw:
            raise GripperConfigError('gripper.type="arm_joint" requires joint_index')
        joint_index = as_int(raw["joint_index"], "gripper.joint_index", minimum=0)
        return ArmJointGripper(joint_index)

    name = raw.get("name")
    if not name:
        raise GripperConfigError(f"gripper.type={kind!r} requires name")
    name = as_str(name, "gripper.name")

    if kind == "servo":
        min_deg = as_float(raw.get("min_deg", _DEFAULT_MIN_DEG), "gripper.min_deg")
        max_deg = as_float(raw.get("max_deg", _DEFAULT_MAX_DEG), "gripper.max_deg")
        return ServoGripper(name, dependencies.get(name), min_deg, max_deg)

    if kind == "do_command":
        for field in ("open_value", "closed_value"):
            if field not in raw:
                raise GripperConfigError(
                    f'gripper.type="do_command" requires {field}; it is the driver-native '
                    "value at that extreme (so-101: 95/0 percent, xarm: 840/2 raw units). "
                    "There is no safe default -- a percentage guess silently saturates a "
                    "raw-unit driver in the first percent of its travel."
                )
        open_value = as_float(raw["open_value"], "gripper.open_value")
        closed_value = as_float(raw["closed_value"], "gripper.closed_value")
        read_key = as_str(raw.get("read_key", _DEFAULT_READ_KEY), "gripper.read_key")
        write_args = raw.get("write_args") or {}
        return DoCommandGripper(
            name, dependencies.get(name), open_value, closed_value, read_key, write_args
        )

    # kind == "gripper"
    mode = as_choice(raw.get("mode", "inputs"), "gripper.mode", GRIPPER_MODES)
    if mode == "inputs":
        if "close_threshold" in raw:
            raise GripperConfigError(
                'close_threshold is only valid with gripper.mode="threshold"'
            )
        return InputsGripper(name, dependencies.get(name))

    close_threshold = as_float(
        raw.get("close_threshold", _DEFAULT_CLOSE_THRESHOLD),
        "gripper.close_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    return ThresholdGripper(name, dependencies.get(name), close_threshold)
