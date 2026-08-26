"""Adapters carrying a policy's continuous gripper channel onto Viam components.

A VLA emits one continuous gripper value per tick. Viam offers several ways
to carry it at different fidelity, so the config picks one explicitly:

  arm_joint   gripper is joint N of the arm; value is a joint angle
              following ``action_units``, like every other joint.
  servo       get_position()/move(angle); both int degrees, so the adapter
              maps normalized 0..1 onto min_deg..max_deg at 1-degree
              resolution.
  do_command  proportional control through DoCommand, for drivers that
              expose it there instead of the typed API
              (``devrel:so101:gripper``, ``viam:ufactory:gripper``).
  none        no gripper channel.

Unit convention: ``arm_joint`` is in degrees (or whatever ``action_units``
is); ``servo`` and ``do_command`` are normalized 0.0-1.0, 0 = fully open,
matching how LeRobot datasets typically encode a gripper channel.

The typed ``Gripper`` API's ``get_current_inputs``/``go_to_inputs`` pair is
deliberately not supported. It is a *frame-system* interface -- one value per
kinematic DOF, in radians or meters -- and ``gripper.MakeModel`` builds a
zero-DOF model, so it carries no aperture at all for any driver here.
``do_command`` is the variant that works against a real proportional gripper.
"""

from __future__ import annotations

import abc
import math
from typing import Any, Mapping

from vla.config_util import ConfigError, VLAError, as_choice, as_float, as_int, as_str

GRIPPER_TYPES = ("arm_joint", "servo", "do_command", "none")

# Variants that resolve their own Viam resource, and so must be named in
# `ControllerConfig.dependencies()`. Lives here rather than in `config.py`
# because it is a fact about the adapters: `dependencies()` has to answer it
# before any adapter exists, so it cannot read `dependency_name` off an
# instance, but it can import the set from the module that owns the variants.
GRIPPER_TYPES_NEEDING_DEPENDENCY = frozenset({"servo", "do_command"})

_DEFAULT_MIN_DEG = 0.0
_DEFAULT_MAX_DEG = 90.0
_DEFAULT_READ_KEY = "position"

# The coercion helpers already raise ConfigError, which is what a bad gripper
# block is. Aliasing rather than defining a second class keeps `except
# GripperConfigError` working at every existing call site without a layer of
# wrappers whose only job was to re-label the exception.
GripperConfigError = ConfigError


class GripperRuntimeError(VLAError, RuntimeError):
    """Raised when a gripper adapter cannot perform a requested operation at
    runtime -- as opposed to a config-time mistake.

    Today this is `DoCommandGripper`'s read path: a driver whose DoCommand
    does not answer with the configured key, answers with a non-numeric or
    non-finite value, or reports a position far outside the configured
    endpoints. Each is only discoverable on an actual awaited call, and a
    bare KeyError/TypeError from deep inside is not an actionable message.
    """


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


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

    @property
    def has_normalized_tail(self) -> bool:
        """Whether this adapter contributes a trailing 0.0-1.0 channel.

        True for every variant whose value is normalized (`servo`,
        `do_command`); False for `arm_joint`, whose channel rides the arm's
        joint vector in ``action_units``, and for `none`, which contributes
        nothing.

        This exists so the fact has one name. It was previously re-derived at
        four call sites in three spellings, two of them De Morgan negations
        of each other -- a variant that set the underlying flags
        inconsistently would have made the safety layer and the tick loop
        silently disagree about units. That is the bug class the write-side
        conversion split exists to prevent; naming the predicate once is
        what keeps it prevented.
        """
        return self.in_state and self.arm_joint_index is None

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
        # Deliberately unvalidated and unclamped, unlike `DoCommandGripper.read`:
        # a servo past its configured `max_deg` returns >1.0 and a driver
        # returning NaN propagates, both straight into the observation vector.
        # This predates the guarded adapter below and is left alone rather than
        # changed here -- but do not read the density of guards one class down
        # as a module-wide discipline. It is not one.
        deg = float(await self._servo.get_position())
        return (deg - self._min) / (self._max - self._min)

    async def write(self, value: float) -> None:
        clamped = _clamp_unit(value)
        await self._servo.move(int(round(self._min + clamped * (self._max - self._min))))


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

    # How far outside [0, 1] a reading may fall before it is refused rather
    # than clamped. The clamp absorbs *calibration* slop -- so-101 declares
    # open at 95 while the servo travels to 100, ~5% of span -- which is small
    # by definition. A reading far outside the band means the configured
    # endpoints do not describe this driver's scale at all (the classic case:
    # so-101's 95/0 percent copied onto a driver reporting raw units), and
    # silently saturating it freezes the policy's gripper channel at a rail
    # forever. A class attribute rather than a module constant because this is
    # the only adapter that validates its reads at all -- `ServoGripper`
    # deliberately does not, so a module-level name would overstate its reach.
    _READ_SLACK = 0.25

    def __init__(
        self,
        name: str,
        gripper: Any,
        *,
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
        # Checked here rather than in the factory because it is an invariant of
        # what `write()` does with `**self._write_args`, not a statement about
        # config shape -- the module's rule is cross-argument invariants in
        # __init__, raw-config coercion in the factory. Direct construction gets
        # it too this way.
        reserved = sorted({"get", "set"} & write_args.keys())
        if reserved:
            raise GripperConfigError(
                f"gripper.write_args must not contain {', '.join(repr(k) for k in reserved)}: "
                'these are the adapter\'s own protocol keys. A "set" entry would replace the '
                'setpoint computed from the policy\'s action; a "get" entry makes a driver '
                "that checks it first treat every write as a read. Either way the gripper "
                "silently stops tracking the policy."
            )
        self.dependency_name = name
        self._gripper = gripper
        self._open_value = open_value
        self._closed_value = closed_value
        self._read_key = read_key
        self._write_args = dict(write_args)

    async def read(self) -> float:
        res = await self._gripper.do_command({"get": True})
        if not isinstance(res, Mapping) or self._read_key not in res:
            got = sorted(res) if isinstance(res, Mapping) else res
            raise GripperRuntimeError(
                f"gripper {self.dependency_name!r} do_command({{'get': True}}) returned "
                f"no {self._read_key!r} key; got {got!r}. Set gripper.read_key to the key "
                "this driver actually returns."
            )
        value = res[self._read_key]
        # `isinstance(True, int)` is True, so bool has to be excluded by hand or a
        # driver returning a flag under this key reads as a legitimate position.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GripperRuntimeError(
                f"gripper {self.dependency_name!r} returned a non-numeric "
                f"{self._read_key!r}: {value!r} ({type(value).__name__})"
            )
        # Kept for its message, not for safety: the slack-band check below
        # already refuses every non-finite reading (nan fails both comparisons;
        # +/-inf normalize to -/+inf and fail too), so _clamp_unit can never see
        # one. But the band's message would read "normalizes to nan", which is
        # actively misleading, where this one names the actual fault.
        if not math.isfinite(value):
            raise GripperRuntimeError(
                f"gripper {self.dependency_name!r} returned a non-finite "
                f"{self._read_key!r}: {value!r}. A non-finite reading would clamp to a "
                "fabricated endpoint and report a confidently wrong aperture to the "
                "policy, so it is refused rather than silently normalized."
            )
        normalized = (float(value) - self._open_value) / (
            self._closed_value - self._open_value
        )
        if not -self._READ_SLACK <= normalized <= 1.0 + self._READ_SLACK:
            raise GripperRuntimeError(
                f"gripper {self.dependency_name!r} reported {self._read_key}={value!r}, "
                f"which normalizes to {normalized:.3f} -- far outside [0, 1]. "
                f"gripper.open_value={self._open_value!r}/closed_value="
                f"{self._closed_value!r} do not describe this driver's scale; "
                "check them against the driver's actual range."
            )
        # Clamped, not refused, inside the band: the endpoints are a
        # *calibration*, not a hard travel limit -- so-101 calls 95 fully open
        # while the servo reaches 100, so an ordinary reading of 98 is a real
        # measurement slightly outside a declared range, not an error.
        return _clamp_unit(normalized)

    async def write(self, value: float) -> None:
        # No validation here, unlike read(): both callers guarantee a finite,
        # in-range input. The tick loop passes float(safe[-1]) from
        # SafetyLayer.apply, which validates and raises SafetyError on any
        # non-finite value in the action vector; _preflight_gripper writes back
        # what read() just returned, already refused-if-non-finite and clamped.
        # The _clamp_unit below is a backstop on an already-clamped value, not
        # a validation point -- do not turn it into one.
        raw = self._open_value + _clamp_unit(value) * (
            self._closed_value - self._open_value
        )
        await self._gripper.do_command({"set": raw, **self._write_args})


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

    kind = as_choice(raw.get("type", "none"), "gripper.type", GRIPPER_TYPES)

    if kind == "none":
        return NoGripper()

    if kind == "arm_joint":
        if "joint_index" not in raw:
            raise GripperConfigError('gripper.type="arm_joint" requires joint_index')
        return ArmJointGripper(as_int(raw["joint_index"], "gripper.joint_index", minimum=0))

    name = raw.get("name")
    if not name:
        raise GripperConfigError(f"gripper.type={kind!r} requires name")
    name = as_str(name, "gripper.name")

    if kind == "servo":
        return ServoGripper(
            name,
            dependencies.get(name),
            as_float(raw.get("min_deg", _DEFAULT_MIN_DEG), "gripper.min_deg"),
            as_float(raw.get("max_deg", _DEFAULT_MAX_DEG), "gripper.max_deg"),
        )

    # kind == "do_command"
    missing = [field for field in ("open_value", "closed_value") if field not in raw]
    if missing:
        raise GripperConfigError(
            f'gripper.type="do_command" requires {", ".join(missing)}; it is the '
            "driver-native value at that extreme (so-101 open/closed: 95/0 percent, "
            "xarm: 840/2 raw units). There is no safe default -- a percentage guess "
            "silently saturates a raw-unit driver in the first percent of its travel."
        )
    write_args = raw.get("write_args", {})
    if not isinstance(write_args, Mapping):
        raise GripperConfigError(f"gripper.write_args must be an object, got {write_args!r}")
    return DoCommandGripper(
        name,
        dependencies.get(name),
        open_value=as_float(raw["open_value"], "gripper.open_value"),
        closed_value=as_float(raw["closed_value"], "gripper.closed_value"),
        read_key=as_str(raw.get("read_key", _DEFAULT_READ_KEY), "gripper.read_key"),
        write_args=write_args,
    )
