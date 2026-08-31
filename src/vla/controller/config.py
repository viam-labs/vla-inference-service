"""Configuration parsing and validation for viam-labs:vla:controller.

Every numeric/choice field goes through `vla.config_util`'s `as_int`/
`as_float`/`as_choice` with its range folded into the call: protobuf `Struct`
delivers every number as a double, so each field needs the same "accept 2.0,
reject 2.5, reject True, reject NaN" treatment.

Two fields carry non-obvious semantics:

  - `max_vel_degs_per_sec` is the operator-facing velocity knob;
    `max_joint_delta_degs` is derivable from it as `max_vel_degs_per_sec /
    fps`, and the safety layer's per-tick delta clamp is what actually
    enforces it (there is no `MoveOptions` path in any released viam-sdk).
    Both may appear in config, but if both are given they must agree --
    silently preferring one over a contradictory other would hide an
    operator mistake on a safety-critical limit.

  - `queue_threshold: None` means "derive it", not "zero". The right value is
    a property of the checkpoint (`n_action_steps`), which config parsing
    cannot see, so `VLAController._build_scheduler` derives `n_action_steps -
    1` once `specs` is known. A sentinel of `0` could not be distinguished
    from an operator who genuinely wants `0`.

`state_units`/`action_units` change *shape* with `action_space`. Under
`"joints"` (the default, and unchanged) they are a single unit string
validated against `units.SUPPORTED_UNITS`, so a "normalized" config fails here
with the field name rather than surviving to the first control tick. Under
`"delta-ee"` they are per-segment objects parsed into a `units.VectorUnits`,
because those vectors mix millimetres with rotation components and no single
string describes them.

`action_space: "delta-ee"` also *rejects* three things it could have ignored:
`state_joint_indices`, a non-`none` `gripper`, and every joint-space `safety`
key. Each would be inert on that path, and an inert safety limit or an inert
state layout is the kind of config an operator reasonably assumes is in
force. Rejection is symmetric: the Cartesian `safety` keys are refused under
`"joints"` for the same reason.

`mode: "async"` overlaps execution with inference instead of stalling the arm
between chunks -- for the case where inference latency approaches or exceeds
chunk duration, where RTC cannot help (it needs `delay < chunk_length`). It is
explicit opt-in: `"auto"` still resolves to `"sequential"`, so an existing
deployment's behavior never changes underneath it. See `AsyncScheduler`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vla.config_util import ConfigError, as_bool, as_choice, as_float, as_int, as_str

from .gripper import GRIPPER_TYPES, GRIPPER_TYPES_NEEDING_DEPENDENCY
from .observation import DEFAULT_DURATION_WARN_S, STALE_FRAME_WARN_S
from .units import (
    ANGLE_UNITS,
    LENGTH_UNITS,
    SUPPORTED_UNITS,
    UnitSegment,
    VectorUnits,
)

__all__ = [
    "ConfigError",
    "SafetyConfig",
    "ControllerConfig",
    "MODES",
    "ENCODINGS",
    "IMAGE_FITS",
    "ACTION_SPACES",
    "JOINTS",
    "DELTA_EE",
]

MODES = ("auto", "sequential", "rtc", "async")
ENCODINGS = ("jpeg", "png", "raw")

# The two action spaces. `joints` is the original and the default: absolute
# joint angles in degrees, written with `move_to_joint_positions`. `delta-ee`
# carries a per-tick end-effector pose delta, composed onto the measured
# `EndPosition` and written with `move_to_position`. They differ in the
# observation they build, the arm call they make, the safety layer that
# clamps them, and the shape of their unit config -- and in nothing else:
# the scheduler, action queue, and RTC/async modes are shared.
JOINTS = "joints"
DELTA_EE = "delta-ee"
ACTION_SPACES = (JOINTS, DELTA_EE)

# "pad" mirrors smolvla's own training-time convention (resize_with_pad,
# lerobot/policies/common/vla_utils.py:219): scale to fit inside the declared
# (h, w) preserving aspect ratio, then pad black on the LEFT and TOP.
# "stretch" is the plain `Image.resize` this module used before that
# convention was matched -- kept as the non-default choice so an existing
# deployment can reproduce its pre-fix output. See observation.py's `_encode`
# and the README's `#controller` section for the measured divergence.
# "stretch_bicubic" is the same geometry as "stretch" with EVO1's resampler,
# and is the default under `action_space="delta-ee"`; see `_default_image_fit`.
IMAGE_FITS = ("pad", "stretch", "stretch_bicubic")

# Sanity ceilings/floors, not physical limits: they turn an obvious typo (a
# negative fps, a fractional queue_threshold) into a config-time error.
_MIN_FPS = 1e-6
_MAX_FPS = 1000.0
_MIN_POSITIVE_DEG = 1e-6
_MIN_POSITIVE_MM = 1e-6
_MIN_POSITIVE_RAD = 1e-9


# Which `safety` keys belong to which action space. Keys are rejected rather
# than ignored when they appear under the wrong one: a `joint_limits_degs`
# silently doing nothing on the `move_to_position` path is exactly the kind of
# safety config an operator would assume is in force.
_JOINTS_ONLY_SAFETY_KEYS = (
    "max_joint_delta_degs",
    "max_start_delta_degs",
    "max_vel_degs_per_sec",
    "joint_limits_degs",
)
_DELTA_EE_ONLY_SAFETY_KEYS = (
    "max_tcp_delta_mm",
    "max_tcp_rot_delta_rads",
    "max_tcp_vel_mms_per_sec",
    "max_tcp_rot_vel_rads_per_sec",
)


@dataclass(frozen=True)
class SafetyConfig:
    max_joint_delta_degs: float = 8.0
    max_start_delta_degs: float = 15.0
    max_vel_degs_per_sec: float | None = None
    joint_limits_degs: list[tuple[float, float]] | None = None
    stop_on_error: bool = True
    # delta-ee only; see CartesianLimits for where the defaults come from.
    max_tcp_delta_mm: float = 40.0
    max_tcp_rot_delta_rads: float = 0.12
    max_tcp_vel_mms_per_sec: float | None = None
    max_tcp_rot_vel_rads_per_sec: float | None = None

    @staticmethod
    def parse(raw: dict[str, Any], *, fps: float, action_space: str = JOINTS) -> "SafetyConfig":
        if not isinstance(raw, dict):
            raise ConfigError(f"safety must be an object, got {raw!r}")

        wrong_space = (
            _DELTA_EE_ONLY_SAFETY_KEYS if action_space == JOINTS else _JOINTS_ONLY_SAFETY_KEYS
        )
        for key in wrong_space:
            if raw.get(key) is not None:
                raise ConfigError(
                    f"safety.{key} does not apply to action_space={action_space!r} and "
                    "would be silently ignored; remove it"
                )

        stop_on_error = as_bool(raw.get("stop_on_error", True), "safety.stop_on_error")
        if action_space == DELTA_EE:
            return SafetyConfig._parse_cartesian(raw, fps=fps, stop_on_error=stop_on_error)

        max_vel = None
        if raw.get("max_vel_degs_per_sec") is not None:
            max_vel = as_float(
                raw["max_vel_degs_per_sec"],
                "safety.max_vel_degs_per_sec",
                minimum=_MIN_POSITIVE_DEG,
            )

        declared_delta = None
        if raw.get("max_joint_delta_degs") is not None:
            declared_delta = as_float(
                raw["max_joint_delta_degs"],
                "safety.max_joint_delta_degs",
                minimum=_MIN_POSITIVE_DEG,
            )

        return SafetyConfig(
            max_joint_delta_degs=_resolve_per_tick(
                max_vel,
                declared_delta,
                fps,
                default=8.0,
                delta_field="safety.max_joint_delta_degs",
                vel_field="safety.max_vel_degs_per_sec",
            ),
            max_start_delta_degs=as_float(
                raw.get("max_start_delta_degs", 15.0), "safety.max_start_delta_degs", minimum=0.0
            ),
            max_vel_degs_per_sec=max_vel,
            joint_limits_degs=_parse_joint_limits(raw.get("joint_limits_degs")),
            stop_on_error=stop_on_error,
        )

    @staticmethod
    def _parse_cartesian(
        raw: dict[str, Any], *, fps: float, stop_on_error: bool
    ) -> "SafetyConfig":
        """The delta-EE half: two magnitude ceilings, each velocity-derivable.

        Structurally identical to the joints half -- an operator-facing speed
        knob, a per-tick ceiling derived from it, and a refusal if both are
        given and disagree -- because the per-tick clamp *is* the speed limit
        in both action spaces. There is no `max_start_*` counterpart: a
        delta-EE action is already a delta, so the first tick's magnitude is
        the same quantity every later tick's clamp measures, and a second
        threshold on it would be the same check under a second name.
        """
        max_tcp_vel = None
        if raw.get("max_tcp_vel_mms_per_sec") is not None:
            max_tcp_vel = as_float(
                raw["max_tcp_vel_mms_per_sec"],
                "safety.max_tcp_vel_mms_per_sec",
                minimum=_MIN_POSITIVE_MM,
            )
        declared_tcp_delta = None
        if raw.get("max_tcp_delta_mm") is not None:
            declared_tcp_delta = as_float(
                raw["max_tcp_delta_mm"], "safety.max_tcp_delta_mm", minimum=_MIN_POSITIVE_MM
            )

        max_rot_vel = None
        if raw.get("max_tcp_rot_vel_rads_per_sec") is not None:
            max_rot_vel = as_float(
                raw["max_tcp_rot_vel_rads_per_sec"],
                "safety.max_tcp_rot_vel_rads_per_sec",
                minimum=_MIN_POSITIVE_RAD,
            )
        declared_rot_delta = None
        if raw.get("max_tcp_rot_delta_rads") is not None:
            declared_rot_delta = as_float(
                raw["max_tcp_rot_delta_rads"],
                "safety.max_tcp_rot_delta_rads",
                minimum=_MIN_POSITIVE_RAD,
            )

        return SafetyConfig(
            stop_on_error=stop_on_error,
            max_tcp_delta_mm=_resolve_per_tick(
                max_tcp_vel,
                declared_tcp_delta,
                fps,
                default=40.0,
                delta_field="safety.max_tcp_delta_mm",
                vel_field="safety.max_tcp_vel_mms_per_sec",
            ),
            max_tcp_rot_delta_rads=_resolve_per_tick(
                max_rot_vel,
                declared_rot_delta,
                fps,
                default=0.12,
                delta_field="safety.max_tcp_rot_delta_rads",
                vel_field="safety.max_tcp_rot_vel_rads_per_sec",
            ),
            max_tcp_vel_mms_per_sec=max_tcp_vel,
            max_tcp_rot_vel_rads_per_sec=max_rot_vel,
        )


def _resolve_per_tick(
    max_vel: float | None,
    declared_delta: float | None,
    fps: float,
    *,
    default: float,
    delta_field: str,
    vel_field: str,
) -> float:
    """Resolve a per-tick ceiling from whichever knob the operator gave.

    The velocity knob is authoritative when both are given, but only after
    confirming the two agree. Both arrive through as_float (already finite
    doubles), so the tolerances below only need to cover ordinary
    floating-point division error, not operator rounding.

    Parameterized by field name because both action spaces need exactly this
    rule -- the per-tick clamp is the velocity limit in each -- and two copies
    would be two chances for the agreement check to drift.
    """
    if max_vel is None:
        return default if declared_delta is None else declared_delta

    derived_delta = max_vel / fps
    if declared_delta is None:
        return derived_delta

    if not math.isclose(declared_delta, derived_delta, rel_tol=1e-6, abs_tol=1e-6):
        raise ConfigError(
            f"{delta_field}={declared_delta} is inconsistent with "
            f"{vel_field} / fps = {max_vel} / {fps} = {derived_delta}; "
            f"{vel_field.split('.')[-1]} is the operator-facing knob -- omit "
            f"{delta_field.split('.')[-1]}, or set it to match the derived value"
        )
    return derived_delta


def _parse_joint_limits(limits: Any) -> list[tuple[float, float]] | None:
    if limits is None:
        return None
    if not isinstance(limits, list):
        raise ConfigError(f"safety.joint_limits_degs must be a list, got {limits!r}")
    parsed: list[tuple[float, float]] = []
    for i, pair in enumerate(limits):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigError(f"safety.joint_limits_degs[{i}] must be [min, max]")
        lo = as_float(pair[0], f"safety.joint_limits_degs[{i}][0]")
        hi = as_float(pair[1], f"safety.joint_limits_degs[{i}][1]")
        if hi <= lo:
            raise ConfigError(
                f"safety.joint_limits_degs[{i}]: min {lo} must be below max {hi}"
            )
        parsed.append((lo, hi))
    return parsed


def _parse_state_joint_indices(raw: Any, *, delta_ee: bool) -> list[int]:
    """Parse `state_joint_indices`: required for joints, forbidden for delta-EE.

    Rejected rather than ignored under delta-EE. The field's whole job is to
    say which arm joints the observation vector is built from, and a delta-EE
    observation is built from `EndPosition` instead -- an operator who left it
    behind while switching action spaces is describing a state layout that no
    longer exists, and would otherwise get no signal at all.
    """
    if delta_ee:
        if raw:
            raise ConfigError(
                f"state_joint_indices does not apply to action_space={DELTA_EE!r}: the "
                "observation is built from the arm's EndPosition, not from selected joints. "
                "Remove it."
            )
        return []

    indices_raw = raw or []
    if not indices_raw:
        raise ConfigError("state_joint_indices is required and must be non-empty")
    indices = [
        as_int(v, f"state_joint_indices[{i}]", minimum=0) for i, v in enumerate(indices_raw)
    ]
    if len(set(indices)) != len(indices):
        raise ConfigError(f"state_joint_indices contains duplicate entries: {indices}")
    return indices


def _default_image_fit(action_space: str) -> str:
    """The `image_fit` an action space gets when the operator names none.

    Different because the two policy families resize differently *inside* the
    policy, and the controller's job is to hand each one the geometry its
    training data had.

    smolvla pads: `resize_with_pad` runs on the dataset frame at train time,
    so a controller that stretched a 16:9 camera into a square declared shape
    would present geometry training never saw. Hence "pad".

    EVO1 does not pad. `_batched_resize_01`
    (lerobot/policies/evo1/internvl3_embedder.py) resizes straight to
    `(image_size, image_size)`, bicubic with antialiasing, explicitly
    mirroring InternVL3's reference PIL preprocessing -- so training saw the
    full dataset frame squashed to a square, with no bars. Padding here would
    feed the policy black borders that then get squashed along with the
    picture. Hence a stretch, with the bicubic resampler the policy itself
    uses so a controller-side resize compounds as little as possible.
    """
    return "stretch_bicubic" if action_space == DELTA_EE else "pad"


# The delta-EE vector layouts, fixed by the dataset contract and not
# configurable: `observation.state` is 3 position components followed by the
# first two ROWS of the tool's rotation matrix, and `action` is a translation
# delta followed by a body-frame axis-angle rotation delta. Only the *units*
# of those segments vary between checkpoints, which is what `state_units` and
# `action_units` describe here.
DELTA_EE_STATE_DIM = 9
DELTA_EE_ACTION_DIM = 6


def _parse_segment_units(
    raw: Any, field_name: str, *, rotation_units: tuple[str, ...], rotation_default: str
) -> VectorUnits:
    """Parse a delta-EE `state_units`/`action_units` block into a `VectorUnits`.

    Rejects the plain-string form the joints path uses. A delta-EE vector
    mixes lengths with rotation components, so a single unit string cannot
    describe it -- and silently reading `"radians"` as "the whole vector" is
    how a translation in millimetres gets multiplied by pi/180.
    """
    if raw is None:
        raw = {}
    if isinstance(raw, str):
        raise ConfigError(
            f"{field_name} must be an object under action_space={DELTA_EE!r} (got the string "
            f"{raw!r}): the vector mixes lengths with rotation components, so one unit cannot "
            'describe it -- use e.g. {"translation": "millimeters", "rotation": '
            f'"{rotation_default}"}}'
        )
    if not isinstance(raw, dict):
        raise ConfigError(f"{field_name} must be an object, got {raw!r}")
    unknown = sorted(set(raw) - {"translation", "rotation"})
    if unknown:
        raise ConfigError(
            f"{field_name} has unknown key(s) {unknown}; expected 'translation' and 'rotation'"
        )

    translation = as_choice(
        raw.get("translation", "millimeters"), f"{field_name}.translation", LENGTH_UNITS
    )
    rotation = as_choice(
        raw.get("rotation", rotation_default), f"{field_name}.rotation", rotation_units
    )
    rotation_size = DELTA_EE_STATE_DIM - 3 if rotation_units == ("unitless",) else 3
    return VectorUnits((UnitSegment(3, translation), UnitSegment(rotation_size, rotation)))


@dataclass(frozen=True)
class ControllerConfig:
    policy_service: str
    arm: str
    cameras: dict[str, str]
    state_joint_indices: list[int]
    gripper: dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    task: str = ""
    fps: float = 10.0
    mode: str = "auto"
    queue_threshold: int | None = None
    starvation_grace_ticks: int = 3
    policy_ready_timeout_s: int = 600
    # A plain unit string under action_space="joints" (a joint vector is one
    # quantity end to end), a per-segment `VectorUnits` under "delta-ee" (its
    # vectors mix lengths with rotation components). See units.py.
    state_units: str | VectorUnits = "degrees"
    action_units: str | VectorUnits = "degrees"
    image_encoding: str = "jpeg"
    jpeg_quality: int = 90
    image_fit: str = "pad"
    action_space: str = JOINTS
    # Operator-configurable rather than fixed module defaults: a checkpoint
    # run at 2 Hz and one at 10 Hz imply very different "this tick is late"
    # and "this frame is stale" thresholds. Defaults match observation.py's
    # own constants, so an operator who never sets these sees no change.
    duration_warn_s: float = DEFAULT_DURATION_WARN_S
    stale_frame_warn_s: float = STALE_FRAME_WARN_S
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @staticmethod
    def parse(raw: dict[str, Any]) -> "ControllerConfig":
        policy_service = raw.get("policy_service")
        if not policy_service:
            raise ConfigError("policy_service is required")
        policy_service = as_str(policy_service, "policy_service")

        arm = raw.get("arm")
        if not arm:
            raise ConfigError("arm is required")
        arm = as_str(arm, "arm")

        cameras_raw = raw.get("cameras") or {}
        if not isinstance(cameras_raw, dict) or not cameras_raw:
            raise ConfigError("cameras must name at least one camera")
        cameras = {
            as_str(k, "cameras key"): as_str(v, f"cameras[{k!r}]")
            for k, v in cameras_raw.items()
        }

        action_space = as_choice(raw.get("action_space", JOINTS), "action_space", ACTION_SPACES)
        delta_ee = action_space == DELTA_EE

        indices = _parse_state_joint_indices(raw.get("state_joint_indices"), delta_ee=delta_ee)

        fps = as_float(raw.get("fps", 10.0), "fps", minimum=_MIN_FPS, maximum=_MAX_FPS)

        gripper = dict(raw.get("gripper") or {"type": "none"})
        gripper_type = as_choice(gripper.get("type", "none"), "gripper.type", GRIPPER_TYPES)
        gripper["type"] = gripper_type
        if delta_ee and gripper_type != "none":
            # The recording captured EndPosition and six joint values, no
            # gripper component and no seventh joint, so the checkpoint has no
            # jaw channel to carry. Configuring an adapter here would make
            # _check_action_dim demand a 7th action dimension the policy never
            # emits; drive the jaws out of band instead.
            raise ConfigError(
                f"gripper.type={gripper_type!r} is not supported under "
                f"action_space={DELTA_EE!r}: the action space has no gripper channel. "
                'Set gripper.type="none" and command the gripper separately.'
            )

        safety = SafetyConfig.parse(
            raw.get("safety") or {}, fps=fps, action_space=action_space
        )

        # The trailing gripper limit pair exists only when the gripper channel
        # is itself in degrees, which is exactly the arm_joint case -- every
        # other gripper variant is normalized [0, 1] and gets its own clamp in
        # safety.py, not a degree-shaped joint limit.
        if safety.joint_limits_degs is not None:
            expected = len(indices) + (1 if gripper_type == "arm_joint" else 0)
            if len(safety.joint_limits_degs) != expected:
                raise ConfigError(
                    f"safety.joint_limits_degs has {len(safety.joint_limits_degs)} entries, "
                    f"expected {expected} (one per action dimension in degrees)"
                )

        return ControllerConfig(
            policy_service=policy_service,
            arm=arm,
            cameras=cameras,
            state_joint_indices=indices,
            gripper=gripper,
            task=as_str(raw.get("task", ""), "task"),
            fps=fps,
            mode=as_choice(raw.get("mode", "auto"), "mode", MODES),
            queue_threshold=(
                as_int(raw["queue_threshold"], "queue_threshold", minimum=0)
                if raw.get("queue_threshold") is not None
                else None
            ),
            starvation_grace_ticks=as_int(
                raw.get("starvation_grace_ticks", 3), "starvation_grace_ticks", minimum=0
            ),
            policy_ready_timeout_s=as_int(
                raw.get("policy_ready_timeout_s", 600), "policy_ready_timeout_s", minimum=1
            ),
            state_units=(
                _parse_segment_units(
                    raw.get("state_units"),
                    "state_units",
                    # The rotation half of the state is two rows of a rotation
                    # matrix -- direction cosines, dimensionless by
                    # construction. It is spelled out as a one-choice field
                    # rather than hidden so that an operator who writes
                    # "radians" here gets told it is wrong instead of having
                    # it quietly ignored.
                    rotation_units=("unitless",),
                    rotation_default="unitless",
                )
                if delta_ee
                else as_choice(raw.get("state_units", "degrees"), "state_units", SUPPORTED_UNITS)
            ),
            action_units=(
                _parse_segment_units(
                    raw.get("action_units"),
                    "action_units",
                    rotation_units=ANGLE_UNITS,
                    rotation_default="radians",
                )
                if delta_ee
                else as_choice(
                    raw.get("action_units", "degrees"), "action_units", SUPPORTED_UNITS
                )
            ),
            image_encoding=as_choice(raw.get("image_encoding", "jpeg"), "image_encoding", ENCODINGS),
            jpeg_quality=as_int(raw.get("jpeg_quality", 90), "jpeg_quality", minimum=0, maximum=100),
            image_fit=as_choice(
                raw.get("image_fit", _default_image_fit(action_space)), "image_fit", IMAGE_FITS
            ),
            action_space=action_space,
            duration_warn_s=as_float(
                raw.get("duration_warn_s", DEFAULT_DURATION_WARN_S), "duration_warn_s", minimum=0.0
            ),
            stale_frame_warn_s=as_float(
                raw.get("stale_frame_warn_s", STALE_FRAME_WARN_S), "stale_frame_warn_s", minimum=0.0
            ),
            safety=safety,
        )

    def dependencies(self) -> list[str]:
        deps = [self.policy_service, self.arm, *self.cameras.values()]
        name = self.gripper.get("name")
        if self.gripper.get("type") in GRIPPER_TYPES_NEEDING_DEPENDENCY and name:
            deps.append(name)
        return deps
