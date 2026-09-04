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
    a property of the checkpoint (`n_action_steps`) or of
    `actions_per_chunk`, neither of which config parsing can see, so
    `VLAController._build_scheduler` derives `effective_chunk - 1` once
    `specs` is known. A sentinel of `0` could not be distinguished from an
    operator who genuinely wants `0`.

  - `actions_per_chunk: None` means "execute the whole chunk". An int
    truncates every chunk to its first N actions before it is queued,
    mirroring lerobot's server-side `actions_per_chunk`
    (`async_inference/policy_server.py`, `chunk[:, : self.actions_per_chunk, :]`).
    A checkpoint whose `chunk_size` covers 5s of motion is 5s of open loop;
    truncating re-observes every N ticks instead. N is floored by inference
    latency -- below `latency * fps` the queue cannot refill in time -- so
    this is only useful with `mode: "async"`, where `sequential`'s
    per-refill stall would otherwise dominate the duty cycle.

`state_units`/`action_units` validate against `units.SUPPORTED_UNITS`, so a
"normalized" config fails here with the field name rather than surviving to
the first control tick.

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
from .units import SUPPORTED_UNITS

__all__ = [
    "ConfigError",
    "SafetyConfig",
    "ControllerConfig",
    "MODES",
    "ENCODINGS",
    "IMAGE_FITS",
    "DEFAULT_ARM_MOVE_EXTRA",
]

MODES = ("auto", "sequential", "rtc", "async")
ENCODINGS = ("jpeg", "png", "raw")
# "pad" mirrors smolvla's own training-time convention (resize_with_pad,
# lerobot/policies/common/vla_utils.py:219): scale to fit inside the declared
# (h, w) preserving aspect ratio, then pad black on the LEFT and TOP.
# "stretch" is the plain `Image.resize` this module used before that
# convention was matched -- kept as the non-default choice so an existing
# deployment can reproduce its pre-fix output. See observation.py's `_encode`
# and the README's `#controller` section for the measured divergence.
IMAGE_FITS = ("pad", "stretch")

# `extra` sent with every `move_to_joint_positions`. The control loop issues a
# new setpoint every tick and the next one supersedes this one, so the driver
# must return WITHOUT waiting for the arm to physically settle. Drivers spell
# that differently and silently ignore keys they do not know, so all three
# spellings ship together rather than making the operator pick:
#
#   wait          so-101's key. Its driver blocks until settled by default.
#   waitAtEnd     ufactory xArm's key (viam-ufactory-xarm arm/xarm.go:701-703).
#                 `wait` is NOT in its parsed set (speed_r, speed_d,
#                 acceleration_r, acceleration_d, direct, waitAtEnd,
#                 interpolate), so it was dropped and the default `true`
#                 applied: every tick ran the client-side interpolation to
#                 completion and then polled GetState every 10ms until the arm
#                 stopped (comm.go:1009-1030). Measured on an xArm at fps=10:
#                 ticks of 0.26-1.49s against a 0.100s budget, i.e. the arm
#                 replayed a 10Hz trajectory at roughly 2Hz.
#   interpolate   also xArm's. `waitAtEnd: false` alone still runs the
#                 client-side interpolation loop, which costs one 1/move_hz
#                 sleep per intermediate step and so still scales with the
#                 delta. `false` sends the goal as a single servo setpoint.
#                 Safe here because safety.py already clamps every per-tick
#                 delta to `max_vel_degs_per_sec / fps`, so a setpoint can
#                 never ask for more motion than one tick's worth.
#
# Override wholesale via the `arm_move_extra` config key for a driver that
# wants something else (xArm also accepts `direct: true` for point-to-point
# instead of servo mode). An empty dict restores stock blocking behavior.
DEFAULT_ARM_MOVE_EXTRA: dict[str, Any] = {
    "wait": False,
    "waitAtEnd": False,
    "interpolate": False,
}


def _parse_arm_move_extra(raw: Any) -> dict[str, Any]:
    """Validate `arm_move_extra`, defaulting to `DEFAULT_ARM_MOVE_EXTRA`.

    Absent means "use the default"; `{}` is a real, distinct choice meaning
    "send nothing", which restores whatever blocking behavior the driver
    defaults to. Keys must be strings because they cross a protobuf Struct.

    Deliberately NOT merged with the default: a driver needing a different
    motion mode may have to *remove* a key, and a merge would make that
    impossible. Whatever is configured is exactly what gets sent.
    """
    if raw is None:
        return dict(DEFAULT_ARM_MOVE_EXTRA)
    if not isinstance(raw, dict):
        raise ConfigError(f"arm_move_extra must be an object, got {type(raw).__name__}")
    bad = sorted(k for k in raw if not isinstance(k, str))
    if bad:
        raise ConfigError(f"arm_move_extra keys must be strings, got {bad}")
    return dict(raw)


# Sanity ceilings/floors, not physical limits: they turn an obvious typo (a
# negative fps, a fractional queue_threshold) into a config-time error.
_MIN_FPS = 1e-6
_MAX_FPS = 1000.0
_MIN_POSITIVE_DEG = 1e-6


@dataclass(frozen=True)
class SafetyConfig:
    max_joint_delta_degs: float = 8.0
    max_start_delta_degs: float = 15.0
    max_vel_degs_per_sec: float | None = None
    joint_limits_degs: list[tuple[float, float]] | None = None
    stop_on_error: bool = True

    @staticmethod
    def parse(raw: dict[str, Any], *, fps: float) -> "SafetyConfig":
        if not isinstance(raw, dict):
            raise ConfigError(f"safety must be an object, got {raw!r}")

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
            max_joint_delta_degs=SafetyConfig._resolve_max_joint_delta(
                max_vel, declared_delta, fps
            ),
            max_start_delta_degs=as_float(
                raw.get("max_start_delta_degs", 15.0), "safety.max_start_delta_degs", minimum=0.0
            ),
            max_vel_degs_per_sec=max_vel,
            joint_limits_degs=_parse_joint_limits(raw.get("joint_limits_degs")),
            stop_on_error=as_bool(raw.get("stop_on_error", True), "safety.stop_on_error"),
        )

    @staticmethod
    def _resolve_max_joint_delta(
        max_vel: float | None, declared_delta: float | None, fps: float
    ) -> float:
        """Resolve max_joint_delta_degs from whichever knob the operator gave.

        max_vel_degs_per_sec is authoritative when both are given, but only
        after confirming the two agree. Both arrive through as_float (already
        finite doubles), so the tolerances below only need to cover ordinary
        floating-point division error, not operator rounding.
        """
        if max_vel is None:
            return 8.0 if declared_delta is None else declared_delta

        derived_delta = max_vel / fps
        if declared_delta is None:
            return derived_delta

        if not math.isclose(declared_delta, derived_delta, rel_tol=1e-6, abs_tol=1e-6):
            raise ConfigError(
                f"safety.max_joint_delta_degs={declared_delta} is inconsistent with "
                f"safety.max_vel_degs_per_sec / fps = {max_vel} / {fps} = {derived_delta}; "
                "max_vel_degs_per_sec is the operator-facing knob -- omit "
                "max_joint_delta_degs, or set it to match the derived value"
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
    actions_per_chunk: int | None = None
    starvation_grace_ticks: int = 3
    policy_ready_timeout_s: int = 600
    state_units: str = "degrees"
    action_units: str = "degrees"
    image_encoding: str = "jpeg"
    jpeg_quality: int = 90
    image_fit: str = "pad"
    arm_move_extra: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_ARM_MOVE_EXTRA)
    )
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

        indices_raw = raw.get("state_joint_indices") or []
        if not indices_raw:
            raise ConfigError("state_joint_indices is required and must be non-empty")
        indices = [
            as_int(v, f"state_joint_indices[{i}]", minimum=0)
            for i, v in enumerate(indices_raw)
        ]
        if len(set(indices)) != len(indices):
            raise ConfigError(f"state_joint_indices contains duplicate entries: {indices}")

        fps = as_float(raw.get("fps", 10.0), "fps", minimum=_MIN_FPS, maximum=_MAX_FPS)

        gripper = dict(raw.get("gripper") or {"type": "none"})
        gripper_type = as_choice(gripper.get("type", "none"), "gripper.type", GRIPPER_TYPES)
        gripper["type"] = gripper_type

        safety = SafetyConfig.parse(raw.get("safety") or {}, fps=fps)

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
            actions_per_chunk=(
                as_int(raw["actions_per_chunk"], "actions_per_chunk", minimum=1)
                if raw.get("actions_per_chunk") is not None
                else None
            ),
            starvation_grace_ticks=as_int(
                raw.get("starvation_grace_ticks", 3), "starvation_grace_ticks", minimum=0
            ),
            policy_ready_timeout_s=as_int(
                raw.get("policy_ready_timeout_s", 600), "policy_ready_timeout_s", minimum=1
            ),
            state_units=as_choice(raw.get("state_units", "degrees"), "state_units", SUPPORTED_UNITS),
            action_units=as_choice(
                raw.get("action_units", "degrees"), "action_units", SUPPORTED_UNITS
            ),
            image_encoding=as_choice(raw.get("image_encoding", "jpeg"), "image_encoding", ENCODINGS),
            jpeg_quality=as_int(raw.get("jpeg_quality", 90), "jpeg_quality", minimum=0, maximum=100),
            image_fit=as_choice(raw.get("image_fit", "pad"), "image_fit", IMAGE_FITS),
            arm_move_extra=_parse_arm_move_extra(raw.get("arm_move_extra")),
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
