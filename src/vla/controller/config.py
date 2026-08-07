"""Configuration parsing and validation for viam-labs:vla:controller.

Uses `vla.config_util` throughout: `ConfigError` is imported from there, not
redefined here, and every numeric/choice field goes through `as_int`/
`as_float`/`as_choice` with its range folded into the call. This config has
about ten numeric fields plus a nested `safety` block -- exactly the
situation those helpers exist for (protobuf `Struct` delivers every number
as a double, so every field needs the same "accept 2.0, reject 2.5, reject
True, reject NaN" treatment).

Two corrections from an earlier plan draft, both load-bearing:

  - `state_units`/`action_units` are validated against `units.SUPPORTED_UNITS`
    (degrees, radians), not `units.UNITS` (which also includes "normalized").
    "normalized" is a config value units.py can represent but never convert
    (it needs per-joint min/max, which is an open question in the design
    doc) -- validating against the narrower set here means a bad config
    fails `validate_config` with the field name, instead of surviving to the
    first control tick and crashing inside `from_degrees`/`to_degrees`.

  - There is no `max_acc_degs_per_sec2` or `max_tcp_speed_m_per_sec` field.
    Both were meant to feed `MoveOptions`, but `move_through_joint_positions`
    (the only method that consumes `MoveOptions`) ships in no released
    viam-sdk -- installed 0.80.0 has only `move_to_joint_positions`, which
    takes no options at all. A config knob with no enforcement path is worse
    than an absent one, so both are gone. `max_vel_degs_per_sec` survives
    because it has a real enforcement path: the safety layer's existing
    per-tick `max_joint_delta_degs` clamp. `max_vel_degs_per_sec` is the
    operator-facing knob (reasoning in degrees/second is what a human can
    actually do); `max_joint_delta_degs` is derived from it as
    `max_vel_degs_per_sec / fps`. Both may still appear in config, but if
    both are given they must agree, or it is a config-time error -- silently
    preferring one over a contradictory other would hide an operator
    mistake instead of surfacing it.

`duration_warn_s` / `stale_frame_warn_s` are operator-configurable rather
than the fixed module defaults `observation.py` used to hardcode: a
checkpoint run at 2 Hz and one run at 10 Hz imply very different "this tick
is late" and "this frame is stale" thresholds (a 100ms duration budget is
generous at 2 Hz but is the *entire* tick at 10 Hz), and the operator is the
only party who knows which regime they are in. Defaults match
`observation.py`'s own constants, so an operator who never sets these two
fields observes no behavior change.

`mode: "async"` (Task 22) overlaps execution with inference instead of
stalling the arm between chunks -- the fix for the measured case where
inference latency approaches or exceeds chunk duration (~5.3s vs. a 5.0s
chunk on Apple Silicon), where RTC cannot help (it needs `delay <
chunk_length`, and here `delay > chunk_length`). It is explicit opt-in,
not a value `"auto"` resolves to: `"auto"` still resolves to `"sequential"`
here, unchanged, so an existing deployment's behavior never changes underneath
it just because this module gained a new mode. An operator picks `"async"`
by name once they have measured that their own inference latency warrants
the discontinuity-at-chunk-boundary tradeoff it accepts in exchange (see
`AsyncScheduler`'s docstring in `scheduler.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vla.config_util import ConfigError, as_bool, as_choice, as_float, as_int, as_str

from .gripper import GRIPPER_TYPES
from .observation import DEFAULT_DURATION_WARN_S, STALE_FRAME_WARN_S
from .units import SUPPORTED_UNITS

__all__ = ["ConfigError", "SafetyConfig", "ControllerConfig", "MODES", "ENCODINGS"]

MODES = ("auto", "sequential", "rtc", "async")
ENCODINGS = ("jpeg", "png", "raw")

# Bounds chosen as sanity ceilings/floors, not physical limits: they exist
# to turn an obvious typo (a negative fps, a fractional queue_threshold)
# into a config-time error rather than a runtime surprise.
_MIN_FPS = 1e-6
_MAX_FPS = 1000.0
_MIN_POSITIVE_DEG = 1e-6
_MIN_POLICY_READY_TIMEOUT_S = 1

# Tolerance for the max_vel_degs_per_sec / fps vs. max_joint_delta_degs
# consistency check. Both arrive through as_float (already finite, already
# doubles), so the only slack needed is for ordinary floating-point
# division/round-trip error -- not for operator "close enough" rounding.
_VEL_DELTA_CONSISTENCY_ABS_TOL = 1e-6
_VEL_DELTA_CONSISTENCY_REL_TOL = 1e-6


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

        max_start_delta = as_float(
            raw.get("max_start_delta_degs", 15.0), "safety.max_start_delta_degs", minimum=0.0
        )
        stop_on_error = as_bool(raw.get("stop_on_error", True), "safety.stop_on_error")

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

        max_joint_delta = SafetyConfig._resolve_max_joint_delta(max_vel, declared_delta, fps)

        return SafetyConfig(
            max_joint_delta_degs=max_joint_delta,
            max_start_delta_degs=max_start_delta,
            max_vel_degs_per_sec=max_vel,
            joint_limits_degs=_parse_joint_limits(raw.get("joint_limits_degs")),
            stop_on_error=stop_on_error,
        )

    @staticmethod
    def _resolve_max_joint_delta(
        max_vel: float | None, declared_delta: float | None, fps: float
    ) -> float:
        """Resolve max_joint_delta_degs from whichever of the two knobs the
        operator gave, applying the precedence documented on this module:
        max_vel_degs_per_sec is authoritative when both are given, but only
        after confirming the two agree.
        """
        if max_vel is None:
            return 8.0 if declared_delta is None else declared_delta

        derived_delta = max_vel / fps
        if declared_delta is None:
            return derived_delta

        if not math.isclose(
            declared_delta,
            derived_delta,
            rel_tol=_VEL_DELTA_CONSISTENCY_REL_TOL,
            abs_tol=_VEL_DELTA_CONSISTENCY_ABS_TOL,
        ):
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
    queue_threshold: int = 30
    starvation_grace_ticks: int = 3
    policy_ready_timeout_s: int = 600
    state_units: str = "degrees"
    action_units: str = "degrees"
    image_encoding: str = "jpeg"
    jpeg_quality: int = 90
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

        mode = as_choice(raw.get("mode", "auto"), "mode", MODES)
        fps = as_float(raw.get("fps", 10.0), "fps", minimum=_MIN_FPS, maximum=_MAX_FPS)
        encoding = as_choice(raw.get("image_encoding", "jpeg"), "image_encoding", ENCODINGS)
        state_units = as_choice(raw.get("state_units", "degrees"), "state_units", SUPPORTED_UNITS)
        action_units = as_choice(
            raw.get("action_units", "degrees"), "action_units", SUPPORTED_UNITS
        )

        gripper = dict(raw.get("gripper") or {"type": "none"})
        gripper_type = as_choice(gripper.get("type", "none"), "gripper.type", GRIPPER_TYPES)
        gripper["type"] = gripper_type

        safety = SafetyConfig.parse(raw.get("safety") or {}, fps=fps)

        # The trailing gripper limit pair exists only when the gripper
        # channel is itself in degrees, which is exactly the arm_joint case
        # -- every other gripper variant is normalized [0, 1] and gets its
        # own clamp in safety.py, not a degree-shaped joint limit.
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
            mode=mode,
            queue_threshold=as_int(raw.get("queue_threshold", 30), "queue_threshold", minimum=0),
            starvation_grace_ticks=as_int(
                raw.get("starvation_grace_ticks", 3), "starvation_grace_ticks", minimum=0
            ),
            policy_ready_timeout_s=as_int(
                raw.get("policy_ready_timeout_s", 600),
                "policy_ready_timeout_s",
                minimum=_MIN_POLICY_READY_TIMEOUT_S,
            ),
            state_units=state_units,
            action_units=action_units,
            image_encoding=encoding,
            jpeg_quality=as_int(raw.get("jpeg_quality", 90), "jpeg_quality", minimum=0, maximum=100),
            duration_warn_s=as_float(
                raw.get("duration_warn_s", DEFAULT_DURATION_WARN_S),
                "duration_warn_s",
                minimum=0.0,
            ),
            stale_frame_warn_s=as_float(
                raw.get("stale_frame_warn_s", STALE_FRAME_WARN_S),
                "stale_frame_warn_s",
                minimum=0.0,
            ),
            safety=safety,
        )

    def dependencies(self) -> list[str]:
        deps = [self.policy_service, self.arm, *self.cameras.values()]
        name = self.gripper.get("name")
        if self.gripper.get("type") in ("servo", "gripper") and name:
            deps.append(name)
        return deps
