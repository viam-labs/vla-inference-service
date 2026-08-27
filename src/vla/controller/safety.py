"""Bounded-motion checks applied to every action before it reaches the arm.

This is the last line of defense between a policy's output and a real robot
arm. There is one layer per action space, because the two command the arm
through different SDK calls with different guarantees:

  - `SafetyLayer` -- `action_space="joints"`, absolute joint targets in
    degrees written with `move_to_joint_positions`.
  - `CartesianSafetyLayer` -- `action_space="delta-ee"`, a per-tick pose
    delta in millimetres and radians composed onto the measured
    `EndPosition` and written with `move_to_position`.

`SafetyLayer`'s order matters and is deliberately fixed:

  1. Reject NaN/inf in the whole vector -- fail the chunk, never clamp it.
  2. Dimension check against the current measured position.
  3. Per-step delta clamp against the *measured* position, not the last
     commanded one -- so a stalled arm cannot accumulate an ever-growing
     command.
  4. Joint limit clamp from the optional `joint_limits_degs` config,
     indexed in action-vector order, with a trailing gripper pair only when
     the gripper channel is itself in degrees (`gripper.type == "arm_joint"`).
  5. There is no driver-side ceiling. `MoveOptions` -- and the
     `move_through_joint_positions` call that would carry it -- ship in no
     released viam-sdk, so the velocity bound is enforced entirely by layer 3:
     `ControllerConfig` derives `max_joint_delta_degs = max_vel_degs_per_sec /
     fps`, making the per-step delta clamp *the* velocity limit rather than a
     redundant backstop. Acceleration and TCP-speed limiting are unavailable;
     `check_start` covers the large-initial-jump case they would have softened.
     This layer only ever produces a target position.
  6. Every clamp is logged and counted in `clamp_counts`, split by which
     layer fired (`delta` / `limit` / `gripper`). Persistent clamping is the
     project's single most likely bug class: wrong units or wrong joint
     order.

The degrees-based clamps (delta and limit) skip the trailing gripper channel
for every variant except `arm_joint` (`servo`, `do_command`): a degree-shaped
limit on a 0.0-1.0 channel would either never fire (useless) or fire
constantly on ordinary gripper motion (worse than useless). It gets its own
`[0, 1]` clamp instead, tracked separately as `clamp_counts["gripper"]`.

That `[0, 1]` clamp acts on the policy's *output* only:
`current[gripper_idx]` is never read (`joint_slice` stops at `gripper_idx`),
so the value the adapter read does not pass through here at all. The one
place it does leak in is `check_start`, which maxes over the whole vector and
so compares the gripper channel against a degrees budget.

Boundary decisions, both deliberate: a value exactly equal to a joint limit
is *not* counted as clamped (it passes through unchanged -- only a value
that `np.clip` actually altered counts), and `check_start`'s budget check is
inclusive (`delta > budget` refuses; `delta == budget` is allowed).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from vla.config_util import VLAError

LOGGER = logging.getLogger(__name__)


class SafetyError(VLAError, RuntimeError):
    """Raised when an action cannot be made safe.

    A `RuntimeError` base (not `ValueError`) because the vector itself is
    often perfectly well-typed -- what makes it unsafe is runtime context
    (the arm's current position, a start-pose mismatch), not a malformed
    argument.
    """


@dataclass(frozen=True)
class SafetyLimits:
    max_joint_delta_degs: float = 8.0
    max_start_delta_degs: float = 15.0
    joint_limits_degs: list[tuple[float, float]] | None = None
    gripper_in_degrees: bool = True


class SafetyLayer:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits
        self.clamp_counts: Counter[str] = Counter()

    def _validate(self, action: np.ndarray, current: np.ndarray) -> None:
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(current)):
            raise SafetyError(
                f"action must be finite (no NaN/inf); got action={action}, current={current}"
            )
        if action.shape != current.shape:
            raise SafetyError(
                f"action dimension {action.shape} does not match arm state dimension "
                f"{current.shape}; check state_joint_indices and the gripper block"
            )

    def _gripper_index(self, n: int) -> int | None:
        if self.limits.gripper_in_degrees or n == 0:
            return None
        return n - 1  # trailing channel: normalized, degree limits do not apply

    def check_start(self, first_action: np.ndarray, current: np.ndarray) -> None:
        """Refuse to start when the first action is far from the current pose.

        A policy handed an unfamiliar initial pose can emit anything.
        Refusing beats moving slowly to a place nobody asked for.
        """
        self._validate(first_action, current)
        delta = float(np.max(np.abs(first_action - current)))
        if delta > self.limits.max_start_delta_degs:
            LOGGER.error(
                "refusing to start: first action is %.2f deg from the current pose, "
                "exceeding max_start_delta_degs=%.2f",
                delta,
                self.limits.max_start_delta_degs,
            )
            raise SafetyError(
                f"first action is {delta:.2f} deg from the current pose, exceeding "
                f"max_start_delta_degs={self.limits.max_start_delta_degs}; "
                "move the arm nearer the expected start pose, or raise the limit deliberately"
            )

    def apply(self, action: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Clamp `action` into a safe range relative to `current`.

        Raises `SafetyError` for non-finite input or a dimension mismatch;
        otherwise always returns a same-shaped array (clamping, never
        rejecting, an out-of-range-but-finite value).
        """
        self._validate(action, current)
        out = np.asarray(action, dtype=np.float64).copy()
        current = np.asarray(current, dtype=np.float64)

        n = out.shape[0]
        gripper_idx = self._gripper_index(n)
        joint_stop = gripper_idx if gripper_idx is not None else n
        joint_slice = slice(0, joint_stop)

        # 3. delta clamp against the measured position.
        delta = out[joint_slice] - current[joint_slice]
        capped = np.clip(
            delta, -self.limits.max_joint_delta_degs, self.limits.max_joint_delta_degs
        )
        if not np.array_equal(delta, capped):
            self.clamp_counts["delta"] += 1
            LOGGER.warning(
                "delta clamp engaged (max %.2f deg); persistent clamping usually means "
                "wrong units or wrong joint order",
                self.limits.max_joint_delta_degs,
            )
        out[joint_slice] = current[joint_slice] + capped

        # 4. joint limit clamp, action-vector order, optional.
        if self.limits.joint_limits_degs:
            for i, (lo, hi) in enumerate(self.limits.joint_limits_degs):
                if i >= joint_stop:
                    break
                clamped = float(np.clip(out[i], lo, hi))
                if clamped != float(out[i]):
                    self.clamp_counts["limit"] += 1
                    LOGGER.warning(
                        "joint %d clamped to configured limit [%.1f, %.1f]", i, lo, hi
                    )
                out[i] = clamped

        # normalized gripper channel: [0, 1] instead of a degree-shaped clamp.
        if gripper_idx is not None:
            g = float(out[gripper_idx])
            clamped = min(1.0, max(0.0, g))
            if clamped != g:
                self.clamp_counts["gripper"] += 1
                LOGGER.warning(
                    "gripper channel clamped to [0, 1] (got %.4f); check that the policy "
                    "and gripper adapter agree on units",
                    g,
                )
            out[gripper_idx] = clamped

        return out


@dataclass(frozen=True)
class CartesianLimits:
    """Per-tick ceilings on a delta-EE action, in working units (mm, radians).

    The defaults come from the recorded per-tick statistics of the dataset
    these checkpoints are trained on (`xarm-open-box-eedelta`, 34,670 frames
    at 10 fps):

    | quantity    | median  | p99     | max     | default here |
    | ----------- | ------- | ------- | ------- | ------------ |
    | translation | 9.31 mm | 28.4 mm | 96.8 mm | 40 mm        |
    | rotation    | 0.0142  | 0.0807  | 0.3246  | 0.12 rad     |

    Each default sits about 1.4x above the p99, so ordinary in-distribution
    motion never clamps and the counter stays a genuine signal, and well
    below the largest single tick in the recording, which at 10 fps would be
    968 mm/s of tool travel. Clamping *below* the observed maximum is the
    deliberate half of that choice: the recorded extremes are a handful of
    frames out of 34,670, so reproducing them at full driver speed buys
    nothing and a policy that emits one every tick is out of distribution,
    not in a hurry.
    """

    max_tcp_delta_mm: float = 40.0
    max_tcp_rot_delta_rads: float = 0.12


class CartesianSafetyLayer:
    """Per-tick magnitude clamp on a delta-EE action. THE velocity limit.

    `move_to_position` carries no notion of "this delta represents one 100 ms
    tick" -- it takes a pose and drives to it at whatever speed the driver
    chooses -- so a policy emitting an out-of-distribution delta would have it
    executed at full driver speed. Bounding the delta is therefore the only
    thing standing between a bad chunk and a fast, large tool motion, exactly
    as `SafetyLayer`'s `max_joint_delta_degs` is for the joints path (see item
    5 of the module docstring). `ControllerConfig` derives these ceilings from
    the operator-facing `max_tcp_vel_mms_per_sec` /
    `max_tcp_rot_vel_rads_per_sec` divided by `fps`, the same way it derives
    `max_joint_delta_degs`.

    Both clamps scale the whole 3-vector by a single factor rather than
    clipping component-wise. Component-wise clipping would change the
    *direction* of the commanded motion -- a delta of (100, 100, 0) mm would
    come out as (40, 40, 0), which happens to stay on the same diagonal, but
    (100, 10, 0) would come out (40, 10, 0), a different heading than the
    policy asked for. In joint space that reshaping is the definition of a
    per-joint limit; in Cartesian space the direction is the physical path of
    the tool, so preserving it and shortening the step is the only clamp that
    means "go the same way, less far". The same argument applies to the
    rotation vector, where the norm is the angle and the direction is the
    axis: scaling keeps the axis and reduces the angle.

    What this layer does NOT cover, stated plainly so nobody assumes
    otherwise: joint limits. `joint_limits_degs` has no Cartesian analogue,
    and an out-of-range or unreachable target is refused by the arm driver's
    own IK rather than here (`move_to_position` is the component method, not
    the motion service, so there is no obstacle avoidance either).
    `ControllerConfig` rejects `joint_limits_degs` under this action space
    rather than accepting a limit it would silently never apply.

    `clamp_counts` is keyed the same way `SafetyLayer`'s is (`translation` /
    `rotation`) so `VLAController._status()` reports both layers identically.
    """

    def __init__(self, limits: CartesianLimits) -> None:
        self.limits = limits
        self.clamp_counts: Counter[str] = Counter()

    def apply(self, delta: np.ndarray) -> np.ndarray:
        """Clamp a 6-dim `[dx, dy, dz, drx, dry, drz]` action in mm and radians.

        Raises `SafetyError` for non-finite input or a wrong dimension;
        otherwise always returns a fresh 6-vector, clamping rather than
        rejecting an out-of-range but finite delta -- the same contract
        `SafetyLayer.apply` keeps.
        """
        out = np.asarray(delta, dtype=np.float64).copy()
        if out.shape != (6,):
            raise SafetyError(
                f"a delta-ee action must have 6 dimensions [dx, dy, dz, drx, dry, drz], "
                f"got shape {out.shape}"
            )
        if not np.all(np.isfinite(out)):
            raise SafetyError(f"action must be finite (no NaN/inf); got {delta}")

        out[0:3] = self._scaled(
            out[0:3], self.limits.max_tcp_delta_mm, "translation", "mm"
        )
        out[3:6] = self._scaled(
            out[3:6], self.limits.max_tcp_rot_delta_rads, "rotation", "rad"
        )
        return out

    def _scaled(
        self, vector: np.ndarray, ceiling: float, kind: str, unit: str
    ) -> np.ndarray:
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= ceiling:
            return vector
        self.clamp_counts[kind] += 1
        LOGGER.warning(
            "%s clamp engaged: |delta| %.4f %s exceeds the per-tick ceiling of %.4f %s, "
            "scaling the step down along the same %s; persistent clamping usually means "
            "wrong units or a checkpoint trained at a different fps",
            kind,
            magnitude,
            unit,
            ceiling,
            unit,
            "heading" if kind == "translation" else "axis",
        )
        return vector * (ceiling / magnitude)
