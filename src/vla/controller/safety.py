"""Bounded-motion checks applied to every action before it reaches the arm.

This is the last line of defense between a policy's output and a real robot
arm. Order matters and is deliberately fixed:

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
for every variant except `arm_joint` (`servo`, `gripper/inputs`,
`gripper/threshold`, `do_command`): a degree-shaped limit on a 0.0-1.0
channel would either never fire (useless) or fire constantly on ordinary
gripper motion (worse than useless). It gets its own `[0, 1]` clamp instead,
tracked separately as `clamp_counts["gripper"]`.

That `[0, 1]` shape is this layer's own contract, not a claim about what the
adapter read: `servo` and `do_command` hand up an already-normalized value,
but `gripper/inputs` and `gripper/threshold` pass the driver's raw
radians/meters through unnormalized (see `_read_first_input` in
`gripper.py`). For those two the clamp does real work rather than
re-asserting an invariant.

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
