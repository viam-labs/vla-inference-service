"""Linear interpolation between successive policy actions.

Densifies one 30 Hz policy action into the ~100 Hz setpoint cadence the xArm's
streamed servo mode wants, so Task 2b can feed `move_through_joint_positions_streamed`
without the policy itself running any faster.
"""

from __future__ import annotations

from typing import Sequence


def interpolate(prev: Sequence[float], target: Sequence[float], n: int) -> list[list[float]]:
    """Return `n` points linearly spaced from `prev` (exclusive) to `target` (inclusive).

    Point k (1-indexed) is `prev + (target - prev) * k / n`. The last point is
    `target` itself, computed via `list(target)` rather than arithmetic so no
    floating-point rounding drift reaches the arm.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if len(prev) != len(target):
        raise ValueError(f"prev and target must be the same length, got {len(prev)} and {len(target)}")

    points = []
    for k in range(1, n):
        points.append([p + (t - p) * k / n for p, t in zip(prev, target)])
    points.append(list(target))
    return points
