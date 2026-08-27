"""Conversion between the units this module works in and a checkpoint's.

Two APIs live here, because the two action spaces have genuinely different
shapes and merging them would make both worse.

`from_degrees`/`to_degrees` serve `action_space="joints"`: a joint vector is
one quantity end to end, so a single unit string describes the whole thing,
and Viam's native unit for it is degrees. Unchanged, and deliberately so --
every joints-mode deployment goes through this pair.

`from_working`/`to_working` serve `action_space="delta-ee"`, whose vectors
*mix* quantities: the 9-dim state is 3 lengths followed by 6 dimensionless
rotation-matrix entries, and the 6-dim action is 3 lengths followed by 3
angles. No single unit string can describe either one, so a `VectorUnits`
names a unit per contiguous segment instead.

"Working units" are the ones this module's own delta-EE math is written in,
and they are fixed per quantity kind rather than configurable:

  - length -> **millimetres**, because `viam.proto.common.Pose` carries mm and
    the dataset stored `EndPosition` mm verbatim.
  - angle -> **radians**, because the action's rotation segment is an
    axis-angle vector consumed by `Rotation.from_rotvec`, and the Cartesian
    per-tick clamp is expressed in radians.
  - dimensionless -> itself.

Note that the angle basis here is radians while `from_degrees`/`to_degrees`
are degrees-based. That is not an inconsistency to tidy away: a joint angle is
a Viam quantity (`JointPositions` is degrees) and an axis-angle rotation
vector is not a Viam quantity at all. Forcing the rotvec segment through
degrees would make `radians` -- the overwhelmingly common case, and the one
the reference converter emits -- a lossy `deg2rad(rad2deg(x))` round trip on a
vector that is about to be turned into arm motion.

Every function returns a fresh float32 ndarray whatever the unit, so the
return type never varies by argument value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla.config_util import VLAError

SUPPORTED_UNITS = ("degrees", "radians")


class UnitError(VLAError, ValueError):
    """Raised for an unsupported unit conversion."""


def _checked(values: np.ndarray, unit: str) -> np.ndarray:
    if unit not in SUPPORTED_UNITS:
        raise UnitError(f"unknown unit {unit!r}, expected one of {SUPPORTED_UNITS}")
    return np.asarray(values, dtype=np.float32)


def from_degrees(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert Viam degrees into the checkpoint's unit."""
    out = _checked(values, unit)
    return out if unit == "degrees" else np.deg2rad(out).astype(np.float32)


def to_degrees(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert the checkpoint's unit back into Viam degrees."""
    out = _checked(values, unit)
    return out if unit == "degrees" else np.rad2deg(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-segment units, for the mixed-quantity delta-EE vectors.
# ---------------------------------------------------------------------------

# Scale factors are "one checkpoint unit expressed in working units", so
# converting *from* the checkpoint multiplies and converting *to* it divides.
# Writing it this way keeps `meters -> 1000.0` readable as "a metre is 1000
# millimetres" rather than as an inverse nobody can check by eye.
_SEGMENT_SCALES: dict[str, float] = {
    "millimeters": 1.0,
    "meters": 1000.0,
    "radians": 1.0,
    "degrees": float(np.pi / 180.0),
    "unitless": 1.0,
}

LENGTH_UNITS = ("millimeters", "meters")
ANGLE_UNITS = ("radians", "degrees")
DIMENSIONLESS_UNITS = ("unitless",)
SEGMENT_UNITS = LENGTH_UNITS + ANGLE_UNITS + DIMENSIONLESS_UNITS


@dataclass(frozen=True)
class UnitSegment:
    """`size` consecutive components that all carry `unit`."""

    size: int
    unit: str


@dataclass(frozen=True)
class VectorUnits:
    """The unit of every component of one vector, run-length encoded.

    Frozen and hashable so a parsed `ControllerConfig` stays frozen, and
    validated at construction rather than at the first control tick: a
    segment list whose sizes do not add up to the vector it describes is a
    config mistake, and the tick loop is the wrong place to discover it.
    """

    segments: tuple[UnitSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise UnitError("a VectorUnits needs at least one segment")
        for segment in self.segments:
            if segment.size <= 0:
                raise UnitError(f"segment size must be positive, got {segment.size}")
            if segment.unit not in _SEGMENT_SCALES:
                raise UnitError(
                    f"unknown unit {segment.unit!r}, expected one of {SEGMENT_UNITS}"
                )

    @property
    def size(self) -> int:
        return sum(segment.size for segment in self.segments)

    def scales(self) -> np.ndarray:
        """Per-component multipliers taking checkpoint units into working units."""
        return np.concatenate(
            [
                np.full(segment.size, _SEGMENT_SCALES[segment.unit], dtype=np.float32)
                for segment in self.segments
            ]
        )


def _checked_vector(values: np.ndarray, units: VectorUnits) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32)
    if out.shape != (units.size,):
        raise UnitError(
            f"expected a vector of {units.size} components to match "
            f"{units.segments}, got shape {out.shape}"
        )
    return out


def from_working(values: np.ndarray, units: VectorUnits) -> np.ndarray:
    """Convert a vector in this module's working units into the checkpoint's."""
    return (_checked_vector(values, units) / units.scales()).astype(np.float32)


def to_working(values: np.ndarray, units: VectorUnits) -> np.ndarray:
    """Convert a vector in the checkpoint's units into this module's working ones."""
    return (_checked_vector(values, units) * units.scales()).astype(np.float32)
