"""Conversion between Viam's native degrees and a checkpoint's state units.

Viam arms report and accept degrees. A checkpoint uses whatever the recording
robot used. Nearly every bug in this module traces back to this boundary,
which is why `_check` deliberately raises for "normalized" units rather than
guessing at a per-joint min/max -- that mapping is an unresolved open
question in the spec (see the design doc), and guessing would silently
produce wrong joint commands instead of a loud config-time error.

Input contract: `values` must be a `numpy.ndarray`. A Python list or a bare
scalar is rejected with `UnitError` rather than being coerced. The
alternative -- accepting anything `np.deg2rad`/`np.rad2deg` tolerate -- would
silently hand back a Python list for the `"degrees"` identity branch (since
that branch returns its input unchanged) while returning an `ndarray` for
every other unit, a type that varies by argument value rather than by
contract. Requiring an `ndarray` up front keeps the return type uniform and
catches a caller that forgot `np.asarray()` immediately instead of letting
the mistake propagate into arithmetic several frames away.
"""

from __future__ import annotations

import numpy as np

from vla.config_util import VLAError

UNITS = ("degrees", "radians", "normalized")


class UnitError(VLAError, ValueError):
    """Raised for an unsupported unit conversion or a malformed input."""


def _check_unit(unit: str) -> None:
    if unit == "normalized":
        raise UnitError(
            "normalized units require per-joint min/max, which is unresolved; "
            "use degrees or radians"
        )
    if unit not in UNITS:
        raise UnitError(f"unknown unit {unit!r}, expected one of {UNITS}")


def _check_values(values: object) -> None:
    if not isinstance(values, np.ndarray):
        raise UnitError(
            f"values must be a numpy ndarray, got {type(values).__name__}: {values!r}"
        )


def from_degrees(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert Viam degrees into the checkpoint's unit."""
    _check_values(values)
    _check_unit(unit)
    if unit == "degrees":
        return values
    return np.deg2rad(values).astype(np.float32)


def to_degrees(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert the checkpoint's unit back into Viam degrees."""
    _check_values(values)
    _check_unit(unit)
    if unit == "degrees":
        return values
    return np.rad2deg(values).astype(np.float32)
