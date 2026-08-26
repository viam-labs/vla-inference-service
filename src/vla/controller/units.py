"""Conversion between Viam's native degrees and a checkpoint's state units.

Viam arms report and accept degrees; a checkpoint uses whatever the recording
robot used. Nearly every bug in this module traces back to this boundary.

Only `degrees` and `radians` convert. `"normalized"` needs per-joint min/max,
an unresolved design question -- `ControllerConfig` validates the configured
unit against `SUPPORTED_UNITS`, so it never reaches here.

Both functions always return a fresh float32 ndarray, whatever the unit, so
the return type never varies by argument value.
"""

from __future__ import annotations

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
