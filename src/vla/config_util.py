"""Shared config-value coercion helpers for viam-labs:vla resources.

Every resource in this module (policy, controller, ...) receives its config
as a plain dict produced by ``struct_to_dict`` on a protobuf ``Struct``.
Struct stores every number as a double, so an int-typed field arrives as
``2.0`` in production but as ``2`` in a hand-written test dict. These
helpers fold that coercion, plus range/membership/type checks, into one
place so every resource's parser raises the same `ConfigError` — not a bare
`ValueError`/`AttributeError` that a caller has to know to catch specially.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


class ConfigError(ValueError):
    """Raised for invalid module configuration."""


def as_int(value: Any, field_name: str, *, minimum: int | float | None = None, maximum: int | float | None = None) -> int:
    """Coerce a protobuf-Struct-shaped value to int, strictly.

    Accepts a plain int, or a float with no fractional part (2.0 -> 2).
    A fractional float (2.5) is a config typo, not a truncation target, so
    it is rejected rather than silently floored. Booleans are technically
    ints in Python but are never a legitimate value for a numeric config
    field, so they are rejected explicitly.
    """
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ConfigError(f"{field_name} must be an integer, got {value!r}")
        result = int(value)
    else:
        raise ConfigError(f"{field_name} must be an integer, got {value!r}")

    if minimum is not None and result < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{field_name} must be <= {maximum}, got {result}")
    return result


def as_float(value: Any, field_name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Coerce a protobuf-Struct-shaped value to float, strictly.

    Booleans are technically numbers in Python but are never a legitimate
    value for a numeric config field, so they are rejected explicitly
    rather than silently becoming 0.0/1.0. NaN and +/-infinity are rejected
    unconditionally: they pass any `<= 0` guard yet poison downstream math
    with no config-time diagnostic.
    """
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        raise ConfigError(f"{field_name} must be a number, got {value!r}")

    if not math.isfinite(result):
        raise ConfigError(f"{field_name} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{field_name} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{field_name} must be <= {maximum}, got {result}")
    return result


def as_bool(value: Any, field_name: str) -> bool:
    """Require an actual bool.

    A hand-edited Viam config JSON can plausibly contain ``"enabled":
    "false"`` (a string). ``bool("false")`` is `True` in Python, so a loose
    check would silently flip the meaning of the field. Only a real bool
    is accepted.
    """
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{field_name} must be a boolean, got {value!r}")


def as_str(value: Any, field_name: str) -> str:
    """Require an actual string.

    Struct has no distinct "string that looks like a number" type: a config
    author's intended string can arrive as a float (e.g. a revision of "1"
    becomes 1.0) and silently propagate into a downstream API call with a
    confusing error far from the actual mistake. This checks type only,
    never format/content.
    """
    if isinstance(value, str):
        return value
    raise ConfigError(f"{field_name} must be a string, got {value!r}")


def as_choice(value: Any, field_name: str, allowed: Sequence[str]) -> str:
    """Require ``value`` to be one of ``allowed``."""
    if value not in allowed:
        raise ConfigError(f"{field_name} must be one of {tuple(allowed)}, got {value!r}")
    return value
