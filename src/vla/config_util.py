"""Shared config-value coercion helpers for viam-labs:vla resources.

Config arrives as a plain dict from ``struct_to_dict`` on a protobuf
``Struct``, which stores every number as a double -- so an int-typed field
arrives as ``2.0`` in production but as ``2`` in a hand-written test dict.
These helpers fold that coercion, plus range/membership/type checks, into
one place so every resource's parser raises the same `ConfigError`.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# POSIX environment variable names are ASCII identifiers, capped here at 64
# chars. This is a shape check, not a secret detector: a real token (e.g.
# Hugging Face's `hf_` + ~34 alphanumerics) is itself a valid identifier and
# passes cleanly. Redaction at every display site is the actual defense.
_MAX_ENV_VAR_NAME_LEN = 64


class VLAError(Exception):
    """Common base for every error this module's own code raises.

    Lets a caller treat "this module rejected the input" (`except VLAError`)
    differently from "this module has a bug" (a bare `AttributeError`,
    `TypeError`, ...), instead of swallowing both in one `except Exception`.
    """


class ConfigError(VLAError, ValueError):
    """Raised for invalid module configuration."""


def as_int(value: Any, field_name: str, *, minimum: int | float | None = None, maximum: int | float | None = None) -> int:
    """Coerce a Struct-shaped value to int, strictly.

    Accepts an int or an integral float (2.0 -> 2). A fractional float is a
    config typo, not a truncation target. Booleans are ints in Python but
    never a legitimate numeric config value, so they are rejected.
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
    """Coerce a Struct-shaped value to float, strictly.

    Booleans are rejected rather than becoming 0.0/1.0. NaN and +/-infinity
    are rejected unconditionally: they pass any `<= 0` guard yet poison
    downstream math with no config-time diagnostic.
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

    A hand-edited config can contain ``"enabled": "false"``, and
    ``bool("false")`` is `True` -- a loose check would flip the field's
    meaning.
    """
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{field_name} must be a boolean, got {value!r}")


def as_str(value: Any, field_name: str) -> str:
    """Require an actual string.

    Struct has no "string that looks like a number" type: an intended
    string can arrive as a float (a revision of "1" becomes 1.0) and
    propagate into a downstream API call. Type only, never format.
    """
    if isinstance(value, str):
        return value
    raise ConfigError(f"{field_name} must be a string, got {value!r}")


def as_choice(value: Any, field_name: str, allowed: Sequence[str]) -> str:
    """Require ``value`` to be one of ``allowed``."""
    if value not in allowed:
        raise ConfigError(f"{field_name} must be one of {tuple(allowed)}, got {value!r}")
    return value


def redact_secret(value: str) -> str:
    """Redact a possibly-sensitive string for safe inclusion in errors/logs.

    Shows at most the first 4 characters plus the total length -- enough for
    an operator to recognize which value they pasted without it being
    reconstructable from a log line.
    """
    return f"{value[:4]}...<{len(value)} chars>"


def as_env_var_name(value: Any, field_name: str) -> str:
    """Require a value shaped like a POSIX environment variable name.

    Fields such as `hf_token_env` name an env var precisely so the secret
    never appears in config or logs. This catches an obviously malformed
    paste (hyphens, dots, spaces, excessive length); it is not a secret
    detector, so a rejected value is still redacted rather than echoed.
    """
    text = as_str(value, field_name)
    if not text.isidentifier() or not text.isascii() or len(text) > _MAX_ENV_VAR_NAME_LEN:
        raise ConfigError(
            f"{field_name} must look like an environment variable name "
            f"(ASCII letters, digits, underscore; not starting with a digit; "
            f"{_MAX_ENV_VAR_NAME_LEN} chars max), got {redact_secret(text)}"
        )
    return text
