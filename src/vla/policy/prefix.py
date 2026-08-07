"""Pad or truncate an RTC prefix to the configured execution horizon.

Port of lerobot `rollout/inference/rtc.py::_normalize_prev_actions_length`.
Kept as pure numpy so it is testable with no torch present.

Verified against lerobot at git SHA
`ff7cc3de1de830f5f3276918a013d04bdf9ea4be`,
`src/lerobot/rollout/inference/rtc.py:83-94`
(`_normalize_prev_actions_length`): same truncation side (keep the first
`target_steps` rows), same zero-padding placement (existing rows first,
zeros after), same dtype handling (`np.zeros(..., dtype=prev_actions.dtype)`
mirroring the tensor version's `dtype=prev_actions.dtype`).

One deliberate divergence: upstream returns the input tensor itself
(aliased) when `steps == target_steps`, and a view
(`prev_actions[:target_steps]`) when truncating. This port always returns a
copy in every branch. The RTC scheduler stores a chunk's raw actions in
`ActionQueue` and this function's output is fed straight back in on a later
tick as `prev_chunk_left_over`; a caller mutating the returned array in
place (unit conversion, clamping, ...) must never be able to corrupt the
queue's stored chunk. See `tests/policy/test_prefix.py` for the enforced
non-aliasing contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla.config_util import ConfigError, VLAError, as_int


class PrefixError(VLAError, ValueError):
    """Raised when an RTC prefix cannot be normalized."""


def normalize_prefix_length(prev_actions: np.ndarray, target_steps: Any) -> np.ndarray:
    if prev_actions.ndim != 2:
        raise PrefixError(f"Expected 2D [T, A] array, got shape={prev_actions.shape}")

    # target_steps arrives as a protobuf-Struct-shaped value in production
    # (a double, never a plain int) wherever it is threaded through from
    # config rather than hardcoded -- as_int accepts an integral float
    # (8.0), rejects a fractional one (8.5) or a bool, and enforces the
    # positive-only bound in one place instead of a bare `<= 0` check that
    # would raise an unhelpful TypeError on a non-numeric or float input.
    # A non-positive horizon is a config bug -- execution_horizon is
    # already validated >= 1 upstream (see PolicyConfig.RTCSettings.parse)
    # -- and a silent (0, dim) result here would be a much nastier way to
    # discover that than an immediate error. ConfigError is translated to
    # this module's own PrefixError so callers only ever need `except
    # PrefixError` (or `except VLAError`) here, never `except ConfigError`.
    try:
        target = as_int(target_steps, "target_steps", minimum=1)
    except ConfigError as exc:
        raise PrefixError(str(exc)) from exc

    steps, action_dim = prev_actions.shape
    if steps == target:
        return prev_actions.copy()
    if steps > target:
        return prev_actions[:target].copy()
    padded = np.zeros((target, action_dim), dtype=prev_actions.dtype)
    padded[:steps] = prev_actions
    return padded
