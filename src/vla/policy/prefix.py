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

import numpy as np

from vla.config_util import VLAError


class PrefixError(VLAError, ValueError):
    """Raised when an RTC prefix cannot be normalized."""


def normalize_prefix_length(prev_actions: np.ndarray, target_steps: int) -> np.ndarray:
    if prev_actions.ndim != 2:
        raise PrefixError(f"Expected 2D [T, A] array, got shape={prev_actions.shape}")
    # A non-positive horizon is a config bug -- execution_horizon is already
    # validated >= 1 upstream (see PolicyConfig.RTCSettings.parse) -- and a
    # silent (0, dim) result here would be a much nastier way to discover
    # that than an immediate error.
    if target_steps <= 0:
        raise PrefixError(f"target_steps must be a positive integer, got {target_steps}")

    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions.copy()
    if steps > target_steps:
        return prev_actions[:target_steps].copy()
    padded = np.zeros((target_steps, action_dim), dtype=prev_actions.dtype)
    padded[:steps] = prev_actions
    return padded
