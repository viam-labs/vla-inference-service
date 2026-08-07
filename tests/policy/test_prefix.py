import numpy as np
import pytest

from vla.config_util import VLAError
from vla.policy.prefix import PrefixError, normalize_prefix_length


def test_exact_length_returns_unchanged():
    x = np.ones((10, 4), dtype=np.float32)
    out = normalize_prefix_length(x, 10)
    np.testing.assert_array_equal(out, x)


def test_longer_prefix_is_truncated():
    x = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    out = normalize_prefix_length(x, 8)
    assert out.shape == (8, 3)
    np.testing.assert_array_equal(out, x[:8])


def test_shorter_prefix_is_zero_padded():
    x = np.ones((3, 4), dtype=np.float32)
    out = normalize_prefix_length(x, 10)
    assert out.shape == (10, 4)
    np.testing.assert_array_equal(out[:3], x)
    np.testing.assert_array_equal(out[3:], np.zeros((7, 4), dtype=np.float32))


def test_preserves_dtype():
    x = np.ones((3, 4), dtype=np.float32)
    assert normalize_prefix_length(x, 6).dtype == np.float32


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        normalize_prefix_length(np.ones((2, 3, 4), dtype=np.float32), 5)


# ---------------------------------------------------------------------------
# Error path must raise this module's own exception type (standing req 5).
# ---------------------------------------------------------------------------


def test_rejects_non_2d_raises_prefix_error_specifically():
    with pytest.raises(PrefixError, match="2D"):
        normalize_prefix_length(np.ones((2, 3, 4), dtype=np.float32), 5)


def test_prefix_error_is_a_vla_error():
    assert issubclass(PrefixError, VLAError)


# ---------------------------------------------------------------------------
# target_steps guard (item 2) -- a non-positive horizon is a config bug,
# not a silent empty-prefix result.
# ---------------------------------------------------------------------------


def test_zero_target_steps_raises():
    with pytest.raises(PrefixError, match="target_steps"):
        normalize_prefix_length(np.ones((3, 4), dtype=np.float32), 0)


def test_negative_target_steps_raises():
    with pytest.raises(PrefixError, match="target_steps"):
        normalize_prefix_length(np.ones((3, 4), dtype=np.float32), -5)


# ---------------------------------------------------------------------------
# An empty prefix is a real input -- the queue returns (0, dim) on the very
# first tick (item 3).
# ---------------------------------------------------------------------------


def test_empty_prefix_is_padded_to_all_zeros():
    x = np.zeros((0, 4), dtype=np.float32)
    out = normalize_prefix_length(x, 6)
    assert out.shape == (6, 4)
    np.testing.assert_array_equal(out, np.zeros((6, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Aliasing (item 4). Chosen behavior: always return a copy, in every branch
# -- including the steps == target_steps and truncation branches, where
# upstream (torch) returns the input itself / a view. The RTC scheduler
# stores a chunk's raw actions in ActionQueue and this function's output is
# fed straight back in as prev_chunk_left_over on a later tick; a caller
# that mutates the returned array in place (unit conversion, clamping, ...)
# must never be able to corrupt the queue's stored chunk.
# ---------------------------------------------------------------------------


def test_exact_length_result_does_not_alias_input():
    x = np.ones((5, 3), dtype=np.float32)
    out = normalize_prefix_length(x, 5)
    out[0, 0] = 999.0
    assert x[0, 0] == 1.0


def test_truncated_result_does_not_alias_input():
    x = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    out = normalize_prefix_length(x, 8)
    out[0, 0] = -1.0
    assert x[0, 0] == 0.0


def test_padded_result_does_not_alias_input():
    x = np.ones((3, 4), dtype=np.float32)
    out = normalize_prefix_length(x, 10)
    out[0, 0] = 999.0
    assert x[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Non-float dtypes and non-contiguous input (item 5).
# ---------------------------------------------------------------------------


def test_int_dtype_is_preserved_and_padded_with_int_zeros():
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = normalize_prefix_length(x, 4)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, np.array([[1, 2], [3, 4], [0, 0], [0, 0]], dtype=np.int32))


def test_non_contiguous_input_is_handled_on_truncation():
    # A transpose is a classic non-contiguous view.
    base = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    x = base.T  # shape (6, 4), non-contiguous
    assert not x.flags["C_CONTIGUOUS"]
    out = normalize_prefix_length(x, 3)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out, x[:3])


def test_non_contiguous_input_is_handled_on_padding():
    base = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    x = base.T  # shape (4, 2), non-contiguous
    assert not x.flags["C_CONTIGUOUS"]
    out = normalize_prefix_length(x, 6)
    assert out.shape == (6, 2)
    np.testing.assert_array_equal(out[:4], x)
    np.testing.assert_array_equal(out[4:], np.zeros((2, 2), dtype=np.float32))
