"""ReferenceBuffer: rows on a time axis, sampled where a target is due."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vla.controller.reference_buffer import ReferenceBuffer, ReferenceBufferError

FPS = 8.0
P = 1.0 / FPS  # 0.125: dyadic, so every time below is exact


def ramp(n, start=0.0, cols=2):
    """Rows whose every column equals start + i: readable back by value."""
    return np.repeat((start + np.arange(n, dtype=np.float64))[:, None], cols, axis=1)


def smooth(x):
    return x * x * (3 - 2 * x)


def test_empty_buffer_is_exhausted_with_no_horizon():
    b = ReferenceBuffer(FPS)
    assert b.exhausted(0.0)
    assert b.horizon() == -math.inf
    assert b.sample(0.0) is None


def test_first_chunk_pins_row_zero_to_the_first_sample():
    b = ReferenceBuffer(FPS)
    b.land(ramp(5), t_obs=0.0)
    assert b.horizon() == math.inf  # unpinned: no horizon to fire against yet
    np.testing.assert_array_equal(b.sample(7.0), [0, 0])
    np.testing.assert_array_equal(b.sample(7.0 + P), [1, 1])
    np.testing.assert_array_equal(b.sample(7.0 + 1.5 * P), [1.5, 1.5])
    assert b.horizon() == 7.0 + 4 * P
    assert b.rebases == 1


def test_sampling_past_the_last_row_is_exhausted():
    b = ReferenceBuffer(FPS)
    b.land(ramp(3), 0.0)
    b.sample(1.0)
    assert b.sample(1.0 + 2 * P) is not None
    assert b.exhausted(1.0 + 2.5 * P)
    assert b.sample(1.0 + 2.5 * P) is None


def test_hard_switch_replaces_everything_after_the_last_sample():
    b = ReferenceBuffer(FPS, blend_s=0.0)
    b.land(ramp(10), 0.0)
    b.sample(P)  # pins with no shift: value = t/P - 1
    np.testing.assert_array_equal(b.sample(2 * P), [1, 1])
    b.land(ramp(10, start=100.0), t_obs=1.5 * P)  # rows at 2.5P, 3.5P, ...
    np.testing.assert_array_equal(b.sample(2.5 * P), [100, 100])
    np.testing.assert_array_equal(b.sample(3.5 * P), [101, 101])
    assert b.overlap_s == 0.0
    assert b.rebases == 1


def test_crossfade_ramps_old_to_new_with_smoothstep_from_the_last_sample():
    b = ReferenceBuffer(FPS, blend_s=4 * P)
    b.land(np.zeros((20, 1)), 0.0)
    b.sample(P)
    b.land(np.ones((20, 1)), t_obs=P)  # rows at 2P, 3P, ...; seam at P
    for t, x in [(2 * P, 0.25), (3 * P, 0.5), (4 * P, 0.75), (5 * P, 1.0), (9 * P, 1.0)]:
        np.testing.assert_allclose(b.sample(t), [smooth(x)], atol=1e-12)
    assert b.overlap_s == 4 * P


def test_an_old_plan_ending_inside_the_window_is_held_not_dropped():
    b = ReferenceBuffer(FPS, blend_s=8 * P)
    b.land(np.array([[0.0], [2.0]]), 0.0)  # rows at P (0) and 2P (2)
    b.sample(P)
    b.land(np.full((20, 1), 10.0), t_obs=P)
    # At 5P the old plan ended at 2P with value 2: blend against 2, not 0.
    w = smooth(0.5)
    np.testing.assert_allclose(b.sample(5 * P), [w * 10 + (1 - w) * 2], atol=1e-12)
    assert b.overlap_s == P


def test_held_columns_switch_at_the_seam_and_never_interpolate():
    b = ReferenceBuffer(FPS, blend_s=4 * P, interp_cols=1)
    b.land(np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]] + [[0.0, 0.0]] * 17), 0.0)
    np.testing.assert_array_equal(b.sample(P), [0, 0])
    # Between rows 1 (gripper 1) and 2 (gripper 0) the gripper holds row 1's value.
    np.testing.assert_array_equal(b.sample(2.5 * P)[1], 1.0)
    b.land(np.ones((20, 2)), t_obs=2 * P)  # rows at 3P, ...; seam at 2.5P
    out = b.sample(3 * P)
    assert 0.0 < out[0] < 1.0  # joint blends
    assert out[1] == 1.0  # gripper takes the new value outright


def test_rebase_resumes_from_the_last_sample_not_from_the_observation():
    b = ReferenceBuffer(FPS)
    b.land(ramp(3, cols=1), 0.0)
    for t in (8 * P, 9 * P, 10 * P):  # pins rows at 8P, 9P, 10P; arm froze at 2
        b.sample(t)
    assert b.exhausted(11 * P)
    # Observed mid-motion at 7.5P: rows at 8.5P (50), 9.5P (51), 10.5P (52), ...
    b.land(ramp(10, start=50.0, cols=1), t_obs=7.5 * P)
    # Freeze at the last sample (10P): the new plan's value there is 51.5.
    np.testing.assert_array_equal(b.sample(24 * P), [51.5])
    np.testing.assert_array_equal(b.sample(24.5 * P), [52.0])
    assert b.rebases == 2


def test_rebase_keeps_row_zero_when_the_observation_follows_the_freeze():
    b = ReferenceBuffer(FPS)
    b.land(ramp(3, cols=1), 0.0)
    for t in (8 * P, 9 * P, 10 * P):
        b.sample(t)
    b.land(ramp(10, start=50.0, cols=1), t_obs=12 * P)  # observed at rest
    np.testing.assert_array_equal(b.sample(30 * P), [50.0])


def test_a_chunk_entirely_behind_the_freeze_is_dropped():
    b = ReferenceBuffer(FPS)
    b.land(ramp(3, cols=1), 0.0)
    for t in (8 * P, 9 * P, 10 * P):
        b.sample(t)
    b.land(ramp(3, start=50.0, cols=1), t_obs=0.0)  # rows at P..3P, all behind 10P
    assert b.exhausted(11 * P)
    assert b.rebases == 1


def test_a_landing_after_the_plan_ran_out_rebases_even_if_rows_remain_past_the_last_sample():
    b = ReferenceBuffer(FPS)
    b.land(ramp(3, cols=1), 0.0)
    for t in (8 * P, 9 * P, 9.5 * P):  # rows at 8P, 9P, 10P; last sample short of 10P
        b.sample(t)
    assert b.exhausted(10.5 * P)  # the tick's sample time passed the last row
    b.land(ramp(10, start=50.0, cols=1), t_obs=9 * P)  # rows at 10P, ...
    # Crossfade would keep these rows at 10P.., already behind the next sample.
    # Rebase: freeze = max(9.5P, 10P) = 10P = row 0, pinned to the next sample.
    np.testing.assert_array_equal(b.sample(20 * P), [50.0])
    assert b.rebases == 2


def test_land_twice_before_any_sample_keeps_the_newest_chunk():
    b = ReferenceBuffer(FPS)
    b.land(ramp(3, cols=1), 0.0)
    b.land(ramp(3, start=9.0, cols=1), 0.0)
    np.testing.assert_array_equal(b.sample(1.0), [9.0])


def test_sampled_obs_t_names_the_observation_behind_the_sample():
    b = ReferenceBuffer(FPS, blend_s=0.0)
    b.land(ramp(10), 0.0)
    b.sample(P)
    assert b.sampled_obs_t == 0.0
    b.land(ramp(10), t_obs=2 * P)
    b.sample(4 * P)
    assert b.sampled_obs_t == 2 * P


@pytest.mark.parametrize(
    "bad", [np.zeros(3), np.zeros((0, 2)), [[1.0, 2.0]]], ids=["1d", "empty", "list"]
)
def test_land_rejects_malformed_chunks(bad):
    with pytest.raises(ReferenceBufferError):
        ReferenceBuffer(FPS).land(bad, 0.0)


def test_land_rejects_a_column_count_change():
    b = ReferenceBuffer(FPS)
    b.land(ramp(3, cols=2), 0.0)
    with pytest.raises(ReferenceBufferError, match="columns"):
        b.land(ramp(3, cols=3), 0.0)
