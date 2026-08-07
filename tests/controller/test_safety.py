import logging

import numpy as np
import pytest

from vla.config_util import VLAError
from vla.controller.safety import SafetyError, SafetyLayer, SafetyLimits


def _layer(**kw):
    defaults = dict(
        max_joint_delta_degs=8.0,
        max_start_delta_degs=15.0,
        joint_limits_degs=None,
        gripper_in_degrees=True,
    )
    defaults.update(kw)
    return SafetyLayer(SafetyLimits(**defaults))


# ---------------------------------------------------------------------------
# 1. reject NaN/inf -- fail the chunk, never clamp
# ---------------------------------------------------------------------------


def test_rejects_nan():
    with pytest.raises(SafetyError, match="finite"):
        _layer().apply(np.array([1.0, np.nan]), current=np.array([0.0, 0.0]))


def test_rejects_inf():
    with pytest.raises(SafetyError, match="finite"):
        _layer().apply(np.array([np.inf, 1.0]), current=np.array([0.0, 0.0]))


def test_rejects_negative_inf():
    with pytest.raises(SafetyError, match="finite"):
        _layer().apply(np.array([-np.inf, 1.0]), current=np.array([0.0, 0.0]))


def test_nan_in_current_is_also_rejected():
    # current comes from a live sensor read; garbage there is just as
    # dangerous as garbage in the action.
    with pytest.raises(SafetyError, match="finite"):
        _layer().apply(np.array([1.0, 1.0]), current=np.array([np.nan, 0.0]))


def test_rejection_does_not_clamp():
    # The contract is "fail the chunk" -- not "fail, but here's a clamped
    # value anyway". A mutant that clamped NaN to 0 and returned it instead
    # of raising must be caught, not just one that returns unclamped NaN.
    with pytest.raises(SafetyError):
        _layer().apply(np.array([1.0, np.nan]), current=np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# 2. dimension check
# ---------------------------------------------------------------------------


def test_rejects_dimension_mismatch():
    with pytest.raises(SafetyError, match="dimension"):
        _layer().apply(np.array([1.0, 2.0, 3.0]), current=np.array([0.0, 0.0]))


def test_rejects_dimension_mismatch_other_direction():
    with pytest.raises(SafetyError, match="dimension"):
        _layer().apply(np.array([1.0]), current=np.array([0.0, 0.0, 0.0]))


def test_check_start_rejects_dimension_mismatch():
    with pytest.raises(SafetyError, match="dimension"):
        _layer().check_start(np.array([1.0, 2.0]), current=np.array([0.0]))


def test_check_start_rejects_nan():
    with pytest.raises(SafetyError, match="finite"):
        _layer().check_start(np.array([np.nan]), current=np.array([0.0]))


# ---------------------------------------------------------------------------
# basic pass-through / defaults
# ---------------------------------------------------------------------------


def test_within_limits_passes_through():
    out = _layer().apply(np.array([2.0, -3.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [2.0, -3.0])


def test_default_limits_values():
    limits = SafetyLimits()
    assert limits.max_joint_delta_degs == 8.0
    assert limits.max_start_delta_degs == 15.0
    assert limits.joint_limits_degs is None
    assert limits.gripper_in_degrees is True


def test_clamp_counts_start_at_zero():
    layer = _layer()
    assert layer.clamp_counts["delta"] == 0
    assert layer.clamp_counts["limit"] == 0
    assert layer.clamp_counts["gripper"] == 0


# ---------------------------------------------------------------------------
# 3. delta clamp against current measured position
# ---------------------------------------------------------------------------


def test_delta_clamped_against_current_measured_position():
    layer = _layer(max_joint_delta_degs=5.0)
    out = layer.apply(np.array([100.0]), current=np.array([10.0]))
    np.testing.assert_allclose(out, [15.0])
    assert layer.clamp_counts["delta"] == 1


def test_delta_clamp_is_symmetric():
    out = _layer(max_joint_delta_degs=5.0).apply(np.array([-100.0]), current=np.array([10.0]))
    np.testing.assert_allclose(out, [5.0])


def test_delta_clamp_uses_current_not_last_commanded():
    # A stalled arm: the previous commanded value is far from where the arm
    # actually sits. Clamping against the last *commanded* value would let
    # the target keep marching upward every tick even though the arm never
    # moved. Clamping against the measured `current` prevents that.
    layer = _layer(max_joint_delta_degs=5.0)
    first = layer.apply(np.array([100.0]), current=np.array([0.0]))
    np.testing.assert_allclose(first, [5.0])
    # Arm is stalled -- still measured at 0.0, not at the 5.0 we asked for.
    second = layer.apply(np.array([100.0]), current=np.array([0.0]))
    np.testing.assert_allclose(second, [5.0])  # not 10.0


def test_clamp_counts_accumulate_for_diagnostics():
    layer = _layer(max_joint_delta_degs=1.0)
    for _ in range(3):
        layer.apply(np.array([100.0]), current=np.array([0.0]))
    assert layer.clamp_counts["delta"] == 3


def test_noop_action_identical_to_current_is_not_counted():
    layer = _layer()
    current = np.array([12.5, -3.2])
    out = layer.apply(current.copy(), current=current)
    np.testing.assert_allclose(out, current)
    assert layer.clamp_counts["delta"] == 0
    assert layer.clamp_counts["limit"] == 0


def test_negative_zero_action_is_not_clamped():
    layer = _layer()
    out = layer.apply(np.array([-0.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [0.0])
    assert layer.clamp_counts["delta"] == 0
    assert layer.clamp_counts["limit"] == 0


def test_many_tiny_deltas_accumulate_no_clamp():
    # A realistic policy emits a long chunk of small per-step deltas; none
    # of them individually should trip the clamp, and repeated legitimate
    # motion must never spuriously increment clamp_counts.
    layer = _layer(max_joint_delta_degs=8.0)
    current = np.array([0.0])
    for _ in range(1000):
        target = current + 0.01
        current = layer.apply(target, current=current)
    assert layer.clamp_counts["delta"] == 0
    np.testing.assert_allclose(current, [10.0], atol=1e-6)


# ---------------------------------------------------------------------------
# 4. joint limit clamp
# ---------------------------------------------------------------------------


def test_joint_limits_clamp():
    layer = _layer(joint_limits_degs=[(-90.0, 90.0)], max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([200.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [90.0])
    assert layer.clamp_counts["limit"] == 1


def test_joint_limits_clamp_lower_bound():
    layer = _layer(joint_limits_degs=[(-90.0, 90.0)], max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([-200.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [-90.0])
    assert layer.clamp_counts["limit"] == 1


def test_limit_layer_skipped_when_unset():
    layer = _layer(joint_limits_degs=None, max_joint_delta_degs=1e7)
    out = layer.apply(np.array([5000.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [5000.0])
    assert layer.clamp_counts["limit"] == 0


def test_joint_limit_exactly_at_boundary_is_not_counted_as_clamped():
    # Decision: the limit clamp uses inclusive bounds (np.clip semantics),
    # so a value exactly equal to the limit passes through unchanged and is
    # not counted -- only a value that would actually be altered counts.
    layer = _layer(joint_limits_degs=[(-90.0, 90.0)], max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([90.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [90.0])
    assert layer.clamp_counts["limit"] == 0


def test_joint_limits_apply_per_joint_independently():
    layer = _layer(
        joint_limits_degs=[(-10.0, 10.0), (-90.0, 90.0)], max_joint_delta_degs=1000.0
    )
    out = layer.apply(np.array([50.0, 50.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [10.0, 50.0])
    assert layer.clamp_counts["limit"] == 1


def test_joint_limits_shorter_than_action_leaves_extra_joints_unclamped():
    # joint_limits_degs is optional per-joint; a shorter list than the
    # action vector must not raise or clamp the joints beyond it.
    layer = _layer(joint_limits_degs=[(-10.0, 10.0)], max_joint_delta_degs=1e7)
    out = layer.apply(np.array([50.0, 5000.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [10.0, 5000.0])
    assert layer.clamp_counts["limit"] == 1


# ---------------------------------------------------------------------------
# clamp interaction: delta vs limit composition
# ---------------------------------------------------------------------------


def test_within_delta_budget_but_outside_joint_limit_clamps_to_limit():
    layer = _layer(max_joint_delta_degs=8.0, joint_limits_degs=[(-5.0, 5.0)])
    out = layer.apply(np.array([7.0]), current=np.array([0.0]))
    # delta of 7 is within the 8-degree budget, so the delta clamp does not
    # fire; the joint-limit clamp catches it instead.
    np.testing.assert_allclose(out, [5.0])
    assert layer.clamp_counts["delta"] == 0
    assert layer.clamp_counts["limit"] == 1


def test_outside_delta_budget_but_within_joint_limit_clamps_to_delta():
    layer = _layer(max_joint_delta_degs=3.0, joint_limits_degs=[(-90.0, 90.0)])
    out = layer.apply(np.array([50.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [3.0])
    assert layer.clamp_counts["delta"] == 1
    assert layer.clamp_counts["limit"] == 0


def test_delta_clamp_runs_before_limit_clamp_order_matters():
    # `current` (20.0) is itself already outside the configured limit
    # (-5..5). Delta clamps relative to the *measured* current position
    # first (20 + 3 = 23), and only then is the result clipped to the
    # joint limit (23 -> 5). If the order were reversed -- limit first,
    # then delta relative to the same unclamped current -- the result
    # would be 17.0 instead: limit-clip(25, -5, 5) = 5, then
    # delta-clip(5 - 20, -3, 3) = -3, giving current + (-3) = 17.0. Asserting
    # the final numeric value pins the documented order, not just "some
    # clamp fired".
    layer = _layer(max_joint_delta_degs=3.0, joint_limits_degs=[(-5.0, 5.0)])
    out = layer.apply(np.array([25.0]), current=np.array([20.0]))
    np.testing.assert_allclose(out, [5.0])
    assert layer.clamp_counts["delta"] == 1
    assert layer.clamp_counts["limit"] == 1


def test_action_violating_neither_clamp_is_unchanged():
    layer = _layer(max_joint_delta_degs=8.0, joint_limits_degs=[(-90.0, 90.0)])
    out = layer.apply(np.array([3.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [3.0])
    assert layer.clamp_counts["delta"] == 0
    assert layer.clamp_counts["limit"] == 0


# ---------------------------------------------------------------------------
# normalized gripper channel: degree clamps must not apply to it
# ---------------------------------------------------------------------------


def test_normalized_gripper_channel_clamped_to_unit_range():
    # Degrees-based limits are meaningless for a 0..1 channel; it gets [0,1] instead.
    layer = _layer(
        gripper_in_degrees=False, joint_limits_degs=[(-90.0, 90.0)], max_joint_delta_degs=1000.0
    )
    out = layer.apply(np.array([10.0, 3.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [10.0, 1.0])


def test_normalized_gripper_channel_clamps_negative_to_zero():
    layer = _layer(gripper_in_degrees=False, max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([10.0, -3.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [10.0, 0.0])
    assert layer.clamp_counts["gripper"] == 1


def test_normalized_gripper_channel_not_delta_clamped():
    # A gripper snapping from fully open to fully closed in one tick (delta
    # of 1.0) is completely normal and must not trip the degree-based delta
    # clamp -- that clamp does not apply to this channel at all.
    layer = _layer(gripper_in_degrees=False, max_joint_delta_degs=0.01, joint_limits_degs=None)
    out = layer.apply(np.array([0.0, 1.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [0.0, 1.0])
    assert layer.clamp_counts["delta"] == 0


def test_normalized_gripper_channel_within_unit_range_is_unchanged():
    layer = _layer(gripper_in_degrees=False, max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([10.0, 0.5]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [10.0, 0.5])
    assert layer.clamp_counts["gripper"] == 0


def test_gripper_in_degrees_true_gripper_channel_uses_degree_limits_not_unit_clamp():
    # arm_joint gripper: the trailing channel IS a joint in degrees, so it
    # must go through the ordinary degree-based delta/limit clamps, not the
    # [0, 1] unit clamp.
    layer = _layer(
        gripper_in_degrees=True,
        joint_limits_degs=[(-90.0, 90.0), (-10.0, 10.0)],
        max_joint_delta_degs=1000.0,
    )
    out = layer.apply(np.array([5.0, 50.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [5.0, 10.0])
    assert layer.clamp_counts["gripper"] == 0
    assert layer.clamp_counts["limit"] == 1


def test_joint_limits_degs_entry_beyond_gripper_index_is_ignored():
    # Defense in depth: config-time validation (elsewhere) is supposed to
    # keep joint_limits_degs from ever carrying a trailing pair for a
    # non-degree gripper channel, but this layer must not rely on that --
    # if such an entry is present anyway, the degree-shaped limit clamp
    # must still never touch the gripper index. Only the [0, 1] unit clamp
    # may.
    layer = _layer(
        gripper_in_degrees=False,
        joint_limits_degs=[(-90.0, 90.0), (-0.3, 0.3)],
        max_joint_delta_degs=1000.0,
    )
    out = layer.apply(np.array([50.0, 0.5]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [50.0, 0.5])
    assert layer.clamp_counts["limit"] == 0
    assert layer.clamp_counts["gripper"] == 0


def test_gripper_only_action_clamps_to_unit_range():
    # Degenerate n=1 case: the whole action vector is the gripper channel.
    layer = _layer(gripper_in_degrees=False, max_joint_delta_degs=1000.0, joint_limits_degs=None)
    out = layer.apply(np.array([5.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [1.0])
    assert layer.clamp_counts["gripper"] == 1
    assert layer.clamp_counts["delta"] == 0


# ---------------------------------------------------------------------------
# check_start: first-move guard
# ---------------------------------------------------------------------------


def test_start_delta_within_budget_allowed():
    _layer(max_start_delta_degs=15.0).check_start(np.array([10.0]), current=np.array([0.0]))


def test_start_delta_exactly_at_budget_is_allowed():
    _layer(max_start_delta_degs=15.0).check_start(np.array([15.0]), current=np.array([0.0]))


def test_start_delta_exceeded_refuses():
    with pytest.raises(SafetyError, match="max_start_delta_degs"):
        _layer(max_start_delta_degs=15.0).check_start(np.array([50.0]), current=np.array([0.0]))


def test_start_delta_just_over_budget_refuses():
    with pytest.raises(SafetyError, match="max_start_delta_degs"):
        _layer(max_start_delta_degs=15.0).check_start(
            np.array([15.001]), current=np.array([0.0])
        )


def test_check_start_does_not_touch_clamp_counts():
    layer = _layer(max_start_delta_degs=15.0)
    layer.check_start(np.array([10.0]), current=np.array([0.0]))
    assert layer.clamp_counts["delta"] == 0
    assert layer.clamp_counts["limit"] == 0
    assert layer.clamp_counts["gripper"] == 0


def test_check_start_uses_max_abs_across_all_joints():
    # A refusal must trigger from the single worst joint, not an average.
    layer = _layer(max_start_delta_degs=15.0)
    with pytest.raises(SafetyError):
        layer.check_start(np.array([1.0, 1.0, 1.0, 50.0]), current=np.array([0.0, 0.0, 0.0, 0.0]))


def test_check_start_symmetric_negative_direction():
    with pytest.raises(SafetyError, match="max_start_delta_degs"):
        _layer(max_start_delta_degs=15.0).check_start(np.array([-50.0]), current=np.array([0.0]))


# ---------------------------------------------------------------------------
# 6. logging whenever a clamp engages
# ---------------------------------------------------------------------------


def test_delta_clamp_logs_warning(caplog):
    layer = _layer(max_joint_delta_degs=1.0)
    with caplog.at_level(logging.WARNING):
        layer.apply(np.array([100.0]), current=np.array([0.0]))
    assert any("clamp" in rec.message.lower() for rec in caplog.records)


def test_limit_clamp_logs_warning(caplog):
    layer = _layer(joint_limits_degs=[(-5.0, 5.0)], max_joint_delta_degs=1000.0)
    with caplog.at_level(logging.WARNING):
        layer.apply(np.array([50.0]), current=np.array([0.0]))
    assert any("clamp" in rec.message.lower() for rec in caplog.records)


def test_gripper_clamp_logs_warning(caplog):
    layer = _layer(gripper_in_degrees=False, max_joint_delta_degs=1000.0)
    with caplog.at_level(logging.WARNING):
        layer.apply(np.array([5.0]), current=np.array([0.0]))
    assert any("clamp" in rec.message.lower() for rec in caplog.records)


def test_no_clamp_does_not_log_warning(caplog):
    layer = _layer()
    with caplog.at_level(logging.WARNING):
        layer.apply(np.array([1.0]), current=np.array([0.0]))
    assert not any("clamp" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# error hierarchy
# ---------------------------------------------------------------------------


def test_safety_error_is_a_vla_error():
    assert issubclass(SafetyError, VLAError)
    assert issubclass(SafetyError, RuntimeError)
