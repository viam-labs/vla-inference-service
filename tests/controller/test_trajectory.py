import pytest

from vla.controller.trajectory import interpolate


def test_n_equals_1_returns_target_only():
    assert interpolate([0, 0], [4, 30], 1) == [[4, 30]]


def test_n_equals_4_linear_spacing():
    assert interpolate([0, 10], [4, 30], 4) == [[1, 15], [2, 20], [3, 25], [4, 30]]


def test_length_equals_n():
    assert len(interpolate([0, 0, 0], [1, 2, 3], 7)) == 7


def test_last_point_equals_target_exactly():
    # 0.29 -> 0.87 over 3 steps is a case where naive arithmetic drifts:
    # 0.29 + (0.87 - 0.29) * 3 / 3 == 0.8700000000000001, not 0.87. The last
    # point must come from `list(target)`, not accumulation.
    prev = [0.29]
    target = [0.87]
    points = interpolate(prev, target, 3)
    assert points[-1] == list(target)


def test_raises_on_n_less_than_1():
    with pytest.raises(ValueError):
        interpolate([0, 0], [1, 1], 0)


def test_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        interpolate([0, 0], [1, 1, 1], 3)
