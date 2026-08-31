import numpy as np
import pytest

from vla.config_util import VLAError
from vla.controller.units import (
    SEGMENT_UNITS,
    UnitError,
    UnitSegment,
    VectorUnits,
    from_degrees,
    from_working,
    to_degrees,
    to_working,
)


def test_degrees_is_identity():
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(from_degrees(x, "degrees"), x)
    np.testing.assert_allclose(to_degrees(x, "degrees"), x)


def test_degrees_to_radians():
    np.testing.assert_allclose(
        from_degrees(np.array([180.0], dtype=np.float32), "radians"), [np.pi], rtol=1e-6
    )


def test_radians_to_degrees():
    np.testing.assert_allclose(
        to_degrees(np.array([np.pi], dtype=np.float32), "radians"), [180.0], rtol=1e-6
    )


def test_roundtrip_is_stable():
    x = np.array([12.5, -90.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(to_degrees(from_degrees(x, "radians"), "radians"), x, rtol=1e-5)


def test_normalized_is_not_yet_supported():
    # Normalized units need per-joint min/max, an open question in the spec.
    # ControllerConfig rejects it at config time; here it is just unknown.
    with pytest.raises(UnitError, match="normalized"):
        from_degrees(np.array([0.0], dtype=np.float32), "normalized")


def test_unknown_unit_errors():
    with pytest.raises(UnitError, match="unknown"):
        from_degrees(np.array([0.0], dtype=np.float32), "furlongs")


def test_unknown_unit_message_lists_only_actually_convertible_units():
    # The message must list only units a caller can actually pass and have
    # work -- never "normalized", which this module cannot convert.
    with pytest.raises(UnitError) as exc_info:
        from_degrees(np.array([0.0], dtype=np.float32), "furlongs")
    message = str(exc_info.value)
    assert "degrees" in message
    assert "radians" in message
    assert "normalized" not in message


# ---------------------------------------------------------------------------
# Standing requirement 5: this module's own exception type, and it must sit
# in the shared VLAError hierarchy like ConfigError/WireError/ResolveError/
# PrefixError so a caller can `except VLAError` uniformly.
# ---------------------------------------------------------------------------


def test_unit_error_is_a_vla_error():
    assert issubclass(UnitError, VLAError)


def test_unit_error_is_a_value_error():
    assert issubclass(UnitError, ValueError)


# ---------------------------------------------------------------------------
# Standing requirement 1: assert every *valid* unit is accepted and produces
# the expected numeric result -- not only that invalid units are rejected.
# Hardcoded literals here (not looping over SUPPORTED_UNITS) per standing
# requirement 7: parametrizing off the module's own constant would make a
# shrunk tuple invisible to this test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit,input_value,expected",
    [
        ("degrees", 45.0, 45.0),
        ("radians", 45.0, np.deg2rad(45.0)),
    ],
)
def test_from_degrees_accepts_every_supported_unit(unit, input_value, expected):
    result = from_degrees(np.array([input_value], dtype=np.float32), unit)
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


@pytest.mark.parametrize(
    "unit,input_value,expected",
    [
        ("degrees", 45.0, 45.0),
        ("radians", np.deg2rad(45.0), 45.0),
    ],
)
def test_to_degrees_accepts_every_supported_unit(unit, input_value, expected):
    result = to_degrees(np.array([input_value], dtype=np.float32), unit)
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


# ---------------------------------------------------------------------------
# Round-trip precision at the boundary angles most likely to expose a sign
# or wrap-around bug: +/-180 degrees, exactly 0, and a small angle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degrees_value", [180.0, -180.0, 0.0, 0.001, -0.001])
def test_roundtrip_stable_at_realistic_joint_angles(degrees_value):
    x = np.array([degrees_value], dtype=np.float32)
    roundtripped = to_degrees(from_degrees(x, "radians"), "radians")
    np.testing.assert_allclose(roundtripped, x, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Per-segment units, for the delta-EE vectors that mix quantities.
#
# The two layouts under test are the only two the module actually builds:
# the 9-dim state (3 lengths + 6 dimensionless rotation-matrix entries) and
# the 6-dim action (3 lengths + 3 angles).
# ---------------------------------------------------------------------------

STATE_UNITS_MM = VectorUnits((UnitSegment(3, "millimeters"), UnitSegment(6, "unitless")))
ACTION_UNITS_MM_RAD = VectorUnits((UnitSegment(3, "millimeters"), UnitSegment(3, "radians")))
ACTION_UNITS_M_DEG = VectorUnits((UnitSegment(3, "meters"), UnitSegment(3, "degrees")))


def test_the_native_delta_ee_layout_is_an_identity_both_ways():
    """A checkpoint recorded the way the converter writes it needs no scaling.

    Millimetres and radians *are* the working units, so this is the case that
    must be exactly the identity -- not merely close. Anything else means a
    stray conversion crept into the common path.
    """
    action = np.array([9.31, -2.0, 0.5, 0.0142, -0.003, 0.0071], dtype=np.float32)
    np.testing.assert_array_equal(to_working(action, ACTION_UNITS_MM_RAD), action)
    np.testing.assert_array_equal(from_working(action, ACTION_UNITS_MM_RAD), action)


def test_each_segment_converts_independently():
    """The whole point: one vector, two different units, no cross-contamination."""
    action = np.array([0.00931, -0.002, 0.0005, 0.8135, -0.1719, 0.4068], dtype=np.float32)
    working = to_working(action, ACTION_UNITS_M_DEG)
    np.testing.assert_allclose(working[:3], [9.31, -2.0, 0.5], rtol=1e-5)
    np.testing.assert_allclose(working[3:], [0.0142, -0.003, 0.0071], rtol=1e-3)


def test_a_unitless_segment_is_never_scaled():
    """The state's rotation rows are direction cosines under every checkpoint.

    Scaling them by a length or angle factor would corrupt the rotation while
    leaving a perfectly well-formed 9-vector, which `state_rotation` would
    happily Gram-Schmidt into a *different* orientation.
    """
    state = np.array(
        [0.3054, -0.01275, 0.2319, 0.2513, 0.9678, 0.0139, 0.9676, -0.2509, -0.0271],
        dtype=np.float32,
    )
    working = to_working(state, VectorUnits((UnitSegment(3, "meters"), UnitSegment(6, "unitless"))))
    np.testing.assert_allclose(working[:3], [305.4, -12.75, 231.9], rtol=1e-5)
    np.testing.assert_array_equal(working[3:], state[3:])


@pytest.mark.parametrize(
    "units",
    [STATE_UNITS_MM, ACTION_UNITS_MM_RAD, ACTION_UNITS_M_DEG],
)
def test_segment_conversion_roundtrips(units):
    values = np.arange(1, units.size + 1, dtype=np.float32) * 0.37
    np.testing.assert_allclose(
        from_working(to_working(values, units), units), values, rtol=1e-5
    )


def test_size_is_the_sum_of_the_segments():
    assert STATE_UNITS_MM.size == 9
    assert ACTION_UNITS_MM_RAD.size == 6


def test_a_wrong_width_vector_is_refused_rather_than_broadcast():
    """numpy would happily broadcast a 3-vector against a 6-scale array.

    Left to itself that turns a truncated policy output into six silently
    wrong numbers instead of an error.
    """
    with pytest.raises(UnitError, match="6 components"):
        to_working(np.zeros(3, dtype=np.float32), ACTION_UNITS_MM_RAD)


def test_an_unknown_segment_unit_is_refused_at_construction():
    with pytest.raises(UnitError, match="furlongs"):
        VectorUnits((UnitSegment(3, "furlongs"),))


def test_a_nonpositive_segment_size_is_refused():
    with pytest.raises(UnitError, match="positive"):
        VectorUnits((UnitSegment(0, "millimeters"),))


def test_an_empty_segment_list_is_refused():
    with pytest.raises(UnitError, match="at least one segment"):
        VectorUnits(())


def test_vector_units_is_hashable_so_a_frozen_config_stays_frozen():
    assert hash(STATE_UNITS_MM) == hash(
        VectorUnits((UnitSegment(3, "millimeters"), UnitSegment(6, "unitless")))
    )


# Hardcoded, not looped off the constant, for the same reason the joints-path
# tests above are: parametrizing off `SEGMENT_UNITS` would make a shrunk tuple
# invisible here.
@pytest.mark.parametrize(
    "unit,value,expected",
    [
        ("millimeters", 1.0, 1.0),
        ("meters", 1.0, 1000.0),
        ("radians", 1.0, 1.0),
        ("degrees", 180.0, np.pi),
        ("unitless", 0.5, 0.5),
    ],
)
def test_every_segment_unit_converts_to_working(unit, value, expected):
    result = to_working(np.array([value], dtype=np.float32), VectorUnits((UnitSegment(1, unit),)))
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


def test_the_supported_segment_units_are_exactly_these_five():
    assert set(SEGMENT_UNITS) == {
        "millimeters",
        "meters",
        "radians",
        "degrees",
        "unitless",
    }


def test_segment_conversion_returns_float32_whatever_the_input():
    result = to_working(np.zeros(6, dtype=np.float64), ACTION_UNITS_MM_RAD)
    assert result.dtype == np.float32
