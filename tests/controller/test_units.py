import numpy as np
import pytest

from vla.config_util import VLAError
from vla.controller.units import UnitError, from_degrees, to_degrees


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
