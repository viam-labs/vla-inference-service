import math

import pytest
from vla.config_util import ConfigError, as_bool, as_choice, as_float, as_int, as_str


# --- as_int ---


def test_as_int_accepts_plain_int():
    assert as_int(2, "field") == 2


def test_as_int_accepts_integral_float():
    assert as_int(2.0, "field") == 2


def test_as_int_rejects_fractional_float():
    with pytest.raises(ConfigError, match="field"):
        as_int(2.5, "field")


def test_as_int_rejects_non_numeric():
    with pytest.raises(ConfigError, match="field"):
        as_int("two", "field")


def test_as_int_rejects_bool():
    with pytest.raises(ConfigError, match="field"):
        as_int(True, "field")


def test_as_int_enforces_minimum():
    with pytest.raises(ConfigError, match="field"):
        as_int(-1, "field", minimum=0)


def test_as_int_enforces_maximum():
    with pytest.raises(ConfigError, match="field"):
        as_int(1000, "field", maximum=100)


def test_as_int_accepts_value_within_bounds():
    assert as_int(5, "field", minimum=0, maximum=10) == 5


# --- as_float ---


def test_as_float_accepts_plain_float():
    assert as_float(1.5, "field") == 1.5


def test_as_float_accepts_int():
    assert as_float(2, "field") == 2.0


def test_as_float_rejects_bool():
    with pytest.raises(ConfigError, match="field"):
        as_float(True, "field")


def test_as_float_rejects_non_numeric():
    with pytest.raises(ConfigError, match="field"):
        as_float("loud", "field")


def test_as_float_rejects_nan():
    with pytest.raises(ConfigError, match="field"):
        as_float(math.nan, "field")


def test_as_float_rejects_infinity():
    with pytest.raises(ConfigError, match="field"):
        as_float(math.inf, "field")


def test_as_float_enforces_minimum():
    with pytest.raises(ConfigError, match="field"):
        as_float(0.0, "field", minimum=1e-6)


def test_as_float_enforces_maximum():
    with pytest.raises(ConfigError, match="field"):
        as_float(1e30, "field", maximum=1000.0)


def test_as_float_accepts_value_within_bounds():
    assert as_float(5.0, "field", minimum=0.0, maximum=10.0) == 5.0


# --- as_bool ---


def test_as_bool_accepts_true():
    assert as_bool(True, "field") is True


def test_as_bool_accepts_false():
    assert as_bool(False, "field") is False


def test_as_bool_rejects_string():
    with pytest.raises(ConfigError, match="field"):
        as_bool("false", "field")


def test_as_bool_rejects_int():
    with pytest.raises(ConfigError, match="field"):
        as_bool(1, "field")


# --- as_str ---


def test_as_str_accepts_string():
    assert as_str("hello", "field") == "hello"


def test_as_str_rejects_float():
    with pytest.raises(ConfigError, match="field"):
        as_str(1.0, "field")


def test_as_str_rejects_int():
    with pytest.raises(ConfigError, match="field"):
        as_str(1, "field")


def test_as_str_rejects_bool():
    with pytest.raises(ConfigError, match="field"):
        as_str(True, "field")


# --- as_choice ---


def test_as_choice_accepts_member():
    assert as_choice("cpu", "field", ("auto", "cpu", "cuda")) == "cpu"


def test_as_choice_rejects_non_member():
    with pytest.raises(ConfigError, match="field"):
        as_choice("tpu", "field", ("auto", "cpu", "cuda"))
