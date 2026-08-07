import math

import pytest
from vla.config_util import (
    ConfigError,
    VLAError,
    as_bool,
    as_choice,
    as_env_var_name,
    as_float,
    as_int,
    as_str,
    redact_secret,
)


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


# --- VLAError base ---


def test_config_error_is_a_vla_error():
    assert issubclass(ConfigError, VLAError)


def test_config_error_is_still_a_value_error():
    # Existing call sites catch ValueError; the new base must not replace it.
    assert issubclass(ConfigError, ValueError)


# --- redact_secret ---


def test_redact_secret_shows_first_four_chars_and_length():
    assert redact_secret("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345") == "ABCD...<32 chars>"


def test_redact_secret_never_contains_the_full_value():
    secret = "hf_AbCdEfGhIjKlMnOpQrStUv123456"
    redacted = redact_secret(secret)
    assert secret not in redacted


def test_redact_secret_handles_short_values():
    assert redact_secret("ab") == "ab...<2 chars>"


# --- as_env_var_name ---


def test_as_env_var_name_accepts_valid_name():
    assert as_env_var_name("HF_TOKEN", "field") == "HF_TOKEN"


def test_as_env_var_name_accepts_leading_underscore():
    assert as_env_var_name("_FOO_BAR", "field") == "_FOO_BAR"


def test_as_env_var_name_rejects_leading_digit():
    with pytest.raises(ConfigError, match="field"):
        as_env_var_name("1FOO", "field")


def test_as_env_var_name_rejects_value_with_hyphen():
    # Shaped like a plausible pasted secret (e.g. an API key), not an env
    # var name -- this is the case the regex is specifically meant to catch.
    with pytest.raises(ConfigError, match="field"):
        as_env_var_name("sk-live-1234567890abcdef1234567890abcdef", "field")


def test_as_env_var_name_rejects_value_over_64_chars():
    with pytest.raises(ConfigError, match="field"):
        as_env_var_name("A" * 65, "field")


def test_as_env_var_name_accepts_value_exactly_64_chars():
    assert as_env_var_name("A" * 64, "field") == "A" * 64


def test_as_env_var_name_rejects_non_string():
    with pytest.raises(ConfigError, match="field"):
        as_env_var_name(1.0, "field")


def test_as_env_var_name_never_echoes_the_full_rejected_value():
    offending = "sk-live-1234567890abcdef1234567890abcdef"
    with pytest.raises(ConfigError) as excinfo:
        as_env_var_name(offending, "field")
    assert offending not in str(excinfo.value)
