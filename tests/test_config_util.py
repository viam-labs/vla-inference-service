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

# Every helper is a coercion table: one accepted-value case and one rejected
# case per rule. Parametrized rather than written out one test per row -- the
# rows ARE the specification, and a table makes a missing rule visible.


@pytest.mark.parametrize(
    "value,expected",
    [
        (2, 2),  # plain int
        (2.0, 2),  # integral float, as protobuf Struct delivers it
        (-3.0, -3),
    ],
)
def test_as_int_accepts(value, expected):
    assert as_int(value, "field") == expected


@pytest.mark.parametrize(
    "value,kwargs",
    [
        (2.5, {}),  # fractional float is a typo, not a truncation target
        ("two", {}),
        (True, {}),  # bool is an int in Python but never a valid config number
        (None, {}),
        (-1, {"minimum": 0}),
        (1000, {"maximum": 100}),
    ],
)
def test_as_int_rejects(value, kwargs):
    with pytest.raises(ConfigError, match="field"):
        as_int(value, "field", **kwargs)


def test_as_int_accepts_value_within_bounds():
    assert as_int(5, "field", minimum=0, maximum=10) == 5


@pytest.mark.parametrize("value,expected", [(1.5, 1.5), (2, 2.0), (-0.5, -0.5)])
def test_as_float_accepts(value, expected):
    assert as_float(value, "field") == expected


@pytest.mark.parametrize(
    "value,kwargs",
    [
        (True, {}),  # would silently become 1.0
        ("loud", {}),
        (None, {}),
        (math.nan, {}),  # passes any `<= 0` guard, then poisons downstream math
        (math.inf, {}),
        (-math.inf, {}),
        (0.0, {"minimum": 1e-6}),
        (1e30, {"maximum": 1000.0}),
    ],
)
def test_as_float_rejects(value, kwargs):
    with pytest.raises(ConfigError, match="field"):
        as_float(value, "field", **kwargs)


def test_as_float_accepts_value_within_bounds():
    assert as_float(5.0, "field", minimum=0.0, maximum=10.0) == 5.0


@pytest.mark.parametrize("value", [True, False])
def test_as_bool_accepts_real_bools(value):
    assert as_bool(value, "field") is value


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None])
def test_as_bool_rejects_everything_else(value):
    # bool("false") is True, so a loose check would flip the field's meaning.
    with pytest.raises(ConfigError, match="field"):
        as_bool(value, "field")


def test_as_str_accepts_string():
    assert as_str("hello", "field") == "hello"


@pytest.mark.parametrize("value", [1.0, 1, True, None, ["a"]])
def test_as_str_rejects_non_strings(value):
    with pytest.raises(ConfigError, match="field"):
        as_str(value, "field")


def test_as_choice_accepts_member():
    assert as_choice("cpu", "field", ("auto", "cpu", "cuda")) == "cpu"


@pytest.mark.parametrize("value", ["tpu", "", None, 1.0])
def test_as_choice_rejects_non_member(value):
    with pytest.raises(ConfigError, match="field"):
        as_choice(value, "field", ("auto", "cpu", "cuda"))


# --- VLAError base ---


def test_config_error_is_a_vla_error():
    assert issubclass(ConfigError, VLAError)


def test_config_error_is_still_a_value_error():
    # Existing call sites catch ValueError; the VLAError base must not replace it.
    assert issubclass(ConfigError, ValueError)


# --- redact_secret ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", "ABCD...<32 chars>"),
        ("ab", "ab...<2 chars>"),  # shorter than the 4-char prefix
        ("", "...<0 chars>"),
    ],
)
def test_redact_secret_shows_only_a_prefix_and_the_length(value, expected):
    assert redact_secret(value) == expected


def test_redact_secret_never_contains_the_full_value():
    secret = "hf_AbCdEfGhIjKlMnOpQrStUv123456"
    assert secret not in redact_secret(secret)


# --- as_env_var_name ---


@pytest.mark.parametrize(
    "value",
    [
        "HF_TOKEN",
        "_FOO_BAR",  # leading underscore is a valid identifier
        "A" * 64,  # exactly at the length cap
        # Documents the known gap: a real Hugging Face token is `hf_` plus
        # ~34 alphanumerics -- a perfectly valid env-var name, so this check
        # cannot and does not reject it. Redaction at display time is the
        # actual defense. This row exists so nobody "fixes" the check later
        # believing it should have caught this.
        "hf_AbCdEfGhIjKlMnOpQrStUv123456",
    ],
)
def test_as_env_var_name_accepts(value):
    assert as_env_var_name(value, "field") == value


@pytest.mark.parametrize(
    "value",
    [
        "1FOO",  # leading digit
        "sk-live-1234567890abcdef1234567890abcdef",  # hyphens
        "FOO BAR",
        "FOO.BAR",
        "A" * 65,  # over the length cap
        "TOKÉN",  # non-ASCII: str.isidentifier() alone would accept it
        1.0,
        "",
    ],
)
def test_as_env_var_name_rejects(value):
    with pytest.raises(ConfigError, match="field"):
        as_env_var_name(value, "field")


def test_as_env_var_name_never_echoes_the_full_rejected_value():
    offending = "sk-live-1234567890abcdef1234567890abcdef"
    with pytest.raises(ConfigError) as excinfo:
        as_env_var_name(offending, "field")
    assert offending not in str(excinfo.value)
