import math

import pytest
from vla.policy.config import PolicyConfig, ConfigError

# Hardcoded, not imported from vla.policy.config: parametrizing off the
# module's own DEVICES/DTYPES/SCHEDULES tuples would mean a mutant that
# shrinks one of those tuples shrinks the test's parametrization along with
# it, so the mutant becomes invisible to the very test meant to catch it.
EXPECTED_DEVICES = ("auto", "cuda", "mps", "cpu")
EXPECTED_DTYPES = ("auto", "float32", "bfloat16", "float16")
EXPECTED_SCHEDULES = ("linear", "exp", "ones", "zeros")


def test_local_path_config():
    cfg = PolicyConfig.parse({"model_path": "/models/smolvla"})
    assert cfg.model_path == "/models/smolvla"
    assert cfg.model_hub_id is None
    assert cfg.device == "auto"
    assert cfg.dtype == "auto"
    assert cfg.warmup_inferences == 2


def test_hub_config():
    cfg = PolicyConfig.parse({"model_hub_id": "lerobot/smolvla_base", "model_revision": "v1"})
    assert cfg.model_hub_id == "lerobot/smolvla_base"
    assert cfg.model_revision == "v1"


def test_requires_exactly_one_source_none_given():
    with pytest.raises(ConfigError, match="exactly one"):
        PolicyConfig.parse({})


def test_requires_exactly_one_source_both_given():
    with pytest.raises(ConfigError, match="exactly one"):
        PolicyConfig.parse({"model_path": "/m", "model_hub_id": "a/b"})


def test_rejects_unknown_device():
    with pytest.raises(ConfigError, match="device"):
        PolicyConfig.parse({"model_path": "/m", "device": "tpu"})


def test_rejects_unknown_dtype():
    with pytest.raises(ConfigError, match="dtype"):
        PolicyConfig.parse({"model_path": "/m", "dtype": "int4"})


def test_rtc_defaults_match_lerobot():
    cfg = PolicyConfig.parse({"model_path": "/m", "rtc": {"enabled": True}})
    assert cfg.rtc.enabled is True
    assert cfg.rtc.execution_horizon == 10
    assert cfg.rtc.max_guidance_weight == 10.0
    assert cfg.rtc.prefix_attention_schedule == "linear"


def test_rtc_disabled_by_default():
    cfg = PolicyConfig.parse({"model_path": "/m"})
    assert cfg.rtc.enabled is False


def test_rejects_nonpositive_guidance_weight():
    with pytest.raises(ConfigError, match="max_guidance_weight"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"max_guidance_weight": 0}})


def test_rejects_negative_warmup():
    with pytest.raises(ConfigError, match="warmup_inferences"):
        PolicyConfig.parse({"model_path": "/m", "warmup_inferences": -1})


# --- protobuf Struct number-handling edge cases ---
# struct_to_dict hands parse() doubles for every number. Production sends
# 2.0 for an int field; a hand-written test dict sends 2. Both must work,
# but a typo'd 2.5 must not silently become 2.


def test_accepts_integral_float_warmup_inferences():
    cfg = PolicyConfig.parse({"model_path": "/m", "warmup_inferences": 2.0})
    assert cfg.warmup_inferences == 2


def test_accepts_integral_float_execution_horizon():
    cfg = PolicyConfig.parse({"model_path": "/m", "rtc": {"execution_horizon": 10.0}})
    assert cfg.rtc.execution_horizon == 10


def test_rejects_fractional_warmup_inferences():
    with pytest.raises(ConfigError, match="warmup_inferences"):
        PolicyConfig.parse({"model_path": "/m", "warmup_inferences": 2.5})


def test_rejects_fractional_execution_horizon():
    with pytest.raises(ConfigError, match="execution_horizon"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"execution_horizon": 10.5}})


def test_rejects_non_numeric_warmup_inferences():
    with pytest.raises(ConfigError, match="warmup_inferences"):
        PolicyConfig.parse({"model_path": "/m", "warmup_inferences": "two"})


def test_rejects_non_numeric_execution_horizon():
    with pytest.raises(ConfigError, match="execution_horizon"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"execution_horizon": "ten"}})


def test_rejects_non_numeric_max_guidance_weight():
    with pytest.raises(ConfigError, match="max_guidance_weight"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"max_guidance_weight": "loud"}})


def test_rejects_non_dict_rtc():
    with pytest.raises(ConfigError, match="rtc"):
        PolicyConfig.parse({"model_path": "/m", "rtc": "yes"})


def test_rejects_boolean_warmup_inferences():
    with pytest.raises(ConfigError, match="warmup_inferences"):
        PolicyConfig.parse({"model_path": "/m", "warmup_inferences": True})


def test_rejects_boolean_execution_horizon():
    with pytest.raises(ConfigError, match="execution_horizon"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"execution_horizon": True}})


# --- coverage gap: "invalid rejected" was tested, "valid accepted" was not ---
# A one-character typo that shrinks an enum tuple (e.g. DTYPES to ("auto",))
# rejects every legitimate value while every existing test stays green,
# because none of them assert that a real value survives parsing.


@pytest.mark.parametrize("device", EXPECTED_DEVICES)
def test_accepts_every_valid_device(device):
    cfg = PolicyConfig.parse({"model_path": "/m", "device": device})
    assert cfg.device == device


@pytest.mark.parametrize("dtype", EXPECTED_DTYPES)
def test_accepts_every_valid_dtype(dtype):
    cfg = PolicyConfig.parse({"model_path": "/m", "dtype": dtype})
    assert cfg.dtype == dtype


@pytest.mark.parametrize("schedule", EXPECTED_SCHEDULES)
def test_accepts_every_valid_schedule(schedule):
    cfg = PolicyConfig.parse({"model_path": "/m", "rtc": {"prefix_attention_schedule": schedule}})
    assert cfg.rtc.prefix_attention_schedule == schedule


def test_rejects_unknown_schedule():
    with pytest.raises(ConfigError, match="prefix_attention_schedule"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"prefix_attention_schedule": "cosine"}})


def test_rejects_zero_execution_horizon():
    with pytest.raises(ConfigError, match="execution_horizon"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"execution_horizon": 0}})


def test_rejects_negative_execution_horizon():
    with pytest.raises(ConfigError, match="execution_horizon"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"execution_horizon": -1}})


def test_default_model_revision_is_main():
    cfg = PolicyConfig.parse({"model_path": "/m"})
    assert cfg.model_revision == "main"


def test_hf_token_env_passthrough():
    cfg = PolicyConfig.parse({"model_path": "/m", "hf_token_env": "HF_TOKEN"})
    assert cfg.hf_token_env == "HF_TOKEN"


def test_rejects_boolean_max_guidance_weight():
    with pytest.raises(ConfigError, match="max_guidance_weight"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"max_guidance_weight": True}})


def test_rejects_nan_max_guidance_weight():
    with pytest.raises(ConfigError, match="max_guidance_weight"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"max_guidance_weight": math.nan}})


# --- item 3: `enabled` must reject truthy-but-not-bool values ---
# bool("false") is True in Python, so a naive cast would silently turn RTC
# *on* for a very plausible hand-edited config value.


def test_rejects_string_enabled():
    with pytest.raises(ConfigError, match="enabled"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"enabled": "false"}})


# --- item 4: `rtc: null` means defaults; any other non-dict must be loud ---


def test_rtc_null_uses_defaults():
    cfg = PolicyConfig.parse({"model_path": "/m", "rtc": None})
    assert cfg.rtc.enabled is False
    assert cfg.rtc.execution_horizon == 10


def test_rejects_falsy_non_dict_rtc():
    with pytest.raises(ConfigError, match="rtc"):
        PolicyConfig.parse({"model_path": "/m", "rtc": False})


# --- as_str type-checking now applied to string fields ---


def test_rejects_non_string_model_revision():
    with pytest.raises(ConfigError, match="model_revision"):
        PolicyConfig.parse({"model_path": "/m", "model_revision": 1.0})


def test_rejects_non_string_hf_token_env():
    with pytest.raises(ConfigError, match="hf_token_env"):
        PolicyConfig.parse({"model_path": "/m", "hf_token_env": 1.0})


# --- hf_token_env must look like an env var name, not a pasted secret ---
# The field exists specifically so the token value never appears in config;
# an operator confusing "name of the env var" with "the token itself" is
# the single most likely mistake, so the accepted shape is checked strictly.


@pytest.mark.parametrize("name", ["HF_TOKEN", "_TOKEN", "MY_TOKEN_2", "a"])
def test_accepts_every_valid_hf_token_env_name(name):
    cfg = PolicyConfig.parse({"model_path": "/m", "hf_token_env": name})
    assert cfg.hf_token_env == name


def test_rejects_hf_token_env_shaped_like_a_pasted_token():
    with pytest.raises(ConfigError, match="hf_token_env"):
        PolicyConfig.parse(
            {
                "model_path": "/m",
                "hf_token_env": "sk-live-1234567890abcdef1234567890abcdef",
            }
        )


def test_rejects_hf_token_env_starting_with_digit():
    with pytest.raises(ConfigError, match="hf_token_env"):
        PolicyConfig.parse({"model_path": "/m", "hf_token_env": "1TOKEN"})


def test_rejects_hf_token_env_over_64_chars():
    with pytest.raises(ConfigError, match="hf_token_env"):
        PolicyConfig.parse({"model_path": "/m", "hf_token_env": "A" * 65})


def test_config_error_never_echoes_full_invalid_hf_token_env_value():
    offending = "sk-live-1234567890abcdef1234567890abcdef"
    with pytest.raises(ConfigError) as excinfo:
        PolicyConfig.parse({"model_path": "/m", "hf_token_env": offending})
    assert offending not in str(excinfo.value)


# --- maximum= bounds prevent pathological config values ---


def test_rejects_excessive_warmup_inferences():
    with pytest.raises(ConfigError, match="warmup_inferences"):
        PolicyConfig.parse({"model_path": "/m", "warmup_inferences": 1e30})


def test_rejects_excessive_execution_horizon():
    with pytest.raises(ConfigError, match="execution_horizon"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"execution_horizon": 1e30}})


def test_rejects_excessive_max_guidance_weight():
    with pytest.raises(ConfigError, match="max_guidance_weight"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"max_guidance_weight": 1e30}})
