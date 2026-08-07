import pytest
from vla.policy.config import PolicyConfig, ConfigError


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
