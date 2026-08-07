"""Configuration parsing and validation for viam-labs:vla:policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vla.config_util import ConfigError, as_bool, as_choice, as_env_var_name, as_float, as_int, as_str, redact_secret

__all__ = ["ConfigError", "RTCSettings", "PolicyConfig", "DEVICES", "DTYPES", "SCHEDULES"]

DEVICES = ("auto", "cuda", "mps", "cpu")
DTYPES = ("auto", "float32", "bfloat16", "float16")
SCHEDULES = ("linear", "exp", "ones", "zeros")

# warmup_inferences runs synchronously before the resource starts serving.
# 100 forward passes is already far more than any real warmup needs; the
# bound exists to turn a typo like 1e30 into a config-time error instead of
# a startup that never completes.
_MAX_WARMUP_INFERENCES = 100

# execution_horizon is the number of actions RTC re-plans per inference
# call. lerobot's default is 10; 1000 is a full two orders of magnitude of
# headroom while still catching an obviously wrong value.
_MAX_EXECUTION_HORIZON = 1000

# max_guidance_weight scales the RTC blending term. lerobot's default is
# 10.0; 1000 is a generous ceiling (100x default) that still catches a
# runaway typo. The lower bound is not 0 but a small positive epsilon:
# the field must be strictly positive (0 disables blending in a way that's
# meaningfully different from "small guidance"), and as_float's `minimum`
# is inclusive, so a tiny-but-positive floor is used to express ">0".
_MIN_GUIDANCE_WEIGHT = 1e-6
_MAX_GUIDANCE_WEIGHT = 1000.0

# load_timeout_s bounds the whole checkpoint-resolve + backend.load +
# warmup sequence so a hung download or deserialize eventually transitions
# the resource to "failed" with an actionable message instead of sitting on
# "loading" forever with no way for an operator to tell "downloading 40 GB"
# from "wedged". 1800s (30 min) is generous for even a large hub download on
# a slow link; 86400s (24h) is a ceiling that still catches a wild typo. The
# floor is not 0 -- a load always takes some nonzero time -- but is set low
# enough that tests can exercise the timeout path in well under a second.
_MIN_LOAD_TIMEOUT_S = 0.001
_MAX_LOAD_TIMEOUT_S = 86400.0
_DEFAULT_LOAD_TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class RTCSettings:
    """Mirrors lerobot RTCConfig field-for-field."""

    enabled: bool = False
    execution_horizon: int = 10
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0

    @staticmethod
    def parse(raw: dict[str, Any]) -> "RTCSettings":
        if not isinstance(raw, dict):
            raise ConfigError(f"rtc must be an object, got {raw!r}")

        schedule = as_choice(
            raw.get("prefix_attention_schedule", "linear"), "rtc.prefix_attention_schedule", SCHEDULES
        )
        horizon = as_int(
            raw.get("execution_horizon", 10),
            "rtc.execution_horizon",
            minimum=1,
            maximum=_MAX_EXECUTION_HORIZON,
        )
        weight = as_float(
            raw.get("max_guidance_weight", 10.0),
            "rtc.max_guidance_weight",
            minimum=_MIN_GUIDANCE_WEIGHT,
            maximum=_MAX_GUIDANCE_WEIGHT,
        )
        enabled = as_bool(raw.get("enabled", False), "rtc.enabled")
        return RTCSettings(
            enabled=enabled,
            execution_horizon=horizon,
            prefix_attention_schedule=schedule,
            max_guidance_weight=weight,
        )


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str | None = None
    model_hub_id: str | None = None
    model_revision: str = "main"
    # Names an env var, not a secret -- but an operator pasting the actual
    # token into this field by mistake is the single most likely failure
    # mode here (see as_env_var_name), and a realistic token is itself a
    # valid env-var name that passes that check cleanly. repr=False keeps
    # the dataclass-generated repr from ever rendering it in full; the
    # explicit __repr__ below still shows a redacted placeholder so the
    # field's presence stays visible for debugging.
    hf_token_env: str | None = field(default=None, repr=False)
    device: str = "auto"
    dtype: str = "auto"
    warmup_inferences: int = 2
    load_timeout_s: float = _DEFAULT_LOAD_TIMEOUT_S
    rtc: RTCSettings = field(default_factory=RTCSettings)

    def __repr__(self) -> str:
        token_repr = "None" if self.hf_token_env is None else redact_secret(self.hf_token_env)
        return (
            "PolicyConfig("
            f"model_path={self.model_path!r}, "
            f"model_hub_id={self.model_hub_id!r}, "
            f"model_revision={self.model_revision!r}, "
            f"hf_token_env={token_repr}, "
            f"device={self.device!r}, "
            f"dtype={self.dtype!r}, "
            f"warmup_inferences={self.warmup_inferences!r}, "
            f"load_timeout_s={self.load_timeout_s!r}, "
            f"rtc={self.rtc!r})"
        )

    @staticmethod
    def parse(raw: dict[str, Any]) -> "PolicyConfig":
        path = raw.get("model_path") or None
        hub = raw.get("model_hub_id") or None
        if bool(path) == bool(hub):
            raise ConfigError(
                "exactly one of model_path or model_hub_id is required "
                f"(got model_path={path!r}, model_hub_id={hub!r})"
            )

        device = as_choice(raw.get("device", "auto"), "device", DEVICES)
        dtype = as_choice(raw.get("dtype", "auto"), "dtype", DTYPES)

        warmup = as_int(
            raw.get("warmup_inferences", 2),
            "warmup_inferences",
            minimum=0,
            maximum=_MAX_WARMUP_INFERENCES,
        )

        load_timeout_s = as_float(
            raw.get("load_timeout_s", _DEFAULT_LOAD_TIMEOUT_S),
            "load_timeout_s",
            minimum=_MIN_LOAD_TIMEOUT_S,
            maximum=_MAX_LOAD_TIMEOUT_S,
        )

        model_revision = as_str(raw.get("model_revision", "main"), "model_revision")

        hf_token_env = raw.get("hf_token_env") or None
        if hf_token_env is not None:
            hf_token_env = as_env_var_name(hf_token_env, "hf_token_env")

        # rtc: null (key present, value None) means "use defaults" — the
        # correct JSON reading of an explicit null. Any other non-dict
        # (rtc: false, rtc: "yes", rtc: 0) is a malformed config and must
        # raise loudly rather than silently falling back to defaults.
        rtc_raw = raw.get("rtc")

        return PolicyConfig(
            model_path=path,
            model_hub_id=hub,
            model_revision=model_revision,
            hf_token_env=hf_token_env,
            device=device,
            dtype=dtype,
            warmup_inferences=warmup,
            load_timeout_s=load_timeout_s,
            rtc=RTCSettings.parse({} if rtc_raw is None else rtc_raw),
        )
