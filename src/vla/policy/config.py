"""Configuration parsing and validation for viam-labs:vla:policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vla.config_util import ConfigError, as_bool, as_choice, as_env_var_name, as_float, as_int, as_str

__all__ = ["ConfigError", "RTCSettings", "PolicyConfig", "DEVICES", "DTYPES", "SCHEDULES"]

DEVICES = ("auto", "cuda", "mps", "cpu")
DTYPES = ("auto", "float32", "bfloat16", "float16")
SCHEDULES = ("linear", "exp", "ones", "zeros")

# unused_image_features exists because of a checkpoint-training quirk, not a
# runtime one: lerobot/configs/train.py:285 refuses a `rename_map` unless a
# pretrained checkpoint is given, so fine-tuning lerobot/smolvla_base (which
# declares observation.images.camera1/2/3) inherits all three image features
# verbatim into the new checkpoint's config.json even when the training
# dataset only recorded one or two cameras. There is no harmless value to
# feed the unused slot -- anything (black, gray, a duplicate) measurably
# shifts the predicted chunk, while omitting the key reproduces upstream
# lerobot's own behavior (modeling_smolvla.py:340-346 drops missing image
# keys and only raises if none are present). This field names the declared
# image features to drop before validating/serving `image_feature_keys`.


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

        return RTCSettings(
            enabled=as_bool(raw.get("enabled", False), "rtc.enabled"),
            # Ceilings are sanity bounds on lerobot's defaults (10 and 10.0),
            # generous enough to never bind in practice but tight enough to
            # turn a runaway typo into a config-time error. The weight's floor
            # expresses ">0": as_float's `minimum` is inclusive, and 0 disables
            # blending in a way that differs meaningfully from small guidance.
            execution_horizon=as_int(
                raw.get("execution_horizon", 10), "rtc.execution_horizon", minimum=1, maximum=1000
            ),
            prefix_attention_schedule=as_choice(
                raw.get("prefix_attention_schedule", "linear"),
                "rtc.prefix_attention_schedule",
                SCHEDULES,
            ),
            max_guidance_weight=as_float(
                raw.get("max_guidance_weight", 10.0),
                "rtc.max_guidance_weight",
                minimum=1e-6,
                maximum=1000.0,
            ),
        )


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str | None = None
    model_hub_id: str | None = None
    model_revision: str = "main"
    # Names an env var, not a secret -- but an operator pasting the actual
    # token here is the most likely mistake (see as_env_var_name), and a
    # realistic token is itself a valid env-var name that passes that check.
    # repr=False keeps it out of the generated repr entirely, so an
    # incidental LOGGER.debug("config: %s", cfg) can never leak it.
    hf_token_env: str | None = field(default=None, repr=False)
    device: str = "auto"
    dtype: str = "auto"
    warmup_inferences: int = 2
    # Bounds the whole resolve + load + warmup sequence so a hung download
    # transitions to "failed" with an actionable message instead of sitting
    # on "loading" forever. 30 min is generous for a large hub download on a
    # slow link; the floor is low enough for tests to exercise the timeout.
    load_timeout_s: float = 1800.0
    rtc: RTCSettings = field(default_factory=RTCSettings)
    # Tuple, not list -- this dataclass is frozen, and a mutable default
    # would be shared across every instance that doesn't override it.
    unused_image_features: tuple[str, ...] = ()

    @staticmethod
    def parse(raw: dict[str, Any]) -> "PolicyConfig":
        path = raw.get("model_path") or None
        hub = raw.get("model_hub_id") or None
        if bool(path) == bool(hub):
            raise ConfigError(
                "exactly one of model_path or model_hub_id is required "
                f"(got model_path={path!r}, model_hub_id={hub!r})"
            )

        hf_token_env = raw.get("hf_token_env") or None
        if hf_token_env is not None:
            hf_token_env = as_env_var_name(hf_token_env, "hf_token_env")

        # rtc: null (key present, value None) means "use defaults" -- the
        # correct JSON reading of an explicit null. Any other non-dict
        # (rtc: false, rtc: "yes", rtc: 0) is malformed and must raise
        # rather than silently falling back to defaults.
        rtc_raw = raw.get("rtc")

        return PolicyConfig(
            model_path=path,
            model_hub_id=hub,
            model_revision=as_str(raw.get("model_revision", "main"), "model_revision"),
            hf_token_env=hf_token_env,
            device=as_choice(raw.get("device", "auto"), "device", DEVICES),
            dtype=as_choice(raw.get("dtype", "auto"), "dtype", DTYPES),
            # 100 forward passes is already far more than any warmup needs;
            # the bound turns a typo like 1e30 into a config-time error
            # instead of a startup that never completes.
            warmup_inferences=as_int(
                raw.get("warmup_inferences", 2), "warmup_inferences", minimum=0, maximum=100
            ),
            load_timeout_s=as_float(
                raw.get("load_timeout_s", 1800.0), "load_timeout_s", minimum=0.001, maximum=86400.0
            ),
            rtc=RTCSettings.parse({} if rtc_raw is None else rtc_raw),
            unused_image_features=_parse_unused_image_features(raw.get("unused_image_features")),
        )


def _parse_unused_image_features(raw: Any) -> tuple[str, ...]:
    # Absent key -> (): most checkpoints declare exactly the cameras they
    # consume and never need this field at all.
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(f"unused_image_features must be a list, got {raw!r}")

    parsed: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        value = as_str(entry, f"unused_image_features[{i}]")
        if not value.strip():
            raise ConfigError(
                f"unused_image_features[{i}] must not be empty or whitespace-only, got {value!r}"
            )
        # Strip, having just rejected whitespace-only: caring about stray
        # whitespace enough to reject an all-blank entry but not enough to
        # trim a padded one would send `" observation.images.camera3 "`
        # through parsing only to fail later at load with "checkpoint does
        # not declare", naming a key that looks correct in the error.
        value = value.strip()
        if value in seen:
            raise ConfigError(f"unused_image_features contains a duplicate entry: {value!r}")
        seen.add(value)
        parsed.append(value)
    return tuple(parsed)
