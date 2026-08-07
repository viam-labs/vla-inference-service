"""Configuration parsing and validation for viam-labs:vla:policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEVICES = ("auto", "cuda", "mps", "cpu")
DTYPES = ("auto", "float32", "bfloat16", "float16")
SCHEDULES = ("linear", "exp", "ones", "zeros")


class ConfigError(ValueError):
    """Raised for invalid module configuration."""


def _as_int(value: Any, field_name: str) -> int:
    """Coerce a protobuf-Struct-shaped value to int, strictly.

    Struct stores every number as a double, so production sends 2.0 for an
    int field while a hand-written test dict sends 2 — both must resolve to
    the same int. A fractional value like 2.5 is a config typo, not a
    truncation target, so it is rejected rather than silently floored.
    Booleans are technically ints in Python but are never a legitimate value
    here, so they are rejected explicitly.
    """
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ConfigError(f"{field_name} must be an integer, got {value!r}")
        return int(value)
    raise ConfigError(f"{field_name} must be an integer, got {value!r}")


def _as_float(value: Any, field_name: str) -> float:
    """Coerce a protobuf-Struct-shaped value to float, strictly.

    Booleans are technically numbers in Python but are never a legitimate
    value here, so they are rejected explicitly rather than silently
    becoming 0.0/1.0.
    """
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    raise ConfigError(f"{field_name} must be a number, got {value!r}")


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

        schedule = raw.get("prefix_attention_schedule", "linear")
        if schedule not in SCHEDULES:
            raise ConfigError(
                f"rtc.prefix_attention_schedule must be one of {SCHEDULES}, got {schedule!r}"
            )
        horizon = _as_int(raw.get("execution_horizon", 10), "rtc.execution_horizon")
        if horizon <= 0:
            raise ConfigError(f"rtc.execution_horizon must be positive, got {horizon}")
        weight = _as_float(raw.get("max_guidance_weight", 10.0), "rtc.max_guidance_weight")
        if weight <= 0:
            raise ConfigError(f"rtc.max_guidance_weight must be positive, got {weight}")
        return RTCSettings(
            enabled=bool(raw.get("enabled", False)),
            execution_horizon=horizon,
            prefix_attention_schedule=schedule,
            max_guidance_weight=weight,
        )


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str | None = None
    model_hub_id: str | None = None
    model_revision: str = "main"
    hf_token_env: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    warmup_inferences: int = 2
    rtc: RTCSettings = field(default_factory=RTCSettings)

    @staticmethod
    def parse(raw: dict[str, Any]) -> "PolicyConfig":
        path = raw.get("model_path") or None
        hub = raw.get("model_hub_id") or None
        if bool(path) == bool(hub):
            raise ConfigError(
                "exactly one of model_path or model_hub_id is required "
                f"(got model_path={path!r}, model_hub_id={hub!r})"
            )

        device = raw.get("device", "auto")
        if device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}, got {device!r}")

        dtype = raw.get("dtype", "auto")
        if dtype not in DTYPES:
            raise ConfigError(f"dtype must be one of {DTYPES}, got {dtype!r}")

        warmup = _as_int(raw.get("warmup_inferences", 2), "warmup_inferences")
        if warmup < 0:
            raise ConfigError(f"warmup_inferences must be >= 0, got {warmup}")

        return PolicyConfig(
            model_path=path,
            model_hub_id=hub,
            model_revision=raw.get("model_revision", "main"),
            hf_token_env=raw.get("hf_token_env") or None,
            device=device,
            dtype=dtype,
            warmup_inferences=warmup,
            rtc=RTCSettings.parse(raw.get("rtc", {}) or {}),
        )
