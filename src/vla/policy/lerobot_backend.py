"""LeRobot-backed implementation of PolicyBackend.

`lerobot` is imported lazily inside methods so the module loads, validates
config, and reports status even when torch is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .backend import PolicyBackend, PolicySpecs
from .prefix import normalize_prefix_length

LOGGER = logging.getLogger(__name__)


def _select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LeRobotBackend(PolicyBackend):
    def __init__(self) -> None:
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._specs: PolicySpecs | None = None
        self._device = "cpu"
        self._execution_horizon = 10
        self._rtc_enabled = False

    def load(self, checkpoint_dir: str, *, device: str, dtype: str, rtc: Any | None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(checkpoint_dir, config=cfg)

        resolved_device = _select_device(device)
        policy.to(resolved_device)
        policy.eval()

        if dtype != "auto":
            # Deliberately not applied. Casting weights with `policy.to(dtype=...)`
            # leaves the preprocessor emitting float32 (its serialized
            # DeviceProcessorStep.float_dtype is None), producing
            # "expected scalar type BFloat16 but found Float". Upstream never
            # casts weights; it uses torch.autocast gated on config.use_amp.
            # Wiring that up is deferred - see "Deferred to follow-up plans".
            LOGGER.warning("dtype=%s is not yet applied; running in the checkpoint's dtype", dtype)

        supports_rtc = bool(policy.supports_rtc())
        if rtc is not None and rtc.enabled:
            if not supports_rtc:
                LOGGER.warning("rtc configured but %s does not support it; ignoring", cfg.type)
            else:
                self._configure_rtc(policy, rtc)
                self._rtc_enabled = True
                self._execution_horizon = rtc.execution_horizon

        # The device override is mandatory, not cosmetic. With `pretrained_path`,
        # the processor pipeline is deserialized from the checkpoint including a
        # DeviceProcessorStep carrying the *training* device. Its __post_init__
        # calls get_safe_torch_device, which RAISES rather than falling back
        # (device_utils.py:48) - so a CUDA-trained checkpoint fails outright on
        # the Apple Silicon dev machine. Even when it does not raise, the
        # preprocessor would move tensors to a different device than the model.
        device_override = {"device": resolved_device}
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=checkpoint_dir,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = resolved_device
        self._specs = self._build_specs(cfg, policy, supports_rtc, preprocessor, resolved_device)

    @staticmethod
    def _configure_rtc(policy, rtc) -> None:
        """Inject an RTCConfig and rewire the processor onto the loaded model.

        A downloaded checkpoint almost always has `rtc_config=None`, and
        `_rtc_enabled()` reads `config.rtc_config.enabled`. `init_rtc_processor`
        exists precisely to rewire an already-constructed model.
        """
        from lerobot.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig

        policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=rtc.execution_horizon,
            max_guidance_weight=rtc.max_guidance_weight,
            # RTCAttentionSchedule members are UPPERCASE ("LINEAR", "EXP",
            # "ONES", "ZEROS"); config accepts the lowercase spelling.
            prefix_attention_schedule=RTCAttentionSchedule(rtc.prefix_attention_schedule.upper()),
        )
        policy.init_rtc_processor()

    @staticmethod
    def _detect_relative_actions(preprocessor) -> bool:
        try:
            from lerobot.processor import RelativeActionsProcessorStep
        except ImportError:  # pragma: no cover - older lerobot
            return False
        return any(
            isinstance(s, RelativeActionsProcessorStep) and getattr(s, "enabled", False)
            for s in preprocessor.steps
        )

    def _build_specs(self, cfg, policy, supports_rtc, preprocessor, device) -> PolicySpecs:
        input_features = {k: list(v.shape) for k, v in cfg.input_features.items()}
        output_features = {k: list(v.shape) for k, v in cfg.output_features.items()}
        action_dim = int(output_features["action"][0])
        state_shape = input_features.get("observation.state")
        state_dim = int(state_shape[0]) if state_shape else None
        # Classify on FeatureType.VISUAL rather than a key-prefix guess: the
        # naming is a checkpoint's choice (smolvla_base uses
        # observation.images.camera1/2/3), the feature type is not.
        image_keys = sorted(cfg.image_features.keys())
        # dtype is deliberately read off a live parameter rather than echoing
        # the requested string: load() never casts weights (see the warning
        # above), so the requested dtype and the checkpoint's actual dtype
        # can diverge and a caller inspecting specs needs the truth.
        dtype = str(next(policy.parameters()).dtype).removeprefix("torch.")
        return PolicySpecs(
            policy_type=cfg.type,
            action_dim=action_dim,
            state_dim=state_dim,
            n_action_steps=int(cfg.n_action_steps),
            input_features=input_features,
            output_features=output_features,
            image_feature_keys=image_keys,
            supports_rtc=supports_rtc,
            rtc_enabled=self._rtc_enabled,
            relative_actions=self._detect_relative_actions(preprocessor),
            device=device,
            dtype=dtype,
        )

    @property
    def specs(self) -> PolicySpecs | None:
        return self._specs

    def predict_chunk(self, images, state, task, rtc_kwargs):
        import torch

        if self._policy is None:
            raise RuntimeError("backend not loaded")

        batch: dict[str, Any] = {"task": task}
        for key, img in images.items():
            # HWC uint8 -> BCHW float in [0, 1]; the preprocessor handles the
            # rest. `np.asarray` + `.copy()`-free `from_numpy` would alias the
            # caller's buffer, but `.float()` always allocates a new tensor,
            # so the caller's array is never mutated by this call.
            t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            batch[key] = t.unsqueeze(0).to(self._device)
        batch["observation.state"] = (
            torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self._device)
        )

        kwargs = {}
        if rtc_kwargs:
            prefix = rtc_kwargs.get("prev_chunk_left_over")
            if prefix is not None and prefix.size:
                # Must match execution_horizon, or denoise_step silently shrinks
                # the horizon to the prefix length and changes the guidance weights.
                prefix = normalize_prefix_length(prefix, self._execution_horizon)
                kwargs["prev_chunk_left_over"] = torch.from_numpy(prefix).to(self._device)
            kwargs["inference_delay"] = int(rtc_kwargs.get("inference_delay", 0))

        with torch.no_grad():
            preprocessed = self._preprocessor(batch)
            actions = self._policy.predict_action_chunk(preprocessed, **kwargs)
            raw = actions.squeeze(0).clone()
            processed = self._postprocessor(actions).squeeze(0)

        return (
            processed.float().cpu().numpy().astype(np.float32),
            raw.float().cpu().numpy().astype(np.float32),
        )

    def reset(self) -> None:
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
