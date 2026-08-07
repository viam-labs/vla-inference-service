"""Deterministic in-memory backend, so everything above it tests without torch."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import PolicyBackend, PolicySpecs


class FakePolicyBackend(PolicyBackend):
    def __init__(
        self,
        *,
        action_dim: int = 6,
        n_action_steps: int = 50,
        supports_rtc: bool = True,
        relative_actions: bool = False,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self._action_dim = action_dim
        self._n_action_steps = n_action_steps
        self._supports_rtc = supports_rtc
        self._relative_actions = relative_actions
        self._image_size = image_size
        self._specs: PolicySpecs | None = None
        self.last_rtc: dict[str, Any] | None = None
        self.call_count = 0

    def load(self, checkpoint_dir: str, *, device: str, dtype: str, rtc: Any | None) -> None:
        h, w = self._image_size
        self._specs = PolicySpecs(
            policy_type="fake",
            action_dim=self._action_dim,
            n_action_steps=self._n_action_steps,
            input_features={"observation.images.top": [3, h, w],
                            "observation.state": [self._action_dim]},
            output_features={"action": [self._action_dim]},
            image_feature_keys=["observation.images.top"],
            supports_rtc=self._supports_rtc,
            rtc_enabled=bool(rtc and getattr(rtc, "enabled", False)),
            relative_actions=self._relative_actions,
            device=device,
        )

    @property
    def specs(self) -> PolicySpecs | None:
        return self._specs

    def predict_chunk(self, images, state, task, rtc_kwargs):
        self.call_count += 1
        self.last_rtc = rtc_kwargs
        steps, dim = self._n_action_steps, self._action_dim
        # A fixed ramp: deterministic, and raw != processed so a controller that
        # swaps the two fails loudly instead of silently passing.
        base = np.arange(steps, dtype=np.float32).reshape(steps, 1)
        raw = np.tile(base, (1, dim)) * 0.01
        processed = raw + 1.0
        return processed, raw
