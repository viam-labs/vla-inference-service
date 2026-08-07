"""Backend abstraction separating the resource layer from any inference runtime."""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicySpecs:
    """Everything a caller needs to know about a loaded policy."""

    policy_type: str
    action_dim: int
    state_dim: int
    n_action_steps: int
    input_features: dict[str, list[int]]
    output_features: dict[str, list[int]]
    image_feature_keys: list[str]
    supports_rtc: bool
    rtc_enabled: bool
    relative_actions: bool
    device: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        # dataclasses.asdict rather than hand-enumerating fields: a field
        # added to the dataclass without a matching line here would
        # otherwise silently never reach Task 7's wire response or Task
        # 17's controller, with no test catching the omission.
        return dataclasses.asdict(self)


class PolicyBackend(abc.ABC):
    """Loads a checkpoint and turns observations into action chunks.

    Implementations must be safe to call concurrently from multiple requests:
    `predict_chunk` carries no state between calls.
    """

    @abc.abstractmethod
    def load(self, checkpoint_dir: str, *, device: str, dtype: str, rtc: Any | None) -> None:
        """Load weights and processors. Blocking; run off the event loop."""

    @property
    @abc.abstractmethod
    def specs(self) -> PolicySpecs | None:
        """Loaded policy specs, or None before load completes."""

    @abc.abstractmethod
    def predict_chunk(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task: str,
        rtc_kwargs: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(processed_actions, raw_actions)`, both `[n_action_steps, action_dim]`.

        `processed_actions` are postprocessed and ready for the robot.
        `raw_actions` are in the policy's own action space and are what an RTC
        caller must feed back as `prev_chunk_left_over`. They are deliberately
        distinct return values because confusing them is the likeliest RTC bug.
        """

    def reset(self) -> None:
        """Clear any cached state. Default: no-op."""
