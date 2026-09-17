"""Backend abstraction separating the resource layer from any inference runtime."""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config_util import ConfigError


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
    # Everything the checkpoint itself declares as an image feature, before
    # `unused_image_features` subtracts the ones it never actually consumes
    # (see PolicyConfig.unused_image_features). `image_feature_keys` above
    # is always a subset of this. Kept distinct rather than reconstructed
    # by the caller so a checkpoint's full declared shape is always visible
    # on the wire, even once a feature has been dropped from the reduced set.
    declared_image_feature_keys: list[str]
    # (height, width) the policy's OWN preprocessing resizes every frame to
    # before the vision encoder sees it, or None when it does no such resize.
    # Distinct from `input_features`, which is only what the checkpoint
    # *declares*: a fine-tune inherits its base model's declared shape
    # verbatim (smolvla_base says 256x256 whatever the fine-tuning dataset
    # actually held), while training fed the dataset's native frames straight
    # into this resize. Feeding the declared shape therefore resamples twice
    # and, from a 720p/1080p recording, throws away three quarters of the
    # pixels the weights were fitted on -- so this is the size a caller
    # should put on the wire.
    preprocess_image_size: list[int] | None
    supports_rtc: bool
    rtc_enabled: bool
    relative_actions: bool
    device: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        # dataclasses.asdict rather than hand-enumerating fields: a field
        # added to the dataclass without a matching line here would
        # otherwise silently never reach the wire response or the
        # controller, with no test catching the omission.
        return dataclasses.asdict(self)


def resolve_image_feature_keys(
    declared: list[str], unused_image_features: frozenset[str]
) -> list[str]:
    """Subtract `unused_image_features` from a checkpoint's declared keys.

    Shared by every `PolicyBackend` implementation so `LeRobotBackend` and
    `FakePolicyBackend` can never silently drift apart on what counts as
    "not declared" or "would leave zero cameras" -- both backends call this
    same function rather than each re-deriving the rule.
    """
    declared_set = set(declared)
    unknown = sorted(k for k in unused_image_features if k not in declared_set)
    if unknown:
        raise ConfigError(
            f"unused_image_features names key(s) {unknown} that this checkpoint does not "
            f"declare (declared image features: {declared})"
        )

    image_keys = [k for k in declared if k not in unused_image_features]
    # Gated on a non-empty `unused_image_features`, not on an empty result.
    # A checkpoint that declares no image features at all (a state-only
    # policy -- this module is generic over PreTrainedConfig, not
    # smolvla-specific) legitimately reduces to zero keys, and loaded fine
    # before this field existed. Raising there would fail such a checkpoint
    # with a message blaming a field the operator never set.
    if unused_image_features and not image_keys:
        raise ConfigError(
            "unused_image_features lists every image feature this checkpoint declares "
            f"({declared}); at least one camera key must remain -- lerobot's own "
            "prepare_images has no path for zero image inputs"
        )
    return image_keys


class PolicyBackend(abc.ABC):
    """Loads a checkpoint and turns observations into action chunks.

    Implementations must be safe to call concurrently from multiple requests:
    `predict_chunk` carries no state between calls.
    """

    @abc.abstractmethod
    def load(
        self,
        checkpoint_dir: str,
        *,
        device: str,
        dtype: str,
        rtc: Any | None,
        unused_image_features: frozenset[str] = frozenset(),
    ) -> None:
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
