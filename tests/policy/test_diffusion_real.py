"""Real DiffusionPolicy through LeRobotBackend.predict_chunk (needs torch, lerobot, diffusers).

`integration`-marked: skipped by `mise run test`. The in-memory test needs no files; the
checkpoint test skips unless DIFFUSION_CHECKPOINT points at a checkpoint dir.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

CHECKPOINT = Path(os.environ["DIFFUSION_CHECKPOINT"]) if "DIFFUSION_CHECKPOINT" in os.environ else None


def _run(backend, n_obs, h, w, d):
    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 255, (h, w, 3), dtype=np.uint8)  # noqa: E731
    vec = lambda: rng.normal(size=d).astype(np.float32)  # noqa: E731
    keys = backend.specs.image_feature_keys
    if n_obs == 1:
        return backend.predict_chunk({k: img() for k in keys}, vec(), "", None)
    return backend.predict_chunk(
        {k: [img() for _ in range(n_obs)] for k in keys},
        [vec() for _ in range(n_obs)],
        "",
        None,
    )


@pytest.mark.skipif(
    CHECKPOINT is None or not CHECKPOINT.exists(), reason="DIFFUSION_CHECKPOINT unset or missing"
)
def test_real_checkpoint_loads_and_predicts():
    from vla.policy.lerobot_backend import LeRobotBackend

    b = LeRobotBackend()
    b.load(str(CHECKPOINT), device="cpu", dtype="auto", rtc=None)
    assert b.specs.n_obs_steps == 1
    c, h, w = b.specs.input_features[b.specs.image_feature_keys[0]]
    processed, raw = _run(b, 1, h, w, b.specs.state_dim)
    assert processed.shape == (20, 10), processed.shape
    assert np.isfinite(processed).all() and np.isfinite(raw).all()


@pytest.mark.parametrize("n_obs", [1, 2])
def test_tiny_in_memory_diffusion(n_obs):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    from vla.policy.lerobot_backend import LeRobotBackend

    cfg = DiffusionConfig(
        n_obs_steps=n_obs,
        horizon=16,
        n_action_steps=8,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
        down_dims=(16, 32),
        n_groups=4,
        spatial_softmax_num_keypoints=8,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        use_group_norm=True,
        num_train_timesteps=10,
        device="cpu",
    )
    def st(*shape):
        return {
            "mean": torch.zeros(*shape), "std": torch.ones(*shape),
            "min": -torch.ones(*shape), "max": torch.ones(*shape),
        }

    stats = {
        "observation.state": st(4),
        "observation.images.top": st(3, 1, 1),
        "action": st(3),
    }
    policy = DiffusionPolicy(cfg).eval()
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)

    b = LeRobotBackend()
    b._policy, b._preprocessor, b._postprocessor = policy, pre, post
    b._time_axis = True
    b._specs = b._build_specs(cfg, policy, False, pre, "cpu", frozenset())
    assert b.specs.n_obs_steps == n_obs
    processed, raw = _run(b, n_obs, 32, 32, 4)
    assert processed.shape == (8, 3) and raw.shape == (8, 3)  # diffusion slices to n_action_steps itself
    assert np.isfinite(processed).all()
    if n_obs == 2:  # single observation is repeated, not an error
        p1, _ = b.predict_chunk({"observation.images.top": np.zeros((32, 32, 3), np.uint8)},
                                np.zeros(4, np.float32), "", None)
        assert p1.shape == (8, 3)
