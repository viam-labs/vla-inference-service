"""Unit tests for LeRobotBackend logic that does not require torch/lerobot.

`LeRobotBackend.load()` imports torch and lerobot lazily inside itself (see
this module's own docstring), but `_build_specs` and the helpers it calls do
not -- `_detect_relative_actions` already guards its `lerobot.processor`
import with `except ImportError`, and `_warn_about_likely_unused_features`
does the same. That means the `unused_image_features` validation this file
exercises -- the same validation `FakePolicyBackend` mirrors via
`resolve_image_feature_keys` -- can be driven directly with plain fakes,
with no `@pytest.mark.integration` and no checkpoint download.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vla.config_util import ConfigError
from vla.policy.lerobot_backend import LeRobotBackend


class _FakeFeature:
    def __init__(self, shape):
        self.shape = shape


class _FakeParam:
    # str(dtype).removeprefix("torch.") is all _build_specs does with this;
    # a plain string stands in fine for a real torch.dtype here.
    dtype = "torch.float32"


class _FakePolicy:
    def parameters(self):
        yield _FakeParam()


def _fake_preprocessor(steps=()):
    return SimpleNamespace(steps=list(steps))


def _fake_cfg(*, image_keys, state_dim=6, action_dim=4, n_action_steps=5):
    image_shape = (3, 224, 224)
    input_features = {k: _FakeFeature(image_shape) for k in image_keys}
    input_features["observation.state"] = _FakeFeature((state_dim,))
    output_features = {"action": _FakeFeature((action_dim,))}
    return SimpleNamespace(
        type="fake_lerobot_policy",
        input_features=input_features,
        output_features=output_features,
        image_features={k: _FakeFeature(image_shape) for k in image_keys},
        n_action_steps=n_action_steps,
    )


def test_build_specs_reduces_image_feature_keys_but_keeps_full_declared_set():
    backend = LeRobotBackend()
    cfg = _fake_cfg(
        image_keys=[
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ]
    )
    specs = backend._build_specs(
        cfg,
        _FakePolicy(),
        True,
        _fake_preprocessor(),
        "cpu",
        frozenset({"observation.images.camera3"}),
    )
    assert specs.image_feature_keys == [
        "observation.images.camera1",
        "observation.images.camera2",
    ]
    assert specs.declared_image_feature_keys == [
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ]


def test_build_specs_with_no_unused_image_features_keeps_everything():
    backend = LeRobotBackend()
    cfg = _fake_cfg(image_keys=["observation.images.top"])
    specs = backend._build_specs(
        cfg, _FakePolicy(), True, _fake_preprocessor(), "cpu", frozenset()
    )
    assert specs.image_feature_keys == ["observation.images.top"]
    assert specs.declared_image_feature_keys == ["observation.images.top"]


def test_build_specs_rejects_a_key_the_checkpoint_never_declared():
    backend = LeRobotBackend()
    cfg = _fake_cfg(image_keys=["observation.images.top"])
    with pytest.raises(ConfigError, match="observation.images.nonexistent"):
        backend._build_specs(
            cfg,
            _FakePolicy(),
            True,
            _fake_preprocessor(),
            "cpu",
            frozenset({"observation.images.nonexistent"}),
        )


def test_build_specs_rejects_listing_every_declared_image_feature():
    backend = LeRobotBackend()
    cfg = _fake_cfg(
        image_keys=["observation.images.camera1", "observation.images.camera2"]
    )
    with pytest.raises(ConfigError, match="every image feature"):
        backend._build_specs(
            cfg,
            _FakePolicy(),
            True,
            _fake_preprocessor(),
            "cpu",
            frozenset({"observation.images.camera1", "observation.images.camera2"}),
        )


def test_build_specs_tolerates_a_preprocessor_with_unrecognized_step_shapes():
    # The advisory unused_image_features check must never fail the load
    # over a processor step it doesn't recognize -- it just skips the hint.
    backend = LeRobotBackend()
    cfg = _fake_cfg(image_keys=["observation.images.top"])
    specs = backend._build_specs(
        cfg, _FakePolicy(), True, _fake_preprocessor([object()]), "cpu", frozenset()
    )
    assert specs.image_feature_keys == ["observation.images.top"]


def test_build_specs_allows_a_checkpoint_that_declares_no_image_features():
    """A state-only checkpoint must still load.

    This module is generic over `PreTrainedConfig` (`policy_type` is read
    from the checkpoint, never configured), so a policy with no VISUAL
    features at all is a legitimate input, and it loaded fine before
    `unused_image_features` existed. The zero-remaining guard is gated on a
    non-empty `unused_image_features` precisely so this case does not fail
    with a message blaming a field the operator never set.
    """
    backend = LeRobotBackend()
    cfg = _fake_cfg(image_keys=[])
    specs = backend._build_specs(
        cfg, _FakePolicy(), True, _fake_preprocessor(), "cpu", frozenset()
    )
    assert specs.image_feature_keys == []
    assert specs.declared_image_feature_keys == []


# ---------------------------------------------------------------------------
# The advisory heuristic's true-positive path. Needs real lerobot processor
# step instances (the check is `isinstance`-based, so SimpleNamespace fakes
# can never reach it) but no checkpoint download -- the steps are built
# directly. Marked `differential` for the same reason every other
# lerobot-requiring test here is: pyproject defines that marker as
# "requires lerobot installed", and the fast suite deselects it.
# ---------------------------------------------------------------------------


@pytest.mark.differential
def test_advisory_warning_names_only_the_camera_with_no_stats_and_no_rename_target(caplog):
    """The shape of a real fine-tune: two cameras renamed onto the base
    model's canonical slots, a third slot inherited from the base and never
    fed. Only that third slot may be named.
    """
    import logging

    normalize_mod = pytest.importorskip("lerobot.processor.normalize_processor")
    rename_mod = pytest.importorskip("lerobot.processor.rename_processor")

    fed = ["observation.images.camera1", "observation.images.camera2"]
    inherited = "observation.images.camera3"

    rename = rename_mod.RenameObservationsProcessorStep(
        rename_map={
            "observation.images.camera_transform": fed[0],
            "observation.images.realsense_cam": fed[1],
        }
    )
    # object.__new__ so this test does not have to track NormalizerProcessorStep's
    # constructor signature across lerobot releases; `stats` is the only
    # attribute the advisory check reads off it.
    normalizer = object.__new__(normalize_mod.NormalizerProcessorStep)
    normalizer.stats = {key: {} for key in fed}

    backend = LeRobotBackend()
    cfg = _fake_cfg(image_keys=[*fed, inherited])
    with caplog.at_level(logging.WARNING, logger="vla.policy.lerobot_backend"):
        specs = backend._build_specs(
            cfg, _FakePolicy(), True, _fake_preprocessor([rename, normalizer]), "cpu", frozenset()
        )

    assert specs.image_feature_keys == [*fed, inherited]
    warnings = [r.message for r in caplog.records if "unused_image_features" in r.message]
    assert len(warnings) == 1
    assert inherited in warnings[0]
    for key in fed:
        assert key not in warnings[0], "a camera with normalizer stats must not be flagged"


@pytest.mark.differential
def test_advisory_warning_is_silent_once_the_camera_is_listed_as_unused(caplog):
    import logging

    normalize_mod = pytest.importorskip("lerobot.processor.normalize_processor")
    rename_mod = pytest.importorskip("lerobot.processor.rename_processor")

    fed = ["observation.images.camera1", "observation.images.camera2"]
    inherited = "observation.images.camera3"
    rename = rename_mod.RenameObservationsProcessorStep(
        rename_map={"observation.images.realsense_cam": fed[1]}
    )
    normalizer = object.__new__(normalize_mod.NormalizerProcessorStep)
    normalizer.stats = {key: {} for key in fed}

    backend = LeRobotBackend()
    cfg = _fake_cfg(image_keys=[*fed, inherited])
    with caplog.at_level(logging.WARNING, logger="vla.policy.lerobot_backend"):
        specs = backend._build_specs(
            cfg,
            _FakePolicy(),
            True,
            _fake_preprocessor([rename, normalizer]),
            "cpu",
            frozenset({inherited}),
        )

    assert specs.image_feature_keys == fed
    assert [r.message for r in caplog.records if "unused_image_features" in r.message] == []
