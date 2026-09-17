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


def _specs(cfg):
    return LeRobotBackend()._build_specs(
        cfg, _FakePolicy(), True, _fake_preprocessor(), "cpu", frozenset()
    )


def test_build_specs_reports_the_size_the_policy_actually_resizes_to():
    cfg = _fake_cfg(image_keys=["observation.images.top"])
    cfg.resize_imgs_with_padding = [512, 512]
    specs = _specs(cfg)
    # The declared shape stays 224 -- these answer different questions, and
    # collapsing them would lose the checkpoint's own claim.
    assert specs.preprocess_image_size == [512, 512]
    assert specs.input_features["observation.images.top"] == [3, 224, 224]


def test_preprocess_image_size_is_height_width_not_the_configs_width_height():
    # lerobot calls resize_with_pad(img, value[1], value[0]) into a
    # (img, height, width) signature, so the config pair is (width, height).
    # Passing it through verbatim transposes every non-square checkpoint --
    # invisible on the 512x512 one in reach today, which is exactly why this
    # is pinned with a non-square value.
    cfg = _fake_cfg(image_keys=["observation.images.top"])
    cfg.resize_imgs_with_padding = [640, 480]
    assert _specs(cfg).preprocess_image_size == [480, 640]


def test_preprocess_image_size_is_none_for_a_policy_that_does_no_resize():
    # No attribute at all: this module is generic over any LeRobot-registered
    # policy, and the caller must fall back to the declared shape rather than
    # get a fabricated one.
    cfg = _fake_cfg(image_keys=["observation.images.top"])
    assert not hasattr(cfg, "resize_imgs_with_padding")
    assert _specs(cfg).preprocess_image_size is None


def test_preprocess_image_size_is_none_for_a_malformed_or_empty_value():
    for bad in (None, [], [512], [512, 512, 512], [0, 512]):
        cfg = _fake_cfg(image_keys=["observation.images.top"])
        cfg.resize_imgs_with_padding = bad
        assert _specs(cfg).preprocess_image_size is None, bad


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


class _FakeTorch:
    """Enough torch surface for `_select_device`, with a controllable device.

    `is_available()` reporting True while a kernel launch raises is not a
    contrived combination -- it is exactly a Jetson Orin (sm_87) running a
    generic aarch64 wheel that carries no kernels for its compute capability.
    """

    class _Backends:
        class _MPS:
            def __init__(self, available):
                self._available = available

            def is_available(self):
                return self._available

        def __init__(self, mps_available):
            self.mps = self._MPS(mps_available)

    def __init__(self, *, cuda_available, cuda_works, mps_available=False, mps_works=True):
        self._works = {"cuda": cuda_works, "mps": mps_works}
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = self._Backends(mps_available)
        self.launched = []

    def ones(self, shape, device):
        self.launched.append(device)
        if not self._works[device]:
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return SimpleNamespace(cpu=lambda: None)

    def mm(self, a, b):
        return a


def _select(monkeypatch, fake):
    import sys

    monkeypatch.setitem(sys.modules, "torch", fake)
    from vla.policy.lerobot_backend import _select_device

    return _select_device


def test_auto_falls_back_to_cpu_when_cuda_cannot_run_a_kernel(monkeypatch):
    fake = _FakeTorch(cuda_available=True, cuda_works=False)
    assert _select(monkeypatch, fake)("auto") == "cpu"
    assert fake.launched == ["cuda"], "must actually attempt a kernel, not just ask is_available()"


def test_auto_uses_cuda_when_a_kernel_actually_runs(monkeypatch):
    fake = _FakeTorch(cuda_available=True, cuda_works=True)
    assert _select(monkeypatch, fake)("auto") == "cuda"


def test_auto_falls_through_to_mps_when_cuda_is_unusable(monkeypatch):
    fake = _FakeTorch(cuda_available=True, cuda_works=False, mps_available=True, mps_works=True)
    assert _select(monkeypatch, fake)("auto") == "mps"
    assert fake.launched == ["cuda", "mps"]


def test_auto_reaches_cpu_when_neither_accelerator_works(monkeypatch):
    fake = _FakeTorch(cuda_available=True, cuda_works=False, mps_available=True, mps_works=False)
    assert _select(monkeypatch, fake)("auto") == "cpu"


def test_an_explicit_device_is_never_probed_or_downgraded(monkeypatch):
    # An operator naming a device wants it to fail loudly, not be silently
    # swapped for something that happens to work.
    fake = _FakeTorch(cuda_available=True, cuda_works=False)
    assert _select(monkeypatch, fake)("cuda") == "cuda"
    assert fake.launched == []
