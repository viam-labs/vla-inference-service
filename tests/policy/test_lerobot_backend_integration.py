"""Integration test: the real LeRobot backend against a real downloaded checkpoint.

Marked `integration` so `mise run test` (`-m "not integration and not differential"`)
skips it by default -- it downloads ~1-2 GB from the Hugging Face hub and requires
torch/lerobot (the `lerobot` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

from vla.policy.backend import PolicySpecs
from vla.policy.fake_backend import FakePolicyBackend

pytestmark = pytest.mark.integration

CHECKPOINT = "lerobot/smolvla_base"
# Deliberately NOT lerobot's own RTCConfig default (10, see
# lerobot/policies/rtc/configuration_rtc.py) -- a backend that forgot to
# propagate the configured execution_horizon and fell back to the library
# default would otherwise pass this suite by coincidence.
EXECUTION_HORIZON = 6


def _images_for(specs: PolicySpecs) -> dict[str, np.ndarray]:
    images = {}
    for key in specs.image_feature_keys:
        c, h, w = specs.input_features[key]
        images[key] = np.zeros((h, w, c), dtype=np.uint8)
    return images


def _state_for(specs: PolicySpecs) -> np.ndarray:
    dim = specs.state_dim if specs.state_dim is not None else specs.action_dim
    return np.zeros(dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures: a checkpoint downloaded once, and two policy loads (plain, RTC).
# Both loads are relatively expensive (network the first time, then a real
# forward pass through torch on every use), so each is module-scoped and
# loaded at most once for the whole file.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint_path():
    from vla.policy.config import PolicyConfig
    from vla.policy.resolver import resolve_checkpoint

    cfg = PolicyConfig.parse({"model_hub_id": CHECKPOINT})
    return resolve_checkpoint(cfg)


@pytest.fixture(scope="module")
def backend(checkpoint_path):
    from vla.policy.lerobot_backend import LeRobotBackend

    b = LeRobotBackend()
    b.load(checkpoint_path, device="auto", dtype="auto", rtc=None)
    return b


@pytest.fixture(scope="module")
def rtc_backend(checkpoint_path):
    from vla.policy.config import RTCSettings
    from vla.policy.lerobot_backend import LeRobotBackend

    b = LeRobotBackend()
    rtc = RTCSettings(
        enabled=True,
        execution_horizon=EXECUTION_HORIZON,
        prefix_attention_schedule="linear",
        max_guidance_weight=10.0,
    )
    b.load(checkpoint_path, device="auto", dtype="auto", rtc=rtc)
    return b


@pytest.fixture
def loaded_fake_backend():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    return b


# ---------------------------------------------------------------------------
# Plan Step 1: specs and a basic finite chunk (plus the required additions:
# state_dim and dtype must be populated for real -- Task 8 review item 1).
# ---------------------------------------------------------------------------


def test_specs_are_populated(backend):
    s = backend.specs
    assert s.policy_type == "smolvla"
    assert s.action_dim > 0
    assert s.n_action_steps == 50
    assert s.supports_rtc is True
    # smolvla_base declares observation.images.camera1/2/3
    assert len(s.image_feature_keys) == 3
    assert all(k in s.input_features for k in s.image_feature_keys)


def test_specs_state_dim_is_populated(backend):
    # Required addition 1: state_dim must come from
    # cfg.input_features["observation.state"], not be left None/omitted.
    s = backend.specs
    assert s.state_dim is not None
    assert s.state_dim == s.input_features["observation.state"][0]
    assert s.state_dim > 0


def test_specs_dtype_reflects_the_actual_checkpoint_not_the_request(backend):
    # Required addition 1: dtype must be the checkpoint's real in-effect
    # dtype (inspected off a live parameter), not merely an echo of the
    # "auto" string that was requested -- load() never casts weights.
    s = backend.specs
    assert s.dtype != "auto"
    assert isinstance(s.dtype, str) and s.dtype
    # Must not carry the "torch." module prefix from str(tensor.dtype).
    assert not s.dtype.startswith("torch.")


def test_specs_every_field_is_populated(backend):
    # Standing requirement 3: every field must be asserted somewhere, so a
    # mutation hardcoding one to None/default is caught.
    s = backend.specs
    assert s.action_dim == 6
    assert s.state_dim == 6
    assert s.n_action_steps == 50
    assert s.output_features == {"action": [6]}
    assert set(s.input_features) == {
        "observation.state",
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    }
    assert s.image_feature_keys == [
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ]
    assert s.supports_rtc is True
    assert s.rtc_enabled is False  # this fixture loaded with rtc=None
    assert s.relative_actions is False
    assert s.device in ("cpu", "cuda", "mps")


def test_predict_chunk_returns_finite_chunk(backend):
    s = backend.specs
    images = _images_for(s)
    state = _state_for(s)

    actions, raw = backend.predict_chunk(images, state, "pick up the red block", None)

    assert actions.shape == (s.n_action_steps, s.action_dim)
    assert raw.shape == (s.n_action_steps, s.action_dim)
    assert np.all(np.isfinite(actions))
    assert np.all(np.isfinite(raw))


# ---------------------------------------------------------------------------
# Required addition 5: predict_chunk must not mutate the caller's image
# buffers -- the controller reuses them across ticks.
# ---------------------------------------------------------------------------


def test_predict_chunk_does_not_mutate_images(backend):
    s = backend.specs
    images = _images_for(s)
    # Non-zero, distinguishable content per camera so any in-place write is
    # detectable regardless of which camera it touches.
    for i, key in enumerate(images):
        images[key][:] = (i + 1) * 10
    snapshots = {k: v.copy() for k, v in images.items()}

    backend.predict_chunk(images, _state_for(s), "pick up the red block", None)

    for key, snapshot in snapshots.items():
        np.testing.assert_array_equal(images[key], snapshot)
        assert images[key].dtype == np.uint8


# ---------------------------------------------------------------------------
# Required addition 2: predict_chunk must raise if the backend was never
# loaded, exactly like FakePolicyBackend (fake_backend.py's own comment:
# "Mirrors LeRobotBackend").
# ---------------------------------------------------------------------------


def test_predict_chunk_before_load_raises():
    from vla.policy.lerobot_backend import LeRobotBackend

    b = LeRobotBackend()
    assert b.specs is None
    with pytest.raises(RuntimeError, match="not loaded"):
        b.predict_chunk({}, np.zeros(1, dtype=np.float32), "t", None)


# ---------------------------------------------------------------------------
# Required addition 3: the fake and the real backend must agree on the
# shared contract. Parametrized over *unloaded* fresh instances of both --
# constructing a LeRobotBackend is cheap (no torch import happens until
# `load()`), so this does not require a second checkpoint load.
# ---------------------------------------------------------------------------


def _fresh_fake():
    return FakePolicyBackend()


def _fresh_real():
    from vla.policy.lerobot_backend import LeRobotBackend

    return LeRobotBackend()


@pytest.mark.parametrize("make_backend", [_fresh_fake, _fresh_real], ids=["fake", "real"])
def test_specs_is_none_before_load(make_backend):
    assert make_backend().specs is None


@pytest.mark.parametrize("make_backend", [_fresh_fake, _fresh_real], ids=["fake", "real"])
def test_predict_chunk_raises_before_load_for_both_backends(make_backend):
    b = make_backend()
    with pytest.raises(RuntimeError, match="not loaded"):
        b.predict_chunk({}, np.zeros(1, dtype=np.float32), "t", None)


@pytest.fixture(params=["fake", "real"])
def loaded_backend(request, loaded_fake_backend, backend):
    return loaded_fake_backend if request.param == "fake" else backend


def test_predict_chunk_contract_shared_by_fake_and_real(loaded_backend):
    s = loaded_backend.specs
    assert s is not None

    images = _images_for(s)
    state = _state_for(s)
    actions, raw = loaded_backend.predict_chunk(images, state, "do the thing", None)

    assert actions.shape == (s.n_action_steps, s.action_dim)
    assert raw.shape == (s.n_action_steps, s.action_dim)
    assert actions.dtype == np.float32
    assert raw.dtype == np.float32
    assert np.all(np.isfinite(actions))
    assert np.all(np.isfinite(raw))
    # The fake backend deliberately returns a different array for each; the
    # real backend must too, or a controller bug that swaps raw/processed
    # would be invisible in both.
    assert raw is not actions


# ---------------------------------------------------------------------------
# Required addition 4: exercise the RTC path for real, including the
# under-verified prefix-normalization branch. `execution_horizon=10` but the
# supplied prefix is 3 rows -- without normalize_prefix_length, lerobot's
# denoise_step silently shrinks the effective execution_horizon to
# `min(execution_horizon, prefix.shape[0])` (see
# lerobot/policies/rtc/modeling_rtc.py's RTCProcessor.denoise_step), which
# changes the guidance weights without raising. Verified directly against
# lerobot 0.6.2 source before writing this test.
# ---------------------------------------------------------------------------


def test_rtc_enabled_flag_is_set_when_configured(rtc_backend):
    s = rtc_backend.specs
    assert s.supports_rtc is True
    assert s.rtc_enabled is True


def test_rtc_predict_chunk_without_prefix_succeeds(rtc_backend):
    s = rtc_backend.specs
    images = _images_for(s)
    state = _state_for(s)

    actions, raw = rtc_backend.predict_chunk(images, state, "pick up the red block", None)

    assert actions.shape == (s.n_action_steps, s.action_dim)
    assert np.all(np.isfinite(actions))
    assert np.all(np.isfinite(raw))


def test_rtc_predict_chunk_normalizes_a_wrong_length_prefix(rtc_backend, monkeypatch):
    import vla.policy.lerobot_backend as lb_module

    s = rtc_backend.specs
    images = _images_for(s)
    state = _state_for(s)

    calls: list[tuple[tuple[int, ...], int]] = []
    original = lb_module.normalize_prefix_length

    def spy(prev_actions, target_steps):
        calls.append((prev_actions.shape, target_steps))
        result = original(prev_actions, target_steps)
        # The whole point of normalization: the output actually fed to the
        # policy must be execution_horizon-long, not the caller's 3 rows.
        assert result.shape == (EXECUTION_HORIZON, s.action_dim)
        return result

    monkeypatch.setattr(lb_module, "normalize_prefix_length", spy)

    # Deliberately wrong length: execution_horizon is 10, this prefix is 3.
    wrong_length_prefix = np.full((3, s.action_dim), 7.0, dtype=np.float32)

    actions, raw = rtc_backend.predict_chunk(
        images,
        state,
        "pick up the red block",
        {"prev_chunk_left_over": wrong_length_prefix, "inference_delay": 2},
    )

    assert actions.shape == (s.n_action_steps, s.action_dim)
    assert np.all(np.isfinite(actions))
    assert np.all(np.isfinite(raw))
    # Proves normalize_prefix_length was actually invoked (not skipped) and
    # invoked with this backend's configured execution_horizon.
    assert calls == [((3, s.action_dim), EXECUTION_HORIZON)]
