"""n_obs_steps: request parsing (service), stacking and repeat (LeRobotBackend), specs."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from tests.policy.test_service import _ready_service
from vla.policy.lerobot_backend import LeRobotBackend
from vla.wire import WireError, encode_image, encode_vector

KEY = "observation.images.top"


def _img(v):
    return encode_image(np.full((224, 224, 3), v, dtype=np.uint8), encoding="png")


def _vec(v):
    return encode_vector(np.full(6, v, dtype=np.float32))


async def test_specs_report_n_obs_steps(tmp_path):
    assert (await (await _ready_service(tmp_path)).do_command({"command": "specs"}))[
        "n_obs_steps"
    ] == 1


async def test_infer_history_lists_reach_backend_oldest_first(tmp_path):
    svc = await _ready_service(tmp_path, n_obs_steps=2)
    await svc.do_command({
        "command": "infer",
        "images": {KEY: [_img(10), _img(20)]},
        "state": [_vec(1), _vec(2)],
    })
    images, state = svc._backend.last_obs
    assert [int(f[0, 0, 0]) for f in images[KEY]] == [10, 20]
    assert [float(v[0]) for v in state] == [1.0, 2.0]


async def test_infer_single_observation_still_accepted_for_history_policy(tmp_path):
    svc = await _ready_service(tmp_path, n_obs_steps=2)
    await svc.do_command({"command": "infer", "images": {KEY: _img(1)}, "state": _vec(1)})
    assert not isinstance(svc._backend.last_obs[1], list)


@pytest.mark.parametrize("bad", ["images", "state"])
async def test_infer_wrong_list_length_errors(tmp_path, bad):
    svc = await _ready_service(tmp_path, n_obs_steps=2)
    images = [_img(1), _img(2)] if bad != "images" else [_img(1)] * 3
    state = [_vec(1), _vec(2)] if bad != "state" else [_vec(1)]
    with pytest.raises(WireError, match="expected 2"):
        await svc.do_command({"command": "infer", "images": {KEY: images}, "state": state})


async def test_infer_list_for_single_obs_policy_must_have_length_one(tmp_path):
    svc = await _ready_service(tmp_path)
    with pytest.raises(WireError, match="expected 1"):
        await svc.do_command(
            {"command": "infer", "images": {KEY: _img(1)}, "state": [_vec(1), _vec(2)]}
        )


# --- backend stacking, with a stub policy/preprocessor (no torch model needed) ---


def _backend(n_obs, time_axis):
    torch = pytest.importorskip("torch")
    b = LeRobotBackend()
    seen = {}

    class Policy:
        def predict_action_chunk(self, batch, **kw):
            seen.update(batch)
            return torch.zeros(1, 4, 2)

    b._policy = Policy()
    b._preprocessor = lambda batch: batch
    b._postprocessor = lambda a: a
    b._specs = SimpleNamespace(n_obs_steps=n_obs, n_action_steps=4)
    b._time_axis = time_axis
    return b, seen


def _frame(v):
    return np.full((8, 8, 3), v, dtype=np.uint8)


def _state(v):
    return np.full(5, v, dtype=np.float32)


def test_n1_time_axis_policy_gets_b1d_state():
    b, seen = _backend(1, True)
    b.predict_chunk({KEY: _frame(255)}, _state(3), "", None)
    assert seen["observation.state"].shape == (1, 1, 5)
    assert seen[KEY].shape == (1, 1, 3, 8, 8)


def test_n1_other_policy_unchanged():
    b, seen = _backend(1, False)
    b.predict_chunk({KEY: _frame(255)}, _state(3), "", None)
    assert seen["observation.state"].shape == (1, 5)
    assert seen[KEY].shape == (1, 3, 8, 8)


def test_n2_stacks_oldest_first():
    b, seen = _backend(2, True)
    b.predict_chunk({KEY: [_frame(0), _frame(255)]}, [_state(1), _state(2)], "", None)
    assert seen["observation.state"].shape == (1, 2, 5)
    assert seen["observation.state"][0, :, 0].tolist() == [1.0, 2.0]
    assert seen[KEY].shape == (1, 2, 3, 8, 8)
    assert seen[KEY][0, :, 0, 0, 0].tolist() == [0.0, 1.0]


def test_n2_single_observation_repeats_and_warns_once(caplog):
    b, seen = _backend(2, True)
    with caplog.at_level(logging.WARNING):
        b.predict_chunk({KEY: _frame(255)}, _state(3), "", None)
        b.predict_chunk({KEY: _frame(255)}, _state(3), "", None)
    assert seen["observation.state"].shape == (1, 2, 5)
    assert seen["observation.state"][0, 0].tolist() == seen["observation.state"][0, 1].tolist()
    assert sum("repeating" in r.message for r in caplog.records) == 1


def test_backend_wrong_list_length_errors():
    b, _ = _backend(2, True)
    with pytest.raises(ValueError, match="n_obs_steps=2"):
        b.predict_chunk({KEY: [_frame(0)] * 3}, [_state(1)] * 2, "", None)


def test_unverified_type_with_history_rejected_at_load():
    from vla.config_util import ConfigError

    cfg = SimpleNamespace(type="g05", n_obs_steps=2)
    with pytest.raises(ConfigError, match="g05"):
        LeRobotBackend._check_history_supported(cfg)
    LeRobotBackend._check_history_supported(SimpleNamespace(type="diffusion", n_obs_steps=2))
    LeRobotBackend._check_history_supported(SimpleNamespace(type="g05", n_obs_steps=1))


async def test_infer_history_frames_must_share_shape(tmp_path):
    svc = await _ready_service(tmp_path, n_obs_steps=2)
    small = encode_image(np.zeros((100, 100, 3), dtype=np.uint8), encoding="png")
    with pytest.raises(WireError, match=KEY):
        await svc.do_command(
            {"command": "infer", "images": {KEY: [_img(1), small]}, "state": [_vec(1), _vec(2)]}
        )
