import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from viam.utils import dict_to_struct, struct_to_dict

from vla.config_util import ConfigError
from vla.policy.backend import PolicyBackend, PolicySpecs
from vla.policy.fake_backend import FakePolicyBackend


def _obs():
    return {"observation.images.top": np.zeros((224, 224, 3), dtype=np.uint8)}


# ---------------------------------------------------------------------------
# Baseline behavior
# ---------------------------------------------------------------------------


def test_specs_before_load_are_none():
    assert FakePolicyBackend().specs is None


def test_load_then_specs():
    b = FakePolicyBackend(action_dim=6, n_action_steps=50)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    specs = b.specs
    assert isinstance(specs, PolicySpecs)
    assert specs.action_dim == 6
    assert specs.n_action_steps == 50
    assert specs.supports_rtc is True
    assert specs.relative_actions is False


def test_predict_chunk_shape():
    b = FakePolicyBackend(action_dim=6, n_action_steps=50)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    actions, raw = b.predict_chunk(_obs(), np.zeros(6, np.float32), "do the thing", None)
    assert actions.shape == (50, 6)
    assert raw.shape == (50, 6)


def test_predict_chunk_is_deterministic():
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    a1, _ = b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    a2, _ = b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    np.testing.assert_array_equal(a1, a2)


def test_raw_and_processed_differ_so_confusion_is_detectable():
    # The two arrays must not be interchangeable, or a controller bug that
    # swaps them would pass every test.
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    actions, raw = b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    assert not np.array_equal(actions, raw)


def test_records_rtc_kwargs_for_assertions():
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    prefix = np.ones((3, 4), dtype=np.float32)
    b.predict_chunk(_obs(), np.zeros(4, np.float32), "t",
                    {"inference_delay": 2, "prev_chunk_left_over": prefix})
    assert b.last_rtc["inference_delay"] == 2
    np.testing.assert_array_equal(b.last_rtc["prev_chunk_left_over"], prefix)


# ---------------------------------------------------------------------------
# predict_chunk before load() must raise, like the real LeRobotBackend
# (review item 1). Task 7 has two ordering hazards -- warmup racing ahead of
# load, and a superseded reconfigure swapping self._backend mid-flight --
# that a fake which silently "succeeds" unloaded would let pass in tests
# and fail only in production.
# ---------------------------------------------------------------------------


def test_predict_chunk_before_load_raises():
    b = FakePolicyBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        b.predict_chunk(_obs(), np.zeros(6, np.float32), "t", None)


# ---------------------------------------------------------------------------
# Constructor defaults must be asserted (standing requirement 2). Mutation-
# verified: action_dim 6->7, n_action_steps 50->51, or image_size
# (224,224)->(256,192) left every prior test green.
# ---------------------------------------------------------------------------


def test_default_constructor_dimensions_are_asserted():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    specs = b.specs
    assert specs.action_dim == 6
    assert specs.n_action_steps == 50
    assert specs.state_dim == 6
    assert specs.input_features["observation.images.top"] == [3, 224, 224]
    assert specs.image_feature_keys == ["observation.images.top"]
    actions, raw = b.predict_chunk(_obs(), np.zeros(6, np.float32), "t", None)
    assert actions.shape == (50, 6)
    assert raw.shape == (50, 6)


# ---------------------------------------------------------------------------
# state_dim must not be conflated with action_dim (review item 2). Task 7's
# _warmup_once comment ("State dim, not action dim: they coincide on
# smolvla_base but need not") exists precisely to guard against this bug --
# a fake that always sets them equal makes that bug undetectable.
# ---------------------------------------------------------------------------


def test_state_dim_defaults_to_action_dim_when_unset():
    b = FakePolicyBackend(action_dim=6)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.state_dim == 6
    assert b.specs.input_features["observation.state"] == [6]


def test_state_dim_distinct_from_action_dim_is_honored():
    # action_dim and state_dim deliberately differ here -- a fake (or a
    # controller) that conflates them would be caught by this test but not
    # by one where the two happen to coincide.
    b = FakePolicyBackend(action_dim=6, state_dim=14)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.action_dim == 6
    assert b.specs.state_dim == 14
    assert b.specs.input_features["observation.state"] == [14]
    assert b.specs.output_features == {"action": [6]}


# ---------------------------------------------------------------------------
# Multi-camera support (review item 3). The real lerobot/smolvla_base
# checkpoint has three cameras (observation.images.camera1/2/3); a fake
# that only ever declares one camera lets Task 7's per-camera decode loop
# and Task 17's image-size mapping go completely unexercised for the
# multi-camera case.
# ---------------------------------------------------------------------------


def test_single_default_camera_key():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.image_feature_keys == ["observation.images.top"]
    assert set(b.specs.input_features) == {"observation.images.top", "observation.state"}


def test_multi_camera_keys_all_appear_in_input_features_and_image_feature_keys():
    keys = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    b = FakePolicyBackend(camera_keys=keys, image_size=(180, 320))
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.image_feature_keys == list(keys)
    for key in keys:
        assert b.specs.input_features[key] == [3, 180, 320]
    # observation.state must still be present alongside the cameras.
    assert "observation.state" in b.specs.input_features
    assert set(b.specs.input_features) == set(keys) | {"observation.state"}


# ---------------------------------------------------------------------------
# Thread safety of the bookkeeping (review item 4). PolicyBackend's
# docstring promises "predict_chunk carries no state between calls", but
# the pre-hardening implementation did read-modify-write on call_count and
# last-writer-wins on last_rtc with no lock, while Task 7 dispatches
# concurrent requests via asyncio.to_thread.
# ---------------------------------------------------------------------------


def test_call_count_is_exact_under_concurrent_calls():
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)

    n_threads = 32
    calls_per_thread = 200

    def hammer():
        for _ in range(calls_per_thread):
            b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(hammer) for _ in range(n_threads)]
        for f in futures:
            f.result()

    assert b.call_count == n_threads * calls_per_thread


def test_last_rtc_stores_a_copy_not_the_callers_dict_reference():
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    caller_dict = {"inference_delay": 1}
    b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", caller_dict)
    caller_dict["inference_delay"] = 999  # mutate after the call returns
    assert b.last_rtc["inference_delay"] == 1


# ---------------------------------------------------------------------------
# PolicySpecs.to_dict must not silently drop fields (item 1)
# ---------------------------------------------------------------------------


def test_to_dict_covers_every_dataclass_field():
    b = FakePolicyBackend(action_dim=3, n_action_steps=7)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    specs = b.specs
    assert set(specs.to_dict()) == {f.name for f in dataclasses.fields(PolicySpecs)}


def test_to_dict_values_match_specs_fields():
    b = FakePolicyBackend(action_dim=3, n_action_steps=7, image_size=(180, 320))
    rtc = SimpleNamespace(enabled=True)
    b.load("/fake", device="mps", dtype="bfloat16", rtc=rtc)
    specs = b.specs
    d = specs.to_dict()
    for f in dataclasses.fields(PolicySpecs):
        assert d[f.name] == getattr(specs, f.name)


# ---------------------------------------------------------------------------
# to_dict must survive a protobuf Struct round trip (item 2) -- this is how
# the specs cross DoCommand in Task 7.
# ---------------------------------------------------------------------------


def test_to_dict_round_trips_through_struct():
    b = FakePolicyBackend(action_dim=5, n_action_steps=12, image_size=(180, 320))
    rtc = SimpleNamespace(enabled=True)
    b.load("/fake", device="cuda", dtype="float16", rtc=rtc)
    original = b.specs.to_dict()

    struct = dict_to_struct(original)
    round_tripped = struct_to_dict(struct)

    # Struct stores every number as a double -- ints come back as floats,
    # not their original int type. Assert that explicitly so it's documented
    # rather than a surprise for Task 17, which has to int() them back.
    assert round_tripped["action_dim"] == 5.0
    assert isinstance(round_tripped["action_dim"], float)
    assert round_tripped["n_action_steps"] == 12.0
    assert isinstance(round_tripped["n_action_steps"], float)

    # Everything else survives with equal (if not identical-typed) values.
    assert round_tripped["policy_type"] == original["policy_type"]
    assert round_tripped["device"] == original["device"]
    assert round_tripped["supports_rtc"] == original["supports_rtc"]
    assert round_tripped["rtc_enabled"] == original["rtc_enabled"]
    assert round_tripped["relative_actions"] == original["relative_actions"]
    assert round_tripped["image_feature_keys"] == original["image_feature_keys"]
    # Nested list-of-int values also come back as floats.
    assert round_tripped["input_features"]["observation.state"] == [5.0]
    assert round_tripped["output_features"]["action"] == [5.0]


# ---------------------------------------------------------------------------
# FakePolicyBackend must be a faithful stand-in for Task 7 (item 3)
# ---------------------------------------------------------------------------


def test_predict_chunk_shape_tracks_nondefault_constructor_args():
    # The baseline tests above only ever exercise the two default-adjacent
    # combinations (6/50 and 4/10). Use values that collide with neither
    # default nor any other hardcoded shape in this file.
    b = FakePolicyBackend(action_dim=9, n_action_steps=17)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    actions, raw = b.predict_chunk(_obs(), np.zeros(9, np.float32), "t", None)
    assert actions.shape == (17, 9)
    assert raw.shape == (17, 9)


def test_specs_image_feature_shape_is_height_width_not_swapped():
    # A non-square image size catches a height/width transposition that a
    # square fixture cannot: if load() ever swapped h and w, this would
    # still look "shaped correctly" with a square image.
    b = FakePolicyBackend(image_size=(180, 320))
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.input_features["observation.images.top"] == [3, 180, 320]


def test_specs_supports_rtc_false_is_honored():
    b = FakePolicyBackend(supports_rtc=False)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.supports_rtc is False


def test_specs_relative_actions_true_is_honored():
    b = FakePolicyBackend(relative_actions=True)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.relative_actions is True


def test_specs_rtc_enabled_true_when_rtc_object_enabled():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=SimpleNamespace(enabled=True))
    assert b.specs.rtc_enabled is True


def test_specs_rtc_enabled_false_when_rtc_object_disabled():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=SimpleNamespace(enabled=False))
    assert b.specs.rtc_enabled is False


def test_specs_rtc_enabled_false_when_rtc_is_none():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.rtc_enabled is False


def test_specs_device_is_passed_through():
    b = FakePolicyBackend()
    b.load("/fake", device="mps", dtype="float32", rtc=None)
    assert b.specs.device == "mps"


def test_specs_dtype_is_passed_through():
    # load() takes a dtype and (per Task 8) deliberately doesn't apply it to
    # the model; a specs/status reader has no other way to learn what
    # precision is actually running unless load() records what it was told.
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="bfloat16", rtc=None)
    assert b.specs.dtype == "bfloat16"


def test_specs_policy_type_is_fake():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.policy_type == "fake"


def test_specs_output_features_and_image_feature_keys():
    b = FakePolicyBackend(action_dim=8)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.output_features == {"action": [8]}
    assert b.specs.image_feature_keys == ["observation.images.top"]


def test_reset_exists_and_is_callable_and_is_noop_by_default():
    b = FakePolicyBackend()
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    b.reset()  # must not raise
    assert b.specs is not None  # reset() does not clear loaded specs


def test_call_count_increments_per_predict_chunk_call():
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.call_count == 0
    b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    assert b.call_count == 1
    b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    assert b.call_count == 2


# ---------------------------------------------------------------------------
# The PolicyBackend contract is the only thing enforcing the seam (item 4)
# ---------------------------------------------------------------------------


def test_backend_missing_load_cannot_be_instantiated():
    class MissingLoad(PolicyBackend):
        @property
        def specs(self):
            return None

        def predict_chunk(self, images, state, task, rtc_kwargs):
            raise NotImplementedError

    with pytest.raises(TypeError):
        MissingLoad()


def test_backend_missing_specs_cannot_be_instantiated():
    class MissingSpecs(PolicyBackend):
        def load(self, checkpoint_dir, *, device, dtype, rtc):
            pass

        def predict_chunk(self, images, state, task, rtc_kwargs):
            raise NotImplementedError

    with pytest.raises(TypeError):
        MissingSpecs()


def test_backend_missing_predict_chunk_cannot_be_instantiated():
    class MissingPredict(PolicyBackend):
        def load(self, checkpoint_dir, *, device, dtype, rtc):
            pass

        @property
        def specs(self):
            return None

    with pytest.raises(TypeError):
        MissingPredict()


def test_backend_implementing_all_abstract_methods_can_be_instantiated():
    class Complete(PolicyBackend):
        def load(self, checkpoint_dir, *, device, dtype, rtc):
            pass

        @property
        def specs(self):
            return None

        def predict_chunk(self, images, state, task, rtc_kwargs):
            raise NotImplementedError

    # Must not raise -- proves the failures above are due to missing
    # abstract methods specifically, not some unrelated constructor issue.
    Complete()


# ---------------------------------------------------------------------------
# unused_image_features: exercises the checkpoint-declares-more-than-it-
# consumes case (the lerobot/smolvla_base inheritance in
# PolicyConfig.unused_image_features) with no torch/lerobot installed.
# ---------------------------------------------------------------------------


def test_declared_image_feature_keys_defaults_to_full_camera_set():
    keys = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    b = FakePolicyBackend(camera_keys=keys)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    assert b.specs.declared_image_feature_keys == list(keys)
    assert b.specs.image_feature_keys == list(keys)


def test_unused_image_features_reduces_image_feature_keys_but_not_declared():
    keys = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    b = FakePolicyBackend(camera_keys=keys)
    b.load(
        "/fake",
        device="cpu",
        dtype="float32",
        rtc=None,
        unused_image_features=frozenset({"observation.images.camera3"}),
    )
    specs = b.specs
    assert specs.image_feature_keys == [
        "observation.images.camera1",
        "observation.images.camera2",
    ]
    assert specs.declared_image_feature_keys == list(keys)


def test_unused_image_features_rejects_a_key_the_checkpoint_never_declared():
    b = FakePolicyBackend(camera_keys=("observation.images.top",))
    with pytest.raises(ConfigError, match="observation.images.nonexistent"):
        b.load(
            "/fake",
            device="cpu",
            dtype="float32",
            rtc=None,
            unused_image_features=frozenset({"observation.images.nonexistent"}),
        )


def test_unused_image_features_rejects_listing_every_declared_feature():
    keys = ("observation.images.camera1", "observation.images.camera2")
    b = FakePolicyBackend(camera_keys=keys)
    with pytest.raises(ConfigError, match="every image feature"):
        b.load(
            "/fake",
            device="cpu",
            dtype="float32",
            rtc=None,
            unused_image_features=frozenset(keys),
        )
