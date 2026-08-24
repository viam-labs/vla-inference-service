import asyncio
import logging
import threading
import time

import numpy as np
import pytest
from viam.proto.app.robot import ServiceConfig
from google.protobuf.struct_pb2 import Struct

from vla.policy.config import PolicyConfig
from vla.policy.fake_backend import FakePolicyBackend
from vla.policy.service import VLAPolicy
from vla.wire import encode_image, encode_matrix, encode_vector, decode_matrix


def _config(attrs: dict) -> ServiceConfig:
    s = Struct()
    s.update(attrs)
    return ServiceConfig(name="p", api="rdk:service:generic", model="viam-labs:vla:policy",
                         attributes=s)


def _make_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("{}")
    return tmp_path


async def _ready_service(tmp_path, **backend_kwargs) -> VLAPolicy:
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(**backend_kwargs)
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    await svc.await_ready()
    return svc


# ---------------------------------------------------------------------------
# Tests given in the plan (Task 7, Step 1)
# ---------------------------------------------------------------------------


def test_validate_returns_no_dependencies(tmp_path):
    required, optional = VLAPolicy.validate_config(_config({"model_path": str(tmp_path)}))
    assert required == []
    assert optional == []


def test_validate_rejects_missing_source():
    with pytest.raises(Exception, match="exactly one"):
        VLAPolicy.validate_config(_config({}))


async def test_reconfigure_returns_before_load_completes(tmp_path):
    """reconfigure must not block on a slow load, or viam-server can time out."""
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    release = threading.Event()

    class SlowBackend(FakePolicyBackend):
        def load(self, *a, **k):
            if not release.wait(timeout=5):
                raise AssertionError("load was never released")
            super().load(*a, **k)

    svc._backend_factory = SlowBackend

    started = time.perf_counter()
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, f"reconfigure blocked for {elapsed:.2f}s"
    assert (await svc.do_command({"command": "status"}))["state"] == "loading"

    release.set()
    await svc.await_ready()
    assert (await svc.do_command({"command": "status"}))["state"] == "ready"


async def test_infer_before_ready_errors(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend
    with pytest.raises(Exception, match="not ready"):
        await svc.do_command({"command": "infer", "images": {}, "state": [], "task": "t"})


async def test_status_reports_ready(tmp_path):
    svc = await _ready_service(tmp_path)
    assert (await svc.do_command({"command": "status"}))["state"] == "ready"


async def test_specs_shape(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=6, n_action_steps=50)
    specs = await svc.do_command({"command": "specs"})
    assert specs["action_dim"] == 6
    assert specs["n_action_steps"] == 50
    assert specs["supports_rtc"] is True
    assert specs["relative_actions"] is False


async def test_infer_returns_both_action_arrays(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=6, n_action_steps=50)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    out = await svc.do_command({
        "command": "infer",
        "images": {"observation.images.top": encode_image(img)},
        "state": encode_vector(np.zeros(6, dtype=np.float32)),
        "task": "pick up the block",
    })
    actions = decode_matrix(out["actions"])
    raw = decode_matrix(out["raw_actions"])
    assert actions.shape == (50, 6)
    assert raw.shape == (50, 6)
    assert not np.array_equal(actions, raw)
    assert out["latency_s"] >= 0


async def test_infer_passes_rtc_kwargs_to_backend(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=10)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    prefix = np.ones((3, 4), dtype=np.float32)
    await svc.do_command({
        "command": "infer",
        "images": {"observation.images.top": encode_image(img)},
        "state": encode_vector(np.zeros(4, dtype=np.float32)),
        "task": "t",
        "rtc": {"inference_delay": 2, "prev_chunk_left_over": encode_matrix(prefix)},
    })
    assert svc._backend.last_rtc["inference_delay"] == 2
    np.testing.assert_array_equal(svc._backend.last_rtc["prev_chunk_left_over"], prefix)


async def test_unknown_command_errors(tmp_path):
    svc = await _ready_service(tmp_path)
    with pytest.raises(Exception, match="unknown command"):
        await svc.do_command({"command": "teleport"})


async def test_failed_load_surfaces_in_status(tmp_path):
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend   # reconfigure constructs the backend eagerly
    svc.reconfigure(_config({"model_path": str(tmp_path / "absent")}), {})
    await svc.await_ready(expect_failure=True)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert "does not exist" in status["error"]


# ---------------------------------------------------------------------------
# Additional coverage per "standing test requirements" / Additional required
# work in the task description.
# ---------------------------------------------------------------------------


# --- 1. close() ---


async def test_close_cancels_in_flight_load_without_waiting_for_it(tmp_path):
    """close() must not hang until the underlying (uncancellable) thread finishes."""
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    release = threading.Event()

    class SlowBackend(FakePolicyBackend):
        def load(self, *a, **k):
            if not release.wait(timeout=5):
                raise AssertionError("load was never released")
            super().load(*a, **k)

    svc._backend_factory = SlowBackend
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})

    # Let the background thread actually start and block on release.wait().
    await asyncio.sleep(0.05)

    started = time.perf_counter()
    await asyncio.wait_for(svc.close(), timeout=1.0)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"close() blocked for {elapsed:.2f}s while a load was in flight"
    # close() must await the task's settlement, not just request cancellation
    # and return: a coroutine with no internal await runs to completion
    # without ever giving the event loop a chance to deliver the
    # CancelledError, so this would still be False if close() forgot to
    # await it.
    assert svc._load_task.done()

    release.set()  # let the orphaned thread finish so it doesn't leak into other tests


async def test_close_after_ready_is_safe(tmp_path):
    svc = await _ready_service(tmp_path)
    await svc.close()  # must not raise


async def test_close_twice_is_safe(tmp_path):
    svc = await _ready_service(tmp_path)
    await svc.close()
    await svc.close()  # must not raise


async def test_close_before_any_reconfigure_is_safe():
    svc = VLAPolicy("p")
    await svc.close()  # no load task was ever created


# --- close() must actually release: refuse further service, drop the
# backend, and log loudly if a load was still running. Per the coordinator:
# viam-server's Python SDK reconfigures a resource by removing it and
# constructing a brand-new instance (remove-then-add), so close() sits on
# the critical path of *every* reconfigure, not just final shutdown -- an
# orphaned download thread here can wedge the REPLACEMENT resource on a
# filelock it still holds in the HF cache.


@pytest.mark.parametrize("command", ["status", "specs", "reset", "infer"])
async def test_do_command_refuses_after_close(tmp_path, command):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    await svc.close()
    with pytest.raises(Exception, match="closed"):
        await svc.do_command({
            "command": command,
            "images": {"observation.images.top": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
        })


async def test_do_command_works_normally_before_close(tmp_path):
    """Sanity check for the parametrized refusal test above: closing must be
    what causes the refusal, not some unrelated regression in do_command."""
    svc = await _ready_service(tmp_path)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "ready"


async def test_close_drops_backend_reference(tmp_path):
    """So a loaded torch model isn't kept alive (holding GPU memory) by a
    resource nothing can reach anymore."""
    svc = await _ready_service(tmp_path)
    assert svc._backend is not None
    await svc.close()
    assert svc._backend is None


async def test_close_logs_warning_when_load_still_running(tmp_path, caplog):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    release = threading.Event()

    class SlowBackend(FakePolicyBackend):
        def load(self, *a, **k):
            release.wait(timeout=5)
            super().load(*a, **k)

    svc._backend_factory = SlowBackend
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    await asyncio.sleep(0.05)  # let the load genuinely be in flight

    with caplog.at_level(logging.WARNING, logger="vla.policy.service"):
        await svc.close()

    release.set()  # let the orphaned thread finish so it doesn't leak

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("still be running" in m or "in progress" in m for m in warnings), warnings


async def test_close_does_not_warn_when_load_already_settled(tmp_path, caplog):
    svc = await _ready_service(tmp_path)
    with caplog.at_level(logging.WARNING, logger="vla.policy.service"):
        await svc.close()
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


# --- 2. generation counter ---


async def test_stale_load_generation_does_not_overwrite_newer_state(tmp_path):
    """Directly exercises the generation guard inside `_load`.

    Simulates a stale load (an old generation number) finishing its work
    after a newer reconfigure has already moved `self._generation` forward.
    The stale load must leave `_state`/`_error` untouched.
    """
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend
    svc._cfg = PolicyConfig.parse({"model_path": str(tmp_path)})
    stale_backend = FakePolicyBackend(action_dim=9, n_action_steps=3)
    svc._backend = stale_backend
    svc._generation = 5
    # Deliberately a value the stale load's own (successful) run would NOT
    # produce on its own -- if the generation guard were a no-op, `_load`
    # would happily overwrite this to "ready"/None, so this baseline is
    # chosen specifically to make that overwrite observable.
    svc._state = "failed"
    svc._error = "some earlier, unrelated failure"

    await svc._load(3, stale_backend)  # stale generation: 3 != current 5

    assert svc._state == "failed"
    assert svc._error == "some earlier, unrelated failure"


async def test_reconfigure_increments_generation_each_call(tmp_path):
    """The increment itself, isolated from cancellation.

    Cancellation alone can mask a broken (e.g. never-incrementing) counter in
    an end-to-end race test, because the cancelled task never reaches the
    comparison at all. This asserts the counter's own arithmetic directly.
    """
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend
    before = svc._generation

    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    assert svc._generation == before + 1

    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    assert svc._generation == before + 2

    await svc.await_ready()


async def test_reconfigure_while_loading_discards_stale_load_result(tmp_path):
    """End-to-end: a slow first load must not win over a faster second reconfigure."""
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    release = threading.Event()
    started = threading.Event()

    class SlowBackend(FakePolicyBackend):
        def load(self, *a, **k):
            started.set()
            if not release.wait(timeout=5):
                raise AssertionError("load was never released")
            super().load(*a, **k)

    svc._backend_factory = SlowBackend
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})

    # Wait until the first load's background thread is actually blocked.
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("first load never started")

    stale_backend = svc._backend

    svc._backend_factory = lambda: FakePolicyBackend(action_dim=9, n_action_steps=3)
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    await svc.await_ready()

    specs = await svc.do_command({"command": "specs"})
    assert specs["action_dim"] == 9
    assert svc._backend is not stale_backend

    release.set()
    await asyncio.sleep(0.1)  # give the stale background thread a chance to finish

    status = await svc.do_command({"command": "status"})
    assert status["state"] == "ready"
    specs = await svc.do_command({"command": "specs"})
    assert specs["action_dim"] == 9
    assert svc._backend is not stale_backend


# --- 3. broad Exception path ---


async def test_load_raising_runtime_error_surfaces_as_internal_error(tmp_path):
    _make_checkpoint(tmp_path)

    class BoomBackend(FakePolicyBackend):
        def load(self, *a, **k):
            raise RuntimeError("boom")

    svc = VLAPolicy("p")
    svc._backend_factory = BoomBackend
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    await svc.await_ready(expect_failure=True)

    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert "internal error" in status["error"]
    assert "boom" in status["error"]


# --- 4. inference_delay coercion via as_int ---


async def test_infer_rtc_inference_delay_accepts_int_and_whole_float(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=10)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for delay in (2, 2.0):
        out = await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(img)},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
            "rtc": {"inference_delay": delay},
        })
        assert out is not None
        assert svc._backend.last_rtc["inference_delay"] == 2


async def test_infer_rtc_inference_delay_rejects_fractional_value(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=10)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(Exception, match="inference_delay"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(img)},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
            "rtc": {"inference_delay": 2.5},
        })


# --- 5. concurrent infer calls ---


async def test_concurrent_infer_calls_are_all_well_formed(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=10)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Default warmup_inferences (2) already ran during load and bumped
    # call_count -- measure the delta caused by this test's own calls, not
    # the absolute count.
    baseline = svc._backend.call_count

    async def one():
        return await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(img)},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
        })

    n = 8
    results = await asyncio.gather(*(one() for _ in range(n)))
    for out in results:
        actions = decode_matrix(out["actions"])
        raw = decode_matrix(out["raw_actions"])
        assert actions.shape == (10, 4)
        assert raw.shape == (10, 4)
    assert svc._backend.call_count - baseline == n


# --- 6. specs/reset before ready must raise ---


async def test_specs_before_ready_errors(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend
    with pytest.raises(Exception, match="not ready"):
        await svc.do_command({"command": "specs"})


async def test_reset_before_ready_errors(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend
    with pytest.raises(Exception, match="not ready"):
        await svc.do_command({"command": "reset"})


# --- 7. warmup uses specs.state_dim ---


async def test_warmup_runs_configured_number_of_times(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(action_dim=4, n_action_steps=5, state_dim=7)
    svc.reconfigure(_config({"model_path": str(tmp_path), "warmup_inferences": 3}), {})
    await svc.await_ready()
    assert svc._backend.call_count == 3


async def test_warmup_zero_skips_warmup_entirely(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(action_dim=4, n_action_steps=5, state_dim=7)
    svc.reconfigure(_config({"model_path": str(tmp_path), "warmup_inferences": 0}), {})
    await svc.await_ready()
    assert svc._backend.call_count == 0


async def test_warmup_skips_gracefully_when_no_state_feature(tmp_path):
    _make_checkpoint(tmp_path)

    class NoStateBackend(FakePolicyBackend):
        def load(self, *a, **k):
            super().load(*a, **k)
            # Simulate a checkpoint with no observation.state feature: the
            # real backend would report state_dim == 0 in that case.
            object.__setattr__(self._specs, "state_dim", 0)

    svc = VLAPolicy("p")
    svc._backend_factory = lambda: NoStateBackend(action_dim=4, n_action_steps=5)
    svc.reconfigure(_config({"model_path": str(tmp_path), "warmup_inferences": 2}), {})
    await svc.await_ready()
    # Warmup must not crash trying to build a zero-length (or wrong) state
    # vector; it should skip gracefully, leaving call_count at 0.
    assert svc._backend.call_count == 0


# --- 8. multi-camera infer ---


async def test_infer_with_multiple_cameras_reaches_backend(tmp_path):
    _make_checkpoint(tmp_path)
    camera_keys = (
        "observation.images.top",
        "observation.images.wrist",
        "observation.images.side",
    )
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(
        action_dim=4, n_action_steps=5, camera_keys=camera_keys
    )
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    await svc.await_ready()

    captured = {}
    real_predict = svc._backend.predict_chunk

    def spy(images, state, task, rtc_kwargs):
        captured.update(images)
        return real_predict(images, state, task, rtc_kwargs)

    svc._backend.predict_chunk = spy

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    await svc.do_command({
        "command": "infer",
        "images": {k: encode_image(img) for k in camera_keys},
        "state": encode_vector(np.zeros(4, dtype=np.float32)),
        "task": "t",
    })
    assert set(captured.keys()) == set(camera_keys)


# --- 9. malformed image payload names its camera ---


async def test_infer_malformed_image_names_the_failing_camera(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    good = encode_image(np.zeros((224, 224, 3), dtype=np.uint8))
    bad = {"encoding": "jpeg", "data": "not-valid-base64!!"}
    with pytest.raises(Exception, match="observation.images.bad"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": good, "observation.images.bad": bad},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
        })


# ---------------------------------------------------------------------------
# Round 2: coordinator-requested hardening.
# ---------------------------------------------------------------------------


# --- 1. the to_thread boundary is the module's headline property and must
# be independently testable: mutating either call site to run synchronously
# on the loop must turn these red, not just "10s slower".


async def test_load_does_not_block_the_event_loop(tmp_path):
    """A slow backend.load() must run off the event loop (via to_thread).

    If _load ever called backend.load() directly on the loop, the entire
    process would freeze for the duration of the (real, OS-level) blocking
    call inside it, since asyncio's cooperative scheduling only progresses
    at await points -- starving even a trivial concurrent coroutine.
    """
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    release = threading.Event()

    class SlowBackend(FakePolicyBackend):
        def load(self, *a, **k):
            release.wait(timeout=1.0)
            super().load(*a, **k)

    svc._backend_factory = SlowBackend
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0.2)
    ticker_task.cancel()
    try:
        await ticker_task
    except asyncio.CancelledError:
        pass

    release.set()
    await svc.await_ready()

    assert ticks > 100, (
        f"event loop only ticked {ticks} times in 0.2s while a slow load "
        "was in flight -- backend.load() is likely running on the loop "
        "instead of via asyncio.to_thread"
    )


async def test_infer_does_not_block_the_event_loop(tmp_path):
    """predict_chunk must run off the event loop (via asyncio.to_thread).

    At the real target's ~100ms-per-forward-pass GPU latency and a 10 Hz
    control loop, calling it directly on the loop would stall every other
    DoCommand and health check for the duration of each inference.
    """
    _make_checkpoint(tmp_path)
    release = threading.Event()

    class SlowInferBackend(FakePolicyBackend):
        def predict_chunk(self, *a, **k):
            release.wait(timeout=1.0)
            return super().predict_chunk(*a, **k)

    svc = VLAPolicy("p")
    svc._backend_factory = lambda: SlowInferBackend(action_dim=4, n_action_steps=5)
    svc.reconfigure(_config({"model_path": str(tmp_path), "warmup_inferences": 0}), {})
    await svc.await_ready()

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    infer_task = asyncio.create_task(svc.do_command({
        "command": "infer",
        "images": {"observation.images.top": encode_image(img)},
        "state": encode_vector(np.zeros(4, dtype=np.float32)),
        "task": "t",
    }))

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0.2)
    ticker_task.cancel()
    try:
        await ticker_task
    except asyncio.CancelledError:
        pass

    release.set()
    await infer_task

    assert ticks > 100, (
        f"event loop only ticked {ticks} times in 0.2s while a slow infer "
        "was in flight -- predict_chunk is likely running on the loop "
        "instead of via asyncio.to_thread"
    )


# --- 3. non-dict payloads must raise WireError naming the field, not a bare
# AttributeError/TypeError escaping from a builtin (standing requirement 5).


async def test_infer_rejects_non_dict_images_field(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="images"):
        await svc.do_command({
            "command": "infer",
            "images": ["not", "a", "dict"],
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
        })


async def test_infer_rejects_non_dict_image_payload_naming_the_camera(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="observation.images.top"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": "oops"},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
        })


async def test_infer_rejects_non_dict_state_field(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="state"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": [0.0, 0.0, 0.0, 0.0],  # bare list, not {"values": [...]}
            "task": "t",
        })


async def test_infer_rejects_non_dict_truthy_rtc(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="rtc"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
            "rtc": "yes",
        })


# --- 4. warmup must operate on the backend captured at dispatch time, not
# self._backend -- which a concurrent reconfigure can swap out from under a
# still-running worker thread. This races in the thread, not on the loop,
# so neither cancellation nor the generation counter helps.


def test_warmup_once_operates_on_the_given_backend_not_self_backend():
    svc = VLAPolicy("p")
    decoy = FakePolicyBackend(action_dim=4, n_action_steps=5)
    decoy.load("/whatever", device="cpu", dtype="float32", rtc=None)
    svc._backend = decoy  # simulate self._backend already swapped by a newer reconfigure

    target = FakePolicyBackend(action_dim=4, n_action_steps=5)
    target.load("/whatever", device="cpu", dtype="float32", rtc=None)

    svc._warmup_once(target)

    assert target.call_count == 1
    assert decoy.call_count == 0


async def test_load_warmup_uses_backend_captured_at_dispatch(tmp_path):
    """End-to-end: a load's own warmup must never touch a backend swapped
    into self._backend after that load's backend.load() call started."""
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    reached_warmup = threading.Event()
    release = threading.Event()

    class PausesBeforeWarmupBackend(FakePolicyBackend):
        def load(self, *a, **k):
            super().load(*a, **k)
            reached_warmup.set()
            release.wait(timeout=5)

    first_backend_holder = {}

    def factory():
        b = PausesBeforeWarmupBackend(action_dim=4, n_action_steps=5, state_dim=4)
        first_backend_holder["backend"] = b
        return b

    svc._backend_factory = factory
    svc.reconfigure(_config({"model_path": str(tmp_path), "warmup_inferences": 1}), {})

    for _ in range(500):
        if reached_warmup.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("first load never reached its post-load pause")

    first_backend = first_backend_holder["backend"]

    # Simulate a concurrent reconfigure swapping self._backend to a new
    # instance while the first load's warmup is about to run in its worker
    # thread -- without going through reconfigure() itself, which would
    # cancel the in-flight task and confuse the scenario being tested.
    second_backend = FakePolicyBackend(action_dim=9, n_action_steps=3, state_dim=9)
    svc._backend = second_backend

    release.set()
    await asyncio.sleep(0.2)  # let the (still generation-1) warmup actually run

    assert first_backend.call_count == 1, "warmup must run against the backend that was loading"
    assert second_backend.call_count == 0, "warmup must not touch a backend swapped in afterward"

    await svc.await_ready()


# --- 5. a bounded overall load timeout is the last remaining route to a
# permanent "loading": every exception path is covered, but a hung download
# just sits there with nothing to distinguish "downloading 40GB" from
# "wedged".


async def test_load_timeout_transitions_to_failed(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    release = threading.Event()

    class HangingBackend(FakePolicyBackend):
        def load(self, *a, **k):
            release.wait(timeout=5)  # far longer than the configured load_timeout_s
            super().load(*a, **k)

    svc._backend_factory = HangingBackend
    svc.reconfigure(_config({"model_path": str(tmp_path), "load_timeout_s": 0.05}), {})
    await svc.await_ready(expect_failure=True)

    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert "timed out" in status["error"]
    assert "0.05" in status["error"]

    release.set()  # let the orphaned thread finish so it doesn't leak


async def test_load_completes_within_generous_timeout(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend
    svc.reconfigure(_config({"model_path": str(tmp_path), "load_timeout_s": 5.0}), {})
    await svc.await_ready()
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "ready"


# --- minors ---


async def test_infer_rtc_inference_delay_rejects_negative(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="inference_delay"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
            "rtc": {"inference_delay": -5},
        })


async def test_infer_rtc_prev_chunk_without_inference_delay_errors(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    prefix = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(Exception, match="inference_delay"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
            "rtc": {"prev_chunk_left_over": encode_matrix(prefix)},
        })


async def test_infer_rejects_state_length_mismatch(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="state_dim"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": encode_vector(np.zeros(7, dtype=np.float32)),  # backend expects 4
            "task": "t",
        })


async def test_infer_rejects_unexpected_image_keys(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="image_feature_keys"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.wrong_key": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
        })


# ---------------------------------------------------------------------------
# unused_image_features: the checkpoint-declares-more-than-it-consumes case
# (PolicyConfig.unused_image_features / the smolvla_base inheritance).
# ---------------------------------------------------------------------------


async def test_specs_reports_reduced_and_declared_image_feature_keys(tmp_path):
    _make_checkpoint(tmp_path)
    camera_keys = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(
        action_dim=4, n_action_steps=5, camera_keys=camera_keys
    )
    svc.reconfigure(
        _config({
            "model_path": str(tmp_path),
            "unused_image_features": ["observation.images.camera3"],
        }),
        {},
    )
    await svc.await_ready()

    specs = await svc.do_command({"command": "specs"})
    assert specs["image_feature_keys"] == [
        "observation.images.camera1",
        "observation.images.camera2",
    ]
    assert specs["declared_image_feature_keys"] == list(camera_keys)


async def test_infer_accepts_reduced_set_and_rejects_declared_set_as_extra(tmp_path):
    _make_checkpoint(tmp_path)
    camera_keys = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(
        action_dim=4, n_action_steps=5, camera_keys=camera_keys
    )
    svc.reconfigure(
        _config({
            "model_path": str(tmp_path),
            "unused_image_features": ["observation.images.camera3"],
        }),
        {},
    )
    await svc.await_ready()

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Exactly the reduced set (image_feature_keys) is accepted.
    await svc.do_command({
        "command": "infer",
        "images": {
            "observation.images.camera1": encode_image(img),
            "observation.images.camera2": encode_image(img),
        },
        "state": encode_vector(np.zeros(4, dtype=np.float32)),
        "task": "t",
    })

    # The full declared set -- including the checkpoint-only camera3 -- is
    # rejected: camera3 shows up as `extra`, exactly the case
    # unused_image_features exists to make unsatisfiable (there is no
    # harmless value to feed that slot; see the config field's docstring).
    with pytest.raises(Exception, match="extra"):
        await svc.do_command({
            "command": "infer",
            "images": {k: encode_image(img) for k in camera_keys},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": "t",
        })


async def test_unused_image_features_unknown_key_fails_load(tmp_path):
    _make_checkpoint(tmp_path)
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(
        camera_keys=("observation.images.top",)
    )
    svc.reconfigure(
        _config({
            "model_path": str(tmp_path),
            "unused_image_features": ["observation.images.nonexistent"],
        }),
        {},
    )
    await svc.await_ready(expect_failure=True)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert "observation.images.nonexistent" in status["error"]


async def test_unused_image_features_listing_every_camera_fails_load(tmp_path):
    _make_checkpoint(tmp_path)
    camera_keys = ("observation.images.camera1", "observation.images.camera2")
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(camera_keys=camera_keys)
    svc.reconfigure(
        _config({
            "model_path": str(tmp_path),
            "unused_image_features": list(camera_keys),
        }),
        {},
    )
    await svc.await_ready(expect_failure=True)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert "every image feature" in status["error"]


async def test_infer_rejects_non_string_task(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    with pytest.raises(Exception, match="task"):
        await svc.do_command({
            "command": "infer",
            "images": {"observation.images.top": encode_image(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )},
            "state": encode_vector(np.zeros(4, dtype=np.float32)),
            "task": 12345,
        })


async def test_infer_accepts_missing_task_defaults_to_empty_string(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=5)
    out = await svc.do_command({
        "command": "infer",
        "images": {"observation.images.top": encode_image(
            np.zeros((224, 224, 3), dtype=np.uint8)
        )},
        "state": encode_vector(np.zeros(4, dtype=np.float32)),
    })
    assert out is not None
