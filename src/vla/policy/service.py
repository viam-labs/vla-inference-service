"""viam-labs:vla:policy — pure VLA inference over a LeRobot checkpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
from typing_extensions import Self
from viam.proto.app.robot import ServiceConfig
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

from ..config_util import VLAError, as_int
from ..wire import WireError, decode_image, decode_matrix, decode_vector, encode_matrix
from .backend import PolicyBackend
from .config import PolicyConfig
from .lerobot_backend import LeRobotBackend
from .resolver import resolve_checkpoint

LOGGER = logging.getLogger(__name__)


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    """Type-check a DoCommand sub-payload before handing it to wire.py.

    wire.py's decode_* helpers assume a dict and call `.get()` on it
    immediately; a caller (including a hand-written DoCommand) that sends a
    bare list or string instead gets a raw AttributeError that escapes this
    module's own WireError contract. Catching the shape mismatch here keeps
    every error on this boundary a WireError that names the offending field.
    """
    if not isinstance(value, dict):
        raise WireError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


class VLAPolicy(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vla"), "policy")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: PolicyConfig | None = None
        self._backend: PolicyBackend | None = None
        self._backend_factory: Callable[[], PolicyBackend] = LeRobotBackend
        self._state = "idle"
        self._error: str | None = None
        self._load_task: asyncio.Task | None = None
        self._generation = 0
        self._closed = False

    @classmethod
    def new(cls, config: ServiceConfig, dependencies: Mapping[Any, ResourceBase]) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> tuple[Sequence[str], Sequence[str]]:
        PolicyConfig.parse(struct_to_dict(config.attributes))
        return [], []

    def reconfigure(self, config: ServiceConfig, dependencies) -> None:
        """Store config and kick off loading in the background.

        Loading must not block: a multi-GB hub download here would stall the
        module's reconfigure loop and can trip viam-server timeouts.
        """
        self._cfg = PolicyConfig.parse(struct_to_dict(config.attributes))
        self._state = "loading"
        self._error = None
        self._closed = False
        backend = self._backend_factory()
        self._backend = backend
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
        # In production, viam-server's Python SDK reconfigures a resource by
        # removing it and constructing a brand-new instance (remove-then-add
        # -- see module.py's reconfigure_resource), so reconfigure() is never
        # actually called twice on the same live object; the race this
        # counter guards against cannot happen there. It is kept as cheap
        # defense-in-depth -- and because this class's own tests (and any
        # other embedder) can call reconfigure() directly, without that SDK
        # guarantee: cancelling `_load_task` only abandons this coroutine's
        # await, and if the awaited to_thread call happens to complete at the
        # exact moment a second reconfigure()/cancel() races it, cancellation
        # alone can lose that race. The generation check inside `_load` is
        # what stops a stale load from writing state after a newer one won
        # in that case.
        self._generation += 1
        self._load_task = asyncio.create_task(self._load(self._generation, backend))

    async def _load(self, generation: int, backend: PolicyBackend) -> None:
        """Resolve, load, and warm up `backend` -- captured at dispatch time.

        `backend` is passed explicitly rather than read from `self._backend`
        so that if a later reconfigure() swaps `self._backend` out from
        under this still-running background task, this load keeps operating
        on the instance it was actually given. That race lives in a worker
        thread (inside asyncio.to_thread), not on the event loop, so neither
        cancellation nor the generation counter above would catch it -- only
        not reading `self._backend` here does.
        """
        cfg = self._cfg
        assert cfg is not None

        def _superseded() -> bool:
            return generation != self._generation

        try:
            await asyncio.wait_for(self._resolve_and_load(cfg, backend), timeout=cfg.load_timeout_s)
            if _superseded():
                LOGGER.info("discarding superseded load (generation %d)", generation)
                return
            self._state = "ready"
            LOGGER.info("policy ready: %s", backend.specs)
        except asyncio.TimeoutError:
            # The last remaining route to a permanent "loading": every other
            # path below reports failure eventually, but a hung download or
            # deserialize just sits there with nothing to tell an operator
            # "downloading 40GB" from "wedged". This does NOT stop the
            # underlying thread (see close()'s docstring for why it can't);
            # it only stops us from waiting on it forever.
            if _superseded():
                LOGGER.info("ignoring timeout from superseded load (generation %d)", generation)
                return
            self._state = "failed"
            self._error = f"load timed out after {cfg.load_timeout_s}s (load_timeout_s)"
            LOGGER.error("policy load timed out after %ss", cfg.load_timeout_s)
        except VLAError as exc:
            # Expected failures: bad config, unresolvable checkpoint, malformed
            # payload. The message is meant for an operator reading `status`.
            if _superseded():
                LOGGER.info("ignoring failure from superseded load: %s", exc)
                return
            self._state = "failed"
            self._error = str(exc)
            LOGGER.error("policy load failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            # Everything else — torch OOM, a CUDA error, or a genuine bug in
            # this module. It must still land in `status`: this runs in a
            # background task, so letting it propagate would leave state stuck
            # on "loading" forever with the traceback swallowed by asyncio.
            # Distinguish it in the message and log the full traceback, so a
            # bug here is never mistaken for a user configuration error.
            if _superseded():
                LOGGER.info("ignoring failure from superseded load: %s", exc)
                return
            self._state = "failed"
            self._error = f"internal error ({type(exc).__name__}): {exc}"
            LOGGER.exception("policy load failed with an unexpected error")

    async def _resolve_and_load(self, cfg: PolicyConfig, backend: PolicyBackend) -> None:
        """The part of `_load` bounded by `cfg.load_timeout_s`."""
        checkpoint = await asyncio.to_thread(resolve_checkpoint, cfg)
        rtc = cfg.rtc if cfg.rtc.enabled else None
        await asyncio.to_thread(
            backend.load,
            checkpoint,
            device=cfg.device,
            dtype=cfg.dtype,
            rtc=rtc,
            unused_image_features=frozenset(cfg.unused_image_features),
        )
        for _ in range(cfg.warmup_inferences):
            await asyncio.to_thread(self._warmup_once, backend)

    def _warmup_once(self, backend: PolicyBackend) -> None:
        """Run one throwaway inference so the first real call is not an outlier.

        Takes `backend` explicitly for the same reason `_load` does -- see
        its docstring.
        """
        specs = backend.specs
        if specs is None:
            return
        images = {}
        for key in specs.image_feature_keys:
            c, h, w = specs.input_features[key]
            images[key] = np.zeros((int(h), int(w), int(c)), dtype=np.uint8)
        # Skip rather than build a bogus zero-length state vector: not every
        # checkpoint declares an observation.state feature, and warmup runs
        # inside _load, so raising here would fail the entire policy.
        if not specs.state_dim:
            LOGGER.warning("no state feature declared; skipping warmup")
            return
        state = np.zeros(int(specs.state_dim), dtype=np.float32)
        backend.predict_chunk(images, state, "warmup", None)

    async def await_ready(self, *, expect_failure: bool = False) -> None:
        """Test helper: wait for the background load to settle."""
        if self._load_task:
            await self._load_task
        if not expect_failure and self._state != "ready":
            raise RuntimeError(f"policy failed to load: {self._error}")

    async def close(self) -> None:
        """Release this resource so a replacement can safely take its place.

        viam-server's Python SDK reconfigures a resource by removing it and
        constructing a brand-new instance (module.py's reconfigure_resource
        is remove-then-add) -- it never calls reconfigure() twice on the
        same live object. That means close() sits on the critical path of
        *every* reconfigure, not just final shutdown: if this leaves an
        orphaned checkpoint download running, the replacement instance
        viam-server constructs immediately afterward can try to download the
        same repo and block on a filelock the orphan still holds in the
        Hugging Face cache -- surfacing as the REPLACEMENT looking
        permanently wedged on "loading" for reasons nothing in its own log
        explains. Separately, asyncio.run's teardown joins the default
        executor's threads, so an orphan can block interpreter exit until
        viam-server's SIGTERM hits.

        We cannot interrupt the worker thread itself: backend.load() and
        resolve_checkpoint() run inside asyncio.to_thread, which has no
        cancellation hook once the call has actually started -- cancelling
        the task here only abandons our own await on it. What we CAN do:
        stop serving requests immediately, drop the backend reference so a
        loaded torch model isn't kept alive (holding GPU memory) by a
        resource nothing can reach anymore, and log loudly and specifically
        when a load was still running at close time, so the
        wedged-replacement failure mode above is diagnosable from this
        resource's own log instead of a silent hang.
        """
        self._closed = True
        task = self._load_task
        still_loading = task is not None and not task.done()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                # _load already handles its own exceptions internally; this
                # is defense in depth so close() itself never raises.
                LOGGER.exception("load task raised while closing")
        if still_loading:
            LOGGER.warning(
                "close() called for %r while a load was still in progress; "
                "cancelling only abandons this resource's own await -- the "
                "underlying checkpoint download/deserialize thread cannot "
                "be interrupted and may still be running to completion in "
                "the background. If viam-server immediately constructs a "
                "replacement resource for this config (its normal "
                "remove-then-add reconfigure), that replacement may block "
                "indefinitely on a filelock the orphaned thread still holds "
                "in the checkpoint cache, appearing wedged on state=loading "
                "for reasons nothing in ITS OWN log will explain.",
                self.name,
            )
        self._backend = None

    async def do_command(self, command: Mapping[str, Any], *, timeout=None, **kwargs) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError(f"policy {self.name!r} is closed")
        name = command.get("command")
        if name == "status":
            return {"state": self._state, "error": self._error or ""}
        if name == "specs":
            self._require_ready()
            return self._backend.specs.to_dict()
        if name == "reset":
            self._require_ready()
            self._backend.reset()
            return {"ok": True}
        if name == "infer":
            return await self._infer(command)
        raise ValueError(f"unknown command {name!r}")

    def _require_ready(self) -> None:
        if self._state != "ready":
            raise RuntimeError(f"policy not ready (state={self._state}, error={self._error})")

    async def _infer(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require_ready()
        # Captured once so this request keeps using the backend that was
        # ready when it started, even if a concurrent reconfigure() swaps
        # self._backend out from under the asyncio.to_thread call below.
        backend = self._backend
        specs = backend.specs

        images_raw = command.get("images")
        if images_raw is None:
            images_raw = {}
        images_raw = _require_dict(images_raw, "images")

        # Wrap per-camera: a bare WireError from decode_image says the payload
        # was malformed but not which feed produced it, and a robot log needs
        # to name the camera.
        images = {}
        for key, payload in images_raw.items():
            try:
                payload = _require_dict(payload, "image payload")
                images[key] = decode_image(payload)
            except WireError as exc:
                raise WireError(f"{key}: {exc}") from exc

        if specs is not None:
            expected_keys = set(specs.image_feature_keys)
            provided_keys = set(images.keys())
            if expected_keys != provided_keys:
                missing = sorted(expected_keys - provided_keys)
                extra = sorted(provided_keys - expected_keys)
                raise WireError(
                    "'images' keys do not match the policy's image_feature_keys "
                    f"(missing={missing}, extra={extra})"
                )

        if "state" not in command:
            raise WireError("infer requires a 'state' vector")
        # decode_vector, not a bare asarray: `command.get("state") or []` would
        # silently yield an empty array for a missing or malformed state, and
        # the shape mismatch would surface deep inside the backend instead.
        state_payload = _require_dict(command["state"], "state")
        state = decode_vector(state_payload)
        if specs is not None and state.shape[0] != specs.state_dim:
            raise WireError(
                f"'state' vector has length {state.shape[0]}, expected "
                f"{specs.state_dim} (specs.state_dim)"
            )

        task_raw = command.get("task")
        if task_raw is None:
            task = ""
        elif isinstance(task_raw, str):
            task = task_raw
        else:
            raise WireError(f"'task' must be a string, got {type(task_raw).__name__}")

        rtc_kwargs = None
        raw_rtc = command.get("rtc")
        if raw_rtc:
            raw_rtc = _require_dict(raw_rtc, "rtc")
            if "prev_chunk_left_over" in raw_rtc and "inference_delay" not in raw_rtc:
                # Almost certainly a caller bug: a prefix with no delay to
                # align it against is not a meaningful RTC request.
                raise WireError(
                    "rtc.prev_chunk_left_over was provided without "
                    "rtc.inference_delay"
                )
            # Protobuf Struct always delivers numbers as doubles, so a client
            # sending 2.5 here is a bug, not a truncation target -- as_int
            # rejects it instead of an `int()` cast silently flooring it.
            # minimum=0: a negative delay has no physical meaning.
            rtc_kwargs = {
                "inference_delay": as_int(
                    raw_rtc.get("inference_delay", 0), "rtc.inference_delay", minimum=0
                )
            }
            prefix = raw_rtc.get("prev_chunk_left_over")
            rtc_kwargs["prev_chunk_left_over"] = decode_matrix(prefix) if prefix else None

        started = time.perf_counter()
        actions, raw = await asyncio.to_thread(backend.predict_chunk, images, state, task, rtc_kwargs)
        latency = time.perf_counter() - started

        return {
            "actions": encode_matrix(actions),
            "raw_actions": encode_matrix(raw),
            "latency_s": latency,
        }
