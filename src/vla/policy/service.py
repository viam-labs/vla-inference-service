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
        self._backend = self._backend_factory()
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
        # Cancelling only abandons the await. The resolver and backend.load run
        # inside asyncio.to_thread, so the worker thread keeps downloading and
        # keeps an executor slot. Bump a generation counter so a superseded load
        # cannot write state after a newer one started; without this, a slow
        # first load can overwrite the result of the config that replaced it.
        self._generation += 1
        self._load_task = asyncio.create_task(self._load(self._generation))

    async def _load(self, generation: int) -> None:
        cfg = self._cfg
        assert cfg is not None

        def _superseded() -> bool:
            return generation != self._generation

        try:
            checkpoint = await asyncio.to_thread(resolve_checkpoint, cfg)
            rtc = cfg.rtc if cfg.rtc.enabled else None
            await asyncio.to_thread(
                self._backend.load, checkpoint, device=cfg.device, dtype=cfg.dtype, rtc=rtc
            )
            for _ in range(cfg.warmup_inferences):
                await asyncio.to_thread(self._warmup_once)
            if _superseded():
                LOGGER.info("discarding superseded load (generation %d)", generation)
                return
            self._state = "ready"
            LOGGER.info("policy ready: %s", self._backend.specs)
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

    def _warmup_once(self) -> None:
        """Run one throwaway inference so the first real call is not an outlier."""
        specs = self._backend.specs
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
        self._backend.predict_chunk(images, state, "warmup", None)

    async def await_ready(self, *, expect_failure: bool = False) -> None:
        """Test helper: wait for the background load to settle."""
        if self._load_task:
            await self._load_task
        if not expect_failure and self._state != "ready":
            raise RuntimeError(f"policy failed to load: {self._error}")

    async def close(self) -> None:
        """Cancel any in-flight load and await its settlement.

        Cancelling only abandons the await -- backend.load runs inside
        asyncio.to_thread, so the worker thread itself keeps running to
        completion in the background. Awaiting the (cancelled) task here just
        waits for the coroutine to unwind, not for that thread to finish.
        """
        if self._load_task is not None:
            self._load_task.cancel()
            try:
                await self._load_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                # _load already handles its own exceptions internally; this
                # is defense in depth so close() itself never raises.
                LOGGER.exception("load task raised while closing")

    async def do_command(self, command: Mapping[str, Any], *, timeout=None, **kwargs) -> Mapping[str, Any]:
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

        # Wrap per-camera: a bare WireError from decode_image says the payload
        # was malformed but not which feed produced it, and a robot log needs
        # to name the camera.
        images = {}
        for key, payload in (command.get("images") or {}).items():
            try:
                images[key] = decode_image(payload)
            except WireError as exc:
                raise WireError(f"{key}: {exc}") from exc
        # decode_vector, not a bare asarray: `command.get("state") or []` would
        # silently yield an empty array for a missing or malformed state, and
        # the shape mismatch would surface deep inside the backend instead.
        state = decode_vector(command["state"]) if "state" in command else None
        if state is None:
            raise WireError("infer requires a 'state' vector")
        task = command.get("task") or ""

        rtc_kwargs = None
        raw_rtc = command.get("rtc")
        if raw_rtc:
            # Protobuf Struct always delivers numbers as doubles, so a client
            # sending 2.5 here is a bug, not a truncation target -- as_int
            # rejects it instead of an `int()` cast silently flooring it.
            rtc_kwargs = {
                "inference_delay": as_int(raw_rtc.get("inference_delay", 0), "rtc.inference_delay")
            }
            prefix = raw_rtc.get("prev_chunk_left_over")
            rtc_kwargs["prev_chunk_left_over"] = decode_matrix(prefix) if prefix else None

        started = time.perf_counter()
        actions, raw = await asyncio.to_thread(
            self._backend.predict_chunk, images, state, task, rtc_kwargs
        )
        latency = time.perf_counter() - started

        return {
            "actions": encode_matrix(actions),
            "raw_actions": encode_matrix(raw),
            "latency_s": latency,
        }
