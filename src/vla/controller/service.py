"""viam-labs:vla:controller -- the observation/inference/actuation loop.

Two corrections from the plan draft, both load-bearing (see the Task 17
"BLOCKER RESOLVED" callout in the design plan):

  - The arm is commanded via ``await arm.move_to_joint_positions(JointPositions(
    values=...))`` -- a single ``JointPositions``, no options. Installed
    viam-sdk 0.80.0 (the latest on PyPI) has no ``move_through_joint_positions``
    method, and nothing consumes ``MoveOptions``; both exist only in an
    unreleased dev checkout. There is no ``_move_options()`` here.

  - The velocity ceiling therefore lives entirely in the safety layer's
    existing per-tick ``max_joint_delta_degs`` clamp (derived from
    ``max_vel_degs_per_sec`` by ``ControllerConfig``), not in a ``MoveOptions``
    the arm has no way to receive. The derived per-tick budget is logged once
    at ``reconfigure()`` time so an operator can see what their velocity limit
    actually implies.

``safety.stop_on_error`` governs exactly one boundary: whether a failure while
*producing the next action* (camera read, observation assembly, or the policy
call itself) halts the loop and stops the arm (``True``, the default) or is
logged and the tick skipped, leaving the loop running for the next tick
(``False``). Every other failure -- a safety refusal (dimension mismatch,
start-pose refusal), or any failure of an actual arm/gripper command -- is
unconditionally fatal regardless of this flag: those happen after inference
already succeeded and are adjacent to physical motion, so there is no
"skip and try again next tick" that would be safe. This is a deliberate
reading of the design doc's error-handling table (whose only qualified row is
inference failure); it is not a retry/backoff mechanism -- skipping a tick is
exactly the loop's ordinary next iteration, with no attempt counter or delay
logic added.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
from typing_extensions import Self
from viam.proto.app.robot import ServiceConfig
from viam.proto.component.arm import JointPositions
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

from ..wire import decode_matrix, encode_vector
from .config import ControllerConfig
from .gripper import GripperAdapter, make_gripper_adapter
from .observation import ObservationBuilder
from .safety import SafetyLayer, SafetyLimits
from .scheduler import SequentialScheduler
from .units import to_degrees

LOGGER = logging.getLogger(__name__)

# States, in the order they normally advance through a run:
#   idle -> waiting_for_policy -> running -> stopped
# with `error` reachable from any of the last three. `idle` is also the
# state produced by reconfigure(), whether or not a run was ever started.
STATES = ("idle", "waiting_for_policy", "running", "stopped", "error")


class VLAController(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vla"), "controller")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: ControllerConfig | None = None
        self._deps: Mapping[str, Any] = {}
        self._state = "idle"
        self._last_error: str | None = None
        self._task: asyncio.Task | None = None
        self._active_task_text = ""
        self._safety: SafetyLayer | None = None
        self._specs: dict[str, Any] | None = None
        self._latencies: list[float] = []
        self._measured_fps = 0.0
        self._scheduler: SequentialScheduler | None = None
        self._stop_task: asyncio.Task | None = None

    @classmethod
    def new(cls, config: ServiceConfig, dependencies: Mapping[Any, ResourceBase]) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> tuple[Sequence[str], Sequence[str]]:
        cfg = ControllerConfig.parse(struct_to_dict(config.attributes))
        return cfg.dependencies(), []

    def reconfigure(self, config: ServiceConfig, dependencies) -> None:
        """Rebuild from config. Never auto-resumes motion after a config change."""
        was_running = self._task is not None and not self._task.done()
        # Capture the arm currently in motion before rebinding _cfg/_deps below,
        # or the scheduled stop would target the *new* arm.
        old_arm = self._deps.get(self._cfg.arm) if (was_running and self._cfg) else None
        self._stop_loop_sync()
        if old_arm is not None:
            # reconfigure is sync, so the stop is scheduled rather than awaited.
            # Cancelling the loop alone leaves the arm executing its last
            # command. Keep a reference: CPython may otherwise garbage-collect
            # a bare task handle before it ever runs.
            self._stop_task = asyncio.create_task(self._stop_arm(old_arm))

        self._cfg = ControllerConfig.parse(struct_to_dict(config.attributes))
        self._deps = {self._key(k): v for k, v in dependencies.items()}
        self._specs = None
        self._scheduler = None
        self._safety = None
        self._latencies = []
        self._measured_fps = 0.0
        self._state = "idle"
        self._last_error = None
        self._log_velocity_budget()

    def _log_velocity_budget(self) -> None:
        s = self._cfg.safety
        if s.max_vel_degs_per_sec is not None:
            LOGGER.info(
                "safety: max_vel_degs_per_sec=%.4f deg/s at fps=%.4f implies a per-tick "
                "budget of max_joint_delta_degs=%.6f deg",
                s.max_vel_degs_per_sec,
                self._cfg.fps,
                s.max_joint_delta_degs,
            )
        else:
            LOGGER.info(
                "safety: max_joint_delta_degs=%.6f deg per tick at fps=%.4f "
                "(no max_vel_degs_per_sec configured)",
                s.max_joint_delta_degs,
                self._cfg.fps,
            )

    @staticmethod
    def _key(dep_key: Any) -> str:
        return getattr(dep_key, "name", str(dep_key))

    def _resource(self, name: str) -> Any:
        if name not in self._deps:
            raise RuntimeError(f"dependency {name!r} was not resolved")
        return self._deps[name]

    async def do_command(self, command: Mapping[str, Any], *, timeout=None, **kwargs):
        name = command.get("command")
        if name == "start":
            return await self._start(command)
        if name == "stop":
            await self._stop()
            return {"ok": True}
        if name == "status":
            return self._status()
        raise ValueError(f"unknown command {name!r}")

    def _status(self) -> dict[str, Any]:
        avg = float(np.mean(self._latencies)) if self._latencies else 0.0
        return {
            "state": self._state,
            "mode": (self._specs or {}).get("_resolved_mode", self._cfg.mode if self._cfg else ""),
            "queue_size": self._scheduler.qsize() if self._scheduler else 0,
            "avg_latency_s": avg,
            "measured_fps": self._measured_fps,
            "clamp_counts": dict(self._safety.clamp_counts) if self._safety else {},
            "last_error": self._last_error or "",
        }

    async def _start(self, command: Mapping[str, Any]) -> dict[str, Any]:
        if self._task and not self._task.done():
            raise RuntimeError("already running")
        assert self._cfg is not None
        self._active_task_text = command.get("task") or self._cfg.task
        if not self._active_task_text:
            raise ValueError("no task instruction: pass 'task' or configure a default")
        self._last_error = None
        self._state = "waiting_for_policy"
        # Ack immediately: waiting out policy_ready_timeout_s inline would
        # exceed the deadline most DoCommand callers use, so the "still
        # loading" error would never reach anyone.
        self._task = asyncio.create_task(self._run())
        return {"ok": True}

    async def _stop(self) -> None:
        # "stopped" is a distinct state from "idle": it means a run was
        # actually halted by this command, not merely that none was ever
        # started (or already halted). Capture that before awaiting the
        # task below, which can itself let a concurrent state change (e.g.
        # a failure landing on "error") run first.
        was_active = self._state in ("waiting_for_policy", "running")
        self._stop_loop_sync()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                # _run() already handles its own exceptions internally
                # (transitioning to state="error"); this is defense in depth
                # so _stop() itself never raises.
                LOGGER.exception("loop task raised while stopping")
        await self._safe_stop_arm()
        if was_active and self._state != "error":
            self._state = "stopped"

    def _stop_loop_sync(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    @staticmethod
    async def _stop_arm(arm: Any) -> None:
        if arm is None:
            return
        try:
            await arm.stop()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("failed to stop arm: %s", exc)

    async def _safe_stop_arm(self) -> None:
        if not self._cfg:
            return
        await self._stop_arm(self._deps.get(self._cfg.arm))

    async def _await_policy(self) -> dict[str, Any]:
        cfg = self._cfg
        policy = self._resource(cfg.policy_service)
        deadline = time.monotonic() + cfg.policy_ready_timeout_s
        delay = 0.05
        while True:
            status = await policy.do_command({"command": "status"})
            state = status.get("state")
            if state == "ready":
                return await policy.do_command({"command": "specs"})
            if state == "failed":
                raise RuntimeError(f"policy failed to load: {status.get('error')}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"policy not ready after {cfg.policy_ready_timeout_s}s")
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, 5.0)

    def _resolve_mode(self, specs: dict[str, Any]) -> str:
        cfg = self._cfg
        if cfg.mode == "sequential":
            return "sequential"
        if cfg.mode == "rtc":
            if not specs.get("supports_rtc"):
                raise RuntimeError("mode=rtc but the policy does not support RTC")
            if specs.get("relative_actions"):
                raise RuntimeError(
                    "mode=rtc but the checkpoint uses relative actions, which requires "
                    "prefix re-anchoring that is not implemented; use mode=sequential"
                )
            raise RuntimeError(
                "mode=rtc is not implemented yet (RTCScheduler is a follow-up plan); "
                "use mode=sequential or mode=auto"
            )
        return "sequential"  # auto: RTCScheduler ships in a follow-up plan

    def _build_safety(self, gripper: GripperAdapter) -> SafetyLayer:
        s = self._cfg.safety
        return SafetyLayer(
            SafetyLimits(
                max_joint_delta_degs=s.max_joint_delta_degs,
                max_start_delta_degs=s.max_start_delta_degs,
                joint_limits_degs=s.joint_limits_degs,
                gripper_in_degrees=(not gripper.in_state) or gripper.uses_degrees,
            )
        )

    def _check_action_dim(self, specs: dict[str, Any], gripper: GripperAdapter) -> None:
        cfg = self._cfg
        expected_dim = len(cfg.state_joint_indices) + (1 if gripper.in_state else 0)
        actual_dim = int(specs["action_dim"])
        if actual_dim == expected_dim:
            return
        if not gripper.in_state and actual_dim == expected_dim + 1:
            # A specific, named case of the mismatch below: the checkpoint's
            # extra dimension is exactly one channel, and gripper.type="none"
            # means there is nowhere configured for it to go. Calling this
            # out by name beats leaving an operator to reverse-engineer it
            # from a bare "expected 5, got 6".
            raise RuntimeError(
                f"the checkpoint emits a gripper channel (action_dim={actual_dim}, "
                f"configured joints={len(cfg.state_joint_indices)}) but gripper.type "
                '="none"; configure a gripper adapter (gripper.type) to receive it'
            )
        raise RuntimeError(
            f"action_dim mismatch: policy emits {actual_dim} dimensions but config "
            f"describes {expected_dim} ({len(cfg.state_joint_indices)} joints + "
            f"{'1 gripper' if gripper.in_state else 'no gripper'})"
        )

    def _image_sizes(self, specs: dict[str, Any]) -> dict[str, tuple[int, int]]:
        cfg = self._cfg
        # int() is mandatory, not defensive. specs arrives via
        # GenericClient.do_command -> struct_to_dict, and protobuf Struct
        # stores every number as a double, so [3, 224, 224] comes back as
        # [3.0, 224.0, 224.0]. PIL's resize() raises "TypeError: integer
        # argument expected, got float" on a bare float.
        sizes = {
            key: (int(specs["input_features"][key][1]), int(specs["input_features"][key][2]))
            for key in specs["image_feature_keys"]
        }
        missing = set(sizes) - set(cfg.cameras)
        if missing:
            raise RuntimeError(
                f"the policy expects camera feature(s) {sorted(missing)} that are not "
                f"mapped in `cameras` (configured: {sorted(cfg.cameras)})"
            )
        return sizes

    async def _run(self) -> None:
        cfg = self._cfg
        try:
            specs = await self._await_policy()
            mode = self._resolve_mode(specs)
            specs = dict(specs)
            specs["_resolved_mode"] = mode
            self._specs = specs

            gripper = make_gripper_adapter(cfg.gripper, self._deps)
            self._check_action_dim(specs, gripper)
            image_sizes = self._image_sizes(specs)

            # Only cameras the policy actually asked for: a camera configured
            # but not among specs.image_feature_keys must never be read or
            # sent -- it has no entry in image_sizes, so ObservationBuilder
            # would fail trying to resize/encode it for a resolution nothing
            # declared, for a feed the policy never asked to see.
            builder = ObservationBuilder(
                cameras={key: self._resource(cfg.cameras[key]) for key in image_sizes},
                arm=self._resource(cfg.arm),
                gripper=gripper,
                state_joint_indices=cfg.state_joint_indices,
                state_units=cfg.state_units,
                image_sizes=image_sizes,
                image_encoding=cfg.image_encoding,
                jpeg_quality=cfg.jpeg_quality,
                duration_warn_s=cfg.duration_warn_s,
                stale_frame_warn_s=cfg.stale_frame_warn_s,
            )
            self._safety = self._build_safety(gripper)
            self._scheduler = SequentialScheduler(lambda rtc: self._infer(builder, rtc))

            self._state = "running"
            await self._loop(self._scheduler, gripper)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._state = "error"
            self._last_error = str(exc)
            LOGGER.error("controller stopped: %s", exc)
            await self._safe_stop_arm()

    async def _infer(
        self, builder: ObservationBuilder, rtc: dict[str, Any] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = await builder.build()
        payload: dict[str, Any] = {
            "command": "infer",
            "images": obs.images,
            "state": encode_vector(obs.state),
            "task": self._active_task_text,
        }
        if rtc:
            payload["rtc"] = rtc
        started = time.perf_counter()
        out = await self._resource(self._cfg.policy_service).do_command(payload)
        self._latencies.append(time.perf_counter() - started)
        del self._latencies[:-50]
        return decode_matrix(out["actions"]), decode_matrix(out["raw_actions"])

    def _record_tick(self, last_tick: float) -> float:
        now = time.perf_counter()
        elapsed = now - last_tick
        if elapsed > 0:
            self._measured_fps = 1.0 / elapsed
        return now

    async def _pace(self, tick_started: float, period: float) -> None:
        remaining = period - (time.perf_counter() - tick_started)
        if remaining > 0:
            await asyncio.sleep(remaining)
        else:
            LOGGER.warning(
                "tick overran its budget: took %.4fs against a %.4fs budget (fps=%.2f); "
                "the loop cannot keep pace and will drift instead of holding fps",
                time.perf_counter() - tick_started,
                period,
                self._cfg.fps,
            )

    async def _loop(self, scheduler: SequentialScheduler, gripper: GripperAdapter) -> None:
        cfg = self._cfg
        arm = self._resource(cfg.arm)
        period = 1.0 / cfg.fps
        first = True
        last_tick = time.perf_counter()

        while True:
            tick_started = time.perf_counter()

            try:
                action = await scheduler.next_action()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # Camera reads, observation assembly, and the policy call
                # itself all surface here (SequentialScheduler.next_action
                # calls straight through to _infer). Gated by stop_on_error:
                # by default this is fatal (stop the arm, halt); an operator
                # who opts out gets the tick skipped and the loop kept alive
                # for the next one -- never a retry counter or a backoff, just
                # the loop's ordinary next iteration.
                self._last_error = str(exc)
                LOGGER.error("tick failed while producing the next action: %s", exc)
                if cfg.safety.stop_on_error:
                    raise
                last_tick = self._record_tick(last_tick)
                await self._pace(tick_started, period)
                continue

            degrees = to_degrees(np.asarray(action, dtype=np.float32), cfg.action_units)

            current_all = list((await arm.get_joint_positions()).values)
            joint_current = [current_all[i] for i in cfg.state_joint_indices]
            if gripper.arm_joint_index is not None:
                joint_current.append(current_all[gripper.arm_joint_index])
            elif gripper.in_state:
                joint_current.append(await gripper.read())
            current = np.asarray(joint_current, dtype=np.float32)

            if first:
                # Unconditionally fatal, not gated by stop_on_error: refusing
                # beats slowly moving somewhere nobody asked for, and there is
                # no tick to "skip" before the very first command.
                self._safety.check_start(degrees, current)
                first = False

            safe = self._safety.apply(degrees, current)

            joint_values = [float(v) for v in safe[: len(cfg.state_joint_indices)]]
            if gripper.arm_joint_index is not None:
                joint_values.append(float(safe[-1]))

            # Arm/gripper command failures are unconditionally fatal too: by
            # this point inference already succeeded and safety already
            # cleared the action, so there is no "try again next tick" that
            # would be safe -- the arm itself is reporting the fault.
            await arm.move_to_joint_positions(JointPositions(values=joint_values))
            if gripper.in_state and gripper.arm_joint_index is None:
                await gripper.write(float(safe[-1]))

            last_tick = self._record_tick(last_tick)
            await self._pace(tick_started, period)

    async def close(self) -> None:
        await self._stop()
