"""viam-labs:vla:controller -- the observation/inference/actuation loop.

Two action spaces share this loop. They differ in four places -- the
observation's state half, the safety layer, the arm call, and the unit
conversion -- and in nothing else: the scheduler, action queue, pacing,
starvation bounds, status reporting, and async/RTC modes are common.

``action_space="joints"`` (the default) commands the arm via ``await
arm.move_to_joint_positions(JointPositions(values=...))`` -- a single
``JointPositions``, no options. Installed viam-sdk 0.80.0 (the latest on
PyPI) has no ``move_through_joint_positions`` and nothing consumes
``MoveOptions``; both exist only in an unreleased dev checkout. The velocity
ceiling therefore lives entirely in the safety layer's per-tick
``max_joint_delta_degs`` clamp (derived from ``max_vel_degs_per_sec`` by
``ControllerConfig``), logged once at ``reconfigure()`` so an operator can
see what their limit implies.

``action_space="delta-ee"`` reads the tool pose from ``get_end_position()``,
builds the 9-dim state the dataset stored, and treats the policy's 6-dim
output as a body-frame pose delta: it is composed onto the *measured* pose
with ``pose.state_compose`` and written with ``arm.move_to_position``. The
same "the per-tick clamp is the velocity limit" argument applies with more
force there, because ``move_to_position`` has no notion of a tick at all --
see ``CartesianSafetyLayer``. Joint limits on that path are enforced by the
arm driver's own IK rather than by this service; ``move_to_position`` is the
arm component method, not the motion service, so there is no obstacle
avoidance either.

A ``move_to_position`` refusal is the one arm-command failure this loop does
not treat as immediately fatal -- see ``_command_pose``.

``safety.stop_on_error`` governs exactly one boundary: whether a failure while
*producing the next action* (camera read, observation assembly, or the policy
call itself) halts the loop and stops the arm (``True``, the default) or is
logged and the tick skipped, leaving the loop running for the next tick
(``False``). Every other failure -- a safety refusal, or any failure of an
actual arm/gripper command -- is unconditionally fatal regardless of this
flag: those happen after inference already succeeded and are adjacent to
physical motion, so there is no "skip and try again next tick" that would be
safe. Skipping a tick is exactly the loop's ordinary next iteration; there is
no retry counter or backoff.

``stop_on_error=False`` is bounded: ``starvation_grace_ticks`` caps
*consecutive* tick failures, and more than that in a row stops the arm and
halts regardless of ``stop_on_error``, so a deployment that opts out of
per-failure halting still cannot spin forever reporting ``running`` while
every tick silently fails.

The write to the arm is a full-width target seeded from the *measured* joint
positions, not a dense positional list: ``state_joint_indices`` is an
arbitrary (possibly non-contiguous, possibly reordered) index list, and
``move_to_joint_positions`` has no notion of "only these joints" -- a
positional write would map clamped slot *i* onto arm joint *i* rather than
``state_joint_indices[i]``, scrambling the mapping and voiding the delta
clamp wherever the two differ (e.g. a 5-DoF policy on a 6-joint arm).
``state_joint_indices`` and ``gripper.joint_index`` are validated against the
arm's actual joint count once at ``_run()`` setup, before any motion --
config parsing has no access to the arm.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
from typing_extensions import Self
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import Pose
from viam.proto.component.arm import JointPositions
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

from ..wire import decode_matrix, encode_vector
from .config import DELTA_EE, DELTA_EE_ACTION_DIM, DELTA_EE_STATE_DIM, ControllerConfig
from .gripper import GripperAdapter, make_gripper_adapter
from .observation import ObservationBuilder
from .pose import orientation_vector, pose_state, state_compose, state_rotation
from .safety import CartesianLimits, CartesianSafetyLayer, SafetyLayer, SafetyLimits
from .scheduler import AsyncScheduler, ChunkScheduler, SequentialScheduler
from .units import to_degrees, to_working

LOGGER = logging.getLogger(__name__)


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
        # Bounded to the most recent 50: ``avg_latency_s`` must reflect
        # recent performance, not a lifetime average a long run dilutes.
        self._latencies: deque[float] = deque(maxlen=50)
        self._measured_fps = 0.0
        self._starved_ticks = 0
        self._scheduler: ChunkScheduler | None = None
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
        self._latencies.clear()
        self._measured_fps = 0.0
        self._starved_ticks = 0
        self._state = "idle"
        self._last_error = None
        self._log_velocity_budget()

    def _log_velocity_budget(self) -> None:
        if self._cfg.action_space == DELTA_EE:
            self._log_cartesian_budget()
            return
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

    def _log_cartesian_budget(self) -> None:
        """The delta-EE half of the same disclosure.

        Logged in both directions -- the configured ceiling and the tool speed
        it implies -- because `move_to_position` accepts no speed argument, so
        the per-tick clamp times fps is the only speed limit there is, and an
        operator has no other place to read it off.
        """
        s = self._cfg.safety
        fps = self._cfg.fps
        LOGGER.info(
            "safety: max_tcp_delta_mm=%.6f mm and max_tcp_rot_delta_rads=%.6f rad per tick "
            "at fps=%.4f, implying at most %.4f mm/s of tool travel and %.4f rad/s of tool "
            "rotation (%s); joint limits on this path are enforced by the arm driver's IK, "
            "not by this service",
            s.max_tcp_delta_mm,
            s.max_tcp_rot_delta_rads,
            fps,
            s.max_tcp_delta_mm * fps,
            s.max_tcp_rot_delta_rads * fps,
            "derived from the configured velocities"
            if (s.max_tcp_vel_mms_per_sec is not None or s.max_tcp_rot_vel_rads_per_sec is not None)
            else "no max_tcp_vel_mms_per_sec / max_tcp_rot_vel_rads_per_sec configured",
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
            # Cumulative count of ticks the loop held position because the
            # queue was empty and a background inference was still in
            # flight (only possible under mode="async") -- distinct from
            # the *consecutive*-run bound that escalates to a halt at
            # starvation_grace_ticks: this is a running total across the
            # whole session, the same shape as clamp_counts, so an operator
            # can see the loop is quietly stalling without reading logs.
            "starved_ticks": self._starved_ticks,
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
        # A restart must not report the *previous* run's telemetry as live:
        # without this, status would show yesterday's measured_fps/
        # avg_latency_s/clamp_counts as if they belonged to a run that
        # hasn't produced a single tick yet.
        self._specs = None
        self._scheduler = None
        self._safety = None
        self._latencies.clear()
        self._measured_fps = 0.0
        self._starved_ticks = 0
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
        if self._scheduler is not None:
            # Cancelling the loop task above does not guarantee an
            # AsyncScheduler's background inference task gets cancelled too
            # -- that only cascades automatically when the loop task
            # happened to be suspended *directly* on it (the blocking
            # first-call path). The non-blocking "return None while starved"
            # path leaves the background task running independently, so
            # without this it would keep running orphaned past the point
            # the controller reports stopped. SequentialScheduler's close()
            # is an inherited no-op, so this is a no-op for it.
            await self._scheduler.close()
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
        if cfg.mode == "async":
            return "async"
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
                "use mode=sequential, mode=async, or mode=auto"
            )
        # auto: "async" is explicit opt-in only (see config.py's docstring) --
        # an existing deployment's behavior must not change underneath it
        # just because this module gained a new mode.
        return "sequential"

    def _build_scheduler(self, mode: str, builder: ObservationBuilder) -> ChunkScheduler:
        if mode == "async":
            threshold = self._resolve_queue_threshold()
            return AsyncScheduler(
                lambda rtc: self._infer(builder, rtc), threshold, self._cfg.fps
            )
        return SequentialScheduler(lambda rtc: self._infer(builder, rtc))

    def _resolve_queue_threshold(self) -> int:
        """`queue_threshold=None` (the config default) means "derive it" --
        config parsing has no access to `specs`, so it cannot pick a value
        itself. `n_action_steps - 1` fires the background refill as early
        as possible (maximizing overlap runway), which is unambiguously
        right in the latency-bound regime this scheduler exists for, and
        harmless in the fast-inference regime, where the queue never drains
        that low before the (fast) refill lands regardless. An explicit
        config value always overrides this -- this is a default, not a
        clamp.
        """
        configured = self._cfg.queue_threshold
        if configured is not None:
            return configured
        n_action_steps = int(self._specs["n_action_steps"])
        derived = max(0, n_action_steps - 1)
        LOGGER.info(
            "queue_threshold not configured; deriving %d from the checkpoint's "
            "n_action_steps=%d (fires the background refill as early as possible, "
            "maximizing overlap runway)",
            derived,
            n_action_steps,
        )
        return derived

    def _build_safety(self, gripper: GripperAdapter) -> SafetyLayer | CartesianSafetyLayer:
        s = self._cfg.safety
        if self._cfg.action_space == DELTA_EE:
            return CartesianSafetyLayer(
                CartesianLimits(
                    max_tcp_delta_mm=s.max_tcp_delta_mm,
                    max_tcp_rot_delta_rads=s.max_tcp_rot_delta_rads,
                )
            )
        return SafetyLayer(
            SafetyLimits(
                max_joint_delta_degs=s.max_joint_delta_degs,
                max_start_delta_degs=s.max_start_delta_degs,
                joint_limits_degs=s.joint_limits_degs,
                gripper_in_degrees=not gripper.has_normalized_tail,
            )
        )

    def _check_action_dim(self, specs: dict[str, Any], gripper: GripperAdapter) -> None:
        cfg = self._cfg
        if cfg.action_space == DELTA_EE:
            self._check_delta_ee_dims(specs)
            return
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

    @staticmethod
    def _check_delta_ee_dims(specs: dict[str, Any]) -> None:
        """Refuse a checkpoint whose vectors are not the delta-EE layout.

        Both widths are fixed by the dataset contract rather than by config,
        so this is a straight equality check and not the negotiation
        `_check_action_dim` performs for joints. It earns its place because
        every downstream mistake it prevents is silent: a 7-dim action would
        have its extra channel dropped by `state_compose`'s slicing, and a
        state of the wrong width would be Gram-Schmidt'd out of whatever six
        numbers happened to land in `state[3:9]` -- both produce arm motion.

        `state_dim` is `None` for a checkpoint that declares no
        `observation.state` feature at all, which cannot be a delta-EE
        checkpoint; it is reported as the mismatch it is rather than skipped.
        """
        action_dim = int(specs["action_dim"])
        if action_dim != DELTA_EE_ACTION_DIM:
            raise RuntimeError(
                f"action_dim mismatch: action_space={DELTA_EE!r} needs a checkpoint emitting "
                f"{DELTA_EE_ACTION_DIM} dimensions [dx, dy, dz, drx, dry, drz], but the policy "
                f"emits {action_dim}"
            )
        state_dim = specs.get("state_dim")
        if state_dim is None or int(state_dim) != DELTA_EE_STATE_DIM:
            raise RuntimeError(
                f"state_dim mismatch: action_space={DELTA_EE!r} needs a checkpoint whose "
                f"observation.state is {DELTA_EE_STATE_DIM} dimensions "
                "[x, y, z, r00, r01, r02, r10, r11, r12], but the policy declares "
                f"{state_dim!r}"
            )

    def _check_joint_indices(self, measured: list[float], gripper: GripperAdapter) -> None:
        """Refuse before any motion if config references a joint the arm
        does not have.

        `state_joint_indices` and `gripper.joint_index` are only checked for
        `>= 0` at config-parse time -- config parsing has no access to the
        arm, so it cannot know how many joints it actually has. This is the
        first point in the lifecycle that does.
        """
        cfg = self._cfg
        n = len(measured)
        bad = sorted({i for i in cfg.state_joint_indices if i >= n})
        if bad:
            raise RuntimeError(
                f"state_joint_indices {bad} exceed the arm's {n} joints "
                f"(arm reports joints 0..{n - 1})"
            )
        if gripper.arm_joint_index is not None and gripper.arm_joint_index >= n:
            raise RuntimeError(
                f"gripper.joint_index {gripper.arm_joint_index} exceeds the arm's "
                f"{n} joints (arm reports joints 0..{n - 1})"
            )

    async def _preflight_gripper(self, gripper: GripperAdapter) -> None:
        """Probe the gripper before any arm motion, so a driver that cannot
        be read or written fails here rather than on the first real tick,
        after the arm has already been commanded once -- the
        refuse-before-motion discipline every other `_run()` check keeps.

        Reads the gripper's own current value and writes it straight back: a
        no-op for `ServoGripper`, and usually one for `DoCommandGripper` --
        but not always, since its `read()` clamps to [0, 1], so a driver
        resting outside its configured endpoints reads as an endpoint rather
        than its true position (an so-101 resting at 98% with open_value=95
        reads 0.0, and writing that back commands 95, a real ~3% aperture
        move). `arm_joint`/`none` have no separate component to probe.
        """
        if not gripper.has_normalized_tail:
            return
        value = await gripper.read()
        await gripper.write(value)

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

            arm = self._resource(cfg.arm)
            if cfg.action_space == DELTA_EE:
                # The delta-EE equivalent of the joint-index check below:
                # refuse before any motion if the arm cannot supply the one
                # reading this action space is built on. A driver that does
                # not implement `get_end_position` (or returns something that
                # is not a Pose) must fail here, not on the first tick after
                # the loop has already reported "running".
                await self._check_end_position(arm)
            else:
                measured = list((await arm.get_joint_positions()).values)
                self._check_joint_indices(measured, gripper)
            await self._preflight_gripper(gripper)

            # Only cameras the policy actually asked for: a camera configured
            # but not among specs.image_feature_keys must never be read or
            # sent -- it has no entry in image_sizes, so ObservationBuilder
            # would fail trying to resize/encode it for a resolution nothing
            # declared, for a feed the policy never asked to see.
            builder = ObservationBuilder(
                cameras={key: self._resource(cfg.cameras[key]) for key in image_sizes},
                arm=arm,
                gripper=gripper,
                state_joint_indices=cfg.state_joint_indices,
                state_units=cfg.state_units,
                image_sizes=image_sizes,
                image_encoding=cfg.image_encoding,
                jpeg_quality=cfg.jpeg_quality,
                image_fit=cfg.image_fit,
                action_space=cfg.action_space,
                duration_warn_s=cfg.duration_warn_s,
                stale_frame_warn_s=cfg.stale_frame_warn_s,
            )
            self._safety = self._build_safety(gripper)
            self._scheduler = self._build_scheduler(mode, builder)

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

    async def _loop(self, scheduler: ChunkScheduler, gripper: GripperAdapter) -> None:
        cfg = self._cfg
        arm = self._resource(cfg.arm)
        period = 1.0 / cfg.fps
        first = True
        last_tick = time.perf_counter()
        consecutive_failures = 0
        consecutive_starved_ticks = 0
        consecutive_rejected_moves = 0

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
                # the loop's ordinary next iteration. But that opt-out is
                # bounded: starvation_grace_ticks caps how many *consecutive*
                # failures are tolerated before this escalates to fatal
                # regardless of stop_on_error, so a bad deployment cannot
                # spin forever reporting "running" while every tick silently
                # fails.
                consecutive_failures += 1
                self._last_error = str(exc)
                LOGGER.error(
                    "tick failed while producing the next action (%d consecutive): %s",
                    consecutive_failures,
                    exc,
                )
                if cfg.safety.stop_on_error:
                    raise
                if consecutive_failures > cfg.starvation_grace_ticks:
                    raise RuntimeError(
                        f"{consecutive_failures} consecutive tick failures exceeded "
                        f"starvation_grace_ticks={cfg.starvation_grace_ticks}; stopping "
                        "regardless of stop_on_error"
                    ) from exc
                last_tick = self._record_tick(last_tick)
                await self._pace(tick_started, period)
                continue

            consecutive_failures = 0

            if action is None:
                # Only AsyncScheduler ever returns this: the queue is empty
                # and a background inference is already working on the next
                # chunk. Hold position -- there is no new action to
                # safety-clamp against, and re-sending the last command
                # would be indistinguishable from "an action" to anyone
                # reading clamp_counts. Bounded the same way tick failures
                # are, by starvation_grace_ticks, but unconditionally (not
                # gated by stop_on_error): an empty tick is not a failure to
                # skip, it is an absence of anything to do, so a deployment
                # cannot spin forever reporting "running" while every tick
                # silently starves either.
                self._starved_ticks += 1
                consecutive_starved_ticks += 1
                if consecutive_starved_ticks > cfg.starvation_grace_ticks:
                    raise RuntimeError(
                        f"{consecutive_starved_ticks} consecutive ticks starved (queue "
                        "empty, inference still in flight) exceeded starvation_grace_ticks="
                        f"{cfg.starvation_grace_ticks}; stopping"
                    )
                last_tick = self._record_tick(last_tick)
                await self._pace(tick_started, period)
                continue

            consecutive_starved_ticks = 0

            if cfg.action_space == DELTA_EE:
                if await self._command_pose(arm, action):
                    consecutive_rejected_moves = 0
                else:
                    consecutive_rejected_moves += 1
                    if consecutive_rejected_moves > cfg.starvation_grace_ticks:
                        raise RuntimeError(
                            f"{consecutive_rejected_moves} consecutive move_to_position "
                            "refusals exceeded starvation_grace_ticks="
                            f"{cfg.starvation_grace_ticks}; stopping"
                        )
            else:
                await self._command_joints(arm, gripper, action, check_start=first)
                first = False

            last_tick = self._record_tick(last_tick)
            await self._pace(tick_started, period)

    async def _command_joints(
        self, arm: Any, gripper: GripperAdapter, action: np.ndarray, *, check_start: bool
    ) -> None:
        """One tick of `action_space="joints"`: clamp, then write joint angles."""
        cfg = self._cfg

        # Convert the *driven joints* only. A normalized gripper channel
        # (every variant except `arm_joint`) is already 0.0-1.0 and must
        # not be scaled: under `action_units="radians"` an unconditional
        # conversion multiplies it by ~57.3, so a policy output of 0.5
        # becomes 28.6, the safety layer clamps it to 1.0, and the gripper
        # sits fully closed on every tick. `observation.py`'s read side
        # already splits this way (see its `# already normalized` tail);
        # the write side has to match or the two disagree about units.
        raw_action = np.asarray(action, dtype=np.float32)
        head = len(raw_action) - (1 if gripper.has_normalized_tail else 0)
        degrees = np.concatenate(
            [to_degrees(raw_action[:head], cfg.action_units), raw_action[head:]]
        )

        # Every joint the arm has, not just the driven ones: `positions`
        # below seeds from the *measured* values so an un-driven joint
        # (state_joint_indices need not be contiguous, or cover every
        # joint -- a 5-DoF policy on a 6-joint arm is a legitimate
        # config) holds its measured position rather than being sent an
        # implicit 0.0, and so each driven value lands on the joint it
        # was actually clamped against.
        measured = list((await arm.get_joint_positions()).values)
        joint_current = [measured[i] for i in cfg.state_joint_indices]
        if gripper.arm_joint_index is not None:
            joint_current.append(measured[gripper.arm_joint_index])
        elif gripper.in_state:
            joint_current.append(await gripper.read())
        current = np.asarray(joint_current, dtype=np.float32)

        if check_start:
            # Unconditionally fatal, not gated by stop_on_error: refusing
            # beats slowly moving somewhere nobody asked for, and there is
            # no tick to "skip" before the very first command.
            self._safety.check_start(degrees, current)

        safe = self._safety.apply(degrees, current)

        # Positional writes silently mis-map: `move_to_joint_positions`
        # has no notion of "only these joints", so a dense list built
        # from `safe` alone would land clamped slot i on arm joint i --
        # not arm joint `state_joint_indices[i]` -- scrambling the
        # mapping and voiding the delta clamp whenever the two differ.
        # Building `target` from `measured` and overwriting only the
        # driven slots keeps every value on the joint it was computed
        # for, and holds every other joint at its measured position.
        target = list(measured)
        for slot, joint_idx in enumerate(cfg.state_joint_indices):
            target[joint_idx] = float(safe[slot])
        if gripper.arm_joint_index is not None:
            target[gripper.arm_joint_index] = float(safe[-1])

        # Arm/gripper command failures are unconditionally fatal too: by
        # this point inference already succeeded and safety already
        # cleared the action, so there is no "try again next tick" that
        # would be safe -- the arm itself is reporting the fault.
        #
        # `wait: False` because the next tick supersedes this setpoint: a driver
        # that blocks until the arm physically settles (so-101's default) spends
        # the entire tick budget waiting for a target we are about to replace.
        # Free-form `extra`, so a driver that does not read the key ignores it.
        await arm.move_to_joint_positions(
            JointPositions(values=target), extra={"wait": False}
        )
        if gripper.has_normalized_tail:
            await gripper.write(float(safe[-1]))

    async def _command_pose(self, arm: Any, action: np.ndarray) -> bool:
        """One tick of `action_space="delta-ee"`. Returns False if the arm refused.

        The composition is delegated to `pose.state_compose` rather than
        rewritten here, and that is load-bearing rather than tidiness. Two
        mistakes it forecloses both produce smooth, plausible, wrong motion:
        reading the state's six rotation numbers as matrix *columns* instead
        of rows (on the recorded data the two differ by 0.05-0.06, so neither
        a no-op nor obviously broken), and LEFT-multiplying the delta, which
        applies it in the world frame when the dataset's action is body-frame.

        Composed against the *measured* pose, freshly read, for the same
        reason the joints path clamps against measured joints: an arm that is
        lagging its setpoint must not have deltas accumulate on top of a
        position it never reached.

        A `move_to_position` refusal is bounded rather than fatal, which is
        the one place this path departs from "arm command failures are
        unconditionally fatal". The joints path can only be refused by
        hardware, because its target is the measured position plus a small
        clamped delta and is therefore always inside joint space. This call
        additionally runs the driver's inverse kinematics, which can refuse a
        kinematically unreachable pose or a singularity while the hardware is
        perfectly healthy -- and refuses *before* commanding motion, so the
        arm is stationary and safe. Since the action is relative, skipping the
        tick self-heals: the next tick re-reads the measured pose and composes
        a fresh delta onto it. `_loop` bounds consecutive refusals by
        `starvation_grace_ticks` (4 ticks, 400 ms at 10 fps, by default), so a
        genuinely stuck arm still halts promptly.
        """
        delta = to_working(np.asarray(action, dtype=np.float32), self._cfg.action_units)
        safe = self._safety.apply(delta)

        pose = await arm.get_end_position()
        current = self._pose_state_of(pose)
        target = state_compose(current, safe)
        vector = orientation_vector(state_rotation(target))
        commanded = Pose(
            x=float(target[0]),
            y=float(target[1]),
            z=float(target[2]),
            o_x=vector["o_x"],
            o_y=vector["o_y"],
            o_z=vector["o_z"],
            theta=vector["theta"],
        )

        try:
            await arm.move_to_position(commanded)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            LOGGER.error(
                "arm refused move_to_position for x=%.3f y=%.3f z=%.3f "
                "o=(%.4f, %.4f, %.4f) theta=%.3f deg: %s; holding position and "
                "recomposing from the measured pose next tick",
                commanded.x,
                commanded.y,
                commanded.z,
                commanded.o_x,
                commanded.o_y,
                commanded.o_z,
                commanded.theta,
                exc,
            )
            return False
        return True

    @staticmethod
    def _pose_state_of(pose: Any) -> np.ndarray:
        """The 9-dim state of a live `Pose`, in working units (mm, unitless).

        Deliberately *not* unit-converted, unlike the copy `ObservationBuilder`
        sends the policy: this one is only ever an operand of `state_compose`,
        whose other operand is the already-converted delta, and both are in
        working units by then. Converting here and back would be a round trip
        with nothing on the far side of it.
        """
        return pose_state(
            {
                "pose": {
                    "x": pose.x,
                    "y": pose.y,
                    "z": pose.z,
                    "o_x": pose.o_x,
                    "o_y": pose.o_y,
                    "o_z": pose.o_z,
                    "theta": pose.theta,
                }
            }
        )

    async def _check_end_position(self, arm: Any) -> None:
        """Refuse before any motion if the arm cannot supply an `EndPosition`.

        The delta-EE counterpart of `_check_joint_indices`: config parsing has
        no access to the arm, so this is the first point in the lifecycle that
        can find out whether the configured arm implements the one read this
        action space depends on. Doing the full decode (not merely calling the
        method) means a driver returning a degenerate orientation vector also
        fails here rather than on the first tick.
        """
        try:
            pose = await arm.get_end_position()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"action_space={DELTA_EE!r} needs the arm's end position, but "
                f"get_end_position() failed: {exc}"
            ) from exc
        try:
            self._pose_state_of(pose)
        except (AttributeError, ValueError) as exc:
            raise RuntimeError(
                f"action_space={DELTA_EE!r} could not read a pose from the arm's "
                f"get_end_position() result {pose!r}: {exc}"
            ) from exc

    async def close(self) -> None:
        await self._stop()
