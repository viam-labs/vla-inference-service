"""action_space="ee-pose": absolute end-effector poses for one or more limbs.

Row layout, state and action alike, with L limbs in config order:

    [limb 0: x y z r00 r01 r02 r10 r11 r12] ... [limb L-1: same] [grip 0 ... grip L-1]

Each pose is the limb's `frame` component in the world frame: position in mm,
then the first two ROWS of its rotation matrix (pose.py's delta-ee state
layout). Each grip is normalized closure, 0 = open (gripper.py's convention).
The grips go last so the reference buffer interpolates the 9L pose columns and
holds the grips, with no change to the buffer.

Poses stream through the builtin motion service's teleop DoCommands: one
pipeline holds every limb, so RDK plans the limbs jointly and checks them
against each other. teleop plays the newest pose it was sent, so a tick that
arrives while RDK is still planning supersedes the previous one rather than
queueing behind it.

The per-tick clamp measures each target against the pose last *commanded*,
not the measured one: teleop trails the stream by a tick or two, and clamping
against a lagging measurement would throttle every tick and let the plan run
ahead of the arm. The first command is the exception: it is checked against
the measured pose and refused outright past `max_start_delta_mm`, as joints
mode refuses a first target past `max_start_delta_degs`. That check bounds
translation only; rotation is clamped per tick from the measured pose onward.

Clamping against the commanded pose leaves it free to run away from the arm:
RDK's teleop planner swallows its own errors (unreachable, IK, collision) and
`teleop_move` still returns true, so while planning fails the commanded pose
walks off, and once the target is reachable again teleop jumps to it in one
unlimited move. So every later tick first compares the previous commanded pose
with the measured one and raises `SafetyError` once it leads by more than
`max_lead_mm`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

import numpy as np
from google.protobuf import json_format
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import MoveRequest

from .config import POSE_DIM, LimbConfig
from .gripper import GripperAdapter
from .observation import pose_state_from_proto
from .pose import orientation_vector, state_compose, state_delta, state_rotation
from .safety import CartesianSafetyLayer, SafetyError

WORLD = "world"


def pose_proto(state: np.ndarray) -> Pose:
    """A 9-vector pose state (mm, rotation rows) as a Viam `Pose`."""
    vector = orientation_vector(state_rotation(state))
    return Pose(
        x=float(state[0]), y=float(state[1]), z=float(state[2]),
        o_x=vector["o_x"], o_y=vector["o_y"], o_z=vector["o_z"], theta=vector["theta"],
    )


async def _bounded(aw: Any, timeout_s: float, what: str) -> Any:
    try:
        return await asyncio.wait_for(aw, timeout_s)
    except asyncio.TimeoutError:
        raise TimeoutError(f"{what} did not return within {timeout_s}s") from None


class Limbs:
    def __init__(
        self,
        limbs: Sequence[LimbConfig],
        *,
        motion: Any,
        motion_name: str,
        arms: Mapping[str, Any],
        grippers: Sequence[GripperAdapter],
        safety: CartesianSafetyLayer,
        start_limit_mm: float,
        lead_limit_mm: float,
        rpc_timeout_s: float,
    ) -> None:
        if len(grippers) != len(limbs):
            raise ValueError("one gripper adapter per limb")
        self._limbs = list(limbs)
        self._motion = motion
        self._motion_name = motion_name
        self._arms = dict(arms)
        self._grippers = list(grippers)
        self._safety = safety
        self._start_limit_mm = start_limit_mm
        self._lead_limit_mm = lead_limit_mm
        self._rpc_s = rpc_timeout_s
        self._last: list[np.ndarray] | None = None  # last commanded pose per limb
        self._sent: np.ndarray | None = None
        self._started = False

    @property
    def last_sent(self) -> np.ndarray | None:
        """The clamped row the last `command` streamed (poses then grips)."""
        return self._sent

    @property
    def width(self) -> int:
        return (POSE_DIM + 1) * len(self._limbs)

    async def _poses(self) -> list[np.ndarray]:
        pifs = await asyncio.gather(
            *(_bounded(self._motion.get_pose(l.frame, WORLD), self._rpc_s, f"motion.get_pose({l.frame!r})")
              for l in self._limbs)
        )
        return [pose_state_from_proto(p.pose) for p in pifs]

    async def read_state(self) -> np.ndarray:
        poses, grips = await asyncio.gather(
            self._poses(),
            asyncio.gather(*(_bounded(g.read(), self._rpc_s, "gripper read") for g in self._grippers)),
        )
        return np.concatenate([*poses, np.asarray(grips, dtype=np.float64)])

    async def start(self) -> None:
        """Clear a pipeline an aborted run left behind, then register every limb at its measured pose."""
        await _bounded(self._motion.do_command({"teleop_stop": True}), self._rpc_s, "teleop_stop")
        for limb, pose in zip(self._limbs, await self._poses()):
            request = MoveRequest(
                name=self._motion_name,
                component_name=limb.frame,
                destination=PoseInFrame(reference_frame=WORLD, pose=pose_proto(pose)),
            )
            await _bounded(
                self._motion.do_command({"teleop_start": json_format.MessageToJson(request)}),
                self._rpc_s, f"teleop_start({limb.frame!r})",
            )
        self._last = None
        self._started = True

    async def command(self, action: np.ndarray, *, first: bool) -> bool:
        """Clamp and stream one row. Returns whether any limb's step was clamped."""
        row = np.asarray(action, dtype=np.float64)
        if row.shape != (self.width,):
            raise SafetyError(f"an ee-pose action must have {self.width} dimensions, got shape {row.shape}")
        if not np.all(np.isfinite(row)):
            raise SafetyError(f"action must be finite (no NaN/inf); got {row}")
        n = len(self._limbs)
        targets = [row[POSE_DIM * i : POSE_DIM * (i + 1)] for i in range(n)]
        raw_grips = row[POSE_DIM * n :]
        grips = np.clip(raw_grips, 0.0, 1.0)
        grips_clipped = bool(np.any(grips != raw_grips))
        if grips_clipped:
            self._safety.clamp_counts["gripper"] += 1

        if first or self._last is None:
            measured = await self._poses()
            for limb, cur, tgt in zip(self._limbs, measured, targets):
                dist = float(np.linalg.norm(tgt[:3] - cur[:3]))
                if dist > self._start_limit_mm:
                    raise SafetyError(
                        f"first target for limb {limb.name!r} is {dist:.1f} mm from its measured pose, "
                        f"over safety.max_start_delta_mm={self._start_limit_mm}; refusing to start"
                    )
            refs = measured
        else:
            refs = self._last
            for limb, ref, cur in zip(self._limbs, refs, await self._poses()):
                lead = float(np.linalg.norm(ref[:3] - cur[:3]))
                if lead > self._lead_limit_mm:
                    raise SafetyError(
                        f"limb {limb.name!r}: commanded pose is {lead:.1f} mm ahead of the arm, over "
                        f"safety.max_lead_mm={self._lead_limit_mm}; the teleop planner may be failing "
                        "(check the motion service's teleop_status)"
                    )

        counts = self._safety.clamp_counts
        before = sum(counts.values())
        commanded = [state_compose(ref, self._safety.apply(state_delta(ref, tgt))) for ref, tgt in zip(refs, targets)]
        clamped = sum(counts.values()) != before or grips_clipped
        self._last = commanded
        self._sent = np.concatenate([*commanded, grips])

        await asyncio.gather(
            *(
                _bounded(
                    self._motion.do_command({
                        "teleop_move": json_format.MessageToJson(
                            PoseInFrame(reference_frame=WORLD, pose=pose_proto(pose))
                        ),
                        "component_name": limb.frame,
                    }),
                    self._rpc_s, f"teleop_move({limb.frame!r})",
                )
                for limb, pose in zip(self._limbs, commanded)
            ),
            *(_bounded(g.write(float(v)), self._rpc_s, "gripper write") for g, v in zip(self._grippers, grips)),
        )
        return clamped

    async def stop(self) -> None:
        """Best effort: end the teleop pipeline (a bare teleop_stop stops every limb in it), then stop every arm."""
        try:
            await _bounded(self._motion.do_command({"teleop_stop": True}), self._rpc_s, "teleop_stop")
        except Exception:  # noqa: BLE001 - stopping must not raise
            pass
        for arm in self._arms.values():
            try:
                await _bounded(arm.stop(), self._rpc_s, "arm.stop")
            except Exception:  # noqa: BLE001
                pass
        self._started = False
        self._last = None
