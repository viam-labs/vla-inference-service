"""Offline smoothness harness: a virtual-time event loop, scripted fakes, metrics.

Drives the real `VLAController` loop with no wall-clock dependence, so a 20 s
session runs in milliseconds and every number is deterministic. See
docs/superpowers/specs/2026-10-23-seam-and-schedule-design.md, section 1.
"""

from __future__ import annotations

import asyncio
import selectors
import math
import random
from dataclasses import dataclass

import numpy as np
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig

from tests.fakes import FakeArm, FakeCamera
from vla.controller.service import VLAController
from vla.wire import encode_matrix


class _VirtualSelector:
    """Wraps a real selector; an idle `select(timeout)` advances the loop's
    virtual clock by `timeout` instead of blocking for it.

    The real selector is still polled (with timeout 0) so the loop's self-pipe
    and `call_soon_threadsafe` keep working. `select(None)` -- nothing ready
    and no timer pending -- would block forever on a real loop; here it is a
    deadlock in the test, and raises.
    """

    def __init__(self, loop: "VirtualLoop") -> None:
        self._loop = loop
        self._real = selectors.DefaultSelector()

    def select(self, timeout=None):
        events = self._real.select(0)
        if events:
            return events
        if timeout is None:
            raise RuntimeError("virtual loop deadlock: nothing ready and no timer pending")
        self._loop._now += max(0.0, timeout)
        return []

    def __getattr__(self, name):
        return getattr(self._real, name)


class VirtualLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps to the next timer instead of waiting.

    `asyncio.sleep`, `wait_for`, and `call_at` all key off `time()`, so they
    become virtual and event-ordered with no further injection.
    """

    def __init__(self) -> None:
        self._now = 0.0
        super().__init__(selector=_VirtualSelector(self))

    def time(self) -> float:
        return self._now


def run_virtual(coro):
    """Run `coro` to completion on a fresh `VirtualLoop`."""
    with asyncio.Runner(loop_factory=VirtualLoop) as runner:
        return runner.run(coro)


FPS = 10.0
PERIOD = 1.0 / FPS
STREAM_HZ = 100.0
N = round(STREAM_HZ / FPS)
STEP = 1.0 / (N * FPS)
CHUNK = 50
# Joints 0-4 follow the reference trajectory. Joint 5 is a bookkeeping tag:
# chunk_id * 1000 + row, so every executed knot names the chunk it came from.
DIM = 6
TAG = 5
# TrajectoryPoint.time is a timedelta: microsecond resolution, each stamp
# rounded on its own, so a gap between two stamps is only good to ~1-2 us.
STAMP_TOL = 2e-6


def reference(t: float) -> np.ndarray:
    """The smooth target the policy is 'trying' to follow: per-joint sinusoids."""
    phases = np.arange(TAG) * 0.7
    return 5.0 * np.sin(2 * math.pi * t / 4.0 + phases)


class ScriptedPolicy:
    """The policy service's DoCommand surface, with deterministic chunks.

    Row i of a chunk is `reference(obs_t + (i + 1) * PERIOD)`, where obs_t is
    when the request arrived, plus the chunk's offset, so consecutive chunks
    disagree by a known amount at every seam. The first inference takes
    `cold_extra_s` longer: the schedule anchors after it, and without the
    offset every later reply would land exactly on a tick boundary.
    """

    def __init__(self, latency_s: float, offsets=(0.0,), cold_extra_s: float = 0.05):
        self.latency_s = latency_s
        self.offsets = tuple(offsets)
        self.cold_extra_s = cold_extra_s
        self.obs_times: list[float] = []
        self.reply_times: list[float] = []

    async def do_command(self, command, **kwargs):
        name = command.get("command")
        if name == "status":
            return {"state": "ready", "error": ""}
        if name == "specs":
            return {
                "policy_type": "scripted",
                "action_dim": float(DIM),
                "state_dim": float(DIM),
                "n_action_steps": float(CHUNK),
                "input_features": {
                    "observation.images.top": [3.0, 224.0, 224.0],
                    "observation.state": [float(DIM)],
                },
                "output_features": {"action": [float(DIM)]},
                "preprocess_image_size": None,
                "image_feature_keys": ["observation.images.top"],
                "supports_rtc": False,
                "rtc_enabled": False,
                "relative_actions": False,
                "device": "cpu",
            }
        if name == "infer":
            loop = asyncio.get_running_loop()
            chunk_id = len(self.obs_times)
            obs_t = loop.time()
            self.obs_times.append(obs_t)
            await asyncio.sleep(self.latency_s + (self.cold_extra_s if chunk_id == 0 else 0.0))
            offset = self.offsets[chunk_id % len(self.offsets)]
            rows = np.empty((CHUNK, DIM), dtype=np.float32)
            for i in range(CHUNK):
                rows[i, :TAG] = reference(obs_t + (i + 1) * PERIOD) + offset
                rows[i, TAG] = chunk_id * 1000 + i
            self.reply_times.append(loop.time())
            return {
                "actions": encode_matrix(rows),
                "raw_actions": encode_matrix(rows),
                "latency_s": self.latency_s,
            }
        raise ValueError(name)


class TrackingArm(FakeArm):
    """A FakeArm whose position follows the stream, with injectable read delays.

    `knots` records (virtual receive time, last point) per streamed batch,
    which is the policy action executed on that tick. `jitter()` returns the
    virtual seconds that each `get_joint_positions` call stalls for. That is
    the one arm call the stream-mode loop awaits every tick; delaying the
    stream consumer instead would fill ArmStream's bounded queue and fault.
    """

    def __init__(self, jitter=None):
        super().__init__(positions=[0.0] * DIM)
        self.jitter = jitter or (lambda: 0.0)
        self.knots: list[tuple[float, list[float]]] = []
        # (virtual receive time, last point's stamp in seconds) per batch.
        self.batch_stamps: list[tuple[float, float]] = []

    async def get_joint_positions(self, **kwargs):
        delay = self.jitter()
        if delay:
            await asyncio.sleep(delay)
        return await super().get_joint_positions(**kwargs)

    async def move_through_joint_positions_streamed(self, batches, *, extra=None, timeout=None, **kwargs):
        async def recording():
            async for batch in batches:
                now = asyncio.get_running_loop().time()
                self.knots.append((now, list(batch[-1].positions)))
                self.batch_stamps.append((now, batch[-1].time.total_seconds()))
                self.positions = list(batch[-1].positions)
                yield batch

        async for update in super().move_through_joint_positions_streamed(
            recording(), extra=extra, timeout=timeout, **kwargs
        ):
            yield update


def seeded_jitter(seed: int, probability: float = 0.2, max_fraction: float = 0.6):
    """Per-call delays: `probability` of calls stall 0.3-`max_fraction` periods."""
    rng = random.Random(seed)

    def jitter() -> float:
        if rng.random() < probability:
            return rng.uniform(0.3, max_fraction) * PERIOD
        return 0.0

    return jitter


def _config(**overrides) -> ServiceConfig:
    attrs = {
        "policy_service": "p",
        "arm": "a",
        "cameras": {"observation.images.top": "cam"},
        "state_joint_indices": list(range(DIM)),
        "fps": FPS,
        "mode": "async",
        "merge": "aligned",
        "arm_write": "stream",
        "stream_hz": STREAM_HZ,
        # Starvation is measured here, not fatal.
        "starvation_grace_ticks": 100000,
        # Never clamp: the tag channel jumps by 1000 between chunks by design.
        "safety": {"max_start_delta_degs": 1e6, "max_vel_degs_per_sec": 1e9},
    }
    attrs.update(overrides)
    s = Struct()
    s.update(attrs)
    return ServiceConfig(
        name="c", api="rdk:service:generic", model="viam-labs:vla:controller", attributes=s
    )


@dataclass
class Session:
    arm: TrackingArm
    policy: ScriptedPolicy
    status: dict


def run_session(*, latency_s, duration_s=20.0, jitter=None, offsets=(0.0,), **config) -> Session:
    """Run the real controller for `duration_s` virtual seconds and return what it did."""

    async def main() -> Session:
        loop = asyncio.get_running_loop()
        arm = TrackingArm(jitter=jitter)
        policy = ScriptedPolicy(latency_s, offsets)
        svc = VLAController("c")
        svc._clock = loop.time
        svc.reconfigure(_config(**config), {"p": policy, "a": arm, "cam": FakeCamera()})
        await svc.do_command({"command": "start", "task": "t"})
        await asyncio.sleep(duration_s)
        status = await svc.do_command({"command": "status"})
        await svc.do_command({"command": "stop"})
        return Session(arm=arm, policy=policy, status=status)

    return run_virtual(main())


# --- metrics -----------------------------------------------------------------


def _gaps(points) -> np.ndarray:
    # Skip the rest point: its gap to the first motion point is the lead, not a
    # stream gap.
    times = np.array([p.time.total_seconds() for p in points[1:]])
    return np.diff(times)


def gap_jitter(points) -> float:
    """Max |gap - STEP| over gaps no longer than one period."""
    d = _gaps(points)
    short = d[d <= PERIOD]
    return float(np.max(np.abs(short - STEP))) if short.size else 0.0


def long_gaps(points) -> int:
    """Gaps longer than a period: holds (starvation or re-anchor), not jitter."""
    return int(np.sum(_gaps(points) > PERIOD))


def lead_error(arm: TrackingArm) -> float:
    """Max deviation of each batch's last stamp from exactly one tick of lead.

    A batch received at virtual time r (relative to the first) carries its
    tick's last point, which should be stamped r + lead + N * STEP, where
    lead is one period. Only meaningful without injected jitter, where the
    receive time is the tick's nominal start.
    """
    r0 = arm.batch_stamps[0][0]
    want = PERIOD + N * STEP
    return max(abs(stamp - (r - r0) - want) for r, stamp in arm.batch_stamps)


def _landing_knots(session: Session, skip: int = 1) -> list[int]:
    """Index of the first knot executed at or after each background landing.

    `skip=1` leaves out the blocking fill, which is not a landing.
    """
    times = np.array([t for t, _ in session.arm.knots])
    out = []
    for reply in session.policy.reply_times[skip:]:
        i = int(np.searchsorted(times, reply))
        if i < len(times):
            out.append(i)
    return out


def landing_obs_age(session: Session) -> float:
    """Max over landings of (exec time - observation time) for the first knot
    after the landing. Observation age, not the README's 'staleness'."""
    ages = []
    for i in _landing_knots(session):
        t, positions = session.arm.knots[i]
        chunk_id = int(round(positions[TAG])) // 1000
        ages.append(t - session.policy.obs_times[chunk_id])
    return max(ages)


def seam_jump(session: Session, window: int = 5) -> float:
    """Max per-joint |second difference| (deg) of executed knots, joints 0-4,
    over the knots from 1 before to `window` after each landing.

    The first background landing is left out too: it supersedes the blocking
    fill, whose rows are offset by the whole frozen first inference, a ~10
    degree jump that would swamp the offset seam being measured.
    """
    pos = np.array([p for _, p in session.arm.knots])[:, :TAG]
    acc = np.abs(np.diff(pos, n=2, axis=0)).max(axis=1)  # acc[m] is centred on knot m+1
    worst = 0.0
    for i in _landing_knots(session, skip=2):
        lo, hi = max(0, i - 2), min(len(acc), i + window)
        if lo < hi:
            worst = max(worst, float(acc[lo:hi].max()))
    return worst


def knot_accel(session: Session) -> float:
    """Max |second difference| over all knots. Baseline for cubic densification (#4)."""
    pos = np.array([p for _, p in session.arm.knots])[:, :TAG]
    return float(np.abs(np.diff(pos, n=2, axis=0)).max())
