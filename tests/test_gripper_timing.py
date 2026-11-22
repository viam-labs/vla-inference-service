"""A `do_command` gripper write fires when the arm reaches its row, not a lead early.

Scenario tests are plain sync functions driving a `VirtualLoop`; see
tests/test_smoothness.py for why.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

import tests.smoothness as sm
from tests.fakes import FakeCamera, FakeDoCommandGripper
from vla.controller.service import VLAController
from vla.wire import decode_matrix, encode_matrix

T_CLOSE = 6.0  # wall time at which the policy's chunk says "closed"


class _TimedGripper(FakeDoCommandGripper):
    def __init__(self):
        super().__init__(0.0)
        self.writes: list[tuple[float, float]] = []

    async def do_command(self, command, **kw):
        if "set" in command:
            self.writes.append((asyncio.get_running_loop().time(), command["set"]))
        return await super().do_command(command, **kw)


def _fire_vs_row_time(arm_write: str, stream_lead_s: float, *, stop_early: bool = False):
    """Returns (time the gripper first closed, plan time of the first closed row)."""
    rows_closed: list[tuple[float, float]] = []  # (reply time, first closed row's time)

    class GripperPolicy(sm.ScriptedPolicy):
        async def do_command(self, command, **kw):
            out = await super().do_command(command, **kw)
            if command.get("command") == "infer" and "compute_s" in out:
                rows = decode_matrix(out["actions"]).copy()
                times = self.obs_times[-1] + (np.arange(rows.shape[0]) + 1) * self.period
                rows[:, sm.TAG] = (times >= T_CLOSE).astype(np.float32)
                closed = times[times >= T_CLOSE]
                if len(closed):
                    rows_closed.append((asyncio.get_running_loop().time(), float(closed[0])))
                out["actions"] = encode_matrix(rows.astype(np.float32))
            return out

    async def main():
        loop = asyncio.get_running_loop()
        arm = sm.TrackingArm()
        arm.positions = [0.0] * 5
        gripper = _TimedGripper()
        svc = VLAController("c")
        svc._clock = loop.time
        cfg = sm._config(
            state_joint_indices=list(range(5)),
            gripper={"type": "do_command", "name": "g", "open_value": 0.0, "closed_value": 1.0},
            arm_write=arm_write,
            **({"stream_lead_s": stream_lead_s} if arm_write == "stream" else {}),
        )
        if arm_write != "stream":
            del cfg.attributes["stream_hz"]
        svc.reconfigure(
            cfg, {"p": GripperPolicy(0.3), "a": arm, "cam": FakeCamera(), "g": gripper}
        )
        await svc.do_command({"command": "start", "task": "t"})
        if stop_early:
            # Run until a close is queued but not yet due, then stop.
            await asyncio.sleep(T_CLOSE - 0.15)
            await svc.do_command({"command": "stop"})
            assert not svc._gripper_queue
            await asyncio.sleep(3.0)
            return gripper, None
        await asyncio.sleep(10.0)
        await svc.do_command({"command": "stop"})
        return gripper, svc

    async def inline(fn, /, *a, **k):
        return fn(*a, **k)

    with patch("vla.controller.observation.to_thread", inline):
        gripper, _ = sm.run_virtual(main())
    if stop_early:
        return gripper, None
    fire = next(t for t, v in gripper.writes if v >= 0.5)
    row_t = [rt for landed, rt in rows_closed if landed <= fire][-1]
    return fire, row_t


@pytest.mark.parametrize(
    "arm_write, stream_lead_s", [("stream", 0.0), ("stream", 0.1), ("setpoint", 0.0)]
)
def test_gripper_write_lands_at_its_rows_time(arm_write, stream_lead_s):
    fire, row_t = _fire_vs_row_time(arm_write, stream_lead_s)
    # Before the fix this was a whole sample lead early: 0.2 s (stream),
    # 0.2 s (stream, stream_lead_s=0.1), 0.1 s (setpoint).
    # Never early (half a tick of slack for float rounding), at most a tick late.
    assert -sm.PERIOD / 2 <= fire - row_t <= sm.PERIOD + 1e-6, (fire, row_t)


def test_no_gripper_write_fires_after_stop():
    gripper, _ = _fire_vs_row_time("stream", 0.0, stop_early=True)
    assert all(v < 0.5 for _, v in gripper.writes), gripper.writes


def test_sequential_regime_gripper_write_fires_at_its_row_not_when_the_next_chunk_lands(monkeypatch):
    """20-row chunks (2.0 s) with 2.6 s latency: the loop idles between chunks, and a
    close in a chunk's last rows used to wait there until the next chunk landed."""
    monkeypatch.setattr(sm, "CHUNK", 20)
    CLOSE_CHUNK, FIRST_CLOSED_ROW = 4, 19
    POL: list = []

    class SeqPolicy(sm.ScriptedPolicy):
        async def do_command(self, command, **kw):
            out = await super().do_command(command, **kw)
            if command.get("command") == "infer" and "compute_s" in out:
                rows = decode_matrix(out["actions"]).copy()
                rows[:, sm.TAG] = 0.0
                if len(self.obs_times) == CLOSE_CHUNK + 1:  # warm-up is not recorded
                    rows[FIRST_CLOSED_ROW:, sm.TAG] = 1.0
                out["actions"] = encode_matrix(rows.astype(np.float32))
            return out

    async def main():
        loop = asyncio.get_running_loop()
        arm = sm.TrackingArm()
        arm.positions = [0.0] * 5
        gripper = _TimedGripper()
        svc = VLAController("c")
        svc._clock = loop.time
        cfg = sm._config(
            state_joint_indices=list(range(5)),
            gripper={"type": "do_command", "name": "g", "open_value": 0.0, "closed_value": 1.0},
            smooth_rows=0,
        )
        svc.reconfigure(
            cfg, {"p": POL.append(SeqPolicy(2.6)) or POL[0], "a": arm, "cam": FakeCamera(), "g": gripper}
        )
        svc._cfg = replace(svc._cfg, blend_s=0.0)
        await svc.do_command({"command": "start", "task": "t"})
        await asyncio.sleep(30.0)
        status = await svc.do_command({"command": "status"})
        await svc.do_command({"command": "stop"})
        return gripper, status, svc

    async def inline(fn, /, *a, **k):
        return fn(*a, **k)

    with patch("vla.controller.observation.to_thread", inline):
        gripper, status, svc = sm.run_virtual(main())
    # The chunk lands after its own rows are due, so the loop executes it from the
    # landing on: its rows' sample times are the trace entries past the landing.
    landing = POL[0].reply_times[CLOSE_CHUNK]
    sample_ts = [r[1] for r in svc._trace if r[1] > landing]
    closed_t = sample_ts[FIRST_CLOSED_ROW]
    assert status["regime"] == "sequential"
    fire = next(t for t, v in gripper.writes if v >= 0.5)
    # Before the fix the write waited for the next chunk, about 2.6 s late.
    assert -sm.PERIOD / 2 <= fire - closed_t <= sm.PERIOD + 1e-6, (fire, closed_t)
