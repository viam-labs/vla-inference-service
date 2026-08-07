"""Real checkpoint, fake robot: load -> preprocess -> infer -> postprocess -> actuate.

This is the spec's phase-1 gate: it proves the whole pipeline end to end with
no hardware, using a real `lerobot/smolvla_base` checkpoint driving a fake
arm and fake cameras.

Measured on this machine (Apple Silicon, MPS): `predict_chunk` takes ~5.3s,
and `n_action_steps` is 50, so at `fps: 10` a chunk buys only 5.0s of
motion -- inference is *slower* than the motion it produces, and a
sequential loop stalls between chunks. `fps: 2.0` here (and a generous
`asyncio.sleep`) gives the loop enough wall-clock time to actually tick
instead of asserting on a loop that never got anywhere.

Actions will be meaningless: smolvla_base was never trained on this fake
arm's joint layout or this task string. That is expected -- this test proves
the *plumbing* (finite numbers flowing end to end), not the behavior.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig

pytestmark = pytest.mark.integration

from tests.fakes import FakeArm, FakeCamera
from vla.controller.service import VLAController
from vla.policy.service import VLAPolicy

CHECKPOINT = "lerobot/smolvla_base"


def _cfg(name: str, model: str, attrs: dict) -> ServiceConfig:
    s = Struct()
    s.update(attrs)
    return ServiceConfig(name=name, api="rdk:service:generic", model=model, attributes=s)


@pytest.fixture(scope="module")
async def policy():
    svc = VLAPolicy("p")
    svc.reconfigure(_cfg("p", "viam-labs:vla:policy", {"model_hub_id": CHECKPOINT}), {})
    await svc.await_ready()
    return svc


async def test_policy_reports_ready_and_specs(policy):
    assert (await policy.do_command({"command": "status"}))["state"] == "ready"
    specs = await policy.do_command({"command": "specs"})
    assert specs["action_dim"] > 0
    assert specs["n_action_steps"] > 0


async def test_full_loop_drives_the_fake_arm(policy):
    specs = await policy.do_command({"command": "specs"})
    dim = int(specs["action_dim"])

    # One extra joint on the fake arm beyond what the policy drives, to prove
    # state_joint_indices selects rather than assumes a dense 0..N-1 layout.
    arm = FakeArm(positions=[0.0] * (dim + 1))
    camera = FakeCamera()
    controller = VLAController("c")
    controller.reconfigure(
        _cfg("c", "viam-labs:vla:controller", {
            "policy_service": "p",
            "arm": "a",
            # smolvla_base wants three cameras; one fake camera serves all
            # three feeds -- the controller only cares that every key the
            # policy declared is mapped to *some* camera resource.
            "cameras": {k: "cam" for k in specs["image_feature_keys"]},
            "state_joint_indices": list(range(dim)),
            "fps": 2.0,
            "task": "pick up the red block",
            # A real policy's first action against an arbitrary fake pose is
            # arbitrary -- generous limits so the safety layer's start-delta
            # refusal doesn't fail this plumbing test for an unrelated reason.
            "safety": {"max_start_delta_degs": 10000.0, "max_joint_delta_degs": 10000.0},
        }),
        {"p": policy, "a": arm, "cam": camera},
    )

    await controller.do_command({"command": "start"})
    await asyncio.sleep(15)
    status = await controller.do_command({"command": "status"})
    await controller.do_command({"command": "stop"})

    assert status["state"] == "running", status["last_error"]
    assert len(arm.moves) >= 1
    assert status["avg_latency_s"] > 0
    for positions in arm.moves:
        assert np.all(np.isfinite(positions.values))
