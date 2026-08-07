"""Minimal fakes standing in for Viam resources.

These are duck-typed stand-ins, not subclasses of the real SDK component
classes -- Tasks 12, 14, and 17 all build on them. Faithfulness matters more
than convenience here: a fake that is too permissive lets a real bug (wrong
units, wrong joint count, an unimplemented SDK method) pass a test that
should have caught it.
"""

from __future__ import annotations

import numpy as np


class FakeArm:
    def __init__(self, positions=None):
        self.positions = list(positions or [0.0] * 6)
        self.moves = []
        self.stopped = 0
        self.fail_next_move = False

    async def get_joint_positions(self, **kwargs):
        from viam.proto.component.arm import JointPositions

        return JointPositions(values=self.positions)

    async def move_through_joint_positions(self, positions, options=None, **kwargs):
        if self.fail_next_move:
            raise RuntimeError("arm move failed")
        self.moves.append((positions, options))
        # Write into the existing vector rather than replacing it: a commanded
        # chunk can be shorter than the arm's joint count (gripper on its own
        # component), and replacing would silently shrink the arm.
        commanded = list(positions[-1].values)
        self.positions[: len(commanded)] = commanded

    async def stop(self, **kwargs):
        self.stopped += 1


class FakeCamera:
    def __init__(self, size=(480, 640), fail=False):
        self.size = size
        self.fail = fail
        self.reads = 0

    async def get_image(self, *args, **kwargs):
        from viam.media.video import ViamImage, CameraMimeType
        import io
        from PIL import Image

        self.reads += 1
        if self.fail:
            raise RuntimeError("camera read failed")
        h, w = self.size
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        return ViamImage(buf.getvalue(), CameraMimeType.JPEG)


class FakeServo:
    def __init__(self, angle=0):
        self.angle = angle
        self.moves = []

    async def get_position(self, **kwargs) -> int:
        return self.angle

    async def move(self, angle: int, **kwargs):
        self.moves.append(angle)
        self.angle = angle


class FakeGripper:
    def __init__(self, inputs=None, supports_inputs=True):
        self.inputs = list(inputs or [0.0])
        self.supports_inputs = supports_inputs
        self.opened = 0
        self.grabbed = 0
        self.sent = []

    async def get_current_inputs(self, **kwargs):
        return list(self.inputs)

    async def go_to_inputs(self, values, **kwargs):
        if not self.supports_inputs:
            raise NotImplementedError("go_to_inputs unimplemented")
        self.sent.append(list(values))
        self.inputs = list(values)

    async def open(self, **kwargs):
        self.opened += 1

    async def grab(self, **kwargs):
        self.grabbed += 1
        return True
