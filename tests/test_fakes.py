from datetime import timedelta

import pytest
from grpclib.const import Status
from grpclib.exceptions import GRPCError
from viam.components.arm import Arm

from tests.fakes import FakeArm, UnstreamableArm


def _point(seconds, positions):
    return Arm.TrajectoryPoint(time=timedelta(seconds=seconds), positions=positions)


async def _batches(*batches):
    for batch in batches:
        yield batch


async def test_streams_all_points_in_order_one_update_per_batch():
    p1 = _point(0, [1, 2])
    p2 = _point(0.01, [3, 4])
    p3 = _point(0.02, [5, 6])
    arm = FakeArm()
    updates = [u async for u in arm.move_through_joint_positions_streamed(_batches([p1], [p2, p3]), extra={"k": "v"})]

    assert arm.stream_points == [p1, p2, p3]
    assert len(updates) == 2
    assert arm.stream_extra == {"k": "v"}
    assert arm.stream_closed is True


async def test_stream_closed_only_flips_after_exhaustion():
    arm = FakeArm()
    gen = arm.move_through_joint_positions_streamed(_batches([_point(0, [1, 2])]))
    await gen.__anext__()
    assert arm.stream_closed is False
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    assert arm.stream_closed is True


async def test_fail_stream_after_points_raises_grpc_aborted_mid_iteration():
    arm = FakeArm()
    arm.fail_stream_after_points = 2
    gen = arm.move_through_joint_positions_streamed(_batches([_point(0, [1, 2])], [_point(0.01, [3, 4])]))
    await gen.__anext__()  # first batch of 1 point: 1 < 2, yields fine
    with pytest.raises(GRPCError, match="arm fault") as exc_info:
        await gen.__anext__()  # second batch pushes point count to 2
    assert exc_info.value.status == Status.ABORTED


async def test_unstreamable_arm_raises_grpc_unimplemented_on_first_iteration():
    arm = UnstreamableArm()
    gen = arm.move_through_joint_positions_streamed(_batches([_point(0, [1, 2])]))
    with pytest.raises(GRPCError) as exc_info:
        await gen.__anext__()
    assert exc_info.value.status == Status.UNIMPLEMENTED


async def test_first_point_must_have_zero_time():
    arm = FakeArm()
    gen = arm.move_through_joint_positions_streamed(_batches([_point(0.01, [1, 2])]))
    with pytest.raises(AssertionError, match=r"point 0"):
        await gen.__anext__()


async def test_point_time_must_strictly_increase_across_batches():
    arm = FakeArm()
    gen = arm.move_through_joint_positions_streamed(_batches([_point(0, [1, 2])], [_point(0, [3, 4])]))
    await gen.__anext__()
    with pytest.raises(AssertionError, match=r"point 1"):
        await gen.__anext__()
