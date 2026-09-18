"""Linear interpolation between successive policy actions, and `ArmStream`,
the streamed arm write built on top of it.

Densifies one policy action per control tick into the higher setpoint cadence
`move_through_joint_positions_streamed` servo mode wants, without the policy
itself running any faster.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import Any, Callable, Sequence

from grpclib.const import Status
from grpclib.exceptions import GRPCError
from viam.components.arm import Arm

LOGGER = logging.getLogger(__name__)


def interpolate(prev: Sequence[float], target: Sequence[float], n: int) -> list[list[float]]:
    """Return `n` points linearly spaced from `prev` (exclusive) to `target` (inclusive).

    Point k (1-indexed) is `prev + (target - prev) * k / n`. The last point is
    `target` itself, computed via `list(target)` rather than arithmetic so no
    floating-point rounding drift reaches the arm.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if len(prev) != len(target):
        raise ValueError(f"prev and target must be the same length, got {len(prev)} and {len(target)}")

    points = []
    for k in range(1, n):
        points.append([p + (t - p) * k / n for p, t in zip(prev, target)])
    points.append(list(target))
    return points


_SENTINEL = object()

# The xArm driver waits for the arm to physically stop moving after the
# batches iterator half-closes, before the streamed RPC itself ends -- this
# has to be long enough to cover that, not just network latency.
_CLOSE_TIMEOUT_S = 5.0


class ArmStream:
    """Feeds `Arm.move_through_joint_positions_streamed` with densified setpoints.

    Driver contract this is built around (xArm): each `TrajectoryPoint` is one
    servo setpoint sent at its stamped time, no interpolation; a past-due point
    is sent immediately; the first point must be `time == 0` at rest; times
    strictly increase across the whole stream; half-closing the batches
    iterator ends the trajectory after the arm stops. Which point the driver
    anchors its clock on is version-dependent, so the rest point ships in the
    same batch as the first tick's motion points and `_t0` is taken there.
    """

    def __init__(
        self,
        arm: Any,
        *,
        fps: float,
        stream_hz: float,
        extra: dict[str, Any],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._arm = arm
        self._fps = fps
        self._stream_hz = stream_hz
        self._extra = extra
        self._clock = clock
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._fault: Exception | None = None
        self._measured: list[float] | None = None
        self._last: list[float] | None = None
        self._last_time = 0.0
        self._t0: float | None = None
        self.points_sent = 0

    async def start(self, measured: list[float]) -> None:
        """Open the RPC; nothing is sent until the first `send()`, which carries the rest point."""
        self._measured = list(measured)
        self._queue = asyncio.Queue(maxsize=4)

        async def batches():
            while True:
                item = await self._queue.get()
                if item is _SENTINEL:
                    return
                yield item

        updates = self._arm.move_through_joint_positions_streamed(batches(), extra=self._extra)

        async def consume() -> None:
            try:
                async for _ in updates:
                    pass
            except Exception as exc:  # noqa: BLE001
                self._fault = exc

        self._task = asyncio.create_task(consume())

    async def send(self, target: Sequence[float]) -> None:
        n = max(1, round(self._stream_hz / self._fps))
        # Tile the tick evenly: 1/stream_hz gives 10, 10, 13.3 ms gaps at 100/30.
        step = 1.0 / (n * self._fps)
        lead = 1.0 / self._fps

        batch: list[Any] = []
        first_send = self._t0 is None
        if first_send:
            self._t0 = self._clock()  # see the class docstring on anchoring
            batch.append(Arm.TrajectoryPoint(time=timedelta(0), positions=list(self._measured)))

        elapsed = self._clock() - self._t0
        prev = self._measured if first_send else self._last

        last_time = self._last_time
        for k, p in enumerate(interpolate(prev, target, n)):
            # Stamp one tick ahead (a past-due point is sent immediately, and a
            # stream of past-due points collapses into bursts at fps); the max()
            # keeps times strictly increasing under a stalled or rewound clock.
            t = max(elapsed + lead + (k + 1) * step, last_time + step)
            batch.append(Arm.TrajectoryPoint(time=timedelta(seconds=t), positions=p))
            last_time = t

        # Bounded: a stalled transport must fail the tick, not back up and replay
        # as a burst. str(QueueFull()) is '', hence the re-raise with a message.
        try:
            self._queue.put_nowait(batch)
        except asyncio.QueueFull:
            raise RuntimeError(
                f"arm {self._arm.name!r} is not draining the trajectory stream "
                f"({self._queue.maxsize} batches queued); the driver or the "
                "transport has stalled"
            ) from None
        self.points_sent += len(batch)
        self._last = list(target)
        self._last_time = last_time

    def check(self) -> None:
        if self._fault is not None:
            fault, self._fault = self._fault, None
            if isinstance(fault, GRPCError) and fault.status is Status.UNIMPLEMENTED:
                raise RuntimeError(
                    f"arm {self._arm.name!r} does not implement "
                    'move_through_joint_positions_streamed; use arm_write: "setpoint"'
                ) from fault
            raise fault
        if self._task is not None and self._task.done():
            raise RuntimeError("arm closed the trajectory stream")

    async def close(self) -> None:
        if self._queue is None or self._task is None:
            return
        if self._task.done():
            if self._fault is not None:
                # About to be discarded by the shutdown this call is part
                # of; it must not vanish silently if nobody called check().
                LOGGER.error("stream closed with an unread fault: %s", self._fault)
                self._fault = None
            return
        try:
            self._queue.put_nowait(_SENTINEL)
            await asyncio.wait_for(self._task, _CLOSE_TIMEOUT_S)
            return
        except (asyncio.QueueFull, asyncio.TimeoutError):
            # Full queue: the stall send() already reported, nothing is draining,
            # a blocking put would hang. Timeout: the driver never finished its
            # motion-stop wait. Either way cancel, and let teardown finish before
            # the caller stops the arm.
            pass
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
            return
        try:
            await asyncio.wait_for(self._task, _CLOSE_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                # Let teardown finish before the caller stops the arm.
                await self._task
            except asyncio.CancelledError:
                pass
