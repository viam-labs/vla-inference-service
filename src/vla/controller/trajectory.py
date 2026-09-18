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

    The xArm driver's streamed RPC sends each `TrajectoryPoint` as an
    absolute-time servo setpoint, with no interpolation of its own -- a point
    that arrives past its stamped time is sent immediately, so a caller that
    wants a higher servo-mode cadence out of slower policy actions has to
    stamp ahead and densify itself (see `interpolate`). The stream's first
    point must land at `time == timedelta(0)` with the arm at rest there, and
    every point after it must be stamped strictly later than the last, across
    the whole stream, not just within one batch.

    The rest point ships in the *same batch* as the first tick's motion
    points, not on its own: which point a driver anchors its wall clock to at
    stream start is driver-version-dependent (some anchor on the t=0 point's
    arrival, some on the first motion point's), so keeping both in one batch
    keeps either anchor within one transport hop of `_t0`. A driver without
    the RPC at all is discovered on `check()`'s first call, before any
    motion -- the stream is opened in `start()`, but nothing is sent until
    the first `send()`. Half-closing the batches iterator (`close()`) is how
    the driver is told the trajectory is over; it waits for the arm to stop
    moving before the RPC itself ends.
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
        """Open the RPC and start the background consumer. Sends nothing.

        The rest point at `time=0` goes out with the first tick's motion
        batch, in `send()` -- see the class docstring for why. `start()`
        does not wait for any acknowledgment either: a driver without the
        RPC is discovered on the first `check()` call instead.
        """
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
        # Ties the spacing to a whole number of points per tick: 1/stream_hz
        # would tile the tick unevenly whenever stream_hz isn't an exact
        # multiple of fps (e.g. 100/30 -> gaps of 10, 10, 13.3ms). Rounding
        # n first and deriving step from it makes every gap in the whole
        # stream equal, at whatever rate n/fps actually works out to (100 at
        # fps=30 runs at 90 Hz, not 100).
        step = 1.0 / (n * self._fps)
        lead = 1.0 / self._fps

        batch: list[Any] = []
        first_send = self._t0 is None
        if first_send:
            # The trajectory clock is anchored to the first *motion* batch,
            # not to start() -- see the class docstring.
            self._t0 = self._clock()
            batch.append(Arm.TrajectoryPoint(time=timedelta(0), positions=list(self._measured)))

        elapsed = self._clock() - self._t0
        prev = self._measured if first_send else self._last

        last_time = self._last_time
        for k, p in enumerate(interpolate(prev, target, n)):
            # One-tick lead: the driver sends a past-due point immediately,
            # and a stream where every point is past due degenerates back
            # into bursts at fps, so every point is stamped ahead of when it
            # is due. The monotonic guard covers a stalled or rewound clock:
            # strict monotonicity is an SDK invariant.
            t = max(elapsed + lead + (k + 1) * step, last_time + step)
            batch.append(Arm.TrajectoryPoint(time=timedelta(seconds=t), positions=p))
            last_time = t

        # Bounded and non-blocking: a stalled transport must surface as a
        # tick failure, not silently back up and replay as a burst once it
        # frees up. str(QueueFull()) is '', so re-raise with a message --
        # otherwise this reaches status.last_error empty.
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
        except asyncio.QueueFull:
            # The same stall send() already turned into a RuntimeError: the
            # queue is stuck full and nothing is draining it, so a blocking
            # put here would hang forever waiting for a graceful half-close
            # that is never coming. Go straight to cancellation instead.
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
