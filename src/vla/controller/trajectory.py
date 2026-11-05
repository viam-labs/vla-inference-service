"""Linear and cubic densification between successive policy actions, and `ArmStream`,
the streamed arm write built on top of them.

Densifies one policy action per control tick into the higher setpoint cadence
`move_through_joint_positions_streamed` servo mode wants, without the policy
itself running any faster.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import timedelta
from typing import Any, Sequence

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


# Fritsch-Carlson radius: tangents (as multiples of the chord) are kept inside
# a circle of this radius, which keeps every point between the two knots.
# ponytail: fixed at 3; drop to 2 to lower peak point speed if a driver faults on servo speed.
_FC_RADIUS = 3.0


def hermite(
    prev: Sequence[float], target: Sequence[float], n: int, m0: Sequence[float], m1: Sequence[float]
) -> list[list[float]]:
    """`n` points of the cubic Hermite segment from `prev` (exclusive) to `target` (inclusive).

    Tangents are in position per tick, so the segment spans s = k/n, k = 1..n.
    `m0 = m1 = target - prev` gives `interpolate`'s points. The last point is
    `list(target)` exactly, as there.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not len(prev) == len(target) == len(m0) == len(m1):
        raise ValueError(
            "prev, target, m0 and m1 must be the same length, got "
            f"{len(prev)}, {len(target)}, {len(m0)} and {len(m1)}"
        )

    points = []
    for k in range(1, n):
        s = k / n
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        points.append(
            [h00 * p + h10 * a + h01 * t + h11 * b for p, t, a, b in zip(prev, target, m0, m1)]
        )
    points.append(list(target))
    return points


def monotone_tangents(
    m0: Sequence[float], m1: Sequence[float], d: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Fritsch-Carlson limiting, per joint, against chord `d`.

    A tangent against the chord, or any tangent on a zero chord, becomes 0;
    (m0/d, m1/d) is then scaled onto the radius-`_FC_RADIUS` circle if outside
    it. The segment then never leaves [prev, target].
    """
    out0, out1 = [], []
    for a, b, c in zip(m0, m1, d):
        ok = abs(c) >= 1e-12  # a negligible chord is a zero chord, never a NaN
        alpha = a / c if ok and a * c > 0 else 0.0
        beta = b / c if ok and b * c > 0 else 0.0
        r = math.hypot(alpha, beta)
        if r > _FC_RADIUS:
            alpha, beta = alpha * _FC_RADIUS / r, beta * _FC_RADIUS / r
        out0.append(alpha * c)
        out1.append(beta * c)
    return out0, out1


def cubic_tick(
    prev: Sequence[float], target: Sequence[float], n: int, tangent: Sequence[float] | None
) -> tuple[list[list[float]], list[float]]:
    """One tick of cubic densification: (points, end tangent to carry).

    The end tangent is this tick's chord (a backward difference: no lookahead);
    the start tangent is the carried one, None meaning at rest. Both are limited
    by `monotone_tangents`, and the limited end tangent is returned for the next tick.
    """
    d = [t - p for p, t in zip(prev, target)]
    m0, m1 = monotone_tangents([0.0] * len(d) if tangent is None else tangent, d, d)
    return hermite(prev, target, n, m0, m1), m1


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
    same batch as the first tick's motion points, and `_t0` is that tick's nominal time.

    `interp='cubic'` replaces linear points with `cubic_tick`'s monotone
    segment; stamps are identical.
    """

    def __init__(
        self,
        arm: Any,
        *,
        fps: float,
        stream_hz: float,
        extra: dict[str, Any],
        interp: str = "linear",
    ) -> None:
        self._arm = arm
        self._fps = fps
        self._stream_hz = stream_hz
        self._extra = extra
        self._interp = interp
        # cubic only: the last segment's (limited) end tangent, per tick; None = at rest.
        self._tangent: list[float] | None = None
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

    async def send(self, target: Sequence[float], nominal_t: float) -> None:
        """Queue one tick's densified points, stamped from the tick's nominal time.

        `nominal_t` is the controller's `TickSchedule.nominal()` for this tick.
        Stamping from it, not from when this call happens to run, keeps every gap
        exactly `step` however late within its lead the tick is.
        """
        n = max(1, round(self._stream_hz / self._fps))
        # Tile the tick evenly: 1/stream_hz gives 10, 10, 13.3 ms gaps at 100/30.
        step = 1.0 / (n * self._fps)
        lead = 1.0 / self._fps

        batch: list[Any] = []
        first_send = self._t0 is None
        if first_send:
            self._t0 = nominal_t  # see the class docstring on anchoring
            batch.append(Arm.TrajectoryPoint(time=timedelta(0), positions=list(self._measured)))

        elapsed = nominal_t - self._t0
        prev = self._measured if first_send else self._last

        tangent = None
        if self._interp == "cubic":
            # Contiguous when this tick starts no later than the last stamp sent:
            # anything later is a hold (starvation restart, re-anchor, skipped
            # tick) the arm spent at rest on `prev`.
            contiguous = not first_send and elapsed + lead <= self._last_time + 1e-6
            points, tangent = cubic_tick(prev, target, n, self._tangent if contiguous else None)
        else:
            points = interpolate(prev, target, n)

        last_time = self._last_time
        for k, p in enumerate(points):
            # Stamp one tick ahead (a past-due point is sent immediately, and a
            # stream of past-due points collapses into bursts at fps); the max()
            # keeps times strictly increasing across a re-anchor or a rewind.
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
        self._tangent = tangent

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
