#!/usr/bin/env python3
"""Replay dumped traces' commanded knots through both stream densifiers.

  uv run python tools/densify_replay.py [--cols 0,1,2,3,4] [--stream-hz 100] run.csv ...

Reads `tools/dump_trace.py` CSVs (nominal_t, cmd0..cmdN). Each tick's
commanded target is a knot; `interpolate` and `cubic_tick` (the functions
ArmStream calls) densify them into round(stream_hz / fps) points per tick.
The trace has no stamps, so a nominal_t gap over 1.5 periods (a hold) splits
the run and restarts the cubic from rest, as ArmStream does. Prints max and p99
|acceleration| (deg/s²) over the densified points and the peak point speed
over the fastest chord's average speed. An arm_joint gripper's outright
switches dominate the max: leave its column out with --cols.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from vla.controller.trajectory import cubic_tick, interpolate


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cmd = sorted((k for k in rows[0] if k.startswith("cmd")), key=lambda k: int(k[3:]))
    t = np.array([float(r["nominal_t"]) for r in rows])
    knots = np.array([[float(r[k]) for k in cmd] for r in rows])
    return t, knots


def densify(knots: np.ndarray, n: int, cubic: bool) -> np.ndarray:
    points, tangent = [list(knots[0])], None
    for prev, target in zip(knots.tolist(), knots[1:].tolist()):
        if cubic:
            seg, tangent = cubic_tick(prev, target, n, tangent)
        else:
            seg = interpolate(prev, target, n)
        points.extend(seg)
    return np.array(points)


def replay(t: np.ndarray, knots: np.ndarray, cols: list[int], stream_hz: float, cubic: bool):
    period = float(np.median(np.diff(t)))
    n = max(1, round(stream_hz * period))
    h = period / n
    runs = np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > 1.5 * period) + 1)
    acc, speed, chord = [], 0.0, 0.0
    for idx in runs:
        if len(idx) < 2:
            continue
        k = knots[idx][:, cols]
        chord = max(chord, float(np.abs(np.diff(k, axis=0)).max()) / period)
        v = np.diff(densify(knots[idx], n, cubic)[:, cols], axis=0) / h
        speed = max(speed, float(np.abs(v).max()))
        if len(v) > 1:
            acc.append(np.abs(np.diff(v, axis=0) / h).max(axis=1))
    a = np.concatenate(acc)
    return float(a.max()), float(np.percentile(a, 99)), speed / chord if chord else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+")
    p.add_argument("--cols", type=lambda v: [int(c) for c in v.split(",")], help="cmd columns to measure, e.g. 0,1,2 (default: all)")
    p.add_argument("--stream-hz", type=float, default=100.0)
    args = p.parse_args()
    for path in args.traces:
        t, knots = load(path)
        cols = args.cols or list(range(knots.shape[1]))
        print(f"{Path(path).name} ({len(t)} ticks)")
        for name, cubic in (("linear", False), ("cubic", True)):
            mx, p99, ratio = replay(t, knots, cols, args.stream_hz, cubic)
            print(f"  {name:6s}  max {mx:8.0f}  p99 {p99:8.0f} deg/s²  peak/chord {ratio:.2f}")


if __name__ == "__main__":
    main()
