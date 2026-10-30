#!/usr/bin/env python3
"""Dump the controller's per-tick trace as CSV and count clamped ticks.

  VIAM_API_KEY=... VIAM_API_KEY_ID=... \
    tools/dump_trace.py --address $FQDN --service vla-controller > run.csv

Run it right after a run, before the next `start` (which clears the trace).
Columns: nominal_t, [sample_t], cmd0..cmdN, meas0..measN, clamped.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys

from viam.robot.client import RobotClient
from viam.services.generic import Generic


async def fetch(address: str, service: str) -> dict:
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"]
    )
    robot = await RobotClient.at_address(address, opts)
    try:
        return await Generic.from_robot(robot, service).do_command({"command": "trace"})
    finally:
        await robot.close()


def write_csv(trace: dict, out) -> int:
    n = len(trace["nominal_t"])
    dof = len(trace["commanded"][0]) if n else 0
    has_sample = "sample_t" in trace
    w = csv.writer(out)
    w.writerow(
        ["nominal_t"]
        + (["sample_t"] if has_sample else [])
        + [f"cmd{j}" for j in range(dof)]
        + [f"meas{j}" for j in range(dof)]
        + ["clamped"]
    )
    for i in range(n):
        w.writerow(
            [trace["nominal_t"][i]]
            + ([trace["sample_t"][i]] if has_sample else [])
            + list(trace["commanded"][i])
            + list(trace["measured"][i])
            + [int(bool(trace["clamped"][i]))]
        )
    return sum(bool(c) for c in trace["clamped"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", required=True, help="machine FQDN (needs VIAM_API_KEY/_ID)")
    p.add_argument("--service", default="vla-controller", help="controller service name")
    args = p.parse_args()
    trace = asyncio.run(fetch(args.address, args.service))
    clamped = write_csv(trace, sys.stdout)
    print(f"{len(trace['nominal_t'])} ticks, {clamped} clamped", file=sys.stderr)


if __name__ == "__main__":
    main()
