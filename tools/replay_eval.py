#!/usr/bin/env python3
"""Offline open-loop action-error replay against a LeRobot training dataset.

Answers one question: given the exact observations the policy was trained on,
does the deployed service predict the actions the dataset recorded? That
separates a bad observation pipeline (units, joint order, camera assignment,
image geometry, task string) from a genuinely weak policy -- on the robot the
two are indistinguishable.

It calls the *deployed* `viam-labs:vla:policy` service over DoCommand rather
than loading the checkpoint locally, so it exercises the same wire codec and
the same backend the controller will use. Point it at whichever machine you
want to characterise; the same run works against the Orin later.

Two comparisons worth running:

  # 1. dataset frames verbatim -- is the MODEL faithful?
  tools/replay_eval.py --address $FQDN --repo-id you/box-flaps --episode 0

  # 2. same frames pushed through your camera geometry and the controller's
  #    own resize/pad/JPEG path -- is your PIPELINE faithful?
  tools/replay_eval.py --address $FQDN --repo-id you/box-flaps --episode 0 \
      --camera-resolution 640x480 --image-fit pad --jpeg-quality 90

If (1) is clean and (2) is not, the fault is entirely in observation wiring.
Compare the two error curves, not either one's absolute value.

Credentials come from the environment, the standard Viam convention:
    export VIAM_API_KEY=...  VIAM_API_KEY_ID=...

The dataset reader needs lerobot's `dataset` extra, which the module's own
`lerobot` extra deliberately does not pull in (it would put HuggingFace
`datasets` on every robot):
    uv pip install 'lerobot[dataset]'

`--self-check` runs the metric and encode paths against synthetic data, with
no network and no dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vla.controller.observation import ObservationBuilder
from vla.wire import encode_image, encode_vector

# --------------------------------------------------------------------------
# The pure core: everything below the I/O edges, so it is testable offline.
# --------------------------------------------------------------------------


def compute_errors(predicted: np.ndarray, recorded: np.ndarray) -> dict:
    """Compare stacked chunks against ground truth.

    Both are (n_samples, horizon, action_dim). Returns the two cuts that
    actually localize a fault: error against horizon offset k (does it start
    wrong, or drift?) and error per action dimension (which joint?).
    """
    if predicted.shape != recorded.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {recorded.shape}")
    if predicted.ndim != 3:
        raise ValueError(f"expected (samples, horizon, dim), got {predicted.shape}")

    err = predicted - recorded
    # L2 across the action vector, per (sample, k) -- the "how wrong is this
    # commanded pose" number, in the action's own units.
    l2 = np.linalg.norm(err, axis=2)
    return {
        "n_samples": int(predicted.shape[0]),
        "horizon": int(predicted.shape[1]),
        "action_dim": int(predicted.shape[2]),
        "l2_by_offset": l2.mean(axis=0),          # (horizon,)
        "mse_by_joint": (err ** 2).mean(axis=(0, 1)),  # (dim,)
        "mae_by_joint": np.abs(err).mean(axis=(0, 1)),
        "l2_mean": float(l2.mean()),
        "l2_first_step": float(l2[:, 0].mean()),
        "l2_p95": float(np.percentile(l2, 95)),
    }


def sample_starts(n_frames: int, horizon: int, stride: int, limit: int) -> list[int]:
    """Choose window start frames, spreading `limit` across the whole episode.

    Stops `horizon` before the end so no window is compared against padding.
    When `limit` truncates, it subsamples EVENLY rather than taking a prefix:
    a prefix biases every sample toward the start of the episode, which is
    precisely the axis errors_by_position measures, and would report the
    early-task error as if it were the whole picture.
    """
    starts = list(range(0, max(0, n_frames - horizon), stride))
    if limit and len(starts) > limit:
        idx = np.linspace(0, len(starts) - 1, limit).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx.tolist()))]
    return starts


def errors_by_position(
    predicted: np.ndarray, truth: np.ndarray, starts: list[int], n_frames: int, buckets: int = 4
) -> list[dict]:
    """Group each sample's error by where in the episode it was taken.

    A policy that has learned the approach but not the finish is worse at the
    end of the episode specifically, and a single mean over the whole run
    hides that completely. Bucketing by start frame is what separates "this
    checkpoint is weak" from "this checkpoint is weak at the last subtask".

    Position is a fraction of the FULL episode, not of the sampled range, so
    the numbers mean what they say -- with the consequence that the final
    `horizon` frames can never be a window start, and the last bucket
    therefore stops short of 1.0. `max_frac` reports where it actually ends
    so a thin top bucket is visible rather than silently reassuring.
    """
    per_sample = np.linalg.norm(predicted - truth, axis=2).mean(axis=1)
    frac = np.asarray(starts, dtype=float) / max(1, n_frames - 1)
    edges = np.linspace(0.0, 1.0, buckets + 1)
    out = []
    for i in range(buckets):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (frac >= lo) & (frac < hi) if i < buckets - 1 else (frac >= lo)
        out.append({
            "lo": lo,
            "hi": hi,
            "n": int(sel.sum()),
            "l2_mean": float(per_sample[sel].mean()) if sel.any() else float("nan"),
            "max_frac": float(frac[sel].max()) if sel.any() else float("nan"),
        })
    return out


def naive_baselines(
    recorded: np.ndarray, states: np.ndarray, starts: list[int], horizon: int
) -> dict[str, np.ndarray]:
    """Predictors that use no policy at all, scored on the same windows.

    Absolute action error is uninterpretable on its own -- whether a mean L2
    of 26 is good depends entirely on how much the arm moves in a chunk. These
    give it a scale, and they are not weak strawmen: on a smooth, quasi-static
    trajectory "don't move" is genuinely hard to beat over a short horizon.
    A policy that does not clear them is not predicting motion, whatever its
    absolute numbers look like.

      hold current state  -- the arm stays exactly where it is
      repeat action at t  -- the true action at t, held (cheats at k=0 by
                             construction; the honest reading is k>0)
      episode-mean action -- the trajectory's centre of mass, ignoring the
                             observation entirely. A policy scoring like THIS
                             one is not reading its inputs.
    """
    tile = lambda rows: np.repeat(rows[:, None, :], horizon, axis=1)
    return {
        "hold current state": tile(states),
        "repeat action at t": tile(recorded[starts]),
        "episode-mean action": np.broadcast_to(
            recorded.mean(axis=0), (len(starts), horizon, recorded.shape[1])
        ).copy(),
    }


def to_hwc_uint8(frame) -> np.ndarray:
    """Normalize a dataset frame to the HWC uint8 that encode_image wants.

    LeRobotDataset hands back CHW float in [0, 1] (a torch tensor); a raw
    numpy dataset may already be HWC uint8. Accept both rather than assuming,
    because getting this wrong silently produces a 1/255-scaled observation
    and an error curve that looks like a broken policy.
    """
    arr = np.asarray(frame.numpy() if hasattr(frame, "numpy") else frame)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def through_camera_pipeline(
    arr: np.ndarray, camera_hw: tuple[int, int] | None, target_hw: tuple[int, int], fit: str
) -> np.ndarray:
    """Simulate the round trip a real camera frame takes to the policy.

    Upsample the dataset frame to the camera's actual resolution, then apply
    the controller's own geometry step back down to the policy's declared
    shape. Reuses ObservationBuilder._resize_with_pad rather than
    re-deriving lerobot's pad convention here -- a second implementation of
    that is exactly how the two paths would silently diverge.
    """
    from PIL import Image

    if camera_hw is not None and arr.shape[:2] != camera_hw:
        arr = np.asarray(
            Image.fromarray(arr).resize((camera_hw[1], camera_hw[0]), Image.BILINEAR),
            dtype=np.uint8,
        )
    if arr.shape[:2] == target_hw:
        return arr
    if fit == "pad":
        return ObservationBuilder._resize_with_pad(arr, target_hw[0], target_hw[1])
    if fit == "stretch":
        return np.asarray(
            Image.fromarray(arr).resize((target_hw[1], target_hw[0]), Image.BILINEAR),
            dtype=np.uint8,
        )
    raise ValueError(f"unknown image_fit {fit!r}")


def resolve_key_map(
    pairs: list[str] | None, slots: list[str], sample
) -> dict[str, str]:
    """Map each policy image slot to the dataset key that feeds it.

    The checkpoint declares canonical slots (`observation.images.camera1`,
    ...) and lerobot's RenameObservationsProcessorStep rewrote the training
    dataset's own key names onto them, so a fine-tuned checkpoint's slot
    names carry no trace of which physical camera each one held. That
    assignment therefore CANNOT be inferred here, and guessing it by
    position would silently swap two views -- a policy fed swapped cameras
    still produces confident, plausible-looking motion, which is the single
    hardest failure mode to spot on a robot. So: require it explicitly, and
    when it is missing, fail with the candidate orderings spelled out.
    """
    available = [k for k in sample if str(k).startswith("observation.images")]

    if pairs:
        mapping = {}
        for pair in pairs:
            if "=" not in pair:
                raise SystemExit(f"--key-map wants DATASET_KEY=POLICY_SLOT, got {pair!r}")
            src, slot = (x.strip() for x in pair.split("=", 1))
            if slot not in slots:
                raise SystemExit(f"--key-map target {slot!r} is not a policy slot; slots are {slots}")
            if src not in available:
                raise SystemExit(f"--key-map source {src!r} is not in the dataset; it has {available}")
            mapping[slot] = src
        unmapped = [s for s in slots if s not in mapping]
        if unmapped:
            raise SystemExit(f"--key-map leaves slot(s) {unmapped} unfed")
        if len(set(mapping.values())) != len(mapping):
            raise SystemExit(f"--key-map feeds one dataset key into two slots: {mapping}")
        return mapping

    identity = [s for s in slots if s in available]
    if len(identity) == len(slots):
        return {s: s for s in slots}

    lines = [
        f"dataset image keys do not match the policy's slots.",
        f"  policy slots : {slots}",
        f"  dataset keys : {available}",
    ]
    if len(available) == len(slots):
        lines += [
            "",
            "Which dataset camera belongs in which slot is recorded in the",
            "checkpoint's rename_map, not in these names -- it cannot be guessed",
            "here. Run BOTH orderings; the correct one has dramatically lower",
            "action error, and that is your ground truth for the controller's",
            "`cameras` config too:",
        ]
        for order in (available, list(reversed(available))):
            flags = " ".join(f"--key-map {src}={slot}" for src, slot in zip(order, slots))
            lines.append(f"    {flags}")
    else:
        lines += [
            "",
            f"Counts differ ({len(available)} dataset keys vs {len(slots)} slots). If the",
            "checkpoint declares a camera this robot never feeds, add it to the policy's",
            "`unused_image_features` -- see the README.",
        ]
    raise SystemExit("\n".join(lines))


def build_infer_command(images: dict, state: np.ndarray, task: str, encoding: str, quality: int) -> dict:
    return {
        "command": "infer",
        "task": task,
        "state": encode_vector(np.asarray(state, dtype=np.float32)),
        "images": {
            key: encode_image(arr, encoding=encoding, quality=quality)
            for key, arr in images.items()
        },
    }


def report(metrics: dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print(
        f"samples={metrics['n_samples']}  horizon={metrics['horizon']}  "
        f"action_dim={metrics['action_dim']}"
    )
    print(f"mean L2 per step      : {metrics['l2_mean']:.4f}")
    print(f"  at offset k=0       : {metrics['l2_first_step']:.4f}   <- immediate action")
    print(f"  p95                 : {metrics['l2_p95']:.4f}")

    curve = metrics["l2_by_offset"]
    idx = sorted({0, 1, 4, 9, 24, len(curve) - 1})
    print("mean L2 by horizon offset:")
    for k in idx:
        bar = "#" * int(round(40 * curve[k] / max(curve.max(), 1e-9)))
        print(f"  k={k:<3} {curve[k]:8.4f}  {bar}")
    if metrics.get("baselines"):
        print("vs. predictors that never call the policy:")
        policy_mean = metrics["l2_mean"]
        for name, b in metrics["baselines"].items():
            verdict = "policy WORSE" if policy_mean >= b["l2_mean"] else "policy better"
            print(
                f"  {name:22s} mean L2 {b['l2_mean']:8.3f}  k=0 {b['l2_first_step']:7.3f}"
                f"   -> {verdict}"
            )
        beaten = [n for n, b in metrics["baselines"].items() if policy_mean >= b["l2_mean"]]
        if beaten:
            print(
                f"  !! the policy does not beat {len(beaten)}/{len(metrics['baselines'])} of them. "
                "It is not predicting motion better than a constant --\n"
                "     re-check the --key-map ordering and the image geometry before "
                "concluding the checkpoint is weak."
            )
    if metrics.get("by_position"):
        print("error by position in the episode (mean L2 per sample):")
        for b in metrics["by_position"]:
            if b["n"] == 0:
                print(f"  [{b['lo']:.2f},{b['hi']:.2f})  no samples")
                continue
            print(f"  [{b['lo']:.2f},{b['hi']:.2f})  n={b['n']:<3d} mean L2 {b['l2_mean']:8.3f}"
                  f"   (starts reach {b['max_frac']:.2f})")
        tail = metrics["by_position"][-1]
        rest = [b for b in metrics["by_position"][:-1] if b["n"]]
        if tail["n"] and rest:
            rest_mean = sum(b["l2_mean"] * b["n"] for b in rest) / sum(b["n"] for b in rest)
            delta = (tail["l2_mean"] - rest_mean) / rest_mean * 100
            print(f"  last bucket vs the rest: {tail['l2_mean']:.3f} vs {rest_mean:.3f}"
                  f"  ({delta:+.0f}%)")
            if delta > 25:
                print("  !! markedly worse at the end of the episode -- the failure is the "
                      "final subtask,\n     not the task as a whole. More demos OF THAT PHASE "
                      "beat more training steps.")
    print("per-joint error (MAE / MSE):")
    for j, (mae, mse) in enumerate(zip(metrics["mae_by_joint"], metrics["mse_by_joint"])):
        print(f"  joint {j}: {mae:8.4f} / {mse:10.4f}")


# --------------------------------------------------------------------------
# I/O edges
# --------------------------------------------------------------------------


def load_episode(repo_id: str, episode: int, root: str | None):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            f"could not import LeRobotDataset ({exc}).\n"
            "Install the dataset extra:  uv pip install 'lerobot[dataset]'"
        ) from exc

    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    frames = [ds[i] for i in range(len(ds))]
    if not frames:
        raise SystemExit(f"episode {episode} of {repo_id} is empty")
    return frames


class CliPolicy:
    """Talk to the policy by shelling out to the viam CLI.

    The SDK path needs an API key; the CLI already holds a login, so this
    makes the tool runnable with nothing but `viam login`. Slower per call
    (a fresh process and WebRTC handshake each time, ~3.5 s), but the metric
    is action error, which does not care about transport, and the latency
    reported by the service is measured server-side either way.
    """

    def __init__(self, part_id: str, service: str):
        self._part_id = part_id
        self._service = service

    async def do_command(self, command: dict) -> dict:
        import json
        import subprocess
        import time

        def call() -> dict:
            # Each call is a fresh process and a fresh WebRTC handshake, and
            # those fail transiently ("host appears to be offline") on a
            # machine that is demonstrably up. A run is dozens of calls over
            # several minutes, so one blip must not discard the whole thing.
            last = ""
            for attempt in range(3):
                proc = subprocess.run(
                    ["viam", "machines", "part", "run",
                     "--part", self._part_id,
                     "--method", "DoCommand",
                     "--component", self._service,
                     "--data", json.dumps({"command": command})],
                    capture_output=True, text=True,
                )
                out = proc.stdout
                start = out.find("{")
                if proc.returncode == 0 and start >= 0:
                    return json.loads(out[start:])["result"]
                last = (proc.stderr or out).strip()[:600]
                if attempt < 2:
                    print(f"    (retry {attempt + 1}/2 after: {last.splitlines()[-1][:120]})",
                          flush=True)
                    time.sleep(5 * (attempt + 1))
            raise SystemExit(f"viam CLI call failed after 3 attempts:\n{last}")

        return await asyncio.to_thread(call)


async def connect(address: str):
    from viam.robot.client import RobotClient

    key = os.environ.get("VIAM_API_KEY")
    key_id = os.environ.get("VIAM_API_KEY_ID")
    if not key or not key_id:
        raise SystemExit("set VIAM_API_KEY and VIAM_API_KEY_ID in the environment")
    return await RobotClient.at_address(
        address,
        RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id),
    )


async def run(args) -> int:
    from viam.services.generic import Generic

    frames = load_episode(args.repo_id, args.episode, args.root)
    print(f"loaded {len(frames)} frames from {args.repo_id} episode {args.episode}")

    robot = None
    if args.part_id:
        policy = CliPolicy(args.part_id, args.service)
        print(f"transport: viam CLI -> part {args.part_id}")
    else:
        robot = await connect(args.address)
        policy = Generic.from_robot(robot, args.service)
        print(f"transport: SDK -> {args.address}")
    try:

        specs = await policy.do_command({"command": "specs"})
        image_keys = list(specs["image_feature_keys"])
        horizon = int(specs["n_action_steps"])
        state_dim = int(specs["state_dim"])
        sizes = {k: tuple(int(v) for v in specs["input_features"][k][1:]) for k in image_keys}
        print(f"policy: {specs['policy_type']} on {specs['device']} ({specs['dtype']})")
        print(f"  image keys {image_keys}, horizon {horizon}, state_dim {state_dim}")

        key_map = resolve_key_map(args.key_map, image_keys, frames[0])
        if key_map != {k: k for k in image_keys}:
            print("dataset -> policy slot mapping:")
            for slot, src in key_map.items():
                print(f"  {src}  ->  {slot}")

        recorded = np.stack([to_np(f["action"]) for f in frames]).astype(np.float32)
        task = args.task if args.task is not None else str(frames[0].get("task", ""))
        print(f"task string: {task!r}")

        # Only starts where a full ground-truth window exists -- a short tail
        # window would silently compare against padding.
        starts = sample_starts(len(frames), horizon, args.stride, args.limit)
        if not starts:
            raise SystemExit(
                f"episode has {len(frames)} frames, shorter than the {horizon}-step "
                "chunk; nothing to compare"
            )
        print(f"sampling {len(starts)} start frames (stride {args.stride})\n")

        camera_hw = None
        if args.camera_resolution:
            w, h = (int(x) for x in args.camera_resolution.lower().split("x"))
            camera_hw = (h, w)

        preds = []
        sampled_states = []
        for n, t in enumerate(starts, 1):
            frame = frames[t]
            images = {}
            for key in image_keys:
                arr = to_hwc_uint8(frame[key_map[key]])
                if camera_hw is not None or arr.shape[:2] != sizes[key]:
                    arr = through_camera_pipeline(arr, camera_hw, sizes[key], args.image_fit)
                images[key] = arr

            state = to_np(frame["observation.state"]).astype(np.float32)
            if state.shape[0] != state_dim:
                raise SystemExit(
                    f"dataset observation.state has {state.shape[0]} dims, policy "
                    f"expects {state_dim} -- state_joint_indices is probably wrong"
                )

            sampled_states.append(state)

            # Independent samples: reset so no per-episode policy state leaks
            # from the previous call and makes the curve look better than it is.
            await policy.do_command({"command": "reset"})
            cmd = build_infer_command(images, state, task, args.image_encoding, args.jpeg_quality)
            result = await policy.do_command(cmd)

            chunk = np.asarray(result["actions"]["rows"], dtype=np.float32)
            preds.append(chunk[:horizon])
            print(
                f"  [{n}/{len(starts)}] frame {t}  latency {result['latency_s']:.2f}s  "
                f"L2(k=0) {np.linalg.norm(chunk[0] - recorded[t]):.4f}",
                flush=True,
            )

        predicted = np.stack(preds)
        truth = np.stack([recorded[t : t + horizon] for t in starts])
        metrics = compute_errors(predicted, truth)
        metrics["by_position"] = errors_by_position(predicted, truth, starts, len(frames))
        metrics["baselines"] = {
            name: compute_errors(pred, truth)
            for name, pred in naive_baselines(
                recorded, np.stack(sampled_states), starts, horizon
            ).items()
        }
        label = "dataset frames verbatim"
        if camera_hw is not None:
            label = f"through {args.camera_resolution} camera + image_fit={args.image_fit}"
        report(metrics, label)
        return 0
    finally:
        if robot is not None:
            await robot.close()


def to_np(x) -> np.ndarray:
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x)


# --------------------------------------------------------------------------


def self_check() -> int:
    """Exercise the metric and geometry paths with no network and no dataset."""
    rng = np.random.default_rng(0)

    truth = rng.normal(size=(4, 50, 6)).astype(np.float32)
    m = compute_errors(truth.copy(), truth)
    assert m["l2_mean"] == 0.0, m["l2_mean"]
    assert m["mse_by_joint"].shape == (6,)

    # A constant offset of c on every one of d dims -> L2 = c*sqrt(d), MSE = c^2.
    off = compute_errors(truth + 0.5, truth)
    assert abs(off["l2_mean"] - 0.5 * np.sqrt(6)) < 1e-5, off["l2_mean"]
    assert np.allclose(off["mse_by_joint"], 0.25, atol=1e-6)
    # Error isolated to one joint must show up in exactly that joint.
    bad = truth.copy()
    bad[:, :, 3] += 2.0
    one = compute_errors(bad, truth)
    assert one["mse_by_joint"].argmax() == 3, one["mse_by_joint"]

    try:
        compute_errors(truth, truth[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise")

    # CHW float [0,1] and HWC uint8 must land on the same array.
    hwc = rng.integers(0, 256, (8, 12, 3), dtype=np.uint8)
    chw = (hwc.transpose(2, 0, 1).astype(np.float32) / 255.0)
    assert np.abs(to_hwc_uint8(chw).astype(int) - hwc.astype(int)).max() <= 1
    assert to_hwc_uint8(hwc).dtype == np.uint8

    # Geometry: aspect-preserving pad keeps content off the padded edges, and
    # the black padding lands on the LEFT/TOP (smolvla's convention).
    wide = np.full((48, 128, 3), 200, dtype=np.uint8)
    out = through_camera_pipeline(wide, None, (64, 64), "pad")
    assert out.shape == (64, 64, 3), out.shape
    assert out[0, 0].max() == 0, "pad must be black on the top-left"
    assert out[-1, -1].max() > 0, "content must reach the bottom-right"
    assert through_camera_pipeline(wide, (480, 640), (64, 64), "stretch").shape == (64, 64, 3)

    # A round trip through the wire codec must preserve the frame's shape.
    from vla.wire import decode_image
    payload = build_infer_command({"c": out}, np.zeros(6, np.float32), "t", "jpeg", 90)
    assert decode_image(payload["images"]["c"]).shape == out.shape
    assert payload["state"]["values"] == [0.0] * 6

    # resolve_key_map: the branchiest thing here, and a silent wrong answer
    # swaps two camera views, so cover every exit.
    slots = ["observation.images.camera1", "observation.images.camera2"]
    ds_keys = {"observation.images.realsense_cam": 0, "observation.images.camera_transform": 0,
               "observation.state": 0, "action": 0}
    same = {k: 0 for k in slots}
    assert resolve_key_map(None, slots, same) == {k: k for k in slots}

    ok = resolve_key_map(
        ["observation.images.realsense_cam=observation.images.camera1",
         "observation.images.camera_transform=observation.images.camera2"], slots, ds_keys)
    assert ok == {slots[0]: "observation.images.realsense_cam",
                  slots[1]: "observation.images.camera_transform"}, ok

    def rejects(pairs, needle):
        try:
            resolve_key_map(pairs, slots, ds_keys)
        except SystemExit as exc:
            assert needle in str(exc), f"{needle!r} not in {exc}"
        else:
            raise AssertionError(f"should have rejected {pairs}")

    rejects(["observation.images.realsense_cam=observation.images.camera1"], "unfed")
    rejects(["observation.images.realsense_cam=observation.images.camera1",
             "observation.images.realsense_cam=observation.images.camera2"],
            "one dataset key into two slots")
    rejects(["observation.images.realsense_cam=nope"], "not a policy slot")
    rejects(["observation.images.nope=observation.images.camera1"], "not in the dataset")
    rejects(["no-equals-sign"], "DATASET_KEY=POLICY_SLOT")
    # No map, equal counts -> must offer BOTH orderings, not pick one.
    try:
        resolve_key_map(None, slots, ds_keys)
    except SystemExit as exc:
        msg = str(exc)
        assert "realsense_cam=observation.images.camera1" in msg
        assert "realsense_cam=observation.images.camera2" in msg, msg
    else:
        raise AssertionError("a missing key-map must not be guessed")
    # Unequal counts -> point at unused_image_features instead of orderings.
    try:
        resolve_key_map(None, slots, {"observation.images.only_one": 0})
    except SystemExit as exc:
        assert "unused_image_features" in str(exc)
    else:
        raise AssertionError("count mismatch must raise")

    # naive_baselines: shapes, and that each predictor is what it claims.
    rec = rng.normal(size=(120, 6)).astype(np.float32)
    sts = rng.normal(size=(3, 6)).astype(np.float32)
    st2 = [0, 10, 20]
    bl = naive_baselines(rec, sts, st2, 50)
    assert set(bl) == {"hold current state", "repeat action at t", "episode-mean action"}
    for name, pred in bl.items():
        assert pred.shape == (3, 50, 6), (name, pred.shape)
        # every predictor is constant across the horizon by construction
        assert np.allclose(pred, pred[:, :1, :]), name
    assert np.allclose(bl["hold current state"][:, 0, :], sts)
    assert np.allclose(bl["repeat action at t"][:, 0, :], rec[st2])
    assert np.allclose(bl["episode-mean action"][0, 0, :], rec.mean(axis=0))
    # "repeat action at t" must be exactly right at k=0 -- that is why the
    # docstring says to read it at k>0.
    truth2 = np.stack([rec[t : t + 50] for t in st2])
    assert compute_errors(bl["repeat action at t"], truth2)["l2_first_step"] == 0.0

    # errors_by_position: bucketing, and that it actually localises error.
    rec2 = np.zeros((400, 6), np.float32)
    pos_starts = [0, 100, 200, 300]
    truth3 = np.stack([rec2[t : t + 50] for t in pos_starts])
    pred3 = truth3.copy()
    pred3[3] += 10.0  # only the last-quarter sample is wrong
    got = errors_by_position(pred3, truth3, pos_starts, 400)
    assert [b["n"] for b in got] == [1, 1, 1, 1], [b["n"] for b in got]
    assert got[0]["l2_mean"] == 0.0 and got[1]["l2_mean"] == 0.0
    assert got[3]["l2_mean"] > 20.0, got[3]["l2_mean"]
    assert abs(got[3]["max_frac"] - 300 / 399) < 1e-6
    # An empty top bucket must be reported, not crash or silently vanish.
    empty = errors_by_position(truth3[:3], truth3[:3], [0, 10, 20], 400)
    assert empty[-1]["n"] == 0 and np.isnan(empty[-1]["l2_mean"])
    assert [b["n"] for b in empty] == [3, 0, 0, 0]

    # sample_starts: never past the ground-truth edge, and --limit must not
    # collapse coverage onto the start of the episode.
    assert sample_starts(400, 50, 20, 0) == list(range(0, 350, 20))
    spread = sample_starts(621, 50, 20, 16)
    assert len(spread) <= 16
    assert max(spread) > 0.75 * (621 - 50), f"--limit truncated coverage: {max(spread)}"
    assert all(t + 50 <= 621 for t in spread)
    assert sample_starts(60, 50, 20, 0) == [0]
    assert sample_starts(40, 50, 20, 0) == []

    print("self-check OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-check", action="store_true", help="run offline checks and exit")
    p.add_argument("--address", help="machine FQDN (SDK transport; needs VIAM_API_KEY/_ID)")
    p.add_argument("--part-id", help="machine part id (viam CLI transport; uses your CLI login)")
    p.add_argument("--service", default="vla-policy", help="policy service name (default: vla-policy)")
    p.add_argument("--repo-id", help="LeRobot dataset repo id or local name")
    p.add_argument("--root", default=None, help="local dataset root, if not the HF cache")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=10, help="frames between sampled starts (default: 10)")
    p.add_argument("--limit", type=int, default=0, help="stop after N samples (0 = all)")
    p.add_argument("--task", default=None, help="override the dataset's task string")
    p.add_argument("--key-map", action="append", metavar="DATASET_KEY=POLICY_SLOT",
                   help="feed a dataset image key into a policy slot; repeatable")
    p.add_argument("--camera-resolution", default=None, metavar="WxH",
                   help="simulate a real camera of this size before the controller's resize")
    p.add_argument("--image-fit", default="pad", choices=("pad", "stretch"))
    p.add_argument("--image-encoding", default="jpeg", choices=("jpeg", "png", "raw"))
    p.add_argument("--jpeg-quality", type=int, default=90)
    args = p.parse_args()

    if args.self_check:
        return self_check()
    if not args.repo_id:
        p.error("--repo-id required (or use --self-check)")
    if bool(args.address) == bool(args.part_id):
        p.error("pass exactly one of --address (SDK) or --part-id (viam CLI)")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
