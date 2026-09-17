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

`--action-space delta-ee` grades an end-effector-pose-delta checkpoint (state 9,
action 6). It changes three things, because a whole-vector L2 would be
meaningless there: the error is reported per SEGMENT (translation in mm,
rotation in radians -- a single L2 over both is a translation metric with the
rotation error rounded away, the two differing by ~500x in scale), the
"don't move" baseline becomes the ZERO DELTA rather than the current state,
and the default `--image-fit` follows the controller's (`stretch_bicubic`).
`auto`, the default, infers it from the policy's declared dims.

Note what this tool does NOT cover on that action space: it grades observation
assembly and the policy, and nothing in the actuation half -- the Cartesian
clamp, composing onto a live pose, the orientation-vector encode, or an IK
refusal. Those need hardware, or the dataset's own recorded motion as a
witness -- neither of which lives in this tool.

Flow-matching policies sample an unseeded noise vector, so the SAME observation
gives a different chunk every call -- on box-opener the draw-to-draw disagreement
is roughly half the ground-truth error. A single-draw run therefore cannot
resolve anything smaller than that. `--repeats N` averages the error over N
draws per window and prints the spread it averaged away:

  tools/replay_eval.py --part-id $PART --repo-id you/box-flaps --repeats 5

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
import contextlib
import io
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


# Action-vector segments, per action space. A segment is (label, start, stop,
# unit), and every error statistic is computed once per segment.
#
# This exists because an L2 across a mixed-unit vector is not a distance. On
# `delta-ee` the first three channels carry millimetres (recorded per-tick std
# ~6-7) and the last three radians (std ~0.011-0.016) -- a ~500x scale gap, so
# a whole-vector L2 is a translation metric with the rotation error rounded
# away entirely. A checkpoint could be badly wrong about the wrist and the
# headline number would not move.
#
# `joints` keeps a single whole-vector segment, so its numbers are bit-identical
# to what this tool reported before segments existed.
SEGMENTS = {
    "delta-ee": (("translation", 0, 3, "mm"), ("rotation", 3, 6, "rad")),
}
DIM_NAMES = {"delta-ee": ["dx", "dy", "dz", "drx", "dry", "drz"]}


def segments_for(action_space: str, action_dim: int):
    return SEGMENTS.get(action_space) or (("action", 0, action_dim, ""),)


def default_image_fit(action_space: str) -> str:
    """The controller's own `image_fit` default for this action space.

    Mirrored rather than imported: `config._default_image_fit` lives in the
    controller package, and this tool must run against a checkpoint without
    a controller configured. Grading through a different resize than the
    controller uses would charge the policy for geometry it never sees.
    """
    return "stretch_bicubic" if action_space == "delta-ee" else "pad"


def compute_errors(
    predicted: np.ndarray,
    recorded: np.ndarray,
    mask: np.ndarray | None = None,
    segments=None,
) -> dict:
    """Compare stacked chunks against ground truth.

    Both are (n_samples, horizon, action_dim). Returns the two cuts that
    actually localize a fault: error against horizon offset k (does it start
    wrong, or drift?) and error per action dimension (which joint?).

    `segments` splits every statistic by unit -- see SEGMENTS. It defaults to
    one whole-vector segment, which reproduces the pre-segment numbers exactly.

    `mask` is (n_samples, horizon) of bools marking which steps have ground
    truth. It is how truncated windows are supported: a window that runs off
    the end of the episode keeps its real steps and masks the rest, instead of
    the window being dropped entirely (which made the last quarter of every
    short episode unmeasurable). Every statistic below is computed over
    unmasked entries only, and `n_by_offset` records how many samples each
    offset actually had, because a truncated series thins out with k and a
    mean over three samples should not read like a mean over thirty.
    """
    if predicted.shape != recorded.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {recorded.shape}")
    if predicted.ndim != 3:
        raise ValueError(f"expected (samples, horizon, dim), got {predicted.shape}")
    if mask is None:
        mask = np.ones(predicted.shape[:2], dtype=bool)
    elif mask.shape != predicted.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != {predicted.shape[:2]}")
    if not mask.any():
        raise ValueError("mask excludes every step; nothing to compare")

    err = np.where(mask[:, :, None], predicted - recorded, 0.0)
    # L2 across the action vector, per (sample, k) -- the "how wrong is this
    # commanded pose" number, in the action's own units.
    l2 = np.linalg.norm(err, axis=2)

    n_by_offset = mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        l2_by_offset = np.where(n_by_offset > 0, l2.sum(axis=0) / n_by_offset, np.nan)
    valid_steps = int(mask.sum())
    flat = l2[mask]

    segs = segments or (("action", 0, predicted.shape[2], ""),)
    seg_stats = {}
    for label, start, stop, unit in segs:
        s_l2 = np.linalg.norm(err[:, :, start:stop], axis=2)
        s_flat = s_l2[mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            s_by_offset = np.where(
                n_by_offset > 0, s_l2.sum(axis=0) / n_by_offset, np.nan
            )
        seg_stats[label] = {
            "unit": unit,
            "cols": (start, stop),
            "l2_mean": float(s_flat.mean()),
            "l2_first_step": (
                float(s_l2[mask[:, 0], 0].mean()) if mask[:, 0].any() else float("nan")
            ),
            "l2_p95": float(np.percentile(s_flat, 95)),
            "l2_by_offset": s_by_offset,
        }
    # per-joint means divide by (valid steps), not by every slot in the array
    sq = np.where(mask[:, :, None], (predicted - recorded) ** 2, 0.0)
    ab = np.where(mask[:, :, None], np.abs(predicted - recorded), 0.0)
    return {
        "n_samples": int(predicted.shape[0]),
        "horizon": int(predicted.shape[1]),
        "action_dim": int(predicted.shape[2]),
        "valid_steps": valid_steps,
        "truncated": bool((~mask).any()),
        "n_by_offset": n_by_offset,
        "l2_by_offset": l2_by_offset,
        "mse_by_joint": sq.sum(axis=(0, 1)) / valid_steps,
        "mae_by_joint": ab.sum(axis=(0, 1)) / valid_steps,
        "l2_mean": float(flat.mean()),
        "l2_first_step": float(l2[mask[:, 0], 0].mean()) if mask[:, 0].any() else float("nan"),
        "l2_p95": float(np.percentile(flat, 95)),
        "segments": seg_stats,
        # Whole-vector statistics are suppressed by `report` when this is set:
        # summing mm and radians under one square root produces a number with
        # no unit and no meaning.
        "mixed_units": len(segs) > 1,
    }


def sample_starts(
    n_frames: int, horizon: int, stride: int, limit: int, min_overlap: int = 0
) -> list[int]:
    """Choose window start frames, spreading `limit` across the whole episode.

    With `min_overlap` 0, stops `horizon` before the end so every window has
    full ground truth. That is safe but blind: a 50-step window at 10 fps
    cannot start in the last 5 s of an episode, so on a 20 s episode the whole
    final quarter is unreachable and reports as "no samples" -- 16 of 125
    box-opener episodes are that short.

    A positive `min_overlap` admits windows that run off the end, requiring
    only that many real steps to grade against; compute_errors masks the rest.
    That trades a comparable metric for reach, so the two are never mixed:
    truncated windows contain fewer high-k steps, and high-k steps are where
    the error lives, so a truncated run scores LOWER for reasons that have
    nothing to do with the policy.

    When `limit` truncates, it subsamples EVENLY rather than taking a prefix:
    a prefix biases every sample toward the start of the episode, which is
    precisely the axis errors_by_position measures, and would report the
    early-task error as if it were the whole picture.
    """
    reach = n_frames - (min_overlap if min_overlap > 0 else horizon)
    starts = list(range(0, max(0, reach), stride))
    if limit and len(starts) > limit:
        idx = np.linspace(0, len(starts) - 1, limit).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx.tolist()))]
    return starts


def repeat_starts(starts: list[int], repeats: int) -> list[int]:
    """Duplicate each window start so it is drawn `repeats` times.

    The policy is stochastic, so the quantity worth reporting is the EXPECTED
    error of one draw -- which means averaging errors across draws, never
    averaging the chunks first. Averaging chunks would measure the error of the
    mean prediction, a different and much more flattering number than anything
    the robot ever executes.

    Expanding the sample list is how that is done here: every downstream
    statistic already means over samples, so N duplicated starts give exactly
    the mean single-draw error, with the standard error cut by sqrt(N), and no
    metric function needs to learn about draws.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    return [t for t in starts for _ in range(repeats)]


def between_draw_spread(
    predicted: np.ndarray, repeats: int, mask: np.ndarray | None = None
) -> float:
    """Mean L2 of a single draw from the mean of its own window's draws.

    This is the noise floor, and without it a run is unreadable: a few percent
    between two conditions is indistinguishable from the sampler. Quoted next
    to the headline it says how much of the headline is real -- with N draws the
    standard error of the mean is about this over sqrt(N).

    Draws of one window are consecutive rows (see repeat_starts), so the
    grouping is a reshape, and every window's draws share one mask.
    """
    if repeats < 2:
        return float("nan")
    n = predicted.shape[0] // repeats
    if mask is None:
        mask = np.ones(predicted.shape[:2], dtype=bool)
    g = predicted.reshape(n, repeats, *predicted.shape[1:])
    gm = mask.reshape(n, repeats, -1)[:, 0]
    dev = np.linalg.norm(g - g.mean(axis=1, keepdims=True), axis=3)
    counts = np.maximum(gm.sum(axis=1), 1)
    per_window = np.where(gm[:, None, :], dev, 0.0).sum(axis=(1, 2)) / (repeats * counts)
    live = gm.any(axis=1)
    return float(per_window[live].mean()) if live.any() else float("nan")


def spread_by_offset(
    predicted: np.ndarray, repeats: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """`between_draw_spread`, resolved per horizon offset instead of pooled.

    This is what separates two very different diagnoses of a rising error
    curve. If the draws AGREE at an offset but are wrong, the policy is
    confidently wrong there and more capacity or training could fix it. If
    the draws DISAGREE, the policy is sampling several plausible futures and
    the error is genuine ambiguity about what happens next -- which no
    amount of retraining removes, because the ground truth is one arbitrary
    sample from that same distribution.

    Returns an array of length `horizon`, nan where a mask left no draws.
    """
    if repeats < 2:
        return np.full(predicted.shape[1], np.nan, dtype=np.float64)
    n = predicted.shape[0] // repeats
    if mask is None:
        mask = np.ones(predicted.shape[:2], dtype=bool)
    g = predicted.reshape(n, repeats, *predicted.shape[1:])
    gm = mask.reshape(n, repeats, -1)[:, 0]
    dev = np.linalg.norm(g - g.mean(axis=1, keepdims=True), axis=3).mean(axis=1)
    counts = gm.sum(axis=0)
    total = np.where(gm, dev, 0.0).sum(axis=0)
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def errors_by_position(
    predicted: np.ndarray, truth: np.ndarray, starts: list[int], n_frames: int,
    buckets: int = 4, mask: np.ndarray | None = None,
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
    if mask is None:
        mask = np.ones(predicted.shape[:2], dtype=bool)
    l2 = np.where(mask, np.linalg.norm(predicted - truth, axis=2), 0.0)
    counts = mask.sum(axis=1)
    per_sample = np.where(counts > 0, l2.sum(axis=1) / np.maximum(counts, 1), np.nan)
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
    recorded: np.ndarray,
    states: np.ndarray,
    starts: list[int],
    horizon: int,
    action_space: str = "joints",
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

    "don't move" is action-space dependent, and getting it wrong here is not a
    subtlety: under `delta-ee` the state is a 9-vector and the action a
    6-vector, so tiling the state as a prediction is a shape error, and the
    thing it was standing in for -- the arm staying put -- is expressed as the
    ZERO DELTA instead. That is also the baseline that matters most for a
    relative action space, and it is a hard one: the recorded median per-tick
    motion on `xarm-open-box-eedelta` is only 9.4 mm, so predicting nothing at
    all already scores well. A policy that does not clear zero is not
    predicting motion.
    """
    tile = lambda rows: np.repeat(rows[:, None, :], horizon, axis=1)
    dim = recorded.shape[1]
    hold = (
        np.zeros((len(starts), horizon, dim), dtype=recorded.dtype)
        if action_space == "delta-ee"
        else tile(states)
    )
    return {
        "zero delta" if action_space == "delta-ee" else "hold current state": hold,
        "repeat action at t": tile(recorded[starts]),
        "episode-mean action": np.broadcast_to(
            recorded.mean(axis=0), (len(starts), horizon, dim)
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
    # `stretch` and `stretch_bicubic` differ only in the resampler, and that is
    # the whole point of having both: measured against EVO1's own internal
    # resize, bicubic differs by 0.13-0.29/255 and BILINEAR by 3.3-13.1/255.
    # Grading a delta-EE checkpoint through `stretch` would therefore charge
    # the policy for a resampler the controller does not use.
    resampler = {"stretch": Image.BILINEAR, "stretch_bicubic": Image.BICUBIC}.get(fit)
    if resampler is None:
        raise ValueError(f"unknown image_fit {fit!r}")
    return np.asarray(
        Image.fromarray(arr).resize((target_hw[1], target_hw[0]), resampler),
        dtype=np.uint8,
    )


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


def _print_curve(curve, counts, truncated: bool, indent: str = "") -> None:
    """The error-against-horizon-offset bars, for one segment."""
    idx = sorted({0, 1, 4, 9, 24, len(curve) - 1})
    peak = np.nanmax(curve) if np.isfinite(curve).any() else 1.0
    for k in idx:
        if not np.isfinite(curve[k]):
            print(f"{indent}  k={k:<3}   (no samples reach this offset)")
            continue
        bar = "#" * int(round(40 * curve[k] / max(peak, 1e-9)))
        tail = f"  n={int(counts[k])}" if counts is not None and truncated else ""
        print(f"{indent}  k={k:<3} {curve[k]:8.4f}  {bar}{tail}")


def report(metrics: dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print(
        f"samples={metrics['n_samples']}  horizon={metrics['horizon']}  "
        f"action_dim={metrics['action_dim']}"
    )
    if metrics.get("truncated"):
        full = metrics["n_samples"] * metrics["horizon"]
        print(
            f"TRUNCATED SERIES: {metrics['valid_steps']}/{full} steps had ground truth.\n"
            "  Windows running off the end of the episode were kept and graded on the\n"
            "  steps that exist. This reaches the end of short episodes, but the numbers\n"
            "  are NOT comparable to a full-window run -- fewer high-k steps means a\n"
            "  lower score for reasons unrelated to the policy. Compare truncated to\n"
            "  truncated only."
        )

    segs = metrics.get("segments") or {}
    reps = int(metrics.get("repeats", 1))
    mixed = bool(metrics.get("mixed_units"))
    if mixed:
        print(
            "\nThis action space mixes units, so there is no whole-vector L2 to quote:\n"
            "  summing millimetres and radians under one square root gives a number with\n"
            "  no unit, dominated by whichever segment happens to have the larger scale\n"
            "  (here translation, by ~500x). Each segment is reported on its own below;\n"
            "  compare a segment only against the same segment."
        )

    for name, seg in segs.items():
        unit = f" {seg['unit']}" if seg["unit"] else ""
        head = f"\n[{name}]{unit and '  (' + seg['unit'] + ')'}" if mixed else ""
        if head:
            print(head)
        ind = "  " if mixed else ""
        print(f"{ind}mean L2 per step      : {seg['l2_mean']:.4f}{unit}")
        spread = seg.get("between_draw_spread")
        if reps > 1 and spread is not None and np.isfinite(spread):
            print(
                f"{ind}  between-draw spread : {spread:.4f}   <- sampler noise, averaged "
                f"over {reps} draws/window\n"
                f"{ind}     the mean above therefore carries a standard error of roughly "
                f"{spread / np.sqrt(reps):.4f} (spread/sqrt(N))."
            )
        print(f"{ind}  at offset k=0       : {seg['l2_first_step']:.4f}   <- immediate action")
        print(f"{ind}  p95                 : {seg['l2_p95']:.4f}")
        print(f"{ind}mean L2 by horizon offset:")
        _print_curve(
            seg["l2_by_offset"], metrics.get("n_by_offset"), bool(metrics.get("truncated")), ind
        )
        sbo = seg.get("spread_by_offset")
        if sbo is not None and np.isfinite(sbo).any():
            # The ratio is the diagnosis. ~1 means the draws disagree with
            # each other about as much as they disagree with the truth --
            # the future is genuinely ambiguous there, and a retrain cannot
            # remove it because the recorded action is itself one arbitrary
            # sample from that distribution. Well under 1 means the policy is
            # confidently wrong, which training CAN fix.
            print(f"{ind}draw disagreement vs. error, by offset:")
            for k in (0, 1, 4, 9, 24, 49):
                if k >= len(sbo) or not np.isfinite(sbo[k]):
                    continue
                err = seg["l2_by_offset"][k]
                ratio = sbo[k] / err if err else float("nan")
                print(
                    f"{ind}  k={k:<4d} spread {sbo[k]:7.4f}  error {err:7.4f}  "
                    f"spread/error {ratio:.2f}"
                )

        if metrics.get("baselines"):
            print(f"{ind}vs. predictors that never call the policy:")
            beaten = []
            for bname, b in metrics["baselines"].items():
                bseg = b["segments"][name]
                worse = seg["l2_mean"] >= bseg["l2_mean"]
                if worse:
                    beaten.append(bname)
                print(
                    f"{ind}  {bname:22s} mean L2 {bseg['l2_mean']:8.3f}  "
                    f"k=0 {bseg['l2_first_step']:7.3f}"
                    f"   -> {'policy WORSE' if worse else 'policy better'}"
                )
            if beaten:
                print(
                    f"{ind}  !! the policy does not beat {len(beaten)}/"
                    f"{len(metrics['baselines'])} of them. It is not predicting motion\n"
                    f"{ind}     better than a constant -- re-check the --key-map ordering "
                    "and the image geometry\n"
                    f"{ind}     before concluding the checkpoint is weak."
                )

        buckets = (metrics.get("by_position") or {}).get(name)
        if buckets:
            print(f"{ind}error by position in the episode (mean L2 per sample):")
            for b in buckets:
                if b["n"] == 0:
                    print(f"{ind}  [{b['lo']:.2f},{b['hi']:.2f})  no samples")
                    continue
                print(
                    f"{ind}  [{b['lo']:.2f},{b['hi']:.2f})  n={b['n']:<3d} "
                    f"mean L2 {b['l2_mean']:8.3f}   (starts reach {b['max_frac']:.2f})"
                )
            tail = buckets[-1]
            rest = [b for b in buckets[:-1] if b["n"]]
            if tail["n"] and rest:
                rest_mean = sum(b["l2_mean"] * b["n"] for b in rest) / sum(b["n"] for b in rest)
                delta = (tail["l2_mean"] - rest_mean) / rest_mean * 100
                print(
                    f"{ind}  last bucket vs the rest: {tail['l2_mean']:.3f} vs "
                    f"{rest_mean:.3f}  ({delta:+.0f}%)"
                )
                if delta > 25:
                    print(
                        f"{ind}  !! markedly worse at the end of the episode -- the failure "
                        "is the final subtask,\n"
                        f"{ind}     not the task as a whole. More demos OF THAT PHASE beat "
                        "more training steps."
                    )

    names = metrics.get("dim_names") or [f"joint {j}" for j in range(metrics["action_dim"])]
    print("\nper-dimension error (MAE / MSE):")
    for name, mae, mse in zip(names, metrics["mae_by_joint"], metrics["mse_by_joint"]):
        print(f"  {name:<10s} {mae:8.4f} / {mse:10.4f}")


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
        # Mirror the controller (`VLAController._image_sizes`): the declared
        # shape is whatever the checkpoint's BASE model advertised, while
        # `preprocess_image_size` is what this policy's own preprocessing
        # actually resizes to. Grading through the declared shape resamples
        # twice and charges the policy for detail the controller no longer
        # discards. Absent -> declared, for an older policy service or a
        # policy that does no resize of its own.
        consumed = specs.get("preprocess_image_size")
        sizes = (
            {k: (int(consumed[0]), int(consumed[1])) for k in image_keys}
            if consumed
            else {k: tuple(int(v) for v in specs["input_features"][k][1:]) for k in image_keys}
        )
        print(f"policy: {specs['policy_type']} on {specs['device']} ({specs['dtype']})")
        print(f"  image keys {image_keys}, horizon {horizon}, state_dim {state_dim}")
        action_space = resolve_action_space(
            args.action_space, int(specs["action_dim"]), state_dim
        )
        image_fit = args.image_fit or default_image_fit(action_space)
        if args.camera_resolution:
            print(f"image_fit: {image_fit}")

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
        starts = sample_starts(len(frames), horizon, args.stride, args.limit, args.min_overlap)
        if not starts:
            raise SystemExit(
                f"episode has {len(frames)} frames, shorter than the {horizon}-step "
                "chunk; nothing to compare"
            )
        draws = repeat_starts(starts, args.repeats)
        extra = f" x {args.repeats} draws each" if args.repeats > 1 else ""
        print(f"sampling {len(starts)} start frames (stride {args.stride}){extra}\n")

        camera_hw = None
        if args.camera_resolution:
            w, h = (int(x) for x in args.camera_resolution.lower().split("x"))
            camera_hw = (h, w)

        preds = []
        sampled_states = []
        for n, t in enumerate(draws, 1):
            frame = frames[t]
            images = {}
            for key in image_keys:
                arr = to_hwc_uint8(frame[key_map[key]])
                if camera_hw is not None or arr.shape[:2] != sizes[key]:
                    arr = through_camera_pipeline(arr, camera_hw, sizes[key], image_fit)
                images[key] = arr

            state = to_np(frame["observation.state"]).astype(np.float32)
            if state.shape[0] != state_dim:
                hint = (
                    "the dataset is not the one this checkpoint was trained on: "
                    "action_space='delta-ee' fixes the state at 9 dims "
                    "[x, y, z, r00, r01, r02, r10, r11, r12]"
                    if action_space == "delta-ee"
                    else "state_joint_indices is probably wrong"
                )
                raise SystemExit(
                    f"dataset observation.state has {state.shape[0]} dims, policy "
                    f"expects {state_dim} -- {hint}"
                )

            sampled_states.append(state)

            # Independent samples: reset so no per-episode policy state leaks
            # from the previous call and makes the curve look better than it is.
            # Per DRAW, not per window -- repeated draws of one window must be
            # as independent as two different windows are.
            await policy.do_command({"command": "reset"})
            cmd = build_infer_command(images, state, task, args.image_encoding, args.jpeg_quality)
            result = await policy.do_command(cmd)

            chunk = np.asarray(result["actions"]["rows"], dtype=np.float32)
            preds.append(chunk[:horizon])
            print(
                f"  [{n}/{len(draws)}] frame {t}  latency {result['latency_s']:.2f}s  "
                f"L2(k=0) {np.linalg.norm(chunk[0] - recorded[t]):.4f}",
                flush=True,
            )

        predicted = np.stack(preds)
        # Windows are padded to `horizon` and masked, rather than trimmed to a
        # ragged list: every statistic then stays a plain array op, and the
        # mask is the single place that knows a window ran off the end.
        truth = np.zeros((len(draws), horizon, predicted.shape[2]), dtype=np.float32)
        mask = np.zeros((len(draws), horizon), dtype=bool)
        for i, t in enumerate(draws):
            avail = min(horizon, len(recorded) - t)
            truth[i, :avail] = recorded[t : t + avail]
            mask[i, :avail] = True
        segs = segments_for(action_space, predicted.shape[2])
        metrics = compute_errors(predicted, truth, mask, segs)
        metrics["repeats"] = args.repeats
        metrics["dim_names"] = DIM_NAMES.get(action_space)
        # Spread and by-position are sliced per segment rather than taught about
        # segments: they take (samples, horizon, dim) arrays, so a column slice
        # is the whole adaptation, and there is one implementation either way.
        metrics["by_position"] = {}
        for name, start, stop, _unit in segs:
            metrics["segments"][name]["between_draw_spread"] = between_draw_spread(
                predicted[:, :, start:stop], args.repeats, mask
            )
            metrics["segments"][name]["spread_by_offset"] = spread_by_offset(
                predicted[:, :, start:stop], args.repeats, mask
            )
            metrics["by_position"][name] = errors_by_position(
                predicted[:, :, start:stop],
                truth[:, :, start:stop],
                draws,
                len(frames),
                mask=mask,
            )
        metrics["baselines"] = {
            name: compute_errors(pred, truth, mask, segs)
            for name, pred in naive_baselines(
                recorded, np.stack(sampled_states), draws, horizon, action_space
            ).items()
        }
        label = "dataset frames verbatim"
        if camera_hw is not None:
            label = f"through {args.camera_resolution} camera + image_fit={image_fit}"
        report(metrics, label)
        return 0
    finally:
        if robot is not None:
            await robot.close()


def resolve_action_space(declared: str, action_dim: int, state_dim: int | None) -> str:
    """Which action space the checkpoint speaks, from `--action-space`.

    `auto` infers it from the declared widths, because the delta-EE dataset
    contract fixes both: action 6, state 9. A joints checkpoint cannot collide
    with that -- its state is the selected joints plus at most one gripper
    channel, so a 6-dim action implies a 6- or 7-dim state, never 9.

    The inference is printed rather than silent. It picks the units every
    number in the report is quoted in, and a wrong guess there would relabel
    radians as millimetres.
    """
    if declared != "auto":
        return declared
    inferred = "delta-ee" if (action_dim == 6 and state_dim == 9) else "joints"
    print(
        f"action space: {inferred} (inferred from action_dim={action_dim}, "
        f"state_dim={state_dim}; pass --action-space to override)"
    )
    return inferred


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

    # Masking: a full mask must reproduce the unmasked result exactly, and
    # masked-out slots must not be able to influence anything.
    t4 = rng.normal(size=(5, 50, 6)).astype(np.float32)
    p4 = t4 + 0.3
    full = np.ones((5, 50), bool)
    a, b = compute_errors(p4, t4), compute_errors(p4, t4, full)
    for key in ("l2_mean", "l2_first_step", "l2_p95"):
        assert abs(a[key] - b[key]) < 1e-6, key
    assert np.allclose(a["mse_by_joint"], b["mse_by_joint"])
    assert not a["truncated"] and a["valid_steps"] == 250

    part = full.copy(); part[:, 30:] = False
    ref = compute_errors(p4[:, :30], t4[:, :30])
    trunc = compute_errors(p4, t4, part)
    assert abs(trunc["l2_mean"] - ref["l2_mean"]) < 1e-6
    assert np.allclose(trunc["mse_by_joint"], ref["mse_by_joint"], atol=1e-6)
    assert trunc["truncated"] and trunc["valid_steps"] == 150
    assert list(trunc["n_by_offset"][:3]) == [5, 5, 5] and trunc["n_by_offset"][40] == 0
    assert not np.isfinite(trunc["l2_by_offset"][40]), "an unreachable offset must be nan"

    # Garbage in masked-out slots must change nothing.
    poisoned = p4.copy(); poisoned[:, 30:] = 1e6
    assert abs(compute_errors(poisoned, t4, part)["l2_mean"] - trunc["l2_mean"]) < 1e-6

    try:
        compute_errors(p4, t4, np.zeros((5, 50), bool))
    except ValueError:
        pass
    else:
        raise AssertionError("an all-false mask must raise, not divide by zero")

    # The whole point: a short episode's last quarter becomes reachable.
    short_n = 187
    assert sample_starts(short_n, 50, 20, 0) == list(range(0, 137, 20))
    assert max(sample_starts(short_n, 50, 20, 0)) / (short_n - 1) < 0.75, "q4 unreachable, as documented"
    deep = sample_starts(short_n, 50, 20, 0, min_overlap=10)
    assert max(deep) / (short_n - 1) > 0.85, max(deep) / (short_n - 1)
    # errors_by_position must then populate the top bucket
    m5 = np.zeros((len(deep), 50), bool)
    for i, t in enumerate(deep):
        m5[i, : min(50, short_n - t)] = True
    zeros = np.zeros((len(deep), 50, 6), np.float32)
    got5 = errors_by_position(zeros, zeros, deep, short_n, mask=m5)
    assert got5[-1]["n"] > 0, "truncation must make the last bucket reachable"

    # --repeats: the expansion must be a pure duplication (repeats=1 changes
    # nothing at all), and the averaged metric must be the mean single-draw
    # error, not the error of the mean chunk -- the whole point of averaging
    # errors instead of chunks.
    base = sample_starts(400, 50, 20, 8)
    assert repeat_starts(base, 1) == base
    r3 = repeat_starts(base, 3)
    assert len(r3) == 3 * len(base)
    assert r3[:3] == [base[0]] * 3, r3[:3]      # draws of one window are adjacent
    assert sorted(r3) == sorted(base * 3)

    t6 = np.zeros((2, 50, 6), np.float32)
    draws_a, draws_b = t6 + 1.0, t6 - 1.0       # equal and opposite errors
    stacked = np.stack([draws_a[0], draws_b[0], draws_a[1], draws_b[1]])
    truth6 = np.stack([t6[0], t6[0], t6[1], t6[1]])
    single = compute_errors(stacked, truth6)["l2_mean"]
    assert abs(single - np.sqrt(6)) < 1e-5, single
    # averaging the CHUNKS first would have given exactly 0 here; it does not.
    assert abs(compute_errors(stacked.reshape(2, 2, 50, 6).mean(axis=1), t6)["l2_mean"]) < 1e-6

    # Between-draw spread: zero for identical draws, positive when they differ,
    # nan when there is nothing to compare.
    assert np.isnan(between_draw_spread(stacked, 1))
    assert between_draw_spread(np.repeat(draws_a, 2, axis=0), 2) == 0.0
    # two draws at +/-1 on 6 dims sit sqrt(6) from their own mean
    assert abs(between_draw_spread(stacked, 2) - np.sqrt(6)) < 1e-5, between_draw_spread(stacked, 2)
    # masked-out steps must not contribute, however wild they are
    m6 = np.ones((4, 50), bool); m6[:, 30:] = False
    wild = stacked.copy(); wild[0, 30:] = 1e6
    assert abs(between_draw_spread(wild, 2, m6) - np.sqrt(6)) < 1e-5

    # spread_by_offset: same quantity as between_draw_spread, resolved per k.
    assert np.isnan(spread_by_offset(stacked, 1)).all()
    assert not spread_by_offset(np.repeat(draws_a, 2, axis=0), 2).any()
    sbo = spread_by_offset(stacked, 2)
    assert sbo.shape == (50,)
    assert np.allclose(sbo, np.sqrt(6)), sbo[:3]
    # Its mean over offsets must agree with the pooled scalar, or the two
    # numbers in one report would contradict each other.
    assert abs(sbo.mean() - between_draw_spread(stacked, 2)) < 1e-5
    # An offset no draw reached is nan, not a silently-zero "they agree!".
    sbo_m = spread_by_offset(wild, 2, m6)
    assert np.allclose(sbo_m[:30], np.sqrt(6)) and np.isnan(sbo_m[30:]).all()

    # ---------------------------------------------------------------- delta-ee
    # Segmented metrics. The point is that a rotation-only error must be
    # visible: with a single whole-vector L2 it is not, which is the bug this
    # segmentation exists to fix.
    segs = segments_for("delta-ee", 6)
    assert segs == (("translation", 0, 3, "mm"), ("rotation", 3, 6, "rad")), segs
    assert segments_for("joints", 7) == (("action", 0, 7, ""),)
    assert default_image_fit("delta-ee") == "stretch_bicubic"
    assert default_image_fit("joints") == "pad"

    t7 = np.zeros((4, 50, 6), np.float32)
    p7 = t7.copy()
    p7[:, :, 3:] += 0.01                      # 0.01 rad on each rotation channel
    m7 = compute_errors(p7, t7, None, segs)
    assert m7["mixed_units"]
    assert m7["segments"]["translation"]["l2_mean"] == 0.0
    assert abs(m7["segments"]["rotation"]["l2_mean"] - 0.01 * np.sqrt(3)) < 1e-6
    # ... and the whole-vector number this replaces would have reported the
    # same 0.017 whether the error were radians or millimetres.
    p8 = t7.copy(); p8[:, :, :3] += 10.0      # 10 mm on each translation channel
    m8 = compute_errors(p8, t7, None, segs)
    assert m8["segments"]["rotation"]["l2_mean"] == 0.0
    assert abs(m8["segments"]["translation"]["l2_mean"] - 10.0 * np.sqrt(3)) < 1e-4
    # A single segment must reproduce the whole-vector statistics exactly.
    one = compute_errors(p8, t7, None, segments_for("joints", 6))
    for key in ("l2_mean", "l2_first_step", "l2_p95"):
        assert abs(one["segments"]["action"][key] - one[key]) < 1e-6, key
    assert not one["mixed_units"]
    # Masking must apply per segment too.
    m9mask = np.ones((4, 50), bool); m9mask[:, 25:] = False
    poisoned = p7.copy(); poisoned[:, 25:] = 1e6
    m9 = compute_errors(poisoned, t7, m9mask, segs)
    assert abs(m9["segments"]["rotation"]["l2_mean"] - 0.01 * np.sqrt(3)) < 1e-6

    # "don't move" under delta-ee is the zero delta, and must not be the state:
    # the state is 9-wide there, so tiling it would be a shape error.
    rec6 = rng.normal(size=(120, 6)).astype(np.float32)
    st9 = rng.normal(size=(4, 9)).astype(np.float32)  # 4 rows: reused by the report smoke test
    bd = naive_baselines(rec6, st9, [0, 10, 20], 50, "delta-ee")
    assert set(bd) == {"zero delta", "repeat action at t", "episode-mean action"}, set(bd)
    assert bd["zero delta"].shape == (3, 50, 6)
    assert not bd["zero delta"].any(), "the zero-delta baseline must be zero"
    # The old baseline would have raised here rather than scoring anything.
    try:
        compute_errors(naive_baselines(rec6, st9, [0, 10, 20], 50)["hold current state"],
                       np.zeros((3, 50, 6), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("a 9-dim state tiled as a 6-dim action must not score")

    # stretch_bicubic must be reachable and must NOT equal stretch -- if the two
    # resamplers agreed there would be no reason for the controller to have both.
    src = rng.integers(0, 256, (450, 800, 3), dtype=np.uint8)
    bic = through_camera_pipeline(src, None, (448, 448), "stretch_bicubic")
    bil = through_camera_pipeline(src, None, (448, 448), "stretch")
    assert bic.shape == bil.shape == (448, 448, 3)
    assert np.abs(bic.astype(int) - bil.astype(int)).mean() > 0.5, "resamplers agree?"
    try:
        through_camera_pipeline(src, None, (448, 448), "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown fit must raise")

    # Action-space inference: only the delta-EE contract's exact widths.
    assert resolve_action_space("auto", 6, 9) == "delta-ee"
    assert resolve_action_space("auto", 6, 7) == "joints"
    assert resolve_action_space("auto", 6, None) == "joints"
    assert resolve_action_space("joints", 6, 9) == "joints", "an explicit flag must win"
    assert resolve_action_space("delta-ee", 7, 7) == "delta-ee"

    # report() does a lot of string formatting over these dicts; run it once on
    # each shape so a KeyError or a format-spec mistake fails here rather than
    # after a twenty-minute run against a robot.
    for space, pred, truth in (("delta-ee", p7, t7), ("joints", p8[:, :, :6], t7)):
        sg = segments_for(space, 6)
        m = compute_errors(pred, truth, None, sg)
        m["repeats"] = 2
        m["dim_names"] = DIM_NAMES.get(space)
        m["by_position"] = {}
        for nm, a, b, _u in sg:
            m["segments"][nm]["between_draw_spread"] = between_draw_spread(
                pred[:, :, a:b], 2
            )
            m["by_position"][nm] = errors_by_position(
                pred[:, :, a:b], truth[:, :, a:b], [0, 1, 2, 3], 200
            )
        m["baselines"] = {
            n: compute_errors(pr, truth, None, sg)
            for n, pr in naive_baselines(
                rec6, st9 if space == "delta-ee" else rec6[:4], [0, 1, 2, 3], 50, space
            ).items()
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(m, f"self-check {space}")
        out = buf.getvalue()
        if space == "delta-ee":
            assert "[translation]" in out and "[rotation]" in out, out
            assert "mean L2 per step" in out
            assert "zero delta" in out, out
            assert "drx" in out, "delta-ee dims must not be labelled 'joint'"
        else:
            assert "[translation]" not in out and "joint 0" in out, out

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
    p.add_argument("--repeats", type=int, default=1, metavar="N",
                   help="draw each window N times and average the ERRORS (default 1). "
                        "The policy's sampler is unseeded, so one draw per window is "
                        "about half noise; N draws cut that by sqrt(N) and the report "
                        "quotes the spread it averaged away.")
    p.add_argument("--task", default=None, help="override the dataset's task string")
    p.add_argument("--key-map", action="append", metavar="DATASET_KEY=POLICY_SLOT",
                   help="feed a dataset image key into a policy slot; repeatable")
    p.add_argument("--camera-resolution", default=None, metavar="WxH",
                   help="simulate a real camera of this size before the controller's resize")
    p.add_argument("--action-space", default="auto",
                   choices=("auto", "joints", "delta-ee"),
                   help="what the checkpoint speaks; `auto` infers it from the declared "
                        "dims. Sets the error segmentation, the 'don't move' baseline, "
                        "and the default --image-fit.")
    p.add_argument("--image-fit", default=None,
                   choices=("pad", "stretch", "stretch_bicubic"),
                   help="controller-side resize to grade through (default: the "
                        "controller's own default for the action space -- pad for "
                        "joints, stretch_bicubic for delta-ee)")
    p.add_argument("--image-encoding", default="jpeg", choices=("jpeg", "png", "raw"))
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--min-overlap", type=int, default=0, metavar="N",
                   help="admit windows running off the episode end, needing N real steps "
                        "to grade (default 0 = full windows only). Reaches the tail of "
                        "short episodes; scores are NOT comparable to a full-window run.")
    args = p.parse_args()

    if args.self_check:
        return self_check()
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if not args.repo_id:
        p.error("--repo-id required (or use --self-check)")
    if bool(args.address) == bool(args.part_id):
        p.error("pass exactly one of --address (SDK) or --part-id (viam CLI)")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
