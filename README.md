# viam-vla-inference-service

A [Viam](https://www.viam.com) module that runs pre-trained
[LeRobot](https://github.com/huggingface/lerobot) Vision-Language-Action (VLA) policies
on a Viam machine. Primary targets are **SmolVLA** and **Evo-1**, both first-class
policies in LeRobot `main`; any other LeRobot-registered policy whose checkpoint is
self-describing (a `config.json` plus `model.safetensors` and processor files) should
load without per-policy code.

The module ships two `rdk:service:generic` resources that communicate over `DoCommand`:

- **`viam-labs:vla:policy`** — pure inference over a LeRobot checkpoint. Owns torch. No
  robot knowledge: one request carries images + joint state + a task string, the
  response carries an entire action chunk.
- **`viam-labs:vla:controller`** — the observation → inference → actuation loop. Owns
  the cameras, the arm, the gripper adapter, the action queue, and all safety clamping.
  Pure numpy; never imports torch.

Splitting them this way means the policy can be tested with no robot attached, and the
control loop can be developed and tested against a fake policy with no GPU and no torch
install at all.

## Setting it up

You configure both resources, in this order:

1. **Configure the policy** with a checkpoint — a local path, a Viam registry package, or
   a Hugging Face repo id. See [worked examples](#worked-config-examples).
2. **Wait for it to load** — poll `{"command": "status"}` until `state` is `ready`. A
   multi-GB checkpoint downloads in the background; `reconfigure()` returns immediately.
3. **Ask it what it needs** — `{"command": "specs"}` reports `image_feature_keys`,
   `state_dim`, and `action_dim`. These tell you what the controller's `cameras`,
   `state_joint_indices`, and `gripper` must line up with.
4. **Configure the controller** to match, then `{"command": "start"}`.

The two settings most likely to bite you are `state_units`/`action_units` (see
[Units](#units)) and the `cameras` map. If the arm moves but moves wrongly, check
`status.clamp_counts` first.

## Requirements

- **Python 3.12+.** LeRobot `main` requires it; a 3.11 venv fails to resolve.
- **Install size.** A full `uv sync --extra lerobot` venv — the one that actually runs
  inference — measured **824 MB on macOS** (Apple Silicon). Breakdown: torch 356 MB,
  cmake 124 MB, cv2 119 MB, transformers 44 MB, sympy 29 MB, viam-sdk 22 MB, numpy 22 MB.
  Linux resolves `torch+cu128`/`torchvision+cu128`, which bundle CUDA runtime libraries
  the macOS wheels omit, so the Linux figure will be **substantially larger** — measure
  on the actual target image (Jetson in particular) before committing to it. The base
  install (no `lerobot` extra — config parsing, the controller's pure-numpy code, the
  wire codec) has no torch dependency at all and measured **81 MB** on the same machine.

## `#policy` — `viam-labs:vla:policy`

Has no dependencies; `validate_config` always returns `([], [])`.

### Config

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_path` | string | — | Local checkpoint directory. Also how Viam registry delivery works: viam-server interpolates `${packages.ml_model.<name>}` into a real path before this module ever sees the config. Exactly one of `model_path` / `model_hub_id` is required. |
| `model_hub_id` | string | — | A Hugging Face Hub repo id, e.g. `lerobot/smolvla_base`. Downloaded once and cached under `$VIAM_MODULE_DATA/checkpoints` (falls back to a fresh, non-persistent temp directory with a logged warning if `VIAM_MODULE_DATA` is unset, which is the normal case on a dev workstation). |
| `model_revision` | string | `"main"` | Hub revision/tag/commit SHA. Ignored for `model_path`. |
| `hf_token_env` | string | — | The **name** of an environment variable holding a Hugging Face token — never the token itself. Machine configs are readable by anyone with fleet access. |
| `device` | `auto` \| `cuda` \| `mps` \| `cpu` | `"auto"` | `auto` tries cuda → mps → cpu, first available. Covers all three deployment targets with no per-machine edits. |
| `dtype` | `auto` \| `float32` \| `bfloat16` \| `float16` | `"auto"` | Parsed and validated but **not yet applied** to loaded weights — see [Limitations](#limitations). |
| `warmup_inferences` | integer, 0–100 | `2` | Throwaway inferences run on synthetic input at load, before `status` reports `ready`. First-call latency on a cold GPU is often several times steady-state. |
| `load_timeout_s` | number, 0.001–86400 | `1800` | Bounds checkpoint-resolve + load + warmup as a whole. A hang (stuck download, wedged deserialize) becomes `state: failed` with a message instead of `loading` forever. |
| `rtc.enabled` | boolean | `false` | Enables the RTC processor on the policy, if the checkpoint supports it (ignored with a logged warning otherwise). |
| `rtc.execution_horizon` | integer, 1–1000 | `10` | Mirrors lerobot's `RTCConfig.execution_horizon`. |
| `rtc.prefix_attention_schedule` | `linear` \| `exp` \| `ones` \| `zeros` | `"linear"` | Mirrors `RTCConfig`. |
| `rtc.max_guidance_weight` | number, >0–1000 | `10.0` | Mirrors `RTCConfig`. |
| `unused_image_features` | array of string | `[]` | Declared image feature keys (from the checkpoint's own `config.json`) to drop before validating/serving `image_feature_keys` — see below. |

`policy_type` is deliberately not configurable — it is read from the checkpoint's own
`config.json`. Making it a config field would only create a way to contradict the
checkpoint.

### `unused_image_features`

**Symptom:** your robot has two cameras, but `specs` reports three
`image_feature_keys` and the controller refuses to start until you map all three.

**Fix:** list the extra key here.

```json
{
  "model_path": "/opt/viam/checkpoints/smolvla-box-bot",
  "unused_image_features": ["observation.images.camera3"]
}
```

`specs.image_feature_keys` now reports the two you actually feed, and the controller
follows automatically. `specs.declared_image_feature_keys` still reports all three, so you
can see what the checkpoint originally claimed. A key the checkpoint never declared fails
at load, naming the real declared keys — so a typo does not quietly drop a camera.

**Why this happens.** A fine-tune of `lerobot/smolvla_base` inherits all three of its base
image features (`observation.images.camera1/2/3`) whatever the dataset actually had:
`lerobot/configs/train.py:285` refuses a `rename_map` unless a pretrained checkpoint is
given, so the base's `input_features` passes through verbatim.

**Why drop it rather than feed it something.** Dropping the key reproduces lerobot's own
behavior exactly — `modeling_smolvla.py:340-346` builds its image batch from whatever keys
are present and only raises if none are — so the policy sees precisely what it saw in
training. Filling the slot does not: measured on a real 2-camera fine-tune, black,
mid-gray, and a duplicate of another camera each shift the predicted 50-step chunk by
**5.7–7.8°** versus omitting it. No value brings that to zero.

> **Measuring this yourself requires a fixed torch seed.** `predict_chunk` does not seed,
> and smolvla's flow-matching sampler draws fresh noise per call, so two calls on
> byte-identical inputs differ by a median of ~15° (up to ~29°) on their own. An unseeded
> before/after comparison measures the sampler, not your change — and the numbers it
> produces look entirely plausible.

Two smaller notes:

- **`input_features` still lists the dropped key**, because it reports what the checkpoint
  declares. `image_feature_keys` is the only field that answers "which cameras do I feed"
  — enumerating from `input_features` reintroduces the problem this field solves.
- **A load-time warning names likely candidates** (a declared feature with no normalizer
  stats and no rename target) but never acts on them: an identity-normalized VISUAL
  feature can legitimately have no stats. Checkpoints with no rename map at all are
  skipped, since that test then flags every camera — including all three of
  `smolvla_base`'s own — and an absent rename map is exactly the case where this
  inheritance cannot have happened.

### DoCommand

Every request names its command in a `"command"` field. `infer`, `specs`, and `reset`
return an error unless `status` reports `ready`.

#### `status`

Whether the policy has finished loading. Always safe to call.

```json
{ "command": "status" }
```

```json
{ "state": "ready", "error": "" }
```

`state` is one of `idle`, `loading`, `ready`, `failed`. `error` is non-empty only when
`state` is `failed`.

#### `specs`

What the loaded checkpoint expects and produces. Call this before configuring a
controller — `image_feature_keys`, `state_dim`, and `action_dim` are what the
controller's `cameras`, `state_joint_indices`, and `gripper` have to line up with.

```json
{ "command": "specs" }
```

```json
{
  "policy_type": "smolvla",
  "action_dim": 6,
  "state_dim": 6,
  "n_action_steps": 50,
  "input_features": {
    "observation.state": [6],
    "observation.images.camera1": [3, 256, 256],
    "observation.images.camera2": [3, 256, 256]
  },
  "output_features": { "action": [6] },
  "image_feature_keys": [
    "observation.images.camera1",
    "observation.images.camera2"
  ],
  "declared_image_feature_keys": [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3"
  ],
  "supports_rtc": true,
  "rtc_enabled": false,
  "relative_actions": false,
  "device": "cuda",
  "dtype": "bfloat16"
}
```

#### `infer`

One observation in, one whole action chunk out. `images` must contain exactly the keys in
`specs.image_feature_keys` — no more, no fewer.

```json
{
  "command": "infer",
  "images": {
    "observation.images.camera1": {
      "encoding": "jpeg",
      "data": "<base64>",
      "height": 256,
      "width": 256,
      "channels": 3
    }
  },
  "state": { "values": [-6.8, -11.3, -46.0, 2.6, 53.0, -30.4] },
  "task": "open box flaps"
}
```

```json
{
  "actions": { "rows": [[-7.1, -10.3, -47.9, 1.7, 56.5, -30.3]] },
  "raw_actions": { "rows": [[0.06, 0.09, -0.12, -0.11, 0.03, 0.05]] },
  "latency_s": 0.32
}
```

`actions` has one row per step (`n_action_steps` of them) and is postprocessed, in the
checkpoint's own units, ready for the robot. `raw_actions` is the same chunk in
policy space — only an RTC caller needs it, to feed back as `prev_chunk_left_over`.
`encoding` may be `jpeg`, `png`, or `raw`; see `src/vla/wire.py`.

RTC callers add an optional `rtc` object:

```json
{
  "command": "infer",
  "images": { "...": {} },
  "state": { "values": [] },
  "task": "open box flaps",
  "rtc": {
    "inference_delay": 5,
    "prev_chunk_left_over": { "rows": [[0.06, 0.09, -0.12, -0.11, 0.03, 0.05]] }
  }
}
```

#### `reset`

Clears any cached backend state. Useful between episodes.

```json
{ "command": "reset" }
```

```json
{ "ok": true }
```

### Worked config examples

Loading happens in a **background task** — `reconfigure()` validates and returns
immediately, even for a multi-GB hub download. `infer`/`specs`/`reset` return an error
until `status` reports `ready`.

**1. Local checkpoint directory:**

```json
{
  "model_path": "/opt/viam/checkpoints/smolvla_base"
}
```

**2. Viam registry package** (`${packages.ml_model.<name>}` is substituted by
viam-server before this module sees the config — from here it's identical to a local
path):

```json
{
  "model_path": "${packages.ml_model.my-smolvla}"
}
```

**3. Hugging Face Hub:**

```json
{
  "model_hub_id": "lerobot/smolvla_base",
  "model_revision": "main",
  "hf_token_env": "HF_TOKEN",
  "device": "auto",
  "warmup_inferences": 3
}
```

## `#controller` — `viam-labs:vla:controller`

Dependencies: `policy_service`, `arm`, every resource named in `cameras`, plus the
gripper resource named in `gripper` (if any) — all computed by `validate_config` and
returned as required dependencies.

The controller tolerates a cold `policy_service`: `reconfigure()` never fails just
because the policy is still `loading` (the normal case on first boot, while a multi-GB
checkpoint downloads). It resolves lazily after the first `start`, polling `status` with
backoff for up to `policy_ready_timeout_s`.

### Config

| Field | Type | Default | Notes |
|---|---|---|---|
| `action_space` | `joints` \| `delta-ee` | `"joints"` | What the checkpoint speaks. `joints` is absolute joint angles written with `move_to_joint_positions`. `delta-ee` is a per-tick end-effector pose delta composed onto the measured `EndPosition` and written with `move_to_position` — see [`action_space`](#action_space). |
| `policy_service` | string | required | Name of the `viam-labs:vla:policy` dependency. |
| `arm` | string | required | Name of the Viam `arm` component to drive. |
| `cameras` | object (feature key → camera name) | required | Must cover every key the policy reports in `specs.image_feature_keys`. |
| `state_joint_indices` | array of integers | required under `action_space: "joints"` | Maps Viam joint order (base → end-effector) onto the state-vector position the checkpoint expects. Indices, not names — Viam joint names are not guaranteed to match LeRobot feature names. Rejected under `delta-ee`, which builds its state from `EndPosition`. |
| `gripper` | object | `{"type": "none"}` | Discriminated union — four variants, see below. |
| `task` | string | `""` | Default task instruction; overridable per `start` call. |
| `fps` | number | `10.0` | Control loop rate. |
| `mode` | `auto` \| `sequential` \| `async` \| `rtc` | `"auto"` | `auto` resolves to `sequential`. Switch to `async` when inference is slower than the motion a chunk buys — see [Performance](#performance). `rtc` is not implemented; configuring it fails at `start`. |
| `queue_threshold` | integer | derived (`n_action_steps - 1`) | `mode: "async"` only: refill fires once the queue has this many actions left. Leave unset — the derived default is the highest useful value. See [Performance](#performance). |
| `starvation_grace_ticks` | integer | `3` | How many consecutive bad ticks the loop tolerates before halting. Counts tick *failures* (when `stop_on_error` is `false`) and, under `mode: "async"`, *empty* ticks with inference still in flight. |
| `policy_ready_timeout_s` | integer | `600` | How long, in the background, `start` waits for a cold policy before giving up. |
| `state_units` | `degrees` \| `radians`, or an object under `delta-ee` | `"degrees"` | Unit of the state vector sent to the policy. `"normalized"` is not yet supported — see [Limitations](#limitations). Under `delta-ee` this is per-segment; see [Units](#units). |
| `action_units` | `degrees` \| `radians`, or an object under `delta-ee` | `"degrees"` | Unit the policy's action is expressed in. Per-segment under `delta-ee`. |
| `image_encoding` | `jpeg` \| `png` \| `raw` | `"jpeg"` | A debugging knob (JPEG artifacts vs. the training distribution), not a tuning one. |
| `jpeg_quality` | integer, 0–100 | `90` | |
| `image_fit` | `pad` \| `stretch` \| `stretch_bicubic` | `"pad"`, or `"stretch_bicubic"` under `delta-ee` | How a camera frame is resized onto the checkpoint's declared `(h, w)` whenever the frame's shape differs from it — see below. |
| `duration_warn_s` | number | `0.1` | Log a warning when observation assembly takes longer than this. |
| `stale_frame_warn_s` | number | `0.5` | Log a warning when a camera frame is older than this. |
| `safety.max_joint_delta_degs` | number | `8.0` | `joints` only. Per-tick clamp against the arm's *measured* position. Derived automatically from `max_vel_degs_per_sec` when that is set instead — see [Safety](#safety). |
| `safety.max_start_delta_degs` | number | `15.0` | `joints` only. `start` refuses outright if the first predicted action is farther than this from the current pose. |
| `safety.max_vel_degs_per_sec` | number | — | `joints` only. Operator-facing velocity knob — see [Safety](#safety). |
| `safety.joint_limits_degs` | array of `[min, max]` | — | `joints` only. One pair per action-vector dimension, **in action-vector order** (not Viam joint order) — see [Safety](#safety). |
| `safety.max_tcp_delta_mm` | number | `40.0` | `delta-ee` only. Per-tick ceiling on `norm(delta[:3])`. Derived from `max_tcp_vel_mms_per_sec` when that is set instead. |
| `safety.max_tcp_rot_delta_rads` | number | `0.12` | `delta-ee` only. Per-tick ceiling on `norm(delta[3:])`. |
| `safety.max_tcp_vel_mms_per_sec` | number | — | `delta-ee` only. Operator-facing tool-speed knob; the per-tick ceiling is `vel / fps`. |
| `safety.max_tcp_rot_vel_rads_per_sec` | number | — | `delta-ee` only. Same, for rotation. |
| `safety.stop_on_error` | boolean | `true` | Whether a failure producing the next action (camera read, observation assembly, the policy call) halts the loop, or is logged and the tick skipped. |

### `action_space`

`"joints"`, the default, is what everything else in this README describes: the policy
observes selected joint angles and emits absolute joint angles.

`"delta-ee"` runs a checkpoint trained on **end-effector pose deltas** — an EVO1
checkpoint from the `viam-labs/viam-sequence-to-lerobot` `action-space-delta-ee`
converter, for instance. The vector layouts are fixed by that dataset contract, not
by config:

| | dims | layout |
|---|---|---|
| `observation.state` | 9 | `x, y, z` in millimetres as `EndPosition` reports them, then the first two **rows** of the tool's 3x3 rotation matrix, row-major |
| `action` | 6 | `dx, dy, dz` in millimetres, then the **body-frame** relative rotation as an axis-angle vector in radians |

Each tick: read `get_end_position()`, build the 9-vector, infer, convert the 6-vector
into millimetres and radians, clamp its magnitude, compose it onto the *measured* pose,
and call `arm.move_to_position`. Everything else — the scheduler, `mode`, chunk pacing,
`starvation_grace_ticks`, `status` — behaves exactly as it does for `joints`.

```json
{
  "action_space": "delta-ee",
  "policy_service": "vla-policy",
  "arm": "xarm",
  "cameras": { "observation.images.top": "cam-top" },
  "fps": 10.0,
  "task": "open the box",
  "safety": { "max_tcp_vel_mms_per_sec": 400.0 }
}
```

Three keys are **rejected** rather than ignored under this action space, because each
would be inert and silence is the worst possible signal for a safety limit or a state
layout:

- `state_joint_indices` — the observation comes from `EndPosition`, not from joints.
- `gripper.type` other than `"none"` — the recording captured no gripper component and
  no seventh joint, so the checkpoint has no jaw channel. Drive the gripper out of band.
- every joint-space `safety` key — no enforcement path on `move_to_position`.

The rejection is symmetric: the `safety.max_tcp_*` keys are refused under `"joints"`.

**Unreachable poses do not kill the run.** `move_to_position` goes through the driver's
inverse kinematics, which can decline a kinematically unreachable target or a
singularity while the hardware is fine — and declines before commanding any motion.
Because the action is relative, the next tick simply recomposes from the measured pose,
so a refusal is logged and the tick skipped. Consecutive refusals are bounded by
`starvation_grace_ticks`, so a genuinely stuck arm still halts.

### `image_fit`

**Leave this at its default for your action space.** Most Viam cameras stream 16:9 while
checkpoints often declare a square resolution, so a frame nearly always has to be
reshaped — and the right reshape depends on what the *policy* does with it internally.

Under `action_space: "joints"` the default is `"pad"`, which scales by
`ratio = max(cur_w / target_w, cur_h / target_h)` and fills the remainder with black on
the **left and top** — matching lerobot's own `resize_with_pad`
(`lerobot/policies/common/vla_utils.py:219`), which is smolvla's training-time convention,
not the centered variant lerobot also ships for openpi. Aspect ratio and padding side both
end up where the checkpoint saw them in training.

`"stretch"` is the plain `Image.resize` this module used before, kept only so an existing
deployment can reproduce its old output byte-for-byte. It squashes a 16:9 frame into a
square, distorting every object's proportions in a way no checkpoint trained on. Measured
against training geometry: **~8.3°** divergence over a 50-step chunk versus **~3.2–4.1°**
for any aspect-preserving fit, about 2.5x worse. (Synthetic texture, so read those as an
ordering rather than a precise bound.)

`"stretch_bicubic"` is the default under `action_space: "delta-ee"`, and it exists because
**EVO1 does not pad.** `_batched_resize_01`
(`lerobot/policies/evo1/internvl3_embedder.py`) resizes straight to
`(image_size, image_size)` — 448 by default — with bicubic interpolation and
antialiasing, explicitly mirroring InternVL3's reference `Image.resize`. So EVO1's
training frames were the *unpadded* dataset frame squashed to a square; padding here
would hand it black bars that then get squashed along with the picture.

`"stretch"` is not a substitute, because it is BILINEAR. Measured against
`_batched_resize_01` on random frames, mean absolute difference per pixel:

| controller fit | difference from EVO1's own resize |
|---|---|
| `stretch_bicubic` | 0.13–0.29 / 255 |
| `stretch` | 3.3–13.1 / 255 |

An order of magnitude apart, which is why there is a third fit rather than a reuse. Both
halves of that comparison are pinned in
`tests/controller/test_observation_differential.py`.

Note this only bites when the camera's resolution differs from the checkpoint's declared
shape. When they match, the controller does not resample at all and parity is exact.


#### The declared resolution may not describe your cameras

The controller takes its wire resolution from `specs.input_features[key]`, which is only
as trustworthy as the checkpoint's own declaration. A fine-tune of `lerobot/smolvla_base`
inherits the base's `[3, 256, 256]` — the same inheritance behind
[`unused_image_features`](#unused_image_features) — so a checkpoint recorded from 1080p
cameras can still claim 256x256. Frames then get downsampled harder than training
downsampled them, on top of whatever `image_fit` does about aspect ratio.

`"pad"` is the fix for the geometry half of that, and it is the half that matters more.
The remaining resolution loss has no config knob: there is deliberately no override for
the declared shape, because the declaration is the right place to fix this.

**Train with `--policy.input_features=null` and no `--rename_map`** and the problem does
not arise: lerobot derives the features from the dataset, so `config.json` records your
cameras' real names and native resolutions. Verified on `viamrobotics/box-opener`, that
yields `observation.images.realsense_cam [3, 720, 1280]` and
`observation.images.camera_transform [3, 1920, 1080]` — two cameras at full resolution
instead of three at 256x256. The controller then sends full-resolution frames with no
extra downsample, and `image_fit` becomes close to a no-op because the aspect already
matches.

Two things to know if you do that:

- **Keep `image_encoding` on `jpeg`** (the default). A raw 1080p frame base64-encodes to
  roughly 8 MB and will hit gRPC message limits; at `jpeg_quality: 90` the same frame is a
  few hundred KB.
- **`fps` budget.** Encoding and shipping a 1080p frame per camera per tick is real work.
  Watch `status.measured_fps` against configured `fps`, and `duration_warn_s` will log
  when observation assembly overruns the tick.

### DoCommand

Every request names its command in a `"command"` field.

#### `start`

Begins the control loop. Returns immediately — it does not wait for the policy to be
ready or for the arm to move, so poll `status` to see what actually happened.

```json
{ "command": "start", "task": "open box flaps" }
```

```json
{ "ok": true }
```

`task` is optional and overrides the configured default for this run.

#### `stop`

Halts the loop. The arm stays where it is.

```json
{ "command": "stop" }
```

```json
{ "ok": true }
```

#### `status`

The one command to watch while a loop runs.

```json
{ "command": "status" }
```

```json
{
  "state": "running",
  "mode": "sequential",
  "queue_size": 43,
  "avg_latency_s": 0.32,
  "measured_fps": 9.94,
  "clamp_counts": { "delta": 0, "limit": 0, "gripper": 0 },
  "starved_ticks": 0,
  "last_error": ""
}
```

What to read it for:

- `state` — `idle`, `waiting_for_policy`, `running`, `stopped`, or `error`.
- `clamp_counts` — **the field to check first.** Counts should stay near zero. A steadily
  climbing `delta` or `limit` almost always means wrong units or wrong joint order, not a
  safety margin doing its job. See [Safety](#safety).
- `measured_fps` — should track configured `fps`. Well below it means inference is not
  keeping up; see [Performance](#performance).
- `starved_ticks` — ticks the loop held position with an empty queue. Only possible under
  `mode: "async"`, and a persistently rising value means `queue_threshold` is too low.
- `last_error` — non-empty after a failed tick, even when `stop_on_error` is `false` and
  the loop kept going.

### Gripper variants

A VLA emits one continuous gripper value per tick; Viam offers several ways to
carry it, at different fidelity.

**`arm_joint`** — recommended default. The gripper rides the arm's own joint vector (SO-100-style
drivers commonly expose it as the last joint), so read and write cost no extra round trip.
Value follows `action_units` like every other joint.

```json
{ "type": "arm_joint", "joint_index": 5 }
```

**`servo`** — bidirectional via `get_position()`/`move(angle)`, both `int` degrees (1°
resolution). Value is normalized `0.0`–`1.0` and mapped onto `[min_deg, max_deg]`.

```json
{ "type": "servo", "name": "grip-servo", "min_deg": 0, "max_deg": 90 }
```

**`do_command`** — proportional control for drivers that expose it through
`DoCommand` rather than the typed API. Requires that the driver implement
`{"get": true}` → `{<read_key>: number}` and `{"set": number}`.
`devrel:so101:gripper` and `viam:ufactory:gripper` both do.

`open_value` and `closed_value` are the driver's own native values at each
extreme and are **required** — there is no safe default, since a percentage
guess would silently saturate a raw-unit driver within the first percent of
its travel. They may run in either direction, which is how this variant
carries a driver whose scale counts *up* toward open.

```json
{ "type": "do_command", "name": "grip",
  "open_value": 95.0, "closed_value": 0.0,
  "write_args": { "wait": false } }
```

> `write_args: {"wait": false}` is worth setting for so-101 and is why
> `write_args` exists. Without it `devrel:so101:gripper` blocks up to ~2s per
> write waiting for the servo to settle — against a 100ms budget at 10 fps. It
> needs `devrel:so101-arm` recent enough to honor the flag (the gripper's
> `set`/`set_position` commands gained it alongside the arm's); an older
> version ignores the key and simply keeps blocking, so setting it is safe
> either way.

```json
{ "type": "do_command", "name": "grip", "read_key": "pos",
  "open_value": 840.0, "closed_value": 2.0 }
```

`name` is required, as for every variant with its own component. `read_key`
defaults to `"position"`. `write_args` must be an object, is merged into the
`set` command, and defaults to `{}`; it may not contain `get` or `set` — those
are the adapter's own protocol keys, and either one silently stops the gripper
tracking the policy.

A reading that normalizes more than 0.25 outside `[0, 1]` is refused rather
than clamped: small excursions are calibration slop (so-101 calls 95 fully
open while the servo reaches 100), but a large one means the configured
endpoints do not describe this driver's scale at all.

> The xarm config above is functionally correct but impractical in a 10 fps
> loop: that driver polls until the jaw settles, up to 10 seconds, with no
> way to opt out.

**`none`** — no gripper channel.

```json
{ "type": "none" }
```

`servo` and `do_command` hand the controller a normalized `0.0`–`1.0` value
(`0` = fully open), matching how LeRobot datasets typically encode a gripper
channel. `arm_joint` carries degrees, per `action_units`.

> **Not supported: the typed `Gripper` API's `get_current_inputs()`/
> `go_to_inputs()` pair.** It is a *frame-system* interface — one value per
> kinematic DOF, in radians or meters — not an aperture channel, so it only
> carries a usable aperture against a driver whose gripper model is
> **jointed**. `gripper.MakeModel` builds a zero-DOF model, and every driver
> we support (`devrel:so101:gripper` included) is zero-DOF, so the adapters
> that used it could never read an aperture at all. `do_command` is the
> variant that works, and normalizes honestly.

### Arm writes do not wait for settle

Every tick's arm command is sent with `extra={"wait": false}`. A VLA replaces
its setpoint on the next tick, so a driver that blocks until the arm
physically settles would spend the whole tick budget — 100 ms at `fps: 10` —
waiting for a target about to be superseded. `devrel:so101:arm` defaults to
waiting, and honours this flag to skip it.

`extra` is a free-form struct, so a driver that does not read `wait` ignores
it and behaves exactly as before. There is no config switch: if you need the
blocking behaviour back, that is a code change.

Note this is the *arm* channel only, and it is passed unconditionally. The
gripper's equivalent is opt-in: set `write_args: {"wait": false}` on a
`do_command` block, as the so-101 example above does. Both flags reach
`devrel:so101-arm` through the same helper on its side, so a version that
honors one honors the other.

### Full worked example

```json
{
  "policy_service": "vla-policy",
  "arm": "my-arm",
  "cameras": {
    "observation.images.top": "cam-top",
    "observation.images.wrist": "cam-wrist"
  },
  "task": "pick up the red block",
  "fps": 10.0,
  "mode": "sequential",
  "state_joint_indices": [0, 1, 2, 3, 4],
  "state_units": "degrees",
  "action_units": "degrees",
  "gripper": { "type": "arm_joint", "joint_index": 5 },
  "safety": {
    "max_start_delta_degs": 15.0,
    "max_vel_degs_per_sec": 60.0,
    "joint_limits_degs": [
      [-110, 110], [-90, 90], [-90, 90], [-90, 90], [-180, 180], [0, 90]
    ],
    "stop_on_error": true
  },
  "image_encoding": "jpeg",
  "jpeg_quality": 90
}
```

Note that `safety.max_joint_delta_degs` is **omitted**, not given some independent
number: it is derived from `max_vel_degs_per_sec / fps` (`60.0 / 10.0 = 6.0` deg/tick
here). If both are given, they must agree, or configuration fails loudly — see [Safety](#safety).

## Units

**This is the section that saves the most debugging time.** Viam arms always report and
accept **degrees** (`JointPositions.values`). A LeRobot checkpoint was trained on
whatever units the recording robot used, which is frequently *not* degrees — SO-100
datasets in particular are commonly recorded in radians, or as normalized joint
positions (this module supports the first two; see [Limitations](#limitations) for `normalized`).

`state_units` and `action_units` tell the controller which unit the checkpoint expects
on each side of inference — the arm's degrees are converted before being sent to the
policy, and the policy's output is converted back to degrees before being sent to the
arm.

**Worked example.** Suppose an SO-100 checkpoint was recorded through a driver that
reports and expects radians, but the Viam SO-100 driver reports and accepts degrees.
Add `state_units`/`action_units` to an otherwise ordinary controller config:

```json
{
  "policy_service": "vla-policy",
  "arm": "my-arm",
  "cameras": { "observation.images.top": "cam-top" },
  "state_joint_indices": [0, 1, 2, 3, 4],
  "state_units": "radians",
  "action_units": "radians"
}
```

The controller reads the arm's joint positions in degrees, converts to radians before
calling `infer`, and converts the returned action back to degrees before commanding the
arm — so a `move_to_joint_positions` call from a policy whose whole world was radians
still lands the arm at the position it actually meant.

Getting this wrong does not usually crash anything — it produces motion that is off by
a factor of ~57 (degrees vs. radians) or wildly rescaled (normalized vs. either), which
the safety layer's delta clamp will silently absorb into constant clamping. That is
exactly why `clamp_counts` exists: see [Safety](#safety) below.

### Per-segment units, under `action_space: "delta-ee"`

A joint vector is one quantity end to end, which is why a single unit string describes
it. A delta-EE vector is not: the 9-dim state is 3 lengths followed by 6 dimensionless
rotation-matrix entries, and the 6-dim action is 3 lengths followed by 3 angles. So
`state_units` and `action_units` become objects there, naming a unit per segment:

```json
{
  "state_units":  { "translation": "millimeters", "rotation": "unitless" },
  "action_units": { "translation": "millimeters", "rotation": "radians" }
}
```

Those are the defaults — what the `viam-sequence-to-lerobot` converter emits — so a
checkpoint built from it needs neither key. Set them only for a checkpoint recorded in
other units, e.g. `{"translation": "meters", "rotation": "degrees"}`.

`translation` accepts `millimeters` or `meters`; the action's `rotation` accepts
`radians` or `degrees`. The state's `rotation` accepts only `unitless`, because those
six numbers are direction cosines under every checkpoint — it is spelled out as a
one-value field rather than hidden so that writing `"radians"` there is reported as an
error instead of quietly ignored.

Passing the plain string form under `delta-ee` is rejected. One unit cannot describe a
vector that mixes millimetres with radians, and applying it to the whole thing is how a
translation gets multiplied by pi/180.

## Safety

Everything in this section describes `action_space: "joints"`; the delta-EE clamps are
below it.

Applied to every action, in this fixed order, before it reaches the arm:

1. **Reject NaN/Inf** in the action vector — fail the whole chunk, never try to clamp it.
2. **Dimension check** against the current measured arm state.
3. **Per-step delta clamp**, against the arm's **currently measured** position (not the
   last commanded one — so a stalled arm cannot accumulate an ever-growing command).
4. **Joint limit clamp**, from the optional `safety.joint_limits_degs`. The list is
   indexed **in action-vector order** — one `[min, max]` pair per action dimension, in
   the same order the policy emits and `state_joint_indices` defines — with a trailing
   gripper pair only when `gripper.type == "arm_joint"`. It is **not** indexed by Viam
   joint order. Config validation enforces
   `len(joint_limits_degs) == len(state_joint_indices) + (1 if gripper.type == "arm_joint" else 0)`.
   When absent, this layer is skipped and a warning is logged once at start, naming the
   arm driver as the sole limit authority.
5. **There is no driver-side kinematic ceiling.** `move_through_joint_positions` and
   `MoveOptions` — which would carry velocity/acceleration/TCP-speed limits — ship in no
   released `viam-sdk` (installed 0.80.0, the latest on PyPI, has only
   `move_to_joint_positions`, which takes no options). The velocity bound is instead
   enforced entirely by layer 3:

   ```
   max_joint_delta_degs = max_vel_degs_per_sec / fps
   ```

   `max_vel_degs_per_sec` is the operator-facing knob (reasoning in degrees/second is
   what a human actually does); the controller derives and logs the per-tick budget this
   implies at `reconfigure()` time. If both `max_vel_degs_per_sec` and
   `max_joint_delta_degs` are configured, they must agree (within floating-point
   tolerance) or configuration fails — silently preferring one over a contradictory
   other would hide an operator mistake instead of surfacing it. Acceleration and
   TCP-speed limiting are unavailable until the SDK ships the newer call;
   `safety.max_start_delta_degs` covers the large-initial-jump case those would
   otherwise soften.
6. **Every clamp is logged and counted**, split by layer, in `status.clamp_counts`
   (`delta` / `limit` / `gripper`). **Persistent clamping is the single most likely
   sign of wrong units or wrong joint order** — it is deliberately loud rather than
   silently "handled."

The trailing gripper channel (`servo`, `do_command` — everything except
`arm_joint`) is exempt
from the degree-shaped delta and limit clamps — it gets its own `[0, 1]` clamp instead,
counted separately as `clamp_counts["gripper"]`.

**First-move guard.** The first action of the first chunk can be arbitrarily far from
the arm's actual current pose — a policy handed an unfamiliar initial pose can output
anything. `start` refuses outright if that first action exceeds
`safety.max_start_delta_degs`. Refusing beats slowly moving the arm somewhere nobody
asked for.

### `action_space: "delta-ee"` — Cartesian clamps

`move_to_position` carries no notion of "this delta represents one 100 ms tick". It takes
a pose and drives to it at whatever speed the driver chooses. So a policy emitting an
out-of-distribution delta would have it executed at full driver speed, and bounding the
delta is the only thing standing between a bad chunk and a fast, large tool motion —
exactly the role `max_joint_delta_degs` plays above.

Two per-tick ceilings, on the magnitude of each half of the action:

| | config key | default | per-tick statistics of the reference training data |
|---|---|---|---|
| translation, `norm(delta[:3])` | `safety.max_tcp_delta_mm` | 40 mm | median 9.31 mm, p99 28.40 mm, max 96.83 mm |
| rotation, `norm(delta[3:])` | `safety.max_tcp_rot_delta_rads` | 0.12 rad | median 0.0142, p99 0.0807, max 0.3246 |

Each default sits about 1.4x above the p99, so in-distribution motion never clamps and
`clamp_counts` stays a real signal — and below the largest single tick in the recording,
which at 10 fps would be 968 mm/s of tool travel. That second half is deliberate: those
extremes are a handful of frames out of 34,670, and a policy emitting one every tick is
out of distribution rather than in a hurry.

As on the joints path, the operator-facing knob is a **velocity**
(`safety.max_tcp_vel_mms_per_sec`, `safety.max_tcp_rot_vel_rads_per_sec`) and the
per-tick ceiling is derived as `vel / fps`. Setting both is allowed only if they agree.
`reconfigure()` logs the ceiling and the tool speed it implies, because
`move_to_position` accepts no speed argument and there is nowhere else to read it off.

Both clamps **scale** the 3-vector by a single factor rather than clipping it
component-wise, so the commanded direction is preserved and only the step length
shortens. Component-wise clipping would turn a `(100, 10, 0)` mm step into `(40, 10, 0)`
— a different heading than the policy asked for. Counted as `clamp_counts["translation"]`
and `clamp_counts["rotation"]`, reported by `status` exactly like the joints counters.

There is no start-delta clamp here. A delta-EE action is already a delta, so the first
tick's magnitude is the same quantity every later tick's clamp measures.

**Joint limits are the arm driver's job on this path.** `safety.joint_limits_degs` has no
Cartesian analogue and is rejected under this action space rather than accepted and
ignored; out-of-range and unreachable targets are refused by the driver's own inverse
kinematics when `move_to_position` is called. Note that `move_to_position` is the arm
*component* method, not the motion service, so there is no obstacle avoidance on this
path either.

## Performance

Measured on Apple Silicon (MPS), `lerobot/smolvla_base`:

| | |
|---|---|
| `predict_chunk` latency | **~5.3 s** |
| `n_action_steps` | 50 |
| Motion a chunk buys at `fps: 10` | 5.0 s |

**Inference is slower than the motion it produces on Apple Silicon.** `mode: "sequential"`
stalls the arm for ~5.3 s between chunks — it moves roughly half the wall-clock time, in
5-second freezes.

**`mode: "async"` is the recommended mode whenever inference latency approaches or
exceeds chunk duration**, exactly the case measured here. It overlaps execution with
inference instead of stalling: the current chunk keeps running while the next one infers
in the background, merged in append mode once it lands. Duty cycle goes from ~50% (in
multi-second freezes) to ~94% (in ~0.3 s hitches at each chunk boundary) at these
measured numbers. RTC cannot do this job instead — RTC needs `delay < chunk_length` to
function at all, and here `delay` (~53 steps) exceeds `chunk_length` (50 steps), so RTC
would discard the entire chunk on every merge (permanent starvation). See
`AsyncScheduler`'s docstring in `src/vla/controller/scheduler.py` for the full
concurrency contract and the discontinuity-at-chunk-boundary tradeoff it accepts in
exchange.

That tradeoff is why `async` is explicit opt-in rather than a default: it is unambiguously
better than `sequential` whenever this latency relationship holds, but a deployment
running comfortably inside its chunk budget has no reason to accept a seam it does not
need. Other levers still apply if you want to avoid the stall/hitch tradeoff entirely:
lower `fps` (a chunk buys more wall-clock time per inference), a smaller/faster
checkpoint, or fewer denoising steps. **For anything resembling a live demo, the x86+CUDA
target remains the practical answer** — the numbers above are Mac-specific, and `async`
narrows the gap without pretending to close it as well as faster hardware would.

### Tuning `queue_threshold` — the single most actionable knob in `mode: "async"`

`queue_threshold` decides *when* the background refill fires (once the queue has this
many or fewer actions left), which decides how much runway the refill gets before the
queue actually drains. A threshold picked too low silently forfeits most of the overlap
benefit — the loop still "works," it just spends a third of its time holding position,
and without a diagnostic it looks indistinguishable from "the policy is slow." Measured
throughput delivering 150 actions at latency ≈ chunk duration × 1.06 (`n_action_steps:
50`):

| `queue_threshold` | wall time | actions/s | vs. `sequential` |
|---|---|---|---|
| — (`sequential`) | 3.20 s | 46.9 | baseline |
| 10 | 2.99 s | 50.2 | +7% |
| 30 (the old fixed default) | 2.55 s | 58.7 | +25% |
| 49 (`n_action_steps - 1`) | 2.16 s | 69.4 | +48% |

The runway a threshold buys is `queue_threshold` ticks; the runway inference *needs* is
`ceil(observed_latency × fps)` ticks. Because the queue can only ever hold at most
`n_action_steps - 1` actions before a refill must have already been requested, the
highest `queue_threshold` can ever usefully be is `n_action_steps - 1` — which is exactly
why that is the derived default (see the config table above) rather than a fixed number:
a fixed default is right for at most one checkpoint's chunk length and measurably wrong
for every other one. Override it explicitly only if you have a specific reason to trade
some of that throughput back for fresher observations at each chunk boundary (a lower
threshold fires the refill later, off a more recent — but riskier — observation).

**Two diagnostics exist so a bad threshold is never silent:**

- `AsyncScheduler` tracks its own recent inference latencies and, once it has a stable
  reading, logs a `WARNING` **once** (never every tick) if `queue_threshold <
  ceil(observed_latency × fps)` — naming the observed latency, the configured and
  required thresholds, and the consequence ("the arm will hold position for ~N tick(s)
  per chunk"), with the concrete remedy (raise `queue_threshold`, lower `fps`, or reduce
  the policy's `num_steps`).
- `status.starved_ticks` is a running total (the same shape as `clamp_counts`) of ticks
  the loop has held position because the queue was empty and a background inference was
  still in flight — distinct from the *consecutive*-run bound that escalates to a halt at
  `starvation_grace_ticks`. A deployment that never escalates can still have a
  persistently nonzero `starved_ticks`, which is exactly the silent-degradation case this
  counter exists to surface.

## Limitations

- **`mode: "rtc"` is not implemented.** `ActionQueue` supports RTC mode and is
  differentially tested against upstream, but `RTCScheduler`, two-delay bookkeeping, and
  `prev_chunk_left_over` wiring are a follow-up plan — deferred until CUDA latency is
  measured, since RTC needs `delay < chunk_length` to function at all (on the measured
  Apple Silicon numbers, `delay > chunk_length`, so it could not be validated there even
  if implemented). Configuring `mode: "rtc"` raises at `start`. `mode: "async"` (see
  Performance) is the currently-shipped answer to slow inference; it is a plain overlap,
  not RTC, and does not smooth the seam at each chunk boundary the way RTC eventually
  would.
- **`state_units`/`action_units: "normalized"` is unsupported.** It needs a source of
  per-joint min/max (dataset stats vs. explicit config) that is an open design question.
  Only `degrees` and `radians` are accepted under `action_space: "joints"`.
- **`action_space: "delta-ee"` has no gripper channel.** The reference recording captured
  the arm's `EndPosition` and six joint values only — no gripper component, no seventh
  joint — so the checkpoint cannot command jaw aperture and a non-`none` `gripper` block
  is rejected. Drive the gripper out of band.
- **Joint limits are not enforced by this service under `action_space: "delta-ee"`.**
  `safety.joint_limits_degs` has no Cartesian analogue on the `move_to_position` path;
  the arm driver's own inverse kinematics refuses out-of-range and unreachable targets.
  `move_to_position` is the arm component method rather than the motion service, so
  there is no obstacle avoidance on that path either.
- **Relative-action checkpoints are refused under RTC**, not silently mishandled — RTC's
  prefix would need re-anchoring against the cached raw state that this module does not
  yet implement, and applying guidance in the wrong coordinate frame would produce
  plausible-looking but wrong motion. Sequential mode is unaffected.
- **No driver-side velocity/acceleration ceilings.** `move_through_joint_positions` (the
  only method that consumes `MoveOptions`) ships in no released `viam-sdk`. The velocity
  bound lives entirely in the safety layer's delta clamp (see [Safety](#safety)); acceleration and
  TCP-speed limiting have no enforcement path at all right now.
- **`dtype` is parsed and validated but not applied.** Casting weights with
  `policy.to(dtype=...)` breaks inference on at least one target (the deserialized
  `DeviceProcessorStep` has `float_dtype=None` and keeps emitting float32 regardless).
  The checkpoint's own dtype is used as-is; a non-`"auto"` setting logs a warning rather
  than silently doing nothing. (`lerobot/smolvla_base` runs `bfloat16` natively, which is
  why this has not blocked phase-1 testing.)

## Development

```bash
mise run test        # fast suite: no torch, no network, seconds
mise run test-all     # everything, including integration/differential (needs the lerobot extra)
uv sync --extra lerobot  # required once before test-all, or before running integration/differential directly
```

- `mise run test` runs `pytest -m 'not integration and not differential'` — this is the
  suite that must stay green (and torch-free) in a base `uv sync`, with no `lerobot`
  extra installed at all. It is the payoff for the `PolicyBackend` seam: everything
  except two files (`lerobot_backend.py` and the integration tests) never imports torch.
- `mise run test-all` runs the full suite, including `@pytest.mark.integration` (a real
  `lerobot/smolvla_base` checkpoint against a fake robot — no hardware) and
  `@pytest.mark.differential` (this module's numpy `ActionQueue` port checked against
  upstream lerobot's torch implementation, on both the pinned SHA and `main`). Both need
  `uv sync --extra lerobot` first.
- `mise run build` / `mise run package` build the wheel and the deployable
  `module.tar.gz` (`meta.json` + `run.sh` + `setup.sh` + the wheel — no `docs/`, so
  internal specs and plans never leak to the module registry).
