# VLA Inference Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Viam Python module with two generic services — `viam-labs:vla:policy` (pure VLA inference over a LeRobot checkpoint) and `viam-labs:vla:controller` (observation → inference → arm actuation loop) — running SmolVLA and Evo-1.

**Architecture:** Two `rdk:service:generic` resources in one module. `policy` is stateless: one request carries images + joint state + a task string, the response carries an entire action chunk. `controller` owns the cameras, the arm, the action queue, and all safety clamping, and never imports torch. A `PolicyBackend` ABC separates the resource layer from LeRobot, so everything except two integration tests runs without torch.

**Tech Stack:** Python 3.12+, `viam-sdk`, `numpy`, `Pillow`, `pytest`, `uv`, `mise`. LeRobot pinned to git SHA `ff7cc3de1de830f5f3276918a013d04bdf9ea4be` as an optional extra.

**Spec:** `docs/superpowers/specs/2026-08-06-vla-inference-service-design.md`

**Scope:** Spec phases 1–2. Delivers a working sequential control loop. `RTCScheduler` (phase 4) is deferred to a follow-up plan — but `ActionQueue` is fully ported here, both RTC and non-RTC modes, because it is a single class and the differential test covers both paths at once.

---

## Background for the implementer

You likely have not worked with either of these systems. Three things will save you time.

**Viam modules.** A module is a subprocess that viam-server starts and talks to over gRPC. You subclass a resource type, implement `new()`, `validate_config()`, `reconfigure()`, and `do_command()`, then register the model. `validate_config()` is a **classmethod** that returns a tuple of `(required_dependencies, optional_dependencies)` as lists of resource-name strings, and raises on invalid config. `reconfigure()` receives resolved dependencies and **must return quickly** — viam-server can time out otherwise.

**LeRobot policies.** A checkpoint directory is self-describing. The real
`lerobot/smolvla_base` contains `config.json`, `model.safetensors`,
`policy_preprocessor.json`, `policy_postprocessor.json`, and per-step normalizer
`.safetensors`. You never need the training dataset at inference time. Loading is
`get_policy_class`/`make_policy` plus `make_pre_post_processors(cfg, pretrained_path=...)`.
(The processor filenames are *not* `preprocessor_config.json` — they come from
`POLICY_PREPROCESSOR_DEFAULT_NAME` in `lerobot/utils/constants.py:60`.)

**Action chunking.** A VLA does not emit one action — it emits a *chunk* of N future actions. The robot executes them in sequence and re-infers before running out. That queue is the `ActionQueue` you will port.

**Units are the main correctness hazard in this project.** Viam arms report and accept **degrees**. LeRobot checkpoints were trained in whatever units the recording robot used. Nearly every bug you hit will be a unit or joint-order mismatch, which is why the safety layer logs loudly whenever a clamp engages.

**Reference sources on this machine** — read them rather than guessing:
- LeRobot: `/Users/nick.hehr/src/lerobot`
- Viam Python SDK: `/Users/nick.hehr/src/viam-python-sdk`

Use @superpowers:test-driven-development throughout. Every task is red → green → commit.

### Standing test requirements

Every task's test list below is a **floor, not a ceiling.** Three separate review
rounds on this project found the specified suites substantially weaker than they
looked (43% mutant survival on Task 3), always for the same reasons. Before
committing any task, add tests covering these unless they genuinely do not apply:

1. **Assert valid values are accepted, not only that invalid ones are rejected.**
   This is the blind spot that keeps recurring. A suite that only tests rejection
   stays green when an enum tuple loses a member — so a one-character typo in
   `DTYPES` would reject every GPU deployment silently. Parametrize over every
   member of every allowed set and assert the value lands on the parsed result.

2. **Assert defaults, not just overrides.** A default that silently changes is
   invisible to a suite that only exercises the override path.

3. **Every field must be asserted somewhere.** A field nothing reads can be
   hardcoded to `None` by a mutation and no test notices.

4. **Numbers arrive from protobuf `Struct` as doubles.** Anything parsed from a
   Viam config or a `DoCommand` payload gets `2.0`, never `2`. Test the float
   form of every integer field, and test that a fractional value is rejected
   rather than silently truncated.

5. **Error paths must raise this module's own exception type**, not a bare
   `ValueError`/`TypeError`/`AttributeError` escaping from a builtin. Callers
   write `except ConfigError` / `except WireError`; anything else walks past them.

6. **Fixtures must be able to fail.** Non-square images catch transposition that
   square ones cannot; a shape used in one test should differ from the shape
   hardcoded anywhere else.

7. **Never parametrize a test off the constant it is testing.** Writing
   `@pytest.mark.parametrize("device", DEVICES)` looks like it covers every
   device, but shrinking `DEVICES` shrinks the test too — the mutant becomes
   invisible and the suite stays green. Hardcode the expected literals in the
   test file so the test and the implementation can actually disagree.

Where mutation testing is cheap, run it — deliberately break a branch and confirm
a test goes red. A test that cannot fail is not evidence.

---

## File Structure

```
src/
  vla/
    __init__.py
    main.py                            entrypoint (must live under vla/ to be packaged)
    wire.py                            shared wire codec (images, matrices, vectors)
    config_util.py                     ConfigError + as_int/as_float/as_bool/as_str/as_choice
    policy/
      __init__.py
      config.py                        PolicyConfig parse + validate
      resolver.py                      checkpoint resolution (local path | HF hub)
      backend.py                       PolicyBackend ABC + PolicySpecs dataclass
      prefix.py                        numpy port of _normalize_prev_actions_length
      fake_backend.py                  FakePolicyBackend (deterministic, no torch)
      lerobot_backend.py               LeRobotBackend (lazy lerobot import)
      service.py                       viam-labs:vla:policy resource
    controller/
      __init__.py
      config.py                        ControllerConfig parse + validate
      units.py                         degrees/radians conversion
      action_queue.py                  numpy port of lerobot ActionQueue
      gripper.py                       5 gripper adapters
      safety.py                        6-layer safety clamp
      observation.py                   observation assembly + encoding
      scheduler.py                     ChunkScheduler ABC + SequentialScheduler
      service.py                       viam-labs:vla:controller resource
.github/workflows/
  differential.yml                     runs the port-vs-upstream test on pinned + main
tests/
  __init__.py                          package markers; policy/ and controller/ too
  fakes.py                             fake arm/camera/gripper/servo
  test_wire.py
  test_integration_full_loop.py        real checkpoint + fake robot
  policy/
    test_config.py  test_resolver.py  test_prefix.py
    test_fake_backend.py  test_service.py
    test_lerobot_backend_integration.py
  controller/
    test_config.py  test_units.py  test_action_queue.py
    test_action_queue_differential.py
    test_gripper.py  test_safety.py  test_observation.py
    test_scheduler.py  test_service.py
```

Why this split: `policy/` and `controller/` never import each other — they communicate only through the wire format in `wire.py`. Everything in `controller/` is pure numpy and can be tested with no model present. `lerobot_backend.py` is the only file that imports `lerobot`, and it does so lazily inside methods.

`config_util.py` holds `ConfigError` and the coercion helpers for **both** config
modules. It exists because protobuf `Struct` delivers every number as a double,
so every numeric field needs the same "accept `2.0`, reject `2.5`, reject `True`,
reject NaN" treatment — and because two separate `ConfigError` classes would mean
`validate_config` has to catch both or let one escape unhandled.

---

## Task 1: Project scaffold

> **Implemented** — `f108749`, `d058c64`. Code below is re-synced with the
> hardening the review added (wheel-only build, script guards, `build.setup`).

**Files:**
- Create: `pyproject.toml`, `mise.toml`, `run.sh`, `setup.sh`, `meta.json`, `.gitignore`,
  `.python-version`
- Create: `src/vla/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "viam-vla-inference-service"
version = "0.1.0"
description = "Run pre-trained LeRobot VLA policies on Viam machines"
requires-python = ">=3.12"
dependencies = [
    "viam-sdk>=0.80.0",
    "numpy>=2.0.0,<2.3.0",
    "pillow>=10.0.0",
    "typing-extensions>=4.12.2",
]

[project.optional-dependencies]
lerobot = [
    "lerobot[smolvla,evo1] @ git+https://github.com/huggingface/lerobot@ff7cc3de1de830f5f3276918a013d04bdf9ea4be",
]

[dependency-groups]
# pytest-asyncio >=1.0: asyncio_default_test_loop_scope only exists from 0.26.
# On older versions pytest merely warns about the unknown key, and the
# module-scoped async fixture in Task 19 then fails with a cross-loop error.
dev = ["pytest>=8.0.0", "pytest-asyncio>=1.0"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "module"
asyncio_default_test_loop_scope = "module"
testpaths = ["tests"]
markers = [
    "integration: requires a real checkpoint and torch (deselect with '-m \"not integration\"')",
    "differential: requires lerobot installed to compare against upstream",
]

[tool.hatch.metadata]
# Required: hatchling refuses to build ANY project whose metadata declares a
# PEP 508 direct reference (the `lerobot @ git+https://...` pin) without this,
# and it validates all declared extras during a plain `uv sync` even though
# that sync never installs them.
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/vla"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

`requires-python = ">=3.12"` is not arbitrary — LeRobot `main` requires it, and a 3.11 venv fails to resolve.

`readme` is deliberately omitted: hatchling raises `OSError: Readme file does not exist` and aborts `uv sync` when the key points at a file that does not exist yet. Task 20 adds the key together with the file.

`packages = ["src/vla"]` means **everything shipped must live under `src/vla/`**, including the entrypoint — hence `src/vla/main.py` in Task 18, not `src/main.py`. A wheel built from this config contains only `vla/`, so a top-level `main.py` would be missing at runtime and `run.sh` would fail with `ModuleNotFoundError`.

- [ ] **Step 2: Create `mise.toml`**

```toml
[tasks.build]
# `--wheel`: uv otherwise also emits an sdist, and the sdist contains docs/ —
# packaging it would publish our internal specs and plans to the registry.
# The clean is inline rather than `depends = ["clean", ...]` because mise runs
# dependencies in parallel and a clean task would race the build.
run = "rm -rf dist && uv build --wheel"

[tasks.clean]
run = "rm -rf dist module.tar.gz"

[tasks.test]
run = "uv run pytest -m 'not integration and not differential' -v"

[tasks.test-all]
run = "uv run pytest -v"

[tasks.package]
# Name the scripts explicitly: `*.sh` would absorb any future dev or CI script
# into the published artifact. `dist/*.whl` rather than `dist` so uv's
# auto-generated dist/.gitignore stays out of the tarball.
run = "tar -czf module.tar.gz meta.json run.sh setup.sh dist/*.whl"
depends = ["build"]
```

- [ ] **Step 3: Create `run.sh` and `setup.sh`**

Both scripts run unattended on deployed robots, so every failure has to be legible
in viam-server's log rather than a bare "no such file or directory" restart loop.

`run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# `:?` rejects empty as well as unset — an empty prefix would silently target /venv.
VENV_NAME="${VIAM_MODULE_DATA:?must be set by viam-server}/venv"
PYTHON="$VENV_NAME/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "run.sh: no venv at $VENV_NAME — first_run (setup.sh) did not complete" >&2
  exit 1
fi

echo "Starting module..."
exec "$PYTHON" -m vla.main "$@"
```

`setup.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_NAME="${VIAM_MODULE_DATA:?must be set by viam-server}/venv"
# Prepend, so a freshly installed uv wins over a stale system one.
export PATH="$HOME/.local/bin:$PATH"

if [ ! "$(command -v uv)" ]; then
  if [ ! "$(command -v curl)" ]; then
    echo "curl is required to install uv."
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# The lerobot extra is a git+https direct reference, so uv shells out to git.
# A minimal robot image may not have it.
if [ ! "$(command -v git)" ]; then
  echo "git is required to install the lerobot extra."
  exit 1
fi

uv venv --python 3.12 "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Two traps here. First, "./dist/"*.whl[lerobot] does not work: bash reads
# [lerobot] as a glob character class, matches nothing, and hands uv the
# literal string. Second, `ls | head -1` sorts lexically, so with 0.1.0 and
# 0.2.0 both present it would silently install the OLDER wheel.
shopt -s nullglob
WHEELS=(./dist/*.whl)
shopt -u nullglob
if [ ${#WHEELS[@]} -ne 1 ]; then
  echo "setup.sh: expected exactly 1 wheel in ./dist, found ${#WHEELS[@]}" >&2
  exit 1
fi
uv pip install "${WHEELS[0]}[lerobot]" -q
```

- [ ] **Step 4: Create `meta.json`**

```json
{
  "$schema": "https://dl.viam.dev/module.schema.json",
  "module_id": "viam-labs:vla",
  "visibility": "private",
  "url": "https://github.com/viam-labs/viam-vla-inference-service",
  "description": "Run pre-trained LeRobot VLA policies (SmolVLA, Evo-1) on Viam machines",
  "build": {
    "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && curl -fsSL https://mise.run | sh",
    "build": "export PATH=\"$HOME/.local/bin:$PATH\" && mise run package",
    "path": "module.tar.gz",
    "arch": ["linux/arm64", "linux/amd64"]
  },
  "models": [
    {
      "api": "rdk:service:generic",
      "model": "viam-labs:vla:policy",
      "short_description": "Pure VLA inference over a LeRobot checkpoint",
      "markdown_link": "README.md#policy"
    },
    {
      "api": "rdk:service:generic",
      "model": "viam-labs:vla:controller",
      "short_description": "Observation-inference-actuation loop driving a Viam arm",
      "markdown_link": "README.md#controller"
    }
  ],
  "entrypoint": "./run.sh",
  "first_run": "./setup.sh"
}
```

`build.setup` is required: Viam's cloud-build containers ship neither `mise` nor
`uv`, so without it `build` fails with `command not found`. The `export PATH` in
`build` is also required — both installers write their PATH line into
`~/.bashrc`/`~/.profile`, which the non-interactive shell running `build` never
sources.

- [ ] **Step 5: Make the scripts executable**

Run: `chmod +x run.sh setup.sh`
`meta.json` points `entrypoint` at `./run.sh`, so a non-executable file fails at deploy time, not build time.

- [ ] **Step 6: Create `.gitignore` and package markers**

`.gitignore`:
```
__pycache__/
*.py[cod]
.venv/
dist/
module.tar.gz
.pytest_cache/
```

Create empty `src/vla/__init__.py` and `tests/__init__.py`.

Also create `.python-version` containing `3.12`. Without it uv picks the newest
interpreter satisfying `>=3.12` (3.13 today) while `setup.sh` provisions 3.12,
and with torch and `numpy<2.3.0` downstream that skew is exactly where "passes
locally, fails on the robot" comes from. Run `uv python install 3.12` first so
the dev venv uses a uv-managed interpreter rather than whatever it discovers.

- [ ] **Step 7: Verify the toolchain works**

Run: `uv sync && uv run python --version && uv run pytest -v`
Expected: Python 3.12.x, then `no tests ran` and exit code 5. That is success —
it proves the venv resolves on the deployed interpreter version and pytest is
wired up.

Then verify packaging does not leak internal docs:

Run: `mise run package && tar -tzf module.tar.gz`
Expected: `meta.json`, `run.sh`, `setup.sh`, and exactly one `.whl`. **No `docs/`.**

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold uv/mise Viam module project"
```

---

## Task 2: Wire codec

> **Implemented** — `25f10ab`, `cddc607`, `e6881ff`. The committed code is the
> source of truth; the code below is the starting point it grew from. Review
> added `encode_vector`/`decode_vector`, validation and `WireError` wrapping on
> the compressed path, and took the suite from 7 to 41 tests.

The controller and policy exchange images, float matrices, and the 1-D state
vector through protobuf `Struct`, which only carries JSON types. This module
owns that translation, and is the only module both resources import.

**Files:**
- Create: `src/vla/wire.py`
- Test: `tests/test_wire.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from vla.wire import (
    encode_image, decode_image, encode_matrix, decode_matrix, WireError,
)


def test_jpeg_roundtrip_preserves_shape():
    # Non-square on purpose: a square fixture cannot catch a height/width
    # transposition, and jpeg is the default encoding.
    img = np.random.randint(0, 255, (224, 160, 3), dtype=np.uint8)
    payload = encode_image(img, encoding="jpeg", quality=90)
    out = decode_image(payload)
    assert out.shape == (224, 160, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("encoding", ["raw", "jpeg", "png"])
def test_decode_image_returns_writable_array(encoding):
    # torch.from_numpy warns on read-only input; every path must be writable.
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding=encoding))
    assert out.flags.writeable
    out[0, 0, 0] = 42  # must not raise


def test_raw_decode_uses_payload_shape_not_a_guess():
    # A different shape from every other fixture: a decoder that hardcodes or
    # infers dimensions passes the other tests but fails this one.
    img = np.random.randint(0, 255, (3, 11, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding="raw"))
    assert out.shape == (3, 11, 3)
    np.testing.assert_array_equal(out, img)


def test_encode_matrix_emits_json_native_floats():
    # The module's whole purpose: numpy scalars are not JSON types and would
    # fail protobuf Struct serialization at runtime.
    payload = encode_matrix(np.array([[1.5, -2.0]], dtype=np.float32))
    assert all(type(v) is float for row in payload["rows"] for v in row)


def test_encode_image_rejects_non_uint8():
    with pytest.raises(WireError, match="uint8"):
        encode_image(np.zeros((4, 4, 3), dtype=np.float32))


def test_encode_image_rejects_wrong_channel_count():
    with pytest.raises(WireError, match="HWC RGB"):
        encode_image(np.zeros((4, 4, 4), dtype=np.uint8))


def test_encode_image_rejects_2d_input():
    with pytest.raises(WireError, match="HWC RGB"):
        encode_image(np.zeros((4, 4), dtype=np.uint8))


def test_decode_image_rejects_truncated_raw_payload():
    payload = encode_image(np.zeros((4, 4, 3), dtype=np.uint8), encoding="raw")
    payload["height"] = 8  # claims more bytes than were sent
    with pytest.raises(WireError, match="bytes"):
        decode_image(payload)


def test_decode_image_missing_data_raises_wire_error():
    with pytest.raises(WireError, match="data"):
        decode_image({"encoding": "jpeg"})


def test_decode_matrix_empty_rows_returns_empty_2d():
    out = decode_matrix({"rows": []})
    assert out.shape == (0, 0)
    assert out.dtype == np.float32


def test_decode_matrix_missing_rows_raises_wire_error():
    with pytest.raises(WireError, match="rows"):
        decode_matrix({})


def test_raw_roundtrip_is_lossless():
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding="raw"))
    np.testing.assert_array_equal(out, img)


def test_raw_payload_carries_explicit_shape():
    img = np.zeros((5, 7, 3), dtype=np.uint8)
    payload = encode_image(img, encoding="raw")
    assert payload["height"] == 5
    assert payload["width"] == 7
    assert payload["channels"] == 3


def test_png_roundtrip_is_lossless():
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding="png"))
    np.testing.assert_array_equal(out, img)


def test_matrix_roundtrip():
    m = np.array([[1.5, -2.0], [0.0, 3.25]], dtype=np.float32)
    out = decode_matrix(encode_matrix(m))
    np.testing.assert_allclose(out, m)
    assert out.dtype == np.float32


def test_decode_matrix_rejects_ragged_rows():
    with pytest.raises(WireError, match="ragged"):
        decode_matrix({"rows": [[1.0, 2.0], [3.0]]})


def test_decode_image_rejects_unknown_encoding():
    with pytest.raises(WireError, match="encoding"):
        decode_image({"encoding": "webp", "data": ""})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wire.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vla.wire'`

- [ ] **Step 3: Implement `src/vla/wire.py`**

```python
"""Wire codec for the protobuf Struct payloads exchanged between resources.

DoCommand carries a protobuf Struct, which holds only JSON types. Images become
base64 strings and float arrays become lists of lists.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


class WireError(ValueError):
    """Raised when a wire payload is malformed."""


def encode_image(img: np.ndarray, *, encoding: str = "jpeg", quality: int = 90) -> dict[str, Any]:
    if img.ndim != 3 or img.shape[2] != 3:
        raise WireError(f"expected HWC RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise WireError(f"expected uint8 image, got {img.dtype}")

    height, width, channels = img.shape

    if encoding == "raw":
        data = base64.b64encode(img.tobytes()).decode("ascii")
    elif encoding in ("jpeg", "png"):
        buf = io.BytesIO()
        pil_format = "JPEG" if encoding == "jpeg" else "PNG"
        kwargs = {"quality": quality} if encoding == "jpeg" else {}
        Image.fromarray(img).save(buf, format=pil_format, **kwargs)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
    else:
        raise WireError(f"unknown encoding {encoding!r}")

    return {
        "encoding": encoding,
        "data": data,
        "height": height,
        "width": width,
        "channels": channels,
    }


def decode_image(payload: dict[str, Any]) -> np.ndarray:
    encoding = payload.get("encoding")
    try:
        raw = base64.b64decode(payload["data"])
    except KeyError as exc:
        raise WireError("image payload missing 'data'") from exc

    if encoding == "raw":
        # Shape is carried explicitly so the decoder never has to infer it.
        try:
            shape = (int(payload["height"]), int(payload["width"]), int(payload["channels"]))
        except KeyError as exc:
            raise WireError("raw image payload missing shape fields") from exc
        expected = shape[0] * shape[1] * shape[2]
        if len(raw) != expected:
            raise WireError(f"raw image payload is {len(raw)} bytes, expected {expected}")
        # .copy() so the result is writable: a bare frombuffer view makes
        # torch.from_numpy warn downstream.
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()

    if encoding in ("jpeg", "png"):
        # np.array (not asarray) copies: np.asarray over a PIL image is
        # read-only, and jpeg is the default encoding.
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)

    raise WireError(f"unknown encoding {encoding!r}")


def encode_matrix(m: np.ndarray) -> dict[str, Any]:
    if m.ndim != 2:
        raise WireError(f"expected 2D matrix, got shape {m.shape}")
    return {"rows": [[float(v) for v in row] for row in m]}


def decode_matrix(payload: dict[str, Any]) -> np.ndarray:
    rows = payload.get("rows")
    if rows is None:
        raise WireError("matrix payload missing 'rows'")
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise WireError(f"ragged matrix rows: widths {sorted(widths)}")
    return np.asarray(rows, dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wire.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/wire.py tests/test_wire.py
git commit -m "feat: add wire codec for image and matrix payloads"
```

---

## Task 3: Policy config

> **Implemented** — `1407a43`, `d9d286d`, `7d486b1`. Review pulled the
> coercion helpers out into `config_util.py` (shared with Task 4) and
> hardened `PolicyConfig.__repr__` to redact `hf_token_env`.

**Files:**
- Create: `src/vla/policy/__init__.py`, `src/vla/policy/config.py`
- Test: `tests/policy/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/policy/__init__.py` (empty) and `tests/policy/test_config.py`:

```python
import pytest
from vla.policy.config import PolicyConfig, ConfigError


def test_local_path_config():
    cfg = PolicyConfig.parse({"model_path": "/models/smolvla"})
    assert cfg.model_path == "/models/smolvla"
    assert cfg.model_hub_id is None
    assert cfg.device == "auto"
    assert cfg.dtype == "auto"
    assert cfg.warmup_inferences == 2


def test_hub_config():
    cfg = PolicyConfig.parse({"model_hub_id": "lerobot/smolvla_base", "model_revision": "v1"})
    assert cfg.model_hub_id == "lerobot/smolvla_base"
    assert cfg.model_revision == "v1"


def test_requires_exactly_one_source_none_given():
    with pytest.raises(ConfigError, match="exactly one"):
        PolicyConfig.parse({})


def test_requires_exactly_one_source_both_given():
    with pytest.raises(ConfigError, match="exactly one"):
        PolicyConfig.parse({"model_path": "/m", "model_hub_id": "a/b"})


def test_rejects_unknown_device():
    with pytest.raises(ConfigError, match="device"):
        PolicyConfig.parse({"model_path": "/m", "device": "tpu"})


def test_rejects_unknown_dtype():
    with pytest.raises(ConfigError, match="dtype"):
        PolicyConfig.parse({"model_path": "/m", "dtype": "int4"})


def test_rtc_defaults_match_lerobot():
    cfg = PolicyConfig.parse({"model_path": "/m", "rtc": {"enabled": True}})
    assert cfg.rtc.enabled is True
    assert cfg.rtc.execution_horizon == 10
    assert cfg.rtc.max_guidance_weight == 10.0
    assert cfg.rtc.prefix_attention_schedule == "linear"


def test_rtc_disabled_by_default():
    cfg = PolicyConfig.parse({"model_path": "/m"})
    assert cfg.rtc.enabled is False


def test_rejects_nonpositive_guidance_weight():
    with pytest.raises(ConfigError, match="max_guidance_weight"):
        PolicyConfig.parse({"model_path": "/m", "rtc": {"max_guidance_weight": 0}})


def test_rejects_negative_warmup():
    with pytest.raises(ConfigError, match="warmup_inferences"):
        PolicyConfig.parse({"model_path": "/m", "warmup_inferences": -1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/policy/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vla.policy'`

- [ ] **Step 3: Implement `src/vla/policy/config.py`**

Create empty `src/vla/policy/__init__.py` first.

```python
"""Configuration parsing and validation for viam-labs:vla:policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEVICES = ("auto", "cuda", "mps", "cpu")
DTYPES = ("auto", "float32", "bfloat16", "float16")
SCHEDULES = ("linear", "exp", "ones", "zeros")


class ConfigError(ValueError):
    """Raised for invalid module configuration."""


@dataclass(frozen=True)
class RTCSettings:
    """Mirrors lerobot RTCConfig field-for-field."""

    enabled: bool = False
    execution_horizon: int = 10
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0

    @staticmethod
    def parse(raw: dict[str, Any]) -> "RTCSettings":
        schedule = raw.get("prefix_attention_schedule", "linear")
        if schedule not in SCHEDULES:
            raise ConfigError(
                f"rtc.prefix_attention_schedule must be one of {SCHEDULES}, got {schedule!r}"
            )
        horizon = int(raw.get("execution_horizon", 10))
        if horizon <= 0:
            raise ConfigError(f"rtc.execution_horizon must be positive, got {horizon}")
        weight = float(raw.get("max_guidance_weight", 10.0))
        if weight <= 0:
            raise ConfigError(f"rtc.max_guidance_weight must be positive, got {weight}")
        return RTCSettings(
            enabled=bool(raw.get("enabled", False)),
            execution_horizon=horizon,
            prefix_attention_schedule=schedule,
            max_guidance_weight=weight,
        )


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str | None = None
    model_hub_id: str | None = None
    model_revision: str = "main"
    hf_token_env: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    warmup_inferences: int = 2
    rtc: RTCSettings = field(default_factory=RTCSettings)

    @staticmethod
    def parse(raw: dict[str, Any]) -> "PolicyConfig":
        path = raw.get("model_path") or None
        hub = raw.get("model_hub_id") or None
        if bool(path) == bool(hub):
            raise ConfigError(
                "exactly one of model_path or model_hub_id is required "
                f"(got model_path={path!r}, model_hub_id={hub!r})"
            )

        device = raw.get("device", "auto")
        if device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}, got {device!r}")

        dtype = raw.get("dtype", "auto")
        if dtype not in DTYPES:
            raise ConfigError(f"dtype must be one of {DTYPES}, got {dtype!r}")

        warmup = int(raw.get("warmup_inferences", 2))
        if warmup < 0:
            raise ConfigError(f"warmup_inferences must be >= 0, got {warmup}")

        return PolicyConfig(
            model_path=path,
            model_hub_id=hub,
            model_revision=raw.get("model_revision", "main"),
            hf_token_env=raw.get("hf_token_env") or None,
            device=device,
            dtype=dtype,
            warmup_inferences=warmup,
            rtc=RTCSettings.parse(raw.get("rtc", {}) or {}),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/policy/test_config.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/policy tests/policy
git commit -m "feat: add policy config parsing and validation"
```

---

## Task 4: Checkpoint resolver

> **Implemented** — `bcc4c23`, `fc9f76d`, `dd3ea1c`. Review added an
> actionable error when `huggingface_hub` is missing and closed a
> token-disclosure path in the resolver's error/log messages.

**Files:**
- Create: `src/vla/policy/resolver.py`
- Test: `tests/policy/test_resolver.py`

Three delivery mechanisms collapse into two config fields: local path (which also covers Viam registry packages, because viam-server interpolates `${packages.ml_model.name}` into a plain path before the module sees it) and Hugging Face hub.

- [ ] **Step 1: Write the failing tests**

```python
import os
import pytest
from vla.policy.config import PolicyConfig
from vla.policy.resolver import resolve_checkpoint, ResolveError

REQUIRED = ["config.json", "model.safetensors"]


def _make_checkpoint(tmp_path):
    for name in REQUIRED:
        (tmp_path / name).write_text("{}")
    return tmp_path


def test_resolves_local_path(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    cfg = PolicyConfig.parse({"model_path": str(ckpt)})
    assert resolve_checkpoint(cfg) == str(ckpt)


def test_missing_local_path_errors(tmp_path):
    cfg = PolicyConfig.parse({"model_path": str(tmp_path / "nope")})
    with pytest.raises(ResolveError, match="does not exist"):
        resolve_checkpoint(cfg)


def test_local_path_missing_config_json_errors(tmp_path):
    (tmp_path / "model.safetensors").write_text("{}")
    cfg = PolicyConfig.parse({"model_path": str(tmp_path)})
    with pytest.raises(ResolveError, match="config.json"):
        resolve_checkpoint(cfg)


def test_hub_download_uses_module_data_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    cfg = PolicyConfig.parse(
        {"model_hub_id": "lerobot/smolvla_base", "model_revision": "abc123"}
    )
    (tmp_path / "dl").mkdir()
    out = resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert captured["repo_id"] == "lerobot/smolvla_base"
    assert captured["revision"] == "abc123"
    assert str(tmp_path) in captured["cache_dir"]
    assert out.endswith("dl")


def test_hub_token_read_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.setenv("MY_HF_TOKEN", "secret-value")
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    (tmp_path / "dl").mkdir()
    cfg = PolicyConfig.parse(
        {"model_hub_id": "a/b", "hf_token_env": "MY_HF_TOKEN"}
    )
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["token"] == "secret-value"


def test_missing_token_env_var_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.delenv("ABSENT_TOKEN", raising=False)
    cfg = PolicyConfig.parse({"model_hub_id": "a/b", "hf_token_env": "ABSENT_TOKEN"})
    with pytest.raises(ResolveError, match="ABSENT_TOKEN"):
        resolve_checkpoint(cfg, snapshot_download=lambda **k: "")


def test_falls_back_to_tempdir_without_module_data(tmp_path, monkeypatch):
    monkeypatch.delenv("VIAM_MODULE_DATA", raising=False)
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl2"))

    (tmp_path / "dl2").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["cache_dir"]  # some writable fallback was chosen
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/policy/test_resolver.py -v`
Expected: FAIL — no module named `vla.policy.resolver`

- [ ] **Step 3: Implement `src/vla/policy/resolver.py`**

```python
"""Resolve a configured model reference to a local checkpoint directory."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

from .config import PolicyConfig

LOGGER = logging.getLogger(__name__)

REQUIRED_FILES = ("config.json", "model.safetensors")


class ResolveError(RuntimeError):
    """Raised when a checkpoint cannot be resolved."""


def _cache_dir() -> str:
    module_data = os.environ.get("VIAM_MODULE_DATA")
    if module_data:
        path = Path(module_data) / "checkpoints"
    else:
        # Viam sets VIAM_MODULE_DATA in deployment; on a dev workstation it is absent.
        LOGGER.warning("VIAM_MODULE_DATA unset; caching checkpoints under the system temp dir")
        path = Path(tempfile.gettempdir()) / "viam-vla-checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _verify(directory: str) -> str:
    d = Path(directory)
    if not d.is_dir():
        raise ResolveError(f"checkpoint directory does not exist: {directory}")
    for name in REQUIRED_FILES:
        if not (d / name).is_file():
            raise ResolveError(f"checkpoint at {directory} is missing {name}")
    return str(d)


def resolve_checkpoint(
    cfg: PolicyConfig,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> str:
    """Return a local directory containing the checkpoint.

    `snapshot_download` is injectable so tests never touch the network.
    """
    if cfg.model_path:
        return _verify(cfg.model_path)

    token = None
    if cfg.hf_token_env:
        token = os.environ.get(cfg.hf_token_env)
        if not token:
            raise ResolveError(
                f"hf_token_env names {cfg.hf_token_env!r} but that variable is unset or empty"
            )

    if snapshot_download is None:  # pragma: no cover - exercised only with network
        from huggingface_hub import snapshot_download as _sd

        snapshot_download = _sd

    LOGGER.info("downloading checkpoint %s@%s", cfg.model_hub_id, cfg.model_revision)
    local = snapshot_download(
        repo_id=cfg.model_hub_id,
        revision=cfg.model_revision,
        cache_dir=_cache_dir(),
        token=token,
    )
    return _verify(local)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/policy/test_resolver.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/policy/resolver.py tests/policy/test_resolver.py
git commit -m "feat: add checkpoint resolver for local and hub sources"
```

---

## Task 5: PolicyBackend ABC and fake backend

> **Implemented** — `b859edb`, `6755c40`. Review found the fake backend
> would let several classes of Task 7 bug pass green: it predicted before
> `load()`, conflated `state_dim` with `action_dim`, declared only one
> camera, and raced `call_count`/`last_rtc` under concurrent calls. `6755c40`
> hardens all four and adds `state_dim`/`dtype` to `PolicySpecs`. Code below
> is re-synced with that hardening.

**Files:**
- Create: `src/vla/policy/backend.py`, `src/vla/policy/fake_backend.py`
- Test: `tests/policy/test_fake_backend.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from vla.policy.backend import PolicySpecs
from vla.policy.fake_backend import FakePolicyBackend


def _obs():
    return {"observation.images.top": np.zeros((224, 224, 3), dtype=np.uint8)}


def test_specs_before_load_are_none():
    assert FakePolicyBackend().specs is None


def test_load_then_specs():
    b = FakePolicyBackend(action_dim=6, n_action_steps=50)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    specs = b.specs
    assert isinstance(specs, PolicySpecs)
    assert specs.action_dim == 6
    assert specs.n_action_steps == 50
    assert specs.supports_rtc is True
    assert specs.relative_actions is False


def test_predict_chunk_shape():
    b = FakePolicyBackend(action_dim=6, n_action_steps=50)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    actions, raw = b.predict_chunk(_obs(), np.zeros(6, np.float32), "do the thing", None)
    assert actions.shape == (50, 6)
    assert raw.shape == (50, 6)


def test_predict_chunk_is_deterministic():
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    a1, _ = b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    a2, _ = b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    np.testing.assert_array_equal(a1, a2)


def test_raw_and_processed_differ_so_confusion_is_detectable():
    # The two arrays must not be interchangeable, or a controller bug that
    # swaps them would pass every test.
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    actions, raw = b.predict_chunk(_obs(), np.zeros(4, np.float32), "t", None)
    assert not np.array_equal(actions, raw)


def test_records_rtc_kwargs_for_assertions():
    b = FakePolicyBackend(action_dim=4, n_action_steps=10)
    b.load("/fake", device="cpu", dtype="float32", rtc=None)
    prefix = np.ones((3, 4), dtype=np.float32)
    b.predict_chunk(_obs(), np.zeros(4, np.float32), "t",
                    {"inference_delay": 2, "prev_chunk_left_over": prefix})
    assert b.last_rtc["inference_delay"] == 2
    np.testing.assert_array_equal(b.last_rtc["prev_chunk_left_over"], prefix)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/policy/test_fake_backend.py -v`
Expected: FAIL — no module named `vla.policy.backend`

- [ ] **Step 3: Implement `src/vla/policy/backend.py`**

```python
"""Backend abstraction separating the resource layer from any inference runtime."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicySpecs:
    """Everything a caller needs to know about a loaded policy."""

    policy_type: str
    action_dim: int
    n_action_steps: int
    input_features: dict[str, list[int]]
    output_features: dict[str, list[int]]
    image_feature_keys: list[str]
    supports_rtc: bool
    rtc_enabled: bool
    relative_actions: bool
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": self.policy_type,
            "action_dim": self.action_dim,
            "n_action_steps": self.n_action_steps,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "image_feature_keys": self.image_feature_keys,
            "supports_rtc": self.supports_rtc,
            "rtc_enabled": self.rtc_enabled,
            "relative_actions": self.relative_actions,
            "device": self.device,
        }


class PolicyBackend(abc.ABC):
    """Loads a checkpoint and turns observations into action chunks.

    Implementations must be safe to call concurrently from multiple requests:
    `predict_chunk` carries no state between calls.
    """

    @abc.abstractmethod
    def load(self, checkpoint_dir: str, *, device: str, dtype: str, rtc: Any | None) -> None:
        """Load weights and processors. Blocking; run off the event loop."""

    @property
    @abc.abstractmethod
    def specs(self) -> PolicySpecs | None:
        """Loaded policy specs, or None before load completes."""

    @abc.abstractmethod
    def predict_chunk(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task: str,
        rtc_kwargs: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(processed_actions, raw_actions)`, both `[n_action_steps, action_dim]`.

        `processed_actions` are postprocessed and ready for the robot.
        `raw_actions` are in the policy's own action space and are what an RTC
        caller must feed back as `prev_chunk_left_over`. They are deliberately
        distinct return values because confusing them is the likeliest RTC bug.
        """

    def reset(self) -> None:
        """Clear any cached state. Default: no-op."""
```

- [ ] **Step 4: Implement `src/vla/policy/fake_backend.py`**

```python
"""Deterministic in-memory backend, so everything above it tests without torch."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import PolicyBackend, PolicySpecs


class FakePolicyBackend(PolicyBackend):
    def __init__(
        self,
        *,
        action_dim: int = 6,
        n_action_steps: int = 50,
        supports_rtc: bool = True,
        relative_actions: bool = False,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self._action_dim = action_dim
        self._n_action_steps = n_action_steps
        self._supports_rtc = supports_rtc
        self._relative_actions = relative_actions
        self._image_size = image_size
        self._specs: PolicySpecs | None = None
        self.last_rtc: dict[str, Any] | None = None
        self.call_count = 0

    def load(self, checkpoint_dir: str, *, device: str, dtype: str, rtc: Any | None) -> None:
        h, w = self._image_size
        self._specs = PolicySpecs(
            policy_type="fake",
            action_dim=self._action_dim,
            n_action_steps=self._n_action_steps,
            input_features={"observation.images.top": [3, h, w],
                            "observation.state": [self._action_dim]},
            output_features={"action": [self._action_dim]},
            image_feature_keys=["observation.images.top"],
            supports_rtc=self._supports_rtc,
            rtc_enabled=bool(rtc and getattr(rtc, "enabled", False)),
            relative_actions=self._relative_actions,
            device=device,
        )

    @property
    def specs(self) -> PolicySpecs | None:
        return self._specs

    def predict_chunk(self, images, state, task, rtc_kwargs):
        self.call_count += 1
        self.last_rtc = rtc_kwargs
        steps, dim = self._n_action_steps, self._action_dim
        # A fixed ramp: deterministic, and raw != processed so a controller that
        # swaps the two fails loudly instead of silently passing.
        base = np.arange(steps, dtype=np.float32).reshape(steps, 1)
        raw = np.tile(base, (1, dim)) * 0.01
        processed = raw + 1.0
        return processed, raw
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/policy/test_fake_backend.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/vla/policy/backend.py src/vla/policy/fake_backend.py tests/policy/test_fake_backend.py
git commit -m "feat: add PolicyBackend ABC and deterministic fake backend"
```

---

## Task 6: Prefix normalization

> **Implemented** — `a3051de`, `3bacaef`. Verified against
> `lerobot/rollout/inference/rtc.py:83-94` at the pinned SHA: same
> truncation/padding semantics, but always returns a copy (upstream
> aliases/views the input) since the RTC scheduler feeds this output back
> into the action queue on a later tick. `3bacaef` closes a bare-`TypeError`
> path on a non-integer `target_steps`. Code below is re-synced with that fix.

This is small but load-bearing. Upstream pads or truncates the RTC prefix to `execution_horizon` before inference. Skip it and `denoise_step` silently shrinks the horizon to the prefix length, changing the guidance weights — wrong motion, no error. See spec section "LeRobotBackend.predict_chunk owns prefix normalization".

**Files:**
- Create: `src/vla/policy/prefix.py`
- Test: `tests/policy/test_prefix.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from vla.policy.prefix import normalize_prefix_length


def test_exact_length_returns_unchanged():
    x = np.ones((10, 4), dtype=np.float32)
    out = normalize_prefix_length(x, 10)
    np.testing.assert_array_equal(out, x)


def test_longer_prefix_is_truncated():
    x = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    out = normalize_prefix_length(x, 8)
    assert out.shape == (8, 3)
    np.testing.assert_array_equal(out, x[:8])


def test_shorter_prefix_is_zero_padded():
    x = np.ones((3, 4), dtype=np.float32)
    out = normalize_prefix_length(x, 10)
    assert out.shape == (10, 4)
    np.testing.assert_array_equal(out[:3], x)
    np.testing.assert_array_equal(out[3:], np.zeros((7, 4), dtype=np.float32))


def test_preserves_dtype():
    x = np.ones((3, 4), dtype=np.float32)
    assert normalize_prefix_length(x, 6).dtype == np.float32


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        normalize_prefix_length(np.ones((2, 3, 4), dtype=np.float32), 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/policy/test_prefix.py -v`
Expected: FAIL — no module named `vla.policy.prefix`

- [ ] **Step 3: Implement `src/vla/policy/prefix.py`**

Mirrors `lerobot/rollout/inference/rtc.py:83` `_normalize_prev_actions_length`.

```python
"""Pad or truncate an RTC prefix to the configured execution horizon.

Port of lerobot `rollout/inference/rtc.py::_normalize_prev_actions_length`.
Kept as pure numpy so it is testable with no torch present.
"""

from __future__ import annotations

import numpy as np


def normalize_prefix_length(prev_actions: np.ndarray, target_steps: int) -> np.ndarray:
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] array, got shape={prev_actions.shape}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = np.zeros((target_steps, action_dim), dtype=prev_actions.dtype)
    padded[:steps] = prev_actions
    return padded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/policy/test_prefix.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/policy/prefix.py tests/policy/test_prefix.py
git commit -m "feat: add RTC prefix length normalization"
```

---

## Task 7: Policy service resource

**Files:**
- Create: `src/vla/policy/service.py`
- Test: `tests/policy/test_service.py`

Read `/Users/nick.hehr/src/viam-python-sdk/src/viam/services/generic/generic.py` before starting so the base class surface is familiar.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio
import threading
import time

import numpy as np
import pytest
from viam.proto.app.robot import ServiceConfig
from google.protobuf.struct_pb2 import Struct

from vla.policy.fake_backend import FakePolicyBackend
from vla.policy.service import VLAPolicy
from vla.wire import encode_image, encode_matrix, decode_matrix


def _config(attrs: dict) -> ServiceConfig:
    s = Struct()
    s.update(attrs)
    return ServiceConfig(name="p", api="rdk:service:generic", model="viam-labs:vla:policy",
                         attributes=s)


async def _ready_service(tmp_path, **backend_kwargs) -> VLAPolicy:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("{}")
    svc = VLAPolicy("p")
    svc._backend_factory = lambda: FakePolicyBackend(**backend_kwargs)
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    await svc.await_ready()
    return svc


def test_validate_returns_no_dependencies(tmp_path):
    required, optional = VLAPolicy.validate_config(_config({"model_path": str(tmp_path)}))
    assert required == []
    assert optional == []


def test_validate_rejects_missing_source():
    with pytest.raises(Exception, match="exactly one"):
        VLAPolicy.validate_config(_config({}))


async def test_reconfigure_returns_before_load_completes(tmp_path):
    """reconfigure must not block on a slow load, or viam-server can time out."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("{}")
    svc = VLAPolicy("p")
    release = threading.Event()

    class SlowBackend(FakePolicyBackend):
        def load(self, *a, **k):
            if not release.wait(timeout=5):
                raise AssertionError("load was never released")
            super().load(*a, **k)

    svc._backend_factory = SlowBackend

    started = time.perf_counter()
    svc.reconfigure(_config({"model_path": str(tmp_path)}), {})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, f"reconfigure blocked for {elapsed:.2f}s"
    assert (await svc.do_command({"command": "status"}))["state"] == "loading"

    release.set()
    await svc.await_ready()
    assert (await svc.do_command({"command": "status"}))["state"] == "ready"


async def test_infer_before_ready_errors(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("{}")
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend
    with pytest.raises(Exception, match="not ready"):
        await svc.do_command({"command": "infer", "images": {}, "state": [], "task": "t"})


async def test_status_reports_ready(tmp_path):
    svc = await _ready_service(tmp_path)
    assert (await svc.do_command({"command": "status"}))["state"] == "ready"


async def test_specs_shape(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=6, n_action_steps=50)
    specs = await svc.do_command({"command": "specs"})
    assert specs["action_dim"] == 6
    assert specs["n_action_steps"] == 50
    assert specs["supports_rtc"] is True
    assert specs["relative_actions"] is False


async def test_infer_returns_both_action_arrays(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=6, n_action_steps=50)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    out = await svc.do_command({
        "command": "infer",
        "images": {"observation.images.top": encode_image(img)},
        "state": encode_vector(np.zeros(6, dtype=np.float32)),
        "task": "pick up the block",
    })
    actions = decode_matrix(out["actions"])
    raw = decode_matrix(out["raw_actions"])
    assert actions.shape == (50, 6)
    assert raw.shape == (50, 6)
    assert not np.array_equal(actions, raw)
    assert out["latency_s"] >= 0


async def test_infer_passes_rtc_kwargs_to_backend(tmp_path):
    svc = await _ready_service(tmp_path, action_dim=4, n_action_steps=10)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    prefix = np.ones((3, 4), dtype=np.float32)
    await svc.do_command({
        "command": "infer",
        "images": {"observation.images.top": encode_image(img)},
        "state": encode_vector(np.zeros(4, dtype=np.float32)),
        "task": "t",
        "rtc": {"inference_delay": 2, "prev_chunk_left_over": encode_matrix(prefix)},
    })
    assert svc._backend.last_rtc["inference_delay"] == 2
    np.testing.assert_array_equal(svc._backend.last_rtc["prev_chunk_left_over"], prefix)


async def test_unknown_command_errors(tmp_path):
    svc = await _ready_service(tmp_path)
    with pytest.raises(Exception, match="unknown command"):
        await svc.do_command({"command": "teleport"})


async def test_failed_load_surfaces_in_status(tmp_path):
    svc = VLAPolicy("p")
    svc._backend_factory = FakePolicyBackend   # reconfigure constructs the backend eagerly
    svc.reconfigure(_config({"model_path": str(tmp_path / "absent")}), {})
    await svc.await_ready(expect_failure=True)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed"
    assert "does not exist" in status["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/policy/test_service.py -v`
Expected: FAIL — no module named `vla.policy.service`

- [ ] **Step 3: Implement `src/vla/policy/service.py`**

```python
"""viam-labs:vla:policy — pure VLA inference over a LeRobot checkpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
from typing_extensions import Self
from viam.proto.app.robot import ServiceConfig
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

from ..config_util import VLAError
from ..wire import WireError, decode_image, decode_matrix, decode_vector, encode_matrix
from .backend import PolicyBackend
from .config import PolicyConfig
from .lerobot_backend import LeRobotBackend
from .resolver import resolve_checkpoint

LOGGER = logging.getLogger(__name__)


class VLAPolicy(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vla"), "policy")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: PolicyConfig | None = None
        self._backend: PolicyBackend | None = None
        self._backend_factory: Callable[[], PolicyBackend] = LeRobotBackend
        self._state = "idle"
        self._error: str | None = None
        self._load_task: asyncio.Task | None = None
        self._generation = 0

    @classmethod
    def new(cls, config: ServiceConfig, dependencies: Mapping[Any, ResourceBase]) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> tuple[Sequence[str], Sequence[str]]:
        PolicyConfig.parse(struct_to_dict(config.attributes))
        return [], []

    def reconfigure(self, config: ServiceConfig, dependencies) -> None:
        """Store config and kick off loading in the background.

        Loading must not block: a multi-GB hub download here would stall the
        module's reconfigure loop and can trip viam-server timeouts.
        """
        self._cfg = PolicyConfig.parse(struct_to_dict(config.attributes))
        self._state = "loading"
        self._error = None
        self._backend = self._backend_factory()
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
        # Cancelling only abandons the await. The resolver and backend.load run
        # inside asyncio.to_thread, so the worker thread keeps downloading and
        # keeps an executor slot. Bump a generation counter so a superseded load
        # cannot write state after a newer one started; without this, a slow
        # first load can overwrite the result of the config that replaced it.
        self._generation += 1
        self._load_task = asyncio.create_task(self._load(self._generation))

    async def _load(self, generation: int) -> None:
        cfg = self._cfg
        assert cfg is not None

        def _superseded() -> bool:
            return generation != self._generation

        try:
            checkpoint = await asyncio.to_thread(resolve_checkpoint, cfg)
            rtc = cfg.rtc if cfg.rtc.enabled else None
            await asyncio.to_thread(
                self._backend.load, checkpoint, device=cfg.device, dtype=cfg.dtype, rtc=rtc
            )
            for _ in range(cfg.warmup_inferences):
                await asyncio.to_thread(self._warmup_once)
            if _superseded():
                LOGGER.info("discarding superseded load (generation %d)", generation)
                return
            self._state = "ready"
            LOGGER.info("policy ready: %s", self._backend.specs)
        except VLAError as exc:
            # Expected failures: bad config, unresolvable checkpoint, malformed
            # payload. The message is meant for an operator reading `status`.
            if _superseded():
                LOGGER.info("ignoring failure from superseded load: %s", exc)
                return
            self._state = "failed"
            self._error = str(exc)
            LOGGER.error("policy load failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            # Everything else — torch OOM, a CUDA error, or a genuine bug in
            # this module. It must still land in `status`: this runs in a
            # background task, so letting it propagate would leave state stuck
            # on "loading" forever with the traceback swallowed by asyncio.
            # Distinguish it in the message and log the full traceback, so a
            # bug here is never mistaken for a user configuration error.
            if _superseded():
                LOGGER.info("ignoring failure from superseded load: %s", exc)
                return
            self._state = "failed"
            self._error = f"internal error ({type(exc).__name__}): {exc}"
            LOGGER.exception("policy load failed with an unexpected error")

    def _warmup_once(self) -> None:
        """Run one throwaway inference so the first real call is not an outlier."""
        specs = self._backend.specs
        if specs is None:
            return
        images = {}
        for key in specs.image_feature_keys:
            c, h, w = specs.input_features[key]
            images[key] = np.zeros((int(h), int(w), int(c)), dtype=np.uint8)
        # State dim, not action dim: they coincide on smolvla_base but need not.
        # Skip rather than KeyError on a state-less checkpoint — warmup runs
        # inside _load, so raising here would fail the entire policy.
        state_feature = specs.input_features.get("observation.state")
        if state_feature is None:
            LOGGER.warning("no observation.state feature; skipping warmup")
            return
        state = np.zeros(int(state_feature[0]), dtype=np.float32)
        self._backend.predict_chunk(images, state, "warmup", None)

    async def await_ready(self, *, expect_failure: bool = False) -> None:
        """Test helper: wait for the background load to settle."""
        if self._load_task:
            await self._load_task
        if not expect_failure and self._state != "ready":
            raise RuntimeError(f"policy failed to load: {self._error}")

    async def do_command(self, command: Mapping[str, Any], *, timeout=None, **kwargs) -> Mapping[str, Any]:
        name = command.get("command")
        if name == "status":
            return {"state": self._state, "error": self._error or ""}
        if name == "specs":
            self._require_ready()
            return self._backend.specs.to_dict()
        if name == "reset":
            self._require_ready()
            self._backend.reset()
            return {"ok": True}
        if name == "infer":
            return await self._infer(command)
        raise ValueError(f"unknown command {name!r}")

    def _require_ready(self) -> None:
        if self._state != "ready":
            raise RuntimeError(f"policy not ready (state={self._state}, error={self._error})")

    async def _infer(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require_ready()

        # Wrap per-camera: a bare WireError from decode_image says the payload
        # was malformed but not which feed produced it, and a robot log needs
        # to name the camera.
        images = {}
        for key, payload in (command.get("images") or {}).items():
            try:
                images[key] = decode_image(payload)
            except WireError as exc:
                raise WireError(f"{key}: {exc}") from exc
        # decode_vector, not a bare asarray: `command.get("state") or []` would
        # silently yield an empty array for a missing or malformed state, and
        # the shape mismatch would surface deep inside the backend instead.
        state = decode_vector(command["state"]) if "state" in command else None
        if state is None:
            raise WireError("infer requires a 'state' vector")
        task = command.get("task") or ""

        rtc_kwargs = None
        raw_rtc = command.get("rtc")
        if raw_rtc:
            rtc_kwargs = {"inference_delay": int(raw_rtc.get("inference_delay", 0))}
            prefix = raw_rtc.get("prev_chunk_left_over")
            rtc_kwargs["prev_chunk_left_over"] = decode_matrix(prefix) if prefix else None

        started = time.perf_counter()
        actions, raw = await asyncio.to_thread(
            self._backend.predict_chunk, images, state, task, rtc_kwargs
        )
        latency = time.perf_counter() - started

        return {
            "actions": encode_matrix(actions),
            "raw_actions": encode_matrix(raw),
            "latency_s": latency,
        }
```

Note the tests monkeypatch `_backend_factory`, so `LeRobotBackend` is never constructed in unit tests. Task 8 creates that module — until then, add a temporary stub so the import resolves:

```python
# src/vla/policy/lerobot_backend.py  (temporary; replaced in Task 8)
class LeRobotBackend:  # pragma: no cover
    def __init__(self):
        raise NotImplementedError("implemented in Task 8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/policy/test_service.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/policy/service.py src/vla/policy/lerobot_backend.py tests/policy/test_service.py
git commit -m "feat: add policy service with background loading"
```

---

## Task 8: LeRobot backend

The only file that imports `lerobot`, and only inside methods so the module still starts when torch is absent.

**Files:**
- Modify: `src/vla/policy/lerobot_backend.py` (replace the stub)
- Test: `tests/policy/test_lerobot_backend_integration.py`

- [ ] **Step 1: Write the failing integration test**

Marked `integration` so the default `mise run test` skips it.

```python
import numpy as np
import pytest

pytestmark = pytest.mark.integration

CHECKPOINT = "lerobot/smolvla_base"


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from vla.policy.config import PolicyConfig
    from vla.policy.lerobot_backend import LeRobotBackend
    from vla.policy.resolver import resolve_checkpoint

    cfg = PolicyConfig.parse({"model_hub_id": CHECKPOINT})
    path = resolve_checkpoint(cfg)
    b = LeRobotBackend()
    b.load(path, device="auto", dtype="auto", rtc=None)
    return b


def test_specs_are_populated(backend):
    s = backend.specs
    assert s.policy_type == "smolvla"
    assert s.action_dim > 0
    assert s.n_action_steps == 50
    assert s.supports_rtc is True
    # smolvla_base declares observation.images.camera1/2/3
    assert len(s.image_feature_keys) == 3
    assert all(k in s.input_features for k in s.image_feature_keys)


def test_predict_chunk_returns_finite_chunk(backend):
    s = backend.specs
    images = {}
    for key in s.image_feature_keys:
        c, h, w = s.input_features[key]
        images[key] = np.zeros((h, w, c), dtype=np.uint8)
    state = np.zeros(s.action_dim, dtype=np.float32)

    actions, raw = backend.predict_chunk(images, state, "pick up the red block", None)

    assert actions.shape == (s.n_action_steps, s.action_dim)
    assert raw.shape == (s.n_action_steps, s.action_dim)
    assert np.all(np.isfinite(actions))
    assert np.all(np.isfinite(raw))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/policy/test_lerobot_backend_integration.py -v -m integration`
Expected: FAIL — `NotImplementedError: implemented in Task 8`

- [ ] **Step 3: Implement `src/vla/policy/lerobot_backend.py`**

Read `/Users/nick.hehr/src/lerobot/src/lerobot/policies/factory.py` alongside this.

```python
"""LeRobot-backed implementation of PolicyBackend.

`lerobot` is imported lazily inside methods so the module loads, validates
config, and reports status even when torch is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .backend import PolicyBackend, PolicySpecs
from .prefix import normalize_prefix_length

LOGGER = logging.getLogger(__name__)


def _select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LeRobotBackend(PolicyBackend):
    def __init__(self) -> None:
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._specs: PolicySpecs | None = None
        self._device = "cpu"
        self._execution_horizon = 10
        self._rtc_enabled = False

    def load(self, checkpoint_dir: str, *, device: str, dtype: str, rtc: Any | None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import (
            get_policy_class, make_pre_post_processors,
        )

        cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(checkpoint_dir, config=cfg)

        resolved_device = _select_device(device)
        policy.to(resolved_device)
        policy.eval()

        if dtype != "auto":
            # Deliberately not applied. Casting weights with `policy.to(dtype=...)`
            # leaves the preprocessor emitting float32 (its serialized
            # DeviceProcessorStep.float_dtype is None), producing
            # "expected scalar type BFloat16 but found Float". Upstream never
            # casts weights; it uses torch.autocast gated on config.use_amp.
            # Wiring that up is deferred - see "Deferred to follow-up plans".
            LOGGER.warning("dtype=%s is not yet applied; running in the checkpoint's dtype", dtype)

        supports_rtc = bool(policy.supports_rtc())
        if rtc is not None and rtc.enabled:
            if not supports_rtc:
                LOGGER.warning("rtc configured but %s does not support it; ignoring", cfg.type)
            else:
                self._configure_rtc(policy, rtc)
                self._rtc_enabled = True
                self._execution_horizon = rtc.execution_horizon

        # The device override is mandatory, not cosmetic. With `pretrained_path`,
        # the processor pipeline is deserialized from the checkpoint including a
        # DeviceProcessorStep carrying the *training* device. Its __post_init__
        # calls get_safe_torch_device, which RAISES rather than falling back
        # (device_utils.py:48) - so a CUDA-trained checkpoint fails outright on
        # the Apple Silicon dev machine. Even when it does not raise, the
        # preprocessor would move tensors to a different device than the model.
        device_override = {"device": resolved_device}
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=checkpoint_dir,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = resolved_device
        self._specs = self._build_specs(cfg, policy, supports_rtc, preprocessor, resolved_device)

    @staticmethod
    def _configure_rtc(policy, rtc) -> None:
        """Inject an RTCConfig and rewire the processor onto the loaded model.

        A downloaded checkpoint almost always has `rtc_config=None`, and
        `_rtc_enabled()` reads `config.rtc_config.enabled`. `init_rtc_processor`
        exists precisely to rewire an already-constructed model.
        """
        from lerobot.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig

        policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=rtc.execution_horizon,
            max_guidance_weight=rtc.max_guidance_weight,
            # RTCAttentionSchedule members are UPPERCASE ("LINEAR", "EXP",
            # "ONES", "ZEROS"); config accepts the lowercase spelling.
            prefix_attention_schedule=RTCAttentionSchedule(
                rtc.prefix_attention_schedule.upper()
            ),
        )
        policy.init_rtc_processor()

    @staticmethod
    def _detect_relative_actions(preprocessor) -> bool:
        try:
            from lerobot.processor import RelativeActionsProcessorStep
        except ImportError:  # pragma: no cover - older lerobot
            return False
        return any(
            isinstance(s, RelativeActionsProcessorStep) and getattr(s, "enabled", False)
            for s in preprocessor.steps
        )

    def _build_specs(self, cfg, policy, supports_rtc, preprocessor, device) -> PolicySpecs:
        input_features = {k: list(v.shape) for k, v in cfg.input_features.items()}
        output_features = {k: list(v.shape) for k, v in cfg.output_features.items()}
        action_dim = int(output_features["action"][0])
        # Classify on FeatureType.VISUAL rather than a key-prefix guess: the
        # naming is a checkpoint's choice (smolvla_base uses
        # observation.images.camera1/2/3), the feature type is not.
        image_keys = sorted(cfg.image_features.keys())
        return PolicySpecs(
            policy_type=cfg.type,
            action_dim=action_dim,
            n_action_steps=int(cfg.n_action_steps),
            input_features=input_features,
            output_features=output_features,
            image_feature_keys=image_keys,
            supports_rtc=supports_rtc,
            rtc_enabled=self._rtc_enabled,
            relative_actions=self._detect_relative_actions(preprocessor),
            device=device,
        )

    @property
    def specs(self) -> PolicySpecs | None:
        return self._specs

    def predict_chunk(self, images, state, task, rtc_kwargs):
        import torch

        if self._policy is None:
            raise RuntimeError("backend not loaded")

        batch: dict[str, Any] = {"task": task}
        for key, img in images.items():
            # HWC uint8 -> BCHW float in [0, 1]; the preprocessor handles the rest.
            t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            batch[key] = t.unsqueeze(0).to(self._device)
        batch["observation.state"] = (
            torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self._device)
        )

        kwargs = {}
        if rtc_kwargs:
            prefix = rtc_kwargs.get("prev_chunk_left_over")
            if prefix is not None and prefix.size:
                # Must match execution_horizon, or denoise_step silently shrinks
                # the horizon to the prefix length and changes the guidance weights.
                prefix = normalize_prefix_length(prefix, self._execution_horizon)
                kwargs["prev_chunk_left_over"] = torch.from_numpy(prefix).to(self._device)
            kwargs["inference_delay"] = int(rtc_kwargs.get("inference_delay", 0))

        with torch.no_grad():
            preprocessed = self._preprocessor(batch)
            actions = self._policy.predict_action_chunk(preprocessed, **kwargs)
            raw = actions.squeeze(0).clone()
            processed = self._postprocessor(actions).squeeze(0)

        return (
            processed.float().cpu().numpy().astype(np.float32),
            raw.float().cpu().numpy().astype(np.float32),
        )

    def reset(self) -> None:
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
```

- [ ] **Step 4: Install the extra and run the integration test**

Run: `uv sync --extra lerobot && uv run pytest tests/policy/test_lerobot_backend_integration.py -v -m integration`
Expected: 2 passed. First run downloads the checkpoint and may take several minutes.

If `cfg.input_features` shapes or `policy.config.rtc_config` do not match what `_build_specs`/`_configure_rtc` expect, read the actual attribute names in `/Users/nick.hehr/src/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py` and adjust. Do not guess.

- [ ] **Step 5: Verify unit tests still pass without the extra**

Run: `uv run pytest -m 'not integration and not differential' -v`
Expected: all pass — nothing above the backend imports torch.

- [ ] **Step 6: Commit**

```bash
git add src/vla/policy/lerobot_backend.py tests/policy/test_lerobot_backend_integration.py
git commit -m "feat: add LeRobot backend with lazy imports and RTC wiring"
```

---

## Task 9: Unit conversion

**Files:**
- Create: `src/vla/controller/__init__.py`, `src/vla/controller/units.py`
- Test: `tests/controller/test_units.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/controller/__init__.py` (empty) and:

```python
import numpy as np
import pytest
from vla.controller.units import from_degrees, to_degrees, UnitError


def test_degrees_is_identity():
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(from_degrees(x, "degrees"), x)
    np.testing.assert_allclose(to_degrees(x, "degrees"), x)


def test_degrees_to_radians():
    np.testing.assert_allclose(
        from_degrees(np.array([180.0], dtype=np.float32), "radians"), [np.pi], rtol=1e-6
    )


def test_radians_to_degrees():
    np.testing.assert_allclose(
        to_degrees(np.array([np.pi], dtype=np.float32), "radians"), [180.0], rtol=1e-6
    )


def test_roundtrip_is_stable():
    x = np.array([12.5, -90.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(to_degrees(from_degrees(x, "radians"), "radians"), x, rtol=1e-5)


def test_normalized_is_not_yet_supported():
    # Deliberate: normalized units need per-joint min/max, an open question in the spec.
    with pytest.raises(UnitError, match="normalized"):
        from_degrees(np.array([0.0], dtype=np.float32), "normalized")


def test_unknown_unit_errors():
    with pytest.raises(UnitError, match="unknown"):
        from_degrees(np.array([0.0], dtype=np.float32), "furlongs")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_units.py -v`
Expected: FAIL — no module named `vla.controller`

- [ ] **Step 3: Implement `src/vla/controller/units.py`**

Create empty `src/vla/controller/__init__.py` first.

```python
"""Conversion between Viam's native degrees and a checkpoint's state units.

Viam arms report and accept degrees. A checkpoint uses whatever the recording
robot used. Nearly every bug in this module traces back to this boundary.
"""

from __future__ import annotations

import numpy as np

UNITS = ("degrees", "radians", "normalized")


class UnitError(ValueError):
    """Raised for an unsupported unit conversion."""


def _check(unit: str) -> None:
    if unit == "normalized":
        raise UnitError(
            "normalized units require per-joint min/max, which is unresolved; "
            "use degrees or radians"
        )
    if unit not in UNITS:
        raise UnitError(f"unknown unit {unit!r}, expected one of {UNITS}")


def from_degrees(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert Viam degrees into the checkpoint's unit."""
    _check(unit)
    if unit == "degrees":
        return values
    return np.deg2rad(values).astype(np.float32)


def to_degrees(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert the checkpoint's unit back into Viam degrees."""
    _check(unit)
    if unit == "degrees":
        return values
    return np.rad2deg(values).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_units.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller tests/controller
git commit -m "feat: add degrees/radians unit conversion"
```

---

## Task 10: ActionQueue numpy port

The highest-risk file in the project. Port `lerobot/policies/rtc/action_queue.py` to numpy, keeping **the same method names** so upstream diffs stay reviewable.

**Files:**
- Create: `src/vla/controller/action_queue.py`
- Test: `tests/controller/test_action_queue.py`

- [ ] **Step 1: Read the upstream source**

Read `/Users/nick.hehr/src/lerobot/src/lerobot/policies/rtc/action_queue.py` in full — all 247 lines. Note that it holds **two** parallel arrays, and that `_check_and_resolve_delays` returns the *unclamped* `real_delay` in its mismatch branch.

- [ ] **Step 2: Write the failing tests**

```python
import numpy as np
import pytest
from vla.controller.action_queue import ActionQueue, QueueSettings


def chunk(n, dim, offset=0.0):
    return (np.arange(n * dim, dtype=np.float32).reshape(n, dim) + offset)


def test_empty_queue_returns_none():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    assert q.get() is None
    assert q.empty()
    assert q.qsize() == 0


def test_append_mode_serves_actions_in_order():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(3, 2), chunk(3, 2, 100.0), real_delay=0)
    np.testing.assert_array_equal(q.get(), [100.0, 101.0])
    np.testing.assert_array_equal(q.get(), [102.0, 103.0])
    np.testing.assert_array_equal(q.get(), [104.0, 105.0])
    assert q.get() is None


def test_get_serves_processed_not_original():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(2, 2), chunk(2, 2, 50.0), real_delay=0)
    np.testing.assert_array_equal(q.get(), [50.0, 51.0])


def test_append_mode_drops_consumed_and_appends():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(3, 1), chunk(3, 1, 10.0), real_delay=0)
    q.get()
    q.merge(chunk(2, 1), chunk(2, 1, 20.0), real_delay=0)
    assert q.qsize() == 4
    assert q.get_action_index() == 0


def test_rtc_mode_replaces_and_trims_by_delay():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(5, 1), chunk(5, 1, 10.0), real_delay=2)
    assert q.qsize() == 3
    np.testing.assert_array_equal(q.get(), [12.0])


def test_rtc_delay_clamped_to_shortest_array():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(3, 1), chunk(3, 1), real_delay=99)
    assert q.qsize() == 0


def test_negative_delay_clamped_to_zero():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(4, 1), chunk(4, 1, 10.0), real_delay=-5)
    assert q.qsize() == 4


def test_get_left_over_returns_original_space():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(4, 1), chunk(4, 1, 100.0), real_delay=0)
    q.get()
    np.testing.assert_array_equal(q.get_left_over(), [[1.0], [2.0], [3.0]])


def test_get_processed_left_over_returns_processed_space():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(4, 1), chunk(4, 1, 100.0), real_delay=0)
    q.get()
    np.testing.assert_array_equal(q.get_processed_left_over(), [[101.0], [102.0], [103.0]])


def test_left_over_is_none_before_any_merge():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    assert q.get_left_over() is None
    assert q.get_processed_left_over() is None


def test_clear_resets_everything():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(3, 1), chunk(3, 1), real_delay=0)
    q.get()
    q.clear()
    assert q.empty()
    assert q.get_action_index() == 0
    assert q.get_left_over() is None


def test_returned_actions_are_copies():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(2, 2), chunk(2, 2, 5.0), real_delay=0)
    a = q.get()
    a[0] = 999.0
    np.testing.assert_array_equal(q.get_processed_left_over()[0], [7.0, 8.0])


def test_index_mismatch_returns_unclamped_delay(caplog):
    # Mirrors upstream _check_and_resolve_delays: warn and return real_delay
    # unchanged when the observed index delta disagrees.
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(6, 1), chunk(6, 1, 10.0), real_delay=0)
    q.get()
    q.get()
    with caplog.at_level("WARNING"):
        q.merge(chunk(6, 1), chunk(6, 1, 20.0), real_delay=1,
                action_index_before_inference=0)
    assert "real delay" in caplog.text.lower()
    assert q.qsize() == 5


def test_concurrent_get_and_merge_do_not_corrupt():
    import threading

    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(200, 1), chunk(200, 1), real_delay=0)
    errors = []

    def consume():
        try:
            for _ in range(100):
                q.get()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def produce():
        try:
            for _ in range(20):
                q.merge(chunk(5, 1), chunk(5, 1), real_delay=0)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=consume), threading.Thread(target=produce)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_action_queue.py -v`
Expected: FAIL — no module named `vla.controller.action_queue`

- [ ] **Step 4: Implement `src/vla/controller/action_queue.py`**

```python
"""Numpy port of lerobot `policies/rtc/action_queue.py::ActionQueue`.

Method names deliberately match upstream so future upstream diffs remain
reviewable against this file. Upstream's only torch usage is `.clone()`,
`torch.cat`, and slicing, so this port is mechanical:
`.clone()` -> `.copy()`, `torch.cat` -> `np.concatenate`.

Two parallel arrays are maintained:
  original_queue  policy-space actions, the source of `prev_chunk_left_over`
  queue           postprocessed actions, what the robot actually executes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueueSettings:
    rtc_enabled: bool = False


class ActionQueue:
    def __init__(self, cfg: QueueSettings) -> None:
        self.queue: np.ndarray | None = None
        self.original_queue: np.ndarray | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> np.ndarray | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        with self.lock:
            return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        """Unconsumed *original* actions — this is `prev_chunk_left_over`."""
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index:].copy()

    def get_processed_left_over(self) -> np.ndarray | None:
        """Unconsumed *processed* actions — what the robot still has queued."""
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index:].copy()

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> None:
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)
            if self.cfg.rtc_enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
                return
            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(
        self, original_actions: np.ndarray, processed_actions: np.ndarray, real_delay: int
    ) -> None:
        """Discard the first `real_delay` actions: they elapsed during inference."""
        clamped = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped:].copy()
        self.queue = processed_actions[clamped:].copy()
        self.last_index = 0

    def _append_actions_queue(
        self, original_actions: np.ndarray, processed_actions: np.ndarray
    ) -> None:
        if self.queue is None:
            self.original_queue = original_actions.copy()
            self.queue = processed_actions.copy()
            return
        self.original_queue = np.concatenate([self.original_queue, original_actions.copy()])
        self.original_queue = self.original_queue[self.last_index:]
        self.queue = np.concatenate([self.queue, processed_actions.copy()])
        self.queue = self.queue[self.last_index:]
        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        effective_delay = max(0, real_delay)
        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                LOGGER.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay
        return effective_delay
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_action_queue.py -v`
Expected: 14 passed

- [ ] **Step 6: Commit**

```bash
git add src/vla/controller/action_queue.py tests/controller/test_action_queue.py
git commit -m "feat: port lerobot ActionQueue to numpy"
```

---

## Task 11: ActionQueue differential test

This is the centerpiece of the test suite. It converts "I hand-ported subtle code" into a mechanically checked property by running identical operation sequences through both implementations.

**Files:**
- Test: `tests/controller/test_action_queue_differential.py`

- [ ] **Step 1: Write the differential test**

```python
"""Run identical operation sequences through the numpy port and upstream torch
ActionQueue, asserting they agree. Requires lerobot: `uv sync --extra lerobot`.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.differential

torch = pytest.importorskip("torch")
upstream_mod = pytest.importorskip("lerobot.policies.rtc.action_queue")
rtc_config_mod = pytest.importorskip("lerobot.policies.rtc.configuration_rtc")

from vla.controller.action_queue import ActionQueue as NumpyQueue, QueueSettings


def _pair(rtc_enabled: bool):
    upstream = upstream_mod.ActionQueue(rtc_config_mod.RTCConfig(enabled=rtc_enabled))
    ours = NumpyQueue(QueueSettings(rtc_enabled=rtc_enabled))
    return upstream, ours


def _chunk(rng, n, dim):
    a = rng.standard_normal((n, dim)).astype(np.float32)
    return a, torch.from_numpy(a.copy())


def _assert_same_state(upstream, ours):
    assert upstream.qsize() == ours.qsize()
    assert upstream.empty() == ours.empty()
    assert upstream.get_action_index() == ours.get_action_index()

    u_left, o_left = upstream.get_left_over(), ours.get_left_over()
    assert (u_left is None) == (o_left is None)
    if u_left is not None:
        np.testing.assert_allclose(u_left.numpy(), o_left, rtol=1e-6)

    u_proc, o_proc = upstream.get_processed_left_over(), ours.get_processed_left_over()
    assert (u_proc is None) == (o_proc is None)
    if u_proc is not None:
        np.testing.assert_allclose(u_proc.numpy(), o_proc, rtol=1e-6)


@pytest.mark.parametrize("rtc_enabled", [False, True])
def test_randomized_operation_sequences_agree(rtc_enabled):
    rng = np.random.default_rng(1234)
    upstream, ours = _pair(rtc_enabled)

    for step in range(200):
        op = rng.integers(0, 10)
        if op < 3:
            n = int(rng.integers(1, 12))
            orig_np, orig_t = _chunk(rng, n, 4)
            proc_np, proc_t = _chunk(rng, n, 4)
            delay = int(rng.integers(-2, 15))
            idx_before = int(rng.integers(0, 5)) if rng.random() < 0.5 else None
            upstream.merge(orig_t, proc_t, delay, idx_before)
            ours.merge(orig_np, proc_np, delay, idx_before)
        elif op < 9:
            u, o = upstream.get(), ours.get()
            assert (u is None) == (o is None), f"divergence at step {step}"
            if u is not None:
                np.testing.assert_allclose(u.numpy(), o, rtol=1e-6)
        else:
            upstream.clear()
            ours.clear()
        _assert_same_state(upstream, ours)


@pytest.mark.parametrize("delay", [-5, 0, 1, 3, 999])
def test_rtc_delay_trimming_agrees(delay):
    upstream, ours = _pair(True)
    rng = np.random.default_rng(7)
    orig_np, orig_t = _chunk(rng, 6, 3)
    proc_np, proc_t = _chunk(rng, 6, 3)
    upstream.merge(orig_t, proc_t, delay)
    ours.merge(orig_np, proc_np, delay)
    _assert_same_state(upstream, ours)


def test_index_mismatch_branch_agrees():
    """The branch that logs and returns the UNCLAMPED real_delay."""
    upstream, ours = _pair(True)
    rng = np.random.default_rng(11)
    orig_np, orig_t = _chunk(rng, 8, 2)
    proc_np, proc_t = _chunk(rng, 8, 2)
    upstream.merge(orig_t, proc_t, 0)
    ours.merge(orig_np, proc_np, 0)
    for _ in range(3):
        upstream.get()
        ours.get()
    orig_np2, orig_t2 = _chunk(rng, 8, 2)
    proc_np2, proc_t2 = _chunk(rng, 8, 2)
    upstream.merge(orig_t2, proc_t2, 1, 0)  # indexes_diff=3 != real_delay=1
    ours.merge(orig_np2, proc_np2, 1, 0)
    _assert_same_state(upstream, ours)
```

- [ ] **Step 2: Run the differential test**

Run: `uv run pytest tests/controller/test_action_queue_differential.py -v -m differential`
Expected: 8 passed

If any case diverges, **fix the port, not the test** — upstream is the reference.

- [ ] **Step 3: Add CI job**

Create `.github/workflows/differential.yml`:

```yaml
name: differential
on: [push, pull_request]
jobs:
  action-queue-vs-upstream:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        lerobot: ["pinned", "main"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra lerobot
      - name: Track lerobot main
        if: matrix.lerobot == 'main'
        run: uv pip install "lerobot[smolvla,evo1] @ git+https://github.com/huggingface/lerobot@main"
      - run: uv run pytest -m differential -v
```

The `main` leg is the point: if upstream changes merge semantics, the build breaks instead of the robot. Allow it to fail loudly rather than silently skipping.

- [ ] **Step 4: Commit**

```bash
git add tests/controller/test_action_queue_differential.py .github/workflows/differential.yml
git commit -m "test: add differential test of ActionQueue port against upstream"
```

---

## Task 12: Gripper adapters

Five variants, because Viam offers three components that can carry a gripper value at different fidelity. See spec "Gripper block".

**Files:**
- Create: `src/vla/controller/gripper.py`
- Test: `tests/controller/test_gripper.py`, `tests/fakes.py`

- [ ] **Step 1: Write fakes**

`tests/fakes.py`:

```python
"""Minimal fakes standing in for Viam resources."""

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
```

- [ ] **Step 2: Write the failing tests**

`tests/controller/test_gripper.py`:

```python
import pytest
from vla.controller.gripper import make_gripper_adapter, GripperConfigError
from tests.fakes import FakeGripper, FakeServo


def test_none_adapter_contributes_nothing():
    a = make_gripper_adapter({"type": "none"}, {})
    assert a.in_state is False
    assert a.dependency_name is None


def test_arm_joint_adapter_is_carried_by_the_arm():
    a = make_gripper_adapter({"type": "arm_joint", "joint_index": 5}, {})
    assert a.in_state is True
    assert a.dependency_name is None       # no separate resource
    assert a.uses_degrees is True
    assert a.arm_joint_index == 5


def test_arm_joint_requires_index():
    with pytest.raises(GripperConfigError, match="joint_index"):
        make_gripper_adapter({"type": "arm_joint"}, {})


async def test_servo_adapter_reads_normalized():
    servo = FakeServo(angle=45)
    a = make_gripper_adapter({"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90},
                             {"s": servo})
    assert await a.read() == pytest.approx(0.5)
    assert a.uses_degrees is False


async def test_servo_adapter_writes_denormalized_int():
    servo = FakeServo()
    a = make_gripper_adapter({"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90},
                             {"s": servo})
    await a.write(0.25)
    assert servo.moves == [22]          # int(0.25 * 90) with 1-degree resolution


async def test_servo_write_clamps_out_of_range():
    servo = FakeServo()
    a = make_gripper_adapter({"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90},
                             {"s": servo})
    await a.write(5.0)
    assert servo.moves == [90]


async def test_gripper_inputs_mode_roundtrip():
    g = FakeGripper(inputs=[0.3])
    a = make_gripper_adapter({"type": "gripper", "name": "g", "mode": "inputs"}, {"g": g})
    assert await a.read() == pytest.approx(0.3)
    await a.write(0.8)
    assert g.sent == [[0.8]]


async def test_gripper_threshold_mode_opens_below_threshold():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g})
    await a.write(0.2)
    assert g.opened == 1
    assert g.grabbed == 0


async def test_gripper_threshold_mode_grabs_above_threshold():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g})
    await a.write(0.9)
    assert g.grabbed == 1


async def test_threshold_mode_does_not_resend_same_state():
    g = FakeGripper(inputs=[0.0])
    a = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "threshold", "close_threshold": 0.5}, {"g": g})
    await a.write(0.9)
    await a.write(0.95)
    assert g.grabbed == 1     # already closed; no redundant command


def test_close_threshold_rejected_outside_threshold_mode():
    with pytest.raises(GripperConfigError, match="close_threshold"):
        make_gripper_adapter(
            {"type": "gripper", "name": "g", "mode": "inputs", "close_threshold": 0.5}, {})


def test_unknown_type_errors():
    with pytest.raises(GripperConfigError, match="unknown"):
        make_gripper_adapter({"type": "tentacle"}, {})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_gripper.py -v`
Expected: FAIL — no module named `vla.controller.gripper`

- [ ] **Step 4: Implement `src/vla/controller/gripper.py`**

```python
"""Adapters carrying a policy's continuous gripper channel onto Viam components.

A VLA emits one continuous gripper value. Viam offers three components that can
carry it at different fidelity, so the config picks one explicitly:

  arm_joint  gripper is joint N of the arm; value is a joint angle in degrees
  servo      get_position()/move(angle); int degrees, so 1-degree resolution
  gripper + inputs     get_current_inputs()/go_to_inputs(); proportional
  gripper + threshold  read inputs, write open()/grab(); binary fallback
  none       no gripper channel
"""

from __future__ import annotations

import abc
from typing import Any, Mapping

GRIPPER_TYPES = ("arm_joint", "servo", "gripper", "none")
GRIPPER_MODES = ("inputs", "threshold")


class GripperConfigError(ValueError):
    """Raised for an invalid gripper block."""


class GripperAdapter(abc.ABC):
    in_state: bool = True
    uses_degrees: bool = False
    dependency_name: str | None = None
    arm_joint_index: int | None = None

    async def read(self) -> float:
        raise NotImplementedError

    async def write(self, value: float) -> None:
        raise NotImplementedError


class NoGripper(GripperAdapter):
    in_state = False

    async def read(self) -> float:  # pragma: no cover - never called
        raise RuntimeError("no gripper configured")

    async def write(self, value: float) -> None:  # pragma: no cover
        raise RuntimeError("no gripper configured")


class ArmJointGripper(GripperAdapter):
    """The arm already carries this channel; read/write ride the arm call."""

    uses_degrees = True

    def __init__(self, joint_index: int) -> None:
        self.arm_joint_index = joint_index


class ServoGripper(GripperAdapter):
    def __init__(self, name: str, servo: Any, min_deg: float, max_deg: float) -> None:
        if max_deg <= min_deg:
            raise GripperConfigError(f"max_deg must exceed min_deg, got {min_deg}..{max_deg}")
        self.dependency_name = name
        self._servo = servo
        self._min = float(min_deg)
        self._max = float(max_deg)

    async def read(self) -> float:
        deg = float(await self._servo.get_position())
        return (deg - self._min) / (self._max - self._min)

    async def write(self, value: float) -> None:
        clamped = min(1.0, max(0.0, float(value)))
        await self._servo.move(int(round(self._min + clamped * (self._max - self._min))))


class InputsGripper(GripperAdapter):
    def __init__(self, name: str, gripper: Any) -> None:
        self.dependency_name = name
        self._gripper = gripper

    async def read(self) -> float:
        values = await self._gripper.get_current_inputs()
        return float(values[0]) if values else 0.0

    async def write(self, value: float) -> None:
        await self._gripper.go_to_inputs([min(1.0, max(0.0, float(value)))])


class ThresholdGripper(GripperAdapter):
    """Binary fallback for drivers without `go_to_inputs`."""

    def __init__(self, name: str, gripper: Any, close_threshold: float) -> None:
        self.dependency_name = name
        self._gripper = gripper
        self._threshold = float(close_threshold)
        self._closed: bool | None = None

    async def read(self) -> float:
        values = await self._gripper.get_current_inputs()
        return float(values[0]) if values else 0.0

    async def write(self, value: float) -> None:
        should_close = float(value) >= self._threshold
        if should_close == self._closed:
            return  # avoid re-commanding the same state every tick
        self._closed = should_close
        if should_close:
            await self._gripper.grab()
        else:
            await self._gripper.open()


def make_gripper_adapter(
    raw: Mapping[str, Any] | None, dependencies: Mapping[str, Any]
) -> GripperAdapter:
    raw = raw or {"type": "none"}
    kind = raw.get("type", "none")

    if kind not in GRIPPER_TYPES:
        raise GripperConfigError(f"unknown gripper type {kind!r}, expected one of {GRIPPER_TYPES}")

    if kind != "gripper" and "close_threshold" in raw:
        raise GripperConfigError("close_threshold is only valid with type=gripper mode=threshold")

    if kind == "none":
        return NoGripper()

    if kind == "arm_joint":
        if "joint_index" not in raw:
            raise GripperConfigError("gripper type=arm_joint requires joint_index")
        return ArmJointGripper(int(raw["joint_index"]))

    name = raw.get("name")
    if not name:
        raise GripperConfigError(f"gripper type={kind} requires name")

    if kind == "servo":
        return ServoGripper(name, dependencies.get(name),
                            float(raw.get("min_deg", 0.0)), float(raw.get("max_deg", 90.0)))

    mode = raw.get("mode", "inputs")
    if mode not in GRIPPER_MODES:
        raise GripperConfigError(f"gripper mode must be one of {GRIPPER_MODES}, got {mode!r}")
    if mode == "inputs":
        if "close_threshold" in raw:
            raise GripperConfigError("close_threshold is only valid with mode=threshold")
        return InputsGripper(name, dependencies.get(name))
    return ThresholdGripper(name, dependencies.get(name), float(raw.get("close_threshold", 0.5)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_gripper.py -v`
Expected: 12 passed

- [ ] **Step 6: Commit**

```bash
git add src/vla/controller/gripper.py tests/controller/test_gripper.py tests/fakes.py
git commit -m "feat: add gripper adapters for arm_joint, servo, and gripper components"
```

---

## Task 13: Safety layer

**Files:**
- Create: `src/vla/controller/safety.py`
- Test: `tests/controller/test_safety.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from vla.controller.safety import SafetyLimits, SafetyLayer, SafetyError


def _layer(**kw):
    defaults = dict(max_joint_delta_degs=8.0, max_start_delta_degs=15.0,
                    joint_limits_degs=None, gripper_in_degrees=True)
    defaults.update(kw)
    return SafetyLayer(SafetyLimits(**defaults))


def test_rejects_nan():
    with pytest.raises(SafetyError, match="finite"):
        _layer().apply(np.array([1.0, np.nan]), current=np.array([0.0, 0.0]))


def test_rejects_inf():
    with pytest.raises(SafetyError, match="finite"):
        _layer().apply(np.array([np.inf, 1.0]), current=np.array([0.0, 0.0]))


def test_rejects_dimension_mismatch():
    with pytest.raises(SafetyError, match="dimension"):
        _layer().apply(np.array([1.0, 2.0, 3.0]), current=np.array([0.0, 0.0]))


def test_within_limits_passes_through():
    out = _layer().apply(np.array([2.0, -3.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [2.0, -3.0])


def test_delta_clamped_against_current_measured_position():
    layer = _layer(max_joint_delta_degs=5.0)
    out = layer.apply(np.array([100.0]), current=np.array([10.0]))
    np.testing.assert_allclose(out, [15.0])
    assert layer.clamp_counts["delta"] == 1


def test_delta_clamp_is_symmetric():
    out = _layer(max_joint_delta_degs=5.0).apply(np.array([-100.0]), current=np.array([10.0]))
    np.testing.assert_allclose(out, [5.0])


def test_joint_limits_clamp():
    layer = _layer(joint_limits_degs=[(-90.0, 90.0)], max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([200.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [90.0])
    assert layer.clamp_counts["limit"] == 1


def test_limit_layer_skipped_when_unset():
    layer = _layer(joint_limits_degs=None, max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([5000.0]), current=np.array([0.0]))
    np.testing.assert_allclose(out, [5000.0])
    assert layer.clamp_counts["limit"] == 0


def test_normalized_gripper_channel_clamped_to_unit_range():
    # Degrees-based limits are meaningless for a 0..1 channel; it gets [0,1] instead.
    layer = _layer(gripper_in_degrees=False, joint_limits_degs=[(-90.0, 90.0)],
                   max_joint_delta_degs=1000.0)
    out = layer.apply(np.array([10.0, 3.0]), current=np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [10.0, 1.0])


def test_start_delta_within_budget_allowed():
    _layer(max_start_delta_degs=15.0).check_start(np.array([10.0]), current=np.array([0.0]))


def test_start_delta_exceeded_refuses():
    with pytest.raises(SafetyError, match="max_start_delta_degs"):
        _layer(max_start_delta_degs=15.0).check_start(np.array([50.0]), current=np.array([0.0]))


def test_clamp_counts_accumulate_for_diagnostics():
    layer = _layer(max_joint_delta_degs=1.0)
    for _ in range(3):
        layer.apply(np.array([100.0]), current=np.array([0.0]))
    assert layer.clamp_counts["delta"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_safety.py -v`
Expected: FAIL — no module named `vla.controller.safety`

- [ ] **Step 3: Implement `src/vla/controller/safety.py`**

```python
"""Bounded-motion checks applied to every action before it reaches the arm.

Order matters: reject non-finite, check dimension, clamp delta against the
*measured* position, then clamp to joint limits. Every clamp is logged, because
persistent clamping is the signature of wrong units or wrong joint order.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


class SafetyError(RuntimeError):
    """Raised when an action cannot be made safe."""


@dataclass(frozen=True)
class SafetyLimits:
    max_joint_delta_degs: float = 8.0
    max_start_delta_degs: float = 15.0
    joint_limits_degs: list[tuple[float, float]] | None = None
    gripper_in_degrees: bool = True


class SafetyLayer:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits
        self.clamp_counts: Counter[str] = Counter()

    def _validate(self, action: np.ndarray, current: np.ndarray) -> None:
        if not np.all(np.isfinite(action)):
            raise SafetyError(f"action contains non-finite values: {action}")
        if action.shape != current.shape:
            raise SafetyError(
                f"action dimension {action.shape} does not match arm state {current.shape}"
            )

    def check_start(self, first_action: np.ndarray, current: np.ndarray) -> None:
        """Refuse to start when the first action is far from the current pose.

        A policy handed an unfamiliar initial pose can emit anything. Refusing
        beats moving slowly to a place nobody asked for.
        """
        self._validate(first_action, current)
        delta = float(np.max(np.abs(first_action - current)))
        if delta > self.limits.max_start_delta_degs:
            raise SafetyError(
                f"first action is {delta:.1f} deg from the current pose, exceeding "
                f"max_start_delta_degs={self.limits.max_start_delta_degs}; "
                "move the arm nearer the expected start pose, or raise the limit deliberately"
            )

    def apply(self, action: np.ndarray, current: np.ndarray) -> np.ndarray:
        self._validate(action, current)
        out = np.asarray(action, dtype=np.float32).copy()

        n = out.shape[0]
        gripper_idx = None
        if not self.limits.gripper_in_degrees and n > 0:
            gripper_idx = n - 1  # normalized channel: degree limits do not apply

        joint_slice = slice(0, gripper_idx if gripper_idx is not None else n)

        delta = out[joint_slice] - current[joint_slice]
        capped = np.clip(delta, -self.limits.max_joint_delta_degs, self.limits.max_joint_delta_degs)
        if not np.array_equal(delta, capped):
            self.clamp_counts["delta"] += 1
            LOGGER.warning(
                "delta clamp engaged (max %.2f deg); persistent clamping usually means "
                "wrong units or wrong joint order",
                self.limits.max_joint_delta_degs,
            )
        out[joint_slice] = current[joint_slice] + capped

        if self.limits.joint_limits_degs:
            for i, (lo, hi) in enumerate(self.limits.joint_limits_degs):
                if i >= (gripper_idx if gripper_idx is not None else n):
                    break
                clamped = float(np.clip(out[i], lo, hi))
                if clamped != float(out[i]):
                    self.clamp_counts["limit"] += 1
                    LOGGER.warning("joint %d clamped to limit [%.1f, %.1f]", i, lo, hi)
                out[i] = clamped

        if gripper_idx is not None:
            g = float(out[gripper_idx])
            clamped = min(1.0, max(0.0, g))
            if clamped != g:
                self.clamp_counts["gripper"] += 1
            out[gripper_idx] = clamped

        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_safety.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/safety.py tests/controller/test_safety.py
git commit -m "feat: add safety layer with delta, limit, and start-pose checks"
```

---

## Task 14: Observation assembly

**Files:**
- Create: `src/vla/controller/observation.py`
- Test: `tests/controller/test_observation.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from vla.controller.observation import ObservationBuilder, ObservationError
from vla.controller.gripper import make_gripper_adapter
from vla.wire import decode_image
from tests.fakes import FakeArm, FakeCamera, FakeServo


def _builder(**kw):
    defaults = dict(
        cameras={"observation.images.top": FakeCamera()},
        arm=FakeArm(positions=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        gripper=make_gripper_adapter({"type": "none"}, {}),
        state_joint_indices=[0, 1, 2, 3, 4],
        state_units="degrees",
        image_sizes={"observation.images.top": (224, 224)},
        image_encoding="jpeg",
        jpeg_quality=90,
    )
    defaults.update(kw)
    return ObservationBuilder(**defaults)


async def test_builds_state_from_selected_indices():
    obs = await _builder().build()
    np.testing.assert_allclose(obs.state, [10.0, 20.0, 30.0, 40.0, 50.0])


async def test_state_converted_to_radians():
    obs = await _builder(state_units="radians").build()
    np.testing.assert_allclose(obs.state, np.deg2rad([10.0, 20.0, 30.0, 40.0, 50.0]), rtol=1e-5)


async def test_joint_reordering_is_respected():
    obs = await _builder(state_joint_indices=[4, 3, 2, 1, 0]).build()
    np.testing.assert_allclose(obs.state, [50.0, 40.0, 30.0, 20.0, 10.0])


async def test_arm_joint_gripper_appended_from_arm():
    obs = await _builder(
        gripper=make_gripper_adapter({"type": "arm_joint", "joint_index": 5}, {})
    ).build()
    np.testing.assert_allclose(obs.state, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])


async def test_servo_gripper_appended_normalized():
    servo = FakeServo(angle=45)
    gripper = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {"s": servo})
    obs = await _builder(gripper=gripper).build()
    assert obs.state[-1] == pytest.approx(0.5)


async def test_images_resized_to_policy_expectation():
    obs = await _builder().build()
    img = decode_image(obs.images["observation.images.top"])
    assert img.shape == (224, 224, 3)


async def test_out_of_range_joint_index_errors():
    with pytest.raises(ObservationError, match="joint index"):
        await _builder(state_joint_indices=[0, 99]).build()


async def test_camera_failure_fails_the_whole_tick():
    # Never substitute a black frame or reuse a stale one: both silently
    # corrupt policy input in ways that look like bad model behavior.
    with pytest.raises(ObservationError, match="camera"):
        await _builder(cameras={"observation.images.top": FakeCamera(fail=True)}).build()


async def test_cameras_are_read_concurrently():
    # Serial reads would take ~3x one camera's latency. At 10 Hz the whole
    # tick budget is 100 ms, so this is a real constraint, not a nicety.
    import asyncio as _asyncio

    class SlowCamera(FakeCamera):
        async def get_image(self, *a, **k):
            await _asyncio.sleep(0.05)
            return await super().get_image(*a, **k)

    cams = {f"observation.images.c{i}": SlowCamera() for i in range(3)}
    b = _builder(cameras=cams, image_sizes={k: (224, 224) for k in cams})

    obs = await b.build()

    assert len(obs.images) == 3
    assert all(c.reads == 1 for c in cams.values())
    assert obs.duration_s < 0.12, f"reads look serial: {obs.duration_s:.3f}s for 3x50ms"


async def test_duration_reflects_actual_assembly_time():
    import asyncio as _asyncio

    class SlowCamera(FakeCamera):
        async def get_image(self, *a, **k):
            await _asyncio.sleep(0.05)
            return await super().get_image(*a, **k)

    obs = await _builder(cameras={"observation.images.top": SlowCamera()}).build()
    assert obs.duration_s >= 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_observation.py -v`
Expected: FAIL — no module named `vla.controller.observation`

- [ ] **Step 3: Implement `src/vla/controller/observation.py`**

```python
"""Assemble one policy observation from Viam cameras and the arm."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..wire import encode_image
from .gripper import GripperAdapter
from .units import from_degrees

LOGGER = logging.getLogger(__name__)


class ObservationError(RuntimeError):
    """Raised when an observation cannot be assembled."""


@dataclass(frozen=True)
class Observation:
    images: dict[str, dict[str, Any]]
    state: np.ndarray
    duration_s: float


class ObservationBuilder:
    def __init__(
        self,
        *,
        cameras: Mapping[str, Any],
        arm: Any,
        gripper: GripperAdapter,
        state_joint_indices: Sequence[int],
        state_units: str,
        image_sizes: Mapping[str, tuple[int, int]],
        image_encoding: str = "jpeg",
        jpeg_quality: int = 90,
    ) -> None:
        self._cameras = dict(cameras)
        self._arm = arm
        self._gripper = gripper
        self._indices = list(state_joint_indices)
        self._units = state_units
        self._sizes = dict(image_sizes)
        self._encoding = image_encoding
        self._quality = jpeg_quality

    async def build(self) -> Observation:
        started = time.perf_counter()
        keys = list(self._cameras)

        # Gathered, not sequential: at 10 Hz the whole tick has a 100 ms budget
        # and two serial camera reads can consume most of it.
        tasks = [self._cameras[k].get_image() for k in keys]
        tasks.append(self._arm.get_joint_positions())
        needs_gripper_read = self._gripper.in_state and self._gripper.arm_joint_index is None
        if needs_gripper_read:
            tasks.append(self._gripper.read())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        gripper_value = None
        if needs_gripper_read:
            gripper_value = results[-1]
            results = results[:-1]
        joints = results[-1]
        frames = results[:-1]

        for key, frame in zip(keys, frames):
            if isinstance(frame, Exception):
                raise ObservationError(f"camera {key!r} read failed: {frame}") from frame
        if isinstance(joints, Exception):
            raise ObservationError(f"arm joint read failed: {joints}") from joints
        if isinstance(gripper_value, Exception):
            raise ObservationError(f"gripper read failed: {gripper_value}") from gripper_value

        images = {
            key: self._encode(key, self._to_array(frame))
            for key, frame in zip(keys, frames)
        }
        state = self._build_state(list(joints.values), gripper_value)

        duration = time.perf_counter() - started
        return Observation(images=images, state=state, duration_s=duration)

    @staticmethod
    def _to_array(frame: Any) -> np.ndarray:
        from viam.media.utils.pil import viam_to_pil_image

        return np.asarray(viam_to_pil_image(frame).convert("RGB"), dtype=np.uint8)

    def _encode(self, key: str, arr: np.ndarray) -> dict[str, Any]:
        size = self._sizes.get(key)
        if size is None:
            # Never forward an unresized frame. A 1080p frame under
            # image_encoding="raw" base64-encodes to ~8.3 MB, which exceeds
            # typical gRPC message limits — and the policy would receive a
            # resolution it was not trained on either way.
            raise ObservationError(
                f"no expected size for {key!r}; policy specs did not declare it"
            )
        if arr.shape[:2] != size:
            arr = np.asarray(
                Image.fromarray(arr).resize((size[1], size[0]), Image.BILINEAR), dtype=np.uint8
            )
        return encode_image(arr, encoding=self._encoding, quality=self._quality)

    def _build_state(self, joint_degrees: list[float], gripper_value: float | None) -> np.ndarray:
        selected = []
        for idx in self._indices:
            if idx < 0 or idx >= len(joint_degrees):
                raise ObservationError(
                    f"state_joint_indices references joint index {idx}, but the arm reports "
                    f"{len(joint_degrees)} joints"
                )
            selected.append(joint_degrees[idx])

        converted = from_degrees(np.asarray(selected, dtype=np.float32), self._units)

        if not self._gripper.in_state:
            return converted

        if self._gripper.arm_joint_index is not None:
            idx = self._gripper.arm_joint_index
            if idx >= len(joint_degrees):
                raise ObservationError(
                    f"gripper joint index {idx} exceeds the arm's {len(joint_degrees)} joints"
                )
            tail = from_degrees(
                np.asarray([joint_degrees[idx]], dtype=np.float32), self._units
            )
        else:
            tail = np.asarray([gripper_value], dtype=np.float32)  # already normalized

        return np.concatenate([converted, tail])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_observation.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/observation.py tests/controller/test_observation.py
git commit -m "feat: add concurrent observation assembly"
```

---

## Task 15: Sequential scheduler

**Files:**
- Create: `src/vla/controller/scheduler.py`
- Test: `tests/controller/test_scheduler.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from vla.controller.scheduler import SequentialScheduler, SchedulerError


class RecordingInfer:
    """Stands in for a call to the policy service."""

    def __init__(self, n=4, dim=2, fail_after=None):
        self.calls = 0
        self.n = n
        self.dim = dim
        self.fail_after = fail_after
        self.last_rtc = "unset"

    async def __call__(self, rtc):
        self.calls += 1
        self.last_rtc = rtc
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("inference exploded")
        base = np.full((self.n, self.dim), float(self.calls), dtype=np.float32)
        return base + 100.0, base


async def test_first_tick_triggers_inference():
    infer = RecordingInfer()
    s = SequentialScheduler(infer)
    action = await s.next_action()
    assert infer.calls == 1
    np.testing.assert_allclose(action, [101.0, 101.0])


async def test_queue_drains_before_reinferring():
    infer = RecordingInfer(n=3)
    s = SequentialScheduler(infer)
    for _ in range(3):
        await s.next_action()
    assert infer.calls == 1
    await s.next_action()
    assert infer.calls == 2


async def test_serves_processed_actions_not_raw():
    infer = RecordingInfer()
    s = SequentialScheduler(infer)
    action = await s.next_action()
    assert action[0] == 101.0     # processed = raw + 100


async def test_sequential_mode_sends_no_rtc_payload():
    infer = RecordingInfer()
    s = SequentialScheduler(infer)
    await s.next_action()
    assert infer.last_rtc is None


async def test_inference_failure_propagates():
    s = SequentialScheduler(RecordingInfer(fail_after=0))
    with pytest.raises(RuntimeError, match="exploded"):
        await s.next_action()


async def test_reset_clears_the_queue():
    infer = RecordingInfer(n=5)
    s = SequentialScheduler(infer)
    await s.next_action()
    s.reset()
    await s.next_action()
    assert infer.calls == 2


async def test_qsize_reflects_remaining():
    infer = RecordingInfer(n=4)
    s = SequentialScheduler(infer)
    await s.next_action()
    assert s.qsize() == 3


async def test_empty_chunk_is_an_error():
    class EmptyInfer:
        async def __call__(self, rtc):
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    s = SequentialScheduler(EmptyInfer())
    with pytest.raises(SchedulerError, match="empty"):
        await s.next_action()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_scheduler.py -v`
Expected: FAIL — no module named `vla.controller.scheduler`

- [ ] **Step 3: Implement `src/vla/controller/scheduler.py`**

```python
"""Chunk scheduling: turn action chunks into one action per control tick.

Only the sequential strategy ships today. RTCScheduler is a follow-up plan; the
ActionQueue underneath already supports both modes.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Awaitable, Callable

import numpy as np

from .action_queue import ActionQueue, QueueSettings

LOGGER = logging.getLogger(__name__)

InferFn = Callable[[dict[str, Any] | None], Awaitable[tuple[np.ndarray, np.ndarray]]]


class SchedulerError(RuntimeError):
    """Raised when the scheduler cannot produce an action."""


class ChunkScheduler(abc.ABC):
    @abc.abstractmethod
    async def next_action(self) -> np.ndarray | None:
        """Return the action for this tick, or None if none is available."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode-scoped state."""

    @abc.abstractmethod
    def qsize(self) -> int:
        """Actions remaining in the queue."""


class SequentialScheduler(ChunkScheduler):
    """Blocking: when the queue drains, infer and refill.

    Inference latency directly stalls the control loop, which is exactly what
    RTC exists to fix — but it is simple, and correct behavior here is the
    baseline the RTC path will be compared against.
    """

    def __init__(self, infer: InferFn) -> None:
        self._infer = infer
        self._queue = ActionQueue(QueueSettings(rtc_enabled=False))

    async def next_action(self) -> np.ndarray:
        action = self._queue.get()
        if action is not None:
            return action

        processed, raw = await self._infer(None)
        if processed.shape[0] == 0:
            raise SchedulerError("policy returned an empty action chunk")
        self._queue.merge(raw, processed, real_delay=0)

        action = self._queue.get()
        if action is None:  # pragma: no cover - guarded by the shape check above
            raise SchedulerError("queue empty immediately after merge")
        return action

    def reset(self) -> None:
        self._queue.clear()

    def qsize(self) -> int:
        return self._queue.qsize()
```

Note this scheduler never returns `None` — it raises instead. The controller's
`starvation_grace_ticks` branch is therefore unreachable in phases 1–2; it exists
for `RTCScheduler`, where a background thread can genuinely leave the queue empty
for a tick. Keep the branch and its config field rather than adding them later.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_scheduler.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/scheduler.py tests/controller/test_scheduler.py
git commit -m "feat: add sequential chunk scheduler"
```

---

## Task 16: Controller config

> **Use `src/vla/config_util.py`.** The code below predates that module and
> hand-rolls its coercion and range checks. Every `float(raw.get(...))`,
> `int(raw.get(...))`, and `if x not in TUPLE` here should be a call to
> `as_float` / `as_int` / `as_choice` with `minimum=`/`maximum=` folded in.
> Import `ConfigError` from `vla.config_util`, do **not** define a second one —
> two `ConfigError` classes means `validate_config` must catch both or let one
> escape unhandled. This config has ~10 numeric fields plus five in the nested
> `safety` block, which is exactly why the helpers were extracted.

**Files:**
- Create: `src/vla/controller/config.py`
- Test: `tests/controller/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from vla.controller.config import ControllerConfig, ConfigError


BASE = {
    "policy_service": "vla-policy",
    "arm": "my-arm",
    "cameras": {"observation.images.top": "cam-top"},
    "state_joint_indices": [0, 1, 2, 3, 4],
}


def test_parses_minimal_config():
    cfg = ControllerConfig.parse(BASE)
    assert cfg.policy_service == "vla-policy"
    assert cfg.arm == "my-arm"
    assert cfg.fps == 10.0
    assert cfg.mode == "auto"
    assert cfg.policy_ready_timeout_s == 600


def test_dependencies_include_policy_arm_and_cameras():
    assert set(ControllerConfig.parse(BASE).dependencies()) == {
        "vla-policy", "my-arm", "cam-top"}


def test_servo_gripper_adds_dependency():
    cfg = ControllerConfig.parse(
        {**BASE, "gripper": {"type": "servo", "name": "grip", "min_deg": 0, "max_deg": 90}})
    assert "grip" in cfg.dependencies()


def test_arm_joint_gripper_adds_no_dependency():
    cfg = ControllerConfig.parse({**BASE, "gripper": {"type": "arm_joint", "joint_index": 5}})
    assert set(cfg.dependencies()) == {"vla-policy", "my-arm", "cam-top"}


def test_requires_policy_service():
    with pytest.raises(ConfigError, match="policy_service"):
        ControllerConfig.parse({k: v for k, v in BASE.items() if k != "policy_service"})


def test_requires_arm():
    with pytest.raises(ConfigError, match="arm"):
        ControllerConfig.parse({k: v for k, v in BASE.items() if k != "arm"})


def test_requires_at_least_one_camera():
    with pytest.raises(ConfigError, match="cameras"):
        ControllerConfig.parse({**BASE, "cameras": {}})


def test_rejects_nonpositive_fps():
    with pytest.raises(ConfigError, match="fps"):
        ControllerConfig.parse({**BASE, "fps": 0})


def test_rejects_unknown_mode():
    with pytest.raises(ConfigError, match="mode"):
        ControllerConfig.parse({**BASE, "mode": "turbo"})


def test_rejects_duplicate_joint_indices():
    with pytest.raises(ConfigError, match="duplicate"):
        ControllerConfig.parse({**BASE, "state_joint_indices": [0, 1, 1]})


def test_joint_limits_length_must_match_arm_joint_gripper():
    with pytest.raises(ConfigError, match="joint_limits_degs"):
        ControllerConfig.parse({
            **BASE,
            "gripper": {"type": "arm_joint", "joint_index": 5},
            "safety": {"joint_limits_degs": [[-90, 90]] * 5},   # needs 6
        })


def test_joint_limits_length_matches_without_degree_gripper():
    cfg = ControllerConfig.parse({
        **BASE,
        "gripper": {"type": "servo", "name": "grip"},
        "safety": {"joint_limits_degs": [[-90, 90]] * 5},        # no trailing pair
    })
    assert len(cfg.safety.joint_limits_degs) == 5


def test_rejects_inverted_joint_limit():
    with pytest.raises(ConfigError, match="min"):
        ControllerConfig.parse({**BASE, "safety": {"joint_limits_degs": [[90, -90]] * 5}})


def test_rejects_unknown_image_encoding():
    with pytest.raises(ConfigError, match="image_encoding"):
        ControllerConfig.parse({**BASE, "image_encoding": "webp"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_config.py -v`
Expected: FAIL — no module named `vla.controller.config`

- [ ] **Step 3: Implement `src/vla/controller/config.py`**

```python
"""Configuration parsing and validation for viam-labs:vla:controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gripper import GRIPPER_TYPES

MODES = ("auto", "sequential", "rtc")
ENCODINGS = ("jpeg", "png", "raw")
UNITS = ("degrees", "radians", "normalized")


class ConfigError(ValueError):
    """Raised for invalid controller configuration."""


@dataclass(frozen=True)
class SafetyConfig:
    max_joint_delta_degs: float = 8.0
    max_start_delta_degs: float = 15.0
    max_vel_degs_per_sec: float | None = None
    max_acc_degs_per_sec2: float | None = None
    max_tcp_speed_m_per_sec: float | None = None
    joint_limits_degs: list[tuple[float, float]] | None = None
    stop_on_error: bool = True

    @staticmethod
    def parse(raw: dict[str, Any]) -> "SafetyConfig":
        limits = raw.get("joint_limits_degs")
        parsed: list[tuple[float, float]] | None = None
        if limits is not None:
            parsed = []
            for i, pair in enumerate(limits):
                if len(pair) != 2:
                    raise ConfigError(f"joint_limits_degs[{i}] must be [min, max]")
                lo, hi = float(pair[0]), float(pair[1])
                if hi <= lo:
                    raise ConfigError(f"joint_limits_degs[{i}]: min {lo} must be below max {hi}")
                parsed.append((lo, hi))

        def _optional_positive(key: str) -> float | None:
            if key not in raw or raw[key] is None:
                return None
            value = float(raw[key])
            if value <= 0:
                raise ConfigError(f"safety.{key} must be positive, got {value}")
            return value

        return SafetyConfig(
            max_joint_delta_degs=float(raw.get("max_joint_delta_degs", 8.0)),
            max_start_delta_degs=float(raw.get("max_start_delta_degs", 15.0)),
            max_vel_degs_per_sec=_optional_positive("max_vel_degs_per_sec"),
            max_acc_degs_per_sec2=_optional_positive("max_acc_degs_per_sec2"),
            max_tcp_speed_m_per_sec=_optional_positive("max_tcp_speed_m_per_sec"),
            joint_limits_degs=parsed,
            stop_on_error=bool(raw.get("stop_on_error", True)),
        )


@dataclass(frozen=True)
class ControllerConfig:
    policy_service: str
    arm: str
    cameras: dict[str, str]
    state_joint_indices: list[int]
    gripper: dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    task: str = ""
    fps: float = 10.0
    mode: str = "auto"
    queue_threshold: int = 30
    starvation_grace_ticks: int = 3
    policy_ready_timeout_s: int = 600
    state_units: str = "degrees"
    action_units: str = "degrees"
    image_encoding: str = "jpeg"
    jpeg_quality: int = 90
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @staticmethod
    def parse(raw: dict[str, Any]) -> "ControllerConfig":
        policy_service = raw.get("policy_service")
        if not policy_service:
            raise ConfigError("policy_service is required")
        arm = raw.get("arm")
        if not arm:
            raise ConfigError("arm is required")
        cameras = dict(raw.get("cameras") or {})
        if not cameras:
            raise ConfigError("cameras must name at least one camera")

        indices = [int(i) for i in (raw.get("state_joint_indices") or [])]
        if not indices:
            raise ConfigError("state_joint_indices is required")
        if len(set(indices)) != len(indices):
            raise ConfigError(f"state_joint_indices contains duplicate entries: {indices}")

        mode = raw.get("mode", "auto")
        if mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")

        fps = float(raw.get("fps", 10.0))
        if fps <= 0:
            raise ConfigError(f"fps must be positive, got {fps}")

        encoding = raw.get("image_encoding", "jpeg")
        if encoding not in ENCODINGS:
            raise ConfigError(f"image_encoding must be one of {ENCODINGS}, got {encoding!r}")

        for key in ("state_units", "action_units"):
            unit = raw.get(key, "degrees")
            if unit not in UNITS:
                raise ConfigError(f"{key} must be one of {UNITS}, got {unit!r}")

        gripper = dict(raw.get("gripper") or {"type": "none"})
        if gripper.get("type", "none") not in GRIPPER_TYPES:
            raise ConfigError(f"gripper.type must be one of {GRIPPER_TYPES}")

        safety = SafetyConfig.parse(raw.get("safety") or {})

        # The trailing gripper limit pair exists only when the gripper channel is
        # in degrees, which is exactly the arm_joint case.
        if safety.joint_limits_degs is not None:
            expected = len(indices) + (1 if gripper.get("type") == "arm_joint" else 0)
            if len(safety.joint_limits_degs) != expected:
                raise ConfigError(
                    f"safety.joint_limits_degs has {len(safety.joint_limits_degs)} entries, "
                    f"expected {expected} (one per action dimension in degrees)"
                )

        return ControllerConfig(
            policy_service=policy_service,
            arm=arm,
            cameras=cameras,
            state_joint_indices=indices,
            gripper=gripper,
            task=raw.get("task", ""),
            fps=fps,
            mode=mode,
            queue_threshold=int(raw.get("queue_threshold", 30)),
            starvation_grace_ticks=int(raw.get("starvation_grace_ticks", 3)),
            policy_ready_timeout_s=int(raw.get("policy_ready_timeout_s", 600)),
            state_units=raw.get("state_units", "degrees"),
            action_units=raw.get("action_units", "degrees"),
            image_encoding=encoding,
            jpeg_quality=int(raw.get("jpeg_quality", 90)),
            safety=safety,
        )

    def dependencies(self) -> list[str]:
        deps = [self.policy_service, self.arm, *self.cameras.values()]
        name = self.gripper.get("name")
        if self.gripper.get("type") in ("servo", "gripper") and name:
            deps.append(name)
        return deps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_config.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/config.py tests/controller/test_config.py
git commit -m "feat: add controller config parsing and validation"
```

---

## Task 17: Controller service

The largest task. Wires everything into a lifecycle and a control loop.

**Files:**
- Create: `src/vla/controller/service.py`
- Test: `tests/controller/test_service.py`

- [ ] **Step 1: Write the failing tests**

```python
import asyncio
import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig

from vla.controller.service import VLAController
from vla.wire import encode_matrix
from tests.fakes import FakeArm, FakeCamera


class FakePolicyClient:
    def __init__(self, action_dim=5, n=4, state="ready", relative=False, supports_rtc=True):
        self.action_dim = action_dim
        self.n = n
        self.state = state
        self.relative = relative
        self.supports_rtc = supports_rtc
        self.infer_calls = 0
        self.fail_infer = False

    async def do_command(self, command, **kwargs):
        name = command.get("command")
        if name == "status":
            return {"state": self.state, "error": ""}
        if name == "specs":
            # Numbers are floats on purpose. A real call crosses gRPC, and
            # protobuf Struct stores every number as a double, so the controller
            # never sees ints here. Returning ints would hide that from tests.
            return {
                "policy_type": "fake",
                "action_dim": float(self.action_dim),
                "n_action_steps": float(self.n),
                "input_features": {"observation.images.top": [3.0, 224.0, 224.0],
                                   "observation.state": [float(self.action_dim)]},
                "output_features": {"action": [float(self.action_dim)]},
                "image_feature_keys": ["observation.images.top"],
                "supports_rtc": self.supports_rtc,
                "rtc_enabled": False,
                "relative_actions": self.relative,
                "device": "cpu",
            }
        if name == "infer":
            self.infer_calls += 1
            if self.fail_infer:
                raise RuntimeError("inference exploded")
            chunk = np.zeros((self.n, self.action_dim), dtype=np.float32)
            return {"actions": encode_matrix(chunk),
                    "raw_actions": encode_matrix(chunk),
                    "latency_s": 0.001}
        raise ValueError(name)


def _config(**overrides):
    attrs = {
        "policy_service": "p",
        "arm": "a",
        "cameras": {"observation.images.top": "cam"},
        "state_joint_indices": [0, 1, 2, 3, 4],
        "fps": 50.0,
        "safety": {"max_start_delta_degs": 1000.0},
    }
    attrs.update(overrides)
    s = Struct()
    s.update(attrs)
    return ServiceConfig(name="c", api="rdk:service:generic",
                         model="viam-labs:vla:controller", attributes=s)


def _deps(policy=None, arm=None, camera=None):
    return {"p": policy or FakePolicyClient(),
            "a": arm or FakeArm(positions=[0.0] * 6),
            "cam": camera or FakeCamera()}


def _svc(config=None, deps=None):
    svc = VLAController("c")
    svc.reconfigure(config or _config(), deps or _deps())
    return svc


async def test_validate_returns_dependencies():
    required, optional = VLAController.validate_config(_config())
    assert set(required) == {"p", "a", "cam"}


async def test_reconfigure_succeeds_with_cold_policy():
    # First boot is the normal case; reconfigure must not fail on it.
    policy = FakePolicyClient(state="loading")
    svc = _svc(deps=_deps(policy=policy))
    assert (await svc.do_command({"command": "status"}))["state"] == "idle"


async def test_status_starts_idle():
    assert (await _svc().do_command({"command": "status"}))["state"] == "idle"


async def test_start_acks_immediately_with_cold_policy():
    policy = FakePolicyClient(state="loading")
    svc = _svc(deps=_deps(policy=policy))
    out = await svc.do_command({"command": "start", "task": "t"})
    assert out["ok"] is True
    assert (await svc.do_command({"command": "status"}))["state"] == "waiting_for_policy"
    await svc.do_command({"command": "stop"})


async def test_loop_runs_and_commands_the_arm():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.3)
    await svc.do_command({"command": "stop"})
    assert policy.infer_calls >= 1
    assert len(arm.moves) >= 1


async def test_arm_commanded_via_move_through_joint_positions_with_options():
    # move_to_joint_positions accepts no MoveOptions, so using it would silently
    # drop every kinematic ceiling in the safety config.
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(config=_config(safety={"max_start_delta_degs": 1000.0,
                                      "max_vel_degs_per_sec": 30.0}),
               deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    positions, options = arm.moves[0]
    assert len(positions) == 1
    assert options is not None
    assert options.HasField("max_vel_degs_per_sec")


async def test_unset_move_options_fields_are_omitted_not_zeroed():
    # An unset scalar reads back as 0.0, indistinguishable from an explicit
    # zero, which would tell the arm not to move. Setting only acceleration
    # must leave velocity genuinely absent.
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(config=_config(safety={"max_start_delta_degs": 1000.0,
                                      "max_acc_degs_per_sec2": 100.0}),
               deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    _, options = arm.moves[0]
    assert options.HasField("max_acc_degs_per_sec2")
    assert not options.HasField("max_vel_degs_per_sec")
    assert not options.HasField("max_tcp_speed")


async def test_no_safety_ceilings_means_no_move_options():
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    _, options = arm.moves[0]
    assert options is None


async def test_stop_halts_the_loop():
    policy = FakePolicyClient()
    svc = _svc(deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    calls = policy.infer_calls
    await asyncio.sleep(0.2)
    assert policy.infer_calls == calls


async def test_inference_failure_stops_arm_and_records_error():
    arm = FakeArm(positions=[0.0] * 6)
    policy = FakePolicyClient()
    policy.fail_infer = True
    svc = _svc(deps=_deps(policy=policy, arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.3)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "error"
    assert "exploded" in status["last_error"]
    assert arm.stopped >= 1


async def test_arm_failure_stops_immediately():
    arm = FakeArm(positions=[0.0] * 6)
    arm.fail_next_move = True
    svc = _svc(deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.3)
    assert (await svc.do_command({"command": "status"}))["state"] == "error"
    assert arm.stopped >= 1


async def test_action_dim_mismatch_refuses_to_start():
    policy = FakePolicyClient(action_dim=99)
    svc = _svc(config=_config(safety={"joint_limits_degs": [[-90, 90]] * 5,
                                      "max_start_delta_degs": 1000.0}),
               deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "error"
    assert "action_dim" in status["last_error"]


async def test_rtc_mode_refuses_relative_action_checkpoint():
    policy = FakePolicyClient(relative=True)
    svc = _svc(config=_config(mode="rtc"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "error"
    assert "relative" in status["last_error"].lower()


async def test_rtc_mode_refuses_policy_without_rtc_support():
    policy = FakePolicyClient(supports_rtc=False)
    svc = _svc(config=_config(mode="rtc"), deps=_deps(policy=policy))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    assert (await svc.do_command({"command": "status"}))["state"] == "error"


async def test_reconfigure_while_running_stops_and_does_not_resume():
    arm = FakeArm(positions=[0.0] * 6)
    deps = _deps(arm=arm)
    svc = _svc(deps=deps)
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    svc.reconfigure(_config(), deps)
    await asyncio.sleep(0.2)
    assert (await svc.do_command({"command": "status"}))["state"] == "idle"


async def test_close_stops_the_arm():
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.close()
    assert arm.stopped >= 1


async def test_status_reports_clamp_counts():
    svc = _svc()
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.2)
    await svc.do_command({"command": "stop"})
    assert "clamp_counts" in await svc.do_command({"command": "status"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/controller/test_service.py -v`
Expected: FAIL — no module named `vla.controller.service`

- [ ] **Step 3: Implement `src/vla/controller/service.py`**

```python
"""viam-labs:vla:controller — the observation/inference/actuation loop."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
from typing_extensions import Self
from viam.proto.app.robot import ServiceConfig
from viam.proto.component.arm import JointPositions, MoveOptions
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

from ..wire import decode_matrix, encode_vector
from .config import ControllerConfig
from .gripper import make_gripper_adapter
from .observation import ObservationBuilder
from .safety import SafetyLayer, SafetyLimits
from .scheduler import SequentialScheduler
from .units import to_degrees

LOGGER = logging.getLogger(__name__)


class VLAController(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vla"), "controller")

    def __init__(self, name: str):
        super().__init__(name)
        self._cfg: ControllerConfig | None = None
        self._deps: Mapping[str, Any] = {}
        self._state = "idle"
        self._last_error: str | None = None
        self._task: asyncio.Task | None = None
        self._active_task_text = ""
        self._safety: SafetyLayer | None = None
        self._specs: dict[str, Any] | None = None
        self._latencies: list[float] = []
        self._measured_fps = 0.0
        self._scheduler = None
        self._stop_task: asyncio.Task | None = None

    @classmethod
    def new(cls, config: ServiceConfig, dependencies: Mapping[Any, ResourceBase]) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> tuple[Sequence[str], Sequence[str]]:
        cfg = ControllerConfig.parse(struct_to_dict(config.attributes))
        return cfg.dependencies(), []

    def reconfigure(self, config: ServiceConfig, dependencies) -> None:
        """Rebuild from config. Never auto-resumes motion after a config change."""
        was_running = self._task is not None and not self._task.done()
        # Capture the arm currently in motion before rebinding _cfg/_deps below,
        # or the scheduled stop would target the *new* arm.
        old_arm = self._deps.get(self._cfg.arm) if (was_running and self._cfg) else None
        self._stop_loop_sync()
        if old_arm is not None:
            # reconfigure is sync, so the stop is scheduled rather than awaited.
            # Cancelling the loop alone leaves the arm executing its last command.
            # Keep a reference: CPython may garbage-collect a bare task handle
            # before it ever runs.
            self._stop_task = asyncio.create_task(self._stop_arm(old_arm))
        self._cfg = ControllerConfig.parse(struct_to_dict(config.attributes))
        self._deps = {self._key(k): v for k, v in dependencies.items()}
        self._specs = None
        self._scheduler = None
        self._state = "idle"
        self._last_error = None

    @staticmethod
    def _key(dep_key: Any) -> str:
        return getattr(dep_key, "name", str(dep_key))

    def _resource(self, name: str) -> Any:
        if name not in self._deps:
            raise RuntimeError(f"dependency {name!r} was not resolved")
        return self._deps[name]

    async def do_command(self, command: Mapping[str, Any], *, timeout=None, **kwargs):
        name = command.get("command")
        if name == "start":
            return await self._start(command)
        if name == "stop":
            await self._stop()
            return {"ok": True}
        if name == "status":
            return self._status()
        raise ValueError(f"unknown command {name!r}")

    def _status(self) -> dict[str, Any]:
        avg = float(np.mean(self._latencies)) if self._latencies else 0.0
        return {
            "state": self._state,
            "mode": (self._specs or {}).get("_resolved_mode", self._cfg.mode if self._cfg else ""),
            "queue_size": self._scheduler.qsize() if self._scheduler else 0,
            "avg_latency_s": avg,
            "measured_fps": self._measured_fps,
            "clamp_counts": dict(self._safety.clamp_counts) if self._safety else {},
            "last_error": self._last_error or "",
        }

    async def _start(self, command: Mapping[str, Any]) -> dict[str, Any]:
        if self._task and not self._task.done():
            raise RuntimeError("already running")
        assert self._cfg is not None
        self._active_task_text = command.get("task") or self._cfg.task
        if not self._active_task_text:
            raise ValueError("no task instruction: pass 'task' or configure a default")
        self._last_error = None
        self._state = "waiting_for_policy"
        # Ack immediately: waiting out policy_ready_timeout_s inline would exceed
        # the deadline most DoCommand callers use.
        self._task = asyncio.create_task(self._run())
        return {"ok": True}

    async def _stop(self) -> None:
        self._stop_loop_sync()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._safe_stop_arm()
        if self._state not in ("error",):
            self._state = "idle"

    def _stop_loop_sync(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    @staticmethod
    async def _stop_arm(arm: Any) -> None:
        if arm is None:
            return
        try:
            await arm.stop()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("failed to stop arm: %s", exc)

    async def _safe_stop_arm(self) -> None:
        if not self._cfg:
            return
        await self._stop_arm(self._deps.get(self._cfg.arm))

    async def _await_policy(self) -> dict[str, Any]:
        cfg = self._cfg
        policy = self._resource(cfg.policy_service)
        deadline = time.monotonic() + cfg.policy_ready_timeout_s
        delay = 0.05
        while time.monotonic() < deadline:
            status = await policy.do_command({"command": "status"})
            if status.get("state") == "ready":
                return await policy.do_command({"command": "specs"})
            if status.get("state") == "failed":
                raise RuntimeError(f"policy failed to load: {status.get('error')}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
        raise TimeoutError(
            f"policy not ready after {cfg.policy_ready_timeout_s}s"
        )

    def _resolve_mode(self, specs: dict[str, Any]) -> str:
        cfg = self._cfg
        if cfg.mode == "sequential":
            return "sequential"
        if cfg.mode == "rtc":
            if not specs.get("supports_rtc"):
                raise RuntimeError("mode=rtc but the policy does not support RTC")
            if specs.get("relative_actions"):
                raise RuntimeError(
                    "mode=rtc but the checkpoint uses relative actions, which requires "
                    "prefix re-anchoring that is not implemented; use mode=sequential"
                )
            raise RuntimeError("mode=rtc is not implemented yet; use mode=sequential or auto")
        return "sequential"  # auto: RTC scheduler ships in a follow-up plan

    def _build_safety(self, gripper) -> SafetyLayer:
        s = self._cfg.safety
        return SafetyLayer(SafetyLimits(
            max_joint_delta_degs=s.max_joint_delta_degs,
            max_start_delta_degs=s.max_start_delta_degs,
            joint_limits_degs=s.joint_limits_degs,
            gripper_in_degrees=(not gripper.in_state) or gripper.uses_degrees,
        ))

    def _move_options(self) -> MoveOptions | None:
        """Build MoveOptions with only configured fields set.

        Unset scalars read back as 0.0, indistinguishable from an explicit zero,
        which would tell the arm not to move.
        """
        s = self._cfg.safety
        kwargs = {}
        if s.max_vel_degs_per_sec is not None:
            kwargs["max_vel_degs_per_sec"] = s.max_vel_degs_per_sec
        if s.max_acc_degs_per_sec2 is not None:
            kwargs["max_acc_degs_per_sec2"] = s.max_acc_degs_per_sec2
        if s.max_tcp_speed_m_per_sec is not None:
            kwargs["max_tcp_speed"] = s.max_tcp_speed_m_per_sec
        return MoveOptions(**kwargs) if kwargs else None

    async def _run(self) -> None:
        cfg = self._cfg
        try:
            specs = await self._await_policy()
            mode = self._resolve_mode(specs)
            specs["_resolved_mode"] = mode
            self._specs = specs

            gripper = make_gripper_adapter(cfg.gripper, self._deps)
            expected_dim = len(cfg.state_joint_indices) + (1 if gripper.in_state else 0)
            if int(specs["action_dim"]) != expected_dim:
                raise RuntimeError(
                    f"action_dim mismatch: policy emits {specs['action_dim']} dimensions but "
                    f"config describes {expected_dim} "
                    f"({len(cfg.state_joint_indices)} joints + "
                    f"{'1 gripper' if gripper.in_state else 'no gripper'})"
                )

            # int() is mandatory, not defensive. specs arrives via
            # GenericClient.do_command -> struct_to_dict, and protobuf Struct
            # stores every number as a double, so [3, 224, 224] comes back as
            # [3.0, 224.0, 224.0]. PIL's resize() raises
            # "TypeError: integer argument expected, got float".
            image_sizes = {
                key: (int(specs["input_features"][key][1]),
                      int(specs["input_features"][key][2]))
                for key in specs["image_feature_keys"]
            }
            missing = set(image_sizes) - set(cfg.cameras)
            if missing:
                raise RuntimeError(
                    f"the policy expects camera feature(s) {sorted(missing)} that are not "
                    f"mapped in `cameras` (configured: {sorted(cfg.cameras)})"
                )
            builder = ObservationBuilder(
                cameras={k: self._resource(v) for k, v in cfg.cameras.items()},
                arm=self._resource(cfg.arm),
                gripper=gripper,
                state_joint_indices=cfg.state_joint_indices,
                state_units=cfg.state_units,
                image_sizes=image_sizes,
                image_encoding=cfg.image_encoding,
                jpeg_quality=cfg.jpeg_quality,
            )
            self._safety = self._build_safety(gripper)
            self._scheduler = SequentialScheduler(lambda rtc: self._infer(builder, rtc))

            self._state = "running"
            await self._loop(self._scheduler, builder, gripper)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._state = "error"
            self._last_error = str(exc)
            LOGGER.error("controller stopped: %s", exc)
            await self._safe_stop_arm()

    async def _infer(self, builder: ObservationBuilder, rtc):
        obs = await builder.build()
        period = 1.0 / self._cfg.fps
        if obs.duration_s > period:
            LOGGER.warning(
                "observation assembly took %.3fs, exceeding the %.3fs tick budget",
                obs.duration_s, period,
            )
        payload = {
            "command": "infer",
            "images": obs.images,
            "state": encode_vector(obs.state),
            "task": self._active_task_text,
        }
        if rtc:
            payload["rtc"] = rtc
        started = time.perf_counter()
        out = await self._resource(self._cfg.policy_service).do_command(payload)
        self._latencies.append(time.perf_counter() - started)
        del self._latencies[:-50]
        return decode_matrix(out["actions"]), decode_matrix(out["raw_actions"])

    async def _loop(self, scheduler, builder: ObservationBuilder, gripper) -> None:
        cfg = self._cfg
        arm = self._resource(cfg.arm)
        period = 1.0 / cfg.fps
        options = self._move_options()
        starved = 0
        first = True
        last_tick = time.perf_counter()

        while True:
            tick_started = time.perf_counter()

            action = await scheduler.next_action()
            if action is None:
                starved += 1
                if starved > cfg.starvation_grace_ticks:
                    raise RuntimeError("action queue starved; stopping")
                await asyncio.sleep(period)
                continue
            starved = 0

            degrees = to_degrees(np.asarray(action, dtype=np.float32), cfg.action_units)
            current_all = list((await arm.get_joint_positions()).values)
            current = np.asarray(
                [current_all[i] for i in cfg.state_joint_indices]
                + ([current_all[gripper.arm_joint_index]]
                   if gripper.arm_joint_index is not None else
                   ([await gripper.read()] if gripper.in_state else [])),
                dtype=np.float32,
            )

            if first:
                self._safety.check_start(degrees, current)
                first = False

            safe = self._safety.apply(degrees, current)

            joint_values = list(safe[: len(cfg.state_joint_indices)])
            if gripper.arm_joint_index is not None:
                joint_values.append(float(safe[-1]))

            await arm.move_through_joint_positions(
                [JointPositions(values=joint_values)], options
            )
            if gripper.in_state and gripper.arm_joint_index is None:
                await gripper.write(float(safe[-1]))

            now = time.perf_counter()
            elapsed = now - last_tick
            if elapsed > 0:
                self._measured_fps = 1.0 / elapsed
            last_tick = now

            remaining = period - (now - tick_started)
            if remaining > 0:
                await asyncio.sleep(remaining)

    async def close(self) -> None:
        await self._stop()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/controller/test_service.py -v`
Expected: 17 passed

If dependency-key handling differs from the fakes, read how a Python module resolves dependencies in `/Users/nick.hehr/src/viam-python-sdk/src/viam/resource/easy_resource.py` and adjust `_key`.

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/service.py tests/controller/test_service.py
git commit -m "feat: add controller service with sequential control loop"
```

---

## Task 18: Module entrypoint

**Files:**
- Create: `src/vla/main.py`

- [ ] **Step 1: Implement `src/vla/main.py`**

Two things make this shorter than it looks. `EasyResource.__init_subclass__` calls `cls.register()` at class-definition time, so **importing** `VLAPolicy` and `VLAController` already registers both models — calling `Registry.register_resource_creator` again raises `DuplicateResourceError: Cannot add resource with duplicate name`. And `Module.run_from_registry()` picks up everything in the registry, so there is nothing to wire by hand.

```python
"""Module entrypoint.

Importing the two service classes is what registers them: EasyResource
registers each subclass at class-definition time. run_from_registry then
serves everything in the registry.
"""

import asyncio

from viam.module.module import Module

from vla.controller.service import VLAController  # noqa: F401 - import registers the model
from vla.policy.service import VLAPolicy  # noqa: F401 - import registers the model

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
```

- [ ] **Step 2: Verify the module imports cleanly and registers exactly once**

Run:
```bash
uv run python -c "
import vla.main
from viam.resource.registry import Registry
from viam.services.generic import Generic
from vla.policy.service import VLAPolicy
from vla.controller.service import VLAController
Registry.lookup_resource_creator(Generic.API, VLAPolicy.MODEL)
Registry.lookup_resource_creator(Generic.API, VLAController.MODEL)
print('ok')
"
```
Expected: `ok`. A `DuplicateResourceError` here means explicit registration crept back in.

- [ ] **Step 3: Verify the entrypoint is actually inside the wheel**

```bash
uv build
uv run python -c "
import zipfile, glob
names = zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist()
assert 'vla/main.py' in names, names
print('ok')
"
```
Expected: `ok`. This is the check that catches the packaging trap — `packages = ["src/vla"]` ships only `vla/`, so an entrypoint outside that tree builds fine and then fails on the machine with `ModuleNotFoundError`.

- [ ] **Step 4: Commit**

```bash
git add src/vla/main.py
git commit -m "feat: add module entrypoint"
```

---

## Task 19: Full-loop integration test

Proves the whole pipeline with a real checkpoint and no hardware — the spec's phase-1 gate.

**Files:**
- Test: `tests/test_integration_full_loop.py`

- [ ] **Step 1: Write the integration test**

```python
"""Real checkpoint, fake robot: load -> preprocess -> infer -> postprocess -> actuate."""

import asyncio
import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig

pytestmark = pytest.mark.integration

from vla.controller.service import VLAController
from vla.policy.service import VLAPolicy
from tests.fakes import FakeArm, FakeCamera

CHECKPOINT = "lerobot/smolvla_base"


def _cfg(name, model, attrs):
    s = Struct()
    s.update(attrs)
    return ServiceConfig(name=name, api="rdk:service:generic", model=model, attributes=s)


@pytest.fixture(scope="module")
async def policy():
    svc = VLAPolicy("p")
    svc.reconfigure(_cfg("p", "viam-labs:vla:policy", {"model_hub_id": CHECKPOINT}), {})
    await svc.await_ready()
    return svc


async def test_policy_reports_ready_and_specs(policy):
    assert (await policy.do_command({"command": "status"}))["state"] == "ready"
    specs = await policy.do_command({"command": "specs"})
    assert specs["action_dim"] > 0
    assert specs["n_action_steps"] > 0


async def test_full_loop_drives_the_fake_arm(policy):
    specs = await policy.do_command({"command": "specs"})
    dim = int(specs["action_dim"])

    arm = FakeArm(positions=[0.0] * (dim + 1))
    camera = FakeCamera()
    controller = VLAController("c")
    controller.reconfigure(
        _cfg("c", "viam-labs:vla:controller", {
            "policy_service": "p",
            "arm": "a",
            # smolvla_base wants three cameras; one fake serves all three feeds.
            "cameras": {k: "cam" for k in specs["image_feature_keys"]},
            "state_joint_indices": list(range(dim)),
            "fps": 2.0,
            "task": "pick up the red block",
            "safety": {"max_start_delta_degs": 10000.0, "max_joint_delta_degs": 10000.0},
        }),
        {"p": policy, "a": arm, "cam": camera},
    )

    await controller.do_command({"command": "start"})
    await asyncio.sleep(15)
    status = await controller.do_command({"command": "status"})
    await controller.do_command({"command": "stop"})

    assert status["state"] == "running", status["last_error"]
    assert len(arm.moves) >= 1
    assert status["avg_latency_s"] > 0
    for positions, _ in arm.moves:
        assert np.all(np.isfinite(positions[0].values))
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/test_integration_full_loop.py -v -m integration`
Expected: 2 passed

Actions will be meaningless on a fake arm that resembles nothing in the training data. That is expected. This test proves the *plumbing*, not the behavior — finite numbers flowing end to end is the bar.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_full_loop.py
git commit -m "test: add full-loop integration test with a real checkpoint"
```

---

## Task 20: README

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write `README.md`**

Cover, in this order:

1. What the module does and which policies it supports (SmolVLA, Evo-1, and any LeRobot-registered policy).
2. **Requirements** — Python 3.12+, and the install-size warning (torch plus CUDA libraries; measure before committing to a Jetson deployment).
3. `#policy` — the full config table, all four DoCommand commands, and worked JSON for each of the three model-delivery paths (local, `${packages.ml_model.name}`, hub).
4. `#controller` — the full config table, the five gripper variants with a worked example each, and `start`/`stop`/`status`.
5. **Units** — Viam reports degrees; state a checkpoint may not. This is the section that saves the most debugging time, so give it a worked SO-100 example.
6. **Safety** — the layer order, and that persistent clamping in `status.clamp_counts` means wrong units or joint order.
7. **Limitations** — `mode: rtc` is not implemented yet; `normalized` state units are unsupported; relative-action checkpoints are refused under RTC.
8. **Development** — `mise run test` (fast, no torch), `mise run test-all`, `uv sync --extra lerobot` for the integration and differential suites.

- [ ] **Step 2: Add the `readme` key now that the file exists**

In `pyproject.toml`, under `[project]`, restore:

```toml
readme = "README.md"
```

Task 1 omitted it deliberately, because hatchling aborts when it points at a missing file.

- [ ] **Step 3: Verify every config example parses**

For each JSON block in the README, confirm `PolicyConfig.parse` / `ControllerConfig.parse` accepts it. Doc examples that do not parse are worse than no examples.

- [ ] **Step 4: Verify the build still works with the readme key**

Run: `uv build`
Expected: succeeds.

- [ ] **Step 5: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs: add module README"
```

---

## Task 21: Final verification

- [ ] **Step 1: Run the fast suite**

Run: `mise run test`
Expected: all pass, no torch imported.

- [ ] **Step 2: Run everything**

Run: `uv sync --extra lerobot && mise run test-all`
Expected: all pass including integration and differential.

- [ ] **Step 3: Confirm the fast suite really runs without torch**

Run: `uv sync && uv run pytest -m 'not integration and not differential' -v`
Expected: all pass in a venv with no torch. This is the payoff for the `PolicyBackend` seam — if it fails, an import leaked out of `lerobot_backend.py`.

- [ ] **Step 4: Build the module tarball**

Run: `mise run package && tar -tzf module.tar.gz | head`
Expected: `meta.json`, the shell scripts, and `dist/`.

- [ ] **Step 5: Measure install size**

This is the spec's main open question and the one thing no architecture choice resolves.

```bash
du -sh .venv
```

Record the number in the README's requirements section. On Linux with CUDA it will be far larger than on macOS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: record measured install size"
```

---

## Deferred to follow-up plans

- **`RTCScheduler`** — background inference with prefix feedback. `ActionQueue` already supports RTC mode and is differentially tested; what remains is the async scheduling, the two-delay bookkeeping, and wiring `prev_chunk_left_over` from `get_left_over()`. The spec says to enable this only after measuring real latency on the target device.
- **Relative-action prefix re-anchoring** — a port of `reanchor_relative_rtc_prefix`, which uses `get_processed_left_over()`. Until then such checkpoints are refused under RTC.
- **`normalized` state units** — needs a decision on where per-joint min/max come from.
- **`dtype` selection** — the config field is parsed and validated but not applied. Casting weights
  with `policy.to(dtype=...)` breaks inference, because the deserialized `DeviceProcessorStep` has
  `float_dtype=None` and keeps emitting float32. The correct fix is `torch.autocast` gated on
  `config.use_amp`, as upstream does. Until then the checkpoint's own dtype is used and a non-`auto`
  setting logs a warning.
- **Hardware smoke test** — a manual checklist with conservative limits, per spec phase 3.
- **Pinned dependency graph on the robot** (pre-deploy, before the first real machine).
  `setup.sh` runs `uv pip install "<wheel>[lerobot]"`, which does not consult `uv.lock`, so
  every robot re-resolves torch and friends at first run. Two machines provisioned a week
  apart can get different builds, and neither matches CI. Fix by shipping `uv.lock` and
  using `uv sync --frozen --extra lerobot`, or by exporting a pinned requirements file at
  package time. Deferred from Task 1 because it changes the deploy contract and the module
  has not been deployed once; it pairs naturally with measuring install size (Task 21).
