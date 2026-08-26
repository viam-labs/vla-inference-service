#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_NAME="${VIAM_MODULE_DATA:?must be set by viam-server}/venv"
export PATH="$HOME/.local/bin:$PATH"

if [ ! "$(command -v git)" ]; then
  echo "git is required to install the lerobot extra (a git+https:// direct reference)."
  exit 1
fi

if [ ! "$(command -v uv)" ]; then
  if [ ! "$(command -v curl)" ]; then
    echo "curl is required to install uv."
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# --allow-existing, not a bare create: viam-server re-runs first_run for every
# new module version, so a reload lands here with the previous reload's venv
# already in place and a plain `uv venv` fails outright ("A virtual environment
# already exists"). Reusing it also makes a reload cheap -- uv updates only what
# changed instead of re-downloading the multi-GB torch stack every time.
uv venv --python 3.12 --allow-existing "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Resolve the wheel path first: "./dist/"*.whl[lerobot] does not work, because
# bash reads [lerobot] as a glob character class, the pattern matches nothing,
# and the literal string is handed to uv.
shopt -s nullglob
WHEELS=(./dist/*.whl)
shopt -u nullglob
if [ ${#WHEELS[@]} -ne 1 ]; then
  echo "setup.sh: expected exactly 1 wheel in ./dist, found ${#WHEELS[@]}" >&2
  exit 1
fi
WHEEL="${WHEELS[0]}"
# --reinstall-package on our own distribution: the version in pyproject.toml is
# static, so on a reload into a reused venv uv sees 0.1.0 already satisfied and
# skips it -- silently running the PREVIOUS build's code while the log says the
# install succeeded. The heavy dependencies are untouched by this and stay cached.
uv pip install --reinstall-package viam-vla-inference-service "${WHEEL}[lerobot]" -q

# Jetson needs torch built for sm_87. The generic aarch64 cu128 wheels the
# resolver picks (PyPI / download.pytorch.org) exclude Orin's compute
# capability outright -- torch itself says so on load ("8.0 which supports
# hardware CC >=8.0,<9.0 except {8.7}") and the first kernel launch dies with
# "no kernel image is available for execution on the device".
#
# NVIDIA's jetson-ai-lab index is the only source of sm_87 builds, and its
# jp6/cu129 line is the only one carrying cp312 wheels -- every other channel
# is cp310, which lerobot (requires-python >=3.12) cannot be installed on at
# all. So the version pair is forced: torch 2.8.0 + torchvision 0.23.0, both
# inside lerobot's own ranges (torch >=2.7,<2.12, torchvision >=0.22,<0.27).
#
# Swapped in after the normal resolve rather than constraining it: --no-deps
# leaves the rest of the graph exactly as resolved and only exchanges these
# two wheels for a different build of a version already known to satisfy
# everything. It re-downloads (~1 GB discarded), which is worth the
# simplicity here -- a constraint file would have to duplicate lerobot's
# pins to save it.
#
# ponytail: jp6/cu129 defaults, overridable -- a different JetPack ships a
# different CUDA and this is exactly the knob that needs turning by hand.
if [ -f /etc/nv_tegra_release ] || grep -qi tegra /proc/device-tree/compatible 2>/dev/null; then
  JETSON_TORCH_INDEX="${JETSON_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu129}"
  JETSON_TORCH_VERSION="${JETSON_TORCH_VERSION:-2.8.0}"
  JETSON_TORCHVISION_VERSION="${JETSON_TORCHVISION_VERSION:-0.23.0}"
  # The cp312 wheels on that index are built against Ubuntu 24.04 and need
  # GLIBC_2.38; JetPack 6 is Ubuntu 22.04 (2.35), where they install fine and
  # then fail to import at all. Skip rather than replace a wheel that at least
  # loads -- and say which upgrade unblocks it, because the cp310 wheels that
  # DO match 22.04 are unusable here (lerobot requires Python >=3.12).
  HAVE_GLIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
  if [ -n "$HAVE_GLIBC" ] && [ "$(printf '%s\n2.38\n' "$HAVE_GLIBC" | sort -V | head -1)" != "2.38" ]; then
    echo "setup.sh: Tegra detected, but glibc is ${HAVE_GLIBC} and the cp312 sm_87" >&2
    echo "  wheels need >= 2.38. Keeping the generic aarch64 torch, which cannot run" >&2
    echo "  kernels on this GPU -- the policy will fall back to CPU. JetPack 7" >&2
    echo "  (Ubuntu 24.04) is what unblocks GPU inference here." >&2
  else
    echo "setup.sh: Tegra detected -- replacing torch with an sm_87 build from ${JETSON_TORCH_INDEX}"
    uv pip install --reinstall --no-deps -q \
      --index-url "$JETSON_TORCH_INDEX" \
      "torch==${JETSON_TORCH_VERSION}" "torchvision==${JETSON_TORCHVISION_VERSION}"
  fi
fi

# The venv survives reloads (see --allow-existing above), so a torch swapped in
# by an EARLIER reload outlives the conditions that justified it -- a Jetson
# wheel installed before the glibc guard existed still satisfies lerobot's
# version range, so nothing above would ever replace it, and the module dies on
# `import torch` every start. Probe the import and re-resolve from the default
# index if it is broken. A good wheel (including a correctly-swapped sm_87 one)
# imports fine and is left alone, so this costs nothing on a healthy venv.
if ! "$VENV_NAME/bin/python" -c "import torch" 2>/dev/null; then
  echo "setup.sh: torch in the venv cannot be imported -- re-resolving from the default index" >&2
  uv pip install --reinstall-package torch --reinstall-package torchvision -q "${WHEEL}[lerobot]"
  "$VENV_NAME/bin/python" -c "import torch; print('setup.sh: torch', torch.__version__, 'imports OK')"
fi
