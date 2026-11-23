#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_NAME="${VIAM_MODULE_DATA:?must be set by viam-server}/venv"
export PATH="$HOME/.local/bin:$PATH"

if [ ! "$(command -v uv)" ]; then
  if [ ! "$(command -v curl)" ]; then
    echo "curl is required to install uv."
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# --allow-existing, not a bare create: viam-server re-runs first_run for every
# new module version, so a reload lands here with the previous venv in place.
# Reusing it also keeps a policy host's lazily installed torch stack.
uv venv --python 3.12 --allow-existing "$VENV_NAME"
source "$VENV_NAME/bin/activate"

shopt -s nullglob
WHEELS=(./dist/*.whl)
shopt -u nullglob
if [ ${#WHEELS[@]} -ne 1 ]; then
  echo "setup.sh: expected exactly 1 wheel in ./dist, found ${#WHEELS[@]}" >&2
  exit 1
fi

# Only the core (viam-sdk, numpy, pillow, scipy, huggingface-hub, uv), pinned
# from uv.lock. Torch, LeRobot and the policy type's extras are installed by the
# policy service on its first load (vla/policy/installer.py), so a machine that
# runs only the controller never downloads them.
uv pip install -q -r requirements/core.txt

# --reinstall-package on our own distribution: the version in pyproject.toml is
# static, so on a reload into a reused venv uv sees 0.1.0 already satisfied and
# skips it -- silently running the PREVIOUS build's code. --no-deps: core.txt
# already pinned everything the wheel needs.
uv pip install -q --no-deps --reinstall-package viam-vla-inference-service "${WHEELS[0]}"
