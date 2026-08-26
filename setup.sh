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
