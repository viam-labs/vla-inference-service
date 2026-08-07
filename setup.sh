#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_NAME="${VIAM_MODULE_DATA}/venv"
export PATH=$PATH:$HOME/.local/bin

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

uv venv --python 3.12 "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Resolve the wheel path first: "./dist/"*.whl[lerobot] does not work, because
# bash reads [lerobot] as a glob character class, the pattern matches nothing,
# and the literal string is handed to uv.
shopt -s nullglob
WHEELS=(./dist/*.whl)
if [ ${#WHEELS[@]} -ne 1 ]; then
  echo "setup.sh: expected exactly 1 wheel in ./dist, found ${#WHEELS[@]}" >&2
  exit 1
fi
WHEEL="${WHEELS[0]}"
uv pip install "${WHEEL}[lerobot]" -q
