#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_NAME="${VIAM_MODULE_DATA:?must be set by viam-server}/venv"
PYTHON="$VENV_NAME/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "run.sh: no venv at $VENV_NAME — first_run (setup.sh) did not complete" >&2
  exit 1
fi

echo "Starting module..."
exec "$PYTHON" -m vla.main "$@"
