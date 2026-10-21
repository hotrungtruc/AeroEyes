#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${AEROEYES_PYTHON:-python3.8}"
VENV_PATH="${AEROEYES_VENV:-.venv}"

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
}

"$PYTHON" -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade "pip<24.1" setuptools wheel
"$VENV_PATH/bin/python" -m pip install -r requirements-jetson.txt

echo
echo "Environment created at $VENV_PATH."
echo "Activate it with: source $VENV_PATH/bin/activate"
echo "Install the JetPack-compatible PyTorch 1.12.1 wheel separately on Jetson."
