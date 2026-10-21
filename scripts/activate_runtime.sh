#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
# shellcheck disable=SC1091
source config/runtime_limits.env
set +a

echo "AeroEyes runtime limits loaded."
echo "Device: ${AEROEYES_DEVICE}; precision: ${AEROEYES_PRECISION}; target FPS: ${AEROEYES_TARGET_FPS}"
