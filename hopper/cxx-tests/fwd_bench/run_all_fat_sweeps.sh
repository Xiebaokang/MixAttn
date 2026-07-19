#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_fat_sweep.py"

run_one() {
  local method="$1"
  local dtype="$2"
  local causal="$3"
  shift 3

  echo "==> method=${method} dtype=${dtype} causal=${causal}"
  "${PYTHON_BIN}" "${SWEEP_SCRIPT}" \
    --method "${method}" \
    --dtype "${dtype}" \
    --causal "${causal}" \
    "$@"
}

for causal in 0 1; do
  run_one fa3 fp8 "${causal}" "$@"
  run_one fa3 fp16 "${causal}" "$@"
  run_one fat fp8 "${causal}" "$@"
  run_one fat fp16 "${causal}" "$@"
done
