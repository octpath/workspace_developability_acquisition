#!/usr/bin/env bash
# Chain Gate B4 Stage1 → Stage2 after screening completes.
set -euo pipefail
ROOT=/workspace_developability_acquisition
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"

echo "[chain] waiting for screening PID ${SCREEN_PID:-none}"
if [[ -n "${SCREEN_PID:-}" ]]; then
  while kill -0 "$SCREEN_PID" 2>/dev/null; do sleep 60; done
  wait "$SCREEN_PID" || true
fi

# Also wait for marker file
while [[ ! -f gate_b4_absolute/logs/01_screen_done.json ]]; do
  # if log shows DONE
  if grep -q "DONE screening" gate_b4_absolute/logs/01_screen.log 2>/dev/null; then
    break
  fi
  if [[ ! -f /proc/${SCREEN_PID:-0} ]] && [[ -n "${SCREEN_PID:-}" ]]; then
    # process gone — check success
    if [[ -f gate_b4_absolute/metrics/all_screening_results.csv ]]; then
      break
    fi
    echo "[chain] screening failed"; exit 1
  fi
  sleep 60
done

echo "[chain] starting nested/finalists/oneshot"
.venv_b1/bin/python -u gate_b4_absolute/scripts/02_nested_finalists_oneshot.py \
  2>&1 | tee gate_b4_absolute/logs/02_nested_oneshot.log

echo "[chain] complete"
