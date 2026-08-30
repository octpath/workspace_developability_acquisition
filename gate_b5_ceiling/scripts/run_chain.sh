#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace_developability_acquisition
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"

wait_log() {
  local marker="$1" log="$2"
  while true; do
    if grep -q "$marker" "$log" 2>/dev/null; then break; fi
    if ! pgrep -f "$3" >/dev/null; then
      if grep -q "$marker" "$log" 2>/dev/null; then break; fi
      echo "Process gone before marker: $3"; tail -40 "$log"; exit 1
    fi
    sleep 60
  done
}

echo "[B5 chain] wait nested"
wait_log "DONE nested" gate_b5_ceiling/logs/01_nested.log "01_nested_ensemble_pearson.py"

echo "[B5 chain] learned pooling"
.venv_b1/bin/python -u gate_b5_ceiling/scripts/02_learned_pooling.py 2>&1 | tee gate_b5_ceiling/logs/02_pooling.log

echo "[B5 chain] light finetune"
.venv_b1/bin/python -u gate_b5_ceiling/scripts/03_light_finetune.py 2>&1 | tee gate_b5_ceiling/logs/03_finetune.log

echo "[B5 chain] finalists + oneshot + FINAL"
.venv_b1/bin/python -u gate_b5_ceiling/scripts/04_freeze_oneshot_final.py 2>&1 | tee gate_b5_ceiling/logs/04_final.log

echo "[B5 chain] COMPLETE"
