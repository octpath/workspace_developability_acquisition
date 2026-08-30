#!/usr/bin/env bash
# Resume from pooling after P0 nested already done
set -euo pipefail
ROOT=/workspace_developability_acquisition
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"

echo "[B5 resume] learned pooling"
.venv_b1/bin/python -u gate_b5_ceiling/scripts/02_learned_pooling.py 2>&1 | tee -a gate_b5_ceiling/logs/02_pooling.log

echo "[B5 resume] light finetune"
.venv_b1/bin/python -u gate_b5_ceiling/scripts/03_light_finetune.py 2>&1 | tee gate_b5_ceiling/logs/03_finetune.log

echo "[B5 resume] finalists + oneshot + FINAL"
.venv_b1/bin/python -u gate_b5_ceiling/scripts/04_freeze_oneshot_final.py 2>&1 | tee gate_b5_ceiling/logs/04_final.log

echo "[B5 resume] COMPLETE"
