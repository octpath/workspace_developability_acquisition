#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace_developability_acquisition
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
PY="$ROOT/.venv_b1/bin/python"
LOG=$ROOT/gate_b2/logs

echo "=== modeling $(date) ===" | tee -a $LOG/04_modeling.log
$PY -u gate_b2/scripts/04_modeling.py 2>&1 | tee -a $LOG/04_modeling.log
echo "=== residuals $(date) ===" | tee -a $LOG/04b_residuals.log
$PY -u gate_b2/scripts/04b_residuals_ensembles.py 2>&1 | tee -a $LOG/04b_residuals.log
echo "=== diagnostics $(date) ===" | tee -a $LOG/05_diagnostics.log
$PY -u gate_b2/scripts/05_diagnostics_final.py 2>&1 | tee -a $LOG/05_diagnostics.log
echo "=== final $(date) ===" | tee -a $LOG/07_final.log
$PY -u gate_b2/scripts/07_write_gate_b2_final.py 2>&1 | tee -a $LOG/07_final.log
echo PIPELINE_TAIL_DONE
