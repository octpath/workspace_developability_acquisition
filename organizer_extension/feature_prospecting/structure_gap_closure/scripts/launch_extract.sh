#!/bin/bash
# Launch gap-closure extractors on cores 16-23 only (leave 0-15 for FeNNix).
set -euo pipefail
CTX=/workspace_developability_acquisition/organizer_extension/feature_prospecting/structure_gap_closure
ST4=/workspace_developability_acquisition/.venv_stage4/bin/python
B1=/workspace_developability_acquisition/.venv_b1/bin/python
LOG=$CTX/cache/logs
mkdir -p "$LOG"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
export GAP_WORKERS=6

# Surface needs freesasa (stage4)
nohup taskset -c 16-23 env GAP_WORKERS=6 OMP_NUM_THREADS=1 \
  "$ST4" "$CTX/scripts/extract_surface_hic.py" \
  > "$LOG/extract_surface.log" 2>&1 &
echo $! > "$LOG/extract_surface.pid"
echo "surface_pid=$(cat "$LOG/extract_surface.pid")"

# Packing/cavity on b1 (Bio.PDB); fewer workers (RAM)
nohup taskset -c 16-23 env GAP_WORKERS=3 OMP_NUM_THREADS=1 \
  "$B1" "$CTX/scripts/extract_fab_packing_iface.py" \
  > "$LOG/extract_fab.log" 2>&1 &
echo $! > "$LOG/extract_fab.pid"
echo "fab_pid=$(cat "$LOG/extract_fab.pid")"
