#!/bin/bash
set -euo pipefail
CTX=/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context
FENNOL=/workspace_developability_acquisition/organizer_extension/feature_prospecting/foundation_stability/envs/fennol/bin/python
PREP=$CTX/cache/prepared_fab
SNAP=$CTX/cache/cpu_cuda_audit/cuda_prepared_snapshot
CPU_SNAP=$CTX/cache/cpu_cuda_audit/cpu_prepared_snapshot
AUDIT=$CTX/cache/cpu_cuda_audit
IDS=(ADI-38501 ADI-38502 ADI-45368 ADI-45370)

run_platform() {
  local TAG=$1
  local SRC=$2
  echo "=== PLATFORM $TAG from $SRC ==="
  for ab in "${IDS[@]}"; do
    cp -f "$SRC/${ab}_prepared.pdb" "$PREP/${ab}_prepared.pdb"
  done
  # shellcheck disable=SC2068
  "$FENNOL" "$CTX/scripts/02_fennix_fab_curvature.py" \
    --ids "${IDS[@]}" \
    --conditions C \
    --audit-tag "curv_$TAG" \
    --fresh
  echo "DONE_$TAG"
}

# Ensure CPU snapshot
mkdir -p "$CPU_SNAP"
for ab in "${IDS[@]}"; do
  if [[ -f $PREP/${ab}_prepared.pdb ]]; then
    cp -f "$PREP/${ab}_prepared.pdb" "$CPU_SNAP/"
  fi
done

run_platform cuda "$SNAP"
# restore CPU prepared before cpu run (cuda overwrote prepared_fab)
for ab in "${IDS[@]}"; do
  cp -f "$CPU_SNAP/${ab}_prepared.pdb" "$PREP/${ab}_prepared.pdb"
done
run_platform cpu "$CPU_SNAP"
echo ALL_CURV_DONE
