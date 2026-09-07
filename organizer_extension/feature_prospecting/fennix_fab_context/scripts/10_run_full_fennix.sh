#!/bin/bash
# Full-cohort FeNNix B/C/M after prep complete. Serial by default; optional shards.
set -euo pipefail
CTX=/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context
FENNOL=/workspace_developability_acquisition/organizer_extension/feature_prospecting/foundation_stability/envs/fennol/bin/python
LOG=$CTX/cache/logs/fennix_full.log
mkdir -p "$CTX/cache/logs" "$CTX/cache/features"
echo "START_FULL $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
# Run all IDs that have prepared.complete (resume-safe inside 02)
# Prefer single process for host stability after prior hangs.
"$FENNOL" "$CTX/scripts/02_fennix_fab_curvature.py" >> "$LOG" 2>&1
echo "DONE_FULL $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
