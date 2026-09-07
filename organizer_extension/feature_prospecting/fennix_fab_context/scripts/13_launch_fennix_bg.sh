#!/bin/bash
# Background launcher for 16-cap FeNNix (avoid interactive pkill self-match issues).
set -euo pipefail
CTX=/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context
PY=/workspace_developability_acquisition/.venv_b1/bin/python
LOGDIR=$CTX/cache/logs
mkdir -p "$LOGDIR"
cd "$CTX"
nohup "$PY" "$CTX/scripts/12_run_fennix_16cap.py" \
  > "$LOGDIR/fennix_16cap_nohup.out" 2>&1 &
echo $! > "$LOGDIR/fennix_16cap.pid"
echo "started pid=$(cat "$LOGDIR/fennix_16cap.pid")"
