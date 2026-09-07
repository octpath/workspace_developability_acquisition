#!/bin/bash
# Resume FeNNix via recycle-10 orchestrator if idle and work remains.
set -euo pipefail
CTX=/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context
PY=/workspace_developability_acquisition/.venv_b1/bin/python
$PY "$CTX/scripts/14_fennix_progress.py" >/dev/null
alive=$($PY -c "import json; print(json.load(open('$CTX/cache/FENNIX_FULL_PROGRESS.json'))['worker_alive'])")
remain=$($PY -c "import json; print(json.load(open('$CTX/cache/FENNIX_FULL_PROGRESS.json'))['remain_pairs'])")
# also treat recycle orchestrator as alive
orch=$($PY - <<'PY'
import os
ok=False
for pid in os.listdir('/proc'):
  if not pid.isdigit(): continue
  try: raw=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\0',b' ').decode()
  except Exception: continue
  toks=raw.split()
  if toks and (toks[0].endswith('python') or '/bin/python' in toks[0]) and '16_run_fennix_recycle_10.py' in raw:
    ok=True; break
print(ok)
PY
)
if [[ "$alive" == "True" || "$orch" == "True" ]]; then
  echo "already running remain=$remain orch=$orch"
  exit 0
fi
if [[ "$remain" -le 0 ]]; then
  echo "complete remain=0"
  exit 0
fi
echo "resuming FeNNix via recycle-10 remain=$remain"
nohup "$PY" "$CTX/scripts/16_run_fennix_recycle_10.py" \
  >> "$CTX/cache/logs/fennix_recycle_nohup.out" 2>&1 &
echo $! > "$CTX/cache/logs/fennix_recycle.pid"
echo "started recycle pid=$(cat "$CTX/cache/logs/fennix_recycle.pid")"
