#!/bin/bash
# BLOCKED: Gate B0.5 BIOEMU_EXTENDED_CHAIN_UNRESOLVED — do not run full cohort.
# Orchestrate BioEmu constant-context pipeline after physical oversampling.
set -euo pipefail
CTX=/workspace_developability_acquisition/organizer_extension/feature_prospecting/bioemu_constant_context
PY=/workspace_developability_acquisition/organizer_extension/feature_prospecting/foundation_stability/envs/bioemu/bin/python
EV=/workspace_developability_acquisition/.venv_stage4/bin/python
export CUDA_VISIBLE_DEVICES=0
cd "$CTX"
mkdir -p results

log() { echo "[$(date -Is)] $*" | tee -a results/orchestrator.log; }

log "=== LIGHT convergence oversample (64 physical) ==="
$PY -u scripts/sample_context.py --convergence --light-only >> results/pilot_oversample_light.log 2>&1
log "=== HEAVY convergence oversample (64 physical) ==="
$PY -u scripts/sample_context.py --convergence --heavy-only >> results/pilot_oversample_heavy.log 2>&1

log "=== Convergence decision ==="
$PY scripts/extract_context_features.py --convergence >> results/convergence_run.log 2>&1
log "Frozen N light=$(cat BIOEMU_CONTEXT_FROZEN_N_LIGHT.txt) heavy=$(cat BIOEMU_CONTEXT_FROZEN_N_HEAVY.txt)"

# Validity quick stats
$PY - <<'PY' >> results/validity_stats.log 2>&1
import mdtraj as md, pandas as pd
from pathlib import Path
CTX=Path('.')
pil=pd.read_csv('pilots/BIOEMU_CONTEXT_CONVERGENCE_ANTIBODIES.csv').id
rows=[]
for ab in pil:
  for arm,suf in [('LIGHT','VLCL'),('HEAVY','VHCH1')]:
    d=CTX/'cache/bioemu_samples'/f'{ab}_{suf}'
    top,xtc=d/'topology.pdb',d/'samples.xtc'
    n=md.load(str(xtc),top=str(top)).n_frames if top.exists() and xtc.exists() else 0
    rows.append(dict(id=ab,arm=arm,physical=n))
pd.DataFrame(rows).to_csv('results/PILOT_PHYSICAL_COUNTS.csv',index=False)
print(pd.DataFrame(rows).groupby('arm').physical.describe())
PY

log "=== FULL LIGHT cohort ==="
$PY -u scripts/sample_context.py --light-only --n "$(cat BIOEMU_CONTEXT_FROZEN_N_LIGHT.txt)" >> results/full_light_sample.log 2>&1

# Heavy full only if pilot median physical >= frozen N
HEAVY_OK=$($EV - <<'PY'
import pandas as pd, sys
df=pd.read_csv('results/PILOT_PHYSICAL_COUNTS.csv')
n=int(open('BIOEMU_CONTEXT_FROZEN_N_HEAVY.txt').read())
med=float(df[df.arm=='HEAVY'].physical.median())
frac=float((df[df.arm=='HEAVY'].physical>=n).mean())
print(med, frac, file=sys.stderr)
print('1' if (med>=n and frac>=0.8) else '0')
PY
)
if [ "$HEAVY_OK" = "1" ]; then
  log "=== FULL HEAVY cohort (viable) ==="
  $PY -u scripts/sample_context.py --heavy-only --n "$(cat BIOEMU_CONTEXT_FROZEN_N_HEAVY.txt)" >> results/full_heavy_sample.log 2>&1
else
  log "=== SKIP full HEAVY cohort (pilot not viable enough); keep diagnostic pilot ==="
fi

log "=== Features ==="
$PY scripts/extract_context_features.py >> results/features_full.log 2>&1
log "=== Static-dynamic ==="
$EV scripts/static_dynamic_compare.py >> results/static_dynamic.log 2>&1 || true
log "=== Eval ==="
$EV scripts/eval_context.py >> results/eval.log 2>&1
log "ORCHESTRATOR_CORE_DONE"
