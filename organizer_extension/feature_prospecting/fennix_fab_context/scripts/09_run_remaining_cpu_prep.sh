#!/bin/bash
set -euo pipefail
CTX=/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context
PY=/workspace_developability_acquisition/.venv_b1/bin/python
BATCH_DIR=$CTX/cache/cpu_prep_batches
LOG=$CTX/cache/logs/cpu_remaining_prep.log
mkdir -p "$CTX/cache/logs"
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
# refresh remaining list
$PY - <<'PY'
import json
from pathlib import Path
import pandas as pd
CTX=Path('/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context')
FAB=Path('/workspace_developability_acquisition/organizer_extension/feature_prospecting/fab_reconstruction')
seq=pd.read_csv(FAB/'sequences/SHEHATA_RECONSTRUCTED_FAB.csv')
prep=CTX/'cache/prepared_fab'
done=set(p.name.replace('_prepared.complete','') for p in prep.glob('*_prepared.complete'))
qc=pd.read_csv(CTX/'FAB_PREP_QC.csv')
for _,r in qc.iterrows():
  if bool(r.ok) and (prep/f'{r.id}_prepared.pdb').exists():
    done.add(str(r.id))
remain=[i for i in seq.id.astype(str).tolist() if i not in done]
outdir=CTX/'cache/cpu_prep_batches'
outdir.mkdir(exist_ok=True)
(outdir/'remaining_ids.json').write_text(json.dumps(remain, indent=2))
for p in outdir.glob('batch_*.json'):
  p.unlink()
for i in range(0,len(remain),15):
  b=remain[i:i+15]
  (outdir/f'batch_{i//15:03d}.json').write_text(json.dumps(b))
print(f'remain={len(remain)} batches={(len(remain)+14)//15}')
PY
shopt -s nullglob
for bf in $(ls "$BATCH_DIR"/batch_*.json | sort); do
  tag=$(basename "$bf" .json)
  mapfile -t IDS < <($PY -c "import json; print('\n'.join(json.load(open('$bf'))))")
  echo "BATCH $tag n=${#IDS[@]} $(date -u +%H:%M:%S)" | tee -a "$LOG"
  # process exit between batches: each batch is a separate python invocation
  "$PY" "$CTX/scripts/06_cpu_batch_prep.py" --ids "${IDS[@]}" --batch-tag "$tag" --attempt 1 \
    >> "$LOG" 2>&1
  echo "BATCH_DONE $tag exit=$? $(date -u +%H:%M:%S)" | tee -a "$LOG"
done
echo "ALL_PREP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
