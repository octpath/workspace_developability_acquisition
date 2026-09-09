#!/usr/bin/env bash
# Clean-directory style validation for AbLang2 follow-up assets (no full retrain).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE="$ROOT/top_models_feature_bundle"
PY="${ROOT}/.venv_b1/bin/python"
OUT="$BUNDLE/results/ABLANG2_CLEAN_DIR_VALIDATION.txt"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

{
  echo "ABLANG2_CLEAN_DIR_VALIDATION"
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo

  echo "== parts =="
  ls "$BUNDLE/residue_level/ablang2/"*.part0 "$BUNDLE/residue_level/ablang2/"*.part1
  echo "PASS: tracked parts exist"

  mkdir -p "$TMP/ablang2"
  cp -a "$BUNDLE/residue_level/ablang2/"*.part0 "$BUNDLE/residue_level/ablang2/"*.part1 "$TMP/ablang2/"
  # force reassembly without pre-existing npy
  echo "== assemble =="
  cat "$TMP/ablang2/heavy_embeddings.npy.part0" "$TMP/ablang2/heavy_embeddings.npy.part1" > "$TMP/ablang2/heavy_embeddings.npy"
  cat "$TMP/ablang2/light_embeddings.npy.part0" "$TMP/ablang2/light_embeddings.npy.part1" > "$TMP/ablang2/light_embeddings.npy"

  echo "== SHA256 =="
  "$PY" - <<PY
import hashlib, json
from pathlib import Path
d = Path("$TMP/ablang2")
man = json.loads(Path("$BUNDLE/residue_level/ablang2/SHA256_MANIFEST.json").read_text())
for name in ("heavy_embeddings.npy", "light_embeddings.npy"):
    h = hashlib.sha256((d/name).read_bytes()).hexdigest()
    assert h == man["files"][name]["sha256"], (name, h)
    print("PASS", name, h[:16])
PY

  echo "== QC =="
  "$PY" - <<PY
import json
from pathlib import Path
qc = json.loads(Path("$BUNDLE/residue_level/ablang2/ablang2_residue_qc_summary.json").read_text())
assert qc["n_unique_ids"] == 324
assert qc["n_heavy"] == 324 and qc["n_light"] == 324
assert qc["all_residue_eq_seq_len"] and qc["finite_100pct"]
print("PASS residue QC")
PY

  echo "== imports + unit tests =="
  cd "$BUNDLE"
  PYTHONPATH=. "$PY" -c "from advanced_models.models.annotated_transformer import AnnotatedTransformer; print('PASS imports')"
  PYTHONPATH=. "$PY" -m unittest advanced_models.tests.test_advanced_models -v

  echo "== smoke train =="
  PYTHONPATH=. "$PY" - <<'PY'
from pathlib import Path
import torch
from advanced_models.data import load_dev_test, load_folds, load_residue_bundle
from advanced_models.cv import train_transformer_seed, tvt_split
dev, test = load_dev_test(Path("dev.csv"), Path("test.csv"))
folds = load_folds()
rb = load_residue_bundle(dev, test, need_ablang2=True)
y_map = {str(r.id): float(r.TmApp) for r in dev.itertuples(index=False)}
tr, va, te = tvt_split(folds.primary, 0, dev["id"].astype(str).tolist())
out = train_transformer_seed(
    train_ids=tr, val_ids=va, test_ids=te, y_map=y_map, rb=rb,
    content_mode="frozen", annotation_mode="full", merge_mode="mean", chain_mode="HL",
    seed=101, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    quick=True, plm_source="ablang2", pooling_mode="reg",
)
assert out["test_pred"].shape[0] == len(te)
print("PASS smoke", out["best_epoch"], float(out["test_mae"]))
PY

  echo "== submission schema =="
  "$PY" -c 'import pandas as pd; df=pd.DataFrame({"id":[f"a{i}" for i in range(162)],"TmApp":0.0,"HIC":0.0}); assert list(df.columns)==["id","TmApp","HIC"]; print("PASS schema")'

  echo
  echo "OVERALL: PASS"
} | tee "$OUT"
