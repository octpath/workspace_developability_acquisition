#!/usr/bin/env python3
"""Export RAW_PREPROCESS feature parquets keyed by experiment_code (EXPxxx)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    ROOT,
    add_split_column,
    concat_recipe_features,
    feature_column_names,
    feature_content_sha256,
    feature_recipe_hash,
    file_sha256,
    load_dev_test_folds,
    recipe_blocks_from_recipes_csv,
)
from experiment_codes import id_to_code  # noqa: E402

# descriptive experiment_id -> source_recipe_id for FULL set
FULL_JOBS = {
    "LIN_TM_ABLINGUA_CDR3_RIDGE": "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "LIN_TM_ABLINGUA_GLOBAL_RIDGE": "TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
    "LIN_TM_BIOEMU_MPNN_RIDGE": "TM_BASE_BIOEMU_MPNN__RIDGE",
    "LIN_HIC_HYDRO_TITRATION_LASSO": "HIC_HYDRO_TITRATION__LASSO",
    "LIN_HIC_CONTINUOUS_SURFACE_LASSO": "HIC_ARO_CONTINUOUS_SURFACE__LASSO",
    "LIN_HIC_ESM2_SEQ_AROMATIC_LASSO": "HIC_ESM2_SEQ_AROMATIC__LASSO",
    "XGB_TM_BIOEMU_MPNN": "TM_BASE_BIOEMU_MPNN__RIDGE",
    "XGB_TM_ABLINGUA_GLOBAL": "TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
    "XGB_TM_ABLINGUA_CDR3": "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "XGB_HIC_CONTINUOUS_SURFACE": "HIC_ARO_CONTINUOUS_SURFACE__LASSO",
    "XGB_HIC_HYDRO_TITRATION": "HIC_HYDRO_TITRATION__LASSO",
    "XGB_HIC_ESM2_SEQ_AROMATIC": "HIC_ESM2_SEQ_AROMATIC__LASSO",
}


def export_one(experiment_id: str, recipe_id: str, compression: str = "zstd") -> dict:
    code = id_to_code(experiment_id)
    dev, test, _ = load_dev_test_folds()
    ids = dev["id"].astype(str).tolist() + test["id"].astype(str).tolist()
    blocks = recipe_blocks_from_recipes_csv(recipe_id)
    raw = concat_recipe_features(blocks, ids)
    raw = add_split_column(raw, dev["id"], test["id"])
    dest = ROOT / "experiments" / "features" / f"{code}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(dest, index=False, compression=compression)
    cols = feature_column_names(raw)
    return {
        "experiment_code": code,
        "experiment_id": experiment_id,
        "source_recipe_id": recipe_id,
        "n_features": len(cols),
        "n_rows": len(raw),
        "path": str(dest.relative_to(ROOT)),
        "feature_sha256": file_sha256(dest),
        "feature_content_sha256": feature_content_sha256(raw),
        "feature_recipe_hash": feature_recipe_hash(blocks, cols),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="optional experiment_id filter")
    args = ap.parse_args()
    jobs = list(FULL_JOBS.items())
    if args.only:
        allow = set(args.only)
        jobs = [j for j in jobs if j[0] in allow]
    rows = []
    for eid, recipe in jobs:
        print(f"[export] {id_to_code(eid)} ({eid}) <- {recipe}", flush=True)
        rows.append(export_one(eid, recipe))
    by_recipe: dict[str, set[str]] = {}
    for r in rows:
        by_recipe.setdefault(r["source_recipe_id"], set()).add(r["feature_content_sha256"])
    for recipe, hashes in by_recipe.items():
        if len(hashes) != 1:
            raise SystemExit(f"content hash mismatch for {recipe}: {hashes}")
    out = ROOT / "results" / "feature_export_meta.json"
    pd.DataFrame(rows).to_json(out, orient="records", indent=2)
    print(f"wrote {len(rows)} parquets; meta -> {out}")


if __name__ == "__main__":
    main()
