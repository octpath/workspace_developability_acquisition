#!/usr/bin/env python3
"""Export RAW_PREPROCESS feature parquets for FULL experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    LIN_TOP6_MAP,
    ROOT,
    XGB_RECIPE,
    add_split_column,
    concat_recipe_features,
    feature_column_names,
    feature_content_sha256,
    feature_recipe_hash,
    file_sha256,
    load_dev_test_folds,
    recipe_blocks_from_recipes_csv,
)

# Linear Top-6: source_model_id -> recipe_id (same)
LIN_FULL = {v: k for k, v in LIN_TOP6_MAP.items()}


def export_one(experiment_id: str, recipe_id: str, compression: str = "zstd") -> dict:
    dev, test, _ = load_dev_test_folds()
    ids = dev["id"].astype(str).tolist() + test["id"].astype(str).tolist()
    blocks = recipe_blocks_from_recipes_csv(recipe_id)
    raw = concat_recipe_features(blocks, ids)
    raw = add_split_column(raw, dev["id"], test["id"])
    dest = ROOT / "experiments" / "features" / f"{experiment_id}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(dest, index=False, compression=compression)
    cols = feature_column_names(raw)
    meta = {
        "experiment_id": experiment_id,
        "recipe_id": recipe_id,
        "blocks": blocks,
        "n_features": len(cols),
        "n_rows": len(raw),
        "path": str(dest.relative_to(ROOT)),
        "feature_sha256": file_sha256(dest),
        "feature_content_sha256": feature_content_sha256(raw),
        "feature_recipe_hash": feature_recipe_hash(blocks, cols),
        "missing_count": int(raw[cols].isna().sum().sum()),
        "missing_fraction": float(raw[cols].isna().mean().mean()) if cols else 0.0,
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="optional experiment_id filter")
    args = ap.parse_args()
    jobs: list[tuple[str, str]] = []
    for eid, recipe in LIN_FULL.items():
        jobs.append((eid, recipe))
    for eid, recipe in XGB_RECIPE.items():
        jobs.append((eid, recipe))
    if args.only:
        allow = set(args.only)
        jobs = [j for j in jobs if j[0] in allow]
    rows = []
    for eid, recipe in jobs:
        print(f"[export] {eid} <- {recipe}", flush=True)
        rows.append(export_one(eid, recipe))
    out = ROOT / "results" / "feature_export_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out, orient="records", indent=2)
    # verify LIN/XGB same-recipe content hashes
    by_recipe: dict[str, list[dict]] = {}
    for r in rows:
        by_recipe.setdefault(r["recipe_id"], []).append(r)
    for recipe, group in by_recipe.items():
        hashes = {g["feature_content_sha256"] for g in group}
        if len(hashes) != 1:
            raise SystemExit(f"content hash mismatch for recipe {recipe}: {hashes}")
        print(f"[ok] recipe {recipe} content hash shared by {len(group)} experiments")
    print(f"wrote {len(rows)} parquets; meta -> {out}")


if __name__ == "__main__":
    main()
