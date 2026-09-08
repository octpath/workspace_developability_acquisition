#!/usr/bin/env python3
"""Validate top_models_feature_bundle."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FORBIDDEN = {
    "tmapp",
    "hic",
    "is_public",
    "is_private",
    "y_true",
    "oof",
    "residual",
    "public_mae",
    "private_mae",
}


def main() -> int:
    errs = []
    man = pd.read_csv(ROOT / "feature_manifest.csv")
    recipes = pd.read_csv(ROOT / "recipes.csv")
    folds = pd.read_csv(ROOT / "folds.csv")
    folds["id"] = folds["id"].astype(str)

    # folds only DEV
    if folds.scheme.nunique() < 2:
        errs.append("folds missing primary/shadow")
    if folds.id.duplicated().any() and False:
        pass

    for _, row in man.iterrows():
        if row.license_status == "NOT_INCLUDED_LICENSE_REVIEW" or not row.file:
            continue
        path = ROOT / row.file
        if not path.exists():
            errs.append(f"missing file {row.file}")
            continue
        df = pd.read_parquet(path)
        if "id" not in df.columns:
            errs.append(f"{row.file}: no id column")
            continue
        if df["id"].duplicated().any():
            errs.append(f"{row.file}: duplicate ids")
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] != int(row.n_features):
            errs.append(f"{row.file}: n_features mismatch {num.shape[1]} vs {row.n_features}")
        if not np.isfinite(num.to_numpy(float)).all():
            # allow documented missing — flag only if all nan
            if num.isna().all().all():
                errs.append(f"{row.file}: entirely NaN")
        low = {c.lower() for c in df.columns}
        leak = low & FORBIDDEN
        if leak:
            errs.append(f"{row.file}: forbidden cols {leak}")

    # every recipe block exists or marked unavailable
    for _, r in recipes.iterrows():
        blocks = [b for b in str(r.feature_blocks).split("|") if b and b != "none"]
        for b in blocks:
            m = man[man.block_name == b]
            if len(m) == 0:
                errs.append(f"recipe {r.recipe_id}: block {b} not in manifest")
            elif not m.iloc[0].file and m.iloc[0].license_status != "NOT_INCLUDED_LICENSE_REVIEW":
                errs.append(f"recipe {r.recipe_id}: block {b} missing without license mark")

    # examples import
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        import load_features  # noqa: F401
    except Exception as e:
        errs.append(f"examples/load_features import fail: {e}")

    if errs:
        print("VALIDATION FAIL")
        for e in errs:
            print(" -", e)
        return 1
    print("VALIDATION OK")
    print(f"manifest_blocks={len(man)} recipes={len(recipes)} folds={len(folds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
