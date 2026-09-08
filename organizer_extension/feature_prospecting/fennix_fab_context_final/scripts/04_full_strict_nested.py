#!/usr/bin/env python3
"""Full-cohort STRICT_NESTED_INCREMENT for frozen FeNNix families."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FINAL = FP / "fennix_fab_context_final/results"
INTERIM = FP / "fennix_fab_context_interim_audit"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"

spec = importlib.util.spec_from_file_location("strict", INTERIM / "scripts/run_strict_nested_increment.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["strict"] = mod
spec.loader.exec_module(mod)

FAMS = [
    "DELTA_GEOM",
    "DELTA_ENV",
    "CONSTANT",
    "INTERFACE",
    "FULL_FAB_NORMALIZED",
    "PREP_RELAX_SENSITIVITY",
    "COMBINED_PREDECLARED",
]


def main():
    usable = set(pd.read_csv(FINAL / "FINAL_FENNIX_COHORT.csv").id.astype(str))
    y = pd.read_csv(mod.OOF_DIR / f"{mod.TMAPP_REF}.csv").set_index("id")
    y.index = y.index.astype(str)
    y_true = y["y_true"].astype(float)
    inc = y["y_pred"].astype(float)

    engine = mod.IncumbentEngine()
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    rows = []
    for fam in FAMS:
        X = mod.prep_X(pd.read_parquet(FINAL / f"FENNIX_{fam}.parquet"))
        cols = [c for c in X.columns if not c.endswith("_n") and not c.endswith("_n_sites")]
        X = X[cols]
        print(f"strict {fam} dim={X.shape[1]}", flush=True)
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            rows.extend(
                mod.eval_family(
                    engine,
                    "FullFeNNix",
                    "TmApp",
                    fam,
                    X,
                    y_true,
                    inc,
                    folds,
                    tag,
                    id_filter=usable,
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(FINAL / "FENNIX_FULL_STRICT_NESTED_RESULTS.csv", index=False)
    sub = out[out.protocol == "STRICT_NESTED_INCREMENT"]
    if len(sub):
        print(sub.groupby(["family", "framework", "cv"])["delta_mae_inc_minus_comb"].mean().to_string())
    print("wrote STRICT", flush=True)


if __name__ == "__main__":
    main()
