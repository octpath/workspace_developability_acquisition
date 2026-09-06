#!/usr/bin/env python3
"""Top-candidate fusion (at most 3 structurally distinct families). Target-blind selection by Primary INC_vs_INCUMBENT delta, then evaluate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
SM = ROOT / "organizer_extension/feature_prospecting/structure_marathon"
sys.path.insert(0, str(SM / "scripts"))
from eval_families import eval_family, nested_ridge, load_y, load_plm, PRIMARY, SHADOW, stack_inc, boot_delta, metrics  # noqa

# Distinct class representatives (predeclared)
CANDIDATES = {
    "TmApp": [
        ("structure_guided_PLM", "S1_PATCH_NEIGHBORS", SM / "structure_guided_pooling/S1_FEATURES.csv", "S1_PATCH_NEIGHBORS_", True),
        ("inverse_folding", "M2_ESM_IF1", SM / "esm_if1/M2_ESMIF1_FEATURES.csv", "IF1_", False),
        ("generator_disagreement", "S2_disagreement", SM / "generator_disagreement/S2_DISAGREEMENT_FEATURES.csv", None, False),
    ],
    "HIC": [
        ("structure_guided_PLM", "S1_STRONGLY_EXPOSED_AROMATIC", SM / "structure_guided_pooling/S1_FEATURES.csv", "S1_STRONGLY_EXPOSED_AROMATIC_", True),
        ("surface_graph", "S3_surface_graph", SM / "surface_patch_graph/S3_FEATURES.csv", None, False),
        ("inverse_folding", "M2_ESM_IF1", SM / "esm_if1/M2_ESMIF1_FEATURES.csv", "IF1_", False),
    ],
}


def load_family(path, prefix, use_status=False):
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df.status.astype(str).str.startswith("SUCCESS")]
    df = df.set_index("id")
    if prefix:
        cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith("_n")
                and (not prefix.startswith("S1_") or c[len(prefix):].split("_")[0].isdigit() or True)]
        if prefix.startswith("S1_"):
            cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith("_n")
                    and c[len(prefix):].split("_")[0].isdigit()]
        X = df[cols]
    else:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:30]
        X = df[cols]
    return X


def main():
    rows = []
    for target, cands in CANDIDATES.items():
        Xs = []
        names = []
        for cls, fam, path, pref, pca in cands:
            if not path.exists():
                continue
            X = load_family(path, pref)
            # standardize column names to avoid collisions
            X = X.add_prefix(f"{fam}__")
            Xs.append(X)
            names.append(fam)
        if not Xs:
            continue
        common = Xs[0].index
        for X in Xs[1:]:
            common = common.intersection(X.index)
        Xc = pd.concat([X.loc[common] for X in Xs], axis=1)
        # drop all-nan columns
        Xc = Xc.dropna(axis=1, how="all")
        print(target, "fusion", Xc.shape, names, flush=True)
        rows.extend(eval_family(f"FUSION_{'+'.join(names)}", Xc, target, use_pca=True))
    out = pd.DataFrame(rows)
    path = SM / "results/STRUCTURE_FAMILY_SCORECARD.csv"
    prev = pd.read_csv(path)
    prev = prev[~prev.family.str.startswith("FUSION_")]
    pd.concat([prev, out], ignore_index=True).to_csv(path, index=False)
    sub = out[(out["mode"] == "INC_vs_INCUMBENT") & (out.cv == "Primary")]
    print(sub[["target", "family", "delta_MAE", "cand_MAE", "boot_lo", "boot_hi", "p_improve"]].to_string(index=False))


if __name__ == "__main__":
    main()
