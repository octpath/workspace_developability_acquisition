#!/usr/bin/env python3
"""Target-blind static ESMFold Fv-vs-Fab vs dynamic BioEmu context deltas."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "bioemu_constant_context"
FAB = FP / "fab_reconstruction"


def main():
    # Prefer existing Fv-vs-Fab comparison table if present
    candidates = list((FAB / "comparisons").glob("*.csv")) + list(FAB.glob("**/*FV*FAB*.csv"))
    static = None
    for p in candidates:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "id" in cols and any("rmsd" in c.lower() for c in df.columns):
            static = df.rename(columns={cols["id"]: "id"})
            static_path = p
            break
    if static is None:
        raise SystemExit(f"No static Fv-vs-Fab table found under {FAB}/comparisons")

    # pick a primary static RMSD column
    rmsd_cols = [c for c in static.columns if "rmsd" in c.lower() and "combined" in c.lower()]
    if not rmsd_cols:
        rmsd_cols = [c for c in static.columns if "rmsd" in c.lower()]
    primary_static = rmsd_cols[0]

    delta = pd.read_csv(CTX / "BIOEMU_LIGHT_CONTEXT_DELTA_FEATURES.csv")
    coup = pd.read_csv(CTX / "BIOEMU_LIGHT_VC_COUPLING_FEATURES.csv") if (CTX / "BIOEMU_LIGHT_VC_COUPLING_FEATURES.csv").exists() else None
    m = static.merge(delta, on="id", how="inner")
    if coup is not None:
        m = m.merge(coup, on="id", how="left")

    dyn_cols = [c for c in m.columns if c.startswith("L3_") or c.startswith("L4_")]
    rows = []
    x = m[primary_static].astype(float)
    for c in dyn_cols:
        y = m[c].astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 20:
            continue
        rows.append(
            {
                "static_feature": primary_static,
                "dynamic_feature": c,
                "spearman": float(spearmanr(x[mask], y[mask]).statistic),
                "pearson": float(pearsonr(x[mask], y[mask]).statistic),
                "n": int(mask.sum()),
                "static_source": str(static_path.relative_to(FP)),
            }
        )
    out = pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    out.to_csv(CTX / "BIOEMU_STATIC_DYNAMIC_CONTEXT_COMPARISON.csv", index=False)
    # summary note
    print("static col", primary_static, "n_dyn", len(dyn_cols), "top", out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
