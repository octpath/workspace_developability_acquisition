#!/usr/bin/env python3
"""Target-blind raw vs prepared Fab sensitivity audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CTX = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/structure_gap_closure")
CACHE = CTX / "cache"
RESULTS = CTX / "results"


def classify(rho, nad):
    if not np.isfinite(rho):
        return "PREP_DEPENDENT"
    if rho >= 0.9 and nad < 0.1:
        return "PREP_ROBUST"
    if rho >= 0.7 and nad < 0.25:
        return "PREP_SENSITIVE"
    return "PREP_DEPENDENT"


def compare(raw: pd.DataFrame, prep: pd.DataFrame, family: str):
    raw = raw.set_index("id")
    prep = prep.set_index("id")
    ids = sorted(set(raw.index) & set(prep.index))
    cols = [c for c in raw.columns if c in prep.columns and pd.api.types.is_numeric_dtype(raw[c])]
    rows = []
    for c in cols:
        a = raw.loc[ids, c].astype(float)
        b = prep.loc[ids, c].astype(float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 20:
            continue
        rho = float(spearmanr(a[m], b[m]).statistic)
        denom = np.maximum(np.abs(a[m]) + np.abs(b[m]), 1e-6)
        nad = float(np.median(np.abs(a[m] - b[m]) / denom))
        rows.append(
            {
                "family": family,
                "feature": c,
                "N": int(m.sum()),
                "spearman": rho,
                "median_nad": nad,
                "median_raw": float(np.median(a[m])),
                "median_prep": float(np.median(b[m])),
                "prep_sensitivity": classify(rho, nad),
            }
        )
    return rows


def main():
    rows = []
    pack_r = RESULTS / "TMAPP_PACKING_CAVITY_FEATURES.csv"
    pack_p = CACHE / "TMAPP_PACKING_CAVITY_PREPARED.csv"
    if pack_r.exists() and pack_p.exists():
        rows += compare(pd.read_csv(pack_r), pd.read_csv(pack_p), "PACKING_CAVITY")
    iface_r = RESULTS / "TMAPP_INTERFACE_FEATURES.csv"
    iface_p = CACHE / "TMAPP_INTERFACE_PREPARED.csv"
    if iface_r.exists() and iface_p.exists():
        rows += compare(pd.read_csv(iface_r), pd.read_csv(iface_p), "FAB_INTERFACE")
    unsat_p = RESULTS / "TMAPP_BURIED_UNSAT_FEATURES.csv"
    unsat_r = CACHE / "TMAPP_BURIED_UNSAT_RAW.csv"
    if unsat_p.exists() and unsat_r.exists():
        rows += compare(pd.read_csv(unsat_r), pd.read_csv(unsat_p), "BURIED_UNSAT")
    surf = RESULTS / "HIC_CONTINUOUS_SURFACE_FEATURES.csv"
    if surf.exists():
        S = pd.read_csv(surf)
        # compare fab_raw vs fab_prep shared feature stems
        raw_cols = [c for c in S.columns if c.startswith("fab_raw__")]
        for rc in raw_cols:
            stem = rc.replace("fab_raw__", "")
            pc = f"fab_prep__{stem}"
            if pc not in S.columns:
                continue
            a = S[["id", rc]].rename(columns={rc: "v"})
            b = S[["id", pc]].rename(columns={pc: "v"})
            rows += compare(a.rename(columns={"v": stem}), b.rename(columns={"v": stem}), "CONT_SURFACE_FAB")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "PREP_SENSITIVITY_AUDIT.csv", index=False)
    if len(df):
        print(df.groupby(["family", "prep_sensitivity"]).size().to_string())
    print("wrote PREP_SENSITIVITY_AUDIT", len(df), flush=True)


if __name__ == "__main__":
    main()
