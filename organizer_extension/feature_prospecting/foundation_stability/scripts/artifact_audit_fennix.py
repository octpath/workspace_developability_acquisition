#!/usr/bin/env python3
"""Correlate FeNNix features vs OpenMM-STRAIN / clash / size proxies (artifact audit)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FS = FP / "foundation_stability"

FENNIX = [
    "ref_F_rms",
    "ref_F_q90",
    "ref_F_max",
    "P1_dE_mean",
    "P1_dE_sd",
    "P1_Frms_mean",
    "P2_dE_mean",
    "P3_dE_mean",
    "P3_Frms_mean",
    "interface_ref_F_rms",
    "n_atoms",
]
STRAIN = [
    "delta_E_per_residue",
    "initial_force_RMS",
    "initial_force_q95",
    "CA_RMSD_pre_post",
    "clash_relief_fraction",
    "severe_clash_count_before",
    "GENERATOR_GEOMETRY_ARTIFACT_DOMINATED_flag",
]


def corr_table(a: pd.DataFrame, b: pd.DataFrame, a_cols, b_cols, label: str):
    ids = sorted(set(a.index) & set(b.index))
    rows = []
    for ca in a_cols:
        if ca not in a.columns:
            continue
        for cb in b_cols:
            if cb not in b.columns:
                continue
            x = a.loc[ids, ca].astype(float)
            y = b.loc[ids, cb].astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 20:
                continue
            pr = float(pearsonr(x[m], y[m]).statistic)
            sp = float(spearmanr(x[m], y[m]).statistic)
            rows.append(
                {
                    "generator": label,
                    "fennix_feature": ca,
                    "proxy": cb,
                    "pearson": pr,
                    "spearman": sp,
                    "n": int(m.sum()),
                }
            )
    return rows


def main():
    out_rows = []
    for gen in ["esmfold", "abodybuilder2"]:
        fx = pd.read_csv(FS / "cache/fennix_features" / f"features_{gen}.csv").set_index("id")
        st = pd.read_parquet(FP / "OPENMM-STRAIN" / f"features_{gen}.parquet").set_index("id")
        # clash from pilot crosswalk if present
        cw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv").set_index("id")
        proxies = st[STRAIN].copy()
        for c in ["n_heavy", "clash_proxy", "clash", "n_residues"]:
            if c in cw.columns:
                proxies[c] = cw[c]
            if c in st.columns:
                proxies[c] = st[c]
        out_rows.extend(corr_table(fx, proxies, FENNIX, list(proxies.columns), gen))

    df = pd.DataFrame(out_rows)
    df.to_csv(FS / "results/FOUNDATION_VS_CLASSICAL_PHYSICS.csv", index=False)
    # summary: max |spearman| of force/dE vs clash vs strain
    summary = []
    for gen, g in df.groupby("generator"):
        clash_like = g[g.proxy.astype(str).str.contains("clash", case=False)]
        strain_like = g[g.proxy.isin(["delta_E_per_residue", "initial_force_RMS", "CA_RMSD_pre_post"])]
        summary.append(
            {
                "generator": gen,
                "max_abs_spearman_vs_clash": float(clash_like.spearman.abs().max()) if len(clash_like) else np.nan,
                "max_abs_spearman_vs_openmm_strain": float(strain_like.spearman.abs().max()) if len(strain_like) else np.nan,
                "median_abs_spearman_vs_openmm": float(strain_like.spearman.abs().median()) if len(strain_like) else np.nan,
            }
        )
    pd.DataFrame(summary).to_csv(FS / "results/FOUNDATION_VS_CLASSICAL_SUMMARY.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print("wrote", FS / "results/FOUNDATION_VS_CLASSICAL_PHYSICS.csv")


if __name__ == "__main__":
    main()
