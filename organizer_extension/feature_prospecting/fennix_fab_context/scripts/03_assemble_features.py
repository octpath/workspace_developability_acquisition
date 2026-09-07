#!/usr/bin/env python3
"""Assemble matched/delta feature tables + target-blind audit (no TmApp yet)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "fennix_fab_context"
V2 = FP / "foundation_stability_v2"
FAB = FP / "fab_reconstruction"
SEQ = FAB / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
CACHE = CTX / "cache/features"

# Shared aggregate columns for variable-region comparison (v2-compatible)
VAR_COLS = [
    "K_all_bb_mean",
    "K_all_bb_median",
    "K_all_bb_q90",
    "K_all_bb_iqr",
    "K_all_bb_frac_neg",
    "K_FW_mean",
    "K_FW_median",
    "K_FW_q90",
    "K_CDR_mean",
    "K_CDR_median",
    "K_CDR_q90",
    "K_HCDR3_mean",
    "K_HCDR3_median",
    "K_HCDR3_q90",
    "K_chi1_mean",
    "K_chi1_median",
    "K_chi1_q90",
]


def _load_feats(pilot: bool):
    if pilot:
        path = CACHE / "pilot_features.csv"
    else:
        # Prefer full-cohort resume artifact from 02_fennix_fab_curvature.py
        cand = CACHE / "features_partial_all.csv"
        path = cand if cand.exists() else CACHE / "features_partial.csv"
    return pd.read_csv(path)


def _load_a():
    a = pd.read_csv(V2 / "FENNIX_V2_CURVATURE_FEATURES.csv")
    a = a[(a.generator == "esmfold") & (a.extraction_status == "SUCCESS")].copy()
    a["condition"] = "A"
    return a


def delta_frame(left: pd.DataFrame, right: pd.DataFrame, cols, name_prefix: str) -> pd.DataFrame:
    L = left.set_index("id")
    R = right.set_index("id")
    ids = sorted(set(L.index) & set(R.index))
    rows = []
    for i in ids:
        row = {"id": i}
        for c in cols:
            if c in L.columns and c in R.columns:
                row[f"{name_prefix}__{c}"] = float(L.loc[i, c]) - float(R.loc[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    import sys

    pilot = "--pilot" in sys.argv
    fab = pd.read_csv(SEQ)
    feats = _load_feats(pilot)
    A = _load_a()
    B = feats[feats.condition == "B"].copy()
    C = feats[feats.condition == "C"].copy()
    M = feats[feats.condition == "M"].copy()

    # Matched variable features (wide)
    matched = A[["id"]].drop_duplicates()
    for label, df in [("A", A), ("B", B), ("M", M), ("C", C)]:
        sub = df.set_index("id")
        for c in VAR_COLS:
            if c in sub.columns:
                matched[f"{label}__{c}"] = matched.id.map(sub[c])
    matched.to_csv(CTX / "FENNIX_FAB_MATCHED_VARIABLE_FEATURES.csv", index=False)

    dgeom = delta_frame(B, A, VAR_COLS, "DELTA_GEOM")
    dgeom.to_csv(CTX / "FENNIX_FAB_DELTA_GEOM_FEATURES.csv", index=False)
    denv = delta_frame(C, M, VAR_COLS, "DELTA_ENV")
    denv.to_csv(CTX / "FENNIX_FAB_DELTA_ENV_FEATURES.csv", index=False)
    # prep sensitivity M - B
    prep_s = delta_frame(M, B, VAR_COLS, "PREP_RELAX_SENS")
    prep_s.to_csv(CTX / "FENNIX_FAB_PREP_RELAX_SENSITIVITY.csv", index=False)

    # Constant / interface / normalized from C
    const_cols = [c for c in C.columns if c.startswith("K_CH1_") or c.startswith("K_CL_")]
    # mean of CH1/CL medians
    out_c = C[["id"]].copy()
    for c in const_cols:
        out_c[c] = C[c].values
    if "K_CH1_median" in C.columns and "K_CL_median" in C.columns:
        out_c["K_CONST_mean_median"] = 0.5 * (C["K_CH1_median"] + C["K_CL_median"])
        out_c["K_CONST_diff_median"] = C["K_CH1_median"] - C["K_CL_median"]
    out_c.to_csv(CTX / "FENNIX_FAB_CONSTANT_FEATURES.csv", index=False)

    iface_cols = [c for c in C.columns if any(c.startswith(p) for p in ("K_VH_CH1_", "K_VL_CL_", "K_CH1_CL_"))]
    C[["id"] + iface_cols].to_csv(CTX / "FENNIX_FAB_INTERFACE_FEATURES.csv", index=False)

    norm = C[["id"]].copy()
    for src, dst in [
        ("K_all_median", "K_full_median"),
        ("K_all_mean", "K_full_mean"),
        ("K_all_q90", "K_full_q90"),
        ("K_all_n", "K_full_n_sites"),
    ]:
        if src in C.columns:
            norm[dst] = C[src].values
    if "K_all_median" in C.columns and "K_all_n" in C.columns:
        # already site-aggregate (not raw sum); keep as-is, add per-site note column
        norm["normalized_note"] = "aggregates_are_site_level_not_raw_sums"
    norm.to_csv(CTX / "FENNIX_FAB_NORMALIZED_FEATURES.csv", index=False)

    ss_cols = [c for c in C.columns if c.startswith("K_SS_neigh_")]
    if ss_cols:
        C[["id"] + ss_cols].to_csv(CTX / "FENNIX_FAB_DISULFIDE_SENSITIVITY.csv", index=False)
    else:
        pd.DataFrame({"id": C.id, "note": "SS_neigh_absent"}).to_csv(CTX / "FENNIX_FAB_DISULFIDE_SENSITIVITY.csv", index=False)

    # Target-blind artifact audit
    prep = pd.read_csv(CTX / "FAB_PREP_QC.csv") if (CTX / "FAB_PREP_QC.csv").exists() else pd.DataFrame()
    if pilot:
        qc_path = CACHE / "pilot_qc.csv"
    else:
        qc_cand = CACHE / "qc_partial_all.csv"
        qc_path = qc_cand if qc_cand.exists() else CACHE / "qc_partial.csv"
    fqc = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()

    audit_rows = []
    families = {
        "DELTA_GEOM": dgeom,
        "DELTA_ENV": denv,
        "CONSTANT": out_c,
        "INTERFACE": C[["id"] + iface_cols] if iface_cols else C[["id"]],
        "NORMALIZED": norm,
    }
    # size proxies
    size = fab.set_index("id")[["heavy_length", "light_length", "VH_len_used", "VL_len_used"]].copy()
    size["n_res"] = size.heavy_length + size.light_length

    lines = ["# FeNNix Fab — Target-blind feature audit", "", "No TmApp used.", ""]
    for fam, df in families.items():
        feat_cols = [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
        lines.append(f"## {fam}")
        lines.append(f"- n_abs={df.id.nunique()} n_features={len(feat_cols)}")
        for c in feat_cols[:12]:
            s = df[c].astype(float)
            miss = float(s.isna().mean())
            # correlate with size / prep
            row = {"family": fam, "feature": c, "missing_frac": miss, "median": float(s.median()), "q90": float(s.quantile(0.9)), "q10": float(s.quantile(0.1))}
            m = df[["id", c]].dropna().merge(size.reset_index(), on="id")
            if len(m) > 5:
                row["spearman_vs_n_res"] = float(spearmanr(m[c], m["n_res"]).statistic)
            else:
                row["spearman_vs_n_res"] = np.nan
            if len(prep):
                m2 = df[["id", c]].dropna().merge(prep, on="id")
                if len(m2) > 5 and "HL_SG_SG_before" in m2.columns:
                    dhl = m2["HL_SG_SG_before"] - m2["HL_SG_SG_after"] if "HL_SG_SG_after" in m2.columns else np.nan
                    if np.isfinite(dhl).sum() > 5:
                        row["spearman_vs_HL_correction"] = float(spearmanr(m2[c], dhl, nan_policy="omit").statistic)
                    else:
                        row["spearman_vs_HL_correction"] = np.nan
                else:
                    row["spearman_vs_HL_correction"] = np.nan
                if "severe_clash_after" in m2.columns and len(m2) > 5:
                    row["spearman_vs_clash"] = float(spearmanr(m2[c], m2["severe_clash_after"], nan_policy="omit").statistic)
                else:
                    row["spearman_vs_clash"] = np.nan
            if len(fqc):
                cq = fqc[fqc.condition == ("C" if fam != "DELTA_GEOM" else "B")]
                m3 = df[["id", c]].dropna().merge(cq, on="id")
                if len(m3) > 5 and "R1_F_rms" in m3.columns:
                    row["spearman_vs_F_rms"] = float(spearmanr(m3[c], m3["R1_F_rms"], nan_policy="omit").statistic)
                else:
                    row["spearman_vs_F_rms"] = np.nan
            audit_rows.append(row)
            lines.append(
                f"- `{c}`: median={row['median']:.4g} miss={miss:.2f} ρ(n_res)={row.get('spearman_vs_n_res', np.nan):.3f}"
            )
        lines.append("")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(CTX / "FENNIX_FAB_ARTIFACT_AUDIT.csv", index=False)
    # flag artifact-like
    if len(audit):
        art = audit[
            (audit.spearman_vs_n_res.abs() > 0.7)
            | (audit.get("spearman_vs_HL_correction", pd.Series(dtype=float)).abs() > 0.7)
            | (audit.get("spearman_vs_clash", pd.Series(dtype=float)).abs() > 0.7)
        ]
        lines.append("## Artifact-like flags (|ρ|>0.7 vs size/clash/HL-correction)")
        lines.append(f"- n_flagged={len(art)}")
        for _, r in art.head(20).iterrows():
            lines.append(f"- {r.family}/{r.feature}")
    (CTX / "FENNIX_FAB_TARGETBLIND_REPORT.md").write_text("\n".join(lines))
    print("wrote feature tables + audit", flush=True)


if __name__ == "__main__":
    main()
