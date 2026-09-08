#!/usr/bin/env python3
"""Final FeNNix Fab cohort assembly + target-blind QC (no full-cohort rerun).

Uses exact frozen definitions from:
  fennix_fab_context/scripts/03_assemble_features.py
  fennix_fab_context_interim_audit/INTERIM_FENNIX_FEATURE_FREEZE.json
  fennix_fab_context/FENNIX_FAB_CONTEXT_SPEC.md
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "fennix_fab_context"
OUT = FP / "fennix_fab_context_final"
RES = OUT / "results"
V2 = FP / "foundation_stability_v2"
CACHE = CTX / "cache"
FEAT_ALL = CACHE / "features/features_partial_all.csv"
QC_ALL = CACHE / "features/qc_partial_all.csv"
SKIP_ID = "ADI-47265"

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def delta_frame(left: pd.DataFrame, right: pd.DataFrame, cols, name_prefix: str) -> pd.DataFrame:
    L = left.set_index("id")
    R = right.set_index("id")
    ids = sorted(set(L.index.astype(str)) & set(R.index.astype(str)))
    rows = []
    for i in ids:
        row = {"id": i}
        for c in cols:
            if c in L.columns and c in R.columns:
                row[f"{name_prefix}__{c}"] = float(L.loc[i, c]) - float(R.loc[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def load_a() -> pd.DataFrame:
    a = pd.read_csv(V2 / "FENNIX_V2_CURVATURE_FEATURES.csv")
    a = a[(a.generator == "esmfold") & (a.extraction_status == "SUCCESS")].copy()
    a["id"] = a["id"].astype(str)
    a["condition"] = "A"
    return a


def assemble():
    RES.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(FEAT_ALL)
    feats["id"] = feats["id"].astype(str)
    assert (feats["extraction_status"] == "SUCCESS").all()

    B = feats[feats.condition == "B"].copy()
    C = feats[feats.condition == "C"].copy()
    M = feats[feats.condition == "M"].copy()
    ids_b, ids_c, ids_m = set(B.id), set(C.id), set(M.id)
    ids = sorted(ids_b & ids_c & ids_m)
    assert SKIP_ID not in ids
    assert len(ids) == 323, len(ids)

    # r1 / matched presence
    rows = []
    for i in ids:
        ok = all((CACHE / "r1" / f"{i}_{c}.npz").is_file() for c in "BCM")
        matched = (CACHE / "matched_fv" / f"{i}_matched.pdb").is_file()
        rows.append(
            {
                "id": i,
                "B_status": "SUCCESS",
                "C_status": "SUCCESS",
                "M_status": "SUCCESS",
                "matched_status": "OK" if matched else "MISSING_MATCHED_PDB",
                "r1_npz_ok": bool(ok),
                "qc_status": "OK" if ok and matched else "ARTIFACT_SUSPECT",
                "accepted": bool(ok and matched),
            }
        )
    cohort = pd.DataFrame(rows)
    # require accepted
    if not cohort["accepted"].all():
        bad = cohort.loc[~cohort.accepted, "id"].tolist()
        raise RuntimeError(f"cohort integrity fail: {bad[:10]}")
    cohort.to_csv(RES / "FINAL_FENNIX_COHORT.csv", index=False)

    A = load_a()
    A = A[A.id.isin(ids)]
    B, C, M = B[B.id.isin(ids)], C[C.id.isin(ids)], M[M.id.isin(ids)]
    if set(A.id) != set(ids):
        raise RuntimeError(f"A missing {set(ids)-set(A.id)}")

    dgeom = delta_frame(B, A, VAR_COLS, "DELTA_GEOM")
    denv = delta_frame(C, M, VAR_COLS, "DELTA_ENV")
    prep = delta_frame(M, B, VAR_COLS, "PREP_RELAX_SENS")

    # Exact frozen CONSTANT definition (includes K_CH1_CL_* via startswith K_CH1_)
    const_cols = [c for c in C.columns if c.startswith("K_CH1_") or c.startswith("K_CL_")]
    out_c = C[["id"]].copy()
    for c in const_cols:
        out_c[c] = C[c].values
    if "K_CH1_median" in C.columns and "K_CL_median" in C.columns:
        out_c["K_CONST_mean_median"] = 0.5 * (C["K_CH1_median"].values + C["K_CL_median"].values)
        out_c["K_CONST_diff_median"] = C["K_CH1_median"].values - C["K_CL_median"].values

    iface_cols = [c for c in C.columns if any(c.startswith(p) for p in ("K_VH_CH1_", "K_VL_CL_", "K_CH1_CL_"))]
    out_i = C[["id"] + iface_cols].copy()

    norm = C[["id"]].copy()
    for src, dst in [
        ("K_all_median", "K_full_median"),
        ("K_all_mean", "K_full_mean"),
        ("K_all_q90", "K_full_q90"),
        ("K_all_n", "K_full_n_sites"),
    ]:
        if src in C.columns:
            norm[dst] = C[src].values

    families = {
        "DELTA_GEOM": dgeom,
        "DELTA_ENV": denv,
        "CONSTANT": out_c,
        "INTERFACE": out_i,
        "FULL_FAB_NORMALIZED": norm,
        "PREP_RELAX_SENSITIVITY": prep,
    }
    for name, df in families.items():
        df.to_csv(RES / f"FENNIX_{name}_FEATURES.csv", index=False)

    # COMBINED_PREDECLARED: inner join all, drop duplicate non-id cols keeping first
    comb = families["DELTA_GEOM"].copy()
    for name in ["DELTA_ENV", "CONSTANT", "INTERFACE", "FULL_FAB_NORMALIZED", "PREP_RELAX_SENSITIVITY"]:
        df = families[name]
        overlap = [c for c in df.columns if c != "id" and c in comb.columns]
        add = df.drop(columns=overlap) if overlap else df
        comb = comb.merge(add, on="id", how="inner")
    comb.to_csv(RES / "FENNIX_COMBINED_PREDECLARED_FEATURES.csv", index=False)
    families["COMBINED_PREDECLARED"] = comb

    # Full wide table = COMBINED (all frozen features)
    full = comb.sort_values("id").reset_index(drop=True)
    # drop non-numeric note-like
    drop = [c for c in full.columns if c != "id" and not pd.api.types.is_numeric_dtype(full[c])]
    full = full.drop(columns=drop)
    assert full["id"].is_unique and len(full) == 323
    full_path = RES / "FENNIX_FAB_FEATURES_FULL.parquet"
    full.to_parquet(full_path, index=False)

    # per-family also as parquet for eval convenience
    for name, df in families.items():
        num = df.copy()
        drop = [c for c in num.columns if c != "id" and not pd.api.types.is_numeric_dtype(num[c])]
        num = num.drop(columns=drop)
        num.to_parquet(RES / f"FENNIX_{name}.parquet", index=False)

    return cohort, families, full, full_path


def write_dictionary(full: pd.DataFrame):
    rows = []
    for c in full.columns:
        if c == "id":
            continue
        if c.startswith("DELTA_GEOM__"):
            fam, region, cond = "DELTA_GEOM", "variable_Fv", "B_minus_A"
            math = "feat(B) − feat(A) on shared VAR_COLS; A=isolated Fv FeNNix-v2, B=Fab-geom Fv R1"
        elif c.startswith("DELTA_ENV__"):
            fam, region, cond = "DELTA_ENV", "variable_Fv", "C_minus_M"
            math = "feat(C_var) − feat(M_var); same Fv coords with/without constant-domain environment (M=matched Fv from C, no re-relax)"
        elif c.startswith("PREP_RELAX_SENS__"):
            fam, region, cond = "PREP_RELAX_SENSITIVITY", "variable_Fv", "M_minus_B"
            math = "feat(M) − feat(B); prep/relax sensitivity control (not primary energy Δ)"
        elif c.startswith("K_full_"):
            fam, region, cond = "FULL_FAB_NORMALIZED", "full_Fab", "C"
            math = "Site-level aggregate curvature stats on full Fab (C); not raw energy sums comparable across different atom counts"
        elif c.startswith("K_VH_CH1_") or c.startswith("K_VL_CL_") or c.startswith("K_CH1_CL_"):
            fam, region, cond = "INTERFACE", "domain_interfaces", "C"
            math = "Region-site curvature aggregates on VH–CH1 / VL–CL / CH1–CL interface sites (condition C)"
        elif c.startswith("K_CH1_") or c.startswith("K_CL_") or c.startswith("K_CONST_"):
            fam, region, cond = "CONSTANT", "constant_domains", "C"
            math = "CH1/CL region-site curvature aggregates (+ CONST mean/diff of medians). Note: frozen startswith also pulls K_CH1_CL_* into CONSTANT"
            region = "constant_domains_and_CH1_CL_prefix_overlap"
        else:
            fam, region, cond = "OTHER", "unknown", "unknown"
            math = "see SPEC"
        rows.append(
            {
                "feature": c,
                "family": fam,
                "molecular_scope": "Fab_reconstructed (VH+CH1 / VL+CL)",
                "region": region,
                "condition_source": cond,
                "mathematical_definition": math,
                "units": "FeNNix local curvature / K aggregates (protocol-defined; not absolute energy Δ across different atom counts)",
                "normalization": "none beyond site-level aggregates in extraction",
                "notes": "target_used=NO; A/B/C/M per FENNIX_FAB_CONTEXT_SPEC",
            }
        )
    dic = pd.DataFrame(rows)
    dic.to_csv(RES / "FENNIX_FAB_FEATURE_DICTIONARY.csv", index=False)
    return dic


def run_qc(cohort, full, families):
    qc_raw = pd.read_csv(QC_ALL)
    qc_raw["id"] = qc_raw["id"].astype(str)
    prep = pd.read_csv(CTX / "FAB_PREP_QC.csv")
    prep["id"] = prep["id"].astype(str)

    lines = ["# FeNNix Fab — Final target-blind QC", "", f"UTC: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.append("## Coverage")
    lines.append(f"- expected IDs: 323 (exclude {SKIP_ID})")
    lines.append(f"- feature rows: {len(full)}")
    lines.append(f"- duplicate IDs: {int(full.id.duplicated().sum())}")
    lines.append(f"- missing IDs vs cohort: {len(set(cohort.id)-set(full.id))}")

    feat_cols = [c for c in full.columns if c != "id"]
    arr = full[feat_cols].to_numpy(dtype=float)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    lines.append(f"- missing feature cells: {n_nan}")
    lines.append(f"- ±inf cells: {n_inf}")

    dist_rows = []
    for c in feat_cols:
        s = full[c].astype(float)
        dist_rows.append(
            {
                "feature": c,
                "min": float(np.nanmin(s)),
                "q01": float(np.nanquantile(s, 0.01)),
                "median": float(np.nanmedian(s)),
                "q99": float(np.nanquantile(s, 0.99)),
                "max": float(np.nanmax(s)),
                "missing": int(s.isna().sum()),
                "zero_variance": bool(np.nanstd(s) < 1e-12),
            }
        )
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(RES / "FENNIX_FAB_FINAL_QC.csv", index=False)
    zv = dist[dist.zero_variance]
    lines.append(f"- zero-variance features: {len(zv)}")

    # B/C/M integrity already in cohort
    lines.append("\n## B/C/M integrity")
    lines.append(f"- all accepted: {bool(cohort.accepted.all())}")
    lines.append(f"- r1 npz ok: {bool(cohort.r1_npz_ok.all())}")

    # technical associations
    tech_rows = []
    qcC = qc_raw[qc_raw.condition == "C"].set_index("id")
    last_prep = prep.groupby("id").tail(1).set_index("id")
    for fam_name, df in families.items():
        if fam_name == "COMBINED_PREDECLARED":
            continue
        cols = [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
        for c in cols[:8]:
            m = df[["id", c]].dropna().copy()
            m["n_atoms"] = m.id.map(qcC["n_atoms"])
            m["n_sites"] = m.id.map(qcC["n_site_evals"])
            m["F_rms"] = m.id.map(qcC["R1_F_rms"])
            m["clash"] = m.id.map(qcC["R1_severe_clash"])
            if "HL_SG_SG_before" in last_prep.columns and "HL_SG_SG_after" in last_prep.columns:
                m["hl_corr"] = m.id.map(last_prep["HL_SG_SG_before"] - last_prep["HL_SG_SG_after"])
            row = {"family": fam_name, "feature": c}
            for tech in ("n_atoms", "n_sites", "F_rms", "clash", "hl_corr"):
                if tech in m.columns and m[tech].notna().sum() > 20:
                    row[f"rho_{tech}"] = float(spearmanr(m[c], m[tech], nan_policy="omit").statistic)
                else:
                    row[f"rho_{tech}"] = np.nan
            tech_rows.append(row)
    tech = pd.DataFrame(tech_rows)
    tech.to_csv(RES / "FENNIX_FAB_TECH_ASSOCIATIONS.csv", index=False)

    # size recoding flags
    suspect = []
    for _, r in tech.iterrows():
        for k in ("rho_n_atoms", "rho_n_sites"):
            if pd.notna(r[k]) and abs(r[k]) > 0.85:
                suspect.append(f"{r.family}:{r.feature} {k}={r[k]:.3f}")
    lines.append("\n## Technical associations (flags |ρ|>0.85 vs n_atoms/n_sites)")
    if suspect:
        for s in suspect[:30]:
            lines.append(f"- {s}")
        verdict = "QC_PASS_WITH_LIMITATIONS"
    else:
        lines.append("- none above |ρ|=0.85 among sampled features")
        verdict = "QC_PASS"
    if n_inf or n_nan or not cohort.accepted.all():
        verdict = "TECHNICAL_FAIL"
    elif len(zv):
        verdict = "QC_PASS_WITH_LIMITATIONS"

    lines.append(f"\n## Overall QC verdict\n\n**{verdict}**\n")
    lines.append("### Conditions A/B/C/M (SPEC)")
    lines.append("- **A**: isolated Fv ESMFold (FeNNix-v2)")
    lines.append("- **B**: Fab-geom Fv (from Fab) + R1 relax")
    lines.append("- **C**: full prepared Fab + R1")
    lines.append("- **M**: matched Fv coords from C (constants removed), no re-relax")
    lines.append("- **DELTA_GEOM** = B−A; **DELTA_ENV** = C_var−M_var (environment, not absolute energy across systems)")
    (RES / "FENNIX_FAB_FINAL_QC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict, dist


def main():
    cohort, families, full, full_path = assemble()
    dic = write_dictionary(full)
    verdict, dist = run_qc(cohort, full, families)
    meta = {
        "assembled_utc": datetime.now(timezone.utc).isoformat(),
        "N": int(len(full)),
        "excluded": [SKIP_ID],
        "feature_dim": int(full.shape[1] - 1),
        "families": {k: int(v.shape[1] - 1) for k, v in families.items()},
        "qc_verdict": verdict,
        "feature_table": str(full_path.relative_to(ROOT)),
        "feature_sha256": sha256_file(full_path),
        "dictionary_sha256": sha256_file(RES / "FENNIX_FAB_FEATURE_DICTIONARY.csv"),
        "spec": "fennix_fab_context/FENNIX_FAB_CONTEXT_SPEC.md",
        "assembly": "exact 03_assemble_features.py + INTERIM_FENNIX_FEATURE_FREEZE families",
    }
    (RES / "ASSEMBLY_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("QC", verdict)


if __name__ == "__main__":
    main()
