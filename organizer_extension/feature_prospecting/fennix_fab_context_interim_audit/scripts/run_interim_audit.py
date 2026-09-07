#!/usr/bin/env python3
"""Interim scientific audit: Gap Closure combination + partial FeNNix signal.

READ-ONLY vs running FeNNix worker / caches (writes only under interim_audit/).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FENNIX = FP / "fennix_fab_context"
GAP = FP / "structure_gap_closure"
OUT = FP / "fennix_fab_context_interim_audit"
RES = OUT / "results"
V2 = FP / "foundation_stability_v2"
SEQ = FP / "fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
OOF_DIR = ROOT / "virtual_participant/stage5_integration/oof"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
EMB = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
TMAPP_REF = "TmApp__META_performance__ridge_100.0"
HIC_REF = "HIC__SIMPLE_blend_seq_surf_adv"
ABLANG_OOF = OOF_DIR / "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt__primary_nested_base.csv"
ESM2_OOF = OOF_DIR / "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt__primary_nested_base.csv"
ARO_CSV = GAP / "cache/aromatic_features_esmfold.csv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_N = 32
RNG = np.random.default_rng(42)
B_BOOT = 10000
TOL_MAX = 1e-4
TOL_RMSD = 1e-6

VAR_COLS = [
    "K_all_bb_mean", "K_all_bb_median", "K_all_bb_q90", "K_all_bb_iqr", "K_all_bb_frac_neg",
    "K_FW_mean", "K_FW_median", "K_FW_q90", "K_CDR_mean", "K_CDR_median", "K_CDR_q90",
    "K_HCDR3_mean", "K_HCDR3_median", "K_HCDR3_q90", "K_chi1_mean", "K_chi1_median", "K_chi1_q90",
]


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.set_index("id") if "id" in df.columns else df.copy()
    X.index = X.index.astype(str)
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    # leave NaN for fold-local median fill
    keep = [c for c in X.columns if X[c].notna().sum() > 10 and float(np.nanstd(X[c].values)) > 1e-12]
    return X[keep]


def select_alpha(X, y, folds, train_ids):
    fold_map = folds.set_index("id")["fold"].to_dict()
    best_a, best = 100.0, np.inf
    for a in ALPHAS:
        maes = []
        for f in sorted({fold_map[i] for i in train_ids}):
            te = [i for i in train_ids if fold_map[i] == f]
            tr = [i for i in train_ids if fold_map[i] != f]
            if len(te) == 0 or len(tr) < 2:
                continue
            med = X.loc[tr].median()
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            Xt = np.nan_to_num(sc.fit_transform(X.loc[tr].fillna(med)), nan=0.0)
            Xe = np.nan_to_num(sc.transform(X.loc[te].fillna(med)), nan=0.0)
            m.fit(Xt, y.loc[tr])
            maes.append(mae(y.loc[te], m.predict(Xe)))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def nested_ridge(X, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        med = X.loc[tr].median()
        Xtr, Xte = X.loc[tr].fillna(med), X.loc[te].fillna(med)
        a = select_alpha(Xtr, y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        Xt = np.nan_to_num(sc.fit_transform(Xtr), nan=0.0)
        Xe = np.nan_to_num(sc.transform(Xte), nan=0.0)
        m.fit(Xt, y.loc[tr])
        oof.loc[te] = m.predict(Xe)
    return oof.astype(float)


def residual_on(base_oof, X_phys, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_phys.index)
    cand = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        resid_tr = y.loc[tr] - base_oof.loc[tr]
        med = X_phys.loc[tr].median()
        X = X_phys.loc[tr].fillna(med)
        a = select_alpha(X, resid_tr, folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(np.nan_to_num(sc.fit_transform(X), nan=0.0), resid_tr)
        pred_r = m.predict(np.nan_to_num(sc.transform(X_phys.loc[te].fillna(med)), nan=0.0))
        cand.loc[te] = base_oof.loc[te] + pred_r
    return cand.astype(float)


def concat_inc(base_oof, X_phys, y, folds):
    Xp = X_phys.copy()
    Xp["__ref__"] = base_oof.reindex(Xp.index).astype(float)
    return nested_ridge(Xp, y, folds)


def plm_pca_oof(X_plm, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_plm.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        sc0 = StandardScaler()
        Ztr = sc0.fit_transform(X_plm.loc[tr])
        Zte = sc0.transform(X_plm.loc[te])
        pca = PCA(n_components=min(PCA_N, max(1, len(tr) - 1)), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        a = select_alpha(pd.DataFrame(Ptr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Ptr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Pte))
    return oof.astype(float)


def fuse_plm_phys(X_plm, X_phys, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_phys.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        sc0 = StandardScaler()
        Ztr = sc0.fit_transform(X_plm.loc[tr])
        Zte = sc0.transform(X_plm.loc[te])
        pca = PCA(n_components=min(PCA_N, max(1, len(tr) - 1)), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        scp = StandardScaler()
        med = X_phys.loc[tr].median()
        Xptr = np.nan_to_num(scp.fit_transform(X_phys.loc[tr].fillna(med)), nan=0.0)
        Xpte = np.nan_to_num(scp.transform(X_phys.loc[te].fillna(med)), nan=0.0)
        Xtr = np.hstack([Ptr, Xptr])
        Xte = np.hstack([Pte, Xpte])
        a = select_alpha(pd.DataFrame(Xtr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Xte))
    return oof.astype(float)


def boot_delta(y, base, cand, B=B_BOOT):
    idx = y.index.intersection(base.index).intersection(cand.index)
    y, base, cand = y.loc[idx], base.loc[idx], cand.loc[idx]
    d0 = mae(y, base) - mae(y, cand)  # >0 cand better
    boots = []
    n = len(y)
    for _ in range(B):
        ix = RNG.integers(0, n, n)
        boots.append(mae(y.iloc[ix], base.iloc[ix]) - mae(y.iloc[ix], cand.iloc[ix]))
    boots = np.asarray(boots)
    return float(d0), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float(np.mean(boots > 0))


def delta_frame(left, right, cols, prefix):
    L, R = left.set_index("id"), right.set_index("id")
    ids = sorted(set(L.index.astype(str)) & set(R.index.astype(str)))
    rows = []
    for i in ids:
        row = {"id": i}
        for c in cols:
            if c in L.columns and c in R.columns:
                row[f"{prefix}__{c}"] = float(L.loc[i, c]) - float(R.loc[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def load_coords(pdb: Path):
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    out = {}
    for a in st.get_atoms():
        res = a.get_parent()
        key = (res.get_parent().id, int(res.id[1]), a.get_name().strip())
        out[key] = np.asarray(a.coord, float)
    return out


def cm_match(ab_id, vh_len, vl_len):
    c_pdb = FENNIX / "cache/r1" / f"{ab_id}_C_r1.pdb"
    m_pdb = FENNIX / "cache/matched_fv" / f"{ab_id}_matched.pdb"
    if not c_pdb.exists() or not m_pdb.exists():
        return {"cm_ok": False, "cm_max_abs": np.nan, "cm_rmsd": np.nan, "cm_n_atoms": 0, "cm_error": "MISSING"}
    Cc, Mm = load_coords(c_pdb), load_coords(m_pdb)

    def var_keys(coords):
        return [k for k in coords if (k[0] == "A" and k[1] <= vh_len) or (k[0] == "B" and k[1] <= vl_len)]

    keys = sorted(set(var_keys(Cc)) & set(var_keys(Mm)))
    if not keys:
        return {"cm_ok": False, "cm_max_abs": np.nan, "cm_rmsd": np.nan, "cm_n_atoms": 0, "cm_error": "NO_OVERLAP"}
    diffs = np.asarray([Cc[k] - Mm[k] for k in keys], float)
    max_abs = float(np.max(np.abs(diffs)))
    rmsd = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
    ok = (max_abs < TOL_MAX) or (rmsd < TOL_RMSD)
    return {"cm_ok": ok, "cm_max_abs": max_abs, "cm_rmsd": rmsd, "cm_n_atoms": len(keys), "cm_error": "" if ok else "MISMATCH"}


def mean_bfactor(pdb: Path) -> float:
    if not pdb.exists():
        return np.nan
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    bfs = [a.get_bfactor() for a in st.get_atoms() if a.element.strip().upper() not in ("H", "")]
    return float(np.mean(bfs)) if bfs else np.nan


# -------------------- Part 2 combination --------------------
def run_combination_closure():
    y_tm = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)
    y_tm.index = y_tm.index.astype(str)
    inc_tm = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_pred"].astype(float)
    inc_tm.index = inc_tm.index.astype(str)
    y_hic = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_true"].astype(float)
    y_hic.index = y_hic.index.astype(str)
    inc_hic = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_pred"].astype(float)
    inc_hic.index = inc_hic.index.astype(str)

    pack = prep_X(pd.read_csv(GAP / "results/TMAPP_PACKING_CAVITY_FEATURES.csv"))
    unsat = prep_X(pd.read_csv(GAP / "results/TMAPP_BURIED_UNSAT_FEATURES.csv"))
    iface = prep_X(
        pd.read_csv(GAP / "results/TMAPP_INTERFACE_FEATURES.csv").drop(
            columns=["shape_complementarity_status"], errors="ignore"
        )
    )
    core = pd.concat([pack, unsat], axis=1, join="inner")
    core = core.loc[:, ~core.columns.duplicated()]
    gap_all = pd.concat([core, iface], axis=1, join="inner")
    gap_all = gap_all.loc[:, ~gap_all.columns.duplicated()]

    surf = pd.read_csv(GAP / "results/HIC_CONTINUOUS_SURFACE_FEATURES.csv")
    scols = [c for c in surf.columns if c.startswith("fv_esmfold__")]
    hic_surf = prep_X(surf[["id"] + scols])
    aro = prep_X(pd.read_csv(ARO_CSV)) if ARO_CSV.exists() else pd.DataFrame()
    hic_all = pd.concat([hic_surf, aro], axis=1, join="inner") if len(aro) else hic_surf
    hic_all = hic_all.loc[:, ~hic_all.columns.duplicated()]

    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    families = [
        ("TmApp", "CORE_DEFECT", core, y_tm, inc_tm),
        ("TmApp", "FAB_INTERFACE", iface, y_tm, inc_tm),
        ("TmApp", "GAP_ALL", gap_all, y_tm, inc_tm),
        ("HIC", "HIC_SURFACE_ALL", hic_all, y_hic, inc_hic),
    ]
    rows = []
    for target, name, X, y, inc in families:
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            ids = sorted(set(X.index) & set(y.index) & set(inc.index) & set(folds.id))
            if len(ids) < 40:
                continue
            Xx, yy, ii = X.loc[ids], y.loc[ids], inc.loc[ids]
            ff = folds[folds.id.isin(ids)].copy()
            # A: incumbent alone
            inc_mae = mae(yy, ii)
            # B: ridge concat
            oof_c = concat_inc(ii, Xx, yy, ff)
            d_c, lo_c, hi_c, p_c = boot_delta(yy, ii, oof_c)
            # C: residual
            oof_r = residual_on(ii, Xx, yy, ff)
            d_r, lo_r, hi_r, p_r = boot_delta(yy, ii, oof_r)
            for framework, oof, d, lo, hi, p in [
                ("ridge_concat_oof_scalar", oof_c, d_c, lo_c, hi_c, p_c),
                ("residual_plus_oof", oof_r, d_r, lo_r, hi_r, p_r),
            ]:
                rows.append(
                    {
                        "target": target,
                        "family": name,
                        "framework": framework,
                        "cv": tag,
                        "N": len(ids),
                        "feature_dim": int(Xx.shape[1]),
                        "incumbent_mae": inc_mae,
                        "combined_mae": mae(yy, oof),
                        "delta_mae_inc_minus_comb": d,  # >0 improvement
                        "boot_ci_low": lo,
                        "boot_ci_high": hi,
                        "p_improve": p,
                        "verdict": (
                            "NO_INCREMENT"
                            if d <= 0
                            else ("WEAK_INCREMENT" if hi > 0 else "CLEAR_INCREMENT")
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "GAP_CLOSURE_COMBINATION_CLOSURE.csv", index=False)
    return out


# -------------------- FeNNix interim --------------------
def build_complete_set():
    feat = pd.read_csv(FENNIX / "cache/features/features_partial_all.csv")
    feat["id"] = feat.id.astype(str)
    ok = feat[feat.extraction_status == "SUCCESS"]
    by = ok.groupby("id")["condition"].apply(lambda s: set(s.astype(str)))
    ids = [i for i, s in by.items() if {"B", "C", "M"} <= s]
    fab = pd.read_csv(SEQ).set_index("id")
    fab.index = fab.index.astype(str)
    y = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv")
    y["id"] = y.id.astype(str)
    dev = set(y.id)
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    # completion order: first row index where id reaches M success after B,C present
    order = []
    seen = set()
    have = {i: set() for i in ids}
    for _, r in ok.iterrows():
        i = str(r.id)
        if i not in have:
            continue
        have[i].add(str(r.condition))
        if i not in seen and {"B", "C", "M"} <= have[i]:
            order.append(i)
            seen.add(i)

    rows = []
    for ab in ids:
        cm = cm_match(ab, int(fab.loc[ab, "VH_len_used"]), int(fab.loc[ab, "VL_len_used"])) if ab in fab.index else {"cm_ok": False}
        matched = (FENNIX / "cache/matched_fv" / f"{ab}_matched.pdb").exists()
        r1 = (FENNIX / "cache/r1" / f"{ab}_C_r1.pdb").exists()
        usable = bool(cm.get("cm_ok")) and matched and r1
        rows.append(
            {
                "id": ab,
                "is_dev": ab in dev,
                "primary_fold": int(primary.set_index("id").loc[ab, "fold"]) if ab in set(primary.id) else np.nan,
                "shadow_fold": int(shadow.set_index("id").loc[ab, "fold"]) if ab in set(shadow.id) else np.nan,
                "cm_ok": cm.get("cm_ok"),
                "cm_max_abs": cm.get("cm_max_abs"),
                "cm_rmsd": cm.get("cm_rmsd"),
                "matched_pdb": matched,
                "r1_pdb": r1,
                "usable_interim": usable,
                "completion_rank": order.index(ab) if ab in order else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "INTERIM_FENNIX_COMPLETE_SET.csv", index=False)
    return df, feat, order


def assemble_features(complete_ids, feat):
    """Deterministic assembly into interim results/ only."""
    A = pd.read_csv(V2 / "FENNIX_V2_CURVATURE_FEATURES.csv")
    A = A[(A.generator == "esmfold") & (A.extraction_status == "SUCCESS")].copy()
    A["id"] = A.id.astype(str)
    B = feat[(feat.condition == "B") & (feat.extraction_status == "SUCCESS")].copy()
    C = feat[(feat.condition == "C") & (feat.extraction_status == "SUCCESS")].copy()
    M = feat[(feat.condition == "M") & (feat.extraction_status == "SUCCESS")].copy()
    for d in (B, C, M):
        d["id"] = d.id.astype(str)
    ids = [i for i in complete_ids if i in set(A.id) and i in set(B.id) and i in set(C.id) and i in set(M.id)]
    A, B, C, M = A[A.id.isin(ids)], B[B.id.isin(ids)], C[C.id.isin(ids)], M[M.id.isin(ids)]

    dgeom = delta_frame(B, A, VAR_COLS, "DELTA_GEOM")
    denv = delta_frame(C, M, VAR_COLS, "DELTA_ENV")
    prep_s = delta_frame(M, B, VAR_COLS, "PREP_RELAX_SENS")
    const_cols = [c for c in C.columns if c.startswith("K_CH1_") or c.startswith("K_CL_")]
    out_c = C[["id"]].copy()
    for c in const_cols:
        out_c[c] = C[c].values
    if "K_CH1_median" in C.columns and "K_CL_median" in C.columns:
        out_c["K_CONST_mean_median"] = 0.5 * (C["K_CH1_median"].values + C["K_CL_median"].values)
        out_c["K_CONST_diff_median"] = C["K_CH1_median"].values - C["K_CL_median"].values
    iface_cols = [c for c in C.columns if any(c.startswith(p) for p in ("K_VH_CH1_", "K_VL_CL_", "K_CH1_CL_"))]
    iface = C[["id"] + iface_cols].copy()
    norm = C[["id"]].copy()
    for src, dst in [("K_all_median", "K_full_median"), ("K_all_mean", "K_full_mean"), ("K_all_q90", "K_full_q90"), ("K_all_n", "K_full_n_sites")]:
        if src in C.columns:
            norm[dst] = C[src].values
    # predeclared combined: concat of all family matrices
    comb = dgeom.set_index("id")
    for df, pref in [(denv, ""), (out_c, ""), (iface, ""), (norm, ""), (prep_s, "")]:
        comb = comb.join(df.set_index("id"), how="inner", rsuffix="_x")
    comb = comb.loc[:, ~comb.columns.duplicated()].reset_index()

    paths = {
        "DELTA_GEOM": dgeom,
        "DELTA_ENV": denv,
        "CONSTANT": out_c,
        "INTERFACE": iface,
        "FULL_FAB_NORMALIZED": norm,
        "PREP_RELAX_SENSITIVITY": prep_s,
        "COMBINED_PREDECLARED": comb,
    }
    for name, df in paths.items():
        df.to_csv(RES / f"INTERIM_{name}_FEATURES.csv", index=False)
    return paths, ids


def write_feature_freeze(paths, usable_ids):
    freeze = {
        "frozen_before_target_labels": True,
        "frozen_utc": pd.Timestamp.now("UTC").isoformat(),
        "source_spec": "fennix_fab_context/FENNIX_FAB_CONTEXT_SPEC.md",
        "N_complete_bcm": len(usable_ids),
        "families": list(paths.keys()),
        "var_cols": VAR_COLS,
        "note": "Assembled deterministically from features_partial_all SUCCESS B/C/M + FeNNix-v2 A; no target-based selection.",
        "writes_only_to": str(OUT),
        "fennix_worker_untouched": True,
    }
    (OUT / "INTERIM_FENNIX_FEATURE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")


def targetblind_qc(paths, usable_ids, feat):
    fab = pd.read_csv(SEQ).set_index("id")
    fab.index = fab.index.astype(str)
    qc = pd.read_csv(FENNIX / "cache/features/qc_partial_all.csv")
    qc["id"] = qc.id.astype(str)
    prep = pd.read_csv(FENNIX / "FAB_PREP_QC.csv")
    prep = prep.groupby("id").tail(1).set_index("id")
    prep.index = prep.index.astype(str)
    rows = []
    for fam, df in paths.items():
        X = prep_X(df)
        X = X.loc[[i for i in usable_ids if i in X.index]]
        finite = float(np.isfinite(X.to_numpy(dtype=float)).mean()) if X.size else np.nan
        # correlate mean abs feature with proxies
        score = X.abs().mean(axis=1)
        for proxy_name, series in [
            ("n_atoms_C", qc[qc.condition == "C"].set_index("id")["n_atoms"]),
            ("R1_F_rms_C", qc[qc.condition == "C"].set_index("id")["R1_F_rms"]),
            ("severe_clash_C", qc[qc.condition == "C"].set_index("id")["R1_severe_clash"]),
            ("n_site_evals_C", qc[qc.condition == "C"].set_index("id")["n_site_evals"] if "n_site_evals" in qc.columns else None),
        ]:
            if series is None:
                continue
            s = series.reindex(score.index).astype(float)
            m = score.notna() & s.notna()
            rho = float(spearmanr(score[m], s[m]).statistic) if m.sum() > 20 else np.nan
            rows.append({"family": fam, "metric": f"spearman_absmean_vs_{proxy_name}", "value": rho, "N": int(m.sum())})
        if "HL_SG_SG_before" in prep.columns:
            hl = (prep["HL_SG_SG_before"] - prep["HL_SG_SG_after"]).reindex(score.index).astype(float)
            m = score.notna() & hl.notna()
            rho = float(spearmanr(score[m], hl[m]).statistic) if m.sum() > 20 else np.nan
            rows.append({"family": fam, "metric": "spearman_absmean_vs_HL_correction", "value": rho, "N": int(m.sum())})
        # pLDDT proxy: mean bfactor raw fab
        plddts = []
        for i in score.index:
            plddts.append(mean_bfactor(FP / "fab_reconstruction/structures/esmfold_fab" / f"{i}.pdb"))
        plddt = pd.Series(plddts, index=score.index)
        m = score.notna() & plddt.notna()
        rho = float(spearmanr(score[m], plddt[m]).statistic) if m.sum() > 20 else np.nan
        rows.append({"family": fam, "metric": "spearman_absmean_vs_mean_bfactor", "value": rho, "N": int(m.sum())})
        rows.append({"family": fam, "metric": "finite_rate", "value": finite, "N": len(X)})
        rows.append({"family": fam, "metric": "feature_dim", "value": float(X.shape[1]), "N": len(X)})
        rows.append({"family": fam, "metric": "outlier_frac_absz_gt5", "value": float((np.abs((X - X.mean()) / X.std(ddof=0)) > 5).any(axis=1).mean()) if X.shape[1] else np.nan, "N": len(X)})
    # empty pools
    for cond in ("B", "C", "M"):
        sub = feat[(feat.id.isin(usable_ids)) & (feat.condition == cond) & (feat.extraction_status == "SUCCESS")]
        for col in [c for c in sub.columns if c.endswith("_n")]:
            empty = float((sub[col].fillna(0) <= 0).mean())
            rows.append({"family": f"pool_{cond}", "metric": f"empty_frac_{col}", "value": empty, "N": len(sub)})
    pd.DataFrame(rows).to_csv(OUT / "INTERIM_FENNIX_TARGETBLIND_QC.csv", index=False)
    return pd.DataFrame(rows)


def completion_bias(complete_df, order):
    ydf = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv")
    ydf["id"] = ydf.id.astype(str)
    ydf = ydf.set_index("id")
    fab = pd.read_csv(SEQ).set_index("id")
    fab.index = fab.index.astype(str)
    primary = pd.read_csv(PRIMARY)
    primary["id"] = primary.id.astype(str)
    complete_dev = set(complete_df.loc[complete_df.is_dev & complete_df.usable_interim, "id"])
    all_dev = set(ydf.index)
    incomplete = all_dev - complete_dev
    lines = ["# Interim FeNNix completion-bias audit (DEV only)", ""]
    lines.append(f"- N_complete_usable_dev = {len(complete_dev)}")
    lines.append(f"- N_not_yet_complete_dev = {len(incomplete)}")
    lines.append(f"- N_all_dev_with_label = {len(all_dev)}")
    lines.append("")

    def summarize(name, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            return f"- {name}: insufficient"
        diff = (np.mean(a) - np.mean(b)) / (np.std(np.concatenate([a, b]), ddof=0) + 1e-9)
        try:
            u = mannwhitneyu(a, b, alternative="two-sided")
            p = float(u.pvalue)
        except Exception:
            p = np.nan
        return f"- {name}: mean_complete={np.mean(a):.4g} mean_rest={np.mean(b):.4g} std_diff={diff:.3f} MW_p={p:.3g}"

    yc = ydf.loc[sorted(complete_dev), "y_true"]
    yi = ydf.loc[sorted(incomplete), "y_true"]
    lines.append(summarize("TmApp_y", yc, yi))
    lines.append(summarize("incumbent_OOF", ydf.loc[sorted(complete_dev), "y_pred"], ydf.loc[sorted(incomplete), "y_pred"]))
    resid_c = ydf.loc[sorted(complete_dev), "y_true"] - ydf.loc[sorted(complete_dev), "y_pred"]
    resid_i = ydf.loc[sorted(incomplete), "y_true"] - ydf.loc[sorted(incomplete), "y_pred"]
    lines.append(summarize("|incumbent_residual|", resid_c.abs(), resid_i.abs()))
    lines.append(summarize("VH_len", fab.loc[sorted(complete_dev), "VH_len_used"], fab.loc[sorted(incomplete & set(fab.index)), "VH_len_used"]))
    lines.append(summarize("VL_len", fab.loc[sorted(complete_dev), "VL_len_used"], fab.loc[sorted(incomplete & set(fab.index)), "VL_len_used"]))
    lines.append(summarize("Fab_len", fab.loc[sorted(complete_dev), "heavy_length"] + fab.loc[sorted(complete_dev), "light_length"],
                           fab.loc[sorted(incomplete & set(fab.index)), "heavy_length"] + fab.loc[sorted(incomplete & set(fab.index)), "light_length"]))
    # kappa/lambda
    if "light_locus" in fab.columns:
        def frac_k(ids):
            s = fab.loc[[i for i in ids if i in fab.index], "light_locus"].astype(str)
            return float((s.str.lower() == "kappa").mean()) if len(s) else np.nan
        lines.append(f"- kappa_frac complete={frac_k(complete_dev):.3f} rest={frac_k(incomplete):.3f}")
    # fold distribution
    pc = primary[primary.id.isin(complete_dev)]["fold"].value_counts().sort_index()
    pi = primary[primary.id.isin(incomplete)]["fold"].value_counts().sort_index()
    lines.append(f"- Primary fold counts complete: {pc.to_dict()}")
    lines.append(f"- Primary fold counts rest: {pi.to_dict()}")
    # classification
    diffs = []
    for a, b in [(yc, yi), (resid_c.abs(), resid_i.abs())]:
        a, b = np.asarray(a, float), np.asarray(b, float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        diffs.append(abs((np.mean(a) - np.mean(b)) / (np.std(np.concatenate([a, b]), ddof=0) + 1e-9)))
    if max(diffs) < 0.25 and len(complete_dev) >= 40:
        label = "COMPLETION_SUBSET_APPROX_REPRESENTATIVE"
    elif max(diffs) > 0.5:
        label = "COMPLETION_SUBSET_BIASED"
    else:
        label = "COMPLETION_BIAS_UNCERTAIN"
    lines.append("")
    lines.append(f"**Classification: `{label}`**")
    (OUT / "INTERIM_FENNIX_COMPLETION_BIAS.md").write_text("\n".join(lines) + "\n")
    return label, complete_dev


def eval_interim(paths, complete_dev, bias_label, qc_df, order):
    y = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")
    y.index = y.index.astype(str)
    y_true = y["y_true"].astype(float)
    inc = y["y_pred"].astype(float)
    ablang = pd.read_csv(ABLANG_OOF).set_index("id")["y_pred"].astype(float)
    ablang.index = ablang.index.astype(str)
    z = np.load(EMB, allow_pickle=True)
    plm = pd.DataFrame(np.asarray(z["ablang2__HL_paired"], float), index=[str(x) for x in z["ids"]])
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    rows = []
    resid_diag = []
    for fam, df in paths.items():
        X = prep_X(df)
        ids = sorted(set(X.index) & complete_dev & set(y_true.index) & set(inc.index) & set(plm.index))
        if len(ids) < 30:
            continue
        Xx = X.loc[ids]
        # drop n-count heavy cols similarly to final eval: take up to 20 numeric
        cols = [c for c in Xx.columns if not c.endswith("_n") and not c.endswith("_n_sites")][:20]
        Xx = Xx[cols]
        yy, ii = y_true.loc[ids], inc.loc[ids]
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            ff = folds[folds.id.isin(ids)].copy()
            if ff["fold"].nunique() < 2 or len(ff) < 30:
                verdict = "INTERIM_UNDERPOWERED"
            else:
                verdict = None
            phys = nested_ridge(Xx, yy, ff)
            plm_oof = plm_pca_oof(plm.loc[ids], yy, ff)
            fuse = fuse_plm_phys(plm.loc[ids], Xx, yy, ff)
            stack = concat_inc(ii, Xx, yy, ff)
            resid = residual_on(ii, Xx, yy, ff)
            med = pd.Series(index=ids, dtype=float)
            fmap = ff.set_index("id")["fold"].to_dict()
            for f in sorted(set(fmap.values())):
                te = [i for i in ids if fmap[i] == f]
                tr = [i for i in ids if fmap[i] != f]
                med.loc[te] = float(np.median(yy.loc[tr]))
            for mode, base, cand in [
                ("standalone_vs_median", med, phys),
                ("incr_ablang2_fuse", plm_oof, fuse),
                ("incr_incumbent_concat", ii, stack),
                ("residual_incumbent", ii, resid),
            ]:
                d, lo, hi, p = boot_delta(yy, base, cand)
                if verdict is None:
                    if mode.startswith("incr_incumbent") or mode.startswith("residual"):
                        # provisional vocabulary
                        if d > 0 and lo > 0:
                            v = "PROVISIONAL_POSITIVE_SIGNAL"
                        elif d > 0:
                            v = "PROVISIONAL_WEAK_SIGNAL"
                        else:
                            v = "PROVISIONAL_NO_SIGNAL_YET"
                    else:
                        v = "PROVISIONAL_WEAK_SIGNAL" if d > 0 else "PROVISIONAL_NO_SIGNAL_YET"
                else:
                    v = verdict
                # artifact override from QC
                art = qc_df[(qc_df.family == fam) & (qc_df.metric.str.contains("spearman_absmean_vs_"))]
                if len(art) and art["value"].abs().max(skipna=True) > 0.7:
                    v = "PROVISIONAL_ARTIFACT_SUSPECT"
                if bias_label == "COMPLETION_SUBSET_BIASED" and mode.startswith("incr_incumbent"):
                    v = "INTERIM_BIASED_SUBSET"
                rows.append(
                    {
                        "family": fam,
                        "cv": tag,
                        "mode": mode,
                        "N": len(ids),
                        "feature_dim": Xx.shape[1],
                        "base_mae": mae(yy, base),
                        "cand_mae": mae(yy, cand),
                        "delta_mae_base_minus_cand": d,
                        "boot_ci_low": lo,
                        "boot_ci_high": hi,
                        "p_improve": p,
                        "provisional_verdict": v,
                        "n_primary_folds": int(ff["fold"].nunique()),
                    }
                )
            # residual diagnostics
            resid_y = yy - ii
            for c in Xx.columns[:12]:
                m = Xx[c].notna() & resid_y.notna()
                if m.sum() < 20:
                    continue
                resid_diag.append(
                    {
                        "family": fam,
                        "feature": c,
                        "spearman_vs_inc_residual": float(spearmanr(Xx.loc[m, c], resid_y[m]).statistic),
                        "N": int(m.sum()),
                    }
                )
            # multivariate residual R2 under nested (diagnostic)
            oof_r = nested_ridge(Xx, resid_y, ff)
            ss_res = float(np.sum((resid_y - oof_r) ** 2))
            ss_tot = float(np.sum((resid_y - resid_y.mean()) ** 2)) + 1e-12
            resid_diag.append(
                {
                    "family": fam,
                    "feature": "__multivariate_ridge__",
                    "spearman_vs_inc_residual": np.nan,
                    "residual_oof_R2": 1.0 - ss_res / ss_tot,
                    "residual_oof_MAE": mae(resid_y, oof_r),
                    "N": len(ids),
                }
            )

    # checkpoints 50 / 75 / all for top families
    # use DELTA_ENV and DELTA_GEOM
    for fam in ("DELTA_GEOM", "DELTA_ENV", "COMBINED_PREDECLARED"):
        X = prep_X(paths[fam])
        cols = [c for c in X.columns if not c.endswith("_n")][:20]
        X = X[cols]
        for ncut in (50, 75, None):
            if ncut is None:
                ids_c = sorted(set(X.index) & complete_dev & set(y_true.index))
                tag = "all_complete_dev"
            else:
                # first ncut usable complete_dev in completion order
                ordered = [i for i in order if i in complete_dev and i in X.index]
                ids_c = ordered[:ncut]
                tag = f"first_{ncut}_dev"
            if len(ids_c) < 40:
                continue
            # require fold coverage
            for folds, cv in [(primary, "Primary"), (shadow, "Shadow")]:
                ff = folds[folds.id.isin(ids_c)].copy()
                if ff["fold"].nunique() < 3:
                    continue
                Xx, yy, ii = X.loc[ids_c], y_true.loc[ids_c], inc.loc[ids_c]
                stack = concat_inc(ii, Xx, yy, ff)
                d, lo, hi, p = boot_delta(yy, ii, stack)
                rows.append(
                    {
                        "family": fam,
                        "cv": cv,
                        "mode": f"checkpoint_incr_incumbent_{tag}",
                        "N": len(ids_c),
                        "feature_dim": Xx.shape[1],
                        "base_mae": mae(yy, ii),
                        "cand_mae": mae(yy, stack),
                        "delta_mae_base_minus_cand": d,
                        "boot_ci_low": lo,
                        "boot_ci_high": hi,
                        "p_improve": p,
                        "provisional_verdict": "INTERIM_UNDERPOWERED" if len(ids_c) < 60 else ("PROVISIONAL_WEAK_SIGNAL" if d > 0 else "PROVISIONAL_NO_SIGNAL_YET"),
                        "n_primary_folds": int(ff["fold"].nunique()),
                    }
                )

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "INTERIM_FENNIX_RESULTS.csv", index=False)
    pd.DataFrame(resid_diag).to_csv(RES / "INTERIM_FENNIX_RESIDUAL_DIAG.csv", index=False)
    return res


def write_report(complete_df, bias_label, results, combo, qc_df):
    usable_dev = complete_df[complete_df.is_dev & complete_df.usable_interim]
    n_dev = len(usable_dev)
    # fold counts
    pf = usable_dev["primary_fold"].value_counts().sort_index().to_dict()
    sf = usable_dev["shadow_fold"].value_counts().sort_index().to_dict()

    def best_row(fam, mode_substr):
        sub = results[(results.family == fam) & (results["mode"].str.contains(mode_substr))]
        if sub.empty:
            return None
        return sub

    lines = ["# Interim FeNNix + Gap Closure Combination — 日本語報告", ""]
    lines.append("**注意:** FeNNix 本計算は未完了。本報告は暫定診断のみ。最終特徴定義・実行中ワーカーは変更していない。")
    lines.append("")
    lines.append("## 冒頭 10 問")
    lines.append("")
    lines.append(f"1. **利用可能な FeNNix Dev 完了 Abs 数は？**  \n   **{n_dev}**（BCM SUCCESS + C↔M 座標一致 + matched/r1）。全完了（Dev+非Dev）usable={int(complete_df.usable_interim.sum())} / BCM完了={len(complete_df)}。")
    lines.append("")
    lines.append(f"2. **完了サブセットは Dev 全体を近似的に代表するか？**  \n   **`{bias_label}`**（詳細: `INTERIM_FENNIX_COMPLETION_BIAS.md`）。")
    lines.append("")

    def summarize_fam(fam, q):
        sub = results[(results.family == fam) & (results["mode"] == "incr_incumbent_concat")]
        if sub.empty:
            return f"{q} **データ不足 / 未評価**"
        p = sub[sub.cv == "Primary"]
        s = sub[sub.cv == "Shadow"]
        if p.empty or s.empty:
            return f"{q} **fold 不足**"
        dp, ds = float(p.iloc[0].delta_mae_base_minus_cand), float(s.iloc[0].delta_mae_base_minus_cand)
        vp, vs = p.iloc[0].provisional_verdict, s.iloc[0].provisional_verdict
        return (
            f"{q} Primary ΔMAE(inc−cand)={dp:.4f} (`{vp}`), Shadow={ds:.4f} (`{vs}`)。"
            f"同方向改善={dp>0 and ds>0}。"
        )

    lines.append("3. " + summarize_fam("DELTA_GEOM", "**DELTA_GEOM に暫定 TmApp 情報は？**  \n  "))
    lines.append("")
    lines.append("4. " + summarize_fam("DELTA_ENV", "**DELTA_ENV に暫定 TmApp 情報は？**  \n  "))
    lines.append("")
    lines.append("5. " + summarize_fam("CONSTANT", "**CONSTANT:**  \n  "))
    lines.append("   " + summarize_fam("INTERFACE", "**INTERFACE:**  \n  "))
    lines.append("   " + summarize_fam("FULL_FAB_NORMALIZED", "**FULL_FAB_NORMALIZED:**  \n  "))
    lines.append("")

    # AbLang2
    def ablang_any():
        sub = results[results["mode"] == "incr_ablang2_fuse"]
        if sub.empty:
            return "データ不足"
        ok = False
        for fam in sub.family.unique():
            p = sub[(sub.family == fam) & (sub.cv == "Primary")]
            s = sub[(sub.family == fam) & (sub.cv == "Shadow")]
            if len(p) and len(s) and float(p.iloc[0].delta_mae_base_minus_cand) > 0 and float(s.iloc[0].delta_mae_base_minus_cand) > 0:
                ok = True
        return "ある（Primary+Shadow 同方向）" if ok else "明確な同方向改善は見えない（暫定）"

    lines.append(f"6. **同一部分コホートで AbLang2 を改善する family は？**  \n   {ablang_any()}。")
    lines.append("")
    lines.append(f"7. **同一部分コホートで TmApp incumbent を改善する family は？**  \n   {summarize_fam('COMBINED_PREDECLARED', '')} 個別は結果 CSV 参照。")
    lines.append("")
    # consistency
    cons = []
    for fam in results.family.unique():
        p = results[(results.family == fam) & (results.cv == "Primary") & (results["mode"] == "incr_incumbent_concat")]
        s = results[(results.family == fam) & (results.cv == "Shadow") & (results["mode"] == "incr_incumbent_concat")]
        if len(p) and len(s):
            cons.append((fam, np.sign(p.iloc[0].delta_mae_base_minus_cand) == np.sign(s.iloc[0].delta_mae_base_minus_cand)))
    lines.append(f"8. **Primary/Shadow 方向一致は？**  \n   " + ", ".join(f"{f}:{'一致' if c else '不一致'}" for f, c in cons))
    lines.append("")
    # artifacts
    art = qc_df[qc_df.metric.str.contains("spearman_absmean")]
    flagged = art[art["value"].abs() > 0.7] if len(art) else art
    lines.append(f"9. **見かけの信号は prep/force/size アーティファクトか？**  \n   |ρ|>0.7 フラグ数={len(flagged)}。詳細は `INTERIM_FENNIX_TARGETBLIND_QC.csv`。DELTA_ENV はサイト数・F_rms・サイズとの相関を必ず確認。")
    lines.append("")
    # Q10 provisional
    any_pos = results[
        (results["mode"].isin(["incr_incumbent_concat", "residual_incumbent"]))
        & (results.provisional_verdict == "PROVISIONAL_POSITIVE_SIGNAL")
    ]
    if len(any_pos):
        expect = "暫定的には、full-Fab FeNNix が incumbent 増分を持ちうる可能性を完全には否定できない（ただし部分コホート・未確定）。"
    elif len(results[results.provisional_verdict == "PROVISIONAL_WEAK_SIGNAL"]):
        expect = "暫定的には弱い信号の可能性はあるが、最終コホートで消失しうる。現時点で科学的に「確定的に面白い」とは言えない。"
    else:
        expect = "現完了 Dev サブセットでは、incumbent 増分の暫定ポジティブ信号は見えていない。最終結果が面白くなる強い根拠はまだ薄い（暫定）。"
    lines.append(f"10. **最終 full-Fab FeNNix が科学的に面白くなりそうか（暫定）？**  \n   {expect}")
    lines.append("")
    lines.append("## Gap Closure combination closure（要約）")
    if len(combo):
        lines.append(combo.groupby(["family", "framework"])["delta_mae_inc_minus_comb"].mean().to_string())
        lines.append("")
        lines.append("両 framework を報告（事後選択なし）。詳細: `GAP_CLOSURE_COMBINATION_CLOSURE.csv`。")
    lines.append("")
    lines.append(f"## 折りたたみ統計")
    lines.append(f"- Primary folds (usable Dev): {pf}")
    lines.append(f"- Shadow folds (usable Dev): {sf}")
    (OUT / "INTERIM_FENNIX_REPORT_JA.md").write_text("\n".join(lines) + "\n")


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    RES.mkdir(parents=True, exist_ok=True)
    print("PART2 combination closure…", flush=True)
    combo = run_combination_closure()
    print("PART3 complete set + CM match…", flush=True)
    complete_df, feat, order = build_complete_set()
    usable = complete_df[complete_df.usable_interim].id.tolist()
    print(f"complete={len(complete_df)} usable={len(usable)}", flush=True)
    print("PART4 assemble + freeze (before target eval)…", flush=True)
    paths, ids_asm = assemble_features(usable, feat)
    write_feature_freeze(paths, usable)
    print("PART5 target-blind QC…", flush=True)
    qc_df = targetblind_qc(paths, usable, feat)
    print("PART6 completion bias…", flush=True)
    bias_label, complete_dev = completion_bias(complete_df, order)
    print("PART7–9 provisional eval…", flush=True)
    # Restrict usable to those with A/B/C/M assembled
    complete_dev = set(complete_dev) & set(ids_asm)
    results = eval_interim(paths, complete_dev, bias_label, qc_df, order)
    write_report(complete_df, bias_label, results, combo, qc_df)
    print("DONE interim audit", flush=True)
    print("combo rows", len(combo), "results", len(results), "bias", bias_label)


if __name__ == "__main__":
    main()
