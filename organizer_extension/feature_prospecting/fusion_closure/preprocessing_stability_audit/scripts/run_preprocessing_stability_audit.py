#!/usr/bin/env python3
"""Fusion preprocessing stability audit (HIC × AROMATIC; target-blind first)."""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "fusion_closure/preprocessing_stability_audit"
RES = OUT / "results"
sys.path.insert(0, str(FP))
from common.ridge_eval import HIC_REF, OOF_DIR, mae, metrics  # noqa: E402

EMB_PATH = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
FOLDS_PATH = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
AROMA_DIR = FP / "AROMATIC-TOPO"
GENS = ["esmfold", "abodybuilder2", "boltz2"]
GEN_LABEL = {"esmfold": "ESMFold", "abodybuilder2": "ABB2", "boltz2": "Boltz2"}
SEED = 42


def load_canonical():
    return list(json.loads((AROMA_DIR / "FEATURE_SPEC.json").read_text())["canonical_features"])


def load_physics(gen: str) -> pd.DataFrame:
    cols = load_canonical()
    df = pd.read_parquet(AROMA_DIR / f"features_{gen}.parquet")
    df = df[df["extraction_status"].astype(str).str.upper().isin(["SUCCESS", "OK"])]
    return df.drop_duplicates("id").set_index("id")[cols].astype(float)


def load_esm2() -> pd.DataFrame:
    z = np.load(EMB_PATH, allow_pickle=True)
    return pd.DataFrame(np.asarray(z["esm2__H"], float), index=[str(x) for x in z["ids"]])


def fill_nan_train_median(Xtr, Xte):
    Xtr = np.asarray(Xtr, float).copy()
    Xte = np.asarray(Xte, float).copy()
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte


def classify_pca(mean_cos2: float, min_cos2: float) -> str:
    if mean_cos2 >= 0.90 and min_cos2 >= 0.70:
        return "STABLE"
    if mean_cos2 >= 0.75 and min_cos2 >= 0.40:
        return "MODERATELY_VARIABLE"
    return "HIGHLY_FOLD_DEPENDENT"


def principal_angles_cos2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A,B: (d, k) orthonormal bases. Return cos^2 of principal angles."""
    # QR for numerical orthonormality
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s**2


def projection_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    Pa = Qa @ Qa.T
    Pb = Qb @ Qb.T
    return float(np.linalg.norm(Pa - Pb, ord="fro"))


def fit_pca_scale_before(Xtr, n_comp, whiten=False):
    sc = StandardScaler()
    Z = sc.fit_transform(Xtr)
    n = min(n_comp, Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=n, whiten=whiten, random_state=SEED)
    scores = pca.fit_transform(Z)
    return sc, pca, scores


def fit_pca_center_only(Xtr, n_comp, whiten=False):
    mean = Xtr.mean(axis=0)
    Z = Xtr - mean
    n = min(n_comp, Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=n, whiten=whiten, random_state=SEED)
    scores = pca.fit_transform(Z)
    return mean, pca, scores


def write_implementation_audit(fold_df, ids):
    lines = []
    lines.append("# PREPROCESSING_IMPLEMENTATION_AUDIT")
    lines.append("")
    lines.append("Source of truth: `fusion_closure/scripts/run_fusion_closure.py` (executed Fusion Closure).")
    lines.append("Historical result files were **not** modified.")
    lines.append("")
    lines.append("## A. PLM branch (HIC / ESM2 Heavy)")
    lines.append("")
    lines.append("| Item | Actual value |")
    lines.append("|------|--------------|")
    lines.append("| Raw embedding key | `esm2__H` from `round1_embeddings.npz` |")
    lines.append("| Raw dimensionality | **1280** |")
    lines.append("| StandardScaler on raw dims | **Yes** (`plm_pca`) |")
    lines.append("| Order | **StandardScaler(raw) → PCA** (scale **BEFORE** PCA) |")
    lines.append("| PCA `n_components` | **32** (`PCA_N = 32`; capped by `min(32, n_train-1, n_features)`) |")
    lines.append("| PCA `whiten` | **False** (sklearn default; not passed) |")
    lines.append("| PCA `random_state` | 42 |")
    lines.append("")
    lines.append("```211:217:organizer_extension/feature_prospecting/fusion_closure/scripts/run_fusion_closure.py")
    lines.append("def plm_pca(X_plm_tr, X_plm_te, n_comp=PCA_N):")
    lines.append("    sc = StandardScaler()")
    lines.append("    Ztr = sc.fit_transform(X_plm_tr)")
    lines.append("    Zte = sc.transform(X_plm_te)")
    lines.append("    n = min(n_comp, Ztr.shape[0] - 1, Ztr.shape[1])")
    lines.append("    pca = PCA(n_components=n, random_state=SEED)")
    lines.append("    return pca.fit_transform(Ztr), pca.transform(Zte)")
    lines.append("```")
    lines.append("")
    lines.append("## B. AROMATIC branch")
    lines.append("")
    lines.append("| Item | Actual value |")
    lines.append("|------|--------------|")
    lines.append("| Feature count | **15** canonical (`AROMATIC-TOPO/FEATURE_SPEC.json`) |")
    lines.append("| StandardScaler | **Yes** (`physics_scale`) |")
    lines.append("| Imputation | train-median fill for non-finite (`fill_nan_train_median`); then scaler |")
    lines.append("| log / clip / other transform | **None** in Fusion Closure path |")
    lines.append("")
    lines.append("## C. Fusion concatenation")
    lines.append("")
    lines.append("| Item | Actual value |")
    lines.append("|------|--------------|")
    lines.append("| Independent block scaling | **Yes**: PLM scaled inside `plm_pca`; AROMATIC scaled in `physics_scale` |")
    lines.append("| Concat | `[PLM_PCA32, physics_scaled]` |")
    lines.append("| Scale again after concat | **Yes**: `fit_predict` / `select_ridge_alpha` apply another `StandardScaler` on the concatenated design before Ridge/SVR |")
    lines.append("")
    lines.append("```295:301:organizer_extension/feature_prospecting/fusion_closure/scripts/run_fusion_closure.py")
    lines.append("def fit_predict(...):")
    lines.append("    a = select_ridge_alpha(Xtr, ytr, groups_tr)")
    lines.append("    sc = StandardScaler()")
    lines.append("    m = Ridge(alpha=a, random_state=0)")
    lines.append("    m.fit(sc.fit_transform(Xtr), ytr)")
    lines.append("```")
    lines.append("")
    lines.append("## D. CV and preprocessing fit sample counts")
    lines.append("")
    lines.append(f"- Primary fold file: `virtual_participant/stage0_cv/cv_primary.csv` (Dev N={len(ids)})")
    lines.append("- Outer: leave-one-fold-out over fold ids {0,1,2,3,4}")
    lines.append("")
    lines.append("### Outer fold sizes")
    lines.append("")
    lines.append("| Outer val fold | N_val | N_outer_train |")
    lines.append("|---------------|-------|---------------|")
    for f in sorted(fold_df["fold"].unique()):
        n_val = int((fold_df["fold"] == f).sum())
        n_tr = int((fold_df["fold"] != f).sum())
        lines.append(f"| {f} | {n_val} | {n_tr} |")
    lines.append("")
    lines.append("### Critical implementation fact (PCA / aromatic scaler)")
    lines.append("")
    lines.append("In `nested_oof_fixed`, **PCA and aromatic StandardScaler are fit on the full outer train**")
    lines.append("(~129–130 samples), **not** on the inner-train subset.")
    lines.append("")
    lines.append("Inner `GroupKFold` is used only inside `select_ridge_alpha` / `select_svr_params`")
    lines.append("on the **already-built** concatenated design matrix (plus a fresh StandardScaler per inner fold).")
    lines.append("")
    lines.append("Because one outer fold is held out, remaining group count = 4, so")
    lines.append("`n_splits = min(5, n_unique_groups) = **4**` (not 5).")
    lines.append("")
    lines.append("| Quantity | Fit sample count in executed code |")
    lines.append("|----------|-----------------------------------|")
    lines.append("| Aromatic scaler (outer model) | outer train ≈ **129–130** |")
    lines.append("| ESM2 StandardScaler (outer model) | outer train ≈ **129–130** |")
    lines.append("| PCA basis (outer model) | outer train ≈ **129–130** |")
    lines.append("| Inner HP selection scaler on concat | inner train ≈ **96–98** |")
    lines.append("| Final Ridge scaler on concat | outer train ≈ **129–130** |")
    lines.append("")
    lines.append("### Typical inner sizes (example outer val fold=0, outer train=130, 4-fold GroupKFold)")
    lines.append("")
    lines.append("| Inner | N_inner_train | N_inner_val |")
    lines.append("|-------|---------------|-------------|")
    lines.append("| 0 | 97 | 33 |")
    lines.append("| 1 | 97 | 33 |")
    lines.append("| 2 | 98 | 32 |")
    lines.append("| 3 | 98 | 32 |")
    lines.append("")
    lines.append("### Direct answers")
    lines.append("")
    lines.append("1. Scalers/PCA for the scored OOF prediction use **~130** outer-train samples.")
    lines.append("2. Nested CV does **not** reduce PCA fitting to ~100; PCA stays on ~130. Nested CV only shrinks the **inner** concat-scaler / α-selection fits to ~96–98.")
    lines.append("3. At PCA fitting: raw ESM2 has **p=1280**, **n≈130** → **p ≫ n** yes.")
    lines.append("")
    (OUT / "PREPROCESSING_IMPLEMENTATION_AUDIT.md").write_text("\n".join(lines) + "\n")


def aromatic_stability(phys: pd.DataFrame, fold_df: pd.DataFrame, gen: str):
    cols = list(phys.columns)
    ids = [i for i in fold_df["id"] if i in phys.index]
    X = phys.loc[ids].to_numpy()
    groups = fold_df.set_index("id").loc[ids, "fold"].to_numpy()
    fold_stats = []
    scalers = {}  # fold_id -> (mean, scale) from StandardScaler on outer train excluding that fold as val
    for f in sorted(set(groups.tolist())):
        tr = np.where(groups != f)[0]
        Xtr, _ = fill_nan_train_median(X[tr], X[tr])
        sc = StandardScaler().fit(Xtr)
        scalers[f] = (sc.mean_.copy(), sc.scale_.copy())
        for j, name in enumerate(cols):
            v = Xtr[:, j]
            fold_stats.append(
                {
                    "generator": GEN_LABEL[gen],
                    "outer_val_fold": int(f),
                    "feature": name,
                    "n_train": int(len(tr)),
                    "mean": float(np.mean(v)),
                    "sd": float(np.std(v, ddof=0)),
                    "median": float(np.median(v)),
                    "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "skewness": float(stats.skew(v)),
                    "fraction_zero": float(np.mean(np.abs(v) < 1e-15)),
                }
            )
    fs = pd.DataFrame(fold_stats)

    # across-fold summaries
    rows = []
    for name in cols:
        sub = fs[fs["feature"] == name]
        means = sub["mean"].to_numpy()
        sds = sub["sd"].to_numpy()
        sd_pos = sds[sds > 0]
        rows.append(
            {
                "generator": GEN_LABEL[gen],
                "feature": name,
                "mean_range": float(means.max() - means.min()),
                "mean_sd": float(np.std(means, ddof=0)),
                "sd_cv": float(np.std(sds, ddof=0) / (np.mean(sds) + 1e-15)),
                "sd_max_min_ratio": float(sds.max() / sds.min()) if sds.min() > 0 else np.inf,
                "median_fraction_zero": float(sub["fraction_zero"].median()),
                "median_skew": float(sub["skewness"].median()),
                "median_sd": float(sub["sd"].median()),
            }
        )
    across = pd.DataFrame(rows)

    # sample-level z instability: for each sample i in fold f (as member of train for other folds' scalers)
    # For each antibody, collect z from every outer-train scaler that includes it (i.e., scalers for outer_val folds != sample's fold)
    abs_vars = {name: [] for name in cols}
    for i, aid in enumerate(ids):
        g_i = int(groups[i])
        zs_by_feat = {name: [] for name in cols}
        for f, (mu, scale) in scalers.items():
            if f == g_i:
                continue  # sample not in that outer train
            x = X[i]
            x = np.where(np.isfinite(x), x, np.nan)
            # use that train's median already baked into scaler via prior fill on train only;
            # for test-like apply: fill nonfinite with mu (approx)
            x = np.where(np.isfinite(x), x, mu)
            z = (x - mu) / (scale + 1e-15)
            for j, name in enumerate(cols):
                zs_by_feat[name].append(z[j])
        for name in cols:
            arr = np.asarray(zs_by_feat[name], float)
            abs_vars[name].append(float(np.max(arr) - np.min(arr)))

    zsum = []
    for name in cols:
        v = np.asarray(abs_vars[name], float)
        zsum.append(
            {
                "generator": GEN_LABEL[gen],
                "feature": name,
                "median_abs_z_variation": float(np.median(v)),
                "q90_abs_z_variation": float(np.quantile(v, 0.90)),
                "max_abs_z_variation": float(np.max(v)),
            }
        )
    zsum = pd.DataFrame(zsum)

    # flags
    flags = []
    for _, r in across.merge(zsum, on=["generator", "feature"]).iterrows():
        reasons = []
        if r["median_fraction_zero"] >= 0.50:
            reasons.append("rare_nonzero")
        if r["median_sd"] < 1e-8 or (np.isfinite(r["sd_max_min_ratio"]) and r["sd_max_min_ratio"] > 5):
            reasons.append("near_zero_variance_or_extreme_sd_ratio")
        if abs(r["median_skew"]) >= 2.0:
            reasons.append("heavy_tail")
        if r["sd_cv"] >= 0.25 or (np.isfinite(r["sd_max_min_ratio"]) and r["sd_max_min_ratio"] >= 2.0):
            reasons.append("fold_sd_unstable")
        if r["median_abs_z_variation"] >= 0.25:
            reasons.append("material_z_variation")
        if r["q90_abs_z_variation"] >= 0.75:
            reasons.append("severe_q90_z_variation")
        flags.append(
            {
                "generator": r["generator"],
                "feature": r["feature"],
                "flags": ";".join(reasons) if reasons else "none",
                "median_abs_z_variation": r["median_abs_z_variation"],
                "q90_abs_z_variation": r["q90_abs_z_variation"],
                "sd_cv": r["sd_cv"],
                "sd_max_min_ratio": r["sd_max_min_ratio"],
            }
        )
    return fs, across, zsum, pd.DataFrame(flags)


def pca_stability_for_method(X_plm, groups, n_comp, method: str):
    """method: scale_before | center_only"""
    bases = {}
    evar_rows = []
    score_stats = []
    for f in sorted(set(groups.tolist())):
        tr = np.where(groups != f)[0]
        Xtr = X_plm[tr]
        if method == "scale_before":
            sc, pca, scores = fit_pca_scale_before(Xtr, n_comp)
            components = pca.components_.T  # (d, k) in scaled space — compare in scaled feature space
            # For subspace comparison across folds, bases live in different scaled coordinate systems.
            # Map components back to raw space: component_raw ∝ component_scaled / scale
            # Or compare in a common space: use W = components.T / scale → directions in raw space.
            scale = sc.scale_
            W = (pca.components_ / scale).T  # (d, k) raw-space loadings (not orthonormal)
            bases[f] = W
        else:
            mean, pca, scores = fit_pca_center_only(Xtr, n_comp)
            bases[f] = pca.components_.T  # (d, k) in centered raw space; orthonormal in raw
        evr = pca.explained_variance_ratio_
        for i, (e, v) in enumerate(zip(evr, pca.explained_variance_), start=1):
            evar_rows.append(
                {
                    "method": method,
                    "n_comp_requested": n_comp,
                    "outer_val_fold": int(f),
                    "pc": i,
                    "explained_variance_ratio": float(e),
                    "eigenvalue": float(v),
                    "cumulative_explained_variance": float(np.sum(evr[:i])),
                }
            )
        score_stats.append(
            {
                "method": method,
                "n_comp": int(pca.n_components_),
                "outer_val_fold": int(f),
                "score_mean_abs": float(np.mean(np.abs(scores))),
                "score_std_mean": float(np.mean(np.std(scores, axis=0))),
                "total_var_explained": float(np.sum(evr)),
            }
        )

    pair_rows = []
    cos2_all = []
    min_cos2_all = []
    for f1, f2 in combinations(sorted(bases.keys()), 2):
        c2 = principal_angles_cos2(bases[f1], bases[f2])
        fro = projection_frobenius(bases[f1], bases[f2])
        # projection similarity: 1 - ||P-Q||_F^2 / (2k) roughly; also report mean cos2
        pair_rows.append(
            {
                "method": method,
                "n_comp_requested": n_comp,
                "fold_a": int(f1),
                "fold_b": int(f2),
                "mean_cos2": float(np.mean(c2)),
                "min_cos2": float(np.min(c2)),
                "projection_frobenius": fro,
            }
        )
        cos2_all.append(float(np.mean(c2)))
        min_cos2_all.append(float(np.min(c2)))
    summary = {
        "method": method,
        "n_comp_requested": n_comp,
        "pairwise_mean_of_mean_cos2": float(np.mean(cos2_all)),
        "pairwise_min_of_min_cos2": float(np.min(min_cos2_all)),
        "class": classify_pca(float(np.mean(cos2_all)), float(np.min(min_cos2_all))),
    }
    return pd.DataFrame(evar_rows), pd.DataFrame(pair_rows), pd.DataFrame(score_stats), summary


def select_ridge_alpha(Xtr, ytr, groups, alphas):
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    best_a, best_mae = alphas[0], np.inf
    for a in alphas:
        maes = []
        for tr, va in gkf.split(Xtr, ytr, groups):
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(Xtr[tr]), ytr[tr])
            maes.append(mae(ytr[va], m.predict(sc.transform(Xtr[va]))))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best_mae - 1e-15 or (abs(score - best_mae) <= 1e-15 and a > best_a):
            best_mae, best_a = score, a
    return best_a


def build_design(recipe: str, X_plm_tr, X_plm_te, Xp_tr, Xp_te, global_pca=None, global_arom_sc=None):
    """Return concatenated train/test designs."""
    Xp_tr, Xp_te = fill_nan_train_median(Xp_tr, Xp_te)

    if recipe == "P0_CURRENT":
        sc = StandardScaler()
        Ztr = sc.fit_transform(X_plm_tr)
        Zte = sc.transform(X_plm_te)
        n = min(32, Ztr.shape[0] - 1, Ztr.shape[1])
        pca = PCA(n_components=n, random_state=SEED)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        asp = StandardScaler()
        Sp_tr, Sp_te = asp.fit_transform(Xp_tr), asp.transform(Xp_te)
        return np.hstack([Ptr, Sp_tr]), np.hstack([Pte, Sp_te])

    if recipe in ("P1_CENTER_PCA32", "P2_CENTER_PCA64", "P4_AROMATIC_UNSCALED_DIAGNOSTIC"):
        n_comp = 64 if recipe == "P2_CENTER_PCA64" else 32
        mean = X_plm_tr.mean(axis=0)
        n = min(n_comp, X_plm_tr.shape[0] - 1, X_plm_tr.shape[1])
        pca = PCA(n_components=n, random_state=SEED)
        Ptr = pca.fit_transform(X_plm_tr - mean)
        Pte = pca.transform(X_plm_te - mean)
        scp = StandardScaler()
        Ptr, Pte = scp.fit_transform(Ptr), scp.transform(Pte)
        if recipe == "P4_AROMATIC_UNSCALED_DIAGNOSTIC":
            return np.hstack([Ptr, Xp_tr]), np.hstack([Pte, Xp_te])
        asp = StandardScaler()
        Sp_tr, Sp_te = asp.fit_transform(Xp_tr), asp.transform(Xp_te)
        return np.hstack([Ptr, Sp_tr]), np.hstack([Pte, Sp_te])

    if recipe == "P3_RAW_ESM2":
        sc = StandardScaler()
        Ptr, Pte = sc.fit_transform(X_plm_tr), sc.transform(X_plm_te)
        asp = StandardScaler()
        Sp_tr, Sp_te = asp.fit_transform(Xp_tr), asp.transform(Xp_te)
        return np.hstack([Ptr, Sp_tr]), np.hstack([Pte, Sp_te])

    if recipe == "P5_GLOBAL_DEV_PREPROCESSING_DIAGNOSTIC_ONLY":
        assert global_pca is not None and global_arom_sc is not None
        mean, pca, scp = global_pca
        Ptr = scp.transform(pca.transform(X_plm_tr - mean))
        Pte = scp.transform(pca.transform(X_plm_te - mean))
        Sp_tr = global_arom_sc.transform(Xp_tr)
        Sp_te = global_arom_sc.transform(Xp_te)
        return np.hstack([Ptr, Sp_tr]), np.hstack([Pte, Sp_te])

    raise ValueError(recipe)


def ridge_oof(recipe, X_plm, Xp, y, groups, alphas, fixed_alpha=None, global_pca=None, global_arom_sc=None):
    n = len(y)
    oof = np.full(n, np.nan)
    alphas_used = {}
    for f in sorted(set(groups.tolist())):
        te = np.where(groups == f)[0]
        tr = np.where(groups != f)[0]
        Xtr, Xte = build_design(
            recipe, X_plm[tr], X_plm[te], Xp[tr], Xp[te], global_pca=global_pca, global_arom_sc=global_arom_sc
        )
        if fixed_alpha is None:
            a = select_ridge_alpha(Xtr, y[tr], groups[tr], alphas)
        else:
            a = fixed_alpha
        alphas_used[int(f)] = float(a)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y[tr])
        oof[te] = m.predict(sc.transform(Xte))
    return oof, alphas_used


def plm_only_oof(X_plm, y, groups, alphas, mode="P0"):
    n = len(y)
    oof = np.full(n, np.nan)
    for f in sorted(set(groups.tolist())):
        te = np.where(groups == f)[0]
        tr = np.where(groups != f)[0]
        if mode == "P0":
            sc = StandardScaler()
            Ztr = sc.fit_transform(X_plm[tr])
            Zte = sc.transform(X_plm[te])
            pca = PCA(n_components=min(32, Ztr.shape[0] - 1), random_state=SEED)
            Xtr, Xte = pca.fit_transform(Ztr), pca.transform(Zte)
        elif mode == "P1":
            mean = X_plm[tr].mean(axis=0)
            pca = PCA(n_components=min(32, len(tr) - 1), random_state=SEED)
            Ptr = pca.fit_transform(X_plm[tr] - mean)
            Pte = pca.transform(X_plm[te] - mean)
            scp = StandardScaler()
            Xtr, Xte = scp.fit_transform(Ptr), scp.transform(Pte)
        else:
            raise ValueError(mode)
        a = select_ridge_alpha(Xtr, y[tr], groups[tr], alphas)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y[tr])
        oof[te] = m.predict(sc.transform(Xte))
    return oof


def kfold_fixed(X_plm, Xp, y, groups_for_align, k, alpha, recipe="P1_CENTER_PCA32"):
    """Supplementary: sklearn KFold k with fixed alpha; ignore group structure for split (diagnostic)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    train_ns = []
    fold_maes = []
    # Also PCA stability across these k folds
    bases = []
    for tr, te in kf.split(X_plm):
        train_ns.append(len(tr))
        Xtr, Xte = build_design(recipe, X_plm[tr], X_plm[te], Xp[tr], Xp[te])
        sc = StandardScaler()
        m = Ridge(alpha=alpha, random_state=0)
        m.fit(sc.fit_transform(Xtr), y[tr])
        pred = m.predict(sc.transform(Xte))
        oof[te] = pred
        fold_maes.append(mae(y[te], pred))
        # PCA basis for stability (center PCA32)
        mean = X_plm[tr].mean(axis=0)
        pca = PCA(n_components=min(32, len(tr) - 1), random_state=SEED)
        pca.fit(X_plm[tr] - mean)
        bases.append(pca.components_.T)
    pair_cos = []
    for i, j in combinations(range(len(bases)), 2):
        pair_cos.append(float(np.mean(principal_angles_cos2(bases[i], bases[j]))))
    return {
        "k": k,
        "mean_train_n": float(np.mean(train_ns)),
        "min_train_n": int(np.min(train_ns)),
        "CV_MAE": mae(y, oof),
        "fold_mae_std": float(np.std(fold_maes, ddof=0)),
        "pca_pairwise_mean_cos2": float(np.mean(pair_cos)),
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    fold_df = pd.read_csv(FOLDS_PATH)
    y_all = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_true"].astype(float)
    esm = load_esm2()
    canon = load_canonical()

    # align Dev
    ids = [i for i in fold_df["id"].tolist() if i in esm.index and i in y_all.index]
    fold_df = fold_df[fold_df["id"].isin(ids)].reset_index(drop=True)
    ids = fold_df["id"].tolist()
    groups = fold_df["fold"].to_numpy()
    X_plm = esm.loc[ids].to_numpy()
    y = y_all.loc[ids].to_numpy()

    write_implementation_audit(fold_df, ids)

    # --- Section 2: aromatic stability (all gens) ---
    all_fs, all_across, all_z, all_flags = [], [], [], []
    for gen in GENS:
        phys = load_physics(gen)
        fs, across, zsum, flags = aromatic_stability(phys, fold_df, gen)
        all_fs.append(fs)
        all_across.append(across)
        all_z.append(zsum)
        all_flags.append(flags)
    pd.concat(all_fs).to_csv(RES / "aromatic_fold_feature_stats.csv", index=False)
    pd.concat(all_across).to_csv(RES / "aromatic_across_fold_summary.csv", index=False)
    pd.concat(all_z).to_csv(RES / "aromatic_z_instability.csv", index=False)
    pd.concat(all_flags).to_csv(RES / "aromatic_instability_flags.csv", index=False)

    # --- Sections 3–4: PCA stability ---
    evar_all, pair_all, score_all, summaries = [], [], [], []
    for method in ["scale_before", "center_only"]:
        for n_comp in [32, 64]:
            evar, pair, scores, summary = pca_stability_for_method(X_plm, groups, n_comp, method)
            evar_all.append(evar)
            pair_all.append(pair)
            score_all.append(scores)
            summaries.append(summary)
            print("PCA", method, n_comp, summary, flush=True)
    pd.concat(evar_all).to_csv(RES / "pca_explained_variance.csv", index=False)
    pd.concat(pair_all).to_csv(RES / "pca_subspace_stability.csv", index=False)
    pd.concat(score_all).to_csv(RES / "pca_score_distribution_summary.csv", index=False)
    pd.DataFrame(summaries).to_csv(RES / "pca_stability_class_summary.csv", index=False)

    # Cross-method subspace: scale_before vs center_only on same outer train (fold0 held out)
    tr0 = np.where(groups != 0)[0]
    sc, pca_s, _ = fit_pca_scale_before(X_plm[tr0], 32)
    mean, pca_c, _ = fit_pca_center_only(X_plm[tr0], 32)
    W_s = (pca_s.components_ / sc.scale_).T
    W_c = pca_c.components_.T
    c2 = principal_angles_cos2(W_s, W_c)
    cross = {
        "mean_cos2_scale_before_vs_center_PCA32": float(np.mean(c2)),
        "min_cos2_scale_before_vs_center_PCA32": float(np.min(c2)),
        "frobenius": projection_frobenius(W_s, W_c),
    }
    (RES / "pca_scale_before_vs_center_same_fold.json").write_text(json.dumps(cross, indent=2) + "\n")
    print("cross method", cross, flush=True)

    # --- Section 6–7: controlled performance (HIC, AROMATIC, ESMFold) ---
    phys = load_physics("esmfold")
    Xp = phys.loc[ids].to_numpy()
    alphas_std = [0.1, 1.0, 10.0, 100.0]
    alphas_raw = [10.0, 100.0, 1000.0, 10000.0]

    # global Dev preprocessing for P5 (LEAKED)
    mean_g = X_plm.mean(axis=0)
    pca_g = PCA(n_components=min(32, len(ids) - 1), random_state=SEED)
    scores_g = pca_g.fit_transform(X_plm - mean_g)
    scp_g = StandardScaler().fit(scores_g)
    Xp_f, _ = fill_nan_train_median(Xp, Xp)
    arom_g = StandardScaler().fit(Xp_f)
    global_pca = (mean_g, pca_g, scp_g)

    perf_rows = []
    # PLM only baselines
    for mode, label in [("P0", "PLM_ONLY_P0style"), ("P1", "PLM_ONLY_P1style")]:
        oof = plm_only_oof(X_plm, y, groups, alphas_std, mode=mode)
        m = metrics(y, oof)
        perf_rows.append(
            {
                "recipe": label,
                "nested": True,
                "fixed_alpha": None,
                "CV_MAE": m["MAE"],
                "Pearson": m["Pearson"],
                "Spearman": m["Spearman"],
                "validity": "VALID_CV",
            }
        )

    p0_alphas = None
    for recipe in [
        "P0_CURRENT",
        "P1_CENTER_PCA32",
        "P2_CENTER_PCA64",
        "P3_RAW_ESM2",
        "P4_AROMATIC_UNSCALED_DIAGNOSTIC",
        "P5_GLOBAL_DEV_PREPROCESSING_DIAGNOSTIC_ONLY",
    ]:
        al = alphas_raw if recipe == "P3_RAW_ESM2" else alphas_std
        gp = global_pca if recipe.startswith("P5") else None
        ga = arom_g if recipe.startswith("P5") else None
        print("perf", recipe, flush=True)
        oof, aused = ridge_oof(recipe, X_plm, Xp, y, groups, al, global_pca=gp, global_arom_sc=ga)
        m = metrics(y, oof)
        validity = "INVALID_AS_CV_PERFORMANCE_EVIDENCE" if recipe.startswith("P5") else "VALID_CV"
        row = {
            "recipe": recipe,
            "nested": True,
            "fixed_alpha": None,
            "CV_MAE": m["MAE"],
            "Pearson": m["Pearson"],
            "Spearman": m["Spearman"],
            "alphas_outer": json.dumps(aused),
            "validity": validity,
            "delta_vs_PLM_ONLY_P0style": m["MAE"] - next(r["CV_MAE"] for r in perf_rows if r["recipe"] == "PLM_ONLY_P0style"),
            "delta_vs_PLM_ONLY_P1style": m["MAE"] - next(r["CV_MAE"] for r in perf_rows if r["recipe"] == "PLM_ONLY_P1style"),
        }
        perf_rows.append(row)
        if recipe == "P0_CURRENT":
            p0_alphas = aused

    # freeze fixed alpha = mode of P0 outer alphas (or median)
    alpha_vals = list(p0_alphas.values())
    # modal
    fixed_alpha = float(stats.mode(alpha_vals, keepdims=True).mode[0])
    freeze = json.loads((OUT / "STABILITY_AUDIT_THRESHOLDS.json").read_text())
    freeze["fixed_alpha_for_nested_vs_nonnested"] = {
        "alpha": fixed_alpha,
        "source": "mode of P0_CURRENT outer-fold selected Ridge alphas",
        "p0_outer_alphas": p0_alphas,
    }
    (OUT / "STABILITY_AUDIT_THRESHOLDS.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print("fixed_alpha", fixed_alpha, "from", p0_alphas, flush=True)

    # Section 7: fixed alpha nested-equivalent outer CV (no inner selection) for P0 and P1
    for recipe in ["P0_CURRENT", "P1_CENTER_PCA32", "P3_RAW_ESM2"]:
        al = alphas_raw if recipe == "P3_RAW_ESM2" else alphas_std
        # for P3 use fixed=1000 if not in p0; use fixed_alpha only for P0/P1
        fa = fixed_alpha if recipe != "P3_RAW_ESM2" else 1000.0
        oof, _ = ridge_oof(recipe, X_plm, Xp, y, groups, al, fixed_alpha=fa)
        m = metrics(y, oof)
        perf_rows.append(
            {
                "recipe": recipe + "_FIXED_ALPHA_OUTER",
                "nested": False,
                "fixed_alpha": fa,
                "CV_MAE": m["MAE"],
                "Pearson": m["Pearson"],
                "Spearman": m["Spearman"],
                "validity": "VALID_CV_FIXED_HP",
                "delta_vs_matched_nested": m["MAE"]
                - next(r["CV_MAE"] for r in perf_rows if r["recipe"] == recipe and r["nested"] is True),
            }
        )

    # Section 8 supplementary 5 vs 10 fold fixed
    supp = []
    for k in [5, 10]:
        supp.append(kfold_fixed(X_plm, Xp, y, groups, k, fixed_alpha, recipe="P1_CENTER_PCA32"))
    pd.DataFrame(supp).to_csv(RES / "supplementary_kfold_fixed_alpha.csv", index=False)

    pd.DataFrame(perf_rows).to_csv(RES / "controlled_preprocessing_performance.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
