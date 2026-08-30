#!/usr/bin/env python3
"""
Gate B5 P0/P1 core:
- B4 ensemble integrity audit (documented)
- Correct nested stacking for HIC & TmApp
- Re-evaluate B4 reference singles with rigorous Pearson summaries
- Residual complementarity
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import (  # noqa: E402
    B4_CONFIG,
    B4_METRICS,
    CONFIG,
    LOGS,
    METRICS,
    PREDS,
    REPORTS,
    MASTER_SEED,
    bootstrap_delta_mae,
    bootstrap_delta_pearson,
    bootstrap_pearson,
    ensure_dirs,
    extended_metrics,
    fisher_z_mean,
    group_kfold_labels,
    load_frozen_train,
    load_representation,
    pearson_fisher_ci,
    read_json,
    set_seeds,
    software_versions,
    write_json,
)
from b4_models import fit_predict_pca_head, fit_predict_sklearn  # noqa: E402
from b4_common import load_bio  # noqa: E402


# Fixed B4-validated params (no retuning)
EN_PARAMS = {"alpha": 0.05, "l1_ratio": 0.5}
RIDGE_ALPHA = 10.0


def fit_base(kind: str, Xtr, ytr, Xte, groups_tr=None, seed=MASTER_SEED):
    """kind encodes representation+model."""
    if kind.endswith("/Ridge"):
        return fit_predict_sklearn("Ridge", {"alpha": RIDGE_ALPHA}, Xtr, ytr, Xte, seed=seed)
    if kind.endswith("/ElasticNet"):
        return fit_predict_sklearn("ElasticNet", EN_PARAMS, Xtr, ytr, Xte, seed=seed)
    if "/PCA" in kind and kind.endswith("/SVR"):
        # e.g. PLM_ESM2/PCA64/SVR
        npc = int(kind.split("/PCA")[1].split("/")[0])
        params = {"n_components": npc, "C": 10.0, "epsilon": 0.1, "gamma": 0.01}
        return fit_predict_pca_head("SVR_RBF", params, Xtr, ytr, Xte, seed=seed)
    if kind.endswith("/CONST_MEDIAN"):
        return np.full(len(Xte), float(np.median(ytr)))
    raise ValueError(kind)


def rep_key(kind: str) -> str:
    if kind.startswith("CONST"):
        return "CONST"
    if "/PCA" in kind:
        return kind.split("/")[0]
    return kind.split("/")[0]


def get_X(cache, tag, ids, train_ids_for_bio=None):
    if tag == "CONST":
        return np.zeros((len(ids), 1))
    if tag == "BIO":
        # freeze categories on train
        return load_bio(ids, category_ids=train_ids_for_bio)
    if tag == "FUSION_ABLANG2_BIO":
        A = load_representation("PLM_ABLANG2", ids)
        B = load_bio(ids, category_ids=train_ids_for_bio)
        return np.concatenate([A, B], axis=1)
    if tag not in cache:
        cache[tag] = load_representation(tag, ids)
    # For BIO-containing we don't use cache the same way
    X = cache[tag]
    # Align if cache built on train only and ids differ — rebuild
    if X.shape[0] != len(ids):
        cache[tag] = load_representation(tag, ids)
        X = cache[tag]
    return X


def inner_oof_predictions(X, y, groups, seed, kind):
    """3-fold grouped OOF on the provided subset."""
    fid = group_kfold_labels(groups, 3, seed=seed)
    oof = np.full(len(y), np.nan)
    for f in range(3):
        te = fid == f
        tr = ~te
        oof[te] = fit_base(kind, X[tr], y[tr], X[te], groups[tr], seed=seed + f)
    return oof


def stack_fit_predict(method: str, M_tr, y_tr, M_te):
    mask = np.all(np.isfinite(M_tr), axis=1) & np.isfinite(y_tr)
    M_tr, y_tr = M_tr[mask], y_tr[mask]
    if method == "mean":
        return np.nanmean(M_te, axis=1), {"method": "mean"}
    if method == "median":
        return np.nanmedian(M_te, axis=1), {"method": "median"}
    if method == "nnls":
        w, _ = nnls(M_tr, y_tr)
        if w.sum() > 0:
            w = w / w.sum()
        return M_te @ w, {"method": "nnls", "weights": w.tolist()}
    if method == "ridge":
        m = Ridge(alpha=1.0, fit_intercept=True).fit(M_tr, y_tr)
        return m.predict(M_te), {"method": "ridge", "coef": m.coef_.tolist(), "intercept": float(m.intercept_)}
    if method == "nn_ridge":
        # nonnegative via NNLS on [M, ones] roughly; use Ridge then clip weights via NNLS on centered
        w, _ = nnls(np.column_stack([M_tr, np.ones(len(y_tr))]), y_tr)
        pred = M_te @ w[:-1] + w[-1]
        return pred, {"method": "nn_ridge", "weights": w.tolist()}
    raise ValueError(method)


def nested_stack(train, outer, target, base_kinds, stack_methods, Xcache, n_repeats=None):
    """Proper outer-CV stacking. Returns fold rows + OOF preds per method."""
    y = train[target].values.astype(float)
    groups = train["sequence_group"].values
    train_ids = list(train["id"])
    fold_infos = outer["folds"] if n_repeats is None else outer["folds"][:n_repeats]
    rows = []
    oof_store = {m: {rep: np.full(len(y), np.nan) for rep in range(len(fold_infos))} for m in stack_methods}
    base_oof_store = {k: {rep: np.full(len(y), np.nan) for rep in range(len(fold_infos))} for k in base_kinds}

    for fold_info in fold_infos:
        rep = fold_info["repeat"]
        fold_id = np.array(fold_info["fold_id"])
        for f in range(outer["n_folds"]):
            te = fold_id == f
            tr = ~te
            # Build inner-OOF for each base on OUTER TRAIN only
            inner_mats = []
            outer_mats = []
            for kind in base_kinds:
                tag = rep_key(kind)
                X_all = get_X(Xcache, tag, train_ids, train_ids_for_bio=train_ids)
                Xtr, ytr, gtr = X_all[tr], y[tr], groups[tr]
                Xte = X_all[te]
                inner = inner_oof_predictions(Xtr, ytr, gtr, seed=MASTER_SEED + 17 * rep + f, kind=kind)
                # Fit base on full outer train → outer val
                outer_pred = fit_base(kind, Xtr, ytr, Xte, gtr, seed=MASTER_SEED + 100 + f)
                inner_mats.append(inner)
                outer_mats.append(outer_pred)
                # also store nested single-model outer pred as base OOF for this rep
                base_oof_store[kind][rep][te] = outer_pred

            M_inner = np.column_stack(inner_mats)
            M_outer = np.column_stack(outer_mats)
            for method in stack_methods:
                pred, meta = stack_fit_predict(method, M_inner, y[tr], M_outer)
                met = extended_metrics(y[te], pred)
                rows.append(
                    {
                        "stage": "nested_stack",
                        "target": target,
                        "tag": f"NESTED_STACK_{method.upper()}",
                        "model": method,
                        "repeat": rep,
                        "fold": f,
                        "bases": "|".join(base_kinds),
                        **met,
                    }
                )
                oof_store[method][rep][te] = pred

            # Also score each base on this outer fold (rigorous nested single)
            for i, kind in enumerate(base_kinds):
                met = extended_metrics(y[te], outer_mats[i])
                rows.append(
                    {
                        "stage": "nested_single",
                        "target": target,
                        "tag": kind,
                        "model": "nested_refit",
                        "repeat": rep,
                        "fold": f,
                        **met,
                    }
                )

    return pd.DataFrame(rows), oof_store, base_oof_store


def eval_simple_cv(train, outer, target, kind, Xcache, n_repeats=3):
    """Standard outer CV for a single model (no stacking)."""
    y = train[target].values.astype(float)
    groups = train["sequence_group"].values
    train_ids = list(train["id"])
    tag = rep_key(kind)
    X = get_X(Xcache, tag, train_ids, train_ids_for_bio=train_ids)
    rows = []
    oofs = {}
    for fold_info in outer["folds"][:n_repeats]:
        rep = fold_info["repeat"]
        fold_id = np.array(fold_info["fold_id"])
        oof = np.full(len(y), np.nan)
        for f in range(outer["n_folds"]):
            te = fold_id == f
            tr = ~te
            if kind.endswith("/CONST_MEDIAN") or kind == "CONST_MEDIAN":
                pred = np.full(te.sum(), float(np.median(y[tr])))
            else:
                pred = fit_base(kind if "/" in kind else f"{kind}/Ridge", X[tr], y[tr], X[te], groups[tr])
            met = extended_metrics(y[te], pred)
            rows.append(
                {
                    "stage": "single_cv",
                    "target": target,
                    "tag": kind,
                    "model": "cv",
                    "repeat": rep,
                    "fold": f,
                    **met,
                }
            )
            oof[te] = pred
        oofs[rep] = oof
    return pd.DataFrame(rows), oofs


def pearson_summaries(name, target, y, oofs_by_rep, fold_rows: pd.DataFrame | None = None) -> dict:
    """Build fold / repeat / aggregated Pearson summaries."""
    out = {"name": name, "target": target}
    # fold-level from rows if available
    if fold_rows is not None and len(fold_rows):
        rs = fold_rows["pearson"].values
        out.update(
            {
                "fold_pearson_mean": float(np.nanmean(rs)),
                "fold_pearson_median": float(np.nanmedian(rs)),
                "fold_pearson_sd": float(np.nanstd(rs)),
                "fold_pearson_min": float(np.nanmin(rs)),
                "fold_pearson_max": float(np.nanmax(rs)),
                "fold_pearson_fisher_z_mean": fisher_z_mean(rs),
                "fold_mae_mean": float(fold_rows["mae"].mean()),
                "fold_rmse_mean": float(fold_rows["rmse"].mean()),
                "fold_spearman_mean": float(fold_rows["spearman"].mean()),
            }
        )
    # repeat-level pooled OOF
    reps = []
    mats = []
    for rep, oof in sorted(oofs_by_rep.items()):
        if np.isfinite(oof).sum() < 10:
            continue
        r = pearsonr(y[np.isfinite(oof)], oof[np.isfinite(oof)])[0]
        reps.append({"repeat": rep, "pearson": float(r), "mae": float(np.mean(np.abs(y - oof)))})
        mats.append(oof)
    if reps:
        out["repeat_pearsons"] = [r["pearson"] for r in reps]
        out["repeat_pearson_mean"] = float(np.mean([r["pearson"] for r in reps]))
        out["repeat_pearson_sd"] = float(np.std([r["pearson"] for r in reps]))
        out["repeat_mae_mean"] = float(np.mean([r["mae"] for r in reps]))
    if mats:
        mean_oof = np.nanmean(np.column_stack(mats), axis=1)
        r_agg = pearsonr(y, mean_oof)[0]
        out["aggregated_repeated_oof_pearson"] = float(r_agg)
        out["aggregated_repeated_oof_mae"] = float(np.mean(np.abs(y - mean_oof)))
        ci = bootstrap_pearson(y, mean_oof, seed=MASTER_SEED)
        out["agg_pearson_ci_lo"] = ci["ci_lo"]
        out["agg_pearson_ci_hi"] = ci["ci_hi"]
        met = extended_metrics(y, mean_oof)
        out["agg_rmse"] = met["rmse"]
        out["agg_spearman"] = met["spearman"]
        out["agg_cal_slope"] = met["cal_slope"]
        out["agg_cal_intercept"] = met["cal_intercept"]
        out["agg_pred_sd"] = met["pred_sd"]
        out["agg_obs_sd"] = met["obs_sd"]
        out["agg_ccc"] = met["ccc"]
        out["mean_oof_pred"] = mean_oof
    return out


def write_audit_report():
    text = """# Ensemble CV integrity audit (Gate B5)

## Verdict

**B4 OOF ensemble CV estimates were OPTIMISTIC (biased high for performance / low MAE).**

They must **not** be treated as unbiased nested-CV ceilings.

## What B4 did

In `gate_b4_absolute/scripts/02_nested_finalists_oneshot.py` → `build_residuals_ensembles`:

1. Base-model OOF predictions were generated (Train-only; legitimate Level-1 OOF).
2. Stackers were fit on **ALL** OOF rows + **ALL** Train labels:
   - `nnls(M[mask], y[mask])` for `ENSEMBLE_NNLS`
   - `Ridge.fit(M[mask], y[mask])` for `ENSEMBLE_RIDGE`
3. Stacker predictions `pred = M @ w` (or Ridge) were then scored by slicing those
   **same** OOF rows into outer folds.

Although Level-1 base predictions are out-of-fold, the **Level-2 stacker is
evaluated in-sample** on the meta-training rows.

Mean/median ensembles of OOF predictions are less biased (no fitted stacker),
but NNLS/Ridge stack scores reported in B4 (~0.469 HIC, ~2.615 TmApp) are
**not** valid outer-CV estimates.

## Correct protocol (this Gate)

For each OUTER fold:

```
OUTER TRAIN
  → inner grouped OOF for each base (OUTER TRAIN only)
  → fit stacker on inner-OOF
  → refit bases on OUTER TRAIN
  → predict OUTER VALIDATION
  → apply frozen stacker
  → score OUTER VALIDATION
```

Outer validation never influences base HPs, membership, weights, or calibration.

## B4 reported (optimistic) values

| Target | Model | Reported CV MAE | Reported CV Pearson |
|--------|-------|----------------:|--------------------:|
| HIC | ENSEMBLE_NNLS/oof_nnls | ≈ 0.469 | ≈ 0.548 |
| TmApp | ENSEMBLE_RIDGE/oof_ridge | ≈ 2.615 | ≈ 0.596 |

Corrected nested values are in `corrected_nested_ensemble_results.md` and
`metrics/corrected_nested_ensemble.csv`.
"""
    (REPORTS / "ensemble_cv_integrity_audit.md").write_text(text)


def main():
    ensure_dirs()
    set_seeds(MASTER_SEED)
    t0 = time.time()
    write_audit_report()

    train, outer = load_frozen_train()
    # verify B3 hashes
    decl = read_json(B4_CONFIG / "TRAIN_ONLY_SEARCH_DECLARATION.json")
    from b5_common import sha256_lines

    assert sha256_lines(train["id"]) == decl["frozen_Train_IDs_sha256"]

    write_json(
        CONFIG / "B5_TRAIN_ONLY_DECLARATION.json",
        {
            "gate": "B5_CEILING",
            "primary_metric": "MAE",
            "secondary_priority": "prediction Pearson r",
            "frozen_Train_IDs_sha256": decl["frozen_Train_IDs_sha256"],
            "frozen_Public_IDs_sha256": decl["frozen_Public_IDs_sha256"],
            "frozen_Private_IDs_sha256": decl["frozen_Private_IDs_sha256"],
            "b4_ensemble_integrity": "OPTIMISTIC_LEVEL2_INSAMPLE",
            "software": software_versions(),
            "gbdt_learning_rate_fixed": 0.03,
        },
    )

    Xcache = {}
    all_rows = []
    pearson_fold_rows = []
    pearson_repeat_rows = []
    pearson_ci_rows = []
    paired_rows = []
    residual_rows = []
    calib_rows = []
    summaries = []

    # ---- HIC ----
    hic_bases = [
        "ESMFN_STRUCTURE/ElasticNet",
        "PLM_ESM2/PCA64/SVR",
        "PLM_ESM2/ElasticNet",
        "FUSION_ESM2_ESMFN/ElasticNet",
        "SEQ_SIMPLE/Ridge",
    ]
    hic_stack_bases = [
        "ESMFN_STRUCTURE/ElasticNet",
        "PLM_ESM2/PCA64/SVR",
        "SEQ_SIMPLE/Ridge",
    ]

    print("=== HIC nested stack ===", flush=True)
    hic_nest, hic_oof_stack, hic_base_oof = nested_stack(
        train, outer, "HIC", hic_stack_bases, ["mean", "median", "nnls", "ridge"], Xcache, n_repeats=3
    )
    all_rows.append(hic_nest)

    # Additional singles with full 3 repeats
    for kind in ["CONST_MEDIAN"] + hic_bases:
        k = kind if kind != "CONST_MEDIAN" else "CONST/CONST_MEDIAN"
        print(f"HIC single {k}", flush=True)
        if kind == "CONST_MEDIAN":
            # manual
            y = train["HIC"].values.astype(float)
            rows = []
            oofs = {}
            for fold_info in outer["folds"][:3]:
                rep = fold_info["repeat"]
                fold_id = np.array(fold_info["fold_id"])
                oof = np.full(len(y), np.nan)
                for f in range(5):
                    te = fold_id == f
                    tr = ~te
                    pred = np.full(te.sum(), float(np.median(y[tr])))
                    met = extended_metrics(y[te], pred)
                    rows.append({"stage": "single_cv", "target": "HIC", "tag": "CONST_MEDIAN", "model": "cv", "repeat": rep, "fold": f, **met})
                    oof[te] = pred
                oofs[rep] = oof
            df_r = pd.DataFrame(rows)
        else:
            df_r, oofs = eval_simple_cv(train, outer, "HIC", kind, Xcache, n_repeats=3)
        all_rows.append(df_r)
        summ = pearson_summaries(kind if kind != "CONST_MEDIAN" else "CONST_MEDIAN", "HIC", train["HIC"].values, oofs, df_r)
        if "mean_oof_pred" in summ:
            np.save(PREDS / "oof" / f"HIC__{kind.replace('/','_')}__meanOOF.npy", summ.pop("mean_oof_pred"))
        summaries.append(summ)

    # Stack summaries
    y_h = train["HIC"].values.astype(float)
    for method, byrep in hic_oof_stack.items():
        tag = f"NESTED_STACK_{method.upper()}"
        sub = hic_nest[(hic_nest.tag == tag)]
        summ = pearson_summaries(tag, "HIC", y_h, byrep, sub)
        if "mean_oof_pred" in summ:
            np.save(PREDS / "oof" / f"HIC__{tag}__meanOOF.npy", summ.pop("mean_oof_pred"))
        summaries.append(summ)

    # ---- TmApp ----
    tm_bases = [
        "BIO/Ridge",
        "GERMLINE_REL/ElasticNet",
        "PLM_ABLANG2/ElasticNet",
        "PLM_ABLANG2/PCA32/SVR",
        "FUSION_ABLANG2_BIO/ElasticNet",
        "SEQ_SIMPLE/Ridge",
        "IMGT_POS_HL/Ridge",
    ]
    tm_stack_bases = [
        "BIO/Ridge",
        "PLM_ABLANG2/ElasticNet",
        "FUSION_ABLANG2_BIO/ElasticNet",
    ]

    print("=== TmApp nested stack ===", flush=True)
    tm_nest, tm_oof_stack, tm_base_oof = nested_stack(
        train, outer, "TmApp", tm_stack_bases, ["mean", "median", "nnls", "ridge"], Xcache, n_repeats=3
    )
    all_rows.append(tm_nest)

    for kind in ["CONST_MEDIAN"] + tm_bases:
        print(f"TmApp single {kind}", flush=True)
        if kind == "CONST_MEDIAN":
            y = train["TmApp"].values.astype(float)
            rows = []
            oofs = {}
            for fold_info in outer["folds"][:3]:
                rep = fold_info["repeat"]
                fold_id = np.array(fold_info["fold_id"])
                oof = np.full(len(y), np.nan)
                for f in range(5):
                    te = fold_id == f
                    tr = ~te
                    pred = np.full(te.sum(), float(np.median(y[tr])))
                    met = extended_metrics(y[te], pred)
                    rows.append({"stage": "single_cv", "target": "TmApp", "tag": "CONST_MEDIAN", "model": "cv", "repeat": rep, "fold": f, **met})
                    oof[te] = pred
                oofs[rep] = oof
            df_r = pd.DataFrame(rows)
        else:
            df_r, oofs = eval_simple_cv(train, outer, "TmApp", kind, Xcache, n_repeats=3)
        all_rows.append(df_r)
        summ = pearson_summaries(kind, "TmApp", train["TmApp"].values, oofs, df_r)
        if "mean_oof_pred" in summ:
            np.save(PREDS / "oof" / f"TmApp__{kind.replace('/','_')}__meanOOF.npy", summ.pop("mean_oof_pred"))
        summaries.append(summ)

    y_t = train["TmApp"].values.astype(float)
    for method, byrep in tm_oof_stack.items():
        tag = f"NESTED_STACK_{method.upper()}"
        sub = tm_nest[(tm_nest.tag == tag)]
        summ = pearson_summaries(tag, "TmApp", y_t, byrep, sub)
        if "mean_oof_pred" in summ:
            np.save(PREDS / "oof" / f"TmApp__{tag}__meanOOF.npy", summ.pop("mean_oof_pred"))
        summaries.append(summ)

    # ---- Residual complementarity (rep0 base OOF from nested) ----
    for target, base_oof, pairs in [
        ("HIC", hic_base_oof, [("ESMFN_STRUCTURE/ElasticNet", "PLM_ESM2/PCA64/SVR")]),
        ("TmApp", tm_base_oof, [("BIO/Ridge", "PLM_ABLANG2/ElasticNet"), ("GERMLINE_REL/ElasticNet", "PLM_ABLANG2/ElasticNet")]),
    ]:
        y = train[target].values.astype(float)
        for a, b in pairs:
            if a not in base_oof or b not in base_oof:
                continue
            pa = base_oof[a][0]
            pb = base_oof[b][0]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            ra, rb = y - pa, y - pb
            # can B predict residuals of A?
            # use simple ridge on features of B to predict ra via CV — use pb as 1-d proxy (prediction complementarity)
            r_resid = pearsonr(ra, rb)[0]
            # residual of A explained by pred B: corr(ra, pb)
            r_ra_pb = pearsonr(ra, pb)[0]
            residual_rows.append(
                {
                    "target": target,
                    "model_a": a,
                    "model_b": b,
                    "corr_residuals": float(r_resid),
                    "corr_residA_vs_predB": float(r_ra_pb),
                    "mae_a": float(np.mean(np.abs(ra))),
                    "mae_b": float(np.mean(np.abs(rb))),
                }
            )

    # ---- Paired comparisons (aggregated OOF) ----
    def load_mean_oof(target, tag):
        p = PREDS / "oof" / f"{target}__{tag.replace('/','_')}__meanOOF.npy"
        return np.load(p) if p.exists() else None

    for target, pairs in [
        ("HIC", [
            ("NESTED_STACK_NNLS", "ESMFN_STRUCTURE/ElasticNet"),
            ("NESTED_STACK_RIDGE", "ESMFN_STRUCTURE/ElasticNet"),
            ("ESMFN_STRUCTURE/ElasticNet", "PLM_ESM2/PCA64/SVR"),
            ("NESTED_STACK_NNLS", "PLM_ESM2/PCA64/SVR"),
        ]),
        ("TmApp", [
            ("NESTED_STACK_RIDGE", "FUSION_ABLANG2_BIO/ElasticNet"),
            ("NESTED_STACK_NNLS", "FUSION_ABLANG2_BIO/ElasticNet"),
            ("FUSION_ABLANG2_BIO/ElasticNet", "BIO/Ridge"),
            ("FUSION_ABLANG2_BIO/ElasticNet", "PLM_ABLANG2/ElasticNet"),
            ("PLM_ABLANG2/ElasticNet", "BIO/Ridge"),
        ]),
    ]:
        y = train[target].values.astype(float)
        for a, b in pairs:
            pa, pb = load_mean_oof(target, a), load_mean_oof(target, b)
            if pa is None or pb is None:
                continue
            dm = bootstrap_delta_mae(y, pa, pb)
            dp = bootstrap_delta_pearson(y, pa, pb)
            paired_rows.append({"target": target, "model_a": a, "model_b": b, **dm, **dp})

    # ---- Persist ----
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(METRICS / "all_ceiling_cv_results.csv", index=False)

    # corrected ensemble table
    corr_rows = []
    b4_old = {
        ("HIC", "ENSEMBLE_NNLS"): {"mae": 0.4691, "pearson": 0.5479, "rmse": 0.6807, "spearman": 0.5268},
        ("TmApp", "ENSEMBLE_RIDGE"): {"mae": 2.6148, "pearson": 0.5956, "rmse": 3.4576, "spearman": 0.5933},
    }
    for s in summaries:
        if not str(s["name"]).startswith("NESTED_STACK"):
            continue
        target = s["target"]
        old_key = ("HIC", "ENSEMBLE_NNLS") if target == "HIC" and "NNLS" in s["name"] else None
        if target == "TmApp" and "RIDGE" in s["name"]:
            old_key = ("TmApp", "ENSEMBLE_RIDGE")
        if target == "HIC" and "RIDGE" in s["name"]:
            old_key = None
        row = {
            "target": target,
            "model": s["name"],
            "corrected_mae": s.get("fold_mae_mean", s.get("aggregated_repeated_oof_mae")),
            "corrected_pearson_fisher_z": s.get("fold_pearson_fisher_z_mean"),
            "corrected_pearson_agg": s.get("aggregated_repeated_oof_pearson"),
            "corrected_rmse": s.get("fold_rmse_mean", s.get("agg_rmse")),
            "corrected_spearman": s.get("fold_spearman_mean", s.get("agg_spearman")),
        }
        if old_key and old_key in b4_old:
            row["b4_reported_mae"] = b4_old[old_key]["mae"]
            row["b4_reported_pearson"] = b4_old[old_key]["pearson"]
            row["delta_mae_corrected_minus_b4"] = row["corrected_mae"] - b4_old[old_key]["mae"]
            row["delta_pearson_corrected_minus_b4"] = (row["corrected_pearson_agg"] or np.nan) - b4_old[old_key]["pearson"]
        corr_rows.append(row)
    pd.DataFrame(corr_rows).to_csv(METRICS / "corrected_nested_ensemble.csv", index=False)

    # foldwise / repeatwise pearson
    fw = all_df[["target", "tag", "model", "repeat", "fold", "pearson", "mae", "rmse", "spearman"]].copy()
    fw.to_csv(METRICS / "foldwise_pearson.csv", index=False)
    rr = []
    for s in summaries:
        for i, r in enumerate(s.get("repeat_pearsons", []) or []):
            rr.append({"target": s["target"], "name": s["name"], "repeat": i, "pearson": r})
    pd.DataFrame(rr).to_csv(METRICS / "repeatwise_pearson.csv", index=False)

    for s in summaries:
        pearson_ci_rows.append(
            {
                "target": s["target"],
                "name": s["name"],
                "agg_pearson": s.get("aggregated_repeated_oof_pearson"),
                "ci_lo": s.get("agg_pearson_ci_lo"),
                "ci_hi": s.get("agg_pearson_ci_hi"),
                "fisher_z_fold_mean": s.get("fold_pearson_fisher_z_mean"),
                "agg_mae": s.get("aggregated_repeated_oof_mae"),
            }
        )
        calib_rows.append(
            {
                "target": s["target"],
                "name": s["name"],
                "cal_slope": s.get("agg_cal_slope"),
                "cal_intercept": s.get("agg_cal_intercept"),
                "pred_sd": s.get("agg_pred_sd"),
                "obs_sd": s.get("agg_obs_sd"),
                "mae": s.get("aggregated_repeated_oof_mae"),
                "pearson": s.get("aggregated_repeated_oof_pearson"),
                "ccc": s.get("agg_ccc"),
            }
        )

    pd.DataFrame(pearson_ci_rows).to_csv(METRICS / "pearson_confidence_intervals.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(METRICS / "calibration_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(METRICS / "paired_model_comparisons.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(METRICS / "residual_complementarity.csv", index=False)
    # drop huge arrays from summaries json
    write_json(METRICS / "pearson_summaries.json", [{k: v for k, v in s.items() if k != "mean_oof_pred"} for s in summaries])

    # Reports
    corr_df = pd.DataFrame(corr_rows)
    (REPORTS / "corrected_nested_ensemble_results.md").write_text(
        "# Corrected nested ensemble results\n\n"
        + corr_df.to_markdown(index=False)
        + "\n\nPositive `delta_mae_corrected_minus_b4` means B4 was optimistic (reported MAE too low).\n"
    )
    (REPORTS / "pearson_protocol.md").write_text(
        """# Pearson evaluation protocol (Gate B5)

## Definitions

- **prediction Pearson r**: Pearson correlation between predicted and measured assay values across antibodies.
- **model-ranking transfer rho**: Spearman/Kendall of *model ranks* across CV/Public/Private (separate diagnostic).

## Summaries computed

1. **Fold-level Pearson** — per outer validation fold; report mean/median/SD/min/max and **Fisher-z mean**.
2. **Repeat-level pooled OOF Pearson** — one OOF vector of length 162 per repeat.
3. **Aggregated repeated-OOF Pearson** — mean OOF across repeats, then one Pearson on Train=162 (descriptive).

## Uncertainty

Fisher-z CI and/or bootstrap (≥5000) on aggregated OOF and holdouts.

## Primary objective

Production selection remains **minimize MAE**. Pearson is a high-priority secondary diagnostic.
"""
    )
    summ_df = pd.DataFrame([{k: v for k, v in s.items() if not isinstance(v, (list, np.ndarray))} for s in summaries])
    (REPORTS / "pearson_ceiling_analysis.md").write_text(
        "# Pearson ceiling analysis\n\n" + summ_df.sort_values(["target", "aggregated_repeated_oof_mae"]).to_markdown(index=False) + "\n"
    )
    (REPORTS / "residual_complementarity.md").write_text(
        "# Residual complementarity\n\n" + pd.DataFrame(residual_rows).to_markdown(index=False) + "\n"
    )
    (REPORTS / "calibration_analysis.md").write_text(
        "# Calibration analysis\n\nConvention: `observed = intercept + slope * predicted`.\n\n"
        + pd.DataFrame(calib_rows).to_markdown(index=False)
        + "\n\nUnder-dispersion: `pred_sd << obs_sd` with slope > 1 often indicates mean-shrinkage.\n"
    )
    (REPORTS / "structure_plm_nested_stack.md").write_text(
        "# HIC PLM + structure nested stacking\n\nBases: ESMFN_STRUCTURE/ElasticNet, PLM_ESM2/PCA64/SVR, SEQ_SIMPLE/Ridge.\n\n"
        + hic_nest[hic_nest.stage == "nested_stack"].groupby(["tag", "model"])[["mae", "pearson", "rmse", "spearman"]]
        .mean()
        .reset_index()
        .to_markdown(index=False)
        + "\n"
    )
    (REPORTS / "bio_plm_nested_stack.md").write_text(
        "# TmApp BIO + PLM nested stacking\n\nBases: BIO/Ridge, PLM_ABLANG2/ElasticNet, FUSION_ABLANG2_BIO/ElasticNet.\n\n"
        + tm_nest[tm_nest.stage == "nested_stack"].groupby(["tag", "model"])[["mae", "pearson", "rmse", "spearman"]]
        .mean()
        .reset_index()
        .to_markdown(index=False)
        + "\n"
    )

    write_json(LOGS / "01_nested_done.json", {"elapsed_s": time.time() - t0, "n_rows": len(all_df)})
    print(f"DONE nested/pearson in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
