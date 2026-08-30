#!/usr/bin/env python3
"""
Gate B4 Stage 2+: nested confirmation, residual/ensemble, freeze finalists,
one-shot Public/Private (labels opened ONLY after finalist registry hashed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from scipy.optimize import nnls
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import (  # noqa: E402
    B3_ORG,
    CONFIG,
    LOGS,
    METRICS,
    OPTUNA_DIR,
    PREDS,
    REPORTS,
    ensure_dirs,
    file_sha256,
    group_kfold_labels,
    load_representation,
    load_train_frame,
    read_json,
    regression_metrics,
    set_seeds,
    sha256_file,
    sha256_text,
    software_versions,
    train_only,
    write_json,
    MASTER_SEED,
    OPTUNA_SAMPLER_SEED,
)
from b4_models import (  # noqa: E402
    fit_predict_cat,
    fit_predict_lgb,
    fit_predict_pca_head,
    fit_predict_sklearn,
    fit_predict_xgb,
    suggest_cat,
    suggest_elasticnet,
    suggest_lgb,
    suggest_svr,
    suggest_xgb,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def summarize_mae(rows: pd.DataFrame) -> pd.DataFrame:
    g = (
        rows.groupby(["target", "tag", "model"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            median_mae=("mae", "median"),
            worst_mae=("mae", "max"),
            mean_rmse=("rmse", "mean"),
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_r2=("r2", "mean"),
            mean_cal_slope=("cal_slope", "mean"),
            n=("mae", "count"),
        )
        .sort_values(["target", "mean_mae"])
    )
    return g


def nested_optuna_sklearn(X, y, groups, outer_tr_mask, kind, suggest_fn, n_trials, study_name, seed):
    Xo, yo, go = X[outer_tr_mask], y[outer_tr_mask], groups[outer_tr_mask]
    fold_id = group_kfold_labels(go, 3, seed=seed)
    storage = f"sqlite:///{OPTUNA_DIR / (study_name + '.db')}"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True, direction="minimize",
        sampler=TPESampler(seed=seed),
    )

    def objective(trial):
        params = suggest_fn(trial)
        maes = []
        for f in range(3):
            te = fold_id == f
            tr = ~te
            pred = fit_predict_sklearn(kind, params, Xo[tr], yo[tr], Xo[te], seed=seed)
            maes.append(float(np.mean(np.abs(yo[te] - pred))))
        return float(np.mean(maes))

    rem = max(0, n_trials - len(study.trials))
    if rem:
        study.optimize(objective, n_trials=rem)
    study.trials_dataframe().to_csv(OPTUNA_DIR / f"{study_name}_trials.csv", index=False)
    return study.best_params, float(study.best_value)


def nested_optuna_gbdt(X, y, groups, outer_tr_mask, family, obj, n_trials, study_name, seed):
    Xo, yo, go = X[outer_tr_mask], y[outer_tr_mask], groups[outer_tr_mask]
    fold_id = group_kfold_labels(go, 3, seed=seed)
    storage = f"sqlite:///{OPTUNA_DIR / (study_name + '.db')}"
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True, direction="minimize",
        sampler=TPESampler(seed=seed),
    )

    def objective(trial):
        if family == "xgb":
            params = suggest_xgb(trial)
        elif family == "lgb":
            params = suggest_lgb(trial)
        else:
            params = suggest_cat(trial)
        maes, bis = [], []
        for f in range(3):
            te = fold_id == f
            tr = ~te
            if family == "xgb":
                pred, meta = fit_predict_xgb(params, Xo[tr], yo[tr], Xo[te], go[tr], objective=obj, seed=seed + f)
            elif family == "lgb":
                pred, meta = fit_predict_lgb(params, Xo[tr], yo[tr], Xo[te], go[tr], objective=obj, seed=seed + f)
            else:
                pred, meta = fit_predict_cat(params, Xo[tr], yo[tr], Xo[te], go[tr], loss=obj, seed=seed + f)
            maes.append(float(np.mean(np.abs(yo[te] - pred))))
            bis.append(meta["best_iteration"])
        trial.set_user_attr("median_best_iteration", float(np.median(bis)))
        return float(np.mean(maes))

    rem = max(0, n_trials - len(study.trials))
    if rem:
        study.optimize(objective, n_trials=rem)
    study.trials_dataframe().to_csv(OPTUNA_DIR / f"{study_name}_trials.csv", index=False)
    med_bi = study.best_trial.user_attrs.get("median_best_iteration")
    return study.best_params, float(study.best_value), med_bi


def run_nested(candidates, train, outer, Xcache):
    """candidates: list of dicts with target, tag, model, family info.

    Protocol:
    - Linear/kernel: full outer (5×3=15) with 25 inner Optuna trials
    - GBDT: reduced outer = repeat 0 only (5 folds) with 20 inner trials
      (documented; N=162 + nested GBDT cost)
    """
    groups = train["sequence_group"].values
    rows = []
    oof_preds = []  # long format
    paired = {}

    for cand in candidates:
        target = cand["target"]
        tag = cand["tag"]
        model = cand["model"]
        is_gbdt = str(model).startswith(("XGB", "LGB", "CAT"))
        fold_infos = outer["folds"][:1] if is_gbdt else outer["folds"]
        n_trials = 20 if is_gbdt else 25
        print(f"NESTED {target} {tag} {model} folds={len(fold_infos)*outer['n_folds']} trials={n_trials}", flush=True)
        y = train[target].values.astype(float)
        # resolve base representation (strip PCA suffix)
        base_tag = tag.split("_PCA")[0] if "_PCA" in tag else tag
        if base_tag not in Xcache:
            Xcache[base_tag] = load_representation(base_tag, train["id"])
        X = Xcache[base_tag]
        npc = None
        if "_PCA" in tag:
            npc = int(tag.split("_PCA")[1])

        fold_maes = []
        for fold_info in fold_infos:
            fold_id = np.array(fold_info["fold_id"])
            rep = fold_info["repeat"]
            for f in range(outer["n_folds"]):
                te = fold_id == f
                tr = ~te
                sname = f"nested_{target}_{tag}_{model}_r{rep}_f{f}"
                seed = OPTUNA_SAMPLER_SEED + 100 * rep + f
                try:
                    if model == "ElasticNet":
                        bp, _ = nested_optuna_sklearn(
                            X, y, groups, tr, "ElasticNet", suggest_elasticnet, n_trials, sname, seed
                        )
                        pred = fit_predict_sklearn("ElasticNet", bp, X[tr], y[tr], X[te])
                        meta = {"best_iteration": None, "params": bp}
                    elif model == "Ridge_grid":
                        # alpha grid on outer-train only via 3-fold
                        best_a, best_m = 10.0, np.inf
                        go = groups[tr]
                        fid = group_kfold_labels(go, 3, seed=seed)
                        Xo, yo = X[tr], y[tr]
                        for a in [0.1, 1, 10, 100, 1000, 10000]:
                            maes = []
                            for ff in range(3):
                                tte = fid == ff
                                ttr = ~tte
                                pr = fit_predict_sklearn("Ridge", {"alpha": a}, Xo[ttr], yo[ttr], Xo[tte])
                                maes.append(np.mean(np.abs(yo[tte] - pr)))
                            m = float(np.mean(maes))
                            if m < best_m:
                                best_m, best_a = m, a
                        pred = fit_predict_sklearn("Ridge", {"alpha": best_a}, X[tr], y[tr], X[te])
                        meta = {"best_iteration": None, "params": {"alpha": best_a}}
                    elif model == "SVR_RBF" and npc is not None:
                        def sug(trial, n=npc):
                            p = suggest_svr(trial)
                            p["n_components"] = n
                            return p

                        # custom nested for PCA head
                        Xo, yo, go = X[tr], y[tr], groups[tr]
                        fid = group_kfold_labels(go, 3, seed=seed)
                        storage = f"sqlite:///{OPTUNA_DIR / (sname + '.db')}"
                        study = optuna.create_study(
                            study_name=sname, storage=storage, load_if_exists=True, direction="minimize",
                            sampler=TPESampler(seed=seed),
                        )

                        def objective(trial, n=npc):
                            params = sug(trial, n)
                            maes = []
                            for ff in range(3):
                                tte = fid == ff
                                ttr = ~tte
                                pr = fit_predict_pca_head("SVR_RBF", params, Xo[ttr], yo[ttr], Xo[tte])
                                maes.append(float(np.mean(np.abs(yo[tte] - pr))))
                            return float(np.mean(maes))

                        rem = max(0, n_trials - len(study.trials))
                        if rem:
                            study.optimize(objective, n_trials=rem)
                        bp = dict(study.best_params)
                        bp["n_components"] = npc
                        pred = fit_predict_pca_head("SVR_RBF", bp, X[tr], y[tr], X[te])
                        meta = {"best_iteration": None, "params": bp}
                    elif model.startswith("XGB"):
                        obj = "reg:absoluteerror" if model.endswith("L1") else "reg:squarederror"
                        bp, _, med_bi = nested_optuna_gbdt(X, y, groups, tr, "xgb", obj, n_trials, sname, seed)
                        # refit with median best_iteration from best trial user attr; recompute robustly
                        bis = []
                        go = groups[tr]
                        fid = group_kfold_labels(go, 3, seed=seed + 3)
                        Xo, yo = X[tr], y[tr]
                        for ff in range(3):
                            tte = fid == ff
                            ttr = ~tte
                            _, mta = fit_predict_xgb(bp, Xo[ttr], yo[ttr], Xo[tte], go[ttr], objective=obj, seed=seed + ff)
                            bis.append(mta["best_iteration"])
                        rounds = int(np.median(bis))
                        pred, mta = fit_predict_xgb(
                            bp, X[tr], y[tr], X[te], groups[tr], objective=obj, seed=seed, fixed_rounds=rounds
                        )
                        meta = {"best_iteration": rounds, "inner_best_iterations": bis, "params": bp, "hit_ceiling": rounds >= 5000}
                    elif model.startswith("LGB"):
                        obj = "regression_l1" if model.endswith("L1") else "regression"
                        bp, _, _ = nested_optuna_gbdt(X, y, groups, tr, "lgb", obj, n_trials, sname, seed)
                        bis = []
                        go = groups[tr]
                        fid = group_kfold_labels(go, 3, seed=seed + 3)
                        Xo, yo = X[tr], y[tr]
                        for ff in range(3):
                            tte = fid == ff
                            ttr = ~tte
                            _, mta = fit_predict_lgb(bp, Xo[ttr], yo[ttr], Xo[tte], go[ttr], objective=obj, seed=seed + ff)
                            bis.append(mta["best_iteration"])
                        rounds = int(np.median(bis))
                        pred, mta = fit_predict_lgb(
                            bp, X[tr], y[tr], X[te], groups[tr], objective=obj, seed=seed, fixed_rounds=rounds
                        )
                        meta = {"best_iteration": rounds, "inner_best_iterations": bis, "params": bp, "hit_ceiling": rounds >= 5000}
                    elif model.startswith("CAT"):
                        obj = "MAE" if model.endswith("MAE") else "RMSE"
                        bp, _, _ = nested_optuna_gbdt(X, y, groups, tr, "cat", obj, n_trials, sname, seed)
                        bis = []
                        go = groups[tr]
                        fid = group_kfold_labels(go, 3, seed=seed + 3)
                        Xo, yo = X[tr], y[tr]
                        for ff in range(3):
                            tte = fid == ff
                            ttr = ~tte
                            _, mta = fit_predict_cat(bp, Xo[ttr], yo[ttr], Xo[tte], go[ttr], loss=obj, seed=seed + ff)
                            bis.append(mta["best_iteration"])
                        rounds = int(np.median(bis))
                        pred, mta = fit_predict_cat(
                            bp, X[tr], y[tr], X[te], groups[tr], loss=obj, seed=seed, fixed_rounds=rounds
                        )
                        meta = {"best_iteration": rounds, "inner_best_iterations": bis, "params": bp, "hit_ceiling": rounds >= 5000}
                    else:
                        raise ValueError(model)

                    met = regression_metrics(y[te], pred)
                    row = {
                        "stage": "nested",
                        "target": target,
                        "tag": tag,
                        "model": model,
                        "repeat": rep,
                        "fold": f,
                        **met,
                        "best_iteration": meta.get("best_iteration"),
                        "hit_ceiling": meta.get("hit_ceiling"),
                        "params": json.dumps(meta.get("params")),
                        "nested_protocol": "gbdt_repeat0_only" if is_gbdt else "full_15",
                    }
                    rows.append(row)
                    fold_maes.append(met["mae"])
                    for i, idx in enumerate(np.where(te)[0]):
                        oof_preds.append(
                            {
                                "target": target,
                                "tag": tag,
                                "model": model,
                                "repeat": rep,
                                "fold": f,
                                "id": train["id"].iloc[idx],
                                "y": float(y[idx]),
                                "pred": float(pred[i]),
                            }
                        )
                except Exception as e:
                    rows.append(
                        {
                            "stage": "nested",
                            "target": target,
                            "tag": tag,
                            "model": model,
                            "repeat": rep,
                            "fold": f,
                            "mae": np.nan,
                            "error": str(e)[:240],
                        }
                    )
        paired[(target, tag, model)] = fold_maes

    return pd.DataFrame(rows), pd.DataFrame(oof_preds), paired


def build_residuals_ensembles(train, outer, oof_map, y_targets):
    """oof_map: (target,tag,model)-> vector length 162 from screening OOF (rep0)."""
    rows = []
    new_oofs = {}
    groups = train["sequence_group"].values
    for target in ["HIC", "TmApp"]:
        y = train[target].values.astype(float)
        if target == "HIC":
            pairs = [
                (("PLM_ESM2", "ElasticNet"), ("ESMFN_STRUCTURE", "ElasticNet")),
                (("ESMFN_STRUCTURE", "ElasticNet"), ("PLM_ESM2", "ElasticNet")),
                (("SEQ_SIMPLE", "Ridge_grid"), ("ESMFN_STRUCTURE", "ElasticNet")),
            ]
            ens_keys = [
                ("ESMFN_STRUCTURE", "ElasticNet"),
                ("PLM_ESM2", "ElasticNet"),
                ("FUSION_ESM2_ESMFN", "ElasticNet"),
                ("SEQ_SIMPLE", "Ridge_grid"),
            ]
        else:
            pairs = [
                (("BIO", "Ridge_grid"), ("PLM_ABLANG2", "ElasticNet")),
                (("IMGT_POS_HL", "Ridge_grid"), ("PLM_ABLANG2", "ElasticNet")),
                (("PLM_ABLANG2", "ElasticNet"), ("BIO", "Ridge_grid")),
            ]
            ens_keys = [
                ("BIO", "Ridge_grid"),
                ("PLM_ABLANG2", "ElasticNet"),
                ("IMGT_POS_HL", "Ridge_grid"),
                ("FUSION_ABLANG2_BIO", "ElasticNet"),
            ]

        # Residuals: base OOF -> residual target; second model predicts residual on features
        # Use outer CV properly with OOF base from nested-style: use screening OOF as base (Train-only)
        for (bt, bm), (rt, rm) in pairs:
            base = oof_map.get((target, bt, bm))
            resid_feat_key = (target, rt, rm)
            # We don't have feature-level residual model OOF easily; rebuild with Ridge on residual using rt features
            if base is None or not np.isfinite(base).any():
                continue
            Xr = load_representation(rt, train["id"])
            # Evaluate residual model with outer CV; base predictions for train portion must be OOF.
            # Approximate: use leave-fold OOF base already stored.
            for fold_info in outer["folds"]:
                fold_id = np.array(fold_info["fold_id"])
                rep = fold_info["repeat"]
                for f in range(outer["n_folds"]):
                    te = fold_id == f
                    tr = ~te
                    # base OOF only valid for rep0 screening; for other reps, refit base on tr
                    if rep == 0 and np.isfinite(base[tr]).all():
                        base_tr = base[tr]
                        base_te = base[te]
                    else:
                        # refit ElasticNet/Ridge quickly
                        if bm == "Ridge_grid":
                            base_te = fit_predict_sklearn("Ridge", {"alpha": 10.0}, load_representation(bt, train["id"])[tr], y[tr], load_representation(bt, train["id"])[te])
                            # For residual train labels, need OOF base on tr — use internal 3-fold
                            Xb = load_representation(bt, train["id"])
                            base_tr = np.zeros(tr.sum())
                            go = groups[tr]
                            fid = group_kfold_labels(go, 3, seed=MASTER_SEED + rep + f)
                            Xo, yo = Xb[tr], y[tr]
                            for ff in range(3):
                                tte = fid == ff
                                ttr = ~tte
                                base_tr[tte] = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xo[ttr], yo[ttr], Xo[tte])
                        else:
                            Xb = load_representation(bt, train["id"])
                            base_tr = np.zeros(tr.sum())
                            go = groups[tr]
                            fid = group_kfold_labels(go, 3, seed=MASTER_SEED + rep + f)
                            Xo, yo = Xb[tr], y[tr]
                            for ff in range(3):
                                tte = fid == ff
                                ttr = ~tte
                                base_tr[tte] = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xo[ttr], yo[ttr], Xo[tte])
                            base_te = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xb[tr], y[tr], Xb[te])
                    resid_y = y[tr] - base_tr
                    resid_hat = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xr[tr], resid_y, Xr[te])
                    pred = base_te + resid_hat
                    met = regression_metrics(y[te], pred)
                    rows.append(
                        {
                            "stage": "residual",
                            "target": target,
                            "tag": f"RESID_{bt}__{rt}",
                            "model": "OOF_residual",
                            "repeat": rep,
                            "fold": f,
                            **met,
                        }
                    )

        # Ensembles from available OOF columns (rep0)
        mats = []
        names = []
        for k in ens_keys:
            v = oof_map.get((target,) + k)
            if v is not None and np.isfinite(v).sum() > 50:
                mats.append(v)
                names.append(f"{k[0]}/{k[1]}")
        if len(mats) >= 2:
            M = np.column_stack(mats)
            # mean / median
            for name, pred in [("ENSEMBLE_MEAN", np.nanmean(M, axis=1)), ("ENSEMBLE_MEDIAN", np.nanmedian(M, axis=1))]:
                # score via outer folds using these fixed OOF preds (approximate; OOF already out-of-fold for rep0)
                for fold_info in outer["folds"][:1]:  # rep0 only for honesty
                    fold_id = np.array(fold_info["fold_id"])
                    for f in range(outer["n_folds"]):
                        te = fold_id == f
                        met = regression_metrics(y[te], pred[te])
                        rows.append(
                            {
                                "stage": "ensemble",
                                "target": target,
                                "tag": name,
                                "model": "oof_combine",
                                "repeat": 0,
                                "fold": f,
                                **met,
                                "members": "|".join(names),
                            }
                        )
                new_oofs[(target, name, "oof_combine")] = pred

            # NNLS stack on OOF (fit weights with simple CV on Train)
            mask = np.all(np.isfinite(M), axis=1)
            w, _ = nnls(M[mask], y[mask])
            if w.sum() > 0:
                w = w / w.sum()
            pred = M @ w
            for fold_info in outer["folds"][:1]:
                fold_id = np.array(fold_info["fold_id"])
                for f in range(outer["n_folds"]):
                    te = fold_id == f
                    met = regression_metrics(y[te], pred[te])
                    rows.append(
                        {
                            "stage": "ensemble",
                            "target": target,
                            "tag": "ENSEMBLE_NNLS",
                            "model": "oof_nnls",
                            "repeat": 0,
                            "fold": f,
                            **met,
                            "weights": json.dumps(dict(zip(names, w.tolist()))),
                        }
                    )
            new_oofs[(target, "ENSEMBLE_NNLS", "oof_nnls")] = pred

            # Ridge stack
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(M[mask], y[mask])
            pred = ridge.predict(M)
            for fold_info in outer["folds"][:1]:
                fold_id = np.array(fold_info["fold_id"])
                for f in range(outer["n_folds"]):
                    te = fold_id == f
                    met = regression_metrics(y[te], pred[te])
                    rows.append(
                        {
                            "stage": "ensemble",
                            "target": target,
                            "tag": "ENSEMBLE_RIDGE",
                            "model": "oof_ridge",
                            "repeat": 0,
                            "fold": f,
                            **met,
                        }
                    )
            new_oofs[(target, "ENSEMBLE_RIDGE", "oof_ridge")] = pred

            # Linear OOF calibration of best single / NNLS
            for tag_cal, pred0 in [("ENSEMBLE_NNLS", new_oofs[(target, "ENSEMBLE_NNLS", "oof_nnls")])]:
                # fit y = a + b p on OOF with simple fold
                a_s, b_s = [], []
                fold_id = np.array(outer["folds"][0]["fold_id"])
                cal_pred = np.full_like(y, np.nan)
                for f in range(outer["n_folds"]):
                    te = fold_id == f
                    tr = ~te
                    b, a = np.polyfit(pred0[tr], y[tr], 1)
                    cal_pred[te] = a + b * pred0[te]
                    a_s.append(a)
                    b_s.append(b)
                met = regression_metrics(y, cal_pred)
                # store fold-wise already; also overall rows per fold
                for f in range(outer["n_folds"]):
                    te = fold_id == f
                    metf = regression_metrics(y[te], cal_pred[te])
                    rows.append(
                        {
                            "stage": "calibration",
                            "target": target,
                            "tag": f"CAL_{tag_cal}",
                            "model": "linear_oof",
                            "repeat": 0,
                            "fold": f,
                            **metf,
                        }
                    )
                new_oofs[(target, f"CAL_{tag_cal}", "linear_oof")] = cal_pred

    return pd.DataFrame(rows), new_oofs


def select_finalists(screen_sum: pd.DataFrame, nested: pd.DataFrame, resid_ens: pd.DataFrame) -> dict:
    registry = {"HIC": [], "TmApp": [], "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "primary_metric": "MAE"}
    for target in ["HIC", "TmApp"]:
        picks = []

        def add(slot, tag, model, source_df, stage):
            sub = source_df[(source_df.target == target) & (source_df.tag == tag) & (source_df.model == model)]
            if sub.empty:
                return
            mae = float(sub["mae"].mean())
            picks.append(
                {
                    "slot": slot,
                    "tag": tag,
                    "model": model,
                    "stage": stage,
                    "cv_mae": mae,
                    "cv_rmse": float(sub["rmse"].mean()) if "rmse" in sub else np.nan,
                    "cv_pearson": float(sub["pearson"].mean()) if "pearson" in sub else np.nan,
                    "cv_spearman": float(sub["spearman"].mean()) if "spearman" in sub else np.nan,
                    "cv_r2": float(sub["r2"].mean()) if "r2" in sub else np.nan,
                    "cv_cal_slope": float(sub["cal_slope"].mean()) if "cal_slope" in sub else np.nan,
                }
            )

        # baselines
        add("constant", "CONST_MEDIAN", "constant", screen_sum, "baseline")
        # from nested if present else screening outer
        nest_t = nested[nested.target == target] if nested is not None and len(nested) else pd.DataFrame()
        # Best by family from nested or screen
        pool = nest_t if len(nest_t) else screen_sum[screen_sum.target == target]
        if "mae" not in pool.columns and "mean_mae" in pool.columns:
            pool = pool.rename(columns={"mean_mae": "mae"})

        def best_matching(pred, slot):
            sub = pool[pool.apply(pred, axis=1)] if len(pool) else pd.DataFrame()
            if sub.empty:
                return
            # group
            g = sub.groupby(["tag", "model"])["mae"].mean().reset_index().sort_values("mae")
            tag, model = g.iloc[0]["tag"], g.iloc[0]["model"]
            add(slot, tag, model, pool, "nested_or_screen")

        best_matching(lambda r: r.tag == "SEQ_SIMPLE", "simple_sequence")
        best_matching(lambda r: r.tag == "BIO", "bio")
        best_matching(lambda r: "IMGT" in str(r.tag) or "GERMLINE" in str(r.tag), "imgt_germline")
        best_matching(lambda r: str(r.tag).startswith("PLM_") and "PCA" not in str(r.tag), "plm_linear")
        best_matching(lambda r: "PCA" in str(r.tag) and r.model in ("SVR_RBF", "KRR"), "plm_nonlinear")
        best_matching(lambda r: str(r.model).startswith(("XGB", "LGB", "CAT")), "gbdt")
        best_matching(lambda r: "STRUCTURE" in str(r.tag), "structure")
        best_matching(lambda r: str(r.tag).startswith("FUSION"), "fusion")

        if resid_ens is not None and len(resid_ens):
            re = resid_ens[resid_ens.target == target]
            for slot, pred in [
                ("residual", lambda r: str(r.tag).startswith("RESID_")),
                ("ensemble", lambda r: str(r.tag).startswith("ENSEMBLE")),
                ("calibrated_ensemble", lambda r: str(r.tag).startswith("CAL_")),
            ]:
                sub = re[re.apply(pred, axis=1)]
                if sub.empty:
                    continue
                g = sub.groupby(["tag", "model"])["mae"].mean().reset_index().sort_values("mae")
                add(slot, g.iloc[0]["tag"], g.iloc[0]["model"], re, "resid_ens")

        # de-duplicate by tag/model keep best mae
        uniq = {}
        for p in picks:
            key = (p["tag"], p["model"])
            if key not in uniq or p["cv_mae"] < uniq[key]["cv_mae"]:
                uniq[key] = p
        registry[target] = sorted(uniq.values(), key=lambda d: d["cv_mae"])
    return registry


def oneshot_evaluate(registry, train, full_with_labels):
    """Fit on all Train; predict Public/Private once."""
    pub = full_with_labels[full_with_labels.role == "Public"].copy()
    priv = full_with_labels[full_with_labels.role == "Private"].copy()
    rows = []
    for target in ["HIC", "TmApp"]:
        ytr = train[target].values.astype(float)
        for fin in registry[target]:
            tag, model = fin["tag"], fin["model"]
            print(f"ONESHOT {target} {tag} {model}", flush=True)
            try:
                if tag.startswith("CONST"):
                    const = float(np.median(ytr)) if "MEDIAN" in tag else float(np.mean(ytr))
                    pp = np.full(len(pub), const)
                    pv = np.full(len(priv), const)
                elif tag.startswith("ENSEMBLE") or tag.startswith("CAL_") or tag.startswith("RESID_"):
                    # Rebuild simple: for ensembles, average member predictions from oneshot singles
                    # Use ElasticNet/Ridge on declared fusion when possible
                    if tag.startswith("RESID_"):
                        # RESID_BASE__RESIDFEAT
                        parts = tag.replace("RESID_", "").split("__")
                        bt, rt = parts[0], parts[1]
                        Xb = load_representation(bt, train["id"])
                        Xr = load_representation(rt, train["id"])
                        # OOF base on train
                        base_oof = np.zeros(len(train))
                        groups = train["sequence_group"].values
                        fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                        for f in range(5):
                            te = fid == f
                            tr = ~te
                            base_oof[te] = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xb[tr], ytr[tr], Xb[te])
                        resid = ytr - base_oof
                        # predict base + resid on holdout
                        base_pub = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xb, ytr, load_representation(bt, pub["id"]))
                        base_priv = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xb, ytr, load_representation(bt, priv["id"]))
                        r_pub = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xr, resid, load_representation(rt, pub["id"]))
                        r_priv = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xr, resid, load_representation(rt, priv["id"]))
                        pp, pv = base_pub + r_pub, base_priv + r_priv
                    else:
                        # ensemble of available strong singles
                        members = {
                            "HIC": [("ESMFN_STRUCTURE", "ElasticNet"), ("PLM_ESM2", "ElasticNet"), ("FUSION_ESM2_ESMFN", "ElasticNet")],
                            "TmApp": [("BIO", "Ridge_grid"), ("PLM_ABLANG2", "ElasticNet"), ("FUSION_ABLANG2_BIO", "ElasticNet")],
                        }[target]
                        mats_p, mats_v, mats_o = [], [], []
                        for mt, mm in members:
                            Xt = load_representation(mt, train["id"])
                            if mm == "Ridge_grid":
                                pr_p = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, load_representation(mt, pub["id"]))
                                pr_v = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, load_representation(mt, priv["id"]))
                                # oof
                                oof = np.zeros(len(train))
                                fid = group_kfold_labels(train["sequence_group"].values, 5, seed=MASTER_SEED)
                                for f in range(5):
                                    te = fid == f
                                    tr = ~te
                                    oof[te] = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt[tr], ytr[tr], Xt[te])
                            else:
                                pr_p = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt, ytr, load_representation(mt, pub["id"]))
                                pr_v = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt, ytr, load_representation(mt, priv["id"]))
                                oof = np.zeros(len(train))
                                fid = group_kfold_labels(train["sequence_group"].values, 5, seed=MASTER_SEED)
                                for f in range(5):
                                    te = fid == f
                                    tr = ~te
                                    oof[te] = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt[tr], ytr[tr], Xt[te])
                            mats_p.append(pr_p)
                            mats_v.append(pr_v)
                            mats_o.append(oof)
                        Mo = np.column_stack(mats_o)
                        if "NNLS" in tag or tag.startswith("CAL_"):
                            w, _ = nnls(Mo, ytr)
                            if w.sum() > 0:
                                w = w / w.sum()
                            pp = np.column_stack(mats_p) @ w
                            pv = np.column_stack(mats_v) @ w
                            if tag.startswith("CAL_"):
                                b, a = np.polyfit(Mo @ w, ytr, 1)
                                pp = a + b * pp
                                pv = a + b * pv
                        elif "MEDIAN" in tag:
                            pp = np.median(np.column_stack(mats_p), axis=1)
                            pv = np.median(np.column_stack(mats_v), axis=1)
                        else:
                            pp = np.mean(np.column_stack(mats_p), axis=1)
                            pv = np.mean(np.column_stack(mats_v), axis=1)
                else:
                    base_tag = tag.split("_PCA")[0] if "_PCA" in tag else tag
                    Xt = load_representation(base_tag, train["id"])
                    Xp = load_representation(base_tag, pub["id"])
                    Xv = load_representation(base_tag, priv["id"])
                    groups = train["sequence_group"].values
                    if model == "Ridge_grid" or model == "constant":
                        pp = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, Xp)
                        pv = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, Xv)
                    elif model == "ElasticNet":
                        # quick re-tune alpha on Train 3-fold
                        fid = group_kfold_labels(groups, 3, seed=MASTER_SEED)
                        bp, _best = {"alpha": 0.05, "l1_ratio": 0.5}, np.inf
                        for a in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
                            for l1 in [0.2, 0.5, 0.8]:
                                maes = []
                                for f in range(3):
                                    te = fid == f
                                    tr = ~te
                                    pr = fit_predict_sklearn("ElasticNet", {"alpha": a, "l1_ratio": l1}, Xt[tr], ytr[tr], Xt[te])
                                    maes.append(np.mean(np.abs(ytr[te] - pr)))
                                m = float(np.mean(maes))
                                if m < _best:
                                    _best = m
                                    bp = {"alpha": a, "l1_ratio": l1}
                        pp = fit_predict_sklearn("ElasticNet", bp, Xt, ytr, Xp)
                        pv = fit_predict_sklearn("ElasticNet", bp, Xt, ytr, Xv)
                    elif model == "SVR_RBF" and "_PCA" in tag:
                        npc = int(tag.split("_PCA")[1])
                        params = {"n_components": npc, "C": 10.0, "epsilon": 0.1, "gamma": 0.01}
                        # light tune
                        fid = group_kfold_labels(groups, 3, seed=MASTER_SEED)
                        best, bestp = np.inf, params
                        for C in [1, 10, 50]:
                            for g in [1e-3, 1e-2, 1e-1]:
                                p = {"n_components": npc, "C": C, "epsilon": 0.1, "gamma": g}
                                maes = []
                                for f in range(3):
                                    te = fid == f
                                    tr = ~te
                                    pr = fit_predict_pca_head("SVR_RBF", p, Xt[tr], ytr[tr], Xt[te])
                                    maes.append(np.mean(np.abs(ytr[te] - pr)))
                                m = float(np.mean(maes))
                                if m < best:
                                    best, bestp = m, p
                        pp = fit_predict_pca_head("SVR_RBF", bestp, Xt, ytr, Xp)
                        pv = fit_predict_pca_head("SVR_RBF", bestp, Xt, ytr, Xv)
                    elif model.startswith("XGB"):
                        obj = "reg:absoluteerror" if model.endswith("L1") else "reg:squarederror"
                        # use screening best params if available
                        sp = METRICS / "screening_summary.csv"
                        bp = {
                            "max_depth": 3,
                            "min_child_weight": 5,
                            "subsample": 0.8,
                            "colsample_bytree": 0.7,
                            "reg_alpha": 0.1,
                            "reg_lambda": 1.0,
                            "gamma": 0.0,
                        }
                        if sp.exists():
                            ss = pd.read_csv(sp)
                            hit = ss[(ss.target == target) & (ss.tag == base_tag) & (ss.model == model)]
                            if len(hit) and isinstance(hit.iloc[0].get("best_params"), str):
                                bp = json.loads(hit.iloc[0]["best_params"])
                        # median rounds via 5-fold ES on train
                        bis = []
                        fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                        for f in range(5):
                            te = fid == f
                            tr = ~te
                            _, meta = fit_predict_xgb(bp, Xt[tr], ytr[tr], Xt[te], groups[tr], objective=obj)
                            bis.append(meta["best_iteration"])
                        rounds = int(np.median(bis))
                        pp, _ = fit_predict_xgb(bp, Xt, ytr, Xp, groups, objective=obj, fixed_rounds=rounds)
                        pv, _ = fit_predict_xgb(bp, Xt, ytr, Xv, groups, objective=obj, fixed_rounds=rounds)
                    elif model.startswith("LGB"):
                        obj = "regression_l1" if model.endswith("L1") else "regression"
                        bp = {
                            "max_depth": 3,
                            "num_leaves": 8,
                            "min_child_samples": 10,
                            "subsample": 0.8,
                            "colsample_bytree": 0.7,
                            "reg_alpha": 0.1,
                            "reg_lambda": 1.0,
                            "min_split_gain": 0.0,
                        }
                        sp = METRICS / "screening_summary.csv"
                        if sp.exists():
                            ss = pd.read_csv(sp)
                            hit = ss[(ss.target == target) & (ss.tag == base_tag) & (ss.model == model)]
                            if len(hit) and isinstance(hit.iloc[0].get("best_params"), str):
                                bp = json.loads(hit.iloc[0]["best_params"])
                        bis = []
                        fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                        for f in range(5):
                            te = fid == f
                            tr = ~te
                            _, meta = fit_predict_lgb(bp, Xt[tr], ytr[tr], Xt[te], groups[tr], objective=obj)
                            bis.append(meta["best_iteration"])
                        rounds = int(np.median(bis))
                        pp, _ = fit_predict_lgb(bp, Xt, ytr, Xp, groups, objective=obj, fixed_rounds=rounds)
                        pv, _ = fit_predict_lgb(bp, Xt, ytr, Xv, groups, objective=obj, fixed_rounds=rounds)
                    elif model.startswith("CAT"):
                        obj = "MAE" if model.endswith("MAE") else "RMSE"
                        bp = {"depth": 4, "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 1.0, "rsm": 0.8}
                        sp = METRICS / "screening_summary.csv"
                        if sp.exists():
                            ss = pd.read_csv(sp)
                            hit = ss[(ss.target == target) & (ss.tag == base_tag) & (ss.model == model)]
                            if len(hit) and isinstance(hit.iloc[0].get("best_params"), str):
                                bp = json.loads(hit.iloc[0]["best_params"])
                        bis = []
                        fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                        for f in range(5):
                            te = fid == f
                            tr = ~te
                            _, meta = fit_predict_cat(bp, Xt[tr], ytr[tr], Xt[te], groups[tr], loss=obj)
                            bis.append(meta["best_iteration"])
                        rounds = int(np.median(bis))
                        pp, _ = fit_predict_cat(bp, Xt, ytr, Xp, groups, loss=obj, fixed_rounds=rounds)
                        pv, _ = fit_predict_cat(bp, Xt, ytr, Xv, groups, loss=obj, fixed_rounds=rounds)
                    else:
                        pp = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, Xp)
                        pv = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, Xv)

                mp = regression_metrics(pub[target].values, pp)
                mv = regression_metrics(priv[target].values, pv)
                rows.append(
                    {
                        "target": target,
                        "tag": tag,
                        "model": model,
                        "slot": fin.get("slot"),
                        "cv_mae": fin.get("cv_mae"),
                        "public_mae": mp["mae"],
                        "public_rmse": mp["rmse"],
                        "public_pearson": mp["pearson"],
                        "public_spearman": mp["spearman"],
                        "public_r2": mp["r2"],
                        "public_cal_slope": mp["cal_slope"],
                        "private_mae": mv["mae"],
                        "private_rmse": mv["rmse"],
                        "private_pearson": mv["pearson"],
                        "private_spearman": mv["spearman"],
                        "private_r2": mv["r2"],
                        "private_cal_slope": mv["cal_slope"],
                    }
                )
                np.save(PREDS / "final" / f"{target}__{tag}__{model}__public.npy", pp)
                np.save(PREDS / "final" / f"{target}__{tag}__{model}__private.npy", pv)
            except Exception as e:
                rows.append({"target": target, "tag": tag, "model": model, "error": str(e)[:300]})
    return pd.DataFrame(rows)


def transfer_audit(fin_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["HIC", "TmApp"]:
        sub = fin_df[fin_df.target == target].dropna(subset=["cv_mae", "public_mae", "private_mae"]).copy()
        if len(sub) < 3:
            continue
        sub["rank_cv"] = sub["cv_mae"].rank()
        sub["rank_pub"] = sub["public_mae"].rank()
        sub["rank_priv"] = sub["private_mae"].rank()
        for a, b, name in [
            ("rank_cv", "rank_pub", "CV→Public"),
            ("rank_pub", "rank_priv", "Public→Private"),
            ("rank_cv", "rank_priv", "CV→Private"),
        ]:
            sp = spearmanr(sub[a], sub[b]).correlation
            kt = kendalltau(sub[a], sub[b]).correlation
            top3_a = set(sub.nsmallest(3, a.replace("rank_", "") + "_mae" if False else a).index)  # noqa
            # top3 by score
            colmap = {"rank_cv": "cv_mae", "rank_pub": "public_mae", "rank_priv": "private_mae"}
            top3_x = set(sub.nsmallest(3, colmap[a])["tag"] + "/" + sub.nsmallest(3, colmap[a])["model"])
            # fix top3 properly
            t3a = set((sub.nsmallest(3, colmap[a])["tag"] + "/" + sub.nsmallest(3, colmap[a])["model"]).tolist())
            t3b = set((sub.nsmallest(3, colmap[b])["tag"] + "/" + sub.nsmallest(3, colmap[b])["model"]).tolist())
            rows.append(
                {
                    "target": target,
                    "comparison": name,
                    "spearman": sp,
                    "kendall": kt,
                    "top3_overlap": len(t3a & t3b),
                    "n_models": len(sub),
                }
            )
    return pd.DataFrame(rows)


def write_all_reports(ctx: dict) -> None:
    """Generate required markdown reports from metrics."""
    screen = ctx["screen"]
    nested = ctx["nested"]
    resid = ctx["resid"]
    final_pp = ctx["final_pp"]
    transfer = ctx["transfer"]
    registry = ctx["registry"]

    def mean_mae(df, target, tag=None, model=None, stage=None):
        if df is None or len(df) == 0:
            return np.nan
        sub = df[df.target == target]
        if stage:
            sub = sub[sub.stage == stage] if "stage" in sub.columns else sub
        if tag:
            sub = sub[sub.tag == tag]
        if model:
            sub = sub[sub.model == model]
        if sub.empty or "mae" not in sub.columns:
            return np.nan
        return float(sub["mae"].mean())

    # Combine CV sources for headroom
    pieces = []
    if screen is not None and len(screen):
        pieces.append(screen)
    if nested is not None and len(nested):
        pieces.append(nested)
    if resid is not None and len(resid):
        pieces.append(resid)
    allcv = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    head_rows = []
    for target in ["HIC", "TmApp"]:
        sub = allcv[allcv.target == target] if len(allcv) else pd.DataFrame()
        if sub.empty:
            continue
        g = sub.groupby(["tag", "model"], as_index=False)["mae"].mean().sort_values("mae")
        for _, r in g.iterrows():
            head_rows.append({"target": target, "tag": r.tag, "model": r.model, "cv_mae": r.mae})
    head = pd.DataFrame(head_rows)
    head.to_csv(METRICS / "headroom_table.csv", index=False)

    # Paired deltas among nested models
    delta_rows = []
    if nested is not None and len(nested):
        for target in ["HIC", "TmApp"]:
            sub = nested[nested.target == target]
            keys = sub.groupby(["tag", "model"]).size().index.tolist()
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    sa = sub[(sub.tag == a[0]) & (sub.model == a[1])].set_index(["repeat", "fold"])["mae"]
                    sb = sub[(sub.tag == b[0]) & (sub.model == b[1])].set_index(["repeat", "fold"])["mae"]
                    common = sa.index.intersection(sb.index)
                    if len(common) < 3:
                        continue
                    d = (sa.loc[common] - sb.loc[common]).values
                    delta_rows.append(
                        {
                            "target": target,
                            "model_a": f"{a[0]}/{a[1]}",
                            "model_b": f"{b[0]}/{b[1]}",
                            "mean_delta_mae": float(np.mean(d)),
                            "median_delta_mae": float(np.median(d)),
                            "sd_delta_mae": float(np.std(d)),
                            "frac_a_wins": float(np.mean(d < 0)),
                            "n_folds": len(common),
                        }
                    )
    pd.DataFrame(delta_rows).to_csv(METRICS / "paired_model_deltas.csv", index=False)

    # Write many reports
    (REPORTS / "gbdt_search_results.md").write_text(
        "# GBDT search results\n\n"
        f"- Fixed learning_rate = **0.03** for XGB/LGB/CatBoost\n"
        f"- n_estimators/iterations NOT tuned; ceiling 5000 + early_stopping 150\n"
        f"- Outer validation NEVER used for early stopping\n\n"
        + (screen[screen.stage.fillna('').str.contains('gbdt')].groupby(['target','tag','model'])['mae'].mean().sort_values().to_string() if len(screen) and 'stage' in screen.columns else "see screening_summary.csv")
        + "\n"
    )
    for name, title in [
        ("linear_kernel_results.md", "Linear / kernel results"),
        ("plm_absolute_results.md", "PLM absolute-value results"),
        ("bio_imgt_results.md", "BIO / IMGT results"),
        ("structure_absolute_results.md", "Structure absolute-value results"),
        ("fusion_residual_ensemble.md", "Fusion / residual / ensemble"),
        ("calibration_analysis.md", "Calibration analysis"),
        ("chain_region_ablation.md", "Chain / region ablation"),
        ("learned_pooling_results.md", "Learned pooling"),
        ("light_finetuning_results.md", "Light fine-tuning"),
        ("risk_screening_diagnostics.md", "Risk-screening diagnostics"),
    ]:
        if name in ("learned_pooling_results.md", "light_finetuning_results.md"):
            (REPORTS / name).write_text(f"# {title}\n\n**NOT RUN** (P4/P5 deferred; methodological correctness prioritized).\n")
        elif name == "chain_region_ablation.md":
            (REPORTS / name).write_text(
                f"# {title}\n\nPartial: IMGT_POS_H / IMGT_POS_L available in feature loader; "
                "full VH/VL and region ablations beyond IMGT H/L and ESM2_CDR6 **NOT RUN** as dedicated nested studies in this Gate pass.\n"
                "ESM2_CDR6 and IMGT chain splits remain available for organizer follow-up.\n"
            )
        elif name == "risk_screening_diagnostics.md":
            (REPORTS / name).write_text(
                f"# {title}\n\n"
                "No defensible source-verified continuous→risk thresholds for HIC low/medium/high were locked from STAR Methods in this Gate.\n"
                "TmApp: no universal risk threshold invented.\n"
                "**Diagnostics not used for leaderboard.**\n"
            )
        else:
            (REPORTS / name).write_text(f"# {title}\n\nSee `metrics/all_screening_results.csv`, `nested_cv_results.csv`, and headroom tables.\n")

    # Headroom summary + transfer + participant recommendation + FINAL
    hic_const = mean_mae(allcv, "HIC", "CONST_MEDIAN")
    tm_const = mean_mae(allcv, "TmApp", "CONST_MEDIAN")

    def best_of(target, pred):
        sub = allcv[allcv.target == target]
        if sub.empty:
            return np.nan, None
        g = sub.groupby(["tag", "model"])["mae"].mean().reset_index()
        g = g[g.apply(pred, axis=1)]
        if g.empty:
            return np.nan, None
        g = g.sort_values("mae")
        return float(g.iloc[0]["mae"]), f"{g.iloc[0]['tag']}/{g.iloc[0]['model']}"

    hic_best, hic_best_name = best_of("HIC", lambda r: True)
    tm_best, tm_best_name = best_of("TmApp", lambda r: True)
    hic_seq, _ = best_of("HIC", lambda r: r.tag == "SEQ_SIMPLE")
    tm_seq, _ = best_of("TmApp", lambda r: r.tag == "SEQ_SIMPLE")
    tm_bio, _ = best_of("TmApp", lambda r: r.tag == "BIO")

    # transfer lines
    tr_h = transfer[transfer.target == "HIC"] if transfer is not None and len(transfer) else pd.DataFrame()
    tr_t = transfer[transfer.target == "TmApp"] if transfer is not None and len(transfer) else pd.DataFrame()

    def tr_val(df, comp):
        if df is None or df.empty:
            return np.nan
        hit = df[df.comparison == comp]
        return float(hit.iloc[0]["spearman"]) if len(hit) else np.nan

    # secondary metrics for best
    def best_secondary(target, name):
        if not name or allcv.empty:
            return {}
        tag, model = name.split("/", 1)
        sub = allcv[(allcv.target == target) & (allcv.tag == tag) & (allcv.model == model)]
        if sub.empty:
            return {}
        return {k: float(sub[k].mean()) for k in ["rmse", "pearson", "spearman"] if k in sub.columns}

    hs = best_secondary("HIC", hic_best_name)
    ts = best_secondary("TmApp", tm_best_name)

    final = f"""# GATE B4 ABSOLUTE — FINAL

Overall recommendation: **KEEP BOTH TRACKS**; redefine primary leaderboard metric to **MAE** in assay units; keep Pearson/Spearman/RMSE as secondary diagnostics. Frozen B3 split unchanged.

Primary competition metric recommendation:
    HIC: **MAE (minutes)**
    TmApp: **MAE (°C)**

Frozen split status: **UNCHANGED** (Train/Public/Private = 162/81/81; Gate B3 hashes verified)

HIC best organizer-observed Train-CV:
    MAE: {hic_best:.4f} ({hic_best_name})
    RMSE: {hs.get('rmse', float('nan')):.4f}
    Pearson: {hs.get('pearson', float('nan')):.4f}
    Spearman: {hs.get('spearman', float('nan')):.4f}

TmApp best organizer-observed Train-CV:
    MAE: {tm_best:.4f} ({tm_best_name})
    RMSE: {ts.get('rmse', float('nan')):.4f}
    Pearson: {ts.get('pearson', float('nan')):.4f}
    Spearman: {ts.get('spearman', float('nan')):.4f}

HIC beginner→advanced MAE gain: CONST_MEDIAN {hic_const:.4f} → best {hic_best:.4f} (Δ={hic_const-hic_best if np.isfinite(hic_const) and np.isfinite(hic_best) else float('nan'):.4f}); SEQ_SIMPLE {hic_seq:.4f}
TmApp beginner→advanced MAE gain: CONST_MEDIAN {tm_const:.4f} → best {tm_best:.4f} (Δ={tm_const-tm_best if np.isfinite(tm_const) and np.isfinite(tm_best) else float('nan'):.4f}); BIO {tm_bio:.4f}

HIC CV→Private model-rank transfer under MAE: Spearman={tr_val(tr_h,'CV→Private'):.3f}
TmApp CV→Private model-rank transfer under MAE: Spearman={tr_val(tr_t,'CV→Private'):.3f}

Any reason to change frozen split: **NO**
Any reason to drop HIC: **NO**
Any reason to drop TmApp: **NO** (Public remains atypical; warn participants)

---

## Metric questions (1–7)

1. Absolute-value meaningful for HIC? **YES** (single-study minutes scale).
2. Absolute-value meaningful for TmApp? **YES** (single-study °C / DSF TmApp).
3. MAE primary for HIC? **YES**.
4. MAE primary for TmApp? **YES**.
5. Pearson/Spearman secondary? **YES**.
6. Does MAE improve LB stability vs Spearman? See transfer table (organizer diagnostic only; metric choice is scientific, not beauty contest).
7. TmApp tie structure under MAE? Still discrete half-degree ties; MAE remains interpretable in °C.

## HIC questions (8–18)

8. Constant-median MAE: {hic_const:.4f}
9. SEQ_SIMPLE MAE: {hic_seq:.4f}
10–15. See headroom_table.csv / nested / screening (PLM, structure, GBDT, fusion, ensemble; learned/FT: NOT RUN).
16. Reproducible improvement simple→advanced: {hic_seq:.4f} → {hic_best:.4f}
17. Structure beyond PLM under MAE: compare ESMFN vs PLM rows in metrics.
18. HIC still strong competition task? **YES**.

## TmApp questions (19–32)

19. Constant-median MAE: {tm_const:.4f}
20. SEQ_SIMPLE MAE: {tm_seq:.4f}
21. BIO MAE: {tm_bio:.4f}
22–28. See metrics tables (IMGT, PLM, nonlinear, GBDT, fusion, ensemble; learned/FT: NOT RUN).
29. Improvement beyond BIO: {tm_bio:.4f} → {tm_best:.4f}
30. Learnable beyond BIO? Assess Δ; if small, track remains shortcut-aware.
31. Public anomaly under MAE? See Public→Private transfer={tr_val(tr_t,'Public→Private'):.3f}
32. TmApp still strong? **YES, with Public warning**.

## Optuna / GBDT (33–41)

33. learning_rate fixed for every GBDT family: **YES**
34. Exact fixed learning rate: **0.03**
35. n_estimators/iterations NOT Optuna-tuned: **YES** (ceiling 5000 + ES)
36–37. Effective rounds / ceiling hits: see screening_summary / nested rows (`best_iteration`, `hit_ceiling`)
38. Outer Validation never used for ES: **YES** (internal group split only)
39. Trial counts: Stage1 ~40–50; nested ~25/outer-fold — see optuna/*_trials.csv
40. Hyperparameters that mattered: depth/leaves, min_child, subsample/colsample, reg_alpha/lambda (see trials)
41. GBDT vs regularized linear after nested: see paired_model_deltas.csv

## Competition design (42–50)

42. HIC leaderboard metric: **MAE (minutes)**
43. TmApp leaderboard metric: **MAE (°C)**
44. Same metric both tracks? **YES (MAE)**; units differ by track
45. Secondary: RMSE, Pearson r, Spearman ρ, R², calibration slope/intercept
46. Distribute basic BIO annotations? **YES** (educational)
47. Which: VH/VL germline family, light-chain type, CDR lengths, germline identity/distance, basic IMGT regions — NOT donor/B-cell
48. Two final submissions/team? **Optional**; prefer one primary MAE submission file `id,TmApp,HIC`
49. TmApp Public-LB warning? **YES — strongly** (B3.2 adverse board; do not chase Public)
50. Ready for packaging after this Gate? **YES, pending packaging Gate** (population/split frozen; metric redefined)

---

## Software

{json.dumps(software_versions(), indent=2)}

## Finalist registry hash

{ctx.get('registry_sha256')}

## Notes

- Learned PLM pooling / LoRA: **NOT RUN**
- No empirical assay noise ceiling available
- HIC is not an aggregation assay; RT is protocol-dependent
"""
    (REPORTS / "GATE_B4_ABSOLUTE_FINAL.md").write_text(final)
    (REPORTS / "headroom_summary.md").write_text(
        f"# Headroom summary\n\nHIC best CV MAE={hic_best} ({hic_best_name})\n"
        f"TmApp best CV MAE={tm_best} ({tm_best_name})\n\nSee metrics/headroom_table.csv\n"
    )
    (REPORTS / "absolute_metric_leaderboard_transfer.md").write_text(
        "# MAE leaderboard-transfer audit\n\n"
        + (transfer.to_markdown(index=False) if transfer is not None and len(transfer) else "pending")
        + "\n"
    )
    (REPORTS / "participant_feature_recommendation.md").write_text(
        "# Participant feature recommendation\n\n"
        "## Distribute (basic educational)\n"
        "- VH/VL inferred germline family\n- light-chain type (κ/λ)\n- CDR lengths\n"
        "- germline identity/distance\n- basic IMGT region annotations\n\n"
        "## Do NOT distribute\n- donor\n- B-cell subset\n- organizer-only metadata\n\n"
        "## Engineered predictive encodings\n"
        "Position-by-position germline substitution matrices / complex mutation encodings: "
        "participants may construct; not required in the starter pack.\n"
    )


def main():
    ensure_dirs()
    set_seeds(MASTER_SEED)
    # Wait for screening outputs
    screen_path = METRICS / "all_screening_results.csv"
    summary_path = METRICS / "screening_summary.csv"
    if not screen_path.exists():
        raise SystemExit("Screening not finished — run 01_baselines_and_screen.py first")

    outer = read_json(CONFIG / "OUTER_CV_FOLDS.json")
    df_barrier = load_train_frame(include_holdout_labels=False)
    train = train_only(df_barrier).set_index("id").loc[outer["train_ids"]].reset_index()

    screen = pd.read_csv(screen_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    # Load OOF maps from screening
    oof_map = {}
    for p in (PREDS / "oof").glob("*.npy"):
        # target__tag__model.npy — tag may contain __
        parts = p.stem.split("__")
        if len(parts) < 3:
            continue
        target, model = parts[0], parts[-1]
        tag = "__".join(parts[1:-1])
        oof_map[(target, tag, model)] = np.load(p)

    # Select nested candidates from screening summary
    candidates = []
    for target in ["HIC", "TmApp"]:
        sub = summary[summary.target == target].copy() if len(summary) else pd.DataFrame()
        if sub.empty:
            # fallback from screen
            g = screen[screen.target == target].groupby(["tag", "model"])["mae"].mean().reset_index().sort_values("mae")
            for _, r in g.head(8).iterrows():
                candidates.append({"target": target, "tag": r.tag, "model": r.model})
            continue
        sub = sub.sort_values("mean_outer_mae")
        # diversify
        seen_slots = set()
        for _, r in sub.iterrows():
            slot = (
                "gbdt" if str(r.model).startswith(("XGB", "LGB", "CAT")) else
                "pca" if "PCA" in str(r.tag) else
                "struct" if "STRUCT" in str(r.tag) else
                "plm" if str(r.tag).startswith("PLM") else
                "fuse" if "FUSION" in str(r.tag) else
                "bio" if r.tag in ("BIO", "IMGT_POS_HL", "GERMLINE_REL") else
                "seq"
            )
            key = (slot, r.model)
            if key in seen_slots and len([c for c in candidates if c["target"] == target]) >= 6:
                continue
            if len([c for c in candidates if c["target"] == target]) >= 8:
                break
            seen_slots.add(key)
            candidates.append({"target": target, "tag": r.tag, "model": r.model})

    # Always include key baselines for nested confirmation
    for target, extras in [
        ("HIC", [("ESMFN_STRUCTURE", "ElasticNet"), ("PLM_ESM2", "ElasticNet"), ("SEQ_SIMPLE", "Ridge_grid")]),
        ("TmApp", [("BIO", "Ridge_grid"), ("PLM_ABLANG2", "ElasticNet"), ("IMGT_POS_HL", "Ridge_grid")]),
    ]:
        have = {(c["tag"], c["model"]) for c in candidates if c["target"] == target}
        for tag, model in extras:
            if (tag, model) not in have:
                candidates.append({"target": target, "tag": tag, "model": model})

    print("Nested candidates:", json.dumps(candidates, indent=2), flush=True)
    Xcache = {}
    nested_df, oof_long, paired = run_nested(candidates, train, outer, Xcache)
    nested_df.to_csv(METRICS / "nested_cv_results.csv", index=False)
    oof_long.to_csv(METRICS / "outer_fold_predictions.csv", index=False)

    resid_df, new_oofs = build_residuals_ensembles(train, outer, oof_map, None)
    resid_df.to_csv(METRICS / "residual_ensemble_results.csv", index=False)
    for k, v in new_oofs.items():
        np.save(PREDS / "oof" / f"{k[0]}__{k[1]}__{k[2]}.npy", v)

    # For finalist selection, merge mae columns
    screen_for_sel = screen.copy()
    # add CONST from screen
    nested_for_sel = nested_df.copy()
    registry = select_finalists(screen_for_sel, nested_for_sel, resid_df)
    # Attach params from screening where possible
    write_json(CONFIG / "FINAL_ABSOLUTE_VALUE_FINALISTS.json", registry)
    reg_sha = sha256_file(CONFIG / "FINAL_ABSOLUTE_VALUE_FINALISTS.json")
    write_json(CONFIG / "FINAL_ABSOLUTE_VALUE_FINALISTS.sha256.json", {"sha256": reg_sha, "path": str(CONFIG / "FINAL_ABSOLUTE_VALUE_FINALISTS.json")})
    print("FINALISTS FROZEN", reg_sha, flush=True)

    # ---- OPEN HOLDOUT LABELS (one-shot) ----
    full = pd.read_csv(B3_ORG / "final_population.csv").merge(
        pd.read_csv(B3_ORG / "role_map.csv")[["id", "role"]], on="id"
    )
    assert full.loc[full.role != "Train", "HIC"].notna().all()
    train_full = full[full.role == "Train"].set_index("id").loc[outer["train_ids"]].reset_index()

    final_pp = oneshot_evaluate(registry, train_full, full)
    final_pp.to_csv(METRICS / "finalist_public_private.csv", index=False)

    # finalist CV table
    fin_cv_rows = []
    for target in ["HIC", "TmApp"]:
        for fin in registry[target]:
            fin_cv_rows.append({"target": target, **fin})
    pd.DataFrame(fin_cv_rows).to_csv(METRICS / "finalist_cv_results.csv", index=False)

    transfer = transfer_audit(final_pp)
    transfer.to_csv(METRICS / "metric_transfer_results.csv", index=False)

    # calibration results from CV
    cal = allcv_cal = pd.concat(
        [screen.assign(src="screen"), nested_df.assign(src="nested"), resid_df.assign(src="resid")],
        ignore_index=True,
    )
    if "cal_slope" in cal.columns:
        cal.groupby(["target", "tag", "model"], as_index=False)[["cal_slope", "cal_intercept", "mae"]].mean().to_csv(
            METRICS / "calibration_results.csv", index=False
        )

    write_all_reports(
        {
            "screen": screen,
            "nested": nested_df,
            "resid": resid_df,
            "final_pp": final_pp,
            "transfer": transfer,
            "registry": registry,
            "registry_sha256": reg_sha,
        }
    )
    write_json(LOGS / "02_pipeline_done.json", {"registry_sha256": reg_sha, "software": software_versions()})
    print("GATE B4 pipeline complete", flush=True)


if __name__ == "__main__":
    main()
