#!/usr/bin/env python3
"""Fold-safe classical CV matching bundle_simple_tvt_v1 (+ ElasticNet/SVR/XGB).

Preprocessing order (documented):
  1. per-block train-median impute
  2. per-block PCA32 only for blocks flagged pca=True (wide PLM), never global concat PCA
  3. StandardScaler on concatenated matrix
  4. estimator with nested fold-local hyperparams
  5. refit on tr+va → predict te

This matches historical AbLingua recipe PCA (per AbLingua block only) and
historical standalone / HIC LASSO (no PCA).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "top_models_feature_bundle"))
from bundle_simple_tvt import (  # noqa: E402
    ALPHAS,
    LASSO_ALPHAS,
    LASSO_MAX_ITER,
    LASSO_TOL,
    PCA_CAP,
    mae,
)

# Column group: list of feature column names + whether to PCA that block
ColumnGroup = tuple[list[str], bool]


def load_folds(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    folds = pd.read_csv(path)
    folds["id"] = folds["id"].astype(str)
    primary = folds[folds["scheme"] == "primary"].set_index("id")["fold"].astype(int).to_dict()
    shadow = folds[folds["scheme"] == "shadow"].set_index("id")["fold"].astype(int).to_dict()
    return primary, shadow


def _impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med


def _impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    out = X.copy()
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(med, inds[1])
    return out


def _default_groups(X_all: pd.DataFrame) -> list[ColumnGroup]:
    cols = [c for c in X_all.columns if c != "id"]
    # No PCA by default (historical standalone); caller should mark PLM blocks
    return [(cols, False)]


def _matrix_cols(df: pd.DataFrame, ids: list[str], cols: list[str]) -> np.ndarray:
    return df.set_index("id").loc[ids, cols].to_numpy(dtype=float)


def _preprocess_fold(
    X_all: pd.DataFrame,
    tr: list[str],
    va: list[str],
    te: list[str],
    groups: list[ColumnGroup],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts_tr, parts_va, parts_te = [], [], []
    for cols, do_pca in groups:
        if not cols:
            continue
        Xtr = _matrix_cols(X_all, tr, cols)
        Xva = _matrix_cols(X_all, va, cols)
        Xte = _matrix_cols(X_all, te, cols)
        med = _impute_fit(Xtr)
        Xtr, Xva, Xte = _impute_apply(Xtr, med), _impute_apply(Xva, med), _impute_apply(Xte, med)
        if do_pca and Xtr.shape[1] > PCA_CAP:
            k = min(PCA_CAP, Xtr.shape[0] - 1, Xtr.shape[1])
            if k >= 1:
                pca = PCA(n_components=k, random_state=0)
                Xtr = pca.fit_transform(Xtr)
                Xva = pca.transform(Xva)
                Xte = pca.transform(Xte)
        parts_tr.append(Xtr)
        parts_va.append(Xva)
        parts_te.append(Xte)
    Xtr = np.hstack(parts_tr) if parts_tr else np.zeros((len(tr), 0))
    Xva = np.hstack(parts_va) if parts_va else np.zeros((len(va), 0))
    Xte = np.hstack(parts_te) if parts_te else np.zeros((len(te), 0))
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    Xte = sc.transform(Xte)
    return Xtr, Xva, Xte


def run_simple_tvt(
    X_all: pd.DataFrame,
    y_map: dict[str, float],
    fold_map: dict[str, int],
    *,
    estimator: str = "ridge",
    groups: list[ColumnGroup] | None = None,
) -> dict:
    """Return primary or shadow OOF + mae. estimator: ridge|lasso|enet|svr|xgb."""
    ids = [i for i in X_all["id"].astype(str).tolist() if i in fold_map and i in y_map]
    groups = groups if groups is not None else _default_groups(X_all)
    oof = pd.Series(index=ids, dtype=float)
    fold_params = []
    for k in range(5):
        te = [i for i in ids if fold_map[i] == k]
        va = [i for i in ids if fold_map[i] == (k + 1) % 5]
        tr = [i for i in ids if fold_map[i] not in (k, (k + 1) % 5)]
        if len(tr) < 20 or len(va) < 5 or len(te) < 5:
            raise RuntimeError("fold sizes")
        Xtr, Xva, Xte = _preprocess_fold(X_all, tr, va, te, groups)
        ytr = np.array([y_map[i] for i in tr], float)
        yva = np.array([y_map[i] for i in va], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if estimator == "ridge":
                best_a, best_m = None, 1e18
                for a in ALPHAS:
                    m = Ridge(alpha=a, random_state=0)
                    m.fit(Xtr, ytr)
                    v = mae(yva, m.predict(Xva))
                    if v < best_m:
                        best_a, best_m = a, v
                Xtv = np.vstack([Xtr, Xva])
                ytv = np.concatenate([ytr, yva])
                # rebuild preprocess on tr+va for te (match historical med2 on ids_tv)
                Xtv2, _, Xte2 = _preprocess_fold(X_all, tr + va, tr + va, te, groups)
                model = Ridge(alpha=best_a, random_state=0).fit(Xtv2, ytv)
                pred = model.predict(Xte2)
                param = {"alpha": best_a}
            elif estimator == "lasso":
                best_a, best_m = None, 1e18
                for a in LASSO_ALPHAS:
                    m = Lasso(alpha=a, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL, random_state=0)
                    m.fit(Xtr, ytr)
                    v = mae(yva, m.predict(Xva))
                    if v < best_m:
                        best_a, best_m = a, v
                ytv = np.concatenate([ytr, yva])
                Xtv2, _, Xte2 = _preprocess_fold(X_all, tr + va, tr + va, te, groups)
                model = Lasso(
                    alpha=best_a, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL, random_state=0
                ).fit(Xtv2, ytv)
                pred = model.predict(Xte2)
                param = {"alpha": best_a}
            elif estimator == "enet":
                grid = [(a, r) for a in [1e-3, 1e-2, 1e-1, 1.0] for r in [0.15, 0.5, 0.85]]
                best, best_m = None, 1e18
                for a, r in grid:
                    m = ElasticNet(alpha=a, l1_ratio=r, max_iter=100_000, tol=1e-3, random_state=0)
                    m.fit(Xtr, ytr)
                    v = mae(yva, m.predict(Xva))
                    if v < best_m:
                        best, best_m = (a, r), v
                ytv = np.concatenate([ytr, yva])
                Xtv2, _, Xte2 = _preprocess_fold(X_all, tr + va, tr + va, te, groups)
                model = ElasticNet(
                    alpha=best[0], l1_ratio=best[1], max_iter=100_000, tol=1e-3, random_state=0
                ).fit(Xtv2, ytv)
                pred = model.predict(Xte2)
                param = {"alpha": best[0], "l1_ratio": best[1]}
            elif estimator == "svr":
                grid = [(C, g, e) for C in [1.0, 10.0, 100.0] for g in ["scale", 0.01] for e in [0.05, 0.1]]
                best, best_m = None, 1e18
                for C, g, e in grid:
                    m = SVR(C=C, gamma=g, epsilon=e, kernel="rbf")
                    m.fit(Xtr, ytr)
                    v = mae(yva, m.predict(Xva))
                    if v < best_m:
                        best, best_m = (C, g, e), v
                ytv = np.concatenate([ytr, yva])
                Xtv2, _, Xte2 = _preprocess_fold(X_all, tr + va, tr + va, te, groups)
                model = SVR(C=best[0], gamma=best[1], epsilon=best[2], kernel="rbf").fit(Xtv2, ytv)
                pred = model.predict(Xte2)
                param = {"C": best[0], "gamma": best[1], "epsilon": best[2]}
            elif estimator == "xgb":
                import xgboost as xgb

                dtr = xgb.DMatrix(Xtr, label=ytr)
                dva = xgb.DMatrix(Xva, label=yva)
                params = dict(
                    max_depth=2,
                    min_child_weight=5,
                    subsample=0.9,
                    colsample_bytree=0.7,
                    eta=0.02,
                    reg_lambda=10,
                    objective="reg:squarederror",
                    eval_metric="mae",
                    tree_method="hist",
                    seed=0,
                )
                booster = xgb.train(
                    params,
                    dtr,
                    num_boost_round=5000,
                    evals=[(dva, "va")],
                    early_stopping_rounds=200,
                    verbose_eval=False,
                )
                best_iter = booster.best_iteration + 1
                ytv = np.concatenate([ytr, yva])
                Xtv2, _, Xte2 = _preprocess_fold(X_all, tr + va, tr + va, te, groups)
                booster2 = xgb.train(
                    params, xgb.DMatrix(Xtv2, label=ytv), num_boost_round=best_iter, verbose_eval=False
                )
                pred = booster2.predict(xgb.DMatrix(Xte2))
                param = {"best_iteration": best_iter}
            else:
                raise ValueError(estimator)
        for ab, p in zip(te, pred):
            oof.loc[ab] = float(p)
        fold_params.append(param)
    y_true = np.array([y_map[i] for i in ids], float)
    y_hat = oof.loc[ids].to_numpy(float)
    return {
        "mae": mae(y_true, y_hat),
        "oof": oof,
        "fold_params": fold_params,
        "ids": ids,
    }


def evaluate_primary_shadow(
    X_all: pd.DataFrame,
    y_map: dict[str, float],
    primary: dict[str, int],
    shadow: dict[str, int],
    estimator: str = "ridge",
    groups: list[ColumnGroup] | None = None,
) -> dict:
    p = run_simple_tvt(X_all, y_map, primary, estimator=estimator, groups=groups)
    s = run_simple_tvt(X_all, y_map, shadow, estimator=estimator, groups=groups)
    return {
        "cv_primary_mae": p["mae"],
        "cv_shadow_mae": s["mae"],
        "cv_mean_mae": 0.5 * (p["mae"] + s["mae"]),
        "cv_worst_mae": max(p["mae"], s["mae"]),
        "oof_primary": p["oof"],
        "oof_shadow": s["oof"],
        "params_primary": p["fold_params"],
        "params_shadow": s["fold_params"],
    }


def fit_full_dev_predict(
    X_all: pd.DataFrame,
    y_map: dict[str, float],
    fold_map: dict[str, int],
    test_ids: list[str],
    *,
    estimator: str = "ridge",
    groups: list[ColumnGroup] | None = None,
) -> np.ndarray:
    """Hyperparams via Dev CV fold0; final fit on all Dev → Test."""
    groups = groups if groups is not None else _default_groups(X_all)
    cv = run_simple_tvt(X_all, y_map, fold_map, estimator=estimator, groups=groups)
    param = cv["fold_params"][0]
    dev_ids = [i for i in X_all["id"].astype(str).tolist() if i in y_map and i in fold_map]
    # preprocess: treat all-dev as train, dummy va=train, te=test
    Xtr, _, Xte = _preprocess_fold(X_all, dev_ids, dev_ids, test_ids, groups)
    ytr = np.array([y_map[i] for i in dev_ids], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if estimator == "ridge":
            return Ridge(alpha=param["alpha"], random_state=0).fit(Xtr, ytr).predict(Xte)
        if estimator == "lasso":
            return (
                Lasso(alpha=param["alpha"], max_iter=LASSO_MAX_ITER, tol=LASSO_TOL, random_state=0)
                .fit(Xtr, ytr)
                .predict(Xte)
            )
        if estimator == "enet":
            return (
                ElasticNet(
                    alpha=param["alpha"],
                    l1_ratio=param["l1_ratio"],
                    max_iter=100_000,
                    tol=1e-3,
                    random_state=0,
                )
                .fit(Xtr, ytr)
                .predict(Xte)
            )
        if estimator == "svr":
            return (
                SVR(C=param["C"], gamma=param["gamma"], epsilon=param["epsilon"], kernel="rbf")
                .fit(Xtr, ytr)
                .predict(Xte)
            )
        if estimator == "xgb":
            import xgboost as xgb

            params_x = dict(
                max_depth=2,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.7,
                eta=0.02,
                reg_lambda=10,
                objective="reg:squarederror",
                eval_metric="mae",
                tree_method="hist",
                seed=0,
            )
            booster = xgb.train(
                params_x,
                xgb.DMatrix(Xtr, label=ytr),
                num_boost_round=int(param["best_iteration"]),
                verbose_eval=False,
            )
            return booster.predict(xgb.DMatrix(Xte))
        raise ValueError(estimator)
