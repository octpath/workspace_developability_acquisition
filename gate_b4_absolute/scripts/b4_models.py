#!/usr/bin/env python3
"""Model fitters for Gate B4 with correct GBDT early-stopping protocol."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

from b4_common import (
    GBDT_EARLY_STOPPING_ROUNDS,
    GBDT_LEARNING_RATE,
    GBDT_N_ESTIMATORS,
    MASTER_SEED,
    es_split_by_group,
    sanitize_pair,
)


def fit_predict_sklearn(kind: str, params: dict, Xtr, ytr, Xte, seed: int = MASTER_SEED):
    Xtr, Xte = sanitize_pair(Xtr, Xte)
    ytr = np.asarray(ytr, float)
    if kind == "Ridge":
        m = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=params.get("alpha", 10.0)))])
    elif kind == "Lasso":
        m = Pipeline(
            [("sc", StandardScaler()), ("m", Lasso(alpha=params.get("alpha", 0.01), max_iter=8000))]
        )
    elif kind == "ElasticNet":
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    ElasticNet(
                        alpha=params.get("alpha", 0.05),
                        l1_ratio=params.get("l1_ratio", 0.5),
                        max_iter=8000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    elif kind == "Huber":
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    HuberRegressor(
                        epsilon=params.get("epsilon", 1.35),
                        alpha=params.get("alpha", 1e-4),
                        max_iter=500,
                    ),
                ),
            ]
        )
    elif kind == "LinearSVR":
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", LinearSVR(C=params.get("C", 1.0), epsilon=params.get("epsilon", 0.0), max_iter=8000)),
            ]
        )
    elif kind == "SVR_RBF":
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    SVR(
                        kernel="rbf",
                        C=params.get("C", 10.0),
                        epsilon=params.get("epsilon", 0.1),
                        gamma=params.get("gamma", "scale"),
                    ),
                ),
            ]
        )
    elif kind == "KRR":
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    KernelRidge(
                        alpha=params.get("alpha", 1.0),
                        kernel="rbf",
                        gamma=params.get("gamma", None),
                    ),
                ),
            ]
        )
    elif kind == "PLS":
        ncomp = int(params.get("n_components", 8))
        ncomp = max(1, min(ncomp, Xtr.shape[0] - 2, Xtr.shape[1]))
        m = Pipeline([("sc", StandardScaler()), ("m", PLSRegression(n_components=ncomp))])
    else:
        raise ValueError(kind)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte)).ravel()


def fit_predict_pca_head(kind: str, params: dict, Xtr, ytr, Xte, seed: int = MASTER_SEED):
    Xtr, Xte = sanitize_pair(Xtr, Xte)
    npc = int(params["n_components"])
    npc = max(1, min(npc, Xtr.shape[0] - 1, Xtr.shape[1]))
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)
    Zte = sc.transform(Xte)
    pca = PCA(n_components=npc, random_state=seed)
    Ztr = pca.fit_transform(Ztr)
    Zte = pca.transform(Zte)
    head = {k: v for k, v in params.items() if k != "n_components"}
    return fit_predict_sklearn(kind, head, Ztr, ytr, Zte, seed=seed)


def _xgb_params(params: dict, objective: str, seed: int) -> dict:
    return {
        "learning_rate": GBDT_LEARNING_RATE,
        "n_estimators": GBDT_N_ESTIMATORS,
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "gamma": float(params.get("gamma", 0.0)),
        "objective": objective,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": 4,
    }


def fit_predict_xgb(
    params: dict,
    Xtr,
    ytr,
    Xte,
    groups_tr,
    objective: str = "reg:squarederror",
    seed: int = MASTER_SEED,
    fixed_rounds: int | None = None,
) -> tuple[np.ndarray, dict]:
    import xgboost as xgb

    Xtr, Xte = sanitize_pair(Xtr, Xte)
    ytr = np.asarray(ytr, float)
    p = _xgb_params(params, objective, seed)
    meta: dict[str, Any] = {"family": "xgboost", "objective": objective, "learning_rate": GBDT_LEARNING_RATE}

    if fixed_rounds is not None:
        p["n_estimators"] = int(fixed_rounds)
        model = xgb.XGBRegressor(**p)
        model.fit(Xtr, ytr)
        meta["best_iteration"] = int(fixed_rounds)
        meta["hit_ceiling"] = int(fixed_rounds) >= GBDT_N_ESTIMATORS
        return np.asarray(model.predict(Xte)).ravel(), meta

    es = es_split_by_group(groups_tr, seed=seed + 91)
    # ensure both sides non-empty
    if es.sum() < 3 or (~es).sum() < 5:
        es = np.zeros(len(ytr), dtype=bool)
        es[-max(3, len(ytr) // 6) :] = True
    p_es = dict(p)
    p_es["early_stopping_rounds"] = GBDT_EARLY_STOPPING_ROUNDS
    model = xgb.XGBRegressor(**p_es)
    model.fit(
        Xtr[~es],
        ytr[~es],
        eval_set=[(Xtr[es], ytr[es])],
        verbose=False,
    )
    # best_iteration is 0-based in xgboost; n_estimators for refit is +1
    bi = int(model.best_iteration) + 1
    bi = max(1, min(bi, GBDT_N_ESTIMATORS))
    meta["best_iteration"] = bi
    meta["hit_ceiling"] = bi >= GBDT_N_ESTIMATORS
    # Refit on ALL Xtr for bi rounds (ES set never touches outer validation)
    p2 = dict(p)
    p2["n_estimators"] = bi
    final = xgb.XGBRegressor(**p2)
    final.fit(Xtr, ytr)
    return np.asarray(final.predict(Xte)).ravel(), meta


def fit_predict_lgb(
    params: dict,
    Xtr,
    ytr,
    Xte,
    groups_tr,
    objective: str = "regression",
    seed: int = MASTER_SEED,
    fixed_rounds: int | None = None,
) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb

    Xtr, Xte = sanitize_pair(Xtr, Xte)
    ytr = np.asarray(ytr, float)
    max_depth = int(params["max_depth"])
    num_leaves = int(params["num_leaves"])
    # prevent invalid huge leaves vs depth
    if max_depth > 0:
        num_leaves = min(num_leaves, 2 ** max_depth)
    p = {
        "learning_rate": GBDT_LEARNING_RATE,
        "n_estimators": GBDT_N_ESTIMATORS if fixed_rounds is None else int(fixed_rounds),
        "max_depth": max_depth,
        "num_leaves": max(2, num_leaves),
        "min_child_samples": int(params["min_child_samples"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "min_split_gain": float(params.get("min_split_gain", 0.0)),
        "objective": objective,
        "random_state": seed,
        "n_jobs": 4,
        "verbosity": -1,
    }
    meta: dict[str, Any] = {"family": "lightgbm", "objective": objective, "learning_rate": GBDT_LEARNING_RATE}
    if fixed_rounds is not None:
        model = lgb.LGBMRegressor(**p)
        model.fit(Xtr, ytr)
        meta["best_iteration"] = int(fixed_rounds)
        meta["hit_ceiling"] = int(fixed_rounds) >= GBDT_N_ESTIMATORS
        return np.asarray(model.predict(Xte)).ravel(), meta

    es = es_split_by_group(groups_tr, seed=seed + 92)
    if es.sum() < 3 or (~es).sum() < 5:
        es = np.zeros(len(ytr), dtype=bool)
        es[-max(3, len(ytr) // 6) :] = True
    model = lgb.LGBMRegressor(**p)
    model.fit(
        Xtr[~es],
        ytr[~es],
        eval_set=[(Xtr[es], ytr[es])],
        callbacks=[
            lgb.early_stopping(GBDT_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    bi = int(model.best_iteration_) if model.best_iteration_ else GBDT_N_ESTIMATORS
    bi = max(1, min(bi, GBDT_N_ESTIMATORS))
    meta["best_iteration"] = bi
    meta["hit_ceiling"] = bi >= GBDT_N_ESTIMATORS
    p2 = dict(p)
    p2["n_estimators"] = bi
    final = lgb.LGBMRegressor(**p2)
    final.fit(Xtr, ytr)
    return np.asarray(final.predict(Xte)).ravel(), meta


def fit_predict_cat(
    params: dict,
    Xtr,
    ytr,
    Xte,
    groups_tr,
    loss: str = "RMSE",
    seed: int = MASTER_SEED,
    fixed_rounds: int | None = None,
) -> tuple[np.ndarray, dict]:
    from catboost import CatBoostRegressor

    Xtr, Xte = sanitize_pair(Xtr, Xte)
    ytr = np.asarray(ytr, float)
    p = {
        "learning_rate": GBDT_LEARNING_RATE,
        "iterations": GBDT_N_ESTIMATORS if fixed_rounds is None else int(fixed_rounds),
        "depth": int(params["depth"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "random_strength": float(params.get("random_strength", 1.0)),
        "bagging_temperature": float(params.get("bagging_temperature", 1.0)),
        "rsm": float(params.get("rsm", 1.0)),
        "loss_function": loss,
        "random_seed": seed,
        "thread_count": 4,
        "verbose": False,
        "allow_writing_files": False,
    }
    meta: dict[str, Any] = {"family": "catboost", "objective": loss, "learning_rate": GBDT_LEARNING_RATE}
    if fixed_rounds is not None:
        model = CatBoostRegressor(**p)
        model.fit(Xtr, ytr)
        meta["best_iteration"] = int(fixed_rounds)
        meta["hit_ceiling"] = int(fixed_rounds) >= GBDT_N_ESTIMATORS
        return np.asarray(model.predict(Xte)).ravel(), meta

    es = es_split_by_group(groups_tr, seed=seed + 93)
    if es.sum() < 3 or (~es).sum() < 5:
        es = np.zeros(len(ytr), dtype=bool)
        es[-max(3, len(ytr) // 6) :] = True
    p_od = dict(p)
    p_od["od_type"] = "Iter"
    p_od["od_wait"] = GBDT_EARLY_STOPPING_ROUNDS
    p_od["use_best_model"] = True
    model = CatBoostRegressor(**p_od)
    model.fit(Xtr[~es], ytr[~es], eval_set=(Xtr[es], ytr[es]))
    bi = int(model.get_best_iteration()) + 1 if model.get_best_iteration() is not None else GBDT_N_ESTIMATORS
    bi = max(1, min(bi, GBDT_N_ESTIMATORS))
    meta["best_iteration"] = bi
    meta["hit_ceiling"] = bi >= GBDT_N_ESTIMATORS
    p2 = dict(p)
    p2["iterations"] = bi
    final = CatBoostRegressor(**p2)
    final.fit(Xtr, ytr)
    return np.asarray(final.predict(Xte)).ravel(), meta


def suggest_xgb(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 300.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
    }


def suggest_lgb(trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 2, 5)
    return {
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 3, min(31, 2 ** max_depth)),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 300.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def suggest_cat(trial) -> dict:
    return {
        "depth": trial.suggest_int("depth", 3, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.35, 1.0),
    }


def suggest_elasticnet(trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
    }


def suggest_svr(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


def suggest_krr(trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }
