#!/usr/bin/env python3
"""Fold-local shallow XGBoost estimator matching Transformer V3 TVT semantics.

Key invariants (must match DL_FOLDLOCAL_COSINE_V3 evaluation meaning):
  - existing Primary / Shadow fold maps
  - per-fold TRAIN/VAL/TEST (held-out Dev) split via tvt_split
  - TRAIN-only median imputation (no StandardScaler)
  - early stopping on VAL only
  - one saved model per (scheme, fold); no full-Dev refit
  - OOF VAL / OOF TEST from the corresponding fold model
  - external Test predictions aggregated as mean/median across fold models
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping

from antibody_transformer.data import FoldMaps, tvt_split
from antibody_transformer.metrics import mae


@dataclass(frozen=True)
class XGBConfig:
    learning_rate: float = 0.02
    max_depth: int = 2
    n_estimators: int = 5000
    early_stopping_rounds: int = 200
    min_child_weight: float = 5.0
    subsample: float = 0.9
    colsample_bytree: float = 0.7
    reg_lambda: float = 10.0
    random_state: int = 101
    objective: str = "reg:squarederror"
    eval_metric: str = "mae"
    tree_method: str = "hist"


def median_impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(np.float64)


def median_impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64).copy()
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(med, inds[1])
    return out


def make_xgb(cfg: XGBConfig) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective=cfg.objective,
        eval_metric=cfg.eval_metric,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        n_estimators=cfg.n_estimators,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        tree_method=cfg.tree_method,
        n_jobs=1,
        callbacks=[
            EarlyStopping(
                rounds=cfg.early_stopping_rounds,
                save_best=True,
                maximize=False,
            )
        ],
    )


@dataclass
class FoldModelArtifact:
    scheme: str
    fold: int
    best_iteration: int
    best_val_mae: float
    n_train: int
    n_val: int
    feature_columns: list[str]
    feature_hash: str
    hyperparams: dict[str, Any]
    median: list[float]
    model_path: str
    seed: int


def train_fold_local_xgb(
    *,
    experiment_code: str,
    X_by_id: pd.DataFrame,
    y_map: dict[str, float],
    folds: FoldMaps,
    test_ids: list[str],
    feature_columns: list[str],
    feature_hash: str,
    cfg: XGBConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """Train Primary+Shadow fold models; write OOF/TEST/external predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    dev_ids = [i for i in X_by_id.index.astype(str).tolist() if i in y_map]
    assert len(dev_ids) == 162

    oof_val = {
        "primary": pd.Series(np.nan, index=dev_ids, dtype=float),
        "shadow": pd.Series(np.nan, index=dev_ids, dtype=float),
    }
    oof_test = {
        "primary": pd.Series(np.nan, index=dev_ids, dtype=float),
        "shadow": pd.Series(np.nan, index=dev_ids, dtype=float),
    }
    ext_folds = {
        "primary": np.zeros((5, len(test_ids)), dtype=float),
        "shadow": np.zeros((5, len(test_ids)), dtype=float),
    }
    artifacts: list[FoldModelArtifact] = []
    selected_rows: list[dict] = []

    X_mat = X_by_id.loc[:, feature_columns]

    for scheme_name, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
        for k in range(5):
            tr, va, te = tvt_split(fmap, k, dev_ids)
            Xtr_raw = X_mat.loc[tr].to_numpy(float)
            Xva_raw = X_mat.loc[va].to_numpy(float)
            Xte_raw = X_mat.loc[te].to_numpy(float)
            Xext_raw = X_mat.loc[test_ids].to_numpy(float)
            ytr = np.asarray([y_map[a] for a in tr], float)
            yva = np.asarray([y_map[a] for a in va], float)
            yte = np.asarray([y_map[a] for a in te], float)

            med = median_impute_fit(Xtr_raw)
            Xtr = median_impute_apply(Xtr_raw, med)
            Xva = median_impute_apply(Xva_raw, med)
            Xte = median_impute_apply(Xte_raw, med)
            Xext = median_impute_apply(Xext_raw, med)

            model = make_xgb(cfg)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            best_iter = int(model.best_iteration)
            pred_va = model.predict(Xva)
            pred_te = model.predict(Xte)
            pred_ext = model.predict(Xext)
            best_val = float(mae(yva, pred_va))

            model_path = model_dir / f"{scheme_name}_k{k}_seed{cfg.random_state}.json"
            model.save_model(model_path)

            art = FoldModelArtifact(
                scheme=scheme_name,
                fold=k,
                best_iteration=best_iter,
                best_val_mae=best_val,
                n_train=len(tr),
                n_val=len(va),
                feature_columns=list(feature_columns),
                feature_hash=feature_hash,
                hyperparams=asdict(cfg),
                median=med.tolist(),
                model_path=str(model_path.relative_to(out_dir)),
                seed=cfg.random_state,
            )
            artifacts.append(art)
            meta_path = model_dir / f"{scheme_name}_k{k}_seed{cfg.random_state}.meta.json"
            meta_path.write_text(json.dumps(asdict(art), indent=2) + "\n")

            oof_val[scheme_name].loc[va] = pred_va
            oof_test[scheme_name].loc[te] = pred_te
            ext_folds[scheme_name][k] = pred_ext
            selected_rows.append(
                {
                    "scheme": scheme_name,
                    "fold": k,
                    "best_iteration": best_iter,
                    "best_val_mae": best_val,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "model_path": str(model_path),
                }
            )
            print(
                f"=== {experiment_code} {scheme_name} fold={k} "
                f"best_iter={best_iter} VAL_MAE={best_val:.6f} ===",
                flush=True,
            )

    y_dev = np.asarray([y_map[a] for a in dev_ids], float)
    scores = {
        "oof_val": {
            "primary": float(mae(y_dev, oof_val["primary"].to_numpy(float))),
            "shadow": float(mae(y_dev, oof_val["shadow"].to_numpy(float))),
        },
        "oof_test": {
            "primary": float(mae(y_dev, oof_test["primary"].to_numpy(float))),
            "shadow": float(mae(y_dev, oof_test["shadow"].to_numpy(float))),
        },
    }
    for split in ("oof_val", "oof_test"):
        p, s = scores[split]["primary"], scores[split]["shadow"]
        scores[split]["mean"] = 0.5 * (p + s)
        scores[split]["worst"] = max(p, s)

    ext = {
        "primary_mean": ext_folds["primary"].mean(axis=0),
        "primary_median": np.median(ext_folds["primary"], axis=0),
        "shadow_mean": ext_folds["shadow"].mean(axis=0),
        "shadow_median": np.median(ext_folds["shadow"], axis=0),
        "primary_folds": ext_folds["primary"],
        "shadow_folds": ext_folds["shadow"],
    }

    best_iters = [a.best_iteration for a in artifacts]
    summary = {
        "experiment_code": experiment_code,
        "estimator": "xgboost_shallow_fold_local",
        "feature_hash": feature_hash,
        "feature_columns": feature_columns,
        "cfg": asdict(cfg),
        "scores": scores,
        "best_iteration_mean": float(np.mean(best_iters)),
        "best_iteration_median": float(np.median(best_iters)),
        "fold_artifacts": [asdict(a) for a in artifacts],
        "no_full_dev_refit": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(selected_rows).to_csv(out_dir / "selected_lr.csv", index=False)
    # naming parallel to Transformer SELECTED_LR even though LR is fixed
    return {
        "summary": summary,
        "selected_df": pd.DataFrame(selected_rows),
        "oof_val": oof_val,
        "oof_test": oof_test,
        "ext": ext,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "y_dev": y_dev,
    }


def reload_fold_model(out_dir: Path, scheme: str, fold: int, seed: int = 101) -> tuple[xgb.XGBRegressor, np.ndarray]:
    meta = json.loads((out_dir / "fold_models" / f"{scheme}_k{fold}_seed{seed}.meta.json").read_text())
    model = xgb.XGBRegressor()
    model.load_model(out_dir / meta["model_path"])
    return model, np.asarray(meta["median"], float)
