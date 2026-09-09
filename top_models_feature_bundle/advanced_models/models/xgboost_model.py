#!/usr/bin/env python3
"""XGBoost GPU/CPU helpers for frozen Top-3 recipes."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import xgboost as xgb

from ..config import load_presets


def make_xgb_params(preset_name: str, device: str = "cuda") -> dict[str, Any]:
    presets = load_presets()["xgboost"]
    p = dict(presets["presets"][preset_name])
    p.update(
        {
            "learning_rate": presets["learning_rate"],
            "n_estimators": presets["n_estimators"],
            "objective": "reg:absoluteerror",
            "eval_metric": presets["eval_metric"],
            "tree_method": "hist",
            "device": "cuda" if device.startswith("cuda") else "cpu",
            "random_state": 0,
            "n_jobs": 4,
        }
    )
    return p


def fit_xgb_early(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    preset_name: str,
    device: str = "cuda",
) -> tuple[xgb.XGBRegressor, int, float]:
    params = make_xgb_params(preset_name, device)
    early = load_presets()["xgboost"]["early_stopping_rounds"]
    model = xgb.XGBRegressor(**params, early_stopping_rounds=early)
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        verbose=False,
    )
    best_iter = int(model.best_iteration)
    # VAL MAE at best
    pred = model.predict(Xva)
    val_mae = float(np.mean(np.abs(pred - yva)))
    return model, best_iter, val_mae


def fit_xgb_rounds(
    X: np.ndarray,
    y: np.ndarray,
    preset_name: str,
    n_estimators: int,
    device: str = "cuda",
) -> xgb.XGBRegressor:
    params = make_xgb_params(preset_name, device)
    params["n_estimators"] = int(n_estimators)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model
