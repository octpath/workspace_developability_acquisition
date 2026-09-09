#!/usr/bin/env python3
"""Metrics and CV selection helpers."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def seed_dispersion(seed_maes: list[float]) -> float:
    if len(seed_maes) < 2:
        return 0.0
    return float(np.std(np.asarray(seed_maes, float), ddof=0))


def cv_worst(primary: float, shadow: float) -> float:
    return float(max(primary, shadow))


def cv_mean(primary: float, shadow: float) -> float:
    return float(0.5 * (primary + shadow))


def select_best_rows(df: pd.DataFrame) -> pd.Series:
    """Select by cv_worst_mae then cv_mean_mae (lower better)."""
    if len(df) == 0:
        raise ValueError("empty selection frame")
    ranked = df.sort_values(["cv_worst_mae", "cv_mean_mae", "variant_id"])
    return ranked.iloc[0]


def score_solution(
    pred: pd.DataFrame,
    solution: pd.DataFrame,
    target: str,
) -> dict:
    """Post-hoc Public/Private MAE. pred: id,prediction."""
    sol = solution.copy()
    sol["id"] = sol["id"].astype(str)
    p = pred.copy()
    p["id"] = p["id"].astype(str)
    m = sol.merge(p, on="id", how="inner")
    if len(m) != len(sol):
        raise RuntimeError("prediction/solution id mismatch")
    y = m[target].to_numpy(float)
    hat = m["prediction"].to_numpy(float)
    pub = m["is_public"].astype(bool).to_numpy()
    priv = m["is_private"].astype(bool).to_numpy()
    if not np.all(pub ^ priv):
        raise RuntimeError("is_public XOR is_private violated")
    return {
        "public_mae": mae(y[pub], hat[pub]),
        "private_mae": mae(y[priv], hat[priv]),
        "overall_test_mae": mae(y, hat),
    }


def guarded_solution_path(path: Optional[str]) -> Optional[str]:
    return path
