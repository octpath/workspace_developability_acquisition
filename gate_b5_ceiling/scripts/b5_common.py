#!/usr/bin/env python3
"""Shared utilities for Gate B5 ceiling check."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b5_ceiling"
B3 = ROOT / "gate_b3"
B4 = ROOT / "gate_b4_absolute"
B4_SCRIPTS = B4 / "scripts"

CONFIG = GATE / "config"
CACHE = GATE / "cache"
METRICS = GATE / "metrics"
PREDS = GATE / "predictions"
REPORTS = GATE / "reports"
LOGS = GATE / "logs"
SCRIPTS = GATE / "scripts"
MODELS = GATE / "models"
OPTUNA_DIR = GATE / "optuna"
PLOTS = GATE / "plots"

B3_ORG = B3 / "frozen" / "organizer"
B4_CONFIG = B4 / "config"
B4_METRICS = B4 / "metrics"

MASTER_SEED = 20260830
CV_SEED = 20260830
N_BOOT = 5000

sys.path.insert(0, str(B4_SCRIPTS))
from b4_common import (  # noqa: E402
    GBDT_LEARNING_RATE,
    group_kfold_labels,
    load_representation,
    load_train_frame,
    regression_metrics,
    sanitize,
    sanitize_pair,
    sha256_file,
    sha256_lines,
    software_versions as b4_software_versions,
    train_only,
)
from b4_models import (  # noqa: E402
    fit_predict_lgb,
    fit_predict_pca_head,
    fit_predict_sklearn,
    fit_predict_xgb,
)


def ensure_dirs():
    for p in [
        CONFIG, CACHE, METRICS, PREDS, REPORTS, LOGS, SCRIPTS, MODELS, OPTUNA_DIR, PLOTS,
        CACHE / "tokens", PREDS / "oof", PREDS / "final",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def set_gpu0():
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def set_seeds(seed: int = MASTER_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)


def fisher_z_mean(rs) -> float:
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    rs = np.clip(rs, -0.999999, 0.999999)
    if len(rs) == 0:
        return float("nan")
    z = np.arctanh(rs)
    return float(np.tanh(np.mean(z)))


def pearson_fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r) or n < 4:
        return float("nan"), float("nan")
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    from scipy.stats import norm
    zcrit = norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def bootstrap_pearson(y, p, n_boot: int = N_BOOT, seed: int = MASTER_SEED) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n = len(y)
    if n < 5:
        return {"r": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}
    r0 = pearsonr(y, p)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        if np.std(yy) < 1e-12 or np.std(pp) < 1e-12:
            continue
        rr = pearsonr(yy, pp)[0]
        if np.isfinite(rr):
            boots.append(rr)
    if not boots:
        lo, hi = pearson_fisher_ci(float(r0), n)
        return {"r": float(r0), "ci_lo": lo, "ci_hi": hi, "n": n, "method": "fisher_z_fallback"}
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"r": float(r0), "ci_lo": float(lo), "ci_hi": float(hi), "n": n, "method": "bootstrap"}


def bootstrap_delta_mae(y, pa, pb, n_boot: int = N_BOOT, seed: int = MASTER_SEED) -> dict:
    y, pa, pb = map(lambda a: np.asarray(a, float), (y, pa, pb))
    m = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb = y[m], pa[m], pb[m]
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    d0 = float(ea.mean() - eb.mean())  # A - B; negative => A better
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        boots.append(float(ea[idx].mean() - eb[idx].mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"delta_mae_A_minus_B": d0, "ci_lo": float(lo), "ci_hi": float(hi), "n": len(y)}


def bootstrap_delta_pearson(y, pa, pb, n_boot: int = N_BOOT, seed: int = MASTER_SEED) -> dict:
    y, pa, pb = map(lambda a: np.asarray(a, float), (y, pa, pb))
    m = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb = y[m], pa[m], pb[m]
    r0a = pearsonr(y, pa)[0]
    r0b = pearsonr(y, pb)[0]
    d0 = float(r0a - r0b)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yy = y[idx]
        if np.std(yy) < 1e-12:
            continue
        ra = pearsonr(yy, pa[idx])[0]
        rb = pearsonr(yy, pb[idx])[0]
        if np.isfinite(ra) and np.isfinite(rb):
            boots.append(float(ra - rb))
    lo, hi = np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan)
    return {"delta_pearson_A_minus_B": d0, "ci_lo": float(lo), "ci_hi": float(hi), "ra": float(r0a), "rb": float(r0b)}


def lins_ccc(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 5:
        return float("nan")
    r = pearsonr(y, p)[0]
    if not np.isfinite(r):
        return float("nan")
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(ddof=1), p.var(ddof=1)
    return float(2 * r * np.sqrt(vy) * np.sqrt(vp) / (vy + vp + (my - mp) ** 2))


def extended_metrics(y, p) -> dict:
    met = regression_metrics(y, p)
    met["ccc"] = lins_ccc(y, p)
    return met


def load_frozen_train():
    """Train frame with labels; holdout labels stripped from frame used during search."""
    outer = read_json(B4_CONFIG / "OUTER_CV_FOLDS.json")
    df = load_train_frame(include_holdout_labels=False)
    train = train_only(df).set_index("id").loc[outer["train_ids"]].reset_index()
    # restore train labels from population (barrier only strips non-train)
    pop = pd.read_csv(B3_ORG / "final_population.csv")
    train = train.drop(columns=["HIC", "TmApp"]).merge(pop[["id", "HIC", "TmApp"]], on="id")
    train = train.set_index("id").loc[outer["train_ids"]].reset_index()
    return train, outer


def software_versions() -> dict:
    out = b4_software_versions()
    try:
        import peft
        out["peft"] = peft.__version__
    except Exception:
        out["peft"] = None
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    return out
