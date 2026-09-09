"""Shared helpers for Top-3-only ensemble / stacking quickcheck.

Reads from top_models_feature_bundle only; never writes into it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
QC_ROOT = HERE.parent
REPO_ROOT = QC_ROOT.parent.parent

TM_MODELS = [
    "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
    "TM_BASE_BIOEMU_MPNN__RIDGE",
]
HIC_MODELS = [
    "HIC_HYDRO_TITRATION__LASSO",
    "HIC_ARO_CONTINUOUS_SURFACE__LASSO",
    "HIC_ESM2_SEQ_AROMATIC__LASSO",
]
MODELS_BY_TARGET = {"TmApp": TM_MODELS, "HIC": HIC_MODELS}
SHORT = {
    "TM_PARENT_ABLINGUA_CDR3__RIDGE": "T1",
    "TM_PARENT_ABLINGUA_GLOBAL__RIDGE": "T2",
    "TM_BASE_BIOEMU_MPNN__RIDGE": "T3",
    "HIC_HYDRO_TITRATION__LASSO": "H1",
    "HIC_ARO_CONTINUOUS_SURFACE__LASSO": "H2",
    "HIC_ESM2_SEQ_AROMATIC__LASSO": "H3",
}
RIDGE_META_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260909
TOL_PASS = 1e-8


class SolutionGuard:
    """Hard-fail if solution labels are read before scoring is enabled."""

    def __init__(self):
        self.enabled = False
        self.accessed_early = False
        self._path: Path | None = None
        self._df: pd.DataFrame | None = None

    def load(self, path: Path | None):
        if path is None:
            return
        self._path = Path(path)
        self._df = None  # lazy — do not read yet

    def enable(self):
        self.enabled = True

    def get(self) -> pd.DataFrame:
        if not self.enabled:
            self.accessed_early = True
            raise RuntimeError("solution.csv accessed before final prediction freeze")
        if self._path is None:
            raise RuntimeError("no solution path configured")
        if self._df is None:
            self._df = pd.read_csv(self._path)
            self._df["id"] = self._df["id"].astype(str)
        return self._df

    @property
    def configured(self) -> bool:
        return self._path is not None


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def ensure_unique_ids(ids: list[str], label: str = "ids"):
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {label}")


def assert_no_bundle_writes(bundle: Path, before_mtimes: dict[str, float]):
    """Fail if any tracked file under bundle changed mtime during the run."""
    for rel, mt0 in before_mtimes.items():
        p = bundle / rel
        if not p.exists():
            continue
        mt1 = p.stat().st_mtime
        if mt1 != mt0:
            raise RuntimeError(f"bundle file modified during run: {rel}")


def snapshot_bundle_mtimes(bundle: Path) -> dict[str, float]:
    keys = [
        "recipes.csv",
        "reproduce_top_recipes.py",
        "bundle_simple_tvt.py",
        "dev.csv",
        "test.csv",
        "folds.csv",
        "EXPECTED_SCORES.json",
        "FULL_DEV_ALPHA_POLICY_BUNDLE.json",
    ]
    out = {}
    for k in keys:
        p = bundle / k
        if p.exists():
            out[k] = p.stat().st_mtime
    return out


def import_bundle(bundle: Path):
    bundle = bundle.resolve()
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))
    import reproduce_top_recipes as rtr  # noqa: WPS433
    import bundle_simple_tvt as tvt  # noqa: WPS433

    return rtr, tvt


def load_tables(bundle: Path):
    dev = pd.read_csv(bundle / "dev.csv")
    test = pd.read_csv(bundle / "test.csv")
    folds = pd.read_csv(bundle / "folds.csv")
    recipes = pd.read_csv(bundle / "recipes.csv")
    expected = json.loads((bundle / "EXPECTED_SCORES.json").read_text())
    for df in (dev, test, folds):
        df["id"] = df["id"].astype(str)
    ensure_unique_ids(dev["id"].tolist(), "dev")
    ensure_unique_ids(test["id"].tolist(), "test")
    if len(dev) != 162 or len(test) != 162:
        raise SystemExit(f"expected N=162; got dev={len(dev)} test={len(test)}")
    return dev, test, folds, recipes, expected


def recipe_row(recipes: pd.DataFrame, recipe_id: str) -> pd.Series:
    hit = recipes[recipes.recipe_id == recipe_id]
    if len(hit) != 1:
        raise KeyError(recipe_id)
    return hit.iloc[0]


def subset_parts(parts: dict, ids: list[str]) -> dict:
    ids = list(ids)
    ensure_unique_ids(ids)
    out = {"mode": parts["mode"], "ids": ids}
    if parts["mode"] == "standalone":
        out["X"] = parts["X"].loc[ids]
    elif parts["mode"] == "base_struct":
        out["base"] = parts["base"].loc[ids]
        out["struct"] = parts["struct"].loc[ids]
    else:
        out["recipe"] = parts["recipe"].loc[ids]
        out["always"] = parts["always"].loc[ids]
        out["extra"] = parts["extra"].loc[ids]
    return out


def all_nonempty_subsets(models: list[str]) -> list[tuple[str, ...]]:
    out = []
    for k in range(1, len(models) + 1):
        for c in combinations(models, k):
            out.append(tuple(c))
    if len(out) != 7:
        raise RuntimeError(f"expected 7 subsets, got {len(out)}")
    return out


def equal_mean_pred(pred_map: dict[str, pd.Series], models: tuple[str, ...], ids) -> pd.Series:
    mats = [pred_map[m].reindex(ids).astype(float) for m in models]
    for i, s in enumerate(mats):
        if s.isna().any():
            raise RuntimeError(f"missing predictions for {models[i]}")
    stacked = np.vstack([s.to_numpy() for s in mats])
    return pd.Series(stacked.mean(axis=0), index=list(ids))


def median3_pred(pred_map: dict[str, pd.Series], models: list[str], ids) -> pd.Series:
    if len(models) != 3:
        raise ValueError("median3 requires exactly 3 models")
    mats = np.vstack([pred_map[m].reindex(ids).astype(float).to_numpy() for m in models])
    if np.isnan(mats).any():
        raise RuntimeError("missing base predictions for median3")
    return pd.Series(np.median(mats, axis=0), index=list(ids))


def convex_mae_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nonnegative weights summing to 1 minimizing MAE; deterministic multi-start."""
    P = np.asarray(P, float)
    y = np.asarray(y, float)
    starts = [
        np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    best_w, best = None, np.inf

    def obj(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    bounds = [(0.0, 1.0)] * 3
    for w0 in starts:
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 2000, "disp": False},
        )
        w = np.asarray(res.x, float)
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        m = obj(w)
        if m < best - 1e-15 or (
            abs(m - best) <= 1e-15
            and best_w is not None
            and tuple(w) < tuple(best_w)
        ) or best_w is None:
            if m < best - 1e-15 or best_w is None:
                best, best_w = m, w
            elif abs(m - best) <= 1e-15 and tuple(w) < tuple(best_w):
                best_w = w
    if best_w is None:
        best_w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    # enforce constraints exactly
    best_w = np.clip(best_w, 0.0, 1.0)
    best_w = best_w / best_w.sum()
    if (best_w < -1e-12).any() or abs(best_w.sum() - 1.0) > 1e-8:
        raise RuntimeError(f"invalid convex weights: {best_w}")
    return best_w


def ridge_select_alpha(P: np.ndarray, y: np.ndarray, alphas=None, n_splits=3, seed=0) -> float:
    """3-fold CV alpha selection on meta VAL only."""
    alphas = list(alphas or RIDGE_META_ALPHAS)
    P = np.asarray(P, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < n_splits * 2:
        # tiny fallback: pick alpha by leave-ish mean absolute residual on full VAL fit
        best_a, best = alphas[0], np.inf
        for a in alphas:
            sc = StandardScaler()
            Z = sc.fit_transform(P)
            m = Ridge(alpha=a, random_state=0)
            m.fit(Z, y)
            score = mae(y, m.predict(Z))
            if score < best:
                best, best_a = score, a
        return float(best_a)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    means = []
    for a in alphas:
        fold_scores = []
        for tr, te in kf.split(P):
            sc = StandardScaler()
            Ztr = sc.fit_transform(P[tr])
            Zte = sc.transform(P[te])
            m = Ridge(alpha=a, random_state=0)
            m.fit(Ztr, y[tr])
            fold_scores.append(mae(y[te], m.predict(Zte)))
        means.append(float(np.mean(fold_scores)))
    # tie-break: larger alpha (more regularized)
    best = min(range(len(alphas)), key=lambda i: (means[i], -alphas[i]))
    return float(alphas[best])


def fit_ridge_meta(P: np.ndarray, y: np.ndarray, alpha: float):
    sc = StandardScaler()
    Z = sc.fit_transform(np.asarray(P, float))
    m = Ridge(alpha=float(alpha), random_state=0)
    m.fit(Z, np.asarray(y, float))
    return sc, m


def predict_ridge_meta(sc: StandardScaler, m: Ridge, P: np.ndarray) -> np.ndarray:
    return m.predict(sc.transform(np.asarray(P, float)))


def paired_bootstrap_mae_diff(
    abs_err_single: np.ndarray,
    abs_err_ens: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Distribution of MAE_single - MAE_ensemble (positive => ensemble better)."""
    a = np.asarray(abs_err_single, float)
    b = np.asarray(abs_err_ens, float)
    if len(a) != len(b):
        raise ValueError("length mismatch")
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(a[idx].mean() - b[idx].mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "mean_improvement": float(diffs.mean()),
        "median_improvement": float(np.median(diffs)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_improvement_gt_0": float(np.mean(diffs > 0)),
    }


def score_pp(pred: pd.Series, solution: pd.DataFrame, target: str) -> dict[str, float]:
    sol = solution.set_index("id")
    common = [i for i in pred.index if i in sol.index]
    y = sol.loc[common, target].astype(float)
    p = pred.loc[common].astype(float)
    pub = [i for i in common if bool(sol.loc[i, "is_public"])]
    priv = [i for i in common if bool(sol.loc[i, "is_private"])]
    return {
        "public_mae": mae(y.loc[pub], p.loc[pub]),
        "private_mae": mae(y.loc[priv], p.loc[priv]),
        "overall_test_mae": mae(y, p),
    }


def cv_stats(primary_mae: float, shadow_mae: float) -> dict[str, float]:
    return {
        "primary_mae": float(primary_mae),
        "shadow_mae": float(shadow_mae),
        "cv_mean_mae": float(0.5 * (primary_mae + shadow_mae)),
        "cv_worst_mae": float(max(primary_mae, shadow_mae)),
    }


def rank_key(row: dict[str, Any], n_models: int | None = None):
    nm = n_models if n_models is not None else row.get("n_models", 99)
    return (row["cv_worst_mae"], row["cv_mean_mae"], nm)


HISTORICAL_CONTEXT = {
    "HIC": {
        "name": "HIC__SIMPLE_blend_seq_surf_adv (organizer historical prediction blend)",
        "cv_primary": 0.4252,
        "cv_shadow": 0.4204,
        "public": 0.4239,
        "private": 0.4222,
        "note": "NOT a Top-3 candidate; descriptive comparison only. Protocol/features differ.",
    },
    "TmApp": {
        "name": "historical organizer meta/blend (descriptive only)",
        "note": "Historical TmApp blends used broader model pools / SVR meta; not comparable as candidates.",
    },
}
