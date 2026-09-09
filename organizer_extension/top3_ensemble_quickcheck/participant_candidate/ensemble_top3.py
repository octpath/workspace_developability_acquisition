#!/usr/bin/env python3
"""Participant-facing Top-3 ensemble helper (candidate; not yet copied into bundle).

Uses only files distributed in top_models_feature_bundle/.
Supports: equal_mean, median3, convex_stack, ridge_stack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Expect to be run with bundle on PYTHONPATH or from bundle after future integration.
def _import_bundle(bundle: Path):
    sys.path.insert(0, str(bundle))
    import reproduce_top_recipes as rtr
    return rtr


def convex_weights(P, y):
    starts = [np.ones(3)/3, np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]]
    best_w, best = None, np.inf
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    for w0 in starts:
        res = minimize(
            lambda w: float(np.mean(np.abs(y - P @ w))),
            w0, method="SLSQP", bounds=[(0,1)]*3, constraints=cons,
            options={"ftol": 1e-14, "maxiter": 2000, "disp": False},
        )
        w = np.clip(res.x, 0, 1); w = w / w.sum()
        m = float(np.mean(np.abs(y - P @ w)))
        if m < best:
            best, best_w = m, w
    return best_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--target", choices=["TmApp", "HIC"], required=True)
    ap.add_argument("--method", choices=["equal_mean", "median3", "convex_stack", "ridge_stack"], required=True)
    ap.add_argument("--subset", default="ALL", help="e.g. T1+T3 or ALL")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bundle = Path(args.bundle).resolve()
    rtr = _import_bundle(bundle)
    # Reproduce OOF + test via bundle API by shelling into reproduce outputs if present,
    # else instruct user to run reproduce_top_recipes.py first.
    out_pred = bundle / "outputs"
    recipes = pd.read_csv(bundle / "recipes.csv")
    recs = recipes[recipes.target == args.target].sort_values("cv_rank")
    models = recs.recipe_id.tolist()
    short = {m: f"{'T' if args.target=='TmApp' else 'H'}{i+1}" for i, m in enumerate(models)}
    inv = {v: k for k, v in short.items()}

    def load_oof(rid, scheme):
        p = out_pred / f"{rid}__cv_predictions_{scheme}.csv"
        if not p.exists():
            raise SystemExit(f"Missing {p}; run reproduce_top_recipes.py first")
        df = pd.read_csv(p); df["id"] = df["id"].astype(str)
        return df.set_index("id")["prediction"].astype(float)

    def load_test(rid):
        p = out_pred / f"{rid}__test_predictions.csv"
        df = pd.read_csv(p); df["id"] = df["id"].astype(str)
        return df.set_index("id")["prediction"].astype(float)

    if args.subset == "ALL":
        use = models
    else:
        use = [inv[s] if s in inv else s for s in args.subset.split("+")]

    test_ids = pd.read_csv(bundle / "test.csv")["id"].astype(str).tolist()
    if args.method == "equal_mean":
        mats = np.vstack([load_test(m).reindex(test_ids).to_numpy() for m in use])
        pred = mats.mean(axis=0)
    elif args.method == "median3":
        mats = np.vstack([load_test(m).reindex(test_ids).to_numpy() for m in models])
        pred = np.median(mats, axis=0)
    elif args.method == "convex_stack":
        dev = pd.read_csv(bundle / "dev.csv"); dev["id"] = dev["id"].astype(str)
        ids = dev["id"].tolist()
        y = dev.set_index("id")[args.target].astype(float).reindex(ids).to_numpy()
        P = np.column_stack([load_oof(m, "primary").reindex(ids).to_numpy() for m in models])
        w = convex_weights(P, y)
        Pte = np.column_stack([load_test(m).reindex(test_ids).to_numpy() for m in models])
        pred = Pte @ w
        print("weights", w.tolist())
    else:
        dev = pd.read_csv(bundle / "dev.csv"); dev["id"] = dev["id"].astype(str)
        ids = dev["id"].tolist()
        y = dev.set_index("id")[args.target].astype(float).reindex(ids).to_numpy()
        P = np.column_stack([load_oof(m, "primary").reindex(ids).to_numpy() for m in models])
        sc = StandardScaler(); Z = sc.fit_transform(P)
        mdl = Ridge(alpha=args.ridge_alpha, random_state=0).fit(Z, y)
        Pte = np.column_stack([load_test(m).reindex(test_ids).to_numpy() for m in models])
        pred = mdl.predict(sc.transform(Pte))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_ids, "prediction": pred}).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
