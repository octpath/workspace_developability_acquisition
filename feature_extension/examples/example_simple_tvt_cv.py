#!/usr/bin/env python3
"""Example: Simple TVT rotation with organizer-recommended folds + Ridge.

For test fold k:
  TEST = fold k
  VAL  = fold (k+1) mod 5
  TRAIN = remaining three folds

Uses a participant-local training CSV (must include `id` and the chosen target).
Does not embed competition labels in this package.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def tvt_masks(folds: pd.Series, k: int, n_folds: int = 5):
    test = folds == k
    val = folds == ((k + 1) % n_folds)
    train = ~(test | val)
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True, help="CSV with id + target column")
    p.add_argument("--target", choices=["TmApp", "HIC"], required=True)
    p.add_argument("--folds", default=str(ROOT / "folds.csv"))
    p.add_argument(
        "--features",
        default=str(ROOT / "data/precomputed_features/aromatic_topology.parquet"),
        help="Feature parquet with id + numeric columns",
    )
    p.add_argument("--fold-column", default="fold_primary", choices=["fold_primary", "fold_shadow"])
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()

    train_df = pd.read_csv(args.train_csv)
    if args.target not in train_df.columns:
        raise SystemExit(f"train CSV missing target column {args.target}")
    folds = pd.read_csv(args.folds)
    feat = pd.read_parquet(args.features)

    df = train_df[["id", args.target]].merge(folds, on="id", how="inner").merge(feat, on="id", how="inner")
    y = df[args.target].to_numpy(dtype=float)
    fold = df[args.fold_column].astype(int)
    xcols = [c for c in feat.columns if c != "id"]
    X = df[xcols].to_numpy(dtype=float)
    # simple impute column median
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    rows = []
    for k in range(5):
        tr, va, te = tvt_masks(fold, k)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        # optional early stop on val (here: just report)
        Xva = scaler.transform(X[va]) if va.sum() else None
        Xte = scaler.transform(X[te])
        model = Ridge(alpha=args.alpha)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        mae = mean_absolute_error(y[te], pred)
        rows.append({"test_fold": k, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "mae": mae})
        if Xva is not None and va.sum():
            rows[-1]["val_mae"] = mean_absolute_error(y[va], model.predict(Xva))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("mean_test_mae", float(out["mae"].mean()))


if __name__ == "__main__":
    main()
