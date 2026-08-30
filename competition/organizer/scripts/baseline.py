#!/usr/bin/env python3
"""Participant-legal baselines using only dev.csv + test_features.csv.

Default: Train-median constant predictor.
Optional: simple AA composition / length descriptors + Ridge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def aa_features(heavy: str, light: str) -> np.ndarray:
    seq = (heavy or "") + (light or "")
    n = max(len(seq), 1)
    counts = np.array([seq.count(a) / n for a in AA20], dtype=float)
    extra = np.array(
        [
            len(heavy or ""),
            len(light or ""),
            len(seq),
            abs(len(heavy or "") - len(light or "")),
        ],
        dtype=float,
    )
    return np.concatenate([counts, extra])


def median_baseline(dev: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    out = test[["id"]].copy()
    out["TmApp"] = float(np.median(dev["TmApp"].values))
    out["HIC"] = float(np.median(dev["HIC"].values))
    return out


def ridge_baseline(dev: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    Xtr = np.vstack([aa_features(h, l) for h, l in zip(dev.heavy, dev.light)])
    Xte = np.vstack([aa_features(h, l) for h, l in zip(test.heavy, test.light)])
    out = test[["id"]].copy()
    for target in ("TmApp", "HIC"):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        pipe.fit(Xtr, dev[target].astype(float).values)
        out[target] = pipe.predict(Xte)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--method",
        choices=["median", "ridge"],
        default="median",
        help="median = Train medians; ridge = AA composition + Ridge",
    )
    args = ap.parse_args()

    dev = pd.read_csv(args.dev)
    test = pd.read_csv(args.test)
    required_dev = {"id", "heavy", "light", "TmApp", "HIC"}
    required_test = {"id", "heavy", "light"}
    assert required_dev.issubset(dev.columns), dev.columns
    assert required_test.issubset(test.columns), test.columns

    if args.method == "median":
        sub = median_baseline(dev, test)
    else:
        sub = ridge_baseline(dev, test)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out} method={args.method} n={len(sub)}")


if __name__ == "__main__":
    main()
