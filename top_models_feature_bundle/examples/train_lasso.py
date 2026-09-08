#!/usr/bin/env python3
"""Minimal Lasso example: StandardScaler -> Lasso, VAL alpha selection, no PCA."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))

def run(train_csv: str, target: str, feature_files: list[str], scheme: str = "primary"):
    train = pd.read_csv(train_csv)
    train["id"] = train["id"].astype(str)
    X = train[["id"]].copy()
    for f in feature_files:
        feat = pd.read_parquet(f)
        feat["id"] = feat["id"].astype(str)
        X = X.merge(feat, on="id", how="left")
    folds = pd.read_csv(ROOT / "folds.csv")
    folds = folds[folds.scheme == scheme]
    fmap = folds.set_index("id")["fold"].astype(int).to_dict()
    ids = [i for i in X["id"] if i in fmap]
    y = train.set_index("id")[target].astype(float)
    num = X.set_index("id").select_dtypes("number")
    oof = pd.Series(index=ids, dtype=float)
    for k in range(5):
        te = [i for i in ids if fmap[i] == k]
        va = [i for i in ids if fmap[i] == (k + 1) % 5]
        tr = [i for i in ids if fmap[i] not in (k, (k + 1) % 5)]
        med = num.loc[tr].median().fillna(0)
        def prep(ix):
            return num.loc[ix].fillna(med).fillna(0.0)
        best_a, best = 1.0, 1e18
        for a in ALPHAS:
            sc = StandardScaler()
            m = Lasso(alpha=a, max_iter=200000, tol=1e-3, random_state=0)
            try:
                m.fit(sc.fit_transform(prep(tr)), y.loc[tr])
            except Exception:
                continue
            s = mae(y.loc[va], m.predict(sc.transform(prep(va))))
            if s < best:
                best, best_a = s, a
        tv = tr + va
        med2 = num.loc[tv].median().fillna(0)
        def prep2(ix):
            return num.loc[ix].fillna(med2).fillna(0.0)
        sc = StandardScaler()
        m = Lasso(alpha=best_a, max_iter=200000, tol=1e-3, random_state=0)
        m.fit(sc.fit_transform(prep2(tv)), y.loc[tv])
        oof.loc[te] = m.predict(sc.transform(prep2(te)))
    print("MAE", mae(y.loc[ids], oof), "alpha_last", best_a)

if __name__ == "__main__":
    print("Provide official train CSV path as argv; see README.")
