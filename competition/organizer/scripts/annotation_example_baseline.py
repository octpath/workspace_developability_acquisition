#!/usr/bin/env python3
"""Tiny example: join sequence-derived annotations by id (not a strong baseline)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CAT = [
    "heavy_v_family",
    "heavy_j_gene",
    "light_v_family",
    "light_j_gene",
    "light_chain_type",
]
NUM = [
    "h_cdr1_length",
    "h_cdr2_length",
    "h_cdr3_length",
    "l_cdr1_length",
    "l_cdr2_length",
    "l_cdr3_length",
    "heavy_germline_identity",
    "light_germline_identity",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--dev-ann", type=Path, required=True)
    ap.add_argument("--test-ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = pd.read_csv(args.dev).merge(pd.read_csv(args.dev_ann), on="id", how="inner", validate="one_to_one")
    test = pd.read_csv(args.test).merge(pd.read_csv(args.test_ann), on="id", how="inner", validate="one_to_one")
    assert len(dev) == len(pd.read_csv(args.dev))
    assert len(test) == len(pd.read_csv(args.test))

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler(), NUM),
        ]
    )
    out = test[["id"]].copy()
    for target in ("TmApp", "HIC"):
        pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
        pipe.fit(dev[CAT + NUM], dev[target].astype(float).values)
        out[target] = pipe.predict(test[CAT + NUM])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} n={len(out)} (annotation join example)")


if __name__ == "__main__":
    main()
