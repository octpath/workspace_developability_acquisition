#!/usr/bin/env python3
"""Example: LEFT JOIN participant train/test CSV with precomputed feature tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True, help="Participant training CSV with an id column")
    p.add_argument(
        "--feature-parquet",
        action="append",
        default=[],
        help="Parquet path(s) under feature_extension (repeatable)",
    )
    p.add_argument("--output", default=None)
    args = p.parse_args()

    train = pd.read_csv(args.train_csv)
    if "id" not in train.columns:
        raise SystemExit("train CSV must contain column `id`")

    defaults = [
        ROOT / "data/bioemu_isolated/features.parquet",
        ROOT / "data/precomputed_features/proteinmpnn.parquet",
        ROOT / "data/precomputed_features/aromatic_topology.parquet",
    ]
    paths = [Path(x) for x in args.feature_parquet] if args.feature_parquet else defaults

    out = train.copy()
    for path in paths:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        feat = pd.read_parquet(path)
        if "id" not in feat.columns:
            raise SystemExit(f"{path} missing id")
        # avoid duplicate non-id columns
        overlap = [c for c in feat.columns if c != "id" and c in out.columns]
        if overlap:
            feat = feat.drop(columns=overlap)
        before = len(out.columns)
        out = out.merge(feat, on="id", how="left")
        print(f"joined {path.name}: +{len(out.columns) - before} cols")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if str(args.output).endswith(".parquet"):
            out.to_parquet(args.output, index=False)
        else:
            out.to_csv(args.output, index=False)
        print(f"wrote {args.output} shape={out.shape}")
    else:
        print(out.head())
        print("shape", out.shape)


if __name__ == "__main__":
    main()
