#!/usr/bin/env python3
"""Score prediction CSVs against local solution.csv (optional)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import ROOT, load_solution, mae  # noqa: E402


def score_file(pred_path: Path, target: str, sol: pd.DataFrame) -> dict:
    pred = pd.read_csv(pred_path)
    pred["id"] = pred["id"].astype(str)
    if target not in pred.columns:
        raise SystemExit(f"{pred_path} missing column {target}")
    m = sol.merge(pred[["id", target]].rename(columns={target: "pred"}), on="id", how="inner")
    if len(m) != 162:
        raise SystemExit(f"expected 162 rows, got {len(m)} for {pred_path}")
    y = m[target].to_numpy(float)
    p = m["pred"].to_numpy(float)
    pub = m["is_public"].astype(bool).to_numpy()
    priv = m["is_private"].astype(bool).to_numpy()
    return {
        "public_mae": mae(y[pub], p[pub]),
        "private_mae": mae(y[priv], p[priv]),
        "overall_mae": mae(y, p),
        "n_public": int(pub.sum()),
        "n_private": int(priv.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--target", required=True, choices=["TmApp", "HIC"])
    args = ap.parse_args()
    sol = load_solution()
    if sol is None:
        raise SystemExit("solution.csv not available locally")
    print(score_file(args.pred, args.target, sol))


if __name__ == "__main__":
    main()
