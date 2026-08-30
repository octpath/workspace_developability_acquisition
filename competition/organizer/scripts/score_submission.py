#!/usr/bin/env python3
"""Score a participant submission against the organizer secret solution.

Primary metric: MAE (lower is better), independently for TmApp and HIC,
on Public and Private subsets.

Diagnostic Pearson / Spearman / RMSE are optional and never combined across targets.

Scientific ranking note (multi-submission):
    exact equal Private MAE values share the same scientific rank.
    Do not break ties with Pearson / Spearman / RMSE / Public / timestamps / IDs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOLUTION = ROOT / "competition" / "data" / "secret" / "solution.csv"


def scientific_ranks_from_private_mae(private_maes: list[float] | np.ndarray) -> list[int]:
    """Dense competition ranks from Private MAE (lower better).

    Exact equal MAE values receive the same rank (ties share rank).
    No secondary keys are used.
    """
    vals = np.asarray(private_maes, float)
    # Sort unique values ascending; map each value to 1-based dense rank
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals), dtype=int)
    rank = 0
    prev = None
    for idx in order:
        v = vals[idx]
        if prev is None or v != prev:
            rank += 1
            prev = v
        ranks[idx] = rank
    return ranks.tolist()


def _as_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda x: str(x).strip().lower() in ("true", "1"))


def _corr(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < 2 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return float("nan")
    return float(pearsonr(y, p)[0])


def _spr(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < 2 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return float("nan")
    r = spearmanr(y, p).correlation
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def validate_submission(sub: pd.DataFrame, sol: pd.DataFrame) -> list[str]:
    errs = []
    required = ["id", "TmApp", "HIC"]
    for c in required:
        if c not in sub.columns:
            errs.append(f"missing column: {c}")
    if errs:
        return errs
    if sub["id"].duplicated().any():
        errs.append("duplicate IDs in submission")
    if len(sub) != len(sol):
        errs.append(f"row count mismatch: submission={len(sub)} solution={len(sol)}")
    if set(sub["id"]) != set(sol["id"]):
        missing = sorted(set(sol["id"]) - set(sub["id"]))
        extra = sorted(set(sub["id"]) - set(sol["id"]))
        errs.append(f"ID set mismatch; missing={len(missing)} extra={len(extra)}")
    for c in ("TmApp", "HIC"):
        try:
            vals = pd.to_numeric(sub[c], errors="coerce")
        except Exception:
            errs.append(f"{c}: not numeric")
            continue
        if vals.isna().any():
            errs.append(f"{c}: NaN / non-numeric present")
        elif not np.isfinite(vals.astype(float)).all():
            errs.append(f"{c}: non-finite values present")
    return errs


def score(submission_path: Path, solution_path: Path = DEFAULT_SOLUTION) -> dict:
    sub = pd.read_csv(submission_path)
    sol = pd.read_csv(solution_path)
    sol["id"] = sol["id"].astype(str)
    sub["id"] = sub["id"].astype(str)

    errs = validate_submission(sub, sol)
    if errs:
        return {"status": "INVALID", "errors": errs}

    # Join by ID (do not rely on row order)
    m = sol.merge(sub[["id", "TmApp", "HIC"]], on="id", how="inner", suffixes=("_true", "_pred"))
    assert len(m) == len(sol)

    pub = _as_bool(m["is_public"])
    priv = _as_bool(m["is_private"])

    out = {
        "status": "OK",
        "primary_metric": "MAE",
        "n_public": int(pub.sum()),
        "n_private": int(priv.sum()),
        "scores": {},
        "diagnostics": {},
    }

    for target in ("TmApp", "HIC"):
        y = m[f"{target}_true"].astype(float).values
        p = m[f"{target}_pred"].astype(float).values
        for split_name, mask in (("public", pub.values), ("private", priv.values)):
            yt, pt = y[mask], p[mask]
            mae = float(np.mean(np.abs(yt - pt)))
            rmse = float(np.sqrt(np.mean((yt - pt) ** 2)))
            out["scores"][f"{target}_{split_name}_mae"] = mae
            out["diagnostics"][f"{target}_{split_name}_rmse"] = rmse
            out["diagnostics"][f"{target}_{split_name}_pearson"] = _corr(yt, pt)
            out["diagnostics"][f"{target}_{split_name}_spearman"] = _spr(yt, pt)

    out["note"] = (
        "Primary results are the four MAE values. "
        "Pearson/Spearman/RMSE are diagnostic only. "
        "TmApp and HIC are never combined into one score."
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--solution", type=Path, default=DEFAULT_SOLUTION)
    ap.add_argument("--json", action="store_true", help="print full JSON")
    args = ap.parse_args()
    result = score(args.submission, args.solution)
    if args.json or result["status"] != "OK":
        print(json.dumps(result, indent=2))
    else:
        s = result["scores"]
        print("status: OK")
        print(f"TmApp Public MAE:  {s['TmApp_public_mae']:.6f}")
        print(f"TmApp Private MAE: {s['TmApp_private_mae']:.6f}")
        print(f"HIC Public MAE:    {s['HIC_public_mae']:.6f}")
        print(f"HIC Private MAE:   {s['HIC_private_mae']:.6f}")
        print("(diagnostics: use --json)")
    sys.exit(0 if result["status"] == "OK" else 1)


if __name__ == "__main__":
    main()
