#!/usr/bin/env python3
"""Independent track scorer for competition submissions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REQUIRED_COLS = {"id", "TmApp", "HIC"}


def validate_submission(sub: pd.DataFrame, expected_ids: set[str]) -> list[str]:
    errs = []
    if not REQUIRED_COLS.issubset(sub.columns):
        errs.append(f"missing columns; need {REQUIRED_COLS}")
        return errs
    if sub["id"].duplicated().any():
        errs.append("duplicate ids")
    got = set(sub["id"])
    if got != expected_ids:
        missing = expected_ids - got
        extra = got - expected_ids
        if missing:
            errs.append(f"missing {len(missing)} ids")
        if extra:
            errs.append(f"extra {len(extra)} ids")
    for col in ["TmApp", "HIC"]:
        if not np.issubdtype(sub[col].dtype, np.number):
            try:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
            except Exception:
                errs.append(f"non-numeric {col}")
        if sub[col].isna().any():
            errs.append(f"NaN in {col}")
        if np.isinf(sub[col].astype(float)).any():
            errs.append(f"Inf in {col}")
    return errs


def score_track(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 2:
        return {"spearman": None, "n": len(y_true), "error": "n<2"}
    if np.nanstd(y_pred) == 0 or np.nanstd(y_true) == 0:
        return {"spearman": None, "n": len(y_true), "error": "constant prediction or labels"}
    sp = spearmanr(y_true, y_pred).correlation
    return {
        "spearman": float(sp) if sp is not None and np.isfinite(sp) else None,
        "n": int(len(y_true)),
        "error": None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score competition submission (independent tracks)")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--hidden_labels", required=True)
    ap.add_argument("--role", required=True, choices=["public", "private", "Public", "Private", "all"])
    ap.add_argument("--track", required=True, choices=["tmapp", "hic", "TmApp", "HIC", "both"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    sub = pd.read_csv(args.submission)
    hid = pd.read_csv(args.hidden_labels)
    role = args.role.capitalize() if args.role.lower() != "all" else "all"
    if role != "all":
        hid = hid[hid["role"].str.lower() == role.lower()]
    expected = set(hid["id"])
    # Allow full test submission; score only the requested role rows
    sub = sub[sub["id"].isin(expected)].copy()
    errs = validate_submission(sub, expected)
    if errs:
        out = {"ok": False, "validation_errors": errs}
        print(json.dumps(out, indent=2) if args.json else out)
        return 1

    m = hid.merge(sub, on="id", suffixes=("_true", "_pred"))
    results = {"ok": True, "role": role, "tracks": {}}
    tracks = []
    t = args.track.lower()
    if t in ("tmapp", "both"):
        tracks.append(("TmApp", "TmApp"))
    if t in ("hic", "both"):
        tracks.append(("HIC", "HIC"))
    # column naming after merge
    for track_name, col in tracks:
        yt = m[f"{col}_true"] if f"{col}_true" in m.columns else m[col]
        # after merge: hidden has TmApp/HIC, sub has TmApp/HIC -> pandas suffixes
        if f"{col}_true" in m.columns:
            yt = m[f"{col}_true"].values
            yp = m[f"{col}_pred"].values
        else:
            # if hidden used role+labels and sub merged without clash
            yt = hid.set_index("id").loc[m["id"], col].values
            yp = sub.set_index("id").loc[m["id"], col].values
        results["tracks"][track_name] = score_track(yt, yp)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
