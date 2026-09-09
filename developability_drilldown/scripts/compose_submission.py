#!/usr/bin/env python3
"""Compose competition submission from Tm + HIC single-target test predictions."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import ROOT  # noqa: E402


def load_exp_table() -> pd.DataFrame:
    path = ROOT / "results" / "experiments.csv"
    if not path.exists():
        raise SystemExit("results/experiments.csv missing; run backfill first")
    return pd.read_csv(path)


def load_test_pred(experiment_id: str, expect_target: str) -> pd.DataFrame:
    path = ROOT / "experiments" / "predictions" / experiment_id / "test.csv"
    if not path.exists():
        raise SystemExit(f"missing test prediction: {path}")
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    if list(df.columns) != ["id", expect_target]:
        raise SystemExit(f"{path} columns {list(df.columns)} != ['id', '{expect_target}']")
    if "submission" in str(path).lower() or "/sub_" in str(path):
        raise SystemExit(f"illegal submission naming in prediction path: {path}")
    return df


def compose(tm_id: str, hic_id: str, register: bool = True) -> Path:
    ex = load_exp_table().set_index("experiment_id")
    if tm_id not in ex.index or hic_id not in ex.index:
        raise SystemExit("unknown experiment id")
    if ex.loc[tm_id, "target"] != "TmApp":
        raise SystemExit(f"{tm_id} target must be TmApp")
    if ex.loc[hic_id, "target"] != "HIC":
        raise SystemExit(f"{hic_id} target must be HIC")

    test_ids = pd.read_csv(ROOT / "data" / "test.csv")["id"].astype(str).tolist()
    tm = load_test_pred(tm_id, "TmApp").set_index("id").reindex(test_ids)
    hic = load_test_pred(hic_id, "HIC").set_index("id").reindex(test_ids)
    if tm["TmApp"].isna().any() or hic["HIC"].isna().any():
        raise SystemExit("prediction ID mismatch vs test.csv")
    out = pd.DataFrame({"id": test_ids, "TmApp": tm["TmApp"].to_numpy(float), "HIC": hic["HIC"].to_numpy(float)})
    if list(out.columns) != ["id", "TmApp", "HIC"]:
        raise SystemExit("bad columns")
    if len(out) != 162 or out["id"].duplicated().any():
        raise SystemExit("bad row count / duplicates")
    if not (out[["TmApp", "HIC"]].apply(pd.to_numeric, errors="coerce").notna().all().all()):
        raise SystemExit("non-finite predictions")

    fname = f"sub__{tm_id}__{hic_id}.csv"
    dest = ROOT / "submissions" / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    if register:
        man_path = ROOT / "submissions" / "submissions.csv"
        row = {
            "submission_id": fname.replace(".csv", ""),
            "tm_experiment_id": tm_id,
            "hic_experiment_id": hic_id,
            "submission_path": str(dest.relative_to(ROOT)),
            "tm_public_mae": ex.loc[tm_id, "public_mae"],
            "tm_private_mae": ex.loc[tm_id, "private_mae"],
            "tm_test_overall_mae": ex.loc[tm_id, "test_overall_mae"],
            "hic_public_mae": ex.loc[hic_id, "public_mae"],
            "hic_private_mae": ex.loc[hic_id, "private_mae"],
            "hic_test_overall_mae": ex.loc[hic_id, "test_overall_mae"],
            "created_from_predictions": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": "",
        }
        if man_path.exists():
            man = pd.read_csv(man_path)
            man = man[man["submission_id"] != row["submission_id"]]
            man = pd.concat([man, pd.DataFrame([row])], ignore_index=True)
        else:
            man = pd.DataFrame([row])
        man.to_csv(man_path, index=False)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", required=True)
    ap.add_argument("--hic", required=True)
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    dest = compose(args.tm, args.hic, register=not args.no_register)
    print("wrote", dest)


if __name__ == "__main__":
    main()
