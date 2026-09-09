#!/usr/bin/env python3
"""Compose competition submission from Tm + HIC single-target test predictions.

Preferred inputs: EXP-Txxx / EXP-Hxxx. Also accepts descriptive experiment_id
and deprecated legacy EXPxxx (with warning). Filename always uses canonical codes.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import ROOT  # noqa: E402
from experiment_codes import CODE_RE, resolve_experiment_ref  # noqa: E402


def load_exp_table() -> pd.DataFrame:
    path = ROOT / "results" / "experiments.csv"
    if not path.exists():
        raise SystemExit("results/experiments.csv missing")
    return pd.read_csv(path)


def load_test_pred(code: str, expect_target: str) -> pd.DataFrame:
    path = ROOT / "experiments" / "predictions" / code / "test.csv"
    if not path.exists():
        raise SystemExit(f"missing test prediction: {path}")
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    if list(df.columns) != ["id", expect_target]:
        raise SystemExit(f"{path} columns {list(df.columns)} != ['id', '{expect_target}']")
    if "submission" in path.name.lower() or "/sub_" in str(path).lower():
        raise SystemExit(f"illegal submission naming in prediction path: {path}")
    return df


def compose(tm_ref: str, hic_ref: str, register: bool = True) -> Path:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        tm_code, tm_id = resolve_experiment_ref(tm_ref)
        hic_code, hic_id = resolve_experiment_ref(hic_ref)
    for w in caught:
        print(f"WARNING: {w.message}", file=sys.stderr)

    if not tm_code.startswith("EXP-T"):
        raise SystemExit(f"--tm must resolve to EXP-T…, got {tm_code}")
    if not hic_code.startswith("EXP-H"):
        raise SystemExit(f"--hic must resolve to EXP-H…, got {hic_code}")
    if tm_code.startswith("EXP-M") or hic_code.startswith("EXP-M"):
        raise SystemExit("EXP-M is reserved for joint multi-target experiments; not supported by compose_submission")

    ex = load_exp_table().set_index("experiment_code")
    if tm_code not in ex.index or hic_code not in ex.index:
        raise SystemExit("unknown experiment code")
    if ex.loc[tm_code, "target"] != "TmApp":
        raise SystemExit(f"{tm_code} ({tm_id}) target must be TmApp")
    if ex.loc[hic_code, "target"] != "HIC":
        raise SystemExit(f"{hic_code} ({hic_id}) target must be HIC")
    if ex.loc[tm_code, "artifact_status"] != "FULL" or ex.loc[hic_code, "artifact_status"] != "FULL":
        raise SystemExit("both experiments must be artifact_status=FULL")

    test_ids = pd.read_csv(ROOT / "data" / "test.csv")["id"].astype(str).tolist()
    tm = load_test_pred(tm_code, "TmApp").set_index("id").reindex(test_ids)
    hic = load_test_pred(hic_code, "HIC").set_index("id").reindex(test_ids)
    if tm["TmApp"].isna().any() or hic["HIC"].isna().any():
        raise SystemExit("prediction ID mismatch vs test.csv")
    out = pd.DataFrame(
        {"id": test_ids, "TmApp": tm["TmApp"].to_numpy(float), "HIC": hic["HIC"].to_numpy(float)}
    )
    if list(out.columns) != ["id", "TmApp", "HIC"] or len(out) != 162 or out["id"].duplicated().any():
        raise SystemExit("bad submission frame")

    fname = f"sub__{tm_code}__{hic_code}.csv"
    dest = ROOT / "submissions" / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    if register:
        man_path = ROOT / "submissions" / "submissions.csv"
        row = {
            "submission_id": fname.replace(".csv", ""),
            "tm_experiment_code": tm_code,
            "tm_experiment_id": tm_id,
            "tm_legacy_experiment_code": ex.loc[tm_code].get("legacy_experiment_code", ""),
            "hic_experiment_code": hic_code,
            "hic_experiment_id": hic_id,
            "hic_legacy_experiment_code": ex.loc[hic_code].get("legacy_experiment_code", ""),
            "submission_path": str(dest.relative_to(ROOT)),
            "tm_public_mae": ex.loc[tm_code, "public_mae"],
            "tm_private_mae": ex.loc[tm_code, "private_mae"],
            "tm_test_overall_mae": ex.loc[tm_code, "test_overall_mae"],
            "hic_public_mae": ex.loc[hic_code, "public_mae"],
            "hic_private_mae": ex.loc[hic_code, "private_mae"],
            "hic_test_overall_mae": ex.loc[hic_code, "test_overall_mae"],
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
    ap.add_argument("--tm", required=True, help="TmApp EXP-Txxx / experiment_id / legacy EXPxxx")
    ap.add_argument("--hic", required=True, help="HIC EXP-Hxxx / experiment_id / legacy EXPxxx")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    dest = compose(args.tm, args.hic, register=not args.no_register)
    print("wrote", dest)


if __name__ == "__main__":
    main()
