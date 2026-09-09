#!/usr/bin/env python3
"""Validate developability_drilldown Phase 1 invariants."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    ARTIFACT_STATUSES,
    EXPERIMENTS_COLUMNS,
    FEATURE_SPACE,
    ROOT,
    feature_column_names,
    feature_content_sha256,
    file_sha256,
    load_dev_test_folds,
)

FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print("FAIL:", msg)


def ok(msg: str) -> None:
    print("OK:", msg)


def main() -> int:
    exp_path = ROOT / "results" / "experiments.csv"
    if not exp_path.exists():
        fail("experiments.csv missing")
        return 1
    df = pd.read_csv(exp_path)
    for c in EXPERIMENTS_COLUMNS:
        if c not in df.columns:
            fail(f"missing column {c}")

    if df["experiment_id"].duplicated().any():
        fail("duplicate experiment_id")
    else:
        ok(f"unique experiment_id n={len(df)}")

    if not set(df["target"]).issubset({"TmApp", "HIC"}):
        fail(f"bad targets {set(df['target'])}")
    if not set(df["artifact_status"]).issubset(ARTIFACT_STATUSES):
        fail(f"bad artifact_status {set(df['artifact_status'])}")

    # terminology: no submission naming under predictions
    pred_root = ROOT / "experiments" / "predictions"
    bad_names = []
    for p in pred_root.rglob("*"):
        name = p.name.lower()
        rel = str(p.relative_to(ROOT)).lower()
        if "submission" in name or name.startswith("sub_") or "/sub_" in rel:
            bad_names.append(str(p))
    if bad_names:
        fail(f"submission naming in predictions: {bad_names}")
    else:
        ok("no submission naming under single-target predictions")

    dev, test, _ = load_dev_test_folds()
    dev_ids = set(dev["id"])
    test_ids = set(test["id"])
    all_ids = dev_ids | test_ids

    full = df[df["artifact_status"] == "FULL"]
    lin_full = full[full["family"] == "LINEAR"]
    xgb_full = full[full["family"] == "XGBOOST"]
    if len(lin_full) != 6:
        fail(f"expected 6 FULL Linear, got {len(lin_full)}")
    else:
        ok("FULL Linear = 6")
    if len(xgb_full) != 6:
        fail(f"expected 6 FULL XGB, got {len(xgb_full)}")
    else:
        ok("FULL XGB = 6")

    # feature content hash equality for shared recipes
    recipe_hashes: dict[str, set[str]] = {}
    for _, r in full.iterrows():
        eid = r["experiment_id"]
        fpath = ROOT / str(r["feature_path"])
        if not fpath.exists():
            fail(f"missing feature {fpath}")
            continue
        feat = pd.read_parquet(fpath)
        if len(feat) != 324:
            fail(f"{eid} feature rows {len(feat)}")
        if set(feat["id"].astype(str)) != all_ids:
            fail(f"{eid} feature id set mismatch")
        if feat["id"].duplicated().any():
            fail(f"{eid} duplicate feature ids")
        if not set(feat["split"]).issubset({"dev", "test"}):
            fail(f"{eid} bad split values")
        cols = feature_column_names(feat)
        if int(r["n_features"]) != len(cols):
            fail(f"{eid} n_features mismatch")
        if r["feature_space"] != FEATURE_SPACE:
            fail(f"{eid} feature_space")
        if r["feature_sha256"] != file_sha256(fpath):
            fail(f"{eid} feature_sha256 mismatch")
        ch = feature_content_sha256(feat)
        if r["feature_content_sha256"] != ch:
            fail(f"{eid} feature_content_sha256 mismatch")
        recipe_hashes.setdefault(str(r["feature_recipe"]), set()).add(ch)

        # predictions
        for kind, n, idset in (
            ("oof_primary", 162, dev_ids),
            ("oof_shadow", 162, dev_ids),
            ("test", 162, test_ids),
        ):
            key = {
                "oof_primary": "oof_primary_path",
                "oof_shadow": "oof_shadow_path",
                "test": "test_prediction_path",
            }[kind]
            p = ROOT / str(r[key])
            if not p.exists():
                fail(f"missing {kind} for {eid}")
                continue
            pred = pd.read_csv(p)
            tgt = r["target"]
            if list(pred.columns) != ["id", tgt]:
                fail(f"{eid} {kind} columns {list(pred.columns)}")
            if len(pred) != n or set(pred["id"].astype(str)) != idset:
                fail(f"{eid} {kind} id mismatch")
            if not np_finite(pred[tgt]):
                fail(f"{eid} {kind} non-finite")

        # derived scores
        pmae = float(r["cv_primary_mae"])
        smae = float(r["cv_shadow_mae"])
        if abs(float(r["cv_mean_mae"]) - (pmae + smae) / 2) > 1e-12:
            fail(f"{eid} cv_mean")
        if abs(float(r["cv_worst_mae"]) - max(pmae, smae)) > 1e-12:
            fail(f"{eid} cv_worst")
        pub, priv = float(r["public_mae"]), float(r["private_mae"])
        if abs(float(r["public_private_delta"]) - (pub - priv)) > 1e-12:
            fail(f"{eid} public_private_delta")
        if abs(float(r["public_private_gap"]) - abs(pub - priv)) > 1e-12:
            fail(f"{eid} public_private_gap")

        cfg = ROOT / str(r["config_path"])
        if not cfg.exists():
            fail(f"missing config {cfg}")

    for recipe, hashes in recipe_hashes.items():
        if len(hashes) != 1:
            fail(f"recipe {recipe} content hash mismatch across experiments: {hashes}")
        else:
            ok(f"shared content hash for recipe {recipe}")

    # XGB audit
    audit_path = ROOT / "results" / "XGB_REPRODUCTION_AUDIT.csv"
    if not audit_path.exists():
        fail("XGB_REPRODUCTION_AUDIT.csv missing")
    else:
        audit = pd.read_csv(audit_path)
        if len(audit) != 6:
            fail(f"audit rows {len(audit)}")
        bad = audit[~audit["reproduction_status"].isin(["PASS", "PASS_CV_ONLY"])]
        if len(bad):
            fail(f"XGB audit not pass: {bad['experiment_id'].tolist()}")
        else:
            ok("XGB reproduction audit PASS/PASS_CV_ONLY for 6/6")

    # submissions
    man = ROOT / "submissions" / "submissions.csv"
    if man.exists():
        m = pd.read_csv(man)
        for _, row in m.iterrows():
            sp = ROOT / str(row["submission_path"])
            if not sp.exists():
                fail(f"missing submission {sp}")
                continue
            sub = pd.read_csv(sp)
            if list(sub.columns) != ["id", "TmApp", "HIC"]:
                fail(f"bad submission columns {sp}")
            if len(sub) != 162 or set(sub["id"].astype(str)) != test_ids:
                fail(f"bad submission ids {sp}")
            if not (np_finite(sub["TmApp"]) and np_finite(sub["HIC"])):
                fail(f"non-finite submission {sp}")
        ok(f"submissions manifest n={len(m)}")
    else:
        fail("submissions.csv missing")

    # solution not present in drilldown
    if (ROOT / "solution.csv").exists():
        fail("solution.csv present under drilldown")
    else:
        ok("no solution.csv in drilldown")

    # FEATURE_LINEAR coverage
    n_lin = int((df["family"] == "LINEAR").sum())
    if n_lin < 40:
        fail(f"expected ~42 LINEAR rows, got {n_lin}")
    else:
        ok(f"LINEAR rows={n_lin}")

    out = ROOT / "results" / "VALIDATION.txt"
    text = "PASS\n" if not FAILS else "FAIL\n" + "\n".join(FAILS) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if FAILS else 0


def np_finite(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce")
    return bool(v.notna().all() and (v.apply(lambda x: math.isfinite(float(x))).all()))


if __name__ == "__main__":
    raise SystemExit(main())
