#!/usr/bin/env python3
"""Validate developability_drilldown registry after EXP-code migration."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    ARTIFACT_STATUSES,
    CODE_RE,
    DRILLDOWN_REPRO,
    EXPERIMENTS_COLUMNS,
    FEATURE_SPACE,
    LICENSE_STATUSES,
    ROOT,
    SOURCE_REPRO,
    feature_column_names,
    feature_content_sha256,
    file_sha256,
    load_dev_test_folds,
)
from experiment_codes import load_codes  # noqa: E402

FAILS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print("FAIL:", msg)


def ok(msg: str) -> None:
    print("OK:", msg)


def np_finite(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce")
    return bool(v.notna().all() and v.apply(lambda x: math.isfinite(float(x))).all())


def main() -> int:
    exp_path = ROOT / "results" / "experiments.csv"
    df = pd.read_csv(exp_path)
    for c in EXPERIMENTS_COLUMNS:
        if c not in df.columns:
            fail(f"missing column {c}")
    if "reproducible" in df.columns and "source_reproducible" not in df.columns:
        fail("ambiguous reproducible column without source_reproducible")
    if "reproducible" in df.columns:
        fail("legacy 'reproducible' column must be removed")

    codes = load_codes()
    if len(df) != 48 or len(codes) != 48:
        fail(f"expected 48 rows/codes, got experiments={len(df)} codes={len(codes)}")
    else:
        ok("48 experiments / 48 codes")

    if df["experiment_code"].duplicated().any() or df["experiment_id"].duplicated().any():
        fail("duplicate experiment_code or experiment_id")
    for c in df["experiment_code"]:
        if not CODE_RE.match(str(c)):
            fail(f"bad code format {c}")
    expected = [f"EXP{i:03d}" for i in range(1, 49)]
    if list(df["experiment_code"]) != expected:
        fail("experiments.csv codes not exactly EXP001–EXP048 in issuance order")
    else:
        ok("EXP001–EXP048 contiguous in registry order")
    if list(codes["experiment_code"]) != expected:
        fail("EXPERIMENT_CODES.csv not EXP001–EXP048")
    merged = df.merge(codes, on=["experiment_code", "experiment_id"], how="inner")
    if len(merged) != 48:
        fail("EXPERIMENT_CODES.csv mapping mismatch vs experiments.csv")
    else:
        ok("code↔id mapping matches EXPERIMENT_CODES.csv")

    if not set(df["target"]).issubset({"TmApp", "HIC"}):
        fail(f"bad targets {set(df['target'])}")
    if not set(df["artifact_status"]).issubset(ARTIFACT_STATUSES):
        fail(f"bad artifact_status {set(df['artifact_status'])}")
    if not set(df["source_reproducible"]).issubset(SOURCE_REPRO):
        fail(f"bad source_reproducible {set(df['source_reproducible'])}")
    if not set(df["drilldown_reproducible"]).issubset(DRILLDOWN_REPRO):
        fail(f"bad drilldown_reproducible {set(df['drilldown_reproducible'])}")
    if not set(df["license_status"]).issubset(LICENSE_STATUSES):
        fail(f"bad license_status {set(df['license_status'])}")
    else:
        ok("repro/license enums valid")
    if (df["license_status"] == "SEE_feature_manifest").any():
        fail("legacy SEE_feature_manifest status remains")

    # terminology
    pred_root = ROOT / "experiments" / "predictions"
    bad = [
        str(p)
        for p in pred_root.rglob("*")
        if "submission" in p.name.lower() or p.name.startswith("sub_")
    ]
    if bad:
        fail(f"submission naming under predictions: {bad}")
    else:
        ok("no submission naming under predictions")

    # no descriptive-id artifact leftovers for FULL
    for leftover_glob in (
        ROOT.glob("experiments/configs/LIN_*.yaml"),
        ROOT.glob("experiments/configs/XGB_*.yaml"),
        ROOT.glob("experiments/features/LIN_*.parquet"),
        ROOT.glob("experiments/features/XGB_*.parquet"),
    ):
        leftovers = list(leftover_glob) if not isinstance(leftover_glob, list) else leftover_glob
    leftovers = (
        list(ROOT.glob("experiments/configs/LIN_*.yaml"))
        + list(ROOT.glob("experiments/configs/XGB_*.yaml"))
        + list(ROOT.glob("experiments/features/LIN_*.parquet"))
        + list(ROOT.glob("experiments/features/XGB_*.parquet"))
        + [p for p in (ROOT / "experiments" / "predictions").iterdir() if p.name.startswith(("LIN_", "XGB_"))]
    )
    if leftovers:
        fail(f"legacy descriptive artifact paths remain: {leftovers[:5]}")
    else:
        ok("artifact paths use EXPxxx only")

    dev, test, _ = load_dev_test_folds()
    dev_ids, test_ids = set(dev["id"]), set(test["id"])
    all_ids = dev_ids | test_ids

    full = df[df["artifact_status"] == "FULL"]
    if len(full) != 12:
        fail(f"FULL count {len(full)}")
    else:
        ok("FULL=12")

    # feature sets
    fs_path = ROOT / "results" / "FEATURE_SETS.csv"
    if not fs_path.exists():
        fail("FEATURE_SETS.csv missing")
    else:
        fs = pd.read_csv(fs_path)
        ok(f"FEATURE_SETS n={len(fs)}")

    recipe_hashes: dict[str, set[str]] = {}
    for _, r in full.iterrows():
        code = r["experiment_code"]
        if not str(r["feature_set_id"]):
            fail(f"{code} missing feature_set_id")
        if not str(r["source_recipe_id"]):
            fail(f"{code} missing source_recipe_id")
        if str(r["source_recipe_id"]).endswith(("__RIDGE", "__LASSO")) and str(r["feature_set_id"]).endswith(
            ("_RIDGE", "_LASSO")
        ):
            fail(f"{code} feature_set_id still looks estimator-suffixed: {r['feature_set_id']}")

        for path_col, expect_suffix in (
            ("config_path", f"experiments/configs/{code}.yaml"),
            ("feature_path", f"experiments/features/{code}.parquet"),
            ("oof_primary_path", f"experiments/predictions/{code}/oof_primary.csv"),
            ("oof_shadow_path", f"experiments/predictions/{code}/oof_shadow.csv"),
            ("test_prediction_path", f"experiments/predictions/{code}/test.csv"),
        ):
            if str(r[path_col]) != expect_suffix:
                fail(f"{code} {path_col}={r[path_col]} expected {expect_suffix}")
            if not (ROOT / str(r[path_col])).exists():
                fail(f"missing {r[path_col]}")

        fpath = ROOT / str(r["feature_path"])
        feat = pd.read_parquet(fpath)
        if len(feat) != 324 or set(feat["id"].astype(str)) != all_ids:
            fail(f"{code} feature id/rows")
        cols = feature_column_names(feat)
        if int(r["n_features"]) != len(cols):
            fail(f"{code} n_features")
        if r["feature_space"] != FEATURE_SPACE:
            fail(f"{code} feature_space")
        if r["feature_sha256"] != file_sha256(fpath):
            fail(f"{code} feature_sha256")
        ch = feature_content_sha256(feat)
        if r["feature_content_sha256"] != ch:
            fail(f"{code} feature_content_sha256")
        recipe_hashes.setdefault(str(r["feature_set_id"]), set()).add(ch)

        for kind, key, idset in (
            ("oof_primary", "oof_primary_path", dev_ids),
            ("oof_shadow", "oof_shadow_path", dev_ids),
            ("test", "test_prediction_path", test_ids),
        ):
            pred = pd.read_csv(ROOT / str(r[key]))
            tgt = r["target"]
            if list(pred.columns) != ["id", tgt] or set(pred["id"].astype(str)) != idset:
                fail(f"{code} {kind} schema/ids")
            if not np_finite(pred[tgt]):
                fail(f"{code} {kind} non-finite")

        pmae, smae = float(r["cv_primary_mae"]), float(r["cv_shadow_mae"])
        if abs(float(r["cv_mean_mae"]) - (pmae + smae) / 2) > 1e-12:
            fail(f"{code} cv_mean")
        if abs(float(r["cv_worst_mae"]) - max(pmae, smae)) > 1e-12:
            fail(f"{code} cv_worst")

    for fsid, hashes in recipe_hashes.items():
        if len(hashes) != 1:
            fail(f"feature_set {fsid} hash mismatch {hashes}")
        else:
            ok(f"feature_set {fsid} shared hash OK")

    # submissions
    man = ROOT / "submissions" / "submissions.csv"
    if not man.exists():
        fail("submissions.csv missing")
    else:
        m = pd.read_csv(man)
        for _, row in m.iterrows():
            if "tm_experiment_code" not in row or "hic_experiment_code" not in row:
                fail("submissions.csv missing code columns")
                break
            sp = ROOT / str(row["submission_path"])
            if not sp.exists():
                fail(f"missing {sp}")
                continue
            if not re.match(r"sub__EXP\d+__EXP\d+\.csv$", sp.name):
                fail(f"submission filename not code-based: {sp.name}")
            sub = pd.read_csv(sp)
            if list(sub.columns) != ["id", "TmApp", "HIC"] or len(sub) != 162:
                fail(f"bad submission {sp}")
            if set(sub["id"].astype(str)) != test_ids:
                fail(f"bad submission ids {sp}")
        ok(f"submissions n={len(m)}")

    if (ROOT / "solution.csv").exists():
        fail("solution.csv present under drilldown")
    else:
        ok("no solution.csv in drilldown")

    out = ROOT / "results" / "VALIDATION.txt"
    text = "PASS\n" if not FAILS else "FAIL\n" + "\n".join(FAILS) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
