#!/usr/bin/env python3
"""Validate developability_drilldown after target-namespaced EXP codes."""
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
    LEGACY_CODE_RE,
    LICENSE_STATUSES,
    ROOT,
    SOURCE_REPRO,
    feature_column_names,
    feature_content_sha256,
    file_sha256,
    load_dev_test_folds,
)
from experiment_codes import load_codes, load_legacy_map, next_code  # noqa: E402

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
    if "reproducible" in df.columns:
        fail("legacy 'reproducible' column must be removed")

    codes = load_codes()
    legacy = load_legacy_map()
    if len(df) != 48 or len(codes) != 48 or len(legacy) != 48:
        fail(f"expected 48 rows; experiments={len(df)} codes={len(codes)} legacy={len(legacy)}")
    else:
        ok("48 experiments / codes / legacy map")

    if df["experiment_code"].duplicated().any() or df["experiment_id"].duplicated().any():
        fail("duplicate experiment_code or experiment_id")
    for _, r in df.iterrows():
        c = str(r["experiment_code"])
        if not CODE_RE.match(c):
            fail(f"bad code format {c}")
        if LEGACY_CODE_RE.match(c):
            fail(f"flat legacy form remains as canonical: {c}")
        if r["target"] == "TmApp" and not c.startswith("EXP-T"):
            fail(f"TmApp row has non-T code {c}")
        if r["target"] == "HIC" and not c.startswith("EXP-H"):
            fail(f"HIC row has non-H code {c}")
        if not str(r.get("legacy_experiment_code") or ""):
            fail(f"missing legacy_experiment_code for {c}")

    m_count = int(df["experiment_code"].astype(str).str.startswith("EXP-M").sum())
    if m_count != 0:
        fail(f"unexpected EXP-M rows: {m_count}")
    else:
        ok("MULTI namespace empty (reserved)")
    if next_code("MULTI") != "EXP-M001":
        fail(f"next MULTI code unexpected: {next_code('MULTI')}")
    else:
        ok("next MULTI reserved EXP-M001")

    t_codes = sorted(
        [c for c in df["experiment_code"] if str(c).startswith("EXP-T")],
        key=lambda x: int(str(x).split("-")[1][1:]),
    )
    h_codes = sorted(
        [c for c in df["experiment_code"] if str(c).startswith("EXP-H")],
        key=lambda x: int(str(x).split("-")[1][1:]),
    )
    t_n, h_n = len(t_codes), len(h_codes)
    if t_codes != [f"EXP-T{i:03d}" for i in range(1, t_n + 1)]:
        fail(f"T codes not contiguous EXP-T001..EXP-T{t_n:03d}")
    else:
        ok(f"TmApp codes EXP-T001..EXP-T{t_n:03d} ({t_n})")
    if h_codes != [f"EXP-H{i:03d}" for i in range(1, h_n + 1)]:
        fail(f"H codes not contiguous")
    else:
        ok(f"HIC codes EXP-H001..EXP-H{h_n:03d} ({h_n})")

    merged = df.merge(
        codes,
        on=["experiment_code", "experiment_id"],
        how="inner",
        suffixes=("", "_c"),
    )
    if len(merged) != 48:
        fail("EXPERIMENT_CODES.csv mismatch")
    else:
        ok("codes table matches experiments.csv")

    leg_m = df.merge(
        legacy,
        left_on=["legacy_experiment_code", "experiment_code", "experiment_id"],
        right_on=["legacy_experiment_code", "experiment_code", "experiment_id"],
        how="inner",
    )
    if len(leg_m) != 48:
        fail("LEGACY_EXPERIMENT_CODE_MAP.csv mismatch")
    else:
        ok("legacy map 48/48")

    if not set(df["artifact_status"]).issubset(ARTIFACT_STATUSES):
        fail("bad artifact_status")
    if not set(df["source_reproducible"]).issubset(SOURCE_REPRO):
        fail("bad source_reproducible")
    if not set(df["drilldown_reproducible"]).issubset(DRILLDOWN_REPRO):
        fail("bad drilldown_reproducible")
    if not set(df["license_status"]).issubset(LICENSE_STATUSES):
        fail("bad license_status")
    else:
        ok("repro/license enums valid")

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

    leftovers = [
        p
        for p in list((ROOT / "experiments" / "configs").glob("EXP[0-9]*.yaml"))
        + list((ROOT / "experiments" / "features").glob("EXP[0-9]*.parquet"))
        + [p for p in pred_root.iterdir() if re.fullmatch(r"EXP[0-9]+", p.name)]
    ]
    if leftovers:
        fail(f"flat EXP0xx paths remain: {leftovers[:5]}")
    else:
        ok("no flat EXP0xx artifact paths")

    dev, test, _ = load_dev_test_folds()
    dev_ids, test_ids = set(dev["id"]), set(test["id"])
    all_ids = dev_ids | test_ids

    full = df[df["artifact_status"] == "FULL"]
    if len(full) != 12:
        fail(f"FULL count {len(full)}")
    else:
        ok("FULL=12")

    recipe_hashes: dict[str, set[str]] = {}
    for _, r in full.iterrows():
        code = r["experiment_code"]
        for path_col, expect in (
            ("config_path", f"experiments/configs/{code}.yaml"),
            ("feature_path", f"experiments/features/{code}.parquet"),
            ("oof_primary_path", f"experiments/predictions/{code}/oof_primary.csv"),
            ("oof_shadow_path", f"experiments/predictions/{code}/oof_shadow.csv"),
            ("test_prediction_path", f"experiments/predictions/{code}/test.csv"),
        ):
            if str(r[path_col]) != expect:
                fail(f"{code} {path_col}={r[path_col]}")
            if not (ROOT / str(r[path_col])).exists():
                fail(f"missing {r[path_col]}")

        fpath = ROOT / str(r["feature_path"])
        feat = pd.read_parquet(fpath)
        if len(feat) != 324 or set(feat["id"].astype(str)) != all_ids:
            fail(f"{code} feature ids")
        cols = feature_column_names(feat)
        if int(r["n_features"]) != len(cols):
            fail(f"{code} n_features")
        if r["feature_space"] != FEATURE_SPACE:
            fail(f"{code} feature_space")
        if r["feature_sha256"] != file_sha256(fpath):
            fail(f"{code} feature_sha256")
        ch = feature_content_sha256(feat)
        if r["feature_content_sha256"] != ch:
            fail(f"{code} content hash")
        recipe_hashes.setdefault(str(r["feature_set_id"]), set()).add(ch)

        for key, idset in (
            ("oof_primary_path", dev_ids),
            ("oof_shadow_path", dev_ids),
            ("test_prediction_path", test_ids),
        ):
            pred = pd.read_csv(ROOT / str(r[key]))
            tgt = r["target"]
            if list(pred.columns) != ["id", tgt] or set(pred["id"].astype(str)) != idset:
                fail(f"{code} {key} schema")
            if not np_finite(pred[tgt]):
                fail(f"{code} {key} non-finite")

    for fsid, hashes in recipe_hashes.items():
        if len(hashes) != 1:
            fail(f"feature_set {fsid} hash mismatch")
        else:
            ok(f"feature_set {fsid} OK")

    man = ROOT / "submissions" / "submissions.csv"
    if not man.exists():
        fail("submissions.csv missing")
    else:
        m = pd.read_csv(man)
        for _, row in m.iterrows():
            sp = ROOT / str(row["submission_path"])
            if not sp.exists():
                fail(f"missing {sp}")
                continue
            if not re.match(r"^sub__EXP-T[0-9]+__EXP-H[0-9]+\.csv$", sp.name):
                fail(f"bad submission filename {sp.name}")
            sub = pd.read_csv(sp)
            if list(sub.columns) != ["id", "TmApp", "HIC"] or len(sub) != 162:
                fail(f"bad submission {sp}")
            if set(sub["id"].astype(str)) != test_ids:
                fail(f"bad submission ids {sp}")
        ok(f"submissions n={len(m)}")

    if (ROOT / "solution.csv").exists():
        fail("solution.csv present")
    else:
        ok("no solution.csv")

    out = ROOT / "results" / "VALIDATION.txt"
    text = "PASS\n" if not FAILS else "FAIL\n" + "\n".join(FAILS) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
