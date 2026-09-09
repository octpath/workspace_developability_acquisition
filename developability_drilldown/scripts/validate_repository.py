#!/usr/bin/env python3
"""Validate developability_drilldown (Phase 1 Linear/XGB + Phase 2A Transformer)."""
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
    N_EXPERIMENTS_TOTAL,
    N_FULL_LINEAR_XGB,
    N_LEGACY_MAP,
    N_TRANSFORMER,
    ROOT,
    SELECTION_POLICIES,
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
    if len(df) != N_EXPERIMENTS_TOTAL or len(codes) != N_EXPERIMENTS_TOTAL:
        fail(
            f"expected {N_EXPERIMENTS_TOTAL} experiments/codes; "
            f"experiments={len(df)} codes={len(codes)}"
        )
    else:
        ok(f"{N_EXPERIMENTS_TOTAL} experiments / codes")
    if len(legacy) != N_LEGACY_MAP:
        fail(f"legacy map expected {N_LEGACY_MAP}, got {len(legacy)}")
    else:
        ok(f"legacy map {N_LEGACY_MAP} (Linear/XGB only)")

    if df["experiment_code"].duplicated().any() or df["experiment_id"].duplicated().any():
        fail("duplicate experiment_code or experiment_id")

    fam = df["family"].value_counts().to_dict()
    if fam.get("LINEAR") != 42 or fam.get("XGBOOST") != 6 or fam.get("TRANSFORMER") != N_TRANSFORMER:
        fail(f"family counts unexpected: {fam}")
    else:
        ok(f"families LINEAR=42 XGBOOST=6 TRANSFORMER={N_TRANSFORMER}")

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
        if r["family"] in ("LINEAR", "XGBOOST") and not str(r.get("legacy_experiment_code") or ""):
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
    if next_code("TmApp") != "EXP-T045":
        fail(f"next TmApp unexpected: {next_code('TmApp')}")
    else:
        ok("next TmApp EXP-T045")
    if next_code("HIC") != "EXP-H034":
        fail(f"next HIC unexpected: {next_code('HIC')}")
    else:
        ok("next HIC EXP-H034")

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
    if len(merged) != N_EXPERIMENTS_TOTAL:
        fail("EXPERIMENT_CODES.csv mismatch")
    else:
        ok("codes table matches experiments.csv")

    leg_m = df[df["family"].isin(["LINEAR", "XGBOOST"])].merge(
        legacy,
        left_on=["legacy_experiment_code", "experiment_code", "experiment_id"],
        right_on=["legacy_experiment_code", "experiment_code", "experiment_id"],
        how="inner",
    )
    if len(leg_m) != N_LEGACY_MAP:
        fail("LEGACY_EXPERIMENT_CODE_MAP.csv mismatch vs Linear/XGB")
    else:
        ok("legacy map matches Linear/XGB")

    if not set(df["artifact_status"]).issubset(ARTIFACT_STATUSES):
        fail("bad artifact_status")
    if not set(df["source_reproducible"]).issubset(SOURCE_REPRO):
        fail("bad source_reproducible")
    if not set(df["drilldown_reproducible"]).issubset(DRILLDOWN_REPRO):
        fail("bad drilldown_reproducible")
    if not set(df["license_status"]).issubset(LICENSE_STATUSES):
        fail("bad license_status")
    if not set(df["selection_policy_at_creation"]).issubset(SELECTION_POLICIES):
        fail("bad selection_policy_at_creation")
    else:
        ok("repro/license/selection enums valid")

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

    # Historical Transformer feature parquet must not exist
    tr = df[df["family"] == "TRANSFORMER"]
    for _, r in tr.iterrows():
        code = r["experiment_code"]
        feat = ROOT / "experiments" / "features" / f"{code}.parquet"
        if feat.exists():
            fail(f"Transformer must not have feature parquet: {feat}")
        if str(r.get("feature_path") or "") not in ("", "nan"):
            fail(f"{code} feature_path must be empty")
        if str(r.get("feature_space") or "") not in ("", "nan"):
            fail(f"{code} feature_space must be empty")
        if r.get("representation_status") != "HISTORICAL_UNAVAILABLE":
            fail(f"{code} representation_status")
        if not str(r.get("input_space") or ""):
            fail(f"{code} missing input_space")
        if not str(r.get("input_asset_ref") or ""):
            fail(f"{code} missing input_asset_ref")
    else:
        ok("Transformer historical feature/representation contracts")

    fs_ids = set(pd.read_csv(ROOT / "results" / "FEATURE_SETS.csv")["feature_set_id"])
    for _, r in tr[tr["transformer_type"] == "FUSION"].iterrows():
        fs = str(r.get("feature_set_id") or "")
        if fs not in fs_ids:
            fail(f"{r['experiment_code']} fusion feature_set_id invalid: {fs}")
    else:
        ok("fusion feature_set FK")

    audit = ROOT / "results" / "TRANSFORMER_BACKFILL_AUDIT.csv"
    if not audit.exists():
        fail("TRANSFORMER_BACKFILL_AUDIT.csv missing")
    else:
        ad = pd.read_csv(audit)
        if len(ad) != N_TRANSFORMER:
            fail(f"audit rows {len(ad)}")
        else:
            ok("TRANSFORMER_BACKFILL_AUDIT.csv")

    dev, test, _ = load_dev_test_folds()
    dev_ids, test_ids = set(dev["id"]), set(test["id"])
    all_ids = dev_ids | test_ids

    full = df[df["artifact_status"] == "FULL"]
    lin_xgb_full = full[full["family"].isin(["LINEAR", "XGBOOST"])]
    tr_full = full[full["family"] == "TRANSFORMER"]
    if len(lin_xgb_full) != N_FULL_LINEAR_XGB:
        fail(f"Linear/XGB FULL count {len(lin_xgb_full)}")
    else:
        ok(f"Linear/XGB FULL={N_FULL_LINEAR_XGB}")
    if len(tr_full) != N_TRANSFORMER:
        fail(f"TRANSFORMER FULL count {len(tr_full)}")
    else:
        ok(f"TRANSFORMER FULL={N_TRANSFORMER}")

    recipe_hashes: dict[str, set[str]] = {}
    for _, r in lin_xgb_full.iterrows():
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

    for _, r in tr_full.iterrows():
        code = r["experiment_code"]
        for path_col, expect in (
            ("config_path", f"experiments/configs/{code}.yaml"),
            ("oof_primary_path", f"experiments/predictions/{code}/oof_primary.csv"),
            ("oof_shadow_path", f"experiments/predictions/{code}/oof_shadow.csv"),
            ("test_prediction_path", f"experiments/predictions/{code}/test.csv"),
        ):
            if str(r[path_col]) != expect:
                fail(f"{code} {path_col}={r[path_col]}")
            if not (ROOT / str(r[path_col])).exists():
                fail(f"missing {r[path_col]}")
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
