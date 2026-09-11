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
    CANONICAL_ELIGIBLE,
    CODE_RE,
    DRILLDOWN_REPRO,
    EXPERIMENTS_COLUMNS,
    FEATURE_SPACE,
    LEGACY_CODE_RE,
    LICENSE_STATUSES,
    N_CLASSICAL_REFINEMENT,
    N_EXPERIMENTS_TOTAL,
    N_FULL_LINEAR_XGB,
    N_HISTORICAL_TRANSFORMER,
    N_LEGACY_MAP,
    N_LINEAR,
    N_TRANSFORMER,
    N_XGBOOST,
    PRESERVATION_SNAPSHOT_77,
    REPRODUCIBILITY_STATUSES,
    ROOT,
    SELECTION_POLICIES,
    SHAREABILITY_STATUSES,
    SOURCE_REPRO,
    feature_column_names,
    feature_content_sha256,
    file_sha256,
    is_classical_refinement_code,
    is_historical_transformer_code,
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
    if (
        fam.get("LINEAR") != N_LINEAR
        or fam.get("XGBOOST") != N_XGBOOST
        or fam.get("TRANSFORMER") != N_TRANSFORMER
    ):
        fail(f"family counts unexpected: {fam}")
    else:
        ok(f"families LINEAR={N_LINEAR} XGBOOST={N_XGBOOST} TRANSFORMER={N_TRANSFORMER}")

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
        leg_raw = r.get("legacy_experiment_code")
        leg = "" if pd.isna(leg_raw) else str(leg_raw).strip()
        legacy_codes = set(legacy["experiment_code"])
        if c in legacy_codes and not leg:
            fail(f"missing legacy_experiment_code for mapped row {c}")
        if leg and c not in legacy_codes:
            fail(f"legacy_experiment_code on unmapped row {c}")

    m_count = int(df["experiment_code"].astype(str).str.startswith("EXP-M").sum())
    if m_count != 0:
        fail(f"unexpected EXP-M rows: {m_count}")
    else:
        ok("MULTI namespace empty (reserved)")
    if next_code("MULTI") != "EXP-M001":
        fail(f"next MULTI code unexpected: {next_code('MULTI')}")
    else:
        ok("next MULTI reserved EXP-M001")
    if next_code("TmApp") != "EXP-T142":
        fail(f"next TmApp unexpected: {next_code('TmApp')}")
    else:
        ok("next TmApp EXP-T142")
    if next_code("HIC") != "EXP-H102":
        fail(f"next HIC unexpected: {next_code('HIC')}")
    else:
        ok("next HIC EXP-H102")

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

    # Historical Transformer feature parquet must not exist; new architecture EXPs may materialize fixed branch
    tr = df[df["family"] == "TRANSFORMER"]
    for _, r in tr.iterrows():
        code = r["experiment_code"]
        feat = ROOT / "experiments" / "features" / f"{code}.parquet"
        if is_historical_transformer_code(code):
            if feat.exists():
                fail(f"Transformer must not have feature parquet: {feat}")
            if str(r.get("feature_path") or "") not in ("", "nan"):
                fail(f"{code} feature_path must be empty")
            if str(r.get("feature_space") or "") not in ("", "nan"):
                fail(f"{code} feature_space must be empty")
            if r.get("representation_status") != "HISTORICAL_UNAVAILABLE":
                fail(f"{code} representation_status")
        else:
            # New architecture Transformer (T065–T067 fusion+RASA/CA; T068+ joint HL)
            space = str(r.get("input_space") or "")
            is_joint_hl = ("JOINT_HL" in space) or space.startswith(
                "JOINT_HL_SINGLE_REG_FROZEN_RESIDUE"
            )
            is_separate_cross = ("CROSS_ATTENTION" in space) or ("SEPARATE_CROSS" in space)
            is_separate_dual = ("SEPARATE_DUAL_REG" in space) or (
                "WITHIN_CHAIN_EXTRA_ATTENTION" in space
            ) or ("REG_ONLY_CROSS" in space)
            is_protocol_v2 = ("PROTOCOL_V2" in space) or space.endswith("_PROTOCOL_V2")
            is_protocol_v3 = (
                ("COSINE_V3" in space)
                or ("FOLDLOCAL_COSINE_V3" in space)
                or space.endswith("_DL_FOLDLOCAL_COSINE_V3")
                or ("DL_FOLDLOCAL_COSINE_V3" in space)
                or space.endswith("_V3")  # V3 architecture/representation sweep
            )
            rasa = ROOT / "experiments" / "inputs" / f"{code}_rasa.parquet"
            ca = ROOT / "experiments" / "inputs" / f"{code}_ca.parquet"
            if (
                is_joint_hl
                or is_separate_cross
                or is_separate_dual
                or is_protocol_v2
                or is_protocol_v3
            ):
                # Joint H/L, separate+cross, or protocol-V2/V3: no fusion feature parquet
                if str(r.get("feature_path") or "") not in ("", "nan"):
                    fail(f"{code} no-fusion arch feature_path must be empty")
                if str(r.get("feature_space") or "") not in ("", "nan"):
                    fail(f"{code} no-fusion arch feature_space must be empty")
                if str(r.get("representation_status") or "") not in ("NOT_EXPORTED", "EXPORTED"):
                    fail(f"{code} representation_status")
                if "CA_DISTANCE" in space:
                    if not ca.exists():
                        fail(f"{code} missing CA coordinate input parquet")
                # else: no rasa / fusion required
            else:
                # Fusion fixed-branch (EXP-T065–T067)
                if not feat.exists():
                    fail(f"{code} missing fusion fixed-branch feature parquet")
                if str(r.get("feature_path") or "") != f"experiments/features/{code}.parquet":
                    fail(f"{code} feature_path")
                if str(r.get("feature_space") or "") != "FUSION_FIXED_BRANCH_RAW":
                    fail(f"{code} feature_space")
                if str(r.get("representation_status") or "") not in ("NOT_EXPORTED", "EXPORTED"):
                    fail(f"{code} representation_status")
                if "CA_DISTANCE" in space:
                    if not ca.exists():
                        fail(f"{code} missing CA coordinate input parquet")
                elif "RASA" in space:
                    if not rasa.exists():
                        fail(f"{code} missing RASA input parquet")
                else:
                    fail(
                        f"{code} unexpected new-architecture input_space "
                        f"without RASA/CA artifact: {space}"
                    )
        if not str(r.get("input_space") or ""):
            fail(f"{code} missing input_space")
        if not str(r.get("input_asset_ref") or ""):
            fail(f"{code} missing input_asset_ref")
    else:
        ok("Transformer feature/representation contracts")

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
        if len(ad) != N_HISTORICAL_TRANSFORMER:
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

    # --- Classical refinement closure checks ---
    freeze = ROOT / "results" / "CLASSICAL_REFINEMENT_FREEZE.yaml"
    if not freeze.exists():
        fail("CLASSICAL_REFINEMENT_FREEZE.yaml missing")
    else:
        ok("classical refinement freeze present")

    cand = ROOT / "results" / "CLASSICAL_REFINEMENT_CANDIDATES.csv"
    if not cand.exists():
        fail("CLASSICAL_REFINEMENT_CANDIDATES.csv missing")
    else:
        cdf = pd.read_csv(cand)
        if len(cdf) != N_CLASSICAL_REFINEMENT:
            fail(f"candidates expected {N_CLASSICAL_REFINEMENT}, got {len(cdf)}")
        else:
            ok(f"classical candidates n={N_CLASSICAL_REFINEMENT}")

    new_codes = df[df["experiment_code"].map(is_classical_refinement_code)]
    if len(new_codes) != N_CLASSICAL_REFINEMENT:
        fail(f"classical new rows expected {N_CLASSICAL_REFINEMENT}, got {len(new_codes)}")
    else:
        ok(f"classical refinement new experiments={N_CLASSICAL_REFINEMENT}")

    ens = new_codes[
        new_codes["ensemble_type"].astype(str).str.len().gt(0)
        | new_codes["member_experiment_codes"].astype(str).str.len().gt(0)
    ]
    if len(ens):
        fail(f"prediction ensemble fields on new rows: {ens['experiment_code'].tolist()[:5]}")
    else:
        ok("no prediction-level ensemble in classical refinement")

    if not all(new_codes["selection_policy_at_creation"] == "CV_SELECTED_POSTCOMP_EVALUATED"):
        fail("classical selection_policy_at_creation not CV_SELECTED_POSTCOMP_EVALUATED")
    else:
        ok("classical selection policy CV_SELECTED_POSTCOMP_EVALUATED")

    snap = pd.read_csv(PRESERVATION_SNAPSHOT_77)
    cur77 = df[df["experiment_code"].isin(snap["experiment_code"])]
    m = snap.merge(cur77, on="experiment_code", suffixes=("_old", "_new"))
    if len(m) != 77:
        fail(f"preservation snapshot merge rows {len(m)}")
    else:
        for c in ("cv_primary_mae", "cv_shadow_mae", "cv_worst_mae", "public_mae", "private_mae"):
            d = (
                pd.to_numeric(m[f"{c}_old"], errors="coerce")
                - pd.to_numeric(m[f"{c}_new"], errors="coerce")
            ).abs().max()
            if float(d) != 0.0 and not pd.isna(d):
                fail(f"preservation delta {c} max={d}")
        ok("existing 77 preservation PASS (score delta 0)")

    # --- Artifact completeness / shareability ---
    for col, allowed in (
        ("reproduction_status", REPRODUCIBILITY_STATUSES),
        ("shareability_status", SHAREABILITY_STATUSES),
        ("canonical_benchmark_eligible", CANONICAL_ELIGIBLE),
    ):
        if col not in df.columns:
            fail(f"missing column {col}")
        else:
            bad = set(df[col].astype(str)) - allowed - {"nan", "<NA>"}
            if bad:
                fail(f"bad {col} values: {bad}")
            else:
                ok(f"{col} enum OK")

    completeness = ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    if not completeness.exists():
        fail("EXPERIMENT_ARTIFACT_COMPLETENESS.csv missing")
    else:
        comp = pd.read_csv(completeness)
        if len(comp) != N_EXPERIMENTS_TOTAL:
            fail(f"completeness rows {len(comp)}")
        else:
            ok("EXPERIMENT_ARTIFACT_COMPLETENESS.csv")

    # New classical 40: T045–T064 and H034–H053 only (not later architecture EXPs)
    new40 = df[df["experiment_code"].map(is_classical_refinement_code)]
    if len(new40) != N_CLASSICAL_REFINEMENT:
        fail(f"classical refinement count {len(new40)}")
    for _, r in new40.iterrows():
        code = r["experiment_code"]
        fp = str(r.get("feature_path") or "")
        if "classical_cache" in fp:
            fail(f"{code} feature_path points to classical_cache")
        feat = ROOT / "experiments" / "features" / f"{code}.parquet"
        if not feat.exists():
            fail(f"{code} missing canonical feature parquet")
        for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
            if not (ROOT / "experiments" / "predictions" / code / name).exists():
                fail(f"{code} missing {name}")
        if str(r.get("shareability_status")) != "SHAREABLE_COMPLETE":
            fail(f"{code} not SHAREABLE_COMPLETE")
        if str(r.get("reproduction_status")) != "REPRODUCED":
            fail(f"{code} not REPRODUCED")
    else:
        ok("new 40 classical shareable+reproduced with canonical features")

    unverified = df[df["reproduction_status"].astype(str) == "UNVERIFIED_HISTORICAL"]
    if (unverified["canonical_benchmark_eligible"].astype(str) == "YES").any():
        fail("UNVERIFIED_HISTORICAL marked canonical_benchmark_eligible=YES")
    else:
        ok("UNVERIFIED_HISTORICAL excluded from canonical eligibility")

    out = ROOT / "results" / "VALIDATION.txt"
    text = "PASS\n" if not FAILS else "FAIL\n" + "\n".join(FAILS) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
