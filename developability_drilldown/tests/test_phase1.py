"""Phase 1 + registry migration tests for developability_drilldown."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from experiment_codes import (  # noqa: E402
    CODE_RE,
    id_to_code,
    load_codes,
    next_code,
    resolve_experiment_ref,
)
from _lib import feature_content_sha256  # noqa: E402


def test_exp_codes_001_048():
    codes = load_codes()
    assert len(codes) == 48
    assert list(codes["experiment_code"]) == [f"EXP{i:03d}" for i in range(1, 49)]
    assert codes["experiment_id"].is_unique
    assert codes["experiment_code"].is_unique


def test_code_id_roundtrip():
    assert id_to_code("LIN_TM_ABLINGUA_CDR3_RIDGE") == "EXP001"
    code, eid = resolve_experiment_ref("EXP001")
    assert code == "EXP001" and eid == "LIN_TM_ABLINGUA_CDR3_RIDGE"
    code2, eid2 = resolve_experiment_ref("LIN_TM_ABLINGUA_CDR3_RIDGE")
    assert code2 == "EXP001" and eid2 == eid


def test_append_only_next_code():
    assert next_code() == "EXP049"


def test_no_renumber_on_catalog_authority():
    """EXPERIMENT_CODES.csv is authority; registry codes must match."""
    codes = load_codes().set_index("experiment_id")["experiment_code"]
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    for _, r in exp.iterrows():
        assert codes[r["experiment_id"]] == r["experiment_code"]


def test_feature_set_equivalence():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    full = exp[exp["artifact_status"] == "FULL"]
    for fsid, g in full.groupby("feature_set_id"):
        hashes = set(g["feature_content_sha256"])
        assert len(hashes) == 1, fsid
        assert not str(fsid).endswith(("_RIDGE", "_LASSO"))


def test_feature_hash_deterministic():
    path = ROOT / "experiments" / "features" / "EXP003.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert feature_content_sha256(df) == feature_content_sha256(df.copy())
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    fs = exp.loc[exp["experiment_code"] == "EXP003", "feature_set_id"].iloc[0]
    twins = exp[
        (exp["feature_set_id"] == fs)
        & (exp["experiment_code"] != "EXP003")
        & (exp["artifact_status"] == "FULL")
    ]
    assert len(twins) >= 1
    other = ROOT / "experiments" / "features" / f"{twins.iloc[0]['experiment_code']}.parquet"
    assert feature_content_sha256(pd.read_parquet(other)) == feature_content_sha256(df)


def test_prediction_schema_full():
    path = ROOT / "experiments" / "predictions" / "EXP001" / "test.csv"
    df = pd.read_csv(path)
    assert list(df.columns) == ["id", "TmApp"]
    assert len(df) == 162


def test_xgb_artifact_completeness():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    xgb = exp[exp["family"] == "XGBOOST"]
    assert len(xgb) == 6
    for _, r in xgb.iterrows():
        assert r["artifact_status"] == "FULL"
        for col in ("oof_primary_path", "oof_shadow_path", "test_prediction_path", "feature_path"):
            assert (ROOT / str(r[col])).exists()
            assert f"/{r['experiment_code']}" in str(r[col]) or str(r[col]).endswith(
                f"{r['experiment_code']}.parquet"
            ) or str(r[col]).endswith(f"{r['experiment_code']}.yaml") or f"/{r['experiment_code']}/" in str(
                r[col]
            )


def test_repro_license_enums():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert "reproducible" not in exp.columns
    assert set(exp["source_reproducible"]).issubset({"YES", "NO", "UNKNOWN"})
    assert set(exp["drilldown_reproducible"]).issubset({"YES", "PARTIAL", "NO"})
    assert set(exp["license_status"]).issubset({"OK", "REVIEW", "RESTRICTED", "UNKNOWN"})


def test_composer_code_and_id():
    import compose_submission as cs

    dest = cs.compose("EXP001", "EXP004", register=False)
    assert dest.name == "sub__EXP001__EXP004.csv"
    df = pd.read_csv(dest)
    assert list(df.columns) == ["id", "TmApp", "HIC"]
    assert len(df) == 162
    # descriptive id also works
    dest2 = cs.compose("LIN_TM_ABLINGUA_CDR3_RIDGE", "LIN_HIC_HYDRO_TITRATION_LASSO", register=False)
    assert dest2.name == "sub__EXP001__EXP004.csv"


def test_composer_rejects_wrong_targets():
    import compose_submission as cs

    with pytest.raises(SystemExit):
        cs.compose("EXP004", "EXP001", register=False)


def test_master_derived_scores():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    full = exp[exp["artifact_status"] == "FULL"]
    for _, r in full.iterrows():
        p, s = float(r["cv_primary_mae"]), float(r["cv_shadow_mae"])
        assert abs(float(r["cv_mean_mae"]) - (p + s) / 2) < 1e-12
        assert abs(float(r["cv_worst_mae"]) - max(p, s)) < 1e-12


def test_no_submission_named_predictions():
    pred = ROOT / "experiments" / "predictions"
    for p in pred.rglob("*"):
        assert "submission" not in p.name.lower()
        assert not p.name.startswith("sub_")


def test_score_columns_present():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert len(exp) == 48
    assert "feature_set_id" in exp.columns and "source_recipe_id" in exp.columns
    assert "feature_recipe" not in exp.columns
