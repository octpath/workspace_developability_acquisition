"""Tests for target-namespaced experiment codes (EXP-T / EXP-H / EXP-M)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_codes import (  # noqa: E402
    CODE_RE,
    id_to_code,
    load_codes,
    load_legacy_map,
    next_code,
    resolve_experiment_ref,
)
from _lib import N_EXPERIMENTS_TOTAL, N_XGBOOST, feature_content_sha256  # noqa: E402


def test_code_format_and_counts():
    codes = load_codes()
    assert len(codes) == N_EXPERIMENTS_TOTAL
    assert all(CODE_RE.match(c) for c in codes["experiment_code"])
    t = [c for c in codes["experiment_code"] if c.startswith("EXP-T")]
    h = [c for c in codes["experiment_code"] if c.startswith("EXP-H")]
    assert len(t) == 141 and len(h) == 139
    assert sorted(t, key=lambda x: int(x.split("-")[1][1:])) == [f"EXP-T{i:03d}" for i in range(1, 142)]
    assert sorted(h, key=lambda x: int(x.split("-")[1][1:])) == [f"EXP-H{i:03d}" for i in range(1, 140)]
    assert not any(c.startswith("EXP-M") for c in codes["experiment_code"])


def test_legacy_map_48():
    leg = load_legacy_map()
    assert len(leg) == 48
    assert leg["legacy_experiment_code"].is_unique
    assert leg["experiment_code"].is_unique
    assert id_to_code("LIN_TM_ABLINGUA_CDR3_RIDGE") == "EXP-T001"
    assert resolve_experiment_ref("EXP-T001")[1] == "LIN_TM_ABLINGUA_CDR3_RIDGE"


def test_legacy_order_preserved_within_target():
    leg = load_legacy_map()
    tm = leg[leg["target"] == "TmApp"].copy()
    tm["ln"] = tm["legacy_experiment_code"].str.replace("EXP", "").astype(int)
    tm["tn"] = tm["experiment_code"].str.replace("EXP-T", "").astype(int)
    tm = tm.sort_values("ln")
    assert list(tm["tn"]) == list(range(1, len(tm) + 1))
    hic = leg[leg["target"] == "HIC"].copy()
    hic["ln"] = hic["legacy_experiment_code"].str.replace("EXP", "").astype(int)
    hic["hn"] = hic["experiment_code"].str.replace("EXP-H", "").astype(int)
    hic = hic.sort_values("ln")
    assert list(hic["hn"]) == list(range(1, len(hic) + 1))


def test_next_code_namespaces():
    assert next_code("TmApp") == "EXP-T142"
    assert next_code("HIC") == "EXP-H140"
    assert next_code("MULTI") == "EXP-M001"
    assert next_code("T") == "EXP-T142"
    assert next_code("H") == "EXP-H140"
    assert next_code("M") == "EXP-M001"


def test_resolver_legacy_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        code, eid = resolve_experiment_ref("EXP001")
        assert code == "EXP-T001"
        assert eid == "LIN_TM_ABLINGUA_CDR3_RIDGE"
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_no_renumber_authority():
    codes = load_codes().set_index("experiment_id")["experiment_code"]
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    for _, r in exp.iterrows():
        assert codes[r["experiment_id"]] == r["experiment_code"]
        if r["family"] in ("LINEAR", "XGBOOST"):
            assert r["legacy_experiment_code"]


def test_feature_set_equivalence():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    full = exp[(exp["artifact_status"] == "FULL") & (exp["family"].isin(["LINEAR", "XGBOOST"]))]
    for fsid, g in full.groupby("feature_set_id"):
        assert len(set(g["feature_content_sha256"])) == 1


def test_feature_hash_deterministic():
    path = ROOT / "experiments" / "features" / "EXP-T003.parquet"
    df = pd.read_parquet(path)
    assert feature_content_sha256(df) == feature_content_sha256(df.copy())
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    fs = exp.loc[exp["experiment_code"] == "EXP-T003", "feature_set_id"].iloc[0]
    twins = exp[
        (exp["feature_set_id"] == fs)
        & (exp["artifact_status"] == "FULL")
        & (exp["family"].isin(["LINEAR", "XGBOOST"]))
        & (exp["experiment_code"] != "EXP-T003")
    ]
    other = ROOT / "experiments" / "features" / f"{twins.iloc[0]['experiment_code']}.parquet"
    assert feature_content_sha256(pd.read_parquet(other)) == feature_content_sha256(df)


def test_prediction_schema():
    df = pd.read_csv(ROOT / "experiments" / "predictions" / "EXP-T001" / "test.csv")
    assert list(df.columns) == ["id", "TmApp"] and len(df) == 162


def test_xgb_full_paths():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    xgb = exp[exp["family"] == "XGBOOST"]
    assert len(xgb) == N_XGBOOST
    for _, r in xgb.iterrows():
        assert r["artifact_status"] == "FULL"
        assert str(r["experiment_code"]).startswith(("EXP-T", "EXP-H"))
        for col in ("feature_path", "oof_primary_path", "test_prediction_path"):
            assert (ROOT / str(r[col])).exists()


def test_composer_th_happy_and_rejects():
    import compose_submission as cs

    dest = cs.compose("EXP-T001", "EXP-H001", register=False)
    assert dest.name == "sub__EXP-T001__EXP-H001.csv"
    df = pd.read_csv(dest)
    assert list(df.columns) == ["id", "TmApp", "HIC"] and len(df) == 162

    dest2 = cs.compose("LIN_TM_ABLINGUA_CDR3_RIDGE", "LIN_HIC_HYDRO_TITRATION_LASSO", register=False)
    assert dest2.name == "sub__EXP-T001__EXP-H001.csv"

    with pytest.raises(SystemExit):
        cs.compose("EXP-H001", "EXP-T001", register=False)


def test_repro_license_enums():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert "reproducible" not in exp.columns
    assert set(exp["source_reproducible"]).issubset({"YES", "NO", "UNKNOWN"})
    assert set(exp["drilldown_reproducible"]).issubset({"YES", "PARTIAL", "NO"})
    assert set(exp["license_status"]).issubset({"OK", "REVIEW", "RESTRICTED", "UNKNOWN"})


def test_master_derived_scores():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    full = exp[exp["artifact_status"] == "FULL"]
    for _, r in full.iterrows():
        p, s = float(r["cv_primary_mae"]), float(r["cv_shadow_mae"])
        assert abs(float(r["cv_mean_mae"]) - (p + s) / 2) < 1e-12


def test_no_submission_named_predictions():
    for p in (ROOT / "experiments" / "predictions").rglob("*"):
        assert "submission" not in p.name.lower()
        assert not p.name.startswith("sub_")


def test_score_schema():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert len(exp) == N_EXPERIMENTS_TOTAL
    assert "feature_set_id" in exp.columns and "legacy_experiment_code" in exp.columns
    assert "feature_recipe" not in exp.columns
    assert "input_space" in exp.columns and "representation_status" in exp.columns
