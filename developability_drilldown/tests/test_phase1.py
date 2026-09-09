"""Phase 1 unit tests for developability_drilldown."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _lib import (  # noqa: E402
    LIN_TOP6_MAP,
    XGB_MAP,
    feature_content_sha256,
    lin_experiment_id,
    xgb_experiment_id,
)


def test_experiment_id_uniqueness():
    ids = [lin_experiment_id(s) for s in LIN_TOP6_MAP]
    ids += [xgb_experiment_id(s) for s in XGB_MAP]
    # also generate a few registry-style ids
    extra = [
        "TM_SEQ_BASIC__RIDGE",
        "HIC_CONSTANT__LASSO",
        "TM_PARENT_ABLINGUA_CDR3__LASSO",
    ]
    ids += [lin_experiment_id(s) for s in extra]
    assert len(ids) == len(set(ids))


def test_top6_friendly_ids():
    assert lin_experiment_id("TM_PARENT_ABLINGUA_CDR3__RIDGE") == "LIN_TM_ABLINGUA_CDR3_RIDGE"
    assert xgb_experiment_id("XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO") == "XGB_HIC_CONTINUOUS_SURFACE"


def test_feature_hash_deterministic():
    path = ROOT / "experiments" / "features" / "LIN_TM_BIOEMU_MPNN_RIDGE.parquet"
    if not path.exists():
        pytest.skip("features not exported yet")
    df = pd.read_parquet(path)
    h1 = feature_content_sha256(df)
    h2 = feature_content_sha256(df.copy())
    assert h1 == h2
    twin = ROOT / "experiments" / "features" / "XGB_TM_BIOEMU_MPNN.parquet"
    if twin.exists():
        assert feature_content_sha256(pd.read_parquet(twin)) == h1


def test_prediction_schema_full_linear():
    path = ROOT / "experiments" / "predictions" / "LIN_TM_ABLINGUA_CDR3_RIDGE" / "test.csv"
    if not path.exists():
        pytest.skip("linear preds not backfilled")
    df = pd.read_csv(path)
    assert list(df.columns) == ["id", "TmApp"]
    assert len(df) == 162


def test_xgb_artifact_completeness():
    exp = ROOT / "results" / "experiments.csv"
    if not exp.exists():
        pytest.skip("registry missing")
    df = pd.read_csv(exp)
    xgb = df[df["family"] == "XGBOOST"]
    assert len(xgb) == 6
    for _, r in xgb.iterrows():
        assert r["artifact_status"] in ("FULL", "INCONSISTENT")
        for col in ("oof_primary_path", "oof_shadow_path", "test_prediction_path", "feature_path"):
            p = ROOT / str(r[col])
            assert p.exists(), p


def test_composer_rejects_wrong_targets(tmp_path):
    # import compose helpers
    sys.path.insert(0, str(SCRIPTS))
    import compose_submission as cs

    exp = ROOT / "results" / "experiments.csv"
    if not exp.exists():
        pytest.skip("registry missing")
    with pytest.raises(SystemExit):
        cs.compose("XGB_HIC_CONTINUOUS_SURFACE", "LIN_TM_ABLINGUA_CDR3_RIDGE", register=False)


def test_composer_roundtrip():
    exp = ROOT / "results" / "experiments.csv"
    if not exp.exists():
        pytest.skip("registry missing")
    import compose_submission as cs

    dest = cs.compose(
        "LIN_TM_ABLINGUA_CDR3_RIDGE",
        "LIN_HIC_HYDRO_TITRATION_LASSO",
        register=False,
    )
    df = pd.read_csv(dest)
    assert list(df.columns) == ["id", "TmApp", "HIC"]
    assert len(df) == 162


def test_master_derived_scores():
    exp = ROOT / "results" / "experiments.csv"
    if not exp.exists():
        pytest.skip("registry missing")
    df = pd.read_csv(exp)
    full = df[df["artifact_status"] == "FULL"]
    for _, r in full.iterrows():
        p, s = float(r["cv_primary_mae"]), float(r["cv_shadow_mae"])
        assert abs(float(r["cv_mean_mae"]) - (p + s) / 2) < 1e-12
        assert abs(float(r["cv_worst_mae"]) - max(p, s)) < 1e-12


def test_no_submission_named_predictions():
    pred = ROOT / "experiments" / "predictions"
    if not pred.exists():
        pytest.skip("no predictions")
    for p in pred.rglob("*"):
        assert "submission" not in p.name.lower()
        assert not p.name.startswith("sub_")
