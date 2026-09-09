"""Tests for experiment artifact completeness / reproduction finalize."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _lib import (  # noqa: E402
    N_CLASSICAL_REFINEMENT,
    N_EXPERIMENTS_TOTAL,
    feature_content_sha256,
)


def _is_new40(code: str) -> bool:
    if code.startswith("EXP-T"):
        return int(code.split("-T")[1]) >= 45
    if code.startswith("EXP-H"):
        return int(code.split("-H")[1]) >= 34
    return False


@pytest.fixture(scope="module")
def exp():
    return pd.read_csv(ROOT / "results" / "experiments.csv")


@pytest.fixture(scope="module")
def audit():
    return pd.read_csv(ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv")


def test_completeness_audit_covers_all(exp, audit):
    assert len(exp) == N_EXPERIMENTS_TOTAL
    assert len(audit) == N_EXPERIMENTS_TOTAL
    assert set(audit["experiment_code"]) == set(exp["experiment_code"])


def test_new40_shareable_complete_and_reproduced(exp):
    n40 = exp[exp["experiment_code"].map(_is_new40)]
    assert len(n40) == N_CLASSICAL_REFINEMENT
    assert (n40["shareability_status"] == "SHAREABLE_COMPLETE").all()
    assert (n40["reproduction_status"] == "REPRODUCED").all()
    assert (n40["canonical_benchmark_eligible"] == "YES").all()
    for code in n40["experiment_code"]:
        feat = ROOT / "experiments" / "features" / f"{code}.parquet"
        assert feat.exists()
        assert "classical_cache" not in str(
            n40.loc[n40["experiment_code"] == code, "feature_path"].iloc[0]
        )
        for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
            assert (ROOT / "experiments" / "predictions" / code / name).exists()


def test_same_feature_set_same_content_hash(exp):
    n40 = exp[exp["experiment_code"].map(_is_new40)]
    g = n40.groupby("feature_set_id")["feature_content_sha256"].nunique()
    assert (g == 1).all()


def test_unverified_not_canonical(exp):
    bad = exp[
        (exp["reproduction_status"] == "UNVERIFIED_HISTORICAL")
        & (exp["canonical_benchmark_eligible"] == "YES")
    ]
    assert bad.empty


def test_fennix_unverified(exp):
    for code in ("EXP-T021", "EXP-T022"):
        row = exp[exp["experiment_code"] == code].iloc[0]
        assert row["reproduction_status"] == "UNVERIFIED_HISTORICAL"
        assert row["canonical_benchmark_eligible"] == "NO"


def test_no_classical_cache_as_canonical_feature_path(exp):
    fps = exp["feature_path"].dropna().astype(str)
    assert not fps.str.contains("classical_cache").any()


def test_catalog_best_excludes_unverified(exp):
    from build_catalog import best_rows

    for target in ("TmApp", "HIC"):
        sub = exp[exp["target"] == target]
        b = best_rows(sub, "cv_worst_mae")
        assert not b.empty
        assert b["reproduction_status"] != "UNVERIFIED_HISTORICAL"
        assert str(b["canonical_benchmark_eligible"]) == "YES"


def test_optuna_unused_for_classical(exp):
    classical = exp[exp["family"].isin(["LINEAR", "XGBOOST"])]
    # stored as string "false" or False
    vals = classical["optuna_used"].astype(str).str.lower()
    assert vals.isin(["false", "0", "nan", "<na>"]).all() or (
        classical["optuna_used"].isna() | (classical["optuna_used"].astype(str) == "false")
    ).all()
