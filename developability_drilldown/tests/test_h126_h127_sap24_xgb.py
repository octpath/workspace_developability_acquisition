#!/usr/bin/env python3
"""Tests for H126–H127 SOURCE_SAP24 fold-local shallow XGBoost."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_source_sap24_exact_24_and_hash():
    from _lib import feature_content_sha256

    src = ROOT.parent / "feature_research/hic_sap_scm_source/features/antibody_source_sap24.parquet"
    df = pd.read_parquet(src)
    cols = [c for c in df.columns if c != "id"]
    assert len(cols) == 24
    prereg = yaml.safe_load((ROOT / "results/H126_H127_SOURCE_SAP24_XGB_PREREGISTRATION.yaml").read_text())
    assert prereg["n_dims"] == 24
    assert prereg["feature_columns"] == cols
    assert prereg["feature_content_sha256"] == feature_content_sha256(df)


def test_xgb_config_shallow_fixed_lr():
    from fold_local_xgb import XGBConfig

    c = XGBConfig(max_depth=2)
    assert c.learning_rate == 0.02
    assert c.max_depth <= 3
    assert c.n_estimators == 5000
    assert c.early_stopping_rounds == 200
    assert c.random_state == 101


def test_median_impute_train_only_semantics():
    from fold_local_xgb import median_impute_apply, median_impute_fit

    Xtr = np.array([[1.0, np.nan], [3.0, 2.0]])
    med = median_impute_fit(Xtr)
    assert med[0] == pytest.approx(2.0)
    assert med[1] == pytest.approx(2.0)
    Xva = np.array([[np.nan, 9.0]])
    out = median_impute_apply(Xva, med)
    assert out[0, 0] == pytest.approx(2.0)


def test_fold_model_roundtrip_if_trained():
    code = "EXP-H126"
    run = ROOT / "results" / f"{code}_run"
    if not run.exists():
        pytest.skip("not trained yet")
    from fold_local_xgb import reload_fold_model

    model, med = reload_fold_model(run, "primary", 0, seed=101)
    X = np.zeros((2, 24))
    pred = model.predict(X)
    assert pred.shape == (2,)
    assert med.shape == (24,)


def test_no_full_dev_refit_flag():
    for code in ("EXP-H126", "EXP-H127"):
        p = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
        if not p.exists():
            pytest.skip("not trained")
        doc = yaml.safe_load(p.read_text())
        assert doc["no_full_dev_refit"] is True


def test_oof_and_external_paths_transformer_semantics():
    code = "EXP-H126"
    pred = ROOT / "experiments" / "predictions" / code
    if not pred.exists():
        pytest.skip("not trained")
    for name in (
        "oof_primary.csv",
        "oof_shadow.csv",
        "oof_val_primary.csv",
        "oof_test_primary.csv",
        "test.csv",
        "test_primary_mean.csv",
        "test_primary_median.csv",
        "test_shadow_mean.csv",
        "test_shadow_median.csv",
        "test_primary_fold0.csv",
    ):
        assert (pred / name).exists(), name
    # fold models exist
    assert len(list((ROOT / f"results/{code}_run/fold_models").glob("*.json"))) >= 20


def test_historical_transformer_untouched_marker():
    cfg = yaml.safe_load((ROOT / "experiments/configs/EXP-H114.yaml").read_text())
    assert cfg["fusion_mode"] == "late_concat_direct"
    assert cfg["fusion_bundle_id"] == "FS_HIC_SOURCE_SAP24"


def test_series_codes():
    from run_exp_h126_h127_sap24_xgb import SERIES

    assert [s["code"] for s in SERIES] == ["EXP-H126", "EXP-H127"]
    assert SERIES[0]["max_depth"] == 2
    assert SERIES[1]["max_depth"] == 3
