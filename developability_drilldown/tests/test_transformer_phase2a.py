"""Phase 2A Transformer tests: codes, equivalence, contracts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))

from experiment_codes import load_codes, next_code  # noqa: E402
from _lib import N_EXPERIMENTS_TOTAL, N_FULL_LINEAR_XGB, N_TRANSFORMER, N_XGBOOST  # noqa: E402
from antibody_transformer.equivalence import (  # noqa: E402
    compare_forward,
    compare_one_step,
    make_pair,
    synthetic_batch,
)


TOL = 1e-6


def test_transformer_code_append_only():
    codes = load_codes()
    assert len(codes) == N_EXPERIMENTS_TOTAL
    t = [c for c in codes["experiment_code"] if c.startswith("EXP-T")]
    h = [c for c in codes["experiment_code"] if c.startswith("EXP-H")]
    assert len(t) == 65 and len(h) == 53
    assert next_code("TmApp") == "EXP-T066"
    assert next_code("HIC") == "EXP-H054"
    assert next_code("MULTI") == "EXP-M001"
    assert not any(c.startswith("EXP-M") for c in codes["experiment_code"])


def test_30_configs_registered():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    tr = exp[exp["family"] == "TRANSFORMER"]
    assert len(tr) == 30
    for _, r in tr.iterrows():
        cfg = ROOT / str(r["config_path"])
        assert cfg.exists(), r["experiment_code"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(content_mode="scratch", annotation_mode="full", merge_mode="concat", chain_mode="HL"),
        dict(content_mode="scratch", annotation_mode="full", merge_mode="mean", chain_mode="HL"),
        dict(
            content_mode="frozen",
            annotation_mode="full",
            merge_mode="concat",
            chain_mode="HL",
            plm_hidden=1280,
        ),
        dict(
            content_mode="scratch",
            annotation_mode="full",
            merge_mode="h_only",
            chain_mode="H_ONLY",
        ),
        dict(
            content_mode="frozen",
            annotation_mode="full",
            merge_mode="concat",
            chain_mode="HL",
            pooling_mode="region_gate",
            plm_hidden=480,
        ),
    ],
)
def test_forward_repr_and_pred_equivalence(kwargs):
    old, new = make_pair(**kwargs, seed=0)
    batch = synthetic_batch(
        content_mode=kwargs.get("content_mode", "scratch"),
        chain_mode=kwargs.get("chain_mode", "HL"),
        annotation_mode=kwargs.get("annotation_mode", "full"),
        plm_hidden=kwargs.get("plm_hidden", 1280),
        seed=99,
    )
    stats = compare_forward(old, new, batch)
    assert stats["repr_max_abs"] <= TOL
    assert stats["pred_max_abs"] <= TOL


def test_fusion_forward_equivalence():
    old, new = make_pair(
        content_mode="scratch",
        annotation_mode="full",
        merge_mode="mean",
        chain_mode="HL",
        fusion=True,
        fixed_dim=32,
        seed=1,
    )
    batch = synthetic_batch(content_mode="scratch", seed=5)
    fixed = torch.randn(batch["heavy_mask"].shape[0], 32)
    stats = compare_forward(old, new, batch, fixed=fixed)
    assert stats["repr_max_abs"] <= TOL
    assert stats["pred_max_abs"] <= TOL


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(content_mode="scratch", merge_mode="concat"),
        dict(content_mode="frozen", plm_hidden=480, pooling_mode="region_gate"),
        dict(content_mode="scratch", chain_mode="H_ONLY", merge_mode="h_only"),
    ],
)
def test_one_step_training_equivalence(kwargs):
    old, new = make_pair(**kwargs, annotation_mode="full", seed=2)
    batch = synthetic_batch(
        content_mode=kwargs.get("content_mode", "scratch"),
        chain_mode=kwargs.get("chain_mode", "HL"),
        plm_hidden=kwargs.get("plm_hidden", 1280),
        seed=11,
    )
    y = torch.randn(batch["heavy_mask"].shape[0])
    stats = compare_one_step(old, new, batch, y=y, seed=7)
    assert stats["initial_pred_max_abs"] <= TOL
    assert stats["loss_abs_delta"] <= TOL
    assert stats["grad_max_abs"] <= TOL
    assert stats["param_max_abs"] <= TOL


def test_fusion_one_step_equivalence():
    old, new = make_pair(fusion=True, fixed_dim=16, seed=3)
    batch = synthetic_batch(seed=12)
    fixed = torch.randn(batch["heavy_mask"].shape[0], 16)
    y = torch.randn(batch["heavy_mask"].shape[0])
    stats = compare_one_step(old, new, batch, fixed=fixed, y=y, seed=8)
    assert stats["loss_abs_delta"] <= TOL
    assert stats["grad_max_abs"] <= TOL
    assert stats["param_max_abs"] <= TOL


def test_input_metadata_by_plm():
    from _lib import is_historical_transformer_code

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    tr = exp[exp["family"] == "TRANSFORMER"]
    hist = tr[tr["experiment_code"].map(is_historical_transformer_code)]
    scratch = hist[hist["plm_source"] == "NONE"]
    assert (scratch["input_space"].isin(["SCRATCH_RESIDUE_SEQUENCE", "RESIDUE_PLUS_FIXED_FEATURES"])).all()
    assert (hist[hist["plm_source"] == "ABLINGUA"]["input_space"].isin(
        ["FROZEN_RESIDUE_EMBEDDING", "RESIDUE_PLUS_FIXED_FEATURES"]
    )).all()
    assert (hist[hist["plm_source"] == "ABLANG2"]["input_space"].isin(
        ["FROZEN_RESIDUE_EMBEDDING", "RESIDUE_PLUS_FIXED_FEATURES"]
    )).all()
    assert (hist[hist["plm_source"] == "ESM2"]["input_space"].isin(
        ["FROZEN_RESIDUE_EMBEDDING", "RESIDUE_PLUS_FIXED_FEATURES"]
    )).all()
    assert (tr["input_asset_ref"].astype(str).str.len() > 0).all()
    t065 = tr[tr["experiment_code"] == "EXP-T065"]
    assert len(t065) == 1
    assert t065.iloc[0]["input_space"] == "RESIDUE_PLUS_FIXED_FEATURES_PLUS_RASA"


def test_fusion_feature_set_fk():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    fs = set(pd.read_csv(ROOT / "results" / "FEATURE_SETS.csv")["feature_set_id"])
    fus = exp[(exp["family"] == "TRANSFORMER") & (exp["transformer_type"] == "FUSION")]
    assert len(fus) == 16  # 15 historical + EXP-T065
    assert set(fus["feature_set_id"]).issubset(fs)


def test_historical_representation_unavailable_contract():
    from _lib import is_historical_transformer_code

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    hist = exp[
        (exp["family"] == "TRANSFORMER")
        & exp["experiment_code"].map(is_historical_transformer_code)
    ]
    assert (hist["representation_status"] == "HISTORICAL_UNAVAILABLE").all()
    for code in hist["experiment_code"]:
        assert not (ROOT / "experiments" / "features" / f"{code}.parquet").exists()
    assert (hist["feature_path"].fillna("") == "").all()
    assert (hist["feature_space"].fillna("") == "").all()
    t065 = exp[exp["experiment_code"] == "EXP-T065"].iloc[0]
    assert t065["representation_status"] == "NOT_EXPORTED"
    assert (ROOT / "experiments" / "features" / "EXP-T065.parquet").exists()
    assert (ROOT / "experiments" / "inputs" / "EXP-T065_rasa.parquet").exists()


def test_transformer_full_and_linear_xgb_unchanged():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert len(exp[exp["artifact_status"] == "FULL"]) == N_FULL_LINEAR_XGB + N_TRANSFORMER
    lin = exp[(exp["family"] == "LINEAR") & (exp["artifact_status"] == "FULL")]
    # 40 historical LINEAR minus 2 SCORE_ONLY OPENMM + 32 classical-refinement LINEAR
    assert len(lin) == 72
    assert (lin["feature_space"] == "RAW_PREPROCESS").all()
    xgb = exp[(exp["family"] == "XGBOOST") & (exp["artifact_status"] == "FULL")]
    assert len(xgb) == N_XGBOOST
    assert (xgb["feature_space"] == "RAW_PREPROCESS").all()


def test_prediction_source_destination_equality():
    audit = pd.read_csv(ROOT / "results" / "TRANSFORMER_BACKFILL_AUDIT.csv")
    assert len(audit) == 29
    assert (audit["oof_value_max_abs_delta"] <= 1e-12).all()
    assert (audit["test_value_max_abs_delta"] <= 1e-12).all()


def test_score_authority_equality():
    from _lib import is_historical_transformer_code

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    mbs = pd.read_csv(REPO / "top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv")
    tr = exp[
        (exp["family"] == "TRANSFORMER")
        & exp["experiment_code"].map(is_historical_transformer_code)
    ]
    for _, r in tr.iterrows():
        row = mbs[mbs["model_id"] == r["source_model_id"]].iloc[0]
        for a, b in [
            ("cv_primary_mae", "cv_primary_mae"),
            ("cv_shadow_mae", "cv_shadow_mae"),
            ("public_mae", "public_mae"),
            ("private_mae", "private_mae"),
            ("test_overall_mae", "overall_test_mae"),
        ]:
            assert abs(float(r[a]) - float(row[b])) < 1e-12, (r["experiment_code"], a)


def test_selection_and_license_enums():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    tr = exp[exp["family"] == "TRANSFORMER"]
    assert set(tr["selection_policy_at_creation"]).issubset(
        {"CV_ONLY", "CV_SELECTED_POSTCOMP_EVALUATED", "POSTCOMP_EXPLORATORY", "UNKNOWN"}
    )
    assert set(tr["license_status"]).issubset({"OK", "REVIEW", "RESTRICTED", "UNKNOWN"})
    assert (tr["current_evaluation_mode"] == "POSTCOMP_EXPLORATORY").all()


def test_catalog_build_has_transformer():
    from build_catalog import catalog_md  # noqa: E402

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    md = catalog_md(exp, "en")
    assert "TRANSFORMER" in md
    assert "TRF_TM_SCRATCH_MIN_CONCAT" in md or "EXP-T026" in md
