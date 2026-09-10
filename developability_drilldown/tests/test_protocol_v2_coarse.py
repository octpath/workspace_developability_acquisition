#!/usr/bin/env python3
"""Tests for EXP-T074 coarse-cosine protocol platform."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.protocol_v2_coarse import (  # noqa: E402
    COARSE_LR_GRID,
    COSINE_EPOCHS,
    DEFAULT_SEED,
    ETA_MIN_FRAC,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    PROTOCOL_ID,
    coarse_lr_grid,
    eta_min_for,
    lr_at_epoch,
)
from antibody_transformer.protocol_v2 import DEFAULT_SEED as V2_SEED  # noqa: E402
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402


def test_platform_constants():
    assert PROTOCOL_ID == "DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2_COARSE_COSINE"
    assert DEFAULT_SEED == 101 == V2_SEED
    assert MAX_EPOCHS == 200
    assert MIN_EPOCHS == 100
    assert PATIENCE == 20
    assert COSINE_EPOCHS == 100
    assert ETA_MIN_FRAC == 0.01
    assert coarse_lr_grid() == pytest.approx([1e-5, 1e-4, 1e-3, 1e-2])
    assert COARSE_LR_GRID == [1e-5, 1e-4, 1e-3, 1e-2]


def test_cosine_then_hold_schedule():
    lr0 = 1e-2
    eta = eta_min_for(lr0)
    assert abs(eta - 1e-4) < 1e-15
    assert abs(lr_at_epoch(1, lr0, eta) - lr0) < 1e-12
    assert abs(lr_at_epoch(101, lr0, eta) - eta) < 1e-12
    assert abs(lr_at_epoch(150, lr0, eta) - eta) < 1e-12
    assert abs(lr_at_epoch(200, lr0, eta) - eta) < 1e-12
    # mid decay strictly between
    mid = lr_at_epoch(51, lr0, eta)
    assert eta < mid < lr0
    # never rises after t=100
    assert lr_at_epoch(120, lr0, eta) == pytest.approx(eta)
    assert lr_at_epoch(180, lr0, eta) == pytest.approx(eta)


def test_no_warmup_in_module():
    import inspect
    import antibody_transformer.protocol_v2_coarse as m

    src = inspect.getsource(m.train_one_candidate_coarse)
    assert "warmup" not in src.lower()
    assert "torch.optim.lr_scheduler.CosineAnnealingLR" not in src
    run_src = inspect.getsource(m.run_protocol_v2_coarse)
    assert "full_dev_transformer_predict" not in run_src
    assert "no_full_dev_refit" in run_src
    assert "nested" not in src.lower()


def test_min_epochs_gate_in_source():
    import inspect
    import antibody_transformer.protocol_v2_coarse as m

    src = inspect.getsource(m.train_one_candidate_coarse)
    assert "min_epochs" in src
    assert "early_ok" in src or "epoch >= min_epochs" in src


def test_t030_architecture_flags():
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
    )
    assert not m.joint_hl_single_reg
    assert not m.joint_hl_dual_reg
    assert not m.use_cross_attention_bridge


def test_runner_guards():
    text = (ROOT / "scripts" / "run_exp_t074_protocol_v2_coarse.py").read_text()
    assert "no_nested_cv" in text
    assert "no_full_dev_refit" in text
    assert "full_dev_transformer_predict" not in text
    assert "1e-5" in text or "coarse_lr_grid" in text


def test_exp_t074_artifacts_if_present():
    import pandas as pd

    code = "EXP-T074"
    if not (ROOT / "experiments" / "configs" / f"{code}.yaml").exists():
        pytest.skip("EXP-T074 not registered yet")
    assert not (ROOT / "experiments" / "features" / f"{code}.parquet").exists()
    pred = ROOT / "experiments" / "predictions" / code
    for name in (
        "oof_val_primary.csv",
        "oof_val_shadow.csv",
        "oof_test_primary.csv",
        "oof_test_shadow.csv",
        "test_primary_mean.csv",
        "test_primary_median.csv",
        "test_shadow_mean.csv",
        "test_shadow_median.csv",
    ):
        assert (pred / name).exists(), name
    hist = pd.read_csv(ROOT / "results" / "EXP-T074_TRAINING_HISTORY.csv")
    required = {
        "experiment_code",
        "scheme",
        "fold",
        "seed",
        "initial_lr",
        "eta_min",
        "epoch",
        "learning_rate",
        "val_mae",
        "early_stop_allowed",
        "numerical_failure",
    }
    assert required.issubset(hist.columns)
    assert hist["initial_lr"].nunique() == 4
    assert hist.groupby(["scheme", "fold", "initial_lr"]).ngroups == 40
    # no early stop before epoch 100 unless numerical failure
    early = hist[hist["stopped_early"].astype(bool) & ~hist["numerical_failure"].astype(bool)]
    if len(early):
        assert early["epoch"].min() >= 100
    sel = pd.read_csv(ROOT / "results" / "EXP-T074_SELECTED_LR.csv")
    assert len(sel) == 10
    assert sel["seed"].nunique() == 1
    assert int(sel["seed"].iloc[0]) == 101
    assert "lr_at_best_epoch" in sel.columns
    assert "selected_initial_lr" in sel.columns
    for p in (
        ROOT / "results" / "EXP-T074_LR_TRAJECTORY_DIAGNOSTICS.md",
        ROOT / "results" / "EXP-T074_PROTOCOL_REPORT.md",
        ROOT / "results" / "EXP-T074_OOF_EVALUATION.yaml",
    ):
        assert p.exists()
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    t073 = exp[exp["experiment_code"] == "EXP-T073"].iloc[0]
    assert abs(float(t073["cv_primary_mae"]) - 3.3619360747160734) < 1e-12
    t030 = exp[exp["experiment_code"] == "EXP-T030"].iloc[0]
    assert abs(float(t030["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
