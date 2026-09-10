#!/usr/bin/env python3
"""Tests for DL_FOLDLOCAL_COSINE_V3 / EXP-T075."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.protocol_v3 import (  # noqa: E402
    BATCH_SIZE,
    COARSE_LR_GRID,
    DEFAULT_SEED,
    ETA_MIN_FRAC,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    PLATFORM_ID,
    SMOOTH_L1_BETA,
    T_MAX,
    WEIGHT_DECAY,
    coarse_lr_grid,
    eta_min_for,
    lr_at_epoch,
    lr_at_t,
)
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402


def test_platform_constants():
    assert PLATFORM_ID == "DL_FOLDLOCAL_COSINE_V3"
    assert DEFAULT_SEED == 101
    assert MAX_EPOCHS == 200
    assert MIN_EPOCHS == 0
    assert PATIENCE == 30
    assert T_MAX == 200
    assert ETA_MIN_FRAC == 0.01
    assert WEIGHT_DECAY == 0.01
    assert BATCH_SIZE == 16
    assert SMOOTH_L1_BETA == 0.5
    assert coarse_lr_grid() == pytest.approx([1e-5, 1e-4, 1e-3, 1e-2])
    assert COARSE_LR_GRID == [1e-5, 1e-4, 1e-3, 1e-2]


def test_cosine_trajectory_known_points():
    lr0 = 1e-2
    eta = eta_min_for(lr0)
    assert abs(eta - 1e-4) < 1e-15
    # t = 0, 50, 100, 150, 200
    assert abs(lr_at_t(0, lr0, eta) - lr0) < 1e-12
    assert abs(lr_at_t(200, lr0, eta) - eta) < 1e-12
    mid = lr_at_t(100, lr0, eta)
    expected_mid = eta + 0.5 * (lr0 - eta) * (1 + math.cos(math.pi * 100 / 200))
    assert abs(mid - expected_mid) < 1e-12
    assert abs(mid - (eta + 0.5 * (lr0 - eta))) < 1e-12  # cos(pi/2)=0
    t50 = lr_at_t(50, lr0, eta)
    t150 = lr_at_t(150, lr0, eta)
    assert eta < t150 < mid < t50 < lr0
    # epoch mapping: epoch 1 => t=0
    assert abs(lr_at_epoch(1, lr0, eta) - lr0) < 1e-12
    assert abs(lr_at_epoch(201, lr0, eta) - eta) < 1e-12


def test_no_min_epochs_gate():
    import inspect
    import antibody_transformer.protocol_v3 as m

    src = inspect.getsource(m.train_one_candidate_v3)
    assert "early_ok" not in src
    assert "epochs_since >= patience" in src
    assert "warmup" not in src.lower()
    assert "torch.optim.lr_scheduler.CosineAnnealingLR" not in src
    assert "full_dev_transformer_predict" not in inspect.getsource(m.run_protocol_v3)
    assert m.MIN_EPOCHS == 0
    assert m.PATIENCE == 30


def test_t030_architecture():
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
    )
    assert not m.joint_hl_single_reg
    assert not m.use_cross_attention_bridge


def test_runner_guards():
    text = (ROOT / "scripts" / "run_exp_t075_final_platform.py").read_text()
    assert "no_nested_cv" in text
    assert "no_full_dev_refit" in text
    assert "DL_FOLDLOCAL_COSINE_V3" in text
    assert "patience" in text
    assert "full_dev_transformer_predict" not in text


def test_exp_t075_artifacts_if_present():
    import pandas as pd

    code = "EXP-T075"
    if not (ROOT / "experiments" / "configs" / f"{code}.yaml").exists():
        pytest.skip("EXP-T075 not registered yet")
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
    hist = pd.read_csv(ROOT / "results" / "EXP-T075_TRAINING_HISTORY.csv")
    assert {"initial_lr", "eta_min", "learning_rate", "val_mae", "numerical_failure"}.issubset(hist.columns)
    assert hist["initial_lr"].nunique() == 4
    assert hist.groupby(["scheme", "fold", "initial_lr"]).ngroups == 40
    # early stop may occur before 100 (no min_epochs)
    sel = pd.read_csv(ROOT / "results" / "EXP-T075_SELECTED_LR.csv")
    assert len(sel) == 10
    assert int(sel["seed"].iloc[0]) == 101
    for p in (
        ROOT / "results" / "EXP-T075_TRAINING_DIAGNOSTICS.md",
        ROOT / "results" / "EXP-T075_PROTOCOL_REPORT.md",
        ROOT / "results" / "EXP-T075_OOF_EVALUATION.yaml",
        ROOT / "results" / "DL_FOLDLOCAL_COSINE_V3_PROTOCOL.md",
        ROOT / "results" / "DL_FOLDLOCAL_COSINE_V3_FREEZE.yaml",
        ROOT / "results" / "T073_T074_T075_PLATFORM_COMPARISON.md",
    ):
        assert p.exists(), p
    freeze = (ROOT / "results" / "DL_FOLDLOCAL_COSINE_V3_FREEZE.yaml").read_text()
    assert "status: FROZEN" in freeze or "status: FROZEN\n" in freeze or '"status": "FROZEN"' in freeze or "FROZEN" in freeze
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert abs(float(exp[exp.experiment_code == "EXP-T074"].iloc[0]["cv_primary_mae"]) - 3.2866556144055026) < 1e-12
    assert abs(float(exp[exp.experiment_code == "EXP-T073"].iloc[0]["cv_primary_mae"]) - 3.3619360747160734) < 1e-12
    assert abs(float(exp[exp.experiment_code == "EXP-T030"].iloc[0]["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
