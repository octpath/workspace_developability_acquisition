#!/usr/bin/env python3
"""Tests for protocol V2 (EXP-T073): fold-local LR, no nested CV, no full-Dev."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.protocol_v2 import (  # noqa: E402
    DEFAULT_SEED,
    ETA_MIN,
    MAX_EPOCHS,
    PATIENCE,
    PROTOCOL_ID,
    build_t030_model,
    lr_grid,
    select_lr,
    state_dict_sha256,
)
from antibody_transformer.config import load_presets  # noqa: E402
from antibody_transformer.equivalence import synthetic_batch  # noqa: E402
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402


def test_protocol_constants():
    assert PROTOCOL_ID == "DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2"
    assert DEFAULT_SEED == 101
    assert MAX_EPOCHS == 200
    assert PATIENCE == 20
    assert ETA_MIN == 0.0
    presets = load_presets()
    assert 101 in presets["neural"]["seeds"]
    lr_ref = float(presets["neural"]["lr"])
    assert abs(lr_ref - 3e-4) < 1e-15
    grid = lr_grid(lr_ref)
    assert grid == pytest.approx([0.1 * lr_ref, 0.3 * lr_ref, 1.0 * lr_ref, 3.0 * lr_ref])


def test_select_lr_tie_prefers_smaller():
    cands = [
        {"lr": 3e-4, "best_val_mae": 1.0},
        {"lr": 9e-5, "best_val_mae": 1.0},
        {"lr": 9e-4, "best_val_mae": 1.2},
    ]
    assert select_lr(cands)["lr"] == 9e-5


def test_t030_architecture_flags():
    # synthetic rb dims via AnnotatedTransformer defaults for unit check
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
    assert not m.joint_hl_chain_specific_dual_reg
    assert not m.use_cross_attention_bridge
    assert m.repr_dim == 256


def test_equivalent_init_hash_same_seed():
    # two models with same seed should match hash after build+seed
    from antibody_transformer.data import ResidueBundle
    # Use tiny synthetic path: hash AnnotatedTransformer after seeded init
    torch.manual_seed(101)
    a = AnnotatedTransformer(content_mode="frozen", plm_hidden=32, d_model=16, n_heads=2, dim_feedforward=32, dropout=0.0)
    ha = state_dict_sha256(a)
    torch.manual_seed(101)
    b = AnnotatedTransformer(content_mode="frozen", plm_hidden=32, d_model=16, n_heads=2, dim_feedforward=32, dropout=0.0)
    hb = state_dict_sha256(b)
    assert ha == hb


def test_cosine_scheduler_exists_in_protocol_module():
    import inspect
    import antibody_transformer.protocol_v2 as p2

    src = inspect.getsource(p2.train_one_candidate)
    assert "CosineAnnealingLR" in src
    assert "eta_min" in src
    assert "full_dev" not in src.lower()
    assert "nested" not in src.lower()


def test_no_nested_cv_in_runner():
    text = (ROOT / "scripts" / "run_exp_t073_protocol_v2.py").read_text()
    assert "nested" in text.lower()  # documented as forbidden
    assert "no_nested_cv" in text
    assert "no_full_dev_refit" in text
    assert "full_dev_transformer_predict" not in text


def test_exp_t073_artifacts_if_present():
    import pandas as pd

    code = "EXP-T073"
    if not (ROOT / "experiments" / "configs" / f"{code}.yaml").exists():
        pytest.skip("EXP-T073 not registered yet")
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
    assert (ROOT / "results" / "EXP-T073_TRAINING_HISTORY.csv").exists()
    assert (ROOT / "results" / "EXP-T073_SELECTED_LR.csv").exists()
    assert (ROOT / "results" / "EXP-T073_TRAINING_DIAGNOSTICS.md").exists()
    assert (ROOT / "results" / "EXP-T073_PROTOCOL_REPORT.md").exists()
    assert (ROOT / "results" / "EXP-T073_OOF_EVALUATION.yaml").exists()
    hist = pd.read_csv(ROOT / "results" / "EXP-T073_TRAINING_HISTORY.csv")
    assert set(["scheme", "fold", "lr_candidate", "epoch", "val_mae", "learning_rate"]).issubset(hist.columns)
    # all 4 LRs present unless quick
    n_lr = hist["lr_candidate"].nunique()
    assert n_lr == 4
    assert hist.groupby(["scheme", "fold", "lr_candidate"]).ngroups == 40
    sel = pd.read_csv(ROOT / "results" / "EXP-T073_SELECTED_LR.csv")
    assert len(sel) == 10  # 5+5
    assert sel["seed"].nunique() == 1
    assert int(sel["seed"].iloc[0]) == 101
    # historical untouched
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    t030 = exp[exp["experiment_code"] == "EXP-T030"].iloc[0]
    assert abs(float(t030["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
    t072 = exp[exp["experiment_code"] == "EXP-T072"].iloc[0]
    assert abs(float(t072["cv_primary_mae"]) - 3.2668468648023574) < 1e-12
    for name in (
        "test_primary_fold0.csv",
        "test_primary_fold4.csv",
        "test_shadow_fold0.csv",
        "test_shadow_fold4.csv",
    ):
        assert (pred / name).exists(), name
    ckpt_dir = ROOT / "results" / "EXP-T073_run" / "checkpoints"
    assert len(list(ckpt_dir.glob("primary_k*_seed101.pt"))) >= 5
    assert len(list(ckpt_dir.glob("shadow_k*_seed101.pt"))) >= 5
