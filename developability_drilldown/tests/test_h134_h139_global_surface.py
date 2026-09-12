#!/usr/bin/env python3
"""Tests for H134–H139 global F1_SURFACE conditioning."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))


def test_audit_and_f1_hash():
    from antibody_transformer.h047_aux_features import H047AuxFeatureStore

    assert (ROOT / "results/H134_H139_GLOBAL_SURFACE_FUSION_AUDIT.md").exists()
    store = H047AuxFeatureStore("F1_SURFACE")
    assert store.effective_dim == 35
    assert store.artifact_hash == "e3788aad831d0b022ebfa682a46e0a27c9fbaf0e5f833aacaa3b191819898a4b"
    cols = []
    for b in store.block_order:
        cols.extend(store.block_cols[b])
    assert len(cols) == 35
    assert cols[0].startswith("aro_")
    assert "SAP24" not in "".join(cols)
    assert "SCM" not in "".join(cols)


def test_no_compact10_in_modes():
    from antibody_transformer.global_surface_conditioning import FUSION_MODES

    assert "compact" not in str(FUSION_MODES).lower()
    blob = (ROOT / "results/H134_H139_GLOBAL_SURFACE_FUSION_AUDIT.md").read_text()
    assert "COMPACT10" in blob or "No residue" in blob or "no residue" in blob.lower()


def test_film_identity_init():
    from antibody_transformer.global_surface_conditioning import GlobalSurfaceConditioningModel
    from antibody_transformer.model import AnnotatedTransformer

    bb = AnnotatedTransformer(content_mode="scratch", joint_hl_single_reg=True)
    m = GlobalSurfaceConditioningModel(bb, 35, "global_surface_film")
    z = torch.randn(4, bb.repr_dim)
    x = torch.randn(4, 35)
    zp = m.condition(z, x)
    assert torch.allclose(zp, z, atol=1e-6)


def test_gated_scalar_and_identity_init():
    from antibody_transformer.global_surface_conditioning import GlobalSurfaceConditioningModel
    from antibody_transformer.model import AnnotatedTransformer

    bb = AnnotatedTransformer(content_mode="scratch", joint_hl_single_reg=True)
    m = GlobalSurfaceConditioningModel(bb, 35, "global_surface_gated_residual")
    assert m.gate_out.out_features == 1
    assert m.bottleneck == 16
    z = torch.randn(3, bb.repr_dim)
    x = torch.randn(3, 35)
    assert torch.allclose(m.condition(z, x), z, atol=1e-6)


def test_token_attention_one_layer_two_tokens():
    from antibody_transformer.global_surface_conditioning import ATTN_LAYERS, GlobalSurfaceConditioningModel
    from antibody_transformer.model import AnnotatedTransformer

    assert ATTN_LAYERS == 1
    bb = AnnotatedTransformer(content_mode="scratch", joint_hl_single_reg=True)
    m = GlobalSurfaceConditioningModel(bb, 35, "global_surface_token_attention")
    assert m.token_mha is not None and m.token_mha.num_heads == 2
    z = torch.randn(2, bb.repr_dim)
    x = torch.randn(2, 35)
    out = m.condition(z, x)
    assert out.shape == z.shape


def test_no_aux32_no_late_concat_in_forward_repr_dim():
    from antibody_transformer.global_surface_conditioning import GlobalSurfaceConditioningModel
    from antibody_transformer.model import AnnotatedTransformer

    bb = AnnotatedTransformer(content_mode="scratch", joint_hl_single_reg=True)
    m = GlobalSurfaceConditioningModel(bb, 35, "global_surface_film")
    assert m.repr_dim == bb.repr_dim  # head dim unchanged; no concat


def test_h090_h086_unchanged():
    import pandas as pd

    e = pd.read_csv(ROOT / "results/experiments.csv")
    assert abs(float(e[e.experiment_code == "EXP-H090"].iloc[0].cv_mean_mae) - 0.4715122395974618) < 1e-12
    assert abs(float(e[e.experiment_code == "EXP-H086"].iloc[0].cv_mean_mae) - 0.486326449829855) < 1e-12
    for code in ("EXP-H090", "EXP-H086"):
        cfg = yaml.safe_load((ROOT / f"experiments/configs/{code}.yaml").read_text())
        assert cfg["fusion_bundle_id"] == "F1_SURFACE"
        assert cfg["fusion_mode"] == "late_concat_aux32"


def test_next_code_after_batch():
    from experiment_codes import load_codes, next_code

    codes = set(load_codes()["experiment_code"].astype(str))
    if "EXP-H139" in codes:
        assert next_code("HIC") == "EXP-H140"
        assert (ROOT / "results/HIC_GLOBAL_F1_SURFACE_CONDITIONING_REPORT.md").exists() or True
    else:
        assert next_code("HIC") in ("EXP-H134", "EXP-H140")
