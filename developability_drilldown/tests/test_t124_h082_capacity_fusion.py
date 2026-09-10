#!/usr/bin/env python3
"""Unit tests for T124 capacity + H047 late fusion (no GPU training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))

from antibody_transformer.h047_aux_features import H047AuxFeatureStore, f4_subblock_slices
from antibody_transformer.late_fusion import LateFusionModel
from antibody_transformer.model import AnnotatedTransformer
from antibody_transformer.protocol_v3_ext import build_platform_model_ext, normalize_capacity


def test_n_layers_3_allowed_for_arch3_and_arch7():
    kw = dict(
        content_mode="scratch",
        plm_hidden=0,
        annotation_mode="full",
        merge_mode="mean",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=128,
        n_heads=4,
        n_layers=3,
        dim_feedforward=256,
    )
    m3 = AnnotatedTransformer(**kw, joint_hl_dual_reg=True)
    assert m3.n_layers == 3
    m7 = AnnotatedTransformer(**kw, use_reg_only_cross_attention=True)
    assert m7.n_layers == 3
    assert m7.cross_attn is not None  # single module


def test_mid_bridge_rejects_n_layers_3():
    with pytest.raises(ValueError, match="n_layers=2"):
        AnnotatedTransformer(
            content_mode="scratch",
            plm_hidden=0,
            annotation_mode="full",
            merge_mode="concat",
            chain_mode="HL",
            pooling_mode="reg",
            n_layers=3,
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
        )


def test_wide_per_head_dim_32():
    cap = normalize_capacity(
        {"d_model": 256, "n_layers": 2, "n_heads": 8, "dim_feedforward": 512}
    )
    assert cap["d_model"] // cap["n_heads"] == 32
    assert cap["dim_feedforward"] == 2 * cap["d_model"]


def test_h047_bundles_dims_no_esm2_h():
    for bid, dim in (
        ("F1_SURFACE", 35),
        ("F2_SEQUENCE_TITRATION", 133),
        ("F3_LOCAL_RASA_CDR3", 32),
        ("F4_H047_AUX_ALL", 200),
    ):
        s = H047AuxFeatureStore(bid)
        assert s.effective_dim == dim
        assert "ESM2_H" not in s.block_order
        prep = s.fit(s.ids[:40])
        X = s.transform(prep, s.ids[:5])
        assert X.shape == (5, prep.effective_dim)
        assert np.isfinite(X).all()


def test_f4_slices_cover_200():
    sl = f4_subblock_slices()
    assert sl["SURFACE"] == slice(0, 35)
    assert sl["SEQUENCE_TITRATION"] == slice(35, 168)
    assert sl["LOCAL_RASA"] == slice(168, 200)


def test_late_fusion_forward_shape():
    backbone = AnnotatedTransformer(
        content_mode="scratch",
        plm_hidden=0,
        annotation_mode="full",
        merge_mode="h_only",
        chain_mode="H_ONLY",
        pooling_mode="reg",
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
    )
    m = LateFusionModel(backbone, aux_dim=35)
    B, L = 2, 10
    batch = {
        "heavy_aa": torch.randint(0, 20, (B, L)),
        "light_aa": torch.randint(0, 20, (B, L)),
        "heavy_mask": torch.ones(B, L, dtype=torch.bool),
        "light_mask": torch.ones(B, L, dtype=torch.bool),
        "heavy_pos": torch.arange(L).unsqueeze(0).expand(B, -1),
        "light_pos": torch.arange(L).unsqueeze(0).expand(B, -1),
        "heavy_imgt": torch.zeros(B, L, dtype=torch.long),
        "light_imgt": torch.zeros(B, L, dtype=torch.long),
        "heavy_region": torch.zeros(B, L, dtype=torch.long),
        "light_region": torch.zeros(B, L, dtype=torch.long),
    }
    fixed = torch.randn(B, 35)
    out = m(batch, fixed)
    assert out.shape == (B,)
