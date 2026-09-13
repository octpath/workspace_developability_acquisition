#!/usr/bin/env python3
"""Tests for T142–T150 C/D H/L interaction architectures."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))


def _batch(B=2, Lh=5, Ll=4):
    return dict(
        heavy_aa=torch.randint(0, 20, (B, Lh)),
        light_aa=torch.randint(0, 20, (B, Ll)),
        heavy_mask=torch.ones(B, Lh, dtype=torch.bool),
        light_mask=torch.ones(B, Ll, dtype=torch.bool),
        heavy_pos=torch.arange(Lh).unsqueeze(0).expand(B, -1),
        light_pos=torch.arange(Ll).unsqueeze(0).expand(B, -1),
        heavy_imgt=torch.zeros(B, Lh, dtype=torch.long),
        light_imgt=torch.zeros(B, Ll, dtype=torch.long),
        heavy_region=torch.zeros(B, Lh, dtype=torch.long),
        light_region=torch.zeros(B, Ll, dtype=torch.long),
    )


def test_controls_unchanged():
    import pandas as pd

    e = pd.read_csv(ROOT / "results/experiments.csv")
    for code, mean in (
        ("EXP-T110", 3.2416051228841143),
        ("EXP-T113", 3.139063058076081),
        ("EXP-T121", 3.137653197771237),
    ):
        row = e[e.experiment_code == code].iloc[0]
        assert abs(float(row.cv_mean_mae) - mean) < 1e-6


def test_c0_shared_cross_attn_params():
    from antibody_transformer.model import AnnotatedTransformer

    m = AnnotatedTransformer(content_mode="scratch", use_reg_only_cross_attention=True, merge_mode="mean")
    assert m.cross_attn is not None
    assert m.reg_cross_variant is None


def test_c_variants_construct_and_share_modules():
    from antibody_transformer.model import AnnotatedTransformer

    batch = _batch()
    for v in ("c1_scalar_gate", "c2_feature_gate", "c3_ffn_adapter", "c4_two_read", "c5_two_query"):
        m = AnnotatedTransformer(
            content_mode="scratch",
            use_reg_only_cross_attention=True,
            merge_mode="mean",
            reg_cross_variant=v,
        )
        y = m(batch)
        assert y.shape == (2,)


def test_c1_zero_init_gate_is_one():
    from antibody_transformer.hl_interaction import SharedScalarCrossGate

    g = SharedScalarCrossGate(32)
    reg = torch.randn(4, 1, 32)
    delta = torch.randn(4, 1, 32)
    out, gate = g(reg, delta)
    assert torch.allclose(gate, torch.ones_like(gate), atol=1e-5)
    assert torch.allclose(out, reg + delta, atol=1e-5)


def test_d3_d4_swap_invariance():
    from antibody_transformer.model import AnnotatedTransformer

    zh = torch.randn(3, 128)
    zl = torch.randn(3, 128)
    for mode in ("d3_symmetric_mlp", "d4_token_attention"):
        m = AnnotatedTransformer(content_mode="scratch", merge_mode="mean", pair_interaction_mode=mode)
        a = m.pair_module(zh, zl)
        b = m.pair_module(zl, zh)
        assert torch.allclose(a, b, atol=1e-5), mode


def test_d1_zero_pair_matches_mean_head():
    from antibody_transformer.model import AnnotatedTransformer

    m = AnnotatedTransformer(content_mode="scratch", merge_mode="mean", pair_interaction_mode="d1_bilinear_score")
    m.eval()
    batch = _batch()
    with torch.no_grad():
        zh, zl = m.encode_separate_regs(batch)
        m_repr = 0.5 * (zh + zl)
        y_head = m.head(m_repr).squeeze(-1)
        y = m(batch)
        assert torch.allclose(y, y_head, atol=1e-5)  # w_pair=0


def test_no_fixed_hl_coefficients_in_c_modules():
    from antibody_transformer.hl_interaction import SharedFeatureCrossGate, SharedScalarCrossGate

    assert not hasattr(SharedScalarCrossGate(16), "w_h")
    assert not hasattr(SharedFeatureCrossGate(16), "w_l")


def test_next_after_batch():
    from experiment_codes import load_codes, next_code

    codes = set(load_codes()["experiment_code"].astype(str))
    if "EXP-T337" in codes:
        assert next_code("TmApp") == "EXP-T338"
    elif "EXP-T160" in codes:
        assert next_code("TmApp") == "EXP-T161"
    elif "EXP-T156" in codes:
        assert next_code("TmApp") == "EXP-T157"
    elif "EXP-T150" in codes:
        assert next_code("TmApp") == "EXP-T151"
    else:
        assert next_code("TmApp") == "EXP-T142"


def test_prereg_exists():
    p = ROOT / "results/T142_T150_CD_ARCHITECTURE_PREREGISTRATION.yaml"
    assert p.exists()
    doc = yaml.safe_load(p.read_text())
    assert len(doc["experiments"]) == 9
