#!/usr/bin/env python3
"""Architecture unit tests for EXP-T080..T104 extensions (no GPU training)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.equivalence import synthetic_batch  # noqa: E402
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.protocol_v3 import (  # noqa: E402
    PLATFORM_ID,
    candidate_config_hash,
    normalize_arch_flags,
)


def _kw(**extra):
    kw = dict(
        content_mode="frozen",
        plm_hidden=64,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=32,
        n_heads=4,
        n_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    kw.update(extra)
    return kw


def _batch(**kw):
    return synthetic_batch(
        content_mode=kw.get("content_mode", "frozen"),
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=kw.get("plm_hidden", 64),
        Lh=6,
        Ll=5,
        B=2,
    )


def test_platform_id_still_v3():
    assert PLATFORM_ID == "DL_FOLDLOCAL_COSINE_V3"
    freeze = ROOT / "results" / "DL_FOLDLOCAL_COSINE_V3_FREEZE.yaml"
    assert freeze.exists()
    assert "FROZEN" in freeze.read_text()


def test_mean_merge_separate_dual_repr_dim():
    m = AnnotatedTransformer(**_kw(merge_mode="mean", d_model=32))
    assert m.repr_dim == 32
    batch = _batch()
    m.eval()
    with torch.no_grad():
        z = m.forward_repr(batch)
        h_h = m.encode_chain(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            mask=batch["heavy_mask"],
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
        )
        h_l = m.encode_chain(
            chain_idx=1,
            aa=batch.get("light_aa"),
            plm=batch.get("light_plm"),
            mask=batch["light_mask"],
            pos=batch["light_pos"],
            imgt=batch.get("light_imgt"),
            region=batch.get("light_region"),
        )
        expected = 0.5 * (h_h + h_l)
    assert z.shape[-1] == 32
    assert torch.allclose(z, expected, atol=1e-6)


def test_joint_dual_mean_repr_dim():
    m = AnnotatedTransformer(**_kw(joint_hl_dual_reg=True, merge_mode="mean", d_model=32))
    assert m.repr_dim == 32
    batch = _batch()
    m.eval()
    with torch.no_grad():
        z = m.forward_repr(batch)
    assert z.shape[-1] == 32


def test_fixed_one_ungated_matches_learned_gate_one():
    torch.manual_seed(0)
    learned = AnnotatedTransformer(
        **_kw(use_cross_attention_bridge=True, cross_gate_mode="learned")
    )
    fixed = AnnotatedTransformer(
        **_kw(use_cross_attention_bridge=True, cross_gate_mode="fixed_one")
    )
    assert learned.cross_gate_h is not None and learned.cross_gate_h.requires_grad
    assert fixed.cross_gate_h is None and fixed.cross_gate_l is None
    assert fixed.param_account()["gates"] == 0
    assert learned.param_account()["gates"] == 2

    sd_l = learned.state_dict()
    sd_f = fixed.state_dict()
    for k, v in sd_l.items():
        if k in sd_f and sd_f[k].shape == v.shape:
            sd_f[k] = v.clone()
    fixed.load_state_dict(sd_f)
    with torch.no_grad():
        learned.cross_gate_h.fill_(1.0)
        learned.cross_gate_l.fill_(1.0)

    batch = _batch()
    learned.eval()
    fixed.eval()
    with torch.no_grad():
        z0 = learned.forward_repr(batch)
        z1 = fixed.forward_repr(batch)
    assert float((z0 - z1).abs().max()) < 1e-5


def test_reg_only_cross_attention():
    m = AnnotatedTransformer(**_kw(use_reg_only_cross_attention=True))
    assert m.cross_attn is not None
    assert m.cross_gate_h is None
    assert m.repr_dim == 64  # concat default
    assert m.param_account()["gates"] == 0
    batch = _batch()
    m.eval()
    with torch.no_grad():
        z = m.forward_repr(batch)
    assert z.shape == (2, 64)


def test_within_chain_extra_attention():
    with pytest.raises(ValueError):
        AnnotatedTransformer(
            **_kw(
                use_within_chain_extra_attention=True,
                use_cross_attention_bridge=True,
            )
        )
    m = AnnotatedTransformer(**_kw(use_within_chain_extra_attention=True))
    assert m.cross_attn is not None
    assert m.cross_gate_h is None
    ungated = AnnotatedTransformer(
        **_kw(use_cross_attention_bridge=True, cross_gate_mode="fixed_one")
    )
    assert m.param_account()["cross_attention"] == ungated.param_account()["cross_attention"]
    batch = _batch()
    m.eval()
    with torch.no_grad():
        z = m.forward_repr(batch)
    assert z.shape[-1] == 64


def test_arch6_vs_arch8_cross_attn_params_equal():
    """ARCH-6 (fixed_one bridge) vs ARCH-8 (within-chain): shared MHA size equal."""
    arch6 = AnnotatedTransformer(
        **_kw(use_cross_attention_bridge=True, cross_gate_mode="fixed_one")
    )
    arch8 = AnnotatedTransformer(**_kw(use_within_chain_extra_attention=True))
    assert arch6.param_account()["cross_attention"] == arch8.param_account()["cross_attention"]
    assert arch6.param_account()["gates"] == 0
    assert arch8.param_account()["gates"] == 0


def test_scratch_vs_frozen_content():
    scratch = AnnotatedTransformer(
        content_mode="scratch",
        plm_hidden=0,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=32,
        n_heads=4,
        n_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    frozen = AnnotatedTransformer(**_kw())
    assert scratch.aa_emb is not None
    assert scratch.plm_proj is None
    assert frozen.aa_emb is None
    assert frozen.plm_proj is not None


def test_normalize_arch_flags_mutex_and_gate_mode():
    with pytest.raises(ValueError):
        normalize_arch_flags(
            {"use_cross_attention_bridge": True, "use_reg_only_cross_attention": True}
        )
    with pytest.raises(ValueError):
        normalize_arch_flags({"cross_gate_mode": "bogus"})
    flags = normalize_arch_flags({"use_cross_attention_bridge": True, "cross_gate_mode": "fixed_one"})
    assert flags["use_cross_attention_bridge"] is True
    assert flags["cross_gate_mode"] == "fixed_one"
    assert flags["use_reg_only_cross_attention"] is False
    h = candidate_config_hash(flags, "frozen", "concat", "ablingua")
    assert isinstance(h, str) and len(h) == 64
