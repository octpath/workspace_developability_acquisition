#!/usr/bin/env python3
"""Unit tests for ARCH-6G H–L Cα RBF geometry bias on residue cross-attention.

HIC experiment codes reserved for this batch: EXP-H054 .. EXP-H081 (28 codes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.cross_geometry import (  # noqa: E402
    DEFAULT_RBF_CENTERS,
    DEFAULT_RBF_SIGMA,
    build_hl_distance,
    rbf_bias,
)
from antibody_transformer.equivalence import synthetic_batch  # noqa: E402
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.protocol_v3 import normalize_arch_flags  # noqa: E402


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


def _batch_with_ca(Lh=6, Ll=5, B=2, plm_hidden=64):
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=plm_hidden,
        Lh=Lh,
        Ll=Ll,
        B=B,
    )
    # Non-degenerate CA: place H and L on offset lattices
    torch.manual_seed(7)
    batch["heavy_ca"] = torch.randn(B, Lh, 3) * 2.0
    batch["light_ca"] = torch.randn(B, Ll, 3) * 2.0 + torch.tensor([8.0, 0.0, 0.0])
    return batch


def test_rbf_centers_and_sigma_exact():
    assert DEFAULT_RBF_CENTERS == [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
    assert DEFAULT_RBF_SIGMA == 2.5
    m = AnnotatedTransformer(
        **_kw(
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
            use_cross_geometry_bias=True,
        )
    )
    assert m.cross_geometry_rbf_centers == DEFAULT_RBF_CENTERS
    assert m.cross_geometry_rbf_sigma == 2.5
    assert list(m.cross_geom_centers.tolist()) == DEFAULT_RBF_CENTERS


def test_zero_init_equivalence_arch6_vs_arch6g():
    torch.manual_seed(0)
    arch6 = AnnotatedTransformer(
        **_kw(use_cross_attention_bridge=True, cross_gate_mode="fixed_one")
    )
    arch6g = AnnotatedTransformer(
        **_kw(
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
            use_cross_geometry_bias=True,
        )
    )
    sd6 = arch6.state_dict()
    sd6g = arch6g.state_dict()
    for k, v in sd6.items():
        if k in sd6g and sd6g[k].shape == v.shape:
            sd6g[k] = v.clone()
    arch6g.load_state_dict(sd6g)
    assert torch.all(arch6g.cross_geom_weight == 0)

    batch = _batch_with_ca()
    arch6.eval()
    arch6g.eval()
    with torch.no_grad():
        y0 = arch6(batch)
        y1 = arch6g(batch)
        z0 = arch6.forward_repr(batch)
        z1 = arch6g.forward_repr(batch)
    assert float((y0 - y1).abs().max()) <= 1e-6
    assert float((z0 - z1).abs().max()) <= 1e-6


def test_geometry_weights_receive_nonzero_grad():
    m = AnnotatedTransformer(
        **_kw(
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
            use_cross_geometry_bias=True,
        )
    )
    batch = _batch_with_ca()
    m.train()
    y = m(batch)
    loss = y.pow(2).mean()
    loss.backward()
    assert m.cross_geom_weight.grad is not None
    g = m.cross_geom_weight.grad
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 0.0


def test_missing_ca_pair_bias_zero():
    B, Lh, Ll, H = 2, 4, 3, 4
    ca_h = torch.randn(B, Lh, 3)
    ca_l = torch.randn(B, Ll, 3)
    ca_h[:, 1, :] = float("nan")
    ca_l[:, 0, :] = float("nan")
    mh = torch.ones(B, Lh, dtype=torch.bool)
    ml = torch.ones(B, Ll, dtype=torch.bool)
    mh[:, -1] = False
    d = build_hl_distance(ca_h, ca_l, mh, ml)
    assert torch.isnan(d[:, 1, :]).all()
    assert torch.isnan(d[:, :, 0]).all()
    assert torch.isnan(d[:, -1, :]).all()
    w = torch.ones(H, 8)
    bias = rbf_bias(d, w, centers=DEFAULT_RBF_CENTERS, sigma=DEFAULT_RBF_SIGMA)
    assert torch.allclose(bias[:, :, 1, :], torch.zeros_like(bias[:, :, 1, :]))
    assert torch.allclose(bias[:, :, :, 0], torch.zeros_like(bias[:, :, :, 0]))
    assert torch.allclose(bias[:, :, -1, :], torch.zeros_like(bias[:, :, -1, :]))
    # Valid pair has non-zero bias when weight=1
    assert float(bias[0, 0, 0, 1].abs()) > 0.0


def test_reg_excluded_from_cross_qkv_with_geometry():
    m = AnnotatedTransformer(
        **_kw(
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
            use_cross_geometry_bias=True,
        )
    )
    batch = _batch_with_ca(Lh=5, Ll=4, B=2)
    captured = {}

    def wrap(q, k, v, key_padding_mask=None, need_weights=False, attn_mask=None):
        captured.setdefault("q_lens", []).append(q.shape[1])
        captured.setdefault("k_lens", []).append(k.shape[1])
        if attn_mask is not None:
            captured.setdefault("attn_shapes", []).append(tuple(attn_mask.shape))
        return torch.zeros_like(q), None

    m.cross_attn.forward = wrap  # type: ignore
    m.eval()
    with torch.no_grad():
        m.forward_repr(batch)
    assert captured["q_lens"] == [5, 4]
    assert captured["k_lens"] == [4, 5]
    # per-head expanded: B*n_heads=8
    assert captured["attn_shapes"] == [(8, 5, 4), (8, 4, 5)]


def test_hl_lh_use_transposed_distances():
    m = AnnotatedTransformer(
        **_kw(
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
            use_cross_geometry_bias=True,
        )
    )
    with torch.no_grad():
        m.cross_geom_weight.fill_(0.5)
    batch = _batch_with_ca(Lh=5, Ll=4, B=1)
    captured = {}

    def wrap(q, k, v, key_padding_mask=None, need_weights=False, attn_mask=None):
        captured.setdefault("masks", []).append(
            None if attn_mask is None else attn_mask.detach().clone()
        )
        return torch.zeros_like(q), None

    m.cross_attn.forward = wrap  # type: ignore
    m.eval()
    with torch.no_grad():
        m.forward_repr(batch)

    mask_hl, mask_lh = captured["masks"]
    # Reconstruct expected from distances
    d = build_hl_distance(
        batch["heavy_ca"], batch["light_ca"], batch["heavy_mask"], batch["light_mask"]
    )
    bias = rbf_bias(
        d,
        m.cross_geom_weight,
        centers=m.cross_geom_centers,
        sigma=m.cross_geometry_rbf_sigma,
    )
    exp_hl = bias.reshape(1 * 4, 5, 4)
    exp_lh = bias.transpose(-2, -1).reshape(1 * 4, 4, 5)
    assert torch.allclose(mask_hl, exp_hl, atol=1e-6)
    assert torch.allclose(mask_lh, exp_lh, atol=1e-6)
    assert torch.allclose(mask_lh, mask_hl.transpose(-2, -1), atol=1e-6)


def test_param_count_plus_32_for_4x8():
    arch6 = AnnotatedTransformer(
        **_kw(use_cross_attention_bridge=True, cross_gate_mode="fixed_one")
    )
    arch6g = AnnotatedTransformer(
        **_kw(
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
            use_cross_geometry_bias=True,
        )
    )
    delta = arch6g.n_trainable_parameters() - arch6.n_trainable_parameters()
    assert delta == 4 * 8
    assert arch6g.param_account()["geometry"] == 32
    assert arch6.param_account().get("geometry", 0) == 0


def test_geometry_requires_bridge_and_mutex_with_ca_encoder():
    with pytest.raises(ValueError):
        AnnotatedTransformer(**_kw(use_cross_geometry_bias=True))
    with pytest.raises(ValueError):
        AnnotatedTransformer(
            **_kw(
                use_cross_attention_bridge=True,
                use_cross_geometry_bias=True,
                use_ca_distance_bias=True,
            )
        )
    with pytest.raises(ValueError):
        normalize_arch_flags({"use_cross_geometry_bias": True})
    flags = normalize_arch_flags(
        {
            "use_cross_attention_bridge": True,
            "cross_gate_mode": "fixed_one",
            "use_cross_geometry_bias": True,
        }
    )
    assert flags["use_cross_geometry_bias"] is True
    assert flags["use_cross_attention_bridge"] is True


def test_h_only_encode_without_geometry():
    """ARCH-H0: Heavy-only path works; geometry not applicable."""
    m = AnnotatedTransformer(**_kw(chain_mode="H_ONLY", merge_mode="h_only"))
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="H_ONLY",
        annotation_mode="full",
        plm_hidden=64,
        Lh=6,
        Ll=5,
        B=2,
    )
    m.eval()
    with torch.no_grad():
        z = m.forward_repr(batch)
    assert z.shape == (2, 32)
    assert m.cross_geom_weight is None
