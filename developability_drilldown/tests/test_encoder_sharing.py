#!/usr/bin/env python3
"""Unit tests for H/L encoder unsharing (EXP-T130–T141)."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from antibody_transformer.model import AnnotatedTransformer


def _kw(**extra):
    base = dict(
        content_mode="scratch",
        plm_hidden=0,
        n_aa=22,
        max_seq_pos=40,
        n_imgt=40,
        n_region=9,
        annotation_mode="full",
        merge_mode="mean",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=32,
        n_heads=4,
        n_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        norm_first=True,
    )
    base.update(extra)
    return base


def _toy_batch(B=2, Lh=5, Ll=4, d_aa=22):
    return {
        "heavy_aa": torch.randint(1, d_aa, (B, Lh)),
        "light_aa": torch.randint(1, d_aa, (B, Ll)),
        "heavy_mask": torch.ones(B, Lh, dtype=torch.bool),
        "light_mask": torch.ones(B, Ll, dtype=torch.bool),
        "heavy_pos": torch.arange(1, Lh + 1).view(1, -1).expand(B, -1),
        "light_pos": torch.arange(1, Ll + 1).view(1, -1).expand(B, -1),
        "heavy_imgt": torch.arange(1, Lh + 1).view(1, -1).expand(B, -1),
        "light_imgt": torch.arange(1, Ll + 1).view(1, -1).expand(B, -1),
        "heavy_region": torch.ones(B, Lh, dtype=torch.long),
        "light_region": torch.ones(B, Ll, dtype=torch.long),
    }


def _align_unshared_to_shared(shared: AnnotatedTransformer, unshared: AnnotatedTransformer) -> None:
    """Copy all matched weights so unshared starts as a functional duplicate of shared."""
    assert shared.share_hl_encoder and not unshared.share_hl_encoder
    sd = shared.state_dict()
    usd = unshared.state_dict()
    new = {}
    for k, v in usd.items():
        if k.startswith("encoder_h."):
            src = "encoder." + k[len("encoder_h.") :]
            new[k] = sd[src].clone()
        elif k.startswith("encoder_l."):
            src = "encoder." + k[len("encoder_l.") :]
            new[k] = sd[src].clone()
        elif k in sd:
            new[k] = sd[k].clone()
        else:
            new[k] = v
    unshared.load_state_dict(new)


def test_unshared_init_hashes_identical():
    m = AnnotatedTransformer(**_kw(share_hl_encoder=False))
    h = m.encoder_init_hashes()
    assert h["encoder_h"] == h["encoder_l"]
    # Distinct Parameter objects
    assert m.encoder_h is not m.encoder_l
    p_h = list(m.encoder_h.parameters())[0]
    p_l = list(m.encoder_l.parameters())[0]
    assert p_h.data_ptr() != p_l.data_ptr()


def test_shared_unshared_initial_forward_equivalence_arch1():
    torch.manual_seed(0)
    shared = AnnotatedTransformer(**_kw(share_hl_encoder=True))
    unshared = AnnotatedTransformer(**_kw(share_hl_encoder=False))
    _align_unshared_to_shared(shared, unshared)
    batch = _toy_batch()
    shared.eval()
    unshared.eval()
    with torch.no_grad():
        y_s = shared(batch)
        y_u = unshared(batch)
    assert float((y_s - y_u).abs().max()) <= 1e-6


def test_shared_unshared_initial_forward_equivalence_arch7():
    torch.manual_seed(1)
    shared = AnnotatedTransformer(
        **_kw(share_hl_encoder=True, use_reg_only_cross_attention=True, merge_mode="concat")
    )
    unshared = AnnotatedTransformer(
        **_kw(share_hl_encoder=False, use_reg_only_cross_attention=True, merge_mode="concat")
    )
    _align_unshared_to_shared(shared, unshared)
    batch = _toy_batch()
    shared.eval()
    unshared.eval()
    with torch.no_grad():
        y_s = shared(batch)
        y_u = unshared(batch)
    assert float((y_s - y_u).abs().max()) <= 1e-6


def test_unshared_encoders_can_diverge_after_asymmetric_update():
    m = AnnotatedTransformer(**_kw(share_hl_encoder=False, dropout=0.0))
    assert m.encoder_init_hashes()["encoder_h"] == m.encoder_init_hashes()["encoder_l"]
    # Asymmetric grad on Heavy encoder only
    p = next(m.encoder_h.parameters())
    p.grad = torch.ones_like(p)
    with torch.no_grad():
        p.add_(-0.1, p.grad)
    h = m.encoder_init_hashes()
    assert h["encoder_h"] != h["encoder_l"]


def test_unshared_param_count_doubles_encoder():
    s = AnnotatedTransformer(**_kw(share_hl_encoder=True))
    u = AnnotatedTransformer(**_kw(share_hl_encoder=False))
    ps = s.param_account()
    pu = u.param_account()
    assert pu["encoder"] == 2 * ps["encoder"]
    assert pu["encoder_h"] == ps["encoder"]
    assert pu["encoder_l"] == ps["encoder"]
    assert pu["other"] == ps["other"]
    assert pu["head"] == ps["head"]


def test_unshared_rejects_joint_arch():
    try:
        AnnotatedTransformer(**_kw(share_hl_encoder=False, joint_hl_dual_reg=True))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "share_hl_encoder=False" in str(e)


def test_cross_attn_still_shared_under_unshared_encoder():
    m = AnnotatedTransformer(**_kw(share_hl_encoder=False, use_reg_only_cross_attention=True))
    assert m.cross_attn is not None
    # single module; both directions use it
    assert isinstance(m.cross_attn, nn.MultiheadAttention)
