#!/usr/bin/env python3
"""Focused tests for joint H/L dual-REG (EXP-T070)."""
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


def _joint_dual(**kwargs):
    return AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=32,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=16,
        n_heads=2,
        n_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        joint_hl_dual_reg=True,
        joint_hl_single_reg=False,
        **kwargs,
    )


def test_dual_reg_tokens_exist_once():
    m = _joint_dual()
    assert m.reg_token is not None and m.reg_token.shape == (2, 16)
    assert m.single_reg_token is None
    assert m.repr_dim == 32  # 2 * d_model


def test_heavy_light_chain_ids_on_residues():
    m = _joint_dual()
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=32,
        Lh=8,
        Ll=6,
        B=2,
    )
    B = batch["heavy_mask"].shape[0]
    Lh = batch["heavy_mask"].shape[1]
    Ll = batch["light_mask"].shape[1]
    assert not torch.allclose(m.chain_emb.weight[0], m.chain_emb.weight[1])
    x_h = m._residue_stream(
        chain_idx=0,
        aa=batch.get("heavy_aa"),
        plm=batch.get("heavy_plm"),
        pos=batch["heavy_pos"],
        imgt=batch.get("heavy_imgt"),
        region=batch.get("heavy_region"),
    )
    x_l = m._residue_stream(
        chain_idx=1,
        aa=batch.get("light_aa"),
        plm=batch.get("light_plm"),
        pos=batch["light_pos"],
        imgt=batch.get("light_imgt"),
        region=batch.get("light_region"),
    )
    assert x_h.shape == (B, Lh, 16) and x_l.shape == (B, Ll, 16)


def test_one_encoder_call_joint_layout_and_no_hl_block():
    m = _joint_dual()
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=32,
        Lh=8,
        Ll=6,
        B=2,
    )
    B, Lh = batch["heavy_mask"].shape
    Ll = batch["light_mask"].shape[1]
    calls = []
    orig = m.encoder.forward

    def wrapped(src, src_key_padding_mask=None, **kwargs):
        calls.append(src.shape)
        assert kwargs.get("src_mask") is None
        return orig(src, src_key_padding_mask=src_key_padding_mask, **kwargs)

    m.encoder.forward = wrapped  # type: ignore
    m.eval()
    with torch.no_grad():
        z = m.forward_repr(batch)
    assert len(calls) == 1
    assert calls[0] == (B, 2 + Lh + Ll, 16)
    assert z.shape == (B, 32)


def test_readout_is_concat_reg_h_reg_l():
    m = _joint_dual()
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=32,
        Lh=8,
        Ll=6,
        B=2,
    )
    B, Lh = batch["heavy_mask"].shape
    m.eval()
    with torch.no_grad():

        def fake_enc(src, src_key_padding_mask=None, **kwargs):
            out = torch.zeros_like(src)
            out[:, 0, :] = 1.0  # REG_H
            out[:, 1 + Lh, :] = 2.0  # REG_L
            return out

        m.encoder.forward = fake_enc  # type: ignore
        z = m.forward_repr(batch)
    assert torch.allclose(z[:, :16], torch.ones(B, 16))
    assert torch.allclose(z[:, 16:], torch.full((B, 16), 2.0))


def test_per_chain_pos_semantics():
    m = _joint_dual()
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=32,
        Lh=6,
        Ll=5,
        B=1,
    )
    assert int(batch["heavy_pos"][0, 0].item()) == int(batch["light_pos"][0, 0].item())
    assert m.pos_emb.num_embeddings > 2


def test_padding_mask_reg_slots_never_pad():
    m = _joint_dual()
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=32,
        Lh=8,
        Ll=6,
        B=2,
    )
    B, Lh = batch["heavy_mask"].shape
    Ll = batch["light_mask"].shape[1]
    batch["heavy_mask"][:, -1] = False
    batch["light_mask"][:, -1] = False
    captured = {}

    def wrap(src, src_key_padding_mask=None, **kwargs):
        captured["pad"] = src_key_padding_mask.clone()
        return torch.zeros_like(src)

    m.encoder.forward = wrap  # type: ignore
    m.forward_repr(batch)
    pad = captured["pad"]
    assert pad.shape == (B, 2 + Lh + Ll)
    assert not pad[:, 0].any()  # REG_H
    assert not pad[:, 1 + Lh].any()  # REG_L
    assert pad[:, Lh].all()  # last H pad
    assert pad[:, -1].all()  # last L pad


def test_xor_single_dual():
    with pytest.raises(ValueError):
        AnnotatedTransformer(
            content_mode="frozen",
            plm_hidden=32,
            joint_hl_single_reg=True,
            joint_hl_dual_reg=True,
        )


def test_no_fusion_recipe_flag_in_train_kwargs_pattern():
    m = _joint_dual()
    assert not hasattr(m, "fixed_proj")


def test_param_counts_ordering():
    kw = dict(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
    )
    t030 = AnnotatedTransformer(**kw)
    t068 = AnnotatedTransformer(**kw, joint_hl_single_reg=True)
    t070 = AnnotatedTransformer(**kw, joint_hl_dual_reg=True)
    assert t030.repr_dim == t070.repr_dim == 256
    assert t068.repr_dim == 128
    assert t070.n_trainable_parameters() > t068.n_trainable_parameters()
    assert t070.n_trainable_parameters() == t030.n_trainable_parameters()


def test_exp_t070_artifacts_if_present():
    import pandas as pd

    code = "EXP-T070"
    cfg = ROOT / "experiments" / "configs" / f"{code}.yaml"
    if not cfg.exists():
        pytest.skip("EXP-T070 not registered yet")
    assert not (ROOT / "experiments" / "features" / f"{code}.parquet").exists()
    pred = ROOT / "experiments" / "predictions" / code
    for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
        assert (pred / name).exists()
    freeze = ROOT / "results" / "EXP-T070_CV_FREEZE.yaml"
    assert freeze.exists()
    from _lib import mae

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    row = exp[exp["experiment_code"] == code].iloc[0]
    assert row["input_space"] == "JOINT_HL_DUAL_REG_FROZEN_RESIDUE"
    t030 = exp[exp["experiment_code"] == "EXP-T030"].iloc[0]
    assert abs(float(t030["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
    dev = pd.read_csv(ROOT / "data" / "dev.csv")
    y = dev.set_index("id")["TmApp"]
    for scheme, col in (("oof_primary.csv", "cv_primary_mae"), ("oof_shadow.csv", "cv_shadow_mae")):
        p = pd.read_csv(pred / scheme)
        p["id"] = p["id"].astype(str)
        ids = dev["id"].astype(str).tolist()
        pred_s = p.set_index("id").loc[ids, "TmApp"]
        assert abs(mae(y.loc[ids].to_numpy(float), pred_s.to_numpy(float)) - float(row[col])) < 1e-10
