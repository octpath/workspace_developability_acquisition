#!/usr/bin/env python3
"""Focused tests for EXP-T071 chain-specific dual-REG attention mask."""
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


def _t071(**kwargs):
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
        joint_hl_chain_specific_dual_reg=True,
        **kwargs,
    )


def test_mask_semantics_synthetic():
    Lh, Ll = 4, 3
    m = AnnotatedTransformer.build_chain_specific_reg_attn_mask(Lh, Ll)
    # True = blocked
    reg_h, reg_l = 0, 1 + Lh
    # REG_H -> L / REG_L blocked; REG_H -> H / self allowed
    assert m[reg_h, reg_l]
    assert m[reg_h, 2 + Lh :].all()
    assert not m[reg_h, reg_h]
    assert not m[reg_h, 1 : 1 + Lh].any()
    # REG_L -> H / REG_H blocked
    assert m[reg_l, reg_h]
    assert m[reg_l, 1 : 1 + Lh].all()
    assert not m[reg_l, reg_l]
    assert not m[reg_l, 2 + Lh :].any()
    # H residue -> REG_L blocked; H->L allowed
    assert m[1 : 1 + Lh, reg_l].all()
    assert not m[1, 2 + Lh]
    # L residue -> REG_H blocked; L->H allowed
    assert m[2 + Lh : 2 + Lh + Ll, reg_h].all()
    assert not m[2 + Lh, 1]


def test_dual_reg_concat_dim():
    m = _t071()
    assert m.reg_token.shape == (2, 16)
    assert m.repr_dim == 32
    batch = synthetic_batch(
        content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=32, Lh=5, Ll=4, B=2
    )
    m.eval()
    with torch.no_grad():
        z = m.forward_repr(batch)
    assert z.shape == (2, 32)


def test_mask_passed_and_blocks_verified_via_manual_score():
    """Do not trust construction alone: verify blocked pairs are True in mask used by encoder."""
    m = _t071()
    batch = synthetic_batch(
        content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=32, Lh=6, Ll=5, B=1
    )
    Lh, Ll = 6, 5
    seen = {}

    def wrap(src, mask=None, src_key_padding_mask=None, **kwargs):
        seen["mask"] = mask
        seen["pad"] = src_key_padding_mask
        return torch.zeros_like(src)

    m.encoder.forward = wrap  # type: ignore
    m.forward_repr(batch)
    assert seen["mask"] is not None
    assert torch.equal(seen["mask"], AnnotatedTransformer.build_chain_specific_reg_attn_mask(Lh, Ll))
    # required blocks
    assert bool(seen["mask"][0, 1 + Lh])  # REG_H -> REG_L
    assert bool(seen["mask"][0, 2 + Lh])  # REG_H -> L
    assert bool(seen["mask"][1 + Lh, 0])  # REG_L -> REG_H
    assert bool(seen["mask"][1 + Lh, 1])  # REG_L -> H
    assert bool(seen["mask"][1, 1 + Lh])  # H -> REG_L
    assert bool(seen["mask"][2 + Lh, 0])  # L -> REG_H
    # allowed cross residue
    assert not bool(seen["mask"][1, 2 + Lh])  # H -> L
    assert not bool(seen["mask"][2 + Lh, 1])  # L -> H


def test_xor_modes():
    with pytest.raises(ValueError):
        AnnotatedTransformer(
            content_mode="frozen",
            plm_hidden=32,
            joint_hl_dual_reg=True,
            joint_hl_chain_specific_dual_reg=True,
        )


def test_exp_t071_artifacts_if_present():
    import pandas as pd

    code = "EXP-T071"
    if not (ROOT / "experiments" / "configs" / f"{code}.yaml").exists():
        pytest.skip("EXP-T071 not registered yet")
    assert not (ROOT / "experiments" / "features" / f"{code}.parquet").exists()
    pred = ROOT / "experiments" / "predictions" / code
    for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
        assert (pred / name).exists()
    freeze = ROOT / "results" / "EXP-T071_CV_FREEZE.yaml"
    assert freeze.exists()
    from _lib import mae

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    row = exp[exp["experiment_code"] == code].iloc[0]
    assert "CHAIN_SPECIFIC" in str(row["input_space"])
    t030 = exp[exp["experiment_code"] == "EXP-T030"].iloc[0]
    assert abs(float(t030["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
    t070 = exp[exp["experiment_code"] == "EXP-T070"].iloc[0]
    assert abs(float(t070["cv_primary_mae"]) - 3.329090573660141) < 1e-12
    dev = pd.read_csv(ROOT / "data" / "dev.csv")
    y = dev.set_index("id")["TmApp"]
    for scheme, col in (("oof_primary.csv", "cv_primary_mae"), ("oof_shadow.csv", "cv_shadow_mae")):
        p = pd.read_csv(pred / scheme)
        p["id"] = p["id"].astype(str)
        ids = dev["id"].astype(str).tolist()
        pred_s = p.set_index("id").loc[ids, "TmApp"]
        assert abs(mae(y.loc[ids].to_numpy(float), pred_s.to_numpy(float)) - float(row[col])) < 1e-10
