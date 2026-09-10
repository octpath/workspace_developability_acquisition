#!/usr/bin/env python3
"""Architecture unit tests for EXP-T076..T079 under V3 platform."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))

from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.protocol_v3 import PLATFORM_ID, normalize_arch_flags  # noqa: E402
from antibody_transformer.equivalence import synthetic_batch  # noqa: E402


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


def test_platform_id_frozen():
    assert PLATFORM_ID == "DL_FOLDLOCAL_COSINE_V3"
    freeze = ROOT / "results" / "DL_FOLDLOCAL_COSINE_V3_FREEZE.yaml"
    assert freeze.exists()
    assert "FROZEN" in freeze.read_text()


def test_t076_single_reg():
    m = AnnotatedTransformer(**_kw(joint_hl_single_reg=True))
    assert m.joint_hl_single_reg
    assert m.repr_dim == 32
    assert not m.joint_hl_dual_reg


def test_t077_dual_unrestricted():
    m = AnnotatedTransformer(**_kw(joint_hl_dual_reg=True))
    assert m.joint_hl_dual_reg
    assert m.repr_dim == 64
    assert not m.joint_hl_chain_specific_dual_reg


def test_t078_chain_specific_mask():
    m = AnnotatedTransformer(**_kw(joint_hl_chain_specific_dual_reg=True))
    assert m.joint_hl_chain_specific_dual_reg
    mask = m.build_chain_specific_reg_attn_mask(4, 3)
    # REG_H at 0 cannot attend REG_L at 5 or L residues 6..
    assert bool(mask[0, 5])  # blocked REG_L
    assert bool(mask[0, 6])  # blocked L
    assert not bool(mask[0, 1])  # H allowed


def test_t079_zero_gate_and_grad():
    torch.manual_seed(0)
    t030 = AnnotatedTransformer(**_kw())
    t079 = AnnotatedTransformer(**_kw(use_cross_attention_bridge=True))
    sd = t079.state_dict()
    for k, v in t030.state_dict().items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.clone()
    t079.load_state_dict(sd)
    assert float(t079.cross_gate_h) == 0.0
    assert t079.cross_gate_h.requires_grad
    batch = synthetic_batch(content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=64, Lh=6, Ll=5, B=2)
    t030.eval()
    t079.eval()
    with torch.no_grad():
        d = float((t030(batch) - t079(batch)).abs().max())
    assert d <= 1e-6


def test_arch_flags_mutex():
    with pytest.raises(ValueError):
        normalize_arch_flags({"joint_hl_single_reg": True, "joint_hl_dual_reg": True})


def test_preregistration_exists_and_hashes():
    p = ROOT / "results" / "T076_T079_ARCHITECTURE_PREREGISTRATION.yaml"
    if not p.exists():
        pytest.skip("prereg not written yet")
    doc = yaml.safe_load(p.read_text())
    assert doc["platform_id"] == PLATFORM_ID
    assert len(doc["experiments"]) == 4
    for row in doc["experiments"]:
        cfg = ROOT / row["config_path"]
        assert cfg.exists()
        import hashlib

        assert hashlib.sha256(cfg.read_bytes()).hexdigest() == row["config_sha256"]
