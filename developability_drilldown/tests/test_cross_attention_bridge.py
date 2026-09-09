#!/usr/bin/env python3
"""Focused tests for EXP-T072 separate self-attn + cross-attention bridge."""
from __future__ import annotations

import copy
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


def _base_kw(**extra):
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


def test_zero_gate_equivalence_vs_t030():
    torch.manual_seed(0)
    t030 = AnnotatedTransformer(**_base_kw())
    t072 = AnnotatedTransformer(**_base_kw(use_cross_attention_bridge=True))
    # copy overlapping weights
    sd030 = t030.state_dict()
    sd072 = t072.state_dict()
    for k, v in sd030.items():
        if k in sd072 and sd072[k].shape == v.shape:
            sd072[k] = v.clone()
    t072.load_state_dict(sd072)
    assert float(t072.cross_gate_h) == 0.0
    assert float(t072.cross_gate_l) == 0.0

    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=64,
        Lh=8,
        Ll=6,
        B=3,
    )
    t030.eval()
    t072.eval()
    with torch.no_grad():
        y0 = t030(batch)
        y1 = t072(batch)
        z0 = t030.forward_repr(batch)
        z1 = t072.forward_repr(batch)
    max_pred = float((y0 - y1).abs().max())
    max_repr = float((z0 - z1).abs().max())
    assert max_pred <= 1e-6, f"pred diff {max_pred}"
    assert max_repr <= 1e-6, f"repr diff {max_repr}"


def test_gates_require_grad_and_in_optimizer():
    m = AnnotatedTransformer(**_base_kw(use_cross_attention_bridge=True))
    assert m.cross_gate_h.requires_grad
    assert m.cross_gate_l.requires_grad
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert "cross_gate_h" in names and "cross_gate_l" in names
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch = synthetic_batch(
        content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=64, Lh=5, Ll=4, B=2
    )
    m.train()
    # force non-zero gates path to get gradient flow into gates
    with torch.no_grad():
        m.cross_gate_h.fill_(0.1)
        m.cross_gate_l.fill_(0.1)
    y = m(batch)
    loss = y.pow(2).mean()
    loss.backward()
    assert m.cross_gate_h.grad is not None
    assert m.cross_gate_l.grad is not None
    opt.step()


def test_reg_excluded_from_cross_qv():
    m = AnnotatedTransformer(**_base_kw(use_cross_attention_bridge=True))
    batch = synthetic_batch(
        content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=64, Lh=5, Ll=4, B=2
    )
    captured = {}

    def wrap(q, k, v, key_padding_mask=None, need_weights=False):
        captured.setdefault("q_lens", []).append(q.shape[1])
        captured.setdefault("k_lens", []).append(k.shape[1])
        # return zeros same shape as q
        return torch.zeros_like(q), None

    m.cross_attn.forward = wrap  # type: ignore
    m.eval()
    with torch.no_grad():
        m.forward_repr(batch)
    # residue lengths only (no REG)
    assert captured["q_lens"] == [5, 4]
    assert captured["k_lens"] == [4, 5]


def test_param_count_increases():
    t030 = AnnotatedTransformer(**_base_kw())
    t072 = AnnotatedTransformer(**_base_kw(use_cross_attention_bridge=True))
    assert t072.n_trainable_parameters() > t030.n_trainable_parameters()
    acc = t072.param_account()
    assert acc["cross_attention"] > 0
    assert acc["gates"] == 2
    assert acc["total"] == t072.n_trainable_parameters()


def test_exp_t072_artifacts_if_present():
    import pandas as pd

    code = "EXP-T072"
    if not (ROOT / "experiments" / "configs" / f"{code}.yaml").exists():
        pytest.skip("EXP-T072 not registered yet")
    assert not (ROOT / "experiments" / "features" / f"{code}.parquet").exists()
    assert (ROOT / "results" / "EXP-T072_CV_FREEZE.yaml").exists()
    assert (ROOT / "results" / "EXP-T072_CROSS_GATES.csv").exists() or (
        ROOT / "results" / "EXP-T072_run" / "cross_gates.csv"
    ).exists()
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    row = exp[exp["experiment_code"] == code].iloc[0]
    assert "CROSS" in str(row["input_space"]).upper()
    t030 = exp[exp["experiment_code"] == "EXP-T030"].iloc[0]
    assert abs(float(t030["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
