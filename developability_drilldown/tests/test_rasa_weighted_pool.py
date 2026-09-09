"""Focused tests for RASA-weighted REG pooling (EXP-T066)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))

from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.equivalence import synthetic_batch  # noqa: E402


def _make_models(seed: int = 0):
    torch.manual_seed(seed)
    ctl = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        use_continuous_rasa=False,
        use_rasa_weighted_pool=False,
    )
    torch.manual_seed(seed)
    pool = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        use_continuous_rasa=False,
        use_rasa_weighted_pool=True,
    )
    # align overlapping weights
    sd = ctl.state_dict()
    pool.load_state_dict(
        {k: v for k, v in sd.items() if k in pool.state_dict() and pool.state_dict()[k].shape == v.shape},
        strict=False,
    )
    assert torch.all(pool.rasa_pool_proj.weight == 0)
    return ctl, pool


def test_zero_init_equivalence_to_control():
    ctl, pool = _make_models(7)
    batch = synthetic_batch(content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=1280)
    B, Lh = batch["heavy_plm"].shape[:2]
    Ll = batch["light_plm"].shape[1]
    batch["heavy_rasa"] = torch.rand(B, Lh)
    batch["light_rasa"] = torch.rand(B, Ll)
    ctl.eval()
    pool.eval()
    with torch.no_grad():
        y0 = ctl(batch)
        y1 = pool(batch)
    assert float((y0 - y1).abs().max()) < 1e-6


def test_shared_projection_h_l():
    _, pool = _make_models(1)
    assert pool.rasa_pool_proj is not None
    # single module used for both chains
    assert isinstance(pool.rasa_pool_proj, torch.nn.Linear)
    assert pool.rasa_pool_proj.bias is None


def test_rasa_weighted_pool_math():
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=32,
        annotation_mode="minimal",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=8,
        use_rasa_weighted_pool=True,
    )
    h = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, False]])
    rasa = torch.tensor([[1.0, 3.0, 9.0]])
    pool = m.rasa_weighted_residue_pool(h, mask, rasa)
    # only first two residues: (1*h0 + 3*h1) / 4 = ([1,0]+[0,6])/4 = [0.25, 1.5]
    expected = torch.tensor([[0.25, 1.5]])
    assert torch.allclose(pool, expected, atol=1e-6)


def test_zero_denominator_fallback():
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=32,
        annotation_mode="minimal",
        merge_mode="h_only",
        chain_mode="H_ONLY",
        pooling_mode="reg",
        d_model=4,
        use_rasa_weighted_pool=True,
    )
    h = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    rasa = torch.zeros(2, 5)
    rasa[0, :] = 0.0
    rasa[1, 0] = 2.0
    m.rasa_pool_zero_denom_count = 0
    pool = m.rasa_weighted_residue_pool(h, mask, rasa)
    assert torch.allclose(pool[0], torch.zeros(4))
    assert m.rasa_pool_zero_denom_count == 1
    assert not torch.allclose(pool[1], torch.zeros(4))


def test_mutually_exclusive_rasa_modes():
    with pytest.raises(ValueError):
        AnnotatedTransformer(
            content_mode="frozen",
            plm_hidden=32,
            use_continuous_rasa=True,
            use_rasa_weighted_pool=True,
        )


def test_exp_t066_artifacts_if_present():
    code = "EXP-T066"
    cfg = ROOT / "experiments" / "configs" / f"{code}.yaml"
    if not cfg.exists():
        pytest.skip("EXP-T066 not registered yet")
    assert (ROOT / "experiments" / "features" / f"{code}.parquet").exists()
    assert (ROOT / "experiments" / "inputs" / f"{code}_rasa.parquet").exists()
    pred = ROOT / "experiments" / "predictions" / code
    for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
        assert (pred / name).exists()
    freeze = ROOT / "results" / "EXP-T066_CV_FREEZE.yaml"
    assert freeze.exists()
    # score recompute
    from _lib import mae  # noqa: E402

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    row = exp[exp["experiment_code"] == code].iloc[0]
    dev = pd.read_csv(ROOT / "data" / "dev.csv")
    y = dev.set_index("id")["TmApp"]
    for scheme, col in (("oof_primary.csv", "cv_primary_mae"), ("oof_shadow.csv", "cv_shadow_mae")):
        p = pd.read_csv(pred / scheme)
        p["id"] = p["id"].astype(str)
        ids = dev["id"].astype(str).tolist()
        pred_s = p.set_index("id").loc[ids, "TmApp"]
        assert abs(mae(y.loc[ids].to_numpy(float), pred_s.to_numpy(float)) - float(row[col])) < 1e-10
    # fixed feature hash vs T065
    t065 = exp[exp["experiment_code"] == "EXP-T065"].iloc[0]
    assert row["feature_content_sha256"] == t065["feature_content_sha256"]
    assert row["feature_set_id"] == t065["feature_set_id"]
