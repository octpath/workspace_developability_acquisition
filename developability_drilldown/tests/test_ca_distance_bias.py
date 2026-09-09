"""Tests for learnable Cα-distance attention bias (EXP-T067)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.equivalence import synthetic_batch  # noqa: E402
from antibody_transformer.distance_bias import (  # noqa: E402
    SharedDistanceBiasTransformerEncoder,
    softplus_inverse,
)
from classical_features.ca_cache import extract_ca_chains  # noqa: E402


def test_softplus_inverse_roundtrip():
    for y in (0.5, 1.0, 5.0, 12.3):
        x = softplus_inverse(y)
        assert abs(math.log1p(math.exp(x)) - y) < 1e-6 or abs(math.log(math.expm1(y) + 1) - math.log1p(math.exp(x))) < 1e-4
        # softplus(x) ~= y
        assert abs(math.log1p(math.exp(x)) - y) < 1e-5


def test_zero_amplitude_equivalence():
    torch.manual_seed(0)
    ctl = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        use_ca_distance_bias=False,
    )
    torch.manual_seed(0)
    dist = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        use_ca_distance_bias=True,
        initial_ell_angstrom=8.0,
    )
    # copy overlapping weights
    sd = ctl.state_dict()
    mapped = {}
    for k, v in sd.items():
        # stock encoder.layers.N -> distance encoder.layers.N
        nk = k
        if k.startswith("encoder.layers."):
            nk = k  # same path in SharedDistanceBiasTransformerEncoder
        if nk in dist.state_dict() and dist.state_dict()[nk].shape == v.shape:
            mapped[nk] = v
    dist.load_state_dict(mapped, strict=False)
    assert torch.all(dist.encoder.a == 0)

    batch = synthetic_batch(content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=1280)
    B, Lh = batch["heavy_plm"].shape[:2]
    Ll = batch["light_plm"].shape[1]
    batch["heavy_ca"] = torch.randn(B, Lh, 3)
    batch["light_ca"] = torch.randn(B, Ll, 3)
    ctl.eval()
    dist.eval()
    with torch.no_grad():
        r0 = ctl.forward_repr(batch)
        r1 = dist.forward_repr(batch)
        y0 = ctl(batch)
        y1 = dist(batch)
    assert float((r0 - r1).abs().max()) < 1e-6
    assert float((y0 - y1).abs().max()) < 1e-6


def test_self_pair_and_reg_zero_bias():
    enc = SharedDistanceBiasTransformerEncoder(
        torch.nn.TransformerEncoderLayer(16, 4, 32, 0.0, batch_first=True, norm_first=True),
        num_layers=1,
        n_heads=4,
    )
    with torch.no_grad():
        enc.a.fill_(1.0)
        enc.set_initial_ell(5.0)
    B, L = 2, 5
    coords = torch.randn(B, L, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    bias = enc.build_attn_bias(coords, mask)  # [B*H, 1+L, 1+L]
    H = 4
    bias4 = bias.view(B, H, 1 + L, 1 + L)
    # REG row/col zero
    assert torch.allclose(bias4[:, :, 0, :], torch.zeros_like(bias4[:, :, 0, :]))
    assert torch.allclose(bias4[:, :, :, 0], torch.zeros_like(bias4[:, :, :, 0]))
    # diagonal among residues zero
    for i in range(L):
        assert torch.allclose(bias4[:, :, 1 + i, 1 + i], torch.zeros(B, H))


def test_ell_positive():
    enc = SharedDistanceBiasTransformerEncoder(
        torch.nn.TransformerEncoderLayer(8, 2, 16, 0.0, batch_first=True, norm_first=True),
        num_layers=1,
        n_heads=2,
    )
    enc.set_initial_ell(7.5)
    ell = enc.length_scales()
    assert torch.all(ell > 0)
    assert abs(float(ell[0]) - 7.5) < 1e-4


def test_ca_euclidean_helper():
    # two points distance
    coords = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]], dtype=torch.float32)
    d = torch.linalg.vector_norm(coords[:, None, :, :] - coords[:, :, None, :], dim=-1)
    assert abs(float(d[0, 0, 1]) - 5.0) < 1e-5


def test_exp_kernel_shape():
    enc = SharedDistanceBiasTransformerEncoder(
        torch.nn.TransformerEncoderLayer(8, 2, 16, 0.0, batch_first=True, norm_first=True),
        num_layers=1,
        n_heads=2,
    )
    with torch.no_grad():
        enc.a.fill_(0.5)
        enc.set_initial_ell(10.0)
    coords = torch.randn(3, 6, 3)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, -1] = False
    bias = enc.build_attn_bias(coords, mask)
    assert bias.shape == (3 * 2, 7, 7)


@pytest.mark.skipif(
    not (Path("/workspace_developability_acquisition/feature_extension/data/esmfold_fv").exists()),
    reason="PDB dir missing",
)
def test_extract_ca_smoke():
    import pandas as pd

    root = Path("/workspace_developability_acquisition")
    dev = pd.read_csv(root / "developability_drilldown/data/dev.csv")
    row = dev.iloc[0]
    pdb = root / "feature_extension/data/esmfold_fv" / f"{row.id}.pdb"
    h, l, mh, ml = extract_ca_chains(pdb, str(row.heavy), str(row.light))
    assert mh and ml
    assert h.shape == (len(row.heavy), 3)
    assert np.isfinite(h).all() and np.isfinite(l).all()


def test_exp_t067_artifacts_if_present():
    import pandas as pd
    import yaml

    sys.path.insert(0, str(ROOT / "scripts"))
    code = "EXP-T067"
    cfg = ROOT / "experiments" / "configs" / f"{code}.yaml"
    if not cfg.exists():
        pytest.skip("EXP-T067 not registered yet")
    assert (ROOT / "experiments" / "features" / f"{code}.parquet").exists()
    assert (ROOT / "experiments" / "inputs" / f"{code}_ca.parquet").exists()
    pred = ROOT / "experiments" / "predictions" / code
    for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
        assert (pred / name).exists()
    freeze_path = ROOT / "results" / "EXP-T067_CV_FREEZE.yaml"
    assert freeze_path.exists()
    freeze = yaml.safe_load(freeze_path.read_text())
    assert freeze["control_experiment_code"] == "EXP-T037"
    assert freeze["config"]["pairwise_geometry"]["cross_chain_distance"] is False
    assert freeze["config"]["pairwise_geometry"]["distance_kernel_shared_across_layers"] is True
    assert "public_mae" in freeze  # recorded after freeze, not used for selection
    assert freeze.get("selection_statement") or freeze.get("cv_scientific_verdict")

    from _lib import mae  # noqa: E402

    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    row = exp[exp["experiment_code"] == code].iloc[0]
    assert row["input_space"] == "RESIDUE_PLUS_FIXED_FEATURES_PLUS_CA_DISTANCE"
    assert row["shareability_status"] == "SHAREABLE_COMPLETE"
    assert row["reproduction_status"] == "REPRODUCED"
    dev = pd.read_csv(ROOT / "data" / "dev.csv")
    y = dev.set_index("id")["TmApp"]
    for scheme, col in (("oof_primary.csv", "cv_primary_mae"), ("oof_shadow.csv", "cv_shadow_mae")):
        p = pd.read_csv(pred / scheme)
        p["id"] = p["id"].astype(str)
        ids = dev["id"].astype(str).tolist()
        pred_s = p.set_index("id").loc[ids, "TmApp"]
        assert abs(mae(y.loc[ids].to_numpy(float), pred_s.to_numpy(float)) - float(row[col])) < 1e-10
    t065 = exp[exp["experiment_code"] == "EXP-T065"].iloc[0]
    assert row["feature_content_sha256"] == t065["feature_content_sha256"]
    assert (ROOT / "results" / "EXP-T067_DISTANCE_PARAMETERS.csv").exists()
