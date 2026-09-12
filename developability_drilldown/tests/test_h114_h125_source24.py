#!/usr/bin/env python3
"""Tests for H114–H125 SOURCE24 fusion ablation wiring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_next_was_h114_at_prereg():
    from experiment_codes import next_code

    # After issuance during train, next may be H126; prereg must record H114
    prereg = yaml.safe_load((ROOT / "results/H114_H125_SOURCE24_FUSION_PREREGISTRATION.yaml").read_text())
    assert prereg["next_code_at_prereg"] == "EXP-H114"
    assert prereg["status"] == "PREREGISTERED_FROZEN"


def test_source_dims_and_hashes():
    from antibody_transformer.source_sap_scm_aux import SourceSapScmAuxFeatureStore

    sap = SourceSapScmAuxFeatureStore("FS_HIC_SOURCE_SAP24")
    scm = SourceSapScmAuxFeatureStore("FS_HIC_SOURCE_SCM24")
    comb = SourceSapScmAuxFeatureStore("FS_HIC_SOURCE_SAP24_SCM24")
    assert sap.effective_dim == 24
    assert scm.effective_dim == 24
    assert comb.effective_dim == 48
    prereg = yaml.safe_load((ROOT / "results/H114_H125_SOURCE24_FUSION_PREREGISTRATION.yaml").read_text())
    assert prereg["feature_blocks"]["FS_HIC_SOURCE_SAP24"]["n_dims"] == 24
    # columns match research artifact
    src = Path(__file__).resolve().parents[2] / "feature_research/hic_sap_scm_source/features/antibody_source_sap24.parquet"
    df = pd.read_parquet(src)
    assert list(df.columns[1:]) == sap.columns


def test_direct_has_no_aux_mlp_and_passes_features():
    from antibody_transformer.late_fusion import DirectLateFusionModel, LateFusionModel

    class BB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.repr_dim = 8
            self.d_model = 16
            self.lin = torch.nn.Linear(1, 8)

        def forward_repr(self, batch):
            b = batch["dummy"].shape[0]
            return self.lin(torch.ones(b, 1))

    bb = BB()
    d = DirectLateFusionModel(bb, aux_dim=24, dropout=0.2)
    assert d.aux_mlp is None
    assert not any("aux_mlp" in n for n, _ in d.named_parameters())
    fixed = torch.randn(4, 24)
    batch = {"dummy": torch.zeros(4, 1)}
    z = d.forward_repr(batch, fixed)
    assert z.shape == (4, 8 + 24)
    # last 24 dims identical to input (direct)
    assert torch.allclose(z[:, 8:], fixed)
    out = d(batch, fixed)
    assert out.shape == (4,)

    a = LateFusionModel(bb, aux_dim=24, dropout=0.2)
    assert a.aux_mlp is not None
    z2 = a.forward_repr(batch, fixed)
    assert z2.shape == (4, 8 + 32)


def test_configs_declare_fusion_mode():
    for i in range(114, 126):
        cfg = yaml.safe_load((ROOT / f"experiments/configs/EXP-H{i}.yaml").read_text())
        assert cfg["fusion_mode"] in ("late_concat_direct", "late_concat_aux32")
        assert cfg["platform_id"] == "DL_FOLDLOCAL_COSINE_V3"
        assert cfg["seed"] == 101


def test_h054_h113_untouched_marker():
    cfg = yaml.safe_load((ROOT / "experiments/configs/EXP-H113.yaml").read_text())
    assert "HSP" in cfg["fusion_bundle_id"]
    assert cfg["fusion_mode"] == "late_concat_aux32"


def test_series_map():
    from preregister_h114_h125_source24 import build_series, install_features

    series = build_series(install_features())
    assert [s["code"] for s in series] == [f"EXP-H{i}" for i in range(114, 126)]
    assert series[0]["fusion_mode"] == "late_concat_direct"
    assert series[1]["fusion_mode"] == "late_concat_aux32"
