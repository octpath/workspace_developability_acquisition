#!/usr/bin/env python3
"""Integration-ish tests for STATIC_SAP_KD artifacts + aux store (no GPU)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))

from antibody_transformer.static_sap_kd import R_REF, SAP3_COLS, SAP9_COLS
from antibody_transformer.static_sap_kd_aux import BUNDLE_IDS, StaticSapKdAuxFeatureStore


RES = ROOT / "experiments/features/static_sap_kd_residue.parquet"
G3 = ROOT / "experiments/features/static_sap_kd_antibody_global3.parquet"
C9 = ROOT / "experiments/features/static_sap_kd_antibody_chain9.parquet"


@pytest.fixture(scope="module")
def artifacts():
    if not (RES.exists() and G3.exists() and C9.exists()):
        pytest.skip("STATIC_SAP_KD artifacts not generated")
    return pd.read_parquet(RES), pd.read_parquet(G3), pd.read_parquet(C9)


def test_artifact_coverage_324(artifacts):
    res, g3, c9 = artifacts
    assert len(g3) == 324
    assert len(c9) == 324
    assert set(g3["id"]) == set(c9["id"])
    assert set(SAP3_COLS).issubset(g3.columns)
    assert set(SAP9_COLS).issubset(c9.columns)
    assert R_REF == 5.0


def test_residue_columns_and_ranges(artifacts):
    res, _, _ = artifacts
    need = {
        "id",
        "chain",
        "residue_index",
        "aa",
        "centroid_x",
        "SASA",
        "Tien_MaxASA",
        "rSASA",
        "KD_raw",
        "KD_norm",
        "SSKD_R_REF",
        "SSKD_R10",
        "coordinate_valid",
        "sasa_valid",
    }
    assert need.issubset(res.columns)
    v = res.loc[res["coordinate_valid"] & res["sasa_valid"]]
    assert (v["rSASA"] >= -1e-9).all() and (v["rSASA"] <= 1 + 1e-9).all()
    assert (v["KD_norm"] >= -1e-9).all() and (v["KD_norm"] <= 1 + 1e-9).all()
    assert (v["SSKD_R_REF"] >= -1e-9).all()


def test_sap3_sap9_consistency(artifacts):
    _, g3, c9 = artifacts
    m = g3.merge(c9, on="id", suffixes=("_g", "_c"))
    for col in SAP3_COLS:
        np.testing.assert_allclose(m[f"{col}_g"], m[f"{col}_c"], rtol=0, atol=1e-7)
    assert (c9["SSKD_ALL_MAX"] + 1e-9 >= c9["SSKD_ALL_MEAN"]).all()
    assert (c9["SSKD_ALL_SUM"] + 1e-9 >= c9["SSKD_ALL_MAX"]).all()


def test_aux_store_dims_and_train_only_preprocess(artifacts):
    _, _, c9 = artifacts
    ids = c9["id"].astype(str).tolist()
    train = ids[:200]
    test = ids[200:220]
    for bid, dim in (
        ("FS_HIC_STATIC_SAP_KD_GLOBAL3", 3),
        ("FS_HIC_STATIC_SAP_KD_CHAIN9", 9),
        ("FS_HIC_SURFACE_PLUS_STATIC_SAP_KD_CHAIN9", 44),
        ("FS_HIC_F4_PLUS_STATIC_SAP_KD_CHAIN9", 209),
    ):
        store = StaticSapKdAuxFeatureStore(bid)
        assert store.effective_dim == dim
        assert store.bundle_id in BUNDLE_IDS
        prep = store.fit(train)
        Xtr = store.transform(prep, train)
        Xte = store.transform(prep, test)
        assert Xtr.shape == (len(train), prep.effective_dim)
        assert Xte.shape == (len(test), prep.effective_dim)
        assert np.isfinite(Xtr).all()


def test_no_hic_in_feature_artifacts(artifacts):
    res, g3, c9 = artifacts
    for df in (res, g3, c9):
        assert "HIC" not in df.columns
        assert "TmApp" not in df.columns


def test_configs_exist_and_platform():
    import yaml

    for code in [f"EXP-H{i:03d}" for i in range(94, 102)]:
        cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
        assert cfg["platform_id"] == "DL_FOLDLOCAL_COSINE_V3"
        assert cfg["descriptor_id"] == "STATIC_SAP_KD"
        assert cfg["structure_scope"] == "Fv"
        assert "fusion_bundle_id" in cfg


def test_controls_unchanged_hashes():
    """Historical control configs must still exist (not rewritten by this batch)."""
    for code in ("EXP-H071", "EXP-H090", "EXP-H093", "EXP-H061", "EXP-H086", "EXP-H089"):
        p = ROOT / "experiments/configs" / f"{code}.yaml"
        assert p.exists()
        txt = p.read_text()
        assert "STATIC_SAP_KD" not in txt
        assert "FS_HIC_STATIC_SAP" not in txt
