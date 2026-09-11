#!/usr/bin/env python3
"""Tests for HSP spatial hydrophobicity research track (no mainline EXP-H)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scales import (  # noqa: E402
    AA20,
    BM_SAP,
    SCALES,
    UNAVAILABLE,
    property_minmax,
    property_raw,
)
from spatial_engine import (  # noqa: E402
    RADII,
    aggregate,
    family_id,
    pairwise_centroid,
    spatial_scores,
)


def test_scales_verified_and_oriented():
    for sid, sc in SCALES.items():
        assert set(sc.values) == set(AA20)
        # hydrophobic Ile should not be the most hydrophilic
        assert sc.values["I"] >= sc.values["R"] or sid in ("WW",)  # WW: I vs R both high; I>=R
        assert sc.values["I"] >= min(sc.values.values())
    assert "JAIN_HIC" in UNAVAILABLE
    mm = property_minmax("KD")
    assert mm["I"] == pytest.approx(1.0)
    assert mm["R"] == pytest.approx(0.0)
    assert abs(BM_SAP["G"]) < 1e-12


def test_radii_and_family_id():
    assert RADII == [4.0, 5.0, 6.0, 7.5, 8.0, 10.0]
    fid = family_id("KD", "MINMAX", "TOTAL_RASA_TIEN", "CENTROID", 5.0)
    assert "R5p0" in fid


def test_spatial_score_self_and_no_decay():
    C = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    D = pairwise_centroid(C)
    exp = np.array([0.5, 1.0, 0.2])
    hydro = np.array([1.0, 0.5, 1.0])
    s = spatial_scores(D, exp, hydro, 5.0)
    assert s[0] == pytest.approx(1.0 * 0.5 + 0.5 * 1.0)
    assert s[2] == pytest.approx(1.0 * 0.2)


def test_topk_when_fewer_than_k():
    scores = np.array([1.0, 2.0])
    ag = aggregate(scores, np.array([True, True]))
    assert ag["TOP5_MEAN"] == pytest.approx(1.5)
    assert ag["TOP3_MEAN"] == pytest.approx(1.5)


def test_exp_h102_unused():
    sys.path.insert(0, str(ROOT.parents[1] / "developability_drilldown" / "scripts"))
    from experiment_codes import next_code

    assert next_code("HIC") == "EXP-H114"


def test_no_hic_in_geometry_if_present():
    p = ROOT / "features" / "cache" / "residue_geometry_sasa.parquet"
    if not p.exists():
        pytest.skip("geometry cache not built")
    import pandas as pd

    df = pd.read_parquet(p)
    assert "HIC" not in df.columns
    assert df["id"].nunique() == 324
    assert (df["total_SASA"] + 1e-9 >= df["sidechain_SASA"]).all()
    assert {"total_rASA_Tien", "sidechain_over_Tien", "Tien_MaxASA"}.issubset(df.columns)
    assert ((df["total_rASA_Tien"] >= 0) & (df["total_rASA_Tien"] <= 1)).all()


def test_q90_q95_and_mask():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    mask = np.ones(10, dtype=bool)
    ag = aggregate(scores, mask)
    assert ag["Q90"] == pytest.approx(8.1)
    assert ag["Q95"] == pytest.approx(8.55)
    assert ag["MAX"] == 9.0
    # region mask: only first two centers
    ag2 = aggregate(scores, np.array([True, True] + [False] * 8))
    assert ag2["MAX"] == 1.0
    assert ag2["MEAN"] == pytest.approx(0.5)


def test_shortlist_rule_preregistered():
    rule = ROOT / "results" / "STAGE1_SHORTLIST_RULE.yaml"
    freeze = ROOT / "results" / "STAGE1_SHORTLIST_FREEZE.yaml"
    assert rule.exists()
    if freeze.exists():
        text = freeze.read_text()
        assert "consumed_EXP_H102: false" in text
        assert "EXP-H102" in text


def test_antibody_atlas_no_hic_labels():
    p = ROOT / "features" / "antibody_spatial_hydrophobicity.parquet"
    if not p.exists():
        pytest.skip("antibody atlas not built")
    import pandas as pd

    df = pd.read_parquet(p)
    assert "HIC" not in df.columns
    assert df.shape[0] == 324
    assert df.shape[1] > 1000
