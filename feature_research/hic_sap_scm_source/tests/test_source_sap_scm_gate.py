#!/usr/bin/env python3
"""Tests for SOURCE SAP24/SCM24 feature-research track."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
TRACK = ROOT / "feature_research" / "hic_sap_scm_source"
RES = TRACK / "results"
FEAT = TRACK / "features"
EXP_DIR = ROOT / "developability_drilldown" / "experiments"
DRILL_RES = ROOT / "developability_drilldown" / "results"

sys.path.insert(0, str(TRACK / "src"))
sys.path.insert(0, str(ROOT / "developability_drilldown" / "scripts"))

from aggregation_engine import (  # noqa: E402
    SOURCE_RADII_A,
    SOURCE_SAP24_NAMES,
    SOURCE_SCM24_NAMES,
    SAP_EXTRA20_NAMES,
    SAP_GLOBAL6_NAMES,
    SCM_EXTRA20_NAMES,
    SCM_GLOBAL6_NAMES,
    local_scores,
    pairwise_centroid,
    positive_sum_mean,
    region_masks,
    top5_mean,
    top5_share_positive,
)
from properties import AA20, CHARGE, CHARGE_HASH, KD_NORM, KD_RAW, TIEN_MAXASA  # noqa: E402
from generate_features import reconstruct_from_residue  # noqa: E402
from experiment_codes import next_code  # noqa: E402


def test_exp_h114_unused_and_next():
    assert next_code("HIC") == "EXP-H114"
    forbidden = list(DRILL_RES.glob("*H114*")) + list(EXP_DIR.rglob("*H114*"))
    assert not forbidden


def test_kd_table_and_minmax():
    assert KD_RAW["I"] == 4.5 and KD_RAW["R"] == -4.5
    assert KD_NORM["I"] == pytest.approx(1.0)
    assert KD_NORM["R"] == pytest.approx(0.0)
    for a in AA20:
        assert 0.0 <= KD_NORM[a] <= 1.0


def test_charge_table_exact():
    assert CHARGE["R"] == 1.0 and CHARGE["K"] == 1.0
    assert CHARGE["D"] == -1.0 and CHARGE["E"] == -1.0
    for a in AA20:
        if a not in "RKDE":
            assert CHARGE[a] == 0.0
    assert len(CHARGE_HASH) == 64


def test_tien_maxasa_matches_repo():
    sys.path.insert(0, str(ROOT / "organizer_extension/feature_prospecting"))
    from common.structure_utils import MAX_ASA

    for a in AA20:
        assert TIEN_MAXASA[a] == MAX_ASA[a]


def test_rasa_clip_and_sap_scm_local_formulas():
    centroids = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    D = pairwise_centroid(centroids)
    prop = np.array([1.0, 0.5, 0.0])
    rasa = np.array([1.0, 0.5, 1.0])
    s = local_scores(D, prop, rasa, 5.0)
    assert s[0] == pytest.approx(1.0 * 1.0 + 0.5 * 0.5)
    charge = np.array([1.0, -1.0, 0.0])
    c = local_scores(D, charge, rasa, 5.0)
    assert c[0] == pytest.approx(1.0 * 1.0 + (-1.0) * 0.5)
    assert c[1] == pytest.approx(1.0 * 1.0 + (-1.0) * 0.5)


def test_radii_exact():
    assert tuple(SOURCE_RADII_A) == (5.0, 10.0)


def test_region_masks_overlap():
    chain = ["H", "H", "L", "L"]
    is_cdr = np.array([True, False, True, False])
    m = region_masks(chain, is_cdr)
    assert m["VH"][0] and m["CDR"][0]
    assert m["FR"][1] and m["VH"][1]
    assert not (m["CDR"] & m["FR"]).any()
    assert m["VH"].sum() + m["VL"].sum() == 4


def test_stats_exact():
    v = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assert positive_sum_mean(v) == pytest.approx((0 + 0 + 0 + 1 + 2 + 3) / 6)
    assert top5_mean(v) == pytest.approx(np.mean([3, 2, 1, 0, -1]))
    assert top5_share_positive(v) == pytest.approx((3 + 2 + 1 + 0 + 0) / (1 + 2 + 3))
    sap = np.array([0.1, 0.2, 0.3, 0.0])
    assert positive_sum_mean(sap) == pytest.approx(float(sap.mean()))
    assert top5_share_positive(np.zeros(4)) == 0.0


def test_dimensions_and_parquets():
    assert len(SOURCE_SAP24_NAMES) == 24
    assert len(SOURCE_SCM24_NAMES) == 24
    assert len(SAP_GLOBAL6_NAMES) == 6 and len(SCM_GLOBAL6_NAMES) == 6
    assert len(SAP_EXTRA20_NAMES) == 20 and len(SCM_EXTRA20_NAMES) == 20
    assert len(SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES) == 30
    assert len(SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES + SAP_EXTRA20_NAMES) == 50
    assert (
        len(
            SOURCE_SAP24_NAMES
            + SAP_GLOBAL6_NAMES
            + SAP_EXTRA20_NAMES
            + SOURCE_SCM24_NAMES
            + SCM_GLOBAL6_NAMES
            + SCM_EXTRA20_NAMES
        )
        == 100
    )
    assert pd.read_parquet(FEAT / "antibody_source_sap24.parquet").shape == (324, 25)
    assert pd.read_parquet(FEAT / "antibody_source_scm24.parquet").shape == (324, 25)
    assert pd.read_parquet(FEAT / "antibody_source_sap_scm48.parquet").shape == (324, 49)
    assert pd.read_parquet(FEAT / "antibody_source_sap30.parquet").shape == (324, 31)
    assert pd.read_parquet(FEAT / "antibody_source_scm30.parquet").shape == (324, 31)
    assert pd.read_parquet(FEAT / "antibody_source_sap_scm60.parquet").shape == (324, 61)
    assert pd.read_parquet(FEAT / "antibody_source_sap50.parquet").shape == (324, 51)
    assert pd.read_parquet(FEAT / "antibody_source_scm50.parquet").shape == (324, 51)
    assert pd.read_parquet(FEAT / "antibody_source_sap_scm100.parquet").shape == (324, 101)
    res = pd.read_parquet(FEAT / "residue_source_sap_scm.parquet")
    assert len(res) == 75057
    assert res["SAP_R5"].min() >= -1e-9


def test_residue_reconstruction():
    wide = pd.read_parquet(FEAT / "antibody_source_sap_scm_wide.parquet")
    res = pd.read_parquet(FEAT / "residue_source_sap_scm.parquet")
    aid = str(wide["id"].iloc[10])
    recon = reconstruct_from_residue(res, aid)
    row = wide.set_index("id").loc[aid]
    for c in SOURCE_SAP24_NAMES + SOURCE_SCM24_NAMES + SAP_GLOBAL6_NAMES + SCM_EXTRA20_NAMES:
        assert abs(float(row[c]) - float(recon[c])) < 1e-8


def test_scm_signed_not_clipped_at_local():
    res = pd.read_parquet(FEAT / "residue_source_sap_scm.parquet")
    assert (res["SCM_R5"] < 0).any()
    assert (res["SCM_R10"] < 0).any()


def test_stage1_hides_test_and_shortlist_rule():
    freeze = yaml.safe_load((RES / "STAGE1_BLOCK_SHORTLIST_FREEZE.yaml").read_text())
    assert freeze["status"] == "STAGE1_BLOCK_SHORTLIST_FROZEN"
    ids = [x["block_id"] for x in freeze["shortlist"]]
    assert ids[:3] == ["B1_SAP24", "B2_SCM24", "B3_SAP24_SCM24"]
    assert len(ids) == 6
    assert freeze["rule"]["no_individual_feature_selection"] is True
    assert freeze["rule"]["no_public_private_selection"] is True
    s1 = pd.read_csv(RES / "STAGE1_VAL_BLOCK_SCREEN.csv")
    assert "TEST" not in "".join(s1.columns)


def test_specs_active():
    sap = yaml.safe_load((RES / "SOURCE_SAP24_FEATURE_SPEC.yaml").read_text())
    scm = yaml.safe_load((RES / "SOURCE_SCM24_FEATURE_SPEC.yaml").read_text())
    assert sap["status"] == "ACTIVE" and sap["dimensions"] == 24
    assert scm["status"] == "ACTIVE" and scm["dimensions"] == 24
    assert "SOURCE_DERIVED_INFERENCE" in scm["exposure"]


def test_audit_provenance_labels():
    text = (RES / "SOURCE_SPEC_AUDIT.md").read_text()
    assert "SOURCE_CONFIRMED" in text
    assert "SOURCE_DERIVED_INFERENCE" in text
    assert "(1/N)" in text or "mean(max" in text


def test_no_h114_in_promotion():
    text = (RES / "MAINLINE_PROMOTION_RECOMMENDATION.md").read_text()
    assert "DO NOT RUN" in text and "EXP-H114" in text
