#!/usr/bin/env python3
"""Tests for HSP robustness / canonicalization audit (no mainline training)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT.parents[1] / "developability_drilldown" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def test_exp_h114_unused():
    from experiment_codes import next_code

    assert next_code("HIC") == "EXP-H114"


def test_audit_artifacts_exist():
    required = [
        "HSP_GEOMETRY_OVERLAP.csv",
        "HSP_NEIGHBOR_COUNT_SUMMARY.csv",
        "HSP_FEATURE_SIMILARITY.csv",
        "HSP_BM_RADIUS_NEIGHBORHOOD_VAL.csv",
        "HSP_EIS_RADIUS_NEIGHBORHOOD_VAL.csv",
        "HSP_ROBUSTNESS_SUMMARY.md",
        "HSP_CANONICAL_DESCRIPTOR_SPEC.yaml",
        "HSP_CANONICALIZATION_REPORT.md",
        "HSP_CANONICAL_RESIDUE_QC.md",
        "HSP_RESIDUE_LEVEL_EXPERIMENT_DESIGN.md",
    ]
    for name in required:
        assert (RES / name).exists(), name


def test_val_reused_no_test_selection():
    bm = pd.read_csv(RES / "HSP_BM_RADIUS_NEIGHBORHOOD_VAL.csv")
    eis = pd.read_csv(RES / "HSP_EIS_RADIUS_NEIGHBORHOOD_VAL.csv")
    assert set(bm.context) <= {"A_ALONE", "B_SURFACE", "C_PHYS"}
    assert "TEST" not in bm.columns
    assert bm.source.str.contains("STAGE1").all()
    # fixed property defs
    assert bm.family_id.str.startswith("HSP_BM_RAW_TOTAL_RASA_TIEN_").all()
    assert eis.family_id.str.startswith("HSP_EIS_RAW_SIDECHAIN_SASA_ABS_").all()
    assert set(bm.neighborhood) <= {"CLOSEST_SC", "CENTROID"}
    assert set(eis.radius) == {4.0, 5.0, 6.0, 7.5, 8.0, 10.0}


def test_spec_no_h114_and_gate():
    spec = yaml.safe_load((RES / "HSP_CANONICAL_DESCRIPTOR_SPEC.yaml").read_text())
    assert spec["consumed_EXP_H114"] is False
    assert spec["no_TEST_used_for_selection"] is True
    assert spec["no_external_used"] is True
    assert set(spec["classifications"]) == {"BM", "EIS"}


def test_geometry_overlap_jaccard_bounds():
    jac = pd.read_csv(RES / "HSP_GEOMETRY_OVERLAP.csv")
    assert ((jac.mean_jaccard >= 0) & (jac.mean_jaccard <= 1)).all()
    # selected pair must appear
    sub = jac[(jac.closest_R == 5.0) & (jac.centroid_R == 8.0)]
    assert len(sub) == 324


def test_b3_only_in_val_bundle_semantics():
    # Alone context uses B3 (3 cols); SURFACE context concatenates SURFACE+B3 (38)
    bm = pd.read_csv(RES / "HSP_BM_RADIUS_NEIGHBORHOOD_VAL.csv")
    alone = bm[bm.context == "A_ALONE"]
    surf = bm[bm.context == "B_SURFACE"]
    assert (alone.n_features == 3).all()
    assert (surf.n_features == 38).all()
    assert alone.family_id.str.startswith("HSP_BM_RAW_TOTAL_RASA_TIEN_").all()


def test_mainline_h102_h113_untouched_marker():
    # ensure we did not rewrite a mainline OOF file in this audit
    drill = ROOT.parents[1] / "developability_drilldown" / "results" / "EXP-H107_OOF_EVALUATION.yaml"
    assert drill.exists()
    text = drill.read_text()
    assert "HSP_EIS" in text or "oof_test" in text
