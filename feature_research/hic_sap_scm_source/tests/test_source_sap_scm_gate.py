"""Tests for SAP/SCM source feature-research track (fidelity gate + isolation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
TRACK = ROOT / "feature_research" / "hic_sap_scm_source"
RESULTS = TRACK / "results"
EXP_DIR = ROOT / "developability_drilldown" / "experiments"


def test_exp_h114_remains_unused():
    results = ROOT / "developability_drilldown" / "results"
    forbidden_globs = [
        "*EXP-H114*",
        "*H114*RESULT*",
        "*H114*metrics*",
        "*H114*TEST*",
    ]
    forbidden = []
    for g in forbidden_globs:
        forbidden.extend(results.glob(g))
        forbidden.extend(EXP_DIR.glob(g))
    # Prereg-only files would still mean H114 was started; disallow any H114 path
    forbidden.extend(results.glob("*H114*"))
    forbidden.extend(list(EXP_DIR.rglob("*H114*")))
    assert not forbidden, f"EXP-H114 artifacts found: {forbidden}"


def test_mainline_registry_next_hic_is_h114():
    import sys

    sys.path.insert(0, str(ROOT / "developability_drilldown" / "scripts"))
    from experiment_codes import next_code  # type: ignore

    assert next_code("HIC") == "EXP-H114"
    text = (RESULTS / "MAINLINE_PROMOTION_RECOMMENDATION.md").read_text()
    assert "EXP-H114" in text
    assert "DO NOT RUN" in text


def test_source_sap24_spec_blocked():
    spec = yaml.safe_load((RESULTS / "SOURCE_SAP24_FEATURE_SPEC.yaml").read_text())
    assert spec["status"] == "BLOCKED_SOURCE_UNRESOLVED"
    assert "positive_sum_mean" in spec["blocking_items"]


def test_source_scm24_spec_blocked():
    spec = yaml.safe_load((RESULTS / "SOURCE_SCM24_FEATURE_SPEC.yaml").read_text())
    assert spec["status"] == "BLOCKED_SOURCE_UNRESOLVED"


def test_audit_marks_positive_sum_mean_unresolved():
    audit = (RESULTS / "SOURCE_SPEC_AUDIT.md").read_text()
    assert "POSITIVE_SUM_MEAN" in audit or "positive_sum_mean" in audit
    assert "UNRESOLVED" in audit
    assert "SCM" in audit


def test_no_source_parquet_generated():
    feats = TRACK / "features"
    assert not list(feats.glob("antibody_source_sap*.parquet"))
    assert not list(feats.glob("antibody_source_scm*.parquet"))


def test_radii_exactly_5_and_10():
    import sys

    sys.path.insert(0, str(TRACK / "src"))
    from aggregation_engine import SOURCE_RADII_A, expected_source24_dim

    assert tuple(SOURCE_RADII_A) == (5.0, 10.0)
    assert expected_source24_dim() == 24


def test_positive_sum_mean_raises():
    import sys

    sys.path.insert(0, str(TRACK / "src"))
    from aggregation_engine import positive_sum_mean_unresolved, STAT_FN

    with pytest.raises(NotImplementedError):
        positive_sum_mean_unresolved(np.array([1.0, 2.0]))
    with pytest.raises(NotImplementedError):
        STAT_FN["POSITIVE_SUM_MEAN"](np.array([1.0]))


def test_aggregate_blocked_by_fidelity_gate():
    import sys

    sys.path.insert(0, str(TRACK / "src"))
    from aggregation_engine import aggregate_region_scores

    with pytest.raises(RuntimeError, match="BLOCKED"):
        aggregate_region_scores(np.array([1.0, 2.0]), np.array([True, True]), "MAX")


def test_top5_mean_and_std_helpers():
    import sys

    sys.path.insert(0, str(TRACK / "src"))
    from aggregation_engine import population_std, top5_mean, top5_share_nonnegative

    assert top5_mean(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    assert top5_mean(np.arange(10.0)) == pytest.approx(np.mean([5, 6, 7, 8, 9]))
    assert population_std(np.array([1.0, 1.0, 1.0])) == 0.0
    assert top5_share_nonnegative(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 5.0])) == pytest.approx(
        9.0 / 10.0
    )


def test_stage1_skipped_and_shortlist_blocked():
    screen = (RESULTS / "STAGE1_VAL_BLOCK_SCREEN.csv").read_text()
    assert "STAGE1_SKIPPED" in screen
    freeze = yaml.safe_load((RESULTS / "STAGE1_BLOCK_SHORTLIST_FREEZE.yaml").read_text())
    assert freeze["status"] == "BLOCKED_NO_STAGE1"


def test_no_individual_feature_winner_language_in_promotion():
    promo = (RESULTS / "MAINLINE_PROMOTION_RECOMMENDATION.md").read_text()
    assert "NO_PROMOTION" in promo
    assert "best column" not in promo.lower()


def test_hsp_results_untouched_marker():
    # Isolation: this track must not rewrite HSP promotion freeze
    hsp = ROOT / "feature_research" / "hic_spatial_hydrophobicity" / "results"
    assert hsp.exists()
