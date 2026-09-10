"""Historical single-model completeness freeze tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_codes import next_code  # noqa: E402
from _lib import N_CLASSICAL_REFINEMENT, N_EXPERIMENTS_TOTAL, PRESERVATION_SNAPSHOT_77  # noqa: E402


def test_existing_77_preservation():
    snap = pd.read_csv(PRESERVATION_SNAPSHOT_77)
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert len(exp) == N_EXPERIMENTS_TOTAL
    m = snap.merge(exp, on="experiment_code", suffixes=("_old", "_new"))
    assert len(m) == 77
    assert (m["experiment_id_old"] == m["experiment_id_new"]).all()
    assert (m["family_old"] == m["family_new"]).all()
    assert (m["target_old"] == m["target_new"]).all()
    for c in ("cv_primary_mae", "cv_shadow_mae", "public_mae", "private_mae", "test_overall_mae"):
        d = (
            pd.to_numeric(m[f"{c}_old"], errors="coerce")
            - pd.to_numeric(m[f"{c}_new"], errors="coerce")
        ).abs()
        assert float(d.max()) == 0.0


def test_audit_classification_complete():
    audit = pd.read_csv(ROOT / "results" / "HISTORICAL_SINGLE_MODEL_COMPLETENESS_AUDIT.csv")
    allowed = {
        "ALREADY_REGISTERED",
        "DUPLICATE_ALIAS",
        "MISSING_BACKFILLABLE_FULL",
        "MISSING_BACKFILLABLE_PARTIAL",
        "MISSING_SCORE_ONLY",
        "ENSEMBLE_EXCLUDED",
        "NONCOMPARABLE",
        "INCONSISTENT",
        "NOT_A_MODEL",
    }
    assert set(audit["classification"]).issubset(allowed)
    assert (audit["classification"] == "ALREADY_REGISTERED").sum() == 77
    assert (audit["classification"].str.startswith("MISSING")).sum() == 0
    assert (audit["classification"] == "INCONSISTENT").sum() == 0


def test_no_missing_comparable_backfill_needed():
    audit = pd.read_csv(ROOT / "results" / "HISTORICAL_SINGLE_MODEL_COMPLETENESS_AUDIT.csv")
    missing = audit[audit["classification"].str.startswith("MISSING", na=False)]
    assert len(missing) == 0


def test_classical_refinement_codes_issued():
    from _lib import is_classical_refinement_code

    codes = pd.read_csv(ROOT / "results" / "EXPERIMENT_CODES.csv")
    assert len(codes) == N_EXPERIMENTS_TOTAL
    assert next_code("TmApp") == "EXP-T130"
    assert next_code("HIC") == "EXP-H094"
    assert next_code("MULTI") == "EXP-M001"
    new = codes[codes["experiment_code"].map(is_classical_refinement_code)]
    assert len(new) == N_CLASSICAL_REFINEMENT


def test_ensemble_inventory_exists_and_excludes_from_registry():
    ens = pd.read_csv(ROOT / "results" / "HISTORICAL_ENSEMBLE_INVENTORY.csv")
    assert len(ens) >= 10
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    # no ensemble families in registry
    assert not exp["family"].astype(str).str.contains("ENSEMBLE", case=False).any()
    assert (exp["ensemble_type"].fillna("") == "").all()


def test_duplicate_alias_hist_winners():
    audit = pd.read_csv(ROOT / "results" / "HISTORICAL_SINGLE_MODEL_COMPLETENESS_AUDIT.csv")
    aliases = set(audit.loc[audit["classification"] == "DUPLICATE_ALIAS", "source_model_id"])
    # HIST winners are aliases to blends / context rows
    assert any("HIST__HIC" in a for a in aliases) or (
        audit["source_model_id"].astype(str).str.contains("HIST__HIC_PUBLIC_WINNER").any()
    )


def test_hic_esmf_svr_noncomparable_documented():
    audit = pd.read_csv(ROOT / "results" / "HISTORICAL_SINGLE_MODEL_COMPLETENESS_AUDIT.csv")
    row = audit[audit["source_model_id"] == "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt"]
    assert len(row) == 1
    assert row.iloc[0]["classification"] == "NONCOMPARABLE"
    assert abs(float(row.iloc[0]["private_mae"]) - 0.438341) < 1e-5


def test_bests_and_readiness_reports():
    bests = (ROOT / "results" / "HISTORICAL_SINGLE_MODEL_BESTS.md").read_text()
    ready = (ROOT / "results" / "NEW_SCIENCE_READINESS.md").read_text()
    assert "NEW_SCIENCE_READY = YES" in ready
    assert "Was a stronger historical single HIC model missing" in bests
    assert "**NO**" in bests
    assert "0.431815" in bests or "0.4318" in bests


def test_no_exp_m_and_catalog_updated():
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    assert len(exp) == N_EXPERIMENTS_TOTAL
    assert not exp["experiment_code"].astype(str).str.startswith("EXP-M").any()
    cat = (ROOT / "results" / "CATALOG.md").read_text()
    assert "EXP-T001" in cat and "EXP-H001" in cat
