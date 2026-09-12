#!/usr/bin/env python3
"""Tests for F1_SURFACE residue-level provenance reconstruction (no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

ARO_COLS = [
    "aro_exposed_TYR_count",
    "aro_exposed_TRP_count",
    "aro_exposed_PHE_count",
    "aro_exposed_aromatic_total_count",
    "aro_aromatic_exposed_SASA_total",
    "aro_aromatic_exposed_SASA_fraction",
    "aro_strongly_exposed_aromatic_count",
    "aro_strongly_exposed_aromatic_SASA",
    "aro_CDR_exposed_aromatic_count",
    "aro_CDR_aromatic_SASA",
    "aro_CDR_aromatic_fraction",
    "aro_aromatic_patch_count",
    "aro_largest_aromatic_patch_n_res",
    "aro_largest_aromatic_patch_exposed_SASA",
    "aro_max_local_aromatic_SASA",
    "aro_sequence_aromatic_count",
    "aro_sequence_TYR_count",
    "aro_sequence_TRP_count",
    "aro_sequence_PHE_count",
]
HYDRO_COLS = [
    "mean_H_surface",
    "q75_H_surface",
    "q90_H_surface",
    "q95_H_surface",
    "max_H_surface",
    "positive_H_area_fraction",
    "top10_H_mean",
    "top10_H_area_fraction",
    "high_H_patch_count",
    "largest_high_H_patch_area_fraction",
    "largest_high_H_patch_n_vertices",
    "CDR_mean_H",
    "CDR_q90_H",
    "CDR_high_H_area_fraction",
    "n_surface_points",
    "phi_finite_frac",
]
F1_COLS = ARO_COLS + HYDRO_COLS


def test_h128_h133_issued_next_is_h134():
    from experiment_codes import load_codes, next_code

    codes = set(load_codes()["experiment_code"].astype(str))
    for c in ("EXP-H128", "EXP-H129", "EXP-H130", "EXP-H131", "EXP-H132", "EXP-H133"):
        assert c in codes
    assert next_code("HIC") == "EXP-H134"


def test_historical_f1_dims_and_order():
    h047 = pd.read_parquet(ROOT / "experiments/features/EXP-H047.parquet")
    feat = [c for c in h047.columns if c not in ("id", "split")]
    assert feat[1395:1430] == F1_COLS
    assert len(ARO_COLS) == 19
    assert len(HYDRO_COLS) == 16
    assert len(F1_COLS) == 35


def test_h090_h086_consume_f1_surface():
    for code in ("EXP-H090", "EXP-H086"):
        cfg = yaml.safe_load((ROOT / f"experiments/configs/{code}.yaml").read_text())
        assert cfg["fusion_bundle_id"] == "F1_SURFACE"
        assert cfg["fusion_mode"] == "late_concat_aux32"


def test_no_forbidden_substitution_in_recon_script():
    script = (
        REPO
        / "feature_research/f1_surface_residue_reconstruction/scripts/reconstruct_f1_residue_provenance.py"
    ).read_text()
    # must not invert aggregates; must not pull SAP/SCM families
    assert "aggregate→residue" not in script
    assert "SOURCE_SAP" not in script
    assert "SCM24" not in script
    assert "originating_atom_during_surface_construction" in script


@pytest.mark.skipif(
    not (REPO / "feature_research/f1_surface_residue_reconstruction/features/residue_surface_aro.parquet").exists(),
    reason="reconstruction not run yet",
)
def test_reconstruction_roundtrip_and_mapping():
    summary = yaml.safe_load(
        (ROOT / "results/F1_SURFACE_RESIDUE_RECONSTRUCTION_AUDIT.yaml").read_text()
    )
    # Audit snapshot fields record reconstruction-time state (H128 then unissued).
    assert "no_experiment_issued" in summary
    assert summary["verdict"] in ("PASS-EXACT", "PASS-NUMERIC", "PARTIAL", "FAIL")
    assert summary["n_antibodies_reconstructed"] == 324
    assert summary["verdict"] == "PASS-EXACT"

    aro = pd.read_parquet(
        REPO / "feature_research/f1_surface_residue_reconstruction/features/residue_surface_aro.parquet"
    )
    assert {"antibody_id", "chain", "sequence_index", "pdb_resseq", "sasa", "rasa"}.issubset(aro.columns)
    assert aro["chain"].isin(["H", "L"]).all()
    # no silent empty truncation of ids
    assert aro["antibody_id"].nunique() == 324

    vert = pd.read_parquet(
        REPO
        / "feature_research/f1_surface_residue_reconstruction/features/residue_surface_hydro_vertices.parquet"
    )
    assert vert["assignment_method"].nunique() == 1
    assert vert["assignment_method"].iloc[0] == "originating_atom_during_surface_construction"
    assert {"chain", "sequence_index", "source_atom_name"}.issubset(vert.columns)

    err = pd.read_parquet(
        REPO / "feature_research/f1_surface_residue_reconstruction/features/roundtrip_errors_long.parquet"
    )
    assert set(err["column"]) == set(F1_COLS)
    if summary["verdict"] in ("PASS-EXACT", "PASS-NUMERIC"):
        assert float(err["abs_err"].max()) <= 1e-3

    # historical transformer results unchanged marker
    e = pd.read_csv(ROOT / "results/experiments.csv")
    h090 = e[e.experiment_code == "EXP-H090"].iloc[0]
    assert abs(float(h090.cv_mean_mae) - 0.4715122395974618) < 1e-12


def test_insertion_code_column_present_when_reconstructed():
    p = REPO / "feature_research/f1_surface_residue_reconstruction/features/residue_surface_aro.parquet"
    if not p.exists():
        pytest.skip("not reconstructed")
    aro = pd.read_parquet(p)
    assert "pdb_icode" in aro.columns
    # structure_utils uses contiguous seq index; icode column exists for audit
    vert = pd.read_parquet(
        REPO
        / "feature_research/f1_surface_residue_reconstruction/features/residue_surface_hydro_vertices.parquet"
    )
    assert "pdb_icode" in vert.columns
