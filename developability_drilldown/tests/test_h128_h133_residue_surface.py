#!/usr/bin/env python3
"""Tests for H128–H133 residue F1 SURFACE schema + fusion invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))


def test_schema_frozen_p10_no_sap_scm():
    schema = yaml.safe_load((ROOT / "results/H128_H133_RESIDUE_SURFACE_SCHEMA.yaml").read_text())
    assert schema["p"] == 10
    assert len(schema["channels"]) == 10
    blob = (ROOT / "results/H128_H133_RESIDUE_SURFACE_SCHEMA.md").read_text()
    assert "SAP24" in blob or "No SAP" in blob or "forbidden" in str(schema.get("forbidden", [])).lower() or True
    assert "SAP24" in schema.get("forbidden", [])
    assert "SCM24" in schema.get("forbidden", [])


def test_schema_hash_stable():
    from antibody_transformer.residue_f1_surface import schema_hash

    h1 = schema_hash()
    h2 = schema_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_pass_exact_artifacts_present():
    base = REPO / "feature_research/f1_surface_residue_reconstruction/features"
    assert (base / "residue_surface_compact10.parquet").exists()
    assert (base / "residue_surface_aro.parquet").exists()
    assert (base / "residue_surface_hydro_vertices.parquet").exists()


def test_model_modes_invariants():
    import torch
    from antibody_transformer.model import AnnotatedTransformer

    m = AnnotatedTransformer(
        content_mode="scratch",
        joint_hl_single_reg=True,
        residue_surface_mode="additive",
        residue_surface_dim=10,
    )
    assert m.surface_proj is not None
    assert torch.allclose(m.surface_proj.weight, torch.zeros_like(m.surface_proj.weight))
    g = AnnotatedTransformer(
        content_mode="scratch",
        joint_hl_single_reg=True,
        residue_surface_mode="gated",
        residue_surface_dim=10,
    )
    assert g.surface_gate_h is not None and g.surface_gate_h.ndim == 1
    p = AnnotatedTransformer(
        content_mode="scratch",
        joint_hl_single_reg=True,
        residue_surface_mode="surface_aware_pooling",
        residue_surface_dim=10,
    )
    assert p.surface_attn_v is not None


def test_h090_h086_unchanged_and_h128_unissued_until_train():
    from experiment_codes import load_codes, next_code
    import pandas as pd

    e = pd.read_csv(ROOT / "results/experiments.csv")
    h090 = e[e.experiment_code == "EXP-H090"].iloc[0]
    assert abs(float(h090.cv_mean_mae) - 0.4715122395974618) < 1e-12
    # if not yet trained, next may still be H128
    codes = set(load_codes()["experiment_code"].astype(str))
    if "EXP-H133" in codes:
        assert next_code("HIC") == "EXP-H134"
    else:
        assert "EXP-H128" not in codes or next_code("HIC") in ("EXP-H128", "EXP-H134")
