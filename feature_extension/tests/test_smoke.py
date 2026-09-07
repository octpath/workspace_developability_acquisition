#!/usr/bin/env python3
"""Smoke tests for feature_extension v1 (no full-cohort extraction)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

FE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FE.parent))


def test_imports():
    from feature_extension.extractors import (
        extract_aromatic,
        extract_interface,
        extract_sasa,
        extract_surface_patch,
    )

    assert callable(extract_sasa)
    assert callable(extract_aromatic)
    assert callable(extract_surface_patch)
    assert callable(extract_interface)


def test_extractors_one_pdb():
    from feature_extension.extractors import extract_aromatic, extract_sasa

    pdbs = sorted((FE / "data/esmfold_fv").glob("*.pdb"))
    assert pdbs, "bundled Fv PDBs required"
    pdb = pdbs[0]
    sasa = extract_sasa(pdb)
    arom = extract_aromatic(pdb)
    assert sasa["sasa_total"] > 0
    assert "exposed_aromatic_total_count" in arom
    assert all(v != -999 for v in sasa.values())
    assert all(v != -999 for v in arom.values())


def test_parquets():
    paths = list((FE / "data/precomputed_features").glob("*.parquet"))
    paths.append(FE / "data/bioemu_isolated/features.parquet")
    assert paths
    for p in paths:
        if not p.is_file():
            continue
        df = pd.read_parquet(p)
        assert "id" in df.columns
        assert df["id"].is_unique


def test_folds():
    folds = pd.read_csv(FE / "folds.csv")
    assert {"id", "fold_primary", "fold_shadow"} <= set(folds.columns)
    assert folds["id"].is_unique
    assert set(folds["fold_primary"].unique()) <= set(range(5))
    assert set(folds["fold_shadow"].unique()) <= set(range(5))
    assert len(folds) == 162


def test_examples_importable():
    # ensure example files exist and reference package name
    for name in (
        "example_extract_features.py",
        "example_join_precomputed.py",
        "example_simple_tvt_cv.py",
    ):
        p = FE / "examples" / name
        assert p.is_file()
        text = p.read_text(encoding="utf-8")
        assert "feature_extension" in text or "folds" in text or "parquet" in text
