#!/usr/bin/env python3
"""Tests for H102–H113 HSP promoted mainline wiring (no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_promoted_columns_match_audit():
    from antibody_transformer.hsp_promoted_aux import HSP_COLS, BUNDLE_IDS, HspPromotedAuxFeatureStore

    assert len(BUNDLE_IDS) == 6
    for bid in BUNDLE_IDS:
        store = HspPromotedAuxFeatureStore(bid)
        assert len(store.ids) == 324
        if "SURFACE_PLUS" in bid:
            assert store.effective_dim == 38
            sl = store.block_slices()
            assert "SURFACE" in sl and "HSP" in sl
        else:
            assert store.effective_dim == 3
            assert list(HSP_COLS) == ["HSP_MAX", "HSP_MEAN", "HSP_SUM"]


def test_canonical_parquets_match_hsp_source():
    import json

    meta = json.loads((ROOT / "results" / "H102_H113_HSP_CANONICAL_FEATURES.json").read_text())
    src = Path(__file__).resolve().parents[2] / "feature_research/hic_spatial_hydrophobicity/features/antibody_spatial_hydrophobicity.parquet"
    ab = pd.read_parquet(src)
    for pid, fam in meta["families"].items():
        pq = ROOT / fam["parquet"]
        df = pd.read_parquet(pq)
        assert len(df) == 324
        for c in fam["columns"]:
            assert (df[c].to_numpy() == ab.set_index("id").loc[df["id"].astype(str), c].to_numpy()).all()


def test_series_codes_contiguous():
    from preregister_h102_h113_hsp import SERIES

    assert [s["code"] for s in SERIES] == [f"EXP-H{i}" for i in range(102, 114)]
    assert all(s["hsp_bundle_id"] == "B3" for s in SERIES)
    assert all(len(s["hsp_columns"]) == 3 for s in SERIES)


def test_h054_h101_configs_untouched_marker():
    # sanity: H101 still STATIC_SAP
    import yaml

    cfg = yaml.safe_load((ROOT / "experiments/configs/EXP-H101.yaml").read_text())
    assert "STATIC_SAP" in cfg["fusion_bundle_id"]
