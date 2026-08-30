#!/usr/bin/env python3
"""Unit tests for RASA, MaxASA freeze, splits, and basic invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b1_common import DATA, MAX_ASA_TIEN2013, SPLITS, TARGET_COLS, clean_aa  # noqa: E402


def test_maxasa_frozen():
    assert MAX_ASA_TIEN2013["A"] == 129.0
    assert MAX_ASA_TIEN2013["W"] == 285.0
    assert len(MAX_ASA_TIEN2013) == 20


def test_rasa_definition():
    # synthetic: SASA=64.5 for Ala → RASA = 64.5/129
    sasa = 64.5
    rasa = sasa / MAX_ASA_TIEN2013["A"]
    assert abs(rasa - 0.5) < 1e-9
    # do not clip >1
    rasa2 = 200.0 / MAX_ASA_TIEN2013["A"]
    assert rasa2 > 1.0


def test_unique_ids_and_counts():
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    assert df["antibody_id"].is_unique
    assert len(df) == 400
    assert df["psr_score"].notna().sum() == 398
    assert df["hic_rt_min"].notna().sum() == 348
    assert df["tm_app_C"].notna().sum() == 346
    assert len(pd.read_csv(DATA / "triple_core.csv")) == 324


def test_seq_validity():
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    for col in ["heavy", "light"]:
        assert df[col].map(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY")).all()


def test_split_disjointness():
    for split in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
        s = pd.read_csv(SPLITS / f"triple_{split}.csv")
        for a, b in [("Dev", "Public"), ("Dev", "Private"), ("Public", "Private")]:
            ga = set(s.loc[s.role == a, "cluster_id"])
            gb = set(s.loc[s.role == b, "cluster_id"])
            assert ga.isdisjoint(gb), f"{split} {a}/{b} overlap"


def test_numbering_concat():
    num = pd.read_csv(DATA / "numbering_germline.csv")
    assert num["H_concat_match"].mean() == 1.0
    assert num["L_concat_match"].mean() == 1.0


if __name__ == "__main__":
    test_maxasa_frozen()
    test_rasa_definition()
    test_unique_ids_and_counts()
    test_seq_validity()
    test_split_disjointness()
    test_numbering_concat()
    print("ALL_TESTS_OK")
