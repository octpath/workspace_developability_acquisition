#!/usr/bin/env python3
"""Unit tests for STATIC_SAP_KD core definition (no HIC labels)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from antibody_transformer.static_sap_kd import (
    INCLUDE_SELF,
    KD_MAX,
    KD_MIN,
    KYTE_DOOLITTLE,
    R_REF,
    SAP3_COLS,
    SAP9_COLS,
    TIEN_MAXASA,
    aggregate_max_mean_sum,
    kd_norm,
    resolve_centroid,
    rsasa,
    sidechain_centroid,
    sskd_scores,
)


def test_kd_table_exact():
    assert KYTE_DOOLITTLE["I"] == 4.5
    assert KYTE_DOOLITTLE["R"] == -4.5
    assert KYTE_DOOLITTLE["G"] == -0.4
    assert len(KYTE_DOOLITTLE) == 20
    assert KD_MIN == -4.5
    assert KD_MAX == 4.5


def test_kd_norm_range_and_formula():
    for aa, v in KYTE_DOOLITTLE.items():
        n = kd_norm(aa)
        assert 0.0 <= n <= 1.0
        assert n == pytest.approx((v - KD_MIN) / (KD_MAX - KD_MIN))
    assert kd_norm("I") == 1.0
    assert kd_norm("R") == 0.0


def test_tien_maxasa_canonical():
    assert TIEN_MAXASA["G"] == 104.0
    assert TIEN_MAXASA["W"] == 285.0
    assert len(TIEN_MAXASA) == 20


def test_rsasa_clip():
    assert rsasa(52.0, "G") == pytest.approx(0.5)
    assert rsasa(1000.0, "G") == 1.0
    assert rsasa(-1.0, "G") == 0.0


def test_sidechain_centroid_and_gly():
    # fake Ala: CB only
    atoms = [
        ("N", np.array([0.0, 0.0, 0.0]), "N"),
        ("CA", np.array([1.0, 0.0, 0.0]), "C"),
        ("C", np.array([2.0, 0.0, 0.0]), "C"),
        ("O", np.array([2.5, 1.0, 0.0]), "O"),
        ("CB", np.array([1.0, 1.5, 0.0]), "C"),
    ]
    c = sidechain_centroid(atoms)
    assert c is not None
    np.testing.assert_allclose(c, [1.0, 1.5, 0.0])
    # Gly fallback
    ca = np.array([3.0, 4.0, 5.0])
    g_atoms = [("N", np.zeros(3), "N"), ("CA", ca, "C"), ("C", np.ones(3), "C"), ("O", np.ones(3), "O")]
    cent, prov = resolve_centroid("G", g_atoms, ca)
    assert prov == "gly_ca_fallback"
    np.testing.assert_allclose(cent, ca)


def test_sskd_formula_no_decay_includes_self():
    # 2 residues within R; identical KD_norm*rSASA
    centroids = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    kd_n = np.array([1.0, 0.5, 1.0])
    rasa = np.array([0.5, 1.0, 0.2])
    valid = np.array([True, True, True])
    scores = sskd_scores(centroids, kd_n, rasa, valid, radius=5.0, include_self=True)
    # residue 0 neighbors: 0 and 1
    assert scores[0] == pytest.approx(1.0 * 0.5 + 0.5 * 1.0)
    assert scores[1] == pytest.approx(1.0 * 0.5 + 0.5 * 1.0)
    # residue 2 only self
    assert scores[2] == pytest.approx(1.0 * 0.2)
    assert INCLUDE_SELF is True
    assert R_REF == 5.0


def test_aggregate_sap3():
    scores = np.array([1.0, 2.0, 3.0, np.nan])
    mx, mn, sm = aggregate_max_mean_sum(scores)
    assert mx == 3.0
    assert mn == pytest.approx(2.0)
    assert sm == 6.0
    assert SAP3_COLS == ["SSKD_ALL_MAX", "SSKD_ALL_MEAN", "SSKD_ALL_SUM"]
    assert len(SAP9_COLS) == 9


def test_no_plddt_in_formula_module():
    import inspect
    import antibody_transformer.static_sap_kd as m

    src = inspect.getsource(m.sskd_scores)
    assert "plddt" not in src.lower()
    assert "rbf" not in src.lower()
    assert "decay" not in src.lower()
