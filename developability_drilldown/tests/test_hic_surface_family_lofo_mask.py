#!/usr/bin/env python3
"""Unit/smoke tests for F1_SURFACE family LOFO post-transform mask."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "models"))

from antibody_transformer.sham_f1_aux import (  # noqa: E402
    TOTAL_DIM,
    make_family_lofo_aux,
    make_surface_prospective_aux,
    BUNDLE_REAL,
)


def test_post_transform_mask_zeros_and_preserves():
    tax = json.loads((ROOT / "results/HIC_SURFACE_FAMILY_TAXONOMY_FROZEN.json").read_text())
    full = make_surface_prospective_aux(BUNDLE_REAL)
    ids = full.ids[:64]
    prep = full.fit(ids)
    X_full = full.transform(prep, ids)
    assert X_full.shape == (64, TOTAL_DIM)

    for fam, idxs in tax["family_indices_0based"].items():
        store = make_family_lofo_aux(fam, idxs)
        # Scaler pathway identical: fit via wrapped FULL
        prep_m = store.fit(ids)
        # prep hashes may differ by outer bundle label; unmasked transform must match FULL
        Xu = store.unmasked_transform(prep, ids)
        assert np.allclose(Xu, X_full, atol=0, rtol=0)

        Xm = store.transform(prep, ids)
        assert Xm.shape == X_full.shape
        assert np.allclose(Xm[:, idxs], 0.0)
        keep = [i for i in range(TOTAL_DIM) if i not in set(idxs)]
        assert np.allclose(Xm[:, keep], X_full[:, keep], atol=0, rtol=0)

        # same mask on a disjoint id set (val-like)
        ids2 = full.ids[64:96]
        X2f = full.transform(prep, ids2)
        X2m = store.transform(prep, ids2)
        assert np.allclose(X2m[:, idxs], 0.0)
        assert np.allclose(X2m[:, keep], X2f[:, keep], atol=0, rtol=0)


def test_mask_not_applied_before_scaler_would_differ_for_nonzero_mean():
    """Document that raw-zero-before-scale is NOT used; post-mask is exact zero."""
    tax = json.loads((ROOT / "results/HIC_SURFACE_FAMILY_TAXONOMY_FROZEN.json").read_text())
    fam = "ARO_EXPOSED_AMOUNT"
    idxs = tax["family_indices_0based"][fam]
    full = make_surface_prospective_aux(BUNDLE_REAL)
    ids = full.ids[:80]
    prep = full.fit(ids)
    store = make_family_lofo_aux(fam, idxs)
    Xm = store.transform(prep, ids)
    assert np.all(Xm[:, idxs] == 0.0)


if __name__ == "__main__":
    test_post_transform_mask_zeros_and_preserves()
    test_mask_not_applied_before_scaler_would_differ_for_nonzero_mean()
    print("OK_FAMILY_LOFO_MASK_SMOKE")
