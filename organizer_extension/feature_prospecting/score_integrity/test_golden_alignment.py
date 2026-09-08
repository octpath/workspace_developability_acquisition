#!/usr/bin/env python3
"""Golden regression tests for canonical Simple TVT ID alignment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
sys.path.insert(0, str(ROOT / "organizer_extension/feature_prospecting/score_integrity"))
from canonical_simple_tvt import (  # noqa: E402
    FeatureAlignmentError,
    align_feature_block,
    concat_blocks,
    load_folds,
    run_standalone,
)


def _toy_block(ids, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(ids), 4)), index=ids, columns=[f"f{i}" for i in range(4)])


def test_a_shuffled_order_same_preds():
    ids = [f"ID{i:03d}" for i in range(40)]
    # fake folds 5-way
    folds = pd.DataFrame({"id": ids, "fold": [i % 5 for i in range(40)]})
    y = pd.Series(np.linspace(50, 80, 40), index=ids)
    X = _toy_block(ids, 1)
    X2 = X.sample(frac=1.0, random_state=0)
    a = align_feature_block(X, ids, "X")
    b = align_feature_block(X2, ids, "Xshuf")
    assert np.allclose(a.to_numpy(), b.to_numpy())
    r1 = run_standalone(a, y, folds, ids, dim_mode="raw")
    r2 = run_standalone(b, y, folds, ids, dim_mode="raw")
    assert abs(r1["mae"] - r2["mae"]) < 1e-12
    assert np.allclose(r1["oof"].to_numpy(), r2["oof"].to_numpy())
    print("TEST A OK")


def test_b_rangeindex_hard_fail_or_align():
    ids = [f"ID{i:03d}" for i in range(10)]
    # RangeIndex numeric features — must fail when aligning to ADI-like ids
    X = pd.DataFrame(np.eye(10), columns=[f"c{i}" for i in range(10)])  # index 0..9
    try:
        align_feature_block(X, ids, "range")
        raise AssertionError("should have failed")
    except FeatureAlignmentError as e:
        assert "missing" in str(e).lower() or "zero" in str(e).lower() or "Missing" in str(e)
    print("TEST B OK")


def test_c_all_nan_hard_fail():
    ids = [f"ID{i:03d}" for i in range(10)]
    X = pd.DataFrame(np.nan, index=ids, columns=["a", "b"])
    try:
        align_feature_block(X, ids, "nanblock")
        raise AssertionError("should have failed")
    except FeatureAlignmentError as e:
        assert "NaN" in str(e) or "nan" in str(e).lower() or "finite" in str(e).lower()
    print("TEST C OK")


def test_d_missing_ids_hard_fail():
    ids = [f"ID{i:03d}" for i in range(10)]
    X = _toy_block(ids[:8])
    try:
        align_feature_block(X, ids, "miss")
        raise AssertionError("should have failed")
    except FeatureAlignmentError:
        pass
    # intersection mode OK
    Xi = align_feature_block(X, ids, "miss_ok", allow_intersection=True)
    assert len(Xi) == 8
    print("TEST D OK")


def test_e_frozen_recipe_repro():
    # Use real AbLingua parent if available
    sys.path.insert(0, str(ROOT / "organizer_extension/feature_prospecting/ablingua600m/scripts"))
    import runpy
    import importlib.util

    ev = runpy.run_path(
        str(ROOT / "organizer_extension/feature_prospecting/ablingua600m/scripts/04_guided_pooling_tvt.py")
    )
    INTERIM = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit"
    spec = importlib.util.spec_from_file_location(
        "tvt", INTERIM / "scripts/run_simple_tvt_rescreen.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ids = [str(x) for x in pd.read_csv(ROOT / "competition/data/distribution/dev.csv")["id"]]
    recipe, HL = ev["load_recipe_and_global"](ids)
    y = mod.load_y("TmApp")
    primary, shadow = load_folds()
    recipe = align_feature_block(recipe, ids, "recipe")
    HL = align_feature_block(HL, ids, "HL")
    from canonical_simple_tvt import run_base_plus_struct

    out_p = run_base_plus_struct(recipe, HL, y, primary, ids, mode="FREE_ALPHA", force_pca_struct=True)
    out_s = run_base_plus_struct(recipe, HL, y, shadow, ids, mode="FREE_ALPHA", force_pca_struct=True)
    assert abs(out_p["plus_mae"] - 2.7466001170280285) < 1e-6
    assert abs(out_s["plus_mae"] - 2.822725206703513) < 1e-6
    # second run identical
    out_p2 = run_base_plus_struct(recipe, HL, y, primary, ids, mode="FREE_ALPHA", force_pca_struct=True)
    assert out_p["prediction_hash"] == out_p2["prediction_hash"]
    print("TEST E OK", out_p["plus_mae"], out_s["plus_mae"])


def test_f_block_order_invariant():
    ids = [f"ID{i:03d}" for i in range(40)]
    folds = pd.DataFrame({"id": ids, "fold": [i % 5 for i in range(40)]})
    y = pd.Series(np.linspace(50, 80, 40), index=ids)
    A = _toy_block(ids, 1)
    B = _toy_block(ids, 2)
    X1, _ = concat_blocks([A, B], ids, ["A", "B"])
    X2, _ = concat_blocks([B, A], ids, ["B", "A"])
    # After proper unique prefixes, column order differs but predictions for standalone
    # on full concat should match if we sort columns — require alignment by name set
    r1 = run_standalone(X1[sorted(X1.columns)], y, folds, ids)
    r2 = run_standalone(X2[sorted(X2.columns)], y, folds, ids)
    assert abs(r1["mae"] - r2["mae"]) < 1e-12
    print("TEST F OK")


def test_g_lasso_alignment_and_no_pca():
    from canonical_simple_tvt import run_standalone_lasso

    ids = [f"ID{i:03d}" for i in range(40)]
    folds = pd.DataFrame({"id": ids, "fold": [i % 5 for i in range(40)]})
    y = pd.Series(np.linspace(50, 80, 40), index=ids)
    X = _toy_block(ids, 7)
    Xs = X.sample(frac=1.0, random_state=1)
    a = align_feature_block(X, ids, "L1")
    b = align_feature_block(Xs, ids, "L2")
    r1 = run_standalone_lasso(a, y, folds, ids)
    r2 = run_standalone_lasso(b, y, folds, ids)
    assert abs(r1["mae"] - r2["mae"]) < 1e-10
    assert "no_PCA" in r1["preprocessing"]
    assert r1["raw_dim"] == 4
    print("TEST G LASSO OK", r1["mae"], r1["alpha_median"])


if __name__ == "__main__":
    test_a_shuffled_order_same_preds()
    test_b_rangeindex_hard_fail_or_align()
    test_c_all_nan_hard_fail()
    test_d_missing_ids_hard_fail()
    test_f_block_order_invariant()
    test_e_frozen_recipe_repro()
    test_g_lasso_alignment_and_no_pca()
    print("ALL GOLDEN TESTS PASSED")
