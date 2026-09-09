"""Unit tests for Top-3 ensemble quickcheck."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BUNDLE = Path(os.environ.get("TOP3_QC_BUNDLE", ROOT.parent.parent / "top_models_feature_bundle"))
OUT = Path(os.environ.get("TOP3_QC_OUT", ROOT))

from core import (  # noqa: E402
    MODELS_BY_TARGET,
    all_nonempty_subsets,
    convex_mae_weights,
    equal_mean_pred,
    ensure_unique_ids,
    median3_pred,
    sha_arr,
)


def test_seven_subsets_exact():
    for models in MODELS_BY_TARGET.values():
        subs = all_nonempty_subsets(models)
        assert len(subs) == 7
        assert len(set(subs)) == 7


def test_equal_mean_correct():
    ids = ["a", "b", "c"]
    pred_map = {
        "A": pd.Series([1.0, 2.0, 3.0], index=ids),
        "B": pd.Series([3.0, 4.0, 5.0], index=ids),
    }
    out = equal_mean_pred(pred_map, ("A", "B"), ids)
    np.testing.assert_allclose(out.to_numpy(), [2.0, 3.0, 4.0])


def test_median3_correct():
    ids = ["a", "b"]
    pred_map = {
        "A": pd.Series([1.0, 10.0], index=ids),
        "B": pd.Series([2.0, 20.0], index=ids),
        "C": pd.Series([100.0, 30.0], index=ids),
    }
    out = median3_pred(pred_map, ["A", "B", "C"], ids)
    np.testing.assert_allclose(out.to_numpy(), [2.0, 20.0])


def test_convex_weights_constraints():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(40, 3))
    true_w = np.array([0.5, 0.3, 0.2])
    y = P @ true_w + rng.normal(scale=0.01, size=40)
    w = convex_mae_weights(P, y)
    assert (w >= -1e-10).all()
    assert abs(w.sum() - 1.0) < 1e-8


def test_duplicate_id_hard_fail():
    with pytest.raises(ValueError):
        ensure_unique_ids(["a", "a", "b"])


def test_missing_base_prediction_hard_fail():
    ids = ["a", "b"]
    pred_map = {"A": pd.Series([1.0, np.nan], index=ids)}
    with pytest.raises(RuntimeError):
        equal_mean_pred(pred_map, ("A",), ids)


def test_id_shuffle_invariance_equal_mean():
    ids = [f"id{i}" for i in range(10)]
    rng = np.random.default_rng(1)
    pred_map = {
        "A": pd.Series(rng.normal(size=10), index=ids),
        "B": pd.Series(rng.normal(size=10), index=ids),
    }
    order1 = ids
    order2 = list(reversed(ids))
    p1 = equal_mean_pred(pred_map, ("A", "B"), order1).reindex(ids)
    p2 = equal_mean_pred(pred_map, ("A", "B"), order2).reindex(ids)
    np.testing.assert_allclose(p1, p2)


def test_base_reproduction_audit_exists_and_pass():
    path = OUT / "BASE_REPRODUCTION_AUDIT.csv"
    if not path.exists():
        pytest.skip("run_all not executed yet")
    df = pd.read_csv(path)
    assert len(df) == 6
    assert bool(df["pass"].all())
    assert (df["max_difference"] <= 1e-8).all()


def test_final_test_n162():
    path = OUT / "final_predictions" / "final_ensemble_submission.csv"
    if not path.exists():
        pytest.skip("run_all not executed yet")
    df = pd.read_csv(path)
    assert len(df) == 162
    assert list(df.columns) == ["id", "TmApp", "HIC"]


def test_stacker_cache_omits_y_test():
    cache = OUT / "strict_meta_cache"
    if not (cache / "INDEX.json").exists():
        pytest.skip("cache missing")
    # fold CSVs for test must not contain y
    for p in cache.glob("*__test.csv"):
        cols = pd.read_csv(p, nrows=1).columns.tolist()
        assert "y" not in cols


def test_solution_not_in_commit_outputs():
    # ensure we did not copy solution into outdir
    assert not (OUT / "solution.csv").exists()


def test_no_bundle_recipe_mutation_marker():
    # recipes still readable and unchanged conceptually
    recipes = pd.read_csv(BUNDLE / "recipes.csv")
    assert len(recipes) == 6


def test_equal_mean_csv_has_seven_per_target():
    path = OUT / "EQUAL_MEAN_ALL_SUBSETS.csv"
    if not path.exists():
        pytest.skip("missing")
    df = pd.read_csv(path)
    for t in ("TmApp", "HIC"):
        assert len(df[df.target == t]) == 7
