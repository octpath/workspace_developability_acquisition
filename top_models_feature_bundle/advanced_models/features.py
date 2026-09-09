#!/usr/bin/env python3
"""Fold-local fixed-length recipe feature preparation (bundle-local)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import BUNDLE_ROOT

sys.path.insert(0, str(BUNDLE_ROOT))
from bundle_simple_tvt import (  # noqa: E402
    FeatureAlignmentError,
    impute_apply,
    impute_fit,
    pca_block,
)
from reproduce_top_recipes import BLOCK_FILE, build_parts, load_block  # noqa: E402


def load_recipes(root: Path | None = None) -> pd.DataFrame:
    root = root or BUNDLE_ROOT
    return pd.read_csv(root / "recipes.csv")


def recipe_row(recipe_id: str) -> pd.Series:
    recipes = load_recipes()
    row = recipes[recipes["recipe_id"] == recipe_id]
    if len(row) != 1:
        raise FeatureAlignmentError(f"unknown recipe {recipe_id}")
    return row.iloc[0]


def build_recipe_parts(recipe_id: str, ids: list[str], root: Path | None = None) -> dict:
    root = root or BUNDLE_ROOT
    r = recipe_row(recipe_id)
    blocks = str(r["feature_blocks"]).split("|")
    parts = build_parts(blocks, ids, root, recipe_id)
    parts["recipe_id"] = recipe_id
    parts["target"] = str(r["target"])
    parts["regressor"] = str(r["regressor"])
    parts["blocks"] = blocks
    return parts


def _fit_transform_matrices(
    mats_fit: list[pd.DataFrame],
    mats_apply: list[list[pd.DataFrame]],
    *,
    do_pca: bool,
) -> list[np.ndarray]:
    """Impute (+ optional PCA per trailing Ablingua mats) + StandardScaler.

    For abl_blocks: mats = [recipe, always, extra] with PCA on always/extra only.
    For base_struct: mats = [base, struct] with PCA on struct only.
    For standalone: mats = [X] no PCA.
    """
    if not mats_fit:
        raise FeatureAlignmentError("empty mats")
    # impute each mat on fit
    meds = [impute_fit(m) for m in mats_fit]
    fit_imp = [impute_apply(m, med) for m, med in zip(mats_fit, meds)]
    apply_imps = [
        [impute_apply(m, med) for m, med in zip(group, meds)] for group in mats_apply
    ]

    if do_pca == "struct":
        # PCA only last matrix
        core_fit = fit_imp[0]
        struct_fit = fit_imp[1]
        struct_apps = [g[1] for g in apply_imps]
        struct_fit2, struct_apps2, _ = pca_block(struct_fit, struct_apps)
        fit_cat = pd.concat([core_fit, struct_fit2], axis=1)
        app_cats = [
            pd.concat([apply_imps[i][0], struct_apps2[i]], axis=1)
            for i in range(len(apply_imps))
        ]
    elif do_pca == "abl":
        recipe_fit = fit_imp[0]
        always_fit = fit_imp[1]
        extra_fit = fit_imp[2]
        always_apps = [g[1] for g in apply_imps]
        extra_apps = [g[2] for g in apply_imps]
        always_fit2, always_apps2, _ = pca_block(always_fit, always_apps)
        extra_fit2, extra_apps2, _ = pca_block(extra_fit, extra_apps)
        fit_cat = pd.concat([recipe_fit, always_fit2, extra_fit2], axis=1)
        app_cats = [
            pd.concat(
                [apply_imps[i][0], always_apps2[i], extra_apps2[i]],
                axis=1,
            )
            for i in range(len(apply_imps))
        ]
    else:
        fit_cat = pd.concat(fit_imp, axis=1)
        app_cats = [pd.concat(g, axis=1) for g in apply_imps]

    sc = StandardScaler()
    X_fit = sc.fit_transform(np.asarray(fit_cat, float))
    outs = [X_fit]
    for ac in app_cats:
        outs.append(sc.transform(np.asarray(ac, float)))
    return outs


def preprocess_parts(
    parts: dict,
    fit_ids: list[str],
    apply_groups: list[list[str]],
) -> list[np.ndarray]:
    """Fit preprocess on fit_ids; transform each apply group. Returns [X_fit, X_a0, ...]."""
    mode = parts["mode"]
    if mode == "standalone":
        mats_fit = [parts["X"].loc[fit_ids]]
        mats_apply = [[parts["X"].loc[g]] for g in apply_groups]
        return _fit_transform_matrices(mats_fit, mats_apply, do_pca=False)
    if mode == "base_struct":
        mats_fit = [parts["base"].loc[fit_ids], parts["struct"].loc[fit_ids]]
        mats_apply = [
            [parts["base"].loc[g], parts["struct"].loc[g]] for g in apply_groups
        ]
        return _fit_transform_matrices(mats_fit, mats_apply, do_pca="struct")
    if mode == "abl_blocks":
        mats_fit = [
            parts["recipe"].loc[fit_ids],
            parts["always"].loc[fit_ids],
            parts["extra"].loc[fit_ids],
        ]
        mats_apply = [
            [
                parts["recipe"].loc[g],
                parts["always"].loc[g],
                parts["extra"].loc[g],
            ]
            for g in apply_groups
        ]
        return _fit_transform_matrices(mats_fit, mats_apply, do_pca="abl")
    raise FeatureAlignmentError(f"unknown mode {mode}")


def raw_concat_parts(parts: dict, ids: list[str]) -> np.ndarray:
    """Unscaled concatenated features (for diagnostics)."""
    mode = parts["mode"]
    if mode == "standalone":
        return np.asarray(parts["X"].loc[ids], float)
    if mode == "base_struct":
        return np.asarray(
            pd.concat([parts["base"].loc[ids], parts["struct"].loc[ids]], axis=1),
            float,
        )
    return np.asarray(
        pd.concat(
            [
                parts["recipe"].loc[ids],
                parts["always"].loc[ids],
                parts["extra"].loc[ids],
            ],
            axis=1,
        ),
        float,
    )
