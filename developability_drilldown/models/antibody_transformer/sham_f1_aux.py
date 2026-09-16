#!/usr/bin/env python3
"""Prospective SURFACE aux stores: SHAM35 vs REAL_F1_SURFACE35.

Duck-typed like H047AuxFeatureStore for protocol_v3_ext late fusion.
SHAM uses the same two-block pathway (ARO19 + HYDRO16) with all zeros.
REAL loads canonical F1_SURFACE from EXP-H047.parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .h047_aux_features import (
    FoldPreprocessor,
    H047AuxFeatureStore,
    H047_PARQUET,
    _impute_apply,
    _impute_fit,
    _sha,
)

ROOT = Path(__file__).resolve().parents[2]

BUNDLE_SHAM = "SHAM35"
BUNDLE_REAL = "REAL_F1_SURFACE35"
BUNDLE_ARO_ONLY = "ARO_ONLY35"
BUNDLE_HYDRO_ONLY = "HYDRO_ONLY35"
ARO_DIM = 19
HYDRO_DIM = 16
TOTAL_DIM = ARO_DIM + HYDRO_DIM  # 35


def _fit_two_block(store, train_ids: list[str], *, expect_zero_blocks: set[str] | None = None) -> FoldPreprocessor:
    Xb = store._slice_blocks(train_ids)
    medians: dict[str, np.ndarray] = {}
    scalers: dict[str, StandardScaler] = {}
    for name in store.block_order:
        X = Xb[name]
        med = _impute_fit(X)
        medians[name] = med
        Xi = _impute_apply(X, med)
        sc = StandardScaler()
        sc.fit(Xi)
        if not np.all(np.isfinite(sc.scale_)) or not np.all(sc.scale_ > 0):
            raise RuntimeError(f"{store.bundle_id} StandardScaler unsafe for block {name}: scale_={sc.scale_}")
        scalers[name] = sc
    prep = FoldPreprocessor(
        medians=medians,
        scalers=scalers,
        pca=None,
        block_order=list(store.block_order),
        effective_dim=store.effective_dim,
        train_ids=list(train_ids),
        hash="",
    )
    prep.hash = _sha(
        {
            "bundle": store.bundle_id,
            "train_ids": train_ids,
            "artifact_hash": store.artifact_hash,
        }
    )
    Xt = prep.transform(store._slice_blocks(train_ids))
    if Xt.shape != (len(train_ids), TOTAL_DIM):
        raise RuntimeError(f"{store.bundle_id} transform shape {Xt.shape}")
    if expect_zero_blocks:
        # After scaling zeros remain zeros; real blocks should not be all-zero
        aro = Xt[:, :ARO_DIM]
        hydro = Xt[:, ARO_DIM:]
        if "AROMATIC_TOPO" in expect_zero_blocks and not np.allclose(aro, 0.0):
            raise RuntimeError(f"{store.bundle_id}: ARO slots expected zero after scale")
        if "HYDRO_FIELD" in expect_zero_blocks and not np.allclose(hydro, 0.0):
            raise RuntimeError(f"{store.bundle_id}: HYDRO slots expected zero after scale")
        if "AROMATIC_TOPO" not in expect_zero_blocks and np.allclose(aro, 0.0):
            raise RuntimeError(f"{store.bundle_id}: ARO slots unexpectedly all zero")
        if "HYDRO_FIELD" not in expect_zero_blocks and np.allclose(hydro, 0.0):
            raise RuntimeError(f"{store.bundle_id}: HYDRO slots unexpectedly all zero")
    return prep


class Sham35AuxFeatureStore:
    """35D constant-zero aux with F1-matched two-block preprocessing pathway."""

    def __init__(self, ids: Optional[list[str]] = None):
        self.bundle_id = BUNDLE_SHAM
        if ids is None:
            df = pd.read_parquet(H047_PARQUET)
            ids = df["id"].astype(str).tolist()
        self.ids = list(ids)
        self.id_to_idx = {a: i for i, a in enumerate(self.ids)}
        n = len(self.ids)
        self.blocks_raw = {
            "AROMATIC_TOPO": np.zeros((n, ARO_DIM), dtype=np.float64),
            "HYDRO_FIELD": np.zeros((n, HYDRO_DIM), dtype=np.float64),
        }
        self.block_order = ["AROMATIC_TOPO", "HYDRO_FIELD"]
        self.uses_pca = False
        self.raw_dim = TOTAL_DIM
        self.effective_dim = TOTAL_DIM
        self.artifact_hash = _sha(
            {
                "bundle": self.bundle_id,
                "n": n,
                "block_order": self.block_order,
                "dims": [ARO_DIM, HYDRO_DIM],
                "content": "all_zeros",
                "no_hic_labels": True,
                "no_sequence_structure": True,
            }
        )

    def _slice_blocks(self, ids: list[str]) -> dict[str, np.ndarray]:
        idx = [self.id_to_idx[a] for a in ids]
        return {k: self.blocks_raw[k][idx] for k in self.block_order}

    def fit(self, train_ids: list[str]) -> FoldPreprocessor:
        prep = _fit_two_block(self, train_ids, expect_zero_blocks={"AROMATIC_TOPO", "HYDRO_FIELD"})
        return prep

    def transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return prep.transform(self._slice_blocks(ids))

    def matrix_for_ids(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self.transform(prep, ids)


class RealF1Surface35AuxFeatureStore:
    """Canonical F1_SURFACE35 with prospective bundle id label."""

    def __init__(self):
        self.bundle_id = BUNDLE_REAL
        self._inner = H047AuxFeatureStore("F1_SURFACE", H047_PARQUET)
        self.ids = self._inner.ids
        self.id_to_idx = self._inner.id_to_idx
        self.blocks_raw = self._inner.blocks_raw
        self.block_order = self._inner.block_order
        self.uses_pca = False
        self.raw_dim = TOTAL_DIM
        self.effective_dim = TOTAL_DIM
        self.artifact_hash = _sha(
            {
                "bundle": self.bundle_id,
                "inner_bundle": "F1_SURFACE",
                "inner_hash": self._inner.artifact_hash,
                "parquet": str(H047_PARQUET),
            }
        )

    def _slice_blocks(self, ids: list[str]) -> dict[str, np.ndarray]:
        idx = [self.id_to_idx[a] for a in ids]
        return {k: self.blocks_raw[k][idx] for k in self.block_order}

    def fit(self, train_ids: list[str]) -> FoldPreprocessor:
        prep = self._inner.fit(train_ids)
        prep.hash = _sha(
            {
                "bundle": self.bundle_id,
                "inner_prep_hash": prep.hash,
                "train_ids": train_ids,
                "artifact_hash": self.artifact_hash,
            }
        )
        if prep.effective_dim != TOTAL_DIM:
            raise RuntimeError(f"REAL F1 effective_dim={prep.effective_dim}, expected {TOTAL_DIM}")
        return prep

    def transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self._inner.transform(prep, ids)

    def matrix_for_ids(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self.transform(prep, ids)


class AroOnly35AuxFeatureStore:
    """[ARO19 | ZERO16] with F1-matched two-block pathway."""

    def __init__(self):
        self.bundle_id = BUNDLE_ARO_ONLY
        inner = H047AuxFeatureStore("F1_SURFACE", H047_PARQUET)
        self.ids = inner.ids
        self.id_to_idx = inner.id_to_idx
        n = len(self.ids)
        self.blocks_raw = {
            "AROMATIC_TOPO": np.array(inner.blocks_raw["AROMATIC_TOPO"], dtype=np.float64, copy=True),
            "HYDRO_FIELD": np.zeros((n, HYDRO_DIM), dtype=np.float64),
        }
        self.block_order = ["AROMATIC_TOPO", "HYDRO_FIELD"]
        self.uses_pca = False
        self.raw_dim = TOTAL_DIM
        self.effective_dim = TOTAL_DIM
        self.artifact_hash = _sha(
            {
                "bundle": self.bundle_id,
                "layout": "[ARO19|ZERO16]",
                "inner_f1_hash": inner.artifact_hash,
            }
        )

    def _slice_blocks(self, ids: list[str]) -> dict[str, np.ndarray]:
        idx = [self.id_to_idx[a] for a in ids]
        return {k: self.blocks_raw[k][idx] for k in self.block_order}

    def fit(self, train_ids: list[str]) -> FoldPreprocessor:
        return _fit_two_block(self, train_ids, expect_zero_blocks={"HYDRO_FIELD"})

    def transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return prep.transform(self._slice_blocks(ids))

    def matrix_for_ids(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self.transform(prep, ids)


class HydroOnly35AuxFeatureStore:
    """[ZERO19 | HYDRO16] with F1-matched two-block pathway."""

    def __init__(self):
        self.bundle_id = BUNDLE_HYDRO_ONLY
        inner = H047AuxFeatureStore("F1_SURFACE", H047_PARQUET)
        self.ids = inner.ids
        self.id_to_idx = inner.id_to_idx
        n = len(self.ids)
        self.blocks_raw = {
            "AROMATIC_TOPO": np.zeros((n, ARO_DIM), dtype=np.float64),
            "HYDRO_FIELD": np.array(inner.blocks_raw["HYDRO_FIELD"], dtype=np.float64, copy=True),
        }
        self.block_order = ["AROMATIC_TOPO", "HYDRO_FIELD"]
        self.uses_pca = False
        self.raw_dim = TOTAL_DIM
        self.effective_dim = TOTAL_DIM
        self.artifact_hash = _sha(
            {
                "bundle": self.bundle_id,
                "layout": "[ZERO19|HYDRO16]",
                "inner_f1_hash": inner.artifact_hash,
            }
        )

    def _slice_blocks(self, ids: list[str]) -> dict[str, np.ndarray]:
        idx = [self.id_to_idx[a] for a in ids]
        return {k: self.blocks_raw[k][idx] for k in self.block_order}

    def fit(self, train_ids: list[str]) -> FoldPreprocessor:
        return _fit_two_block(self, train_ids, expect_zero_blocks={"AROMATIC_TOPO"})

    def transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return prep.transform(self._slice_blocks(ids))

    def matrix_for_ids(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self.transform(prep, ids)


class PostTransformMaskedF1SurfaceAuxStore:
    """FULL35 with post-StandardScaler zero-mask on selected coordinates.

    Mask is applied in *model-input* space after fold-local impute+scale so that
    removed family slots are exactly neutral zero (not re-centered nonzero).
    """

    def __init__(self, zero_indices: list[int], *, family_id: str):
        idxs = sorted({int(i) for i in zero_indices})
        if not idxs or idxs[0] < 0 or idxs[-1] >= TOTAL_DIM:
            raise ValueError(f"invalid zero_indices={zero_indices}")
        if len(idxs) != len(zero_indices):
            raise ValueError("duplicate zero_indices")
        self.family_id = family_id
        self.zero_indices = idxs
        self._inner = RealF1Surface35AuxFeatureStore()
        self.bundle_id = f"FULL35_MINUS_{family_id}"
        self.ids = self._inner.ids
        self.id_to_idx = self._inner.id_to_idx
        self.blocks_raw = self._inner.blocks_raw
        self.block_order = self._inner.block_order
        self.uses_pca = False
        self.raw_dim = TOTAL_DIM
        self.effective_dim = TOTAL_DIM
        self.artifact_hash = _sha(
            {
                "bundle": self.bundle_id,
                "family_id": family_id,
                "zero_indices": idxs,
                "mask_stage": "post_standardscaler_model_input",
                "inner_hash": self._inner.artifact_hash,
            }
        )

    def fit(self, train_ids: list[str]) -> FoldPreprocessor:
        # Fit on unmasked FULL35 pathway (identical scaler to FULL baseline).
        return self._inner.fit(train_ids)

    def transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        X = self._inner.transform(prep, ids)
        X = np.array(X, dtype=np.float32, copy=True)
        X[:, self.zero_indices] = 0.0
        return X

    def matrix_for_ids(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self.transform(prep, ids)

    def unmasked_transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self._inner.transform(prep, ids)


def make_surface_prospective_aux(bundle_id: str):
    if bundle_id == BUNDLE_SHAM:
        return Sham35AuxFeatureStore()
    if bundle_id == BUNDLE_REAL:
        return RealF1Surface35AuxFeatureStore()
    if bundle_id == BUNDLE_ARO_ONLY:
        return AroOnly35AuxFeatureStore()
    if bundle_id == BUNDLE_HYDRO_ONLY:
        return HydroOnly35AuxFeatureStore()
    raise ValueError(bundle_id)


def make_family_lofo_aux(family_id: str, zero_indices: list[int]) -> PostTransformMaskedF1SurfaceAuxStore:
    return PostTransformMaskedF1SurfaceAuxStore(zero_indices, family_id=family_id)
