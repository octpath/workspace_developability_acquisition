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
ARO_DIM = 19
HYDRO_DIM = 16
TOTAL_DIM = ARO_DIM + HYDRO_DIM  # 35


class Sham35AuxFeatureStore:
    """35D constant-zero aux with F1-matched two-block preprocessing pathway."""

    def __init__(self, ids: Optional[list[str]] = None):
        self.bundle_id = BUNDLE_SHAM
        if ids is None:
            # Align id universe with H047 / F1 coverage (324 antibodies)
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
        Xb = self._slice_blocks(train_ids)
        medians: dict[str, np.ndarray] = {}
        scalers: dict[str, StandardScaler] = {}
        for name in self.block_order:
            X = Xb[name]
            med = _impute_fit(X)
            medians[name] = med
            Xi = _impute_apply(X, med)
            sc = StandardScaler()
            sc.fit(Xi)
            # Constant features: sklearn sets scale_=1.0; assert finite
            if not np.all(np.isfinite(sc.scale_)) or not np.all(sc.scale_ > 0):
                raise RuntimeError(f"SHAM35 StandardScaler unsafe for block {name}: scale_={sc.scale_}")
            scalers[name] = sc
        prep = FoldPreprocessor(
            medians=medians,
            scalers=scalers,
            pca=None,
            block_order=list(self.block_order),
            effective_dim=self.effective_dim,
            train_ids=list(train_ids),
            hash="",
        )
        prep.hash = _sha(
            {
                "bundle": self.bundle_id,
                "train_ids": train_ids,
                "artifact_hash": self.artifact_hash,
            }
        )
        # smoke transform
        Xt = prep.transform(self._slice_blocks(train_ids))
        if Xt.shape != (len(train_ids), TOTAL_DIM):
            raise RuntimeError(f"SHAM35 transform shape {Xt.shape}")
        if not np.allclose(Xt, 0.0):
            raise RuntimeError("SHAM35 expected zeros after scale of zeros")
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

    def fit(self, train_ids: list[str]) -> FoldPreprocessor:
        prep = self._inner.fit(train_ids)
        # Relabel hash to include outer bundle id
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


def make_surface_prospective_aux(bundle_id: str):
    if bundle_id == BUNDLE_SHAM:
        return Sham35AuxFeatureStore()
    if bundle_id == BUNDLE_REAL:
        return RealF1Surface35AuxFeatureStore()
    raise ValueError(bundle_id)
