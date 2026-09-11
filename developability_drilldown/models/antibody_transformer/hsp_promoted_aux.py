#!/usr/bin/env python3
"""Late-fusion aux stores for promoted HSP antibody-level B3 blocks (H102–H113).

Frozen families (source commit 210a270d, bundle B3 = ALL_FV MAX/MEAN/SUM):
  FS_HIC_HSP_BM_R5_PROMOTED
  FS_HIC_HSP_FP_R5_PROMOTED
  FS_HIC_HSP_EIS_R8_PROMOTED
  FS_HIC_SURFACE_PLUS_HSP_BM_R5_PROMOTED
  FS_HIC_SURFACE_PLUS_HSP_FP_R5_PROMOTED
  FS_HIC_SURFACE_PLUS_HSP_EIS_R8_PROMOTED
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
    _impute_apply,
    _impute_fit,
    _sha,
)

ROOT = Path(__file__).resolve().parents[2]
H047_PARQUET = ROOT / "experiments" / "features" / "EXP-H047.parquet"

HSP_COLS = ("HSP_MAX", "HSP_MEAN", "HSP_SUM")

BUNDLE_TO_PARQUET = {
    "FS_HIC_HSP_BM_R5_PROMOTED": "hsp_bm_r5_promoted.parquet",
    "FS_HIC_SURFACE_PLUS_HSP_BM_R5_PROMOTED": "hsp_bm_r5_promoted.parquet",
    "FS_HIC_HSP_FP_R5_PROMOTED": "hsp_fp_r5_promoted.parquet",
    "FS_HIC_SURFACE_PLUS_HSP_FP_R5_PROMOTED": "hsp_fp_r5_promoted.parquet",
    "FS_HIC_HSP_EIS_R8_PROMOTED": "hsp_eis_r8_promoted.parquet",
    "FS_HIC_SURFACE_PLUS_HSP_EIS_R8_PROMOTED": "hsp_eis_r8_promoted.parquet",
}

BUNDLE_IDS = tuple(BUNDLE_TO_PARQUET.keys())


class HspPromotedAuxFeatureStore:
    """Duck-typed like H047AuxFeatureStore / StaticSapKdAuxFeatureStore."""

    def __init__(self, bundle_id: str):
        if bundle_id not in BUNDLE_IDS:
            raise ValueError(bundle_id)
        self.bundle_id = bundle_id
        pq = ROOT / "experiments" / "features" / BUNDLE_TO_PARQUET[bundle_id]
        df = pd.read_parquet(pq)
        if len(df) != 324:
            raise RuntimeError(f"expected 324 rows, got {len(df)} for {pq}")
        self.ids = df["id"].astype(str).tolist()
        self.id_to_idx = {a: i for i, a in enumerate(self.ids)}

        self.blocks_raw: dict[str, np.ndarray] = {}
        self.block_order: list[str] = []
        self.uses_pca = False
        self.h047: Optional[H047AuxFeatureStore] = None

        hsp = df[list(HSP_COLS)].to_numpy(np.float64)
        if bundle_id.startswith("FS_HIC_SURFACE_PLUS_"):
            self.h047 = H047AuxFeatureStore("F1_SURFACE", H047_PARQUET)
            h_idx = [self.h047.id_to_idx[a] for a in self.ids]
            for name in self.h047.block_order:
                self.blocks_raw[name] = self.h047.blocks_raw[name][h_idx]
            self.blocks_raw["HSP3"] = hsp
            self.block_order = list(self.h047.block_order) + ["HSP3"]
            self.raw_dim = 35 + 3
            self.effective_dim = 38
        else:
            self.blocks_raw["HSP3"] = hsp
            self.block_order = ["HSP3"]
            self.raw_dim = 3
            self.effective_dim = 3

        self.artifact_hash = _sha(
            {
                "bundle": bundle_id,
                "parquet": str(pq),
                "n": len(self.ids),
                "block_order": self.block_order,
                "hsp_cols": list(HSP_COLS),
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
        prep.effective_dim = int(sum(self.blocks_raw[n].shape[1] for n in self.block_order))
        prep.hash = _sha(
            {
                "bundle": self.bundle_id,
                "train_ids": train_ids,
                "artifact_hash": self.artifact_hash,
            }
        )
        return prep

    def transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return prep.transform(self._slice_blocks(ids))

    def matrix_for_ids(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self.transform(prep, ids)

    def block_slices(self, effective_dim: Optional[int] = None) -> dict[str, slice]:
        out: dict[str, slice] = {}
        start = 0
        for n in self.block_order:
            d = self.blocks_raw[n].shape[1]
            out[n] = slice(start, start + d)
            start += d
        if "AROMATIC_TOPO" in out and "HYDRO_FIELD" in out:
            out["SURFACE"] = slice(out["AROMATIC_TOPO"].start, out["HYDRO_FIELD"].stop)
        if "HSP3" in out:
            out["HSP"] = out["HSP3"]
        if "SURFACE" in out and "HSP" in out:
            out["SURFACE_PLUS_HSP"] = slice(out["SURFACE"].start, out["HSP"].stop)
        out["ALL_AUX"] = slice(0, start)
        return out
