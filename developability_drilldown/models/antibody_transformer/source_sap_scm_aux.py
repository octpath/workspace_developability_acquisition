#!/usr/bin/env python3
"""Late-fusion aux stores for SOURCE SAP24 / SCM24 / COMBINED48 (H114–H125).

Frozen research artifacts under feature_research/hic_sap_scm_source/features/.
Canonical copies live in experiments/features/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .h047_aux_features import FoldPreprocessor, _impute_apply, _impute_fit, _sha

ROOT = Path(__file__).resolve().parents[2]
SRC_FEAT = ROOT.parent / "feature_research" / "hic_sap_scm_source" / "features"
EXP_FEAT = ROOT / "experiments" / "features"

BUNDLE_TO_SOURCE = {
    "FS_HIC_SOURCE_SAP24": "antibody_source_sap24.parquet",
    "FS_HIC_SOURCE_SCM24": "antibody_source_scm24.parquet",
    "FS_HIC_SOURCE_SAP24_SCM24": "antibody_source_sap_scm48.parquet",
}

BUNDLE_IDS = tuple(BUNDLE_TO_SOURCE.keys())


def _columns_without_id(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "id"]


class SourceSapScmAuxFeatureStore:
    """Duck-typed like H047AuxFeatureStore / HspPromotedAuxFeatureStore."""

    def __init__(self, bundle_id: str):
        if bundle_id not in BUNDLE_IDS:
            raise ValueError(bundle_id)
        self.bundle_id = bundle_id
        pq = EXP_FEAT / BUNDLE_TO_SOURCE[bundle_id]
        if not pq.exists():
            # fall back to research artifact
            pq = SRC_FEAT / BUNDLE_TO_SOURCE[bundle_id]
        df = pd.read_parquet(pq)
        if len(df) != 324:
            raise RuntimeError(f"expected 324 rows, got {len(df)} for {pq}")
        self.ids = df["id"].astype(str).tolist()
        self.id_to_idx = {a: i for i, a in enumerate(self.ids)}
        self.columns = _columns_without_id(df)
        X = df[self.columns].to_numpy(np.float64)

        self.blocks_raw: dict[str, np.ndarray] = {}
        if bundle_id == "FS_HIC_SOURCE_SAP24":
            assert len(self.columns) == 24
            self.blocks_raw["SAP24"] = X
            self.block_order = ["SAP24"]
        elif bundle_id == "FS_HIC_SOURCE_SCM24":
            assert len(self.columns) == 24
            self.blocks_raw["SCM24"] = X
            self.block_order = ["SCM24"]
        else:
            assert len(self.columns) == 48
            self.blocks_raw["SAP24"] = X[:, :24]
            self.blocks_raw["SCM24"] = X[:, 24:]
            self.block_order = ["SAP24", "SCM24"]

        self.raw_dim = int(X.shape[1])
        self.effective_dim = self.raw_dim
        self.uses_pca = False
        self.parquet_path = str(pq)
        self.artifact_hash = _sha(
            {
                "bundle": bundle_id,
                "parquet": str(pq),
                "n": len(self.ids),
                "columns": self.columns,
                "block_order": self.block_order,
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
        prep.effective_dim = self.effective_dim
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
        if "SAP24" in out and "SCM24" in out:
            out["COMBINED48"] = slice(out["SAP24"].start, out["SCM24"].stop)
        out["ALL_AUX"] = slice(0, start)
        return out
