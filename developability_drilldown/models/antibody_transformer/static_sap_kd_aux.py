#!/usr/bin/env python3
"""Late-fusion aux stores for STATIC_SAP_KD antibody-level bundles (H094–H101).

Bundles:
  FS_HIC_STATIC_SAP_KD_GLOBAL3              SAP3 (3)
  FS_HIC_STATIC_SAP_KD_CHAIN9               SAP9 (9)
  FS_HIC_SURFACE_PLUS_STATIC_SAP_KD_CHAIN9  SURFACE(35) + SAP9(9) = 44
  FS_HIC_F4_PLUS_STATIC_SAP_KD_CHAIN9       F4(200) + SAP9(9) = 209 effective
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .h047_aux_features import (
    FoldPreprocessor,
    H047AuxFeatureStore,
    PCA_CAP,
    PCA_RANDOM_STATE,
    _impute_apply,
    _impute_fit,
    _sha,
)
from .static_sap_kd import SAP3_COLS, SAP9_COLS

ROOT = Path(__file__).resolve().parents[2]
G3_PATH = ROOT / "experiments" / "features" / "static_sap_kd_antibody_global3.parquet"
C9_PATH = ROOT / "experiments" / "features" / "static_sap_kd_antibody_chain9.parquet"
H047_PARQUET = ROOT / "experiments" / "features" / "EXP-H047.parquet"

BUNDLE_IDS = (
    "FS_HIC_STATIC_SAP_KD_GLOBAL3",
    "FS_HIC_STATIC_SAP_KD_CHAIN9",
    "FS_HIC_SURFACE_PLUS_STATIC_SAP_KD_CHAIN9",
    "FS_HIC_F4_PLUS_STATIC_SAP_KD_CHAIN9",
)


class StaticSapKdAuxFeatureStore:
    """Duck-typed like H047AuxFeatureStore: fit / transform / effective_dim / artifact_hash."""

    def __init__(self, bundle_id: str):
        if bundle_id not in BUNDLE_IDS:
            raise ValueError(bundle_id)
        self.bundle_id = bundle_id

        g3 = pd.read_parquet(G3_PATH)
        c9 = pd.read_parquet(C9_PATH)
        if len(g3) != 324 or len(c9) != 324:
            raise RuntimeError(f"expected 324 rows, got g3={len(g3)} c9={len(c9)}")
        self.ids = c9["id"].astype(str).tolist()
        self.id_to_idx = {a: i for i, a in enumerate(self.ids)}

        self.blocks_raw: dict[str, np.ndarray] = {}
        self.block_order: list[str] = []
        self.uses_pca = False
        self.h047: Optional[H047AuxFeatureStore] = None

        if bundle_id == "FS_HIC_STATIC_SAP_KD_GLOBAL3":
            self.blocks_raw["SAP3"] = g3[SAP3_COLS].to_numpy(np.float64)
            self.block_order = ["SAP3"]
            self.raw_dim = 3
            self.effective_dim = 3
        elif bundle_id == "FS_HIC_STATIC_SAP_KD_CHAIN9":
            self.blocks_raw["SAP9"] = c9[SAP9_COLS].to_numpy(np.float64)
            self.block_order = ["SAP9"]
            self.raw_dim = 9
            self.effective_dim = 9
        elif bundle_id == "FS_HIC_SURFACE_PLUS_STATIC_SAP_KD_CHAIN9":
            self.h047 = H047AuxFeatureStore("F1_SURFACE", H047_PARQUET)
            # align ids
            assert self.h047.ids == self.ids or set(self.h047.ids) == set(self.ids)
            # reindex H047 to SAP id order
            h_idx = [self.h047.id_to_idx[a] for a in self.ids]
            for name in self.h047.block_order:
                self.blocks_raw[name] = self.h047.blocks_raw[name][h_idx]
            self.blocks_raw["SAP9"] = c9[SAP9_COLS].to_numpy(np.float64)
            self.block_order = list(self.h047.block_order) + ["SAP9"]
            self.raw_dim = 35 + 9
            self.effective_dim = 44
        else:  # F4 + SAP9
            self.h047 = H047AuxFeatureStore("F4_H047_AUX_ALL", H047_PARQUET)
            h_idx = [self.h047.id_to_idx[a] for a in self.ids]
            for name in self.h047.block_order:
                self.blocks_raw[name] = self.h047.blocks_raw[name][h_idx]
            self.blocks_raw["SAP9"] = c9[SAP9_COLS].to_numpy(np.float64)
            self.block_order = list(self.h047.block_order) + ["SAP9"]
            self.uses_pca = True
            self.raw_dim = 35 + 133 + 1280 + 9
            self.effective_dim = 200 + 9  # 209

        self.artifact_hash = _sha(
            {
                "bundle": bundle_id,
                "g3": str(G3_PATH),
                "c9": str(C9_PATH),
                "n": len(self.ids),
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
        pca = None
        for name in self.block_order:
            X = Xb[name]
            med = _impute_fit(X)
            medians[name] = med
            Xi = _impute_apply(X, med)
            if name == "FB_ESM2_RASA_CDR3":
                n_comp = min(PCA_CAP, max(1, len(train_ids) - 1), Xi.shape[1])
                pca = PCA(n_components=n_comp, random_state=PCA_RANDOM_STATE)
                Xi = pca.fit_transform(Xi)
            sc = StandardScaler()
            sc.fit(Xi)
            scalers[name] = sc
        prep = FoldPreprocessor(
            medians=medians,
            scalers=scalers,
            pca=pca,
            block_order=list(self.block_order),
            effective_dim=self.effective_dim,
            train_ids=list(train_ids),
            hash="",
        )
        if pca is not None:
            eff = 0
            for n in self.block_order:
                if n == "FB_ESM2_RASA_CDR3":
                    eff += int(pca.n_components_)
                else:
                    eff += self.blocks_raw[n].shape[1]
            prep.effective_dim = eff
        else:
            prep.effective_dim = int(sum(self.blocks_raw[n].shape[1] for n in self.block_order))
        prep.hash = _sha(
            {
                "bundle": self.bundle_id,
                "train_ids": train_ids,
                "artifact_hash": self.artifact_hash,
                "pca_n": None if pca is None else int(pca.n_components_),
            }
        )
        return prep

    def transform(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return prep.transform(self._slice_blocks(ids))

    def matrix_for_ids(self, prep: FoldPreprocessor, ids: list[str]) -> np.ndarray:
        return self.transform(prep, ids)

    def block_slices(self, effective_dim: Optional[int] = None) -> dict[str, slice]:
        """Column slices inside the effective (preprocessed) aux vector."""
        # Build cumulative dims assuming PCA_CAP for FB when present
        dims = []
        names = []
        for n in self.block_order:
            if n == "FB_ESM2_RASA_CDR3":
                d = PCA_CAP
            else:
                d = self.blocks_raw[n].shape[1]
            names.append(n)
            dims.append(d)
        # collapse SURFACE / SEQUENCE_TITRATION / LOCAL_RASA / SAP for reporting
        out: dict[str, slice] = {}
        # raw named blocks
        start = 0
        for n, d in zip(names, dims):
            out[n] = slice(start, start + d)
            start += d
        # aliases
        if "AROMATIC_TOPO" in out and "HYDRO_FIELD" in out:
            out["SURFACE"] = slice(out["AROMATIC_TOPO"].start, out["HYDRO_FIELD"].stop)
        if "SEQ_ALL" in out and "TITRATION_SHAPE" in out:
            out["SEQUENCE_TITRATION"] = slice(out["SEQ_ALL"].start, out["TITRATION_SHAPE"].stop)
        if "FB_ESM2_RASA_CDR3" in out:
            out["LOCAL_RASA"] = out["FB_ESM2_RASA_CDR3"]
        if "SAP3" in out:
            out["SAP"] = out["SAP3"]
        if "SAP9" in out:
            out["SAP"] = out["SAP9"]
        if "SURFACE" in out and "SAP" in out and "FS_HIC_SURFACE" in self.bundle_id:
            out["SURFACE_PLUS_SAP"] = slice(out["SURFACE"].start, out["SAP"].stop)
        out["ALL_AUX"] = slice(0, start)
        return out
