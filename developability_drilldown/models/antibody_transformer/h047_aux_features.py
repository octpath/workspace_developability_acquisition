#!/usr/bin/env python3
"""H047-derived auxiliary feature bundles for late fusion (EXP-H082–H093).

Bundles (global ESM2_H deliberately excluded):
  F1 SURFACE             : AROMATIC_TOPO19 + HYDRO_FIELD16 = 35
  F2 SEQUENCE_TITRATION  : SEQ_ALL115 + TITRATION_SHAPE18 = 133
  F3 LOCAL_RASA_CDR3     : FB_ESM2_RASA_CDR3 1280 -> TRAIN PCA32
  F4 H047_AUX_ALL        : F1 + F2 + PCA32(F3) = 200 effective

Preprocessing is TRAIN-only (median impute, StandardScaler, PCA).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
H047_PARQUET = ROOT / "experiments" / "features" / "EXP-H047.parquet"
PCA_CAP = 32
PCA_RANDOM_STATE = 0

BUNDLE_IDS = ("F1_SURFACE", "F2_SEQUENCE_TITRATION", "F3_LOCAL_RASA_CDR3", "F4_H047_AUX_ALL")


def _sha(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(np.float64)


def _impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    out = np.array(X, dtype=np.float64, copy=True)
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = np.take(med, np.where(mask)[1])
    return out


@dataclass
class FoldPreprocessor:
    """Fitted on TRAIN only; transform is apply-only."""

    medians: dict[str, np.ndarray]
    scalers: dict[str, StandardScaler]
    pca: Optional[PCA]
    block_order: list[str]
    effective_dim: int
    train_ids: list[str]
    hash: str

    def transform(self, X_blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for name in self.block_order:
            X = _impute_apply(X_blocks[name], self.medians[name])
            if name == "FB_ESM2_RASA_CDR3":
                assert self.pca is not None
                X = self.pca.transform(X)
            X = self.scalers[name].transform(X)
            parts.append(X.astype(np.float32))
        return np.concatenate(parts, axis=1)


class H047AuxFeatureStore:
    """Indexed raw feature matrices for one fusion bundle."""

    def __init__(self, bundle_id: str, parquet_path: Path = H047_PARQUET):
        if bundle_id not in BUNDLE_IDS:
            raise ValueError(bundle_id)
        self.bundle_id = bundle_id
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path)
        if "id" not in df.columns:
            raise ValueError("H047 parquet missing id")
        self.ids = df["id"].astype(str).tolist()
        self.id_to_idx = {a: i for i, a in enumerate(self.ids)}
        # Feature columns in canonical H047 parquet order (exclude id/split).
        feat_cols = [c for c in df.columns if c not in ("id", "split")]
        if len(feat_cols) != 2728:
            raise RuntimeError(f"expected 2728 H047 features, got {len(feat_cols)}")

        # Exact pca_groups ranges from EXP-H047.yaml (0-index within feat_cols):
        # ESM2_H 1280 | SEQ 115 | ARO 19 | HYDRO 16 | TITR 18 | FB 1280
        slices = {
            "ESM2_H": slice(0, 1280),
            "SEQ_ALL": slice(1280, 1395),
            "AROMATIC_TOPO": slice(1395, 1414),
            "HYDRO_FIELD": slice(1414, 1430),
            "TITRATION_SHAPE": slice(1430, 1448),
            "FB_ESM2_RASA_CDR3": slice(1448, 2728),
        }
        # Sanity: do not silently include global ESM2_H in any fusion bundle.
        assert feat_cols[0].startswith("esm2h_")
        assert feat_cols[1280].startswith("seqA_")
        assert feat_cols[1395].startswith("aro_")
        assert feat_cols[1448].startswith("FB_ESM2_RASA_CDR3_")

        self.block_cols = {k: feat_cols[sl] for k, sl in slices.items() if k != "ESM2_H"}
        for name, expected in (
            ("SEQ_ALL", 115),
            ("AROMATIC_TOPO", 19),
            ("HYDRO_FIELD", 16),
            ("TITRATION_SHAPE", 18),
            ("FB_ESM2_RASA_CDR3", 1280),
        ):
            if len(self.block_cols[name]) != expected:
                raise RuntimeError(
                    f"{name} expected {expected}, got {len(self.block_cols[name])}"
                )

        self.blocks_raw: dict[str, np.ndarray] = {
            k: df[cols].to_numpy(dtype=np.float64) for k, cols in self.block_cols.items()
        }
        if bundle_id == "F1_SURFACE":
            self.block_order = ["AROMATIC_TOPO", "HYDRO_FIELD"]
            self.uses_pca = False
            self.raw_dim = 35
            self.effective_dim = 35
        elif bundle_id == "F2_SEQUENCE_TITRATION":
            self.block_order = ["SEQ_ALL", "TITRATION_SHAPE"]
            self.uses_pca = False
            self.raw_dim = 133
            self.effective_dim = 133
        elif bundle_id == "F3_LOCAL_RASA_CDR3":
            self.block_order = ["FB_ESM2_RASA_CDR3"]
            self.uses_pca = True
            self.raw_dim = 1280
            self.effective_dim = PCA_CAP
        else:  # F4
            self.block_order = [
                "AROMATIC_TOPO",
                "HYDRO_FIELD",
                "SEQ_ALL",
                "TITRATION_SHAPE",
                "FB_ESM2_RASA_CDR3",
            ]
            self.uses_pca = True
            self.raw_dim = 35 + 133 + 1280
            self.effective_dim = 35 + 133 + PCA_CAP  # 200

        self.artifact_hash = _sha(
            {
                "parquet": str(parquet_path),
                "bundle": bundle_id,
                "n_rows": len(self.ids),
                "blocks": {k: self.block_cols[k] for k in self.block_order},
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
            effective_dim=self.effective_dim if not self.uses_pca else (
                sum(
                    PCA_CAP if n == "FB_ESM2_RASA_CDR3" else self.blocks_raw[n].shape[1]
                    for n in self.block_order
                )
            ),
            train_ids=list(train_ids),
            hash="",
        )
        # Recompute effective dim from fitted PCA components
        if pca is not None:
            eff = 0
            for n in self.block_order:
                if n == "FB_ESM2_RASA_CDR3":
                    eff += int(pca.n_components_)
                else:
                    eff += self.blocks_raw[n].shape[1]
            prep.effective_dim = eff
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


def f4_subblock_slices(effective_dim: int = 200) -> dict[str, slice]:
    """Column slices inside F4 effective vector after per-block preprocess.

    Order: ARO(19) + HYDRO(16) + SEQ(115) + TITR(18) + PCA32 = 200
    """
    return {
        "SURFACE": slice(0, 35),
        "SEQUENCE_TITRATION": slice(35, 168),
        "LOCAL_RASA": slice(168, 168 + 32),
    }
