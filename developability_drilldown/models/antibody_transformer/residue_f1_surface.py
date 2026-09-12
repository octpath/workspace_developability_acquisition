#!/usr/bin/env python3
"""Compact residue-level F1_SURFACE features for H128–H133.

Schema: FS_HIC_RESIDUE_F1_SURFACE_COMPACT10 (p=10), frozen before training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
SCHEMA_YAML = ROOT / "results" / "H128_H133_RESIDUE_SURFACE_SCHEMA.yaml"
COMPACT_PQ = (
    REPO
    / "feature_research/f1_surface_residue_reconstruction/features/residue_surface_compact10.parquet"
)

CHANNELS = [
    "rasa",
    "sasa",
    "is_aromatic",
    "is_exposed",
    "is_strongly_exposed",
    "hydro_area_sum",
    "hydro_H_awmean",
    "hydro_pos_H_area",
    "hydro_neg_H_area",
    "availability_hydro",
]
CONTINUOUS = [
    "rasa",
    "sasa",
    "hydro_area_sum",
    "hydro_H_awmean",
    "hydro_pos_H_area",
    "hydro_neg_H_area",
]
BINARY = ["is_aromatic", "is_exposed", "is_strongly_exposed", "availability_hydro"]
HYDRO_CONT = ["hydro_area_sum", "hydro_H_awmean", "hydro_pos_H_area", "hydro_neg_H_area"]
P = len(CHANNELS)
AVAIL_IDX = CHANNELS.index("availability_hydro")


def load_schema() -> dict:
    return yaml.safe_load(SCHEMA_YAML.read_text())


def schema_hash() -> str:
    doc = load_schema()
    blob = json.dumps(
        {"channels": doc["channels"], "content_sha256": doc["compact_artifact"]["content_sha256"]},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class FoldSurfacePrep:
    mean: np.ndarray  # [p]
    scale: np.ndarray  # [p]
    channels: list[str]
    train_ids: list[str]
    hash: str

    def transform_chain(self, X: np.ndarray) -> np.ndarray:
        """X [L,p] float; returns standardized. Hydro-missing rows already 0 on hydro cont."""
        out = np.array(X, dtype=np.float64, copy=True)
        out = (out - self.mean) / self.scale
        avail = X[:, AVAIL_IDX] > 0.5
        for name in HYDRO_CONT:
            j = CHANNELS.index(name)
            out[~avail, j] = 0.0
        for name in BINARY:
            j = CHANNELS.index(name)
            out[:, j] = X[:, j]
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "channels": list(self.channels),
            "train_ids": list(self.train_ids),
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FoldSurfacePrep":
        return cls(
            mean=np.asarray(d["mean"], float),
            scale=np.asarray(d["scale"], float),
            channels=list(d["channels"]),
            train_ids=list(d["train_ids"]),
            hash=str(d["hash"]),
        )


def _rows_for_ids(compact: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return compact[compact["antibody_id"].astype(str).isin(set(map(str, ids)))]


def fit_fold_surface_prep(compact: pd.DataFrame, train_ids: list[str]) -> FoldSurfacePrep:
    sub = _rows_for_ids(compact, train_ids)
    mean = np.zeros(P, dtype=np.float64)
    scale = np.ones(P, dtype=np.float64)
    for name in CONTINUOUS:
        j = CHANNELS.index(name)
        if name in HYDRO_CONT:
            vals = sub.loc[sub["availability_hydro"] > 0.5, name].to_numpy(float)
        else:
            vals = sub[name].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        mu = float(np.mean(vals)) if len(vals) else 0.0
        sd = float(np.std(vals)) if len(vals) else 1.0
        if sd < 1e-8:
            sd = 1.0
        mean[j] = mu
        scale[j] = sd
    h = hashlib.sha256(
        json.dumps({"mean": mean.tolist(), "scale": scale.tolist(), "train_ids": list(train_ids)}, sort_keys=True).encode()
    ).hexdigest()
    return FoldSurfacePrep(mean=mean, scale=scale, channels=list(CHANNELS), train_ids=list(train_ids), hash=h)


def attach_residue_surface_compact(rb, compact: Optional[pd.DataFrame] = None) -> None:
    """Attach raw [N,L,p] heavy/light surface arrays onto ResidueBundle (pad=0)."""
    if compact is None:
        compact = pd.read_parquet(COMPACT_PQ)
    assert list(load_schema()["channels"]) == CHANNELS
    N = len(rb.ids)
    Lh = rb.heavy_aa.shape[1]
    Ll = rb.light_aa.shape[1]
    heavy = np.zeros((N, Lh, P), dtype=np.float32)
    light = np.zeros((N, Ll, P), dtype=np.float32)
    id_to_i = {str(a): i for i, a in enumerate(rb.ids)}
    vals = compact[CHANNELS].to_numpy(float)
    aids = compact["antibody_id"].astype(str).to_numpy()
    chains = compact["chain"].astype(str).to_numpy()
    sis = compact["sequence_index"].to_numpy(int)
    for row, aid, ch, si in zip(vals, aids, chains, sis):
        i = id_to_i.get(aid)
        if i is None:
            continue
        if ch == "H":
            if 0 <= si < Lh and rb.heavy_mask[i, si]:
                heavy[i, si] = row
        elif ch == "L":
            if 0 <= si < Ll and rb.light_mask[i, si]:
                light[i, si] = row
    rb.heavy_surface = heavy
    rb.light_surface = light
    rb.residue_surface_dim = P
    rb.residue_surface_channels = list(CHANNELS)
    rb.residue_surface_schema_hash = schema_hash()


def apply_prep_to_sample(X: np.ndarray, prep: FoldSurfacePrep) -> np.ndarray:
    return prep.transform_chain(X)
