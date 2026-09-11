#!/usr/bin/env python3
"""Static SAP-like Kyte–Doolittle (STATIC_SAP_KD / SSKD) antibody-level descriptors.

Fv ESMFold structures; whole-residue Shrake–Rupley SASA / Tien MaxASA;
side-chain-centroid neighborhoods; min-max Kyte–Doolittle.
NOT MD-averaged Chennamsetty SAP; NOT Black–Mould STATIC-SAP.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Standard Kyte–Doolittle hydrophobicity (Kyte & Doolittle 1982)
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

_KD_VALS = np.asarray(list(KYTE_DOOLITTLE.values()), dtype=np.float64)
KD_MIN = float(_KD_VALS.min())  # -4.5 (R)
KD_MAX = float(_KD_VALS.max())  # 4.5 (I)

# Tien 2013 MaxASA (same as structure_utils.MAX_ASA)
TIEN_MAXASA: dict[str, float] = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

PROBE = 1.4
N_POINTS_PRIMARY = 100  # REPOSITORY_DEFAULT (H047 / structure_utils)
N_POINTS_QC = 960
R_REF = 5.0  # CANONICAL_SAP_RADIUS_FALLBACK (source R not recovered)
R_SENSITIVITY = 10.0
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
INCLUDE_SELF = True  # matches existing STATIC-SAP d<=R including self

SAP3_COLS = ["SSKD_ALL_MAX", "SSKD_ALL_MEAN", "SSKD_ALL_SUM"]
SAP9_COLS = [
    "SSKD_ALL_MAX",
    "SSKD_ALL_MEAN",
    "SSKD_ALL_SUM",
    "SSKD_H_MAX",
    "SSKD_H_MEAN",
    "SSKD_H_SUM",
    "SSKD_L_MAX",
    "SSKD_L_MEAN",
    "SSKD_L_SUM",
]


def kd_norm(aa: str) -> float:
    """Min-max Kyte–Doolittle over the canonical 20 AA → [0,1]."""
    if aa not in KYTE_DOOLITTLE:
        return float("nan")
    return (KYTE_DOOLITTLE[aa] - KD_MIN) / (KD_MAX - KD_MIN)


def rsasa(sasa: float, aa: str) -> float:
    """Whole-residue SASA / Tien MaxASA, clipped to [0,1]."""
    maxasa = TIEN_MAXASA.get(aa)
    if maxasa is None or not np.isfinite(sasa):
        return float("nan")
    return float(np.clip(sasa / maxasa, 0.0, 1.0))


def sidechain_centroid(atoms: list[tuple[str, np.ndarray, str]]) -> Optional[np.ndarray]:
    """Arithmetic mean of non-hydrogen side-chain coords; Gly → None (caller uses CA)."""
    coords = []
    for name, coord, element in atoms:
        n = name.strip().upper()
        el = (element or "").strip().upper()
        if n in BACKBONE_ATOMS:
            continue
        if el == "H" or n.startswith("H"):
            continue
        coords.append(np.asarray(coord, dtype=np.float64))
    if not coords:
        return None
    return np.mean(np.stack(coords, axis=0), axis=0)


def resolve_centroid(aa: str, atoms, ca: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], str]:
    """Return (centroid, provenance): sidechain | gly_ca_fallback | ca_fallback | invalid."""
    sc = sidechain_centroid(atoms)
    if sc is not None:
        return sc, "sidechain"
    if aa == "G" and ca is not None:
        return np.asarray(ca, dtype=np.float64), "gly_ca_fallback"
    if ca is not None:
        return np.asarray(ca, dtype=np.float64), "ca_fallback"
    return None, "invalid"


def sskd_scores(
    centroids: np.ndarray,
    kd_n: np.ndarray,
    rasa: np.ndarray,
    valid: np.ndarray,
    radius: float,
    *,
    include_self: bool = INCLUDE_SELF,
) -> np.ndarray:
    """Per-residue STATIC_SAP_KD_i(R) for valid centers; NaN for invalid."""
    n = len(centroids)
    out = np.full(n, np.nan, dtype=np.float64)
    # Precompute pairwise among valid
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out
    C = centroids[idx]
    # distances
    d2 = np.sum((C[:, None, :] - C[None, :, :]) ** 2, axis=-1)
    r2 = float(radius) ** 2
    neigh = d2 <= r2
    if not include_self:
        np.fill_diagonal(neigh, False)
    contrib = kd_n[idx] * rasa[idx]
    # for each center i, sum contrib[j] where neigh
    scores = neigh.astype(np.float64) @ contrib
    out[idx] = scores
    return out


def aggregate_max_mean_sum(scores: np.ndarray) -> tuple[float, float, float]:
    v = scores[np.isfinite(scores)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.max()), float(v.mean()), float(v.sum())
