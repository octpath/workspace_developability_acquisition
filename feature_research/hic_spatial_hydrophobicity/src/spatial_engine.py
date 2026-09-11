#!/usr/bin/env python3
"""Generic spatial hydrophobicity engine + aggregations."""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

from scales import AA20, property_minmax, property_raw

RADII = [4.0, 5.0, 6.0, 7.5, 8.0, 10.0]
EXPOSURES = ("TOTAL_RASA_TIEN", "SIDECHAIN_OVER_TIEN", "SIDECHAIN_SASA_ABS")
NEIGHBORHOODS = ("CENTROID", "CLOSEST_SC")
TRANSFORMS = ("RAW", "MINMAX")
SCOPES = ("ALL_FV", "HEAVY", "LIGHT", "ALL_CDR", "H_CDR3", "L_CDR3")
AGGS = ("MAX", "MEAN", "SUM", "Q90", "Q95", "TOP3_MEAN", "TOP5_MEAN")


def exposure_vector(ab: dict, mode: str) -> np.ndarray:
    if mode == "TOTAL_RASA_TIEN":
        return ab["total_rASA_Tien"].copy()
    if mode == "SIDECHAIN_OVER_TIEN":
        return ab["sidechain_over_Tien"].copy()
    if mode == "SIDECHAIN_SASA_ABS":
        return ab["sidechain_SASA"].copy()
    raise ValueError(mode)


def hydro_vector(aas: list[str], scale_id: str, transform: str) -> np.ndarray:
    tab = property_raw(scale_id) if transform == "RAW" else property_minmax(scale_id)
    return np.asarray([tab[a] for a in aas], dtype=np.float64)


def pairwise_centroid(centroids: np.ndarray) -> np.ndarray:
    # (n,n) distances
    d = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def pairwise_closest_sc(sc_list: list[np.ndarray]) -> np.ndarray:
    n = len(sc_list)
    D = np.full((n, n), np.inf)
    for i in range(n):
        ai = sc_list[i]
        if len(ai) == 0:
            continue
        D[i, i] = 0.0
        for j in range(i + 1, n):
            aj = sc_list[j]
            if len(aj) == 0:
                continue
            # (ni, nj)
            dif = ai[:, None, :] - aj[None, :, :]
            mij = float(np.sqrt((dif * dif).sum(-1)).min())
            D[i, j] = D[j, i] = mij
    return D


def spatial_scores(D: np.ndarray, exposure: np.ndarray, hydro: np.ndarray, R: float) -> np.ndarray:
    """P_i = sum_j I[d_ij<=R] * exposure_j * hydro_j  (self included)."""
    valid = np.isfinite(exposure) & np.isfinite(hydro) & np.isfinite(np.diag(D) * 0 + 1)
    # treat invalid coords as non-neighbors
    contrib = exposure * hydro
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)
    neigh = (D <= R) & np.isfinite(D)
    # zero out invalid centers/neighbors with nan centroids -> inf distance already
    scores = neigh.astype(np.float64) @ contrib
    # invalidate centers with non-finite centroid (D[i,i] was set 0 only if valid)
    # marks: if centroid invalid, row was all-inf except we need flag
    return scores


def scope_mask(ab: dict, scope: str) -> np.ndarray:
    chain = np.asarray(ab["chain"])
    if scope == "ALL_FV":
        return np.ones(len(chain), dtype=bool)
    if scope == "HEAVY":
        return chain == "H"
    if scope == "LIGHT":
        return chain == "L"
    if scope == "ALL_CDR":
        return ab["is_cdr"]
    if scope == "H_CDR3":
        return ab["is_h_cdr3"]
    if scope == "L_CDR3":
        return ab["is_l_cdr3"]
    raise ValueError(scope)


def aggregate(scores: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    v = scores[mask & np.isfinite(scores)]
    if len(v) == 0:
        return {a: float("nan") for a in AGGS}
    out = {
        "MAX": float(v.max()),
        "MEAN": float(v.mean()),
        "SUM": float(v.sum()),
        "Q90": float(np.quantile(v, 0.90)),
        "Q95": float(np.quantile(v, 0.95)),
    }
    order = np.sort(v)[::-1]
    out["TOP3_MEAN"] = float(order[: min(3, len(order))].mean())
    out["TOP5_MEAN"] = float(order[: min(5, len(order))].mean())
    return out


def family_id(scale: str, transform: str, exposure: str, neigh: str, radius: float) -> str:
    rtag = str(radius).replace(".", "p")
    return f"HSP_{scale}_{transform}_{exposure}_{neigh}_R{rtag}"


def feature_name(family: str, scope: str, agg: str) -> str:
    return f"{family}__{scope}__{agg}"


def spec_hash(**kwargs) -> str:
    blob = json.dumps(kwargs, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
