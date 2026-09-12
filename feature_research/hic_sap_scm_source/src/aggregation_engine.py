"""Generic antibody-level aggregation: local scores → region × radius × stats."""
from __future__ import annotations

from typing import Sequence

import numpy as np

SOURCE_REGIONS = ("VH", "VL", "CDR", "FR")
EXTENSION_REGIONS = ("VH", "VL", "CDR", "FR", "ALL_FV")
SOURCE_RADII_A = (5.0, 10.0)
SOURCE_STATS = ("MAX", "TOP5_MEAN", "POSITIVE_SUM_MEAN")
EXTENSION_STATS = ("STD", "TOP5_SHARE_POSITIVE")


def region_masks(chain: Sequence[str], is_cdr: np.ndarray) -> dict[str, np.ndarray]:
    chain = np.asarray(chain)
    is_cdr = np.asarray(is_cdr, dtype=bool)
    return {
        "VH": chain == "H",
        "VL": chain == "L",
        "CDR": is_cdr.copy(),
        "FR": ~is_cdr,
        "ALL_FV": np.ones(len(chain), dtype=bool),
    }


def pairwise_centroid(centroids: np.ndarray) -> np.ndarray:
    d = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def local_scores(D: np.ndarray, prop: np.ndarray, rasa_clip: np.ndarray, R: float) -> np.ndarray:
    """score_i = Σ_j I[d_ij<=R] * prop_j * rasa_clip_j  (self included)."""
    contrib = prop * rasa_clip
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)
    neigh = (D <= R) & np.isfinite(D)
    scores = neigh.astype(np.float64) @ contrib
    return scores


def max_stat(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.max(v))


def top5_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    k = min(5, v.size)
    return float(np.mean(np.sort(v)[::-1][:k]))


def positive_sum_mean(values: np.ndarray) -> float:
    """mean(max(x,0)) over ALL valid residues N (SOURCE_CONFIRMED)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(v, 0.0)))


def population_std(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.std(v, ddof=0))


def top5_share_positive(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    pos = np.maximum(v, 0.0)
    total = float(np.sum(pos))
    if total == 0.0:
        return 0.0
    k = min(5, pos.size)
    return float(np.sum(np.sort(pos)[::-1][:k]) / total)


STAT_FN = {
    "MAX": max_stat,
    "TOP5_MEAN": top5_mean,
    "POSITIVE_SUM_MEAN": positive_sum_mean,
    "STD": population_std,
    "TOP5_SHARE_POSITIVE": top5_share_positive,
}


def rtag(R: float) -> str:
    return f"R{int(R)}" if float(R).is_integer() else f"R{R}"


def feature_names(prefix: str, regions: Sequence[str], radii: Sequence[float], stats: Sequence[str]) -> list[str]:
    names: list[str] = []
    for region in regions:
        for R in radii:
            for stat in stats:
                names.append(f"{prefix}_{region}_{rtag(R)}_{stat}")
    return names


SOURCE_SAP24_NAMES = feature_names("SAP", SOURCE_REGIONS, SOURCE_RADII_A, SOURCE_STATS)
SOURCE_SCM24_NAMES = feature_names("SCM", SOURCE_REGIONS, SOURCE_RADII_A, SOURCE_STATS)
SAP_GLOBAL6_NAMES = feature_names("SAP", ("ALL_FV",), SOURCE_RADII_A, SOURCE_STATS)
SCM_GLOBAL6_NAMES = feature_names("SCM", ("ALL_FV",), SOURCE_RADII_A, SOURCE_STATS)
SAP_EXTRA20_NAMES = feature_names("SAP", EXTENSION_REGIONS, SOURCE_RADII_A, EXTENSION_STATS)
SCM_EXTRA20_NAMES = feature_names("SCM", EXTENSION_REGIONS, SOURCE_RADII_A, EXTENSION_STATS)


def aggregate_block(
    scores_by_radius: dict[float, np.ndarray],
    masks: dict[str, np.ndarray],
    prefix: str,
    regions: Sequence[str],
    radii: Sequence[float],
    stats: Sequence[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for region in regions:
        mask = masks[region]
        n = int(np.sum(mask))
        if n == 0:
            raise RuntimeError(f"empty region {region} for {prefix}")
        for R in radii:
            scores = scores_by_radius[R]
            vals = scores[mask]
            # only finite among region; empty finite after filter still N>0 expected
            for stat in stats:
                name = f"{prefix}_{region}_{rtag(R)}_{stat}"
                out[name] = STAT_FN[stat](vals)
                if not np.isfinite(out[name]) and n > 0 and np.isfinite(vals).any():
                    # still allow nan only if all invalid scores in region
                    pass
                if not np.isfinite(out[name]) and not np.isfinite(vals).any():
                    raise RuntimeError(f"all-invalid scores in {name}")
    return out
