"""Generic antibody-level aggregation engine (geometry → regions → radii → stats).

SOURCE_SAP24 / SOURCE_SCM24 generation is gated on resolved source definitions.
Do not emit SOURCE_* feature matrices while fidelity gate fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

FIDELITY_GATE = {
    "SOURCE_SAP24": "BLOCKED_SOURCE_UNRESOLVED",
    "SOURCE_SCM24": "BLOCKED_SOURCE_UNRESOLVED",
    "blocking": ("positive_sum_mean", "scm_charge_semantics"),
}

SOURCE_REGIONS = ("VH", "VL", "CDR", "FR")
SOURCE_RADII_A = (5.0, 10.0)
SOURCE_STATS = ("MAX", "TOP5_MEAN", "POSITIVE_SUM_MEAN")
EXTENSION_REGIONS = ("VH", "VL", "CDR", "FR", "ALL_FV")
EXTENSION_STATS = ("STD", "TOP5_SHARE")


@dataclass(frozen=True)
class AggregationConfig:
    regions: Sequence[str]
    radii_a: Sequence[float]
    statistics: Sequence[str]


SOURCE24_CONFIG = AggregationConfig(
    regions=SOURCE_REGIONS,
    radii_a=SOURCE_RADII_A,
    statistics=SOURCE_STATS,
)


def assert_source_fidelity_allows(block: str) -> None:
    status = FIDELITY_GATE.get(block)
    if status and status.startswith("BLOCKED"):
        raise RuntimeError(
            f"{block} is {status}; unresolved: {FIDELITY_GATE['blocking']}. "
            "See results/SOURCE_SPEC_AUDIT.md — do not invent formulas."
        )


def top5_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    k = min(5, v.size)
    return float(np.mean(np.partition(v, -k)[-k:]))


def population_std(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.std(v, ddof=0))


def top5_share_nonnegative(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    if np.any(v < 0):
        raise ValueError("TOP5_SHARE requires nonnegative channel")
    denom = float(np.sum(v))
    if denom == 0.0:
        return 0.0
    k = min(5, v.size)
    return float(np.sum(np.partition(v, -k)[-k:]) / denom)


def positive_sum_mean_unresolved(*_args, **_kwargs) -> float:
    """Placeholder — formula not source-confirmed."""
    raise NotImplementedError(
        "positive_sum_mean is UNRESOLVED; see SOURCE_SPEC_AUDIT.md"
    )


STAT_FN: dict[str, Callable[[np.ndarray], float]] = {
    "MAX": lambda v: float(np.nanmax(v)) if np.isfinite(v).any() else float("nan"),
    "TOP5_MEAN": top5_mean,
    "POSITIVE_SUM_MEAN": positive_sum_mean_unresolved,
    "STD": population_std,
    "TOP5_SHARE": top5_share_nonnegative,
}


def feature_names(prefix: str, cfg: AggregationConfig) -> list[str]:
    names: list[str] = []
    for region in cfg.regions:
        for r in cfg.radii_a:
            rtag = f"R{int(r)}" if float(r).is_integer() else f"R{r}"
            for stat in cfg.statistics:
                names.append(f"{prefix}_{region}_{rtag}_{stat}")
    return names


def expected_source24_dim() -> int:
    return len(SOURCE_REGIONS) * len(SOURCE_RADII_A) * len(SOURCE_STATS)


def aggregate_region_scores(
    per_residue_scores: np.ndarray,
    mask: np.ndarray,
    statistic: str,
) -> float:
    """Aggregate one region for one radius column of per-residue scores."""
    assert_source_fidelity_allows("SOURCE_SAP24")  # always blocked until audit clears
    fn = STAT_FN[statistic]
    return fn(per_residue_scores[mask.astype(bool)])
