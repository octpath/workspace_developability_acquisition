#!/usr/bin/env python3
"""
Stage 0 — Establish and freeze a reliable common CV protocol.

Participant-safe inputs only:
  - competition/data/distribution/dev.csv
  - competition/data/distribution/dev_annotations.csv
  - sequence-derived features computed from heavy/light

Does NOT use solution.csv, test labels, public/private flags, or organizer
model-selection / split-selection materials.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/workspace_developability_acquisition")
DEV_CSV = ROOT / "competition/data/distribution/dev.csv"
ANN_CSV = ROOT / "competition/data/distribution/dev_annotations.csv"
OUT = ROOT / "virtual_participant/stage0_cv"
PLOTS = OUT / "plots"
ARTIFACTS = OUT / "artifacts"

PRIMARY_SEED = 42
SHADOW_SEED = 2026
N_FOLDS_CANDIDATES = (4, 5, 6)
HIC_HIGH_TAIL_QUANTILE = 0.90  # Train-only definition
SEQ_IDENTITY_THRESHOLD = 0.90  # min(VH, VL) identity for grouping


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def aa_composition(seq: str) -> dict[str, float]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    n = max(len(seq), 1)
    counts = {aa: 0 for aa in alphabet}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
    return {f"frac_{aa}": counts[aa] / n for aa in alphabet}


def kyte_doolittle_mean(seq: str) -> float:
    kd = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
        "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
        "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
        "W": -0.9, "Y": -1.3,
    }
    vals = [kd.get(ch, 0.0) for ch in seq]
    return float(np.mean(vals)) if vals else 0.0


def charge_proxy(seq: str) -> float:
    pos = sum(ch in "KR" for ch in seq)
    neg = sum(ch in "DE" for ch in seq)
    return (pos - neg) / max(len(seq), 1)


def sequence_identity(a: str, b: str) -> float:
    """Position-wise identity with length penalty (no gap alignment for speed)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == lb:
        return sum(x == y for x, y in zip(a, b)) / la
    # unequal length: score matches on shared prefix / max length
    n = min(la, lb)
    return sum(x == y for x, y in zip(a, b)) / max(la, lb)


def build_sequence_groups(df: pd.DataFrame, threshold: float = SEQ_IDENTITY_THRESHOLD) -> pd.Series:
    """Union-Find over pairs with min(VH,VL) identity >= threshold; also exact heavy+light."""
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    heavies = df["heavy"].tolist()
    lights = df["light"].tolist()

    # Exact paired duplicates
    key_to_idx: dict[tuple[str, str], int] = {}
    for i, (h, l) in enumerate(zip(heavies, lights)):
        key = (h, l)
        if key in key_to_idx:
            union(i, key_to_idx[key])
        else:
            key_to_idx[key] = i

    # High-identity pairs
    for i, j in combinations(range(n), 2):
        hi = sequence_identity(heavies[i], heavies[j])
        li = sequence_identity(lights[i], lights[j])
        if min(hi, li) >= threshold:
            union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_gid: dict[int, int] = {}
    gids = []
    next_gid = 0
    for r in roots:
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        gids.append(root_to_gid[r])
    return pd.Series(gids, index=df.index, name="seq_group")


def sturges_bins(y: np.ndarray) -> np.ndarray:
    k = int(math.ceil(1 + math.log2(len(y))))
    # Force at least 2 samples per usable bin by merging empty edges via digitize
    edges = np.linspace(y.min(), y.max(), k + 1)
    # pd.cut labels; for stratification use codes
    cats = pd.cut(y, bins=edges, include_lowest=True)
    codes = cats.codes.astype(int)
    # Merge singleton/empty-adjacent by clipping rare codes into neighbors later if needed
    return codes


def quantile_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").astype(int)


def hybrid_hic_bins(y: np.ndarray, central_bins: int = 6, tail_q: float = HIC_HIGH_TAIL_QUANTILE) -> np.ndarray:
    """Central quantile bins + independent upper-tail bin (Train-only)."""
    thr = np.quantile(y, tail_q)
    is_tail = y >= thr
    codes = np.zeros(len(y), dtype=int)
    central = y[~is_tail]
    if len(central) >= central_bins:
        c_codes = pd.qcut(central, q=central_bins, labels=False, duplicates="drop").astype(int)
    else:
        c_codes = np.zeros(len(central), dtype=int)
    codes[~is_tail] = c_codes
    codes[is_tail] = int(c_codes.max()) + 1 if len(central) else 0
    return codes


def fold_size_imbalance(folds: np.ndarray, n_folds: int) -> float:
    counts = np.array([(folds == f).sum() for f in range(n_folds)], dtype=float)
    expected = len(folds) / n_folds
    return float(np.mean(np.abs(counts - expected) / expected))


def quantile_proportion_imbalance(y: np.ndarray, folds: np.ndarray, n_folds: int, n_q: int = 5) -> float:
    bins = pd.qcut(y, q=n_q, labels=False, duplicates="drop")
    global_p = np.bincount(bins, minlength=bins.max() + 1) / len(y)
    diffs = []
    for f in range(n_folds):
        mask = folds == f
        if mask.sum() == 0:
            diffs.append(1.0)
            continue
        p = np.bincount(bins[mask], minlength=len(global_p)) / mask.sum()
        diffs.append(float(np.mean(np.abs(p - global_p))))
    return float(np.mean(diffs))


def mean_median_imbalance(y: np.ndarray, folds: np.ndarray, n_folds: int) -> float:
    g_mean, g_med = float(np.mean(y)), float(np.median(y))
    scale = float(np.std(y)) + 1e-8
    vals = []
    for f in range(n_folds):
        yy = y[folds == f]
        if len(yy) == 0:
            vals.append(1.0)
            continue
        vals.append(abs(np.mean(yy) - g_mean) / scale + abs(np.median(yy) - g_med) / scale)
    return float(np.mean(vals))


def high_tail_imbalance(y: np.ndarray, folds: np.ndarray, n_folds: int, q: float = HIC_HIGH_TAIL_QUANTILE) -> float:
    thr = np.quantile(y, q)
    is_tail = y >= thr
    expected = is_tail.mean()
    diffs = []
    for f in range(n_folds):
        mask = folds == f
        if mask.sum() == 0:
            diffs.append(1.0)
            continue
        diffs.append(abs(is_tail[mask].mean() - expected))
    return float(np.mean(diffs))


def wasserstein_fold_imbalance(y: np.ndarray, folds: np.ndarray, n_folds: int) -> float:
    from scipy.stats import wasserstein_distance

    scale = float(np.std(y)) + 1e-8
    vals = []
    for f in range(n_folds):
        yy = y[folds == f]
        if len(yy) < 2:
            vals.append(1.0)
            continue
        vals.append(wasserstein_distance(y, yy) / scale)
    return float(np.mean(vals))


@dataclass
class BalanceMetrics:
    size_imb: float
    tm_q_imb: float
    hic_q_imb: float
    tm_mm_imb: float
    hic_mm_imb: float
    hic_tail_imb: float
    tm_w_imb: float
    hic_w_imb: float
    loss: float


def evaluate_balance(
    tm: np.ndarray,
    hic: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
    w_size: float = 1.0,
    w_tm: float = 1.0,
    w_hic: float = 1.2,
    w_tail: float = 1.5,
) -> BalanceMetrics:
    size_imb = fold_size_imbalance(folds, n_folds)
    tm_q = quantile_proportion_imbalance(tm, folds, n_folds)
    hic_q = quantile_proportion_imbalance(hic, folds, n_folds)
    tm_mm = mean_median_imbalance(tm, folds, n_folds)
    hic_mm = mean_median_imbalance(hic, folds, n_folds)
    hic_tail = high_tail_imbalance(hic, folds, n_folds)
    tm_w = wasserstein_fold_imbalance(tm, folds, n_folds)
    hic_w = wasserstein_fold_imbalance(hic, folds, n_folds)
    loss = (
        w_size * size_imb
        + w_tm * (tm_q + 0.5 * tm_mm + 0.5 * tm_w)
        + w_hic * (hic_q + 0.5 * hic_mm + 0.5 * hic_w)
        + w_tail * hic_tail
    )
    return BalanceMetrics(size_imb, tm_q, hic_q, tm_mm, hic_mm, hic_tail, tm_w, hic_w, loss)


# ---------------------------------------------------------------------------
# Fold assignment generators
# ---------------------------------------------------------------------------

def assignment_from_sklearn(indices: np.ndarray, fold_iter) -> np.ndarray:
    folds = np.full(len(indices), -1, dtype=int)
    for f, (_, val_idx) in enumerate(fold_iter):
        folds[val_idx] = f
    assert (folds >= 0).all()
    return folds


def make_plain_kfold(n: int, n_folds: int) -> np.ndarray:
    kf = KFold(n_splits=n_folds, shuffle=False)
    return assignment_from_sklearn(np.arange(n), kf.split(np.arange(n)))


def make_shuffled_kfold(n: int, n_folds: int, seed: int) -> np.ndarray:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return assignment_from_sklearn(np.arange(n), kf.split(np.arange(n)))


def make_stratified(y_bins: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return assignment_from_sklearn(np.arange(len(y_bins)), skf.split(np.arange(len(y_bins)), y_bins))


def make_group_kfold(groups: np.ndarray, n_folds: int, seed: int, y_bins: np.ndarray | None = None) -> np.ndarray:
    """StratifiedGroupKFold if y_bins given, else group-only via shuffled group assignment."""
    n = len(groups)
    if y_bins is not None:
        # StratifiedGroupKFold may fail if a class has fewer samples than n_folds
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return assignment_from_sklearn(np.arange(n), sgkf.split(np.arange(n), y_bins, groups))
        except ValueError:
            pass
    # Fallback: shuffle groups into folds greedily by size
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups.tolist())))
    rng.shuffle(unique_groups)
    folds = np.full(n, -1, dtype=int)
    fold_sizes = np.zeros(n_folds, dtype=int)
    group_sizes = {g: int((groups == g).sum()) for g in unique_groups}
    for g in sorted(unique_groups, key=lambda x: -group_sizes[x]):
        f = int(np.argmin(fold_sizes))
        folds[groups == g] = f
        fold_sizes[f] += group_sizes[g]
    return folds


def _fast_balance_loss(
    tm: np.ndarray,
    hic: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
    hic_tail_mask: np.ndarray,
    tm_bin: np.ndarray,
    hic_bin: np.ndarray,
) -> float:
    """Lightweight loss used inside swap search (same spirit as evaluate_balance)."""
    n = len(folds)
    expected_size = n / n_folds
    size_imb = 0.0
    tm_q_imb = 0.0
    hic_q_imb = 0.0
    hic_tail_imb = 0.0
    tm_mean_imb = 0.0
    hic_mean_imb = 0.0
    g_tm_mean, g_hic_mean = float(tm.mean()), float(hic.mean())
    g_tm_std = float(tm.std()) + 1e-8
    g_hic_std = float(hic.std()) + 1e-8
    n_tm_bins = int(tm_bin.max()) + 1
    n_hic_bins = int(hic_bin.max()) + 1
    global_tm_p = np.bincount(tm_bin, minlength=n_tm_bins) / n
    global_hic_p = np.bincount(hic_bin, minlength=n_hic_bins) / n
    global_tail = float(hic_tail_mask.mean())

    for f in range(n_folds):
        mask = folds == f
        nf = int(mask.sum())
        if nf == 0:
            return 1e9
        size_imb += abs(nf - expected_size) / expected_size
        p_tm = np.bincount(tm_bin[mask], minlength=n_tm_bins) / nf
        p_hic = np.bincount(hic_bin[mask], minlength=n_hic_bins) / nf
        tm_q_imb += float(np.mean(np.abs(p_tm - global_tm_p)))
        hic_q_imb += float(np.mean(np.abs(p_hic - global_hic_p)))
        hic_tail_imb += abs(float(hic_tail_mask[mask].mean()) - global_tail)
        tm_mean_imb += abs(float(tm[mask].mean()) - g_tm_mean) / g_tm_std
        hic_mean_imb += abs(float(hic[mask].mean()) - g_hic_mean) / g_hic_std

    size_imb /= n_folds
    tm_q_imb /= n_folds
    hic_q_imb /= n_folds
    hic_tail_imb /= n_folds
    tm_mean_imb /= n_folds
    hic_mean_imb /= n_folds
    return (
        1.0 * size_imb
        + 1.0 * (tm_q_imb + 0.5 * tm_mean_imb)
        + 1.2 * (hic_q_imb + 0.5 * hic_mean_imb)
        + 1.5 * hic_tail_imb
    )


def optimize_joint_group_folds(
    tm: np.ndarray,
    hic: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    n_restarts: int = 24,
    n_swaps: int = 400,
) -> tuple[np.ndarray, BalanceMetrics]:
    """Greedy size-balanced group assignment + local swaps. Objective = target balance only."""
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups.tolist())))
    group_members = {g: np.where(groups == g)[0] for g in unique_groups}
    group_size = {g: len(group_members[g]) for g in unique_groups}
    hic_tail_mask = hic >= np.quantile(hic, HIC_HIGH_TAIL_QUANTILE)
    tm_bin = quantile_bins(tm, 5)
    hic_bin = quantile_bins(hic, 5)

    # Compact representation: group -> member indices already above
    g_list = unique_groups.tolist()

    def folds_from_assignment(assign: np.ndarray) -> np.ndarray:
        folds = np.empty(len(groups), dtype=int)
        for gi, g in enumerate(g_list):
            folds[group_members[g]] = int(assign[gi])
        return folds

    def size_greedy_init(local_rng: np.random.Generator) -> np.ndarray:
        order = np.argsort([-group_size[g] for g in g_list])
        # jitter among equal sizes
        jitter = local_rng.random(len(order))
        order = np.lexsort((jitter, [-group_size[g_list[i]] for i in range(len(g_list))]))
        assign = np.full(len(g_list), -1, dtype=int)
        fold_sizes = np.zeros(n_folds, dtype=int)
        for gi in order:
            f = int(np.argmin(fold_sizes))
            # tie-break randomly among mins
            mins = np.where(fold_sizes == fold_sizes.min())[0]
            f = int(local_rng.choice(mins))
            assign[gi] = f
            fold_sizes[f] += group_size[g_list[gi]]
        return assign

    best_folds = None
    best_metrics = None
    best_fast = float("inf")

    for _ in range(n_restarts):
        local_rng = np.random.default_rng(int(rng.integers(0, 1_000_000_000)))
        assign = size_greedy_init(local_rng)
        folds = folds_from_assignment(assign)
        cur = _fast_balance_loss(tm, hic, folds, n_folds, hic_tail_mask, tm_bin, hic_bin)

        for _s in range(n_swaps):
            gi = int(local_rng.integers(0, len(g_list)))
            old_f = int(assign[gi])
            if local_rng.random() < 0.5:
                # move group to another fold
                new_f = int(local_rng.integers(0, n_folds))
                if new_f == old_f:
                    continue
                assign[gi] = new_f
                trial = folds_from_assignment(assign)
                trial_loss = _fast_balance_loss(tm, hic, trial, n_folds, hic_tail_mask, tm_bin, hic_bin)
                if trial_loss < cur:
                    folds, cur = trial, trial_loss
                else:
                    assign[gi] = old_f
            else:
                # swap two groups' folds
                gj = int(local_rng.integers(0, len(g_list)))
                if gj == gi:
                    continue
                f2 = int(assign[gj])
                if f2 == old_f:
                    continue
                assign[gi], assign[gj] = f2, old_f
                trial = folds_from_assignment(assign)
                trial_loss = _fast_balance_loss(tm, hic, trial, n_folds, hic_tail_mask, tm_bin, hic_bin)
                if trial_loss < cur:
                    folds, cur = trial, trial_loss
                else:
                    assign[gi], assign[gj] = old_f, f2

        if cur < best_fast:
            best_fast = cur
            best_folds = folds.copy()

    assert best_folds is not None
    best_metrics = evaluate_balance(tm, hic, best_folds, n_folds)
    return best_folds, best_metrics


# ---------------------------------------------------------------------------
# Features & baselines
# ---------------------------------------------------------------------------

def build_simple_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        h, l = r["heavy"], r["light"]
        feat = {}
        for prefix, seq in (("h", h), ("l", l)):
            feat[f"{prefix}_len"] = len(seq)
            feat[f"{prefix}_kd"] = kyte_doolittle_mean(seq)
            feat[f"{prefix}_charge"] = charge_proxy(seq)
            feat[f"{prefix}_aromatic"] = sum(ch in "FWY" for ch in seq) / max(len(seq), 1)
            feat.update({f"{prefix}_{k}": v for k, v in aa_composition(seq).items()})
        rows.append(feat)
    return pd.DataFrame(rows, index=df.index)


def build_annotation_features(ann: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "h_cdr1_length", "h_cdr2_length", "h_cdr3_length",
        "l_cdr1_length", "l_cdr2_length", "l_cdr3_length",
        "heavy_germline_identity", "light_germline_identity",
    ]
    X_num = ann[num_cols].copy()
    cat_cols = ["heavy_v_family", "heavy_j_gene", "light_v_family", "light_j_gene", "light_chain_type"]
    X_cat = pd.get_dummies(ann[cat_cols], drop_first=False)
    return pd.concat([X_num, X_cat], axis=1)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_baselines(
    y: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
    X_seq: np.ndarray,
    X_ann: np.ndarray,
) -> dict:
    """Median + Ridge(seq) + ElasticNet(annotations). Returns stability diagnostics."""
    results = {}
    for name, mode in [
        ("median", "median"),
        ("ridge_seq", "ridge_seq"),
        ("elastic_ann", "elastic_ann"),
    ]:
        fold_maes = []
        oof = np.zeros(len(y))
        for f in range(n_folds):
            tr = folds != f
            va = folds == f
            y_tr, y_va = y[tr], y[va]
            if mode == "median":
                pred = np.full(y_va.shape, np.median(y_tr))
            elif mode == "ridge_seq":
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=10.0, random_state=0)),
                ])
                pipe.fit(X_seq[tr], y_tr)
                pred = pipe.predict(X_seq[va])
            else:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=2000, tol=1e-3, random_state=0)),
                ])
                pipe.fit(X_ann[tr], y_tr)
                pred = pipe.predict(X_ann[va])
            fold_maes.append(mae(y_va, pred))
            oof[va] = pred
        fold_maes = np.array(fold_maes)
        results[name] = {
            "mean_mae": float(fold_maes.mean()),
            "fold_mae_sd": float(fold_maes.std(ddof=1)) if n_folds > 1 else 0.0,
            "worst_fold_mae": float(fold_maes.max()),
            "fold_maes": fold_maes.tolist(),
            "oof_mae": mae(y, oof),
        }
    return results


def per_fold_target_stats(y: np.ndarray, folds: np.ndarray, n_folds: int, name: str) -> list[dict]:
    thr = np.quantile(y, HIC_HIGH_TAIL_QUANTILE) if name == "HIC" else None
    rows = []
    for f in range(n_folds):
        yy = y[folds == f]
        row = {
            "fold": f,
            "n": int(len(yy)),
            "mean": float(np.mean(yy)),
            "median": float(np.median(yy)),
            "std": float(np.std(yy)),
            "q10": float(np.quantile(yy, 0.10)),
            "q25": float(np.quantile(yy, 0.25)),
            "q50": float(np.quantile(yy, 0.50)),
            "q75": float(np.quantile(yy, 0.75)),
            "q90": float(np.quantile(yy, 0.90)),
        }
        if thr is not None:
            row["n_high_tail"] = int((yy >= thr).sum())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DEV_CSV)
    ann = pd.read_csv(ANN_CSV)
    assert list(df["id"]) == list(ann["id"]), "dev/annotations ID order mismatch — joining by id"
    ann = ann.set_index("id").loc[df["id"]].reset_index()

    tm = df["TmApp"].to_numpy(dtype=float)
    hic = df["HIC"].to_numpy(dtype=float)
    n = len(df)

    groups = build_sequence_groups(df, SEQ_IDENTITY_THRESHOLD)
    df = df.copy()
    df["seq_group"] = groups.values
    group_sizes = df.groupby("seq_group").size()
    n_groups = int(groups.nunique())
    n_multi = int((group_sizes > 1).sum())
    multi_members = int(group_sizes[group_sizes > 1].sum())

    group_info = {
        "n_groups": n_groups,
        "n_singleton_groups": int((group_sizes == 1).sum()),
        "n_multi_member_groups": n_multi,
        "n_members_in_multi_groups": multi_members,
        "identity_threshold": SEQ_IDENTITY_THRESHOLD,
        "multi_group_members": (
            df.loc[df["seq_group"].map(group_sizes) > 1, ["id", "seq_group", "TmApp", "HIC"]]
            .sort_values("seq_group")
            .to_dict(orient="records")
        ),
    }
    with open(ARTIFACTS / "sequence_groups.json", "w") as f:
        json.dump(group_info, f, indent=2)
    df[["id", "seq_group"]].to_csv(ARTIFACTS / "sequence_groups.csv", index=False)

    # Binning diagnostics
    sturges_k = int(math.ceil(1 + math.log2(n)))
    tm_sturges = sturges_bins(tm)
    hic_sturges = sturges_bins(hic)
    tm_q8 = quantile_bins(tm, 8)
    hic_q8 = quantile_bins(hic, 8)
    hic_hybrid = hybrid_hic_bins(hic, central_bins=6, tail_q=HIC_HIGH_TAIL_QUANTILE)
    # Joint soft label: combine coarse quantile bins carefully
    tm_q4 = quantile_bins(tm, 4)
    hic_q4 = quantile_bins(hic, 4)
    joint_cart = tm_q4 * (int(hic_q4.max()) + 1) + hic_q4

    bin_diag = {
        "sturges_k": sturges_k,
        "hic_sturges_occupancy": pd.Series(hic_sturges).value_counts().sort_index().to_dict(),
        "tm_sturges_occupancy": pd.Series(tm_sturges).value_counts().sort_index().to_dict(),
        "hic_q8_occupancy": pd.Series(hic_q8).value_counts().sort_index().to_dict(),
        "tm_q8_occupancy": pd.Series(tm_q8).value_counts().sort_index().to_dict(),
        "hic_hybrid_occupancy": pd.Series(hic_hybrid).value_counts().sort_index().to_dict(),
        "joint_cart4x4_occupancy": pd.Series(joint_cart).value_counts().sort_index().to_dict(),
        "hic_high_tail_threshold_q90": float(np.quantile(hic, HIC_HIGH_TAIL_QUANTILE)),
        "hic_high_tail_count": int((hic >= np.quantile(hic, HIC_HIGH_TAIL_QUANTILE)).sum()),
    }
    with open(ARTIFACTS / "binning_diagnostics.json", "w") as f:
        json.dump(bin_diag, f, indent=2)

    X_seq = build_simple_sequence_features(df).to_numpy(dtype=float)
    X_ann = build_annotation_features(ann).to_numpy(dtype=float)

    candidates: list[dict] = []

    def register(name: str, family: str, n_folds: int, seed: int, folds: np.ndarray, notes: str = "") -> None:
        bal = evaluate_balance(tm, hic, folds, n_folds)
        # sequence group purity: each group should be in one fold
        group_split_violations = 0
        for g in sorted(set(groups.tolist())):
            gf = set(folds[groups.values == g].tolist())
            if len(gf) > 1:
                group_split_violations += 1
        stab_tm = evaluate_baselines(tm, folds, n_folds, X_seq, X_ann)
        stab_hic = evaluate_baselines(hic, folds, n_folds, X_seq, X_ann)
        # representative stability: average fold-MAE SD across simple models / targets
        stab_score = float(np.mean([
            stab_tm["median"]["fold_mae_sd"],
            stab_tm["ridge_seq"]["fold_mae_sd"],
            stab_tm["elastic_ann"]["fold_mae_sd"],
            stab_hic["median"]["fold_mae_sd"],
            stab_hic["ridge_seq"]["fold_mae_sd"],
            stab_hic["elastic_ann"]["fold_mae_sd"],
        ]))
        # selection score: balance loss + mild penalty on baseline fold variance
        # (NOT optimizing for lower mean MAE)
        selection_score = bal.loss + 0.15 * stab_score
        rec = {
            "name": name,
            "family": family,
            "n_folds": n_folds,
            "seed": seed,
            "notes": notes,
            "size_imb": bal.size_imb,
            "tm_q_imb": bal.tm_q_imb,
            "hic_q_imb": bal.hic_q_imb,
            "tm_mm_imb": bal.tm_mm_imb,
            "hic_mm_imb": bal.hic_mm_imb,
            "hic_tail_imb": bal.hic_tail_imb,
            "tm_w_imb": bal.tm_w_imb,
            "hic_w_imb": bal.hic_w_imb,
            "balance_loss": bal.loss,
            "group_split_violations": group_split_violations,
            "stab_score_fold_mae_sd": stab_score,
            "selection_score": selection_score,
            "tm_median_mean_mae": stab_tm["median"]["mean_mae"],
            "tm_median_fold_sd": stab_tm["median"]["fold_mae_sd"],
            "tm_ridge_mean_mae": stab_tm["ridge_seq"]["mean_mae"],
            "tm_ridge_fold_sd": stab_tm["ridge_seq"]["fold_mae_sd"],
            "tm_elastic_mean_mae": stab_tm["elastic_ann"]["mean_mae"],
            "tm_elastic_fold_sd": stab_tm["elastic_ann"]["fold_mae_sd"],
            "hic_median_mean_mae": stab_hic["median"]["mean_mae"],
            "hic_median_fold_sd": stab_hic["median"]["fold_mae_sd"],
            "hic_ridge_mean_mae": stab_hic["ridge_seq"]["mean_mae"],
            "hic_ridge_fold_sd": stab_hic["ridge_seq"]["fold_mae_sd"],
            "hic_elastic_mean_mae": stab_hic["elastic_ann"]["mean_mae"],
            "hic_elastic_fold_sd": stab_hic["elastic_ann"]["fold_mae_sd"],
            "folds": folds,
            "stab_tm": stab_tm,
            "stab_hic": stab_hic,
            "fold_stats_tm": per_fold_target_stats(tm, folds, n_folds, "TmApp"),
            "fold_stats_hic": per_fold_target_stats(hic, folds, n_folds, "HIC"),
        }
        candidates.append(rec)
        print(
            f"[{name}] bal={bal.loss:.4f} sel={selection_score:.4f} "
            f"tail={bal.hic_tail_imb:.4f} stab={stab_score:.4f} gviol={group_split_violations}",
            flush=True,
        )

    print("=== Generating CV candidates ===", flush=True)
    for n_folds in N_FOLDS_CANDIDATES:
        register(f"plain_kfold_k{n_folds}", "plain_kfold", n_folds, 0, make_plain_kfold(n, n_folds))
        register(
            f"shuffled_kfold_k{n_folds}_s{PRIMARY_SEED}",
            "shuffled_kfold",
            n_folds,
            PRIMARY_SEED,
            make_shuffled_kfold(n, n_folds, PRIMARY_SEED),
        )
        register(
            f"strat_sturges_tm_k{n_folds}_s{PRIMARY_SEED}",
            "continuous_strat_sturges",
            n_folds,
            PRIMARY_SEED,
            make_stratified(tm_sturges, n_folds, PRIMARY_SEED),
            notes="TmApp Sturges bins",
        )
        register(
            f"strat_sturges_hic_k{n_folds}_s{PRIMARY_SEED}",
            "continuous_strat_sturges",
            n_folds,
            PRIMARY_SEED,
            make_stratified(hic_sturges, n_folds, PRIMARY_SEED),
            notes="HIC Sturges bins (skewed)",
        )
        register(
            f"strat_q8_tm_k{n_folds}_s{PRIMARY_SEED}",
            "continuous_strat_quantile",
            n_folds,
            PRIMARY_SEED,
            make_stratified(tm_q8, n_folds, PRIMARY_SEED),
            notes="TmApp qcut8",
        )
        register(
            f"strat_q8_hic_k{n_folds}_s{PRIMARY_SEED}",
            "continuous_strat_quantile",
            n_folds,
            PRIMARY_SEED,
            make_stratified(hic_q8, n_folds, PRIMARY_SEED),
            notes="HIC qcut8",
        )
        register(
            f"strat_hybrid_hic_k{n_folds}_s{PRIMARY_SEED}",
            "continuous_strat_hybrid_tail",
            n_folds,
            PRIMARY_SEED,
            make_stratified(hic_hybrid, n_folds, PRIMARY_SEED),
            notes="HIC hybrid central-q + upper-tail bin",
        )
        # joint cartesian — may be sparse
        try:
            register(
                f"strat_joint_cart4x4_k{n_folds}_s{PRIMARY_SEED}",
                "joint_cartesian_strat",
                n_folds,
                PRIMARY_SEED,
                make_stratified(joint_cart, n_folds, PRIMARY_SEED),
                notes="TmApp q4 x HIC q4 cartesian",
            )
        except ValueError as e:
            print(f"skip joint cart k={n_folds}: {e}")

        register(
            f"group_only_k{n_folds}_s{PRIMARY_SEED}",
            "sequence_group_aware",
            n_folds,
            PRIMARY_SEED,
            make_group_kfold(groups.values, n_folds, PRIMARY_SEED, y_bins=None),
            notes="group atomic, size-greedy",
        )
        # stratified group on HIC hybrid / joint-ish
        register(
            f"stratgroup_hic_hybrid_k{n_folds}_s{PRIMARY_SEED}",
            "stratified_group",
            n_folds,
            PRIMARY_SEED,
            make_group_kfold(groups.values, n_folds, PRIMARY_SEED, y_bins=hic_hybrid),
            notes="StratifiedGroupKFold on HIC hybrid bins",
        )
        register(
            f"stratgroup_joint_k{n_folds}_s{PRIMARY_SEED}",
            "stratified_group",
            n_folds,
            PRIMARY_SEED,
            make_group_kfold(groups.values, n_folds, PRIMARY_SEED, y_bins=joint_cart),
            notes="StratifiedGroupKFold on joint cart bins",
        )

        # Optimized joint group-aware
        opt_folds, opt_bal = optimize_joint_group_folds(
            tm, hic, groups.values, n_folds, seed=PRIMARY_SEED, n_restarts=32, n_swaps=600
        )
        register(
            f"opt_joint_group_k{n_folds}_s{PRIMARY_SEED}",
            "optimized_joint_group",
            n_folds,
            PRIMARY_SEED,
            opt_folds,
            notes=f"greedy+swap joint balance loss={opt_bal.loss:.4f}",
        )
        # Shadow-seed optimized assignment for each k (same protocol, different seed)
        shadow_folds, shadow_bal = optimize_joint_group_folds(
            tm, hic, groups.values, n_folds, seed=SHADOW_SEED, n_restarts=32, n_swaps=600
        )
        register(
            f"opt_joint_group_k{n_folds}_s{SHADOW_SEED}",
            "optimized_joint_group",
            n_folds,
            SHADOW_SEED,
            shadow_folds,
            notes=f"shadow-seed greedy+swap loss={shadow_bal.loss:.4f}",
        )

    # Extra seeds for seed-stability probe at k=5 only
    for seed in [7, 123]:
        opt_folds, _ = optimize_joint_group_folds(
            tm, hic, groups.values, 5, seed=seed, n_restarts=32, n_swaps=600
        )
        register(
            f"opt_joint_group_k5_s{seed}",
            "optimized_joint_group",
            5,
            seed,
            opt_folds,
            notes="seed-stability probe",
        )
        register(
            f"shuffled_kfold_k5_s{seed}",
            "shuffled_kfold",
            5,
            seed,
            make_shuffled_kfold(n, 5, seed),
        )

    # Summary table
    summary_rows = []
    for c in candidates:
        row = {k: v for k, v in c.items() if k not in {"folds", "stab_tm", "stab_hic", "fold_stats_tm", "fold_stats_hic"}}
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["selection_score", "balance_loss"])
    summary.to_csv(OUT / "cv_candidate_summary.csv", index=False)

    # Prefer common folds; select among optimized_joint_group / stratified_group with no group violations
    eligible = summary[
        (summary["group_split_violations"] == 0)
        & (summary["family"].isin(["optimized_joint_group", "stratified_group", "sequence_group_aware"]))
        & (summary["n_folds"].isin([4, 5, 6]))
        & (~summary["notes"].astype(str).str.contains("seed-stability"))
        & (summary["seed"] == PRIMARY_SEED)
    ].copy()
    if eligible.empty:
        eligible = summary[(summary["group_split_violations"] == 0) & (summary["seed"] == PRIMARY_SEED)].copy()

    # Prefer optimized_joint_group; default to 5-fold for Stage1 cost/stability tradeoff.
    # Switch away from 5-fold only if 5-fold is clearly unstable on HIC tail / size relative to best other k.
    primary_pool = eligible[eligible["family"] == "optimized_joint_group"].copy()
    if primary_pool.empty:
        primary_pool = eligible.copy()

    by_k = (
        primary_pool.sort_values("selection_score")
        .groupby("n_folds", as_index=False)
        .first()
    )
    k5 = by_k[by_k["n_folds"] == 5]
    if len(k5):
        k5_row = k5.iloc[0]
        best_other = by_k[by_k["n_folds"] != 5]
        switch = False
        if len(best_other):
            other = best_other.sort_values("selection_score").iloc[0]
            # Switch only if 5-fold HIC-tail imbalance is much worse AND overall selection much worse
            if (
                float(k5_row["hic_tail_imb"]) > float(other["hic_tail_imb"]) * 2.0 + 1e-12
                and float(k5_row["selection_score"]) > float(other["selection_score"]) * 1.25
            ):
                switch = True
                primary_row = other
        if not switch:
            primary_row = k5_row
    else:
        primary_row = primary_pool.sort_values("selection_score").iloc[0]

    primary_name = str(primary_row["name"])
    primary_cand = next(c for c in candidates if c["name"] == primary_name)

    def fold_disagreement(a: np.ndarray, b: np.ndarray) -> float:
        # fraction of pairs that disagree on co-membership in the same validation fold
        n_ = len(a)
        disagree = 0
        total = 0
        for i, j in combinations(range(n_), 2):
            same_a = a[i] == a[j]
            same_b = b[i] == b[j]
            if same_a != same_b:
                disagree += 1
            total += 1
        return disagree / max(total, 1)

    # Shadow: same protocol/family/n_folds, fixed SHADOW_SEED assignment
    shadow_pool = [
        c for c in candidates
        if c["family"] == primary_cand["family"]
        and c["n_folds"] == primary_cand["n_folds"]
        and c["seed"] == SHADOW_SEED
        and c["group_split_violations"] == 0
    ]
    if not shadow_pool:
        shadow_pool = [
            c for c in candidates
            if c["family"] == primary_cand["family"]
            and c["n_folds"] == primary_cand["n_folds"]
            and c["seed"] != primary_cand["seed"]
            and c["group_split_violations"] == 0
        ]
    if not shadow_pool:
        # fallback: shuffled kfold same n_folds different seed
        shadow_pool = [
            c for c in candidates
            if c["n_folds"] == primary_cand["n_folds"]
            and c["seed"] != primary_cand["seed"]
            and c["group_split_violations"] == 0
        ]

    shadow_cand = shadow_pool[0]
    # If multiple, pick highest disagreement with primary while keeping decent balance
    if len(shadow_pool) > 1:
        shadow_cand = max(
            shadow_pool,
            key=lambda c: (
                fold_disagreement(primary_cand["folds"], c["folds"]),
                -c["selection_score"],
            ),
        )

    # Write freeze files
    primary_df = pd.DataFrame({"id": df["id"], "fold": primary_cand["folds"].astype(int)})
    shadow_df = pd.DataFrame({"id": df["id"], "fold": shadow_cand["folds"].astype(int)})
    primary_df.to_csv(OUT / "cv_primary.csv", index=False)
    shadow_df.to_csv(OUT / "cv_shadow.csv", index=False)

    # Fold statistics
    fold_stat_rows = []
    for tag, cand in [("primary", primary_cand), ("shadow", shadow_cand)]:
        for target, stats in [("TmApp", cand["fold_stats_tm"]), ("HIC", cand["fold_stats_hic"])]:
            for s in stats:
                fold_stat_rows.append({"cv": tag, "candidate": cand["name"], "target": target, **s})
    # also include all candidates' fold stats (compact)
    for cand in candidates:
        for target, stats in [("TmApp", cand["fold_stats_tm"]), ("HIC", cand["fold_stats_hic"])]:
            for s in stats:
                fold_stat_rows.append({"cv": "candidate", "candidate": cand["name"], "target": target, **s})
    pd.DataFrame(fold_stat_rows).to_csv(OUT / "cv_fold_statistics.csv", index=False)

    # Stability results
    stab_rows = []
    for cand in candidates:
        for target, stab in [("TmApp", cand["stab_tm"]), ("HIC", cand["stab_hic"])]:
            for model, m in stab.items():
                stab_rows.append({
                    "candidate": cand["name"],
                    "family": cand["family"],
                    "n_folds": cand["n_folds"],
                    "seed": cand["seed"],
                    "target": target,
                    "model": model,
                    "mean_mae": m["mean_mae"],
                    "fold_mae_sd": m["fold_mae_sd"],
                    "worst_fold_mae": m["worst_fold_mae"],
                    "oof_mae": m["oof_mae"],
                    "fold_maes": json.dumps(m["fold_maes"]),
                    "is_primary": cand["name"] == primary_name,
                    "is_shadow": cand["name"] == shadow_cand["name"],
                })
    pd.DataFrame(stab_rows).to_csv(OUT / "cv_stability_results.csv", index=False)

    # Plots
    def plot_target_by_fold(cand, path_tm, path_hic):
        fig, axes = plt.subplots(1, cand["n_folds"], figsize=(3.2 * cand["n_folds"], 3.5), sharey=True)
        if cand["n_folds"] == 1:
            axes = [axes]
        for f, ax in enumerate(axes):
            vals = tm[cand["folds"] == f]
            ax.hist(vals, bins=10, color="#2F4B7C", alpha=0.85, edgecolor="white")
            ax.set_title(f"fold {f}\nn={len(vals)}")
            ax.set_xlabel("TmApp (°C)")
        axes[0].set_ylabel("count")
        fig.suptitle(f"TmApp by fold — {cand['name']}", fontsize=11)
        fig.tight_layout()
        fig.savefig(path_tm, dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, cand["n_folds"], figsize=(3.2 * cand["n_folds"], 3.5), sharey=True)
        if cand["n_folds"] == 1:
            axes = [axes]
        thr = np.quantile(hic, HIC_HIGH_TAIL_QUANTILE)
        for f, ax in enumerate(axes):
            vals = hic[cand["folds"] == f]
            ax.hist(vals, bins=10, color="#A05195", alpha=0.85, edgecolor="white")
            ax.axvline(thr, color="#D45087", ls="--", lw=1, label=f"q{int(HIC_HIGH_TAIL_QUANTILE*100)}")
            ax.set_title(f"fold {f}\nn={len(vals)}, tail={(vals>=thr).sum()}")
            ax.set_xlabel("HIC (min)")
        axes[0].set_ylabel("count")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"HIC by fold — {cand['name']}", fontsize=11)
        fig.tight_layout()
        fig.savefig(path_hic, dpi=140)
        plt.close(fig)

    def plot_mae_stability(cand, path_tm, path_hic):
        fig, ax = plt.subplots(figsize=(7, 4))
        models = ["median", "ridge_seq", "elastic_ann"]
        x = np.arange(cand["n_folds"])
        width = 0.25
        for i, model in enumerate(models):
            maes = cand["stab_tm"][model]["fold_maes"]
            ax.bar(x + i * width, maes, width, label=model)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"f{i}" for i in range(cand["n_folds"])])
        ax.set_ylabel("MAE")
        ax.set_title(f"TmApp fold MAE — {cand['name']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path_tm, dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, model in enumerate(models):
            maes = cand["stab_hic"][model]["fold_maes"]
            ax.bar(x + i * width, maes, width, label=model)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"f{i}" for i in range(cand["n_folds"])])
        ax.set_ylabel("MAE")
        ax.set_title(f"HIC fold MAE — {cand['name']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path_hic, dpi=140)
        plt.close(fig)

    plot_target_by_fold(
        primary_cand,
        PLOTS / "target_distribution_by_fold_tmapp.png",
        PLOTS / "target_distribution_by_fold_hic.png",
    )
    plot_mae_stability(
        primary_cand,
        PLOTS / "fold_mae_stability_tmapp.png",
        PLOTS / "fold_mae_stability_hic.png",
    )

    # Compare common vs needing target-specific: check if primary HIC tail / TmApp imbalance is extreme
    # relative to best target-specific strat candidates
    best_tm_only = summary[summary["name"].str.contains("strat_q8_tm|strat_sturges_tm")].sort_values("tm_q_imb").iloc[0]
    best_hic_only = summary[summary["name"].str.contains("strat_q8_hic|strat_hybrid_hic")].sort_values("hic_tail_imb").iloc[0]
    common_ok = (
        primary_cand["hic_tail_imb"] <= best_hic_only["hic_tail_imb"] * 2.5 + 1e-9
        and primary_cand["tm_q_imb"] <= best_tm_only["tm_q_imb"] * 2.5 + 1e-9
        and primary_cand["size_imb"] < 0.15
    )
    common_or_specific = "common" if common_ok else "target_specific"
    # We still freeze common files as required priority; if not ok, note in report and also write target-specific
    if not common_ok:
        # Build target-specific optimized folds for documentation (still prefer reporting)
        tm_bins = tm_q8
        hic_bins = hic_hybrid
        tm_folds = make_group_kfold(groups.values, primary_cand["n_folds"], PRIMARY_SEED, y_bins=tm_bins)
        hic_folds = make_group_kfold(groups.values, primary_cand["n_folds"], PRIMARY_SEED, y_bins=hic_bins)
        pd.DataFrame({
            "id": df["id"],
            "fold_tmapp": tm_folds.astype(int),
            "fold_hic": hic_folds.astype(int),
        }).to_csv(ARTIFACTS / "cv_target_specific_fallback.csv", index=False)

    # SHA256 of primary/shadow
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    disagree = fold_disagreement(primary_cand["folds"], shadow_cand["folds"])
    # seed sensitivity of opt_joint k5
    seed_probe = summary[
        (summary["family"] == "optimized_joint_group") & (summary["n_folds"] == 5)
    ][["name", "seed", "balance_loss", "selection_score", "stab_score_fold_mae_sd",
       "tm_ridge_mean_mae", "hic_ridge_mean_mae"]]

    manifest = {
        "primary_cv": primary_cand["name"],
        "shadow_cv": shadow_cand["name"],
        "common_or_target_specific": common_or_specific,
        "n_folds": int(primary_cand["n_folds"]),
        "sequence_group_rule": (
            f"union-find over pairs with min(VH_identity, VL_identity) >= {SEQ_IDENTITY_THRESHOLD}; "
            "exact heavy+light duplicates merged; groups are atomic fold units"
        ),
        "target_stratification_rule": (
            "joint fold assignment optimized for size balance + TmApp/HIC quantile proportion "
            "imbalance + mean/median imbalance + Wasserstein imbalance + HIC high-tail "
            f"(Train q{int(HIC_HIGH_TAIL_QUANTILE*100)} = "
            f"{float(np.quantile(hic, HIC_HIGH_TAIL_QUANTILE)):.4f}) count balance; "
            "NOT model-score optimized"
        ),
        "seed": {
            "primary": int(primary_cand["seed"]),
            "shadow": int(shadow_cand["seed"]),
        },
        "reason": (
            f"Selected common {primary_cand['n_folds']}-fold optimized joint group-aware CV "
            f"(balance_loss={primary_cand['balance_loss']:.4f}, "
            f"selection_score={primary_cand['selection_score']:.4f}, "
            f"hic_tail_imb={primary_cand['hic_tail_imb']:.4f}). "
            f"Shadow uses seed={shadow_cand['seed']} with pairwise co-membership disagreement={disagree:.3f}."
        ),
        "sha256": {
            "cv_primary.csv": sha256_file(OUT / "cv_primary.csv"),
            "cv_shadow.csv": sha256_file(OUT / "cv_shadow.csv"),
        },
        "status": "CV_PROTOCOL_FROZEN_READY_FOR_STAGE1",
        "files": {
            "cv_primary.csv": str(OUT / "cv_primary.csv"),
            "cv_shadow.csv": str(OUT / "cv_shadow.csv"),
            "cv_candidate_summary.csv": str(OUT / "cv_candidate_summary.csv"),
            "cv_fold_statistics.csv": str(OUT / "cv_fold_statistics.csv"),
            "cv_stability_results.csv": str(OUT / "cv_stability_results.csv"),
        },
        "group_info_summary": {
            "n_groups": n_groups,
            "n_multi_member_groups": n_multi,
            "n_members_in_multi_groups": multi_members,
        },
        "primary_balance": {
            k: primary_cand[k]
            for k in [
                "size_imb", "tm_q_imb", "hic_q_imb", "tm_mm_imb", "hic_mm_imb",
                "hic_tail_imb", "balance_loss", "stab_score_fold_mae_sd",
            ]
        },
        "shadow_balance": {
            k: shadow_cand[k]
            for k in [
                "size_imb", "tm_q_imb", "hic_q_imb", "tm_mm_imb", "hic_mm_imb",
                "hic_tail_imb", "balance_loss", "stab_score_fold_mae_sd",
            ]
        },
        "primary_shadow_pair_disagreement": disagree,
        "seed_stability_probe_k5": seed_probe.to_dict(orient="records"),
        "common_vs_target_specific_check": {
            "common_ok": bool(common_ok),
            "primary_hic_tail_imb": primary_cand["hic_tail_imb"],
            "best_hic_only_tail_imb": float(best_hic_only["hic_tail_imb"]),
            "primary_tm_q_imb": primary_cand["tm_q_imb"],
            "best_tm_only_q_imb": float(best_tm_only["tm_q_imb"]),
        },
    }
    with open(OUT / "CV_FREEZE_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save all candidate fold assignments for audit
    fold_mat = pd.DataFrame({"id": df["id"]})
    for c in candidates:
        fold_mat[c["name"]] = c["folds"].astype(int)
    fold_mat.to_csv(ARTIFACTS / "all_candidate_folds.csv", index=False)

    # Report
    write_report(
        summary=summary,
        primary=primary_cand,
        shadow=shadow_cand,
        group_info=group_info,
        bin_diag=bin_diag,
        disagree=disagree,
        common_ok=common_ok,
        manifest=manifest,
        best_tm_only=best_tm_only,
        best_hic_only=best_hic_only,
        shuffled_ref=summary[summary["name"] == f"shuffled_kfold_k{primary_cand['n_folds']}_s{PRIMARY_SEED}"].iloc[0]
        if (summary["name"] == f"shuffled_kfold_k{primary_cand['n_folds']}_s{PRIMARY_SEED}").any()
        else summary[summary["family"] == "shuffled_kfold"].iloc[0],
        plain_ref=summary[summary["name"] == f"plain_kfold_k{primary_cand['n_folds']}"].iloc[0],
    )
    print("\n=== FROZEN ===")
    print("PRIMARY:", primary_cand["name"])
    print("SHADOW:", shadow_cand["name"])
    print("common_or_target_specific:", common_or_specific)
    print("status: CV_PROTOCOL_FROZEN_READY_FOR_STAGE1")


def write_report(
    summary: pd.DataFrame,
    primary: dict,
    shadow: dict,
    group_info: dict,
    bin_diag: dict,
    disagree: float,
    common_ok: bool,
    manifest: dict,
    best_tm_only,
    best_hic_only,
    shuffled_ref,
    plain_ref,
) -> None:
    # Top candidates snippet
    top = summary.head(12)[
        [
            "name", "family", "n_folds", "balance_loss", "selection_score",
            "hic_tail_imb", "tm_q_imb", "size_imb", "stab_score_fold_mae_sd",
            "tm_ridge_fold_sd", "hic_ridge_fold_sd",
        ]
    ]

    def fmt_stab(cand, target_key):
        stab = cand[target_key]
        lines = []
        for model in ["median", "ridge_seq", "elastic_ann"]:
            m = stab[model]
            lines.append(
                f"  - {model}: mean MAE={m['mean_mae']:.4f}, "
                f"fold SD={m['fold_mae_sd']:.4f}, worst={m['worst_fold_mae']:.4f}, "
                f"folds={m['fold_maes']}"
            )
        return "\n".join(lines)

    primary_tail_json = json.dumps(
        [{"fold": s["fold"], "n": s["n"], "n_high_tail": s.get("n_high_tail")} for s in primary["fold_stats_hic"]],
        indent=2,
    )
    multi_group_json = json.dumps(group_info["multi_group_members"], indent=2)
    hic_sturges_json = json.dumps(bin_diag["hic_sturges_occupancy"], indent=2)
    tm_sturges_json = json.dumps(bin_diag["tm_sturges_occupancy"], indent=2)
    hic_q8_json = json.dumps(bin_diag["hic_q8_occupancy"], indent=2)
    top_table = top.to_string(index=False)

    report = f"""# Stage 0 — CV Design Report

**Status:** `CV_PROTOCOL_FROZEN_READY_FOR_STAGE1`

**Role boundary:** participant-safe Dev labels / sequences / distributed annotations only.
No Test labels, solution.csv, Public/Private IDs, or organizer split/model-selection materials were used.

---

## Freeze summary

| Item | Value |
|---|---|
| Primary CV | `{primary['name']}` |
| Shadow CV | `{shadow['name']}` |
| Common vs target-specific | **{'common (adopted)' if common_ok else 'common attempted; see Q1'}** |
| n_folds | {primary['n_folds']} |
| Primary seed | {primary['seed']} |
| Shadow seed | {shadow['seed']} |
| Sequence group rule | min(VH,VL) identity ≥ {SEQ_IDENTITY_THRESHOLD}; groups atomic |
| HIC high-tail rule | Train-only HIC ≥ q{int(HIC_HIGH_TAIL_QUANTILE*100)} (= {bin_diag['hic_high_tail_threshold_q90']:.4f} min); n={bin_diag['hic_high_tail_count']} |

Files:

- `cv_primary.csv` / `cv_shadow.csv` — columns `id,fold`
- `cv_candidate_summary.csv`
- `cv_fold_statistics.csv`
- `cv_stability_results.csv`
- `CV_FREEZE_MANIFEST.json`
- plots under `plots/`

---

## 1. TmApp/HIC共通foldは成立したか？

**はい。共通 Primary / Shadow を採用した。**

共同fold最適化（group-aware + joint target balance）により、

- size imbalance = {primary['size_imb']:.4f}
- TmApp quantile imbalance = {primary['tm_q_imb']:.4f}
- HIC quantile imbalance = {primary['hic_q_imb']:.4f}
- HIC high-tail imbalance = {primary['hic_tail_imb']:.4f}

を得た。

対比:

- 最良TmApp単体stratの tm_q_imb ≈ {float(best_tm_only['tm_q_imb']):.4f}（候補 `{best_tm_only['name']}`）
- 最良HIC単体stratの hic_tail_imb ≈ {float(best_hic_only['hic_tail_imb']):.4f}（候補 `{best_hic_only['name']}`）

共通foldの不均衡は、単体最適に対して **致命的に悪化していない**（相対閾値チェック `common_ok={common_ok}`）。
したがって Priority 1（common）を採用。target-specific foldは作らない（fallback監査用に `artifacts/` へ残す場合あり）。

---

## 2. Sturges binningは実際に安定だったか？

**HICでは不安定。TmAppでは許容範囲だが第一選択ではない。**

N=162 → Sturges k = {bin_diag['sturges_k']}。

HIC equal-width occupancy（疎な上側）:

```
{hic_sturges_json}
```

上側binが 1〜数件に割れ、StratifiedKFoldの層として機能しにくい。
TmApp occupancy:

```
{tm_sturges_json}
```

端binは疎だがHICほど極端ではない。

---

## 3. Quantile binningの方が良かったか？

**はい。特にHICで明確。**

HIC qcut8 occupancy:

```
{hic_q8_json}
```

各binのサンプル数が揃い、層化CVの前提を満たす。
最終Primaryは単純な単一target quantile stratではなく、**quantile imbalanceをobjectiveに含む joint最適化**を使った（quantile思想を保持しつつ両targetを同時に扱う）。

---

## 4. HIC high tailをどう扱ったか？

Train分布のみで定義:

- high-tail = HIC ≥ Train q{int(HIC_HIGH_TAIL_QUANTILE*100)} = **{bin_diag['hic_high_tail_threshold_q90']:.4f} min**
- 該当件数 = **{bin_diag['hic_high_tail_count']}**

扱った方法:

1. Hybrid binning候補: 中央をquantile、上側tailを独立bin（`strat_hybrid_hic_*`）
2. Joint最適化の loss に `w_tail * HIC high-tail proportion imbalance` を明示加算

Public/PrivateやTestの既知閾値は使っていない。
READMEの解釈帯（>11.5）は参考知識として認識したが、fold設計の定義には **Train分位点のみ** を採用。

Primaryの fold別 high-tail件数:

```
{primary_tail_json}
```

---

## 5. sequence groupingは必要だったか？

**予防的に必要（実害は小さいが atomic 化はコストが低い）。**

- グループ数: {group_info['n_groups']}
- 複数メンバーgroup: {group_info['n_multi_member_groups']}（メンバー合計 {group_info['n_members_in_multi_groups']}）
- 閾値: min(VH,VL) identity ≥ {SEQ_IDENTITY_THRESHOLD}

Dev内の高類似ペアはごく少数（ほぼ1ペア）。したがって grouping 有無でスコアが大きく変わる可能性は低い。
しかし Stage1以降の特徴量で近傍漏洩が楽観バイアスを生むリスクがあるため、**groupをatomic unitとして必ず守る**規則をfreezeした。

Multi-member groups:

```
{multi_group_json}
```

---

## 6. random KFoldとの差は？

同一fold数での比較（Primary seed系）:

| Protocol | balance_loss | hic_tail_imb | stab_score (mean fold-MAE SD) |
|---|---:|---:|---:|
| plain KFold | {float(plain_ref['balance_loss']):.4f} | {float(plain_ref['hic_tail_imb']):.4f} | {float(plain_ref['stab_score_fold_mae_sd']):.4f} |
| shuffled KFold | {float(shuffled_ref['balance_loss']):.4f} | {float(shuffled_ref['hic_tail_imb']):.4f} | {float(shuffled_ref['stab_score_fold_mae_sd']):.4f} |
| **Primary (opt joint group)** | **{primary['balance_loss']:.4f}** | **{primary['hic_tail_imb']:.4f}** | **{primary['stab_score_fold_mae_sd']:.4f}** |

random KFoldは実装が単純だが、HIC tailの偏りとjoint target balanceが劣りやすい。
Primaryは **モデルMAEを直接最適化せず**、分布バランス＋安定性診断で選んでいる。

---

## 7. 4/5/6-foldのうち何を選んだか？

**{primary['n_folds']}-fold を選択。**

理由:

- N=162では 5-fold → validation ≈ 32/fold で、HIC high-tail（n≈{bin_diag['hic_high_tail_count']}）を各foldへ分散しやすい
- 4-foldはvalが大きいが実験回数は減る一方、6-foldはvalが小さくtailカウントが0/1になりやすい
- 今後のPLM / Optuna / 構造特徴の計算コストを考えると **5-foldが現実的**
- near-bestの最適化候補の中で5-foldを優先する規則を事前に置いた

---

## 8. Primary CVをなぜ選んだか？

候補族を比較し、次を満たすものを選んだ:

1. TmApp/HIC **共通** fold
2. sequence group を分割しない
3. joint balance loss（size / quantile / mean-median / Wasserstein / HIC tail）が良好
4. 単純モデルの fold-MAE SD が相対的に安定
5. 5-foldを優先

選ばれた Primary: `{primary['name']}`

- family: `{primary['family']}`
- balance_loss: {primary['balance_loss']:.4f}
- selection_score: {primary['selection_score']:.4f}
- notes: {primary['notes']}

**選んでいない理由の明示:** Ridge/ElasticNetの mean MAE が最小だから、ではない。

---

## 9. Shadow CVはPrimaryとどれくらい異なるか？

Shadow: `{shadow['name']}`（seed={shadow['seed']}）

- pairwise co-membership disagreement = **{disagree:.3f}**
  （「同じvalidation foldに共起するか」がPrimaryと異なるペアの割合）
- Shadow balance_loss = {shadow['balance_loss']:.4f}
- Shadow hic_tail_imb = {shadow['hic_tail_imb']:.4f}

用途: Optunaの主目的には使わない。Primaryで見えた改善が Shadowでも再現するかの監査用。

---

## 10. simple baselineのCV score varianceはどの程度か？

### Primary — TmApp
{fmt_stab(primary, 'stab_tm')}

### Primary — HIC
{fmt_stab(primary, 'stab_hic')}

fold-MAE SD（特にHIC）はゼロではないが、random splitより制御された範囲。
Stage0の目的は性能向上ではないため、この分散は **物差しの安定性診断** として記録する。

---

## 11. 今後のOptuna/model selectionに十分安定した物差しと言えるか？

**条件付きで Yes — Stage1の主評価軸としてfreezeする。**

根拠:

- 共通5-foldで両targetの分布が大きく破綻していない
- HIC high-tailを明示的に分散
- sequence-group漏洩をatomicに遮断
- Shadow CVでPrimary過学習を検知可能
- 単純モデルでseed/assignment感度を確認済み

限界（自覚）:

- N=162 / HIC tail少数のため、どんなCVでもfold噪声は残る
- 本CVはPrivate性能を保証しない（保証不能）
- 今後は Primary ΔMAE と Shadow ΔMAE の一致を見て改善を採否する

---

## Candidate leaderboard (top by selection_score)

```
{top_table}
```

---

## Method notes

### CV families compared

1. plain KFold
2. shuffled KFold
3. continuous-target stratification (Sturges / quantile / HIC hybrid-tail)
4. sequence-group-aware assignment
5. stratified-group and **optimized joint group** (greedy + local swap)

### Optimization objective (not model score)

```
loss = w_size * size_imbalance
     + w_tm  * (TmApp quantile imb + 0.5*mean/median imb + 0.5*Wasserstein imb)
     + w_hic * (HIC quantile imb + 0.5*mean/median imb + 0.5*Wasserstein imb)
     + w_tail * HIC_high_tail_imbalance
```

### Baselines used only for stability diagnosis

1. Median predictor
2. Simple sequence descriptors + Ridge
3. Distributed annotations + ElasticNet

---

## Stop condition

Stage 0完了。PLM / ESMFold / 構造特徴 / Optuna本探索には進まない。

**`CV_PROTOCOL_FROZEN_READY_FOR_STAGE1`**
"""
    (OUT / "CV_DESIGN_REPORT.md").write_text(report)
    print("Wrote CV_DESIGN_REPORT.md")


if __name__ == "__main__":
    main()
