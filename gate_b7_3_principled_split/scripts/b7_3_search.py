"""Gate B7.3 search methods A/B/C — model-blind split generation."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from b7_3_common import (
    PUBLIC_N,
    SEED_BASE,
    check_hard,
    log,
    mask_hash,
    pub_from_group_mask,
    sha256_ids,
)
from b7_3_metrics import score_mask


def _init_greedy(data: dict, rng: np.random.Generator, tilt: np.ndarray | None = None) -> np.ndarray | None:
    """Randomized greedy: place multi-groups first by tilted incremental cost, then fill."""
    sizes = data["group_sizes"]
    high = data["group_high"]
    med = data["group_med"]
    n = len(sizes)
    order = rng.permutation(n)
    # prefer placing larger / high-bearing groups earlier with noise
    priority = -sizes.astype(float) + rng.normal(0, 2.0, size=n)
    if tilt is not None:
        priority = priority + tilt
    order = np.argsort(priority)

    mask = np.zeros(n, dtype=int)
    rem = PUBLIC_N
    for i in order:
        s = int(sizes[i])
        if s > rem:
            continue
        # score heuristic: prefer balancing HIGH/MED provisional
        cur_high = int(mask @ high)
        cur_med = int(mask @ med)
        # remaining capacity after placing
        after = rem - s
        # decide pub vs leave for priv
        place_pub = False
        if after == 0:
            place_pub = True
        else:
            # biased coin toward needed HIGH/MED
            need_high = 3.5  # target ~3.5
            need_med = 3.0
            score_pub = -abs((cur_high + high[i]) - need_high) - 0.5 * abs((cur_med + med[i]) - need_med)
            score_priv = -abs(cur_high - need_high) - 0.5 * abs(cur_med - need_med)
            # also size pressure
            score_pub += rng.normal(0, 0.4)
            score_priv += rng.normal(0, 0.4)
            if rem >= s and (rem - s) <= sum(sizes[mask == 0]) - s:
                place_pub = score_pub >= score_priv
            else:
                place_pub = rem >= s
        if place_pub:
            mask[i] = 1
            rem -= s
            if rem == 0:
                break
    if rem != 0 or not check_hard(mask, data):
        # repair attempt: random restarts via knapsack
        return _knapsack_random(data, rng)
    return mask


def _knapsack_random(data: dict, rng: np.random.Generator, max_try: int = 80) -> np.ndarray | None:
    sizes = data["group_sizes"]
    high = data["group_high"]
    n = len(sizes)
    for _ in range(max_try):
        order = rng.permutation(n)
        mask = np.zeros(n, dtype=int)
        rem = PUBLIC_N
        for i in order:
            if sizes[i] <= rem and (rem - sizes[i] == 0 or rng.random() < 0.55):
                mask[i] = 1
                rem -= int(sizes[i])
                if rem == 0:
                    break
        if rem == 0 and int(mask @ high) in (3, 4):
            return mask
    return None


def _neighbor_swap(mask: np.ndarray, data: dict, rng: np.random.Generator) -> np.ndarray | None:
    """Swap one Public group with one Private group preserving size+HIGH when possible."""
    sizes = data["group_sizes"]
    high = data["group_high"]
    pub_ix = np.where(mask == 1)[0]
    priv_ix = np.where(mask == 0)[0]
    if len(pub_ix) == 0 or len(priv_ix) == 0:
        return None
    # try several random pairs
    for _ in range(12):
        i = int(rng.choice(pub_ix))
        j = int(rng.choice(priv_ix))
        if sizes[i] != sizes[j]:
            # allow unequal if net size preserved via another swap — skip unequal for simplicity
            # unless we can find compensating — keep equal-size only for fast moves
            continue
        trial = mask.copy()
        trial[i] = 0
        trial[j] = 1
        if check_hard(trial, data):
            return trial
    # try single group move with compensating singleton chain
    for _ in range(8):
        i = int(rng.choice(pub_ix))
        j = int(rng.choice(priv_ix))
        delta = int(sizes[j] - sizes[i])
        trial = mask.copy()
        trial[i] = 0
        trial[j] = 1
        # now size is off by delta; fix with additional equal moves of size |delta| if possible
        if delta == 0:
            if check_hard(trial, data):
                return trial
            continue
        # need to move groups totaling |delta| the other way
        need = abs(delta)
        side = 1 if delta < 0 else 0  # if delta>0 public grew; need move pub->priv of size delta
        cand = np.where(trial == side)[0]
        rng.shuffle(cand)
        acc = []
        rem_need = need
        for k in cand:
            if k in (i, j):
                continue
            if sizes[k] <= rem_need:
                acc.append(k)
                rem_need -= int(sizes[k])
                if rem_need == 0:
                    break
        if rem_need != 0:
            continue
        for k in acc:
            trial[k] = 1 - side
        if check_hard(trial, data):
            return trial
    return None


def _lex(m: dict) -> tuple:
    return m["lex_key"]


def local_improve(
    mask: np.ndarray,
    data: dict,
    energy_scale: float,
    rng: np.random.Generator,
    n_swaps: int = 250,
) -> tuple[np.ndarray, dict]:
    best = mask.copy()
    best_m = score_mask(best, data, energy_scale)
    for _ in range(n_swaps):
        nb = _neighbor_swap(best, data, rng)
        if nb is None:
            continue
        m = score_mask(nb, data, energy_scale)
        if _lex(m) < _lex(best_m):
            best, best_m = nb, m
    return best, best_m


def method_a_start(seed: int, data: dict, energy_scale: float) -> dict | None:
    rng = np.random.default_rng(seed)
    mask = _init_greedy(data, rng)
    if mask is None:
        return None
    mask, metrics = local_improve(mask, data, energy_scale, rng, n_swaps=300)
    if not metrics.get("feasible"):
        return None
    return _pack(mask, data, metrics, method="A_greedy_swap", seed=seed)


def method_b_start(seed: int, data: dict, energy_scale: float) -> dict | None:
    """Simulated annealing on lexicographic key (scalarized for SA acceptance)."""
    rng = np.random.default_rng(seed)
    mask = _init_greedy(data, rng)
    if mask is None:
        mask = _knapsack_random(data, rng)
    if mask is None:
        return None
    cur = mask.copy()
    cur_m = score_mask(cur, data, energy_scale)
    best, best_m = cur.copy(), cur_m
    T0 = 2.0
    n_steps = 600
    for t in range(n_steps):
        T = T0 * (1.0 - t / n_steps) + 1e-4
        nb = _neighbor_swap(cur, data, rng)
        if nb is None:
            continue
        m = score_mask(nb, data, energy_scale)
        # scalarized energy for Metropolis
        def scalar(mm):
            return mm["L1"] + 5.0 * mm["L2"] + 3.0 * mm["L3"] + mm["L4"]

        d = scalar(m) - scalar(cur_m)
        if d <= 0 or rng.random() < np.exp(-d / T):
            cur, cur_m = nb, m
            if _lex(cur_m) < _lex(best_m):
                best, best_m = cur.copy(), cur_m
    if not best_m.get("feasible"):
        return None
    return _pack(best, data, best_m, method="B_simulated_annealing", seed=seed)


def method_c_milp(
    seed: int,
    data: dict,
    energy_scale: float,
    time_limit: float = 8.0,
) -> dict | None:
    """MILP on linearized count objectives under group binaries + size + HIGH."""
    rng = np.random.default_rng(seed)
    sizes = data["group_sizes"].astype(float)
    high = data["group_high"].astype(float)
    med = data["group_med"].astype(float)
    n = len(sizes)
    feat = data["feat"]

    # Objective tilt: random weights on group features (linear proxy)
    # Use per-group contribution to q8 imbalance via expected counts
    tm_q = data["qbins"][8]["TmApp"]
    hic_q = data["qbins"][8]["HIC"]
    joint = data["tm_quart"] * 4 + data["hic_quart"]

    # group -> count vector for bins
    def group_bin_counts(bins, n_bins):
        mat = np.zeros((n, n_bins), float)
        for k, g in enumerate(data["group_ids"]):
            for i in data["group_map"][g]:
                r = data["id_to_row"][i]
                mat[k, bins[r]] += 1
        return mat

    G_tm = group_bin_counts(tm_q, 8)
    G_hic = group_bin_counts(hic_q, 8)
    G_j = group_bin_counts(joint, 16)

    # Random objective: minimize sum_b |2*pub_b - total_b| linearized
    # Using auxiliary vars is heavy; use weighted linear tilt:
    # prefer groups whose local composition matches a random target mix
    w = rng.normal(0, 1.0, size=n)
    # tilt toward MEDIUM-bearing groups sometimes
    if rng.random() < 0.6:
        w = w - 0.8 * med
    if rng.random() < 0.5:
        # diversify HIGH placement: soft preference for target high_pub
        target_h = 3 if rng.random() < 0.5 else 4
        # can't enforce soft in c; hard constraint below
        _ = target_h

    # Also add tilt from random target public counts per bin
    tgt_tm = G_tm.sum(0) / 2.0 + rng.normal(0, 0.3, size=8)
    # linear proxy: -sum_g x_g * (alignment)
    align = (G_tm * tgt_tm).sum(1) + 0.7 * (G_hic * (G_hic.sum(0) / 2)).sum(1)
    align += 0.4 * (G_j * (G_j.sum(0) / 2)).sum(1)
    c = -align + 0.15 * w  # maximize alignment => minimize -align

    # Constraints: size == 81, HIGH in {3,4}
    # For HIGH in {3,4}: solve twice
    best_pack = None
    best_lex = None
    for high_target in (3, 4):
        # x binary
        integrality = np.ones(n)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        A = np.vstack([sizes, high])
        lb = np.array([PUBLIC_N, high_target], float)
        ub = np.array([PUBLIC_N, high_target], float)
        constraints = LinearConstraint(A, lb, ub)
        try:
            res = milp(
                c=c,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints,
                options={"time_limit": time_limit, "disp": False},
            )
        except Exception:
            continue
        if res.x is None:
            continue
        mask = (np.asarray(res.x) > 0.5).astype(int)
        if not check_hard(mask, data):
            # repair via local search from this seed
            mask2 = mask.copy()
            # if size wrong, abort
            if int(mask2 @ data["group_sizes"]) != PUBLIC_N:
                continue
            if not check_hard(mask2, data):
                continue
        # polish with local swaps
        mask, metrics = local_improve(mask, data, energy_scale, rng, n_swaps=120)
        if not metrics.get("feasible"):
            continue
        pack = _pack(mask, data, metrics, method="C_milp", seed=seed)
        pack["milp_high_target"] = high_target
        pack["milp_success"] = bool(res.success)
        if best_lex is None or pack["lex_key"] < best_lex:
            best_pack, best_lex = pack, pack["lex_key"]
    return best_pack


def _pack(mask, data, metrics, method, seed) -> dict:
    pub, priv = pub_from_group_mask(mask, data)
    return {
        "split_id": f"{method.split('_')[0]}_{seed}",
        "seed": int(seed),
        "method": method,
        "public_ids": pub,
        "private_ids": priv,
        "public_hash": sha256_ids(pub),
        "private_hash": sha256_ids(priv),
        "mask_hash": mask_hash(pub),
        "group_mask": mask.astype(int).tolist(),
        "is_control": False,
        **{k: metrics[k] for k in metrics if k != "lex_key"},
        "lex_key": metrics["lex_key"],
    }


def run_search(
    data: dict,
    energy_scale: float,
    n_a: int = 400,
    n_b: int = 400,
    n_c: int = 80,
    max_total: int = 3000,
    min_total: int = 500,
) -> tuple[list[dict], dict]:
    """Run Methods A/B/C; return unique valid packs + summary."""
    archive: dict[str, dict] = {}
    stats = {
        "n_a_starts": 0,
        "n_b_starts": 0,
        "n_c_starts": 0,
        "n_a_valid": 0,
        "n_b_valid": 0,
        "n_c_valid": 0,
        "n_failed": 0,
        "runtime_sec": 0.0,
    }
    t0 = time.time()

    def ingest(pack: dict | None, method_key: str):
        if pack is None:
            stats["n_failed"] += 1
            return
        stats[f"n_{method_key}_valid"] += 1
        h = pack["mask_hash"]
        if h not in archive or pack["lex_key"] < archive[h]["lex_key"]:
            archive[h] = pack

    log(f"METHOD A — randomized greedy+swaps ({n_a} starts)")
    for i in range(n_a):
        seed = SEED_BASE + i
        stats["n_a_starts"] += 1
        ingest(method_a_start(seed, data, energy_scale), "a")
        if (i + 1) % 50 == 0:
            log(f"  A progress {i+1}/{n_a}; unique={len(archive)}")

    log(f"METHOD B — simulated annealing ({n_b} starts)")
    for i in range(n_b):
        seed = SEED_BASE + 10000 + i
        stats["n_b_starts"] += 1
        ingest(method_b_start(seed, data, energy_scale), "b")
        if (i + 1) % 50 == 0:
            log(f"  B progress {i+1}/{n_b}; unique={len(archive)}")

    log(f"METHOD C — scipy milp linearized ({n_c} solves)")
    for i in range(n_c):
        seed = SEED_BASE + 20000 + i
        stats["n_c_starts"] += 1
        ingest(method_c_milp(seed, data, energy_scale, time_limit=6.0), "c")
        if (i + 1) % 10 == 0:
            log(f"  C progress {i+1}/{n_c}; unique={len(archive)}")

    # Ensure minimum starts if too few unique
    total_starts = stats["n_a_starts"] + stats["n_b_starts"] + stats["n_c_starts"]
    extra = 0
    while len(archive) < 50 and total_starts + extra < max_total and extra < 200:
        seed = SEED_BASE + 30000 + extra
        ingest(method_a_start(seed, data, energy_scale), "a")
        stats["n_a_starts"] += 1
        extra += 1
        total_starts += 1

    stats["runtime_sec"] = time.time() - t0
    stats["n_unique_valid"] = len(archive)
    stats["n_starts_total"] = stats["n_a_starts"] + stats["n_b_starts"] + stats["n_c_starts"]
    log(
        f"Search done: starts={stats['n_starts_total']} unique_valid={len(archive)} "
        f"runtime={stats['runtime_sec']:.1f}s"
    )
    packs = sorted(archive.values(), key=lambda p: p["lex_key"])
    # assign stable split_ids by rank
    for i, p in enumerate(packs):
        p["split_id"] = f"GEN_{i:04d}_{p['method'].split('_')[0]}_{p['seed']}"
        p["premodel_rank"] = i + 1
    return packs, stats
