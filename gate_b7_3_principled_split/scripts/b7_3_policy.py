"""Gate B7.3 participant policy stress simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from b7_3_common import SEED_BASE, log, mae


POLICIES = ("CV_FIRST", "BALANCED", "PUBLIC_FIRST")


def _choose_final(policy: str, history: list[dict]) -> dict:
    if policy == "CV_FIRST":
        return min(history, key=lambda h: (h["cv_mae"], h["public_mae"]))
    if policy == "PUBLIC_FIRST":
        return min(history, key=lambda h: (h["public_mae"], h["cv_mae"]))
    # BALANCED: sum of ranks
    cv_sorted = sorted(history, key=lambda h: h["cv_mae"])
    pub_sorted = sorted(history, key=lambda h: h["public_mae"])
    cv_rank = {h["name"]: i for i, h in enumerate(cv_sorted)}
    pub_rank = {h["name"]: i for i, h in enumerate(pub_sorted)}
    return min(
        history,
        key=lambda h: (cv_rank[h["name"]] + pub_rank[h["name"]], h["public_mae"]),
    )


def simulate_trajectories(
    entry: dict,
    data: dict,
    store: dict,
    target: str,
    n_traj: int = 500,
    catalog_size: int = 8,
    seed: int = SEED_BASE,
) -> list[dict]:
    """Random development trajectories over bank models; return per-policy aggregates."""
    pub_ids = entry["public_ids"]
    priv_ids = entry["private_ids"]
    split_id = entry.get("split_id") or entry.get("candidate_id")
    id_to_ix = {i: k for k, i in enumerate(data["test_ids"])}
    pub_ix = [id_to_ix[i] for i in pub_ids]
    priv_ix = [id_to_ix[i] for i in priv_ids]
    y_train = data["pop"].loc[data["train_ids"], target].astype(float).values
    y_pub = data["pop"].loc[pub_ids, target].astype(float).values
    y_priv = data["pop"].loc[priv_ids, target].astype(float).values

    keys = [k for k in store if k.startswith(f"{target}::")]
    models = []
    for k in keys:
        d = store[k]
        models.append({
            "name": d["name"],
            "cv_mae": mae(y_train, d["oof"]),
            "public_mae": mae(y_pub, d["test_full"][pub_ix]),
            "private_mae": mae(y_priv, d["test_full"][priv_ix]),
        })
    if len(models) < 3:
        return []

    best_priv_global = min(m["private_mae"] for m in models)
    rng = np.random.default_rng(seed)
    # collect regrets per policy
    regrets = {p: [] for p in POLICIES}
    bsf_regrets = {p: [] for p in POLICIES}

    for t in range(n_traj):
        order = rng.permutation(len(models))
        # progressive reveal of catalog_size models (or all)
        n_show = min(catalog_size, len(models))
        revealed = [models[i] for i in order[:n_show]]
        for policy in POLICIES:
            # stepwise: at each step k=2..n_show, choose among revealed[:k]
            best_so_far_priv = float("inf")
            final = None
            for k in range(2, n_show + 1):
                hist = revealed[:k]
                final = _choose_final(policy, hist)
                best_so_far_priv = min(best_so_far_priv, final["private_mae"])
            assert final is not None
            regrets[policy].append(final["private_mae"] - best_priv_global)
            bsf_regrets[policy].append(best_so_far_priv - best_priv_global)

    rows = []
    for policy in POLICIES:
        r = np.asarray(regrets[policy], float)
        b = np.asarray(bsf_regrets[policy], float)
        rows.append({
            "split_id": split_id,
            "target": target,
            "policy": policy,
            "n_traj": n_traj,
            "mean_private_regret": float(r.mean()),
            "median_private_regret": float(np.median(r)),
            "p90_private_regret": float(np.quantile(r, 0.90)),
            "mean_bsf_private_regret": float(b.mean()),
            "frac_zero_regret": float(np.mean(r <= 1e-12)),
        })
    return rows


def run_policy_stress(
    entries: list[dict],
    data: dict,
    store: dict,
    n_traj: int = 500,
) -> pd.DataFrame:
    rows = []
    for entry in entries:
        sid = entry.get("split_id") or entry.get("candidate_id")
        log(f"  policy stress {sid}")
        for target in ("TmApp", "HIC"):
            seed = SEED_BASE + (hash(sid + target) % 100000)
            rows.extend(
                simulate_trajectories(entry, data, store, target, n_traj=n_traj, seed=seed)
            )
    return pd.DataFrame(rows)
