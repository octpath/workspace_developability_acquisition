#!/usr/bin/env python3
"""
Freeze final competition population (HIC∩TmApp) and ONE Train/Public/Private split.

CRITICAL: selection uses ONLY pre-model diagnostics. No PLM/structure/model scores.
Target geometry: ~50% Train / ~25% Public / ~25% Private.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import (  # noqa: E402
    B1_DATA,
    CONFIG,
    ORG,
    PART,
    REPORTS,
    ensure_dirs,
    file_sha256,
    sha256_lines,
    write_json,
)

# ---------------------------------------------------------------------------
# FROZEN quality weights — set BEFORE ranking; never change after model results
# ---------------------------------------------------------------------------
WEIGHTS = {
    "pub_priv_count_diff": 3.0,
    "size_deviation_50_25_25": 2.5,
    "hic_wass_train_pub": 1.5,
    "hic_wass_train_priv": 1.5,
    "hic_wass_pub_priv": 2.0,
    "tm_wass_train_pub": 1.5,
    "tm_wass_train_priv": 1.5,
    "tm_wass_pub_priv": 2.0,
    "hic_quantile_imbalance": 1.2,
    "tm_quantile_imbalance": 1.2,
    "len_vh_wass": 0.8,
    "len_vl_wass": 0.8,
    "vh_family_js": 1.0,
    "vl_family_js": 1.0,
    "cluster_size_imbalance": 1.0,
    "nn_similarity_asymmetry": 1.5,
    "bcell_js_secondary": 0.3,  # organizer audit only — small weight
    "donor_js_secondary": 0.2,
}

NOMINAL = {"Train": 0.50, "Public": 0.25, "Private": 0.25}
N_CANDIDATES = 2500
CLUSTER_THR = 0.90
MASTER_SEED = 20260829


def identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / max(len(a), len(b))


class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def cluster_paired(heavies, lights, thr=CLUSTER_THR):
    n = len(heavies)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            if identity(heavies[i], heavies[j]) >= thr or identity(lights[i], lights[j]) >= thr:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([uniq[r] for r in roots], dtype=int)


def js_categorical(series, roles, role_a, role_b=None):
    """Jensen-Shannon-like L1 between role fractions vs full (or role_a vs role_b)."""
    def frac(mask):
        vc = series[mask].value_counts(normalize=True)
        return vc

    if role_b is None:
        full = series.value_counts(normalize=True)
        sub = frac(roles == role_a)
        keys = full.index.union(sub.index)
        a = full.reindex(keys, fill_value=0).values
        b = sub.reindex(keys, fill_value=0).values
    else:
        fa = frac(roles == role_a)
        fb = frac(roles == role_b)
        keys = fa.index.union(fb.index)
        a = fa.reindex(keys, fill_value=0).values
        b = fb.reindex(keys, fill_value=0).values
    m = 0.5 * (a + b)
    # JS via KL with epsilon
    eps = 1e-12
    def kl(p, q):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return float(np.sum(p * np.log(p / q)))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def quantile_imbalance(y, roles):
    """Mean abs diff of role quantiles vs Train quantiles."""
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    yt = y[roles == "Train"]
    if len(yt) < 5:
        return 1.0
    qt = np.quantile(yt, qs)
    s = 0.0
    for role in ["Public", "Private"]:
        yr = y[roles == role]
        if len(yr) < 3:
            return 1.0
        s += float(np.mean(np.abs(np.quantile(yr, qs) - qt)))
    return s / 2.0


def assign_roles_grouped(groups, seed, n, frac_pub=0.25, frac_priv=0.25):
    """Greedy fill Public then Private by shuffled clusters to hit mass targets."""
    roles = np.array(["Train"] * n, dtype=object)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = uniq.copy()
    rng.shuffle(order)
    sizes = {g: int((groups == g).sum()) for g in uniq}
    # Sort strategies: sometimes largest-first residual fill
    strategy = int(seed) % 3
    if strategy == 1:
        order = np.array(sorted(order, key=lambda g: -sizes[g]))
        # reshuffle within size bands lightly
        rng2 = np.random.default_rng(seed + 7)
        order = order[rng2.permutation(len(order))]
    elif strategy == 2:
        order = np.array(sorted(order, key=lambda g: sizes[g]))
        rng2 = np.random.default_rng(seed + 11)
        order = order[rng2.permutation(len(order))]

    pub_mass = priv_mass = 0
    target_pub, target_priv = frac_pub * n, frac_priv * n
    pub_gs, priv_gs = set(), set()
    for g in order:
        if pub_mass + sizes[g] <= target_pub * 1.15 and pub_mass < target_pub:
            pub_gs.add(int(g))
            pub_mass += sizes[g]
        elif priv_mass + sizes[g] <= target_priv * 1.15 and priv_mass < target_priv:
            priv_gs.add(int(g))
            priv_mass += sizes[g]
    # residual fill if under
    for g in order:
        g = int(g)
        if g in pub_gs or g in priv_gs:
            continue
        if pub_mass < target_pub:
            pub_gs.add(g)
            pub_mass += sizes[g]
        elif priv_mass < target_priv:
            priv_gs.add(g)
            priv_mass += sizes[g]
    for i, g in enumerate(groups):
        if int(g) in pub_gs:
            roles[i] = "Public"
        elif int(g) in priv_gs:
            roles[i] = "Private"
    return roles


def nn_asymmetry(df, roles, sample_max=80):
    """|mean max-id Train→Pub - Train→Priv| using sampled pairs."""
    train = df[roles == "Train"]
    pub = df[roles == "Public"]
    priv = df[roles == "Private"]
    if len(train) == 0 or len(pub) == 0 or len(priv) == 0:
        return 1.0
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(train), size=min(sample_max, len(train)), replace=False)
    def mean_nn(target):
        vals = []
        t_idx = rng.choice(len(target), size=min(sample_max, len(target)), replace=False)
        for j in t_idx:
            th, tl = target.iloc[j]["heavy"], target.iloc[j]["light"]
            best = 0.0
            for i in tr_idx:
                best = max(
                    best,
                    identity(train.iloc[i]["heavy"], th),
                    identity(train.iloc[i]["light"], tl),
                )
            vals.append(best)
        return float(np.mean(vals))
    return abs(mean_nn(pub) - mean_nn(priv))


def score_split(df, roles, groups, include_nn=False):
    n = len(df)
    counts = {k: int((roles == k).sum()) for k in ["Train", "Public", "Private"]}
    components = {}
    components["pub_priv_count_diff"] = abs(counts["Public"] - counts["Private"]) / max(n, 1)
    size_pen = 0.0
    for role, frac in NOMINAL.items():
        size_pen += abs(counts[role] / n - frac)
    components["size_deviation_50_25_25"] = size_pen

    hic = df["HIC"].values.astype(float)
    tm = df["TmApp"].values.astype(float)
    for name, y in [("hic", hic), ("tm", tm)]:
        for a, b, key in [
            ("Train", "Public", f"{name}_wass_train_pub"),
            ("Train", "Private", f"{name}_wass_train_priv"),
            ("Public", "Private", f"{name}_wass_pub_priv"),
        ]:
            ya, yb = y[roles == a], y[roles == b]
            components[key] = float(wasserstein_distance(ya, yb)) if len(ya) and len(yb) else 1.0
    components["hic_quantile_imbalance"] = quantile_imbalance(hic, roles)
    components["tm_quantile_imbalance"] = quantile_imbalance(tm, roles)

    for col, key in [("vh_len", "len_vh_wass"), ("vl_len", "len_vl_wass")]:
        y = df[col].values.astype(float)
        components[key] = 0.5 * (
            float(wasserstein_distance(y[roles == "Train"], y[roles == "Public"]))
            + float(wasserstein_distance(y[roles == "Train"], y[roles == "Private"]))
        )

    components["vh_family_js"] = js_categorical(df["vh_family"], roles, "Public", "Private")
    components["vl_family_js"] = js_categorical(df["vl_family"], roles, "Public", "Private")

    cs = pd.Series(groups).value_counts()
    role_cs = {}
    for role in ["Train", "Public", "Private"]:
        gset = set(groups[roles == role])
        role_cs[role] = np.array([cs[g] for g in gset]) if gset else np.array([0.0])
    components["cluster_size_imbalance"] = float(
        abs(role_cs["Public"].mean() - role_cs["Private"].mean()) / max(cs.mean(), 1)
    )
    components["nn_similarity_asymmetry"] = nn_asymmetry(df, roles) if include_nn else 0.0
    components["bcell_js_secondary"] = js_categorical(df["b_cell_subset"].astype(str), roles, "Public", "Private")
    components["donor_js_secondary"] = js_categorical(df["donor"].astype(str), roles, "Public", "Private")

    hard = 0.0
    if counts["Public"] == 0 or counts["Private"] == 0 or counts["Train"] == 0:
        hard += 100.0
    for g in np.unique(groups):
        rs = set(roles[groups == g])
        if len(rs) > 1:
            hard += 100.0
            break

    total = hard
    for k, w in WEIGHTS.items():
        if k == "nn_similarity_asymmetry" and not include_nn:
            continue
        total += w * float(components.get(k, 0.0))
    return total, components, counts


def main():
    ensure_dirs()
    t0 = time.time()
    core = pd.read_csv(B1_DATA / "triple_core.csv")
    assert core["hic_rt_min"].notna().all() and core["tm_app_C"].notna().all()
    num = pd.read_csv(B1_DATA / "numbering_germline.csv")

    pop = core.copy()
    pop = pop.rename(columns={"antibody_id": "id", "hic_rt_min": "HIC", "tm_app_C": "TmApp"})
    # donor if present
    if "donor" not in pop.columns:
        # try from full shehata
        full = B1_DATA / "shehata_b1_full.csv"
        if full.exists():
            f = pd.read_csv(full)
            if "donor" in f.columns:
                pop = pop.merge(f[["antibody_id", "donor"]].rename(columns={"antibody_id": "id"}), on="id", how="left")
            else:
                pop["donor"] = "UNK"
        else:
            pop["donor"] = "UNK"
    pop = pop.merge(
        num.rename(columns={"antibody_id": "id"})[
            ["id", "ORG_author_vh_family", "ORG_author_vl_family", "PL_combined_germline_distance"]
        ],
        on="id",
        how="left",
    )
    pop["vh_family"] = pop["ORG_author_vh_family"].fillna("UNK").astype(str)
    pop["vl_family"] = pop["ORG_author_vl_family"].fillna("UNK").astype(str)
    pop["b_cell_subset"] = pop["b_cell_subset"].fillna("UNK").astype(str)
    pop["donor"] = pop["donor"].fillna("UNK").astype(str)

    groups = cluster_paired(pop["heavy"].tolist(), pop["light"].tolist(), CLUSTER_THR)
    pop["sequence_group"] = groups
    print(f"POPULATION n={len(pop)} n_groups={len(np.unique(groups))}", flush=True)

    # Save population
    pop_out = pop[
        [
            "id", "heavy", "light", "HIC", "TmApp", "sequence_group",
            "b_cell_subset", "donor", "vh_family", "vl_family",
            "vh_germline", "vl_germline", "vh_len", "vl_len", "seq_pair_hash",
        ]
    ].copy()
    pop_path = ORG / "final_population.csv"
    pop_out.to_csv(pop_path, index=False)
    pop_sha = file_sha256(pop_path)

    # Generate candidates — phase 1 without expensive NN, phase 2 refine top-K
    candidates = []
    print(f"Generating {N_CANDIDATES} candidates (phase-1, no NN)…", flush=True)
    for i in range(N_CANDIDATES):
        seed = MASTER_SEED + i * 17 + (i % 97)
        roles = assign_roles_grouped(groups, seed, len(pop), 0.25, 0.25)
        total, components, counts = score_split(pop, roles, groups, include_nn=False)
        # Prefer near-equal pub/priv early
        if abs(counts["Public"] - counts["Private"]) > 12:
            total += 2.0
        candidates.append(
            {
                "seed": seed,
                "score": total,
                "n_train": counts["Train"],
                "n_public": counts["Public"],
                "n_private": counts["Private"],
                "components": components,
                "roles": roles,
            }
        )
        if (i + 1) % 500 == 0:
            print(f"  scored {i+1}/{N_CANDIDATES}", flush=True)

    candidates.sort(key=lambda c: c["score"])
    topk = candidates[:80]
    print(f"Phase-2 NN refine on top {len(topk)}…", flush=True)
    refined = []
    for c in topk:
        total, components, counts = score_split(pop, c["roles"], groups, include_nn=True)
        c2 = dict(c)
        c2["score"] = total
        c2["components"] = components
        c2["n_train"], c2["n_public"], c2["n_private"] = counts["Train"], counts["Public"], counts["Private"]
        refined.append(c2)
    refined.sort(key=lambda c: c["score"])
    best = refined[0]
    roles = best["roles"]
    print(
        f"BEST seed={best['seed']} score={best['score']:.4f} "
        f"N={best['n_train']}/{best['n_public']}/{best['n_private']}",
        flush=True,
    )

    pop["role"] = roles
    # Write ID lists
    for role, fname in [("Train", "train_ids.csv"), ("Public", "public_ids.csv"), ("Private", "private_ids.csv")]:
        ids = pop.loc[pop["role"] == role, "id"].sort_values()
        path = ORG / fname
        ids.to_frame("id").to_csv(path, index=False)

    train_sha = sha256_lines(pop.loc[pop.role == "Train", "id"])
    pub_sha = sha256_lines(pop.loc[pop.role == "Public", "id"])
    priv_sha = sha256_lines(pop.loc[pop.role == "Private", "id"])

    # Participant staging
    train_df = pop[pop.role == "Train"][["id", "heavy", "light", "TmApp", "HIC"]].sort_values("id")
    test_df = pop[pop.role != "Train"][["id", "heavy", "light"]].sort_values("id")
    sample = test_df[["id"]].copy()
    sample["TmApp"] = 0.0
    sample["HIC"] = 0.0
    hidden = pop[pop.role != "Train"][["id", "role", "TmApp", "HIC"]].sort_values("id")
    train_df.to_csv(PART / "train.csv", index=False)
    test_df.to_csv(PART / "test.csv", index=False)
    sample.to_csv(PART / "sample_submission.csv", index=False)
    hidden.to_csv(ORG / "test_labels_hidden.csv", index=False)

    # Full role map for organizer
    pop[["id", "role", "sequence_group", "HIC", "TmApp"]].to_csv(ORG / "role_map.csv", index=False)

    code_hash = file_sha256(Path(__file__))
    manifest = {
        "STATUS": "FINAL SPLIT FROZEN — DO NOT CHANGE BASED ON MODEL PERFORMANCE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "population": {
            "n": len(pop),
            "filter": "both HIC and TmApp non-null (triple_core)",
            "source": str(B1_DATA / "triple_core.csv"),
            "sha256": pop_sha,
            "path": str(pop_path),
        },
        "geometry": {"Train": 0.50, "Public": 0.25, "Private": 0.25},
        "counts": {
            "Train": best["n_train"],
            "Public": best["n_public"],
            "Private": best["n_private"],
            "pub_priv_diff": abs(best["n_public"] - best["n_private"]),
        },
        "clustering": {"threshold_identity": CLUSTER_THR, "n_groups": int(len(np.unique(groups)))},
        "candidate_pool_size": N_CANDIDATES,
        "selected_seed": best["seed"],
        "master_seed": MASTER_SEED,
        "selection_score": best["score"],
        "component_scores": best["components"],
        "quality_weights_frozen": WEIGHTS,
        "id_list_sha256": {"train": train_sha, "public": pub_sha, "private": priv_sha},
        "split_generation_code_sha256": code_hash,
        "runtime_s": time.time() - t0,
        "prohibited_note": "No PLM/structure/CV/model scores used in selection.",
    }
    write_json(CONFIG / "FINAL_SPLIT_MANIFEST.json", manifest)

    # Report
    def summarize(y, mask):
        v = y[mask]
        return {
            "n": int(mask.sum()),
            "mean": float(v.mean()),
            "sd": float(v.std()),
            "q10": float(np.quantile(v, 0.1)),
            "q50": float(np.quantile(v, 0.5)),
            "q90": float(np.quantile(v, 0.9)),
        }

    lines = [
        "# Final split freeze",
        "",
        "## FINAL SPLIT FROZEN",
        "## DO NOT CHANGE BASED ON MODEL PERFORMANCE",
        "",
        f"- Population N={len(pop)} (HIC∩TmApp)",
        f"- Train/Public/Private = **{best['n_train']} / {best['n_public']} / {best['n_private']}**",
        f"- |Public−Private| = {abs(best['n_public']-best['n_private'])}",
        f"- Sequence groups: {len(np.unique(groups))} @ {CLUSTER_THR:.0%} paired identity",
        f"- Selected seed: {best['seed']}  score={best['score']:.4f}",
        f"- Candidate pool: {N_CANDIDATES}",
        "",
        "### HIC by role",
        f"- Train: {summarize(pop['HIC'].values, roles=='Train')}",
        f"- Public: {summarize(pop['HIC'].values, roles=='Public')}",
        f"- Private: {summarize(pop['HIC'].values, roles=='Private')}",
        "",
        "### TmApp by role",
        f"- Train: {summarize(pop['TmApp'].values, roles=='Train')}",
        f"- Public: {summarize(pop['TmApp'].values, roles=='Public')}",
        f"- Private: {summarize(pop['TmApp'].values, roles=='Private')}",
        "",
        "### Hashes",
        f"- population sha256: `{pop_sha}`",
        f"- train ids: `{train_sha}`",
        f"- public ids: `{pub_sha}`",
        f"- private ids: `{priv_sha}`",
        "",
        "Pre-model only. Model evaluation must not alter this split.",
        "",
    ]
    (REPORTS / "final_split_freeze.md").write_text("\n".join(lines) + "\n")

    # Quick leakage assert
    for g in np.unique(groups):
        assert len(set(roles[groups == g])) == 1
    assert set(pop.loc[pop.role == "Public", "id"]) == set(hidden.loc[hidden.role == "Public", "id"])
    print("SPLIT_FROZEN", manifest["counts"])


if __name__ == "__main__":
    main()
