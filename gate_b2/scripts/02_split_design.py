#!/usr/bin/env python3
"""Pre-model split redesign: generate candidate pool, score, freeze canonical+5 shadows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import (  # noqa: E402
    B1_DATA,
    B2_TARGETS,
    CONFIG,
    DATA,
    REPORTS,
    SPLITS,
    ensure_dirs,
    pair_hash,
    sha256_file,
    write_json,
)

# Reuse B1 clustering
sys.path.insert(0, str(Path("/workspace_developability_acquisition/gate_b1/scripts")))
from importlib import import_module

# inline minimal cluster from B1
sys.path.insert(0, "/workspace_developability_acquisition/gate_b1/scripts")


def identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


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


def cluster_paired(heavies, lights, thr=0.90):
    n = len(heavies)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            if identity(heavies[i], heavies[j]) >= thr or identity(lights[i], lights[j]) >= thr:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([uniq[r] for r in roots], dtype=int)


def assign_roles(groups, seed, frac_pub=0.20, frac_priv=0.20):
    n = len(groups)
    roles = np.array(["Dev"] * n, dtype=object)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = uniq.copy()
    rng.shuffle(order)
    sizes = {g: int((groups == g).sum()) for g in uniq}
    pub_mass = priv_mass = 0
    target_pub, target_priv = frac_pub * n, frac_priv * n
    pub_gs, priv_gs = set(), set()
    for g in order:
        if pub_mass < target_pub:
            pub_gs.add(g)
            pub_mass += sizes[g]
        elif priv_mass < target_priv:
            priv_gs.add(g)
            priv_mass += sizes[g]
    for i, g in enumerate(groups):
        if g in pub_gs:
            roles[i] = "Public"
        elif g in priv_gs:
            roles[i] = "Private"
    return roles


def wasserstein_1d(a, b):
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    # approximate via quantile grid
    qs = np.linspace(0, 1, 50)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def cat_balance(series, roles, role):
    full = series.value_counts(normalize=True)
    sub = series[roles == role].value_counts(normalize=True)
    keys = full.index.union(sub.index)
    return float(np.sum(np.abs(full.reindex(keys, fill_value=0) - sub.reindex(keys, fill_value=0))))


def split_quality(df, roles, groups, target_col):
    n = len(df)
    counts = {k: int((roles == k).sum()) for k in ["Dev", "Public", "Private"]}
    # size balance vs 60/20/20
    size_pen = (
        abs(counts["Dev"] / n - 0.60)
        + abs(counts["Public"] / n - 0.20)
        + abs(counts["Private"] / n - 0.20)
    )
    y = df[target_col].values
    mean_pen = sd_pen = q_pen = 0.0
    for role in ["Public", "Private"]:
        yr = y[roles == role]
        yd = y[roles == "Dev"]
        mean_pen += abs(yr.mean() - yd.mean()) / (yd.std() + 1e-6)
        sd_pen += abs(yr.std() - yd.std()) / (yd.std() + 1e-6)
        q_pen += wasserstein_1d(yr, yd) / (yd.std() + 1e-6)
    germ_pen = 0.0
    if "ORG_author_vh_family" in df.columns:
        for role in ["Public", "Private"]:
            germ_pen += cat_balance(df["ORG_author_vh_family"], roles, role)
    subset_pen = 0.0
    if "b_cell_subset" in df.columns:
        for role in ["Public", "Private"]:
            subset_pen += cat_balance(df["b_cell_subset"], roles, role)
    # nn similarity Dev→Public/Private
    nn_pen = 0.0
    train_h = df.loc[roles == "Dev", "heavy"].tolist()
    train_l = df.loc[roles == "Dev", "light"].tolist()
    for role in ["Public", "Private"]:
        sims = []
        for _, r in df[roles == role].iterrows():
            best = 0.0
            for h, l in zip(train_h, train_l):
                best = max(best, identity(r["heavy"], h), identity(r["light"], l))
            sims.append(best)
        # prefer moderate leakage (not too high)
        nn_pen += max(0.0, float(np.mean(sims)) - 0.85) * 2.0
    # cluster size: avoid tiny Public
    g_pub = len(np.unique(groups[roles == "Public"]))
    g_priv = len(np.unique(groups[roles == "Private"]))
    cluster_pen = 0.0
    if g_pub < 8:
        cluster_pen += (8 - g_pub) * 0.05
    if g_priv < 8:
        cluster_pen += (8 - g_priv) * 0.05
    # lower is better
    score = (
        3.0 * size_pen
        + 1.5 * mean_pen
        + 1.0 * sd_pen
        + 1.5 * q_pen
        + 1.0 * germ_pen
        + 0.8 * subset_pen
        + 1.2 * nn_pen
        + cluster_pen
    )
    return {
        "score": float(score),
        "counts": counts,
        "size_pen": size_pen,
        "mean_pen": mean_pen,
        "sd_pen": sd_pen,
        "q_pen": q_pen,
        "germ_pen": germ_pen,
        "subset_pen": subset_pen,
        "nn_pen": nn_pen,
        "n_groups_pub": g_pub,
        "n_groups_priv": g_priv,
    }


def design_for_target(name: str, csv_name: str, n_candidates: int = 120):
    df = pd.read_csv(B1_DATA / csv_name)
    num = pd.read_csv(B1_DATA / "numbering_germline.csv")[
        ["antibody_id", "ORG_author_vh_family", "ORG_author_vl_family", "PL_combined_germline_distance"]
    ]
    df = df.merge(num, on="antibody_id", how="left")
    groups = cluster_paired(df["heavy"].tolist(), df["light"].tolist(), 0.90)
    df = df.copy()
    df["cluster_id"] = groups
    col = B2_TARGETS[name]

    geometries = [
        (0.20, 0.20),  # 60/20/20
        (0.15, 0.25),  # 60/15/25
        (0.20, 0.25),  # 55/20/25 approx via greedy
    ]
    cands = []
    seed0 = 20260829 + hash(name) % 10000
    for gi, (fp, fpr) in enumerate(geometries):
        for k in range(n_candidates // len(geometries)):
            seed = seed0 + gi * 1000 + k
            roles = assign_roles(groups, seed, frac_pub=fp, frac_priv=fpr)
            # integrity
            sets = {r: set(groups[roles == r].tolist()) for r in ["Dev", "Public", "Private"]}
            if sets["Dev"] & sets["Public"] or sets["Dev"] & sets["Private"] or sets["Public"] & sets["Private"]:
                continue
            q = split_quality(df, roles, groups, col)
            q.update(
                {
                    "seed": seed,
                    "frac_pub": fp,
                    "frac_priv": fpr,
                    "geometry": f"{1-fp-fpr:.2f}/{fp:.2f}/{fpr:.2f}",
                }
            )
            cands.append((q, roles))

    cands.sort(key=lambda x: x[0]["score"])
    # select top distinct-ish: canonical + 5 shadows with different seeds
    selected = []
    used_seeds = set()
    for q, roles in cands:
        if q["seed"] in used_seeds:
            continue
        # diversify geometries a bit for shadows
        selected.append((q, roles))
        used_seeds.add(q["seed"])
        if len(selected) >= 6:
            break

    names = ["canonical", "shadow_1", "shadow_2", "shadow_3", "shadow_4", "shadow_5"]
    manifest = {"target": name, "threshold": 0.90, "n_candidates_scored": len(cands), "selected": {}}
    for nm, (q, roles) in zip(names, selected):
        out = df[["antibody_id", "cluster_id"]].copy()
        out["role"] = roles
        out["split"] = nm
        path = SPLITS / f"{name.lower()}_{nm}.csv"
        out.to_csv(path, index=False)
        manifest["selected"][nm] = {
            "path": str(path),
            "sha256": sha256_file(path),
            **{k: v for k, v in q.items() if k != "counts"},
            "counts": q["counts"],
        }
        print(name, nm, q["score"], q["counts"], q["geometry"], flush=True)

    # save candidate scoreboard head
    pd.DataFrame([c[0] for c in cands[:50]]).to_csv(DATA / f"{name.lower()}_split_candidates_top50.csv", index=False)
    df[["antibody_id", "cluster_id"]].to_csv(DATA / f"{name.lower()}_clusters.csv", index=False)
    return manifest, df


def main():
    ensure_dirs()
    all_m = {}
    for name, csvn in [("HIC", "hic_full.csv"), ("TmApp", "tmapp_full.csv")]:
        print("DESIGN", name, flush=True)
        m, _ = design_for_target(name, csvn, n_candidates=120)
        all_m[name] = m
    # also overlap triple for apples-to-apples (reuse same logic lightly)
    write_json(CONFIG / "split_manifest.json", all_m)

    lines = [
        "# Split design (pre-model)",
        "",
        "Splits selected **only** from sequence/target/germline/subset diagnostics — no model scores.",
        "",
        "Clustering: paired VH/VL connected components at 90% identity (either chain).",
        "",
        "Candidate pool: 120 seeds × geometries approximating 60/20/20, 60/15/25, 55/20/25.",
        "",
        "Quality score (lower better): size balance + mean/SD/quantile target balance + germline/subset TV distance + NN similarity penalty + min group counts.",
        "",
    ]
    for name, m in all_m.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Candidates scored: {m['n_candidates_scored']}")
        for nm, s in m["selected"].items():
            lines.append(
                f"- **{nm}**: score={s['score']:.3f} counts={s['counts']} geometry≈{s['geometry']} seed={s['seed']}"
            )
        lines.append("")
    lines += [
        "## Rationale",
        "",
        "Canonical = best pre-model score. Shadows = next-best distinct seeds for robustness.",
        "Larger Private fractions included among candidates because N≈350 makes 20% Public noisy for ranking.",
        "Final geometry of each frozen split is whatever the selected seed produced under group integrity.",
        "",
    ]
    (REPORTS / "split_design.md").write_text("\n".join(lines) + "\n")
    print("SPLITS_OK")


if __name__ == "__main__":
    main()
