#!/usr/bin/env python3
"""Paired-antibody clustering + competition-style splits."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    CONFIG,
    DATA,
    REPORTS,
    SPLITS,
    TARGET_COLS,
    ensure_dirs,
    sha256_file,
    write_json,
)


def identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) == len(b):
        return sum(x == y for x, y in zip(a, b)) / len(a)
    # global identity with simple ungapped best length
    n = min(len(a), len(b))
    matches = sum(x == y for x, y in zip(a[:n], b[:n]))
    return matches / max(len(a), len(b))


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
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


def cluster_paired(heavies: list[str], lights: list[str], thr: float) -> np.ndarray:
    n = len(heavies)
    uf = UnionFind(n)
    # O(n^2) acceptable for n~400
    for i in range(n):
        for j in range(i + 1, n):
            if identity(heavies[i], heavies[j]) >= thr or identity(lights[i], lights[j]) >= thr:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    # remap to 0..G-1
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([uniq[r] for r in roots], dtype=int)


def cluster_stats(labels: np.ndarray) -> dict:
    _, counts = np.unique(labels, return_counts=True)
    return {
        "n_groups": int(len(counts)),
        "largest_group": int(counts.max()),
        "median_group_size": float(np.median(counts)),
        "mean_group_size": float(counts.mean()),
        "singleton_fraction": float((counts == 1).mean()),
        "n_singletons": int((counts == 1).sum()),
    }


def assign_split_by_groups(
    groups: np.ndarray,
    seed: int,
    frac_public: float = 0.2,
    frac_private: float = 0.2,
) -> np.ndarray:
    """Return array of roles: Dev/Public/Private without splitting groups."""
    n = len(groups)
    roles = np.array(["Dev"] * n, dtype=object)
    uniq_groups = np.unique(groups)
    rng = np.random.default_rng(seed)

    # weight groups by size for approximate mass fractions
    g_sizes = {g: int((groups == g).sum()) for g in uniq_groups}
    order = uniq_groups.copy()
    rng.shuffle(order)

    target_pub = frac_public * n
    target_priv = frac_private * n
    pub_mass = priv_mass = 0
    pub_gs, priv_gs = set(), set()
    for g in order:
        # assign greedily
        if pub_mass < target_pub:
            pub_gs.add(g)
            pub_mass += g_sizes[g]
        elif priv_mass < target_priv:
            priv_gs.add(g)
            priv_mass += g_sizes[g]
        # else Dev
    for i, g in enumerate(groups):
        if g in pub_gs:
            roles[i] = "Public"
        elif g in priv_gs:
            roles[i] = "Private"
        else:
            roles[i] = "Dev"
    return roles


def nearest_train_test_sim(df: pd.DataFrame, roles: np.ndarray) -> dict:
    train = df[roles == "Dev"]
    out = {}
    for test_name in ["Public", "Private"]:
        test = df[roles == test_name]
        if len(test) == 0 or len(train) == 0:
            out[test_name] = None
            continue
        sims = []
        th = train["heavy"].tolist()
        tl = train["light"].tolist()
        for _, r in test.iterrows():
            best = 0.0
            for h, l in zip(th, tl):
                s = max(identity(r["heavy"], h), identity(r["light"], l))
                if s > best:
                    best = s
            sims.append(best)
        sims = np.array(sims)
        out[test_name] = {
            "mean_nn_sim": float(sims.mean()),
            "median_nn_sim": float(np.median(sims)),
            "p90_nn_sim": float(np.quantile(sims, 0.9)),
            "max_nn_sim": float(sims.max()),
            "frac_nn_sim_ge_0.9": float((sims >= 0.9).mean()),
        }
    return out


def split_diagnostics(df: pd.DataFrame, roles: np.ndarray, groups: np.ndarray, name: str) -> dict:
    d = {
        "split": name,
        "n": {k: int((roles == k).sum()) for k in ["Dev", "Public", "Private"]},
        "n_groups": {
            k: int(len(np.unique(groups[roles == k]))) for k in ["Dev", "Public", "Private"]
        },
        "targets": {},
        "b_cell_subset": {},
        "vh_family": {},
        "nn_similarity": nearest_train_test_sim(df, roles),
    }
    for tname, col in TARGET_COLS.items():
        d["targets"][tname] = {
            k: {
                "mean": float(df.loc[roles == k, col].mean()),
                "sd": float(df.loc[roles == k, col].std()),
                "median": float(df.loc[roles == k, col].median()),
            }
            for k in ["Dev", "Public", "Private"]
        }
    if "b_cell_subset" in df.columns:
        for k in ["Dev", "Public", "Private"]:
            d["b_cell_subset"][k] = df.loc[roles == k, "b_cell_subset"].value_counts().to_dict()
    if "ORG_author_vh_family" in df.columns:
        for k in ["Dev", "Public", "Private"]:
            d["vh_family"][k] = (
                df.loc[roles == k, "ORG_author_vh_family"].value_counts().head(10).to_dict()
            )
    # group disjointness
    sets = {k: set(groups[roles == k].tolist()) for k in ["Dev", "Public", "Private"]}
    d["group_overlap"] = {
        "Dev∩Public": len(sets["Dev"] & sets["Public"]),
        "Dev∩Private": len(sets["Dev"] & sets["Private"]),
        "Public∩Private": len(sets["Public"] & sets["Private"]),
    }
    # exact pair leakage
    def pairs(mask):
        return set(zip(df.loc[mask, "heavy"], df.loc[mask, "light"]))

    d["exact_pair_overlap"] = {
        "Dev∩Public": len(pairs(roles == "Dev") & pairs(roles == "Public")),
        "Dev∩Private": len(pairs(roles == "Dev") & pairs(roles == "Private")),
        "Public∩Private": len(pairs(roles == "Public") & pairs(roles == "Private")),
    }
    return d


def make_splits_for_dataset(
    df: pd.DataFrame,
    prefix: str,
    thr: float,
    seeds: dict[str, int],
) -> dict:
    heavies = df["heavy"].tolist()
    lights = df["light"].tolist()
    groups = cluster_paired(heavies, lights, thr)
    df = df.copy()
    df["cluster_id"] = groups
    stats = cluster_stats(groups)

    manifests = {}
    diags = []
    for split_name, seed in seeds.items():
        roles = assign_split_by_groups(groups, seed=seed)
        out = df[["antibody_id", "cluster_id"]].copy()
        out["role"] = roles
        out["split"] = split_name
        path = SPLITS / f"{prefix}_{split_name}.csv"
        out.to_csv(path, index=False)
        manifests[split_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "seed": seed,
            "n": {k: int((roles == k).sum()) for k in ["Dev", "Public", "Private"]},
            "n_groups": {
                k: int(len(np.unique(groups[roles == k]))) for k in ["Dev", "Public", "Private"]
            },
        }
        # merge organizer cols for diagnostics if present
        diag_df = df.merge(
            pd.read_csv(DATA / "numbering_germline.csv")[
                ["antibody_id", "ORG_author_vh_family", "ORG_author_vl_family"]
            ],
            on="antibody_id",
            how="left",
        )
        diags.append(split_diagnostics(diag_df, roles, groups, split_name))
        assert manifests[split_name]["n"]  # noqa
        # assertions
        assert diags[-1]["group_overlap"]["Dev∩Public"] == 0
        assert diags[-1]["group_overlap"]["Dev∩Private"] == 0
        assert diags[-1]["group_overlap"]["Public∩Private"] == 0
        assert diags[-1]["exact_pair_overlap"]["Dev∩Public"] == 0

    return {"cluster_stats": stats, "manifests": manifests, "diagnostics": diags, "threshold": thr}


def main() -> None:
    ensure_dirs()
    triple = pd.read_csv(DATA / "triple_core.csv")
    num = pd.read_csv(DATA / "numbering_germline.csv")

    # Threshold evaluation on triple only (sequence topology, not labels)
    thr_report = {}
    for thr in [0.85, 0.90, 0.95]:
        labels = cluster_paired(triple["heavy"].tolist(), triple["light"].tolist(), thr)
        thr_report[str(thr)] = cluster_stats(labels)
    write_json(DATA / "cluster_threshold_scan.json", thr_report)

    # Choose canonical threshold
    canonical_thr = 0.90
    reason = "default 90% identity connected-components on either chain"
    s90 = thr_report["0.9"]
    if s90["largest_group"] > 0.4 * len(triple) or s90["n_groups"] < 40:
        # too collapsed or too few groups
        if thr_report["0.95"]["n_groups"] >= 40 and thr_report["0.95"]["largest_group"] <= 0.4 * len(
            triple
        ):
            canonical_thr = 0.95
            reason = (
                f"90% produced largest_group={s90['largest_group']} n_groups={s90['n_groups']}; "
                "raised to 95% based on sequence topology only"
            )
        elif thr_report["0.85"]["n_groups"] >= 30:
            canonical_thr = 0.85
            reason = (
                f"90% topology unsuitable (n_groups={s90['n_groups']}, "
                f"largest={s90['largest_group']}); using 85%"
            )

    seeds = {
        "canonical": 20260329,
        "shadow_1": 101,
        "shadow_2": 202,
        "shadow_3": 303,
    }

    triple_res = make_splits_for_dataset(triple, "triple", canonical_thr, seeds)

    # Secondary full-target splits (one canonical each)
    full_res = {}
    for name in ["psr", "hic", "tmapp"]:
        d = pd.read_csv(DATA / f"{name}_full.csv")
        full_res[name] = make_splits_for_dataset(
            d, f"{name}_full", canonical_thr, {"canonical": 20260329 + hash(name) % 1000}
        )

    manifest = {
        "canonical_threshold": canonical_thr,
        "threshold_reason": reason,
        "threshold_scan": thr_report,
        "triple_core": triple_res,
        "full_targets": {
            k: {"cluster_stats": v["cluster_stats"], "manifests": v["manifests"]}
            for k, v in full_res.items()
        },
    }
    write_json(CONFIG / "split_manifest.json", manifest)

    # Save cluster ids on triple
    groups = cluster_paired(triple["heavy"].tolist(), triple["light"].tolist(), canonical_thr)
    cl = triple[["antibody_id"]].copy()
    cl["cluster_id"] = groups
    cl.to_csv(DATA / "triple_clusters.csv", index=False)

    md = [
        "# Sequence diversity and splits",
        "",
        f"## Threshold scan (TRIPLE_CORE n={len(triple)})",
        "",
        "| thr | n_groups | largest | median | singleton_frac |",
        "|-----|----------:|--------:|-------:|---------------:|",
    ]
    for thr, st in thr_report.items():
        md.append(
            f"| {thr} | {st['n_groups']} | {st['largest_group']} | {st['median_group_size']:.1f} | {st['singleton_fraction']:.2f} |"
        )
    md += [
        "",
        f"**Canonical threshold: {canonical_thr}** — {reason}",
        "",
        "## Triple-core splits",
        "",
    ]
    for d in triple_res["diagnostics"]:
        md += [
            f"### {d['split']}",
            "",
            f"- N Dev/Public/Private: {d['n']}",
            f"- N groups: {d['n_groups']}",
            f"- Group overlap: {d['group_overlap']}",
            f"- Exact pair overlap: {d['exact_pair_overlap']}",
            f"- NN similarity Public: {d['nn_similarity'].get('Public')}",
            f"- NN similarity Private: {d['nn_similarity'].get('Private')}",
            "",
        ]
    (REPORTS / "sequence_diversity_and_splits.md").write_text("\n".join(md) + "\n")
    print("SPLITS_OK", canonical_thr, triple_res["cluster_stats"])


if __name__ == "__main__":
    main()
