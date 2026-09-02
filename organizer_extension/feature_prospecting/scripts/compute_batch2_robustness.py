#!/usr/bin/env python3
"""Target-blind 3-generator robustness for Advanced Batch2 families."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
GENS = ["esmfold", "abodybuilder2", "boltz2"]
PAIRS = [("esmfold", "abodybuilder2"), ("esmfold", "boltz2"), ("abodybuilder2", "boltz2")]


def jaccard_extreme(a: pd.Series, b: pd.Series, top=True, frac=0.2):
    n = max(1, int(round(frac * len(a))))
    if top:
        sa, sb = set(a.nlargest(n).index), set(b.nlargest(n).index)
    else:
        sa, sb = set(a.nsmallest(n).index), set(b.nsmallest(n).index)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan


def scalar_robustness(family: str, features: list[str]):
    fam = FP / family
    dfs = {
        g: pd.read_parquet(fam / f"features_{g}.parquet").query("extraction_status == 'SUCCESS'").set_index("id")
        for g in GENS
    }
    rows = []
    for feat in features:
        for g1, g2 in PAIRS:
            common = dfs[g1].index.intersection(dfs[g2].index)
            a, b = dfs[g1].loc[common, feat].astype(float), dfs[g2].loc[common, feat].astype(float)
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            if len(a) < 10:
                continue
            pr = float(pearsonr(a, b).statistic) if np.std(a) and np.std(b) else np.nan
            sr = float(spearmanr(a, b).statistic) if np.std(a) and np.std(b) else np.nan
            rows.append(
                {
                    "feature": feat,
                    "pair": f"{g1}_vs_{g2}",
                    "n": int(len(a)),
                    "pearson": pr,
                    "spearman": sr,
                    "top20_jaccard": jaccard_extreme(a, b, True),
                    "bottom20_jaccard": jaccard_extreme(a, b, False),
                }
            )
    rdf = pd.DataFrame(rows)
    rdf.to_csv(fam / "structure_robustness.csv", index=False)
    pair_med = {
        r["pair"]: float(rdf.loc[rdf.pair == r["pair"], "spearman"].median())
        for r in rows
        if True
    }
    # unique pairs
    pair_med = {p: float(rdf.loc[rdf.pair == p, "spearman"].median()) for p in rdf.pair.unique()}
    med = float(np.median(list(pair_med.values()))) if pair_med else np.nan
    if med >= 0.7:
        cls = "ROBUST"
    elif med >= 0.4:
        cls = "MODERATE"
    else:
        cls = "FRAGILE"
    summary = {"robustness_class": cls, "pairwise_median_spearman": pair_med, "overall_median_spearman": med}
    (fam / "structure_robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def di_robustness():
    fam = FP / "3DI-FROZEN"
    spec = json.loads((fam / "FEATURE_SPEC.json").read_text())
    thr = spec["embedding_robustness_thresholds_prespecified"]
    tokens = {g: pd.read_parquet(fam / f"tokens_{g}.parquet").set_index("id") for g in GENS}
    embs = {}
    for g in GENS:
        f = pd.read_parquet(fam / f"features_{g}.parquet").query("extraction_status == 'SUCCESS'").set_index("id")
        cols = [c for c in f.columns if c.startswith("hl_")]
        embs[g] = f[cols]

    def token_id(a, b):
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        return float(np.mean([x == y for x, y in zip(a, b)]))

    pair_stats = {}
    for g1, g2 in PAIRS:
        common = tokens[g1].index.intersection(tokens[g2].index).intersection(embs[g1].index).intersection(embs[g2].index)
        tok_H, tok_L, cos_H, cos_L, cos_HL = [], [], [], [], []
        for aid in common:
            t1, t2 = tokens[g1].loc[aid], tokens[g2].loc[aid]
            tok_H.append(token_id(t1["H_3di"], t2["H_3di"]))
            tok_L.append(token_id(t1["L_3di"], t2["L_3di"]))
            e1, e2 = embs[g1].loc[aid].values.astype(float), embs[g2].loc[aid].values.astype(float)
            h1, h2 = e1[:1024], e2[:1024]
            l1, l2 = e1[1024:], e2[1024:]
            def cos(u, v):
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                return float(np.dot(u, v) / (nu * nv)) if nu and nv else np.nan
            cos_H.append(cos(h1, h2))
            cos_L.append(cos(l1, l2))
            cos_HL.append(cos(e1, e2))
        # distance matrices
        X1 = embs[g1].loc[common].values.astype(float)
        X2 = embs[g2].loc[common].values.astype(float)
        # pairwise euclidean upper triangle
        def upper_dist(X):
            n = len(X)
            d = []
            for i in range(n):
                for j in range(i + 1, n):
                    d.append(np.linalg.norm(X[i] - X[j]))
            return np.asarray(d)
        d1, d2 = upper_dist(X1), upper_dist(X2)
        sp = float(spearmanr(d1, d2).statistic) if len(d1) > 5 else np.nan
        pair_stats[f"{g1}_vs_{g2}"] = {
            "median_token_identity_H": float(np.median(tok_H)),
            "median_token_identity_L": float(np.median(tok_L)),
            "median_token_identity_HL_mean": float(np.median(0.5 * (np.array(tok_H) + np.array(tok_L)))),
            "median_cosine_H": float(np.nanmedian(cos_H)),
            "median_cosine_L": float(np.nanmedian(cos_L)),
            "median_cosine_HL": float(np.nanmedian(cos_HL)),
            "distance_matrix_spearman": sp,
            "n": int(len(common)),
        }

    def meet(stats, level):
        t = thr[level]
        return (
            stats["median_token_identity_HL_mean"] >= t["median_token_identity_HL_mean"]
            and stats["median_cosine_HL"] >= t["median_cosine_HL"]
            and stats["distance_matrix_spearman"] >= t["distance_matrix_spearman"]
        )

    labels = {}
    for p, st in pair_stats.items():
        if meet(st, "ROBUST"):
            labels[p] = "ROBUST"
        elif meet(st, "MODERATE"):
            labels[p] = "MODERATE"
        else:
            labels[p] = "FRAGILE"
    order = {"ROBUST": 2, "MODERATE": 1, "FRAGILE": 0}
    overall = min(labels.values(), key=lambda x: order[x])
    summary = {
        "robustness_class": overall,
        "per_pair_class": labels,
        "pair_stats": pair_stats,
        "thresholds": thr,
    }
    (fam / "structure_robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    flat = []
    for p, st in pair_stats.items():
        flat.append({"pair": p, "class": labels[p], **st})
    pd.DataFrame(flat).to_csv(fam / "structure_robustness.csv", index=False)
    return summary


def main():
    hydro = json.loads((FP / "HYDRO-FIELD/FEATURE_SPEC.json").read_text())["canonical_features"]
    cop = json.loads((FP / "ELEC-HYDRO-COPATCH/FEATURE_SPEC.json").read_text())["canonical_features"]
    iface = json.loads((FP / "INTERFACE-ENERGY/FEATURE_SPEC.json").read_text())["canonical_features"]
    out = {
        "HYDRO-FIELD": scalar_robustness("HYDRO-FIELD", hydro),
        "ELEC-HYDRO-COPATCH": scalar_robustness("ELEC-HYDRO-COPATCH", cop),
        "INTERFACE-ENERGY": scalar_robustness("INTERFACE-ENERGY", iface),
        "3DI-FROZEN": di_robustness(),
    }
    print(json.dumps({k: v.get("robustness_class") for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
