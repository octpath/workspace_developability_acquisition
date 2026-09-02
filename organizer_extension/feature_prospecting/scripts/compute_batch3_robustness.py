#!/usr/bin/env python3
"""Target-blind robustness for Late Batch3 families."""
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


def jaccard_extreme(a, b, top=True, frac=0.2):
    n = max(1, int(round(frac * len(a))))
    sa = set((a.nlargest(n) if top else a.nsmallest(n)).index)
    sb = set((b.nlargest(n) if top else b.nsmallest(n)).index)
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
            rows.append(
                {
                    "feature": feat,
                    "pair": f"{g1}_vs_{g2}",
                    "n": int(len(a)),
                    "pearson": float(pearsonr(a, b).statistic) if np.std(a) and np.std(b) else np.nan,
                    "spearman": float(spearmanr(a, b).statistic) if np.std(a) and np.std(b) else np.nan,
                    "top20_jaccard": jaccard_extreme(a, b, True),
                    "bottom20_jaccard": jaccard_extreme(a, b, False),
                }
            )
    rdf = pd.DataFrame(rows)
    rdf.to_csv(fam / "structure_robustness.csv", index=False)
    pair_med = {p: float(rdf.loc[rdf.pair == p, "spearman"].median()) for p in rdf.pair.unique()}
    med = float(np.median(list(pair_med.values()))) if pair_med else np.nan
    cls = "ROBUST" if med >= 0.7 else ("MODERATE" if med >= 0.4 else "FRAGILE")
    summary = {"robustness_class": cls, "pairwise_median_spearman": pair_med, "overall_median_spearman": med}
    # titration curve metrics
    if family == "TITRATION-SHAPE":
        curve_stats = {}
        for g1, g2 in PAIRS:
            c1 = pd.read_parquet(fam / f"titration_curves_{g1}.parquet")
            c2 = pd.read_parquet(fam / f"titration_curves_{g2}.parquet")
            rmses, corrs = [], []
            common = set(c1.id) & set(c2.id)
            for aid in common:
                q1 = c1.loc[c1.id == aid].sort_values("pH")["Q_sidechain"].values
                q2 = c2.loc[c2.id == aid].sort_values("pH")["Q_sidechain"].values
                if len(q1) != len(q2) or len(q1) < 5:
                    continue
                rmses.append(float(np.sqrt(np.mean((q1 - q2) ** 2))))
                corrs.append(float(np.corrcoef(q1, q2)[0, 1]) if np.std(q1) and np.std(q2) else np.nan)
            curve_stats[f"{g1}_vs_{g2}"] = {
                "median_curve_RMSE": float(np.median(rmses)) if rmses else np.nan,
                "median_curve_corr": float(np.nanmedian(corrs)) if corrs else np.nan,
                "n": len(rmses),
            }
        summary["curve_agreement"] = curve_stats
    (fam / "structure_robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    out = {}
    for fam in ["TITRATION-SHAPE", "OPENMM-STRAIN"]:
        feats = json.loads((FP / fam / "FEATURE_SPEC.json").read_text())["canonical_features"]
        out[fam] = scalar_robustness(fam, feats)
        print(fam, out[fam]["robustness_class"])
    # SURFACE-DL blocked
    blocked = {
        "robustness_class": "METHOD_BLOCKED",
        "note": "No embeddings extracted",
    }
    (FP / "SURFACE-DL" / "structure_robustness_summary.json").write_text(json.dumps(blocked, indent=2) + "\n")
    out["SURFACE-DL"] = blocked
    print(json.dumps({k: v.get("robustness_class") for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
