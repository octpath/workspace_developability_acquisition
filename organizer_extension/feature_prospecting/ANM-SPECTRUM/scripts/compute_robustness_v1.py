#!/usr/bin/env python3
"""Target-blind 3-generator robustness + confound correlations for ANM-SPECTRUM_v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
FAMILY = ROOT / "organizer_extension/feature_prospecting/ANM-SPECTRUM"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
GENS = ["esmfold", "abodybuilder2", "boltz2"]
PAIRS = [("esmfold", "abodybuilder2"), ("esmfold", "boltz2"), ("abodybuilder2", "boltz2")]
CONFOUNDS = ["n_CA", "n_edges", "mean_degree", "structure_confidence"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jaccard_topk(a: pd.Series, b: pd.Series, k: int, bottom: bool = False) -> float:
    if bottom:
        ia = set(a.nsmallest(k).index)
        ib = set(b.nsmallest(k).index)
    else:
        ia = set(a.nlargest(k).index)
        ib = set(b.nlargest(k).index)
    inter = len(ia & ib)
    union = len(ia | ib)
    return float(inter / union) if union else np.nan


def main() -> None:
    dfs = {}
    for g in GENS:
        df = pd.read_parquet(FAMILY / f"features_{g}.parquet")
        df = df[df["extraction_status"] == "SUCCESS"].set_index("id")
        dfs[g] = df
        print(g, len(df))

    # common IDs
    ids = set(dfs["esmfold"].index)
    for g in GENS[1:]:
        ids &= set(dfs[g].index)
    ids = sorted(ids)
    print("common", len(ids))

    rows = []
    pair_medians = {f"{a}_vs_{b}": [] for a, b in PAIRS}
    for feat in CANON:
        row = {"feature": feat}
        rhos = []
        for a, b in PAIRS:
            x = dfs[a].loc[ids, feat].astype(float)
            y = dfs[b].loc[ids, feat].astype(float)
            mask = x.notna() & y.notna()
            xx, yy = x[mask], y[mask]
            pr = float(pearsonr(xx, yy).statistic) if len(xx) > 2 else np.nan
            sr = float(spearmanr(xx, yy).statistic) if len(xx) > 2 else np.nan
            key = f"{a}_vs_{b}"
            row[f"{key}_pearson"] = pr
            row[f"{key}_spearman"] = sr
            row[f"{key}_top20_jaccard"] = jaccard_topk(xx, yy, 20, False)
            row[f"{key}_bottom20_jaccard"] = jaccard_topk(xx, yy, 20, True)
            pair_medians[key].append(sr)
            rhos.append(sr)
        row["median_pairwise_spearman"] = float(np.nanmedian(rhos))
        row["min_pairwise_spearman"] = float(np.nanmin(rhos))
        rows.append(row)

    rob = pd.DataFrame(rows)
    pair_family_median = {k: float(np.nanmedian(v)) for k, v in pair_medians.items()}
    min_pair = float(min(pair_family_median.values()))
    if all(v >= 0.8 for v in pair_family_median.values()):
        klass = "ROBUST"
    elif all(v >= 0.5 for v in pair_family_median.values()):
        klass = "MODERATE"
    else:
        klass = "FRAGILE"

    summary = {
        "n_common_ids": len(ids),
        "pairwise_median_spearman": pair_family_median,
        "minimum_pairwise_median_spearman": min_pair,
        "median_of_pairwise_median_spearman": float(np.median(list(pair_family_median.values()))),
        "robustness_class": klass,
    }
    rob.to_csv(FAMILY / "structure_robustness.csv", index=False)
    (FAMILY / "structure_robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # confounds
    conf_rows = []
    for g in GENS:
        df = dfs[g]
        for feat in CANON:
            for c in CONFOUNDS:
                if c not in df.columns:
                    continue
                x = df[feat].astype(float)
                y = df[c].astype(float)
                mask = x.notna() & y.notna()
                sr = float(spearmanr(x[mask], y[mask]).statistic) if mask.sum() > 2 else np.nan
                conf_rows.append({"generator": g, "feature": feat, "confound": c, "spearman": sr})
    pd.DataFrame(conf_rows).to_csv(FAMILY / "confound_spearman.csv", index=False)

    # update hash freeze
    freeze_path = FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json"
    freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else {"files": {}}
    freeze["state"] = "ANM_SPECTRUM_V1_FEATURES_TARGET_BLIND_FROZEN"
    freeze["target_scoring_started_after_feature_freeze"] = False
    freeze["robustness_summary"] = summary
    for p in [
        FAMILY / "structure_robustness.csv",
        FAMILY / "structure_robustness_summary.json",
        FAMILY / "confound_spearman.csv",
        FAMILY / "extraction_qc.csv",
        FAMILY / "FEATURE_MANIFEST.csv",
        FAMILY / "FEATURE_SPEC.json",
        FAMILY / "features_esmfold.parquet",
        FAMILY / "features_abodybuilder2.parquet",
        FAMILY / "features_boltz2.parquet",
    ]:
        freeze["files"][str(p.relative_to(ROOT))] = {
            "sha256": sha256_file(p),
            "nbytes": p.stat().st_size,
        }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
