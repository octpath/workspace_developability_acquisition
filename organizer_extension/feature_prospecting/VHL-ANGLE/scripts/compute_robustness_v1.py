#!/usr/bin/env python3
"""Target-blind robustness for VHL-ANGLE_v1 (ABangle parameters)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
FAMILY = ROOT / "organizer_extension/feature_prospecting/VHL-ANGLE"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
ANGULAR = set(SPEC["angular_features"])
GENS = ["esmfold", "abodybuilder2", "boltz2"]
PAIRS = [("esmfold", "abodybuilder2"), ("esmfold", "boltz2"), ("abodybuilder2", "boltz2")]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ang_diff(a, b):
    """Smallest absolute difference for degrees in [-180,180]-like space."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = (d + 180.0) % 360.0 - 180.0
    return np.abs(d)


def main():
    dfs = {}
    for g in GENS:
        df = pd.read_parquet(FAMILY / f"features_{g}.parquet")
        df = df[df.extraction_status == "SUCCESS"].set_index("id")
        dfs[g] = df
        print(g, len(df))
    ids = sorted(set.intersection(*(set(dfs[g].index) for g in GENS)))
    print("common", len(ids))

    rows = []
    pair_medians = {f"{a}_vs_{b}": [] for a, b in PAIRS}
    for feat in CANON:
        row = {"feature": feat, "is_angular": feat in ANGULAR}
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
            if feat in ANGULAR:
                ad = ang_diff(xx, yy)
                row[f"{key}_mean_abs_ang_diff_deg"] = float(np.mean(ad))
                row[f"{key}_median_abs_ang_diff_deg"] = float(np.median(ad))
            else:
                row[f"{key}_mean_abs_diff"] = float(np.mean(np.abs(xx - yy)))
                row[f"{key}_median_abs_diff"] = float(np.median(np.abs(xx - yy)))
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
        "note_angular": "Spearman reported on raw ABangle degrees; also mean/median absolute circular differences for angles",
    }
    rob.to_csv(FAMILY / "structure_robustness.csv", index=False)
    (FAMILY / "structure_robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    freeze_path = FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["state"] = "VHL_ANGLE_V1_FEATURES_TARGET_BLIND_FROZEN"
    freeze["robustness_summary"] = summary
    for p in [
        FAMILY / "structure_robustness.csv",
        FAMILY / "structure_robustness_summary.json",
        FAMILY / "FEATURE_SPEC.json",
        FAMILY / "FEATURE_MANIFEST.csv",
        FAMILY / "extraction_qc.csv",
        FAMILY / "features_esmfold.parquet",
        FAMILY / "features_abodybuilder2.parquet",
        FAMILY / "features_boltz2.parquet",
    ]:
        freeze["files"][str(p.relative_to(ROOT))] = {"sha256": sha256_file(p), "nbytes": p.stat().st_size}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
