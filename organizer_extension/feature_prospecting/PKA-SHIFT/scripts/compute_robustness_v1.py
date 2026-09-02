#!/usr/bin/env python3
"""Target-blind three-generator robustness for PKA-SHIFT_v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
FAMILY = ROOT / "organizer_extension/feature_prospecting/PKA-SHIFT"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
GENS = ["esmfold", "abodybuilder2", "boltz2"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def top_jaccard(a: pd.Series, b: pd.Series, k=20):
    ids = a.index.intersection(b.index)
    if len(ids) < k:
        return np.nan
    ta = set(a.loc[ids].nlargest(k).index)
    tb = set(b.loc[ids].nlargest(k).index)
    return len(ta & tb) / len(ta | tb)


def bottom_jaccard(a: pd.Series, b: pd.Series, k=20):
    ids = a.index.intersection(b.index)
    if len(ids) < k:
        return np.nan
    ta = set(a.loc[ids].nsmallest(k).index)
    tb = set(b.loc[ids].nsmallest(k).index)
    return len(ta & tb) / len(ta | tb)


def main():
    feats = {
        g: pd.read_parquet(FAMILY / f"features_{g}.parquet")
        .query("extraction_status == 'SUCCESS'")
        .set_index("id")
        for g in GENS
    }
    rows = []
    pair_med = {}
    for a, b, key in [
        ("esmfold", "abodybuilder2", "esmfold_vs_abodybuilder2"),
        ("esmfold", "boltz2", "esmfold_vs_boltz2"),
        ("abodybuilder2", "boltz2", "abodybuilder2_vs_boltz2"),
    ]:
        ids = feats[a].index.intersection(feats[b].index)
        spears = []
        for f in CANON:
            xa = feats[a].loc[ids, f].astype(float)
            xb = feats[b].loc[ids, f].astype(float)
            mask = xa.notna() & xb.notna()
            if mask.sum() < 10:
                pr = sr = np.nan
            else:
                pr = float(pearsonr(xa[mask], xb[mask]).statistic)
                sr = float(spearmanr(xa[mask], xb[mask]).statistic)
            if np.isfinite(sr) and not (xa[mask].nunique() <= 1 or xb[mask].nunique() <= 1):
                spears.append(sr)
            diff = (xa - xb).abs()
            rows.append(
                {
                    "pair": key,
                    "feature": f,
                    "n": int(mask.sum()),
                    "pearson": pr,
                    "spearman": sr,
                    "abs_diff_median": float(diff[mask].median()) if mask.any() else np.nan,
                    "abs_diff_p95": float(diff[mask].quantile(0.95)) if mask.any() else np.nan,
                    "top20_jaccard": top_jaccard(xa, xb),
                    "bottom20_jaccard": bottom_jaccard(xa, xb),
                }
            )
        pair_med[key] = float(np.median(spears)) if spears else np.nan

    # residue-level delta_pKa consistency
    res = {g: pd.read_parquet(FAMILY / f"residue_pka_{g}.parquet") for g in GENS}
    res_rows = []
    keys = ["id", "chain", "sequence_index", "residue_type"]
    for a, b, key in [
        ("esmfold", "abodybuilder2", "esmfold_vs_abodybuilder2"),
        ("esmfold", "boltz2", "esmfold_vs_boltz2"),
        ("abodybuilder2", "boltz2", "abodybuilder2_vs_boltz2"),
    ]:
        ma = res[a][keys + ["delta_pKa"]].drop_duplicates(keys).rename(columns={"delta_pKa": "da"})
        mb = res[b][keys + ["delta_pKa"]].drop_duplicates(keys).rename(columns={"delta_pKa": "db"})
        merged = ma.merge(mb, on=keys, how="inner")
        pooled_sp = (
            float(spearmanr(merged["da"], merged["db"]).statistic) if len(merged) > 10 else np.nan
        )
        per = []
        for aid, g in merged.groupby("id"):
            if len(g) < 5:
                continue
            per.append(float(spearmanr(g["da"], g["db"]).statistic))
        res_rows.append(
            {
                "pair": key,
                "n_residue_pairs": int(len(merged)),
                "pooled_spearman_delta_pKa": pooled_sp,
                "per_ab_median_spearman": float(np.nanmedian(per)) if per else np.nan,
                "per_ab_mean_spearman": float(np.nanmean(per)) if per else np.nan,
                "n_antibodies_with_ge5": int(len(per)),
            }
        )

    rob = pd.DataFrame(rows)
    rob.to_csv(FAMILY / "structure_robustness.csv", index=False)
    pd.DataFrame(res_rows).to_csv(FAMILY / "residue_level_robustness.csv", index=False)

    vals = list(pair_med.values())
    if all(v >= 0.8 for v in vals):
        cls = "ROBUST"
    elif all(v >= 0.5 for v in vals):
        cls = "MODERATE"
    else:
        cls = "FRAGILE"

    summary = {
        "robustness_class": cls,
        "pairwise_median_spearman": pair_med,
        "residue_level": res_rows,
        "note": "FRAGILE is scientific result for structure-sensitive pKa; do not discard",
    }
    (FAMILY / "structure_robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # update freeze hashes
    freeze = json.loads((FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json").read_text())
    for p in [
        FAMILY / "structure_robustness.csv",
        FAMILY / "structure_robustness_summary.json",
        FAMILY / "residue_level_robustness.csv",
    ]:
        freeze["files"][str(p.relative_to(ROOT))] = {"sha256": sha256_file(p), "nbytes": p.stat().st_size}
    freeze["target_scoring_started_after_feature_freeze"] = False
    (FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(cls, pair_med)


if __name__ == "__main__":
    main()
