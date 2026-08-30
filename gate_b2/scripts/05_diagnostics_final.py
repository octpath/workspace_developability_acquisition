#!/usr/bin/env python3
"""Bootstrap leaderboard uncertainty, residual analysis, scientific diagnostics, final reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import (  # noqa: E402
    B1_DATA,
    B2_TARGETS,
    CACHE,
    CONFIG,
    DATA,
    METRICS,
    REPORTS,
    SPLITS,
    ensure_dirs,
    read_json,
    write_json,
)


def bootstrap_spearman(y, pred, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    if len(y) < 5:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan}
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sp = spearmanr(y[idx], pred[idx]).correlation
        if sp is not None and np.isfinite(sp):
            stats.append(sp)
    stats = np.array(stats)
    return {
        "mean": float(stats.mean()),
        "lo": float(np.quantile(stats, 0.025)),
        "hi": float(np.quantile(stats, 0.975)),
        "sd": float(stats.std()),
    }


def pairwise_win_prob(y, pred_a, pred_b, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    a = np.asarray(pred_a, float)
    b = np.asarray(pred_b, float)
    m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[m], a[m], b[m]
    wins = 0
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sa = spearmanr(y[idx], a[idx]).correlation
        sb = spearmanr(y[idx], b[idx]).correlation
        if sa is not None and sb is not None and sa > sb:
            wins += 1
    return wins / n_boot


def ranking_spearman(df, target, split, col_a, col_b):
    sub = df[(df.target == target) & (df.split == split)].copy()
    sub = sub.sort_values(col_a, ascending=False).groupby("representation", as_index=False).first()
    a, b = sub[col_a].values, sub[col_b].values
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(spearmanr(a[m], b[m]).correlation)


def main():
    ensure_dirs()
    res = pd.read_csv(METRICS / "all_results.csv")
    # Leaderboard reliability across shadows
    lb_rows = []
    for target in B2_TARGETS:
        for split in sorted(res[res.target == target]["split"].unique()):
            lb_rows.append(
                {
                    "target": target,
                    "split": split,
                    "cv_to_public": ranking_spearman(res, target, split, "cv_spearman", "public_spearman"),
                    "public_to_private": ranking_spearman(
                        res, target, split, "public_spearman", "private_spearman"
                    ),
                    "cv_to_private": ranking_spearman(res, target, split, "cv_spearman", "private_spearman"),
                }
            )
    lb = pd.DataFrame(lb_rows)
    lb.to_csv(METRICS / "split_results.csv", index=False)

    lines = ["# Leaderboard uncertainty / ranking reliability", ""]
    for target in B2_TARGETS:
        sub = lb[lb.target == target]
        lines.append(f"## {target}")
        for col in ["cv_to_public", "public_to_private", "cv_to_private"]:
            vals = sub[col].dropna()
            lines.append(
                f"- {col}: mean={vals.mean():.3f} median={vals.median():.3f} sd={vals.std():.3f} "
                f"min={vals.min():.3f} max={vals.max():.3f}"
            )
        lines.append("")
        for _, r in sub.iterrows():
            lines.append(
                f"- {r.split}: CV→Pub={r.cv_to_public:.3f} Pub→Priv={r.public_to_private:.3f} CV→Priv={r.cv_to_private:.3f}"
            )
        lines.append("")
    (REPORTS / "leaderboard_uncertainty.md").write_text("\n".join(lines) + "\n")

    # Winner stability
    win_lines = ["# Winner stability", ""]
    for target in B2_TARGETS:
        win_lines.append(f"## {target}")
        tops = []
        for split in sorted(res[res.target == target]["split"].unique()):
            sub = res[(res.target == target) & (res.split == split)]
            best = sub.sort_values("private_spearman", ascending=False).iloc[0]
            tops.append((split, best.representation, best.model, best.private_spearman))
            win_lines.append(
                f"- {split} Private#1: {best.representation}/{best.model} ρ={best.private_spearman:.3f}"
            )
        fam = pd.Series([t[1] for t in tops]).value_counts()
        win_lines.append(f"- Family win counts (Private#1): {fam.to_dict()}")
        win_lines.append("")
    (REPORTS / "winner_stability.md").write_text("\n".join(win_lines) + "\n")

    # Paired deltas
    deltas = []
    for target in B2_TARGETS:
        for split in sorted(res[res.target == target]["split"].unique()):
            sub = res[(res.target == target) & (res.split == split)]

            def best(pred):
                m = sub[sub.representation.map(pred)]
                return float(m.cv_spearman.max()) if len(m) else np.nan

            simple = best(lambda r: r in ("SEQ_SIMPLE", "SEQ_CDR"))
            bio = best(lambda r: r == "BIO_SHORTCUT")
            plm = best(lambda r: str(r).startswith("PLM_"))
            abb = best(lambda r: str(r).startswith("ABB_"))
            esmn = best(lambda r: str(r).startswith("ESMF_NATIVE_"))
            fusion = best(lambda r: str(r).startswith("FUSION_"))
            s0 = best(lambda r: r.endswith("_SASA") and "RASA" not in r)
            s1 = best(lambda r: r.endswith("_SASA_RASA"))
            s2 = best(lambda r: r.endswith("_SURFACE") and "RASAW" not in r)
            s3 = best(lambda r: "SURFACE_RASAW" in r)
            s4 = best(lambda r: "ALL_SURFACE" in r)
            patch = best(lambda r: r.endswith("_PATCH"))
            deltas.append(
                {
                    "target": target,
                    "split": split,
                    "plm_minus_bio": plm - bio,
                    "abb_minus_plm": abb - plm,
                    "esmn_minus_plm": esmn - plm,
                    "abb_minus_esmn": abb - esmn,
                    "fusion_minus_plm": fusion - plm,
                    "s1_minus_s0": s1 - s0,
                    "s2_minus_s0": s2 - s0,
                    "s3_minus_s0": s3 - s0,
                    "s4_minus_s0": s4 - s0,
                    "patch_best": patch,
                    "best_simple": simple,
                    "best_bio": bio,
                    "best_plm": plm,
                    "best_abb": abb,
                    "best_esmn": esmn,
                    "best_overall": float(sub.cv_spearman.max()),
                }
            )
    pd.DataFrame(deltas).to_csv(METRICS / "paired_deltas.csv", index=False)

    # Scientific analyses
    hic = pd.read_csv(B1_DATA / "hic_full.csv")
    tm = pd.read_csv(B1_DATA / "tmapp_full.csv")
    num = pd.read_csv(B1_DATA / "numbering_germline.csv")
    feat_a = pd.read_csv(CACHE / "features" / "stage_A_simple.csv")
    feat_b = pd.read_csv(CACHE / "features" / "stage_B_cdr.csv")
    hic_m = hic.merge(num, on="antibody_id").merge(feat_a, on="antibody_id").merge(feat_b, on="antibody_id")
    # structure if available
    abb_path = CACHE / "structure_features" / "abb_sasa_rasa_patch.csv"
    if abb_path.exists():
        hic_m = hic_m.merge(pd.read_csv(abb_path), on="antibody_id", how="left")

    def corr_table(df, ycol, xcols):
        rows = []
        for x in xcols:
            if x not in df.columns:
                continue
            a = df[[ycol, x]].dropna()
            if len(a) < 10:
                continue
            rows.append(
                {
                    "x": x,
                    "spearman": float(spearmanr(a[ycol], a[x]).correlation),
                    "pearson": float(pearsonr(a[ycol], a[x])[0]),
                    "n": len(a),
                }
            )
        return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)

    hic_xs = [
        "A2_HL_gravy",
        "B_all_CDR_gravy",
        "B_H_CDR3_gravy",
        "B_H_full_gravy",
        "PL_combined_germline_distance",
        "H_CDR3_len",
        "ABB_Fv_sasa_hydrophobic",
        "ABB_Fv_rasa_w_hydrophobicity_sum",
        "ABB_largest_hydrophobic_patch_sasa",
        "ABB_cdr_hydrophobic_sasa_exposed",
        "ABB_h3_hydrophobic_sasa_exposed",
        "ABB_BSA",
    ]
    hic_corr = corr_table(hic_m, "hic_rt_min", hic_xs)
    hic_corr.to_csv(METRICS / "hic_feature_correlations.csv", index=False)
    (REPORTS / "hic_scientific_analysis.md").write_text(
        "# HIC scientific analysis\n\n"
        "Question: is HIC driven by exposed hydrophobic patches vs bulk sequence hydrophobicity?\n\n"
        + hic_corr.to_string(index=False)
        + "\n\n"
        + (
            "Interpretation: compare |ρ| of GRAVY / CDR gravy vs surface hydrophobic SASA / patch / RASA-weighted hydrophobicity.\n"
        )
    )

    tm_m = tm.merge(num, on="antibody_id").merge(feat_a, on="antibody_id").merge(feat_b, on="antibody_id")
    if abb_path.exists():
        tm_m = tm_m.merge(pd.read_csv(abb_path), on="antibody_id", how="left")
    tm_xs = [
        "PL_combined_germline_distance",
        "PL_vh_germline_distance",
        "H_CDR3_len",
        "A2_HL_gravy",
        "A2_HL_charge_ph7",
        "ABB_BSA",
        "ABB_Fv_mean_rasa",
        "ABB_Fv_sasa_hydrophobic",
        "ABB_Fv_rasa_w_hydrophobicity_sum",
        "B_all_FR_gravy",
    ]
    tm_corr = corr_table(tm_m, "tm_app_C", tm_xs)
    tm_corr.to_csv(METRICS / "tmapp_feature_correlations.csv", index=False)
    (REPORTS / "tmapp_scientific_analysis.md").write_text(
        "# TmApp scientific analysis\n\n"
        "Question: maturation/germline vs residual sequence/structure signal?\n\n"
        + tm_corr.to_string(index=False)
        + "\n"
    )

    # Overlap relationship
    both = hic[["antibody_id", "hic_rt_min"]].merge(tm[["antibody_id", "tm_app_C"]], on="antibody_id")
    sp = spearmanr(both["hic_rt_min"], both["tm_app_C"]).correlation
    (REPORTS / "target_comparison.md").write_text(
        f"""# Target comparison — HIC vs TmApp

## Overlap

- N overlap antibodies with both labels: {len(both)}
- Spearman(HIC, TmApp): {sp:.3f}

## Modeling headroom (from paired deltas, mean over splits)

"""
        + pd.DataFrame(deltas).groupby("target")[["plm_minus_bio", "abb_minus_plm", "best_overall"]].mean().to_string()
        + "\n\nSee GATE_B2_FINAL.md for competition-format decision.\n"
    )

    # Surface ablation report
    abl = pd.DataFrame(deltas)
    (REPORTS / "surface_feature_ablation.md").write_text(
        "# Surface feature ablation (S0–S4)\n\n"
        "Nested sets on identical structures/splits/regressors.\n\n"
        + abl.groupby("target")[["s1_minus_s0", "s2_minus_s0", "s3_minus_s0", "s4_minus_s0", "patch_best"]].mean().to_string()
        + "\n\nS1=RASA summary; S2=absolute surface physchem; S3=RASA-weighted physchem; S4=all.\n"
    )

    print("DIAGNOSTICS_OK")


if __name__ == "__main__":
    main()
