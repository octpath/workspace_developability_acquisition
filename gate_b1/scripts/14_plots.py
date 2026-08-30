#!/usr/bin/env python3
"""Lightweight diagnostic plots for Gate B1."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import DATA, METRICS, PLOTS, REPORTS, TARGET_COLS, ensure_dirs  # noqa: E402


def main():
    ensure_dirs()
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "triple_core.csv")
    num = pd.read_csv(DATA / "numbering_germline.csv")
    m = df.merge(num, on="antibody_id")
    feat = pd.read_csv(Path(__file__).resolve().parents[1] / "cache" / "features" / "stage_A_simple.csv")
    m = m.merge(feat[["antibody_id", "A2_HL_gravy", "A2_HL_charge_ph7"]], on="antibody_id")

    # target distributions
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, (name, col) in zip(axes, TARGET_COLS.items()):
        ax.hist(m[col].dropna(), bins=30, color="#2c5f4a", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(PLOTS / "target_distributions.png", dpi=120)
    plt.close()

    # correlations
    fig, ax = plt.subplots(figsize=(4, 3.5))
    corr = m[list(TARGET_COLS.values())].corr(method="spearman")
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Target Spearman corr")
    fig.tight_layout()
    fig.savefig(PLOTS / "target_correlations.png", dpi=120)
    plt.close()

    # vs germline distance / CDR-H3 / gravy / charge
    pairs = [
        ("PL_combined_germline_distance", "germline_distance"),
        ("H_CDR3_len", "cdrh3_len"),
        ("A2_HL_gravy", "gravy"),
        ("A2_HL_charge_ph7", "charge"),
    ]
    for xcol, tag in pairs:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, (name, col) in zip(axes, TARGET_COLS.items()):
            ax.scatter(m[xcol], m[col], s=8, alpha=0.5, c="#2c5f4a")
            ax.set_xlabel(xcol)
            ax.set_ylabel(name)
        fig.tight_layout()
        fig.savefig(PLOTS / f"target_vs_{tag}.png", dpi=120)
        plt.close()

    # model family scores if available
    res_path = METRICS / "all_results.csv"
    if not res_path.exists():
        frames = []
        for p in [METRICS / "stage_ABC_results.csv", METRICS / "stage_PLM_STR_results.csv"]:
            if p.exists():
                frames.append(pd.read_csv(p))
        if frames:
            res = pd.concat(frames, ignore_index=True)
        else:
            res = None
    else:
        res = pd.read_csv(res_path)

    if res is not None:
        can = res[(res.split == "canonical") & (res.feature_class != "ILLEGAL_FOR_PARTICIPANTS")]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, t in zip(axes, TARGET_COLS):
            sub = can[can.target == t].sort_values("cv_spearman", ascending=False).groupby("representation", as_index=False).first()
            sub = sub.head(12)
            ax.barh(sub["representation"], sub["cv_spearman"], color="#2c5f4a")
            ax.set_title(t)
            ax.set_xlabel("CV Spearman")
        fig.tight_layout()
        fig.savefig(PLOTS / "model_family_cv_scores.png", dpi=120)
        plt.close()

        # shadow distributions of best overall
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, t in zip(axes, TARGET_COLS):
            vals = []
            for split in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
                sub = res[(res.target == t) & (res.split == split) & (res.feature_class != "ILLEGAL_FOR_PARTICIPANTS")]
                vals.append(sub["cv_spearman"].max() if len(sub) else np.nan)
            ax.bar(["can", "s1", "s2", "s3"], vals, color="#2c5f4a")
            ax.set_ylim(0, 1)
            ax.set_title(f"{t} best CV ρ")
        fig.tight_layout()
        fig.savefig(PLOTS / "shadow_best_scores.png", dpi=120)
        plt.close()

    print("PLOTS_OK", list(PLOTS.glob('*.png')))


if __name__ == "__main__":
    main()
