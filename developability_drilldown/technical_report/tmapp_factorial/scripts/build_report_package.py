#!/usr/bin/env python3
"""Build human-readable TmApp factorial technical-report package (no training)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]  # developability_drilldown
PKG = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATA = PKG / "data"
FIG = PKG / "figures"
TAB = PKG / "tables"
NOTES = PKG / "notes"

TERM = yaml.safe_load((PKG / "terminology.yaml").read_text())
TOPO_ORDER = TERM["topology_order"]
ANN_ORDER = TERM["annotation_order"]
REP_ORDER = TERM["representation_order"]

LEGACY_TO_ID = {k: v["topology_id"] for k, v in TERM["topology"].items()}
LEGACY_TO_DISP = {k: v["topology_display"] for k, v in TERM["topology"].items()}
ANN_DISP = {k: v["annotation_display"] for k, v in TERM["annotation"].items()}
REP_DISP = {k: v["representation_display"] for k, v in TERM["representation"].items()}
REP_DOMAIN = {k: v["domain"] for k, v in TERM["representation"].items()}

# For PAIR-SEPARATE: map family to (paired_raw, separate_raw) by true context
PAIR_MAP = {
    "AbLang2": ("ablang2_unpaired", "ablang2_paired"),  # PAIRED_NATIVE, SEPARATE_CHAIN
    "CurrAb": ("currab_paired", "currab_unpaired"),
}


def savefig(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_master() -> pd.DataFrame:
    src = pd.read_csv(RES / "TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv")
    assert len(src) == 200, len(src)
    rows = []
    for _, r in src.iterrows():
        raw = r["representation"]
        leg = r["topology"]
        rows.append(
            {
                "experiment_code": r["experiment_code"],
                "effective_code": r.get("effective_code", r["experiment_code"]),
                "representation_raw": raw,
                "representation_display": REP_DISP[raw],
                "representation_context": r["representation_context"],
                "plm_family": r.get("plm_family"),
                "paired_match_group": r.get("paired_match_group"),
                "topology_legacy": leg,
                "topology_id": LEGACY_TO_ID[leg],
                "topology_display": LEGACY_TO_DISP[leg],
                "annotation_id": r["annotation"],
                "annotation_display": ANN_DISP[r["annotation"]],
                "primary_mae": float(r["primary_mae"]),
                "shadow_mae": float(r["shadow_mae"]),
                "mean_ps": float(r["mean_ps"]),
                "worst_ps": float(r["worst_ps"]),
                "abs_ps": float(r["abs_ps"]),
                "raw_dim": r.get("raw_dim"),
                "execution_status": r["execution_status"],
                "reused_or_new": "reused" if r["execution_status"] == "REUSE" else "new",
                "reuse_source": r.get("reuse_source") or "",
            }
        )
    m = pd.DataFrame(rows)
    # verify numeric identity
    for col in ("primary_mae", "shadow_mae", "mean_ps", "worst_ps"):
        assert np.allclose(
            m.sort_values("experiment_code")[col].to_numpy(),
            src.sort_values("experiment_code")[col].to_numpy(),
        )
    assert len(m) == 200
    assert set(m.topology_id) == set(TOPO_ORDER)
    # AbLang2 context naming
    a2s = m[m.representation_raw == "ablang2_paired"].iloc[0]
    a2p = m[m.representation_raw == "ablang2_unpaired"].iloc[0]
    assert a2s.representation_context == "SEPARATE_CHAIN"
    assert a2p.representation_context == "PAIRED_NATIVE"
    assert "separate-chain" in a2s.representation_display
    assert "paired H/L" in a2p.representation_display
    c_p = m[m.representation_raw == "currab_paired"].iloc[0]
    c_s = m[m.representation_raw == "currab_unpaired"].iloc[0]
    assert c_p.representation_context == "PAIRED_NATIVE"
    assert c_s.representation_context == "SEPARATE_CHAIN"
    m.to_csv(DATA / "tmapp_factorial_human_master.csv", index=False)
    return m


def figure1_schematic() -> None:
    fig = plt.figure(figsize=(12.5, 8.2))
    # Panel A
    ax = fig.add_axes([0.05, 0.72, 0.9, 0.24])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("A. Overall factorial pipeline", loc="left", fontsize=13, fontweight="bold")
    boxes = [
        (0.3, 1.0, "Residue\nrepresentation"),
        (2.7, 1.0, "Annotation\ninjection"),
        (5.1, 1.0, "H/L\ntopology"),
        (7.5, 1.0, "TmApp\nprediction"),
    ]
    for x, y, t in boxes:
        ax.add_patch(
            FancyBboxPatch((x, y), 1.8, 1.2, boxstyle="round,pad=0.04,rounding_size=0.15",
                           facecolor="#E8EEF5", edgecolor="#2C3E50", linewidth=1.4)
        )
        ax.text(x + 0.9, y + 0.6, t, ha="center", va="center", fontsize=10)
    for x0 in (2.1, 4.5, 6.9):
        ax.annotate("", xy=(x0 + 0.55, 1.6), xytext=(x0, 1.6),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.6))
    ax.text(5, 0.25,
            "Controlled comparison after frozen residue embeddings (or learned AA for Scratch).\n"
            "REG = a learned chain-level token that gathers information from residues in the downstream Transformer.",
            ha="center", va="center", fontsize=8.5, color="#333")

    # Panel B — five topologies
    axb = fig.add_axes([0.04, 0.04, 0.92, 0.64])
    axb.set_xlim(0, 10)
    axb.set_ylim(0, 7)
    axb.axis("off")
    axb.set_title("B. Downstream H/L topologies (presentation IDs)", loc="left", fontsize=13, fontweight="bold")

    def draw_topo(ax, x0, y0, title, mode):
        ax.text(x0 + 1.7, y0 + 2.85, title, ha="center", fontsize=10, fontweight="bold")
        # H residues (blue), L residues (orange), REG_H, REG_L
        h_box = FancyBboxPatch((x0 + 0.15, y0 + 1.5), 1.3, 1.0, boxstyle="round,pad=0.02",
                               facecolor="#D6EAF8", edgecolor="#1A5276")
        l_box = FancyBboxPatch((x0 + 1.95, y0 + 1.5), 1.3, 1.0, boxstyle="round,pad=0.02",
                               facecolor="#FDEBD0", edgecolor="#9A7B4F")
        ax.add_patch(h_box)
        ax.add_patch(l_box)
        ax.text(x0 + 0.8, y0 + 2.0, "H residues", ha="center", va="center", fontsize=7.5)
        ax.text(x0 + 2.6, y0 + 2.0, "L residues", ha="center", va="center", fontsize=7.5)
        rh = FancyBboxPatch((x0 + 0.35, y0 + 0.35), 0.9, 0.7, boxstyle="round,pad=0.02",
                            facecolor="#AED6F1", edgecolor="#1A5276", linewidth=1.3)
        rl = FancyBboxPatch((x0 + 2.15, y0 + 0.35), 0.9, 0.7, boxstyle="round,pad=0.02",
                            facecolor="#F5CBA7", edgecolor="#9A7B4F", linewidth=1.3)
        ax.add_patch(rh)
        ax.add_patch(rl)
        ax.text(x0 + 0.8, y0 + 0.7, "REG_H", ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.text(x0 + 2.6, y0 + 0.7, "REG_L", ha="center", va="center", fontsize=7.5, fontweight="bold")
        # within-chain residue->REG
        ax.annotate("", xy=(x0 + 0.8, y0 + 1.05), xytext=(x0 + 0.8, y0 + 1.5),
                    arrowprops=dict(arrowstyle="->", color="#1A5276", lw=1.0))
        ax.annotate("", xy=(x0 + 2.6, y0 + 1.05), xytext=(x0 + 2.6, y0 + 1.5),
                    arrowprops=dict(arrowstyle="->", color="#9A7B4F", lw=1.0))

        if mode == "SEP":
            ax.text(x0 + 1.7, y0 + 1.35, "no H↔L path", ha="center", fontsize=7, color="#922B21")
        elif mode == "JOINT":
            # residue-residue joint + REG joint
            ax.annotate("", xy=(x0 + 1.95, y0 + 2.15), xytext=(x0 + 1.45, y0 + 2.15),
                        arrowprops=dict(arrowstyle="<->", color="#6C3483", lw=1.6))
            ax.annotate("", xy=(x0 + 2.15, y0 + 0.7), xytext=(x0 + 1.25, y0 + 0.7),
                        arrowprops=dict(arrowstyle="<->", color="#6C3483", lw=1.6))
            ax.annotate("", xy=(x0 + 2.1, y0 + 1.7), xytext=(x0 + 1.2, y0 + 1.0),
                        arrowprops=dict(arrowstyle="<->", color="#6C3483", lw=1.0, connectionstyle="arc3,rad=0.2"))
            ax.text(x0 + 1.7, y0 - 0.05, "unrestricted joint", ha="center", fontsize=7, color="#6C3483")
        elif mode == "REG-SEP":
            ax.annotate("", xy=(x0 + 1.95, y0 + 2.15), xytext=(x0 + 1.45, y0 + 2.15),
                        arrowprops=dict(arrowstyle="<->", color="#6C3483", lw=1.6))
            ax.text(x0 + 1.7, y0 + 1.2, "REGs stay\nchain-specific", ha="center", fontsize=7, color="#1A5276")
        elif mode == "XREG":
            ax.annotate("", xy=(x0 + 1.95, y0 + 1.9), xytext=(x0 + 1.25, y0 + 0.9),
                        arrowprops=dict(arrowstyle="->", color="#117A65", lw=1.5))
            ax.annotate("", xy=(x0 + 1.45, y0 + 1.9), xytext=(x0 + 2.15, y0 + 0.9),
                        arrowprops=dict(arrowstyle="->", color="#117A65", lw=1.5))
            ax.text(x0 + 1.7, y0 - 0.05, "REG↔opposite residues", ha="center", fontsize=7, color="#117A65")
        elif mode == "FUSE":
            ax.annotate("", xy=(x0 + 2.15, y0 + 0.7), xytext=(x0 + 1.25, y0 + 0.7),
                        arrowprops=dict(arrowstyle="<->", color="#B9770E", lw=1.8))
            ax.text(x0 + 1.7, y0 - 0.05, "REG_H ↔ REG_L fusion", ha="center", fontsize=7, color="#B9770E")

    specs = [
        (0.2, 3.6, "SEP — Separate H/L", "SEP"),
        (3.5, 3.6, "JOINT — Full Joint", "JOINT"),
        (6.8, 3.6, "REG-SEP — Joint residues,\nseparate REGs", "REG-SEP"),
        (1.8, 0.25, "XREG — Cross-REG Read", "XREG"),
        (5.2, 0.25, "FUSE — REG Fusion", "FUSE"),
    ]
    for args in specs:
        draw_topo(axb, *args)
    savefig(fig, "Figure_01_factorial_topology_schematic")


def figure2_best_per_rep(m: pd.DataFrame) -> None:
    for metric, stem, title in [
        ("mean_ps", "Figure_02_best_per_representation_mean", "Best configuration per representation (by mean Primary/Shadow MAE)"),
        ("worst_ps", "Figure_02_best_per_representation_worst", "Best configuration per representation (by worst Primary/Shadow MAE)"),
    ]:
        rows = []
        for disp in REP_ORDER:
            sub = m[m.representation_display == disp]
            r = sub.sort_values([metric, "mean_ps", "abs_ps"]).iloc[0]
            rows.append(r)
        df = pd.DataFrame(rows).sort_values(metric, ascending=True)
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        y = np.arange(len(df))
        ax.barh(y, df[metric], color="#5D6D7E", height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(df.representation_display)
        ax.set_xlabel(f"{metric.replace('_', ' ')}  (lower is better)")
        ax.set_title(title, fontsize=12)
        for i, r in enumerate(df.itertuples()):
            ax.text(r._asdict()[metric] + 0.01, i,
                    f"{r.topology_id} / {r.annotation_id}  ({r.experiment_code})  {r._asdict()[metric]:.3f}",
                    va="center", fontsize=8)
        ax.set_xlim(df[metric].min() - 0.05, df[metric].max() + 0.55)
        ax.invert_yaxis()
        fig.tight_layout()
        savefig(fig, stem)


def _heatmap(ax, mat, title, cmap, vmin, vmax, center=None, cbar=True, annotate=True):
    if center is not None:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(TOPO_ORDER)))
    ax.set_xticklabels(TOPO_ORDER, fontsize=8)
    ax.set_yticks(range(len(ANN_ORDER)))
    ax.set_yticklabels(ANN_ORDER, fontsize=8)
    ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="black" if abs(v - (vmin + vmax) / 2) < (vmax - vmin) * 0.35 else "white")
    return im


def pivot_abs(m: pd.DataFrame, disp: str) -> np.ndarray:
    sub = m[m.representation_display == disp]
    mat = sub.pivot_table(index="annotation_id", columns="topology_id", values="mean_ps")
    mat = mat.reindex(index=ANN_ORDER, columns=TOPO_ORDER)
    return mat.to_numpy(float)


def figure3_absolute_landscape(m: pd.DataFrame) -> None:
    vals = [pivot_abs(m, d) for d in REP_ORDER]
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    groups = [
        ("Figure_03A_absolute_scratch", [("Scratch", "Scratch control")]),
        ("Figure_03B_absolute_general_protein", [
            ("ESM-1b", "ESM-1b"), ("ESM-2", "ESM-2"), ("ESM-C 600M", "ESM-C 600M")
        ]),
        ("Figure_03C_absolute_antibody_ablang1_ablingua", [
            ("AbLang1", "AbLang1"), ("AbLingua", "AbLingua")
        ]),
        ("Figure_03D_absolute_ablang2", [
            ("AbLang2 (separate-chain)", "AbLang2 (separate-chain)"),
            ("AbLang2 (paired H/L)", "AbLang2 (paired H/L)"),
        ]),
        ("Figure_03E_absolute_currab", [
            ("CurrAb (separate-chain)", "CurrAb (separate-chain)"),
            ("CurrAb (paired H/L)", "CurrAb (paired H/L)"),
        ]),
    ]
    for stem, items in groups:
        n = len(items)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1.2, 3.8), squeeze=False)
        im = None
        for ax, (disp, title) in zip(axes[0], items):
            mat = pivot_abs(m, disp)
            im = _heatmap(ax, mat, title, "viridis_r", vmin, vmax)
            # individual export
            fig_i, ax_i = plt.subplots(figsize=(5.2, 3.8))
            im_i = _heatmap(ax_i, mat, f"{disp}: mean(P,S)", "viridis_r", vmin, vmax)
            fig_i.colorbar(im_i, ax=ax_i, fraction=0.046, pad=0.04, label="mean(P,S) MAE")
            fig_i.tight_layout()
            safe = disp.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            savefig(fig_i, f"Figure_03_individual_{safe}")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="mean(P,S) MAE")
        fig.suptitle("Absolute performance landscape (shared color scale)", fontsize=12, y=1.02)
        fig.tight_layout()
        savefig(fig, stem)


def figure4_topology_effects(m: pd.DataFrame) -> None:
    # Δ = MAE(topo) - MAE(SEP) for each rep × annotation
    rows = []
    for disp in REP_ORDER:
        for ann in ANN_ORDER:
            base = m[(m.representation_display == disp) & (m.annotation_id == ann) & (m.topology_id == "SEP")]
            if len(base) != 1:
                continue
            b = float(base.iloc[0].mean_ps)
            for tid in ["JOINT", "REG-SEP", "XREG", "FUSE"]:
                x = m[(m.representation_display == disp) & (m.annotation_id == ann) & (m.topology_id == tid)]
                if len(x) != 1:
                    continue
                rows.append({"representation_display": disp, "annotation_id": ann, "topology_id": tid,
                             "delta_mean": float(x.iloc[0].mean_ps) - b})
    g = pd.DataFrame(rows)
    g.to_csv(DATA / "topology_gains_vs_SEP_human.csv", index=False)
    lim = float(np.nanmax(np.abs(g.delta_mean)))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    for ax, ann in zip(axes.ravel(), ANN_ORDER):
        mat = g[g.annotation_id == ann].pivot_table(
            index="representation_display", columns="topology_id", values="delta_mean"
        ).reindex(index=REP_ORDER, columns=["JOINT", "REG-SEP", "XREG", "FUSE"])
        im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["JOINT", "REG-SEP", "XREG", "FUSE"], fontsize=8)
        ax.set_yticks(range(len(REP_ORDER)))
        ax.set_yticklabels(REP_ORDER, fontsize=7.5)
        ax.set_title(f"Annotation = {ann}", fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.to_numpy(float)[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                 label="ΔMAE = MAE(topology) − MAE(SEP)\n(negative = better than SEP)")
    fig.suptitle("Topology effects relative to SEP", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "Figure_04_topology_effects_vs_SEP")


def figure5_annotation_effects(m: pd.DataFrame) -> None:
    rows = []
    for disp in REP_ORDER:
        for tid in TOPO_ORDER:
            base = m[(m.representation_display == disp) & (m.topology_id == tid) & (m.annotation_id == "BASE")]
            if len(base) != 1:
                continue
            b = float(base.iloc[0].mean_ps)
            for ann in ["IMGT", "REGION", "FULL"]:
                x = m[(m.representation_display == disp) & (m.topology_id == tid) & (m.annotation_id == ann)]
                if len(x) != 1:
                    continue
                rows.append({"representation_display": disp, "topology_id": tid, "annotation_id": ann,
                             "delta_mean": float(x.iloc[0].mean_ps) - b})
    g = pd.DataFrame(rows)
    g.to_csv(DATA / "annotation_gains_vs_BASE_human.csv", index=False)
    lim = float(np.nanmax(np.abs(g.delta_mean)))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.8), sharey=True)
    for ax, ann in zip(axes, ["IMGT", "REGION", "FULL"]):
        mat = g[g.annotation_id == ann].pivot_table(
            index="representation_display", columns="topology_id", values="delta_mean"
        ).reindex(index=REP_ORDER, columns=TOPO_ORDER)
        im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_xticks(range(5))
        ax.set_xticklabels(TOPO_ORDER, fontsize=8)
        ax.set_yticks(range(len(REP_ORDER)))
        ax.set_yticklabels(REP_ORDER, fontsize=8)
        ax.set_title(f"Δ vs BASE: {ann}", fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.to_numpy(float)[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                 label="ΔMAE = MAE(annotation) − MAE(BASE)\n(negative = better than BASE)")
    fig.suptitle("Annotation effects relative to BASE (by topology)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "Figure_05_annotation_effects_vs_BASE")


def figure6_paired_vs_separate(m: pd.DataFrame) -> None:
    boot = pd.read_csv(RES / "TMAPP_REP_TOPO_ANNOT_BOOTSTRAP.csv")
    # Map bootstrap family labels: ablang2/currab with paired_minus_unpaired used allocation names
    # Rebuild PAIR-SEPARATE from master using true contexts.
    mats = {}
    sig = {}
    for family, (paired_raw, sep_raw) in PAIR_MAP.items():
        rows = []
        markers = []
        for ann in ANN_ORDER:
            for tid in TOPO_ORDER:
                p = m[(m.representation_raw == paired_raw) & (m.annotation_id == ann) & (m.topology_id == tid)]
                s = m[(m.representation_raw == sep_raw) & (m.annotation_id == ann) & (m.topology_id == tid)]
                assert len(p) == 1 and len(s) == 1
                d = float(p.iloc[0].mean_ps) - float(s.iloc[0].mean_ps)
                rows.append({"annotation_id": ann, "topology_id": tid, "delta": d})
                # bootstrap CI from existing table if available (allocation-based paired_minus_unpaired)
                # For AbLang2: allocation paired=SEPARATE, unpaired=PAIRED => allocation Δ = SEP-PAIR = -(PAIR-SEP)
                # For CurrAb: allocation paired=PAIRED, unpaired=SEPARATE => allocation Δ = PAIR-SEP
                fam_key = "ablang2" if family == "AbLang2" else "currab"
                leg = {v: k for k, v in LEGACY_TO_ID.items()}[tid]
                bsub = boot[
                    (boot.kind == "paired_minus_unpaired")
                    & (boot.family == fam_key)
                    & (boot.annotation == ann)
                    & (boot.topology == leg)
                    & (boot.scheme == "primary")
                ]
                mark = False
                if len(bsub):
                    lo, hi = float(bsub.iloc[0].ci95_lo), float(bsub.iloc[0].ci95_hi)
                    # convert to PAIR-SEPARATE direction
                    if family == "AbLang2":
                        lo, hi = -hi, -lo
                    mark = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
                markers.append(mark)
        mat = pd.DataFrame(rows).pivot(index="annotation_id", columns="topology_id", values="delta")
        mat = mat.reindex(index=ANN_ORDER, columns=TOPO_ORDER)
        mats[family] = mat
        sig[family] = np.array(markers).reshape(4, 5)

    lim = float(max(np.nanmax(np.abs(mats["AbLang2"].to_numpy())), np.nanmax(np.abs(mats["CurrAb"].to_numpy()))))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, family in zip(axes, ["AbLang2", "CurrAb"]):
        mat = mats[family].to_numpy(float)
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_xticks(range(5))
        ax.set_xticklabels(TOPO_ORDER, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(ANN_ORDER, fontsize=9)
        ax.set_title(f"{family}: PAIR − SEPARATE", fontsize=11)
        for i in range(4):
            for j in range(5):
                v = mat[i, j]
                star = " *" if sig[family][i, j] else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=8)
        mats[family].to_csv(DATA / f"pair_minus_separate_{family.lower()}.csv")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03,
                 label="MAE(paired H/L) − MAE(separate-chain)\n(negative = paired better)")
    fig.suptitle("Matched checkpoint: paired H/L context vs separate-chain context\n(* = primary paired-bootstrap 95% CI excludes 0; descriptive only)",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "Figure_06_paired_minus_separate")


def figure7_robustness(m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6))
    ax = axes[0]
    ax.scatter(m.primary_mae, m.shadow_mae, s=22, alpha=0.65, c="#5D6D7E", edgecolors="none")
    lims = [min(m.primary_mae.min(), m.shadow_mae.min()) - 0.05,
            max(m.primary_mae.max(), m.shadow_mae.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1)
    # label best overall + best per rep
    best = m.sort_values(["mean_ps", "abs_ps"]).iloc[0]
    ax.scatter([best.primary_mae], [best.shadow_mae], s=80, c="#C0392B", zorder=5, label="overall best")
    ax.annotate(f"{best.representation_display}\n{best.topology_id}/{best.annotation_id}",
                (best.primary_mae, best.shadow_mae), textcoords="offset points", xytext=(8, 8), fontsize=7,
                color="#C0392B")
    for disp in REP_ORDER:
        r = m[m.representation_display == disp].sort_values(["mean_ps", "abs_ps"]).iloc[0]
        if r.experiment_code == best.experiment_code:
            continue
        ax.annotate(disp.split("(")[0].strip()[:10], (r.primary_mae, r.shadow_mae),
                    textcoords="offset points", xytext=(4, -8), fontsize=6, alpha=0.85)
    # high disagreement
    hi = m.nlargest(5, "abs_ps")
    ax.scatter(hi.primary_mae, hi.shadow_mae, s=40, facecolors="none", edgecolors="#F39C12", linewidths=1.5,
               label="highest |P−S|")
    ax.set_xlabel("Primary MAE")
    ax.set_ylabel("Shadow MAE")
    ax.set_title("Primary vs Shadow (all 200 cells)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=7, loc="upper left")

    ax2 = axes[1]
    ax2.scatter(m.mean_ps, m.abs_ps, s=22, alpha=0.65, c="#5D6D7E", edgecolors="none")
    ax2.scatter([best.mean_ps], [best.abs_ps], s=80, c="#C0392B", zorder=5)
    ax2.set_xlabel("mean(P,S) MAE  (lower better)")
    ax2.set_ylabel("|Primary − Shadow|")
    ax2.set_title("Strength vs P/S disagreement")
    for _, r in hi.iterrows():
        ax2.annotate(r.experiment_code, (r.mean_ps, r.abs_ps), fontsize=6, xytext=(4, 4),
                     textcoords="offset points")
    fig.tight_layout()
    savefig(fig, "Figure_07_primary_shadow_robustness")


def write_tables(m: pd.DataFrame) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    # Table 1
    t1 = []
    for raw, meta in TERM["representation"].items():
        ctx = m[m.representation_raw == raw].iloc[0].representation_context
        dim = m[m.representation_raw == raw].iloc[0].raw_dim
        t1.append({
            "Representation": meta["representation_display"],
            "Domain": meta["domain"],
            "Context": ctx,
            "Raw dimension": dim,
            "representation_raw": raw,
            "Notes": meta["notes"].strip().replace("\n", " "),
        })
    t1 = pd.DataFrame(t1)
    t1["Representation"] = pd.Categorical(t1["Representation"], REP_ORDER, ordered=True)
    t1 = t1.sort_values("Representation")
    t1.to_csv(TAB / "Table_01_representation_definitions.csv", index=False)

    # Table 2
    t2 = []
    for leg, meta in TERM["topology"].items():
        t2.append({
            "topology_id": meta["topology_id"],
            "topology_display": meta["topology_display"],
            "legacy_id": leg,
            "residue_level_HL_communication": meta["residue_hl_communication"],
            "REG_cross_chain_communication": meta["reg_cross_chain_communication"],
            "point_of_HL_interaction": meta["interaction_point"],
            "definition": meta["definition"].strip().replace("\n", " "),
        })
    t2 = pd.DataFrame(t2)
    t2["topology_id"] = pd.Categorical(t2["topology_id"], TOPO_ORDER, ordered=True)
    t2 = t2.sort_values("topology_id")
    t2.to_csv(TAB / "Table_02_topology_definitions.csv", index=False)

    # Table 3
    t3 = []
    for aid, meta in TERM["annotation"].items():
        t3.append({
            "annotation_id": aid,
            "annotation_display": meta["annotation_display"],
            "sequence_position": meta["sequence_position"],
            "chain_id": meta["chain_id"],
            "IMGT_position": meta["imgt_position"],
            "CDR_FR_region": meta["cdr_fr_region"],
        })
    pd.DataFrame(t3).to_csv(TAB / "Table_03_annotation_definitions.csv", index=False)

    # Table 4 best per rep
    rows = []
    for disp in REP_ORDER:
        r = m[m.representation_display == disp].sort_values(["mean_ps", "worst_ps", "abs_ps"]).iloc[0]
        rows.append({
            "representation": disp,
            "topology": r.topology_id,
            "annotation": r.annotation_id,
            "experiment": r.experiment_code,
            "Primary": r.primary_mae,
            "Shadow": r.shadow_mae,
            "mean_PS": r.mean_ps,
            "worst_PS": r.worst_ps,
            "abs_PS": r.abs_ps,
        })
    t4 = pd.DataFrame(rows)
    t4.to_csv(TAB / "Table_04_best_cell_per_representation.csv", index=False)

    # Table 5 top configs
    top = m.sort_values(["mean_ps", "worst_ps"]).head(15)[
        ["experiment_code", "representation_display", "topology_id", "annotation_id",
         "primary_mae", "shadow_mae", "mean_ps", "worst_ps", "abs_ps"]
    ]
    top.to_csv(TAB / "Table_05_top15_by_mean_ps.csv", index=False)
    topw = m.sort_values(["worst_ps", "mean_ps"]).head(15)[
        ["experiment_code", "representation_display", "topology_id", "annotation_id",
         "primary_mae", "shadow_mae", "mean_ps", "worst_ps", "abs_ps"]
    ]
    topw.to_csv(TAB / "Table_05b_top15_by_worst_ps.csv", index=False)

    # markdown versions
    def to_md(df, path, title):
        lines = [f"# {title}", "", df.to_markdown(index=False), ""]
        path.write_text("\n".join(lines), encoding="utf-8")

    to_md(t1.drop(columns=["Notes"]).assign(Notes=t1["Notes"].str.slice(0, 80)), TAB / "Table_01_representation_definitions.md", "Table 1 — Representation definitions")
    to_md(t2.drop(columns=["definition"]), TAB / "Table_02_topology_definitions.md", "Table 2 — Topology definitions")
    to_md(pd.read_csv(TAB / "Table_03_annotation_definitions.csv"), TAB / "Table_03_annotation_definitions.md", "Table 3 — Annotation definitions")
    to_md(t4.round(4), TAB / "Table_04_best_cell_per_representation.md", "Table 4 — Best cell per representation")
    to_md(top.round(4), TAB / "Table_05_top15_by_mean_ps.md", "Table 5 — Top 15 by mean(P,S)")
    to_md(topw.round(4), TAB / "Table_05b_top15_by_worst_ps.md", "Table 5b — Top 15 by worst(P,S)")


def write_notes(m: pd.DataFrame) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    best = m.sort_values("mean_ps").iloc[0]
    scratch_best = m[m.representation_display == "Scratch"].sort_values("mean_ps").iloc[0]
    a2s = m[m.representation_display == "AbLang2 (separate-chain)"].sort_values("mean_ps").iloc[0]
    a2p = m[m.representation_display == "AbLang2 (paired H/L)"].sort_values("mean_ps").iloc[0]
    esmc = m[m.representation_display == "ESM-C 600M"].sort_values("mean_ps").iloc[0]
    esm1 = m[m.representation_display == "ESM-1b"].sort_values("mean_ps").iloc[0]
    esm2 = m[m.representation_display == "ESM-2"].sort_values("mean_ps").iloc[0]

    notes = {
        "Figure_01_interpretation.md": f"""# Figure 1 — Factorial / topology schematic

## What the figure directly shows

The panel defines the controlled pipeline after residue representations are fixed, and diagrams the five downstream H/L topologies using presentation IDs SEP, JOINT, REG-SEP, XREG, and FUSE. REG is defined as a learned chain-level token that gathers information from residues processed by the downstream Transformer.

## Strongest patterns

Not a results figure; it encodes design distinctions that later figures quantify.

## Robustness

N/A (schematic).

## What we should NOT claim

That any topology diagram is a physical model of antibody H/L interfaces.

## Possible technical interpretation

Interpretation: these diagrams are inductive-bias choices in a shared downstream platform, not claims about PLM internal mechanisms.
""",
        "Figure_02_interpretation.md": f"""# Figure 2 — Best configuration per representation

## What the figure directly shows

For each representation, the single lowest mean(P,S) cell (and separately worst(P,S)) among the 20 annotation×topology combinations. Overall best in this matrix: {best.experiment_code} = {best.representation_display} / {best.topology_id} / {best.annotation_id} (mean={best.mean_ps:.3f}). Scratch best: {scratch_best.topology_id}/{scratch_best.annotation_id}. AbLang2 separate-chain best: {a2s.topology_id}/{a2s.annotation_id}; AbLang2 paired H/L best: {a2p.topology_id}/{a2p.annotation_id}.

## Strongest patterns

AbLang2 contexts occupy the leading end of the ranking. Best topology/annotation is not shared across representations (e.g., Scratch REG-SEP+FULL; ESM-C JOINT+BASE; ESM-2 FUSE+FULL; ESM-1b XREG+BASE).

## Robustness

Ordering by mean(P,S) vs worst(P,S) can differ for mid-ranked representations; small MAE gaps should not be treated as decisive without bootstrap.

## What we should NOT claim

That the top-ranked representation “contains structure” or is universally best outside this OOF matrix.

## Possible technical interpretation

Interpretation: optimal downstream inductive bias (topology/annotation) appears representation-dependent under a frozen training protocol.
""",
        "Figure_03_interpretation.md": f"""# Figure 3 — Absolute performance landscape

## What the figure directly shows

4×5 heatmaps of mean(P,S) for every annotation×topology cell, one representation per panel group, on a shared absolute MAE color scale.

## Strongest patterns

Within AbLang2 panels, XREG+REGION cells are among the darkest (lowest MAE). ESM-C shows a clear JOINT+BASE bright spot relative to its other cells. Landscapes are not uniform copies of each other.

## Robustness

Absolute MAE heatmaps do not themselves provide uncertainty; compare with Primary/Shadow and bootstrap tables for contrasts.

## What we should NOT claim

That color intensity maps to mechanistic “understanding” of antibodies.

## Possible technical interpretation

Interpretation: the same downstream design space yields different absolute landscapes once residue inputs change, consistent with representation-dependent requirements.
""",
        "Figure_04_interpretation.md": f"""# Figure 4 — Topology effects vs SEP

## What the figure directly shows

ΔMAE = MAE(topology) − MAE(SEP) for JOINT, REG-SEP, XREG, FUSE within each representation×annotation. Negative values indicate improvement over SEP.

## Strongest patterns

Scratch shows large negative deltas for REG-SEP/JOINT/XREG under several annotations (especially FULL). ESM-2 shows broad negative deltas including FUSE under FULL. ESM-C often shows positive deltas (SEP preferred) under FULL. Antibody PLMs are mixed: AbLang2 contexts favor XREG/JOINT in several annotations, not uniformly.

## Robustness

Prefer patterns that repeat across Primary and Shadow in the bootstrap/topology-gain tables. Single-panel outliers should stay exploratory.

## What we should NOT claim

That negative ΔMAE proves the topology “implements” biological H/L pairing, or that REG tokens correspond to physical interface contacts.

## Possible technical interpretation

Interpretation: pretrained residue information can change how much explicit downstream H/L communication helps relative to Scratch, without implying equivalence of mechanisms.
""",
        "Figure_05_interpretation.md": f"""# Figure 5 — Annotation effects vs BASE

## What the figure directly shows

ΔMAE = MAE(annotation) − MAE(BASE) for IMGT, REGION, and FULL, stratified by representation and topology. Sequence-position and chain-ID embeddings are present in all four annotation conditions.

## Strongest patterns

Annotation effects are not uniform: some representation×topology cells improve with REGION/FULL, others worsen. Topology dependence is visible (columns differ within a row).

## Robustness

Check annotation-gain bootstrap CIs before treating small signed deltas as stable. Primary/Shadow agreement matters for claims.

## What we should NOT claim

That a near-zero IMGT delta means “IMGT is already encoded in the PLM,” or that REGION gains mean the model learned CDR biology.

## Possible technical interpretation

Interpretation: explicit IMGT/CDR-FR features are an optional inductive bias whose value depends on both the residue representation and the H/L topology.
""",
        "Figure_06_interpretation.md": f"""# Figure 6 — Paired vs separate PLM context

## What the figure directly shows

For matched checkpoints, heatmaps of MAE(paired H/L context) − MAE(separate-chain context) over annotation×topology. Negative = paired context better. AbLang2 and CurrAb share the same color scale. Stars mark primary paired-bootstrap CIs excluding zero (descriptive).

## Strongest patterns

CurrAb shows a more consistently negative (paired-better) field on average. AbLang2 is more mixed by annotation/topology; best cells for both AbLang2 contexts remain near XREG+REGION. Average paired advantage and best-cell near-ties can coexist (especially CurrAb).

## Robustness

Use bootstrap markers cautiously; many cells will have CIs including zero. Require Primary/Shadow agreement for stronger wording.

## What we should NOT claim

That paired inference “learned the interface,” or that downstream XREG substitutes for PLM-internal pairing in a mechanistic sense.

## Possible technical interpretation

Interpretation: holding weights fixed, changing only inference-time H/L context alters the downstream error surface; AbLang2 and CurrAb do not show identical PAIR−SEPARATE maps.
""",
        "Figure_07_interpretation.md": f"""# Figure 7 — Primary / Shadow robustness

## What the figure directly shows

Scatter of all 200 cells in Primary vs Shadow MAE space, plus mean(P,S) vs |P−S|. Overall best and high-disagreement cells are highlighted.

## Strongest patterns

Most cells track near y=x. A minority show larger |P−S|, indicating that low mean alone is insufficient for robustness ranking.

## Robustness

This figure is the direct P/S diagnostic; combine with worst(P,S) rankings when selecting “stable” cells.

## What we should NOT claim

That agreement of P and S validates external generalization, or that disagreement implies a bug.

## Possible technical interpretation

Interpretation: fold-scheme sensitivity remains non-negligible for some representation×topology×annotation combinations even under a frozen protocol.
""",
    }
    for name, text in notes.items():
        (NOTES / name).write_text(text, encoding="utf-8")

    qnotes = f"""# First-pass scientific questions (figure-oriented)

REG definition used throughout: a learned chain-level token that gathers information from the residues processed by the downstream Transformer.

## Q1. How strongly does optimal H/L topology depend on residue representation?

Supported by Figures 2–4: best topologies differ across representations (Scratch REG-SEP; AbLang2 XREG; ESM-C JOINT; ESM-2 FUSE; ESM-1b XREG). Topology Δ vs SEP maps are not interchangeable.

## Q2. Does explicit antibody annotation help uniformly?

No. Figure 5 shows signed Δ vs BASE that change with both representation and topology. REGION/FULL are not universal improvements.

## Q3. Does pretrained residue information reduce the need for downstream H/L interaction vs Scratch?

Partially consistent with Figure 4: Scratch often shows large negative Δ for interaction topologies; some PLMs (e.g., ESM-C under FULL) prefer SEP. This is comparative predictive evidence, not a mechanistic reduction proof.

## Q4. Does pretrained information reduce need for explicit IMGT/CDR-FR?

Not uniformly (Figure 5). Some PLM cells improve under BASE relative to FULL; others benefit from REGION. Avoid claiming that annotations are “already inside” the PLM.

## Q5. Within the same checkpoint, what changes with paired H/L inference context?

Figure 6: PAIR−SEPARATE surfaces are non-flat for AbLang2 and CurrAb. Absolute best cells can be similar (AbLang2 XREG+REGION in both contexts) while average Δ still favors one context.

## Q6. Are AbLang2 and CurrAb paired-vs-separate patterns similar?

Figure 6 suggests **not identical**: CurrAb’s average paired advantage is clearer; AbLang2 is more annotation/topology-dependent. Shared scale enables direct visual comparison.

## Q7. Simple monotonic ordering by newer/general/antibody PLMs?

Figure 2/3: **no simple monotonic story**. Antibody-oriented AbLang2 leads this matrix, but ESM-C can beat several antibody PLMs depending on cell choice; age/domain labels are descriptive only.

## Special cases (examples, not proof points)

- Scratch: REG-SEP + FULL best ({scratch_best.experiment_code}, mean={scratch_best.mean_ps:.3f})
- AbLang2: XREG + REGION leads both contexts ({a2s.experiment_code}, {a2p.experiment_code})
- ESM-C: JOINT + BASE ({esmc.experiment_code}, mean={esmc.mean_ps:.3f}) far better than a C+FULL-only smoke reading
- ESM-1b: XREG + BASE ({esm1.experiment_code})
- ESM-2: FUSE + FULL ({esm2.experiment_code})
- CurrAb: average paired-context advantage with near-tie of best separate/paired cells
"""
    (NOTES / "first_pass_scientific_questions.md").write_text(qnotes, encoding="utf-8")


def write_outline() -> None:
    text = """# TECHNICAL_REPORT_OUTLINE.md

Proposed structure for a later technical report. **Do not treat this as polished manuscript prose.**

1. Motivation
2. Dataset and prediction task
3. Experimental design (Representation × Topology × Annotation factorial)
4. Residue representations (display names; `representation_context` as ground truth)
5. H/L topology definitions (SEP / JOINT / REG-SEP / XREG / FUSE; REG definition)
6. Annotation definitions (BASE / IMGT / REGION / FULL; position+chain always on)
7. Evaluation protocol (DL_FOLDLOCAL_COSINE_V3; Primary/Shadow; seed 101)
8. Overall performance landscape (Figures 2–3; Tables 4–5)
9. Representation-dependent topology effects (Figure 4)
10. Representation-dependent annotation effects (Figure 5)
11. Paired vs separate PLM context (Figure 6; matched checkpoints)
12. Scratch as control
13. Robustness and uncertainty (Figure 7; bootstrap tables)
14. External diagnostic (post-internal-freeze only)
15. Limitations
16. Conclusions (predictive inductive-bias evidence; non-claims)

Figure/table inventory lives in `README.md`.
"""
    (PKG / "TECHNICAL_REPORT_OUTLINE.md").write_text(text, encoding="utf-8")


def write_readme(m: pd.DataFrame) -> None:
    figs = sorted(p.name for p in FIG.glob("Figure_*.png"))
    tabs = sorted(p.name for p in TAB.glob("Table_*.csv"))
    text = f"""# TmApp factorial — technical report analysis package

Presentation-layer analysis only. No new training. Historical experiment IDs and result files are not rewritten.

## Source of truth

- Machine results: `results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv`
- Human master: `data/tmapp_factorial_human_master.csv` ({len(m)} rows)
- Terminology: `terminology.yaml`

## REG definition

A REG is **a learned chain-level token that gathers information from the residues processed by the downstream Transformer**.

## Topology display IDs

| legacy | ID | display |
| --- | --- | --- |
| A | SEP | Separate H/L |
| B1 | JOINT | Full Joint |
| B2 | REG-SEP | Joint Residues, Separate REGs |
| C | XREG | Cross-REG Read |
| D | FUSE | REG Fusion |

## AbLang2 / CurrAb naming

Uses `representation_context`, not historical allocation labels:

- AbLang2 (separate-chain) ← `ablang2_paired` / SEPARATE_CHAIN
- AbLang2 (paired H/L) ← `ablang2_unpaired` / PAIRED_NATIVE
- CurrAb (paired H/L) ← `currab_paired` / PAIRED_NATIVE
- CurrAb (separate-chain) ← `currab_unpaired` / SEPARATE_CHAIN

## Figures

{chr(10).join(f'- `{f}` (+ PDF)' for f in figs)}

## Tables

{chr(10).join(f'- `{t}`' for t in tabs)}

## Notes

- `notes/Figure_XX_interpretation.md`
- `notes/first_pass_scientific_questions.md`
- `TECHNICAL_REPORT_OUTLINE.md`

## Rebuild

```bash
PYTHONPATH=scripts:models ../.venv_b1/bin/python technical_report/tmapp_factorial/scripts/build_report_package.py
```
"""
    (PKG / "README.md").write_text(text, encoding="utf-8")


def consistency_scan() -> list[str]:
    """Scan NEW reporting prose for discouraged primary-axis legacy labels / summary."""
    issues = []
    allow_files = {
        "terminology.yaml",
        "Table_02_topology_definitions.csv",
        "Table_02_topology_definitions.md",
        "tmapp_factorial_human_master.csv",
        "README.md",  # mapping table
        "TECHNICAL_REPORT_OUTLINE.md",
    }
    # patterns that are bad in narrative axes when used as primary labels
    bad_axis = re.compile(r"\bB1\b|\bB2\b")
    summary_re = re.compile(r"\bsummary\b", re.I)
    for path in PKG.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".md", ".yaml", ".csv", ".txt"}:
            continue
        if path.name in allow_files or "legacy" in path.name:
            # still forbid summary in most places except quoting
            if path.suffix == ".md" and path.name not in {"terminology.yaml"}:
                txt = path.read_text(encoding="utf-8", errors="ignore")
                # allow 'summary' only if we somehow need it - check
                for i, line in enumerate(txt.splitlines(), 1):
                    if summary_re.search(line) and "summary" in line.lower():
                        # allow in 'do not use summary' instructional lines
                        if "summary" in line.lower() and ("avoid" in line.lower() or "not" in line.lower() or "prefer" in line.lower()):
                            continue
                        if path.name == "terminology.yaml":
                            continue
                        # Figure notes should not say summary token
                        if "summary token" in line.lower() or "summary." in line.lower():
                            issues.append(f"{path.relative_to(PKG)}:{i}: summary wording")
            continue
        if path.suffix in {".md"}:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(txt.splitlines(), 1):
                if summary_re.search(line):
                    if any(w in line.lower() for w in ("avoid", "do not", "not call", "prefer", "instead")):
                        continue
                    issues.append(f"{path.relative_to(PKG)}:{i}: contains 'summary'")
                # legacy as primary axis in headings
                if line.strip().startswith("#") and bad_axis.search(line) and "legacy" not in line.lower():
                    issues.append(f"{path.relative_to(PKG)}:{i}: legacy ID in heading")
    return issues


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    print("building master...", flush=True)
    m = build_master()
    print("figure 1...", flush=True)
    figure1_schematic()
    print("figure 2...", flush=True)
    figure2_best_per_rep(m)
    print("figure 3...", flush=True)
    figure3_absolute_landscape(m)
    print("figure 4...", flush=True)
    figure4_topology_effects(m)
    print("figure 5...", flush=True)
    figure5_annotation_effects(m)
    print("figure 6...", flush=True)
    figure6_paired_vs_separate(m)
    print("figure 7...", flush=True)
    figure7_robustness(m)
    print("tables...", flush=True)
    write_tables(m)
    print("notes...", flush=True)
    write_notes(m)
    write_outline()
    write_readme(m)
    issues = consistency_scan()
    (NOTES / "terminology_consistency_scan.txt").write_text(
        "\n".join(issues) if issues else "PASS: no unexpected summary/legacy-axis issues in new narrative files.\n",
        encoding="utf-8",
    )
    print("consistency issues:", len(issues), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
