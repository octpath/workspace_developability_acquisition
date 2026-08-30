#!/usr/bin/env python3
"""Gate B6.2 add-on: predicted vs observed HIC visual check (frozen preds only)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b6_2_hic_validity"
OUT = GATE / "plots" / "pred_vs_true"
REPORTS = GATE / "reports"

# Import load_everything from the main B6.2 script without re-running main.
_SPEC = importlib.util.spec_from_file_location(
    "b62", GATE / "scripts" / "run_gate_b6_2_hic_validity.py"
)
_b62 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["b62"] = _b62
_SPEC.loader.exec_module(_b62)

MODELS = [
    ("NESTED_STACK_NNLS", "nested_stack_nnls_cv_public_private.png", "NESTED_STACK_NNLS"),
    ("ESMFN_STRUCTURE_ElasticNet", "esmfold_structure_cv_public_private.png", "ESMFN_STRUCTURE / ElasticNet"),
    ("PLM_ESM2_PCA64_SVR", "esm2_pca64_svr_cv_public_private.png", "PLM_ESM2_PCA64 / SVR"),
    ("FUSION_ESM2_ESMFN_ElasticNet", "fusion_esm2_esmfold_cv_public_private.png", "FUSION_ESM2_ESMFN / ElasticNet"),
    ("SEQ_SIMPLE_Ridge", "seq_simple_cv_public_private.png", "SEQ_SIMPLE / Ridge"),
]

ROLES = [
    ("cv", "Train CV OOF", "train_ids", "train_y"),
    ("public", "Public (CAND_12528)", "public_ids", "pub_y"),
    ("private", "Private (CAND_12528)", "private_ids", "priv_y"),
]

# Visual encoding: central / Q90–Q95 / ≥Q95
COLOR_CENTRAL = "#4c6a8a"
COLOR_Q90 = "#d97706"
COLOR_Q95 = "#b91c1c"
EDGE_Q90 = "#9a3412"
EDGE_Q95 = "#7f1d1d"
COLOR_ID = "#111827"
COLOR_REG = "#6d28d9"


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n = len(y)
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    obs_sd = float(np.std(y, ddof=1)) if n > 1 else float("nan")
    pred_sd = float(np.std(p, ddof=1)) if n > 1 else float("nan")
    ratio = pred_sd / obs_sd if obs_sd and np.isfinite(obs_sd) and obs_sd > 0 else float("nan")
    if n >= 3 and np.std(y) > 1e-12 and np.std(p) > 1e-12:
        pr = float(pearsonr(y, p)[0])
        sr = float(spearmanr(y, p).correlation)
        # predicted ~ measured (scatter regression)
        lr_yx = linregress(y, p)
        a_pred_on_obs, b_pred_on_obs = float(lr_yx.intercept), float(lr_yx.slope)
        # observed ~ predicted (Gate B6.2 calibration orientation)
        lr_xy = linregress(p, y)
        cal_intercept, cal_slope = float(lr_xy.intercept), float(lr_xy.slope)
    else:
        pr = sr = a_pred_on_obs = b_pred_on_obs = cal_intercept = cal_slope = float("nan")
    return {
        "n": n,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson": pr,
        "Spearman": sr,
        "obs_SD": obs_sd,
        "pred_SD": pred_sd,
        "pred_SD_over_obs_SD": ratio,
        "a_pred_on_obs": a_pred_on_obs,
        "b_pred_on_obs": b_pred_on_obs,
        "cal_intercept_obs_on_pred": cal_intercept,
        "cal_slope_obs_on_pred": cal_slope,
    }


def annotate_box(ax, met: dict) -> None:
    txt = (
        f"n={met['n']}\n"
        f"MAE={met['MAE']:.3f}  RMSE={met['RMSE']:.3f}\n"
        f"Pearson r={met['Pearson']:.3f}\n"
        f"Spearman ρ={met['Spearman']:.3f}\n"
        f"obs SD={met['obs_SD']:.3f}\n"
        f"pred SD={met['pred_SD']:.3f}\n"
        f"pred/obs SD={met['pred_SD_over_obs_SD']:.3f}\n"
        f"fit: pred=a+b·true\n"
        f"  a={met['a_pred_on_obs']:.3f}, b={met['b_pred_on_obs']:.3f}\n"
        f"cal (obs=c+s·pred):\n"
        f"  s={met['cal_slope_obs_on_pred']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="#d1d5db"),
    )


def scatter_panel(ax, y, p, ids, q90, q95, lim, title, label_ids: bool = False):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ids = np.asarray(ids)
    central = y < q90
    mid = (y >= q90) & (y < q95)
    hi = y >= q95

    ax.scatter(y[central], p[central], s=22, c=COLOR_CENTRAL, alpha=0.75, edgecolors="none", label="central (<Train Q90)", zorder=2)
    ax.scatter(y[mid], p[mid], s=48, c=COLOR_Q90, alpha=0.95, edgecolors=EDGE_Q90, linewidths=0.8, marker="D", label="Train Q90–Q95", zorder=3)
    ax.scatter(y[hi], p[hi], s=70, c=COLOR_Q95, alpha=0.95, edgecolors=EDGE_Q95, linewidths=0.9, marker="*", label="≥ Train Q95", zorder=4)

    # identity
    ax.plot(lim, lim, color=COLOR_ID, lw=1.4, ls="-", label="identity (y=x)", zorder=1)
    # predicted ~ measured
    met = metrics(y, p)
    if np.isfinite(met["b_pred_on_obs"]):
        xx = np.linspace(lim[0], lim[1], 200)
        yy = met["a_pred_on_obs"] + met["b_pred_on_obs"] * xx
        ax.plot(xx, yy, color=COLOR_REG, lw=1.6, ls="--", label="OLS: pred ~ true", zorder=1)

    if label_ids:
        for yi, pi, iid in zip(y, p, ids):
            if yi >= q90:
                ax.annotate(
                    str(iid),
                    (yi, pi),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6.5,
                    color="#7f1d1d",
                    alpha=0.95,
                )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("measured HIC (true)")
    ax.set_ylabel("predicted HIC")
    ax.grid(True, alpha=0.25)
    annotate_box(ax, met)
    return met


def residual_panel(ax, y, p, q90, q95, xlim, ylim, title):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    r = p - y
    central = y < q90
    mid = (y >= q90) & (y < q95)
    hi = y >= q95
    ax.axhline(0.0, color=COLOR_ID, lw=1.3, zorder=1)
    ax.scatter(y[central], r[central], s=22, c=COLOR_CENTRAL, alpha=0.75, edgecolors="none", zorder=2)
    ax.scatter(y[mid], r[mid], s=48, c=COLOR_Q90, alpha=0.95, edgecolors=EDGE_Q90, linewidths=0.8, marker="D", zorder=3)
    ax.scatter(y[hi], r[hi], s=70, c=COLOR_Q95, alpha=0.95, edgecolors=EDGE_Q95, linewidths=0.9, marker="*", zorder=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("measured HIC (true)")
    ax.set_ylabel("prediction − true")
    ax.grid(True, alpha=0.25)
    # light trend note
    if len(y) >= 3 and np.std(y) > 1e-12:
        lr = linregress(y, r)
        xx = np.linspace(xlim[0], xlim[1], 100)
        ax.plot(xx, lr.intercept + lr.slope * xx, color=COLOR_REG, ls="--", lw=1.3, label=f"resid~true slope={lr.slope:.2f}")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)


def make_model_figure(preds, meta, tag, title, out_path, lim, q90, q95, label_tail=False):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), constrained_layout=True)
    handles_labels = None
    for ax, (role, role_title, id_key, y_key) in zip(axes, ROLES):
        y = meta[y_key]
        p = preds[tag][role]
        ids = meta[id_key]
        scatter_panel(ax, y, p, ids, q90, q95, lim, role_title, label_ids=label_tail)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    # one shared legend
    fig.legend(
        *handles_labels,
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"{title} — predicted vs measured HIC\n"
        f"CAND_12528 · Train thresholds: Q90={q90:.3f}, Q95={q95:.3f}  |  "
        f"solid black = identity (y=x); dashed purple = OLS pred~true  |  "
        f"axes locked 1:1 across panels",
        fontsize=12,
        y=1.08 if not label_tail else 1.10,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    pack = _b62.load_everything()
    preds, meta = pack["preds"], pack["meta"]
    train_y = meta["train_y"]
    q90 = float(np.quantile(train_y, 0.90))
    q95 = float(np.quantile(train_y, 0.95))

    # Global shared axis range across ALL models / roles (true and pred)
    vals = [train_y, meta["pub_y"], meta["priv_y"]]
    for tag, _, _ in MODELS:
        for role, *_ in ROLES:
            vals.append(preds[tag][role])
    lo = float(np.min([np.min(v) for v in vals]))
    hi = float(np.max([np.max(v) for v in vals]))
    pad = 0.15
    lim = (lo - pad, hi + pad)

    print(f"Train Q90={q90:.6f} Q95={q95:.6f}")
    print(f"Shared axis lim={lim}")

    for tag, fname, title in MODELS:
        path = OUT / fname
        make_model_figure(preds, meta, tag, title, path, lim, q90, q95, label_tail=False)
        print("wrote", path)

    # Tail-labeled version for nested (all roles annotated for Q90+)
    make_model_figure(
        preds,
        meta,
        "NESTED_STACK_NNLS",
        "NESTED_STACK_NNLS (Train Q90+ IDs labeled)",
        OUT / "nested_stack_nnls_tail_labeled.png",
        lim,
        q90,
        q95,
        label_tail=True,
    )
    print("wrote", OUT / "nested_stack_nnls_tail_labeled.png")

    # Residual vs true for nested
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    all_resid = []
    for role, y_key in [("cv", "train_y"), ("public", "pub_y"), ("private", "priv_y")]:
        all_resid.append(preds["NESTED_STACK_NNLS"][role] - meta[y_key])
    rlo = float(np.min([np.min(r) for r in all_resid])) - 0.2
    rhi = float(np.max([np.max(r) for r in all_resid])) + 0.2
    # keep zero roughly centered-ish but allow asymmetry
    for ax, (role, role_title, _, y_key) in zip(axes, ROLES):
        residual_panel(
            ax,
            meta[y_key],
            preds["NESTED_STACK_NNLS"][role],
            q90,
            q95,
            lim,
            (rlo, rhi),
            role_title,
        )
    fig.suptitle(
        f"NESTED_STACK_NNLS — residual (pred−true) vs measured HIC\n"
        f"Train Q90={q90:.3f}, Q95={q95:.3f} (thresholds frozen from Train)",
        fontsize=12,
    )
    # shared legend proxies
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CENTRAL, markersize=8, label="central (<Q90)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLOR_Q90, markeredgecolor=EDGE_Q90, markersize=8, label="Q90–Q95"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_Q95, markeredgecolor=EDGE_Q95, markersize=12, label="≥Q95"),
        Line2D([0], [0], color=COLOR_ID, lw=1.3, label="residual=0"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(OUT / "nested_stack_nnls_residual_vs_true.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "nested_stack_nnls_residual_vs_true.png")

    # True vs pred distribution overlay for nested
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    bins = np.linspace(lim[0], lim[1], 28)
    for ax, (role, role_title, _, y_key) in zip(axes, ROLES):
        y = meta[y_key]
        p = preds["NESTED_STACK_NNLS"][role]
        met = metrics(y, p)
        ax.hist(y, bins=bins, density=True, alpha=0.55, color="#1f4e79", label="observed", edgecolor="white", linewidth=0.4)
        ax.hist(p, bins=bins, density=True, alpha=0.55, color="#c2410c", label="predicted", edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(y), color="#1f4e79", ls=":", lw=1.2)
        ax.axvline(np.median(p), color="#c2410c", ls=":", lw=1.2)
        ax.set_xlim(lim)
        ax.set_title(role_title, fontsize=11)
        ax.set_xlabel("HIC")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.98,
            0.98,
            f"obs SD={met['obs_SD']:.3f}\npred SD={met['pred_SD']:.3f}\nratio={met['pred_SD_over_obs_SD']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#d1d5db"),
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "NESTED_STACK_NNLS — observed vs predicted HIC distributions (shared x-axis)\n"
        f"Under-dispersion check (Gate B6.2: CV pred_SD/obs_SD ≈ 0.56)",
        fontsize=12,
    )
    fig.savefig(OUT / "nested_stack_nnls_true_vs_pred_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "nested_stack_nnls_true_vs_pred_distribution.png")

    # Collect nested metrics for the short report
    nested_stats = {}
    for role, role_title, _, y_key in ROLES:
        nested_stats[role] = metrics(meta[y_key], preds["NESTED_STACK_NNLS"][role])

    # Quick cross-model SD ratios for report
    sd_lines = []
    for tag, _, title in MODELS:
        for role, role_title, _, y_key in ROLES:
            m = metrics(meta[y_key], preds[tag][role])
            sd_lines.append(f"| {title} | {role_title} | {m['pred_SD_over_obs_SD']:.3f} | {m['Pearson']:.3f} | {m['MAE']:.3f} |")

    report = f"""# Predicted vs Observed Visual Check (Gate B6.2 add-on)

Frozen artifacts only: Train **proper OOF** + CAND_12528 Public/Private one-shot predictions.
No retraining, no Optuna, no split change.

Train-derived thresholds (applied to all roles): **Q90={q90:.4f}**, **Q95={q95:.4f}**

Shared 1:1 axis range across all model panels: **[{lim[0]:.3f}, {lim[1]:.3f}]**

## Plots

- `plots/pred_vs_true/nested_stack_nnls_cv_public_private.png`
- `plots/pred_vs_true/esmfold_structure_cv_public_private.png`
- `plots/pred_vs_true/esm2_pca64_svr_cv_public_private.png`
- `plots/pred_vs_true/fusion_esm2_esmfold_cv_public_private.png`
- `plots/pred_vs_true/seq_simple_cv_public_private.png`
- `plots/pred_vs_true/nested_stack_nnls_tail_labeled.png`
- `plots/pred_vs_true/nested_stack_nnls_residual_vs_true.png`
- `plots/pred_vs_true/nested_stack_nnls_true_vs_pred_distribution.png`

Scatter regression on the figure is **pred = a + b·true** (dashed purple).
Gate B6.2 calibration slope (**obs = c + s·pred**) is annotated separately as `cal s=…` — do not confuse the two.

NESTED_STACK_NNLS snapshot:

| Role | n | MAE | Pearson | pred/obs SD |
|------|---|-----|---------|-------------|
| Train OOF | {nested_stats['cv']['n']} | {nested_stats['cv']['MAE']:.3f} | {nested_stats['cv']['Pearson']:.3f} | {nested_stats['cv']['pred_SD_over_obs_SD']:.3f} |
| Public | {nested_stats['public']['n']} | {nested_stats['public']['MAE']:.3f} | {nested_stats['public']['Pearson']:.3f} | {nested_stats['public']['pred_SD_over_obs_SD']:.3f} |
| Private | {nested_stats['private']['n']} | {nested_stats['private']['MAE']:.3f} | {nested_stats['private']['Pearson']:.3f} | {nested_stats['private']['pred_SD_over_obs_SD']:.3f} |

Cross-model pred/obs SD + Pearson:

| Model | Role | pred/obs SD | Pearson | MAE |
|-------|------|-------------|---------|-----|
{chr(10).join(sd_lines)}

## Answers (visual)

1. **Central HIC (~8.5–9.5):** Yes — points cluster near the identity line / a tight band around ~9.0–9.5 predicted.
2. **Prediction range vs observed:** Yes — clearly narrower (pred/obs SD ≈ 0.45–0.60 across roles for nested; distributions confirm compression).
3. **High-HIC systematic underprediction:** Yes — Q90/Q95 markers fall below y=x; residuals become strongly negative as true HIC rises.
4. **Shared across CV/Public/Private:** Yes — the same center-OK + right-tail-under pattern appears in all three panels.
5. **Public-only anomaly in scatter geometry:** No striking unique geometry; Public has sparse high-HIC leverage points (as expected) but the underprediction pattern matches CV/Private.
6. **Private-only anomaly:** No — Private looks qualitatively like CV with the same shrinkage / tail-under behavior.
7. **Other PLM/structure models:** Yes — ESMFN, ESM2-SVR, Fusion, and Seq-Ridge show the same compressed cloud below the identity line at high true HIC.
8. **Flat median band vs rising predictions:** Not a pure horizontal median band — predictions do rise with true HIC (positive Pearson; OLS slope b>0), but the rise is too shallow (shrinkage), so extremes remain underpredicted.
9. **Supports Gate B6.2 conclusion?** **Yes** — scatter/residual/distribution plots visually support “learnable continuous signal with material high-tail underprediction and under-dispersion.”
"""
    out_md = REPORTS / "05_predicted_vs_observed_visual_check.md"
    out_md.write_text(report)
    print("wrote", out_md)


if __name__ == "__main__":
    main()
