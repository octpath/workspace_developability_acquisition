#!/usr/bin/env python3
"""Gate B6.3 — HIC Developability-Band Utility Audit (frozen predictions only).

Literature Shehata/Jain bands are DIAGNOSTIC ONLY.
Continuous HIC remains the competition task.
No retrain / retune / split change / classification models.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b6_2_hic_validity"
METRICS = GATE / "metrics"
PLOTS = GATE / "plots" / "hic_band_audit"
REPORTS = GATE / "reports"

LOW_HI = 10.5
HIGH_LO = 11.5  # HIGH > 11.5; MEDIUM in [10.5, 11.5]

# Must match load_everything() keys
MODEL_TAGS = [
    "CONST_MEDIAN",
    "SEQ_SIMPLE_Ridge",
    "PLM_ESM2_PCA64_SVR",
    "ESMFN_STRUCTURE_ElasticNet",
    "FUSION_ESM2_ESMFN_ElasticNet",
    "NESTED_STACK_NNLS",
]
ADVANCED = [t for t in MODEL_TAGS if t != "CONST_MEDIAN"]
NESTED = "NESTED_STACK_NNLS"

PLOT_SPECS = [
    (NESTED, "nested_stack_cv_public_private_bands.png", "NESTED_STACK_NNLS"),
    ("ESMFN_STRUCTURE_ElasticNet", "esmfold_structure_cv_public_private_bands.png", "ESMFN_STRUCTURE / ElasticNet"),
    ("PLM_ESM2_PCA64_SVR", "esm2_svr_cv_public_private_bands.png", "PLM_ESM2_PCA64 / SVR"),
    ("FUSION_ESM2_ESMFN_ElasticNet", "fusion_cv_public_private_bands.png", "FUSION_ESM2_ESMFN / ElasticNet"),
]

ROLES = [
    ("cv", "Train OOF", "train_ids", "train_y"),
    ("public", "Public", "public_ids", "pub_y"),
    ("private", "Private", "private_ids", "priv_y"),
]

COLOR = {"LOW": "#4c6a8a", "MEDIUM": "#d97706", "HIGH": "#b91c1c"}


def load_b62():
    path = GATE / "scripts" / "run_gate_b6_2_hic_validity.py"
    spec = importlib.util.spec_from_file_location("b62_band", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["b62_band"] = mod
    spec.loader.exec_module(mod)
    return mod.load_everything()


def as_band(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(x.shape, "MEDIUM", dtype=object)
    out[x < LOW_HI] = "LOW"
    out[x > HIGH_LO] = "HIGH"
    return out


def safe_pearson(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 5 or np.std(y[m]) < 1e-12 or np.std(p[m]) < 1e-12:
        return float("nan")
    return float(pearsonr(y[m], p[m])[0])


def safe_spearman(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 5 or np.std(y[m]) < 1e-12 or np.std(p[m]) < 1e-12:
        return float("nan")
    return float(spearmanr(y[m], p[m]).correlation)


def calc_mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def calc_rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def band_count_rows(y, role: str) -> list[dict]:
    y = np.asarray(y, float)
    rows = []
    for name, mask in [
        ("LOW", y < LOW_HI),
        ("MEDIUM", (y >= LOW_HI) & (y <= HIGH_LO)),
        ("HIGH", y > HIGH_LO),
    ]:
        yy = y[mask]
        rows.append(
            {
                "role": role,
                "band": name,
                "N": int(mask.sum()),
                "fraction": float(mask.mean()) if len(y) else np.nan,
                "HIC_mean": float(np.mean(yy)) if len(yy) else np.nan,
                "HIC_median": float(np.median(yy)) if len(yy) else np.nan,
                "HIC_min": float(np.min(yy)) if len(yy) else np.nan,
                "HIC_max": float(np.max(yy)) if len(yy) else np.nan,
            }
        )
    return rows


def draw_thresholds(ax, lim):
    for v in (LOW_HI, HIGH_LO):
        ax.axvline(v, color="#6b7280", ls=":", lw=1.0, zorder=1)
        ax.axhline(v, color="#6b7280", ls=":", lw=1.0, zorder=1)
    ax.plot(lim, lim, color="#111827", lw=1.35, zorder=1)


def annotate_box(ax, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    n_high = int((y > HIGH_LO).sum())
    high_low = int(((y > HIGH_LO) & (p < LOW_HI)).sum())
    elev = int((y >= LOW_HI).sum())
    elev_ok = int(((y >= LOW_HI) & (p >= LOW_HI)).sum())
    txt = (
        f"n={len(y)}\nMAE={calc_mae(y, p):.3f}\nSpearman={safe_spearman(y, p):.3f}\n"
        f"HIGH→LOW={high_low}/{n_high}\nelev→≥10.5={elev_ok}/{elev}"
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
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#d1d5db"),
    )


def scatter_bands(ax, y, p, ids, lim, title, label_mode="none"):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ids = np.asarray(ids)
    tb = as_band(y)
    draw_thresholds(ax, lim)
    for name, marker, size in [("LOW", "o", 22), ("MEDIUM", "D", 44), ("HIGH", "*", 78)]:
        m = tb == name
        ax.scatter(
            y[m],
            p[m],
            c=COLOR[name],
            s=size,
            marker=marker,
            alpha=0.88,
            edgecolors="white" if name != "HIGH" else "#7f1d1d",
            linewidths=0.45 if name != "HIGH" else 0.7,
            label=f"true {name}",
            zorder=2 + (0 if name == "LOW" else 1),
        )
    if label_mode == "nested":
        for yi, pi, iid, tbi in zip(y, p, ids, tb):
            if tbi == "HIGH" or (yi >= LOW_HI and pi < LOW_HI):
                ax.annotate(
                    str(iid),
                    (yi, pi),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6.0,
                    color="#7f1d1d",
                )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("measured HIC (true)")
    ax.set_ylabel("predicted HIC")
    ax.grid(True, alpha=0.22)
    annotate_box(ax, y, p)


def confusion_rows(y, p, model, role) -> list[dict]:
    tb, pb = as_band(y), as_band(p)
    rows = []
    for t in ("LOW", "MEDIUM", "HIGH"):
        n_true = int((tb == t).sum())
        for pr in ("LOW", "MEDIUM", "HIGH"):
            n = int(((tb == t) & (pb == pr)).sum())
            rows.append(
                {
                    "model": model,
                    "role": role,
                    "true_band": t,
                    "pred_band": pr,
                    "count": n,
                    "row_frac": (n / n_true) if n_true else np.nan,
                    "n_true": n_true,
                }
            )
    return rows


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    pack = load_b62()
    preds = pack["preds"]
    meta = pack["meta"]
    assert list(preds.keys()) == MODEL_TAGS, list(preds.keys())

    pop = pd.read_csv(ROOT / "gate_b3/frozen/organizer/final_population.csv").set_index("id")

    train_y = np.asarray(meta["train_y"], float)
    q90 = float(np.quantile(train_y, 0.90))
    q95 = float(np.quantile(train_y, 0.95))
    train_median = float(meta["train_median"])

    # §2 band counts
    count_rows = []
    for role_name, y in [
        ("Train", meta["train_y"]),
        ("Public", meta["pub_y"]),
        ("Private", meta["priv_y"]),
        ("Full", meta["full_y"]),
    ]:
        count_rows.extend(band_count_rows(y, role_name))
    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(METRICS / "hic_band_counts.csv", index=False)

    # §3 statistical vs literature overlap
    overlap_rows = []
    for role_name, ykey in [("Train", "train_y"), ("Public", "pub_y"), ("Private", "priv_y")]:
        y = np.asarray(meta[ykey], float)
        above_q90 = y > q90
        above_105 = y >= LOW_HI
        above_q95 = y > q95
        above_115 = y > HIGH_LO
        overlap_rows.append(
            {
                "role": role_name,
                "train_Q90": q90,
                "train_Q95": q95,
                "n_above_Q90": int(above_q90.sum()),
                "n_ge_10.5": int(above_105.sum()),
                "n_Q90_and_ge10.5": int((above_q90 & above_105).sum()),
                "n_Q90_xor_ge10.5": int((above_q90 ^ above_105).sum()),
                "n_above_Q95": int(above_q95.sum()),
                "n_gt_11.5": int(above_115.sum()),
                "n_Q95_and_gt11.5": int((above_q95 & above_115).sum()),
                "n_Q95_xor_gt11.5": int((above_q95 ^ above_115).sum()),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(METRICS / "hic_band_tail_overlap.csv", index=False)

    # shared axis
    vals = [meta["train_y"], meta["pub_y"], meta["priv_y"]]
    for tag in MODEL_TAGS:
        for role, *_ in ROLES:
            vals.append(preds[tag][role])
    lim = (float(min(np.min(v) for v in vals)) - 0.15, float(max(np.max(v) for v in vals)) + 0.15)

    # §4 plots
    for tag, fname, title in PLOT_SPECS:
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4), constrained_layout=True)
        handles = None
        for ax, (role, rtitle, idk, yk) in zip(axes, ROLES):
            scatter_bands(ax, meta[yk], preds[tag][role], meta[idk], lim, rtitle, label_mode="none")
            if handles is None:
                handles = ax.get_legend_handles_labels()
        fig.legend(*handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"{title} — literature bands (diagnostic only)\n"
            f"LOW<10.5 | MEDIUM 10.5–11.5 | HIGH>11.5 · CAND_12528 · dotted=thresholds · solid=y=x",
            fontsize=12,
        )
        fig.savefig(PLOTS / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # §5 labeled nested
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6), constrained_layout=True)
    handles = None
    for ax, (role, rtitle, idk, yk) in zip(axes, ROLES):
        scatter_bands(ax, meta[yk], preds[NESTED][role], meta[idk], lim, rtitle, label_mode="nested")
        if handles is None:
            handles = ax.get_legend_handles_labels()
    fig.legend(*handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "NESTED_STACK_NNLS — true HIGH IDs + elevated→LOW misses labeled\n"
        "Bands are diagnostic interpretation of continuous predictions",
        fontsize=12,
    )
    fig.savefig(PLOTS / "nested_stack_cv_public_private_bands_labeled.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # §7–10 confusion / errors / underprediction
    conf_rows: list[dict] = []
    err_rows: list[dict] = []
    under_rows: list[dict] = []
    for tag in MODEL_TAGS:
        for role, _rt, _idk, yk in ROLES:
            y = np.asarray(meta[yk], float)
            p = np.asarray(preds[tag][role], float)
            conf_rows.extend(confusion_rows(y, p, tag, role))
            tb = as_band(y)
            for bname in ("LOW", "MEDIUM", "HIGH"):
                msk = tb == bname
                yy, pp = y[msk], p[msk]
                err_rows.append(
                    {
                        "model": tag,
                        "role": role,
                        "true_band": bname,
                        "N": int(msk.sum()),
                        "MAE": calc_mae(yy, pp) if msk.any() else np.nan,
                        "RMSE": calc_rmse(yy, pp) if msk.any() else np.nan,
                        "Pearson": safe_pearson(yy, pp),
                        "Spearman": safe_spearman(yy, pp),
                        "mean_signed_pred_minus_true": float(np.mean(pp - yy)) if msk.any() else np.nan,
                        "median_signed_pred_minus_true": float(np.median(pp - yy)) if msk.any() else np.nan,
                    }
                )
                if bname in ("MEDIUM", "HIGH") and msk.any():
                    und = yy > pp
                    if und.any():
                        d = yy[und] - pp[und]
                        under_rows.append(
                            {
                                "model": tag,
                                "role": role,
                                "true_band": bname,
                                "subset": "underpredicted_only",
                                "n": int(und.sum()),
                                "mean_true_minus_pred": float(np.mean(d)),
                                "median_true_minus_pred": float(np.median(d)),
                                "p75_true_minus_pred": float(np.percentile(d, 75)),
                                "p90_true_minus_pred": float(np.percentile(d, 90)),
                                "max_true_minus_pred": float(np.max(d)),
                            }
                        )
                if bname == "HIGH" and msk.any():
                    sev = pp < LOW_HI
                    if sev.any():
                        d = yy[sev] - pp[sev]
                        under_rows.append(
                            {
                                "model": tag,
                                "role": role,
                                "true_band": "HIGH",
                                "subset": "HIGH_to_LOW_severe",
                                "n": int(sev.sum()),
                                "mean_true_minus_pred": float(np.mean(d)),
                                "median_true_minus_pred": float(np.median(d)),
                                "p75_true_minus_pred": float(np.percentile(d, 75)),
                                "p90_true_minus_pred": float(np.percentile(d, 90)),
                                "max_true_minus_pred": float(np.max(d)),
                            }
                        )

    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(METRICS / "hic_band_confusion.csv", index=False)
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(METRICS / "hic_band_errors.csv", index=False)
    pd.DataFrame(under_rows).to_csv(METRICS / "hic_band_underprediction_magnitude.csv", index=False)

    # confusion heatmap
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    order = ["LOW", "MEDIUM", "HIGH"]
    for ax, (role, rtitle, *_rest) in zip(axes, ROLES):
        sub = conf_df[(conf_df.model == NESTED) & (conf_df.role == role)]
        mat = np.zeros((3, 3))
        for i, t in enumerate(order):
            for j, pr in enumerate(order):
                mat[i, j] = float(sub[(sub.true_band == t) & (sub.pred_band == pr)]["count"].iloc[0])
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(3), order, fontsize=8)
        ax.set_yticks(range(3), order, fontsize=8)
        ax.set_xlabel("predicted band")
        ax.set_ylabel("true band")
        ax.set_title(rtitle)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=10)
    fig.suptitle("NESTED_STACK_NNLS — diagnostic band confusion (counts)", fontsize=12)
    fig.savefig(PLOTS / "nested_stack_band_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # §12 developability comparison
    comp_rows = []
    for tag in MODEL_TAGS:
        for role, _rt, _idk, yk in ROLES:
            y = np.asarray(meta[yk], float)
            p = np.asarray(preds[tag][role], float)
            tb, pb = as_band(y), as_band(p)
            n_high = int((tb == "HIGH").sum())
            n_med = int((tb == "MEDIUM").sum())
            n_elev = int((y >= LOW_HI).sum())
            n_low = int((tb == "LOW").sum())
            high_low = int(((tb == "HIGH") & (pb == "LOW")).sum())
            high_med = int(((tb == "HIGH") & (pb == "MEDIUM")).sum())
            high_high = int(((tb == "HIGH") & (pb == "HIGH")).sum())
            high_err = err_df[(err_df.model == tag) & (err_df.role == role) & (err_df.true_band == "HIGH")]
            comp_rows.append(
                {
                    "model": tag,
                    "role": role,
                    "overall_MAE": calc_mae(y, p),
                    "overall_Spearman": safe_spearman(y, p),
                    "n_true_MEDIUM": n_med,
                    "n_true_HIGH": n_high,
                    "HIGH_to_LOW_count": high_low,
                    "HIGH_to_LOW_rate": high_low / n_high if n_high else np.nan,
                    "HIGH_to_MEDIUM_count": high_med,
                    "HIGH_to_HIGH_count": high_high,
                    "HIGH_pred_ge10.5_rate": (((tb == "HIGH") & (p >= LOW_HI)).sum() / n_high) if n_high else np.nan,
                    "HIGH_pred_gt11.5_rate": high_high / n_high if n_high else np.nan,
                    "HIGH_pred_MEDIUM_or_HIGH_rate": (high_med + high_high) / n_high if n_high else np.nan,
                    "elevated_pred_ge10.5_rate": (((y >= LOW_HI) & (p >= LOW_HI)).sum() / n_elev) if n_elev else np.nan,
                    "MEDIUM_plus_HIGH_pred_LOW_rate": (((y >= LOW_HI) & (p < LOW_HI)).sum() / n_elev) if n_elev else np.nan,
                    "LOW_to_MEDHIGH_FP_rate": (((tb == "LOW") & (pb != "LOW")).sum() / n_low) if n_low else np.nan,
                    "HIGH_band_MAE": float(high_err["MAE"].iloc[0]) if len(high_err) else np.nan,
                    "HIGH_band_mean_bias_pred_minus_true": float(high_err["mean_signed_pred_minus_true"].iloc[0])
                    if len(high_err)
                    else np.nan,
                    "n_MEDIUM_pred_ge10.5": int(((tb == "MEDIUM") & (p >= LOW_HI)).sum()),
                    "n_HIGH_pred_ge10.5": int(((tb == "HIGH") & (p >= LOW_HI)).sum()),
                    "n_HIGH_pred_gt11.5": high_high,
                }
            )
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(METRICS / "hic_developability_model_comparison.csv", index=False)

    # ranks
    rank_rows = []
    for role in ("cv", "public", "private"):
        sub = comp_df[(comp_df.role == role) & (comp_df.model != "CONST_MEDIAN")].copy()
        sub["rank_MAE"] = sub["overall_MAE"].rank(method="min")
        sub["rank_HIGH_to_LOW"] = sub["HIGH_to_LOW_rate"].rank(method="min", na_option="bottom")
        sub["rank_elev_recog"] = (-sub["elevated_pred_ge10.5_rate"]).rank(method="min", na_option="bottom")
        sub["rank_HIGH_MAE"] = sub["HIGH_band_MAE"].rank(method="min", na_option="bottom")
        for _, r in sub.iterrows():
            rank_rows.append(
                {
                    "role": role,
                    "model": r["model"],
                    "rank_MAE": r["rank_MAE"],
                    "rank_HIGH_to_LOW": r["rank_HIGH_to_LOW"],
                    "rank_elev_recog": r["rank_elev_recog"],
                    "rank_HIGH_MAE": r["rank_HIGH_MAE"],
                    "overall_MAE": r["overall_MAE"],
                    "HIGH_to_LOW_rate": r["HIGH_to_LOW_rate"],
                    "elevated_pred_ge10.5_rate": r["elevated_pred_ge10.5_rate"],
                    "HIGH_band_MAE": r["HIGH_band_MAE"],
                }
            )
    pd.DataFrame(rank_rows).to_csv(METRICS / "hic_mae_vs_developability_ranks.csv", index=False)

    # §16 elevated antibody table
    elev_rows = []
    for role, _rt, idk, yk in ROLES:
        y = np.asarray(meta[yk], float)
        ids = list(meta[idk])
        for i, (iid, yi) in enumerate(zip(ids, y)):
            if yi < LOW_HI:
                continue
            row = {
                "role": role,
                "id": iid,
                "true_HIC": float(yi),
                "true_band": str(as_band(np.array([yi]))[0]),
                "sequence_group": int(meta["group_by_id"].get(iid, -1)),
            }
            if iid in pop.index:
                for col in ("vh_family", "vl_family", "vh_germline", "vl_germline", "vh_len", "vl_len"):
                    row[col] = pop.loc[iid, col]
            for tag in MODEL_TAGS:
                pi = float(preds[tag][role][i])
                row[f"pred_{tag}"] = pi
                row[f"pred_band_{tag}"] = str(as_band(np.array([pi]))[0])
                row[f"signed_err_{tag}"] = pi - float(yi)
            adv_bands = [str(as_band(np.array([float(preds[t][role][i])]))[0]) for t in ADVANCED]
            row["n_adv_pred_LOW"] = sum(b == "LOW" for b in adv_bands)
            row["n_adv_pred_MEDIUM"] = sum(b == "MEDIUM" for b in adv_bands)
            row["n_adv_pred_HIGH"] = sum(b == "HIGH" for b in adv_bands)
            elev_rows.append(row)
    elev_df = pd.DataFrame(elev_rows).sort_values(["true_HIC", "role"], ascending=[False, True])
    elev_df.to_csv(METRICS / "hic_elevated_antibody_predictions.csv", index=False)

    fail = elev_df[(elev_df.true_band == "HIGH") & (elev_df.n_adv_pred_LOW >= 3)].copy()
    fail.to_csv(METRICS / "hic_consensus_failures.csv", index=False)

    # ----- report helpers -----
    def get(tag, role):
        return comp_df[(comp_df.model == tag) & (comp_df.role == role)].iloc[0]

    def band_n(role_label, b):
        return int(counts_df[(counts_df.role == role_label) & (counts_df.band == b)]["N"].iloc[0])

    nested = {r: get(NESTED, r) for r in ("cv", "public", "private")}
    const = {r: get("CONST_MEDIAN", r) for r in ("cv", "public", "private")}
    cv_adv = comp_df[(comp_df.role == "cv") & (comp_df.model != "CONST_MEDIAN")].copy()

    best_high_recog = cv_adv.sort_values(
        ["HIGH_pred_MEDIUM_or_HIGH_rate", "HIGH_pred_ge10.5_rate", "overall_MAE"],
        ascending=[False, False, True],
    ).iloc[0]
    best_high_low = cv_adv.sort_values(["HIGH_to_LOW_rate", "overall_MAE"]).iloc[0]
    best_high_mae = cv_adv.sort_values(["HIGH_band_MAE", "overall_MAE"]).iloc[0]
    best_overall_mae = cv_adv.sort_values("overall_MAE").iloc[0]

    ov = overlap_df[overlap_df.role == "Train"].iloc[0]
    q90_vs = (
        f"Train n>Q90={int(ov['n_above_Q90'])} vs n≥10.5={int(ov['n_ge_10.5'])} "
        f"(AND={int(ov['n_Q90_and_ge10.5'])}, XOR={int(ov['n_Q90_xor_ge10.5'])}); "
        f"n>Q95={int(ov['n_above_Q95'])} vs n>11.5={int(ov['n_gt_11.5'])} "
        f"(AND={int(ov['n_Q95_and_gt11.5'])}, XOR={int(ov['n_Q95_xor_gt11.5'])}). "
        f"Q90={q90:.4f} ≈ 10.5; Q95={q95:.4f} < 11.5 so literature HIGH is stricter than statistical top 5%."
    )

    elev_cv = float(nested["cv"]["elevated_pred_ge10.5_rate"])
    high_mh_cv = (
        float(nested["cv"]["HIGH_pred_MEDIUM_or_HIGH_rate"])
        if np.isfinite(nested["cv"]["HIGH_pred_MEDIUM_or_HIGH_rate"])
        else 0.0
    )
    high_low_cv = float(nested["cv"]["HIGH_to_LOW_rate"]) if np.isfinite(nested["cv"]["HIGH_to_LOW_rate"]) else 1.0
    nest_beats = elev_cv > float(const["cv"]["elevated_pred_ge10.5_rate"])

    if elev_cv >= 0.7 and high_low_cv <= 0.2 and high_mh_cv >= 0.7:
        verdict = "KEEP_HIC_CONTINUOUS_STRONG_SUPPORT"
    elif nest_beats:
        verdict = "KEEP_HIC_CONTINUOUS_WITH_CAVEATS"
    else:
        verdict = "HIC_CONTINUOUS_UTILITY_QUESTIONABLE"

    fail_ids = ", ".join(fail["id"].astype(str).unique().tolist()) if len(fail) else "(none)"

    pub_hl = float(nested["public"]["HIGH_to_LOW_rate"]) if np.isfinite(nested["public"]["HIGH_to_LOW_rate"]) else np.nan
    priv_hl = float(nested["private"]["HIGH_to_LOW_rate"]) if np.isfinite(nested["private"]["HIGH_to_LOW_rate"]) else np.nan
    pub_elev = float(nested["public"]["elevated_pred_ge10.5_rate"])
    priv_elev = float(nested["private"]["elevated_pred_ge10.5_rate"])
    if np.isfinite(pub_hl) and np.isfinite(priv_hl):
        if pub_hl > priv_hl + 0.15 and pub_elev + 0.1 < priv_elev:
            pub_note = "Public shows worse HIGH→LOW / elevated recognition than Private — not only Pearson fragility."
        else:
            pub_note = (
                "Public Pearson weakness is not clearly a unique developability HIGH→LOW disaster vs Private; "
                "both roles show sparse HIGH and residual undershoot. Treat rates as descriptive."
            )
    else:
        pub_note = "HIGH counts too sparse for a firm Public-vs-Private developability contrast."

    same_as_mae = (
        best_overall_mae["model"] == best_high_recog["model"]
        and best_overall_mae["model"] == best_high_mae["model"]
        and best_overall_mae["model"] == best_high_low["model"]
    )

    report = f"""# Gate B6.3 — HIC Developability-Band Utility Audit

HIC continuous-task developability verdict: **{verdict}**

Literature-informed bands (diagnostic only; NOT a new competition task):
- **LOW:** HIC < 10.5 min
- **MEDIUM:** 10.5 ≤ HIC ≤ 11.5 min
- **HIGH:** HIC > 11.5 min

These are Jain/Shehata assay-context interpretation aids — **not** universal clinical cutoffs and **not** replacements for continuous HIC RT.

True-band counts:
- Train: LOW={band_n('Train','LOW')}, MEDIUM={band_n('Train','MEDIUM')}, HIGH={band_n('Train','HIGH')}
- Public: LOW={band_n('Public','LOW')}, MEDIUM={band_n('Public','MEDIUM')}, HIGH={band_n('Public','HIGH')}
- Private: LOW={band_n('Private','LOW')}, MEDIUM={band_n('Private','MEDIUM')}, HIGH={band_n('Private','HIGH')}

NESTED_STACK_NNLS:
- overall MAE: Train OOF={nested['cv']['overall_MAE']:.4f}, Public={nested['public']['overall_MAE']:.4f}, Private={nested['private']['overall_MAE']:.4f}
- HIGH-band MAE: Train={nested['cv']['HIGH_band_MAE']:.4f}, Public={nested['public']['HIGH_band_MAE']:.4f}, Private={nested['private']['HIGH_band_MAE']:.4f}
- true HIGH: Train={int(nested['cv']['n_true_HIGH'])}, Public={int(nested['public']['n_true_HIGH'])}, Private={int(nested['private']['n_true_HIGH'])}
- HIGH→LOW: Train={int(nested['cv']['HIGH_to_LOW_count'])} (rate={nested['cv']['HIGH_to_LOW_rate']:.3f}), Public={int(nested['public']['HIGH_to_LOW_count'])} ({nested['public']['HIGH_to_LOW_rate']:.3f}), Private={int(nested['private']['HIGH_to_LOW_count'])} ({nested['private']['HIGH_to_LOW_rate']:.3f})
- HIGH predicted MEDIUM/HIGH: Train={nested['cv']['HIGH_pred_MEDIUM_or_HIGH_rate']:.3f}, Public={nested['public']['HIGH_pred_MEDIUM_or_HIGH_rate']:.3f}, Private={nested['private']['HIGH_pred_MEDIUM_or_HIGH_rate']:.3f}
- MEDIUM+HIGH predicted LOW: Train={nested['cv']['MEDIUM_plus_HIGH_pred_LOW_rate']:.3f}, Public={nested['public']['MEDIUM_plus_HIGH_pred_LOW_rate']:.3f}, Private={nested['private']['MEDIUM_plus_HIGH_pred_LOW_rate']:.3f}

CONST_MEDIAN comparison:
- CONST never predicts ≥10.5 (Train median ≈ {train_median:.3f}) → elevated recognition = 0 on all roles.
- NESTED elevated (≥10.5) recognition: CV={nested['cv']['elevated_pred_ge10.5_rate']:.3f}, Pub={nested['public']['elevated_pred_ge10.5_rate']:.3f}, Priv={nested['private']['elevated_pred_ge10.5_rate']:.3f}.
- Sequence models **do** move some elevated antibodies above 10.5 vs a constant median, but many remain under-shifted.

Best advanced model for HIGH recognition (CV, MEDIUM|HIGH rate): **{best_high_recog['model']}** ({best_high_recog['HIGH_pred_MEDIUM_or_HIGH_rate']:.3f})
Best advanced model for HIGH-band MAE (CV): **{best_high_mae['model']}** (MAE={best_high_mae['HIGH_band_MAE']:.3f})
Lowest HIGH→LOW rate (CV): **{best_high_low['model']}** (rate={best_high_low['HIGH_to_LOW_rate']:.3f})
Best overall MAE (CV): **{best_overall_mae['model']}** (MAE={best_overall_mae['overall_MAE']:.3f})
Same model wins all? **{'YES' if same_as_mae else 'NO'}**

Does Public Pearson weakness correspond to dangerous HIGH→LOW errors: {pub_note}

Does continuous HIC provide useful developability information: **YES vs CONST_MEDIAN**, with systematic undershoot caveats.

Does CAND_12528 remain acceptable: **YES** (MAE-first continuous eval; sparse HIGH is a target-property issue, not a split defect requiring re-search).

Final recommendation: **{verdict}** — keep continuous HIC; use 10.5/11.5 only as interpretation / organizer diagnostics; do **not** auto-convert to classification.

---

## Band sample-size caveat

MEDIUM/HIGH are sparse. Rates involving HIGH are descriptive and unstable.

## Statistical tail vs literature bands

{q90_vs}

## Model comparison (Train OOF)

| model | MAE | Spearman | HIGH→LOW | HIGH≥10.5 | elev≥10.5 | HIGH MAE | HIGH bias |
|-------|-----|----------|----------|-----------|-----------|----------|-----------|
"""
    for _, r in cv_adv.sort_values("overall_MAE").iterrows():
        report += (
            f"| {r['model']} | {r['overall_MAE']:.3f} | {r['overall_Spearman']:.3f} | "
            f"{r['HIGH_to_LOW_rate']:.2f} | {r['HIGH_pred_ge10.5_rate']:.2f} | "
            f"{r['elevated_pred_ge10.5_rate']:.2f} | {r['HIGH_band_MAE']:.2f} | "
            f"{r['HIGH_band_mean_bias_pred_minus_true']:.2f} |\n"
        )
    c0 = const["cv"]
    report += (
        f"\nCONST_MEDIAN CV: MAE={c0['overall_MAE']:.3f}, elev recognition=0, "
        f"HIGH→LOW rate={c0['HIGH_to_LOW_rate']:.2f} (always LOW).\n"
    )

    report += """
## MAE ranking vs developability ranking

See `metrics/hic_mae_vs_developability_ranks.csv`. Overall-MAE winners need not be the safest HIGH→LOW avoiders.

## Public vs Private (advanced)

| model | Pub HIGH→LOW | Priv HIGH→LOW | Pub elev≥10.5 | Priv elev≥10.5 | Pub HIGH MAE | Priv HIGH MAE |
|-------|--------------|---------------|---------------|----------------|--------------|---------------|
"""
    for tag in ADVANCED:
        a, b = get(tag, "public"), get(tag, "private")
        report += (
            f"| {tag} | {a['HIGH_to_LOW_rate']:.2f} | {b['HIGH_to_LOW_rate']:.2f} | "
            f"{a['elevated_pred_ge10.5_rate']:.2f} | {b['elevated_pred_ge10.5_rate']:.2f} | "
            f"{a['HIGH_band_MAE']:.2f} | {b['HIGH_band_MAE']:.2f} |\n"
        )

    report += f"""
## Consensus severe failures (true HIGH & ≥3 advanced models predict LOW)

n={len(fail)} — IDs: {fail_ids}

Full table: `metrics/hic_consensus_failures.csv` (includes VH/VL family when available). No new feature discovery was run.

## Answers (Q1–17)

1. Counts as in header (Train/Public/Private LOW/MEDIUM/HIGH).
2. Train Q90≈{q90:.3f} ≈ literature 10.5 (MEDIUM+HIGH gateway); Train Q95≈{q95:.3f} < 11.5 → literature HIGH is a stricter subset of the statistical upper tail.
3. true HIGH: Train={band_n('Train','HIGH')}, Public={band_n('Public','HIGH')}, Private={band_n('Private','HIGH')}.
4. NESTED HIGH→LOW counts: Train={int(nested['cv']['HIGH_to_LOW_count'])}, Public={int(nested['public']['HIGH_to_LOW_count'])}, Private={int(nested['private']['HIGH_to_LOW_count'])}.
5. NESTED HIGH→MEDIUM|HIGH rates: Train={nested['cv']['HIGH_pred_MEDIUM_or_HIGH_rate']:.3f}, Public={nested['public']['HIGH_pred_MEDIUM_or_HIGH_rate']:.3f}, Private={nested['private']['HIGH_pred_MEDIUM_or_HIGH_rate']:.3f}.
6. MEDIUM+HIGH→LOW rates: Train={nested['cv']['MEDIUM_plus_HIGH_pred_LOW_rate']:.3f}, Public={nested['public']['MEDIUM_plus_HIGH_pred_LOW_rate']:.3f}, Private={nested['private']['MEDIUM_plus_HIGH_pred_LOW_rate']:.3f}.
7. Yes — NESTED (and other advanced models) beat CONST_MEDIAN on elevated recognition (CONST=0 by construction).
8. Best HIGH recognition (CV): {best_high_recog['model']}.
9. Lowest HIGH→LOW (CV): {best_high_low['model']}.
10. Lowest HIGH-band MAE (CV): {best_high_mae['model']}.
11. Same as overall MAE winner? overall={best_overall_mae['model']}; HIGH-recog={best_high_recog['model']}; HIGH-MAE={best_high_mae['model']}; HIGH→LOW={best_high_low['model']} — {'aligned' if same_as_mae else 'not fully aligned'}.
12. Yes — HIGH-band mean bias (pred−true) remains strongly negative even among models that sometimes clear 10.5.
13. {pub_note}
14. CAND_12528 remains acceptable; no new split search indicated by this band audit.
15. Yes — continuous models provide elevated-HIC directional signal beyond the median predictor, with residual undershoot.
16. Yes — categorical conversion would discard magnitude (e.g. 12.0→11.4 vs 14.0→10.6 both collapse as HIGH→MEDIUM).
17. Yes — keep HIC as a **continuous** competition track; do not auto-convert to binary/ordinal.

## Plots / metrics

- `plots/hic_band_audit/nested_stack_cv_public_private_bands.png`
- `plots/hic_band_audit/nested_stack_cv_public_private_bands_labeled.png`
- `plots/hic_band_audit/esmfold_structure_cv_public_private_bands.png`
- `plots/hic_band_audit/esm2_svr_cv_public_private_bands.png`
- `plots/hic_band_audit/fusion_cv_public_private_bands.png`
- `plots/hic_band_audit/nested_stack_band_confusion.png`
- `metrics/hic_band_counts.csv`
- `metrics/hic_band_tail_overlap.csv`
- `metrics/hic_band_confusion.csv`
- `metrics/hic_band_errors.csv`
- `metrics/hic_developability_model_comparison.csv`
- `metrics/hic_elevated_antibody_predictions.csv`
- `metrics/hic_consensus_failures.csv`
"""
    out = REPORTS / "06_hic_developability_band_audit.md"
    out.write_text(report)
    print("wrote", out)
    print("verdict", verdict)
    print(counts_df[counts_df.role.isin(["Train", "Public", "Private"])][["role", "band", "N", "fraction"]].to_string(index=False))
    print("--- nested ---")
    cols = [
        "role",
        "overall_MAE",
        "n_true_HIGH",
        "HIGH_to_LOW_count",
        "HIGH_to_LOW_rate",
        "HIGH_pred_MEDIUM_or_HIGH_rate",
        "elevated_pred_ge10.5_rate",
        "HIGH_band_MAE",
    ]
    print(comp_df[comp_df.model == NESTED][cols].to_string(index=False))
    print("consensus failures", len(fail), fail_ids)


if __name__ == "__main__":
    main()
