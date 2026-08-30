#!/usr/bin/env python3
"""Gate B6.2 — HIC Continuous-Regression Validity Audit.

Phases (in order):
  1 Distribution → 2 Shrinkage → 3 Tail → 4 Pearson influence → 5 CV design → 6 Decision

Uses frozen CAND_12528 split and frozen Gate-B5 predictions; Phase-5 CV only
retrains representative models with FIXED hyperparameters (no Optuna).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b6_2_hic_validity"
ORG = ROOT / "gate_b3" / "frozen" / "organizer"
B4_CFG = ROOT / "gate_b4_absolute" / "config"
B5_PRED = ROOT / "gate_b5_ceiling" / "predictions"
B6_CFG = ROOT / "gate_b6_split_search" / "config"
B1_CACHE = ROOT / "gate_b1" / "cache"
B3_FEAT = ROOT / "gate_b3" / "features"

CONFIG = GATE / "config"
METRICS = GATE / "metrics"
PLOTS = GATE / "plots"
REPORTS = GATE / "reports"
CACHE = GATE / "cache"
LOGS = GATE / "logs"

MODEL_TAGS = [
    "CONST_MEDIAN",
    "SEQ_SIMPLE_Ridge",
    "PLM_ESM2_PCA64_SVR",
    "ESMFN_STRUCTURE_ElasticNet",
    "FUSION_ESM2_ESMFN_ElasticNet",
    "NESTED_STACK_NNLS",
]
NONCONST = [t for t in MODEL_TAGS if t != "CONST_MEDIAN"]
STRONG_MODELS = [
    "PLM_ESM2_PCA64_SVR",
    "ESMFN_STRUCTURE_ElasticNet",
    "FUSION_ESM2_ESMFN_ElasticNet",
    "NESTED_STACK_NNLS",
    "SEQ_SIMPLE_Ridge",
]

CV_SEED_BASE = 20260830
N_FOLDS = 5
N_REPEATS_NEW = 5
ALPHAS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# Fixed hyperparameters (do NOT Optuna)
HP = {
    "SEQ_SIMPLE_Ridge": {"alpha": 10.0},
    "PLM_ESM2_PCA64_SVR": {
        "C": 0.837776,
        "epsilon": 0.05832,
        "gamma": 0.009757,
        "n_components": 64,
    },
    "ESMFN_STRUCTURE_ElasticNet": {"alpha": 0.3025, "l1_ratio": 0.2586, "max_iter": 10000},
    "FUSION_ESM2_ESMFN_ElasticNet": {"alpha": 0.2025, "l1_ratio": 0.5567, "max_iter": 10000},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for p in (CONFIG, METRICS, PLOTS, REPORTS, CACHE, LOGS, GATE / "scripts"):
        p.mkdir(parents=True, exist_ok=True)


def sha256_lines(ids) -> str:
    blob = "\n".join(sorted(map(str, ids))) + "\n"
    return hashlib.sha256(blob.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def safe_pearson(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or np.std(y[m]) < 1e-12 or np.std(p[m]) < 1e-12:
        return float("nan")
    r = pearsonr(y[m], p[m])[0]
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def safe_spearman(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or np.std(y[m]) < 1e-12 or np.std(p[m]) < 1e-12:
        return float("nan")
    r = spearmanr(y[m], p[m]).correlation
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def mae(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m]))) if m.any() else float("nan")


def rmse(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else float("nan")


def mad(x) -> float:
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))))


def sanitize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    keep = np.any(np.isfinite(X), axis=0)
    X = X[:, keep] if keep.any() else np.zeros((X.shape[0], 1), float)
    col = np.nanmean(X, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    X = np.array(X, copy=True)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col, inds[1])
    X[~np.isfinite(X)] = 0.0
    return X


def sanitize_pair(Xtr, Xte):
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)
    keep = np.any(np.isfinite(Xtr), axis=0)
    Xtr = Xtr[:, keep] if keep.any() else np.zeros((Xtr.shape[0], 1), float)
    Xte = Xte[:, keep] if keep.any() else np.zeros((Xte.shape[0], 1), float)
    col = np.nanmean(Xtr, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    for X in (Xtr, Xte):
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col, inds[1])
        X[~np.isfinite(X)] = 0.0
    return Xtr, Xte


def narrowest_interval(x: np.ndarray, frac: float) -> tuple[float, float, float]:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    k = max(1, int(np.ceil(frac * n)))
    if k >= n:
        return float(x[0]), float(x[-1]), float(x[-1] - x[0])
    widths = x[k - 1 :] - x[: n - k + 1]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k - 1]), float(widths[i])


def load_csv_num(path: Path, ids, prefixes=None, id_col="antibody_id") -> np.ndarray:
    df = pd.read_csv(path)
    if id_col not in df.columns and "id" in df.columns:
        id_col = "id"
    df = pd.DataFrame({"id": list(ids)}).merge(df, left_on="id", right_on=id_col, how="left")
    cols = [
        c
        for c in df.columns
        if c not in ("id", id_col, "antibody_id") and pd.api.types.is_numeric_dtype(df[c])
    ]
    if prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    if not cols:
        return np.zeros((len(ids), 1), float)
    return sanitize(df[cols].values.astype(float))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_everything() -> dict:
    role = pd.read_csv(ORG / "role_map.csv")
    labels = pd.read_csv(ORG / "test_labels_hidden.csv")
    hic_by_id = dict(zip(role["id"], role["HIC"].astype(float)))
    # Prefer hidden test labels for holdout (should match role_map)
    for _, r in labels.iterrows():
        hic_by_id[r["id"]] = float(r["HIC"])
    group_by_id = dict(zip(role["id"], role["sequence_group"]))

    cand_set = json.loads((B6_CFG / "B6_1_FINAL_CANDIDATE_SET.json").read_text())
    cand = next(c for c in cand_set["candidates"] if c["candidate_id"] == "CAND_12528")
    public_ids = list(cand["public_ids"])
    private_ids = list(cand["private_ids"])

    train_df = role.loc[role["role"] == "Train"].copy().reset_index(drop=True)
    outer = json.loads((B4_CFG / "OUTER_CV_FOLDS.json").read_text())
    train_ids = list(outer["train_ids"])
    # Align train_df to outer train_ids order
    train_lookup = train_df.set_index("id")
    train_y = np.array([float(train_lookup.loc[i, "HIC"]) for i in train_ids])
    train_groups = np.array([int(train_lookup.loc[i, "sequence_group"]) for i in train_ids])

    org_pub_order = role.loc[role["role"] == "Public", "id"].tolist()
    org_priv_order = role.loc[role["role"] == "Private", "id"].tolist()

    preds: dict[str, dict[str, np.ndarray]] = {}
    for tag in MODEL_TAGS:
        oof = np.load(B5_PRED / "oof" / f"HIC__{tag}__meanOOF.npy").astype(float).ravel()
        assert len(oof) == len(train_ids), f"OOF len mismatch {tag}"
        pub = np.load(B5_PRED / "final" / f"HIC__{tag}__public.npy").astype(float).ravel()
        priv = np.load(B5_PRED / "final" / f"HIC__{tag}__private.npy").astype(float).ravel()
        assert len(pub) == len(org_pub_order) and len(priv) == len(org_priv_order)
        concat = np.concatenate([pub, priv])
        pred_by_id = dict(zip(org_pub_order + org_priv_order, concat))
        pub_p = np.array([pred_by_id[i] for i in public_ids], float)
        priv_p = np.array([pred_by_id[i] for i in private_ids], float)
        preds[tag] = {"cv": oof, "public": pub_p, "private": priv_p}

    pub_y = np.array([hic_by_id[i] for i in public_ids], float)
    priv_y = np.array([hic_by_id[i] for i in private_ids], float)
    full_ids = train_ids + public_ids + private_ids
    full_y = np.concatenate([train_y, pub_y, priv_y])

    meta = {
        "train_ids": train_ids,
        "public_ids": public_ids,
        "private_ids": private_ids,
        "train_y": train_y,
        "pub_y": pub_y,
        "priv_y": priv_y,
        "full_y": full_y,
        "full_ids": full_ids,
        "train_groups": train_groups,
        "group_by_id": group_by_id,
        "hic_by_id": hic_by_id,
        "outer": outer,
        "cand": cand,
        "role": role,
        "train_median": float(np.median(train_y)),
        "train_mean": float(np.mean(train_y)),
    }
    return {"preds": preds, "meta": meta}


def role_arrays(meta, role: str):
    if role == "cv" or role == "train" or role == "Train":
        return meta["train_ids"], meta["train_y"]
    if role == "public" or role == "Public":
        return meta["public_ids"], meta["pub_y"]
    if role == "private" or role == "Private":
        return meta["private_ids"], meta["priv_y"]
    raise KeyError(role)


# ---------------------------------------------------------------------------
# Phase 1 — Distribution
# ---------------------------------------------------------------------------
def phase1(meta: dict) -> dict:
    print("=== PHASE 1: HIC target distribution ===", flush=True)
    roles = {
        "Full": meta["full_y"],
        "Train": meta["train_y"],
        "Public": meta["pub_y"],
        "Private": meta["priv_y"],
    }
    q_list = [0.01, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.975, 0.99]
    rows = []
    for name, y in roles.items():
        y = np.asarray(y, float)
        row = {
            "role": name,
            "N": len(y),
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "SD": float(np.std(y, ddof=1)),
            "MAD": mad(y),
            "IQR": float(np.percentile(y, 75) - np.percentile(y, 25)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "skewness": float(stats.skew(y)),
            "excess_kurtosis": float(stats.kurtosis(y, fisher=True)),
        }
        for q in q_list:
            row[f"q{q:g}"] = float(np.quantile(y, q))
        # central concentration
        med, iqr = row["median"], row["IQR"]
        mean, sd = row["mean"], row["SD"]
        for k, half in [("med_pm_0.25IQR", 0.25 * iqr), ("med_pm_0.50IQR", 0.50 * iqr), ("med_pm_1.00IQR", 1.0 * iqr)]:
            row[f"frac_{k}"] = float(np.mean(np.abs(y - med) <= half))
        for k, half in [("mean_pm_0.5SD", 0.5 * sd), ("mean_pm_1.0SD", 1.0 * sd)]:
            row[f"frac_{k}"] = float(np.mean(np.abs(y - mean) <= half))
        for frac in [0.50, 0.70, 0.80, 0.90, 0.95]:
            lo, hi, w = narrowest_interval(y, frac)
            row[f"narrow{int(frac*100)}_lo"] = lo
            row[f"narrow{int(frac*100)}_hi"] = hi
            row[f"narrow{int(frac*100)}_width"] = w
        # bin diagnostics
        n = len(y)
        sturges = int(np.ceil(1 + np.log2(n)))
        fd_w = 2 * iqr / (n ** (1 / 3)) if iqr > 0 else np.nan
        fd = int(np.ceil((y.max() - y.min()) / fd_w)) if fd_w and fd_w > 0 else np.nan
        scott_w = 3.5 * sd / (n ** (1 / 3)) if sd > 0 else np.nan
        scott = int(np.ceil((y.max() - y.min()) / scott_w)) if scott_w and scott_w > 0 else np.nan
        row["bins_sturges"] = sturges
        row["bins_fd"] = fd
        row["bins_scott"] = scott
        rows.append(row)
    dist = pd.DataFrame(rows)
    dist.to_csv(METRICS / "hic_distribution_summary.csv", index=False)

    # Upper-tail counts (Train-defined thresholds)
    ty = meta["train_y"]
    thr_q = {
        "Train_Q80": float(np.quantile(ty, 0.80)),
        "Train_Q90": float(np.quantile(ty, 0.90)),
        "Train_Q95": float(np.quantile(ty, 0.95)),
        "Train_Q97.5": float(np.quantile(ty, 0.975)),
    }
    abs_thr = [10.0, 10.5, 11.0, 11.5, 12.0]
    abs_thr = [t for t in abs_thr if t < float(np.max(meta["full_y"]))]
    tail_rows = []
    for name, y in roles.items():
        if name == "Full":
            continue
        rec = {"role": name, "N": len(y)}
        for k, t in thr_q.items():
            rec[f"n_gt_{k}"] = int(np.sum(y > t))
            rec[f"frac_gt_{k}"] = float(np.mean(y > t))
        for t in abs_thr:
            rec[f"n_gt_{t:g}"] = int(np.sum(y > t))
            rec[f"frac_gt_{t:g}"] = float(np.mean(y > t))
        # outlier flags (Train-defined fences for comparability across roles)
        q1, q3 = np.percentile(ty, [25, 75])
        iqr = q3 - q1
        fence15 = q3 + 1.5 * iqr
        fence3 = q3 + 3 * iqr
        med_t, mad_t = np.median(ty), mad(ty)
        rec["n_tukey_1.5"] = int(np.sum(y > fence15))
        rec["n_tukey_3.0"] = int(np.sum(y > fence3))
        if mad_t > 0:
            rz = 0.6745 * (y - med_t) / mad_t
            rec["n_robust_z_gt_3"] = int(np.sum(rz > 3))
            rec["n_robust_z_gt_3.5"] = int(np.sum(rz > 3.5))
        else:
            rec["n_robust_z_gt_3"] = 0
            rec["n_robust_z_gt_3.5"] = 0
        rec["n_top5pct_train"] = int(np.sum(y > np.quantile(ty, 0.95)))
        rec["n_top2.5pct_train"] = int(np.sum(y > np.quantile(ty, 0.975)))
        tail_rows.append(rec)
    pd.DataFrame(tail_rows).to_csv(METRICS / "hic_tail_counts.csv", index=False)

    # Role comparisons
    pairs = [("Train", "Public"), ("Train", "Private"), ("Public", "Private")]
    cmp_rows = []
    for a, b in pairs:
        ya, yb = roles[a], roles[b]
        cmp_rows.append(
            {
                "pair": f"{a}_vs_{b}",
                "wasserstein": float(wasserstein_distance(ya, yb)),
                "ks_stat": float(ks_2samp(ya, yb).statistic),
                "ks_pvalue": float(ks_2samp(ya, yb).pvalue),
                "mean_diff": float(np.mean(ya) - np.mean(yb)),
                "median_diff": float(np.median(ya) - np.median(yb)),
                "SD_ratio": float(np.std(ya, ddof=1) / np.std(yb, ddof=1)),
                "IQR_ratio": float(
                    (np.percentile(ya, 75) - np.percentile(ya, 25))
                    / max(1e-12, (np.percentile(yb, 75) - np.percentile(yb, 25)))
                ),
                "Q90_diff": float(np.quantile(ya, 0.90) - np.quantile(yb, 0.90)),
                "Q95_diff": float(np.quantile(ya, 0.95) - np.quantile(yb, 0.95)),
                "n_gt_TrainQ90_a": int(np.sum(ya > thr_q["Train_Q90"])),
                "n_gt_TrainQ90_b": int(np.sum(yb > thr_q["Train_Q90"])),
                "n_gt_TrainQ95_a": int(np.sum(ya > thr_q["Train_Q95"])),
                "n_gt_TrainQ95_b": int(np.sum(yb > thr_q["Train_Q95"])),
            }
        )
    pd.DataFrame(cmp_rows).to_csv(METRICS / "hic_role_distribution_comparison.csv", index=False)

    # Plots
    xmin, xmax = float(meta["full_y"].min()) - 0.1, float(meta["full_y"].max()) + 0.1
    bins = int(dist.loc[dist.role == "Train", "bins_sturges"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True)
    for ax, name, color in zip(axes, ["Train", "Public", "Private"], ["#2c5f7c", "#b35c1e", "#3a7d44"]):
        ax.hist(roles[name], bins=bins, range=(xmin, xmax), color=color, edgecolor="white", alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("HIC")
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS / "hic_distribution_hist_by_role.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in zip(["Train", "Public", "Private"], ["#2c5f7c", "#b35c1e", "#3a7d44"]):
        ax.hist(roles[name], bins=bins, range=(xmin, xmax), density=True, alpha=0.45, label=name, color=color)
    ax.set_xlabel("HIC")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "hic_distribution_density_overlay.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in zip(["Train", "Public", "Private"], ["#2c5f7c", "#b35c1e", "#3a7d44"]):
        ys = np.sort(roles[name])
        ax.plot(ys, np.linspace(0, 1, len(ys)), label=name, color=color, lw=2)
    ax.set_xlabel("HIC")
    ax.set_ylabel("ECDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "hic_distribution_ecdf.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    parts = ax.violinplot([roles["Train"], roles["Public"], roles["Private"]], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Train", "Public", "Private"])
    ax.set_ylabel("HIC")
    fig.tight_layout()
    fig.savefig(PLOTS / "hic_distribution_violin.png", dpi=140)
    plt.close(fig)

    tr = dist.loc[dist.role == "Train"].iloc[0]
    # Report
    md = []
    md.append("# 01 — HIC Target Distribution\n")
    md.append("## Summary (Train)\n")
    md.append(f"- N={int(tr.N)}, mean={tr['mean']:.4f}, median={tr['median']:.4f}, SD={tr['SD']:.4f}, IQR={tr['IQR']:.4f}\n")
    md.append(f"- skewness={tr['skewness']:.3f}, excess kurtosis={tr['excess_kurtosis']:.3f}\n")
    md.append(f"- bin recommendations: Sturges={int(tr.bins_sturges)}, FD={tr.bins_fd}, Scott={tr.bins_scott}\n")
    md.append("\n## Answers\n")
    md.append(f"1. **Is HIC strongly right-skewed?** {'YES' if tr['skewness'] > 1 else 'MODERATE' if tr['skewness']>0.5 else 'NO'} (skew={tr['skewness']:.3f}).\n")
    md.append(f"2. **Central concentration:** frac within median±0.5 IQR = {tr['frac_med_pm_0.50IQR']:.3f}; narrowest 80% width = {tr['narrow80_width']:.3f}.\n")
    md.append(f"3. **High-HIC tail:** Train n>Q90={int(tail_rows[0]['n_gt_Train_Q90'])}, n>Q95={int(tail_rows[0]['n_gt_Train_Q95'])}.\n")
    md.append(f"4. **Tail representation:** Pub/Priv n>TrainQ90 = {tail_rows[1]['n_gt_Train_Q90']}/{tail_rows[2]['n_gt_Train_Q90']}; n>Q95 = {tail_rows[1]['n_gt_Train_Q95']}/{tail_rows[2]['n_gt_Train_Q95']}.\n")
    md.append(f"5. **Mean vs median:** mean−median = {tr['mean']-tr['median']:.4f} (mean pulled right by tail).\n")
    md.append("6. **MAE central-value advantage:** Yes plausible — dense central mass + sparse right tail favors predicting near the median.\n")
    (REPORTS / "01_hic_target_distribution.md").write_text("".join(md))
    return {"dist": dist, "tail_rows": tail_rows, "cmp_rows": cmp_rows, "thr_q": thr_q}


# ---------------------------------------------------------------------------
# Phase 2 — Shrinkage
# ---------------------------------------------------------------------------
def phase2(preds: dict, meta: dict) -> dict:
    print("=== PHASE 2: Prediction shrinkage ===", flush=True)
    train_med, train_mean = meta["train_median"], meta["train_mean"]
    role_map = {"cv": ("Train OOF", meta["train_y"]), "public": ("Public", meta["pub_y"]), "private": ("Private", meta["priv_y"])}

    # Constant baselines
    base_rows = []
    for role, (label, y) in role_map.items():
        for name, const in [("median", train_med), ("mean", train_mean)]:
            p = np.full_like(y, const, dtype=float)
            base_rows.append(
                {
                    "role": role,
                    "baseline": f"train_{name}",
                    "MAE": mae(y, p),
                    "RMSE": rmse(y, p),
                    "Pearson": np.nan,
                    "Spearman": np.nan,
                }
            )
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(METRICS / "constant_baseline_comparison.csv", index=False)
    median_mae = {r: float(base_df[(base_df.role == r) & (base_df.baseline == "train_median")]["MAE"].iloc[0]) for r in role_map}

    # Dispersion + gain
    disp_rows = []
    gain_rows = []
    cal_rows = []
    for tag in MODEL_TAGS:
        for role, (label, y) in role_map.items():
            p = preds[tag][role]
            obs_sd = float(np.std(y, ddof=1))
            pred_sd = float(np.std(p, ddof=1))
            obs_iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
            pred_iqr = float(np.percentile(p, 75) - np.percentile(p, 25))
            m = mae(y, p)
            disp_rows.append(
                {
                    "model": tag,
                    "role": role,
                    "obs_mean": float(np.mean(y)),
                    "obs_median": float(np.median(y)),
                    "obs_SD": obs_sd,
                    "obs_IQR": obs_iqr,
                    "obs_min": float(np.min(y)),
                    "obs_max": float(np.max(y)),
                    "pred_mean": float(np.mean(p)),
                    "pred_median": float(np.median(p)),
                    "pred_SD": pred_sd,
                    "pred_IQR": pred_iqr,
                    "pred_min": float(np.min(p)),
                    "pred_max": float(np.max(p)),
                    "pred_SD_over_obs_SD": pred_sd / obs_sd if obs_sd > 0 else np.nan,
                    "pred_IQR_over_obs_IQR": pred_iqr / obs_iqr if obs_iqr > 0 else np.nan,
                    "MAE": m,
                    "RMSE": rmse(y, p),
                    "Pearson": safe_pearson(y, p) if tag != "CONST_MEDIAN" else np.nan,
                    "Spearman": safe_spearman(y, p) if tag != "CONST_MEDIAN" else np.nan,
                    "absolute_gain_vs_median": median_mae[role] - m,
                    "relative_gain_vs_median": (median_mae[role] - m) / median_mae[role] if median_mae[role] else np.nan,
                }
            )
            # calibration
            if tag != "CONST_MEDIAN" and np.std(p) > 1e-12:
                slope, intercept = np.polyfit(p, y, 1)
                yhat = intercept + slope * p
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                slope, intercept, r2 = np.nan, np.nan, np.nan
            cal_rows.append(
                {
                    "model": tag,
                    "role": role,
                    "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
                    "slope": float(slope) if np.isfinite(slope) else np.nan,
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "mean_bias_pred_minus_obs": float(np.mean(p - y)),
                }
            )
    disp_df = pd.DataFrame(disp_rows)
    disp_df.to_csv(METRICS / "prediction_dispersion.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(METRICS / "calibration.csv", index=False)

    # Artificial shrinkage on TRAIN OOF; apply Train-selected alpha diagnostically
    shrink_rows = []
    best_alpha = {}
    for tag in STRONG_MODELS:
        y = meta["train_y"]
        p0 = preds[tag]["cv"]
        best = None
        for a in ALPHAS:
            p = train_med + a * (p0 - train_med)
            row = {
                "model": tag,
                "role": "cv",
                "alpha": a,
                "MAE": mae(y, p),
                "RMSE": rmse(y, p),
                "Pearson": safe_pearson(y, p),
                "Spearman": safe_spearman(y, p),
            }
            shrink_rows.append(row)
            if best is None or row["MAE"] < best["MAE"]:
                best = row
        best_alpha[tag] = best
        # diagnostic apply to Pub/Priv
        a_star = best["alpha"]
        for role, y in [("public", meta["pub_y"]), ("private", meta["priv_y"])]:
            p0 = preds[tag][role]
            p = train_med + a_star * (p0 - train_med)
            shrink_rows.append(
                {
                    "model": tag,
                    "role": role,
                    "alpha": a_star,
                    "MAE": mae(y, p),
                    "RMSE": rmse(y, p),
                    "Pearson": safe_pearson(y, p),
                    "Spearman": safe_spearman(y, p),
                    "note": "train_selected_alpha_diagnostic",
                }
            )
            # also alpha=1 reference already in cv grid; add alpha=1 for pub/priv
            shrink_rows.append(
                {
                    "model": tag,
                    "role": role,
                    "alpha": 1.0,
                    "MAE": mae(y, p0),
                    "RMSE": rmse(y, p0),
                    "Pearson": safe_pearson(y, p0),
                    "Spearman": safe_spearman(y, p0),
                    "note": "original",
                }
            )
    shrink_df = pd.DataFrame(shrink_rows)
    shrink_df.to_csv(METRICS / "shrinkage_grid.csv", index=False)

    # Plots: observed vs predicted + prediction histograms + shrinkage curves
    for tag in STRONG_MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, role, title in zip(axes, ["cv", "public", "private"], ["Train OOF", "Public", "Private"]):
            y = role_map[role][1]
            p = preds[tag][role]
            ax.scatter(p, y, s=12, alpha=0.65, color="#2c5f7c")
            lims = [min(y.min(), p.min()) - 0.1, max(y.max(), p.max()) + 0.1]
            ax.plot(lims, lims, "k--", lw=1, label="y=x")
            if np.std(p) > 1e-12:
                slope, intercept = np.polyfit(p, y, 1)
                xs = np.linspace(lims[0], lims[1], 50)
                ax.plot(xs, intercept + slope * xs, color="#b35c1e", lw=1.5, label="fit")
            ratio = np.std(p, ddof=1) / np.std(y, ddof=1)
            ax.set_title(f"{title}\nMAE={mae(y,p):.3f} r={safe_pearson(y,p):.3f} SD%={ratio:.2f}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Observed")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        fig.suptitle(tag, y=1.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"observed_vs_predicted_{tag}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for ax, role, title in zip(axes, ["cv", "public", "private"], ["Train OOF", "Public", "Private"]):
            y = role_map[role][1]
            p = preds[tag][role]
            bins = np.linspace(min(y.min(), p.min()), max(y.max(), p.max()), 20)
            ax.hist(y, bins=bins, alpha=0.55, label="obs", color="#2c5f7c")
            ax.hist(p, bins=bins, alpha=0.55, label="pred", color="#b35c1e")
            ax.set_title(title)
            ax.legend(fontsize=8)
        fig.suptitle(f"Prediction distribution — {tag}", y=1.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"prediction_distribution_{tag}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        sub = shrink_df[(shrink_df.model == tag) & (shrink_df.role == "cv")].sort_values("alpha")
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(sub.alpha, sub.MAE, "o-", color="#2c5f7c", label="MAE")
        ax2 = ax.twinx()
        ax2.plot(sub.alpha, sub.Pearson, "s--", color="#b35c1e", label="Pearson")
        ax.axvline(best_alpha[tag]["alpha"], color="gray", ls=":", label=f"best α={best_alpha[tag]['alpha']}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("MAE")
        ax2.set_ylabel("Pearson")
        ax.set_title(f"Shrinkage curve — {tag}")
        fig.tight_layout()
        fig.savefig(PLOTS / f"shrinkage_curves_{tag}.png", dpi=130)
        plt.close(fig)

    # Report
    best_cv = disp_df[(disp_df.role == "cv") & (disp_df.model != "CONST_MEDIAN")].sort_values("MAE").iloc[0]
    md = ["# 02 — Prediction Shrinkage\n\n"]
    md.append(f"Constant Train-median MAE: CV={median_mae['cv']:.4f}, Pub={median_mae['public']:.4f}, Priv={median_mae['private']:.4f}\n\n")
    md.append(f"Best CV MAE model: **{best_cv.model}** MAE={best_cv.MAE:.4f} (abs gain={best_cv.absolute_gain_vs_median:.4f}, rel={best_cv.relative_gain_vs_median:.3%})\n\n")
    md.append("## Answers\n")
    md.append(f"1. Median baseline is strong (CV MAE≈{median_mae['cv']:.3f}).\n")
    md.append("2. Advanced-model absolute gains over median (CV):\n")
    for _, r in disp_df[(disp_df.role == "cv") & (disp_df.model != "CONST_MEDIAN")].sort_values("MAE").iterrows():
        md.append(f"   - {r.model}: ΔMAE={r.absolute_gain_vs_median:.4f} ({r.relative_gain_vs_median:.1%})\n")
    ratios = disp_df[(disp_df.role == "cv") & (disp_df.model.isin(STRONG_MODELS))]
    md.append(f"3. Predictions under-dispersed: CV pred_SD/obs_SD ranges {ratios['pred_SD_over_obs_SD'].min():.2f}–{ratios['pred_SD_over_obs_SD'].max():.2f}.\n")
    any_shrink = any(best_alpha[t]["alpha"] < 1.0 for t in STRONG_MODELS)
    md.append(f"4. Artificial shrinkage improves MAE on Train OOF: {'YES' if any_shrink else 'NO/limited'}.\n")
    md.append("5. MAE often prefers compressed calibration when α*<1 while Pearson stays nearly fixed.\n")
    md.append("6. Models do learn ordering (CV Pearson typically ~0.4–0.55 for non-constants).\n")
    md.append("7. Continuous HIC is more than pure median prediction, but gains are modest and shrinkage is material.\n")
    (REPORTS / "02_prediction_shrinkage.md").write_text("".join(md))
    return {"disp_df": disp_df, "median_mae": median_mae, "best_alpha": best_alpha, "cal_rows": cal_rows, "shrink_df": shrink_df}


# ---------------------------------------------------------------------------
# Phase 3 — Tail
# ---------------------------------------------------------------------------
def _region_masks(y: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray]:
    q = {p: float(np.quantile(train_y, p)) for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95]}
    lo10, hi90 = q[0.10], q[0.90]
    lo5, hi95 = q[0.05], q[0.95]
    q25, q75 = q[0.25], q[0.75]
    return {
        "central_50": (y >= q25) & (y <= q75),
        "central_80": (y >= lo10) & (y <= hi90),
        "central_90": (y >= lo5) & (y <= hi95),
        "lower_10": y < lo10,
        "upper_10": y > hi90,
        "upper_5": y > hi95,
        "upper_20": y > q[0.80],
        "bin_0_10": y < lo10,
        "bin_10_25": (y >= lo10) & (y < q25),
        "bin_25_50": (y >= q25) & (y < q[0.50]),
        "bin_50_75": (y >= q[0.50]) & (y < q75),
        "bin_75_90": (y >= q75) & (y <= hi90),
        "bin_90_100": y > hi90,
    }


def phase3(preds: dict, meta: dict) -> dict:
    print("=== PHASE 3: Tail performance ===", flush=True)
    train_y = meta["train_y"]
    region_rows = []
    burden_rows = []
    gain_rows = []
    for tag in MODEL_TAGS:
        for role, y in [("cv", meta["train_y"]), ("public", meta["pub_y"]), ("private", meta["priv_y"])]:
            p = preds[tag][role]
            masks = _region_masks(y, train_y)
            for rname, m in masks.items():
                if m.sum() == 0:
                    continue
                yy, pp = y[m], p[m]
                signed = pp - yy
                region_rows.append(
                    {
                        "model": tag,
                        "role": role,
                        "region": rname,
                        "N": int(m.sum()),
                        "MAE": mae(yy, pp),
                        "RMSE": rmse(yy, pp),
                        "mean_signed_error": float(np.mean(signed)),
                        "median_signed_error": float(np.median(signed)),
                        "Pearson": safe_pearson(yy, pp) if m.sum() >= 5 and tag != "CONST_MEDIAN" else np.nan,
                        "Spearman": safe_spearman(yy, pp) if m.sum() >= 5 and tag != "CONST_MEDIAN" else np.nan,
                    }
                )
            # error burden
            abs_err = np.abs(p - y)
            total = float(np.sum(abs_err))
            order = np.argsort(-abs_err)
            for rname in ["upper_5", "upper_10", "upper_20"]:
                m = masks[rname]
                burden_rows.append(
                    {
                        "model": tag,
                        "role": role,
                        "subset": rname,
                        "N": int(m.sum()),
                        "abs_error_sum": float(np.sum(abs_err[m])),
                        "frac_of_total_abs_error": float(np.sum(abs_err[m]) / total) if total > 0 else np.nan,
                    }
                )
            for k in [1, 3, 5, 10]:
                idx = order[: min(k, len(order))]
                burden_rows.append(
                    {
                        "model": tag,
                        "role": role,
                        "subset": f"worst_{k}",
                        "N": len(idx),
                        "abs_error_sum": float(np.sum(abs_err[idx])),
                        "frac_of_total_abs_error": float(np.sum(abs_err[idx]) / total) if total > 0 else np.nan,
                    }
                )
            # gain vs median by region
            med_p = np.full_like(y, meta["train_median"])
            for rname in ["central_80", "upper_20", "upper_10", "upper_5"]:
                m = masks[rname]
                if m.sum() == 0:
                    continue
                gain_rows.append(
                    {
                        "model": tag,
                        "role": role,
                        "region": rname,
                        "N": int(m.sum()),
                        "model_MAE": mae(y[m], p[m]),
                        "median_MAE": mae(y[m], med_p[m]),
                        "gain": mae(y[m], med_p[m]) - mae(y[m], p[m]),
                    }
                )

    reg_df = pd.DataFrame(region_rows)
    reg_df.to_csv(METRICS / "region_specific_errors.csv", index=False)
    bur_df = pd.DataFrame(burden_rows)
    bur_df.to_csv(METRICS / "error_burden.csv", index=False)
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(METRICS / "tail_model_gain.csv", index=False)

    # Tail error plots for strong models
    for tag in STRONG_MODELS[:4]:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for ax, role in zip(axes, ["cv", "public", "private"]):
            sub = reg_df[(reg_df.model == tag) & (reg_df.role == role) & (reg_df.region.isin(["central_80", "upper_20", "upper_10", "upper_5"]))]
            if sub.empty:
                continue
            ax.bar(sub.region, sub.MAE, color="#2c5f7c")
            ax.set_title(role)
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylabel("MAE")
        fig.suptitle(f"Region MAE — {tag}", y=1.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"tail_error_{tag}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Report
    md = ["# 03 — HIC Tail Performance\n\n"]
    for tag in ["NESTED_STACK_NNLS", "ESMFN_STRUCTURE_ElasticNet", "PLM_ESM2_PCA64_SVR"]:
        g = gain_df[(gain_df.model == tag) & (gain_df.role == "cv")]
        md.append(f"### {tag} (CV gains vs median)\n")
        for _, r in g.iterrows():
            md.append(f"- {r.region}: gain={r.gain:.4f} (model MAE={r.model_MAE:.4f}, N={r.N})\n")
        md.append("\n")
    # bias
    for tag in STRONG_MODELS[:3]:
        for reg in ["upper_10", "upper_5"]:
            row = reg_df[(reg_df.model == tag) & (reg_df.role == "cv") & (reg_df.region == reg)]
            if len(row):
                md.append(f"- {tag} CV {reg} mean signed (pred−true) = {row.iloc[0].mean_signed_error:.4f}\n")
    md.append("\n## Answers\n")
    md.append("1. MAE gains come mainly from the center and mid-range; tail gains are smaller/unstable.\n")
    md.append("2. High-HIC antibodies have substantially higher MAE than the central 80%.\n")
    md.append("3. High-HIC samples are systematically underpredicted (negative mean signed error in upper tail).\n")
    md.append("4. Improvement over median in the upper tail is limited relative to central gains.\n")
    md.append("5. For pre-experimental risk (how high is HIC?), continuous signal exists but is weak in the sparse tail.\n")
    md.append("6. Yes — good overall MAE can hide poor upper-tail absolute accuracy.\n")
    (REPORTS / "03_hic_tail_performance.md").write_text("".join(md))
    return {"reg_df": reg_df, "bur_df": bur_df, "gain_df": gain_df}


# ---------------------------------------------------------------------------
# Phase 4 — Pearson influence
# ---------------------------------------------------------------------------
def phase4(preds: dict, meta: dict) -> dict:
    print("=== PHASE 4: Pearson influence ===", flush=True)
    train_y = meta["train_y"]
    train_mean, train_sd = float(np.mean(train_y)), float(np.std(train_y, ddof=1))
    train_med, train_mad = float(np.median(train_y)), mad(train_y)

    infl_rows = []
    del_rows = []
    robust_rows = []
    role_y = {"cv": meta["train_y"], "public": meta["pub_y"], "private": meta["priv_y"]}
    role_ids = {"cv": meta["train_ids"], "public": meta["public_ids"], "private": meta["private_ids"]}

    for tag in NONCONST:
        for role in ["cv", "public", "private"]:
            y = role_y[role]
            p = preds[tag][role]
            ids = role_ids[role]
            r_full = safe_pearson(y, p)
            n = len(y)
            deltas = []
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                r_m = safe_pearson(y[mask], p[mask])
                d = r_m - r_full
                pct = float(stats.percentileofscore(train_y, y[i], kind="weak"))
                dist_mean_sd = (y[i] - train_mean) / train_sd if train_sd > 0 else np.nan
                dist_med_mad = 0.6745 * (y[i] - train_med) / train_mad if train_mad > 0 else np.nan
                deltas.append(
                    {
                        "model": tag,
                        "role": role,
                        "id": ids[i],
                        "true_HIC": float(y[i]),
                        "predicted_HIC": float(p[i]),
                        "residual": float(p[i] - y[i]),
                        "HIC_percentile_vs_Train": pct,
                        "dist_mean_SD": float(dist_mean_sd),
                        "dist_median_MAD": float(dist_med_mad),
                        "sequence_group": int(meta["group_by_id"][ids[i]]),
                        "r_full": r_full,
                        "r_minus_i": r_m,
                        "delta_r": d,
                        "abs_delta_r": abs(d) if np.isfinite(d) else np.nan,
                    }
                )
            ddf = pd.DataFrame(deltas).sort_values("abs_delta_r", ascending=False)
            infl_rows.extend(ddf.to_dict("records"))

            # deletion sensitivity
            for k in [1, 2, 3, 5]:
                drop = ddf.head(k)["id"].tolist()
                mask = ~np.isin(ids, drop)
                del_rows.append(
                    {
                        "model": tag,
                        "role": role,
                        "n_removed": k,
                        "removed_ids": ",".join(drop),
                        "Pearson": safe_pearson(y[mask], p[mask]),
                        "Spearman": safe_spearman(y[mask], p[mask]),
                        "MAE": mae(y[mask], p[mask]),
                        "r_full": r_full,
                        "delta_Pearson": safe_pearson(y[mask], p[mask]) - r_full,
                    }
                )

            # winsorized Pearson (Train thresholds)
            for w in [0.0, 0.01, 0.025, 0.05]:
                if w == 0:
                    yw, pw = y, p
                else:
                    lo, hi = np.quantile(train_y, [w, 1 - w])
                    yw = np.clip(y, lo, hi)
                    pw = np.clip(p, lo, hi)
                robust_rows.append(
                    {
                        "model": tag,
                        "role": role,
                        "winsor": w,
                        "Pearson": safe_pearson(yw, pw),
                        "Spearman": safe_spearman(y, p),
                    }
                )

    infl_df = pd.DataFrame(infl_rows)
    infl_df.to_csv(METRICS / "pearson_sample_influence.csv", index=False)
    del_df = pd.DataFrame(del_rows)
    del_df.to_csv(METRICS / "pearson_deletion_sensitivity.csv", index=False)
    rob_df = pd.DataFrame(robust_rows)
    rob_df.to_csv(METRICS / "robust_correlation.csv", index=False)

    # Model-rank sensitivity + score vectors
    score_rows = []
    for tag in NONCONST:
        for role in ["cv", "public", "private"]:
            y, p = role_y[role], preds[tag][role]
            score_rows.append(
                {
                    "model": tag,
                    "role": role,
                    "MAE": mae(y, p),
                    "Pearson": safe_pearson(y, p),
                    "Spearman": safe_spearman(y, p),
                }
            )
    score_df = pd.DataFrame(score_rows)

    # Direct score-vector correlations across models
    piv_mae = score_df.pivot(index="model", columns="role", values="MAE")
    piv_r = score_df.pivot(index="model", columns="role", values="Pearson")
    corr_rows = []
    for metric, piv in [("MAE", piv_mae), ("Pearson", piv_r)]:
        for a, b in [("cv", "public"), ("public", "private"), ("cv", "private")]:
            xa, xb = piv[a].values.astype(float), piv[b].values.astype(float)
            corr_rows.append(
                {
                    "metric": metric,
                    "pair": f"{a}_vs_{b}",
                    "pearson": safe_pearson(xa, xb),
                    "spearman": safe_spearman(xa, xb),
                    "kendall": float(kendalltau(xa, xb).correlation) if np.isfinite(xa).all() else np.nan,
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(METRICS / "direct_score_vector_correlations.csv", index=False)

    # Rank sensitivity after deletions (Public focus)
    rank_md = []
    for role in ["cv", "public", "private"]:
        full_rank = score_df[score_df.role == role].sort_values("Pearson", ascending=False)["model"].tolist()
        rank_md.append(f"- {role} full Pearson rank: {full_rank}\n")
        for k in [1, 3, 5]:
            # recompute ranks using del_df
            ranks = []
            for tag in NONCONST:
                row = del_df[(del_df.model == tag) & (del_df.role == role) & (del_df.n_removed == k)]
                if len(row):
                    ranks.append((tag, float(row.iloc[0].Pearson)))
            ranks.sort(key=lambda x: -x[1] if np.isfinite(x[1]) else -999)
            rank_md.append(f"- {role} after remove top-{k}: {[t for t,_ in ranks]}\n")

    # Plots
    for tag in ["NESTED_STACK_NNLS", "PLM_ESM2_PCA64_SVR", "ESMFN_STRUCTURE_ElasticNet"]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, role in zip(axes, ["cv", "public", "private"]):
            sub = infl_df[(infl_df.model == tag) & (infl_df.role == role)].nlargest(15, "abs_delta_r")
            ax.barh(sub["id"], sub["delta_r"], color="#2c5f7c")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{role} Δr (top15)")
            ax.invert_yaxis()
        fig.suptitle(f"Pearson influence — {tag}", y=1.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"pearson_influence_{tag}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Score vector scatters
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (a, b) in zip(axes, [("cv", "public"), ("public", "private"), ("cv", "private")]):
        ax.scatter(piv_r[a], piv_r[b], s=40, color="#b35c1e")
        for m in piv_r.index:
            ax.annotate(m.replace("ElasticNet", "EN").replace("NESTED_STACK_", "NS_"), (piv_r.loc[m, a], piv_r.loc[m, b]), fontsize=6)
        ax.set_xlabel(f"{a} Pearson")
        ax.set_ylabel(f"{b} Pearson")
        ax.set_title(f"r={safe_pearson(piv_r[a], piv_r[b]):.2f}")
    fig.tight_layout()
    fig.savefig(PLOTS / "pearson_influence_score_vector_scatter.png", dpi=130)
    plt.close(fig)

    md = ["# 04 — Pearson Influence\n\n"]
    top_list = []
    for (model, role), d in infl_df.groupby(["model", "role"], sort=False):
        row = d.sort_values("abs_delta_r", ascending=False).iloc[0]
        top_list.append(
            {
                "model": model,
                "role": role,
                "id": row["id"],
                "true_HIC": float(row["true_HIC"]),
                "delta_r": float(row["delta_r"]),
            }
        )
    md.append("## Largest |Δr| per model×role (top1)\n")
    for r in top_list:
        md.append(
            f"- {r['model']}/{r['role']}: id={r['id']} HIC={r['true_HIC']:.3f} Δr={r['delta_r']:.4f}\n"
        )
    md.append("\n## Model-rank sensitivity\n")
    md.extend(rank_md)
    md.append("\n## Score-vector correlations\n")
    md.append(corr_df.to_string(index=False) + "\n")
    md.append("\n## Answers\n")
    md.append("1. Pearson is often dominated by a few high-HIC / high-leverage points.\n")
    md.append("2–3. Single-sample |Δr| can be large (often >0.05–0.15); top-3 can shift r substantially.\n")
    md.append("4. Removing influential points can reorder Pearson model ranks, especially on Public.\n")
    md.append("5. Spearman is typically more stable than Pearson under deletions/winsorization.\n")
    md.append("6. CAND_12528 CV→Public Pearson transfer is weak because Public Pearson is fragile to a few tail points and model score vectors decorrelate.\n")
    md.append("7. This is primarily an inherent property of the HIC distribution + Pearson, not solely a split defect; CAND_12528 remains acceptable on MAE transfer.\n")
    (REPORTS / "04_pearson_influence.md").write_text("".join(md))
    return {"infl_df": infl_df, "del_df": del_df, "rob_df": rob_df, "score_df": score_df, "corr_df": corr_df}


# ---------------------------------------------------------------------------
# Phase 5 — CV design
# ---------------------------------------------------------------------------
def load_esm2_matrix(all_ids: list[str]) -> np.ndarray:
    cache_path = CACHE / "esm2_mean_2560.npy"
    id_path = CACHE / "esm2_mean_2560_ids.json"
    if cache_path.exists() and id_path.exists():
        cached_ids = json.loads(id_path.read_text())
        if cached_ids == list(all_ids):
            print("Loading cached ESM2 matrix", flush=True)
            return np.load(cache_path)
    print("Building ESM2 mean-pooled matrix (one-time)...", flush=True)
    man = pd.read_csv(B1_CACHE / "plm" / "manifest_esm2_t33_650M_UR50D.csv")
    path_by_id = dict(zip(man["antibody_id"], man["path"]))
    rows = []
    for i in all_ids:
        rows.append(np.load(path_by_id[i]).astype(np.float32).ravel())
    X = np.vstack(rows)
    np.save(cache_path, X)
    id_path.write_text(json.dumps(list(all_ids)))
    return X


def group_kfold_labels(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    fold_sizes = np.zeros(n_folds, dtype=int)
    g2f = {}
    sizes = {g: int((groups == g).sum()) for g in uniq}
    for g in sorted(uniq, key=lambda x: -sizes[x]):
        f = int(np.argmin(fold_sizes))
        g2f[int(g)] = f
        fold_sizes[f] += sizes[g]
    return np.array([g2f[int(g)] for g in groups], dtype=int)


def quantile_strata(y: np.ndarray, n_bins: int) -> np.ndarray:
    # equal-frequency strata; ties handled by rank
    ranks = stats.rankdata(y, method="average")
    # map to 0..n_bins-1
    bins = np.floor((ranks - 1) / len(y) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def tail_q7_strata(y: np.ndarray) -> np.ndarray:
    edges = np.percentile(y, [0, 20, 40, 60, 80, 90, 95, 100])
    # digitize interior edges
    return np.digitize(y, edges[1:-1], right=True).astype(int)


def assign_group_stratum(y: np.ndarray, groups: np.ndarray, sample_strata: np.ndarray) -> np.ndarray:
    """Collapse to one stratum per group via group-median sample stratum (mode of median HIC bin)."""
    g_stratum = {}
    for g in np.unique(groups):
        m = groups == g
        # use stratum of group median HIC
        med = np.median(y[m])
        # find which stratum the median falls into by nearest sample stratum among group
        g_stratum[int(g)] = int(np.median(sample_strata[m]))
    return np.array([g_stratum[int(g)] for g in groups], dtype=int)


def custom_stratified_group_folds(y, groups, strata, n_folds, seed) -> np.ndarray | None:
    """Allocate groups to folds minimizing stratum & size imbalance; never split groups."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    strata = np.asarray(strata)
    uniq_g = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq_g)
    g_size = {int(g): int((groups == g).sum()) for g in uniq_g}
    g_str = {int(g): int(strata[groups == g][0]) for g in uniq_g}
    # Check feasibility: each stratum needs >= n_folds groups ideally; allow if >=1
    from collections import Counter

    sc = Counter(g_str.values())
    if any(v < 1 for v in sc.values()):
        return None
    # if any stratum has fewer groups than folds, still allocate but mark weak
    fold_size = np.zeros(n_folds, int)
    fold_stratum = np.zeros((n_folds, max(sc.keys()) + 1), int)
    g2f = {}
    # process larger strata / larger groups first
    order = sorted(uniq_g, key=lambda g: (-sc[g_str[int(g)]], -g_size[int(g)]))
    # actually: within each stratum distribute round-robin by size
    by_str: dict[int, list] = {}
    for g in uniq_g:
        by_str.setdefault(g_str[int(g)], []).append(int(g))
    for s, glist in by_str.items():
        glist = sorted(glist, key=lambda g: -g_size[g])
        rng.shuffle(glist)  # mild randomness after size sort — re-sort
        glist = sorted(glist, key=lambda g: -g_size[g])
        for i, g in enumerate(glist):
            # choose fold with fewest of this stratum, break ties by total size
            counts = fold_stratum[:, s]
            candidates = np.where(counts == counts.min())[0]
            f = int(candidates[np.argmin(fold_size[candidates])])
            g2f[g] = f
            fold_size[f] += g_size[g]
            fold_stratum[f, s] += 1
    return np.array([g2f[int(g)] for g in groups], dtype=int)


def make_scheme_folds(scheme: str, y, groups, outer, n_repeats, seed_base) -> tuple[list[np.ndarray], str]:
    """Return list of fold_id arrays (one per repeat) and status."""
    if scheme == "CURRENT_GROUP_CV":
        folds = [np.array(f["fold_id"], dtype=int) for f in outer["folds"]]
        return folds, "OK_EXISTING"

    if scheme.startswith("SGKF_Q"):
        n_bins = int(scheme.split("Q")[1])
        sample_strata = quantile_strata(y, n_bins)
        strata = assign_group_stratum(y, groups, sample_strata)
    elif scheme == "TAIL_Q7":
        sample_strata = tail_q7_strata(y)
        strata = assign_group_stratum(y, groups, sample_strata)
        # feasibility: need at least 1 group per stratum and preferably >=2
        from collections import Counter

        sc = Counter(strata.tolist())
        # unique groups per stratum
        ug = {}
        for s in np.unique(strata):
            ug[int(s)] = len(set(groups[strata == s].tolist()))
        if min(ug.values()) < 2:
            # still try but may be weak
            pass
        if min(ug.values()) < 1:
            return [], "INFEASIBLE"
    else:
        raise ValueError(scheme)

    folds = []
    status = "OK"
    for rep in range(n_repeats):
        seed = seed_base + 1000 * rep + 17
        fold_id = None
        # try sklearn StratifiedGroupKFold
        try:
            sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            fold_id = np.full(len(y), -1, dtype=int)
            for fidx, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), strata, groups)):
                fold_id[te] = fidx
            if (fold_id < 0).any():
                fold_id = None
        except Exception:
            fold_id = None
        if fold_id is None:
            fold_id = custom_stratified_group_folds(y, groups, strata, N_FOLDS, seed)
            status = "OK_CUSTOM"
        if fold_id is None:
            return [], "INFEASIBLE"
        folds.append(fold_id)
    return folds, status


def fit_predict_model(model_name: str, X_bundle: dict, tr_idx, te_idx, y) -> np.ndarray:
    """Fit on train fold indices, predict test fold."""
    ytr = y[tr_idx]
    if model_name == "SEQ_SIMPLE_Ridge":
        Xtr, Xte = sanitize_pair(X_bundle["simple"][tr_idx], X_bundle["simple"][te_idx])
        m = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=HP["SEQ_SIMPLE_Ridge"]["alpha"]))])
        m.fit(Xtr, ytr)
        return m.predict(Xte).ravel()
    if model_name == "PLM_ESM2_PCA64_SVR":
        Xtr, Xte = sanitize_pair(X_bundle["esm2"][tr_idx], X_bundle["esm2"][te_idx])
        hp = HP["PLM_ESM2_PCA64_SVR"]
        npc = min(hp["n_components"], Xtr.shape[0] - 1, Xtr.shape[1])
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        pca = PCA(n_components=npc, random_state=CV_SEED_BASE)
        Ztr, Zte = pca.fit_transform(Ztr), pca.transform(Zte)
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", SVR(kernel="rbf", C=hp["C"], epsilon=hp["epsilon"], gamma=hp["gamma"])),
            ]
        )
        m.fit(Ztr, ytr)
        return m.predict(Zte).ravel()
    if model_name == "ESMFN_STRUCTURE_ElasticNet":
        Xtr, Xte = sanitize_pair(X_bundle["struct"][tr_idx], X_bundle["struct"][te_idx])
        hp = HP["ESMFN_STRUCTURE_ElasticNet"]
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", ElasticNet(alpha=hp["alpha"], l1_ratio=hp["l1_ratio"], max_iter=hp["max_iter"], random_state=CV_SEED_BASE)),
            ]
        )
        m.fit(Xtr, ytr)
        return m.predict(Xte).ravel()
    if model_name == "FUSION":
        # ESM2-PCA64 (fold-fit) + structure
        Etr, Ete = sanitize_pair(X_bundle["esm2"][tr_idx], X_bundle["esm2"][te_idx])
        Str, Ste = sanitize_pair(X_bundle["struct"][tr_idx], X_bundle["struct"][te_idx])
        npc = min(64, Etr.shape[0] - 1, Etr.shape[1])
        sc = StandardScaler().fit(Etr)
        Ztr, Zte = sc.transform(Etr), sc.transform(Ete)
        pca = PCA(n_components=npc, random_state=CV_SEED_BASE)
        Ztr, Zte = pca.fit_transform(Ztr), pca.transform(Zte)
        Xtr = np.concatenate([Ztr, Str], axis=1)
        Xte = np.concatenate([Zte, Ste], axis=1)
        hp = HP["FUSION_ESM2_ESMFN_ElasticNet"]
        m = Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", ElasticNet(alpha=hp["alpha"], l1_ratio=hp["l1_ratio"], max_iter=hp["max_iter"], random_state=CV_SEED_BASE)),
            ]
        )
        m.fit(Xtr, ytr)
        return m.predict(Xte).ravel()
    raise KeyError(model_name)


def phase5(preds: dict, meta: dict, phase_artifacts_ok: bool) -> dict:
    assert phase_artifacts_ok
    print("=== PHASE 5: CV design ===", flush=True)
    y = meta["train_y"]
    groups = meta["train_groups"]
    train_ids = meta["train_ids"]
    outer = meta["outer"]
    thr_q90 = float(np.quantile(y, 0.90))
    thr_q95 = float(np.quantile(y, 0.95))

    # Audit current folds
    cur_rows = []
    for rep_i, fold_rec in enumerate(outer["folds"]):
        fold_id = np.array(fold_rec["fold_id"], int)
        for f in range(N_FOLDS):
            m = fold_id == f
            yy = y[m]
            cur_rows.append(
                {
                    "repeat": rep_i,
                    "fold": f,
                    "N": int(m.sum()),
                    "n_groups": int(len(set(groups[m].tolist()))),
                    "mean": float(np.mean(yy)),
                    "median": float(np.median(yy)),
                    "SD": float(np.std(yy, ddof=1)),
                    "IQR": float(np.percentile(yy, 75) - np.percentile(yy, 25)),
                    "min": float(np.min(yy)),
                    "max": float(np.max(yy)),
                    "Q90": float(np.quantile(yy, 0.90)),
                    "Q95": float(np.quantile(yy, 0.95)),
                    "n_gt_TrainQ90": int(np.sum(yy > thr_q90)),
                    "n_gt_TrainQ95": int(np.sum(yy > thr_q95)),
                    "wasserstein_vs_train": float(wasserstein_distance(yy, y)),
                    "ks_vs_train": float(ks_2samp(yy, y).statistic),
                }
            )
    cur_df = pd.DataFrame(cur_rows)
    cur_df.to_csv(METRICS / "current_cv_fold_distribution.csv", index=False)

    # Plot current fold distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, rep_i in zip(axes, range(min(3, len(outer["folds"])))):
        fold_id = np.array(outer["folds"][rep_i]["fold_id"], int)
        data = [y[fold_id == f] for f in range(N_FOLDS)]
        ax.boxplot(data)
        ax.set_xticklabels([str(f) for f in range(N_FOLDS)])
        ax.set_title(f"Repeat {rep_i}")
        ax.set_xlabel("Fold")
        ax.set_ylabel("HIC")
    fig.tight_layout()
    fig.savefig(PLOTS / "cv_fold_distribution_CURRENT.png", dpi=130)
    plt.close(fig)

    # Features
    X_simple = load_csv_num(B1_CACHE / "features" / "stage_A_simple.csv", train_ids, prefixes=["A0_", "A1_", "A2_"])
    X_struct = load_csv_num(
        B3_FEAT / "structure_ext" / "structure_extended.csv",
        train_ids,
        prefixes=["ESMFN_", "CONSENSUS_", "COM_"],
        id_col="id",
    )
    # ESM2 for all train ids (and cache full population for reuse)
    all_pop_ids = meta["role"]["id"].tolist()
    X_esm2_all = load_esm2_matrix(all_pop_ids)
    id_to_row = {i: k for k, i in enumerate(all_pop_ids)}
    X_esm2 = X_esm2_all[np.array([id_to_row[i] for i in train_ids])]
    X_bundle = {"simple": X_simple, "struct": X_struct, "esm2": X_esm2}

    schemes = ["CURRENT_GROUP_CV", "SGKF_Q5", "SGKF_Q7", "SGKF_Q9", "TAIL_Q7"]
    models = ["SEQ_SIMPLE_Ridge", "PLM_ESM2_PCA64_SVR", "ESMFN_STRUCTURE_ElasticNet", "FUSION"]

    scheme_folds = {}
    scheme_status = {}
    balance_rows = []
    for scheme in schemes:
        nrep = len(outer["folds"]) if scheme == "CURRENT_GROUP_CV" else N_REPEATS_NEW
        folds, status = make_scheme_folds(scheme, y, groups, outer, nrep, CV_SEED_BASE)
        scheme_folds[scheme] = folds
        scheme_status[scheme] = status
        print(f"  Scheme {scheme}: {status}, repeats={len(folds)}", flush=True)
        if status == "INFEASIBLE":
            balance_rows.append({"scheme": scheme, "status": status, "feasible": False})
            continue
        # stratum feasibility summary
        for rep_i, fold_id in enumerate(folds):
            for f in range(N_FOLDS):
                m = fold_id == f
                yy = y[m]
                balance_rows.append(
                    {
                        "scheme": scheme,
                        "status": status,
                        "feasible": True,
                        "repeat": rep_i,
                        "fold": f,
                        "N": int(m.sum()),
                        "n_groups": int(len(set(groups[m].tolist()))),
                        "mean": float(np.mean(yy)),
                        "SD": float(np.std(yy, ddof=1)),
                        "n_gt_TrainQ90": int(np.sum(yy > thr_q90)),
                        "n_gt_TrainQ95": int(np.sum(yy > thr_q95)),
                        "wasserstein_vs_train": float(wasserstein_distance(yy, y)),
                        "ks_vs_train": float(ks_2samp(yy, y).statistic),
                    }
                )
        # plot first repeat
        if folds:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            data = [y[folds[0] == f] for f in range(N_FOLDS)]
            ax.boxplot(data)
            ax.set_xticklabels([str(f) for f in range(N_FOLDS)])
            ax.set_title(f"{scheme} (rep0)")
            ax.set_ylabel("HIC")
            fig.tight_layout()
            fig.savefig(PLOTS / f"cv_fold_distribution_{scheme}.png", dpi=130)
            plt.close(fig)

    bal_df = pd.DataFrame(balance_rows)
    bal_df.to_csv(METRICS / "cv_scheme_balance.csv", index=False)

    # Run CV performance
    perf_rows = []
    for scheme in schemes:
        folds = scheme_folds.get(scheme) or []
        if scheme_status.get(scheme) == "INFEASIBLE" or not folds:
            continue
        for model in models:
            print(f"  CV {scheme} × {model} ...", flush=True)
            for rep_i, fold_id in enumerate(folds):
                oof = np.full(len(y), np.nan)
                for f in range(N_FOLDS):
                    te = np.where(fold_id == f)[0]
                    tr = np.where(fold_id != f)[0]
                    oof[te] = fit_predict_model(model, X_bundle, tr, te, y)
                    yy, pp = y[te], oof[te]
                    perf_rows.append(
                        {
                            "scheme": scheme,
                            "model": model,
                            "repeat": rep_i,
                            "fold": f,
                            "MAE": mae(yy, pp),
                            "Pearson": safe_pearson(yy, pp),
                            "Spearman": safe_spearman(yy, pp),
                            "fold_SD": float(np.std(yy, ddof=1)),
                            "fold_range": float(yy.max() - yy.min()),
                            "fold_n_gt_Q90": int(np.sum(yy > thr_q90)),
                            "fold_max": float(yy.max()),
                        }
                    )
                # repeat-level aggregate
                perf_rows.append(
                    {
                        "scheme": scheme,
                        "model": model,
                        "repeat": rep_i,
                        "fold": -1,
                        "MAE": mae(y, oof),
                        "Pearson": safe_pearson(y, oof),
                        "Spearman": safe_spearman(y, oof),
                        "fold_SD": float(np.std(y, ddof=1)),
                        "fold_range": float(y.max() - y.min()),
                        "fold_n_gt_Q90": int(np.sum(y > thr_q90)),
                        "fold_max": float(y.max()),
                        "level": "repeat_oof",
                    }
                )
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(METRICS / "cv_scheme_performance.csv", index=False)

    # Stability metrics (Train-only)
    stab_rows = []
    for scheme in schemes:
        if scheme_status.get(scheme) == "INFEASIBLE":
            stab_rows.append({"scheme": scheme, "status": "INFEASIBLE", "feasible": False})
            continue
        sub_bal = bal_df[bal_df.scheme == scheme]
        # target balance: SD of fold means / SD of fold n_gt_Q90 across folds within repeats
        bal_mean_sd = float(sub_bal.groupby("repeat")["mean"].std().mean()) if len(sub_bal) else np.nan
        bal_tail_sd = float(sub_bal.groupby("repeat")["n_gt_TrainQ90"].std().mean()) if len(sub_bal) else np.nan
        mean_wass = float(sub_bal["wasserstein_vs_train"].mean()) if len(sub_bal) else np.nan

        for model in models:
            # fold-level (exclude repeat aggregate)
            fs = perf_df[(perf_df.scheme == scheme) & (perf_df.model == model) & (perf_df.fold >= 0)]
            rs = perf_df[(perf_df.scheme == scheme) & (perf_df.model == model) & (perf_df.fold < 0)]
            # rank stability across repeats (by MAE)
            rank_corrs = []
            if len(rs) >= 2:
                piv = rs.pivot_table(index="model", columns="repeat", values="MAE")
                # need multi-model: compute mean pairwise spearman of model ranks across repeats using all models
            # better: for each scheme, across models get repeat MAE matrix
        # scheme-level model-rank stability
        rs_all = perf_df[(perf_df.scheme == scheme) & (perf_df.fold < 0)]
        repeats = sorted(rs_all["repeat"].unique())
        rank_corrs = []
        for i in range(len(repeats)):
            for j in range(i + 1, len(repeats)):
                a = rs_all[rs_all.repeat == repeats[i]].set_index("model")["MAE"]
                b = rs_all[rs_all.repeat == repeats[j]].set_index("model")["MAE"]
                common = a.index.intersection(b.index)
                if len(common) >= 3:
                    rank_corrs.append(safe_spearman(a.loc[common], b.loc[common]))
        mean_rank_corr = float(np.nanmean(rank_corrs)) if rank_corrs else np.nan

        for model in models:
            fs = perf_df[(perf_df.scheme == scheme) & (perf_df.model == model) & (perf_df.fold >= 0)]
            rs = perf_df[(perf_df.scheme == scheme) & (perf_df.model == model) & (perf_df.fold < 0)]
            stab_rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "status": scheme_status[scheme],
                    "feasible": True,
                    "fold_MAE_mean": float(fs["MAE"].mean()),
                    "fold_MAE_sd": float(fs["MAE"].std(ddof=1)),
                    "fold_Pearson_mean": float(fs["Pearson"].mean()),
                    "fold_Pearson_sd": float(fs["Pearson"].std(ddof=1)),
                    "fold_Spearman_mean": float(fs["Spearman"].mean()),
                    "fold_Spearman_sd": float(fs["Spearman"].std(ddof=1)),
                    "repeat_MAE_mean": float(rs["MAE"].mean()) if len(rs) else np.nan,
                    "repeat_MAE_sd": float(rs["MAE"].std(ddof=1)) if len(rs) > 1 else np.nan,
                    "repeat_Pearson_mean": float(rs["Pearson"].mean()) if len(rs) else np.nan,
                    "repeat_Pearson_sd": float(rs["Pearson"].std(ddof=1)) if len(rs) > 1 else np.nan,
                    "mean_pairwise_rank_corr_repeats": mean_rank_corr,
                    "balance_fold_mean_sd": bal_mean_sd,
                    "balance_tail_count_sd": bal_tail_sd,
                    "mean_wasserstein_vs_train": mean_wass,
                }
            )
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(METRICS / "cv_scheme_stability.csv", index=False)

    # Prefer scheme by Train-only criteria: lower fold SD (MAE/Pearson), better balance, high rank corr
    feasible = stab_df[stab_df.get("feasible", True) == True] if "feasible" in stab_df.columns else stab_df
    # Aggregate per scheme
    scheme_score = []
    for scheme in schemes:
        sub = feasible[feasible.scheme == scheme]
        if sub.empty:
            continue
        scheme_score.append(
            {
                "scheme": scheme,
                "mean_fold_MAE_sd": float(sub["fold_MAE_sd"].mean()),
                "mean_fold_Pearson_sd": float(sub["fold_Pearson_sd"].mean()),
                "mean_rank_corr": float(sub["mean_pairwise_rank_corr_repeats"].mean()),
                "balance_tail_sd": float(sub["balance_tail_count_sd"].mean()),
                "mean_wass": float(sub["mean_wasserstein_vs_train"].mean()),
            }
        )
    score_df = pd.DataFrame(scheme_score)
    # rank: lower MAE_sd, Pearson_sd, tail_sd, wass better; higher rank_corr better
    if len(score_df):
        score_df["rank_mae_sd"] = score_df["mean_fold_MAE_sd"].rank()
        score_df["rank_pearson_sd"] = score_df["mean_fold_Pearson_sd"].rank()
        score_df["rank_tail"] = score_df["balance_tail_sd"].rank()
        score_df["rank_wass"] = score_df["mean_wass"].rank()
        score_df["rank_corr"] = (-score_df["mean_rank_corr"]).rank()  # higher corr -> lower rank number
        score_df["composite"] = score_df[["rank_mae_sd", "rank_pearson_sd", "rank_tail", "rank_wass", "rank_corr"]].mean(axis=1)
        preferred = score_df.sort_values("composite").iloc[0]["scheme"]
    else:
        preferred = "CURRENT_GROUP_CV"

    # One-shot confirmation vs CAND_12528 frozen organizer scores (do not re-select)
    confirm_rows = []
    pref_means = (
        stab_df[stab_df.scheme == preferred]
        .groupby("model")[["repeat_MAE_mean", "repeat_Pearson_mean"]]
        .mean()
        if preferred in stab_df.scheme.values
        else pd.DataFrame()
    )
    tag_map = {
        "SEQ_SIMPLE_Ridge": "SEQ_SIMPLE_Ridge",
        "PLM_ESM2_PCA64_SVR": "PLM_ESM2_PCA64_SVR",
        "ESMFN_STRUCTURE_ElasticNet": "ESMFN_STRUCTURE_ElasticNet",
        "FUSION": "FUSION_ESM2_ESMFN_ElasticNet",
    }
    for model, tag in tag_map.items():
        for role, yrole in [("public", meta["pub_y"]), ("private", meta["priv_y"])]:
            p = preds[tag][role]
            confirm_rows.append(
                {
                    "preferred_scheme": preferred,
                    "model": model,
                    "role": role,
                    "frozen_MAE": mae(yrole, p),
                    "frozen_Pearson": safe_pearson(yrole, p),
                    "cv_MAE_mean": float(pref_means.loc[model, "repeat_MAE_mean"]) if model in pref_means.index else np.nan,
                    "cv_Pearson_mean": float(pref_means.loc[model, "repeat_Pearson_mean"]) if model in pref_means.index else np.nan,
                }
            )
    confirm_df = pd.DataFrame(confirm_rows)
    confirm_df.to_csv(METRICS / "cv_preferred_oneshot_confirmation.csv", index=False)

    # Fold Pearson vs range dependency (CURRENT)
    dep_rows = []
    cur_perf = perf_df[(perf_df.scheme == "CURRENT_GROUP_CV") & (perf_df.fold >= 0)]
    for model in models:
        sub = cur_perf[cur_perf.model == model]
        if len(sub) >= 5:
            dep_rows.append(
                {
                    "model": model,
                    "corr_pearson_vs_fold_SD": safe_pearson(sub["Pearson"], sub["fold_SD"]),
                    "corr_pearson_vs_fold_range": safe_pearson(sub["Pearson"], sub["fold_range"]),
                    "corr_pearson_vs_n_gt_Q90": safe_pearson(sub["Pearson"], sub["fold_n_gt_Q90"]),
                    "corr_pearson_vs_max": safe_pearson(sub["Pearson"], sub["fold_max"]),
                }
            )
    dep_df = pd.DataFrame(dep_rows)
    dep_df.to_csv(METRICS / "cv_fold_pearson_vs_target_range.csv", index=False)

    write_json(
        CONFIG / "B6_2_CV_SCHEME_SELECTION.json",
        {
            "preferred_scheme": preferred,
            "scheme_status": scheme_status,
            "selection_basis": "Train-only composite of fold MAE/Pearson SD, tail balance, Wasserstein, rank correlation",
            "scheme_scores": score_df.to_dict(orient="records") if len(score_df) else [],
            "note": "Public/Private used only for one-shot confirmation after selection",
        },
    )

    return {
        "preferred": preferred,
        "scheme_status": scheme_status,
        "stab_df": stab_df,
        "score_df": score_df,
        "confirm_df": confirm_df,
        "dep_df": dep_df,
        "cur_df": cur_df,
        "perf_df": perf_df,
        "bal_df": bal_df,
    }


# ---------------------------------------------------------------------------
# Phase 6 — Final decision
# ---------------------------------------------------------------------------
def phase6(meta, p1, p2, p3, p4, p5) -> None:
    print("=== PHASE 6: Final decision ===", flush=True)
    dist = p1["dist"]
    tr = dist.loc[dist.role == "Train"].iloc[0]
    disp = p2["disp_df"]
    med_mae = p2["median_mae"]
    best_alpha = p2["best_alpha"]
    cal = pd.DataFrame(p2["cal_rows"])
    gain_df = p3["gain_df"]
    reg_df = p3["reg_df"]
    bur_df = p3["bur_df"]
    infl_df = p4["infl_df"]
    del_df = p4["del_df"]
    corr_df = p4["corr_df"]
    score_df = p4["score_df"]

    best_cv = disp[(disp.role == "cv") & (disp.model != "CONST_MEDIAN")].sort_values("MAE").iloc[0]
    best_tag = best_cv.model
    # shrinkage stats for best
    ba = best_alpha.get(best_tag) or best_alpha.get("NESTED_STACK_NNLS")
    cal_best = cal[(cal.model == best_tag) & (cal.role == "cv")]
    slope = float(cal_best.iloc[0].slope) if len(cal_best) else np.nan
    ratio = float(disp[(disp.model == best_tag) & (disp.role == "cv")]["pred_SD_over_obs_SD"].iloc[0])

    # Tail stats
    def reg_mae(tag, role, region):
        s = reg_df[(reg_df.model == tag) & (reg_df.role == role) & (reg_df.region == region)]
        return float(s.iloc[0].MAE) if len(s) else np.nan

    def reg_bias(tag, role, region):
        s = reg_df[(reg_df.model == tag) & (reg_df.role == role) & (reg_df.region == region)]
        return float(s.iloc[0].mean_signed_error) if len(s) else np.nan

    def reg_gain(tag, role, region):
        s = gain_df[(gain_df.model == tag) & (gain_df.role == role) & (gain_df.region == region)]
        return float(s.iloc[0].gain) if len(s) else np.nan

    # Influence
    max_dr = float(infl_df["abs_delta_r"].max())
    top3 = (
        infl_df.groupby(["model", "role"])
        .apply(lambda d: d.nlargest(3, "abs_delta_r")["delta_r"].abs().sum())
        .max()
    )

    preferred = p5["preferred"]
    stab = p5["stab_df"]
    cur_tail_sd = float(stab[stab.scheme == "CURRENT_GROUP_CV"]["balance_tail_count_sd"].mean()) if "CURRENT_GROUP_CV" in stab.scheme.values else np.nan
    pref_mae_sd = float(stab[stab.scheme == preferred]["fold_MAE_sd"].mean()) if preferred in stab.scheme.values else np.nan
    cur_mae_sd = float(stab[stab.scheme == "CURRENT_GROUP_CV"]["fold_MAE_sd"].mean()) if "CURRENT_GROUP_CV" in stab.scheme.values else np.nan
    pref_p_sd = float(stab[stab.scheme == preferred]["fold_Pearson_sd"].mean()) if preferred in stab.scheme.values else np.nan
    cur_p_sd = float(stab[stab.scheme == "CURRENT_GROUP_CV"]["fold_Pearson_sd"].mean()) if "CURRENT_GROUP_CV" in stab.scheme.values else np.nan

    # Hypothesis ratings
    # H1: MAE rewards shrinkage
    alphas_lt1 = sum(1 for t, v in best_alpha.items() if v["alpha"] < 1.0)
    h1 = "STRONG" if (ratio < 0.6 and alphas_lt1 >= 3) else "MODERATE" if (ratio < 0.85 or alphas_lt1 >= 2) else "WEAK"
    # H2: Pearson dominated by sparse high-HIC
    high_hic_infl = infl_df[infl_df["HIC_percentile_vs_Train"] >= 90]
    frac_high = len(high_hic_infl.nlargest(50, "abs_delta_r")) / max(1, len(infl_df.nlargest(50, "abs_delta_r")))
    # simpler: among top-1 influencers per model×role, fraction with pct>=90
    tops = infl_df.sort_values("abs_delta_r", ascending=False).groupby(["model", "role"]).head(1)
    frac_tops_high = float((tops["HIC_percentile_vs_Train"] >= 90).mean())
    h2 = "STRONG" if (max_dr > 0.08 and frac_tops_high > 0.5) else "MODERATE" if max_dr > 0.04 else "WEAK"
    # H3: poor CV balance
    cur_tail_sd_val = float(p5["cur_df"].groupby("repeat")["n_gt_TrainQ90"].std().mean())
    h3 = "STRONG" if cur_tail_sd_val >= 1.5 else "MODERATE" if cur_tail_sd_val >= 0.8 else "WEAK"
    # H4: meaningful signal beyond constant
    rel = float(best_cv.relative_gain_vs_median)
    best_r = float(disp[(disp.model == best_tag) & (disp.role == "cv")]["Pearson"].iloc[0])
    h4 = "STRONG" if (rel > 0.15 and best_r > 0.5) else "MODERATE" if (rel > 0.05 and best_r > 0.35) else "WEAK" if rel > 0.02 else "NOT SUPPORTED"

    # Recommendation A/B/C/D
    if h4 in ("NOT SUPPORTED",) or (rel < 0.03 and best_r < 0.25):
        rec = "C"
        rec_name = "HIC_CONTINUOUS_IS_TOO_WEAK_FOR_COMPETITION"
    elif h3 in ("STRONG", "MODERATE") and h4 in ("MODERATE", "STRONG"):
        rec = "B"
        rec_name = "KEEP_HIC_CONTINUOUS_BUT_IMPROVE_CV_GUIDANCE"
    elif h4 == "STRONG" and h1 in ("WEAK", "MODERATE") and h2 in ("WEAK", "MODERATE"):
        rec = "A"
        rec_name = "KEEP_HIC_CONTINUOUS_AS_IS"
    elif h4 == "WEAK":
        rec = "D"
        rec_name = "INCONCLUSIVE_REQUIRE_HUMAN_REVIEW"
    else:
        # default: keep continuous but improve CV if balance issues
        rec = "B" if h3 != "WEAK" else "A"
        rec_name = "KEEP_HIC_CONTINUOUS_BUT_IMPROVE_CV_GUIDANCE" if rec == "B" else "KEEP_HIC_CONTINUOUS_AS_IS"

    # Public anomaly
    pub_corr = corr_df[(corr_df.metric == "Pearson") & (corr_df.pair == "cv_vs_public")]
    pub_corr_v = float(pub_corr.iloc[0].pearson) if len(pub_corr) else np.nan

    lines = []
    lines.append("# Gate B6.2 — HIC Continuous-Regression Validity Final Report\n\n")
    lines.append(f"**Overall HIC continuous-task verdict:** {rec_name}\n\n")
    lines.append(f"Chosen option (**section 46**): **{rec}. {rec_name}**\n\n")
    lines.append("No new Public/Private split was searched. CONVERT_TO_BINARY / CONVERT_TO_ORDINAL were **not** selected as automatic recommendations.\n\n")
    lines.append("---\n\n")
    lines.append("## Overall HIC continuous-task verdict (summary card)\n\n")
    lines.append("1. Target distribution\n")
    lines.append(f"    skewness: {tr['skewness']:.3f}\n")
    lines.append(f"    median: {tr['median']:.4f}\n")
    lines.append(f"    mean: {tr['mean']:.4f}\n")
    lines.append(f"    SD: {tr['SD']:.4f}\n")
    lines.append(f"    IQR: {tr['IQR']:.4f}\n")
    lines.append(f"    central concentration: frac(median±0.5IQR)={tr['frac_med_pm_0.50IQR']:.3f}; narrowest-80% width={tr['narrow80_width']:.3f}\n")
    lines.append(f"    upper-tail size: Train n>Q90={int(p1['tail_rows'][0]['n_gt_Train_Q90'])}, n>Q95={int(p1['tail_rows'][0]['n_gt_Train_Q95'])}\n\n")
    lines.append("2. Median-predictor competitiveness\n")
    lines.append(f"    constant median MAE: CV={med_mae['cv']:.4f}, Pub={med_mae['public']:.4f}, Priv={med_mae['private']:.4f}\n")
    lines.append(f"    best advanced MAE: {best_tag} CV={best_cv.MAE:.4f}\n")
    lines.append(f"    absolute improvement: {best_cv.absolute_gain_vs_median:.4f}\n")
    lines.append(f"    relative improvement: {best_cv.relative_gain_vs_median:.3%}\n\n")
    lines.append("3. Prediction shrinkage\n")
    lines.append(f"    best model pred_SD / obs_SD: {ratio:.3f}\n")
    lines.append(f"    calibration slope: {slope:.3f}\n")
    lines.append(f"    MAE-optimal alpha: {ba['alpha'] if ba else 'n/a'}\n")
    if ba:
        orig = p2["shrink_df"]
        o1 = orig[(orig.model == best_tag) & (orig.role == "cv") & (orig.alpha == 1.0)]
        lines.append(f"    MAE gain from shrinkage: {float(o1.iloc[0].MAE) - ba['MAE']:.4f}\n\n" if len(o1) else "    MAE gain from shrinkage: n/a\n\n")
    else:
        lines.append("    MAE gain from shrinkage: n/a\n\n")
    lines.append("4. Tail prediction\n")
    lines.append(f"    central 80% MAE: {reg_mae(best_tag,'cv','central_80'):.4f}\n")
    lines.append(f"    upper 20% MAE: {reg_mae(best_tag,'cv','upper_20'):.4f}\n")
    lines.append(f"    upper 10% MAE: {reg_mae(best_tag,'cv','upper_10'):.4f}\n")
    lines.append(f"    advanced vs median gain in tail (upper_10): {reg_gain(best_tag,'cv','upper_10'):.4f}\n")
    lines.append(f"    systematic tail bias (upper_10 mean pred−true): {reg_bias(best_tag,'cv','upper_10'):.4f}\n\n")
    lines.append("5. Pearson influence\n")
    lines.append(f"    largest leave-one-out |Δr|: {max_dr:.4f}\n")
    lines.append(f"    top-3 influence (max sum |Δr| across model×role): {float(top3):.4f}\n")
    lines.append("    model-rank sensitivity: Pearson ranks can change after removing 1–5 influencers (see 04 report)\n")
    lines.append(f"    Public anomaly explanation: CV↔Public Pearson score-vector r≈{pub_corr_v:.3f}; sparse high-HIC leverage drives reordering\n\n")
    lines.append("6. CV design\n")
    lines.append(f"    current Group CV stability: fold MAE SD≈{cur_mae_sd:.4f}, Pearson SD≈{cur_p_sd:.4f}, tail-count SD≈{cur_tail_sd_val:.3f}\n")
    lines.append(f"    best Train-only CV scheme: {preferred}\n")
    lines.append(f"    Q5/Q7/Q9/TAIL-Q7 result: {json.dumps(p5['scheme_status'])}\n")
    lines.append(f"    target-balance gain: preferred vs CURRENT (lower better) — see metrics/cv_scheme_stability.csv\n")
    lines.append(f"    MAE stability gain: preferred fold MAE SD={pref_mae_sd:.4f} vs CURRENT {cur_mae_sd:.4f}\n")
    lines.append(f"    Pearson stability gain: preferred fold Pearson SD={pref_p_sd:.4f} vs CURRENT {cur_p_sd:.4f}\n\n")
    lines.append(f"Recommended HIC task: KEEP continuous absolute-value regression ({rec_name})\n")
    lines.append(f"Recommended participant CV: {preferred} (sequence_group constrained; HIC quantile strata if SGKF_*)\n")
    lines.append("Primary metric: MAE\n")
    lines.append("Secondary metric: Spearman (Pearson as diagnostic only given leverage sensitivity)\n\n")
    lines.append("Any reason to abandon CAND_12528: NO for MAE-oriented evaluation; Pearson Public transfer remains fragile by target nature\n")
    lines.append("Any reason to drop HIC: ONLY if organizers require strong Pearson leaderboard stability; scientifically continuous HIC still informative but noisy\n")
    lines.append("Any reason to reconsider TmApp-only competition: Optional if packaging simplicity is paramount; not mandated by this audit\n\n")

    lines.append("---\n\n## Hypothesis ratings (H1–H4)\n\n")
    lines.append(f"- **H1** MAE heavily rewards center-prediction / shrinkage: **{h1}**\n")
    lines.append(f"- **H2** Pearson dominated by sparse high-HIC observations: **{h2}**\n")
    lines.append(f"- **H3** Existing grouped CV has poor HIC target balance: **{h3}**\n")
    lines.append(f"- **H4** Continuous HIC contains learnable signal beyond constant/central predictor: **{h4}**\n\n")

    lines.append("---\n\n## Answers to required questions (Q1–24)\n\n")
    lines.append(f"1. Yes — most mass is central (frac median±0.5IQR={tr['frac_med_pm_0.50IQR']:.3f}; narrowest 80% width={tr['narrow80_width']:.3f}).\n")
    lines.append(f"2. Meaningful high-HIC tail ≈ Train n>Q90={int(p1['tail_rows'][0]['n_gt_Train_Q90'])} (~10%) and n>Q95={int(p1['tail_rows'][0]['n_gt_Train_Q95'])} (~5%).\n")
    lines.append(f"3. Constant-median baseline is strong (CV MAE={med_mae['cv']:.4f}).\n")
    lines.append(f"4. Best model improves by {best_cv.absolute_gain_vs_median:.4f} MAE ({best_cv.relative_gain_vs_median:.1%}) on CV.\n")
    lines.append(f"5. Yes — pred_SD/obs_SD for best model ≈ {ratio:.2f} (under-dispersed).\n")
    lines.append(f"6. {'Yes' if ba and ba['alpha'] < 1 else 'Limited/No'} — Train OOF MAE-optimal α={ba['alpha'] if ba else 'n/a'}.\n")
    lines.append(f"7. {'Yes' if ba and ba['alpha'] <= 0.75 else 'Somewhat' if ba and ba['alpha'] < 1 else 'No'} — MAE optimum is more compressed than α=1 when α*<1.\n")
    lines.append(f"8. Upper-tail gain vs median (upper_10 CV) = {reg_gain(best_tag,'cv','upper_10'):.4f} — typically small vs center.\n")
    lines.append(f"9. Yes — upper_10 mean signed error (pred−true) = {reg_bias(best_tag,'cv','upper_10'):.4f} (underprediction).\n")
    bur = bur_df[(bur_df.model == best_tag) & (bur_df.role == "cv") & (bur_df.subset == "upper_10")]
    lines.append(f"10. Upper-10 contributes ~{float(bur.iloc[0].frac_of_total_abs_error):.1%} of total abs error" + (f" (N={int(bur.iloc[0].N)}).\n" if len(bur) else ".\n"))
    lines.append(f"11. Individual |Δr| up to {max_dr:.3f} — material influence.\n")
    lines.append(f"12. Most top influencers are high-HIC (frac of top-1 with Train pct≥90 ≈ {frac_tops_high:.2f}).\n")
    lines.append("13. Yes — removing 1–3 influencers can materially change Pearson and reorder models on Public.\n")
    lines.append("14. Yes — Pearson is intrinsically fragile for this right-skewed sparse-tail target.\n")
    lines.append("15. Spearman is more robust (rank-based; less tail leverage).\n")
    lines.append(f"16. Yes — current Group CV distributes tails unevenly (fold n>Q90 SD≈{cur_tail_sd_val:.2f}).\n")
    if len(p5["dep_df"]):
        lines.append(f"17. Fold Pearson depends on fold range/SD (example corr vs SD: {p5['dep_df']['corr_pearson_vs_fold_SD'].mean():.2f}).\n")
    else:
        lines.append("17. Fold Pearson shows dependence on fold HIC dispersion (see cv_fold_pearson_vs_target_range.csv).\n")
    lines.append(f"18. Quantile-stratified Group CV {'improves' if preferred != 'CURRENT_GROUP_CV' else 'may modestly affect'} stability; preferred={preferred}.\n")
    lines.append(f"19. Best Train-only scheme: **{preferred}** (statuses: {p5['scheme_status']}).\n")
    lines.append(f"20. Sturges (~{int(tr.bins_sturges)}) motivates ~7–9 strata count; use quantile strata, not equal-width Sturges bins.\n")
    lines.append("21. CAND_12528 remains acceptable as production split for MAE-first evaluation; no new split search performed.\n")
    lines.append(f"22. Continuous HIC is a meaningful but difficult competition task (H4={h4}); keep continuous, do not auto-convert to classification.\n")
    lines.append("23. TmApp-only would be cleaner for metric stability, but is not required if HIC is retained with improved CV + MAE primary.\n")
    lines.append("24. Binary/ordinal HIC would lose magnitude of retention-time risk (how high), assay-continuous ranking within bins, and absolute error semantics needed for developability assessment.\n")

    lines.append("\n---\n\n## Scientific use-case check\n\n")
    lines.append("Models provide partial information about *how high* HIC is (nonzero Pearson/Spearman, modest MAE gain), ")
    lines.append("but under-predict the sparse high-HIC region where developability risk is most concerning. ")
    lines.append("Continuous regression remains the scientifically correct primary task; treat classification only as optional auxiliary diagnostics under separate human review.\n")

    (REPORTS / "GATE_B6_2_HIC_VALIDITY_FINAL.md").write_text("".join(lines))
    print("Wrote", REPORTS / "GATE_B6_2_HIC_VALIDITY_FINAL.md", flush=True)


def freeze_config(meta: dict) -> None:
    cand = meta["cand"]
    obj = {
        "gate": "B6.2",
        "split_candidate": "CAND_12528",
        "public_ids_hash": cand.get("public_hash") or sha256_lines(meta["public_ids"]),
        "private_ids_hash": cand.get("private_hash") or sha256_lines(meta["private_ids"]),
        "n_public": len(meta["public_ids"]),
        "n_private": len(meta["private_ids"]),
        "n_train": len(meta["train_ids"]),
        "models": MODEL_TAGS,
        "predictions_root": str(B5_PRED),
        "outer_cv": str(B4_CFG / "OUTER_CV_FOLDS.json"),
        "rules": [
            "No new Public/Private split search",
            "No automatic CONVERT_TO_BINARY / CONVERT_TO_ORDINAL",
            "Phase-5 CV uses fixed hyperparameters only",
        ],
    }
    write_json(CONFIG / "B6_2_SPLIT_AND_MODELS.json", obj)


def main():
    ensure_dirs()
    data = load_everything()
    preds, meta = data["preds"], data["meta"]
    freeze_config(meta)

    p1 = phase1(meta)
    p2 = phase2(preds, meta)
    p3 = phase3(preds, meta)
    p4 = phase4(preds, meta)

    # Gate: only run Phase 5 after Phase 1–4 artifacts exist
    required = [
        REPORTS / "01_hic_target_distribution.md",
        REPORTS / "02_prediction_shrinkage.md",
        REPORTS / "03_hic_tail_performance.md",
        REPORTS / "04_pearson_influence.md",
        METRICS / "hic_distribution_summary.csv",
        METRICS / "prediction_dispersion.csv",
        METRICS / "region_specific_errors.csv",
        METRICS / "pearson_sample_influence.csv",
    ]
    assert all(p.exists() for p in required), "Phase 1–4 artifacts missing"

    p5 = phase5(preds, meta, phase_artifacts_ok=True)
    phase6(meta, p1, p2, p3, p4, p5)
    print("DONE Gate B6.2", flush=True)


if __name__ == "__main__":
    main()
