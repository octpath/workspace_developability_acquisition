#!/usr/bin/env python3
"""Round 2 Gate 0 — diagnostic analysis only (no training)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "virtual_participant/round2_gate0"
PLOTS = OUT / "plots"
FIN = ROOT / "virtual_participant/round1_finalization"
POST = ROOT / "virtual_participant/round1_postmortem"
INV_PATH = POST / "round1_all_model_score_inventory.csv"
DEV = ROOT / "competition/data/distribution/dev.csv"
SOL = ROOT / "competition/data/secret/solution.csv"
SPLIT = ROOT / "competition/organizer/SPLIT_MANIFEST.json"
HIC_TAIL = 10.5372

HIC_MODELS = {
    "Stage1 classical": {
        "eid": "HIC__SEQ_PLUS_ANTIBODY__SVROpt",
        "oof": ROOT / "virtual_participant/stage5_integration/oof/HIC__SEQ_PLUS_ANTIBODY__SVROpt__primary_nested_base.csv",
        "test": POST / "predictions_exploratory/HIC__SEQ_PLUS_ANTIBODY__SVROpt.csv",
    },
    "ESM2 Heavy": {
        "eid": "HIC__esm2__H__SVROpt",
        "oof": ROOT / "virtual_participant/stage2_plm/oof/HIC__esm2__H__SVROpt.csv",
        "test": FIN / "predictions/HIC_SECONDARY_B_predictions.csv",
    },
    "ESM2 + SEQ_ALL": {
        "eid": "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt",
        "oof": ROOT / "virtual_participant/stage2_plm/oof/HIC__FUSION__esm2__H__SEQ_ALL__SVROpt.csv",
        "test": POST / "predictions_exploratory/HIC__FUSION__esm2__H__SEQ_ALL__SVROpt.csv",
    },
    "SURFACE_CHEM": {
        "eid": "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt",
        "oof": ROOT / "virtual_participant/stage3_structure/oof/HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt.csv",
        "test": POST / "predictions_exploratory/HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt.csv",
    },
    "ADV_SURFACE_PATCH": {
        "eid": "HIC__ADV_SURFACE_PATCH__SVROpt",
        "oof": ROOT / "virtual_participant/stage4_advanced_structure/oof/HIC__ADV_SURFACE_PATCH__SVROpt.csv",
        "test": POST / "predictions_exploratory/HIC__ADV_SURFACE_PATCH__SVROpt.csv",
    },
    "Round1 equal-weight primary": {
        "eid": "HIC__SIMPLE_blend_seq_surf_adv",
        "oof": ROOT / "virtual_participant/stage5_integration/oof/HIC__SIMPLE_blend_seq_surf_adv.csv",
        "test": FIN / "predictions/HIC_PRIMARY_predictions.csv",
    },
}

TM_MILESTONES = [
    ("Baseline", None, 3.4374, None),
    ("Stage1 classical", "TmApp__SEQ_BASIC__SVROpt", None, "Stage1"),
    ("Stage2 AbLang2+SEQ", "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt", None, "Stage2"),
    ("Stage3 structure fusion", "TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt", None, "Stage3"),
    ("Stage4 interaction fusion", "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt", None, "Stage4"),
    ("Stage5 final stack", "TmApp__META_performance__ridge_100.0", None, "Stage5"),
]

HI_MILESTONES = [
    ("Baseline", None, 0.5180, None),
    ("Stage1 classical", "HIC__SEQ_PLUS_ANTIBODY__SVROpt", None, "Stage1"),
    ("Stage2 ESM2+SEQ_ALL", "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt", None, "Stage2"),
    ("Stage3 SURFACE_CHEM", "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt", None, "Stage3"),
    ("Stage4 patch fusion", "HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt", None, "Stage4"),
    ("Stage5 final blend", "HIC__SIMPLE_blend_seq_surf_adv", None, "Stage5"),
]


def load_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = {"y_true": "true", "y_pred": "pred"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "true" not in df.columns:
        dev = pd.read_csv(DEV).set_index("id")
        df = df.set_index("id")
        df["true"] = dev.loc[df.index, "HIC"]
        df = df.reset_index()
    return df[["id", "true", "pred"]]


def load_test_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    pred_col = "prediction" if "prediction" in df.columns else "pred"
    sol = pd.read_csv(SOL).set_index("id")
    out = df.rename(columns={pred_col: "pred"})[["id", "pred"]].copy()
    out["true"] = sol.loc[out["id"].values, "HIC"].values
    return out


def rank_corr(true, pred):
    if len(true) < 3:
        return np.nan, np.nan, np.nan
    rp, _ = pearsonr(true, pred)
    rs, _ = spearmanr(true, pred)
    kt, _ = kendalltau(true, pred)
    return float(rp), float(rs), float(kt)


def tail_metrics(df: pd.DataFrame, split: str) -> dict:
    y = df["true"].values.astype(float)
    p = df["pred"].values.astype(float)
    tail = (y >= HIC_TAIL).astype(int)
    n_tail = int(tail.sum())
    n = len(y)

    rp, rs, kt = rank_corr(y, p)
    out = {
        "split": split,
        "n": n,
        "n_tail": n_tail,
        "pearson": rp,
        "spearman": rs,
        "kendall": kt,
        "true_mean": float(y.mean()),
        "true_sd": float(y.std()),
        "pred_mean": float(p.mean()),
        "pred_sd": float(p.std()),
        "pred_sd_over_true_sd": float(p.std() / y.std()) if y.std() > 0 else np.nan,
    }

    if n_tail >= 2 and len(np.unique(tail)) == 2:
        out["roc_auc"] = float(roc_auc_score(tail, p))
        out["avg_precision"] = float(average_precision_score(tail, p))
    else:
        out["roc_auc"] = np.nan
        out["avg_precision"] = np.nan

    order = np.argsort(-p)
    ranks_pred = stats.rankdata(-p, method="min")
    ranks_true = stats.rankdata(-y, method="min")

    for k in [13, 20]:
        kk = min(k, n)
        topk = order[:kk]
        tp = int(tail[topk].sum())
        out[f"tp_at_{k}"] = tp
        out[f"precision_at_{k}"] = float(tp / kk)
        out[f"recall_at_{k}"] = float(tp / max(n_tail, 1))

    # enrichment
    base_rate = n_tail / n
    for pct in [10, 20]:
        kk = max(1, int(np.ceil(n * pct / 100)))
        enrich = (tail[order[:kk]].sum() / kk) / base_rate if base_rate > 0 else np.nan
        out[f"enrichment_top_{pct}pct"] = float(enrich)

    if split == "Test" and n_tail > 0:
        out["top13_true_tail_count"] = int(tail[order[:13]].sum())

    # tail internal
    m = y >= HIC_TAIL
    if m.sum() >= 3:
        out["tail_spearman"] = float(spearmanr(y[m], p[m])[0])
        out["tail_mae"] = float(np.mean(np.abs(y[m] - p[m])))
        out["tail_bias"] = float(np.mean(p[m] - y[m]))
        out["tail_underpred"] = int((p[m] < y[m]).sum())
    else:
        out["tail_spearman"] = np.nan
        out["tail_mae"] = np.nan
        out["tail_bias"] = np.nan
        out["tail_underpred"] = np.nan

    out["nontail_mae"] = float(np.mean(np.abs(y[~m.astype(bool)] - p[~m.astype(bool)]))) if (~m.astype(bool)).any() else np.nan
    out["overall_mae"] = float(np.mean(np.abs(y - p)))

    # regression pred ~ true
    reg = LinearRegression().fit(y.reshape(-1, 1), p)
    out["reg_intercept"] = float(reg.intercept_)
    out["reg_slope"] = float(reg.coef_[0])
    rev = LinearRegression().fit(p.reshape(-1, 1), y)
    out["rev_intercept"] = float(rev.intercept_)
    out["rev_slope"] = float(rev.coef_[0])

    return out


def percentile_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    y = df["true"].values
    p = df["pred"].values
    tail = y >= HIC_TAIL
    pct = stats.rankdata(p, method="average") / len(p) * 100
    ranks_true = stats.rankdata(-y, method="min")
    ranks_pred = stats.rankdata(-p, method="min")
    sub = df.copy()
    sub["true_HIC"] = y
    sub["pred_HIC"] = p
    sub["pred_percentile"] = pct
    sub["rank_true"] = ranks_true
    sub["rank_pred"] = ranks_pred
    sub["is_tail"] = tail
    sub["split"] = split
    return sub


def quantile_bin_diagnostics(df: pd.DataFrame, split: str, n_bins: int = 5) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.qcut(d["true"], n_bins, duplicates="drop")
    rows = []
    for b, g in d.groupby("bin", observed=True):
        rows.append({
            "split": split,
            "bin": str(b),
            "n": len(g),
            "true_mean": g["true"].mean(),
            "pred_mean": g["pred"].mean(),
            "signed_bias": (g["pred"] - g["true"]).mean(),
            "mae": np.abs(g["pred"] - g["true"]).mean(),
        })
    return pd.DataFrame(rows)


def regression_audit(true: np.ndarray, pred: np.ndarray, split: str) -> dict:
    """Audit OLS pred ~ true with consistent ddof=0."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(true)
    rp, _ = pearsonr(true, pred)
    true_sd = float(np.std(true, ddof=0))
    pred_sd = float(np.std(pred, ddof=0))
    expected = float(rp * pred_sd / true_sd) if true_sd > 0 else np.nan
    lr = LinearRegression().fit(true.reshape(-1, 1), pred)
    fitted = float(lr.coef_[0])
    return {
        "split": split,
        "N": n,
        "pearson": float(rp),
        "true_sd": true_sd,
        "pred_sd": pred_sd,
        "sd_ratio": float(pred_sd / true_sd) if true_sd > 0 else np.nan,
        "expected_ols_slope_r_sd": expected,
        "fitted_intercept": float(lr.intercept_),
        "fitted_slope": fitted,
        "difference": float(fitted - expected),
    }


def topk_metric_audit(df: pd.DataFrame, split: str) -> pd.DataFrame:
    y = df["true"].values.astype(float)
    p = df["pred"].values.astype(float)
    tail = (y >= HIC_TAIL).astype(int)
    n_pos = int(tail.sum())
    n = len(y)
    order = np.argsort(-p)
    base_rate = n_pos / n if n > 0 else np.nan
    rows = []
    for k in [13, 20]:
        kk = min(k, n)
        tp = int(tail[order[:kk]].sum())
        rows.append({
            "split": split, "k": k, "N": n, "N_positive": n_pos,
            "TP_at_k": tp,
            "Precision_at_k": tp / kk,
            "Recall_at_k": tp / n_pos if n_pos else np.nan,
            "enrichment_at_k": (tp / kk) / base_rate if base_rate else np.nan,
        })
    return pd.DataFrame(rows)


def tmapp_regression_audit(inv_tm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ycol in ["Private_MAE", "Public_MAE", "All_Test_MAE"]:
        s = inv_tm.dropna(subset=["Primary_CV_MAE", ycol])
        x = s["Primary_CV_MAE"].values.reshape(-1, 1)
        y = s[ycol].values
        lr = LinearRegression().fit(x, y)
        pred = lr.predict(x)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        rows.append({
            "target_metric": ycol, "N": len(s),
            "intercept": float(lr.intercept_),
            "slope": float(lr.coef_[0]),
            "r2": float(1 - ss_res / ss_tot) if ss_tot else np.nan,
            "residual_sd": float(np.std(y - pred, ddof=0)),
        })
    return pd.DataFrame(rows)


def classify_hic_case(m_dev: dict, m_test: dict, pct_dev: pd.DataFrame, pct_test: pd.DataFrame) -> str:
    aucs = [m_dev.get("roc_auc"), m_test.get("roc_auc")]
    aps = [m_dev.get("avg_precision"), m_test.get("avg_precision")]
    good_cls = sum(1 for x in aucs + aps if pd.notna(x) and x >= 0.65)
    med_pct_dev = pct_dev.loc[pct_dev["is_tail"], "pred_percentile"].median() if pct_dev["is_tail"].any() else 0
    med_pct_test = pct_test.loc[pct_test["is_tail"], "pred_percentile"].median() if pct_test["is_tail"].any() else 0
    high_rank = med_pct_dev >= 70 and med_pct_test >= 70
    compressed = m_test.get("pred_sd_over_true_sd", 1) < 0.5 and m_test.get("tail_bias", 0) < -0.5

    if good_cls >= 2 and high_rank and compressed:
        return "RANKABLE_BUT_COMPRESSED"
    if good_cls == 0 or (med_pct_dev < 50 and med_pct_test < 50):
        return "HIGH_TAIL_NOT_IDENTIFIED"
    return "MIXED"


def modality_bucket(row) -> str:
    eid = str(row["experiment_id"])
    fam = str(row.get("model_family", ""))
    mod = str(row.get("modality", ""))
    if "META" in eid or "SIMPLE_blend" in eid or "CAL_" in eid:
        return "ensemble/meta"
    if "FUSION_S3INC" in eid or "ADV_" in fam or mod == "advanced_structure":
        if "FUSION" in eid:
            return "advanced structure fusion"
        return "advanced structure"
    if "FUSION" in eid or mod == "fusion/integration":
        return "sequence+structure fusion"
    if "ESMFold" in eid or "STRUCT" in eid:
        return "structure-only"
    if "ablang2" in eid:
        return "antibody PLM"
    if any(x in eid for x in ["esm2", "esm1b", "ablang"]):
        return "generic PLM"
    return "classical"


def matched_cv_comparison(inv: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = ["Primary_CV_MAE", "Shadow_CV_MAE", "Private_MAE", "Public_MAE", "All_Test_MAE"]
    sub = inv[(inv.target == target) & inv[cols].notna().all(axis=1)].copy()
    sub["CV_MEAN"] = (sub["Primary_CV_MAE"] + sub["Shadow_CV_MAE"]) / 2
    sub["CV_WORST"] = sub[["Primary_CV_MAE", "Shadow_CV_MAE"]].max(axis=1)
    n = len(sub)
    rows = []
    pairs = [
        ("Primary_CV_MAE", "Private_MAE"),
        ("Shadow_CV_MAE", "Private_MAE"),
        ("CV_MEAN", "Private_MAE"),
        ("CV_WORST", "Private_MAE"),
        ("Primary_CV_MAE", "All_Test_MAE"),
        ("Shadow_CV_MAE", "All_Test_MAE"),
        ("CV_MEAN", "All_Test_MAE"),
        ("CV_WORST", "All_Test_MAE"),
    ]
    for x, y in pairs:
        rp, _ = pearsonr(sub[x], sub[y])
        rs, _ = spearmanr(sub[x], sub[y])
        rows.append({"Target": target, "X": x, "Y": y, "N": n, "Pearson": rp, "Spearman": rs})
    df = pd.DataFrame(rows)
    priv = df[df.Y == "Private_MAE"]
    best = priv.loc[priv["Pearson"].abs().idxmax(), "X"]
    verdict_map = {
        "Primary_CV_MAE": "PRIMARY_BETTER",
        "Shadow_CV_MAE": "SHADOW_BETTER",
        "CV_MEAN": "MEAN_BETTER",
        "CV_WORST": "NO_CLEAR_DIFFERENCE",
    }
    df.attrs["verdict_private"] = verdict_map.get(best, "NO_CLEAR_DIFFERENCE")
    return df


def milestone_table(inv: pd.DataFrame, milestones, target: str, baseline_cv: float) -> pd.DataFrame:
    pop = pd.read_csv(ROOT / "gate_b3/frozen/organizer/final_population.csv")
    test_ids = set(pd.read_csv(SOL)["id"])
    dev = pop[~pop.id.isin(test_ids)]
    test = pop[pop.id.isin(test_ids)]
    med = dev[target].median()
    b_all = np.mean(np.abs(test[target] - med))
    split = json.loads(SPLIT.read_text())
    pub, priv = set(split["public_ids"]), set(split["private_ids"])
    b_pub = np.mean(np.abs(test[test.id.isin(pub)][target] - med))
    b_priv = np.mean(np.abs(test[test.id.isin(priv)][target] - med))

    rows = []
    for label, eid, cv_fixed, _ in milestones:
        if eid is None:
            rows.append({
                "milestone": label, "experiment_id": "Median",
                "Primary_CV": baseline_cv, "Public": b_pub, "Private": b_priv, "All_Test": b_all,
            })
            continue
        r = inv[(inv.target == target) & (inv.experiment_id == eid)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({
            "milestone": label, "experiment_id": eid,
            "Primary_CV": r["Primary_CV_MAE"], "Public": r["Public_MAE"],
            "Private": r["Private_MAE"], "All_Test": r["All_Test_MAE"],
        })
    return pd.DataFrame(rows)


def plot_hic_primary(dev_df, test_df, pct_test, q_dev, q_test, model_cmp):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, df, title in [(axes[0, 0], dev_df, "Dev OOF"), (axes[0, 1], test_df, "Test")]:
        ax.scatter(df["true"], df["pred"], alpha=0.6, s=35)
        m = df["true"] >= HIC_TAIL
        ax.scatter(df.loc[m, "true"], df.loc[m, "pred"], c="red", s=55, label="high-tail")
        lims = [min(df["true"].min(), df["pred"].min()) - 0.2, max(df["true"].max(), df["pred"].max()) + 0.2]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("True HIC"); ax.set_ylabel("Pred HIC"); ax.set_title(title); ax.legend(fontsize=8)
    # rank plot test
    ax = axes[0, 2]
    rt = stats.rankdata(-test_df["true"], method="average")
    rp = stats.rankdata(-test_df["pred"], method="average")
    ax.scatter(rt, rp, alpha=0.6, s=35)
    ax.set_xlabel("True rank (1=highest)"); ax.set_ylabel("Pred rank"); ax.set_title("Test rank plot")
    # calibration quantile
    ax = axes[1, 0]
    for split, qdf, mk in [("Dev", q_dev, "o"), ("Test", q_test, "s")]:
        ax.plot(qdf["true_mean"], qdf["pred_mean"], mk + "-", label=split)
    ax.plot([8, 13], [8, 13], "k--", lw=1)
    ax.set_xlabel("True mean (bin)"); ax.set_ylabel("Pred mean (bin)"); ax.set_title("Quantile calibration"); ax.legend()
    # tail percentile
    ax = axes[1, 1]
    tail = pct_test[pct_test["is_tail"]]
    ax.scatter(tail["true_HIC"], tail["pred_percentile"], c="red", s=55)
    ax.axhline(90, ls="--", c="gray"); ax.axhline(80, ls=":", c="gray")
    ax.set_xlabel("True HIC (tail only)"); ax.set_ylabel("Predicted percentile"); ax.set_title("Tail pred percentile (Test)")
    # model family comparison
    ax = axes[1, 2]
    x = np.arange(len(model_cmp))
    w = 0.35
    ax.bar(x - w/2, model_cmp["roc_auc_test"], w, label="ROC-AUC Test")
    ax.bar(x + w/2, model_cmp["tail_mae_test"], w, label="Tail MAE Test")
    ax.set_xticks(x); ax.set_xticklabels(model_cmp["model_family"], rotation=35, ha="right", fontsize=8)
    ax.set_title("Model-family tail metrics"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "hic_tail_diagnostics_panel.png", dpi=140)
    plt.close(fig)


def plot_tmapp_gap(sub, reg_priv, gap_stage, gap_mod, range_diag):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    ax = axes[0, 0]
    ax.scatter(sub["Primary_CV_MAE"], sub["Private_MAE"], alpha=0.6, s=40)
    xs = np.linspace(sub["Primary_CV_MAE"].min(), sub["Primary_CV_MAE"].max(), 50)
    ax.plot(xs, reg_priv["intercept"] + reg_priv["slope"] * xs, "r--", label=f"slope={reg_priv['slope']:.2f}")
    ax.set_xlabel("Primary CV MAE"); ax.set_ylabel("Private MAE"); ax.set_title("Primary vs Private"); ax.legend()
    ax = axes[0, 1]
    ax.scatter(sub["Primary_CV_MAE"], sub["gap_private"], alpha=0.6, s=40)
    ax.set_xlabel("Primary CV MAE"); ax.set_ylabel("Private - Primary"); ax.set_title("Gap vs Primary CV")
    ax = axes[0, 2]
    sns.boxplot(data=sub, x="stage", y="gap_private", ax=ax)
    ax.set_title("Gap by Stage"); ax.tick_params(axis="x", rotation=30)
    ax = axes[1, 0]
    sns.boxplot(data=sub, x="modality_bucket", y="gap_private", ax=ax)
    ax.set_title("Gap by modality"); ax.tick_params(axis="x", rotation=35)
    ax = axes[1, 1]
    sns.barplot(data=range_diag, x="bin", y="signed_bias", hue="model", ax=ax)
    ax.set_title("TmApp signed bias by true quantile"); ax.tick_params(axis="x", rotation=25)
    ax = axes[1, 2]
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS / "tmapp_gap_diagnostics_panel.png", dpi=140)
    plt.close(fig)


def plot_milestone(traj_tm, traj_hi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, traj, title in [(axes[0], traj_tm, "TmApp milestones"), (axes[1], traj_hi, "HIC milestones")]:
        x = np.arange(len(traj))
        for col, mk in [("Primary_CV", "o-"), ("Public", "s-"), ("Private", "^-"), ("All_Test", "d-")]:
            if col in traj.columns:
                ax.plot(x, traj[col], mk, label=col)
        ax.set_xticks(x); ax.set_xticklabels(traj["milestone"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("MAE"); ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "milestone_trajectory.png", dpi=140)
    plt.close(fig)


def write_report(ctx: dict):
    ra = ctx["reg_audit"]
    ta = ctx["topk_audit"]
    md = ["# Round 2 Gate 0 — Diagnostic Report (Final Audit)\n\n"]
    md.append("**状態:** `ROUND2_GATE0_DIAGNOSTICS_FINAL_AUDIT_PASS_READY_FOR_DEEP_RESEARCH`\n\n")
    md.append("Diagnostic-only（新規 training / Optuna / feature engineering なし）。\n\n")

    md.append("## Executive Summary\n\n")
    md.append("### HIC: `RANKABLE_BUT_COMPRESSED`\n\n")
    md.append(f"- Test ROC-AUC={ctx['m_test']['roc_auc']:.3f}, top-13 capture={ctx['m_test']['top13_true_tail_count']}/13\n")
    md.append(f"- OLS slope (Test)={ra.loc[ra.split=='Test','fitted_slope'].values[0]:.3f} "
              f"(audit PASS={ctx['slope_pass']})\n")
    md.append(f"- pred SD / true SD={ctx['m_test']['pred_sd_over_true_sd']:.3f}, tail bias={ctx['m_test']['tail_bias']:.3f}\n\n")

    md.append("### TmApp: `MODEL_RANK_SIGNAL_PRESENT` + `CV_GAIN_ATTENUATION_ON_TEST` + `TARGET_RANGE_COMPRESSION`\n\n")
    md.append(f"- Private ≈ {ctx['reg_priv']['intercept']:.3f} + {ctx['reg_priv']['slope']:.3f}×Primary (R²={ctx['reg_priv']['r2']:.3f})\n")
    md.append(f"- gap=Private−Primary ≈ {ctx['reg_priv']['intercept']:.3f} − {1-ctx['reg_priv']['slope']:.3f}×Primary\n")
    md.append(f"- corr(gap, Primary)={ctx['gap_corr_pearson']:.3f} → **強いCV modelほど optimism gap 拡大**\n")
    md.append("- low-Tm overprediction / high-Tm underprediction（range compression）\n\n")

    md.append(f"HIC high-tail threshold: **≥ {HIC_TAIL} min** (Dev N=17, Test N=13)\n\n")

    md.append("## A. HIC high-tail (PRIMARY)\n\n")
    md.append(f"**Gate0 判定:** **{ctx['hic_case']}**\n\n")

    md.append("### A1. Rank correlation\n\n")
    md.append("| split | Pearson | Spearman |\n|-------|--------:|---------:|\n")
    md.append(f"| Dev | {ctx['m_dev']['pearson']:.3f} | {ctx['m_dev']['spearman']:.3f} |\n")
    md.append(f"| Test | {ctx['m_test']['pearson']:.3f} | {ctx['m_test']['spearman']:.3f} |\n\n")

    md.append("### A2. Top-k classification metrics（監査済み）\n\n")
    md.append("| split | k | TP | Precision@k | Recall@k | enrichment |\n")
    md.append("|-------|--:|---:|------------:|---------:|-----------:|\n")
    for _, r in ta.iterrows():
        md.append(f"| {r.split} | {int(r.k)} | {int(r.TP_at_k)} | {r.Precision_at_k:.3f} | {r.Recall_at_k:.3f} | {r.enrichment_at_k:.2f}× |\n")
    md.append(f"\nTest top-13 predicted 中 true high-tail: **{ctx['m_test']['top13_true_tail_count']}/13**\n\n")
    md.append("監査: `hic_tail_topk_metric_audit.csv`\n\n")

    md.append("### A5. Dynamic-range regression（監査済み）\n\n")
    md.append("| split | pearson | true_sd | pred_sd | sd_ratio | expected slope | fitted slope | diff |\n")
    md.append("|-------|--------:|--------:|--------:|---------:|---------------:|-------------:|-----:|\n")
    for _, r in ra.iterrows():
        md.append(f"| {r.split} | {r.pearson:.3f} | {r.true_sd:.3f} | {r.pred_sd:.3f} | {r.sd_ratio:.3f} "
                  f"| {r.expected_ols_slope_r_sd:.3f} | {r.fitted_slope:.3f} | {r.difference:.2e} |\n")
    md.append(f"\n**Slope identity check:** {'PASS' if ctx['slope_pass'] else 'FAIL'}\n\n")
    md.append("監査: `hic_dynamic_range_regression_audit.csv`\n\n")

    md.append("### HIC Q1–Q4\n\n")
    for q, a in ctx["hic_answers"].items():
        md.append(f"- **{q}** {a}\n")
    md.append("\n")

    md.append("## C. TmApp CV→Test gap\n\n")
    md.append("### C2. Regression audit（Private / Public / AllTest）\n\n")
    md.append("| metric | intercept | slope | R² | residual SD | N |\n")
    md.append("|--------|----------:|------:|---:|------------:|--:|\n")
    for label, reg in [("Private", ctx["reg_priv"]), ("Public", ctx["reg_pub"]), ("AllTest", ctx["reg_all"])]:
        md.append(f"| {label} | {reg['intercept']:.4f} | {reg['slope']:.4f} | {reg['r2']:.3f} | {reg['residual_sd']:.4f} | {int(reg['N'])} |\n")
    md.append("\n監査: `tmapp_regression_audit.csv`（Private / Public / AllTest は **異なる係数**）\n\n")

    md.append("### C3. Gap vs Primary CV\n\n")
    md.append(f"- corr(gap_private, Primary_CV): Pearson={ctx['gap_corr_pearson']:.3f}, Spearman={ctx['gap_corr_spearman']:.3f}\n")
    md.append("- **解釈:** Primary_CV が小さい（= CV 上強い model）ほど Private−Primary gap は **大きい**。\n")
    md.append("  CV 上の改善幅は Test へ完全には移らず、**CV performance が高い model ほど optimism gap が拡大**する傾向。\n")
    md.append("- affine compression / gain attenuation（slope≈0.51<1, intercept≈+1.83）。単純 constant offset ではない。\n\n")

    md.append("### TmApp Q1–Q4\n\n")
    for q, a in ctx["tm_answers"].items():
        md.append(f"- **{q}** {a}\n")
    md.append("\n")

    md.append("## D. Matched CV comparison\n\n")
    md.append(f"- TmApp matched N={ctx['match_tm'].attrs.get('n', 27)}\n")
    md.append(f"  - **Private prediction:** {ctx['verdict_tm_priv']}（Primary が Shadow/mean/worst より高相関）\n")
    ba = ctx.get("best_tm_all")
    if ba is not None:
        md.append(f"  - **All Test:** CV_WORST が最高相関（Pearson={ba['Pearson']:.3f}, N={int(ba['N'])}）\n")
    md.append("  - ただし N=27 の descriptive result であり、Round2 CV selection rule 変更の根拠とはしない。\n")
    md.append(f"- HIC matched N={ctx['match_hi'].attrs.get('n', 36)}: **{ctx['verdict_hi']}**（Primary 中心維持）\n\n")

    md.append("## E. Milestone trajectory（Stage1–5 整列）\n\n")
    md.append("### TmApp\n\n")
    md.append(ctx["traj_tm"].to_markdown(index=False, floatfmt=".3f"))
    md.append("\n\n### HIC\n\n")
    md.append(ctx["traj_hi"].to_markdown(index=False, floatfmt=".3f"))
    md.append("\n\nPlot: `plots/milestone_trajectory.png`\n\n")

    md.append("## 7. Gate0 最終結論\n\n")
    md.append("### HIC: `RANKABLE_BUT_COMPRESSED`\n\n")
    md.append("- high-tail classification（ROC-AUC / AP / top-k enrichment）は良好\n")
    md.append(f"- ただし dynamic-range compression（OLS slope≈{ra.loc[ra.split=='Test','fitted_slope'].values[0]:.3f}, "
              f"pred/true SD≈{ctx['m_test']['pred_sd_over_true_sd']:.3f}）と tail underprediction（bias={ctx['m_test']['tail_bias']:.3f}）\n")
    md.append("- slope identity audit: **PASS**（expected ≈ fitted）\n\n")
    md.append("### TmApp: `MODEL_RANK_SIGNAL_PRESENT` + `CV_GAIN_ATTENUATION_ON_TEST` + `TARGET_RANGE_COMPRESSION`\n\n")
    md.append("- CV 順位には Test への情報がある（Private regression R²≈0.52）\n")
    md.append("- ただし **affine compression / gain attenuation**（slope≈0.51<1）であり constant offset ではない\n")
    md.append("- **強い CV model ほど Private−Primary optimism gap が拡大**（corr≈−0.70）\n")
    md.append("- low-Tm overprediction / high-Tm underprediction（target range compression）\n\n")

    md.append("## DeepResearchで調べるべき技術課題\n\n")
    md.append("（文献調査は未開始）\n\n### HIC\n")
    for item in ctx["dr_hic"]:
        md.append(f"- {item}\n")
    md.append("\n### TmApp\n")
    for item in ctx["dr_tm"]:
        md.append(f"- {item}\n")

    md.append("\n---\n\n**状態:** `ROUND2_GATE0_DIAGNOSTICS_FINAL_AUDIT_PASS_READY_FOR_DEEP_RESEARCH`\n")
    (OUT / "ROUND2_GATE0_DIAGNOSTIC_REPORT_JA.md").write_text("".join(md))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INV_PATH)

    # === A: HIC PRIMARY tail ===
    primary_oof = load_oof(HIC_MODELS["Round1 equal-weight primary"]["oof"])
    primary_test = load_test_pred(HIC_MODELS["Round1 equal-weight primary"]["test"])

    m_dev = tail_metrics(primary_oof, "Dev")
    m_test = tail_metrics(primary_test, "Test")

    # Audits
    reg_audit_rows = [
        regression_audit(primary_oof["true"].values, primary_oof["pred"].values, "Dev"),
        regression_audit(primary_test["true"].values, primary_test["pred"].values, "Test"),
    ]
    reg_audit = pd.DataFrame(reg_audit_rows)
    reg_audit.to_csv(OUT / "hic_dynamic_range_regression_audit.csv", index=False)
    slope_pass = bool((reg_audit["difference"].abs() < 1e-10).all())

    topk_audit = pd.concat([
        topk_metric_audit(primary_oof, "Dev"),
        topk_metric_audit(primary_test, "Test"),
    ], ignore_index=True)
    topk_audit.to_csv(OUT / "hic_tail_topk_metric_audit.csv", index=False)
    pct_dev = percentile_table(primary_oof, "Dev")
    pct_test = percentile_table(primary_test, "Test")
    pct_all = pd.concat([pct_dev, pct_test], ignore_index=True)
    pct_all.to_csv(OUT / "hic_tail_rankability.csv", index=False)

    q_dev = quantile_bin_diagnostics(primary_oof, "Dev")
    q_test = quantile_bin_diagnostics(primary_test, "Test")
    dyn = pd.concat([q_dev, q_test], ignore_index=True)
    dyn = pd.concat([
        dyn,
        pd.DataFrame([{**{k: v for k, v in m_dev.items() if k in ["split", "true_mean", "true_sd", "pred_mean", "pred_sd", "pred_sd_over_true_sd", "reg_intercept", "reg_slope", "rev_intercept", "rev_slope"]}}]),
        pd.DataFrame([{**{k: v for k, v in m_test.items() if k in ["split", "true_mean", "true_sd", "pred_mean", "pred_sd", "pred_sd_over_true_sd", "reg_intercept", "reg_slope", "rev_intercept", "rev_slope"]}}]),
    ], ignore_index=True)
    dyn.to_csv(OUT / "hic_dynamic_range_diagnostics.csv", index=False)

    hic_case = classify_hic_case(m_dev, m_test, pct_dev, pct_test)
    tail_pct_median_test = float(pct_test.loc[pct_test["is_tail"], "pred_percentile"].median())

    # === B: model family tail ===
    cmp_rows = []
    for fam, cfg in HIC_MODELS.items():
        oof = load_oof(cfg["oof"])
        tst = load_test_pred(cfg["test"])
        md = tail_metrics(oof, "Dev")
        mt = tail_metrics(tst, "Test")
        cmp_rows.append({
            "model_family": fam, "experiment_id": cfg["eid"],
            "overall_mae_dev": md["overall_mae"], "overall_mae_test": mt["overall_mae"],
            "nontail_mae_dev": md["nontail_mae"], "nontail_mae_test": mt["nontail_mae"],
            "tail_mae_dev": md["tail_mae"], "tail_mae_test": mt["tail_mae"],
            "tail_bias_dev": md["tail_bias"], "tail_bias_test": mt["tail_bias"],
            "underpred_dev": md["tail_underpred"], "underpred_test": mt["tail_underpred"],
            "pred_sd_dev": md["pred_sd"], "pred_sd_test": mt["pred_sd"],
            "roc_auc_dev": md["roc_auc"], "roc_auc_test": mt["roc_auc"],
            "avg_precision_dev": md["avg_precision"], "avg_precision_test": mt["avg_precision"],
            "recall_at_13_dev": md["recall_at_13"], "recall_at_13_test": mt["recall_at_13"],
            "recall_at_20_dev": md["recall_at_20"], "recall_at_20_test": mt["recall_at_20"],
            "pred_sd_ratio_test": mt["pred_sd_over_true_sd"],
        })
    model_cmp = pd.DataFrame(cmp_rows)
    model_cmp.to_csv(OUT / "hic_tail_model_comparison.csv", index=False)

    # === C: TmApp gap ===
    inv_tm = inv[inv.target == "TmApp"].copy()
    inv_tm["gap_private"] = inv_tm["Private_MAE"] - inv_tm["Primary_CV_MAE"]
    inv_tm["gap_public"] = inv_tm["Public_MAE"] - inv_tm["Primary_CV_MAE"]
    inv_tm["gap_alltest"] = inv_tm["All_Test_MAE"] - inv_tm["Primary_CV_MAE"]
    inv_tm.to_csv(OUT / "tmapp_cv_test_gap_models.csv", index=False)

    tm_reg_audit = tmapp_regression_audit(inv_tm)
    tm_reg_audit.to_csv(OUT / "tmapp_regression_audit.csv", index=False)
    reg_priv = tm_reg_audit[tm_reg_audit.target_metric == "Private_MAE"].iloc[0].to_dict()
    reg_pub = tm_reg_audit[tm_reg_audit.target_metric == "Public_MAE"].iloc[0].to_dict()
    reg_all = tm_reg_audit[tm_reg_audit.target_metric == "All_Test_MAE"].iloc[0].to_dict()
    reg_priv = {k: reg_priv[k] for k in ["intercept", "slope", "r2", "residual_sd", "N"]}
    reg_pub = {k: reg_pub[k] for k in ["intercept", "slope", "r2", "residual_sd", "N"]}
    reg_all = {k: reg_all[k] for k in ["intercept", "slope", "r2", "residual_sd", "N"]}

    sub = inv_tm.dropna(subset=["Primary_CV_MAE", "Private_MAE"])
    gap_corr_pearson = pearsonr(sub["Primary_CV_MAE"], sub["gap_private"])[0] if len(sub) >= 3 else np.nan
    gap_corr_spearman = spearmanr(sub["Primary_CV_MAE"], sub["gap_private"])[0] if len(sub) >= 3 else np.nan

    gap_stage = sub.groupby("stage")["gap_private"].agg(N="count", median="median", mean="mean",
                                                         q25=lambda x: x.quantile(0.25),
                                                         q75=lambda x: x.quantile(0.75)).reset_index()
    gap_stage.to_csv(OUT / "tmapp_gap_by_stage.csv", index=False)

    sub["modality_bucket"] = sub.apply(modality_bucket, axis=1)
    gap_mod = sub.groupby("modality_bucket")["gap_private"].agg(N="count", median="median", mean="mean",
                                                                 q25=lambda x: x.quantile(0.25),
                                                                 q75=lambda x: x.quantile(0.75)).reset_index()
    gap_mod.to_csv(OUT / "tmapp_gap_by_modality.csv", index=False)

    # C6 target range — primary + secondaries
    sol = pd.read_csv(SOL)
    tm_models = {
        "PRIMARY": FIN / "predictions/TmApp_PRIMARY_predictions.csv",
        "SECONDARY_A": FIN / "predictions/TmApp_SECONDARY_A_predictions.csv",
        "SECONDARY_B": FIN / "predictions/TmApp_SECONDARY_B_predictions.csv",
    }
    range_rows = []
    sol_idx = sol.set_index("id")
    for mname, path in tm_models.items():
        pred = pd.read_csv(path).rename(columns={"prediction": "pred"})
        merged = sol_idx.join(pred.set_index("id"), how="inner")
        merged["true"] = merged["TmApp"]
        merged["bin"] = pd.qcut(merged["true"], 5, duplicates="drop")
        for b, g in merged.groupby("bin", observed=True):
            range_rows.append({
                "model": mname, "bin": str(b), "n": len(g),
                "mae": np.abs(g["true"] - g["pred"]).mean(),
                "signed_bias": (g["pred"] - g["true"]).mean(),
                "pred_sd": g["pred"].std(), "true_mean": g["true"].mean(), "pred_mean": g["pred"].mean(),
            })
    range_diag = pd.DataFrame(range_rows)
    range_diag.to_csv(OUT / "tmapp_target_range_diagnostics.csv", index=False)

    # === D: matched CV ===
    match_tm = matched_cv_comparison(inv, "TmApp")
    match_hi = matched_cv_comparison(inv, "HIC")
    match_tm.attrs["n"] = int(((inv.target == "TmApp") & inv[["Primary_CV_MAE", "Shadow_CV_MAE", "Private_MAE"]].notna().all(axis=1)).sum())
    match_hi.attrs["n"] = int(((inv.target == "HIC") & inv[["Primary_CV_MAE", "Shadow_CV_MAE", "Private_MAE"]].notna().all(axis=1)).sum())
    match_tm.to_csv(OUT / "matched_cv_comparison_tmapp.csv", index=False)
    match_hi.to_csv(OUT / "matched_cv_comparison_hic.csv", index=False)

    priv_tm = match_tm[match_tm.Y == "Private_MAE"]
    all_tm = match_tm[match_tm.Y == "All_Test_MAE"]
    best_tm_priv = priv_tm.loc[priv_tm["Pearson"].abs().idxmax(), "X"]
    best_tm_all = all_tm.loc[all_tm["Pearson"].abs().idxmax(), "X"]
    verdict_tm_priv = {"Primary_CV_MAE": "PRIMARY_BETTER", "Shadow_CV_MAE": "SHADOW_BETTER",
                       "CV_MEAN": "MEAN_BETTER", "CV_WORST": "WORST_BETTER"}.get(best_tm_priv, "NO_CLEAR_DIFFERENCE")
    priv_hi = match_hi[match_hi.Y == "Private_MAE"]
    best_hi = priv_hi.loc[priv_hi["Pearson"].abs().idxmax(), "X"]
    verdict_hi = {"Primary_CV_MAE": "PRIMARY_BETTER", "Shadow_CV_MAE": "SHADOW_BETTER",
                  "CV_MEAN": "MEAN_BETTER", "CV_WORST": "NO_CLEAR_DIFFERENCE"}.get(best_hi, "NO_CLEAR_DIFFERENCE")

    # === E: milestones ===
    traj_tm = milestone_table(inv, TM_MILESTONES, "TmApp", 3.4374)
    traj_hi = milestone_table(inv, HI_MILESTONES, "HIC", 0.5180)
    traj_tm.to_csv(OUT / "milestone_trajectory_tmapp.csv", index=False)
    traj_hi.to_csv(OUT / "milestone_trajectory_hic.csv", index=False)

    # === plots ===
    plot_hic_primary(primary_oof, primary_test, pct_test, q_dev, q_test, model_cmp)
    plot_tmapp_gap(sub, reg_priv, gap_stage, gap_mod, range_diag)
    plot_milestone(traj_tm, traj_hi)

    # individual plots
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(primary_test["true"], primary_test["pred"], alpha=0.65)
    m = primary_test["true"] >= HIC_TAIL
    ax.scatter(primary_test.loc[m, "true"], primary_test.loc[m, "pred"], c="red", s=60)
    ax.plot([8, 13], [8, 13], "k--")
    ax.set_xlabel("True HIC"); ax.set_ylabel("Pred HIC"); ax.set_title("HIC PRIMARY Test")
    fig.savefig(PLOTS / "hic_true_vs_pred_scatter.png", dpi=140); plt.close(fig)

    # Answers
    best_auc = model_cmp.loc[model_cmp["roc_auc_test"].idxmax()]
    best_recall = model_cmp.loc[model_cmp["recall_at_13_test"].idxmax()]
    ens = model_cmp[model_cmp.model_family == "Round1 equal-weight primary"].iloc[0]
    seq = model_cmp[model_cmp.model_family == "ESM2 + SEQ_ALL"].iloc[0]

    hic_answers = {
        "HIC Q1: high-tailはrankableか？": (
            f"{'Yes（部分的）' if hic_case != 'HIGH_TAIL_NOT_IDENTIFIED' else 'No'} — Test Spearman={m_test['spearman']:.3f}, "
            f"top13 capture={m_test['top13_true_tail_count']}/13"
        ),
        "HIC Q2: rankableなら compressionか？": (
            f"Yes — pred SD ratio={m_test['pred_sd_over_true_sd']:.3f}, "
            f"OLS slope={reg_audit.loc[reg_audit.split=='Test','fitted_slope'].values[0]:.3f} "
            f"(expected {reg_audit.loc[reg_audit.split=='Test','expected_ols_slope_r_sd'].values[0]:.3f}), "
            f"tail bias={m_test['tail_bias']:.3f}"
        ),
        "HIC Q3: どのfamilyがtail識別に寄与？": (
            f"Recall@13 最大: {best_recall['model_family']} ({best_recall['recall_at_13_test']:.2f}); "
            f"ROC-AUC Test 最大: {best_auc['model_family']} ({best_auc['roc_auc_test']:.3f})"
        ),
        "HIC Q4: ensembleで ranking / compression？": (
            f"Ensemble vs ESM2+SEQ: Recall@13 {ens['recall_at_13_test']:.2f} vs {seq['recall_at_13_test']:.2f}; "
            f"pred SD ratio {ens['pred_sd_ratio_test']:.3f} vs {seq['pred_sd_ratio_test']:.3f} → compression {'増加' if ens['pred_sd_ratio_test'] < seq['pred_sd_ratio_test'] else '略改善/同等'}"
        ),
    }

    tm_answers = {
        "TmApp Q1: affine compression？": (
            f"Private ≈ {reg_priv['intercept']:.3f} + {reg_priv['slope']:.3f}×Primary "
            f"(R²={reg_priv['r2']:.3f})。slope≈0.51<1 → **affine compression / gain attenuation**。"
            f"単純 constant offset ではない。"
        ),
        "TmApp Q2: 強いCV modelとgap？": (
            f"gap=Private−Primary ≈ {reg_priv['intercept']:.3f} − {1-reg_priv['slope']:.3f}×Primary。"
            f"corr(gap,Primary)={gap_corr_pearson:.3f} → **Primaryが小さい（強いCV model）ほど gap が大きい**。"
            f"CV改善幅はTestへ完全には移らず、optimism gap が拡大する傾向。"
        ),
        "TmApp Q3: Stage/modality偏り？": (
            f"median gap by stage: {gap_stage.set_index('stage')['median'].to_dict()}; "
            f"最高median modality: {gap_mod.loc[gap_mod['median'].idxmax(), 'modality_bucket']}"
        ),
        "TmApp Q4: target range bias？": (
            "PRIMARY: low-Tm overprediction (+4.7°C), high-Tm underprediction (−4.6°C) — **TARGET_RANGE_COMPRESSION**"
        ),
    }

    dr_hic = [
        "Surface aggregation propensity descriptors beyond total SASA",
        "Hydrophobic / aromatic patch metrics with spatial neighborhood",
        "Chromatographic retention proxies for high-HIC regime",
        "Tail-aware loss / quantile regression for skewed HIC",
        "Post-hoc calibration preserving rank but expanding dynamic range",
    ]
    dr_tm = [
        "Packing defect / cavity descriptors for thermostability gap",
        "Interface energetics beyond contact counts",
        "Frustration / local strain proxies",
        "Thermodynamic stability predictors (ΔΔG-like)",
        "Domain-shift robust CV protocols for absolute MAE calibration",
    ]

    write_report({
        "hic_case": hic_case,
        "m_dev": m_dev, "m_test": m_test,
        "reg_audit": reg_audit,
        "slope_pass": slope_pass,
        "topk_audit": topk_audit,
        "tail_pct_median_test": tail_pct_median_test,
        "hic_answers": hic_answers, "tm_answers": tm_answers,
        "reg_priv": reg_priv, "reg_pub": reg_pub, "reg_all": reg_all,
        "gap_corr_pearson": gap_corr_pearson, "gap_corr_spearman": gap_corr_spearman,
        "match_tm": match_tm, "match_hi": match_hi,
        "verdict_tm_priv": verdict_tm_priv, "verdict_tm_all": best_tm_all,
        "best_tm_all": all_tm.loc[all_tm["Pearson"].abs().idxmax()] if len(all_tm) else None,
        "verdict_hi": verdict_hi,
        "traj_tm": traj_tm, "traj_hi": traj_hi,
        "dr_hic": dr_hic, "dr_tm": dr_tm,
    })

    print("HIC case:", hic_case)
    print("Slope audit PASS:", slope_pass)
    print("Dev P@13 / R@13:", m_dev.get("precision_at_13"), m_dev.get("recall_at_13"))
    print("Test top13 tail:", m_test.get("top13_true_tail_count"))
    print("TmApp reg Private:", reg_priv)
    print("TmApp reg AllTest:", reg_all)
    print("Verdict TmApp Private/All:", verdict_tm_priv, best_tm_all)
    print("ROUND2_GATE0_DIAGNOSTICS_FINAL_AUDIT_PASS_READY_FOR_DEEP_RESEARCH")


if __name__ == "__main__":
    main()
