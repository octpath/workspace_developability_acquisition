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
    ("Baseline median", None, 3.4374, None),
    ("Stage1 classical", "TmApp__SEQ_BASIC__SVROpt", None, "Stage1"),
    ("Stage2 AbLang2 PLM", "TmApp__ablang2__HL_paired__SVROpt", None, "Stage2"),
    ("Stage2 AbLang2+SEQ", "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt", None, "Stage2"),
    ("Stage3 RASA fusion", "TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt", None, "Stage3"),
    ("Stage4 interaction", "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt", None, "Stage4"),
    ("Stage5 final stack", "TmApp__META_performance__ridge_100.0", None, "Stage5"),
]

HI_MILESTONES = [
    ("Baseline median", None, 0.5180, None),
    ("Stage1 classical", "HIC__SEQ_PLUS_ANTIBODY__SVROpt", None, "Stage1"),
    ("Stage2 ESM2 Heavy", "HIC__esm2__H__SVROpt", None, "Stage2"),
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
        out[f"precision_at_{k}"] = float(tail[topk].sum() / kk)
        out[f"recall_at_{k}"] = float(tail[topk].sum() / max(n_tail, 1))

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
    md = ["# Round 2 Gate 0 — Diagnostic Report\n\n"]
    md.append(f"**状態:** `ROUND2_GATE0_DIAGNOSTICS_COMPLETE_READY_FOR_DEEP_RESEARCH`\n\n")
    md.append(f"HIC high-tail threshold: **≥ {HIC_TAIL} min** (Dev N=17, Test N=13)\n\n")

    md.append("## A. HIC high-tail diagnosis (PRIMARY)\n\n")
    md.append(f"**Gate0 判定:** **{ctx['hic_case']}**\n\n")
    md.append(f"- Dev Spearman: {ctx['m_dev']['spearman']:.3f}, Test Spearman: {ctx['m_test']['spearman']:.3f}\n")
    md.append(f"- Test ROC-AUC: {ctx['m_test']['roc_auc']:.3f}, AP: {ctx['m_test']['avg_precision']:.3f}\n")
    md.append(f"- Test top-13 predicted に true high-tail **{ctx['m_test']['top13_true_tail_count']}/13** 件\n")
    md.append(f"- Test pred SD / true SD: {ctx['m_test']['pred_sd_over_true_sd']:.3f}, tail bias: {ctx['m_test']['tail_bias']:.3f}\n")
    md.append(f"- True high-tail median predicted percentile (Test): {ctx['tail_pct_median_test']:.1f}\n\n")

    md.append("### HIC Q1–Q4\n\n")
    for q, a in ctx["hic_answers"].items():
        md.append(f"- **{q}:** {a}\n")
    md.append("\n")

    md.append("## C. TmApp CV→Test gap\n\n")
    md.append(f"- Private = {ctx['reg_priv']['intercept']:.3f} + {ctx['reg_priv']['slope']:.3f}×Primary (R²={ctx['reg_priv']['r2']:.3f})\n")
    md.append(f"- gap vs Primary Pearson: {ctx['gap_corr_pearson']:.3f}, Spearman: {ctx['gap_corr_spearman']:.3f}\n\n")
    md.append("### TmApp Q1–Q4\n\n")
    for q, a in ctx["tm_answers"].items():
        md.append(f"- **{q}:** {a}\n")
    md.append("\n")

    md.append("## D. Matched CV comparison\n\n")
    md.append(f"- TmApp (N={ctx['match_tm'].attrs.get('n', '')}): **{ctx['verdict_tm']}**\n")
    md.append(f"- HIC (N={ctx['match_hi'].attrs.get('n', '')}): **{ctx['verdict_hi']}**\n\n")

    md.append("## DeepResearchで調べるべき技術課題\n\n")
    md.append("### HIC\n")
    for item in ctx["dr_hic"]:
        md.append(f"- {item}\n")
    md.append("\n### TmApp\n")
    for item in ctx["dr_tm"]:
        md.append(f"- {item}\n")

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

    sub = inv_tm.dropna(subset=["Primary_CV_MAE", "Private_MAE"])
    reg_priv = {"intercept": np.nan, "slope": np.nan, "r2": np.nan, "resid_sd": np.nan}
    reg_all = reg_priv.copy()
    if len(sub) >= 5:
        for name, ycol, store in [("Private", "Private_MAE", "reg_priv"), ("AllTest", "All_Test_MAE", "reg_all")]:
            s = inv_tm.dropna(subset=["Primary_CV_MAE", ycol])
            lr = LinearRegression().fit(s["Primary_CV_MAE"].values.reshape(-1, 1), s[ycol].values)
            pred = lr.predict(s["Primary_CV_MAE"].values.reshape(-1, 1))
            ss_res = np.sum((s[ycol].values - pred) ** 2)
            ss_tot = np.sum((s[ycol].values - s[ycol].mean()) ** 2)
            store_dict = store if isinstance(store, dict) else reg_priv
            if name == "Private":
                reg_priv.update({"intercept": lr.intercept_, "slope": lr.coef_[0],
                                 "r2": 1 - ss_res / ss_tot if ss_tot else np.nan,
                                 "resid_sd": float(np.std(s[ycol].values - pred))})
            else:
                reg_all.update({"intercept": lr.intercept_, "slope": lr.coef_[0],
                                "r2": 1 - ss_res / ss_tot if ss_tot else np.nan,
                                "resid_sd": float(np.std(s[ycol].values - pred))})

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
    best_tm = priv_tm.loc[priv_tm["Pearson"].abs().idxmax(), "X"]
    verdict_tm = {"Primary_CV_MAE": "PRIMARY_BETTER", "Shadow_CV_MAE": "SHADOW_BETTER",
                  "CV_MEAN": "MEAN_BETTER", "CV_WORST": "NO_CLEAR_DIFFERENCE"}.get(best_tm, "NO_CLEAR_DIFFERENCE")
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
            f"{'Yes' if hic_case == 'RANKABLE_BUT_COMPRESSED' else 'Partial/mixed'} — pred SD ratio={m_test['pred_sd_over_true_sd']:.3f}, tail bias={m_test['tail_bias']:.3f}"
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
        "TmApp Q1: constant offset？": (
            f"slope={reg_priv['slope']:.3f}, intercept={reg_priv['intercept']:.3f} — "
            f"{'approximately constant offset' if 0.85 <= reg_priv['slope'] <= 1.15 and reg_priv['intercept'] > 0.2 else 'not pure constant offset'}"
        ),
        "TmApp Q2: 強いmodelほどgap大？": (
            f"corr(gap, Primary) Pearson={gap_corr_pearson:.3f} — "
            f"{'weak/no trend' if abs(gap_corr_pearson) < 0.3 else 'positive' if gap_corr_pearson > 0 else 'negative'}"
        ),
        "TmApp Q3: Stage/modality偏り？": (
            f"median gap by stage: {gap_stage.set_index('stage')['median'].to_dict()}; "
            f"最高median modality: {gap_mod.loc[gap_mod['median'].idxmax(), 'modality_bucket']}"
        ),
        "TmApp Q4: target range bias？": (
            f"PRIMARY signed bias varies by quantile bin — see tmapp_target_range_diagnostics.csv"
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
        "tail_pct_median_test": tail_pct_median_test,
        "hic_answers": hic_answers, "tm_answers": tm_answers,
        "reg_priv": reg_priv, "gap_corr_pearson": gap_corr_pearson, "gap_corr_spearman": gap_corr_spearman,
        "match_tm": match_tm, "match_hi": match_hi,
        "verdict_tm": verdict_tm, "verdict_hi": verdict_hi,
        "dr_hic": dr_hic, "dr_tm": dr_tm,
    })

    print("HIC case:", hic_case)
    print("Test top13 tail:", m_test.get("top13_true_tail_count"))
    print("TmApp reg:", reg_priv)
    print("Verdict TmApp/HIC:", verdict_tm, verdict_hi)
    print("ROUND2_GATE0_DIAGNOSTICS_COMPLETE_READY_FOR_DEEP_RESEARCH")


if __name__ == "__main__":
    main()
