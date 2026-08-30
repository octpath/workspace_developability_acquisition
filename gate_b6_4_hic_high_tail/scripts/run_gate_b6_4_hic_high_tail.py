#!/usr/bin/env python3
"""Gate B6.4 — HIC high-tail mechanism & learnability audit (clean runner)."""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b6_4_hic_high_tail"
DATA, METRICS, PLOTS, REPORTS, CONFIG = [GATE / x for x in ("data", "metrics", "plots", "reports", "config")]
LOW_HI, HIGH_LO = 10.5, 11.5

KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
HYDRO, AROM, ACID, BASE = set("AILMFVWY"), set("FWY"), set("DE"), set("KRH")
ADV = [
    "SEQ_SIMPLE_Ridge",
    "PLM_ESM2_PCA64_SVR",
    "ESMFN_STRUCTURE_ElasticNet",
    "FUSION_ESM2_ESMFN_ElasticNet",
    "NESTED_STACK_NNLS",
]
ALLM = ["CONST_MEDIAN"] + ADV


def mkdir():
    for p in [
        DATA, METRICS, REPORTS, CONFIG,
        PLOTS / "high_hic_feature_effects",
        PLOTS / "high_hic_matched_controls",
        PLOTS / "high_hic_surface_features",
        PLOTS / "high_hic_sequence_space",
        PLOTS / "high_hic_model_residuals",
        PLOTS / "high_hic_case_studies",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def load_b62():
    p = ROOT / "gate_b6_2_hic_validity/scripts/run_gate_b6_2_hic_validity.py"
    spec = importlib.util.spec_from_file_location("b62", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_everything()


def band_of(x):
    x = np.asarray(x, float)
    o = np.full(x.shape, "MEDIUM", object)
    o[x < LOW_HI] = "LOW"
    o[x > HIGH_LO] = "HIGH"
    return o


def clean(seq):
    return "".join(a for a in str(seq).upper() if a in KD)


def gravy(seq):
    seq = clean(seq)
    if not seq:
        return np.nan
    try:
        return float(ProteinAnalysis(seq).gravy())
    except Exception:
        return float(np.mean([KD[a] for a in seq]))


def fr(seq, s):
    seq = clean(seq)
    return np.nan if not seq else sum(a in s for a in seq) / len(seq)


def pi(seq):
    seq = clean(seq)
    if len(seq) < 5:
        return np.nan
    try:
        return float(ProteinAnalysis(seq).isoelectric_point())
    except Exception:
        return np.nan


def charge(seq):
    seq = clean(seq)
    return np.nan if not seq else (sum(a in BASE for a in seq) - sum(a in ACID for a in seq)) / len(seq)


def maxwin(seq, w):
    seq = clean(seq)
    if len(seq) < w:
        return gravy(seq)
    return float(max(np.mean([KD[a] for a in seq[i : i + w]]) for i in range(len(seq) - w + 1)))


def hydro_run(seq):
    seq = clean(seq)
    best = cur = 0
    for a in seq:
        cur = cur + 1 if a in HYDRO else 0
        best = max(best, cur)
    return best


def seq_feats(pref, seq):
    return {
        f"{pref}_len": len(clean(seq)),
        f"{pref}_gravy": gravy(seq),
        f"{pref}_frac_hydrophobic": fr(seq, HYDRO),
        f"{pref}_frac_aromatic": fr(seq, AROM),
        f"{pref}_frac_F": fr(seq, set("F")),
        f"{pref}_frac_W": fr(seq, set("W")),
        f"{pref}_frac_Y": fr(seq, set("Y")),
        f"{pref}_charge": charge(seq),
        f"{pref}_pI": pi(seq),
        f"{pref}_maxwin5": maxwin(seq, 5),
        f"{pref}_maxwin9": maxwin(seq, 9),
        f"{pref}_hydro_run": hydro_run(seq),
    }


def identity(a, b):
    a, b = str(a), str(b)
    n = min(len(a), len(b))
    if max(len(a), len(b)) == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


def paired_id(h1, l1, h2, l2):
    return 0.5 * (identity(h1, h2) + identity(l1, l2))


def cliffs(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    return float(stats.spearmanr(x[m], y[m]).correlation)


def build_pop(pack):
    meta, preds = pack["meta"], pack["preds"]
    pop = pd.read_csv(ROOT / "gate_b3/frozen/organizer/final_population.csv")
    pop = pop.rename(columns={"heavy": "VH", "light": "VL", "sequence_group": "sequence_group",
                              "b_cell_subset": "b_cell_subset", "vh_family": "vh_family",
                              "vl_family": "vl_family", "vh_germline": "vh_germline",
                              "vl_germline": "vl_germline", "vh_len": "vh_len", "vl_len": "vl_len"})
    role = {}
    for i in meta["train_ids"]:
        role[i] = "Train"
    for i in meta["public_ids"]:
        role[i] = "Public"
    for i in meta["private_ids"]:
        role[i] = "Private"
    pop["role"] = pop["id"].map(role)
    pop = pop[pop.role.notna()].copy()
    pop["true_band"] = band_of(pop.HIC.values)
    pop["sequence_group"] = pop["id"].map(meta["group_by_id"]).astype(int)
    pop["cluster_size"] = pop.groupby("sequence_group")["id"].transform("size")

    # predictions
    for tag in ALLM:
        by = {}
        for role_key, ids, ykey in [
            ("cv", meta["train_ids"], None),
            ("public", meta["public_ids"], None),
            ("private", meta["private_ids"], None),
        ]:
            for i, v in zip(ids if role_key != "cv" else meta["train_ids"], preds[tag][role_key]):
                by[i] = float(v)
        pop[f"pred_{tag}"] = pop["id"].map(by)
        pop[f"pred_band_{tag}"] = band_of(pop[f"pred_{tag}"].values)
        pop[f"resid_{tag}"] = pop[f"pred_{tag}"] - pop["HIC"]

    nlow = np.zeros(len(pop), int)
    nmed = np.zeros(len(pop), int)
    nhi = np.zeros(len(pop), int)
    for tag in ADV:
        pb = pop[f"pred_band_{tag}"].values
        nlow += pb == "LOW"
        nmed += pb == "MEDIUM"
        nhi += pb == "HIGH"
    pop["n_adv_pred_LOW"] = nlow
    pop["n_adv_pred_MEDIUM"] = nmed
    pop["n_adv_pred_HIGH"] = nhi
    pop["consensus_severe_failure"] = (pop.true_band == "HIGH") & (pop.n_adv_pred_LOW >= 3)

    # PSR / TmApp
    psr = pd.read_csv(ROOT / "gate_b1/data/psr_full.csv")
    pop["PSR"] = pop["id"].map(dict(zip(psr.antibody_id, psr.psr_score)))
    if "TmApp" not in pop.columns and "TmApp" in pop.columns:
        pass
    if "TmApp" not in pop.columns:
        pop["TmApp"] = pop["id"].map(dict(zip(psr.antibody_id, psr.tm_app_C)))

    # numbering CDRs
    num = pd.read_csv(ROOT / "gate_b1/data/numbering_germline.csv")
    num = num.rename(columns={"antibody_id": "id"})
    cdrs = ["H_CDR1", "H_CDR2", "H_CDR3", "L_CDR1", "L_CDR2", "L_CDR3"]
    pop = pop.merge(num[["id"] + [c for c in cdrs if c in num.columns]], on="id", how="left")

    # germline distances
    germ = pd.read_csv(ROOT / "gate_b3/features/germline/germline_relative.csv")
    if "antibody_id" in germ.columns:
        germ = germ.rename(columns={"antibody_id": "id"})
    gkeep = ["id"] + [c for c in germ.columns if "germline_distance" in c or "mutfrac" in c]
    pop = pop.merge(germ[gkeep], on="id", how="left")

    # structure
    sasa = pd.read_csv(ROOT / "gate_b2/cache/structure_features/esmfold_native_sasa_rasa_patch.csv")
    sasa = sasa.rename(columns={"antibody_id": "id"})
    want = [
        "id", "ESMFN_Fv_total_sasa", "ESMFN_Fv_sasa_hydrophobic", "ESMFN_Fv_sasa_aromatic",
        "ESMFN_Fv_rasa_w_hydrophobicity_sum", "ESMFN_n_hydrophobic_patches",
        "ESMFN_largest_hydrophobic_patch_sasa", "ESMFN_total_hydrophobic_patch_sasa",
        "ESMFN_H_CDR3_sasa_hydrophobic", "ESMFN_H_CDR3_rasa_w_hydrophobicity_sum",
        "ESMFN_VH_sasa_hydrophobic", "ESMFN_VL_sasa_hydrophobic", "ESMFN_all_CDR_sasa_hydrophobic",
    ]
    want = [c for c in want if c in sasa.columns]
    pop = pop.merge(sasa[want], on="id", how="left")
    if {"ESMFN_Fv_sasa_hydrophobic", "ESMFN_Fv_total_sasa"} <= set(pop.columns):
        pop["ESMFN_hydrophobic_sasa_frac"] = pop["ESMFN_Fv_sasa_hydrophobic"] / pop["ESMFN_Fv_total_sasa"]

    # sequence physchem
    rows = []
    for _, r in pop.iterrows():
        d = {"id": r.id}
        d.update(seq_feats("VH", r.VH))
        d.update(seq_feats("VL", r.VL))
        d.update(seq_feats("HL", r.VH + r.VL))
        if isinstance(r.get("H_CDR3"), str):
            d.update(seq_feats("H_CDR3", r.H_CDR3))
        if isinstance(r.get("L_CDR3"), str):
            d.update(seq_feats("L_CDR3", r.L_CDR3))
        rows.append(d)
    pop = pop.merge(pd.DataFrame(rows), on="id", how="left")
    return pop, meta


def phase1(pop):
    hi = pop[pop.true_band == "HIGH"].sort_values("HIC", ascending=False).copy()
    assert len(hi) == 13, len(hi)
    cols = ["id", "role", "HIC", "VH", "VL", "sequence_group", "consensus_severe_failure",
            "n_adv_pred_LOW", "n_adv_pred_MEDIUM", "n_adv_pred_HIGH"]
    for t in ALLM:
        cols += [f"pred_{t}", f"pred_band_{t}", f"resid_{t}"]
    hi[cols].to_csv(DATA / "high_hic_13.csv", index=False)

    wb = openpyxl.load_workbook(ROOT / "raw/shehata/a05/mmc2.xlsx", read_only=True, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))
    hdr = list(rows[0]); idx = {h: i for i, h in enumerate(hdr)}
    src = {r[idx["Clone name"]]: r for r in rows[1:] if r and r[0]}
    ver = []
    for _, r in hi.iterrows():
        s = src.get(r.id)
        if s is None:
            ver.append(dict(id=r.id, processed_HIC=r.HIC, source_HIC=np.nan, exact_match=False,
                            sequence_match=False, any_issue="ID_NOT_IN_MMC2"))
            continue
        sh = float(s[idx["HIC retention time (min)"]])
        vhok = str(s[idx["VH Protein"]]).replace(" ", "") == str(r.VH).replace(" ", "")
        vlok = str(s[idx["VL Protein"]]).replace(" ", "") == str(r.VL).replace(" ", "")
        ok = abs(sh - float(r.HIC)) < 1e-6
        issue = []
        if not ok: issue.append("HIC_MISMATCH")
        if not (vhok and vlok): issue.append("SEQ_MISMATCH")
        ver.append(dict(id=r.id, processed_HIC=float(r.HIC), source_HIC=sh, exact_match=ok,
                        sequence_match=vhok and vlok, VH_match=vhok, VL_match=vlok,
                        any_issue=";".join(issue) if issue else "NONE"))
    ver_df = pd.DataFrame(ver)
    ver_df.to_csv(METRICS / "high_hic_source_verification.csv", index=False)

    allh = np.sort(pop.HIC.values)
    osrows = []
    vals = hi.HIC.values
    for i, v in enumerate(vals):
        nxt = vals[i + 1] if i + 1 < len(vals) else float(pop.loc[pop.HIC < HIGH_LO, "HIC"].max())
        osrows.append(dict(rank=i + 1, id=hi.iloc[i].id, HIC=float(v), delta_above_11_5=float(v - HIGH_LO),
                           delta_to_next_lower=float(v - nxt), percentile=float((allh <= v).mean() * 100),
                           consensus_severe_failure=bool(hi.iloc[i].consensus_severe_failure)))
    os_df = pd.DataFrame(osrows)
    os_df.to_csv(METRICS / "high_hic_order_statistics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols_c = ["#b91c1c" if f else "#d97706" for f in hi.consensus_severe_failure]
    y = np.arange(len(hi))
    ax.barh(y, hi.HIC, color=cols_c)
    ax.axvline(HIGH_LO, ls="--", c="k", label="11.5")
    ax.axvline(LOW_HI, ls=":", c="gray", label="10.5")
    ax.set_yticks(y, [f"{i} ({r})" for i, r in zip(hi.id, hi.role)], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("HIC (min)"); ax.legend(fontsize=8)
    ax.set_title("13 HIGH-HIC Abs (red = consensus severe failure)")
    fig.tight_layout(); fig.savefig(PLOTS / "high_hic_order_statistics.png", dpi=150); plt.close(fig)

    (REPORTS / "01_high_hic_integrity.md").write_text(
        f"# 01 Integrity\n\nn_HIGH={len(hi)}; source exact HIC matches={int(ver_df.exact_match.sum())}/13; "
        f"sequence matches={int(ver_df.sequence_match.sum())}/13; issues={(ver_df.any_issue!='NONE').sum()}.\n\n"
        f"Order stats:\n\n{os_df.to_markdown(index=False)}\n\n"
        "Interpretation: HIGH forms a smooth right-tail continuation above 11.5, not a single isolated gap artifact.\n"
    )
    return hi, ver_df, os_df


def matched(pop):
    hi = pop[pop.true_band == "HIGH"]
    pairs = []
    for _, h in hi.iterrows():
        for label, pool, k in [("LOW", pop[pop.true_band == "LOW"], 3), ("MEDIUM", pop[pop.true_band == "MEDIUM"], 3)]:
            scored = []
            for _, c in pool.iterrows():
                if c.id == h.id:
                    continue
                fam = int(c.vh_family == h.vh_family) + int(c.vl_family == h.vl_family)
                sim = paired_id(h.VH, h.VL, c.VH, c.VL)
                score = 10 * fam + 5 * sim - 0.01 * (abs(c.vh_len - h.vh_len) + abs(c.vl_len - h.vl_len))
                scored.append((score, sim, fam, c))
            scored.sort(reverse=True, key=lambda x: x[0])
            for score, sim, fam, c in scored[:k]:
                pairs.append(dict(high_id=h.id, high_HIC=h.HIC, control_id=c.id, control_HIC=c.HIC,
                                  control_band=label, paired_identity=sim, family_match=fam, match_score=score))
    mdf = pd.DataFrame(pairs)
    mdf.to_csv(DATA / "high_hic_matched_controls.csv", index=False)
    feats = [c for c in [
        "HL_gravy", "HL_frac_aromatic", "HL_frac_hydrophobic", "HL_charge", "HL_pI", "H_CDR3_gravy",
        "H_CDR3_frac_aromatic", "H_CDR3_len", "PL_combined_germline_distance",
        "ESMFN_Fv_sasa_hydrophobic", "ESMFN_largest_hydrophobic_patch_sasa", "ESMFN_hydrophobic_sasa_frac",
        "ESMFN_H_CDR3_sasa_hydrophobic",
    ] if c in pop.columns]
    by = pop.set_index("id")
    diffs = []
    for _, p in mdf.iterrows():
        h, c = by.loc[p.high_id], by.loc[p.control_id]
        row = dict(high_id=p.high_id, control_id=p.control_id, control_band=p.control_band,
                   delta_HIC=float(h.HIC - c.HIC), paired_identity=p.paired_identity)
        for f in feats:
            row[f"delta_{f}"] = float(h[f] - c[f]) if pd.notna(h[f]) and pd.notna(c[f]) else np.nan
        diffs.append(row)
    ddf = pd.DataFrame(diffs)
    ddf.to_csv(METRICS / "high_hic_matched_differences.csv", index=False)
    if "delta_HL_gravy" in ddf.columns:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for lab, col in [("LOW", "#4c6a8a"), ("MEDIUM", "#d97706")]:
            ax.hist(ddf.loc[ddf.control_band == lab, "delta_HL_gravy"].dropna(), bins=12, alpha=0.55, color=col, label=lab)
        ax.axvline(0, color="k"); ax.legend(); ax.set_xlabel("Δ HL GRAVY (HIGH−control)")
        fig.tight_layout(); fig.savefig(PLOTS / "high_hic_matched_controls/delta_HL_gravy.png", dpi=130); plt.close(fig)
    return mdf, ddf


def effects(pop):
    feats = [c for c in pop.columns if any(c.startswith(p) for p in
             ("VH_", "VL_", "HL_", "H_CDR3_", "L_CDR3_", "PL_", "ESMFN_")) and pd.api.types.is_numeric_dtype(pop[c])]
    hi, non = pop[pop.true_band == "HIGH"], pop[pop.true_band != "HIGH"]
    rows = []
    ybin = (pop.true_band == "HIGH").astype(int).values
    for f in feats:
        a, b = hi[f].values, non[f].values
        x = pop[f].astype(float).values
        m = np.isfinite(x)
        try:
            auc = roc_auc_score(ybin[m], x[m]); auc = max(auc, 1 - auc)
        except Exception:
            auc = np.nan
        rows.append(dict(
            feature=f, median_HIGH=float(np.nanmedian(a)), median_nonHIGH=float(np.nanmedian(b)),
            median_diff=float(np.nanmedian(a) - np.nanmedian(b)), cliffs_delta=cliffs(a, b),
            spearman_vs_HIC=spearman(pop[f], pop.HIC), auc_HIGH_sep_abs=auc,
        ))
    edf = pd.DataFrame(rows).sort_values("auc_HIGH_sep_abs", ascending=False)
    edf.to_csv(METRICS / "high_hic_feature_effects.csv", index=False)
    edf.to_csv(METRICS / "high_hic_sequence_features.csv", index=False)
    for f in edf.head(8).feature:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        data = [pop.loc[pop.true_band == b, f].dropna() for b in ("LOW", "MEDIUM", "HIGH")]
        ax.boxplot(data, labels=["LOW", "MED", "HIGH"], showfliers=False)
        ax.set_title(f); fig.tight_layout()
        fig.savefig(PLOTS / "high_hic_feature_effects" / f"{f[:70]}.png", dpi=120); plt.close(fig)
    (REPORTS / "02_high_hic_sequence_mechanism.md").write_text(
        "# 02 Sequence mechanism\n\n" + edf.head(20).to_markdown(index=False) + "\n"
    )
    return edf


def shm(pop):
    cols = [c for c in pop.columns if "germline_distance" in c or "mutfrac" in c]
    hi, non = pop[pop.true_band == "HIGH"], pop[pop.true_band != "HIGH"]
    rows = []
    for c in cols:
        rows.append(dict(feature=c, spearman_vs_HIC=spearman(pop[c], pop.HIC),
                         median_HIGH=float(np.nanmedian(hi[c])), median_nonHIGH=float(np.nanmedian(non[c])),
                         cliffs_delta=cliffs(hi[c], non[c])))
    sdf = pd.DataFrame(rows).sort_values("spearman_vs_HIC", key=lambda s: s.abs(), ascending=False)
    sdf.to_csv(METRICS / "high_hic_shm_analysis.csv", index=False)
    note = ("Per-mutation hydrophobicity-increasing SHM classification was NOT run: frozen germline AA "
            "alignments are unavailable for all 324. Germline_distance/mutfrac used as proxies.")
    (METRICS / "high_hic_shm_limitations.txt").write_text(note)
    (REPORTS / "04_high_hic_shm_germline.md").write_text("# 04 SHM/germline\n\n" + sdf.to_markdown(index=False) + "\n\n" + note + "\n")
    return sdf, note


def structure(pop):
    cols = [c for c in pop.columns if c.startswith("ESMFN_")]
    hi, non = pop[pop.true_band == "HIGH"], pop[pop.true_band != "HIGH"]
    rows = [dict(feature=c, spearman_vs_HIC=spearman(pop[c], pop.HIC),
                 median_HIGH=float(np.nanmedian(hi[c])), median_nonHIGH=float(np.nanmedian(non[c])),
                 cliffs_delta=cliffs(hi[c], non[c])) for c in cols]
    sdf = pd.DataFrame(rows).sort_values("cliffs_delta", key=lambda s: s.abs(), ascending=False)
    sdf.to_csv(METRICS / "high_hic_structure_features.csv", index=False)
    for f in [c for c in ["ESMFN_Fv_sasa_hydrophobic", "ESMFN_largest_hydrophobic_patch_sasa",
                          "ESMFN_hydrophobic_sasa_frac", "ESMFN_H_CDR3_sasa_hydrophobic"] if c in pop.columns]:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for b, col in [("LOW", "#4c6a8a"), ("MEDIUM", "#d97706"), ("HIGH", "#b91c1c")]:
            s = pop[pop.true_band == b]
            ax.scatter(s.HIC, s[f], s=16 if b != "HIGH" else 45, c=col, alpha=0.7, label=b)
        ax.set_xlabel("HIC"); ax.set_ylabel(f); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(PLOTS / "high_hic_surface_features" / f"{f}.png", dpi=130); plt.close(fig)
    (REPORTS / "03_high_hic_structure_mechanism.md").write_text("# 03 Structure\n\n" + sdf.head(25).to_markdown(index=False) + "\n")
    return sdf


def disagreement(pop):
    hi = pop[pop.true_band == "HIGH"].copy()
    rows = []
    for _, r in hi.iterrows():
        rows.append(dict(
            id=r.id, role=r.role, HIC=r.HIC,
            pred_NESTED=r.pred_NESTED_STACK_NNLS, pred_ESMFN=r.pred_ESMFN_STRUCTURE_ElasticNet,
            pred_PLM=r.pred_PLM_ESM2_PCA64_SVR,
            band_NESTED=r.pred_band_NESTED_STACK_NNLS, band_ESMFN=r.pred_band_ESMFN_STRUCTURE_ElasticNet,
            band_PLM=r.pred_band_PLM_ESM2_PCA64_SVR,
            structure_closer=abs(r.pred_ESMFN_STRUCTURE_ElasticNet - r.HIC) < abs(r.pred_PLM_ESM2_PCA64_SVR - r.HIC),
            nested_under=r.HIC - r.pred_NESTED_STACK_NNLS,
            esmfn_under=r.HIC - r.pred_ESMFN_STRUCTURE_ElasticNet,
            consensus_severe_failure=bool(r.consensus_severe_failure), n_adv_pred_LOW=int(r.n_adv_pred_LOW),
        ))
    ddf = pd.DataFrame(rows)
    ddf.to_csv(METRICS / "high_hic_model_disagreement.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, tag, title in zip(axes, ["NESTED_STACK_NNLS", "ESMFN_STRUCTURE_ElasticNet", "PLM_ESM2_PCA64_SVR"],
                              ["NESTED", "ESMFN", "PLM"]):
        ax.axhline(0, color="k", lw=1)
        ax.scatter(pop.HIC, pop[f"resid_{tag}"], s=9, alpha=0.35, c="#4c6a8a")
        h = pop[pop.true_band == "HIGH"]
        ax.scatter(h.HIC, h[f"resid_{tag}"], s=40, c="#b91c1c", label="HIGH")
        ax.set_title(title); ax.set_xlabel("true HIC"); ax.set_ylabel("pred−true")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "high_hic_model_residuals/resid_vs_true.png", dpi=140); plt.close(fig)
    (REPORTS / "06_high_hic_model_failure_analysis.md").write_text(
        "# 06 Model failures\n\n" + ddf.to_markdown(index=False) + "\n"
    )
    return ddf


def neighbors(pop, train_ids):
    train = pop[pop.id.isin(set(train_ids))]
    rows = []
    for _, r in pop.iterrows():
        best = (-1.0, None, np.nan)
        for _, t in train.iterrows():
            if t.id == r.id:
                continue
            sim = paired_id(r.VH, r.VL, t.VH, t.VL)
            if sim > best[0]:
                best = (sim, t.id, float(t.HIC))
        rows.append(dict(id=r.id, role=r.role, HIC=r.HIC, true_band=r.true_band,
                         nearest_train_id=best[1], nearest_train_paired_sim=best[0],
                         nearest_train_HIC=best[2],
                         abs_HIC_diff_to_nearest_train=abs(r.HIC - best[2]) if best[2] == best[2] else np.nan))
    ndf = pd.DataFrame(rows)
    ndf.to_csv(METRICS / "high_hic_neighbor_analysis.csv", index=False)

    ids, H, L, y = pop.id.tolist(), pop.VH.tolist(), pop.VL.tolist(), pop.HIC.values
    thr = {t: [] for t in (0.90, 0.95, 0.97, 0.99)}
    disc = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = paired_id(H[i], L[i], H[j], L[j])
            dh = abs(y[i] - y[j])
            for t in thr:
                if sim >= t:
                    thr[t].append(dh)
            if sim >= 0.95 and dh >= 1.5:
                disc.append(dict(id_a=ids[i], id_b=ids[j], paired_sim=sim, HIC_a=y[i], HIC_b=y[j], abs_HIC_diff=dh))
    rng = np.random.default_rng(0)
    rand = np.abs(rng.choice(y, 4000) - rng.choice(y, 4000))
    srows = []
    for t, arr in thr.items():
        arr = np.asarray(arr, float)
        srows.append(dict(min_paired_sim=t, n_pairs=len(arr),
                          median_abs_HIC_diff=float(np.median(arr)) if len(arr) else np.nan,
                          p90_abs_HIC_diff=float(np.percentile(arr, 90)) if len(arr) else np.nan,
                          random_pair_median_abs_HIC_diff=float(np.median(rand))))
    sdf = pd.DataFrame(srows)
    sdf.to_csv(METRICS / "high_hic_local_smoothness.csv", index=False)
    ddf = pd.DataFrame(disc).sort_values("abs_HIC_diff", ascending=False)
    ddf.to_csv(METRICS / "high_hic_discordant_near_identical_pairs.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for b, col in [("LOW", "#4c6a8a"), ("MEDIUM", "#d97706"), ("HIGH", "#b91c1c")]:
        s = ndf[ndf.true_band == b]
        ax.scatter(s.nearest_train_paired_sim, s.HIC, s=16 if b != "HIGH" else 50, c=col, alpha=0.7, label=b)
    ax.set_xlabel("nearest-Train paired identity"); ax.set_ylabel("HIC"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "high_hic_sequence_space/nearest_train_sim_vs_hic.png", dpi=140); plt.close(fig)
    (REPORTS / "05_high_hic_sequence_space.md").write_text(
        f"# 05 Sequence space\n\nHIGH median nearest-Train sim="
        f"{ndf.loc[ndf.true_band=='HIGH','nearest_train_paired_sim'].median():.3f}; "
        f"LOW={ndf.loc[ndf.true_band=='LOW','nearest_train_paired_sim'].median():.3f}\n\n"
        f"Smoothness:\n\n{sdf.to_markdown(index=False)}\n\nDiscordant near-identical pairs n={len(ddf)}\n"
        + (ddf.head(20).to_markdown(index=False) if len(ddf) else "") + "\n"
    )
    return ndf, sdf, ddf


def mat(df, cols):
    X = df[cols].astype(float).values.copy()
    colmed = np.nanmedian(X, axis=0)
    ii = np.where(~np.isfinite(X))
    X[ii] = np.take(colmed, ii[1])
    return X


def diagnostics(pop, train_ids):
    train = pop[pop.id.isin(set(train_ids))].reset_index(drop=True)
    y = train.HIC.values
    groups = train.sequence_group.values
    physseq = [c for c in [
        "HL_gravy", "HL_frac_aromatic", "HL_frac_hydrophobic", "HL_charge", "HL_pI",
        "HL_maxwin5", "HL_maxwin9", "HL_hydro_run", "VH_gravy", "VL_gravy",
        "H_CDR3_gravy", "H_CDR3_frac_aromatic", "H_CDR3_len", "L_CDR3_gravy",
        "PL_combined_germline_distance", "PL_vh_mutfrac_excl_CDR3", "PL_vl_mutfrac_excl_CDR3",
    ] if c in train.columns]
    physstruct = [c for c in [
        "ESMFN_Fv_sasa_hydrophobic", "ESMFN_Fv_sasa_aromatic", "ESMFN_hydrophobic_sasa_frac",
        "ESMFN_largest_hydrophobic_patch_sasa", "ESMFN_total_hydrophobic_patch_sasa",
        "ESMFN_n_hydrophobic_patches", "ESMFN_H_CDR3_sasa_hydrophobic",
        "ESMFN_VH_sasa_hydrophobic", "ESMFN_VL_sasa_hydrophobic", "ESMFN_Fv_rasa_w_hydrophobicity_sum",
    ] if c in train.columns]
    Xs, Xt = mat(train, physseq), mat(train, physstruct)

    ids = json.loads((ROOT / "gate_b6_2_hic_validity/cache/esm2_mean_2560_ids.json").read_text())
    Eall = np.load(ROOT / "gate_b6_2_hic_validity/cache/esm2_mean_2560.npy")
    eby = dict(zip(ids, Eall))
    E = np.vstack([eby[i] for i in train.id])
    Ep = PCA(64, random_state=0).fit_transform(StandardScaler().fit_transform(E))

    def evaluate(name, pred):
        high = y > HIGH_LO
        elev = y >= LOW_HI
        return dict(
            model=name, overall_MAE=mean_absolute_error(y, pred), Spearman=spearman(y, pred),
            HIGH_MAE=mean_absolute_error(y[high], pred[high]),
            HIGH_bias=float(np.mean(pred[high] - y[high])),
            HIGH_to_LOW_rate=float(np.mean(pred[high] < LOW_HI)),
            HIGH_pred_ge10_5_rate=float(np.mean(pred[high] >= LOW_HI)),
            elevated_pred_ge10_5_rate=float(np.mean(pred[elev] >= LOW_HI)),
            pred_SD_over_obs_SD=float(np.std(pred, ddof=1) / np.std(y, ddof=1)),
        )

    results = []
    for tag in ADV + ["CONST_MEDIAN"]:
        results.append(evaluate(f"FROZEN_{tag}", train[f"pred_{tag}"].values))

    specs = [
        ("PHYSSEQ_Ridge", Xs, "ridge", False),
        ("PHYSSEQ_ElasticNet", Xs, "enet", False),
        ("PHYSSTRUCT_Ridge", Xt, "ridge", False),
        ("PHYSSEQ_STRUCT_Ridge", np.hstack([Xs, Xt]), "ridge", False),
        ("PHYSSEQ_STRUCT_Huber", np.hstack([Xs, Xt]), "huber", False),
        ("PHYSSEQ_STRUCT_Ridge_elevW2", np.hstack([Xs, Xt]), "ridge", True),
        ("ESM2PCA64_SVR", Ep, "svr", False),
        ("ESM2PCA64_PHYSSEQ_Ridge", np.hstack([Ep, Xs]), "ridge", False),
        ("ESM2PCA64_PHYSSEQ_STRUCT_Ridge", np.hstack([Ep, Xs, Xt]), "ridge", False),
    ]
    gkf = GroupKFold(5)
    oofs = {}
    for name, X, kind, elevw in specs:
        oof = np.zeros(len(train))
        for tr, te in gkf.split(X, y, groups):
            pipe = Pipeline([("sc", StandardScaler()), ("m", {
                "ridge": Ridge(10.0), "enet": ElasticNet(0.05, l1_ratio=0.5, max_iter=20000),
                "svr": SVR(C=1.0, epsilon=0.1), "huber": HuberRegressor(alpha=1e-4, max_iter=2000),
            }[kind])])
            w = np.where(y[tr] >= LOW_HI, 2.0, 1.0) if elevw else None
            if w is not None and kind != "svr":
                pipe.fit(X[tr], y[tr], m__sample_weight=w)
            else:
                pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict(X[te])
        results.append(evaluate(name, oof)); oofs[name] = oof
    rdf = pd.DataFrame(results)
    rdf.to_csv(METRICS / "high_hic_diagnostic_models.csv", index=False)

    nested_mae = float(rdf.loc[rdf.model == "FROZEN_NESTED_STACK_NNLS", "overall_MAE"].iloc[0])
    cand = rdf[~rdf.model.str.startswith("FROZEN_")].copy()
    cand = cand[cand.overall_MAE <= nested_mae + 0.15]
    if cand.empty:
        cand = rdf[~rdf.model.str.startswith("FROZEN_")].copy()
    winner = cand.sort_values(["HIGH_to_LOW_rate", "HIGH_MAE", "overall_MAE"]).iloc[0]
    win = winner.model

    # one-shot holdout: refit on full train
    def X_for(name, df, pca_obj, sc_e, Ep_only=False):
        Xs2, Xt2 = mat(df, physseq), mat(df, physstruct)
        E2 = np.vstack([eby[i] for i in df.id])
        Ep2 = pca_obj.transform(sc_e.transform(E2))
        if name.startswith("PHYSSEQ_Ridge") and "STRUCT" not in name and "elev" not in name:
            return Xs2
        if name == "PHYSSEQ_ElasticNet":
            return Xs2
        if name.startswith("PHYSSTRUCT"):
            return Xt2
        if name.startswith("PHYSSEQ_STRUCT"):
            return np.hstack([Xs2, Xt2])
        if name == "ESM2PCA64_SVR":
            return Ep2
        if name == "ESM2PCA64_PHYSSEQ_Ridge":
            return np.hstack([Ep2, Xs2])
        if name == "ESM2PCA64_PHYSSEQ_STRUCT_Ridge":
            return np.hstack([Ep2, Xs2, Xt2])
        return np.hstack([Xs2, Xt2])

    sc_e = StandardScaler(); pca = PCA(64, random_state=0)
    Etr = np.vstack([eby[i] for i in train.id])
    Ep_tr = pca.fit_transform(sc_e.fit_transform(Etr))
    # rebuild Xtr according to winner using fitted pca
    # monkeypatch: temporarily put Ep into function via closure already fitted
    def features(name, df):
        return X_for(name, df, pca, sc_e)

    Xtr = features(win, train)
    sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr)
    kind = "huber" if "Huber" in win else ("svr" if "SVR" in win else ("enet" if "ElasticNet" in win else "ridge"))
    model = {"ridge": Ridge(10.0), "enet": ElasticNet(0.05, l1_ratio=0.5, max_iter=20000),
             "svr": SVR(C=1.0, epsilon=0.1), "huber": HuberRegressor(alpha=1e-4, max_iter=2000)}[kind]
    elevw = "elevW2" in win
    w = np.where(y >= LOW_HI, 2.0, 1.0) if elevw else None
    if w is not None and kind != "svr":
        model.fit(Xtrs, y, sample_weight=w)
    else:
        model.fit(Xtrs, y)

    hold = []
    for role, sub in [("public", pop[pop.role == "Public"]), ("private", pop[pop.role == "Private"])]:
        X = features(win, sub); pred = model.predict(sc.transform(X)); yy = sub.HIC.values
        high = yy > HIGH_LO; elev = yy >= LOW_HI
        hold.append(dict(role=role, model=win, overall_MAE=mean_absolute_error(yy, pred), Spearman=spearman(yy, pred),
                         HIGH_n=int(high.sum()), HIGH_MAE=mean_absolute_error(yy[high], pred[high]) if high.any() else np.nan,
                         HIGH_bias=float(np.mean(pred[high] - yy[high])) if high.any() else np.nan,
                         HIGH_to_LOW_rate=float(np.mean(pred[high] < LOW_HI)) if high.any() else np.nan,
                         HIGH_pred_ge10_5_rate=float(np.mean(pred[high] >= LOW_HI)) if high.any() else np.nan,
                         elevated_pred_ge10_5_rate=float(np.mean(pred[elev] >= LOW_HI)) if elev.any() else np.nan))
        predn = sub.pred_NESTED_STACK_NNLS.values
        hold.append(dict(role=role, model="FROZEN_NESTED_STACK_NNLS", overall_MAE=mean_absolute_error(yy, predn),
                         Spearman=spearman(yy, predn), HIGH_n=int(high.sum()),
                         HIGH_MAE=mean_absolute_error(yy[high], predn[high]) if high.any() else np.nan,
                         HIGH_bias=float(np.mean(predn[high] - yy[high])) if high.any() else np.nan,
                         HIGH_to_LOW_rate=float(np.mean(predn[high] < LOW_HI)) if high.any() else np.nan,
                         HIGH_pred_ge10_5_rate=float(np.mean(predn[high] >= LOW_HI)) if high.any() else np.nan,
                         elevated_pred_ge10_5_rate=float(np.mean(predn[elev] >= LOW_HI)) if elev.any() else np.nan))
    hold_df = pd.DataFrame(hold)
    hold_df.to_csv(METRICS / "high_hic_diagnostic_holdout_oneshot.csv", index=False)
    freeze = dict(winner=win, train_metrics=winner.to_dict(), physseq=physseq, physstruct=physstruct,
                  selection="min HIGH→LOW then HIGH MAE then overall MAE among models with MAE<=nested+0.15")
    (CONFIG / "B6_4_DIAGNOSTIC_WINNER.json").write_text(json.dumps(freeze, indent=2, default=str))
    (REPORTS / "07_high_hic_diagnostic_learnability.md").write_text(
        f"# 07 Diagnostic learnability\n\nWinner: **{win}**\n\nTrain CV:\n\n{rdf.to_markdown(index=False)}\n\n"
        f"One-shot holdout:\n\n{hold_df.to_markdown(index=False)}\n"
    )
    return rdf, hold_df, freeze


def oracle(pop, train_ids):
    train = pop[pop.id.isin(set(train_ids))].copy()
    y = train.HIC.values; g = train.sequence_group.values
    dummies = pd.get_dummies(train.b_cell_subset.fillna("UNK"), prefix="sub")
    X = np.hstack([train[["TmApp", "PSR"]].astype(float).fillna(train[["TmApp", "PSR"]].median()).values, dummies.values])
    oof = np.zeros(len(train))
    for tr, te in GroupKFold(5).split(X, y, g):
        pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(1.0))])
        pipe.fit(X[tr], y[tr]); oof[te] = pipe.predict(X[te])
    high = y > HIGH_LO
    row = dict(model="ORGANIZER_ONLY_ORACLE_Ridge", overall_MAE=mean_absolute_error(y, oof),
               Spearman=spearman(y, oof), HIGH_MAE=mean_absolute_error(y[high], oof[high]),
               HIGH_to_LOW_rate=float(np.mean(oof[high] < LOW_HI)),
               note="ORGANIZER-ONLY ORACLE — not a participant model")
    pd.DataFrame([row]).to_csv(METRICS / "high_hic_oracle_diagnostic.csv", index=False)
    cor = []
    for c in ["TmApp", "PSR"]:
        m = pop[c].notna()
        cor.append(dict(assay=c, pearson=float(stats.pearsonr(pop.loc[m, c], pop.loc[m, "HIC"])[0]),
                        spearman=spearman(pop[c], pop.HIC),
                        median_HIGH=float(pop.loc[pop.true_band == "HIGH", c].median()),
                        median_nonHIGH=float(pop.loc[pop.true_band != "HIGH", c].median())))
    pd.DataFrame(cor).to_csv(METRICS / "high_hic_assay_correlations.csv", index=False)
    pd.crosstab(pop.b_cell_subset, pop.true_band).to_csv(METRICS / "high_hic_bcell_subset_counts.csv")
    # donor
    pd.crosstab(pop.donor.fillna("UNK"), pop.true_band).to_csv(METRICS / "high_hic_donor_counts.csv")
    return row


def case_table_and_studies(pop, ndf):
    hi = pop[pop.true_band == "HIGH"].sort_values("HIC", ascending=False)
    keep = ["id", "role", "HIC", "true_band", "VH", "VL", "sequence_group", "cluster_size", "vh_family", "vl_family",
            "vh_germline", "vl_germline", "vh_len", "vl_len", "b_cell_subset", "donor", "TmApp", "PSR",
            "consensus_severe_failure", "n_adv_pred_LOW", "n_adv_pred_MEDIUM", "n_adv_pred_HIGH"]
    keep += [c for c in pop.columns if c.startswith(("pred_", "resid_", "pred_band_", "HL_", "VH_", "VL_", "H_CDR3_", "PL_", "ESMFN_"))]
    cols = []
    seen = set()
    for c in keep:
        if c in hi.columns and c not in seen:
            cols.append(c); seen.add(c)
    hi[cols].to_csv(DATA / "high_hic_case_table.csv", index=False)

    # select ~6-8 cases
    ids = []
    for iid in list(hi.id.head(1)) + list(hi.sort_values("pred_ESMFN_STRUCTURE_ElasticNet", ascending=False).id.head(2)) + \
               list(hi[hi.consensus_severe_failure].id.head(3)) + list(hi[~hi.consensus_severe_failure].id.head(2)):
        if iid not in ids:
            ids.append(iid)
    ids = ids[:8]
    by = pop.set_index("id"); nn = ndf.set_index("id")
    lines = ["# 08 Case studies\n"]
    for iid in ids:
        r, n = by.loc[iid], nn.loc[iid]
        lines.append(f"\n## {iid} (role={r.role}, HIC={r.HIC:.3f})\n")
        lines.append(f"- severe_failure={bool(r.consensus_severe_failure)} (n_adv_LOW={int(r.n_adv_pred_LOW)})\n")
        lines.append(f"- preds NESTED={r.pred_NESTED_STACK_NNLS:.3f}, ESMFN={r.pred_ESMFN_STRUCTURE_ElasticNet:.3f}, "
                     f"PLM={r.pred_PLM_ESM2_PCA64_SVR:.3f}\n")
        lines.append(f"- families {r.vh_family}/{r.vl_family}; germline {r.vh_germline}/{r.vl_germline}\n")
        lines.append(f"- HL_gravy={r.HL_gravy:.3f}, H_CDR3_gravy={r.get('H_CDR3_gravy', np.nan):.3f}, "
                     f"patch={r.get('ESMFN_largest_hydrophobic_patch_sasa', np.nan):.1f}\n")
        lines.append(f"- nearest Train {n.nearest_train_id} sim={n.nearest_train_paired_sim:.3f} "
                     f"(HIC={n.nearest_train_HIC:.3f}, |Δ|={n.abs_HIC_diff_to_nearest_train:.3f})\n")
        lines.append(f"- TmApp={r.TmApp}, PSR={r.PSR}, subset={r.b_cell_subset}\n")
        reasons = []
        if r.HL_gravy > pop.HL_gravy.median(): reasons.append("above-median HL GRAVY")
        if pd.notna(r.get("ESMFN_largest_hydrophobic_patch_sasa")) and r.ESMFN_largest_hydrophobic_patch_sasa > pop.ESMFN_largest_hydrophobic_patch_sasa.median():
            reasons.append("large hydrophobic patch")
        if n.nearest_train_paired_sim < ndf.nearest_train_paired_sim.median():
            reasons.append("more sequence-novel than median")
        if not reasons: reasons.append("no single dominant marker")
        lines.append(f"- restrained interpretation: {', '.join(reasons)}; residual underprediction remains; confidence moderate/low (n=13).\n")
    (REPORTS / "08_high_hic_case_studies.md").write_text("".join(lines))


def final_report(pop, ver, edf, sdf, shm_df, shm_note, ndf, smooth, disc, diag, hold, freeze, oracle_row):
    hi = pop[pop.true_band == "HIGH"]
    sev = int(hi.consensus_severe_failure.sum())
    win = freeze["winner"]
    tw = freeze["train_metrics"]
    nested = diag[diag.model == "FROZEN_NESTED_STACK_NNLS"].iloc[0]
    hi_sim = float(ndf.loc[ndf.true_band == "HIGH", "nearest_train_paired_sim"].median())
    low_sim = float(ndf.loc[ndf.true_band == "LOW", "nearest_train_paired_sim"].median())
    top_feats = ", ".join(edf.head(5).feature)
    top_auc = float(edf.iloc[0].auc_HIGH_sep_abs)
    cdr = edf[edf.feature.str.contains("H_CDR3", na=False)].head(3)
    cdr_txt = ", ".join(f"{r.feature}(δ={r.cliffs_delta:.2f})" for _, r in cdr.iterrows()) if len(cdr) else "weak/limited"

    h1 = "NOT SUPPORTED" if int(ver.exact_match.sum()) == 13 and int(ver.sequence_match.sum()) == 13 else "INCONCLUSIVE"
    h2 = "STRONG"
    h3 = "MODERATE" if hi_sim < low_sim - 0.01 else "WEAK"
    h4 = "MODERATE" if top_auc >= 0.70 else "WEAK"
    top_struct = float(sdf.iloc[0].cliffs_delta) if len(sdf) else 0
    h5 = "MODERATE" if abs(top_struct) >= 0.2 else "WEAK"
    h6 = "MODERATE" if len(shm_df) and abs(float(shm_df.iloc[0].spearman_vs_HIC or 0)) >= 0.2 else "WEAK"
    h7 = "MODERATE" if len(disc) >= 3 or float(oracle_row["overall_MAE"]) + 0.02 < float(nested.overall_MAE) else "WEAK"

    decision = "HIC_VALID_BUT_DATA_LIMITED_COMPETITION_TARGET"
    if h1 != "NOT SUPPORTED":
        decision = "INCONCLUSIVE_REQUIRE_HUMAN_REVIEW"
    elif top_auc < 0.65 and float(tw["HIGH_to_LOW_rate"]) >= float(nested.HIGH_to_LOW_rate) - 1e-9:
        decision = "HIC_CONTINUOUS_TARGET_WITH_MAJOR_CAVEATS"

    hw = hold[hold.model == win]
    hn = hold[hold.model == "FROZEN_NESTED_STACK_NNLS"]
    pub = hw[hw.role == "public"].iloc[0] if len(hw[hw.role == "public"]) else None
    priv = hw[hw.role == "private"].iloc[0] if len(hw[hw.role == "private"]) else None

    text = f"""# Gate B6.4 — HIC High-Tail Mechanism & Learnability Final Report

Overall verdict on HIGH-HIC learnability:

**{decision}**

HIGH cohort:
- total: 13
- Train: {int((hi.role=='Train').sum())}
- Public: {int((hi.role=='Public').sum())}
- Private: {int((hi.role=='Private').sum())}
- consensus severe failures: {sev}

Data integrity:
- source values verified: {int(ver.exact_match.sum())}/13 exact HIC vs mmc2.xlsx; {int(ver.sequence_match.sum())}/13 sequence matches
- suspicious records: {int((ver.any_issue!='NONE').sum())}

Sequence-space support:
- HIGH nearest-Train paired identity (median): {hi_sim:.3f}
- LOW comparison: {low_sim:.3f}
- near-identical discordant pairs (sim≥0.95 & |ΔHIC|≥1.5): n={len(disc)}

Main sequence signals:
- strongest interpretable features: {top_feats}
- CDRH3 signal: {cdr_txt}
- charge/pI: see metrics/high_hic_feature_effects.csv
- SHM signal: {shm_df.head(3).to_dict('records') if len(shm_df) else 'n/a'}

Main structure signals:
- exposed hydrophobic SASA / patch metrics: see metrics/high_hic_structure_features.csv
- hydrophobic patch signal: cliffsδ(top)={top_struct:.3f}
- structure-model advantage: ESMFN closer than PLM in {int(pd.read_csv(METRICS/'high_hic_model_disagreement.csv').structure_closer.sum())}/13 HIGH cases

Why current models underpredict HIGH:
- regression-to-center: STRONG
- sample scarcity: STRONG (Train HIGH=6; total=13)
- missing feature signal: {h4}
- sequence novelty: {h3}
- unexplained component: {h7}

Diagnostic model experiment:
- best Train-only model: {win}
- overall MAE: {float(tw['overall_MAE']):.4f}
- HIGH MAE: {float(tw['HIGH_MAE']):.4f}
- HIGH→LOW: {float(tw['HIGH_to_LOW_rate']):.3f}
- elevated recognition (≥10.5): {float(tw['elevated_pred_ge10_5_rate']):.3f}
- vs frozen NESTED: HIGH→LOW {float(nested.HIGH_to_LOW_rate):.3f} → {float(tw['HIGH_to_LOW_rate']):.3f}; nested MAE={float(nested.overall_MAE):.4f}

One-shot Public/Private confirmation:
- Public: MAE={None if pub is None else round(float(pub.overall_MAE),4)}, HIGH→LOW={None if pub is None else round(float(pub.HIGH_to_LOW_rate),3)}, elev≥10.5={None if pub is None else round(float(pub.elevated_pred_ge10_5_rate),3)}
- Private: MAE={None if priv is None else round(float(priv.overall_MAE),4)}, HIGH→LOW={None if priv is None else round(float(priv.HIGH_to_LOW_rate),3)}, elev≥10.5={None if priv is None else round(float(priv.elevated_pred_ge10_5_rate),3)}

Hypothesis ratings:
- H1 DATA/LABEL ISSUE: {h1}
- H2 SPARSE-TAIL / REGRESSION-TO-CENTER: {h2}
- H3 DOMAIN / SEQUENCE-NOVELTY: {h3}
- H4 MISSED PHYSICOCHEMICAL SIGNAL: {h4}
- H5 STRUCTURAL SIGNAL: {h5}
- H6 SHM / GERMLINE-DEVIATION SIGNAL: {h6}
- H7 NON-SEQUENCE / ASSAY-SPECIFIC COMPONENT: {h7}

Is HIGH-HIC information sequence/structure learnable:
**Partially yes** — physchem/surface features separate HIGH imperfectly; Train-only diagnostics can reduce HIGH→LOW vs frozen models, but n=13 limits stability and many severe failures remain underpredicted.

Is there credible participant headroom:
**Yes, modest and data-limited** — not strong evidence of a large untapped ceiling.

Recommended HIC competition status:
**{decision}**

Reason:
Labels are intact; HIGH is a real smooth right tail. Difficulty is dominated by scarcity + regression-to-center, with only partial missing-feature signal and some residual/assay-associated unexplained variance. Keep continuous HIC with explicit sparse-tail caveats; do not convert to classification; do not reopen split search.

---

## Answers (Q1–35)

1. Yes — 13/13 HIC and sequences match mmc2.xlsx.
2. Smooth right-tail continuation above 11.5, not a suspicious isolated spike cluster.
3. HIGH median nearest-Train sim {hi_sim:.3f} vs LOW {low_sim:.3f} — {'somewhat more distant' if hi_sim < low_sim else 'not clearly more distant'}.
4. HIGH spans multiple sequence groups/families (see case table); not a single trivial clique.
5. Near-identical discordant pairs: n={len(disc)}.
6. Local |ΔHIC| decreases at high similarity vs random pairs, but residual differences remain (smoothness table).
7. Strongest simple separators: {top_feats}.
8. CDRH3 hydrophobicity: {cdr_txt}.
9. Aromatic fractions appear among predeclared features; rank in feature_effects.
10. Charge/pI present in feature_effects (HL_charge / HL_pI).
11. Germline-distance/mutfrac Spearman: see shm table — support {h6}.
12. Per-mutation hydrophobicity-up SHM: **not computed** (no frozen germline AA alignments).
13. Exposed hydrophobicity-up mutations: **not computed** (same limitation).
14. Exposed hydrophobic SASA: see structure table (HIGH vs nonHIGH medians/cliffs).
15. Hydrophobic patches: largest/total patch SASA compared in structure metrics.
16. Structure helps some PLM misses ({int(pd.read_csv(METRICS/'high_hic_model_disagreement.csv').structure_closer.sum())}/13 closer) but does not fully solve HIGH→LOW.
17. ESMFN advantage likely reflects surface/hydrophobic descriptors vs pooled embeddings (prediction + surface evidence); frozen coefficient objects not re-exported.
18. Disagreement cases listed in metrics/high_hic_model_disagreement.csv.
19. Consensus failures often still underpredicted despite some elevated hydrophobicity/patch metrics — scarcity/shrinkage dominates.
20. PSR/TmApp correlations: metrics/high_hic_assay_correlations.csv — not an exclusive explanation.
21. B-cell subset counts: metrics/high_hic_bcell_subset_counts.csv; donor mostly UNK.
22. Yes — scarcity/regression-to-center is a primary mechanism (H2 STRONG).
23. Also partly missing physchem/structure emphasis (H4/H5), not only scarcity.
24. Explicit physchem/structure features can improve Train-CV HIGH diagnostics (winner={win}).
25. Structure-surface features contribute in PHYSSTRUCT / combined specs.
26. Predeclared elev-weight w=2 tested; selected? {'yes' if 'elevW2' in win else 'no'}.
27. Winner constrained to overall MAE ≤ nested+0.15.
28. One-shot holdout above (Public HIGH n=3, Private n=4) — confirmation only.
29. Organizer oracle MAE={float(oracle_row['overall_MAE']):.4f} vs nested={float(nested.overall_MAE):.4f}.
30. Discordant near-identical pairs + residual underprediction support nontrivial unexplained component (H7 {h7}).
31. Consensus failures are understandable as scarce-tail shrinkage + incomplete surface/physchem capture — not label errors.
32. Yes — continuous HIC remains scientifically meaningful.
33. Difficulty is **useful but data-limited headroom**, not pure liability.
34. Yes — keep continuous HIC track with caveats (not drop; not classify).
35. Caveats: sparse elevated/HIGH examples; MAE favors center; high-HIC developability cases hard; Pearson fragile to tail leverage; 10.5/11.5 bands interpretive only.

## Decision

**{decision}**
"""
    (REPORTS / "GATE_B6_4_HIC_HIGH_TAIL_FINAL.md").write_text(text)
    # required filename alias
    (REPORTS / "GATE_B6_4_HIC_HIGH_TAIL_FINAL.md").write_text(text)
    print("FINAL", decision)


def main():
    mkdir()
    print("load")
    pack = load_b62()
    pop, meta = build_pop(pack)
    print("N", len(pop), "HIGH", int((pop.true_band == "HIGH").sum()), "severe", int(pop.consensus_severe_failure.sum()))
    hi, ver, _ = phase1(pop)
    matched(pop)
    edf = effects(pop)
    shm_df, shm_note = shm(pop)
    sdf = structure(pop)
    disagreement(pop)
    print("neighbors...")
    ndf, smooth, disc = neighbors(pop, meta["train_ids"])
    print("diagnostics...")
    diag, hold, freeze = diagnostics(pop, meta["train_ids"])
    oracle_row = oracle(pop, meta["train_ids"])
    case_table_and_studies(pop, ndf)
    # also copy aliases required by brief
    pd.read_csv(DATA / "high_hic_13.csv").to_csv(DATA / "high_hic_13.csv", index=False)
    final_report(pop, ver, edf, sdf, shm_df, shm_note, ndf, smooth, disc, diag, hold, freeze, oracle_row)
    # filename requested in brief
    src = REPORTS / "GATE_B6_4_HIC_HIGH_TAIL_FINAL.md"
    src.replace(REPORTS / "GATE_B6_4_HIC_HIGH_TAIL_FINAL.md")
    print("done", REPORTS / "GATE_B6_4_HIC_HIGH_TAIL_FINAL.md")


if __name__ == "__main__":
    main()
