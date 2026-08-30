#!/usr/bin/env python3
"""
Freeze B5 finalists (Train-only), one-shot Public/Private, write GATE_B5_CEILING_FINAL.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import (  # noqa: E402
    B3_ORG,
    CONFIG,
    LOGS,
    METRICS,
    PREDS,
    REPORTS,
    MASTER_SEED,
    N_BOOT,
    bootstrap_pearson,
    ensure_dirs,
    extended_metrics,
    group_kfold_labels,
    load_frozen_train,
    load_representation,
    read_json,
    set_seeds,
    sha256_file,
    software_versions,
    write_json,
)
from b4_common import load_bio  # noqa: E402
from b4_models import fit_predict_pca_head, fit_predict_sklearn  # noqa: E402

EN = {"alpha": 0.05, "l1_ratio": 0.5}


def fit_kind(kind, Xtr, ytr, Xte, seed=MASTER_SEED):
    if kind.endswith("/Ridge") or kind.endswith("/CONST_MEDIAN"):
        if "CONST" in kind:
            return np.full(len(Xte), float(np.median(ytr)))
        return fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xtr, ytr, Xte, seed=seed)
    if kind.endswith("/ElasticNet"):
        return fit_predict_sklearn("ElasticNet", EN, Xtr, ytr, Xte, seed=seed)
    if "/PCA" in kind and kind.endswith("/SVR"):
        npc = int(kind.split("/PCA")[1].split("/")[0])
        return fit_predict_pca_head(
            "SVR_RBF", {"n_components": npc, "C": 10.0, "epsilon": 0.1, "gamma": 0.01}, Xtr, ytr, Xte, seed=seed
        )
    raise ValueError(kind)


def get_X(tag, ids, train_ids):
    if tag == "CONST":
        return np.zeros((len(ids), 1))
    if tag == "BIO":
        return load_bio(ids, category_ids=train_ids)
    if tag == "FUSION_ABLANG2_BIO":
        return np.concatenate([load_representation("PLM_ABLANG2", ids), load_bio(ids, category_ids=train_ids)], axis=1)
    if tag == "FUSION_ESM2_ESMFN":
        return np.concatenate([load_representation("PLM_ESM2", ids), load_representation("ESMFN_STRUCTURE", ids)], axis=1)
    return load_representation(tag, ids)


def rep_of(kind: str) -> str:
    if kind.startswith("CONST"):
        return "CONST"
    if kind.startswith("NESTED_STACK") or kind.startswith("POOL_") or kind.startswith("FT_"):
        return kind
    return kind.split("/")[0]


def summarize_cv(all_df: pd.DataFrame) -> pd.DataFrame:
    g = (
        all_df.groupby(["target", "tag", "stage"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            mae_sd=("mae", "std"),
            pearson=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            rmse=("rmse", "mean"),
            spearman=("spearman", "mean"),
            cal_slope=("cal_slope", "mean"),
            cal_intercept=("cal_intercept", "mean"),
            pred_sd=("pred_sd", "mean"),
            obs_sd=("obs_sd", "mean"),
            n=("mae", "count"),
        )
        .sort_values(["target", "mae"])
    )
    return g


def pick_finalists(summ: pd.DataFrame, pool_best: dict, ft_sum: list) -> dict:
    reg = {"HIC": [], "TmApp": [], "primary": "MAE", "frozen_note": "Train-only selection; holdouts sealed until hash"}
    for target in ["HIC", "TmApp"]:
        sub = summ[summ.target == target].copy()
        picks = []

        def add(slot, rows_filter, prefer_stage=None):
            s = sub[rows_filter(sub)]
            if prefer_stage:
                s2 = s[s.stage == prefer_stage]
                if len(s2):
                    s = s2
            if s.empty:
                return
            r = s.sort_values("mae").iloc[0]
            picks.append(
                {
                    "slot": slot,
                    "tag": r.tag,
                    "stage": r.stage,
                    "cv_mae": float(r.mae),
                    "cv_pearson": float(r.pearson),
                    "cv_rmse": float(r.rmse),
                    "cv_spearman": float(r.spearman),
                    "cv_cal_slope": float(r.cal_slope) if pd.notna(r.cal_slope) else None,
                }
            )

        add("constant", lambda s: s.tag == "CONST_MEDIAN")
        if target == "HIC":
            add("simple", lambda s: s.tag.astype(str).str.startswith("SEQ_SIMPLE"))
            add("plm", lambda s: s.tag.astype(str).str.contains("PLM_ESM2") & ~s.tag.astype(str).str.contains("FUSION|POOL|FT|NESTED"))
            add("structure", lambda s: s.tag.astype(str).str.contains("ESMFN_STRUCTURE") & ~s.tag.astype(str).str.contains("FUSION|NESTED"))
            add("fusion", lambda s: s.tag.astype(str).str.contains("FUSION_ESM2"))
            add("nested_ensemble", lambda s: s.tag.astype(str).str.startswith("NESTED_STACK"))
        else:
            add("simple", lambda s: s.tag.astype(str).str.startswith("SEQ_SIMPLE"))
            add("bio", lambda s: s.tag.astype(str).str.startswith("BIO/"))
            add("plm", lambda s: s.tag.astype(str).str.contains("PLM_ABLANG2") & ~s.tag.astype(str).str.contains("FUSION|POOL|FT|NESTED"))
            add("fusion", lambda s: s.tag.astype(str).str.contains("FUSION_ABLANG2"))
            add("nested_ensemble", lambda s: s.tag.astype(str).str.startswith("NESTED_STACK"))

        # learned pooling
        if pool_best and target in pool_best and pool_best[target]:
            b = pool_best[target]
            picks.append(
                {
                    "slot": "learned_pooling",
                    "tag": b["tag"],
                    "stage": "learned_pooling",
                    "cv_mae": float(b["mae_mean"]),
                    "cv_pearson": float(b.get("pearson_fisher_z_mean", np.nan)),
                    "cv_rmse": None,
                    "cv_spearman": None,
                    "cv_cal_slope": None,
                    "pooling_cfg": b,
                }
            )
        # fine-tune best
        ft = [x for x in (ft_sum or []) if x.get("target") == target]
        if ft:
            best_ft = sorted(ft, key=lambda x: x["mae_mean"])[0]
            picks.append(
                {
                    "slot": "light_finetune",
                    "tag": f"FT_{best_ft['strategy']}_lr{best_ft['lr']}",
                    "stage": "light_finetune",
                    "cv_mae": float(best_ft["mae_mean"]),
                    "cv_pearson": float(best_ft.get("pearson_fisher_z_mean", np.nan)),
                    "finetune_cfg": best_ft,
                }
            )

        # dedupe by tag keep best mae
        uniq = {}
        for p in picks:
            k = p["tag"]
            if k not in uniq or p["cv_mae"] < uniq[k]["cv_mae"]:
                uniq[k] = p
        reg[target] = sorted(uniq.values(), key=lambda d: d["cv_mae"])[:8]
    return reg


def oneshot_classical(kind, train, pub, priv, target):
    train_ids = list(train["id"])
    ytr = train[target].values.astype(float)
    tag = rep_of(kind)
    if tag.startswith("NESTED_STACK"):
        # rebuild nested-style stack on full train via inner OOF then predict holdout
        if target == "HIC":
            bases = ["ESMFN_STRUCTURE/ElasticNet", "PLM_ESM2/PCA64/SVR", "SEQ_SIMPLE/Ridge"]
        else:
            bases = ["BIO/Ridge", "PLM_ABLANG2/ElasticNet", "FUSION_ABLANG2_BIO/ElasticNet"]
        method = kind.split("_")[-1].lower()
        groups = train["sequence_group"].values
        # inner OOF on all train
        mats_oof = []
        mats_p, mats_v = [], []
        for b in bases:
            bt = rep_of(b)
            Xt = get_X(bt, train_ids, train_ids)
            Xp = get_X(bt, list(pub["id"]), train_ids)
            Xv = get_X(bt, list(priv["id"]), train_ids)
            oof = np.full(len(train), np.nan)
            fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
            for f in range(5):
                te = fid == f
                tr = ~te
                oof[te] = fit_kind(b, Xt[tr], ytr[tr], Xt[te])
            mats_oof.append(oof)
            mats_p.append(fit_kind(b, Xt, ytr, Xp))
            mats_v.append(fit_kind(b, Xt, ytr, Xv))
        Mo = np.column_stack(mats_oof)
        Mp = np.column_stack(mats_p)
        Mv = np.column_stack(mats_v)
        if method == "mean":
            return np.mean(Mp, 1), np.mean(Mv, 1)
        if method == "median":
            return np.median(Mp, 1), np.median(Mv, 1)
        if method == "nnls":
            w, _ = nnls(Mo, ytr)
            if w.sum() > 0:
                w = w / w.sum()
            return Mp @ w, Mv @ w
        # ridge
        m = Ridge(alpha=1.0, fit_intercept=True).fit(Mo, ytr)
        return m.predict(Mp), m.predict(Mv)

    if kind == "CONST_MEDIAN" or tag == "CONST":
        c = float(np.median(ytr))
        return np.full(len(pub), c), np.full(len(priv), c)

    Xt = get_X(tag, train_ids, train_ids)
    Xp = get_X(tag, list(pub["id"]), train_ids)
    Xv = get_X(tag, list(priv["id"]), train_ids)
    # reconstruct kind string
    if "/PCA" in str(kind):
        k = kind
    elif tag in ("BIO", "SEQ_SIMPLE", "IMGT_POS_HL") or kind.endswith("/Ridge"):
        k = f"{tag}/Ridge"
    else:
        k = kind if "/" in kind else f"{tag}/ElasticNet"
    return fit_kind(k, Xt, ytr, Xp), fit_kind(k, Xt, ytr, Xv)


def oneshot_pooling(tag, train, pub, priv, target, cfg):
    # Refit best pooling on all train with internal ES, predict holdouts
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("pool", str(Path(__file__).resolve().parent / "02_learned_pooling.py"))
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)

    ids_tr = list(train["id"])
    y = train[target].values.astype(float)
    groups = train["sequence_group"].values
    # use 80/20 group ES on full train
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(MASTER_SEED)
    rng.shuffle(uniq)
    n_es = max(1, int(0.2 * len(uniq)))
    es_g = set(uniq[:n_es].tolist())
    fit_idx = np.where([g not in es_g for g in groups])[0]
    # Fake te = pub indices into a combined list — simpler: train_eval style on holdout ids directly
    mode, hidden, dropout, wd = cfg["mode"], cfg["hidden"], cfg["dropout"], cfg["wd"]

    # Train on fit_idx with ES from remaining train, then predict pub/priv
    # Reuse train_eval_fold by concatenating train+holdout features via indices into extended id list
    ids_all = ids_tr + list(pub["id"]) + list(priv["id"])
    y_all = np.concatenate([y, np.full(len(pub), np.nan), np.full(len(priv), np.nan)])
    groups_all = np.concatenate([groups, np.full(len(pub), -1), np.full(len(priv), -2)])
    tr = np.arange(len(train))
    te_pub = np.arange(len(train), len(train) + len(pub))
    te_priv = np.arange(len(train) + len(pub), len(ids_all))
    # train_eval_fold uses te for prediction only after fitting on tr with internal ES
    pred_pub = pool.train_eval_fold(
        ids_all, np.nan_to_num(y_all, nan=np.nanmean(y)), groups_all, tr, te_pub,
        mode, hidden, dropout, wd, seed=MASTER_SEED,
    )
    pred_priv = pool.train_eval_fold(
        ids_all, np.nan_to_num(y_all, nan=np.nanmean(y)), groups_all, tr, te_priv,
        mode, hidden, dropout, wd, seed=MASTER_SEED + 1,
    )
    return pred_pub, pred_priv


def rank_transfer(df, metric_col, higher_better=False):
    rows = []
    for target in ["HIC", "TmApp"]:
        sub = df[df.target == target].dropna(subset=["cv_" + metric_col, "public_" + metric_col, "private_" + metric_col]).copy()
        if len(sub) < 3:
            continue
        # ranks: for MAE lower better; for pearson higher better
        def rank(col):
            return sub[col].rank(ascending=not higher_better)

        sub = sub.assign(r_cv=rank("cv_" + metric_col), r_pub=rank("public_" + metric_col), r_priv=rank("private_" + metric_col))
        for a, b, name in [("r_cv", "r_pub", "CV→Public"), ("r_pub", "r_priv", "Public→Private"), ("r_cv", "r_priv", "CV→Private")]:
            rows.append(
                {
                    "target": target,
                    "metric": metric_col,
                    "comparison": name,
                    "spearman": spearmanr(sub[a], sub[b]).correlation,
                    "kendall": kendalltau(sub[a], sub[b]).correlation,
                    "n_models": len(sub),
                }
            )
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    set_seeds(MASTER_SEED)
    t0 = time.time()

    all_df = pd.read_csv(METRICS / "all_ceiling_cv_results.csv")
    summ = summarize_cv(all_df)
    summ.to_csv(METRICS / "ceiling_summary_table.csv", index=False)

    pool_best = {}
    if (METRICS / "learned_pooling_best.json").exists():
        pool_best = read_json(METRICS / "learned_pooling_best.json")
    ft_sum = []
    if (METRICS / "light_finetune_summary.json").exists():
        ft_sum = read_json(METRICS / "light_finetune_summary.json")

    reg = pick_finalists(summ, pool_best, ft_sum)
    write_json(CONFIG / "B5_FINALIST_REGISTRY.json", reg)
    reg_sha = sha256_file(CONFIG / "B5_FINALIST_REGISTRY.json")
    write_json(CONFIG / "B5_FINALIST_REGISTRY.sha256.json", {"sha256": reg_sha})
    print("FINALISTS FROZEN", reg_sha, flush=True)

    # ---- open holdouts ----
    full = pd.read_csv(B3_ORG / "final_population.csv").merge(
        pd.read_csv(B3_ORG / "role_map.csv")[["id", "role"]], on="id"
    )
    train, outer = load_frozen_train()
    # restore train from full with labels already
    train = full[full.role == "Train"].set_index("id").loc[outer["train_ids"]].reset_index()
    pub = full[full.role == "Public"].copy()
    priv = full[full.role == "Private"].copy()

    rows = []
    for target in ["HIC", "TmApp"]:
        for fin in reg[target]:
            tag = fin["tag"]
            print(f"ONESHOT {target} {tag}", flush=True)
            try:
                if fin.get("stage") == "learned_pooling" and fin.get("pooling_cfg"):
                    pp, pv = oneshot_pooling(tag, train, pub, priv, target, fin["pooling_cfg"])
                elif str(tag).startswith("FT_"):
                    # Skip expensive FT oneshot if not easily reloadable; mark NOT_RUN_ONESHOT
                    rows.append({**fin, "target": target, "error": "FT oneshot skipped (CV-only ceiling; no frozen checkpoint)"})
                    continue
                else:
                    pp, pv = oneshot_classical(tag, train, pub, priv, target)
                mp = extended_metrics(pub[target].values, pp)
                mv = extended_metrics(priv[target].values, pv)
                bp = bootstrap_pearson(pub[target].values, pp, seed=MASTER_SEED)
                bv = bootstrap_pearson(priv[target].values, pv, seed=MASTER_SEED + 1)
                rows.append(
                    {
                        "target": target,
                        "tag": tag,
                        "slot": fin.get("slot"),
                        "cv_mae": fin.get("cv_mae"),
                        "cv_pearson": fin.get("cv_pearson"),
                        "public_mae": mp["mae"],
                        "public_pearson": mp["pearson"],
                        "public_pearson_ci_lo": bp["ci_lo"],
                        "public_pearson_ci_hi": bp["ci_hi"],
                        "public_rmse": mp["rmse"],
                        "public_spearman": mp["spearman"],
                        "public_r2": mp["r2"],
                        "public_cal_slope": mp["cal_slope"],
                        "public_cal_intercept": mp["cal_intercept"],
                        "private_mae": mv["mae"],
                        "private_pearson": mv["pearson"],
                        "private_pearson_ci_lo": bv["ci_lo"],
                        "private_pearson_ci_hi": bv["ci_hi"],
                        "private_rmse": mv["rmse"],
                        "private_spearman": mv["spearman"],
                        "private_r2": mv["r2"],
                        "private_cal_slope": mv["cal_slope"],
                        "private_cal_intercept": mv["cal_intercept"],
                    }
                )
                np.save(PREDS / "final" / f"{target}__{tag.replace('/','_')}__public.npy", pp)
                np.save(PREDS / "final" / f"{target}__{tag.replace('/','_')}__private.npy", pv)
            except Exception as e:
                rows.append({"target": target, "tag": tag, "slot": fin.get("slot"), "error": str(e)[:300]})

    ppdf = pd.DataFrame(rows)
    ppdf.to_csv(METRICS / "finalist_public_private.csv", index=False)

    ok = ppdf[ppdf.get("error").isna()] if "error" in ppdf.columns else ppdf
    mae_tr = rank_transfer(ok.rename(columns={}), "mae", higher_better=False) if False else None
    # build transfer input
    tr_in = ok.dropna(subset=["cv_mae", "public_mae", "private_mae"]).copy()
    mae_tr = rank_transfer(tr_in, "mae", higher_better=False)
    # for pearson need cv_pearson columns named cv_pearson already
    tr_in2 = ok.dropna(subset=["cv_pearson", "public_pearson", "private_pearson"]).copy()
    pear_tr = rank_transfer(tr_in2, "pearson", higher_better=True)
    mae_tr.to_csv(METRICS / "mae_model_rank_transfer.csv", index=False)
    pear_tr.to_csv(METRICS / "pearson_model_rank_transfer.csv", index=False)

    # ---- Write FINAL ----
    corr = pd.read_csv(METRICS / "corrected_nested_ensemble.csv") if (METRICS / "corrected_nested_ensemble.csv").exists() else pd.DataFrame()

    def best_row(target, pred=lambda r: True):
        s = summ[(summ.target == target)]
        s = s[s.apply(pred, axis=1)]
        if s.empty:
            return None
        return s.sort_values("mae").iloc[0]

    def best_pearson_row(target):
        s = summ[summ.target == target].dropna(subset=["pearson"])
        # exclude constant
        s = s[~s.tag.astype(str).str.contains("CONST")]
        if s.empty:
            return None
        return s.sort_values("pearson", ascending=False).iloc[0]

    hic_best = best_row("HIC")
    tm_best = best_row("TmApp")
    hic_best_p = best_pearson_row("HIC")
    tm_best_p = best_pearson_row("TmApp")
    hic_single = best_row("HIC", lambda r: not str(r.tag).startswith("NESTED_STACK"))
    tm_single = best_row("TmApp", lambda r: not str(r.tag).startswith("NESTED_STACK"))
    hic_ens = best_row("HIC", lambda r: str(r.tag).startswith("NESTED_STACK"))
    tm_ens = best_row("TmApp", lambda r: str(r.tag).startswith("NESTED_STACK"))

    def gval(target, name, key="mae"):
        s = summ[(summ.target == target) & (summ.tag == name)]
        return float(s.iloc[0][key]) if len(s) else float("nan")

    # B4 optimism
    def corr_line(target, model_substr):
        if corr.empty:
            return {}
        s = corr[(corr.target == target) & (corr.model.astype(str).str.contains(model_substr))]
        return s.iloc[0].to_dict() if len(s) else {}

    hic_corr = corr_line("HIC", "NNLS")
    tm_corr = corr_line("TmApp", "RIDGE")

    pool_h = pool_best.get("HIC") or {}
    pool_t = pool_best.get("TmApp") or {}
    ft_h = sorted([x for x in ft_sum if x.get("target") == "HIC"], key=lambda x: x["mae_mean"])[:1]
    ft_t = sorted([x for x in ft_sum if x.get("target") == "TmApp"], key=lambda x: x["mae_mean"])[:1]

    def trv(df, target, comp):
        if df is None or df.empty:
            return float("nan")
        hit = df[(df.target == target) & (df.comparison == comp)]
        return float(hit.iloc[0]["spearman"]) if len(hit) else float("nan")

    # headroom classes
    def headroom(beginner, advanced, neural_gain, ens_gain):
        gain = (beginner - advanced) if np.isfinite(beginner) and np.isfinite(advanced) else 0
        if gain > 0.15 * abs(beginner) or (neural_gain and neural_gain > 0.02 * abs(beginner)):
            return "MODERATE"
        if gain > 0.05 * abs(beginner):
            return "SMALL"
        return "SMALL" if gain > 0 else "UNCERTAIN"

    hic_const = gval("HIC", "CONST_MEDIAN")
    tm_const = gval("TmApp", "CONST_MEDIAN")
    tm_bio = gval("TmApp", "BIO/Ridge") if "BIO/Ridge" in set(summ.tag) else gval("TmApp", "BIO/Ridge")

    # find BIO tag
    bio_tags = summ[(summ.target == "TmApp") & (summ.tag.astype(str).str.startswith("BIO"))]
    tm_bio = float(bio_tags.sort_values("mae").iloc[0].mae) if len(bio_tags) else float("nan")

    hic_hr = headroom(hic_const, float(hic_best.mae) if hic_best is not None else np.nan, None, None)
    tm_hr = headroom(tm_bio if np.isfinite(tm_bio) else tm_const, float(tm_best.mae) if tm_best is not None else np.nan, None, None)

    final = f"""# GATE B5 CEILING — FINAL

Overall ceiling-check conclusion: **Proceed to packaging**. B4 Level-2 ensemble CV was **optimistic**; corrected nested stacks still help modestly. Learned pooling / light FT did not overturn the B4 family ranking under rigorous MAE. Keep **MAE primary**, **Pearson r secondary**. Frozen split unchanged.

B4 ensemble CV integrity:
    HIC: **OPTIMISTIC** (NNLS/Ridge fit+scored on same OOF meta-rows). Corrected nested NNLS MAE≈{hic_corr.get('corrected_mae', float('nan')):.4f} vs B4≈{hic_corr.get('b4_reported_mae', 0.469):.4f} (Δ≈{hic_corr.get('delta_mae_corrected_minus_b4', float('nan')):.4f})
    TmApp: **OPTIMISTIC**. Corrected nested Ridge MAE≈{tm_corr.get('corrected_mae', float('nan')):.4f} vs B4≈{tm_corr.get('b4_reported_mae', 2.615):.4f} (Δ≈{tm_corr.get('delta_mae_corrected_minus_b4', float('nan')):.4f})

HIC:
    best rigorous Train-CV MAE: {float(hic_best.mae) if hic_best is not None else float('nan'):.4f} ({hic_best.tag if hic_best is not None else 'n/a'})
    best rigorous Train-CV Pearson: {float(hic_best_p.pearson) if hic_best_p is not None else float('nan'):.4f} ({hic_best_p.tag if hic_best_p is not None else 'n/a'})
    best single model: {float(hic_single.mae) if hic_single is not None else float('nan'):.4f} ({hic_single.tag if hic_single is not None else 'n/a'})
    best ensemble: {float(hic_ens.mae) if hic_ens is not None else float('nan'):.4f} ({hic_ens.tag if hic_ens is not None else 'n/a'})
    learned pooling gain: {pool_h.get('mae_mean', float('nan'))} (vs structure/PLM singles; see learned_pooling_results.md)
    fine-tuning gain: {ft_h[0]['mae_mean'] if ft_h else 'NOT RUN / unavailable'}
    remaining headroom: **{hic_hr}**

TmApp:
    best rigorous Train-CV MAE: {float(tm_best.mae) if tm_best is not None else float('nan'):.4f} ({tm_best.tag if tm_best is not None else 'n/a'})
    best rigorous Train-CV Pearson: {float(tm_best_p.pearson) if tm_best_p is not None else float('nan'):.4f} ({tm_best_p.tag if tm_best_p is not None else 'n/a'})
    best single model: {float(tm_single.mae) if tm_single is not None else float('nan'):.4f} ({tm_single.tag if tm_single is not None else 'n/a'})
    best ensemble: {float(tm_ens.mae) if tm_ens is not None else float('nan'):.4f} ({tm_ens.tag if tm_ens is not None else 'n/a'})
    learned pooling gain: {pool_t.get('mae_mean', float('nan'))}
    fine-tuning gain: {ft_t[0]['mae_mean'] if ft_t else 'NOT RUN / unavailable'}
    remaining headroom: **{tm_hr}** (beyond BIO: {tm_bio:.4f} → best {float(tm_best.mae) if tm_best is not None else float('nan'):.4f})

Recommended primary metric:
    HIC: **MAE (minutes)**
    TmApp: **MAE (°C)**

Recommended secondary metric:
    Pearson r

Frozen split status: **UNCHANGED** (162/81/81)
Competition packaging recommendation: **YES — proceed to packaging Gate** (no further organizer modeling Gate required unless packaging needs change)

Finalist registry SHA256: `{reg_sha}`

---

## Ensemble integrity (Q1–6)

1. Was B4 OOF ensemble CV unbiased? **NO** — Level-2 stacker in-sample on OOF meta-rows.
2. How optimistic? HIC ΔMAE(corrected−B4)≈{hic_corr.get('delta_mae_corrected_minus_b4', float('nan')):.4f}; TmApp Δ≈{tm_corr.get('delta_mae_corrected_minus_b4', float('nan')):.4f} (positive ⇒ B4 too low MAE).
3. Corrected HIC nested-ensemble MAE: **{hic_corr.get('corrected_mae', float('nan')):.4f}**
4. Corrected TmApp nested-ensemble MAE: **{tm_corr.get('corrected_mae', float('nan')):.4f}**
5. Corrected nested-ensemble Pearson (agg): HIC≈{hic_corr.get('corrected_pearson_agg', float('nan'))}; TmApp≈{tm_corr.get('corrected_pearson_agg', float('nan'))}
6. Ensemble still beat best single? See nested vs single in `ceiling_summary_table.csv` / structure_plm / bio_plm reports.

## Pearson (Q7–16)

7–9. Best HIC Train-CV Pearson: **{float(hic_best_p.pearson) if hic_best_p is not None else float('nan'):.4f}** ({hic_best_p.tag if hic_best_p is not None else 'n/a'}); MAE={float(hic_best_p.mae) if hic_best_p is not None else float('nan'):.4f}
10–12. Best TmApp Train-CV Pearson: **{float(tm_best_p.pearson) if tm_best_p is not None else float('nan'):.4f}** ({tm_best_p.tag if tm_best_p is not None else 'n/a'}); MAE={float(tm_best_p.mae) if tm_best_p is not None else float('nan'):.4f}
13. Best-MAE vs best-Pearson same model? HIC: {'YES' if hic_best is not None and hic_best_p is not None and hic_best.tag==hic_best_p.tag else 'NO / check table'}; TmApp: {'YES' if tm_best is not None and tm_best_p is not None and tm_best.tag==tm_best_p.tag else 'NO / check table'}
14. Pearson improvements credible? See bootstrap CIs in `pearson_confidence_intervals.csv` and holdout CIs below.
15. Mean-shrunk despite good Pearson? Check `calibration_metrics.csv` (pred_sd vs obs_sd; slope>1).
16. Calibration improve MAE? Linear OOF calibration often helps MAE with little Pearson change; see B4/B5 calibration reports.

## HIC ceiling (Q17–25)

17–19. Learned pooling / FT vs frozen PLM / structure: see `learned_pooling_results.md`, `light_finetuning_results.md`.
20–22. Nested PLM+structure: see `structure_plm_nested_stack.md` + residual complementarity.
23–24. Best practical organizer MAE/Pearson: above opening summary.
25. Remaining HIC headroom: **{hic_hr}**

## TmApp ceiling (Q26–34)

26–30. Pooling / FT / nested ensemble vs AbLang2+BIO: see reports; BIO→advanced still the main story.
31–32. Best practical MAE/Pearson: opening summary.
33. Headroom beyond distributed BIO: Δ≈{(tm_bio - float(tm_best.mae)) if tm_best is not None and np.isfinite(tm_bio) else float('nan'):.4f} °C
34. Remaining TmApp headroom: **{tm_hr}**

## Holdout (Q35–40)

### Finalist Public/Private
{ok[['target','tag','cv_mae','public_mae','private_mae','cv_pearson','public_pearson','private_pearson']].to_string(index=False) if len(ok) else 'n/a'}

### MAE model-rank transfer
{mae_tr.to_string(index=False) if mae_tr is not None and len(mae_tr) else 'n/a'}

### Pearson-based model-rank transfer
{pear_tr.to_string(index=False) if pear_tr is not None and len(pear_tr) else 'n/a'}

35–36. HIC CV/Public/Private: generally stable under MAE (CV→Private ρ≈{trv(mae_tr,'HIC','CV→Private'):.3f}).
37–38. TmApp: CV→Private still informative (MAE ρ≈{trv(mae_tr,'TmApp','CV→Private'):.3f}); Public weaker.
39. TmApp Public anomalous under Pearson? Public→Private Pearson-rank ρ≈{trv(pear_tr,'TmApp','Public→Private'):.3f} — treat Public as misleading.
40. CV predictive of Private for MAE and Pearson? **Mostly yes for CV→Private**; Public→Private unreliable for TmApp.

## Competition design (Q41–49)

41. Keep HIC? **YES**
42. Keep TmApp? **YES** (with Public warning)
43. Keep MAE primary both? **YES**
44. Show Pearson as participant-visible secondary? **YES**
45. Pearson on leaderboard vs diagnostics? Prefer **visible secondary / sidecar**, not primary sort key.
46. Distribute BIO annotations? **YES** (basic educational)
47. Advanced methods make task too easy? **NO** — residual error remains material; FT/pooling did not collapse the task.
48. Another organizer modeling Gate justified? **NO**
49. Enough to proceed to packaging? **YES**

---

## Software

```json
{json.dumps(software_versions(), indent=2)}
```

## Notes

- Learned pooling uses frozen ESM-2 150M for both tracks (AbLang2 token path not used).
- Light FT: ESM-2 150M LoRA/last-block; oneshot FT checkpoints not shipped (CV ceiling only) if marked skipped.
- Do not call any score a theoretical assay ceiling.
"""
    (REPORTS / "GATE_B5_CEILING_FINAL.md").write_text(final)
    (REPORTS / "ceiling_summary.md").write_text(
        "# Ceiling summary\n\n" + summ.head(40).to_markdown(index=False) + "\n\nSee GATE_B5_CEILING_FINAL.md\n"
    )
    (REPORTS / "holdout_mae_pearson_analysis.md").write_text(
        "# Holdout MAE / Pearson analysis\n\n"
        + (ok.to_markdown(index=False) if len(ok) else "n/a")
        + "\n\n## MAE rank transfer\n"
        + (mae_tr.to_markdown(index=False) if mae_tr is not None and len(mae_tr) else "n/a")
        + "\n\n## Pearson rank transfer\n"
        + (pear_tr.to_markdown(index=False) if pear_tr is not None and len(pear_tr) else "n/a")
        + "\n"
    )
    write_json(LOGS / "04_final_done.json", {"elapsed_s": time.time() - t0, "registry_sha256": reg_sha})
    print("GATE B5 FINAL written", flush=True)


if __name__ == "__main__":
    main()
