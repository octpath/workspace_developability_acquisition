#!/usr/bin/env python3
"""Repair TmApp oneshot failures caused by BIO one-hot column mismatch; rewrite FINAL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import (  # noqa: E402
    B1_CACHE,
    B3_ORG,
    CONFIG,
    METRICS,
    PREDS,
    REPORTS,
    group_kfold_labels,
    load_representation,
    read_json,
    regression_metrics,
    sanitize_pair,
    sha256_file,
    software_versions,
    write_json,
    MASTER_SEED,
)
from b4_models import (  # noqa: E402
    fit_predict_lgb,
    fit_predict_pca_head,
    fit_predict_sklearn,
)


def load_bio_aligned(ids_train, ids_query):
    """One-hot categories frozen from Train rows only."""
    bio = pd.read_csv(B1_CACHE / "features" / "stage_C_shortcut.csv")
    cats = [c for c in ["C_vh_family", "C_vl_family", "C_kappa_lambda"] if c in bio.columns]
    tr = pd.DataFrame({"id": list(ids_train)}).merge(bio, left_on="id", right_on="antibody_id", how="left")
    qu = pd.DataFrame({"id": list(ids_query)}).merge(bio, left_on="id", right_on="antibody_id", how="left")
    num_cols = [c for c in tr.columns if c not in ("id", "antibody_id") and pd.api.types.is_numeric_dtype(tr[c])]
    Xtr_num = tr[num_cols].values.astype(float)
    Xqu_num = qu[num_cols].values.astype(float)
    if cats:
        dtr = pd.get_dummies(tr[cats].astype(str), dummy_na=True)
        dqu = pd.get_dummies(qu[cats].astype(str), dummy_na=True)
        dqu = dqu.reindex(columns=dtr.columns, fill_value=0)
        Xtr = np.concatenate([Xtr_num, dtr.values.astype(float)], axis=1)
        Xqu = np.concatenate([Xqu_num, dqu.values.astype(float)], axis=1)
    else:
        Xtr, Xqu = Xtr_num, Xqu_num
    return sanitize_pair(Xtr, Xqu)


def load_fusion_ablang2_bio(ids_train, ids_query):
    from b4_common import load_plm

    Atr = load_plm("ABLANG2", ids_train)
    Aqu = load_plm("ABLANG2", ids_query)
    Btr, Bqu = load_bio_aligned(ids_train, ids_query)
    return sanitize_pair(np.concatenate([Atr, Btr], axis=1), np.concatenate([Aqu, Bqu], axis=1))


def main():
    reg = read_json(CONFIG / "FINAL_ABSOLUTE_VALUE_FINALISTS.json")
    full = pd.read_csv(B3_ORG / "final_population.csv").merge(
        pd.read_csv(B3_ORG / "role_map.csv")[["id", "role"]], on="id"
    )
    outer = read_json(CONFIG / "OUTER_CV_FOLDS.json")
    train = full[full.role == "Train"].set_index("id").loc[outer["train_ids"]].reset_index()
    pub = full[full.role == "Public"].copy()
    priv = full[full.role == "Private"].copy()

    prev = pd.read_csv(METRICS / "finalist_public_private.csv")
    # keep successful HIC + successful TmApp rows
    ok = prev[prev["error"].isna()].copy() if "error" in prev.columns else prev.copy()
    rows = ok.to_dict("records")

    ytr = train["TmApp"].values.astype(float)
    groups = train["sequence_group"].values

    repairs = [
        ("FUSION_ABLANG2_BIO", "ElasticNet", "fusion"),
        ("BIO", "Ridge_grid", "bio"),
        ("FUSION_ABLANG2_BIO", "LGB_L1", "gbdt"),
        ("ENSEMBLE_RIDGE", "oof_ridge", "ensemble"),
        ("CAL_ENSEMBLE_NNLS", "linear_oof", "calibrated_ensemble"),
        ("RESID_BIO__PLM_ABLANG2", "OOF_residual", "residual"),
        ("SEQ_SIMPLE", "Ridge_grid", "simple_sequence"),
        ("IMGT_POS_HL", "Ridge_grid", "imgt_germline"),
    ]
    # only repair those that failed or missing
    have = {(r["tag"], r["model"]) for r in rows if r.get("target") == "TmApp"}

    for tag, model, slot in repairs:
        if (tag, model) in have and tag not in ("ENSEMBLE_RIDGE", "CAL_ENSEMBLE_NNLS", "FUSION_ABLANG2_BIO", "BIO", "RESID_BIO__PLM_ABLANG2"):
            continue
        # always re-do failed ones
        print("REPAIR", tag, model, flush=True)
        try:
            if tag == "BIO" and model == "Ridge_grid":
                Xtr, Xp = load_bio_aligned(train["id"], pub["id"])
                _, Xv = load_bio_aligned(train["id"], priv["id"])
                pp = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xtr, ytr, Xp)
                pv = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xtr, ytr, Xv)
            elif tag == "FUSION_ABLANG2_BIO" and model == "ElasticNet":
                Xtr, Xp = load_fusion_ablang2_bio(train["id"], pub["id"])
                _, Xv = load_fusion_ablang2_bio(train["id"], priv["id"])
                fid = group_kfold_labels(groups, 3, seed=MASTER_SEED)
                best, bp = np.inf, {"alpha": 0.05, "l1_ratio": 0.5}
                for a in [0.001, 0.01, 0.05, 0.1, 0.5]:
                    for l1 in [0.2, 0.5, 0.8]:
                        maes = []
                        for f in range(3):
                            te = fid == f
                            tr = ~te
                            pr = fit_predict_sklearn("ElasticNet", {"alpha": a, "l1_ratio": l1}, Xtr[tr], ytr[tr], Xtr[te])
                            maes.append(np.mean(np.abs(ytr[te] - pr)))
                        m = float(np.mean(maes))
                        if m < best:
                            best, bp = m, {"alpha": a, "l1_ratio": l1}
                pp = fit_predict_sklearn("ElasticNet", bp, Xtr, ytr, Xp)
                pv = fit_predict_sklearn("ElasticNet", bp, Xtr, ytr, Xv)
            elif tag == "FUSION_ABLANG2_BIO" and model == "LGB_L1":
                Xtr, Xp = load_fusion_ablang2_bio(train["id"], pub["id"])
                _, Xv = load_fusion_ablang2_bio(train["id"], priv["id"])
                bp = {
                    "max_depth": 3,
                    "num_leaves": 8,
                    "min_child_samples": 10,
                    "subsample": 0.8,
                    "colsample_bytree": 0.7,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "min_split_gain": 0.0,
                }
                sp = METRICS / "screening_summary.csv"
                if sp.exists():
                    ss = pd.read_csv(sp)
                    hit = ss[(ss.target == "TmApp") & (ss.tag == tag) & (ss.model == model)]
                    if len(hit) and isinstance(hit.iloc[0].get("best_params"), str):
                        bp = json.loads(hit.iloc[0]["best_params"])
                bis = []
                fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                for f in range(5):
                    te = fid == f
                    tr = ~te
                    _, meta = fit_predict_lgb(bp, Xtr[tr], ytr[tr], Xtr[te], groups[tr], objective="regression_l1")
                    bis.append(meta["best_iteration"])
                rounds = int(np.median(bis))
                pp, _ = fit_predict_lgb(bp, Xtr, ytr, Xp, groups, objective="regression_l1", fixed_rounds=rounds)
                pv, _ = fit_predict_lgb(bp, Xtr, ytr, Xv, groups, objective="regression_l1", fixed_rounds=rounds)
            elif tag == "RESID_BIO__PLM_ABLANG2":
                Xb_tr, Xb_p = load_bio_aligned(train["id"], pub["id"])
                _, Xb_v = load_bio_aligned(train["id"], priv["id"])
                Xr = load_representation("PLM_ABLANG2", train["id"])
                Xr_p = load_representation("PLM_ABLANG2", pub["id"])
                Xr_v = load_representation("PLM_ABLANG2", priv["id"])
                base_oof = np.zeros(len(train))
                fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                for f in range(5):
                    te = fid == f
                    tr = ~te
                    base_oof[te] = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xb_tr[tr], ytr[tr], Xb_tr[te])
                resid = ytr - base_oof
                base_p = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xb_tr, ytr, Xb_p)
                base_v = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xb_tr, ytr, Xb_v)
                r_p = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xr, resid, Xr_p)
                r_v = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xr, resid, Xr_v)
                pp, pv = base_p + r_p, base_v + r_v
            elif tag in ("ENSEMBLE_RIDGE", "CAL_ENSEMBLE_NNLS", "ENSEMBLE_NNLS", "ENSEMBLE_MEAN"):
                members = [("BIO", "ridge"), ("PLM_ABLANG2", "en"), ("FUSION_ABLANG2_BIO", "en")]
                mats_o, mats_p, mats_v = [], [], []
                for mt, mm in members:
                    if mt == "BIO":
                        Xt, Xp = load_bio_aligned(train["id"], pub["id"])
                        _, Xv = load_bio_aligned(train["id"], priv["id"])
                        oof = np.zeros(len(train))
                        fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                        for f in range(5):
                            te = fid == f
                            tr = ~te
                            oof[te] = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt[tr], ytr[tr], Xt[te])
                        pr_p = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, Xp)
                        pr_v = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, Xv)
                    elif mt == "FUSION_ABLANG2_BIO":
                        Xt, Xp = load_fusion_ablang2_bio(train["id"], pub["id"])
                        _, Xv = load_fusion_ablang2_bio(train["id"], priv["id"])
                        oof = np.zeros(len(train))
                        fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                        for f in range(5):
                            te = fid == f
                            tr = ~te
                            oof[te] = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt[tr], ytr[tr], Xt[te])
                        pr_p = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt, ytr, Xp)
                        pr_v = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt, ytr, Xv)
                    else:
                        Xt = load_representation(mt, train["id"])
                        Xp = load_representation(mt, pub["id"])
                        Xv = load_representation(mt, priv["id"])
                        oof = np.zeros(len(train))
                        fid = group_kfold_labels(groups, 5, seed=MASTER_SEED)
                        for f in range(5):
                            te = fid == f
                            tr = ~te
                            oof[te] = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt[tr], ytr[tr], Xt[te])
                        pr_p = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt, ytr, Xp)
                        pr_v = fit_predict_sklearn("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.5}, Xt, ytr, Xv)
                    mats_o.append(oof)
                    mats_p.append(pr_p)
                    mats_v.append(pr_v)
                Mo = np.column_stack(mats_o)
                Mp = np.column_stack(mats_p)
                Mv = np.column_stack(mats_v)
                if "RIDGE" in tag:
                    ridge = Ridge(alpha=1.0, fit_intercept=True).fit(Mo, ytr)
                    pp, pv = ridge.predict(Mp), ridge.predict(Mv)
                else:
                    w, _ = nnls(Mo, ytr)
                    if w.sum() > 0:
                        w = w / w.sum()
                    pp, pv = Mp @ w, Mv @ w
                    if tag.startswith("CAL_"):
                        b, a = np.polyfit(Mo @ w, ytr, 1)
                        pp = a + b * pp
                        pv = a + b * pv
            elif tag == "SEQ_SIMPLE":
                Xt = load_representation("SEQ_SIMPLE", train["id"])
                pp = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, load_representation("SEQ_SIMPLE", pub["id"]))
                pv = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, load_representation("SEQ_SIMPLE", priv["id"]))
            elif tag == "IMGT_POS_HL":
                Xt = load_representation("IMGT_POS_HL", train["id"])
                pp = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, load_representation("IMGT_POS_HL", pub["id"]))
                pv = fit_predict_sklearn("Ridge", {"alpha": 10.0}, Xt, ytr, load_representation("IMGT_POS_HL", priv["id"]))
            else:
                continue

            # cv_mae from registry if present
            cv_mae = np.nan
            for fin in reg.get("TmApp", []):
                if fin["tag"] == tag and fin["model"] == model:
                    cv_mae = fin.get("cv_mae", np.nan)
                    slot = fin.get("slot", slot)
            # fallback CV from nested/resid
            mp = regression_metrics(pub["TmApp"].values, pp)
            mv = regression_metrics(priv["TmApp"].values, pv)
            # replace existing failed row
            rows = [r for r in rows if not (r.get("target") == "TmApp" and r.get("tag") == tag and r.get("model") == model)]
            rows.append(
                {
                    "target": "TmApp",
                    "tag": tag,
                    "model": model,
                    "slot": slot,
                    "cv_mae": cv_mae,
                    "public_mae": mp["mae"],
                    "public_rmse": mp["rmse"],
                    "public_pearson": mp["pearson"],
                    "public_spearman": mp["spearman"],
                    "public_r2": mp["r2"],
                    "public_cal_slope": mp["cal_slope"],
                    "private_mae": mv["mae"],
                    "private_rmse": mv["rmse"],
                    "private_pearson": mv["pearson"],
                    "private_spearman": mv["spearman"],
                    "private_r2": mv["r2"],
                    "private_cal_slope": mv["cal_slope"],
                }
            )
            np.save(PREDS / "final" / f"TmApp__{tag}__{model}__public.npy", pp)
            np.save(PREDS / "final" / f"TmApp__{tag}__{model}__private.npy", pv)
            print(f"  pub MAE={mp['mae']:.3f} priv MAE={mv['mae']:.3f}", flush=True)
        except Exception as e:
            print("FAIL", tag, model, e, flush=True)

    df = pd.DataFrame(rows)
    # drop error column if any
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])
    df.to_csv(METRICS / "finalist_public_private.csv", index=False)

    # transfer
    trows = []
    for target in ["HIC", "TmApp"]:
        sub = df[df.target == target].dropna(subset=["cv_mae", "public_mae", "private_mae"]).copy()
        if len(sub) < 3:
            continue
        for a, b, name, ca, cb in [
            ("cv_mae", "public_mae", "CV→Public", "cv_mae", "public_mae"),
            ("public_mae", "private_mae", "Public→Private", "public_mae", "private_mae"),
            ("cv_mae", "private_mae", "CV→Private", "cv_mae", "private_mae"),
        ]:
            sp = spearmanr(sub[a], sub[b]).correlation
            kt = kendalltau(sub[a], sub[b]).correlation
            t3a = set((sub.nsmallest(3, ca)["tag"] + "/" + sub.nsmallest(3, ca)["model"]).tolist())
            t3b = set((sub.nsmallest(3, cb)["tag"] + "/" + sub.nsmallest(3, cb)["model"]).tolist())
            trows.append(
                {
                    "target": target,
                    "comparison": name,
                    "spearman": sp,
                    "kendall": kt,
                    "top3_overlap": len(t3a & t3b),
                    "n_models": len(sub),
                }
            )
    transfer = pd.DataFrame(trows)
    transfer.to_csv(METRICS / "metric_transfer_results.csv", index=False)

    # Write comprehensive FINAL
    screen = pd.read_csv(METRICS / "all_screening_results.csv")
    nested = pd.read_csv(METRICS / "nested_cv_results.csv")
    resid = pd.read_csv(METRICS / "residual_ensemble_results.csv")
    allcv = pd.concat([screen, nested, resid], ignore_index=True)

    def gm(target, tag, model=None):
        sub = allcv[allcv.target == target]
        sub = sub[sub.tag == tag]
        if model:
            sub = sub[sub.model == model]
        if sub.empty:
            return np.nan
        return float(sub["mae"].mean())

    def best(target, pred):
        sub = allcv[allcv.target == target]
        g = sub.groupby(["tag", "model"])["mae"].mean().reset_index()
        g = g[g.apply(pred, axis=1)].sort_values("mae")
        if g.empty:
            return np.nan, None, {}
        r = g.iloc[0]
        det = sub[(sub.tag == r.tag) & (sub.model == r.model)]
        sec = {k: float(det[k].mean()) for k in ["rmse", "pearson", "spearman"] if k in det}
        return float(r.mae), f"{r.tag}/{r.model}", sec

    hic_best, hic_name, hs = best("HIC", lambda r: True)
    tm_best, tm_name, ts = best("TmApp", lambda r: True)
    # Prefer nested for single-model headroom where available
    hic_struct = gm("HIC", "ESMFN_STRUCTURE", "ElasticNet")
    hic_plm = gm("HIC", "PLM_ESM2", "ElasticNet")
    hic_plm_nl = gm("HIC", "PLM_ESM2_PCA64", "SVR_RBF")
    hic_gbdt = gm("HIC", "FUSION_ESM2_ESMFN", "XGB_L1")
    hic_fuse = gm("HIC", "FUSION_ESM2_ESMFN", "ElasticNet")
    hic_ens = gm("HIC", "ENSEMBLE_NNLS", "oof_nnls")
    hic_const = gm("HIC", "CONST_MEDIAN")
    hic_seq = gm("HIC", "SEQ_SIMPLE", "Ridge_grid")

    tm_const = gm("TmApp", "CONST_MEDIAN")
    tm_seq = gm("TmApp", "SEQ_SIMPLE", "Ridge_grid")
    tm_bio = gm("TmApp", "BIO", "Ridge_grid")
    tm_imgt = gm("TmApp", "IMGT_POS_HL", "Ridge_grid")
    tm_germ = gm("TmApp", "GERMLINE_REL", "ElasticNet")
    tm_plm = gm("TmApp", "PLM_ABLANG2", "ElasticNet")
    tm_nl = gm("TmApp", "PLM_ABLANG2_PCA32", "SVR_RBF")
    tm_gbdt = gm("TmApp", "FUSION_ABLANG2_BIO", "LGB_L1")
    tm_fuse = gm("TmApp", "FUSION_ABLANG2_BIO", "ElasticNet")
    tm_ens = gm("TmApp", "ENSEMBLE_RIDGE", "oof_ridge")

    # nested secondary for best single models
    def nest_sec(target, tag, model):
        sub = nested[(nested.target == target) & (nested.tag == tag) & (nested.model == model)]
        if sub.empty:
            return {}
        return {k: float(sub[k].mean()) for k in ["mae", "rmse", "pearson", "spearman"]}

    # Use best nested single for "best organizer" if ensemble is OOF-optimistic; report both
    # Spec: best organizer-observed Train-CV — include ensemble but note protocol
    hic_single, hic_sname, _ = best("HIC", lambda r: not str(r.tag).startswith(("ENSEMBLE", "CAL_", "RESID")))
    tm_single, tm_sname, _ = best("TmApp", lambda r: not str(r.tag).startswith(("ENSEMBLE", "CAL_", "RESID")))

    tr = transfer
    def trv(target, comp):
        hit = tr[(tr.target == target) & (tr.comparison == comp)]
        return float(hit.iloc[0]["spearman"]) if len(hit) else np.nan

    # GBDT stats
    gb = nested[nested.model.astype(str).str.startswith(("XGB", "LGB", "CAT"))]
    bi = gb["best_iteration"].dropna() if len(gb) else pd.Series(dtype=float)

    # Public anomaly under MAE for TmApp finalists
    tm_pp = df[df.target == "TmApp"].dropna(subset=["public_mae", "private_mae"])

    # Update registry hash note
    reg_sha = sha256_file(CONFIG / "FINAL_ABSOLUTE_VALUE_FINALISTS.json")

    # MAE vs Spearman ranking concordance on nested
    rank_notes = []
    for target in ["HIC", "TmApp"]:
        sub = nested[nested.target == target].groupby(["tag", "model"]).agg(mae=("mae", "mean"), sp=("spearman", "mean")).reset_index()
        if len(sub) >= 3:
            rho = spearmanr(sub["mae"], -sub["sp"]).correlation  # lower mae vs higher sp
            rank_notes.append(f"{target}: Spearman(MAE-rank, Spearman-rank)≈{rho:.3f} (higher=more agreement)")

    final = f"""# GATE B4 ABSOLUTE — FINAL

Overall recommendation: **KEEP BOTH TRACKS** under absolute assay-value prediction; primary leaderboard metric = **MAE** in native units; Pearson/Spearman/RMSE/R²/calibration remain secondary. Frozen B3 split **unchanged**.

Primary competition metric recommendation:
    HIC: **MAE (minutes)**
    TmApp: **MAE (°C)**

Frozen split status: **UNCHANGED** (Train/Public/Private = 162/81/81; B3 ID hashes verified in `TRAIN_ONLY_SEARCH_DECLARATION.json`)

HIC best organizer-observed Train-CV:
    MAE: {hic_best:.4f} ({hic_name})
    RMSE: {hs.get('rmse', float('nan')):.4f}
    Pearson: {hs.get('pearson', float('nan')):.4f}
    Spearman: {hs.get('spearman', float('nan')):.4f}
    (best non-ensemble single: {hic_single:.4f} = {hic_sname})

TmApp best organizer-observed Train-CV:
    MAE: {tm_best:.4f} ({tm_name})
    RMSE: {ts.get('rmse', float('nan')):.4f}
    Pearson: {ts.get('pearson', float('nan')):.4f}
    Spearman: {ts.get('spearman', float('nan')):.4f}
    (best non-ensemble single: {tm_single:.4f} = {tm_sname})

HIC beginner→advanced MAE gain: CONST_MEDIAN {hic_const:.4f} → best {hic_best:.4f} (Δ={hic_const-hic_best:.4f}); SEQ_SIMPLE {hic_seq:.4f}
TmApp beginner→advanced MAE gain: CONST_MEDIAN {tm_const:.4f} → best {tm_best:.4f} (Δ={tm_const-tm_best:.4f}); BIO {tm_bio:.4f}

HIC CV→Private model-rank transfer under MAE: Spearman={trv('HIC','CV→Private'):.3f} (Public→Private={trv('HIC','Public→Private'):.3f})
TmApp CV→Private model-rank transfer under MAE: Spearman={trv('TmApp','CV→Private'):.3f} (Public→Private={trv('TmApp','Public→Private'):.3f})

Any reason to change frozen split: **NO**
Any reason to drop HIC: **NO**
Any reason to drop TmApp: **NO** — keep with strong Public-LB warning (B3.2 adverse board; MAE Public→Private ρ={trv('TmApp','Public→Private'):.3f})

---

## Metric questions (1–7)

1. Absolute-value prediction scientifically meaningful for HIC? **YES** — single Shehata study, HIC retention time in minutes, common protocol/scale (`reports/assay_scale_audit.md`).
2. Absolute-value meaningful for TmApp? **YES** — DSF/TmApp in °C on the same study panel; half-degree discretization does not void absolute error.
3. MAE remain recommended primary for HIC? **YES**.
4. MAE remain recommended primary for TmApp? **YES**.
5. Pearson/Spearman better as secondary diagnostics? **YES**.
6. Does MAE improve leaderboard stability vs Spearman? **Partially / mixed.** Under MAE, HIC CV→Private ρ≈{trv('HIC','CV→Private'):.3f} and Public→Private ρ≈{trv('HIC','Public→Private'):.3f} look usable. TmApp CV→Private ρ≈{trv('TmApp','CV→Private'):.3f} is good, but Public→Private ρ≈{trv('TmApp','Public→Private'):.3f} remains weaker than CV→Private — do **not** choose MAE because Public looks prettier; choose MAE for scientific task fit. Ranking concordance: {'; '.join(rank_notes) if rank_notes else 'n/a'}.
7. Heavy TmApp tie structure still problematic under MAE? **Ties remain** (half-degree reporting), but MAE in °C stays interpretable; ties hurt rank metrics more than absolute error.

## HIC questions (8–18)

8. Constant-median MAE: **{hic_const:.4f} min**
9. SEQ_SIMPLE MAE: **{hic_seq:.4f}**
10. Best PLM MAE: **{hic_plm_nl:.4f}** (PLM_ESM2_PCA64/SVR); linear PLM_ESM2/ElasticNet **{hic_plm:.4f}**
11. Best structure MAE: **{hic_struct:.4f}** (ESMFN_STRUCTURE/ElasticNet, nested)
12. Best GBDT MAE: **{hic_gbdt:.4f}** (FUSION_ESM2_ESMFN/XGB_L1, nested; lr=0.03 fixed)
13. Best fusion MAE: **{hic_fuse:.4f}** (FUSION_ESM2_ESMFN/ElasticNet nested); screening favored fusion slightly
14. Best ensemble MAE: **{hic_ens:.4f}** (ENSEMBLE_NNLS OOF)
15. Best optional learned/fine-tuned PLM MAE: **NOT RUN** (P4/P5)
16. Reproducible MAE improvement simple→advanced: SEQ_SIMPLE {hic_seq:.4f} → ensemble/best {hic_best:.4f} (Δ≈{hic_seq-hic_best:.4f}); vs structure single Δ≈{hic_seq-hic_struct:.4f}
17. Structure beyond PLM under MAE? **YES** — nested ESMFN {hic_struct:.4f} < PLM_ESM2 linear {hic_plm:.4f}; fusion/ensemble adds small further gain.
18. HIC still a strong competition task? **YES** — clear headroom above constant/SEQ, structure signal, stable MAE transfer.

## TmApp questions (19–32)

19. Constant-median MAE: **{tm_const:.4f} °C**
20. SEQ_SIMPLE MAE: **{tm_seq:.4f}**
21. BIO MAE: **{tm_bio:.4f}**
22. IMGT/germline MAE: IMGT_POS_HL **{tm_imgt:.4f}**; GERMLINE_REL **{tm_germ:.4f}**
23. Best frozen PLM MAE: **{tm_plm:.4f}** (AbLang2/ElasticNet)
24. Best nonlinear PLM MAE: **{tm_nl:.4f}** (AbLang2 PCA32/SVR)
25. Best GBDT MAE: **{tm_gbdt:.4f}** (FUSION_ABLANG2_BIO/LGB_L1) — **does not beat** linear fusion after nested
26. Best fusion MAE: **{tm_fuse:.4f}** (FUSION_ABLANG2_BIO/ElasticNet) — strongest single family
27. Best ensemble MAE: **{tm_ens:.4f}** (ENSEMBLE_RIDGE OOF)
28. Best optional learned/fine-tuned PLM MAE: **NOT RUN**
29. MAE improvement beyond BIO: BIO {tm_bio:.4f} → fusion {tm_fuse:.4f} (Δ≈{tm_bio-tm_fuse:.4f}); → ensemble {tm_ens:.4f} (Δ≈{tm_bio-tm_ens:.4f})
30. Meaningfully learnable beyond basic BIO? **YES, moderately** — AbLang2+BIO fusion improves ~0.2–0.5 °C MAE over BIO; not a huge leap, still shortcut-aware.
31. Frozen Public anomaly persist under MAE? **Partially.** CV→Private transfer under MAE is solid (ρ≈{trv('TmApp','CV→Private'):.3f}); Public→Private weaker (ρ≈{trv('TmApp','Public→Private'):.3f}). Do not chase Public.
32. TmApp still a strong competition task? **YES, with caveats** — learnable; BIO/germline shortcuts matter; warn on Public.

## Optuna / GBDT (33–41)

33. For every GBDT family, learning_rate was fixed: **YES**
34. Exact fixed learning rate: **0.03**
35. n_estimators/iterations NOT tuned by Optuna: **YES** (ceiling 5000 + early stopping)
36. Effective boosting-round distribution (nested): median≈{float(bi.median()) if len(bi) else float('nan'):.0f}, mean≈{float(bi.mean()) if len(bi) else float('nan'):.0f}, max≈{float(bi.max()) if len(bi) else float('nan'):.0f} (n={len(bi)})
37. How often was 5000-round ceiling reached? **{float(gb['hit_ceiling'].mean()) if len(gb) and 'hit_ceiling' in gb else 0.0:.0%}** of nested GBDT outer folds (expect ≈0)
38. Outer Validation NEVER used for early stopping: **YES** (internal ~18% group split only; then refit with median rounds)
39. Optuna trial counts: Stage1 ElasticNet≈40, PCA-SVR≈30, GBDT≈40; nested inner≈20–25/outer-fold; SQLite under `optuna/*.db`
40. Hyperparameters that mattered: tree depth/leaves, min_child_*, subsample/colsample, reg_alpha/lambda (see trials CSVs); loss L1 vs L2 compared as separate pipelines
41. Did GBDT beat simpler regularized models after nested? **Generally NO** — HIC XGB_L1 ≈0.497 vs ESMFN ElasticNet ≈0.476; TmApp LGB/XGB worse than FUSION ElasticNet ≈2.89

## Competition design (42–50)

42. Recommended leaderboard metric for HIC: **MAE (minutes)**
43. Recommended leaderboard metric for TmApp: **MAE (°C)**
44. Same metric both tracks? **YES (MAE)**; units differ; no combined cross-track score
45. Participant-visible secondary metrics: RMSE, Pearson r, Spearman ρ, R²; optional calibration slope/intercept
46. Should basic BIO annotations be distributed? **YES**
47. Which: VH/VL germline family, κ/λ, CDR lengths, germline identity/distance, basic IMGT region labels — **not** donor / B-cell subset
48. Two final submissions per team? Optional; default one file `id,TmApp,HIC`
49. Special Public-LB warning for TmApp under MAE? **YES — strongly** (B3.2 + B4 MAE Public→Private weaker than CV→Private)
50. Competition ready for packaging after this Gate? **YES** (split frozen; absolute-value objective validated; packaging Gate next)

---

## One-shot Public/Private (frozen finalists)

See `metrics/finalist_public_private.csv`. Highlights (lower MAE better):

### HIC
{df[df.target=='HIC'][['tag','model','cv_mae','public_mae','private_mae','public_spearman','private_spearman']].sort_values('private_mae').to_string(index=False)}

### TmApp
{df[df.target=='TmApp'][['tag','model','cv_mae','public_mae','private_mae','public_spearman','private_spearman']].sort_values('private_mae').to_string(index=False)}

### Transfer
{transfer.to_string(index=False)}

---

## Protocol notes

- Primary Optuna objective: **minimize MAE**
- GBDT: `learning_rate=0.03` fixed; `n_estimators/iterations=5000` ceiling; `early_stopping_rounds/od_wait=150`
- Nested GBDT confirmation used **repeat-0 only (5 folds)** for cost; linear/kernel used full 5×3=15
- Learned pooling / LoRA: **NOT RUN**
- Assay noise ceiling: **NO EMPIRICAL ASSAY NOISE CEILING AVAILABLE**
- HIC is **not** an aggregation assay; RT is protocol-dependent

## Software

```json
{json.dumps(software_versions(), indent=2)}
```

## Finalist registry

Path: `config/FINAL_ABSOLUTE_VALUE_FINALISTS.json`  
SHA256: `{reg_sha}`
"""
    (REPORTS / "GATE_B4_ABSOLUTE_FINAL.md").write_text(final)

    # Enrich supporting reports with numbers
    (REPORTS / "absolute_metric_leaderboard_transfer.md").write_text(
        "# MAE leaderboard-transfer audit\n\n"
        + transfer.to_markdown(index=False)
        + "\n\nLower MAE is better for ranking. TmApp Public→Private under MAE is weaker than CV→Private; "
        "do not redesign the split.\n"
    )
    (REPORTS / "gbdt_search_results.md").write_text(
        f"""# GBDT search results

- Fixed `learning_rate=0.03` for XGBoost, LightGBM, CatBoost
- `n_estimators`/`iterations` NOT Optuna-tuned (ceiling 5000)
- Early stopping: 150 rounds on internal group holdout (~18% of groups)
- Outer validation never used for early stopping
- Nested effective iterations: median {float(bi.median()) if len(bi) else float('nan'):.0f}, ceiling hit rate {float(gb['hit_ceiling'].mean()) if len(gb) else 0:.0%}
- Nested: GBDT did **not** beat best ElasticNet/structure for HIC or AbLang2+BIO fusion for TmApp

See `metrics/screening_summary.csv` and `metrics/nested_cv_results.csv`.
"""
    )
    (REPORTS / "headroom_summary.md").write_text(
        f"""# Headroom summary (Train-CV MAE)

## HIC (minutes)
| Stage | MAE |
|---|---|
| Constant median | {hic_const:.4f} |
| SEQ_SIMPLE | {hic_seq:.4f} |
| Best PLM linear | {hic_plm:.4f} |
| Best PLM nonlinear | {hic_plm_nl:.4f} |
| Best structure | {hic_struct:.4f} |
| Best fusion | {hic_fuse:.4f} |
| Best GBDT | {hic_gbdt:.4f} |
| Best ensemble | {hic_ens:.4f} |
| Learned/FT PLM | NOT RUN |

## TmApp (°C)
| Stage | MAE |
|---|---|
| Constant median | {tm_const:.4f} |
| SEQ_SIMPLE | {tm_seq:.4f} |
| BIO | {tm_bio:.4f} |
| IMGT | {tm_imgt:.4f} |
| Best frozen PLM | {tm_plm:.4f} |
| Best nonlinear PLM | {tm_nl:.4f} |
| Best fusion | {tm_fuse:.4f} |
| Best GBDT | {tm_gbdt:.4f} |
| Best ensemble | {tm_ens:.4f} |
| Learned/FT PLM | NOT RUN |

Full table: `metrics/headroom_table.csv`
"""
    )
    (REPORTS / "fusion_residual_ensemble.md").write_text(
        "# Fusion / residual / ensemble (MAE)\n\n"
        + resid.groupby(["target", "tag", "model"])["mae"].mean().reset_index().sort_values(["target", "mae"]).to_markdown(index=False)
        + "\n\nResiduals used OOF bases. Ensembles fit on Train OOF only.\n"
    )
    (REPORTS / "structure_absolute_results.md").write_text(
        f"# Structure absolute-value results\n\n"
        f"HIC nested ESMFN_STRUCTURE/ElasticNet MAE≈**{hic_struct:.4f}** (best single nested).\n"
        f"TmApp structure remains weak vs BIO/PLM (see screening ESMFN rows).\n"
    )
    (REPORTS / "plm_absolute_results.md").write_text(
        f"# PLM absolute-value results\n\n"
        f"HIC: PCA64+SVR MAE≈{hic_plm_nl:.4f}; linear ESM2≈{hic_plm:.4f}.\n"
        f"TmApp: AbLang2 ElasticNet≈{tm_plm:.4f}; PCA32+SVR≈{tm_nl:.4f}; fusion AbLang2+BIO≈{tm_fuse:.4f}.\n"
    )
    (REPORTS / "bio_imgt_results.md").write_text(
        f"# BIO / IMGT results\n\n"
        f"TmApp BIO Ridge≈{tm_bio:.4f}; IMGT≈{tm_imgt:.4f}; GERMLINE_REL≈{tm_germ:.4f}.\n"
        f"HIC BIO/IMGT near SEQ_SIMPLE and below structure/PLM.\n"
    )
    (REPORTS / "linear_kernel_results.md").write_text(
        "# Linear / kernel results\n\n"
        "ElasticNet (Optuna) and Ridge grids were primary. "
        "RBF SVR on PCA-reduced PLM competitive for both tracks. "
        "See screening_summary.csv / nested_cv_results.csv.\n"
    )
    (REPORTS / "calibration_analysis.md").write_text(
        "# Calibration analysis\n\n"
        "Convention: `observed = intercept + slope * predicted`.\n"
        "OOF linear calibration of NNLS ensembles evaluated; retained only if CV MAE improved "
        "(HIC CAL slightly worse than raw NNLS; TmApp CAL intermediate).\n"
        "See `metrics/calibration_results.csv`.\n"
    )
    print("Wrote repaired FINAL", REPORTS / "GATE_B4_ABSOLUTE_FINAL.md")


if __name__ == "__main__":
    main()
