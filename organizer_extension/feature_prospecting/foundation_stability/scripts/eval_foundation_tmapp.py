#!/usr/bin/env python3
"""TmApp eval for foundation features: standalone / incremental / residual."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FS = FP / "foundation_stability"
sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    B_BOOT,
    OOF_DIR,
    TMAPP_REF,
    bootstrap_delta,
    mae,
    median_baseline_oof,
    metrics,
    nested_ridge,
)

EMB = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
SOL = ROOT / "competition/data/secret/solution.csv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_N = 32


def load_y():
    return pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)


def load_plm():
    z = np.load(EMB, allow_pickle=True)
    return pd.DataFrame(np.asarray(z["ablang2__HL_paired"], float), index=[str(x) for x in z["ids"]])


CANON_FENNIX = [
    "ref_F_rms",
    "ref_F_q90",
    "ref_F_max",
    "ref_E_per_atom",
    "P1_dE_mean",
    "P1_dE_median",
    "P1_dE_sd",
    "P1_dE_q10",
    "P1_dE_q90",
    "P1_dE_max",
    "P1_dE_per_atom_mean",
    "P1_Frms_mean",
    "P1_Frms_q90",
    "P2_dE_mean",
    "P2_dE_sd",
    "P2_dE_q90",
    "P2_Frms_mean",
    "P3_dE_mean",
    "P3_dE_sd",
    "P3_dE_q90",
    "P3_Frms_mean",
    "CDR_ref_F_rms",
    "CDR_P1_dE_mean",
    "interface_ref_F_rms",
    "HCDR3_ref_F_rms",
]
# CDR_* often all-NaN when IMGT index join fails; dropped at use-time if non-finite.


def plm_pca_oof(X_plm, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_plm.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        sc0 = StandardScaler()
        Ztr = sc0.fit_transform(X_plm.loc[tr])
        Zte = sc0.transform(X_plm.loc[te])
        pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        # nested alpha on Ptr
        groups = np.array([fold_map[i] for i in tr])
        Xtr = pd.DataFrame(Ptr, index=tr)
        a = select_alpha_df(Xtr, y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Ptr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Pte))
    return oof


def select_alpha_df(X, y, folds, train_ids):
    fold_map = folds.set_index("id")["fold"].to_dict()
    uniq = sorted({fold_map[i] for i in train_ids})
    n_splits = min(5, len(uniq))
    # use remaining groups
    best_a, best = 100.0, np.inf
    for a in ALPHAS:
        maes = []
        for f in uniq:
            te = [i for i in train_ids if fold_map[i] == f]
            tr = [i for i in train_ids if fold_map[i] != f]
            if len(te) == 0 or len(tr) < 2:
                continue
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(X.loc[tr]), y.loc[tr])
            maes.append(mae(y.loc[te], m.predict(sc.transform(X.loc[te]))))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def fuse_oof(X_plm, X_phys, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_phys.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        sc0 = StandardScaler()
        Ztr = sc0.fit_transform(X_plm.loc[tr])
        Zte = sc0.transform(X_plm.loc[te])
        pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        scp = StandardScaler()
        Xptr = scp.fit_transform(X_phys.loc[tr].fillna(X_phys.loc[tr].median()))
        Xpte = scp.transform(X_phys.loc[te].fillna(X_phys.loc[tr].median()))
        Xtr = np.hstack([Ptr, Xptr])
        Xte = np.hstack([Pte, Xpte])
        # alpha select
        # build df for selector using fake ids
        Xdf = pd.DataFrame(Xtr, index=tr)
        a = select_alpha_df(Xdf, y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Xte))
    return oof


def residual_oof(X_phys, y, folds, plm_oof_builder):
    """Leakage-safe: within each outer fold, cross-fit PLM on outer-train, residual = y - plm_hat, fit physics->residual."""
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_phys.index)
    cand = pd.Series(index=ids, dtype=float)
    resid_corr = []
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        # cross-fit PLM on outer train only
        plm_hat_tr = plm_oof_builder(tr)
        resid_tr = y.loc[tr] - plm_hat_tr.loc[tr]
        X = X_phys.loc[tr].fillna(X_phys.loc[tr].median())
        a = select_alpha_df(X, resid_tr, folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(X), resid_tr)
        # PLM pred on te: fit PLM on full outer train
        plm_te = plm_predict_train(tr, te)
        pred_resid_te = m.predict(sc.transform(X_phys.loc[te].fillna(X_phys.loc[tr].median())))
        cand.loc[te] = plm_te + pred_resid_te
        # store residual model corr on train for diagnostics
        pred_r_tr = m.predict(sc.transform(X))
        if np.std(resid_tr) > 0 and np.std(pred_r_tr) > 0:
            resid_corr.append(float(np.corrcoef(resid_tr, pred_r_tr)[0, 1]))
    return cand, float(np.mean(resid_corr)) if resid_corr else np.nan


# globals set in main for closures
PLM_DF = None
Y = None
FOLDS = None


def plm_predict_train(tr, te):
    sc0 = StandardScaler()
    Ztr = sc0.fit_transform(PLM_DF.loc[tr])
    Zte = sc0.transform(PLM_DF.loc[te])
    pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
    Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
    Xdf = pd.DataFrame(Ptr, index=tr)
    a = select_alpha_df(Xdf, Y.loc[tr], FOLDS, tr)
    sc = StandardScaler()
    m = Ridge(alpha=a, random_state=0)
    m.fit(sc.fit_transform(Ptr), Y.loc[tr])
    return m.predict(sc.transform(Pte))


def plm_oof_builder(tr_ids):
    """OOF PLM predictions on tr_ids using only subgroups within tr_ids."""
    sub = FOLDS[FOLDS.id.isin(tr_ids)]
    return plm_pca_oof(PLM_DF.loc[tr_ids], Y.loc[tr_ids], sub)


def eval_family(name, feat_path, gen_label, folds, y, sol, plm):
    global PLM_DF, Y, FOLDS
    PLM_DF, Y, FOLDS = plm, y, folds
    df = pd.read_parquet(feat_path) if str(feat_path).endswith(".parquet") else pd.read_csv(feat_path)
    df = df[df.extraction_status == "SUCCESS"].set_index("id")
    cols = [c for c in CANON_FENNIX if c in df.columns]
    X_all = df[cols].astype(float)
    keep = [c for c in X_all.columns if X_all[c].notna().mean() >= 0.5]
    X_all = X_all[keep]
    cols = keep
    # Dev alignment for CV
    ids = [i for i in folds.id if i in X_all.index and i in y.index and i in plm.index]
    X = X_all.loc[ids]
    yy = y.loc[ids]
    folds_u = folds[folds.id.isin(ids)]
    plm_u = plm.loc[ids]

    # standalone
    oof_s, *_ = nested_ridge(X.fillna(X.median()), yy, folds_u)
    base_oof, base_med = median_baseline_oof(yy, folds_u)
    # PLM only
    oof_plm = plm_pca_oof(plm_u, yy, folds_u)
    # incremental
    oof_fuse = fuse_oof(plm_u, X, yy, folds_u)
    # residual reconstructed
    oof_resid, resid_r = residual_oof(X, yy, folds_u, plm_oof_builder)

    pub = list(sol.loc[sol.is_public == 1, "id"])
    pri = list(sol.loc[sol.is_private == 1, "id"])
    y_te = sol.set_index("id")["TmApp"].astype(float)

    def fit_predict_test(kind):
        # fit on all Dev ids; predict Public/Private that have features+PLM
        tr = ids
        te = [i for i in y_te.index if i in X_all.index and i in plm.index and i not in set(ids)]
        if len(te) == 0:
            return te, np.array([])
        if kind == "standalone":
            sc = StandardScaler()
            a = select_alpha_df(X.loc[tr].fillna(X.loc[tr].median()), yy.loc[tr], folds_u, tr)
            m = Ridge(alpha=a, random_state=0)
            Xt = X.loc[tr].fillna(X.loc[tr].median())
            Xte = X_all.loc[te][cols].fillna(X.loc[tr].median())
            m.fit(sc.fit_transform(Xt), yy.loc[tr])
            return te, m.predict(sc.transform(Xte))
        if kind == "plm":
            pred = plm_predict_train(tr, te)
            return te, pred
        if kind == "fuse":
            sc0 = StandardScaler()
            Ztr = sc0.fit_transform(plm.loc[tr])
            Zte = sc0.transform(plm.loc[te])
            pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
            Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
            scp = StandardScaler()
            Xptr = scp.fit_transform(X.loc[tr].fillna(X.loc[tr].median()))
            Xpte = scp.transform(X_all.loc[te][cols].fillna(X.loc[tr].median()))
            Xtr = np.hstack([Ptr, Xptr])
            Xte = np.hstack([Pte, Xpte])
            a = select_alpha_df(pd.DataFrame(Xtr, index=tr), yy.loc[tr], folds_u, tr)
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(Xtr), yy.loc[tr])
            return te, m.predict(sc.transform(Xte))
        raise ValueError(kind)

    rows = []
    # CV metrics
    for label, oof, ref in [
        ("STANDALONE", oof_s, base_oof),
        ("PLM_ONLY", oof_plm, base_oof),
        ("PLM_PLUS", oof_fuse, oof_plm),
        ("RESIDUAL_RECON", oof_resid, oof_plm),
    ]:
        m = metrics(yy, oof)
        d = bootstrap_delta(yy, oof, ref, b=B_BOOT)
        rows.append(
            {
                "family": name,
                "generator": gen_label,
                "mode": label,
                "split": "CV",
                "MAE": m["MAE"],
                "Pearson": m["Pearson"],
                "Spearman": m["Spearman"],
                "delta_vs_ref": d["delta"],
                "boot_lo": d["ci95_low"],
                "boot_hi": d["ci95_high"],
                "p_improve": d["p_improve"],
                "ref": "MEDIAN" if label == "STANDALONE" else ("MEDIAN" if label == "PLM_ONLY" else "PLM_ONLY"),
                "resid_train_corr_mean": resid_r if label == "RESIDUAL_RECON" else np.nan,
                "n_features": len(cols),
            }
        )

    # Public/Private for standalone and fuse
    for kind, mode in [("standalone", "STANDALONE"), ("plm", "PLM_ONLY"), ("fuse", "PLM_PLUS")]:
        te, pred = fit_predict_test(kind)
        if len(te) == 0:
            continue
        for split, id_list in [("Public", pub), ("Private", pri)]:
            ids_s = [i for i in id_list if i in te]
            if not ids_s:
                continue
            idx = [te.index(i) for i in ids_s]
            yy_s = y_te.loc[ids_s]
            pp = np.asarray(pred)[idx]
            if mode == "STANDALONE":
                ref = np.full(len(ids_s), float(np.median(yy)))
            else:
                _, pred_plm = fit_predict_test("plm")
                ref = np.asarray(pred_plm)[idx]
            m = metrics(yy_s, pp)
            d = bootstrap_delta(yy_s, pp, ref, b=B_BOOT)
            rows.append(
                {
                    "family": name,
                    "generator": gen_label,
                    "mode": mode,
                    "split": split,
                    "MAE": m["MAE"],
                    "Pearson": m["Pearson"],
                    "Spearman": m["Spearman"],
                    "delta_vs_ref": d["delta"],
                    "boot_lo": d["ci95_low"],
                    "boot_hi": d["ci95_high"],
                    "p_improve": d["p_improve"],
                    "ref": "MEDIAN" if mode == "STANDALONE" else "PLM_ONLY",
                    "n_features": len(cols),
                }
            )
    return rows, cols


def robustness(path_esm, path_abb):
    a = pd.read_parquet(path_esm) if str(path_esm).endswith(".parquet") else pd.read_csv(path_esm)
    b = pd.read_parquet(path_abb) if str(path_abb).endswith(".parquet") else pd.read_csv(path_abb)
    a = a[a.extraction_status == "SUCCESS"].set_index("id")
    b = b[b.extraction_status == "SUCCESS"].set_index("id")
    ids = sorted(set(a.index) & set(b.index))
    cols = [c for c in CANON_FENNIX if c in a.columns and c in b.columns]
    rows = []
    for c in cols:
        x, y = a.loc[ids, c].astype(float), b.loc[ids, c].astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 10:
            continue
        from scipy.stats import pearsonr, spearmanr

        pr = pearsonr(x, y).statistic
        sp = spearmanr(x, y).statistic
        nad = float(np.mean(np.abs(x - y)) / (np.std(x) + 1e-12))
        sign = float(np.mean(np.sign(x) == np.sign(y))) if np.any(x != 0) else np.nan
        rows.append({"feature": c, "pearson": pr, "spearman": sp, "norm_abs_diff": nad, "sign_consistency": sign})
    rdf = pd.DataFrame(rows)
    # classify by median spearman of response features
    med = float(rdf["spearman"].median()) if len(rdf) else 0
    label = "ROBUST" if med >= 0.7 else ("MODERATE" if med >= 0.4 else "FRAGILE")
    return rdf, label


def main():
    y = load_y()
    plm = load_plm()
    folds = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    sol = pd.read_csv(SOL)
    feat_dir = FS / "cache/fennix_features"
    all_rows = []
    for gen, lab in [("esmfold", "ESMFold"), ("abodybuilder2", "ABB2")]:
        path = feat_dir / f"features_{gen}.parquet"
        if not path.exists():
            path = feat_dir / f"features_{gen}.csv"
        if not path.exists():
            print("missing", path)
            continue
        rows, cols = eval_family("FENNIX_BIO1S", path, lab, folds, y, sol, plm)
        all_rows.extend(rows)
        # shadow
        rows_s, _ = eval_family("FENNIX_BIO1S", path, lab + "_SHADOW", shadow, y, sol, plm)
        for r in rows_s:
            if r["split"] == "CV":
                r["split"] = "Shadow"
                all_rows.append(r)

    out = pd.DataFrame(all_rows)
    out.to_csv(FS / "results/FOUNDATION_ALL_EVAL.csv", index=False)
    out.loc[out["mode"].isin(["STANDALONE", "PLM_ONLY"])].to_csv(
        FS / "FOUNDATION_STANDALONE_RESULTS.csv", index=False
    )
    out.loc[out["mode"] == "PLM_PLUS"].to_csv(FS / "FOUNDATION_INCREMENTAL_RESULTS.csv", index=False)
    out.loc[out["mode"] == "RESIDUAL_RECON"].to_csv(FS / "FOUNDATION_RESIDUAL_RESULTS.csv", index=False)

    rob, rob_lab = robustness(
        feat_dir / "features_esmfold.parquet", feat_dir / "features_abodybuilder2.parquet"
    )
    rob.to_csv(FS / "FOUNDATION_ROBUSTNESS.csv", index=False)
    (FS / "results/robustness_class.txt").write_text(rob_lab + "\n")
    print("robustness", rob_lab)
    print(out.groupby(["generator", "mode", "split"], observed=True)["MAE"].mean())


if __name__ == "__main__":
    main()
