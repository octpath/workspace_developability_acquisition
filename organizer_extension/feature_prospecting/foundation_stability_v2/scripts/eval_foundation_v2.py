#!/usr/bin/env python3
"""TmApp eval for Foundation Stability v2 (FeNNix curvature + BioEmu ensemble)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
V1 = FP / "foundation_stability"
V2 = FP / "foundation_stability_v2"
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

PLM_DF = None
Y = None
FOLDS = None


def load_y():
    return pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)


def load_plm():
    z = np.load(EMB, allow_pickle=True)
    return pd.DataFrame(np.asarray(z["ablang2__HL_paired"], float), index=[str(x) for x in z["ids"]])


def select_alpha_df(X, y, folds, train_ids):
    fold_map = folds.set_index("id")["fold"].to_dict()
    uniq = sorted({fold_map[i] for i in train_ids})
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
        a = select_alpha_df(pd.DataFrame(Ptr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Ptr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Pte))
    return oof


def plm_predict_train(tr, te):
    sc0 = StandardScaler()
    Ztr = sc0.fit_transform(PLM_DF.loc[tr])
    Zte = sc0.transform(PLM_DF.loc[te])
    pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
    Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
    a = select_alpha_df(pd.DataFrame(Ptr, index=tr), Y.loc[tr], FOLDS, tr)
    sc = StandardScaler()
    m = Ridge(alpha=a, random_state=0)
    m.fit(sc.fit_transform(Ptr), Y.loc[tr])
    return m.predict(sc.transform(Pte))


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
        med = X_phys.loc[tr].median()
        Xptr = scp.fit_transform(X_phys.loc[tr].fillna(med))
        Xpte = scp.transform(X_phys.loc[te].fillna(med))
        Xtr = np.hstack([Ptr, Xptr])
        Xte = np.hstack([Pte, Xpte])
        a = select_alpha_df(pd.DataFrame(Xtr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Xte))
    return oof


def residual_oof(X_phys, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_phys.index)
    cand = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        sub = folds[folds.id.isin(tr)]
        plm_hat_tr = plm_pca_oof(PLM_DF.loc[tr], y.loc[tr], sub)
        resid_tr = y.loc[tr] - plm_hat_tr.loc[tr]
        med = X_phys.loc[tr].median()
        X = X_phys.loc[tr].fillna(med)
        a = select_alpha_df(X, resid_tr, folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(X), resid_tr)
        plm_te = plm_predict_train(tr, te)
        pred_r = m.predict(sc.transform(X_phys.loc[te].fillna(med)))
        cand.loc[te] = plm_te + pred_r
    return cand


def fennix_feature_cols(df):
    prefer = [
        "K_all_bb_mean",
        "K_all_bb_median",
        "K_all_bb_q10",
        "K_all_bb_q90",
        "K_all_bb_iqr",
        "K_all_bb_frac_neg",
        "K_FW_mean",
        "K_FW_median",
        "K_FW_q90",
        "K_CDR_mean",
        "K_CDR_median",
        "K_CDR_q90",
        "K_HCDR3_mean",
        "K_HCDR3_median",
        "K_HCDR3_q90",
        "K_chi1_mean",
        "K_chi1_median",
        "K_chi1_q10",
        "K_chi1_q90",
        "K_chi1_iqr",
        "K_chi1_frac_neg",
    ]
    cols = [c for c in prefer if c in df.columns and df[c].notna().mean() >= 0.5]
    return cols[:25]


def eval_matrix(name, X_all, folds, y, sol, plm, gen_label="NA"):
    global PLM_DF, Y, FOLDS
    PLM_DF, Y, FOLDS = plm, y, folds
    ids = [i for i in folds.id if i in X_all.index and i in y.index and i in plm.index]
    X = X_all.loc[ids]
    yy = y.loc[ids]
    folds_u = folds[folds.id.isin(ids)]
    plm_u = plm.loc[ids]

    oof_s, *_ = nested_ridge(X.fillna(X.median()), yy, folds_u)
    base_oof, _ = median_baseline_oof(yy, folds_u)
    oof_plm = plm_pca_oof(plm_u, yy, folds_u)
    oof_fuse = fuse_oof(plm_u, X, yy, folds_u)
    oof_resid = residual_oof(X, yy, folds_u)

    rows = []
    for label, oof, ref, refname in [
        ("STANDALONE", oof_s, base_oof, "MEDIAN"),
        ("PLM_ONLY", oof_plm, base_oof, "MEDIAN"),
        ("PLM_PLUS", oof_fuse, oof_plm, "PLM_ONLY"),
        ("RESIDUAL_RECON", oof_resid, oof_plm, "PLM_ONLY"),
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
                "ref": refname,
                "n_features": X.shape[1],
                "n": len(ids),
            }
        )

    # Public/Private
    pub = list(sol.loc[sol.is_public == 1, "id"])
    pri = list(sol.loc[sol.is_private == 1, "id"])
    y_te = sol.set_index("id")["TmApp"].astype(float)
    te = [i for i in y_te.index if i in X_all.index and i in plm.index and i not in set(ids)]

    def fit_kind(kind):
        tr = ids
        if kind == "standalone":
            sc = StandardScaler()
            med = X.median()
            a = select_alpha_df(X.fillna(med), yy, folds_u, tr)
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(X.fillna(med)), yy)
            return m.predict(sc.transform(X_all.loc[te].fillna(med)))
        if kind == "plm":
            return plm_predict_train(tr, te)
        sc0 = StandardScaler()
        Ztr = sc0.fit_transform(plm.loc[tr])
        Zte = sc0.transform(plm.loc[te])
        pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        scp = StandardScaler()
        med = X.median()
        Xptr = scp.fit_transform(X.fillna(med))
        Xpte = scp.transform(X_all.loc[te].fillna(med))
        Xtr = np.hstack([Ptr, Xptr])
        Xte = np.hstack([Pte, Xpte])
        a = select_alpha_df(pd.DataFrame(Xtr, index=tr), yy, folds_u, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), yy)
        return m.predict(sc.transform(Xte))

    if te:
        preds = {k: fit_kind(k) for k in ("standalone", "plm", "fuse")}
        for kind, mode in [("standalone", "STANDALONE"), ("plm", "PLM_ONLY"), ("fuse", "PLM_PLUS")]:
            for split, id_list in [("Public", pub), ("Private", pri)]:
                ids_s = [i for i in id_list if i in te]
                if not ids_s:
                    continue
                idx = [te.index(i) for i in ids_s]
                pp = np.asarray(preds[kind])[idx]
                ref = np.full(len(ids_s), float(np.median(yy))) if mode == "STANDALONE" else np.asarray(preds["plm"])[idx]
                m = metrics(y_te.loc[ids_s], pp)
                d = bootstrap_delta(y_te.loc[ids_s], pp, ref, b=B_BOOT)
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
                        "n_features": X.shape[1],
                        "n": len(ids_s),
                    }
                )
    return rows


def artifact_and_robustness():
    v2 = pd.read_csv(V2 / "FENNIX_V2_CURVATURE_FEATURES.csv")
    qc = pd.read_csv(V2 / "FENNIX_V2_RELAXATION_QC.csv")
    v1 = pd.read_csv(V1 / "cache/fennix_features/features_esmfold.csv")
    # also abb2 v1
    v1b = pd.read_csv(V1 / "cache/fennix_features/features_abodybuilder2.csv")
    strain = {
        "esmfold": pd.read_parquet(FP / "OPENMM-STRAIN/features_esmfold.parquet"),
        "abodybuilder2": pd.read_parquet(FP / "OPENMM-STRAIN/features_abodybuilder2.parquet"),
    }
    rows = []
    for gen in ["esmfold", "abodybuilder2"]:
        f2 = v2[v2.generator == gen].set_index("id")
        q = qc[qc.generator == gen].set_index("id")
        st = strain[gen].set_index("id")
        v1f = (v1 if gen == "esmfold" else v1b).set_index("id")
        cols_v2 = fennix_feature_cols(f2.reset_index())
        cols_v1 = [c for c in ["P1_dE_mean", "P3_dE_mean", "ref_F_rms", "P1_Frms_mean"] if c in v1f.columns]
        proxies = {
            "severe_clash_before": q["R0_severe_clash"] if "R0_severe_clash" in q else st.get("severe_clash_count_before"),
            "severe_clash_after": q["R1_severe_clash"] if "R1_severe_clash" in q else np.nan,
            "n_atoms": q["n_atoms"] if "n_atoms" in q else v1f.get("n_atoms"),
            "initial_force_RMS_openmm": st["initial_force_RMS"],
            "v1_ref_F_rms": v1f["ref_F_rms"] if "ref_F_rms" in v1f else np.nan,
        }
        for feat in cols_v1 + cols_v2:
            src = v1f if feat in v1f.columns else f2
            if feat not in src.columns:
                continue
            for pname, pser in proxies.items():
                if pser is None:
                    continue
                ids = sorted(set(src.index) & set(pser.index if hasattr(pser, "index") else []))
                if not ids:
                    continue
                x = src.loc[ids, feat].astype(float)
                y = pd.Series(pser).loc[ids].astype(float)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 20:
                    continue
                rows.append(
                    {
                        "generator": gen,
                        "feature": feat,
                        "family": "v1" if feat in cols_v1 else "v2",
                        "proxy": pname,
                        "spearman": float(spearmanr(x[m], y[m]).statistic),
                        "pearson": float(pearsonr(x[m], y[m]).statistic),
                        "n": int(m.sum()),
                    }
                )
    art = pd.DataFrame(rows)
    art.to_csv(V2 / "FENNIX_V1_V2_ARTIFACT_COMPARISON.csv", index=False)

    # generator robustness v2
    a = v2[v2.generator == "esmfold"].set_index("id")
    b = v2[v2.generator == "abodybuilder2"].set_index("id")
    ids = sorted(set(a.index) & set(b.index))
    rob = []
    for c in fennix_feature_cols(v2):
        x, y = a.loc[ids, c].astype(float), b.loc[ids, c].astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            continue
        rob.append(
            {
                "feature": c,
                "pearson": float(pearsonr(x[m], y[m]).statistic),
                "spearman": float(spearmanr(x[m], y[m]).statistic),
                "norm_abs_diff": float(np.mean(np.abs(x[m] - y[m]) / (np.std(x[m]) + 1e-12))),
            }
        )
    rdf = pd.DataFrame(rob)
    rdf.to_csv(V2 / "FENNIX_V2_GENERATOR_ROBUSTNESS.csv", index=False)
    med = float(rdf.spearman.median()) if len(rdf) else 0
    label = "ROBUST" if med >= 0.7 else ("MODERATE" if med >= 0.4 else "FRAGILE")
    (V2 / "results/fennix_v2_robustness_class.txt").write_text(label + "\n")
    return art, rdf, label


def main():
    y = load_y()
    plm = load_plm()
    folds = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    sol = pd.read_csv(SOL)
    all_rows = []

    # FeNNix v2
    feat = pd.read_csv(V2 / "FENNIX_V2_CURVATURE_FEATURES.csv")
    feat = feat[feat.extraction_status == "SUCCESS"]
    for gen, lab in [("esmfold", "ESMFold"), ("abodybuilder2", "ABB2")]:
        g = feat[feat.generator == gen].set_index("id")
        cols = fennix_feature_cols(g.reset_index())
        X = g[cols].astype(float)
        all_rows.extend(eval_matrix("FENNIX_V2", X, folds, y, sol, plm, lab))
        # shadow CV only
        sh = eval_matrix("FENNIX_V2", X, shadow, y, sol, plm, lab + "_SHADOW")
        for r in sh:
            if r["split"] == "CV":
                r["split"] = "Shadow"
                all_rows.append(r)

    # BioEmu
    be_path = V2 / "BIOEMU_V12_FEATURES.csv"
    if be_path.exists():
        be = pd.read_csv(be_path).set_index("id")
        cols = [c for c in be.columns if c not in ("id", "extraction_status", "n_features") and be[c].notna().mean() >= 0.5]
        cols = cols[:30]
        X = be[cols].astype(float)
        all_rows.extend(eval_matrix("BIOEMU_V12", X, folds, y, sol, plm, "sequence"))
        sh = eval_matrix("BIOEMU_V12", X, shadow, y, sol, plm, "sequence_SHADOW")
        for r in sh:
            if r["split"] == "CV":
                r["split"] = "Shadow"
                all_rows.append(r)

    out = pd.DataFrame(all_rows)
    out.to_csv(V2 / "results/FOUNDATION_V2_ALL_EVAL.csv", index=False)
    out.loc[out["mode"].isin(["STANDALONE", "PLM_ONLY"])].to_csv(V2 / "FOUNDATION_V2_STANDALONE_RESULTS.csv", index=False)
    out.loc[out["mode"] == "PLM_PLUS"].to_csv(V2 / "FOUNDATION_V2_INCREMENTAL_RESULTS.csv", index=False)
    out.loc[out["mode"] == "RESIDUAL_RECON"].to_csv(V2 / "FOUNDATION_V2_RESIDUAL_RESULTS.csv", index=False)
    # family-specific
    out.loc[out.family.str.startswith("FENNIX")].to_csv(V2 / "FENNIX_V2_RESULTS.csv", index=False)
    if (out.family == "BIOEMU_V12").any():
        out.loc[out.family == "BIOEMU_V12"].to_csv(V2 / "BIOEMU_V12_RESULTS.csv", index=False)

    # bootstrap summary key rows
    out.loc[out.split == "CV"].to_csv(V2 / "FOUNDATION_V2_BOOTSTRAP.csv", index=False)

    if (V2 / "FENNIX_V2_CURVATURE_FEATURES.csv").exists() and (V2 / "FENNIX_V2_RELAXATION_QC.csv").exists():
        try:
            artifact_and_robustness()
        except Exception as e:
            print("artifact skipped", e)

    print(out.groupby(["family", "generator", "mode", "split"], observed=True)["MAE"].mean())


if __name__ == "__main__":
    main()
