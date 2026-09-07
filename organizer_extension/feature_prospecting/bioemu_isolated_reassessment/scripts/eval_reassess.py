#!/usr/bin/env python3
"""TmApp eval for BioEmu isolated reassessment + OLD vs NEW comparison."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
R = FP / "bioemu_isolated_reassessment"
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
PLM_DF = Y = FOLDS = None


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
        Ztr, Zte = sc0.fit_transform(X_plm.loc[tr]), sc0.transform(X_plm.loc[te])
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
    Ztr, Zte = sc0.fit_transform(PLM_DF.loc[tr]), sc0.transform(PLM_DF.loc[te])
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
        Ztr, Zte = sc0.fit_transform(X_plm.loc[tr]), sc0.transform(X_plm.loc[te])
        pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        scp = StandardScaler()
        med = X_phys.loc[tr].median()
        Xtr = np.hstack([Ptr, scp.fit_transform(X_phys.loc[tr].fillna(med))])
        Xte = np.hstack([Pte, scp.transform(X_phys.loc[te].fillna(med))])
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
        cand.loc[te] = plm_te + m.predict(sc.transform(X_phys.loc[te].fillna(med)))
    return cand


def eval_matrix(name, X_all, folds, y, sol, plm, label="reassess"):
    global PLM_DF, Y, FOLDS
    PLM_DF, Y, FOLDS = plm, y, folds
    ids = [i for i in folds.id if i in X_all.index and i in y.index and i in plm.index]
    X, yy = X_all.loc[ids], y.loc[ids]
    folds_u, plm_u = folds[folds.id.isin(ids)], plm.loc[ids]
    oof_s, *_ = nested_ridge(X.fillna(X.median()), yy, folds_u)
    base_oof, _ = median_baseline_oof(yy, folds_u)
    oof_plm = plm_pca_oof(plm_u, yy, folds_u)
    oof_fuse = fuse_oof(plm_u, X, yy, folds_u)
    oof_resid = residual_oof(X, yy, folds_u)
    rows = []
    for mode, oof, ref, rn in [
        ("STANDALONE", oof_s, base_oof, "MEDIAN"),
        ("PLM_ONLY", oof_plm, base_oof, "MEDIAN"),
        ("PLM_PLUS", oof_fuse, oof_plm, "PLM_ONLY"),
        ("RESIDUAL_RECON", oof_resid, oof_plm, "PLM_ONLY"),
    ]:
        m = metrics(yy, oof)
        d = bootstrap_delta(yy, oof, ref, b=B_BOOT)
        rows.append(
            dict(
                family=name,
                generator=label,
                mode=mode,
                split="CV",
                MAE=m["MAE"],
                Pearson=m["Pearson"],
                Spearman=m["Spearman"],
                delta_vs_ref=d["delta"],
                boot_lo=d["ci95_low"],
                boot_hi=d["ci95_high"],
                p_improve=d["p_improve"],
                ref=rn,
                n_features=X.shape[1],
                n=len(ids),
            )
        )
    # Public/Private
    pub = list(sol.loc[sol.is_public == 1, "id"])
    pri = list(sol.loc[sol.is_private == 1, "id"])
    y_te = sol.set_index("id")["TmApp"].astype(float)
    te = [i for i in y_te.index if i in X_all.index and i in plm.index and i not in set(ids)]

    def fit_kind(kind):
        tr = ids
        if kind == "standalone":
            sc, med = StandardScaler(), X.median()
            a = select_alpha_df(X.fillna(med), yy, folds_u, tr)
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(X.fillna(med)), yy)
            return m.predict(sc.transform(X_all.loc[te].fillna(med)))
        if kind == "plm":
            return plm_predict_train(tr, te)
        sc0 = StandardScaler()
        Ztr, Zte = sc0.fit_transform(plm.loc[tr]), sc0.transform(plm.loc[te])
        pca = PCA(n_components=min(PCA_N, len(tr) - 1), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        scp, med = StandardScaler(), X.median()
        Xtr = np.hstack([Ptr, scp.fit_transform(X.fillna(med))])
        Xte = np.hstack([Pte, scp.transform(X_all.loc[te].fillna(med))])
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
                    dict(
                        family=name,
                        generator=label,
                        mode=mode,
                        split=split,
                        MAE=m["MAE"],
                        Pearson=m["Pearson"],
                        Spearman=m["Spearman"],
                        delta_vs_ref=d["delta"],
                        boot_lo=d["ci95_low"],
                        boot_hi=d["ci95_high"],
                        p_improve=d["p_improve"],
                        ref="MEDIAN" if mode == "STANDALONE" else "PLM_ONLY",
                        n_features=X.shape[1],
                        n=len(ids_s),
                    )
                )
    return rows, {"standalone": oof_s, "plm": oof_plm, "fuse": oof_fuse}


def numeric_family(df, prefixes, max_n=30):
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes) and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() >= 0.5][:max_n]
    return cols


def main():
    y, plm = load_y(), load_plm()
    folds, shadow, sol = pd.read_csv(PRIMARY), pd.read_csv(SHADOW), pd.read_csv(SOL)
    new = pd.read_csv(R / "BIOEMU_ISOLATED_REASSESS_FEATURES.csv").set_index("id")
    old = pd.read_csv(V2 / "BIOEMU_V12_FEATURES.csv").set_index("id")

    families = {
        "NEW_COMBINED": new[numeric_family(new.reset_index(), ["mean_", "max_", "absdiff_", "VH_", "VL_"], 30)],
        "NEW_CONTACT": new[numeric_family(new.reset_index(), ["mean_contact", "VH_contact", "VL_contact", "max_contact", "absdiff_contact"], 15)],
        "NEW_FLEX": new[numeric_family(new.reset_index(), ["mean_ca_rmsf", "mean_rmsf", "VH_ca_rmsf", "VL_ca_rmsf", "VH_rmsf", "VL_rmsf"], 15)],
        "NEW_PAIRWISE": new[numeric_family(new.reset_index(), ["mean_ca_rmsd", "VH_ca_rmsd", "VL_ca_rmsd"], 10)],
        "NEW_SHAPE": new[numeric_family(new.reset_index(), ["mean_rg", "VH_rg", "VL_rg"], 10)],
        "OLD_BIOEMU": old[[c for c in old.columns if c != "n_features" and old[c].notna().mean() >= 0.5][:30]],
    }

    all_rows = []
    oofs = {}
    for name, X in families.items():
        if X.shape[1] == 0:
            continue
        print("eval", name, X.shape, flush=True)
        rows, oo = eval_matrix(name, X, folds, y, sol, plm)
        all_rows.extend(rows)
        oofs[name] = oo
        for r in eval_matrix(name, X, shadow, y, sol, plm, "reassess_SHADOW")[0]:
            if r["split"] == "CV":
                r["split"] = "Shadow"
                all_rows.append(r)

    # OLD vs NEW standalone bootstrap
    ids = [i for i in folds.id if i in families["OLD_BIOEMU"].index and i in families["NEW_COMBINED"].index and i in y.index]
    oof_old, *_ = nested_ridge(families["OLD_BIOEMU"].loc[ids].fillna(families["OLD_BIOEMU"].loc[ids].median()), y.loc[ids], folds[folds.id.isin(ids)])
    oof_new, *_ = nested_ridge(families["NEW_COMBINED"].loc[ids].fillna(families["NEW_COMBINED"].loc[ids].median()), y.loc[ids], folds[folds.id.isin(ids)])
    d = bootstrap_delta(y.loc[ids], oof_new, oof_old, b=B_BOOT)
    m = metrics(y.loc[ids], oof_new)
    all_rows.append(
        dict(
            family="NEW_VS_OLD",
            generator="reassess",
            mode="INCREMENTAL_OLD_BIOEMU",
            split="CV",
            MAE=m["MAE"],
            Pearson=m["Pearson"],
            Spearman=m["Spearman"],
            delta_vs_ref=d["delta"],
            boot_lo=d["ci95_low"],
            boot_hi=d["ci95_high"],
            p_improve=d["p_improve"],
            ref="OLD_BIOEMU",
            n_features=families["NEW_COMBINED"].shape[1],
            n=len(ids),
        )
    )

    # feature correlation old vs new overlapping names
    ov = []
    for c in families["NEW_COMBINED"].columns:
        # map mean_X to mean_X style in old
        if c in old.columns:
            a, b = new[c], old[c]
            msk = a.notna() & b.notna()
            if msk.sum() > 20:
                ov.append(dict(feature=c, spearman=float(spearmanr(a[msk], b[msk]).statistic), pearson=float(pearsonr(a[msk], b[msk]).statistic), n=int(msk.sum())))
    pd.DataFrame(ov).to_csv(R / "BIOEMU_ISOLATED_OLD_VS_NEW.csv", index=False)

    out = pd.DataFrame(all_rows)
    out.to_csv(R / "results/ALL_EVAL.csv", index=False)
    out.loc[out["mode"].isin(["STANDALONE", "PLM_ONLY"])].to_csv(R / "BIOEMU_ISOLATED_REASSESS_STANDALONE.csv", index=False)
    out.loc[out["mode"].isin(["PLM_PLUS", "INCREMENTAL_OLD_BIOEMU"])].to_csv(R / "BIOEMU_ISOLATED_REASSESS_INCREMENTAL.csv", index=False)
    out.loc[out["mode"] == "RESIDUAL_RECON"].to_csv(R / "BIOEMU_ISOLATED_REASSESS_RESIDUAL.csv", index=False)
    out.loc[out["split"] == "CV"].to_csv(R / "BIOEMU_ISOLATED_REASSESS_BOOTSTRAP.csv", index=False)
    print(out.groupby(["family", "mode", "split"], observed=True)["MAE"].mean())


if __name__ == "__main__":
    main()
