#!/usr/bin/env python3
"""TmApp evaluation for BioEmu constant-context feature families."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "bioemu_constant_context"
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


def eval_matrix(name, X_all, folds, y, sol, plm, gen_label="NA", ref_oof=None, refname="PLM_ONLY"):
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
    for label, oof, ref, rn in [
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
                "ref": rn,
                "n_features": X.shape[1],
                "n": len(ids),
            }
        )

    if ref_oof is not None:
        # incremental vs old BioEmu: fuse old+new already in X; also compare standalone of X vs ref
        # PLM_PLUS already vs PLM; add OLD_BIOEMU_PLUS by treating ref_oof as baseline for a fused model
        # Here: evaluate concat(old,new) already passed as X when caller builds it; extra mode vs old-only oof
        oof_vs_old = oof_s  # if X is context-only, compare context standalone to old bioemu oof
        # Better: nested ridge on X when X includes both; caller handles.
        pass

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


def numeric_cols(df, prefixes=None, max_n=15):
    cols = [c for c in df.columns if c != "id" and pd.api.types.is_numeric_dtype(df[c])]
    if prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    cols = [c for c in cols if df[c].notna().mean() >= 0.5]
    return cols[:max_n]


def main():
    y = load_y()
    plm = load_plm()
    folds = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    sol = pd.read_csv(SOL)
    old = pd.read_csv(V2 / "BIOEMU_V12_FEATURES.csv").set_index("id")
    old_cols = [c for c in old.columns if c not in ("n_features",) and old[c].notna().mean() >= 0.5][:30]

    families = {}
    L1 = CTX / "BIOEMU_FULL_LIGHT_FEATURES.csv"
    L3 = CTX / "BIOEMU_LIGHT_CONTEXT_DELTA_FEATURES.csv"
    L4 = CTX / "BIOEMU_LIGHT_VC_COUPLING_FEATURES.csv"
    L5 = CTX / "BIOEMU_LIGHT_NATIVE_LIKENESS_FEATURES.csv"
    H = CTX / "BIOEMU_UNPAIRED_HEAVY_FEATURES.csv"

    if L1.exists():
        d = pd.read_csv(L1).set_index("id")
        families["L1_FULL_LIGHT"] = d[numeric_cols(d.reset_index(), ["L1_"], 15)]
        families["L2_VL_IN_CONTEXT"] = d[numeric_cols(d.reset_index(), ["L2_"], 15)]
    if L3.exists():
        d = pd.read_csv(L3).set_index("id")
        families["L3_CONTEXT_DELTA"] = d[numeric_cols(d.reset_index(), ["L3_"], 15)]
    if L4.exists():
        d = pd.read_csv(L4).set_index("id")
        families["L4_VL_CL_COUPLING"] = d[numeric_cols(d.reset_index(), ["L4_"], 15)]
    if L5.exists():
        d = pd.read_csv(L5).set_index("id")
        families["L5_ESMFOLD_NATIVE_LIKENESS"] = d[numeric_cols(d.reset_index(), ["L5_"], 15)]

    # combined LIGHT <= 30
    light_parts = []
    for k in ("L1_FULL_LIGHT", "L2_VL_IN_CONTEXT", "L3_CONTEXT_DELTA", "L4_VL_CL_COUPLING", "L5_ESMFOLD_NATIVE_LIKENESS"):
        if k in families:
            light_parts.append(families[k])
    if light_parts:
        comb = pd.concat(light_parts, axis=1)
        comb = comb.loc[:, ~comb.columns.duplicated()]
        # cap 30 cols by coverage
        cols = [c for c in comb.columns if comb[c].notna().mean() >= 0.5][:30]
        families["LIGHT_COMBINED"] = comb[cols]

    if H.exists():
        d = pd.read_csv(H).set_index("id")
        families["H1_UNPAIRED_VH_CONTEXT_DELTA"] = d[numeric_cols(d.reset_index(), ["H1_"], 15)]
        families["H2_UNPAIRED_CH1_NATIVE_LIKENESS"] = d[numeric_cols(d.reset_index(), ["H2_"], 15)]
        families["H3_UNPAIRED_VH_CH1_COUPLING"] = d[numeric_cols(d.reset_index(), ["H3_"], 15)]
        families["HEAVY_DIAGNOSTIC"] = d[numeric_cols(d.reset_index(), ["H1_", "H2_", "H3_"], 30)]

    # old BioEmu + LIGHT
    if "LIGHT_COMBINED" in families:
        both = pd.concat([old[old_cols], families["LIGHT_COMBINED"]], axis=1, join="inner")
        both = both.loc[:, ~both.columns.duplicated()]
        families["OLD_BIOEMU_PLUS_LIGHT"] = both.iloc[:, :60]
        families["OLD_BIOEMU_ONLY"] = old[old_cols]

    all_rows = []
    for name, X in families.items():
        print("eval", name, X.shape, flush=True)
        all_rows.extend(eval_matrix(name, X, folds, y, sol, plm, "context"))
        sh = eval_matrix(name, X, shadow, y, sol, plm, "context_SHADOW")
        for r in sh:
            if r["split"] == "CV":
                r["split"] = "Shadow"
                all_rows.append(r)

    # incremental vs old BioEmu: compare OLD_BIOEMU_PLUS_LIGHT PLM_PLUS? Better:
    # STANDALONE of OLD vs OLD+LIGHT via bootstrap of nested ridge
    if "OLD_BIOEMU_ONLY" in families and "OLD_BIOEMU_PLUS_LIGHT" in families:
        Xold = families["OLD_BIOEMU_ONLY"]
        Xnew = families["OLD_BIOEMU_PLUS_LIGHT"]
        ids = [i for i in folds.id if i in Xold.index and i in Xnew.index and i in y.index]
        oof_old, *_ = nested_ridge(Xold.loc[ids].fillna(Xold.loc[ids].median()), y.loc[ids], folds[folds.id.isin(ids)])
        oof_new, *_ = nested_ridge(Xnew.loc[ids].fillna(Xnew.loc[ids].median()), y.loc[ids], folds[folds.id.isin(ids)])
        d = bootstrap_delta(y.loc[ids], oof_new, oof_old, b=B_BOOT)
        m = metrics(y.loc[ids], oof_new)
        all_rows.append(
            {
                "family": "OLD_BIOEMU_PLUS_LIGHT_VS_OLD",
                "generator": "context",
                "mode": "INCREMENTAL_OLD_BIOEMU",
                "split": "CV",
                "MAE": m["MAE"],
                "Pearson": m["Pearson"],
                "Spearman": m["Spearman"],
                "delta_vs_ref": d["delta"],
                "boot_lo": d["ci95_low"],
                "boot_hi": d["ci95_high"],
                "p_improve": d["p_improve"],
                "ref": "OLD_BIOEMU_ONLY",
                "n_features": Xnew.shape[1],
                "n": len(ids),
            }
        )

    out = pd.DataFrame(all_rows)
    out.to_csv(CTX / "results/BIOEMU_CONTEXT_ALL_EVAL.csv", index=False)
    out.loc[out.mode.isin(["STANDALONE", "PLM_ONLY"])].to_csv(CTX / "BIOEMU_CONTEXT_STANDALONE_RESULTS.csv", index=False)
    out.loc[out.mode.isin(["PLM_PLUS", "INCREMENTAL_OLD_BIOEMU"])].to_csv(CTX / "BIOEMU_CONTEXT_INCREMENTAL_RESULTS.csv", index=False)
    out.loc[out.mode == "RESIDUAL_RECON"].to_csv(CTX / "BIOEMU_CONTEXT_RESIDUAL_RESULTS.csv", index=False)
    out.loc[out.split == "CV"].to_csv(CTX / "BIOEMU_CONTEXT_BOOTSTRAP.csv", index=False)
    print(out.groupby(["family", "mode", "split"], observed=True)["MAE"].mean())


if __name__ == "__main__":
    main()
