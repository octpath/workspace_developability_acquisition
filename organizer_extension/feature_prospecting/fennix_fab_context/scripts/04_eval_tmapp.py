#!/usr/bin/env python3
"""TmApp scoring for Fab FeNNix families (standalone / AbLang2 / incumbent)."""
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
CTX = FP / "fennix_fab_context"
sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    B_BOOT,
    OOF_DIR,
    TMAPP_REF,
    bootstrap_delta,
    mae,
    median_baseline_oof,
    metrics,
)

EMB = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_N = 32
RNG = np.random.default_rng(42)

Y = None
PLM = None
INC = None


def load_y():
    return pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)


def load_plm():
    z = np.load(EMB, allow_pickle=True)
    return pd.DataFrame(np.asarray(z["ablang2__HL_paired"], float), index=[str(x) for x in z["ids"]])


def load_incumbent():
    df = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")
    return df["y_pred"].astype(float)


def select_alpha(X, y, folds, train_ids):
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


def nested_ridge(X, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        med = X.loc[tr].median()
        Xtr = X.loc[tr].fillna(med)
        Xte = X.loc[te].fillna(med)
        a = select_alpha(Xtr, y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Xte))
    return oof


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
        a = select_alpha(pd.DataFrame(Ptr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Ptr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Pte))
    return oof


def fuse_plm_phys(X_plm, X_phys, y, folds):
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
        a = select_alpha(pd.DataFrame(Xtr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Xte))
    return oof


def stack_incumbent(inc_oof, phys_oof, y, folds):
    """Leakage-safe linear stack of frozen incumbent OOF + structural OOF."""
    X = pd.DataFrame({"inc": inc_oof, "phys": phys_oof}).dropna()
    common = X.index.intersection(y.index)
    return nested_ridge(X.loc[common], y.loc[common], folds[folds.id.isin(common)])


def residual_on(base_oof, X_phys, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_phys.index)
    cand = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        resid_tr = y.loc[tr] - base_oof.loc[tr]
        med = X_phys.loc[tr].median()
        X = X_phys.loc[tr].fillna(med)
        a = select_alpha(X, resid_tr, folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(X), resid_tr)
        pred_r = m.predict(sc.transform(X_phys.loc[te].fillna(med)))
        cand.loc[te] = base_oof.loc[te] + pred_r
    return cand


def boot_mae_delta(y, base, cand, B=B_BOOT):
    y, base, cand = y.align(base, join="inner")[0], y.align(base, join="inner")[1], y.align(cand, join="inner")[1]
    # align cand
    idx = y.index.intersection(base.index).intersection(cand.index)
    y, base, cand = y.loc[idx], base.loc[idx], cand.loc[idx]
    d0 = mae(y, base) - mae(y, cand)  # positive => cand better
    boots = []
    n = len(y)
    for _ in range(B):
        ix = RNG.integers(0, n, n)
        boots.append(mae(y.iloc[ix], base.iloc[ix]) - mae(y.iloc[ix], cand.iloc[ix]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(d0), float(lo), float(hi)


def family_X(name: str) -> pd.DataFrame:
    paths = {
        "DELTA_GEOM": CTX / "FENNIX_FAB_DELTA_GEOM_FEATURES.csv",
        "DELTA_ENV": CTX / "FENNIX_FAB_DELTA_ENV_FEATURES.csv",
        "CONSTANT": CTX / "FENNIX_FAB_CONSTANT_FEATURES.csv",
        "INTERFACE": CTX / "FENNIX_FAB_INTERFACE_FEATURES.csv",
        "NORMALIZED": CTX / "FENNIX_FAB_NORMALIZED_FEATURES.csv",
        "VARIABLE_C": CTX / "FENNIX_FAB_MATCHED_VARIABLE_FEATURES.csv",
    }
    df = pd.read_csv(paths[name]).set_index("id")
    if name == "VARIABLE_C":
        cols = [c for c in df.columns if c.startswith("C__") and not c.endswith("_n")]
        # cap dims
        cols = cols[:15]
        return df[cols]
    cols = [c for c in df.columns if c != "normalized_note" and pd.api.types.is_numeric_dtype(df[c])]
    # drop n-count columns
    cols = [c for c in cols if not c.endswith("_n") and not c.endswith("_n_sites")][:15]
    return df[cols]


def eval_cv(name, X, folds, tag):
    common = X.index.intersection(Y.index)
    X = X.loc[common]
    y = Y.loc[common]
    folds = folds[folds.id.isin(common)]
    med = median_baseline_oof(y, folds)
    phys = nested_ridge(X, y, folds)
    plm = plm_pca_oof(PLM.loc[common], y, folds)
    fuse = fuse_plm_phys(PLM.loc[common], X, y, folds)
    inc = INC.loc[common]
    stack = stack_incumbent(inc, phys, y, folds)
    resid_plm = residual_on(plm, X, y, folds)
    resid_inc = residual_on(inc, X, y, folds)

    rows = []
    boots = []
    for mode, base, cand in [
        ("STANDALONE_vs_median", med, phys),
        ("INCREMENTAL_ABLANG2", plm, fuse),
        ("INCREMENTAL_INCUMBENT_STACK", inc, stack),
        ("RESIDUAL_ABLANG2", plm, resid_plm),
        ("RESIDUAL_INCUMBENT", inc, resid_inc),
    ]:
        m_base = metrics(y, base)
        m_cand = metrics(y, cand)
        d, lo, hi = boot_mae_delta(y, base, cand)
        rows.append(
            {
                "family": name,
                "cv": tag,
                "mode": mode,
                "base_MAE": m_base["MAE"],
                "cand_MAE": m_cand["MAE"],
                "delta_MAE_base_minus_cand": d,
                "boot_lo": lo,
                "boot_hi": hi,
                "cand_Spearman": m_cand["Spearman"],
                "n": len(y),
            }
        )
        boots.append(
            {
                "family": name,
                "cv": tag,
                "mode": mode,
                "delta_MAE": d,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )
    return rows, boots


def main():
    global Y, PLM, INC
    Y = load_y()
    PLM = load_plm()
    INC = load_incumbent()
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)

    stand, incr_a, incr_i, resid, boots = [], [], [], [], []
    for fam in ["DELTA_GEOM", "DELTA_ENV", "CONSTANT", "INTERFACE", "NORMALIZED", "VARIABLE_C"]:
        path_map = {
            "DELTA_GEOM": CTX / "FENNIX_FAB_DELTA_GEOM_FEATURES.csv",
            "DELTA_ENV": CTX / "FENNIX_FAB_DELTA_ENV_FEATURES.csv",
            "CONSTANT": CTX / "FENNIX_FAB_CONSTANT_FEATURES.csv",
            "INTERFACE": CTX / "FENNIX_FAB_INTERFACE_FEATURES.csv",
            "NORMALIZED": CTX / "FENNIX_FAB_NORMALIZED_FEATURES.csv",
            "VARIABLE_C": CTX / "FENNIX_FAB_MATCHED_VARIABLE_FEATURES.csv",
        }
        if not path_map[fam].exists():
            print("skip missing", fam)
            continue
        X = family_X(fam)
        if X.shape[1] == 0 or X.shape[0] < 20:
            print("skip small", fam, X.shape)
            continue
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            rows, b = eval_cv(fam, X, folds, tag)
            for r in rows:
                if r["mode"] == "STANDALONE_vs_median":
                    stand.append(r)
                elif r["mode"] == "INCREMENTAL_ABLANG2":
                    incr_a.append(r)
                elif r["mode"] == "INCREMENTAL_INCUMBENT_STACK":
                    incr_i.append(r)
                else:
                    resid.append(r)
            boots.extend(b)
        print("scored", fam, flush=True)

    pd.DataFrame(stand).to_csv(CTX / "FENNIX_FAB_STANDALONE_RESULTS.csv", index=False)
    pd.DataFrame(incr_a).to_csv(CTX / "FENNIX_FAB_INCREMENTAL_ABLANG2.csv", index=False)
    pd.DataFrame(incr_i).to_csv(CTX / "FENNIX_FAB_INCREMENTAL_INCUMBENT.csv", index=False)
    pd.DataFrame(resid).to_csv(CTX / "FENNIX_FAB_RESIDUAL_RESULTS.csv", index=False)
    pd.DataFrame(boots).to_csv(CTX / "FENNIX_FAB_BOOTSTRAP.csv", index=False)
    print("DONE scoring", flush=True)


if __name__ == "__main__":
    main()
