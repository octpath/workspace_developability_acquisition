#!/usr/bin/env python3
"""Fusion Closure Gates 3A–3C: PLM↔physics audit + early/orthogonal fusion."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "fusion_closure"
RES = OUT / "results"
sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    B_BOOT,
    HIC_REF,
    OOF_DIR,
    TMAPP_REF,
    bootstrap_delta,
    mae,
    metrics,
)

EMB_PATH = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
FOLDS_PATH = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SOL_PATH = ROOT / "competition/data/secret/solution.csv"

GEN_MAP = {"esmfold": "ESMFold", "abodybuilder2": "ABB2", "boltz2": "Boltz2"}
GEN_FILES = {
    "esmfold": "features_esmfold.parquet",
    "abodybuilder2": "features_abodybuilder2.parquet",
    "boltz2": "features_boltz2.parquet",
}

FAMILIES = {
    "POLAR-SAT": {
        "target": "TmApp",
        "dir": FP / "POLAR-SAT",
        "role": "primary",
        "semantic": [
            "buried_unsatisfied_polar_count",
            "fraction_buried_polar_unsatisfied",
            "CDR_buried_unsat_count",
            "interface_buried_unsat_count",
        ],
    },
    "PKA-SHIFT_CORRECTED": {
        "target": "TmApp",
        "dir": FP / "PKA-SHIFT/TECHNICAL_CORRECTION",
        "role": "primary",
        "spec": FP / "PKA-SHIFT/FEATURE_SPEC.json",
        "semantic": [
            "mean_abs_delta_pKa",
            "cdr_mean_abs_delta",
            "interface_mean_abs_delta",
            "buried_mean_abs_delta",
            "buried_frac_abs_ge_1",
        ],
    },
    "VOID-EXPLICIT": {
        "target": "TmApp",
        "dir": FP / "VOID-EXPLICIT",
        "role": "secondary_control",
        "semantic": [],
    },
    "AROMATIC-TOPO": {
        "target": "HIC",
        "dir": FP / "AROMATIC-TOPO",
        "role": "primary",
        "semantic": [
            "aromatic_exposed_SASA_total",
            "CDR_aromatic_SASA",
            "largest_aromatic_patch_exposed_SASA",
            "exposed_aromatic_total_count",
        ],
    },
    "STATIC-SAP": {
        "target": "HIC",
        "dir": FP / "STATIC-SAP",
        "role": "negative_control",
        "semantic": [],
    },
    "HYDRO-FIELD": {
        "target": "HIC",
        "dir": FP / "HYDRO-FIELD",
        "role": "negative_control",
        "semantic": [],
    },
}

STANDALONE = {
    "POLAR-SAT": "PROMISING_BUT_REDUNDANT/FRAGILE",
    "PKA-SHIFT_CORRECTED": "MIXED (ESMFold-leaning)",
    "VOID-EXPLICIT": "MIXED/FRAGILE",
    "AROMATIC-TOPO": "PROMISING_BUT_REDUNDANT",
    "STATIC-SAP": "NO_EVIDENCE",
    "HYDRO-FIELD": "NO_EVIDENCE",
}

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
SVR_GRID = list(
    ParameterGrid(
        {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.05, 0.1, 0.2]}
    )
)
PCA_N = 32
SEED = 42
RNG = np.random.default_rng(42)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_canonical(family: str) -> list[str]:
    meta = FAMILIES[family]
    spec_path = meta.get("spec", meta["dir"] / "FEATURE_SPEC.json")
    return list(json.loads(spec_path.read_text())["canonical_features"])


def load_physics(family: str, gen_key: str) -> pd.DataFrame:
    path = FAMILIES[family]["dir"] / GEN_FILES[gen_key]
    cols = load_canonical(family)
    df = pd.read_parquet(path)
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.upper().isin(["SUCCESS", "OK"])].copy()
    out = df.drop_duplicates("id", keep="first").set_index("id")
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].astype(float)


def load_embeddings():
    z = np.load(EMB_PATH, allow_pickle=True)
    return {
        "ids": [str(x) for x in z["ids"]],
        "ablang2__HL_paired": np.asarray(z["ablang2__HL_paired"], dtype=np.float64),
        "esm2__H": np.asarray(z["esm2__H"], dtype=np.float64),
    }


def emb_frame(emb, key: str) -> pd.DataFrame:
    return pd.DataFrame(emb[key], index=emb["ids"])


def predictability_label(median_r2: float) -> str:
    if median_r2 >= 0.50:
        return "HIGHLY_PLM_PREDICTABLE"
    if median_r2 >= 0.20:
        return "PARTLY_PLM_PREDICTABLE"
    if median_r2 > 0:
        return "WEAKLY_PLM_PREDICTABLE"
    return "NOT_USEFULLY_PLM_PREDICTABLE"


def fusion_class(d_cv, d_pub, d_pri, boots):
    improves = [d_cv < 0, d_pub < 0, d_pri < 0]
    n_imp = sum(improves)
    if n_imp == 3:
        return "REPRODUCIBLE_FUSION_GAIN"
    if n_imp == 2:
        fail = "CV" if d_cv >= 0 else ("Public" if d_pub >= 0 else "Private")
        b = boots[fail]
        lo, hi = b["ci95_low"], b["ci95_high"]
        if lo <= 0 <= hi:
            return "WEAK_OR_MIXED_FUSION"
        if lo > 0:
            return "DIRECTION_REVERSAL"
        return "WEAK_OR_MIXED_FUSION"
    return "NO_FUSION_GAIN"


def generator_consistency(class_by_gen: dict) -> str:
    gains = [c == "REPRODUCIBLE_FUSION_GAIN" for c in class_by_gen.values()]
    n = sum(gains)
    if n == 3:
        return "FUSION_GAIN_ALL3"
    if n == 2:
        return "FUSION_GAIN_2OF3"
    if n == 1:
        return "GENERATOR_SPECIFIC_FUSION"
    return "NO_FUSION_GAIN"


def fill_nan_train_median(Xtr: np.ndarray, Xte: np.ndarray):
    Xtr = np.asarray(Xtr, dtype=np.float64).copy()
    Xte = np.asarray(Xte, dtype=np.float64).copy()
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte


def plm_pca(X_plm_tr, X_plm_te, n_comp=PCA_N):
    sc = StandardScaler()
    Ztr = sc.fit_transform(X_plm_tr)
    Zte = sc.transform(X_plm_te)
    n = min(n_comp, Ztr.shape[0] - 1, Ztr.shape[1])
    pca = PCA(n_components=n, random_state=SEED)
    return pca.fit_transform(Ztr), pca.transform(Zte)


def physics_scale(Xp_tr, Xp_te):
    Xp_tr, Xp_te = fill_nan_train_median(Xp_tr, Xp_te)
    sc = StandardScaler()
    return sc.fit_transform(Xp_tr), sc.transform(Xp_te)


def select_ridge_alpha(Xtr, ytr, groups):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    best_a, best_mae = RIDGE_ALPHAS[0], np.inf
    for a in RIDGE_ALPHAS:
        maes = []
        for tr, va in gkf.split(Xtr, ytr, groups):
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(Xtr[tr]), ytr[tr])
            maes.append(mae(ytr[va], m.predict(sc.transform(Xtr[va]))))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best_mae - 1e-15 or (abs(score - best_mae) <= 1e-15 and a > best_a):
            best_mae, best_a = score, a
    return best_a


def select_svr_params(Xtr, ytr, groups):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    best_p, best_mae = SVR_GRID[0], np.inf
    for p in SVR_GRID:
        maes = []
        for tr, va in gkf.split(Xtr, ytr, groups):
            sc = StandardScaler()
            m = SVR(kernel="rbf", **p)
            m.fit(sc.fit_transform(Xtr[tr]), ytr[tr])
            maes.append(mae(ytr[va], m.predict(sc.transform(Xtr[va]))))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best_mae:
            best_mae, best_p = score, p
    return best_p


def residualize_physics(X_plm_tr, X_plm_te, Xp_tr, Xp_te):
    Ptr, Pte = plm_pca(X_plm_tr, X_plm_te)
    Xp_tr_s, Xp_te_s = physics_scale(Xp_tr, Xp_te)
    kf = KFold(n_splits=min(5, max(2, len(Ptr) // 20)), shuffle=True, random_state=SEED)
    best_a, best_mse = 1.0, np.inf
    for a in RIDGE_ALPHAS:
        mses = []
        for tr, va in kf.split(Ptr):
            scy = StandardScaler()
            Y1 = scy.fit_transform(Xp_tr_s[tr])
            Y2 = scy.transform(Xp_tr_s[va])
            m = Ridge(alpha=a, random_state=0)
            m.fit(Ptr[tr], Y1)
            mses.append(np.mean((Y2 - m.predict(Ptr[va])) ** 2))
        mse = float(np.mean(mses))
        if mse < best_mse:
            best_mse, best_a = mse, a
    scy = StandardScaler()
    Ytr = scy.fit_transform(Xp_tr_s)
    m = Ridge(alpha=best_a, random_state=0)
    m.fit(Ptr, Ytr)
    hat_tr = scy.inverse_transform(m.predict(Ptr))
    hat_te = scy.inverse_transform(m.predict(Pte))
    return Xp_tr_s - hat_tr, Xp_te_s - hat_te


def build_design(X_plm_tr, X_plm_te, Xp_tr, Xp_te, mode: str):
    Ptr, Pte = plm_pca(X_plm_tr, X_plm_te)
    if mode == "plm_only":
        return Ptr, Pte
    if mode == "plus":
        Sp_tr, Sp_te = physics_scale(Xp_tr, Xp_te)
        return np.hstack([Ptr, Sp_tr]), np.hstack([Pte, Sp_te])
    Rtr, Rte = residualize_physics(X_plm_tr, X_plm_te, Xp_tr, Xp_te)
    return np.hstack([Ptr, Rtr]), np.hstack([Pte, Rte])


def fit_predict(Xtr, Xte, ytr, groups_tr, model_kind: str):
    if model_kind == "ridge":
        a = select_ridge_alpha(Xtr, ytr, groups_tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    p = select_svr_params(Xtr, ytr, groups_tr)
    sc = StandardScaler()
    m = SVR(kernel="rbf", **p)
    m.fit(sc.fit_transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def nested_oof(X_plm, Xp, y, groups, mode: str, model_kind: str):
    n = len(y)
    oof = np.full(n, np.nan)
    for fid in sorted(set(groups.tolist())):
        te = np.where(groups == fid)[0]
        tr = np.where(groups != fid)[0]
        Xtr, Xte = build_design(X_plm[tr], X_plm[te], Xp[tr] if Xp is not None else None, Xp[te] if Xp is not None else None, mode)
        # plm_only: Xp unused — pass dummy
        if mode == "plm_only":
            Xtr, Xte = build_design(X_plm[tr], X_plm[te], np.zeros((len(tr), 1)), np.zeros((len(te), 1)), mode)
        oof[te] = fit_predict(Xtr, Xte, y[tr], groups[tr], model_kind)
    return oof


def nested_oof_fixed(X_plm, Xp, y, groups, mode, model_kind):
    """Correct nested OOF with proper Xp handling."""
    n = len(y)
    oof = np.full(n, np.nan)
    for fid in sorted(set(groups.tolist())):
        te = np.where(groups == fid)[0]
        tr = np.where(groups != fid)[0]
        if mode == "plm_only":
            Xtr, Xte = plm_pca(X_plm[tr], X_plm[te])
        elif mode == "plus":
            Ptr, Pte = plm_pca(X_plm[tr], X_plm[te])
            Sp_tr, Sp_te = physics_scale(Xp[tr], Xp[te])
            Xtr, Xte = np.hstack([Ptr, Sp_tr]), np.hstack([Pte, Sp_te])
        else:
            Ptr, Pte = plm_pca(X_plm[tr], X_plm[te])
            Rtr, Rte = residualize_physics(X_plm[tr], X_plm[te], Xp[tr], Xp[te])
            Xtr, Xte = np.hstack([Ptr, Rtr]), np.hstack([Pte, Rte])
        oof[te] = fit_predict(Xtr, Xte, y[tr], groups[tr], model_kind)
    return oof


def predict_test(X_plm_tr, X_plm_te, Xp_tr, Xp_te, y_tr, groups_tr, mode, model_kind):
    if mode == "plm_only":
        Xtr, Xte = plm_pca(X_plm_tr, X_plm_te)
    elif mode == "plus":
        Ptr, Pte = plm_pca(X_plm_tr, X_plm_te)
        Sp_tr, Sp_te = physics_scale(Xp_tr, Xp_te)
        Xtr, Xte = np.hstack([Ptr, Sp_tr]), np.hstack([Pte, Sp_te])
    else:
        Ptr, Pte = plm_pca(X_plm_tr, X_plm_te)
        Rtr, Rte = residualize_physics(X_plm_tr, X_plm_te, Xp_tr, Xp_te)
        Xtr, Xte = np.hstack([Ptr, Rtr]), np.hstack([Pte, Rte])
    return fit_predict(Xtr, Xte, y_tr, groups_tr, model_kind)


def gate3a(X_plm, Xp, groups, feat_names):
    """Predict physics from PLM; metrics on original physics scale."""
    n, p = Xp.shape
    oof = np.full((n, p), np.nan)
    for fid in sorted(set(groups.tolist())):
        te = np.where(groups == fid)[0]
        tr = np.where(groups != fid)[0]
        Ptr, Pte = plm_pca(X_plm[tr], X_plm[te])
        Xp_tr, Xp_te = fill_nan_train_median(Xp[tr], Xp[te])
        kf = KFold(n_splits=min(5, max(2, len(tr) // 20)), shuffle=True, random_state=SEED)
        best_a, best_mse = 1.0, np.inf
        for a in RIDGE_ALPHAS:
            mses = []
            for i1, i2 in kf.split(Ptr):
                scy = StandardScaler()
                Y1 = scy.fit_transform(Xp_tr[i1])
                Y2 = scy.transform(Xp_tr[i2])
                m = Ridge(alpha=a, random_state=0)
                m.fit(Ptr[i1], Y1)
                mses.append(np.mean((Y2 - m.predict(Ptr[i2])) ** 2))
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse, best_a = mse, a
        scy = StandardScaler()
        Ytr = scy.fit_transform(Xp_tr)
        m = Ridge(alpha=best_a, random_state=0)
        m.fit(Ptr, Ytr)
        oof[te] = scy.inverse_transform(m.predict(Pte))

    rows = []
    for j, name in enumerate(feat_names):
        yt, yp = Xp[:, j], oof[:, j]
        msk = np.isfinite(yt) & np.isfinite(yp)
        if msk.sum() < 10:
            continue
        yt, yp = yt[msk], yp[msk]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        pr = float(pearsonr(yt, yp).statistic) if len(yt) > 2 and np.std(yp) > 0 else np.nan
        sp = float(spearmanr(yt, yp).statistic) if len(yt) > 2 and np.std(yp) > 0 else np.nan
        nrmse = float(np.sqrt(np.mean((yt - yp) ** 2)) / (np.std(yt) + 1e-12))
        rows.append({"feature": name, "cv_R2": float(r2), "pearson": pr, "spearman": sp, "nrmse": nrmse})
    return pd.DataFrame(rows)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(FOLDS_PATH)
    sol = pd.read_csv(SOL_PATH)
    emb = load_embeddings()
    fold_map = folds.set_index("id")["fold"].to_dict()

    freeze = {
        "state": "FUSION_CLOSURE_SPEC_FROZEN_BEFORE_SCORING",
        "spec_sha256": sha256(OUT / "FUSION_CLOSURE_SPEC.json"),
        "embedding_npz_sha256": sha256(EMB_PATH),
        "note": "hashed before fusion target scoring",
    }
    (OUT / "FUSION_CLOSURE_SPEC_HASH.json").write_text(json.dumps(freeze, indent=2) + "\n")

    # targets
    y_tm = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)
    y_hic = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_true"].astype(float)
    y_test_tm = sol.set_index("id")["TmApp"].astype(float)
    y_test_hic = sol.set_index("id")["HIC"].astype(float)
    pub = list(sol.loc[sol.is_public == 1, "id"])
    pri = list(sol.loc[sol.is_private == 1, "id"])

    plm_df = {
        "TmApp": emb_frame(emb, "ablang2__HL_paired"),
        "HIC": emb_frame(emb, "esm2__H"),
    }
    plm_name = {"TmApp": "AbLang2_HL_paired", "HIC": "ESM2_H"}
    y_dev = {"TmApp": y_tm, "HIC": y_hic}
    y_te = {"TmApp": y_test_tm, "HIC": y_test_hic}

    # --- PLM_ONLY once per target ---
    plm_only_cache = {}
    for target in ["TmApp", "HIC"]:
        ids = [i for i in y_dev[target].index if i in plm_df[target].index]
        X_plm = plm_df[target].loc[ids].to_numpy()
        y = y_dev[target].loc[ids].to_numpy()
        groups = np.asarray([int(fold_map[i]) for i in ids])
        print(f"PLM_ONLY {target} ridge/svr", flush=True)
        for mk in ["ridge", "svr"]:
            oof = nested_oof_fixed(X_plm, None, y, groups, "plm_only", mk)
            # test fit
            te_ids = [i for i in y_te[target].index if i in plm_df[target].index]
            X_te = plm_df[target].loc[te_ids].to_numpy()
            pred_te = predict_test(X_plm, X_te, None, None, y, groups, "plm_only", mk)
            plm_only_cache[(target, mk)] = {
                "ids": ids,
                "y": y,
                "groups": groups,
                "oof": oof,
                "te_ids": te_ids,
                "pred_te": pred_te,
                "y_te": y_te[target].loc[te_ids].to_numpy(),
                "X_plm": X_plm,
            }

    pred_summaries = []
    fusion_rows = []
    registry_rows = []

    for family, fmeta in FAMILIES.items():
        target = fmeta["target"]
        feat_names = load_canonical(family)
        pname = plm_name[target]

        for gen_key, gen_label in GEN_MAP.items():
            print(f"=== {family} / {gen_label} / {target} ===", flush=True)
            phys = load_physics(family, gen_key)
            base = plm_only_cache[(target, "ridge")]  # for id alignment reference
            ids = [i for i in base["ids"] if i in phys.index]
            # realign if physics missing some
            X_plm = plm_df[target].loc[ids].to_numpy()
            Xp = phys.loc[ids, feat_names].to_numpy()
            y = y_dev[target].loc[ids].to_numpy()
            groups = np.asarray([int(fold_map[i]) for i in ids])

            # Gate 3A
            pred_df = gate3a(X_plm, Xp, groups, feat_names)
            pred_df["family"] = family
            pred_df["generator"] = gen_label
            pred_df["target"] = target
            pred_df["plm"] = pname
            pred_df.to_csv(RES / f"gate3a_{family}_{gen_key}.csv", index=False)
            med_r2 = float(pred_df["cv_R2"].median())
            med_sp = float(pred_df["spearman"].median())
            plabel = predictability_label(med_r2)
            psum = {
                "family": family,
                "generator": gen_label,
                "target": target,
                "plm": pname,
                "median_cv_R2": med_r2,
                "median_spearman": med_sp,
                "frac_R2_gt_0": float((pred_df["cv_R2"] > 0).mean()),
                "frac_R2_gt_0.25": float((pred_df["cv_R2"] > 0.25).mean()),
                "frac_R2_gt_0.50": float((pred_df["cv_R2"] > 0.50).mean()),
                "predictability_class": plabel,
                "n_features": int(len(pred_df)),
            }
            for s in fmeta.get("semantic", []):
                if s in set(pred_df["feature"]):
                    row = pred_df.loc[pred_df["feature"] == s].iloc[0]
                    psum[f"semantic_{s}_R2"] = float(row["cv_R2"])
                    psum[f"semantic_{s}_spearman"] = float(row["spearman"])
            pred_summaries.append(psum)

            te_ids = [i for i in y_te[target].index if i in phys.index and i in plm_df[target].index]
            X_plm_te = plm_df[target].loc[te_ids].to_numpy()
            Xp_te = phys.loc[te_ids, feat_names].to_numpy()
            y_test = y_te[target].loc[te_ids].to_numpy()
            pub_g = [i for i in pub if i in te_ids]
            pri_g = [i for i in pri if i in te_ids]
            te_index = {a: i for i, a in enumerate(te_ids)}

            preds = {}
            for mk in ["ridge", "svr"]:
                # reuse PLM_only if same id set
                cache = plm_only_cache[(target, mk)]
                if cache["ids"] == ids:
                    oof_plm = cache["oof"]
                    pred_te_plm = cache["pred_te"]
                    # te_ids may match
                    if cache["te_ids"] == te_ids:
                        pass
                    else:
                        pred_te_plm = predict_test(
                            X_plm, X_plm_te, None, None, y, groups, "plm_only", mk
                        )
                else:
                    print(f"  recompute plm_only {mk} (id mismatch)", flush=True)
                    oof_plm = nested_oof_fixed(X_plm, None, y, groups, "plm_only", mk)
                    pred_te_plm = predict_test(X_plm, X_plm_te, None, None, y, groups, "plm_only", mk)
                preds[("plm_only", mk)] = {"oof": oof_plm, "pred_te": pred_te_plm}

                for mode in ["plus", "ortho"]:
                    print(f"  CV {mode}/{mk}", flush=True)
                    oof = nested_oof_fixed(X_plm, Xp, y, groups, mode, mk)
                    pred_te = predict_test(X_plm, X_plm_te, Xp, Xp_te, y, groups, mode, mk)
                    preds[(mode, mk)] = {"oof": oof, "pred_te": pred_te}

            for mode in ["plus", "ortho"]:
                for mk in ["ridge", "svr"]:
                    base_p = preds[("plm_only", mk)]
                    fus_p = preds[(mode, mk)]
                    m_cv_b = metrics(y, base_p["oof"])
                    m_cv_f = metrics(y, fus_p["oof"])
                    d_cv = m_cv_f["MAE"] - m_cv_b["MAE"]

                    def split_delta(id_list):
                        idx = [te_index[i] for i in id_list]
                        yy = y_test[idx]
                        return (
                            mae(yy, fus_p["pred_te"][idx]) - mae(yy, base_p["pred_te"][idx]),
                            yy,
                            fus_p["pred_te"][idx],
                            base_p["pred_te"][idx],
                        )

                    d_pub, y_pub, pf_pub, pb_pub = split_delta(pub_g)
                    d_pri, y_pri, pf_pri, pb_pri = split_delta(pri_g)
                    d_all = mae(y_test, fus_p["pred_te"]) - mae(y_test, base_p["pred_te"])

                    boots = {
                        "CV": bootstrap_delta(y, fus_p["oof"], base_p["oof"], b=B_BOOT),
                        "Public": bootstrap_delta(y_pub, pf_pub, pb_pub, b=B_BOOT),
                        "Private": bootstrap_delta(y_pri, pf_pri, pb_pri, b=B_BOOT),
                    }
                    fclass = fusion_class(d_cv, d_pub, d_pri, boots)
                    row = {
                        "family": family,
                        "generator": gen_label,
                        "target": target,
                        "plm": pname,
                        "mode": mode,
                        "model": mk,
                        "PLM_only_CV_MAE": m_cv_b["MAE"],
                        "fusion_CV_MAE": m_cv_f["MAE"],
                        "delta_CV": d_cv,
                        "delta_Public": d_pub,
                        "delta_Private": d_pri,
                        "delta_AllTest": d_all,
                        "boot_CV_lo": boots["CV"]["ci95_low"],
                        "boot_CV_hi": boots["CV"]["ci95_high"],
                        "boot_Public_lo": boots["Public"]["ci95_low"],
                        "boot_Public_hi": boots["Public"]["ci95_high"],
                        "boot_Private_lo": boots["Private"]["ci95_high"],
                        "boot_Private_hi": boots["Private"]["ci95_high"],
                        "fusion_reproducibility": fclass,
                        "plm_predictability_median_R2": med_r2,
                        "plm_predictability_class": plabel,
                    }
                    # fix Private lo typo
                    row["boot_Private_lo"] = boots["Private"]["ci95_low"]
                    fusion_rows.append(row)
                    registry_rows.append(
                        {
                            "experiment_id": f"{family}__{gen_label}__{mode}__{mk}",
                            "family": family,
                            "generator": gen_label,
                            "target": target,
                            "mode": mode,
                            "model": mk,
                            "status": "scored",
                        }
                    )

            np.savez_compressed(
                RES / f"preds_{family}_{gen_key}.npz",
                ids=np.asarray(ids),
                y=y,
                te_ids=np.asarray(te_ids),
                **{f"{m}_{k}_oof": preds[(m, k)]["oof"] for m in ["plm_only", "plus", "ortho"] for k in ["ridge", "svr"]},
                **{f"{m}_{k}_te": preds[(m, k)]["pred_te"] for m in ["plm_only", "plus", "ortho"] for k in ["ridge", "svr"]},
            )

    pd.DataFrame(pred_summaries).to_csv(RES / "GATE3A_PREDICTABILITY_SUMMARY.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(RES / "FUSION_RESULTS_ALL.csv", index=False)
    pd.DataFrame(registry_rows).to_csv(OUT / "FUSION_EXPERIMENT_REGISTRY.csv", index=False)

    # wide summary
    wide = []
    for family in FAMILIES:
        for gen_label in GEN_MAP.values():
            subset = [r for r in fusion_rows if r["family"] == family and r["generator"] == gen_label]
            if not subset:
                continue
            plus_r = next(r for r in subset if r["mode"] == "plus" and r["model"] == "ridge")
            plus_s = next(r for r in subset if r["mode"] == "plus" and r["model"] == "svr")
            ort_r = next(r for r in subset if r["mode"] == "ortho" and r["model"] == "ridge")
            ort_s = next(r for r in subset if r["mode"] == "ortho" and r["model"] == "svr")
            best = plus_r if plus_r["delta_CV"] <= plus_s["delta_CV"] else plus_s
            pred_good = plus_r["plm_predictability_median_R2"] >= 0.20
            gain = best["fusion_reproducibility"] == "REPRODUCIBLE_FUSION_GAIN"
            if pred_good and not gain:
                interp = "LIKELY_REPRESENTATIONAL_REDUNDANCY"
            elif (not pred_good) and gain:
                interp = "STRONG_COMPLEMENTARY_PHYSICS"
            elif (not pred_good) and not gain:
                interp = "DIFFERENT_INFORMATION_BUT_NOT_TARGET_USEFUL"
            else:
                interp = "PARTIAL_REDUNDANCY_WITH_USEFUL_NONLINEAR_OR_FINE_SIGNAL"

            combined = {}
            for g in GEN_MAP.values():
                rows_g = [r for r in fusion_rows if r["family"] == family and r["generator"] == g and r["mode"] == "plus"]
                if not rows_g:
                    continue
                cr = next(r["fusion_reproducibility"] for r in rows_g if r["model"] == "ridge")
                cs = next(r["fusion_reproducibility"] for r in rows_g if r["model"] == "svr")
                combined[g] = (
                    "REPRODUCIBLE_FUSION_GAIN"
                    if "REPRODUCIBLE_FUSION_GAIN" in (cr, cs)
                    else (cr if cr != "NO_FUSION_GAIN" else cs)
                )
            med_sp = next(
                p["median_spearman"]
                for p in pred_summaries
                if p["family"] == family and p["generator"] == gen_label
            )
            wide.append(
                {
                    "target": plus_r["target"],
                    "family": family,
                    "physical_generator": gen_label,
                    "mechanistic_prior": FAMILIES[family]["role"],
                    "standalone_physics_verdict": STANDALONE.get(family, ""),
                    "PLM_name": plus_r["plm"],
                    "plm_predictability_median_R2": plus_r["plm_predictability_median_R2"],
                    "plm_predictability_median_spearman": med_sp,
                    "PLM_only_CV_MAE": plus_r["PLM_only_CV_MAE"],
                    "fusion_Ridge_CV_MAE": plus_r["fusion_CV_MAE"],
                    "fusion_Ridge_delta_CV": plus_r["delta_CV"],
                    "fusion_RBF_CV_MAE": plus_s["fusion_CV_MAE"],
                    "fusion_RBF_delta_CV": plus_s["delta_CV"],
                    "delta_Public": best["delta_Public"],
                    "delta_Private": best["delta_Private"],
                    "orthogonal_fusion_delta_CV": min(ort_r["delta_CV"], ort_s["delta_CV"]),
                    "orthogonal_fusion_delta_Public": ort_r["delta_Public"]
                    if ort_r["delta_CV"] <= ort_s["delta_CV"]
                    else ort_s["delta_Public"],
                    "orthogonal_fusion_delta_Private": ort_r["delta_Private"]
                    if ort_r["delta_CV"] <= ort_s["delta_CV"]
                    else ort_s["delta_Private"],
                    "fusion_reproducibility": best["fusion_reproducibility"],
                    "generator_consistency": generator_consistency(combined),
                    "final_interpretation": interp,
                    "paper_url": "",
                    "repository_url": "",
                    "fusion_Ridge_delta_Public": plus_r["delta_Public"],
                    "fusion_Ridge_delta_Private": plus_r["delta_Private"],
                    "fusion_RBF_delta_Public": plus_s["delta_Public"],
                    "fusion_RBF_delta_Private": plus_s["delta_Private"],
                    "ortho_Ridge_delta_CV": ort_r["delta_CV"],
                    "ortho_RBF_delta_CV": ort_s["delta_CV"],
                    "ridge_reproducibility": plus_r["fusion_reproducibility"],
                    "rbf_reproducibility": plus_s["fusion_reproducibility"],
                }
            )

    pd.DataFrame(wide).to_csv(OUT / "FEATURE_FUSION_SUMMARY_CURRENT.csv", index=False)

    audit = {
        "scaler_fit": "outer_train_only",
        "pca_fit": "outer_train_only",
        "physics_residualizer_fit": "outer_train_only",
        "ridge_svr_hp_selection": "inner_GroupKFold_on_outer_train_only",
        "public_private_in_tuning": False,
        "identical_outer_folds_matched_comparisons": True,
        "primary_cv": "opt_joint_group_k5_s42 via stage0_cv/cv_primary.csv",
        "bootstrap_B": B_BOOT,
        "gate3d_status": "PRIOR_STRUCTURE_GUIDED_POOLING_SPEC_NOT_AVAILABLE",
        "no_optuna": True,
        "no_plm_finetune": True,
        "family_by_family_only": True,
        "plm_embeddings": str(EMB_PATH.relative_to(ROOT)),
        "spec_hash_file": "FUSION_CLOSURE_SPEC_HASH.json",
    }
    (OUT / "FUSION_LEAKAGE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
