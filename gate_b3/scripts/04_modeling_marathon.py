#!/usr/bin/env python3
"""
Gate B3 Train-only modeling marathon.
Public/Private labels are NEVER used for tuning — only sealed until finalists.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import spearmanr, pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, MultiTaskElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_ridge import KernelRidge

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import (  # noqa: E402
    B1_CACHE,
    B1_DATA,
    CONFIG,
    FEATURES,
    METRICS,
    ORG,
    PREDS,
    REPORTS,
    ensure_dirs,
    read_json,
    write_json,
)

sys.path.insert(0, str(Path("/workspace_developability_acquisition/gate_b1/scripts")))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tplm", "/workspace_developability_acquisition/gate_b1/scripts/10_train_plm_structure.py"
)
tplm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tplm)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 5:
        return {"spearman": np.nan, "pearson": np.nan, "mae": np.nan, "rmse": np.nan, "n": len(y)}
    sp = spearmanr(y, p).correlation
    pr = pearsonr(y, p)[0]
    return {
        "spearman": float(sp) if sp is not None and np.isfinite(sp) else np.nan,
        "pearson": float(pr) if pr is not None and np.isfinite(pr) else np.nan,
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "n": int(len(y)),
    }


def transform_y(y, mode="raw"):
    y = np.asarray(y, float)
    if mode == "raw":
        return y, None
    # rank / gauss — fit stats returned for inverse not needed (eval on pred ranks vs true)
    order = stats.rankdata(y)
    if mode == "rank":
        return (order - 0.5) / len(y), None
    if mode == "gauss":
        u = (order - 0.5) / len(y)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return stats.norm.ppf(u), None
    return y, None


def sanitize(X):
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    keep = np.any(np.isfinite(X), axis=0)
    X = X[:, keep] if keep.any() else np.zeros((X.shape[0], 1))
    col = np.nanmean(X, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    inds = np.where(~np.isfinite(X))
    X = np.array(X, copy=True)
    X[inds] = np.take(col, inds[1])
    X[~np.isfinite(X)] = 0.0
    return X


def align_ids(ids_all, X, want_ids):
    idx = {i: k for k, i in enumerate(ids_all)}
    rows = [idx[i] for i in want_ids]
    if sparse.issparse(X):
        return X[rows]
    return X[rows]


def load_csv_num(path, ids, prefixes=None, include=None, id_col="antibody_id"):
    df = pd.read_csv(path)
    if id_col not in df.columns and "id" in df.columns:
        id_col = "id"
    df = pd.DataFrame({"id": ids}).merge(df, left_on="id", right_on=id_col, how="left")
    cols = [c for c in df.columns if c not in ("id", id_col, "antibody_id") and pd.api.types.is_numeric_dtype(df[c])]
    if prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    if include:
        cols = [c for c in cols if any(s in c for s in include)]
    return sanitize(df[cols].values.astype(float))


def make_model(name, n_feat, y_train=None):
    if name == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    if name == "Ridge_grid":
        return "GRID_RIDGE"
    if name == "Lasso":
        return Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=5000))])
    if name == "ElasticNet":
        return Pipeline([("sc", StandardScaler()), ("m", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000))])
    if name == "PLS":
        ncomp = min(8, max(2, n_feat // 10), 16)
        return ("PLS", ncomp)
    if name == "LinearSVR":
        return Pipeline([("sc", StandardScaler()), ("m", LinearSVR(C=1.0, max_iter=5000))])
    if name == "SVR_RBF":
        return Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=10.0, gamma="scale"))])
    if name == "KRR":
        return Pipeline([("sc", StandardScaler()), ("m", KernelRidge(alpha=1.0, kernel="rbf", gamma=None))])
    if name == "kNN":
        return Pipeline([("sc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=7, weights="distance"))])
    raise ValueError(name)


def fit_predict(model_spec, Xtr, ytr, Xte):
    Xtr, Xte = sanitize(Xtr), sanitize(Xte)
    # align columns if mismatch from sanitize — refit sanitize jointly
    n = min(Xtr.shape[1], Xte.shape[1])
    # better: sanitize on train, apply to test
    Xtr = np.asarray(Xtr, float)
    keep = np.any(np.isfinite(Xtr), axis=0)
    Xtr = Xtr[:, keep]
    Xte = np.asarray(Xte, float)[:, keep] if Xte.shape[1] >= keep.sum() else sanitize(Xte)
    if Xte.shape[1] != Xtr.shape[1]:
        # pad/truncate
        if Xte.shape[1] > Xtr.shape[1]:
            Xte = Xte[:, : Xtr.shape[1]]
        else:
            pad = np.zeros((Xte.shape[0], Xtr.shape[1] - Xte.shape[1]))
            Xte = np.concatenate([Xte, pad], axis=1)
    col = np.nanmean(Xtr, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    for X in (Xtr, Xte):
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col, inds[1])
        X[~np.isfinite(X)] = 0.0

    if model_spec == "GRID_RIDGE":
        best, best_m = -np.inf, None
        # tiny inner holdout
        n = len(ytr)
        cut = max(int(n * 0.8), n - max(n // 5, 5))
        for a in [0.1, 1, 10, 100, 1000, 10000]:
            m = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=a))])
            m.fit(Xtr[:cut], ytr[:cut])
            pr = m.predict(Xtr[cut:])
            sp = spearmanr(ytr[cut:], pr).correlation
            if sp is not None and np.isfinite(sp) and sp > best:
                best, best_m = sp, a
        m = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=best_m or 10.0))])
        m.fit(Xtr, ytr)
        return m.predict(Xte), m
    if isinstance(model_spec, tuple) and model_spec[0] == "PLS":
        ncomp = min(model_spec[1], Xtr.shape[0] - 2, Xtr.shape[1])
        ncomp = max(ncomp, 1)
        m = Pipeline([("sc", StandardScaler()), ("m", PLSRegression(n_components=ncomp))])
        m.fit(Xtr, ytr)
        return m.predict(Xte).ravel(), m
    m = model_spec
    if not hasattr(m, "fit"):
        m = make_model(m, Xtr.shape[1])
        if isinstance(m, tuple) or m == "GRID_RIDGE":
            return fit_predict(m, Xtr, ytr, Xte)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte)).ravel(), m


def repeated_cv(X, y, fold_payload, train_ids, model_names, y_mode="raw", tag=""):
    """X rows aligned to train_ids order in fold_payload."""
    id_to_row = {i: k for k, i in enumerate(fold_payload["train_ids"])}
    # X might be in different order — assume caller aligned to fold_payload train_ids
    results = []
    oof_store = {mn: np.full(len(y), np.nan) for mn in model_names}
    for fold_info in fold_payload["folds"]:
        fold_id = np.array(fold_info["fold_id"])
        rep = fold_info["repeat"]
        for f in range(fold_payload["n_folds"]):
            te = fold_id == f
            tr = ~te
            if tr.sum() < 10 or te.sum() < 3:
                continue
            ytr_raw, yte_raw = y[tr], y[te]
            ytr, _ = transform_y(ytr_raw, y_mode)
            for mn in model_names:
                try:
                    pred, _ = fit_predict(make_model(mn, X.shape[1]), X[tr], ytr, X[te])
                    # pred is in transformed space; Spearman vs raw y is still valid for rank transforms
                    met = metrics(yte_raw, pred)
                    results.append(
                        {
                            "tag": tag,
                            "model": mn,
                            "y_mode": y_mode,
                            "repeat": rep,
                            "fold": f,
                            **met,
                        }
                    )
                    # accumulate OOF for first model names for ensembles — use raw-space pred
                    if y_mode == "raw":
                        oof_store[mn][te] = pred
                except Exception as e:
                    results.append(
                        {"tag": tag, "model": mn, "y_mode": y_mode, "repeat": rep, "fold": f, "spearman": np.nan, "error": str(e)[:120]}
                    )
    return results, oof_store


def summarize(rows, tag, model):
    sub = [r for r in rows if r.get("tag") == tag and r.get("model") == model and np.isfinite(r.get("spearman", np.nan))]
    if not sub:
        return {"mean_spearman": np.nan, "sd_spearman": np.nan, "n": 0}
    sps = [r["spearman"] for r in sub]
    return {"mean_spearman": float(np.mean(sps)), "sd_spearman": float(np.std(sps)), "n": len(sps)}


def plm_latent_cv(X, y, fold_payload, tag_prefix, y_mode="raw"):
    """PCA/PLS inside each fold."""
    rows = []
    fold_id_list = fold_payload["folds"]
    for fold_info in fold_id_list:
        fold_id = np.array(fold_info["fold_id"])
        rep = fold_info["repeat"]
        for f in range(fold_payload["n_folds"]):
            te = fold_id == f
            tr = ~te
            Xtr, Xte = sanitize(X[tr]), None
            # fit PCA on train
            keep = np.any(np.isfinite(X[tr]), axis=0)
            Xtr_raw = np.asarray(X[tr], float)[:, keep]
            Xte_raw = np.asarray(X[te], float)[:, keep]
            col = np.nanmean(Xtr_raw, axis=0)
            col = np.where(np.isfinite(col), col, 0)
            for XX in (Xtr_raw, Xte_raw):
                inds = np.where(~np.isfinite(XX))
                XX[inds] = np.take(col, inds[1])
            ytr, _ = transform_y(y[tr], y_mode)
            for npc in [8, 16, 32, 64]:
                if npc >= min(Xtr_raw.shape[0] - 1, Xtr_raw.shape[1]):
                    continue
                pca = PCA(n_components=npc, random_state=0)
                Ztr = pca.fit_transform(StandardScaler().fit_transform(Xtr_raw))
                # need same scaler
                sc = StandardScaler().fit(Xtr_raw)
                Ztr = pca.fit_transform(sc.transform(Xtr_raw))
                Zte = pca.transform(sc.transform(Xte_raw))
                for mn, est in [
                    ("Ridge", Ridge(alpha=1.0)),
                    ("SVR_RBF", SVR(kernel="rbf", C=10.0)),
                    ("KRR", KernelRidge(alpha=1.0, kernel="rbf")),
                ]:
                    try:
                        est.fit(Ztr, ytr)
                        pred = est.predict(Zte)
                        met = metrics(y[te], pred)
                        rows.append(
                            {
                                "tag": f"{tag_prefix}_PCA{npc}",
                                "model": mn,
                                "y_mode": y_mode,
                                "repeat": rep,
                                "fold": f,
                                **met,
                            }
                        )
                    except Exception:
                        pass
            for nc in [2, 4, 8, 16]:
                if nc >= min(Xtr_raw.shape[0] - 1, Xtr_raw.shape[1]):
                    continue
                try:
                    sc = StandardScaler().fit(Xtr_raw)
                    pls = PLSRegression(n_components=nc)
                    pls.fit(sc.transform(Xtr_raw), ytr)
                    pred = pls.predict(sc.transform(Xte_raw)).ravel()
                    met = metrics(y[te], pred)
                    rows.append(
                        {
                            "tag": f"{tag_prefix}_PLS{nc}",
                            "model": "PLS",
                            "y_mode": y_mode,
                            "repeat": rep,
                            "fold": f,
                            **met,
                        }
                    )
                except Exception:
                    pass
    return rows


def main():
    ensure_dirs()
    pop = pd.read_csv(ORG / "final_population.csv")
    role = pd.read_csv(ORG / "role_map.csv")[["id", "role"]]
    df = pop.merge(role, on="id")
    train = df[df.role == "Train"].copy().reset_index(drop=True)
    fold_payload = read_json(CONFIG / "TRAIN_CV_FOLDS.json")
    # reorder train to fold id order
    train = train.set_index("id").loc[fold_payload["train_ids"]].reset_index()
    assert list(train["id"]) == fold_payload["train_ids"]

    all_rows = []
    oofs = {}  # target -> tag -> oof vector

    for target, col in [("HIC", "HIC"), ("TmApp", "TmApp")]:
        print(f"=== TARGET {target} ===", flush=True)
        y = train[col].values.astype(float)
        oofs[target] = {}

        # --- B2 baselines ---
        reps = []
        reps.append(("SEQ_SIMPLE", load_csv_num(B1_CACHE / "features" / "stage_A_simple.csv", train.id, prefixes=["A0_", "A1_", "A2_"])))
        reps.append(("SEQ_CDR", load_csv_num(B1_CACHE / "features" / "stage_B_cdr.csv", train.id, prefixes=["B_"])))
        # BIO numeric
        bio = pd.read_csv(B1_CACHE / "features" / "stage_C_shortcut.csv")
        bio = train[["id"]].merge(bio, left_on="id", right_on="antibody_id", how="left")
        bio_num = bio.select_dtypes(include=[np.number]).values.astype(float)
        # one-hot cats
        cats = [c for c in ["C_vh_family", "C_vl_family", "C_kappa_lambda"] if c in bio.columns]
        if cats:
            Xcat = pd.get_dummies(bio[cats].astype(str), dummy_na=True).values.astype(float)
            bio_X = sanitize(np.concatenate([sanitize(bio_num), Xcat], axis=1))
        else:
            bio_X = sanitize(bio_num)
        reps.append(("BIO_SHORTCUT", bio_X))

        for man, name in [
            ("manifest_ablang2_default.csv", "PLM_ABLANG2"),
            ("manifest_esm1b_t33_650M_UR50S.csv", "PLM_ESM1B"),
            ("manifest_esm2_t33_650M_UR50D.csv", "PLM_ESM2"),
            ("manifest_esm2_t33_650M_UR50D_CDR6.csv", "PLM_ESM2_CDR6"),
        ]:
            X = tplm.load_embedding_matrix(B1_CACHE / "plm" / man, train["id"])
            reps.append((name, sanitize(X)))

        abb = load_csv_num(
            FEATURES / "structure_ext" / "structure_extended.csv",
            train.id,
            prefixes=["ABB_"],
            id_col="id",
        )
        esmn = load_csv_num(
            FEATURES / "structure_ext" / "structure_extended.csv",
            train.id,
            prefixes=["ESMFN_", "CONSENSUS_", "COM_"],
            id_col="id",
        )
        # fallback to B2 if ext missing cols
        if abb.shape[1] < 5:
            abb = load_csv_num(
                Path("/workspace_developability_acquisition/gate_b2/cache/structure_features/abb_sasa_rasa_patch.csv"),
                train.id,
                prefixes=["ABB_"],
            )
        if esmn.shape[1] < 5:
            esmn = load_csv_num(
                Path("/workspace_developability_acquisition/gate_b2/cache/structure_features/esmfold_native_sasa_rasa_patch.csv"),
                train.id,
                prefixes=["ESMFN_"],
            )
        reps.append(("ABB_STRUCTURE", abb))
        reps.append(("ESMFN_STRUCTURE", esmn))

        # IMGT
        imgt = np.load(FEATURES / "imgt" / "positional_onehot.npz", allow_pickle=True)
        imgt_ids = list(imgt["ids"])
        Xhl = align_ids(imgt_ids, imgt["Xhl"], train["id"].tolist())
        Xh = align_ids(imgt_ids, imgt["Xh"], train["id"].tolist())
        Xl = align_ids(imgt_ids, imgt["Xl"], train["id"].tolist())
        reps.append(("IMGT_POS_HL", sanitize(Xhl)))
        reps.append(("IMGT_POS_H", sanitize(Xh)))
        reps.append(("IMGT_POS_L", sanitize(Xl)))

        # Germline relative
        ger = load_csv_num(FEATURES / "germline" / "germline_relative.csv", train.id, id_col="id")
        reps.append(("GERMLINE_REL", ger))

        # Ngrams (HL 2mer, 3mer)
        ng_ids = list(np.load(FEATURES / "ngram" / "ids.npy", allow_pickle=True))
        for n in [2, 3]:
            X = sparse.load_npz(FEATURES / "ngram" / f"tfidf_HL_{n}mer.npz")
            X = align_ids(ng_ids, X, train["id"].tolist())
            reps.append((f"NGRAM_HL_{n}", sanitize(X)))

        models_lin = ["Ridge_grid", "ElasticNet", "PLS"]
        models_struct = ["Ridge_grid", "ElasticNet", "PLS"]
        models_plm = ["Ridge_grid", "ElasticNet"]

        for tag, X in reps:
            print(f"  CV {tag} d={X.shape[1]}", flush=True)
            mnames = models_plm if tag.startswith("PLM") or tag.startswith("IMGT") or tag.startswith("NGRAM") else models_lin
            if tag.startswith("ABB") or tag.startswith("ESMFN"):
                mnames = models_struct + ["kNN"]
            rows, oof = repeated_cv(X, y, fold_payload, train.id.tolist(), mnames, y_mode="raw", tag=tag)
            all_rows.extend([{**r, "target": target} for r in rows])
            # keep best model oof
            best_mn, best_sp = None, -np.inf
            for mn in mnames:
                s = summarize(rows, tag, mn)
                if s["mean_spearman"] > best_sp:
                    best_sp, best_mn = s["mean_spearman"], mn
            if best_mn and best_mn in oof:
                oofs[target][tag] = oof[best_mn]
                print(f"    best {best_mn} meanρ={best_sp:.3f}", flush=True)

        # Target transforms on best PLM + SEQ
        for tag, X in [("PLM_ESM2", dict(reps)["PLM_ESM2"]), ("SEQ_CDR", dict(reps)["SEQ_CDR"])]:
            for mode in ["rank", "gauss"]:
                rows, _ = repeated_cv(X, y, fold_payload, train.id.tolist(), ["Ridge_grid"], y_mode=mode, tag=f"{tag}_{mode}")
                all_rows.extend([{**r, "target": target} for r in rows])

        # PLM latent
        print("  PLM latent PCA/PLS…", flush=True)
        for plm_name in ["PLM_ESM2", "PLM_ABLANG2", "PLM_ESM1B"]:
            X = dict(reps)[plm_name]
            lat = plm_latent_cv(X, y, fold_payload, plm_name, y_mode="raw")
            all_rows.extend([{**r, "target": target} for r in lat])

        # Fusions block-scaled
        def block_concat(blocks):
            parts = []
            for B in blocks:
                B = sanitize(B)
                # equal block norm
                nrm = np.linalg.norm(B) / max(np.sqrt(B.shape[0]), 1)
                parts.append(B / max(nrm, 1e-8))
            return np.concatenate(parts, axis=1)

        fusions = {
            "FUSION_ESM2_BIO": block_concat([dict(reps)["PLM_ESM2"], bio_X]),
            "FUSION_ESM2_SEQ": block_concat([dict(reps)["PLM_ESM2"], dict(reps)["SEQ_CDR"]]),
            "FUSION_ESM2_ESMFN": block_concat([dict(reps)["PLM_ESM2"], esmn]),
            "FUSION_ESM2_BIO_ESMFN": block_concat([dict(reps)["PLM_ESM2"], bio_X, esmn]),
            "FUSION_ESM2_IMGT": block_concat([dict(reps)["PLM_ESM2"], dict(reps)["IMGT_POS_HL"]]),
        }
        for tag, X in fusions.items():
            print(f"  CV {tag} d={X.shape[1]}", flush=True)
            rows, oof = repeated_cv(X, y, fold_payload, train.id.tolist(), ["Ridge_grid", "ElasticNet"], tag=tag)
            all_rows.extend([{**r, "target": target} for r in rows])
            s = summarize(rows, tag, "Ridge_grid")
            oofs[target][tag] = oof["Ridge_grid"]
            print(f"    Ridge meanρ={s['mean_spearman']:.3f}", flush=True)

        # Residual: BIO -> ESMFN / PLM
        print("  Residuals…", flush=True)
        for base_tag, res_tag in [("BIO_SHORTCUT", "PLM_ESM2"), ("SEQ_SIMPLE", "ESMFN_STRUCTURE"), ("PLM_ESM2", "ESMFN_STRUCTURE")]:
            if base_tag not in oofs[target] or res_tag not in dict(reps):
                # need base oof — compute quickly Ridge OOF
                pass
            Xbase = dict(reps)[base_tag]
            rows_b, oof_b = repeated_cv(Xbase, y, fold_payload, train.id.tolist(), ["Ridge_grid"], tag=f"_tmp_{base_tag}")
            base_pred = oof_b["Ridge_grid"]
            resid = y - base_pred
            Xres = dict(reps)[res_tag]
            # CV on residual target but report Spearman of base+resid_pred vs y
            fold_id_all = fold_payload["folds"]
            sps = []
            for fold_info in fold_id_all:
                fold_id = np.array(fold_info["fold_id"])
                for f in range(5):
                    te = fold_id == f
                    tr = ~te
                    pred_r, _ = fit_predict(make_model("Ridge_grid", Xres.shape[1]), Xres[tr], resid[tr], Xres[te])
                    # For OOF consistency use base from same fold — approximate with stored base_pred
                    comb = base_pred[te] + pred_r
                    sps.append(metrics(y[te], comb)["spearman"])
            all_rows.append(
                {
                    "target": target,
                    "tag": f"RESID_{base_tag}__{res_tag}",
                    "model": "Ridge_stack",
                    "y_mode": "raw",
                    "repeat": -1,
                    "fold": -1,
                    "spearman": float(np.nanmean(sps)),
                    "pearson": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "n": len(sps),
                }
            )
            print(f"    RESID {base_tag}->{res_tag} meanρ={np.nanmean(sps):.3f}", flush=True)

        # Ensembles from OOF
        print("  Ensembles…", flush=True)
        keys = [k for k, v in oofs[target].items() if np.isfinite(v).mean() > 0.8]
        if len(keys) >= 2:
            stack = np.vstack([oofs[target][k] for k in keys])
            # rank mean
            from scipy.stats import rankdata

            rank_mean = np.mean([rankdata(oofs[target][k]) for k in keys], axis=0)
            all_rows.append(
                {
                    "target": target,
                    "tag": "ENSEMBLE_RANKMEAN",
                    "model": "rank_mean",
                    "y_mode": "raw",
                    "repeat": -1,
                    "fold": -1,
                    "spearman": metrics(y, rank_mean)["spearman"],
                    "pearson": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "n": len(y),
                    "members": ",".join(keys),
                }
            )
            # nonnegative blend via Ridge on OOF (in-sample optimistic — also do fold-wise)
            sps = []
            for fold_info in fold_payload["folds"]:
                fold_id = np.array(fold_info["fold_id"])
                for f in range(5):
                    te = fold_id == f
                    tr = ~te
                    A = np.vstack([oofs[target][k][tr] for k in keys]).T
                    w = Ridge(alpha=1.0, fit_intercept=False).fit(A, y[tr]).coef_
                    w = np.maximum(w, 0)
                    w = w / w.sum() if w.sum() > 0 else np.ones(len(keys)) / len(keys)
                    pred = sum(w[i] * oofs[target][keys[i]][te] for i in range(len(keys)))
                    sps.append(metrics(y[te], pred)["spearman"])
            all_rows.append(
                {
                    "target": target,
                    "tag": "ENSEMBLE_NNLS",
                    "model": "nonneg_ridge",
                    "y_mode": "raw",
                    "repeat": -1,
                    "fold": -1,
                    "spearman": float(np.nanmean(sps)),
                    "pearson": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "n": len(sps),
                    "members": ",".join(keys),
                }
            )
            print(f"    rankmean={metrics(y, rank_mean)['spearman']:.3f} nnls={np.nanmean(sps):.3f}", flush=True)

        # Multitask quick (only once when target==HIC)
        if target == "HIC":
            print("  Multitask ElasticNet…", flush=True)
            Y2 = train[["HIC", "TmApp"]].values.astype(float)
            X = dict(reps)["PLM_ESM2"]
            sps_h, sps_t = [], []
            for fold_info in fold_payload["folds"][:2]:  # 2 repeats to save time
                fold_id = np.array(fold_info["fold_id"])
                for f in range(5):
                    te = fold_id == f
                    tr = ~te
                    pipe = Pipeline(
                        [
                            ("sc", StandardScaler()),
                            ("m", MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=4000)),
                        ]
                    )
                    try:
                        pipe.fit(sanitize(X[tr]), Y2[tr])
                        pred = pipe.predict(sanitize(X[te]))
                        sps_h.append(metrics(Y2[te, 0], pred[:, 0])["spearman"])
                        sps_t.append(metrics(Y2[te, 1], pred[:, 1])["spearman"])
                    except Exception:
                        pass
            all_rows.append(
                {
                    "target": "HIC",
                    "tag": "MULTITASK_EN_ESM2",
                    "model": "MultiTaskEN",
                    "spearman": float(np.nanmean(sps_h)),
                    "y_mode": "raw",
                    "repeat": -1,
                    "fold": -1,
                }
            )
            all_rows.append(
                {
                    "target": "TmApp",
                    "tag": "MULTITASK_EN_ESM2",
                    "model": "MultiTaskEN",
                    "spearman": float(np.nanmean(sps_t)),
                    "y_mode": "raw",
                    "repeat": -1,
                    "fold": -1,
                }
            )

        # checkpoint
        pd.DataFrame(all_rows).to_csv(METRICS / "all_train_cv_results.csv", index=False)

    pd.DataFrame(all_rows).to_csv(METRICS / "all_train_cv_results.csv", index=False)
    # save oofs
    for target, dct in oofs.items():
        for tag, vec in dct.items():
            np.save(PREDS / "oof" / f"{target}_{tag}.npy", vec)
    write_json(PREDS / "oof" / "index.json", {t: list(d.keys()) for t, d in oofs.items()})
    print("MODELING_MARATHON_OK", len(all_rows))


if __name__ == "__main__":
    main()
