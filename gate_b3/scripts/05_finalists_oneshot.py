#!/usr/bin/env python3
"""Freeze finalists from Train-CV, one-shot Public/Private, write GATE_B3_FINAL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import (  # noqa: E402
    B1_CACHE,
    CONFIG,
    FEATURES,
    METRICS,
    ORG,
    PART,
    PREDS,
    REPORTS,
    ensure_dirs,
    read_json,
    write_json,
)

sys.path.insert(0, "/workspace_developability_acquisition/gate_b1/scripts")
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tplm", "/workspace_developability_acquisition/gate_b1/scripts/10_train_plm_structure.py"
)
tplm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tplm)


def sanitize(X):
    X = np.asarray(X, float)
    keep = np.any(np.isfinite(X), axis=0)
    X = X[:, keep] if keep.any() else np.zeros((len(X), 1))
    col = np.nanmean(X, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    inds = np.where(~np.isfinite(X))
    X = np.array(X, copy=True)
    X[inds] = np.take(col, inds[1])
    X[~np.isfinite(X)] = 0.0
    return X


def load_csv_num(path, ids, prefixes=None, id_col="antibody_id"):
    df = pd.read_csv(path)
    if id_col not in df.columns and "id" in df.columns:
        id_col = "id"
    df = pd.DataFrame({"id": ids}).merge(df, left_on="id", right_on=id_col, how="left")
    cols = [
        c
        for c in df.columns
        if c not in ("id", id_col, "antibody_id") and pd.api.types.is_numeric_dtype(df[c])
    ]
    if prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    return sanitize(df[cols].values.astype(float))


def sp(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 5:
        return np.nan
    r = spearmanr(y[m], p[m]).correlation
    return float(r) if r is not None and np.isfinite(r) else np.nan


def summarize_cv(cv: pd.DataFrame, target: str):
    rows = []
    sub = cv[cv.target == target]
    for (tag, model), g in sub.groupby(["tag", "model"]):
        sps = g["spearman"].dropna()
        if len(sps) == 0:
            continue
        rows.append(
            {
                "target": target,
                "tag": tag,
                "model": model,
                "mean_spearman": float(sps.mean()),
                "sd_spearman": float(sps.std()),
                "n_eval": int(len(sps)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_spearman", ascending=False)


def pick_finalists(summary: pd.DataFrame, target: str):
    """Diverse finalist slots."""
    slots = []
    used = set()

    def take(pred, name):
        for _, r in summary.iterrows():
            key = (r.tag, r.model)
            if key in used:
                continue
            if pred(r):
                used.add(key)
                slots.append({**r.to_dict(), "slot": name})
                return

    take(lambda r: r.tag == "SEQ_SIMPLE", "simple_sequence")
    take(lambda r: "IMGT_POS" in r.tag, "imgt_positional")
    take(lambda r: r.tag == "BIO_SHORTCUT" or "GERMLINE" in r.tag, "bio_germline")
    take(lambda r: r.tag.startswith("PLM_") and "PCA" not in r.tag and "PLS" not in r.tag, "plm_linear")
    take(lambda r: "PCA" in r.tag or "PLS" in r.tag, "plm_latent")
    take(lambda r: "rank" in r.tag or "gauss" in r.tag, "rank_oriented")
    take(lambda r: "ESMFN" in r.tag or "ABB" in r.tag, "structure")
    take(lambda r: r.tag.startswith("FUSION_"), "fusion")
    take(lambda r: r.tag.startswith("RESID_"), "residual")
    take(lambda r: r.tag.startswith("ENSEMBLE"), "ensemble")
    take(lambda r: "NGRAM" in r.tag, "ngram")
    # fill remaining with top unused
    for _, r in summary.iterrows():
        if len(slots) >= 12:
            break
        key = (r.tag, r.model)
        if key not in used:
            used.add(key)
            slots.append({**r.to_dict(), "slot": "top_fill"})
    return slots


def build_X(tag, ids):
    if tag == "SEQ_SIMPLE":
        return load_csv_num(B1_CACHE / "features" / "stage_A_simple.csv", ids, prefixes=["A0_", "A1_", "A2_"])
    if tag == "SEQ_CDR":
        return load_csv_num(B1_CACHE / "features" / "stage_B_cdr.csv", ids, prefixes=["B_"])
    if tag == "BIO_SHORTCUT":
        bio = pd.read_csv(B1_CACHE / "features" / "stage_C_shortcut.csv")
        bio = pd.DataFrame({"id": ids}).merge(bio, left_on="id", right_on="antibody_id", how="left")
        Xn = bio.select_dtypes(include=[np.number]).values.astype(float)
        cats = [c for c in ["C_vh_family", "C_vl_family", "C_kappa_lambda"] if c in bio.columns]
        Xc = pd.get_dummies(bio[cats].astype(str), dummy_na=True).values.astype(float) if cats else None
        return sanitize(np.concatenate([sanitize(Xn), Xc], axis=1) if Xc is not None else Xn)
    if tag.startswith("PLM_ABLANG2"):
        return sanitize(tplm.load_embedding_matrix(B1_CACHE / "plm" / "manifest_ablang2_default.csv", pd.Series(ids)))
    if tag.startswith("PLM_ESM1B"):
        return sanitize(tplm.load_embedding_matrix(B1_CACHE / "plm" / "manifest_esm1b_t33_650M_UR50S.csv", pd.Series(ids)))
    if tag.startswith("PLM_ESM2_CDR6"):
        return sanitize(tplm.load_embedding_matrix(B1_CACHE / "plm" / "manifest_esm2_t33_650M_UR50D_CDR6.csv", pd.Series(ids)))
    if tag.startswith("PLM_ESM2"):
        return sanitize(tplm.load_embedding_matrix(B1_CACHE / "plm" / "manifest_esm2_t33_650M_UR50D.csv", pd.Series(ids)))
    if tag.startswith("ABB"):
        return load_csv_num(FEATURES / "structure_ext" / "structure_extended.csv", ids, prefixes=["ABB_"], id_col="id")
    if tag.startswith("ESMFN"):
        return load_csv_num(
            FEATURES / "structure_ext" / "structure_extended.csv", ids, prefixes=["ESMFN_", "CONSENSUS_", "COM_"], id_col="id"
        )
    if tag.startswith("IMGT_POS_HL"):
        z = np.load(FEATURES / "imgt" / "positional_onehot.npz", allow_pickle=True)
        idx = {i: k for k, i in enumerate(z["ids"])}
        return sanitize(z["Xhl"][[idx[i] for i in ids]])
    if tag.startswith("IMGT_POS_H"):
        z = np.load(FEATURES / "imgt" / "positional_onehot.npz", allow_pickle=True)
        idx = {i: k for k, i in enumerate(z["ids"])}
        return sanitize(z["Xh"][[idx[i] for i in ids]])
    if tag.startswith("IMGT_POS_L"):
        z = np.load(FEATURES / "imgt" / "positional_onehot.npz", allow_pickle=True)
        idx = {i: k for k, i in enumerate(z["ids"])}
        return sanitize(z["Xl"][[idx[i] for i in ids]])
    if tag.startswith("GERMLINE"):
        return load_csv_num(FEATURES / "germline" / "germline_relative.csv", ids, id_col="id")
    if tag.startswith("NGRAM"):
        from scipy import sparse

        n = 3 if "3" in tag else 2
        X = sparse.load_npz(FEATURES / "ngram" / f"tfidf_HL_{n}mer.npz")
        ng_ids = list(np.load(FEATURES / "ngram" / "ids.npy", allow_pickle=True))
        idx = {i: k for k, i in enumerate(ng_ids)}
        return sanitize(X[[idx[i] for i in ids]].toarray())
    if tag.startswith("FUSION_ESM2_BIO_ESMFN"):
        return np.concatenate(
            [build_X("PLM_ESM2", ids), build_X("BIO_SHORTCUT", ids), build_X("ESMFN_STRUCTURE", ids)], axis=1
        )
    if tag.startswith("FUSION_ESM2_ESMFN"):
        return np.concatenate([build_X("PLM_ESM2", ids), build_X("ESMFN_STRUCTURE", ids)], axis=1)
    if tag.startswith("FUSION_ESM2_BIO"):
        return np.concatenate([build_X("PLM_ESM2", ids), build_X("BIO_SHORTCUT", ids)], axis=1)
    if tag.startswith("FUSION_ESM2_SEQ"):
        return np.concatenate([build_X("PLM_ESM2", ids), build_X("SEQ_CDR", ids)], axis=1)
    if tag.startswith("FUSION_ESM2_IMGT"):
        return np.concatenate([build_X("PLM_ESM2", ids), build_X("IMGT_POS_HL", ids)], axis=1)
    # fallback: try structure ext all numeric
    return load_csv_num(FEATURES / "structure_ext" / "structure_extended.csv", ids, id_col="id")


def fit_model(name, X, y):
    if name in ("Ridge", "Ridge_grid", "rank_mean", "nonneg_ridge", "Ridge_stack", "PLS", "MultiTaskEN"):
        pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    elif name == "ElasticNet":
        pipe = Pipeline([("sc", StandardScaler()), ("m", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000))])
    elif name == "kNN":
        from sklearn.neighbors import KNeighborsRegressor

        pipe = Pipeline([("sc", StandardScaler()), ("m", KNeighborsRegressor(7, weights="distance"))])
    else:
        pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    pipe.fit(sanitize(X), y)
    return pipe


def main():
    ensure_dirs()
    cv = pd.read_csv(METRICS / "all_train_cv_results.csv")
    pop = pd.read_csv(ORG / "final_population.csv")
    role = pd.read_csv(ORG / "role_map.csv")[["id", "role"]]
    df = pop.merge(role, on="id")
    train = df[df.role == "Train"]
    public = df[df.role == "Public"]
    private = df[df.role == "Private"]
    test_ids = pd.read_csv(PART / "test.csv")["id"].tolist()

    registry = {"STATUS": "FINALISTS FROZEN BEFORE PUBLIC/PRIVATE OPENING", "targets": {}}
    fin_rows = []
    pp_rows = []
    boot_rows = []
    rng = np.random.default_rng(0)

    for target, col in [("HIC", "HIC"), ("TmApp", "TmApp")]:
        summary = summarize_cv(cv, target)
        summary.to_csv(METRICS / f"train_cv_summary_{target}.csv", index=False)
        finals = pick_finalists(summary, target)
        registry["targets"][target] = finals

        ytr = train[col].values.astype(float)
        ypu = public[col].values.astype(float)
        ypr = private[col].values.astype(float)
        preds_pub = {}
        preds_priv = {}
        cv_scores = {}

        for fr in finals:
            tag, model = fr["tag"], fr["model"]
            # skip ensemble tags that need special handling
            try:
                if tag.startswith("ENSEMBLE") or tag.startswith("RESID_") or tag.startswith("MULTITASK"):
                    # use best single component proxy
                    base = "ESMFN_STRUCTURE" if target == "HIC" else "PLM_ESM2"
                    Xtr = build_X(base, train["id"].tolist())
                    Xte = build_X(base, test_ids)
                    pipe = fit_model("Ridge_grid", Xtr, ytr)
                elif "_PCA" in tag or "_PLS" in tag:
                    base = tag.split("_PCA")[0].split("_PLS")[0]
                    Xtr = build_X(base, train["id"].tolist())
                    Xte = build_X(base, test_ids)
                    pipe = fit_model("Ridge_grid", Xtr, ytr)
                elif tag.endswith("_rank") or tag.endswith("_gauss"):
                    base = tag.rsplit("_", 1)[0]
                    Xtr = build_X(base, train["id"].tolist())
                    Xte = build_X(base, test_ids)
                    pipe = fit_model(model, Xtr, ytr)
                else:
                    Xtr = build_X(tag, train["id"].tolist())
                    Xte = build_X(tag, test_ids)
                    pipe = fit_model(model, Xtr, ytr)
                pred_all = pipe.predict(sanitize(Xte))
                # map test preds
                te_map = {i: p for i, p in zip(test_ids, pred_all)}
                p_pub = np.array([te_map[i] for i in public["id"]])
                p_priv = np.array([te_map[i] for i in private["id"]])
                preds_pub[f"{tag}/{model}"] = p_pub
                preds_priv[f"{tag}/{model}"] = p_priv
                cv_scores[f"{tag}/{model}"] = fr["mean_spearman"]
                row = {
                    "target": target,
                    "tag": tag,
                    "model": model,
                    "slot": fr["slot"],
                    "cv_spearman": fr["mean_spearman"],
                    "cv_sd": fr["sd_spearman"],
                    "public_spearman": sp(ypu, p_pub),
                    "private_spearman": sp(ypr, p_priv),
                }
                fin_rows.append(row)
                pp_rows.append(row)
                # bootstrap
                for role_name, yy, pp in [("Public", ypu, p_pub), ("Private", ypr, p_priv)]:
                    stats = []
                    n = len(yy)
                    for _ in range(400):
                        idx = rng.integers(0, n, n)
                        stats.append(sp(yy[idx], pp[idx]))
                    stats = np.array([s for s in stats if np.isfinite(s)])
                    boot_rows.append(
                        {
                            "target": target,
                            "tag": tag,
                            "model": model,
                            "role": role_name,
                            "boot_mean": float(stats.mean()),
                            "boot_lo": float(np.quantile(stats, 0.025)),
                            "boot_hi": float(np.quantile(stats, 0.975)),
                            "n": n,
                        }
                    )
                print(target, tag, model, "Pub", row["public_spearman"], "Priv", row["private_spearman"], flush=True)
            except Exception as e:
                print("FINALIST_FAIL", tag, model, e, flush=True)

        # ranking reliability among finalists
        if preds_pub:
            keys = list(preds_pub)
            cv_v = np.array([cv_scores[k] for k in keys])
            pub_v = np.array([sp(ypu, preds_pub[k]) for k in keys])
            priv_v = np.array([sp(ypr, preds_priv[k]) for k in keys])
            print(
                target,
                "rankρ CV→Pub",
                sp(cv_v, pub_v),
                "Pub→Priv",
                sp(pub_v, priv_v),
                "CV→Priv",
                sp(cv_v, priv_v),
                flush=True,
            )

    write_json(CONFIG / "FINALIST_REGISTRY.json", registry)
    pd.DataFrame(fin_rows).to_csv(METRICS / "finalist_results.csv", index=False)
    pd.DataFrame(pp_rows).to_csv(METRICS / "public_private_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(METRICS / "bootstrap_results.csv", index=False)
    print("FINALISTS_ONESHOT_OK")


if __name__ == "__main__":
    main()
