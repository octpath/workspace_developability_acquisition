#!/usr/bin/env python3
"""Residual headroom + rank-average / Dev-OOF linear blend ensembles."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import B1_CACHE, B1_DATA, B2_TARGETS, CACHE, METRICS, PREDS, SPLITS, ensure_dirs  # noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tplm", "/workspace_developability_acquisition/gate_b1/scripts/10_train_plm_structure.py"
)
tplm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tplm)


def metrics(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 5:
        return np.nan, np.nan
    return float(spearmanr(y[m], p[m]).correlation), float(np.corrcoef(y[m], p[m])[0, 1])


def sanitize_X(X, fit_mask=None):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return np.zeros((len(X), 1))
    keep = np.any(np.isfinite(X), axis=0)
    X = X[:, keep]
    if fit_mask is not None:
        keep2 = np.any(np.isfinite(X[fit_mask]), axis=0)
        X = X[:, keep2]
    if X.shape[1] == 0:
        return np.zeros((X.shape[0], 1))
    col = np.nanmean(X, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    inds = np.where(~np.isfinite(X))
    X = np.array(X, copy=True)
    X[inds] = np.take(col, inds[1])
    X[~np.isfinite(X)] = 0.0
    return X


def oof_ridge(X, y, groups):
    X = sanitize_X(X)
    oof = np.full(len(y), np.nan)
    n_outer = min(5, len(np.unique(groups)))
    while n_outer > 2 and len(np.unique(groups)) // n_outer < 2:
        n_outer -= 1
    for tr, te in GroupKFold(n_splits=n_outer).split(X, y, groups):
        m = Ridge(alpha=10.0)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def load_csv(path, ids, include=None, prefixes=None):
    feat = pd.read_csv(path)
    feat = ids.to_frame("antibody_id").merge(feat, on="antibody_id", how="left")
    cols = [c for c in feat.columns if c != "antibody_id"]
    if prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    if include:
        cols = [c for c in cols if any(s in c for s in include)]
    return feat[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)


def main():
    ensure_dirs()
    PREDS.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, col in B2_TARGETS.items():
        df = pd.read_csv(B1_DATA / f"{'hic' if target == 'HIC' else 'tmapp'}_full.csv")
        sp = pd.read_csv(SPLITS / f"{target.lower()}_canonical.csv")
        m = sp.merge(df, on="antibody_id")
        y = m[col].values.astype(float)
        roles = m["role"].values
        groups = m["cluster_id"].values
        ids = m["antibody_id"]
        dev = roles == "Dev"

        Xsimple = load_csv(CACHE / "features" / "stage_A_simple.csv", ids, prefixes=["A2_"])
        Xbio = load_csv(CACHE / "features" / "stage_C_shortcut.csv", ids)
        # numeric only for bio
        bio = pd.read_csv(CACHE / "features" / "stage_C_shortcut.csv")
        bio = ids.to_frame("antibody_id").merge(bio, on="antibody_id", how="left")
        Xbio = bio.select_dtypes(include=[np.number]).values.astype(float)
        Xplm = tplm.load_embedding_matrix(B1_CACHE / "plm" / "manifest_esm2_t33_650M_UR50D.csv", ids)
        abb_p = CACHE / "structure_features" / "abb_sasa_rasa_patch.csv"
        esm_p = CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv"
        Xabb = load_csv(abb_p, ids, include=["sasa", "rasa", "patch", "BSA"]) if abb_p.exists() else None
        Xesm = load_csv(esm_p, ids, include=["sasa", "rasa", "patch", "BSA"]) if esm_p.exists() else None

        # baselines
        oof_s = oof_ridge(Xsimple[dev], y[dev], groups[dev])
        oof_b = oof_ridge(Xbio[dev], y[dev], groups[dev])
        resid_s = y[dev] - oof_s
        resid_b = y[dev] - oof_b

        for name, X in [("PLM_ESM2", Xplm), ("ABB", Xabb), ("ESMFN", Xesm)]:
            if X is None:
                continue
            pred_rs = oof_ridge(X[dev], resid_s, groups[dev])
            pred_rb = oof_ridge(X[dev], resid_b, groups[dev])
            sp_s, pe_s = metrics(resid_s, pred_rs)
            sp_b, pe_b = metrics(resid_b, pred_rb)
            rows.append(
                {
                    "target": target,
                    "split": "canonical",
                    "baseline": "SEQ_SIMPLE",
                    "residual_model": name,
                    "residual_spearman": sp_s,
                    "residual_pearson": pe_s,
                }
            )
            rows.append(
                {
                    "target": target,
                    "split": "canonical",
                    "baseline": "BIO_SHORTCUT",
                    "residual_model": name,
                    "residual_spearman": sp_b,
                    "residual_pearson": pe_b,
                }
            )

        # Ensembles from Dev-OOF of several families (Public/Private never used for weights)
        oofs = {}
        for name, X in [
            ("SEQ_SIMPLE", Xsimple),
            ("BIO", Xbio),
            ("PLM", Xplm),
            ("ABB", Xabb),
            ("ESMFN", Xesm),
        ]:
            if X is None:
                continue
            oofs[name] = oof_ridge(X[dev], y[dev], groups[dev])
            # full-dev fit for holdout roles
            Xf = sanitize_X(X, fit_mask=dev)
            model = Ridge(alpha=10.0).fit(Xf[dev], y[dev])
            oofs[name + "_fullpred"] = model.predict(Xf)

        # rank average on Public/Private using fullpred
        keys = [k for k in oofs if k.endswith("_fullpred")]
        if len(keys) >= 2:
            stack = np.vstack([rankdata(oofs[k]) for k in keys])
            rank_mean = stack.mean(axis=0)
            for role in ["Public", "Private"]:
                mask = roles == role
                sp, _ = metrics(y[mask], rank_mean[mask])
                rows.append(
                    {
                        "target": target,
                        "split": "canonical",
                        "baseline": "ENSEMBLE",
                        "residual_model": f"rank_mean/{role}",
                        "residual_spearman": sp,
                        "residual_pearson": np.nan,
                    }
                )
            # linear blend weights from Dev OOF only
            names = [k for k in oofs if not k.endswith("_fullpred")]
            A = np.vstack([oofs[n] for n in names]).T
            # fit nonneg-ish ridge on OOF
            w = Ridge(alpha=1.0, fit_intercept=False).fit(A, y[dev]).coef_
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            blend = sum(w[i] * oofs[names[i] + "_fullpred"] for i in range(len(names)))
            for role in ["Public", "Private", "Dev"]:
                mask = roles == role if role != "Dev" else dev
                sp, pe = metrics(y[mask], blend[mask])
                rows.append(
                    {
                        "target": target,
                        "split": "canonical",
                        "baseline": "ENSEMBLE",
                        "residual_model": f"oof_linear_blend/{role}",
                        "residual_spearman": sp,
                        "residual_pearson": pe,
                        "weights": str(dict(zip(names, map(float, w)))),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "residual_ensemble_results.csv", index=False)
    # append ensemble rows into all_results if present
    allp = METRICS / "all_results.csv"
    if allp.exists():
        base = pd.read_csv(allp)
        ens = out[out.baseline == "ENSEMBLE"].copy()
        add = []
        for _, r in ens.iterrows():
            role = str(r.residual_model).split("/")[-1]
            if role not in ("Public", "Private"):
                continue
            # store as synthetic representation
            add.append(
                {
                    "representation": r.residual_model.split("/")[0],
                    "model": "ensemble",
                    "target": r.target,
                    "split": "canonical",
                    "cv_spearman": np.nan,
                    "public_spearman": r.residual_spearman if role == "Public" else np.nan,
                    "private_spearman": r.residual_spearman if role == "Private" else np.nan,
                    "feature_class": "PARTICIPANT_LEGAL",
                }
            )
        if add:
            pd.concat([base, pd.DataFrame(add)], ignore_index=True).to_csv(allp, index=False)
    print("RESIDUAL_ENSEMBLE_OK", len(out))


if __name__ == "__main__":
    main()
