#!/usr/bin/env python3
"""Shared utilities for Gate B4 absolute-value rebuild."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b4_absolute"
B1 = ROOT / "gate_b1"
B2 = ROOT / "gate_b2"
B3 = ROOT / "gate_b3"

CONFIG = GATE / "config"
CACHE = GATE / "cache"
FEATURES = GATE / "features"
OPTUNA_DIR = GATE / "optuna"
MODELS = GATE / "models"
PREDS = GATE / "predictions"
METRICS = GATE / "metrics"
PLOTS = GATE / "plots"
SCRIPTS = GATE / "scripts"
REPORTS = GATE / "reports"
LOGS = GATE / "logs"

B1_CACHE = B1 / "cache"
B1_DATA = B1 / "data"
B2_CACHE = B2 / "cache"
B3_ORG = B3 / "frozen" / "organizer"
B3_FEATURES = B3 / "features"
B3_CONFIG = B3 / "config"

# GBDT protocol (NON-NEGOTIABLE)
GBDT_LEARNING_RATE = 0.03
GBDT_N_ESTIMATORS = 5000
GBDT_EARLY_STOPPING_ROUNDS = 150
GBDT_ES_GROUP_FRAC = 0.18

MASTER_SEED = 20260829
OPTUNA_SAMPLER_SEED = 20260829
CV_SEED = 20260829

sys.path.insert(0, str(B1 / "scripts"))


def ensure_dirs() -> None:
    for p in [
        CONFIG, CACHE, FEATURES, OPTUNA_DIR, MODELS, PREDS, METRICS, PLOTS, SCRIPTS, REPORTS, LOGS,
        PREDS / "oof", PREDS / "final", CACHE / "features",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def set_gpu0() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(ids) -> str:
    blob = "\n".join(sorted(map(str, ids))) + "\n"
    return hashlib.sha256(blob.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def set_seeds(seed: int = MASTER_SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def sanitize(X: np.ndarray) -> np.ndarray:
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    keep = np.any(np.isfinite(X), axis=0)
    X = X[:, keep] if keep.any() else np.zeros((X.shape[0], 1), float)
    col = np.nanmean(X, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    X = np.array(X, copy=True)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col, inds[1])
    X[~np.isfinite(X)] = 0.0
    return X


def sanitize_pair(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)
    keep = np.any(np.isfinite(Xtr), axis=0)
    Xtr = Xtr[:, keep] if keep.any() else np.zeros((Xtr.shape[0], 1), float)
    Xte = Xte[:, keep] if keep.any() else np.zeros((Xte.shape[0], 1), float)
    col = np.nanmean(Xtr, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    for X in (Xtr, Xte):
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col, inds[1])
        X[~np.isfinite(X)] = 0.0
    return Xtr, Xte


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n = int(len(y))
    out = {
        "n": n,
        "mae": np.nan,
        "rmse": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
        "r2": np.nan,
        "cal_slope": np.nan,
        "cal_intercept": np.nan,
        "pred_mean": np.nan,
        "pred_sd": np.nan,
        "obs_mean": np.nan,
        "obs_sd": np.nan,
    }
    if n < 3:
        return out
    resid = y - p
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pr = pearsonr(y, p)[0]
    sp = spearmanr(y, p).correlation
    # observed = intercept + slope * predicted
    if np.std(p) > 1e-12:
        slope, intercept = np.polyfit(p, y, 1)
    else:
        slope, intercept = np.nan, float(np.mean(y))
    out.update(
        {
            "mae": mae,
            "rmse": rmse,
            "pearson": float(pr) if pr is not None and np.isfinite(pr) else np.nan,
            "spearman": float(sp) if sp is not None and np.isfinite(sp) else np.nan,
            "r2": r2,
            "cal_slope": float(slope) if np.isfinite(slope) else np.nan,
            "cal_intercept": float(intercept) if np.isfinite(intercept) else np.nan,
            "pred_mean": float(np.mean(p)),
            "pred_sd": float(np.std(p, ddof=1)) if n > 1 else 0.0,
            "obs_mean": float(np.mean(y)),
            "obs_sd": float(np.std(y, ddof=1)) if n > 1 else 0.0,
        }
    )
    return out


def load_train_frame(include_holdout_labels: bool = False) -> pd.DataFrame:
    """Load population + role. Holdout labels stripped unless explicitly allowed."""
    pop = pd.read_csv(B3_ORG / "final_population.csv")
    role = pd.read_csv(B3_ORG / "role_map.csv")[["id", "role"]]
    df = pop.merge(role, on="id", how="inner")
    assert len(df) == 324
    if not include_holdout_labels:
        # Barrier: zero-out Public/Private assay labels so accidental use fails loudly
        mask = df["role"] != "Train"
        df = df.copy()
        df.loc[mask, "HIC"] = np.nan
        df.loc[mask, "TmApp"] = np.nan
    return df


def train_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["role"] == "Train"].copy().reset_index(drop=True)


def load_csv_num(path: Path, ids, prefixes=None, include=None, id_col="antibody_id") -> np.ndarray:
    df = pd.read_csv(path)
    if id_col not in df.columns and "id" in df.columns:
        id_col = "id"
    df = pd.DataFrame({"id": list(ids)}).merge(df, left_on="id", right_on=id_col, how="left")
    cols = [
        c
        for c in df.columns
        if c not in ("id", id_col, "antibody_id") and pd.api.types.is_numeric_dtype(df[c])
    ]
    if prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    if include:
        cols = [c for c in cols if any(s in c for s in include)]
    if not cols:
        return np.zeros((len(ids), 1), float)
    return sanitize(df[cols].values.astype(float))


def load_bio(ids, category_ids=None) -> np.ndarray:
    """BIO features. If category_ids given, freeze one-hot columns from that set (usually Train)."""
    bio = pd.read_csv(B1_CACHE / "features" / "stage_C_shortcut.csv")
    cats = [c for c in ["C_vh_family", "C_vl_family", "C_kappa_lambda"] if c in bio.columns]
    query = pd.DataFrame({"id": list(ids)}).merge(bio, left_on="id", right_on="antibody_id", how="left")
    bio_num = query.select_dtypes(include=[np.number]).values.astype(float)
    if not cats:
        return sanitize(bio_num)
    ref_ids = list(category_ids) if category_ids is not None else list(ids)
    ref = pd.DataFrame({"id": ref_ids}).merge(bio, left_on="id", right_on="antibody_id", how="left")
    dref = pd.get_dummies(ref[cats].astype(str), dummy_na=True)
    dqu = pd.get_dummies(query[cats].astype(str), dummy_na=True).reindex(columns=dref.columns, fill_value=0)
    return sanitize(np.concatenate([sanitize(bio_num), dqu.values.astype(float)], axis=1))


def load_plm(name: str, ids) -> np.ndarray:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tplm", str(B1 / "scripts" / "10_train_plm_structure.py")
    )
    tplm = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tplm)
    man = {
        "ABLANG2": "manifest_ablang2_default.csv",
        "ESM1B": "manifest_esm1b_t33_650M_UR50S.csv",
        "ESM2": "manifest_esm2_t33_650M_UR50D.csv",
        "ESM2_CDR6": "manifest_esm2_t33_650M_UR50D_CDR6.csv",
    }[name]
    X = tplm.load_embedding_matrix(B1_CACHE / "plm" / man, pd.Series(list(ids)))
    return sanitize(X)


def load_imgt(ids, which: str = "Xhl") -> np.ndarray:
    z = np.load(B3_FEATURES / "imgt" / "positional_onehot.npz", allow_pickle=True)
    imgt_ids = list(z["ids"])
    idx = {i: k for k, i in enumerate(imgt_ids)}
    rows = [idx[i] for i in ids]
    X = z[which]
    if sparse.issparse(X):
        X = X.toarray()
    return sanitize(np.asarray(X)[rows])


def load_representation(tag: str, ids) -> np.ndarray:
    ids = list(ids)
    if tag == "SEQ_SIMPLE":
        return load_csv_num(B1_CACHE / "features" / "stage_A_simple.csv", ids, prefixes=["A0_", "A1_", "A2_"])
    if tag == "SEQ_CDR":
        return load_csv_num(B1_CACHE / "features" / "stage_B_cdr.csv", ids, prefixes=["B_"])
    if tag == "BIO":
        return load_bio(ids)
    if tag == "GERMLINE_REL":
        return load_csv_num(B3_FEATURES / "germline" / "germline_relative.csv", ids, id_col="id")
    if tag == "IMGT_POS_HL":
        return load_imgt(ids, "Xhl")
    if tag == "IMGT_POS_H":
        return load_imgt(ids, "Xh")
    if tag == "IMGT_POS_L":
        return load_imgt(ids, "Xl")
    if tag == "PLM_ABLANG2":
        return load_plm("ABLANG2", ids)
    if tag == "PLM_ESM1B":
        return load_plm("ESM1B", ids)
    if tag == "PLM_ESM2":
        return load_plm("ESM2", ids)
    if tag == "PLM_ESM2_CDR6":
        return load_plm("ESM2_CDR6", ids)
    if tag == "ABB_STRUCTURE":
        return load_csv_num(
            B3_FEATURES / "structure_ext" / "structure_extended.csv", ids, prefixes=["ABB_"], id_col="id"
        )
    if tag == "ESMFN_STRUCTURE":
        return load_csv_num(
            B3_FEATURES / "structure_ext" / "structure_extended.csv",
            ids,
            prefixes=["ESMFN_", "CONSENSUS_", "COM_"],
            id_col="id",
        )
    if tag == "FUSION_ESM2_ESMFN":
        return np.concatenate(
            [load_representation("PLM_ESM2", ids), load_representation("ESMFN_STRUCTURE", ids)], axis=1
        )
    if tag == "FUSION_ESM2_SEQCDR_ESMFN":
        return np.concatenate(
            [
                load_representation("PLM_ESM2", ids),
                load_representation("SEQ_CDR", ids),
                load_representation("ESMFN_STRUCTURE", ids),
            ],
            axis=1,
        )
    if tag == "FUSION_ABLANG2_BIO":
        return np.concatenate(
            [load_representation("PLM_ABLANG2", ids), load_representation("BIO", ids)], axis=1
        )
    if tag == "FUSION_ABLANG2_IMGT":
        return np.concatenate(
            [load_representation("PLM_ABLANG2", ids), load_representation("IMGT_POS_HL", ids)], axis=1
        )
    if tag == "FUSION_BIO_IMGT":
        return np.concatenate(
            [load_representation("BIO", ids), load_representation("IMGT_POS_HL", ids)], axis=1
        )
    if tag.startswith("NGRAM_"):
        n = int(tag.split("_")[-1].replace("mer", ""))
        X = sparse.load_npz(B3_FEATURES / "ngram" / f"tfidf_HL_{n}mer.npz")
        ng_ids = list(np.load(B3_FEATURES / "ngram" / "ids.npy", allow_pickle=True))
        idx = {i: k for k, i in enumerate(ng_ids)}
        rows = [idx[i] for i in ids]
        return sanitize(X[rows].toarray())
    raise KeyError(f"Unknown representation: {tag}")


def group_kfold_labels(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign fold ids to samples without splitting groups; balance by group count."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    # greedy balance by sample count
    fold_sizes = np.zeros(n_folds, dtype=int)
    g2f = {}
    # larger groups first
    sizes = {g: int((groups == g).sum()) for g in uniq}
    for g in sorted(uniq, key=lambda x: -sizes[x]):
        f = int(np.argmin(fold_sizes))
        g2f[g] = f
        fold_sizes[f] += sizes[g]
    return np.array([g2f[g] for g in groups], dtype=int)


def make_outer_cv(train_df: pd.DataFrame, n_folds: int = 5, n_repeats: int = 3, seed: int = CV_SEED) -> dict:
    train_ids = list(train_df["id"])
    groups = train_df["sequence_group"].values
    folds = []
    for rep in range(n_repeats):
        fold_id = group_kfold_labels(groups, n_folds, seed=seed + 1000 * rep + 17)
        folds.append({"repeat": rep, "fold_id": fold_id.tolist()})
    return {
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "seed": seed,
        "train_ids": train_ids,
        "protocol": "5-fold grouped × 3 repeats; groups never split; greedy size balance",
        "folds": folds,
    }


def es_split_by_group(groups: np.ndarray, seed: int, frac: float = GBDT_ES_GROUP_FRAC):
    """Return boolean mask: True = early-stop set (~frac of groups)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    n_es = max(1, int(round(len(uniq) * frac)))
    es_groups = set(uniq[:n_es].tolist())
    return np.array([g in es_groups for g in groups], dtype=bool)


def software_versions() -> dict:
    import sklearn
    import xgboost

    out = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    try:
        import lightgbm

        out["lightgbm"] = lightgbm.__version__
    except Exception:
        out["lightgbm"] = None
    try:
        import catboost

        out["catboost"] = catboost.__version__
    except Exception:
        out["catboost"] = None
    try:
        import optuna

        out["optuna"] = optuna.__version__
    except Exception:
        out["optuna"] = None
    return out
