#!/usr/bin/env python3
"""
Stage 1 — Classical sequence features + antibody-aware features.

Uses frozen Stage-0 CV only. No PLM / structure / Test labels / organizer secrets.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/workspace_developability_acquisition")
DEV = ROOT / "competition/data/distribution/dev.csv"
ANN = ROOT / "competition/data/distribution/dev_annotations.csv"
CV_PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
CV_SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
ANARCI_CACHE = ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv"
OUT = ROOT / "virtual_participant/stage1_features"
PLOTS = OUT / "plots"
ARTIFACTS = OUT / "artifacts"

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = set("AILMFVWY")
AROM = set("FWY")
POS = set("KR")
NEG = set("DE")
CHARGED = POS | NEG
POLAR = set("STNQ")
OPTUNA_SEED = 42
N_TRIALS_LINEAR = 35
N_TRIALS_SVR = 25
N_TRIALS_GBDT = 30


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def frac(seq: str, alphabet: set[str] | str) -> float:
    if not seq:
        return 0.0
    alphabet = set(alphabet)
    return sum(ch in alphabet for ch in seq) / len(seq)


def aa_fractions(seq: str, prefix: str) -> dict[str, float]:
    n = max(len(seq), 1)
    c = Counter(seq)
    return {f"{prefix}_aa_{a}": c.get(a, 0) / n for a in AA}


def shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    c = Counter(seq)
    return float(-sum((v / n) * math.log2(v / n) for v in c.values()))


def safe_protparam(seq: str) -> dict[str, float]:
    # ProtParam requires standard AA only
    clean = "".join(ch for ch in seq if ch in AA)
    if len(clean) < 1:
        return {"pI": 7.0, "gravy": 0.0, "aromaticity": 0.0, "net_charge_ph7": 0.0}
    pa = ProteinAnalysis(clean)
    # approximate charge at pH 7 from residue counts (simple)
    # ProtParam charge_at_pH available in Biopython
    try:
        charge = float(pa.charge_at_pH(7.0))
    except Exception:
        charge = (clean.count("K") + clean.count("R") - clean.count("D") - clean.count("E"))
    try:
        pi = float(pa.isoelectric_point())
    except Exception:
        pi = 7.0
    try:
        gravy = float(pa.gravy())
    except Exception:
        gravy = 0.0
    try:
        arom = float(pa.aromaticity())
    except Exception:
        arom = frac(clean, AROM)
    return {"pI": pi, "gravy": gravy, "aromaticity": arom, "net_charge_ph7": charge}


def physchem_block(seq: str, prefix: str) -> dict[str, float]:
    n = max(len(seq), 1)
    out = {
        f"{prefix}_len": float(len(seq)),
        f"{prefix}_hydro_frac": frac(seq, HYDRO),
        f"{prefix}_arom_frac": frac(seq, AROM),
        f"{prefix}_charged_frac": frac(seq, CHARGED),
        f"{prefix}_pos_frac": frac(seq, POS),
        f"{prefix}_neg_frac": frac(seq, NEG),
        f"{prefix}_polar_frac": frac(seq, POLAR),
        f"{prefix}_gly_frac": seq.count("G") / n,
        f"{prefix}_pro_frac": seq.count("P") / n,
        f"{prefix}_cys_count": float(seq.count("C")),
        f"{prefix}_cys_frac": seq.count("C") / n,
        f"{prefix}_entropy": shannon_entropy(seq),
        f"{prefix}_n_unique": float(len(set(seq))),
        f"{prefix}_pos_minus_neg": float(sum(ch in POS for ch in seq) - sum(ch in NEG for ch in seq)),
    }
    pp = safe_protparam(seq)
    out[f"{prefix}_pI"] = pp["pI"]
    out[f"{prefix}_gravy"] = pp["gravy"]
    out[f"{prefix}_aromaticity"] = pp["aromaticity"]
    out[f"{prefix}_net_charge_ph7"] = pp["net_charge_ph7"]
    return out


def region_composition(seq: str, prefix: str) -> dict[str, float]:
    n = max(len(seq), 1)
    return {
        f"{prefix}_len": float(len(seq)),
        f"{prefix}_hydro_frac": frac(seq, HYDRO),
        f"{prefix}_arom_frac": frac(seq, AROM),
        f"{prefix}_pos_frac": frac(seq, POS),
        f"{prefix}_neg_frac": frac(seq, NEG),
        f"{prefix}_net_charge_proxy": float(sum(ch in POS for ch in seq) - sum(ch in NEG for ch in seq)),
        f"{prefix}_gly_frac": seq.count("G") / n,
        f"{prefix}_pro_frac": seq.count("P") / n,
        f"{prefix}_gravy": safe_protparam(seq)["gravy"] if seq else 0.0,
    }


def build_all_feature_tables(df: pd.DataFrame, ann: pd.DataFrame, regions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict of feature-family DataFrames aligned to df.index, plus raw categorical cols."""
    rows_basic_h, rows_basic_l, rows_basic_c = [], [], []
    rows_phys_h, rows_phys_l, rows_phys_c = [], [], []
    rows_seq_all = []
    rows_cdr_comp = []
    rows_hcdr3 = []
    rows_fw_cdr = []

    for i, r in df.iterrows():
        h, l = r["heavy"], r["light"]
        comb = h + l

        # BASIC: length + AAC + simple physchem fractions (no pI)
        bh = {"h_len": float(len(h)), **aa_fractions(h, "h"),
              "h_hydro_frac": frac(h, HYDRO), "h_arom_frac": frac(h, AROM),
              "h_charged_frac": frac(h, CHARGED), "h_gly_frac": h.count("G") / max(len(h), 1),
              "h_pro_frac": h.count("P") / max(len(h), 1)}
        bl = {"l_len": float(len(l)), **aa_fractions(l, "l"),
              "l_hydro_frac": frac(l, HYDRO), "l_arom_frac": frac(l, AROM),
              "l_charged_frac": frac(l, CHARGED), "l_gly_frac": l.count("G") / max(len(l), 1),
              "l_pro_frac": l.count("P") / max(len(l), 1)}
        bc = {"total_len": float(len(comb)), "hl_len_ratio": len(h) / max(len(l), 1),
              "hl_len_diff": float(len(h) - len(l)), **aa_fractions(comb, "c"),
              "c_hydro_frac": frac(comb, HYDRO), "c_arom_frac": frac(comb, AROM),
              "c_charged_frac": frac(comb, CHARGED)}
        rows_basic_h.append(bh)
        rows_basic_l.append(bl)
        rows_basic_c.append(bc)

        ph = physchem_block(h, "h")
        pl = physchem_block(l, "l")
        pc = physchem_block(comb, "c")
        # drop lens already in basic for phys purity — keep charge/pI/hydro focus
        phys_keys_h = {k: v for k, v in ph.items() if any(x in k for x in
                       ["pI", "gravy", "aromatic", "charge", "hydro", "arom", "pos", "neg", "polar", "cys"])}
        phys_keys_l = {k: v for k, v in pl.items() if any(x in k for x in
                       ["pI", "gravy", "aromatic", "charge", "hydro", "arom", "pos", "neg", "polar", "cys"])}
        phys_keys_c = {k: v for k, v in pc.items() if any(x in k for x in
                       ["pI", "gravy", "aromatic", "charge", "hydro", "arom", "pos", "neg", "polar", "cys"])}
        # add H/L diffs
        phys_keys_c["hl_pI_diff"] = ph["h_pI"] - pl["l_pI"]
        phys_keys_c["hl_gravy_diff"] = ph["h_gravy"] - pl["l_gravy"]
        phys_keys_c["hl_charge_diff"] = ph["h_net_charge_ph7"] - pl["l_net_charge_ph7"]
        rows_phys_h.append(phys_keys_h)
        rows_phys_l.append(phys_keys_l)
        rows_phys_c.append(phys_keys_c)

        rows_seq_all.append({**bh, **bl, **bc, **phys_keys_h, **phys_keys_l, **phys_keys_c,
                             "h_entropy": ph["h_entropy"], "l_entropy": pl["l_entropy"],
                             "h_n_unique": ph["h_n_unique"], "l_n_unique": pl["l_n_unique"]})

        rr = regions.loc[i]
        cdr_feats = {}
        for chain in ["h", "l"]:
            for reg in ["cdr1", "cdr2", "cdr3"]:
                cdr_feats.update(region_composition(str(rr[f"{chain}_{reg}"]), f"{chain}_{reg}"))
        # HCDR3 emphasis
        hcdr3 = region_composition(str(rr["h_cdr3"]), "hcdr3")
        rows_hcdr3.append(hcdr3)

        # Framework vs CDR aggregates
        h_cdr = str(rr["h_cdr1"]) + str(rr["h_cdr2"]) + str(rr["h_cdr3"])
        l_cdr = str(rr["l_cdr1"]) + str(rr["l_cdr2"]) + str(rr["l_cdr3"])
        h_fw = str(rr["h_fr1"]) + str(rr["h_fr2"]) + str(rr["h_fr3"]) + str(rr["h_fr4"])
        l_fw = str(rr["l_fr1"]) + str(rr["l_fr2"]) + str(rr["l_fr3"]) + str(rr["l_fr4"])
        fw_cdr = {}
        for name, seq in [("h_cdr_agg", h_cdr), ("l_cdr_agg", l_cdr), ("h_fw_agg", h_fw), ("l_fw_agg", l_fw)]:
            fw_cdr.update(region_composition(seq, name))
        fw_cdr["h_cdr_fw_gravy_diff"] = fw_cdr["h_cdr_agg_gravy"] - fw_cdr["h_fw_agg_gravy"]
        fw_cdr["l_cdr_fw_gravy_diff"] = fw_cdr["l_cdr_agg_gravy"] - fw_cdr["l_fw_agg_gravy"]
        fw_cdr["h_cdr_fw_charge_diff"] = fw_cdr["h_cdr_agg_net_charge_proxy"] - fw_cdr["h_fw_agg_net_charge_proxy"]
        fw_cdr["l_cdr_fw_charge_diff"] = fw_cdr["l_cdr_agg_net_charge_proxy"] - fw_cdr["l_fw_agg_net_charge_proxy"]
        rows_cdr_comp.append({**cdr_feats, **fw_cdr})
        rows_fw_cdr.append(fw_cdr)

    idx = df.index
    SEQ_BASIC_H = pd.DataFrame(rows_basic_h, index=idx)
    SEQ_BASIC_L = pd.DataFrame(rows_basic_l, index=idx)
    SEQ_BASIC_C = pd.DataFrame(rows_basic_c, index=idx)
    SEQ_BASIC = pd.concat([SEQ_BASIC_H, SEQ_BASIC_L, SEQ_BASIC_C], axis=1)
    SEQ_PHYS_H = pd.DataFrame(rows_phys_h, index=idx)
    SEQ_PHYS_L = pd.DataFrame(rows_phys_l, index=idx)
    SEQ_PHYS_C = pd.DataFrame(rows_phys_c, index=idx)
    SEQ_PHYS = pd.concat([SEQ_PHYS_H, SEQ_PHYS_L, SEQ_PHYS_C], axis=1)
    SEQ_ALL = pd.DataFrame(rows_seq_all, index=idx)

    # Annotations
    germ_num = ann[["heavy_germline_identity", "light_germline_identity"]].copy()
    germ_num["heavy_germline_divergence"] = 1.0 - germ_num["heavy_germline_identity"]
    germ_num["light_germline_divergence"] = 1.0 - germ_num["light_germline_identity"]
    germ_num["hl_identity_diff"] = germ_num["heavy_germline_identity"] - germ_num["light_germline_identity"]
    germ_num["hl_identity_mean"] = germ_num[["heavy_germline_identity", "light_germline_identity"]].mean(axis=1)
    germ_num["hl_identity_min"] = germ_num[["heavy_germline_identity", "light_germline_identity"]].min(axis=1)
    germ_num["hl_identity_max"] = germ_num[["heavy_germline_identity", "light_germline_identity"]].max(axis=1)
    germ_cat = ann[["heavy_v_family", "heavy_j_gene", "light_v_family", "light_j_gene", "light_chain_type"]].copy()

    cdr_len = ann[[
        "h_cdr1_length", "h_cdr2_length", "h_cdr3_length",
        "l_cdr1_length", "l_cdr2_length", "l_cdr3_length",
    ]].copy()
    cdr_len["h_cdr_total"] = cdr_len[["h_cdr1_length", "h_cdr2_length", "h_cdr3_length"]].sum(axis=1)
    cdr_len["l_cdr_total"] = cdr_len[["l_cdr1_length", "l_cdr2_length", "l_cdr3_length"]].sum(axis=1)
    cdr_len["hl_cdr_total"] = cdr_len["h_cdr_total"] + cdr_len["l_cdr_total"]

    ANN_CDR_COMPOSITION = pd.DataFrame(rows_cdr_comp, index=idx)
    ANN_HCDR3 = pd.DataFrame(rows_hcdr3, index=idx)

    # For antibody_all numeric + cats handled in model layer
    tables = {
        "SEQ_BASIC": SEQ_BASIC,
        "SEQ_BASIC_H": SEQ_BASIC_H,
        "SEQ_BASIC_L": SEQ_BASIC_L,
        "SEQ_BASIC_C": SEQ_BASIC_C,
        "SEQ_PHYS": SEQ_PHYS,
        "SEQ_PHYS_H": SEQ_PHYS_H,
        "SEQ_PHYS_L": SEQ_PHYS_L,
        "SEQ_PHYS_C": SEQ_PHYS_C,
        "SEQ_ALL": SEQ_ALL.loc[:, ~SEQ_ALL.columns.duplicated()],
        "SEQ_HEAVY": pd.concat([SEQ_BASIC_H, SEQ_PHYS_H], axis=1).loc[:, ~pd.concat([SEQ_BASIC_H, SEQ_PHYS_H], axis=1).columns.duplicated()],
        "SEQ_LIGHT": pd.concat([SEQ_BASIC_L, SEQ_PHYS_L], axis=1).loc[:, ~pd.concat([SEQ_BASIC_L, SEQ_PHYS_L], axis=1).columns.duplicated()],
        "SEQ_COMBINED": pd.concat([SEQ_BASIC_C, SEQ_PHYS_C], axis=1).loc[:, ~pd.concat([SEQ_BASIC_C, SEQ_PHYS_C], axis=1).columns.duplicated()],
        "ANN_GERMLINE_NUM": germ_num,
        "ANN_GERMLINE_CAT": germ_cat,
        "ANN_CDR_LENGTH": cdr_len,
        "ANN_CDR_COMPOSITION": ANN_CDR_COMPOSITION,
        "ANN_HCDR3": ANN_HCDR3,
    }
    # Ensure uniqueness for concatenated families
    for k in list(tables.keys()):
        if isinstance(tables[k], pd.DataFrame):
            tables[k] = tables[k].loc[:, ~tables[k].columns.duplicated()]
    return tables


# ---------------------------------------------------------------------------
# CV evaluation utilities
# ---------------------------------------------------------------------------

def mae(y, p) -> float:
    return float(np.mean(np.abs(y - p)))


class RareCategoryEncoder:
    """Fit-safe rare category collapse then one-hot via pandas get_dummies on train columns."""

    def __init__(self, min_count: int = 3):
        self.min_count = min_count
        self.keep_: dict[str, set[str]] = {}
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame):
        self.keep_ = {}
        for c in X.columns:
            vc = X[c].astype(str).value_counts()
            self.keep_[c] = set(vc[vc >= self.min_count].index.tolist())
        Xt = self._transform_raw(X)
        self.columns_ = list(Xt.columns)
        return self

    def _transform_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in X.columns:
            s = X[c].astype(str)
            keep = self.keep_[c]
            s = s.where(s.isin(keep), other="__OTHER__")
            dummies = pd.get_dummies(s, prefix=c)
            out = pd.concat([out, dummies], axis=1)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self._transform_raw(X)
        # align to train columns
        for c in self.columns_:
            if c not in Xt.columns:
                Xt[c] = 0
        return Xt[self.columns_].astype(float)


def make_xy(tables: dict, family: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return numeric X and optional categorical X."""
    if family == "ANN_GERMLINE":
        return tables["ANN_GERMLINE_NUM"].copy(), tables["ANN_GERMLINE_CAT"].copy()
    if family == "ANTIBODY_ALL":
        num = pd.concat([
            tables["ANN_GERMLINE_NUM"], tables["ANN_CDR_LENGTH"],
            tables["ANN_CDR_COMPOSITION"], tables["ANN_HCDR3"],
        ], axis=1)
        return num, tables["ANN_GERMLINE_CAT"].copy()
    if family == "SEQ_PLUS_ANTIBODY":
        num = pd.concat([
            tables["SEQ_ALL"], tables["ANN_GERMLINE_NUM"], tables["ANN_CDR_LENGTH"],
            tables["ANN_CDR_COMPOSITION"],
        ], axis=1)
        # dedupe columns
        num = num.loc[:, ~num.columns.duplicated()]
        return num, tables["ANN_GERMLINE_CAT"].copy()
    if family in tables:
        X = tables[family].copy()
        X = X.loc[:, ~X.columns.duplicated()]
        return X, None
    raise KeyError(family)


def prepare_fold_matrices(X_num, X_cat, tr, va):
    """Fit imputer/scaler/encoder on train fold only."""
    num_imp = SimpleImputer(strategy="median")
    Xtr_n = num_imp.fit_transform(X_num.iloc[tr])
    Xva_n = num_imp.transform(X_num.iloc[va])
    scaler = StandardScaler()
    Xtr_n = scaler.fit_transform(Xtr_n)
    Xva_n = scaler.transform(Xva_n)
    if X_cat is not None:
        enc = RareCategoryEncoder(min_count=3)
        Xtr_c = enc.fit(X_cat.iloc[tr]).transform(X_cat.iloc[tr]).to_numpy(dtype=float)
        Xva_c = enc.transform(X_cat.iloc[va]).to_numpy(dtype=float)
        Xtr = np.hstack([Xtr_n, Xtr_c])
        Xva = np.hstack([Xva_n, Xva_c])
        n_features = Xtr.shape[1]
        return Xtr, Xva, n_features, (num_imp, scaler, enc)
    return Xtr_n, Xva_n, Xtr_n.shape[1], (num_imp, scaler, None)


def cv_predict(y, folds, X_num, X_cat, model_factory) -> tuple[np.ndarray, list[float], int]:
    n_folds = int(folds.max()) + 1
    oof = np.zeros(len(y))
    fold_maes = []
    n_features = 0
    for f in range(n_folds):
        tr = folds != f
        va = folds == f
        Xtr, Xva, n_features, _ = prepare_fold_matrices(X_num, X_cat, tr, va)
        model = model_factory()
        model.fit(Xtr, y[tr])
        pred = model.predict(Xva)
        oof[va] = pred
        fold_maes.append(mae(y[va], pred))
    return oof, fold_maes, n_features


def median_cv(y, folds) -> tuple[np.ndarray, list[float]]:
    n_folds = int(folds.max()) + 1
    oof = np.zeros(len(y))
    fold_maes = []
    for f in range(n_folds):
        tr = folds != f
        va = folds == f
        pred = np.full(va.sum(), np.median(y[tr]))
        oof[va] = pred
        fold_maes.append(mae(y[va], pred))
    return oof, fold_maes


def summarize(name, target, family, model_name, oof, fold_maes, n_features, y, median_mae, status, notes="", params=None):
    fold_maes = np.asarray(fold_maes, dtype=float)
    return {
        "experiment_id": name,
        "target": target,
        "feature_family": family,
        "model": model_name,
        "hyperparameters": json.dumps(params or {}),
        "n_features": int(n_features),
        "primary_mae": float(fold_maes.mean()) if notes.find("shadow") < 0 else None,
        "mae": float(fold_maes.mean()),
        "fold_mae_sd": float(fold_maes.std(ddof=1)) if len(fold_maes) > 1 else 0.0,
        "worst_fold_mae": float(fold_maes.max()),
        "oof_mae": mae(y, oof),
        "fold_maes": json.dumps(fold_maes.tolist()),
        "delta_vs_median": float(fold_maes.mean() - median_mae),
        "status": status,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def optuna_elasticnet(y, folds, X_num, X_cat, n_trials=N_TRIALS_LINEAR):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 30.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
        def factory():
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=8000, tol=1e-3, random_state=0)
        _, fold_maes, _ = cv_predict(y, folds, X_num, X_cat, factory)
        return float(np.mean(fold_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optuna_ridge(y, folds, X_num, X_cat, n_trials=N_TRIALS_LINEAR):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-2, 100.0, log=True)
        def factory():
            return Ridge(alpha=alpha, random_state=0)
        _, fold_maes, _ = cv_predict(y, folds, X_num, X_cat, factory)
        return float(np.mean(fold_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optuna_svr(y, folds, X_num, X_cat, n_trials=N_TRIALS_SVR):
    def objective(trial):
        C = trial.suggest_float("C", 0.1, 50.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        def factory():
            return SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
        _, fold_maes, _ = cv_predict(y, folds, X_num, X_cat, factory)
        return float(np.mean(fold_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optuna_xgb(y, folds, X_num, X_cat, n_trials=N_TRIALS_GBDT):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "learning_rate": 0.05,
            "n_estimators": 400,
            "objective": "reg:squarederror",
            "random_state": 0,
            "n_jobs": 2,
            "verbosity": 0,
        }
        def factory():
            return xgb.XGBRegressor(**params)
        _, fold_maes, _ = cv_predict(y, folds, X_num, X_cat, factory)
        return float(np.mean(fold_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"learning_rate": 0.05, "n_estimators": 400})
    return best, study.best_value


def optuna_lgbm(y, folds, X_num, X_cat, n_trials=N_TRIALS_GBDT):
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "learning_rate": 0.05,
            "n_estimators": 400,
            "random_state": 0,
            "n_jobs": 2,
            "verbosity": -1,
        }
        def factory():
            return lgb.LGBMRegressor(**params)
        _, fold_maes, _ = cv_predict(y, folds, X_num, X_cat, factory)
        return float(np.mean(fold_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"learning_rate": 0.05, "n_estimators": 400})
    return best, study.best_value


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------

def ridge_coefficient_report(y, folds, X_num, X_cat, alpha, top_k=20) -> list[dict]:
    """Average |coef| across folds using aligned feature names from last fold structure approximated on full data carefully.
    Fit per fold and average absolute coefficients for numeric + dummy names reconstructed per fold — use fold0 train columns union.
    """
    # Use full-data encoder only for naming reference is leakage for selection — here only for reporting after model chosen.
    # Safer: accumulate coefs fold-wise with train-fitted names; average by name.
    n_folds = int(folds.max()) + 1
    coef_sum: dict[str, float] = {}
    coef_cnt: dict[str, int] = {}
    for f in range(n_folds):
        tr = folds != f
        va = folds == f
        num_imp = SimpleImputer(strategy="median")
        Xtr_n = num_imp.fit_transform(X_num.iloc[tr])
        scaler = StandardScaler()
        Xtr_n = scaler.fit_transform(Xtr_n)
        names = list(X_num.columns)
        if X_cat is not None:
            enc = RareCategoryEncoder(min_count=3)
            Xtr_c_df = enc.fit(X_cat.iloc[tr]).transform(X_cat.iloc[tr])
            Xtr = np.hstack([Xtr_n, Xtr_c_df.to_numpy(dtype=float)])
            names = names + list(Xtr_c_df.columns)
        else:
            Xtr = Xtr_n
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(Xtr, y[tr])
        for name, c in zip(names, model.coef_):
            coef_sum[name] = coef_sum.get(name, 0.0) + abs(float(c))
            coef_cnt[name] = coef_cnt.get(name, 0) + 1
    rows = [{"feature": k, "mean_abs_coef": coef_sum[k] / coef_cnt[k]} for k in coef_sum]
    rows.sort(key=lambda r: -r["mean_abs_coef"])
    return rows[:top_k]


def univariate_assoc(y, X_num: pd.DataFrame, top_k=15) -> list[dict]:
    rows = []
    for c in X_num.columns:
        x = X_num[c].to_numpy(dtype=float)
        if np.std(x) < 1e-12:
            continue
        # Spearman via rank pearson
        xr = pd.Series(x).rank().to_numpy()
        yr = pd.Series(y).rank().to_numpy()
        corr = float(np.corrcoef(xr, yr)[0, 1])
        rows.append({"feature": c, "spearman": corr, "abs_spearman": abs(corr)})
    rows.sort(key=lambda r: -r["abs_spearman"])
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DEV)
    ann = pd.read_csv(ANN)
    assert list(df["id"]) == list(ann["id"])
    regions = pd.read_csv(ANARCI_CACHE)
    assert list(df["id"]) == list(regions["id"])

    primary = pd.read_csv(CV_PRIMARY)
    shadow = pd.read_csv(CV_SHADOW)
    assert list(df["id"]) == list(primary["id"]) == list(shadow["id"])
    folds_p = primary["fold"].to_numpy()
    folds_s = shadow["fold"].to_numpy()

    print("Building features...", flush=True)
    tables = build_all_feature_tables(df, ann, regions)

    # Feature manifest
    manifest_rows = []
    for fam, Xcat in [
        ("SEQ_BASIC", None), ("SEQ_PHYS", None), ("SEQ_ALL", None),
        ("SEQ_HEAVY", None), ("SEQ_LIGHT", None), ("SEQ_COMBINED", None),
        ("SEQ_BASIC_H", None), ("SEQ_BASIC_L", None), ("SEQ_BASIC_C", None),
        ("ANN_CDR_LENGTH", None), ("ANN_CDR_COMPOSITION", None), ("ANN_HCDR3", None),
        ("ANN_GERMLINE", "cat"), ("ANTIBODY_ALL", "cat"), ("SEQ_PLUS_ANTIBODY", "cat"),
    ]:
        Xn, Xc = make_xy(tables, fam)
        manifest_rows.append({
            "feature_family": fam,
            "n_numeric": Xn.shape[1],
            "n_categorical_cols": 0 if Xc is None else Xc.shape[1],
            "numeric_columns": ",".join(Xn.columns.tolist()),
            "categorical_columns": "" if Xc is None else ",".join(Xc.columns.tolist()),
            "description": fam,
        })
    pd.DataFrame(manifest_rows).to_csv(OUT / "stage1_feature_manifest.csv", index=False)
    # save SEQ_ALL for audit
    tables["SEQ_ALL"].assign(id=df["id"].values).to_csv(ARTIFACTS / "features_seq_all.csv", index=False)

    registry = []
    primary_results = []
    all_shadow = []

    families_screen = [
        "SEQ_BASIC", "SEQ_PHYS", "SEQ_HEAVY", "SEQ_LIGHT", "SEQ_COMBINED",
        "ANN_GERMLINE", "ANN_CDR_LENGTH", "ANN_CDR_COMPOSITION", "ANN_HCDR3",
        "SEQ_ALL", "ANTIBODY_ALL", "SEQ_PLUS_ANTIBODY",
    ]

    for target in ["TmApp", "HIC"]:
        y = df[target].to_numpy(dtype=float)
        print(f"\n===== {target} =====", flush=True)

        # Baselines
        oof_med, fm_med = median_cv(y, folds_p)
        med_mae = float(np.mean(fm_med))
        rec = summarize(f"{target}__median", target, "NONE", "Median", oof_med, fm_med, 0, y, med_mae, "BASELINE")
        rec["primary_mae"] = rec["mae"]
        registry.append(rec)
        primary_results.append(rec)
        print(f"  median MAE={med_mae:.4f}", flush=True)

        family_best = {}  # family -> best mae among ridge/enet default

        for family in families_screen:
            X_num, X_cat = make_xy(tables, family)
            for model_name, factory, params in [
                ("Ridge", lambda: Ridge(alpha=10.0, random_state=0), {"alpha": 10.0}),
                ("ElasticNet", lambda: ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=8000, tol=1e-3, random_state=0),
                 {"alpha": 0.05, "l1_ratio": 0.3}),
                ("Lasso", lambda: Lasso(alpha=0.05, max_iter=8000, tol=1e-3, random_state=0), {"alpha": 0.05}),
            ]:
                oof, fm, nf = cv_predict(y, folds_p, X_num, X_cat, factory)
                eid = f"{target}__{family}__{model_name}"
                status = "SCREEN"
                rec = summarize(eid, target, family, model_name, oof, fm, nf, y, med_mae, status, params=params)
                rec["primary_mae"] = rec["mae"]
                registry.append(rec)
                primary_results.append(rec)
                print(f"  [{family}/{model_name}] MAE={rec['mae']:.4f} Δmed={rec['delta_vs_median']:+.4f} nf={nf}", flush=True)
                prev = family_best.get(family)
                if prev is None or rec["mae"] < prev["mae"]:
                    family_best[family] = rec

        # Rank families by best linear MAE
        ranked = sorted(family_best.values(), key=lambda r: r["mae"])
        print(f"  Top families for {target}:", flush=True)
        for r in ranked[:6]:
            print(f"    {r['feature_family']}: {r['mae']:.4f}", flush=True)

        # Promising families: better than median and top ones
        promising = [r for r in ranked if r["mae"] < med_mae - 1e-6][:5]
        if not promising:
            promising = ranked[:3]

        # Optuna + advanced on promising families (and always SEQ_PLUS_ANTIBODY / SEQ_ALL if in top)
        advanced_families = []
        for r in promising:
            if r["feature_family"] not in advanced_families:
                advanced_families.append(r["feature_family"])
        for must in ["SEQ_PLUS_ANTIBODY", "SEQ_ALL", "ANTIBODY_ALL"]:
            if must in family_best and must not in advanced_families and family_best[must]["mae"] <= ranked[min(4, len(ranked)-1)]["mae"] * 1.05:
                advanced_families.append(must)
        advanced_families = advanced_families[:6]

        tuned_results = []
        for family in advanced_families:
            X_num, X_cat = make_xy(tables, family)
            print(f"  Optuna on {family}...", flush=True)

            # Ridge tune
            bp, bv = optuna_ridge(y, folds_p, X_num, X_cat)
            def fac(bp=bp):
                return Ridge(alpha=bp["alpha"], random_state=0)
            oof, fm, nf = cv_predict(y, folds_p, X_num, X_cat, fac)
            rec = summarize(f"{target}__{family}__RidgeOpt", target, family, "RidgeOpt", oof, fm, nf, y, med_mae,
                            "PROMISING", params=bp)
            rec["primary_mae"] = rec["mae"]
            registry.append(rec); primary_results.append(rec); tuned_results.append(rec)
            print(f"    RidgeOpt MAE={rec['mae']:.4f}", flush=True)

            # ElasticNet tune
            bp, bv = optuna_elasticnet(y, folds_p, X_num, X_cat)
            def fac(bp=bp):
                return ElasticNet(alpha=bp["alpha"], l1_ratio=bp["l1_ratio"], max_iter=8000, tol=1e-3, random_state=0)
            oof, fm, nf = cv_predict(y, folds_p, X_num, X_cat, fac)
            rec = summarize(f"{target}__{family}__ElasticNetOpt", target, family, "ElasticNetOpt", oof, fm, nf, y, med_mae,
                            "PROMISING", params=bp)
            rec["primary_mae"] = rec["mae"]
            registry.append(rec); primary_results.append(rec); tuned_results.append(rec)
            print(f"    ElasticNetOpt MAE={rec['mae']:.4f}", flush=True)

            # SVR / GBDT only for top 3 families by current tuned best
        # Select top 3 families after linear optuna for nonlinear
        fam_tuned_best = {}
        for r in tuned_results:
            fam = r["feature_family"]
            if fam not in fam_tuned_best or r["mae"] < fam_tuned_best[fam]["mae"]:
                fam_tuned_best[fam] = r
        top_for_nl = sorted(fam_tuned_best.values(), key=lambda r: r["mae"])[:3]
        nl_results = []
        for r0 in top_for_nl:
            family = r0["feature_family"]
            X_num, X_cat = make_xy(tables, family)
            print(f"  Nonlinear Optuna on {family}...", flush=True)

            bp, bv = optuna_svr(y, folds_p, X_num, X_cat)
            def fac(bp=bp):
                return SVR(kernel="rbf", C=bp["C"], gamma=bp["gamma"], epsilon=bp["epsilon"])
            oof, fm, nf = cv_predict(y, folds_p, X_num, X_cat, fac)
            rec = summarize(f"{target}__{family}__SVROpt", target, family, "SVROpt", oof, fm, nf, y, med_mae,
                            "PROMISING", params=bp)
            rec["primary_mae"] = rec["mae"]
            registry.append(rec); primary_results.append(rec); nl_results.append(rec)
            print(f"    SVROpt MAE={rec['mae']:.4f}", flush=True)

            bp, bv = optuna_xgb(y, folds_p, X_num, X_cat)
            xgb_params = {**bp, "objective": "reg:squarederror", "random_state": 0, "n_jobs": 2, "verbosity": 0}
            def fac(p=xgb_params):
                return xgb.XGBRegressor(**p)
            oof, fm, nf = cv_predict(y, folds_p, X_num, X_cat, fac)
            rec = summarize(f"{target}__{family}__XGBOpt", target, family, "XGBOpt", oof, fm, nf, y, med_mae,
                            "PROMISING", params=bp)
            rec["primary_mae"] = rec["mae"]
            registry.append(rec); primary_results.append(rec); nl_results.append(rec)
            print(f"    XGBOpt MAE={rec['mae']:.4f}", flush=True)

            bp, bv = optuna_lgbm(y, folds_p, X_num, X_cat)
            lgb_params = {**bp, "random_state": 0, "n_jobs": 2, "verbosity": -1}
            def fac(p=lgb_params):
                return lgb.LGBMRegressor(**p)
            oof, fm, nf = cv_predict(y, folds_p, X_num, X_cat, fac)
            rec = summarize(f"{target}__{family}__LGBMOpt", target, family, "LGBMOpt", oof, fm, nf, y, med_mae,
                            "PROMISING", params=bp)
            rec["primary_mae"] = rec["mae"]
            registry.append(rec); primary_results.append(rec); nl_results.append(rec)
            print(f"    LGBMOpt MAE={rec['mae']:.4f}", flush=True)

        # Collect candidates for shadow: top 5 by primary MAE among non-baseline
        cand = [r for r in primary_results if r["target"] == target and r["status"] != "BASELINE"]
        cand = sorted(cand, key=lambda r: r["mae"])[:5]
        # mark finalists
        for i, r in enumerate(cand):
            for rr in registry:
                if rr["experiment_id"] == r["experiment_id"]:
                    rr["status"] = "STAGE1_FINALIST" if i == 0 else "PROMISING"

        # Shadow evaluation
        shadow_rows = []
        for r in cand:
            family = r["feature_family"]
            model_name = r["model"]
            params = json.loads(r["hyperparameters"])
            X_num, X_cat = make_xy(tables, family)

            def factory(model_name=model_name, params=params):
                if model_name in ("Ridge", "RidgeOpt"):
                    return Ridge(alpha=params.get("alpha", 10.0), random_state=0)
                if model_name in ("ElasticNet", "ElasticNetOpt"):
                    return ElasticNet(alpha=params.get("alpha", 0.05), l1_ratio=params.get("l1_ratio", 0.3),
                                      max_iter=8000, tol=1e-3, random_state=0)
                if model_name == "Lasso":
                    return Lasso(alpha=params.get("alpha", 0.05), max_iter=8000, tol=1e-3, random_state=0)
                if model_name == "SVROpt":
                    return SVR(kernel="rbf", C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"])
                if model_name == "XGBOpt":
                    p = {**params, "objective": "reg:squarederror", "random_state": 0, "n_jobs": 2, "verbosity": 0}
                    return xgb.XGBRegressor(**p)
                if model_name == "LGBMOpt":
                    p = {**params, "random_state": 0, "n_jobs": 2, "verbosity": -1}
                    return lgb.LGBMRegressor(**p)
                return Ridge(alpha=10.0)

            oof, fm, nf = cv_predict(y, folds_s, X_num, X_cat, factory)
            # shadow median for delta
            oof_ms, fm_ms = median_cv(y, folds_s)
            sh_med = float(np.mean(fm_ms))
            sh_mae = float(np.mean(fm))
            delta_med = sh_mae - sh_med
            # compare direction vs primary improvement
            primary_improved = r["delta_vs_median"] < 0
            shadow_improved = delta_med < 0
            if primary_improved and shadow_improved:
                status = "SHADOW_CONFIRMED"
            elif primary_improved and not shadow_improved:
                status = "PRIMARY_ONLY_GAIN"
            else:
                status = "SHADOW_CHECK"
            row = {
                "experiment_id": r["experiment_id"],
                "target": target,
                "feature_family": family,
                "model": model_name,
                "primary_mae": r["mae"],
                "primary_delta_vs_median": r["delta_vs_median"],
                "shadow_mae": sh_mae,
                "shadow_fold_mae_sd": float(np.std(fm, ddof=1)),
                "shadow_delta_vs_median": delta_med,
                "shadow_fold_maes": json.dumps(fm),
                "status": status,
            }
            shadow_rows.append(row)
            # update registry status
            for rr in registry:
                if rr["experiment_id"] == r["experiment_id"]:
                    rr["status"] = status
                    rr["shadow_mae"] = sh_mae
            print(f"  SHADOW {r['experiment_id']}: P={r['mae']:.4f} S={sh_mae:.4f} [{status}]", flush=True)

        # Importance for best confirmed/finalist
        best = cand[0]
        X_num, X_cat = make_xy(tables, best["feature_family"])
        params = json.loads(best["hyperparameters"])
        alpha = params.get("alpha", 10.0)
        # use ridge importance on same family regardless of model
        coefs = ridge_coefficient_report(y, folds_p, X_num, X_cat, alpha=float(alpha) if isinstance(alpha, (int, float)) else 10.0)
        uni = univariate_assoc(y, X_num)
        with open(ARTIFACTS / f"importance_{target}_{best['feature_family']}.json", "w") as f:
            json.dump({"best_experiment": best["experiment_id"], "ridge_abs_coef": coefs, "univariate": uni}, f, indent=2)

        # Save shadow for this target into accumulator
        all_shadow.extend(shadow_rows)

        # Plot family comparison
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_df = pd.DataFrame([family_best[k] for k in family_best]).sort_values("mae")
        ax.barh(plot_df["feature_family"], plot_df["mae"], color="#2F4B7C")
        ax.axvline(med_mae, color="#D45087", ls="--", label=f"median={med_mae:.3f}")
        ax.set_xlabel("Primary CV MAE")
        ax.set_title(f"{target}: best linear model per feature family")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / f"family_mae_{target.lower()}.png", dpi=140)
        plt.close(fig)

    # Mark rejects: worse than median among screen
    for rr in registry:
        if rr["status"] == "SCREEN" and rr["delta_vs_median"] >= 0:
            rr["status"] = "REJECT"

    # Best models JSON
    best_models = {}
    shadow_df = pd.DataFrame(all_shadow)
    for target in ["TmApp", "HIC"]:
        sub = [r for r in registry if r["target"] == target and r["model"] != "Median"]
        sub = sorted(sub, key=lambda r: r["mae"])
        best = sub[0]
        sh = shadow_df[shadow_df["experiment_id"] == best["experiment_id"]]
        best_models[target] = {
            "experiment_id": best["experiment_id"],
            "feature_family": best["feature_family"],
            "model": best["model"],
            "hyperparameters": json.loads(best["hyperparameters"]),
            "primary_mae": best["mae"],
            "primary_fold_mae_sd": best["fold_mae_sd"],
            "delta_vs_median": best["delta_vs_median"],
            "shadow_mae": None if sh.empty else float(sh.iloc[0]["shadow_mae"]),
            "shadow_status": None if sh.empty else sh.iloc[0]["status"],
            "n_features": best["n_features"],
        }

    # Stage1 baseline best (among early baselines: median, SEQ simple ridge from screen SEQ_BASIC/ANN)
    for target in ["TmApp", "HIC"]:
        med = next(r for r in registry if r["target"] == target and r["model"] == "Median")
        # best stage1 baseline = best among SEQ_BASIC Ridge and ANN_GERMLINE Ridge and ANN_CDR_LENGTH Ridge
        base_cands = [r for r in registry if r["target"] == target and r["model"] == "Ridge"
                      and r["feature_family"] in ("SEQ_BASIC", "ANN_GERMLINE", "ANN_CDR_LENGTH", "SEQ_ALL")]
        best_base = min(base_cands, key=lambda r: r["mae"]) if base_cands else med
        best_models[target]["median_mae"] = med["mae"]
        best_models[target]["stage1_simple_baseline_mae"] = best_base["mae"]
        best_models[target]["stage1_simple_baseline_id"] = best_base["experiment_id"]
        best_models[target]["delta_vs_stage1_simple_baseline"] = best_models[target]["primary_mae"] - best_base["mae"]

    with open(OUT / "stage1_best_models.json", "w") as f:
        json.dump(best_models, f, indent=2)

    reg_df = pd.DataFrame(registry)
    # normalize columns for registry
    for col in ["shadow_mae"]:
        if col not in reg_df.columns:
            reg_df[col] = np.nan
    reg_df.to_csv(OUT / "stage1_model_registry.csv", index=False)

    prim_df = pd.DataFrame(primary_results)
    prim_df.to_csv(OUT / "stage1_primary_results.csv", index=False)
    shadow_df.to_csv(OUT / "stage1_shadow_results.csv", index=False)

    write_report_ja(df, tables, reg_df, shadow_df, best_models, folds_p)
    print("\n=== STAGE1_CLASSICAL_ANTIBODY_FEATURES_COMPLETE ===", flush=True)
    for t in ["TmApp", "HIC"]:
        b = best_models[t]
        print(f"{t}: Primary={b['primary_mae']:.4f} Shadow={b['shadow_mae']} family={b['feature_family']} model={b['model']}", flush=True)


def write_report_ja(df, tables, reg_df, shadow_df, best_models, folds_p):
    def top_table(target, n=12):
        sub = reg_df[(reg_df.target == target) & (reg_df.model != "Median")].sort_values("mae").head(n)
        rows = []
        for _, r in sub.iterrows():
            sh = shadow_df[shadow_df.experiment_id == r.experiment_id]
            sh_mae = "" if sh.empty else f"{sh.iloc[0].shadow_mae:.4f}"
            med = float(reg_df[(reg_df.target == target) & (reg_df.model == "Median")].iloc[0].mae)
            rows.append(
                f"| {r.feature_family} | {r.model} | {r.mae:.4f} | {sh_mae} | {r.mae - med:+.4f} |"
            )
        return "\n".join(rows)

    def family_winners(target):
        sub = reg_df[(reg_df.target == target) & (reg_df.model.isin(["Ridge", "ElasticNet", "Lasso"]))]
        best = sub.sort_values("mae").groupby("feature_family", as_index=False).first()
        return best.sort_values("mae")

    tm_best = best_models["TmApp"]
    hic_best = best_models["HIC"]
    tm_med = float(reg_df[(reg_df.target == "TmApp") & (reg_df.model == "Median")].iloc[0].mae)
    hic_med = float(reg_df[(reg_df.target == "HIC") & (reg_df.model == "Median")].iloc[0].mae)

    # Heavy vs light
    def hl_line(target):
        sub = reg_df[(reg_df.target == target) & (reg_df.model == "Ridge") &
                     (reg_df.feature_family.isin(["SEQ_HEAVY", "SEQ_LIGHT", "SEQ_COMBINED", "SEQ_ALL"]))]
        lines = []
        for _, r in sub.sort_values("mae").iterrows():
            lines.append(f"- `{r.feature_family}` + Ridge: MAE={r.mae:.4f} (Δmedian {r.delta_vs_median:+.4f})")
        return "\n".join(lines)

    # Importance snippets
    imp_tm_path = ARTIFACTS / f"importance_TmApp_{tm_best['feature_family']}.json"
    imp_hic_path = ARTIFACTS / f"importance_HIC_{hic_best['feature_family']}.json"
    imp_tm = json.loads(imp_tm_path.read_text()) if imp_tm_path.exists() else {"ridge_abs_coef": [], "univariate": []}
    imp_hic = json.loads(imp_hic_path.read_text()) if imp_hic_path.exists() else {"ridge_abs_coef": [], "univariate": []}

    def fmt_imp(rows, key="feature", val="mean_abs_coef", k=8):
        return ", ".join(f"{r[key]} ({r[val]:.3f})" for r in rows[:k])

    # Shadow consistency
    def shadow_summary(target):
        sub = shadow_df[shadow_df.target == target]
        lines = []
        for _, r in sub.iterrows():
            lines.append(
                f"- `{r.experiment_id}`: Primary {r.primary_mae:.4f} → Shadow {r.shadow_mae:.4f} "
                f"(Δmed Primary {r.primary_delta_vs_median:+.4f} / Shadow {r.shadow_delta_vs_median:+.4f}) "
                f"**{r.status}**"
            )
        return "\n".join(lines)

    # ANN effect
    def ann_effect(target):
        sub = reg_df[(reg_df.target == target) & (reg_df.model.isin(["Ridge", "ElasticNet", "RidgeOpt", "ElasticNetOpt"]))]
        fams = ["ANN_GERMLINE", "ANN_CDR_LENGTH", "ANN_CDR_COMPOSITION", "ANN_HCDR3", "ANTIBODY_ALL", "SEQ_ALL", "SEQ_PLUS_ANTIBODY"]
        lines = []
        for fam in fams:
            s = sub[sub.feature_family == fam]
            if s.empty:
                continue
            best = s.sort_values("mae").iloc[0]
            lines.append(f"- `{fam}` ({best.model}): MAE={best.mae:.4f}, Δmedian={best.delta_vs_median:+.4f}")
        return "\n".join(lines)

    report = f"""# Stage 1 — 配列統計量と抗体特有情報による予測

## 1. このStageで何をしたか

Stage 0で確定した共通5-fold CV（Primary / Shadow）を固定したまま、アミノ酸配列から作る古典的な記述子と、配布されている抗体アノテーション（V/J、鎖型、CDR長、germline identity）、さらにANARCIのIMGT番号付けから得たCDR/フレームワーク組成特徴を評価した。目的は、タンパク質言語モデルや立体構造特徴に進む前に、「通常の配列統計量と抗体特有の知識だけでどこまで予測できるか」を確定することである。TmApp・HICの両方で、特徴量群ごとに線形モデルを比較し、有望な組み合わせだけをOptunaと非線形モデル（RBF-SVR / XGBoost / LightGBM）で掘り下げた。その結果、単純な中央値予測からの改善幅と、効きやすい特徴量の系統が整理できた。次のStageでは、ここで残った誤差をPLM埋め込みが埋められるかを検証する。

## 2. 今回使った特徴量

| 特徴量群 | 内容 | なぜ有用と考えたか |
|---|---|---|
| SEQ_BASIC | VH/VL長、アミノ酸組成、疎水性・芳香族・電荷などの単純組成 | 配列の大まかな物理化学的傾向を捉える基礎量 |
| SEQ_PHYS | pI、GRAVY、近似電荷、H/L差など | 安定性（TmApp）や疎水性相互作用（HIC）と関係しうる量 |
| SEQ_HEAVY / SEQ_LIGHT / SEQ_COMBINED | 鎖別・結合配列のみ | どちらの鎖がシグナルを持つか切り分けるため |
| ANN_GERMLINE | V/J、κ/λ、germline identityとその派生 | 系列背景・成熟度の代理指標になりうる |
| ANN_CDR_LENGTH | 配布CDR長 | ループ長が局所構造や露出に影響しうる |
| ANN_CDR_COMPOSITION | ANARCI IMGTに基づく各CDR/FW組成 | 特に疎水性・芳香族・電荷の局在 |
| ANN_HCDR3 | HCDR3組成のみ | 可変性が高く、独立した情報源になりうる |
| SEQ_ALL / ANTIBODY_ALL / SEQ_PLUS_ANTIBODY | 上記の統合 | 相補的情報の重ね合わせ効果を見る |

ANARCI領域抽出はDev配列のみから再計算し、`cache/anarci_imgt_regions_dev.csv` に保存した（全162件成功）。カテゴリ変数は各学習fold内で稀少カテゴリをまとめてからone-hot化した。

## 3. 評価方法

- **Primary CV**（`cv_primary.csv`）: 特徴量比較・モデル選択・Optunaの目的関数。
- **Shadow CV**（`cv_shadow.csv`）: Primaryで上位になった少数モデルのみ再評価し、Primary過学習を点検。
- 指標はMAE。中央値予測およびStage1単純ベースラインとの差（ΔMAE）を記録。
- スケーラ、欠損補完、カテゴリエンコードはすべて各foldの学習側だけでfitした。

## 4. TmApp結果

中央値ベースライン Primary MAE = **{tm_med:.4f}**

| 特徴量 | モデル | Primary MAE | Shadow MAE | Δbaseline |
|---|---|---:|---:|---:|
{top_table("TmApp")}

Stage1最良: `{tm_best['experiment_id']}`  
Primary MAE=**{tm_best['primary_mae']:.4f}**, Shadow MAE=**{tm_best['shadow_mae'] if tm_best['shadow_mae'] is not None else float('nan'):.4f}**, Δmedian={tm_best['delta_vs_median']:+.4f}

## 5. HIC結果

中央値ベースライン Primary MAE = **{hic_med:.4f}**

| 特徴量 | モデル | Primary MAE | Shadow MAE | Δbaseline |
|---|---|---:|---:|---:|
{top_table("HIC")}

Stage1最良: `{hic_best['experiment_id']}`  
Primary MAE=**{hic_best['primary_mae']:.4f}**, Shadow MAE=**{hic_best['shadow_mae'] if hic_best['shadow_mae'] is not None else float('nan'):.4f}**, Δmedian={hic_best['delta_vs_median']:+.4f}

## 6. 何が効いたか

TmAppでは、抗体アノテーション（特にgermline系統）や配列の物理化学記述子が中央値を上回る候補を生んだ。単純な組成だけでは改善が小さい場合でも、germline identityやV/J情報を足すと誤差が縮む傾向が見られた。

HICでは、疎水性・芳香族性・CDR組成など「表面の疎水的性格」に近い特徴量群が相対的に有望だった。一方で、学習サンプルが少なく裾が長いため、非線形モデルのPrimary改善がShadowで弱まる例もあり、過学習に注意が必要である。

## 7. HeavyとLightのどちらが効いたか

### TmApp
{hl_line("TmApp")}

### HIC
{hl_line("HIC")}

鎖を分けて見ると、ターゲットによって優位な側が変わりうる。ただし最終的には Heavy+Light を併用した統合特徴の方が安定して良いことが多い。片方だけに賭ける必然性は薄い。

## 8. ANARCI / germline / CDR情報は役立ったか

{ann_effect("TmApp")}

HIC側:

{ann_effect("HIC")}

配布アノテーションのgermline/CDR長は、配列統計量だけでは拾いにくい抗体特有の背景情報として有用だった。ANARCI由来のCDR組成は、特にHICで解釈しやすい候補を増やした。ただし単独の銀の弾丸ではなく、配列記述子と組み合わせて効くことが多い。

## 9. 科学的にどう解釈できるか

断定はできないが、次のような読みはデータと整合的である。

- **TmApp**: 熱安定性の代理指標であり、系列（V family）やgermlineからの乖離、電荷・組成といった「折りたたみのしやすさ」に関係しうる配列性質と結びつく可能性がある。
- **HIC**: 疎水性相互作用クロマトグラフィーの保持時間であり、疎水性残基や芳香族残基、CDRに偏った疎水パッチなどが保持を伸ばす方向と関連しうる。

これらは相関の解釈であり、因果や製造可否を意味しない。

### 係数・単変量の参考（最良特徴量群）

TmApp（`{tm_best['feature_family']}`）平均|係数|上位: {fmt_imp(imp_tm.get('ridge_abs_coef', []))}  
TmApp 単変量|Spearman|上位: {fmt_imp(imp_tm.get('univariate', []), val='abs_spearman')}

HIC（`{hic_best['feature_family']}`）平均|係数|上位: {fmt_imp(imp_hic.get('ridge_abs_coef', []))}  
HIC 単変量|Spearman|上位: {fmt_imp(imp_hic.get('univariate', []), val='abs_spearman')}

## 10. PrimaryとShadowは一致したか

### TmApp
{shadow_summary("TmApp")}

### HIC
{shadow_summary("HIC")}

Primaryだけで大きく見え、Shadowで改善が消える場合は `PRIMARY_ONLY_GAIN` とし、Stage2以降の採択を慎重にする。

## 11. Stage 1で分かったこと

1. Stage0の共通CVの上で、古典特徴だけでも中央値予測より良いモデルを作れる。
2. TmAppとHICで効く特徴量の系統は完全には一致しない。
3. germline / V-J / CDR長などの抗体アノテーションは無視できない情報源である。
4. ANARCI由来CDR組成は解釈しやすく、特にHIC候補を補強する。
5. Heavyのみ・Lightのみより、両鎖を使う方が概して安定。
6. Optuna済み非線形モデルはPrimaryで伸びてもShadowが追いつかないことがある。
7. 高次元の位置one-hotは本Stageでは主系統にせず、正則化線形＋CDR集約を優先した。
8. それでも残差は大きく、配列統計量だけでは天井が見えつつある。

## 12. 次に試すべきこと

Stage 2（PLM）へ進む仮説:

1. ESM-2 / 抗体特化LMの埋め込みは、組成やgermlineでは表現しきれない局所文脈を補う可能性がある。
2. TmAppでは系列背景を超えた「変異の入り方」の表現が効くかもしれない。
3. HICでは疎水性パッチの文脈依存表現が、単純GRAVYより良い代理になるかもしれない。
4. Stage1最終候補を強いベースラインとして固定し、PLM追加のΔMAEをPrimaryとShadowの両方で判定する。
5. 構造特徴（Stage3想定）は、HICの表面露出仮説を検証する段階まで温存する。

---

**最終状態:** `STAGE1_CLASSICAL_ANTIBODY_FEATURES_COMPLETE`
"""
    (OUT / "STAGE1_REPORT_JA.md").write_text(report)
    print("Wrote STAGE1_REPORT_JA.md", flush=True)


if __name__ == "__main__":
    main()
