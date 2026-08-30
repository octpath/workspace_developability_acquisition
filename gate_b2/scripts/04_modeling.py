#!/usr/bin/env python3
"""Gate B2 modeling: SEQ/PLM/structure ablations, fusions, residuals, nonlinear."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import (  # noqa: E402
    B1_CACHE,
    B1_DATA,
    B2_TARGETS,
    CACHE,
    CONFIG,
    DATA,
    METRICS,
    PREDS,
    REPORTS,
    SPLITS,
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


def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 3:
        return {"spearman": np.nan, "pearson": np.nan, "mae": np.nan, "rmse": np.nan, "n": int(len(y_true))}
    sp = spearmanr(y_true, y_pred).correlation
    pr = pearsonr(y_true, y_pred)[0]
    return {
        "spearman": float(sp) if sp is not None and np.isfinite(sp) else np.nan,
        "pearson": float(pr) if pr is not None and np.isfinite(pr) else np.nan,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n": int(len(y_true)),
    }


def nested_cv_dense(X, y, groups, model_name="Ridge", n_outer=5, fixed_folds=None):
    """Faster grids for high-dim; still nested GroupKFold."""
    X = np.asarray(X, float)
    n_feat = X.shape[1]
    n_groups = len(np.unique(groups))
    n_outer = min(n_outer, n_groups)
    while n_outer > 2 and n_groups // n_outer < 2:
        n_outer -= 1
    if n_feat > 200:
        grids = {
            "Ridge": {"model__alpha": [1, 10, 100, 1000, 10000]},
            "Lasso": {"model__alpha": [1e-3, 1e-2, 0.1]},
            "ElasticNet": {"model__alpha": [1e-2, 0.1, 1], "model__l1_ratio": [0.5]},
        }
        max_iter = 4000
    else:
        grids = {
            "Ridge": {"model__alpha": [0.1, 1, 10, 100, 1000]},
            "Lasso": {"model__alpha": [1e-3, 1e-2, 0.1, 1.0]},
            "ElasticNet": {"model__alpha": [1e-3, 1e-2, 0.1, 1], "model__l1_ratio": [0.2, 0.5, 0.8]},
        }
        max_iter = 8000
    ctors = {
        "Ridge": Ridge,
        "Lasso": lambda **k: Lasso(max_iter=max_iter, **k),
        "ElasticNet": lambda **k: ElasticNet(max_iter=max_iter, **k),
    }
    outer = GroupKFold(n_splits=n_outer)
    oof = np.full(len(y), np.nan)
    best_params_last = None
    for tr, te in outer.split(X, y, groups):
        best_score, best_p = -np.inf, None
        gtr = groups[tr]
        n_inner = min(3, len(np.unique(gtr)))
        if n_inner < 2:
            n_inner = 2
        for params in ParameterGrid(grids[model_name]):
            scores = []
            for itr, iva in GroupKFold(n_splits=n_inner).split(X[tr], y[tr], gtr):
                pipe = Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("model", ctors[model_name]()),
                    ]
                )
                pipe.set_params(**params)
                try:
                    pipe.fit(X[tr][itr], y[tr][itr])
                    pred = pipe.predict(X[tr][iva])
                    sc = spearmanr(y[tr][iva], pred).correlation
                    if sc is not None and np.isfinite(sc):
                        scores.append(sc)
                except Exception:
                    continue
            if scores and np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_p = params
        if best_p is None:
            best_p = list(ParameterGrid(grids[model_name]))[0]
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", ctors[model_name]()),
            ]
        )
        pipe.set_params(**best_p)
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
        best_params_last = best_p
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ctors[model_name]()),
        ]
    )
    pipe.set_params(**best_params_last)
    pipe.fit(X, y)
    return {"oof": oof, "cv": metrics_dict(y, oof), "model": pipe, "params": best_params_last}


def models_for(rep_name, split_name, n_feat):
    """Canonical: fuller model set. Shadows: Ridge (+ EN for low-dim structure/seq)."""
    is_plm = rep_name.startswith("PLM_") or rep_name.startswith("FUSION_")
    is_struct = rep_name.startswith("ABB_") or rep_name.startswith("ESMF_NATIVE_")
    if split_name == "canonical":
        if is_plm or n_feat > 200:
            return ["Ridge", "ElasticNet"]  # skip slow Lasso on high-dim
        return ["Ridge", "Lasso", "ElasticNet"]
    # shadows: ranking reliability needs breadth of reps, not every regularizer
    if is_plm or n_feat > 200:
        return ["Ridge"]
    if is_struct or rep_name.startswith("SEQ_") or rep_name == "BIO_SHORTCUT":
        return ["Ridge", "ElasticNet"]
    return ["Ridge"]


def load_csv_features(path, ids, prefixes=None, include=None, exclude=None, cols=None):
    feat = pd.read_csv(path)
    feat = ids.to_frame("antibody_id").merge(feat, on="antibody_id", how="left")
    if cols:
        use = [c for c in cols if c in feat.columns]
    else:
        use = [c for c in feat.columns if c != "antibody_id"]
        if prefixes:
            use = [c for c in use if any(c.startswith(p) for p in prefixes)]
        if include:
            use = [c for c in use if any(s in c for s in include)]
        if exclude:
            use = [c for c in use if not any(s in c for s in exclude)]
    X = feat[use].apply(pd.to_numeric, errors="coerce")
    return X


def load_emb(manifest, ids):
    # resolve symlink to B1
    man = Path(manifest)
    if not man.exists():
        man = B1_CACHE / "plm" / Path(manifest).name
    return tplm.load_embedding_matrix(man, ids)


def tree_cv(X, y, groups, kind="RF"):
    n_groups = len(np.unique(groups))
    n_outer = min(5, n_groups)
    while n_outer > 2 and n_groups // n_outer < 2:
        n_outer -= 1
    X = np.asarray(X, float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(col_mean, inds[1])
    outer = GroupKFold(n_splits=n_outer)
    oof = np.full(len(y), np.nan)
    for tr, te in outer.split(X, y, groups):
        if kind == "RF":
            model = RandomForestRegressor(
                n_estimators=400, max_depth=6, min_samples_leaf=3, n_jobs=8, random_state=0
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=400, max_depth=6, min_samples_leaf=3, n_jobs=8, random_state=0
            )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    if kind == "RF":
        final = RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3, n_jobs=8, random_state=0
        )
    else:
        final = ExtraTreesRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3, n_jobs=8, random_state=0
        )
    final.fit(X, y)
    return {"oof": oof, "cv": metrics_dict(y, oof), "model": final}


def light_xgb(X, y, groups):
    import xgboost as xgb

    n_groups = len(np.unique(groups))
    n_outer = min(5, n_groups)
    while n_outer > 2 and n_groups // n_outer < 2:
        n_outer -= 1
    X = np.asarray(X, float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(col_mean, inds[1])
    oof = np.full(len(y), np.nan)
    params = dict(
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.5,
        learning_rate=0.03,
        n_estimators=1500,
        n_jobs=8,
        objective="reg:squarederror",
    )
    for tr, te in GroupKFold(n_splits=n_outer).split(X, y, groups):
        model = xgb.XGBRegressor(**params)
        model.fit(X[tr], y[tr], verbose=False)
        oof[te] = model.predict(X[te])
    final = xgb.XGBRegressor(**params)
    final.fit(X, y, verbose=False)
    return {"oof": oof, "cv": metrics_dict(y, oof), "model": final, "params": params}


def eval_roles(model, X, y, roles, impute_from=None):
    out = {}
    X = np.asarray(X, float)
    if impute_from is not None:
        col_mean = np.nanmean(np.asarray(impute_from, float), axis=0)
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_mean, inds[1])
    for role in ["Public", "Private"]:
        mask = roles == role
        if mask.sum() == 0:
            continue
        pred = model.predict(X[mask])
        out[role] = metrics_dict(y[mask], pred)
    return out


def build_registry():
    feat_a = CACHE / "features" / "stage_A_simple.csv"
    feat_b = CACHE / "features" / "stage_B_cdr.csv"
    feat_c = CACHE / "features" / "stage_C_shortcut.csv"
    abb = CACHE / "structure_features" / "abb_sasa_rasa_patch.csv"
    esmn = CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv"
    reg = {
        "SEQ_SIMPLE": {"kind": "csv", "path": feat_a, "prefixes": ["A0_", "A1_", "A2_"]},
        "SEQ_CDR": {"kind": "csv", "path": feat_b, "prefixes": ["B_"]},
        "BIO_SHORTCUT": {
            "kind": "csv_cat",
            "path": feat_c,
            "prefixes": ["C_"],
            "categorical": ["C_vh_family", "C_vl_family", "C_family_pair", "C_kappa_lambda"],
        },
        "PLM_ABLANG2": {"kind": "emb", "manifest": B1_CACHE / "plm" / "manifest_ablang2_default.csv"},
        "PLM_ESM1B": {"kind": "emb", "manifest": B1_CACHE / "plm" / "manifest_esm1b_t33_650M_UR50S.csv"},
        "PLM_ESM2": {"kind": "emb", "manifest": B1_CACHE / "plm" / "manifest_esm2_t33_650M_UR50D.csv"},
        "PLM_ESM2_CDR6": {
            "kind": "emb",
            "manifest": B1_CACHE / "plm" / "manifest_esm2_t33_650M_UR50D_CDR6.csv",
        },
    }
    # Structure nested ablation S0-S4 for both predictors
    for tag, path, pref in [("ABB", abb, "ABB"), ("ESMF_NATIVE", esmn, "ESMFN")]:
        if not path.exists():
            continue
        reg[f"{tag}_SASA"] = {
            "kind": "csv",
            "path": path,
            "include": ["_total_sasa", "_mean_sasa", "_sasa_"],
            "exclude": ["rasa", "patch", "BSA", "iface", "iso", "w_"],
            "ablation": "S0",
        }
        reg[f"{tag}_SASA_RASA"] = {
            "kind": "csv",
            "path": path,
            "include": ["sasa", "rasa"],
            "exclude": ["patch", "BSA", "iface", "iso", "w_", "rasa_w"],
            "ablation": "S1",
        }
        reg[f"{tag}_SURFACE"] = {
            "kind": "csv",
            "path": path,
            "include": ["sasa_hydrophobic", "sasa_aromatic", "sasa_positive", "sasa_negative", "sasa_polar", "total_sasa", "mean_sasa"],
            "exclude": ["rasa", "patch", "BSA", "iface"],
            "ablation": "S2",
        }
        reg[f"{tag}_SURFACE_RASAW"] = {
            "kind": "csv",
            "path": path,
            "include": ["total_sasa", "mean_sasa", "rasa_w_", "exposed_"],
            "exclude": ["patch", "BSA", "iface"],
            "ablation": "S3",
        }
        reg[f"{tag}_ALL_SURFACE"] = {
            "kind": "csv",
            "path": path,
            "include": ["sasa", "rasa", "rasa_w", "exposed_"],
            "exclude": ["patch", "BSA", "iface", "iso"],
            "ablation": "S4",
        }
        reg[f"{tag}_PATCH"] = {
            "kind": "csv",
            "path": path,
            "include": ["patch", "hydrophobic_patch", "cdr_hydrophobic", "h3_hydrophobic"],
            "exclude": [],
        }
        reg[f"{tag}_INTERFACE"] = {
            "kind": "csv",
            "path": path,
            "include": ["BSA", "iface", "iso"],
            "exclude": [],
        }
        reg[f"{tag}_ALL"] = {
            "kind": "csv",
            "path": path,
            "prefixes": [f"{pref}_"],
            "exclude": ["error"],
        }
    write_json(CONFIG / "pipeline_registry.json", {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()} for k, v in reg.items()})
    return reg


def load_rep(meta, ids):
    kind = meta["kind"]
    if kind == "emb":
        return load_emb(meta["manifest"], ids), None, None
    if kind == "csv":
        X = load_csv_features(
            meta["path"],
            ids,
            prefixes=meta.get("prefixes"),
            include=meta.get("include"),
            exclude=meta.get("exclude"),
            cols=meta.get("cols"),
        )
        return X.values.astype(float), list(X.columns), []
    if kind == "csv_cat":
        # handled specially
        return None, None, None
    raise ValueError(kind)


def run_csv_cat(meta, m, y, roles, groups, rep_name, target, split_name, rows):
    """BIO_SHORTCUT with one-hot via Ridge/Lasso/ElasticNet using B1 helper path."""
    # Use numeric C_ columns + simple hash encoding for cats
    feat = pd.read_csv(meta["path"])
    feat = m[["antibody_id"]].merge(feat, on="antibody_id", how="left")
    cat_cols = [c for c in meta.get("categorical", []) if c in feat.columns]
    num_cols = [c for c in feat.columns if c.startswith("C_") and c not in cat_cols and c != "antibody_id"]
    Xnum = feat[num_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    # one-hot cats manually with pandas
    Xcat = pd.get_dummies(feat[cat_cols].astype(str), dummy_na=True).values.astype(float)
    X = np.concatenate([Xnum, Xcat], axis=1)
    dev = roles == "Dev"
    for mn in models_for(rep_name, split_name, X.shape[1]):
        print(f"RUN {split_name} {target} {rep_name} {mn}", flush=True)
        cv = nested_cv_dense(X[dev], y[dev], groups[dev], mn)
        role_m = eval_roles(cv["model"], X, y, roles)
        rows.append(record(rep_name, mn, target, split_name, cv, role_m))


def record(rep, model, target, split, cv, role_m, extra=None):
    d = {
        "representation": rep,
        "model": model,
        "target": target,
        "split": split,
        "cv_spearman": cv["cv"]["spearman"],
        "cv_pearson": cv["cv"]["pearson"],
        "cv_mae": cv["cv"]["mae"],
        "cv_rmse": cv["cv"]["rmse"],
        "public_spearman": role_m.get("Public", {}).get("spearman"),
        "private_spearman": role_m.get("Private", {}).get("spearman"),
        "public_mae": role_m.get("Public", {}).get("mae"),
        "private_mae": role_m.get("Private", {}).get("mae"),
        "feature_class": "PARTICIPANT_LEGAL",
    }
    if extra:
        d.update(extra)
    return d


def main():
    ensure_dirs()
    reg = build_registry()
    rows = []
    splits = ["canonical", "shadow_1", "shadow_2", "shadow_3", "shadow_4", "shadow_5"]

    for target, col in B2_TARGETS.items():
        df = pd.read_csv(B1_DATA / f"{'hic' if target=='HIC' else 'tmapp'}_full.csv")
        for split_name in splits:
            sp = pd.read_csv(SPLITS / f"{target.lower()}_{split_name}.csv")
            m = sp.merge(df, on="antibody_id", how="left")
            y = m[col].values.astype(float)
            roles = m["role"].values
            groups = m["cluster_id"].values
            ids = m["antibody_id"]
            dev = roles == "Dev"

            for rep_name, meta in reg.items():
                if meta["kind"] == "csv_cat":
                    run_csv_cat(meta, m, y, roles, groups, rep_name, target, split_name, rows)
                    continue
                if meta["kind"] == "csv" and not Path(meta["path"]).exists():
                    continue
                if meta["kind"] == "emb" and not Path(meta["manifest"]).exists():
                    continue
                try:
                    X, _, _ = load_rep(meta, ids)
                    if X is None or X.shape[1] == 0:
                        continue
                except Exception as e:
                    print("SKIP", rep_name, e, flush=True)
                    continue
                models = models_for(rep_name, split_name, X.shape[1])
                # nonlinear on structure-ish only (canonical core families)
                do_trees = split_name == "canonical" and rep_name in (
                    "ABB_ALL",
                    "ABB_SURFACE",
                    "ABB_PATCH",
                    "ESMF_NATIVE_ALL",
                    "ESMF_NATIVE_SURFACE",
                    "ESMF_NATIVE_PATCH",
                )
                for mn in models:
                    print(f"RUN {split_name} {target} {rep_name} {mn} d={X.shape[1]}", flush=True)
                    cv = nested_cv_dense(X[dev], y[dev], groups[dev], mn)
                    role_m = eval_roles(cv["model"], X, y, roles)
                    rows.append(record(rep_name, mn, target, split_name, cv, role_m))
                    # incremental save
                    if len(rows) % 20 == 0:
                        pd.DataFrame(rows).to_csv(METRICS / "all_results.csv", index=False)
                if do_trees:
                    for kind in ["RF", "ET"]:
                        print(f"TREE {split_name} {target} {rep_name} {kind}", flush=True)
                        cv = tree_cv(X[dev], y[dev], groups[dev], kind=kind)
                        role_m = eval_roles(cv["model"], X, y, roles, impute_from=X[dev])
                        rows.append(record(rep_name, kind, target, split_name, cv, role_m))
                if split_name == "canonical" and rep_name in (
                    "PLM_ESM2",
                    "ABB_ALL",
                    "ESMF_NATIVE_ALL",
                    "SEQ_CDR",
                ):
                    try:
                        print(f"XGB {split_name} {target} {rep_name}", flush=True)
                        cv = light_xgb(X[dev], y[dev], groups[dev])
                        role_m = eval_roles(cv["model"], X, y, roles, impute_from=X[dev])
                        rows.append(record(rep_name, "XGBoost", target, split_name, cv, role_m))
                    except Exception as e:
                        print("XGB_FAIL", e)

            # Fusions on canonical + shadows (linear only)
            try:
                Xplm = load_emb(B1_CACHE / "plm" / "manifest_esm2_t33_650M_UR50D.csv", ids)
                parts = [("FUSION_ESM2_SEQ", CACHE / "features" / "stage_A_simple.csv", ["A2_"])]
                abb_path = CACHE / "structure_features" / "abb_sasa_rasa_patch.csv"
                esmn_path = CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv"
                if abb_path.exists():
                    parts.append(("FUSION_ESM2_ABB", abb_path, None))
                if esmn_path.exists():
                    parts.append(("FUSION_ESM2_ESMFN", esmn_path, None))
                for fname, path, prefs in parts:
                    if prefs:
                        Xs = load_csv_features(path, ids, prefixes=prefs).values.astype(float)
                    else:
                        Xs = load_csv_features(
                            path, ids, include=["sasa", "rasa", "patch", "BSA", "rasa_w"], exclude=["error"]
                        ).values.astype(float)
                    X = np.concatenate([Xplm, Xs], axis=1)
                    for mn in models_for(fname, split_name, X.shape[1]):
                        print(f"FUSION {split_name} {target} {fname} {mn}", flush=True)
                        cv = nested_cv_dense(X[dev], y[dev], groups[dev], mn)
                        role_m = eval_roles(cv["model"], X, y, roles)
                        rows.append(record(fname, mn, target, split_name, cv, role_m))
                # PLM + structure + BIO numeric — Ridge on all splits; EN on canonical
                bio = pd.read_csv(CACHE / "features" / "stage_C_shortcut.csv")
                bio = ids.to_frame("antibody_id").merge(bio, on="antibody_id", how="left")
                bnum = bio.select_dtypes(include=[np.number]).values.astype(float)
                if abb_path.exists():
                    Xs = load_csv_features(
                        abb_path, ids, include=["sasa", "rasa", "patch", "rasa_w"], exclude=["error"]
                    ).values.astype(float)
                    X = np.concatenate([Xplm, Xs, bnum], axis=1)
                    for mn in (["Ridge", "ElasticNet"] if split_name == "canonical" else ["Ridge"]):
                        cv = nested_cv_dense(X[dev], y[dev], groups[dev], mn)
                        role_m = eval_roles(cv["model"], X, y, roles)
                        rows.append(record("FUSION_ESM2_ABB_BIO", mn, target, split_name, cv, role_m))
            except Exception as e:
                print("FUSION_FAIL", e)

            pd.DataFrame(rows).to_csv(METRICS / "all_results.csv", index=False)
            print(f"CHECKPOINT {target} {split_name} n={len(rows)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "all_results.csv", index=False)
    print("MODELING_OK", len(out))


if __name__ == "__main__":
    main()
