#!/usr/bin/env python3
"""Train linear models on PLM embeddings and structure features; optional XGBoost finalists."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    CACHE,
    CONFIG,
    DATA,
    METRICS,
    PREDS,
    REPORTS,
    SPLITS,
    TARGET_COLS,
    ensure_dirs,
    read_json,
    write_json,
)
from importlib import import_module

# reuse metrics from 05
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

spec = importlib.util.spec_from_file_location("train_abc", Path(__file__).resolve().parent / "05_train_ABC.py")
train_abc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_abc)
metrics_dict = train_abc.metrics_dict


def load_embedding_matrix(manifest_csv: Path, ids: pd.Series) -> np.ndarray:
    man = pd.read_csv(manifest_csv).set_index("antibody_id")
    vecs = []
    for aid in ids:
        path = Path(man.loc[aid, "path"])
        vecs.append(np.load(path))
    X = np.vstack(vecs)
    return X


def load_structure_features(feat_csv: Path, ids: pd.Series, include=None, exclude=None) -> pd.DataFrame:
    feat = pd.read_csv(feat_csv)
    feat = ids.to_frame("antibody_id").merge(feat, on="antibody_id", how="left")
    cols = [c for c in feat.columns if c != "antibody_id"]
    if include:
        cols = [c for c in cols if any(s in c for s in include)]
    if exclude:
        cols = [c for c in cols if not any(s in c for s in exclude)]
    # drop non-numeric
    X = feat[cols].apply(pd.to_numeric, errors="coerce")
    return X


def nested_cv_dense(X, y, groups, model_name="Ridge", n_outer=5):
    n_groups = len(np.unique(groups))
    n_outer = min(n_outer, n_groups)
    while n_outer > 2 and n_groups // n_outer < 2:
        n_outer -= 1
    grids = {
        "Ridge": {"model__alpha": [0.1, 1, 10, 100, 1000, 10000]},
        "Lasso": {"model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0]},
        "ElasticNet": {"model__alpha": [1e-3, 1e-2, 0.1, 1], "model__l1_ratio": [0.2, 0.5, 0.8]},
    }
    ctors = {
        "Ridge": Ridge,
        "Lasso": lambda **k: Lasso(max_iter=20000, **k),
        "ElasticNet": lambda **k: ElasticNet(max_iter=20000, **k),
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
            inner = GroupKFold(n_splits=n_inner)
            for itr, iva in inner.split(X[tr], y[tr], gtr):
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


def run_xgb(X, y, groups, n_outer=5):
    import xgboost as xgb

    n_groups = len(np.unique(groups))
    n_outer = min(n_outer, n_groups)
    while n_outer > 2 and n_groups // n_outer < 2:
        n_outer -= 1
    # modest search
    grid = [
        {"max_depth": d, "min_child_weight": w, "subsample": s, "colsample_bytree": c, "reg_lambda": l, "reg_alpha": a}
        for d in [2, 3, 4]
        for w in [1, 5]
        for s in [0.7, 1.0]
        for c in [0.5, 0.8]
        for l in [1.0, 5.0]
        for a in [0.0, 0.5]
    ]
    rng = np.random.default_rng(0)
    grid = [grid[i] for i in rng.choice(len(grid), size=min(20, len(grid)), replace=False)]

    outer = GroupKFold(n_splits=n_outer)
    oof = np.full(len(y), np.nan)
    best_last = grid[0]
    X = np.asarray(X, dtype=float)
    # impute
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    for tr, te in outer.split(X, y, groups):
        best_score, best_p = -np.inf, None
        gtr = groups[tr]
        n_inner = min(3, len(np.unique(gtr)))
        if n_inner < 2:
            n_inner = 2
        for params in grid:
            scores = []
            inner = GroupKFold(n_splits=n_inner)
            for itr, iva in inner.split(X[tr], y[tr], gtr):
                model = xgb.XGBRegressor(
                    n_estimators=4000,
                    learning_rate=0.03,
                    objective="reg:squarederror",
                    n_jobs=4,
                    **params,
                )
                model.fit(
                    X[tr][itr],
                    y[tr][itr],
                    eval_set=[(X[tr][iva], y[tr][iva])],
                    verbose=False,
                )
                # manual early stop approx: use best iteration if available
                pred = model.predict(X[tr][iva])
                sc = spearmanr(y[tr][iva], pred).correlation
                if sc is not None and np.isfinite(sc):
                    scores.append(sc)
            if scores and np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_p = params
        best_p = best_p or grid[0]
        model = xgb.XGBRegressor(
            n_estimators=4000, learning_rate=0.03, objective="reg:squarederror", n_jobs=4, **best_p
        )
        # use a holdout from train for early stopping
        model.fit(X[tr], y[tr], verbose=False)
        oof[te] = model.predict(X[te])
        best_last = best_p
    final = xgb.XGBRegressor(
        n_estimators=4000, learning_rate=0.03, objective="reg:squarederror", n_jobs=4, **best_last
    )
    final.fit(X, y, verbose=False)
    return {"oof": oof, "cv": metrics_dict(y, oof), "model": final, "params": best_last}


def eval_roles(model, X, y, roles, ids, is_xgb=False):
    out = {}
    for role in ["Public", "Private"]:
        mask = roles == role
        if mask.sum() == 0:
            continue
        Xp = X[mask] if isinstance(X, np.ndarray) else X.iloc[mask]
        pred = model.predict(Xp)
        out[role] = metrics_dict(y[mask], pred)
        pd.DataFrame({"antibody_id": ids[mask], "y_true": y[mask], "y_pred": pred}).to_csv(
            PREDS / f"_tmp_{role}.csv", index=False
        )
    return out


def main():
    ensure_dirs()
    df = pd.read_csv(DATA / "triple_core.csv")

    # Define representation loaders available
    reps = []
    for mid in ["ablang2_default", "ablang_original", "esm1b_t33_650M_UR50S", "esm2_t33_650M_UR50D"]:
        man = CACHE / "plm" / f"manifest_{mid}.csv"
        if man.exists():
            reps.append(("PLM_" + mid, "emb", man, None))
    man_cdr = CACHE / "plm" / "manifest_esm2_t33_650M_UR50D_CDR6.csv"
    if man_cdr.exists():
        reps.append(("PLM_esm2_CDR6", "emb", man_cdr, None))

    for pred, pref in [("abodybuilder2", "ABB"), ("esmfold", "ESMF")]:
        path = CACHE / "structure_features" / f"{pred}_sasa_rasa.csv"
        if not path.exists():
            continue
        reps.append((f"STR_{pref}_SASA", "str", path, {"include": ["_sasa", "total_sasa", "mean_sasa"], "exclude": ["rasa", "weighted", "BSA", "interface"]}))
        reps.append((f"STR_{pref}_SASA_RASA", "str", path, {"include": ["sasa", "rasa"], "exclude": ["weighted", "BSA"]}))
        reps.append((f"STR_{pref}_SURFACE_PHYS", "str", path, {"include": ["rasa_weighted", "exposed_", "sasa_hydrophobic", "sasa_aromatic"], "exclude": []}))
        reps.append((f"STR_{pref}_INTERFACE", "str", path, {"include": ["BSA", "interface", "isolated"], "exclude": []}))

    all_rows = []
    for split_name in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
        split_df = pd.read_csv(SPLITS / f"triple_{split_name}.csv")
        # split_df already carries cluster_id
        m = split_df.merge(df, on="antibody_id", how="left")
        for target, col in TARGET_COLS.items():
            y_all = m[col].values.astype(float)
            roles = m["role"].values
            groups = m["cluster_id"].values
            ids = m["antibody_id"].values
            dev = roles == "Dev"
            for rep_name, kind, path, filt in reps:
                print(f"RUN {split_name} {target} {rep_name}", flush=True)
                try:
                    if kind == "emb":
                        X = load_embedding_matrix(path, m["antibody_id"])
                    else:
                        Xdf = load_structure_features(path, m["antibody_id"], **(filt or {}))
                        X = Xdf.values.astype(float)
                except Exception as e:
                    print("SKIP", rep_name, e)
                    continue
                for mn in ["Ridge", "Lasso", "ElasticNet"]:
                    cv = nested_cv_dense(X[dev], y_all[dev], groups[dev], model_name=mn)
                    model = cv["model"]
                    # public/private
                    role_m = {}
                    for role in ["Public", "Private"]:
                        mask = roles == role
                        pred = model.predict(X[mask])
                        role_m[role] = metrics_dict(y_all[mask], pred)
                        pd.DataFrame(
                            {"antibody_id": ids[mask], "y_true": y_all[mask], "y_pred": pred}
                        ).to_csv(PREDS / f"{split_name}_{target}_{rep_name}_{mn}_{role}.csv", index=False)
                    pd.DataFrame(
                        {"antibody_id": ids[dev], "y_true": y_all[dev], "y_pred": cv["oof"]}
                    ).to_csv(PREDS / f"{split_name}_{target}_{rep_name}_{mn}_oof.csv", index=False)
                    all_rows.append(
                        {
                            "representation": rep_name,
                            "model": mn,
                            "target": target,
                            "split": split_name,
                            "cv_spearman": cv["cv"]["spearman"],
                            "cv_pearson": cv["cv"]["pearson"],
                            "cv_mae": cv["cv"]["mae"],
                            "cv_rmse": cv["cv"]["rmse"],
                            "public_spearman": role_m.get("Public", {}).get("spearman"),
                            "private_spearman": role_m.get("Private", {}).get("spearman"),
                            "public_mae": role_m.get("Public", {}).get("mae"),
                            "private_mae": role_m.get("Private", {}).get("mae"),
                            "feature_class": "PARTICIPANT_LEGAL",
                            "best_params": json.dumps(cv["params"]),
                        }
                    )

    out = pd.DataFrame(all_rows)
    out.to_csv(METRICS / "stage_PLM_STR_results.csv", index=False)
    # reports
    lines = ["# PLM results", ""]
    can = out[out["split"] == "canonical"]
    for target in TARGET_COLS:
        lines.append(f"## {target}")
        sub = can[can["target"] == target].sort_values("cv_spearman", ascending=False)
        lines.append("| rep | model | CV ρ | Pub ρ | Priv ρ |")
        lines.append("|-----|-------|------:|------:|-------:|")
        for _, r in sub.head(20).iterrows():
            lines.append(
                f"| {r['representation']} | {r['model']} | {r['cv_spearman']:.3f} | {r['public_spearman']:.3f} | {r['private_spearman']:.3f} |"
            )
        lines.append("")
    (REPORTS / "plm_results.md").write_text("\n".join(lines) + "\n")
    print("PLM_STR_TRAIN_OK", len(out))


if __name__ == "__main__":
    main()
