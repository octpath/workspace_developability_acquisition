#!/usr/bin/env python3
"""Grouped nested CV training utilities + Stage A–C model runs."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

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


def metrics_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": np.nan, "pearson": np.nan, "mae": np.nan, "rmse": np.nan, "n": int(len(y_true))}
    sp = spearmanr(y_true, y_pred).correlation
    pr = pearsonr(y_true, y_pred)[0]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "spearman": float(sp) if sp is not None and np.isfinite(sp) else np.nan,
        "pearson": float(pr) if pr is not None and np.isfinite(pr) else np.nan,
        "mae": mae,
        "rmse": rmse,
        "n": int(len(y_true)),
    }


def load_feature_matrix(rep_name: str, registry: dict, ids: pd.Series) -> tuple[pd.DataFrame, list, list]:
    meta = registry[rep_name]
    path = CACHE / "features" / meta["file"]
    feat = pd.read_csv(path)
    feat = ids.to_frame("antibody_id").merge(feat, on="antibody_id", how="left")
    cat_cols = list(meta.get("categorical") or [])
    if "cols" in meta:
        cols = list(meta["cols"])
    else:
        prefixes = meta.get("cols_prefix") or []
        cols = [c for c in feat.columns if c != "antibody_id" and any(c.startswith(p) for p in prefixes)]
    # keep only existing
    cols = [c for c in cols if c in feat.columns]
    cat_cols = [c for c in cat_cols if c in cols]
    num_cols = [c for c in cols if c not in cat_cols]
    X = feat[cols].copy()
    return X, num_cols, cat_cols


def make_pipe(num_cols, cat_cols, model):
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    pre = ColumnTransformer(transformers)
    return Pipeline([("pre", pre), ("model", model)])


def nested_group_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    num_cols: list,
    cat_cols: list,
    model_name: str,
    n_outer: int = 5,
    n_inner: int = 3,
    random_state: int = 0,
) -> dict:
    n_groups = len(np.unique(groups))
    n_outer = min(n_outer, n_groups)
    if n_outer < 2:
        n_outer = 2
    # adjust if too few groups
    while n_outer > 2 and n_groups // n_outer < 2:
        n_outer -= 1

    param_grids = {
        "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
        "Lasso": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
        "ElasticNet": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
        "SVR": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", 0.01],
            "model__epsilon": [0.05, 0.1],
        },
    }
    model_ctors = {
        "Ridge": lambda: Ridge(random_state=random_state),
        "Lasso": lambda: Lasso(max_iter=20000, random_state=random_state),
        "ElasticNet": lambda: ElasticNet(max_iter=20000, random_state=random_state),
        "SVR": lambda: SVR(kernel="rbf"),
    }

    outer = GroupKFold(n_splits=n_outer)
    oof = np.full(len(y), np.nan)
    fold_stats = []
    best_params_folds = []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        gtr = groups[tr]
        n_inner_eff = min(n_inner, len(np.unique(gtr)))
        if n_inner_eff < 2:
            n_inner_eff = 2
        # grid search manual on inner
        best_score, best_p = -np.inf, None
        grid = list(ParameterGrid(param_grids[model_name]))
        # subsample grid if huge
        if len(grid) > 20:
            rng = np.random.default_rng(random_state + fold)
            idx = rng.choice(len(grid), size=20, replace=False)
            grid = [grid[i] for i in idx]
        for params in grid:
            inner = GroupKFold(n_splits=n_inner_eff)
            scores = []
            for itr, iva in inner.split(Xtr, ytr, gtr):
                pipe = make_pipe(num_cols, cat_cols, model_ctors[model_name]())
                pipe.set_params(**params)
                try:
                    pipe.fit(Xtr.iloc[itr], ytr[itr])
                    pred = pipe.predict(Xtr.iloc[iva])
                    sc = spearmanr(ytr[iva], pred).correlation
                    if sc is not None and np.isfinite(sc):
                        scores.append(sc)
                except Exception:
                    continue
            if scores and np.nanmean(scores) > best_score:
                best_score = float(np.nanmean(scores))
                best_p = params
        if best_p is None:
            best_p = list(ParameterGrid(param_grids[model_name]))[0]
        pipe = make_pipe(num_cols, cat_cols, model_ctors[model_name]())
        pipe.set_params(**best_p)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        oof[te] = pred
        fold_stats.append(metrics_dict(yte, pred))
        best_params_folds.append(best_p)

    # refit best avg params on all (mode of alphas approx: use last fold's best)
    # choose params with best mean by reusing majority / last
    final_params = best_params_folds[-1] if best_params_folds else {}
    pipe = make_pipe(num_cols, cat_cols, model_ctors[model_name]())
    pipe.set_params(**final_params)
    pipe.fit(X, y)
    return {
        "oof": oof,
        "cv_metrics": metrics_dict(y, oof),
        "fold_metrics": fold_stats,
        "best_params": final_params,
        "n_outer": n_outer,
        "model": pipe,
    }


def eval_split_roles(model, X_all, y_all, roles, ids):
    out = {}
    preds = {}
    for role in ["Dev", "Public", "Private"]:
        mask = roles == role
        if mask.sum() == 0:
            continue
        # For Dev report OOF separately; here Dev is refit train score (optimistic) — skip for freeze protocol
        if role == "Dev":
            continue
        pred = model.predict(X_all.iloc[mask])
        out[role] = metrics_dict(y_all[mask], pred)
        preds[role] = pd.DataFrame(
            {"antibody_id": ids[mask].values, "y_true": y_all[mask], "y_pred": pred}
        )
    return out, preds


def run_rep_target(
    rep_name,
    target_name,
    split_name,
    registry,
    df,
    split_df,
    model_names=("Ridge", "Lasso", "ElasticNet"),
):
    col = TARGET_COLS[target_name]
    # align
    m = split_df.merge(df[["antibody_id", col, "cluster_id"] if "cluster_id" in df.columns else ["antibody_id", col]], on="antibody_id")
    # ensure cluster
    if "cluster_id" not in m.columns:
        cl = pd.read_csv(DATA / "triple_clusters.csv")
        m = m.merge(cl, on="antibody_id")
    roles = m["role"].values
    dev_mask = roles == "Dev"
    ids = m["antibody_id"]
    X, num_cols, cat_cols = load_feature_matrix(rep_name, registry, ids)
    y = m[col].values.astype(float)
    groups = m["cluster_id"].values

    results = []
    best = None
    for mn in model_names:
        cv = nested_group_cv(
            X.iloc[dev_mask].reset_index(drop=True),
            y[dev_mask],
            groups[dev_mask],
            num_cols,
            cat_cols,
            mn,
        )
        # refit on all Dev with returned model already fit on Dev
        model = cv["model"]
        role_metrics, role_preds = eval_split_roles(model, X, y, roles, ids)
        rec = {
            "representation": rep_name,
            "model": mn,
            "target": target_name,
            "split": split_name,
            "cv_spearman": cv["cv_metrics"]["spearman"],
            "cv_pearson": cv["cv_metrics"]["pearson"],
            "cv_mae": cv["cv_metrics"]["mae"],
            "cv_rmse": cv["cv_metrics"]["rmse"],
            "public_spearman": role_metrics.get("Public", {}).get("spearman"),
            "public_mae": role_metrics.get("Public", {}).get("mae"),
            "private_spearman": role_metrics.get("Private", {}).get("spearman"),
            "private_mae": role_metrics.get("Private", {}).get("mae"),
            "best_params": json.dumps(cv["best_params"]),
            "feature_class": registry[rep_name].get("class"),
            "n_dev": int(dev_mask.sum()),
            "n_features": int(X.shape[1]),
        }
        results.append(rec)
        # save OOF on Dev
        oof_df = pd.DataFrame(
            {
                "antibody_id": ids[dev_mask].values,
                "y_true": y[dev_mask],
                "y_pred": cv["oof"],
            }
        )
        oof_path = PREDS / f"{split_name}_{target_name}_{rep_name}_{mn}_oof.csv"
        oof_df.to_csv(oof_path, index=False)
        for role, pdf in role_preds.items():
            pdf.to_csv(PREDS / f"{split_name}_{target_name}_{rep_name}_{mn}_{role}.csv", index=False)
        if best is None or (rec["cv_spearman"] is not None and rec["cv_spearman"] > best["cv_spearman"]):
            best = rec
            best["_model_obj"] = model
            best["_X"] = X
            best["_y"] = y
            best["_roles"] = roles
            best["_ids"] = ids
    return results, best


def main():
    ensure_dirs()
    registry = read_json(CONFIG / "representation_registry.json")
    df = pd.read_csv(DATA / "triple_core.csv")
    cl = pd.read_csv(DATA / "triple_clusters.csv")
    df = df.merge(cl, on="antibody_id")

    reps = [
        "A0_length",
        "A1_aa_comp",
        "A2_physchem",
        "B_cdr_descriptors",
        "B_cdr_hydrophobicity_only",
        "C_BIO_SHORTCUT",
        "ORG_subset_only",
        "ORG_germline_author",
        "ORG_subset_plus_germline",
    ]
    all_rows = []
    for split_name in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
        split_df = pd.read_csv(SPLITS / f"triple_{split_name}.csv")
        for target in ["PSR", "HIC", "TmApp"]:
            for rep in reps:
                print(f"RUN {split_name} {target} {rep}", flush=True)
                rows, _ = run_rep_target(rep, target, split_name, registry, df, split_df)
                all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    out.to_csv(METRICS / "stage_ABC_results.csv", index=False)
    # markdown summary for canonical
    can = out[out["split"] == "canonical"]
    lines = ["# Simple and shortcut baselines", "", "## Canonical TRIPLE_CORE (Dev CV Spearman)", ""]
    for target in ["PSR", "HIC", "TmApp"]:
        lines.append(f"### {target}")
        lines.append("")
        sub = can[can["target"] == target].sort_values("cv_spearman", ascending=False)
        lines.append("| rep | model | CV ρ | Public ρ | Private ρ | class |")
        lines.append("|-----|-------|------:|---------:|----------:|-------|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['representation']} | {r['model']} | {r['cv_spearman']:.3f} | "
                f"{r['public_spearman'] if pd.notna(r['public_spearman']) else float('nan'):.3f} | "
                f"{r['private_spearman'] if pd.notna(r['private_spearman']) else float('nan'):.3f} | {r['feature_class']} |"
            )
        lines.append("")
    (REPORTS / "simple_and_shortcut_baselines.md").write_text("\n".join(lines) + "\n")
    print("STAGE_ABC_OK", len(out))


if __name__ == "__main__":
    main()
