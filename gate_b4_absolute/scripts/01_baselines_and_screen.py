#!/usr/bin/env python3
"""
Gate B4 Stage 1: absolute baselines + Optuna screening (Train-only).
Outer validation never used for early stopping / HP selection within a trial's
inner fit; screening scores use frozen outer CV for ranking candidates only.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import (  # noqa: E402
    CONFIG,
    LOGS,
    METRICS,
    OPTUNA_DIR,
    PREDS,
    REPORTS,
    ensure_dirs,
    group_kfold_labels,
    load_representation,
    load_train_frame,
    read_json,
    regression_metrics,
    set_seeds,
    software_versions,
    train_only,
    write_json,
    MASTER_SEED,
    OPTUNA_SAMPLER_SEED,
)
from b4_models import (  # noqa: E402
    fit_predict_cat,
    fit_predict_lgb,
    fit_predict_pca_head,
    fit_predict_sklearn,
    fit_predict_xgb,
    suggest_cat,
    suggest_elasticnet,
    suggest_krr,
    suggest_lgb,
    suggest_svr,
    suggest_xgb,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def ridge_alpha_grid_cv(X, y, fold_id, n_folds):
    best_alpha, best_mae = 10.0, np.inf
    for a in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        maes = []
        for f in range(n_folds):
            te = fold_id == f
            tr = ~te
            pred = fit_predict_sklearn("Ridge", {"alpha": a}, X[tr], y[tr], X[te])
            maes.append(np.mean(np.abs(y[te] - pred)))
        m = float(np.mean(maes))
        if m < best_mae:
            best_mae, best_alpha = m, a
    return best_alpha, best_mae


def eval_outer(X, y, groups, outer, predict_fn, tag, model_name):
    """Evaluate on frozen outer folds; predict_fn(Xtr,ytr,Xte,gtr)->(pred, meta)."""
    rows = []
    oof = np.full(len(y), np.nan)
    train_ids = outer["train_ids"]
    for fold_info in outer["folds"]:
        fold_id = np.array(fold_info["fold_id"])
        rep = fold_info["repeat"]
        for f in range(outer["n_folds"]):
            te = fold_id == f
            tr = ~te
            if tr.sum() < 10 or te.sum() < 3:
                continue
            try:
                pred, meta = predict_fn(X[tr], y[tr], X[te], groups[tr])
                met = regression_metrics(y[te], pred)
                rows.append(
                    {
                        "stage": "screen_outer",
                        "tag": tag,
                        "model": model_name,
                        "repeat": rep,
                        "fold": f,
                        **met,
                        "best_iteration": meta.get("best_iteration"),
                        "hit_ceiling": meta.get("hit_ceiling"),
                    }
                )
                if rep == 0:
                    oof[te] = pred
            except Exception as e:
                rows.append(
                    {
                        "stage": "screen_outer",
                        "tag": tag,
                        "model": model_name,
                        "repeat": rep,
                        "fold": f,
                        "mae": np.nan,
                        "error": str(e)[:200],
                    }
                )
    return rows, oof


def optuna_screen_sklearn(study_name, X, y, groups, kind, suggest_fn, n_trials, seed):
    # fixed 3-fold grouped on all train for screening HP
    fold_id = group_kfold_labels(groups, 3, seed=seed + 7)
    storage = f"sqlite:///{OPTUNA_DIR / (study_name + '.db')}"
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    def objective(trial):
        params = suggest_fn(trial)
        maes = []
        for f in range(3):
            te = fold_id == f
            tr = ~te
            # early-stop not needed; sklearn
            # inner-val is evaluation only — no ES
            pred = fit_predict_sklearn(kind, params, X[tr], y[tr], X[te], seed=seed)
            maes.append(float(np.mean(np.abs(y[te] - pred))))
        return float(np.mean(maes))

    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    study.trials_dataframe().to_csv(OPTUNA_DIR / f"{study_name}_trials.csv", index=False)
    return study.best_params, float(study.best_value), len(study.trials)


def optuna_screen_gbdt(study_name, X, y, groups, family, objective_name, n_trials, seed):
    fold_id = group_kfold_labels(groups, 3, seed=seed + 11)
    storage = f"sqlite:///{OPTUNA_DIR / (study_name + '.db')}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=seed),
    )

    def objective(trial):
        if family == "xgb":
            params = suggest_xgb(trial)
        elif family == "lgb":
            params = suggest_lgb(trial)
        else:
            params = suggest_cat(trial)
        maes = []
        for f in range(3):
            te = fold_id == f
            tr = ~te
            # ES uses split of tr only; te untouched
            if family == "xgb":
                pred, _ = fit_predict_xgb(params, X[tr], y[tr], X[te], groups[tr], objective=objective_name, seed=seed + f)
            elif family == "lgb":
                pred, _ = fit_predict_lgb(params, X[tr], y[tr], X[te], groups[tr], objective=objective_name, seed=seed + f)
            else:
                pred, _ = fit_predict_cat(params, X[tr], y[tr], X[te], groups[tr], loss=objective_name, seed=seed + f)
            maes.append(float(np.mean(np.abs(y[te] - pred))))
        return float(np.mean(maes))

    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    study.trials_dataframe().to_csv(OPTUNA_DIR / f"{study_name}_trials.csv", index=False)
    return study.best_params, float(study.best_value), len(study.trials)


def main():
    ensure_dirs()
    set_seeds(MASTER_SEED)
    t0 = time.time()
    outer = read_json(CONFIG / "OUTER_CV_FOLDS.json")
    df = load_train_frame(include_holdout_labels=False)
    train = train_only(df)
    # align to outer train_ids
    train = train.set_index("id").loc[outer["train_ids"]].reset_index()
    groups = train["sequence_group"].values
    assert list(train["id"]) == outer["train_ids"]

    all_rows = []
    oof_bank = {}  # (target, tag, model) -> oof
    screen_summary = []

    # Representations by target priority
    reps_common = ["SEQ_SIMPLE", "SEQ_CDR", "BIO", "GERMLINE_REL", "IMGT_POS_HL"]
    reps_plm = ["PLM_ABLANG2", "PLM_ESM1B", "PLM_ESM2", "PLM_ESM2_CDR6"]
    reps_str = ["ABB_STRUCTURE", "ESMFN_STRUCTURE"]
    reps_fuse_h = ["FUSION_ESM2_ESMFN", "FUSION_ESM2_SEQCDR_ESMFN"]
    reps_fuse_t = ["FUSION_ABLANG2_BIO", "FUSION_ABLANG2_IMGT", "FUSION_BIO_IMGT"]

    # Preload all needed
    needed = sorted(set(reps_common + reps_plm + reps_str + reps_fuse_h + reps_fuse_t))
    Xcache = {}
    for tag in needed:
        print(f"load {tag}", flush=True)
        Xcache[tag] = load_representation(tag, train["id"])
        print(f"  shape {Xcache[tag].shape}", flush=True)

    for target, col in [("HIC", "HIC"), ("TmApp", "TmApp")]:
        print(f"\n===== TARGET {target} =====", flush=True)
        y = train[col].values.astype(float)
        y_sd = float(np.std(y, ddof=1))
        y_iqr = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))

        # ---- Baselines ----
        med = float(np.median(y))
        mean = float(np.mean(y))
        for name, const in [("CONST_MEDIAN", med), ("CONST_MEAN", mean)]:
            pred = np.full_like(y, const)
            # fake outer eval
            for fold_info in outer["folds"]:
                fold_id = np.array(fold_info["fold_id"])
                rep = fold_info["repeat"]
                for f in range(outer["n_folds"]):
                    te = fold_id == f
                    met = regression_metrics(y[te], pred[te])
                    all_rows.append(
                        {
                            "stage": "baseline",
                            "target": target,
                            "tag": name,
                            "model": "constant",
                            "repeat": rep,
                            "fold": f,
                            **met,
                            "mae_over_sd": met["mae"] / y_sd if y_sd else np.nan,
                            "mae_over_iqr": met["mae"] / y_iqr if y_iqr else np.nan,
                        }
                    )
            oof_bank[(target, name, "constant")] = pred.copy()

        # Ridge grid baselines on SEQ/BIO
        for tag in ["SEQ_SIMPLE", "SEQ_CDR", "BIO", "GERMLINE_REL", "IMGT_POS_HL"]:
            X = Xcache[tag]
            # pick alpha on repeat0 folds only (Train-only)
            fold0 = np.array(outer["folds"][0]["fold_id"])
            alpha, _ = ridge_alpha_grid_cv(X, y, fold0, outer["n_folds"])

            def pred_fn(Xtr, ytr, Xte, gtr, a=alpha):
                return fit_predict_sklearn("Ridge", {"alpha": a}, Xtr, ytr, Xte), {"best_iteration": None}

            rows, oof = eval_outer(X, y, groups, outer, pred_fn, tag, "Ridge_grid")
            for r in rows:
                r["target"] = target
                r["stage"] = "baseline"
                r["mae_over_sd"] = r["mae"] / y_sd if np.isfinite(r.get("mae", np.nan)) else np.nan
                r["mae_over_iqr"] = r["mae"] / y_iqr if np.isfinite(r.get("mae", np.nan)) else np.nan
                r["alpha"] = alpha
            all_rows.extend(rows)
            oof_bank[(target, tag, "Ridge_grid")] = oof
            screen_summary.append(
                {
                    "target": target,
                    "tag": tag,
                    "model": "Ridge_grid",
                    "alpha": alpha,
                    "mean_mae": float(np.nanmean([r["mae"] for r in rows])),
                }
            )

        # Priority reps for Optuna
        if target == "HIC":
            screen_reps = ["SEQ_SIMPLE", "BIO", "PLM_ESM2", "PLM_ESM1B", "PLM_ABLANG2", "ESMFN_STRUCTURE", "ABB_STRUCTURE", "FUSION_ESM2_ESMFN"]
        else:
            screen_reps = ["SEQ_SIMPLE", "BIO", "IMGT_POS_HL", "GERMLINE_REL", "PLM_ABLANG2", "PLM_ESM2", "ESMFN_STRUCTURE", "FUSION_ABLANG2_BIO"]

        # ElasticNet Optuna (40 trials)
        for tag in screen_reps:
            X = Xcache[tag]
            sname = f"screen_{target}_{tag}_ElasticNet"
            print(f"Optuna {sname}", flush=True)
            try:
                best_params, best_val, ntr = optuna_screen_sklearn(
                    sname, X, y, groups, "ElasticNet", suggest_elasticnet, n_trials=40, seed=OPTUNA_SAMPLER_SEED
                )
            except Exception as e:
                print("FAIL", sname, e)
                traceback.print_exc()
                continue

            def pred_fn(Xtr, ytr, Xte, gtr, bp=best_params):
                return fit_predict_sklearn("ElasticNet", bp, Xtr, ytr, Xte), {}

            rows, oof = eval_outer(X, y, groups, outer, pred_fn, tag, "ElasticNet")
            for r in rows:
                r["target"] = target
                r["stage"] = "screen"
                r["mae_over_sd"] = r["mae"] / y_sd if np.isfinite(r.get("mae", np.nan)) else np.nan
            all_rows.extend(rows)
            oof_bank[(target, tag, "ElasticNet")] = oof
            screen_summary.append(
                {
                    "target": target,
                    "tag": tag,
                    "model": "ElasticNet",
                    "optuna_best_inner_mae": best_val,
                    "n_trials": ntr,
                    "best_params": json.dumps(best_params),
                    "mean_outer_mae": float(np.nanmean([r["mae"] for r in rows])),
                }
            )

        # PCA+SVR / KRR for top PLMs
        plm_tags = ["PLM_ABLANG2", "PLM_ESM2"] if target == "TmApp" else ["PLM_ESM2", "PLM_ESM1B"]
        for tag in plm_tags:
            X = Xcache[tag]
            for npc in [16, 32, 64]:
                sname = f"screen_{target}_{tag}_PCA{npc}_SVR"
                print(f"Optuna {sname}", flush=True)

                def suggest_pca_svr(trial, n=npc):
                    p = suggest_svr(trial)
                    p["n_components"] = n
                    return p

                fold_id = group_kfold_labels(groups, 3, seed=OPTUNA_SAMPLER_SEED + 21)
                storage = f"sqlite:///{OPTUNA_DIR / (sname + '.db')}"
                study = optuna.create_study(
                    study_name=sname,
                    storage=storage,
                    load_if_exists=True,
                    direction="minimize",
                    sampler=TPESampler(seed=OPTUNA_SAMPLER_SEED),
                )

                def objective(trial, n=npc):
                    params = suggest_pca_svr(trial, n)
                    maes = []
                    for f in range(3):
                        te = fold_id == f
                        tr = ~te
                        pred = fit_predict_pca_head("SVR_RBF", params, X[tr], y[tr], X[te])
                        maes.append(float(np.mean(np.abs(y[te] - pred))))
                    return float(np.mean(maes))

                rem = max(0, 30 - len(study.trials))
                if rem:
                    study.optimize(objective, n_trials=rem)
                study.trials_dataframe().to_csv(OPTUNA_DIR / f"{sname}_trials.csv", index=False)
                bp = study.best_params
                bp["n_components"] = npc

                def pred_fn(Xtr, ytr, Xte, gtr, params=bp):
                    return fit_predict_pca_head("SVR_RBF", params, Xtr, ytr, Xte), {}

                rows, oof = eval_outer(X, y, groups, outer, pred_fn, f"{tag}_PCA{npc}", "SVR_RBF")
                for r in rows:
                    r["target"] = target
                    r["stage"] = "screen"
                all_rows.extend(rows)
                oof_bank[(target, f"{tag}_PCA{npc}", "SVR_RBF")] = oof
                screen_summary.append(
                    {
                        "target": target,
                        "tag": f"{tag}_PCA{npc}",
                        "model": "SVR_RBF",
                        "optuna_best_inner_mae": float(study.best_value),
                        "n_trials": len(study.trials),
                        "best_params": json.dumps(bp),
                        "mean_outer_mae": float(np.nanmean([r["mae"] for r in rows])),
                    }
                )

        # GBDT screening on priority reps (60 trials each, limited set)
        # Pre-registered loss families; keep a lean but diverse Stage-1 matrix
        if target == "HIC":
            gbdt_reps = ["ESMFN_STRUCTURE", "PLM_ESM2", "FUSION_ESM2_ESMFN"]
        else:
            gbdt_reps = ["BIO", "PLM_ABLANG2", "FUSION_ABLANG2_BIO"]

        gbdt_specs = [
            ("xgb", "reg:absoluteerror", "XGB_L1"),
            ("xgb", "reg:squarederror", "XGB_L2"),
            ("lgb", "regression_l1", "LGB_L1"),
            ("cat", "MAE", "CAT_MAE"),
        ]
        for tag in gbdt_reps:
            X = Xcache[tag]
            for family, obj, mname in gbdt_specs:
                sname = f"screen_{target}_{tag}_{mname}"
                print(f"Optuna GBDT {sname}", flush=True)
                try:
                    bp, bv, ntr = optuna_screen_gbdt(
                        sname, X, y, groups, family, obj, n_trials=40, seed=OPTUNA_SAMPLER_SEED
                    )
                except Exception as e:
                    print("FAIL GBDT", sname, e)
                    traceback.print_exc()
                    continue

                def make_pred_fn(fam=family, objective=obj, params=bp):
                    def pred_fn(Xtr, ytr, Xte, gtr):
                        if fam == "xgb":
                            return fit_predict_xgb(params, Xtr, ytr, Xte, gtr, objective=objective)
                        if fam == "lgb":
                            return fit_predict_lgb(params, Xtr, ytr, Xte, gtr, objective=objective)
                        return fit_predict_cat(params, Xtr, ytr, Xte, gtr, loss=objective)

                    return pred_fn

                rows, oof = eval_outer(X, y, groups, outer, make_pred_fn(), tag, mname)
                for r in rows:
                    r["target"] = target
                    r["stage"] = "screen_gbdt"
                all_rows.extend(rows)
                oof_bank[(target, tag, mname)] = oof
                screen_summary.append(
                    {
                        "target": target,
                        "tag": tag,
                        "model": mname,
                        "optuna_best_inner_mae": bv,
                        "n_trials": ntr,
                        "best_params": json.dumps(bp),
                        "mean_outer_mae": float(np.nanmean([r["mae"] for r in rows])),
                        "median_best_iteration": float(
                            np.nanmedian([r.get("best_iteration") for r in rows if r.get("best_iteration")])
                        ),
                        "frac_hit_ceiling": float(
                            np.nanmean([1.0 if r.get("hit_ceiling") else 0.0 for r in rows])
                        ),
                    }
                )

    # Persist
    res = pd.DataFrame(all_rows)
    res.to_csv(METRICS / "all_screening_results.csv", index=False)
    pd.DataFrame(screen_summary).to_csv(METRICS / "screening_summary.csv", index=False)

    # Save OOF for later residual/ensemble
    for (target, tag, model), vec in oof_bank.items():
        np.save(PREDS / "oof" / f"{target}__{tag}__{model}.npy", vec)

    # Absolute baselines report
    base = res[res.stage == "baseline"]
    lines = ["# Absolute-value baselines (Train outer CV MAE)\n\n"]
    for target in ["HIC", "TmApp"]:
        lines.append(f"## {target}\n\n")
        sub = base[base.target == target]
        g = sub.groupby(["tag", "model"])["mae"].agg(["mean", "std", "median"]).reset_index()
        g = g.sort_values("mean")
        lines.append(g.to_markdown(index=False) + "\n\n")
    (REPORTS / "absolute_baselines.md").write_text("".join(lines))

    # Protocol stub
    (REPORTS / "optuna_protocol.md").write_text(
        "# Optuna protocol (Gate B4)\n\n"
        "- Primary objective: minimize inner grouped-CV MAE\n"
        "- Sampler: TPESampler with recorded seed\n"
        "- Persistent SQLite under `optuna/*.db`\n"
        "- No pruning for sklearn/GBDT screening\n"
        "- GBDT: learning_rate fixed 0.03; n_estimators ceiling 5000; ES rounds 150 on internal group split\n"
        "- Outer validation never used for early stopping\n\n"
        f"Software: {json.dumps(software_versions())}\n"
    )

    elapsed = time.time() - t0
    write_json(LOGS / "01_screen_done.json", {"elapsed_s": elapsed, "n_rows": len(res), "n_oof": len(oof_bank)})
    print(f"DONE screening in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
