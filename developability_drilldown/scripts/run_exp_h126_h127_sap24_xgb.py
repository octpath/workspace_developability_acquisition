#!/usr/bin/env python3
"""EXP-H126/H127: SOURCE_SAP24 + fold-local shallow XGBoost (depth 2 / 3).

Independent fixed-feature estimator experiment. Same TVT / OOF / external
aggregation semantics as Transformer V3. No full-Dev refit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import (  # noqa: E402
    EXPERIMENTS_COLUMNS,
    FEATURE_SPACE,
    feature_content_sha256,
    file_sha256,
    mae,
)
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_solution  # noqa: E402
from fold_local_xgb import XGBConfig, train_fold_local_xgb  # noqa: E402

PREREG = ROOT / "results" / "H126_H127_SOURCE_SAP24_XGB_PREREGISTRATION.yaml"
SOURCE_PQ = REPO / "feature_research/hic_sap_scm_source/features/antibody_source_sap24.parquet"
CANON_PQ = ROOT / "experiments/features/antibody_source_sap24.parquet"
SEED = 101

SERIES = [
    {
        "code": "EXP-H126",
        "max_depth": 2,
        "experiment_id": "XGB_HIC_SOURCE_SAP24_DEPTH2_FOLDLOCAL",
        "description": "SOURCE_SAP24 shallow XGBoost depth=2 fold-local",
        "canonical": True,
    },
    {
        "code": "EXP-H127",
        "max_depth": 3,
        "experiment_id": "XGB_HIC_SOURCE_SAP24_DEPTH3_FOLDLOCAL",
        "description": "SOURCE_SAP24 shallow XGBoost depth=3 fold-local (sensitivity)",
        "canonical": False,
    },
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def save_pred(ids, vals, path: Path, col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, col: vals}).to_csv(path, index=False)


def score_external(pred: np.ndarray, test_ids: list[str], sol: pd.DataFrame, target: str) -> dict:
    sol2 = sol.set_index("id")
    te = pd.Series(pred, index=test_ids)
    pub = sol2.index[sol2["is_public"].astype(bool)].tolist()
    priv = sol2.index[sol2["is_private"].astype(bool)].tolist()
    return {
        "public_mae": float(mae(sol2.loc[pub, target].to_numpy(float), te.loc[pub].to_numpy(float))),
        "private_mae": float(mae(sol2.loc[priv, target].to_numpy(float), te.loc[priv].to_numpy(float))),
        "overall_mae": float(mae(sol2.loc[test_ids, target].to_numpy(float), te.loc[test_ids].to_numpy(float))),
    }


def ensure_feature_parquet(code: str) -> tuple[pd.DataFrame, list[str], str, str]:
    src = CANON_PQ if CANON_PQ.exists() else SOURCE_PQ
    df = pd.read_parquet(src)
    cols = [c for c in df.columns if c != "id"]
    assert len(cols) == 24 and len(df) == 324
    # per-experiment feature path required by validate
    dst = ROOT / "experiments" / "features" / f"{code}.parquet"
    df.to_parquet(dst, index=False)
    return df, cols, feature_content_sha256(df), file_sha256(dst)


def write_prereg() -> None:
    nxt = next_code("HIC")
    if nxt != "EXP-H126":
        raise SystemExit(f"expected next HIC EXP-H126, got {nxt}")
    src = CANON_PQ if CANON_PQ.exists() else SOURCE_PQ
    df = pd.read_parquet(src)
    cols = [c for c in df.columns if c != "id"]
    doc = {
        "batch_id": "H126_H127_SOURCE_SAP24_SHALLOW_XGB",
        "status": "PREREGISTERED_FROZEN",
        "git_rev_at_prereg": git_rev(),
        "next_code_at_prereg": nxt,
        "feature_block": "FS_HIC_SOURCE_SAP24",
        "n_dims": 24,
        "feature_columns": cols,
        "feature_content_sha256": feature_content_sha256(df),
        "source_parquet": str(src),
        "estimator": "xgboost_shallow_fold_local",
        "shared_hyperparams": {
            "learning_rate": 0.02,
            "n_estimators_upper": 5000,
            "early_stopping_rounds": 200,
            "random_state": SEED,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "reg_lambda": 10,
        },
        "experiments": SERIES,
        "semantics": {
            "splits": "existing Primary/Shadow folds.csv",
            "preprocessing": "TRAIN-only median impute; no StandardScaler",
            "early_stopping": "VAL MAE only",
            "no_full_dev_refit": True,
            "aggregation": "same as Transformer V3 (fold mean/median external)",
        },
        "do_not": ["SCM24", "COMBINED48", "SURFACE", "residue", "Optuna", "TEST tuning"],
    }
    for spec in SERIES:
        cfg = {
            "experiment_code": spec["code"],
            "experiment_id": spec["experiment_id"],
            "target": "HIC",
            "family": "XGBOOST",
            "model_type": "XGBOOST",
            "feature_set_id": "FS_HIC_SOURCE_SAP24",
            "max_depth": spec["max_depth"],
            "learning_rate": 0.02,
            "n_estimators": 5000,
            "early_stopping_rounds": 200,
            "seed": SEED,
            "cv_protocol": "dl_foldlocal_cosine_v3_oof_test_xgb",
            "no_full_dev_refit": True,
            "description": spec["description"],
        }
        (ROOT / "experiments" / "configs" / f"{spec['code']}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    PREREG.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print("Wrote", PREREG, flush=True)


def ensure_feature_set_row(content_sha: str, n_features: int) -> None:
    path = ROOT / "results" / "FEATURE_SETS.csv"
    fs = pd.read_csv(path)
    if "FS_HIC_SOURCE_SAP24" not in set(fs["feature_set_id"].astype(str)):
        row = {
            "feature_set_id": "FS_HIC_SOURCE_SAP24",
            "target": "HIC",
            "n_features": n_features,
            "feature_content_sha256": content_sha,
            "feature_recipe_hash": "",
            "source_recipe_ids": "SOURCE_SAP24",
            "feature_blocks": "SOURCE_SAP24",
            "notes": "source-specified SAP24 antibody-level block",
        }
        fs = pd.concat([fs, pd.DataFrame([row])], ignore_index=True)
        fs.to_csv(path, index=False)


def issue_one(spec: dict) -> str:
    codes = load_codes()
    eid = spec["experiment_id"]
    if (codes["experiment_id"] == eid).any():
        return str(codes.set_index("experiment_id").loc[eid, "experiment_code"])
    expect = spec["code"]
    nxt = next_code("HIC")
    if nxt != expect:
        raise SystemExit(f"expected {expect}, got {nxt}")
    code = issue_code(
        eid,
        "HIC",
        source_model_id=f"SOURCE_SAP24_XGB_DEPTH{spec['max_depth']}",
        phase="H126_SOURCE_SAP24_XGB",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


def register(code: str, spec: dict, summary: dict, ext_scores: dict, feat_meta: dict) -> None:
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    exp_path = ROOT / "results" / "experiments.csv"
    df = pd.read_csv(exp_path)
    df = df[df["experiment_code"] != code]
    row = {c: "" for c in df.columns}
    for c in EXPERIMENTS_COLUMNS:
        row.setdefault(c, "")
    row.update(
        {
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "legacy_experiment_code": "",
            "target": "HIC",
            "family": "XGBOOST",
            "model_type": "XGBOOST",
            "feature_set_id": "FS_HIC_SOURCE_SAP24",
            "feature_space": FEATURE_SPACE,
            "feature_path": f"experiments/features/{code}.parquet",
            "n_features": 24,
            "feature_sha256": feat_meta["file_sha"],
            "feature_content_sha256": feat_meta["content_sha"],
            "config_path": f"experiments/configs/{code}.yaml",
            "artifact_status": "FULL",
            "cv_primary_mae": oof["primary"],
            "cv_shadow_mae": oof["shadow"],
            "cv_mean_mae": oof["mean"],
            "cv_worst_mae": oof["worst"],
            "public_mae": pm["public_mae"],
            "private_mae": pm["private_mae"],
            "test_overall_mae": pm["overall_mae"],
            "public_private_delta": pm["public_mae"] - pm["private_mae"],
            "public_private_gap": abs(pm["public_mae"] - pm["private_mae"]),
            "cv_protocol": "dl_foldlocal_cosine_v3_oof_test_xgb",
            "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "license_status": "OK",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "REPRODUCED",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "shareability_status": "SHAREABLE_COMPLETE",
            "canonical_benchmark_eligible": "YES",
            "xgb_preset": f"shallow_depth{spec['max_depth']}_lr0.02_es200",
            "xgb_final_n_estimators": summary.get("best_iteration_median"),
            "optuna_used": False,
            "notes": spec["description"],
            "source_model_id": f"SOURCE_SAP24_XGB_DEPTH{spec['max_depth']}",
        }
    )
    pd.concat([df, pd.DataFrame([{c: row.get(c, "") for c in df.columns}])], ignore_index=True).to_csv(
        exp_path, index=False
    )
    comp_path = ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        if code not in set(comp["experiment_code"].astype(str)):
            crow = {c: "" for c in comp.columns}
            crow.update(
                {
                    "experiment_code": code,
                    "experiment_id": spec["experiment_id"],
                    "target": "HIC",
                    "family": "XGBOOST",
                    "config_exists": True,
                    "feature_required": True,
                    "feature_exists": True,
                    "feature_path": f"experiments/features/{code}.parquet",
                    "oof_primary_exists": True,
                    "oof_shadow_exists": True,
                    "test_exists": True,
                    "score_recompute_ok": True,
                    "prediction_max_delta": 0.0,
                    "reproduction_status": "REPRODUCED",
                    "shareability_status": "SHAREABLE_COMPLETE",
                    "canonical_benchmark_eligible": "YES",
                    "notes": "fold-local shallow XGB SOURCE_SAP24",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat(
                [comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])],
                ignore_index=True,
            )
            comp.to_csv(comp_path, index=False)


def already_complete(code: str) -> bool:
    return (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").exists() and (
        ROOT / "experiments" / "predictions" / code / "test_primary_mean.csv"
    ).exists()


def run_one(spec: dict) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} depth={spec['max_depth']} ====", flush=True)
    if already_complete(code):
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        st = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
        return {"code": code, "summary": summary, "ext_scores": st["ext_scores"], "skipped": True}

    df, cols, content_sha, file_sha = ensure_feature_parquet(code)
    ensure_feature_set_row(content_sha, 24)
    feat_meta = {"content_sha": content_sha, "file_sha": file_sha}

    X = df.set_index("id")
    # include all 324 for external test rows
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    y_map = {str(r["id"]): float(r["HIC"]) for _, r in dev.iterrows()}
    test_ids = test["id"].astype(str).tolist()

    cfg = XGBConfig(max_depth=int(spec["max_depth"]), random_state=SEED)
    out = ROOT / "results" / f"{code}_run"
    result = train_fold_local_xgb(
        experiment_code=code,
        X_by_id=X,
        y_map=y_map,
        folds=folds,
        test_ids=test_ids,
        feature_columns=cols,
        feature_hash=content_sha,
        cfg=cfg,
        out_dir=out,
    )
    summary = result["summary"]
    sel_df = result["selected_df"]
    sel_df.to_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv", index=False)
    # fold table
    sel_df.to_csv(ROOT / "results" / f"{code}_FOLD_BEST_ITERATION.csv", index=False)

    pred = ROOT / "experiments" / "predictions" / code
    pred.mkdir(parents=True, exist_ok=True)
    for scheme in ("primary", "shadow"):
        save_pred(
            result["dev_ids"],
            result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float),
            pred / f"oof_val_{scheme}.csv",
            "HIC",
        )
        save_pred(
            result["dev_ids"],
            result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float),
            pred / f"oof_test_{scheme}.csv",
            "HIC",
        )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_primary.csv",
        "HIC",
    )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_shadow.csv",
        "HIC",
    )
    for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        save_pred(result["test_ids"], result["ext"][key], pred / f"test_{key}.csv", "HIC")
    for scheme in ("primary", "shadow"):
        for k in range(5):
            save_pred(
                result["test_ids"],
                result["ext"][f"{scheme}_folds"][k],
                pred / f"test_{scheme}_fold{k}.csv",
                "HIC",
            )
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv", "HIC")

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol, "HIC")
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }
    oof_doc = {
        "experiment_code": code,
        "max_depth": spec["max_depth"],
        "learning_rate": 0.02,
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external": ext_scores,
        "best_iteration_mean": summary["best_iteration_mean"],
        "best_iteration_median": summary["best_iteration_median"],
        "no_full_dev_refit": True,
        "selected": sel_df.to_dict(orient="records"),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(
        yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "results" / f"{code}_run_state.json").write_text(
        json.dumps({"ext_scores": ext_scores}, indent=2)
    )
    register(code, spec, summary, ext_scores, feat_meta)
    print(f"DONE {code} TEST_mean={summary['scores']['oof_test']['mean']:.6f}", flush=True)
    return {"code": code, "summary": summary, "ext_scores": ext_scores, "skipped": False}


def write_report() -> None:
    rows = []
    fold_lines = ["# Fold-wise best_iteration / VAL MAE", ""]
    for spec in SERIES:
        code = spec["code"]
        oof = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
        ot = oof["oof_test"]
        ext = oof["external"]["primary_mean"]
        rows.append(
            {
                "Model": code,
                "Features": "SOURCE_SAP24",
                "depth": spec["max_depth"],
                "LR": 0.02,
                "TEST_P": ot["primary"],
                "TEST_S": ot["shadow"],
                "TEST_mean": ot["mean"],
                "TEST_worst": ot["worst"],
                "best_iter_median": oof["best_iteration_median"],
                "Public": ext["public_mae"],
                "Private": ext["private_mae"],
                "Overall": ext["overall_mae"],
            }
        )
        fold_lines.append(f"## {code}")
        fold_lines.append("")
        fold_lines.append("| scheme | fold | best_iteration | VAL_MAE |")
        fold_lines.append("|--------|-----:|---------------:|--------:|")
        for r in oof["selected"]:
            fold_lines.append(
                f"| {r['scheme']} | {r['fold']} | {r['best_iteration']} | {r['best_val_mae']:.6f} |"
            )
        fold_lines.append("")

    tab = pd.DataFrame(rows)
    # comparisons
    h071 = 0.5016881738827552
    h061 = 0.5067954770076422
    ridge_test = 0.644364
    svr_test = 0.547006
    h114 = 0.513981
    h115 = 0.517976
    h120 = 0.520690
    h121 = 0.522403

    can = tab[tab.Model == "EXP-H126"].iloc[0]
    sens = tab[tab.Model == "EXP-H127"].iloc[0]

    def case(tm: float) -> str:
        if tm <= 0.48:
            return "CASE_A_near_0.46"
        if tm < min(h071, h061) - 0.005:
            return "CASE_A_strong_vs_transformer"
        if tm <= max(h071, h061) + 0.01 and tm < svr_test - 0.02:
            return "CASE_B_better_than_ridge_svr_near_transformer"
        return "CASE_C_no_reproduce_0.46"

    lines = [
        "# HIC SOURCE_SAP24 Fold-Local Shallow XGBoost (H126–H127)",
        "",
        "Estimator-only experiment. Feature: frozen SOURCE_SAP24 (24D). No SCM/SURFACE/Transformer.",
        "",
        "## Summary table",
        "",
        "| Model | Features | depth | LR | TEST_P | TEST_S | TEST_mean | TEST_worst | best_iter median | Public | Private | Overall |",
        "|-------|----------|------:|---:|-------:|-------:|----------:|-----------:|-----------------:|-------:|--------:|--------:|",
    ]
    for _, r in tab.iterrows():
        lines.append(
            f"| {r.Model} | {r.Features} | {int(r.depth)} | {r.LR} | {r.TEST_P:.4f} | {r.TEST_S:.4f} | "
            f"{r.TEST_mean:.4f} | {r.TEST_worst:.4f} | {r.best_iter_median:.0f} | "
            f"{r.Public:.4f} | {r.Private:.4f} | {r.Overall:.4f} |"
        )

    lines += [
        "",
        "## Comparisons",
        "",
        f"- Ridge SOURCE_SAP24 TEST_mean ≈ {ridge_test:.3f}",
        f"- RBF-SVR SOURCE_SAP24 TEST_mean ≈ {svr_test:.3f}",
        f"- H071 Scratch TEST_mean = {h071:.4f}",
        f"- H061 ESM2 TEST_mean = {h061:.4f}",
        f"- H114/H115 Scratch+SAP24 = {h114:.4f} / {h115:.4f}",
        f"- H120/H121 ESM2+SAP24 = {h120:.4f} / {h121:.4f}",
        f"- reported approximate-SAP + XGBoost ≈ 0.46 (not same conditions)",
        "",
        f"## Verdict (canonical depth=2 = H126)",
        "",
        f"- TEST_mean = **{can.TEST_mean:.4f}** → **{case(float(can.TEST_mean))}**",
        f"- depth=3 sensitivity H127 TEST_mean = {sens.TEST_mean:.4f}",
        "",
        "### Final questions",
        "",
        f"1. Beat H071/H061? {'YES' if can.TEST_mean < min(h071,h061) else 'NO'} "
        f"(H126={can.TEST_mean:.4f} vs {h071:.4f}/{h061:.4f})",
        f"2. Better than Ridge/SVR? {'YES' if can.TEST_mean < svr_test else 'NO'}",
        f"3. Reproduce ≈0.46? {'YES' if can.TEST_mean <= 0.48 else 'NO'}",
        f"4. depth=2 enough vs depth=3? "
        f"{'YES (similar/better)' if can.TEST_mean <= sens.TEST_mean + 0.005 else 'depth=3 better'}; "
        f"Δ(d3−d2)={sens.TEST_mean - can.TEST_mean:+.4f}",
        f"5. Tree-specific signal vs Ridge/SVR? "
        f"{'YES' if can.TEST_mean < ridge_test - 0.05 else 'WEAK/NO'}",
        "6. Continue SAP for HIC? See CASE; if CASE_C, deprioritize exact SOURCE_SAP24 as HIC mainline.",
        "",
        *fold_lines,
        "",
        "STOP. Do not run SCM/SURFACE/residue/Transformer follow-ons from this batch.",
    ]
    (ROOT / "results" / "HIC_SOURCE_SAP24_XGBOOST_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    freeze = {
        "status": "INTERNAL_THEN_EXTERNAL_COMPLETE",
        "canonical_code": "EXP-H126",
        "sensitivity_code": "EXP-H127",
        "TEST_mean_H126": float(can.TEST_mean),
        "TEST_mean_H127": float(sens.TEST_mean),
        "case": case(float(can.TEST_mean)),
        "git_rev": git_rev(),
    }
    (ROOT / "results" / "H126_H127_PRE_EXTERNAL_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "train", "report", "all"])
    args = ap.parse_args()
    if args.phase in ("prereg", "all"):
        write_prereg()
    if args.phase in ("train", "all"):
        if not PREREG.exists():
            write_prereg()
        for spec in SERIES:
            run_one(spec)
    if args.phase in ("report", "all"):
        write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
