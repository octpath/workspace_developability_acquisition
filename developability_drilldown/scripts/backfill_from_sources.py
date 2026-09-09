#!/usr/bin/env python3
"""Backfill experiments.csv, configs, Linear predictions from authoritative sources."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    BUNDLE,
    CV_PROTOCOL,
    EXPERIMENTS_COLUMNS,
    FEATURE_SPACE,
    LIN_TOP6_MAP,
    ORG,
    ROOT,
    XGB_MAP,
    XGB_RECIPE,
    XGB_SOURCE_BY_EXP,
    derived_scores,
    feature_column_names,
    feature_content_sha256,
    feature_recipe_hash,
    file_sha256,
    lin_experiment_id,
    load_dev_test_folds,
    load_solution,
    mae,
    normalize_prediction_csv,
    recipe_blocks_from_recipes_csv,
    write_yaml,
    xgb_experiment_id,
)


def _f(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def empty_row() -> dict[str, Any]:
    return {c: "" for c in EXPERIMENTS_COLUMNS}


def fill_derived(row: dict[str, Any]) -> None:
    p, s = _f(row.get("cv_primary_mae")), _f(row.get("cv_shadow_mae"))
    pub, priv = _f(row.get("public_mae")), _f(row.get("private_mae"))
    if p is not None and s is not None:
        cv_mean, cv_worst, overall_from_pp, delta, gap = derived_scores(p, s, pub, priv)
        row["cv_mean_mae"] = cv_mean
        row["cv_worst_mae"] = cv_worst
        row["public_private_delta"] = delta if pub is not None else ""
        row["public_private_gap"] = gap if pub is not None else ""
        # overall: prefer solution-based if already set; else from PP mean; else authoritative
        if row.get("test_overall_mae") in ("", None) and pub is not None and priv is not None:
            row["test_overall_mae"] = overall_from_pp
            if not row.get("notes"):
                row["notes"] = ""
            if "overall_derivation" not in str(row.get("notes")):
                row["notes"] = (str(row.get("notes") or "") + " overall_derivation=mean(public,private)").strip()


def score_test_with_solution(pred_path: Path, target: str, sol: pd.DataFrame) -> tuple[float, float, float]:
    pred = pd.read_csv(pred_path)
    pred["id"] = pred["id"].astype(str)
    m = sol.merge(pred[["id", target]].rename(columns={target: "pred"}), on="id", how="inner")
    if len(m) != 162:
        raise RuntimeError(f"expected 162 test rows for scoring {pred_path}, got {len(m)}")
    y_true = m[target].to_numpy(float)
    y_pred = m["pred"].to_numpy(float)
    pub = m["is_public"].astype(bool).to_numpy()
    priv = m["is_private"].astype(bool).to_numpy()
    return mae(y_true[pub], y_pred[pub]), mae(y_true[priv], y_pred[priv]), mae(y_true, y_pred)


def attach_feature_meta(row: dict[str, Any], experiment_id: str) -> None:
    path = ROOT / "experiments" / "features" / f"{experiment_id}.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    cols = feature_column_names(df)
    row["feature_path"] = str(path.relative_to(ROOT))
    row["n_features"] = len(cols)
    row["feature_space"] = FEATURE_SPACE
    row["feature_sha256"] = file_sha256(path)
    row["feature_content_sha256"] = feature_content_sha256(df)
    # recipe hash from config or recompute via recipes
    recipe = row["feature_recipe"]
    try:
        blocks = recipe_blocks_from_recipes_csv(recipe)
    except KeyError:
        blocks = []
    row["feature_recipe_hash"] = feature_recipe_hash(blocks, cols)
    row["feature_source"] = "top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS"


def write_linear_full_config(eid: str, source_id: str, recipes: pd.DataFrame, alpha: float) -> Path:
    r = recipes[recipes["recipe_id"] == source_id].iloc[0]
    blocks = str(r["feature_blocks"]).split("|")
    target = str(r["target"])
    model_type = str(r["regressor"]).title() if str(r["regressor"]).upper() in ("RIDGE", "LASSO") else str(r["regressor"])
    data = {
        "experiment_id": eid,
        "target": target,
        "family": "LINEAR",
        "model_type": str(r["regressor"]).upper(),
        "source_model_id": source_id,
        "feature_recipe": source_id,
        "feature_blocks": blocks,
        "preprocessing": {
            "description": str(r["preprocessing"]),
            "fold_local": True,
            "applied_in_feature_parquet": False,
            "feature_parquet_space": FEATURE_SPACE,
        },
        "hyperparameters": {"alpha": alpha},
        "cv_protocol": CV_PROTOCOL,
        "selection_policy_at_creation": "CV_ONLY",
        "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
        "artifact_status": "FULL",
        "reproducibility": "EXACT_REPRODUCTION_from_bundle",
        "license_status": "SEE_feature_manifest",
        "source_paths": {
            "scores": "top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv",
            "oof_primary": f"organizer_extension/top3_ensemble_quickcheck/base_predictions/{source_id}__oof_primary.csv",
            "oof_shadow": f"organizer_extension/top3_ensemble_quickcheck/base_predictions/{source_id}__oof_shadow.csv",
            "test": f"organizer_extension/top3_ensemble_quickcheck/base_predictions/{source_id}__test.csv",
            "recipes": "top_models_feature_bundle/recipes.csv",
            "alpha_policy": "top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json",
        },
    }
    path = ROOT / "experiments" / "configs" / f"{eid}.yaml"
    write_yaml(path, data)
    return path


def write_xgb_full_config(eid: str, adv_row: pd.Series) -> Path:
    source_id = XGB_SOURCE_BY_EXP[eid]
    recipe = XGB_RECIPE[eid]
    blocks = recipe_blocks_from_recipes_csv(recipe)
    target = target_of_xgb(eid)
    data = {
        "experiment_id": eid,
        "target": target,
        "family": "XGBOOST",
        "model_type": "XGBRegressor",
        "source_model_id": source_id,
        "feature_recipe": recipe,
        "feature_blocks": blocks,
        "preprocessing": {
            "description": "fold-local median impute (+PCA32 on AbLingua blocks when applicable) + StandardScaler; same as Linear recipe path",
            "fold_local": True,
            "applied_in_feature_parquet": False,
            "feature_parquet_space": FEATURE_SPACE,
        },
        "hyperparameters": {
            "objective": "reg:absoluteerror",
            "learning_rate": 0.02,
            "early_stopping_rounds": 200,
            "n_estimators_cap": 10000,
            "random_state": 0,
            "tree_method": "hist",
            "xgb_preset": str(adv_row["final_preset"]),
            "final_n_estimators": int(float(adv_row["final_n_estimators"])),
        },
        "xgb_preset": str(adv_row["final_preset"]),
        "final_n_estimators": int(float(adv_row["final_n_estimators"])),
        "cv_protocol": CV_PROTOCOL,
        "selection_policy_at_creation": "CV_ONLY",
        "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
        "artifact_status": "FULL",
        "reproducibility": "FROZEN_ADVANCED_SUITE",
        "license_status": "SEE_feature_manifest",
        "source_paths": {
            "scores": "top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv",
            "benchmark_summary": "top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv",
            "presets": "top_models_feature_bundle/advanced_models/presets.json",
            "test_predictions": f"top_models_feature_bundle/advanced_outputs/predictions/{target}__xgboost__{recipe}__test.csv",
            "oof_rebuild": "developability_drilldown/scripts/rebuild_xgb_oof.py",
        },
    }
    path = ROOT / "experiments" / "configs" / f"{eid}.yaml"
    write_yaml(path, data)
    return path


def target_of_xgb(eid: str) -> str:
    return "TmApp" if "_TM_" in eid else "HIC"


def backfill_linear_top6(summary: pd.DataFrame, recipes: pd.DataFrame, alpha_pol: dict, sol) -> list[dict]:
    base_pred = ORG / "top3_ensemble_quickcheck" / "base_predictions"
    expected = json.loads((BUNDLE / "EXPECTED_SCORES.json").read_text())
    rows = []
    for source_id, eid in LIN_TOP6_MAP.items():
        srow = summary[(summary["model_id"] == source_id) & (summary["family"] == "LINEAR")]
        if len(srow) != 1:
            raise RuntimeError(f"missing SUMMARY row for {source_id}")
        s = srow.iloc[0]
        target = str(s["target"])
        rmeta = recipes[recipes["recipe_id"] == source_id].iloc[0]
        alpha_key_base = source_id.rsplit("__", 1)[0]
        reg = str(rmeta["regressor"]).upper()
        alpha = float(alpha_pol["final_alpha_by_recipe_regressor"][f"{alpha_key_base}::{reg}"])
        cfg = write_linear_full_config(eid, source_id, recipes, alpha)

        dev, test, _ = load_dev_test_folds()
        pred_dir = ROOT / "experiments" / "predictions" / eid
        normalize_prediction_csv(
            base_pred / f"{source_id}__oof_primary.csv",
            target,
            dev["id"].tolist(),
            pred_dir / "oof_primary.csv",
        )
        normalize_prediction_csv(
            base_pred / f"{source_id}__oof_shadow.csv",
            target,
            dev["id"].tolist(),
            pred_dir / "oof_shadow.csv",
        )
        normalize_prediction_csv(
            base_pred / f"{source_id}__test.csv",
            target,
            test["id"].tolist(),
            pred_dir / "test.csv",
        )

        # cross-check expected
        exp = expected[source_id]
        assert abs(float(s["cv_primary_mae"]) - exp["expected_primary_mae"]) < 1e-9

        row = empty_row()
        row.update(
            {
                "experiment_id": eid,
                "target": target,
                "family": "LINEAR",
                "model_type": reg,
                "source_model_id": source_id,
                "feature_recipe": source_id,
                "cv_primary_mae": float(s["cv_primary_mae"]),
                "cv_shadow_mae": float(s["cv_shadow_mae"]),
                "public_mae": float(s["public_mae"]),
                "private_mae": float(s["private_mae"]),
                "test_overall_mae": float(s["overall_test_mae"]),
                "cv_protocol": CV_PROTOCOL,
                "selection_policy_at_creation": "CV_ONLY",
                "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
                "artifact_status": "FULL",
                "config_path": str(cfg.relative_to(ROOT)),
                "oof_primary_path": str((pred_dir / "oof_primary.csv").relative_to(ROOT)),
                "oof_shadow_path": str((pred_dir / "oof_shadow.csv").relative_to(ROOT)),
                "test_prediction_path": str((pred_dir / "test.csv").relative_to(ROOT)),
                "linear_alpha": alpha,
                "score_source": "top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv",
                "prediction_source": "organizer_extension/top3_ensemble_quickcheck/base_predictions/",
                "reproducible": "YES",
                "reproduction_status": "PASS",
                "license_status": "SEE_feature_manifest",
                "notes": "canonical Top-3 Linear; overall_derivation=authoritative_overall_test_mae",
            }
        )
        if sol is not None:
            pub_r, priv_r, overall_r = score_test_with_solution(pred_dir / "test.csv", target, sol)
            # keep authoritative scores; note reconstructed
            row["notes"] += f" local_solution_check pub={pub_r:.6g} priv={priv_r:.6g} overall={overall_r:.6g}"
        fill_derived(row)
        # restore authoritative overall (fill_derived may overwrite)
        row["test_overall_mae"] = float(s["overall_test_mae"])
        attach_feature_meta(row, eid)
        rows.append(row)
    return rows


def backfill_feature_linear_rest(existing_ids: set[str]) -> list[dict]:
    reg_path = ORG / "linear_model_closure" / "LINEAR_MODEL_MASTER_REGISTRY.csv"
    reg = pd.read_csv(reg_path)
    fl = reg[(reg["result_class"] == "FEATURE_LINEAR") & (reg["cv_protocol"] == "canonical_simple_tvt_v1")]
    rows = []
    for _, r in fl.iterrows():
        source_id = str(r["model_id"])
        eid = lin_experiment_id(source_id)
        if eid in existing_ids:
            continue  # Top-6 already FULL
        target = str(r["target"])
        regn = str(r["regressor"]).upper()
        pub, priv = _f(r["public_mae"]), _f(r["private_mae"])
        row = empty_row()
        status = "SCORE_ONLY"
        notes = []
        if str(r.get("validity")) == "CV_ONLY_NO_TEST_FEATURES":
            notes.append("CV_ONLY_NO_TEST_FEATURES; public/private NA")
            status = "SCORE_ONLY"
        if "CONSTANT" in source_id or str(r.get("feature_blocks_tokens")) in ("none", "nan", ""):
            notes.append("constant/no-feature baseline")
        if int(r.get("n_dev") or 162) != 162:
            notes.append(f"n_dev={r.get('n_dev')} (nonstandard cohort)")
        # endgame test preds exist for many — mark RECONSTRUCTABLE if prediction file exists
        endgame = ORG / "endgame_model_benchmark" / "predictions" / f"{source_id}.csv"
        if endgame.exists() and status == "SCORE_ONLY":
            status = "RECONSTRUCTABLE"
            notes.append("test predictions available under endgame_model_benchmark/predictions (not materialized in Phase 1)")

        row.update(
            {
                "experiment_id": eid,
                "target": target,
                "family": "LINEAR",
                "model_type": regn,
                "source_model_id": source_id,
                "feature_recipe": source_id,
                "cv_primary_mae": float(r["cv_primary_mae"]),
                "cv_shadow_mae": float(r["cv_shadow_mae"]),
                "public_mae": pub if pub is not None else "",
                "private_mae": priv if priv is not None else "",
                "cv_protocol": CV_PROTOCOL,
                "selection_policy_at_creation": "CV_ONLY",
                "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
                "artifact_status": status,
                "linear_alpha": _f(r["final_alpha"]) if _f(r["final_alpha"]) is not None else "",
                "score_source": "organizer_extension/linear_model_closure/LINEAR_MODEL_MASTER_REGISTRY.csv",
                "prediction_source": "",
                "feature_source": str(r.get("feature_source_files") or ""),
                "reproducible": "YES" if str(r.get("canonical_replay_status")) == "REPLAYED_CANONICAL" else "PARTIAL",
                "reproduction_status": str(r.get("canonical_replay_status") or ""),
                "license_status": "SEE_feature_manifest",
                "notes": "; ".join(notes),
            }
        )
        fill_derived(row)
        if pub is not None and priv is not None and row.get("test_overall_mae") not in ("", None):
            pass
        rows.append(row)
    return rows


def backfill_xgb(summary: pd.DataFrame, adv: pd.DataFrame, sol) -> list[dict]:
    rows = []
    for source_id, eid in XGB_MAP.items():
        arow = adv[adv["variant_id"] == source_id]
        if len(arow) != 1:
            raise RuntimeError(f"missing ADVANCED row {source_id}")
        a = arow.iloc[0]
        srow = summary[summary["model_id"] == source_id]
        if len(srow) != 1:
            raise RuntimeError(f"missing SUMMARY row {source_id}")
        s = srow.iloc[0]
        target = str(s["target"])
        cfg = write_xgb_full_config(eid, a)
        pred_dir = ROOT / "experiments" / "predictions" / eid
        for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
            if not (pred_dir / name).exists():
                raise FileNotFoundError(
                    f"missing {pred_dir / name}; run rebuild_xgb_oof.py first"
                )
        row = empty_row()
        row.update(
            {
                "experiment_id": eid,
                "target": target,
                "family": "XGBOOST",
                "model_type": "XGBRegressor",
                "source_model_id": source_id,
                "feature_recipe": XGB_RECIPE[eid],
                "cv_primary_mae": float(s["cv_primary_mae"]),
                "cv_shadow_mae": float(s["cv_shadow_mae"]),
                "public_mae": float(s["public_mae"]),
                "private_mae": float(s["private_mae"]),
                "test_overall_mae": float(s["overall_test_mae"]),
                "cv_protocol": CV_PROTOCOL,
                "selection_policy_at_creation": "CV_ONLY",
                "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
                "artifact_status": "FULL",
                "config_path": str(cfg.relative_to(ROOT)),
                "oof_primary_path": str((pred_dir / "oof_primary.csv").relative_to(ROOT)),
                "oof_shadow_path": str((pred_dir / "oof_shadow.csv").relative_to(ROOT)),
                "test_prediction_path": str((pred_dir / "test.csv").relative_to(ROOT)),
                "xgb_preset": str(a["final_preset"]),
                "xgb_final_n_estimators": int(float(a["final_n_estimators"])),
                "score_source": "top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv",
                "prediction_source": "advanced_outputs test + rebuild_xgb_oof OOF",
                "reproducible": "YES",
                "reproduction_status": "PENDING_AUDIT",
                "license_status": "SEE_feature_manifest",
                "notes": "overall_derivation=authoritative_overall_test_mae",
            }
        )
        if sol is not None:
            pub_r, priv_r, overall_r = score_test_with_solution(pred_dir / "test.csv", target, sol)
            row["notes"] += f" local_solution_check pub={pub_r:.6g} priv={priv_r:.6g} overall={overall_r:.6g}"
        fill_derived(row)
        row["test_overall_mae"] = float(s["overall_test_mae"])
        attach_feature_meta(row, eid)
        rows.append(row)
    return rows


def main() -> None:
    codes_path = ROOT / "results" / "EXPERIMENT_CODES.csv"
    if codes_path.exists():
        raise SystemExit(
            "EXPERIMENT_CODES.csv exists — Phase 1 backfill is frozen. "
            "Do not re-run backfill_from_sources.py (would risk renumbering). "
            "Use migrate_registry_exp_codes.py / catalog rebuild / export_features instead."
        )
    summary = pd.read_csv(BUNDLE / "results" / "MODEL_BENCHMARK_SUMMARY.csv")
    recipes = pd.read_csv(BUNDLE / "recipes.csv")
    alpha_pol = json.loads((BUNDLE / "FULL_DEV_ALPHA_POLICY_BUNDLE.json").read_text())
    adv = pd.read_csv(BUNDLE / "advanced_models" / "frozen_results" / "ADVANCED_MODEL_RESULTS.csv")
    sol = load_solution()

    linear_full = backfill_linear_top6(summary, recipes, alpha_pol, sol)
    existing = {r["experiment_id"] for r in linear_full}
    linear_rest = backfill_feature_linear_rest(existing)
    # XGB requires predictions already present
    xgb_rows = backfill_xgb(summary, adv, sol)

    all_rows = linear_full + linear_rest + xgb_rows
    ids = [r["experiment_id"] for r in all_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate experiment_id")
    df = pd.DataFrame(all_rows, columns=EXPERIMENTS_COLUMNS)
    out = ROOT / "results" / "experiments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")
    print(df["artifact_status"].value_counts().to_string())
    print(df["family"].value_counts().to_string())


if __name__ == "__main__":
    main()
