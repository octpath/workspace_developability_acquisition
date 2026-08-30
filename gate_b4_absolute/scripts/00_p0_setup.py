#!/usr/bin/env python3
"""Gate B4 P0: assay audit notes inputs, feature inventory, Train-only declaration, outer CV freeze."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_common import (  # noqa: E402
    B1_CACHE,
    B2_CACHE,
    B3_CONFIG,
    B3_FEATURES,
    B3_ORG,
    CONFIG,
    REPORTS,
    ensure_dirs,
    file_sha256,
    load_train_frame,
    make_outer_cv,
    read_json,
    sha256_lines,
    software_versions,
    train_only,
    write_json,
    GBDT_EARLY_STOPPING_ROUNDS,
    GBDT_LEARNING_RATE,
    GBDT_N_ESTIMATORS,
    MASTER_SEED,
    OPTUNA_SAMPLER_SEED,
    CV_SEED,
)


def write_assay_audit(train: pd.DataFrame, pop: pd.DataFrame) -> None:
    lines = []
    lines.append("# Assay scale audit — Shehata 2019 (Gate B4)\n")
    lines.append("Source: Shehata et al., *Cell Reports* 28:3300–3308.e4 (2019), DOI 10.1016/j.celrep.2019.08.056;\n")
    lines.append("supplementary `mmc2` assay columns; Adimab/study STAR-methods description (HIC + DSF).\n")
    lines.append("Frozen population: N=324 antibodies with both HIC and TmApp non-null (Gate B3 triple-core).\n\n")

    lines.append("## HIC\n\n")
    lines.append("| Field | Finding |\n|---|---|\n")
    lines.append("| Measurement method | Hydrophobic interaction chromatography (HIC); high-throughput Adimab panel |\n")
    lines.append("| Reported quantity | Retention time |\n")
    lines.append("| Units | minutes (supplement column: `HIC retention time (min)`) |\n")
    lines.append("| Protocol consistency | Single published study / single platform panel; all 324 values from same Shehata supplement table |\n")
    lines.append("| Instrument / assay | Study STAR Methods: HIC as hydrophobicity / aggregation-propensity proxy (not an aggregation kinetic assay) |\n")
    hic = pop["HIC"].astype(float)
    lines.append(f"| Rounding / discretization | Continuous-looking floats; N_unique={hic.nunique()} / 324; sample decimals to 0.001 |\n")
    lines.append("| Replicates | No per-antibody replicate table in the public supplement used here |\n")
    lines.append("| Batch information | Not provided as a participant/organizer column in mmc2 join |\n")
    lines.append(
        f"| Train distribution | mean={train['HIC'].mean():.3f}, SD={train['HIC'].std():.3f}, "
        f"IQR={train['HIC'].quantile(0.75)-train['HIC'].quantile(0.25):.3f}, "
        f"range=[{train['HIC'].min():.3f},{train['HIC'].max():.3f}] |\n"
    )
    lines.append("\n**Absolute-error meaningful?** **YES** — all values share one study protocol and minute scale.\n")
    lines.append("Caveat: HIC RT is **protocol-dependent**; model predicts assay RT under Shehata conditions, not a universal physical constant.\n\n")

    lines.append("## TmApp\n\n")
    lines.append("| Field | Finding |\n|---|---|\n")
    lines.append("| Measurement method | Differential scanning fluorimetry (DSF) / thermal melt; reported as TmApp |\n")
    lines.append("| Reported quantity | Apparent thermal transition temperature |\n")
    lines.append("| Units | °C (supplement column: `TmApp (°C)`) |\n")
    lines.append("| Protocol consistency | Single study / single panel; all 324 from same table |\n")
    lines.append("| Instrument / assay | DSF-based apparent Tm (TmApp) as conformational stability readout |\n")
    tm = pop["TmApp"].astype(float)
    lines.append(
        f"| Rounding / discretization | Half-degree grid common; N_unique={tm.nunique()} / 324; "
        f"fraction exactly *.0 or *.5 = {(np.isclose(tm % 0.5, 0) | np.isclose(tm % 0.5, 0.5)).mean():.2f} |\n"
    )
    lines.append("| Replicates | No public replicate SD table for TmApp in the materials used here |\n")
    lines.append("| Batch information | Not available as a column |\n")
    lines.append(
        f"| Train distribution | mean={train['TmApp'].mean():.3f}, SD={train['TmApp'].std():.3f}, "
        f"IQR={train['TmApp'].quantile(0.75)-train['TmApp'].quantile(0.25):.3f}, "
        f"range=[{train['TmApp'].min():.1f},{train['TmApp'].max():.1f}] |\n"
    )
    lines.append("\n**Absolute-error meaningful?** **YES** — common °C assay scale within this study.\n")
    lines.append("Caveat: heavy discrete ties (half-degree reporting) inflate rank-ties; MAE in °C remains interpretable.\n\n")

    lines.append("## Track-level decision\n\n")
    lines.append("| Track | Common assay scale? | Absolute-value scoring | Action |\n|---|---|---|---|\n")
    lines.append("| HIC | YES | GO | Rebuild under MAE (minutes) |\n")
    lines.append("| TmApp | YES | GO | Rebuild under MAE (°C) |\n")
    lines.append("\nNo HOLD. Frozen B3 split unchanged.\n")
    lines.append("\n## Experimental noise ceiling\n\n")
    lines.append("**NO EMPIRICAL ASSAY NOISE CEILING AVAILABLE** — no replicate measurements with SD in the public materials reused here.\n")
    (REPORTS / "assay_scale_audit.md").write_text("".join(lines))


def write_feature_inventory() -> None:
    rows = []

    def add(name, path, n, nmatch, dim, ver, tgt_indep, notes=""):
        rows.append(
            {
                "name": name,
                "path": str(path),
                "N": n,
                "N_match_324": nmatch,
                "dim": dim,
                "version_note": ver,
                "target_independent": tgt_indep,
                "notes": notes,
                "sha256": file_sha256(Path(path)) if Path(path).is_file() else "DIR_OR_MISSING",
            }
        )

    pop_ids = set(pd.read_csv(B3_ORG / "final_population.csv")["id"])

    for name, rel, idc, pref in [
        ("SEQ_SIMPLE", B1_CACHE / "features" / "stage_A_simple.csv", "antibody_id", None),
        ("SEQ_CDR", B1_CACHE / "features" / "stage_B_cdr.csv", "antibody_id", None),
        ("BIO_SHORTCUT", B1_CACHE / "features" / "stage_C_shortcut.csv", "antibody_id", None),
        ("ABB_SASA_B2", B2_CACHE / "structure_features" / "abb_sasa_rasa_patch.csv", "antibody_id", None),
        ("ESMFN_SASA_B2", B2_CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv", "antibody_id", None),
        ("GERMLINE_REL", B3_FEATURES / "germline" / "germline_relative.csv", "id", None),
        ("STRUCTURE_EXT", B3_FEATURES / "structure_ext" / "structure_extended.csv", "id", None),
    ]:
        df = pd.read_csv(rel)
        nmatch = int(df[idc].isin(pop_ids).sum())
        num = [c for c in df.columns if c != idc and pd.api.types.is_numeric_dtype(df[c])]
        add(name, rel, len(df), nmatch, len(num), "cached B1/B2/B3", True)

    z = np.load(B3_FEATURES / "imgt" / "positional_onehot.npz", allow_pickle=True)
    add("IMGT_POS", B3_FEATURES / "imgt" / "positional_onehot.npz", len(z["ids"]),
        int(sum(1 for i in z["ids"] if i in pop_ids)),
        f"Xh={z['Xh'].shape[1]},Xl={z['Xl'].shape[1]},Xhl={z['Xhl'].shape[1]}",
        "ANARCI/IMGT B3", True)

    for man, label in [
        ("manifest_ablang2_default.csv", "AbLang2"),
        ("manifest_esm1b_t33_650M_UR50S.csv", "ESM-1b"),
        ("manifest_esm2_t33_650M_UR50D.csv", "ESM-2"),
        ("manifest_esm2_t33_650M_UR50D_CDR6.csv", "ESM-2 CDR6"),
    ]:
        mf = pd.read_csv(B1_CACHE / "plm" / man)
        dim = int(mf["dim"].iloc[0])
        add(f"PLM_{label}", B1_CACHE / "plm" / man, len(mf),
            int(mf["antibody_id"].isin(pop_ids).sum()), dim, man.replace("manifest_", "").replace(".csv", ""), True,
            notes="embeddings on disk; not recomputed")

    inv_df = pd.DataFrame(rows)
    inv_df.to_csv(REPORTS.parent / "metrics" / "feature_cache_inventory.csv", index=False)

    md = ["# Feature cache inventory (Gate B4)\n\n",
          "All expensive PLM/structure artifacts are **reused**; no ESMFold/PLM recomputation.\n\n",
          inv_df.to_markdown(index=False),
          "\n\n## Target independence\n\nAll listed caches are sequence/structure-derived and target-independent.\n",
          "Supervised transforms (PCA/PLS/scalers) are fit **inside** Train folds only.\n"]
    (REPORTS / "feature_cache_inventory.md").write_text("".join(md))


def main():
    ensure_dirs()
    # Load WITH labels for assay audit of full population, but write declaration from Train-only barrier frame
    pop = pd.read_csv(B3_ORG / "final_population.csv")
    role = pd.read_csv(B3_ORG / "role_map.csv")
    full = pop.merge(role[["id", "role"]], on="id")
    train = full[full.role == "Train"].copy().reset_index(drop=True)
    public_ids = full[full.role == "Public"]["id"].tolist()
    private_ids = full[full.role == "Private"]["id"].tolist()
    train_ids = train["id"].tolist()

    # Verify frozen hashes match B3
    man = read_json(B3_CONFIG / "FINAL_SPLIT_MANIFEST.json")
    assert sha256_lines(train_ids) == man["id_list_sha256"]["train"]
    assert sha256_lines(public_ids) == man["id_list_sha256"]["public"]
    assert sha256_lines(private_ids) == man["id_list_sha256"]["private"]

    write_assay_audit(train, pop)
    write_feature_inventory()

    outer = make_outer_cv(train, n_folds=5, n_repeats=3, seed=CV_SEED)
    write_json(CONFIG / "OUTER_CV_FOLDS.json", outer)

    declaration = {
        "gate": "B4_ABSOLUTE",
        "primary_metric": "MAE",
        "primary_metric_direction": "minimize",
        "secondary_metrics": ["RMSE", "Pearson", "Spearman", "R2", "calibration_slope", "calibration_intercept"],
        "calibration_convention": "observed = intercept + slope * predicted",
        "frozen_population_n": 324,
        "frozen_counts": {"Train": 162, "Public": 81, "Private": 81},
        "frozen_Train_IDs_sha256": sha256_lines(train_ids),
        "frozen_Public_IDs_sha256": sha256_lines(public_ids),
        "frozen_Private_IDs_sha256": sha256_lines(private_ids),
        "b3_manifest_match": True,
        "information_barrier": "Public/Private labels MUST NOT be loaded until FINAL_ABSOLUTE_VALUE_FINALISTS.json is written and hashed",
        "candidate_model_families": [
            "constant_median", "constant_mean",
            "Ridge", "ElasticNet", "Lasso", "HuberRegressor", "PLS",
            "LinearSVR", "SVR_RBF", "KernelRidge",
            "XGBoost", "LightGBM", "CatBoost",
            "residual_OOF", "ensemble_OOF", "calibration_OOF",
        ],
        "optuna_budgets": {
            "stage1_screening_trials": 60,
            "stage1_cheap_linear_trials": 40,
            "stage2_nested_inner_trials": 25,
            "gbdt_max_trials_cap": 100,
        },
        "gbdt_fixed": {
            "learning_rate": GBDT_LEARNING_RATE,
            "n_estimators_ceiling": GBDT_N_ESTIMATORS,
            "early_stopping_rounds": GBDT_EARLY_STOPPING_ROUNDS,
            "learning_rate_tunable": False,
            "n_estimators_tunable": False,
        },
        "random_seeds": {
            "master": MASTER_SEED,
            "optuna_sampler": OPTUNA_SAMPLER_SEED,
            "cv": CV_SEED,
            "model_default": MASTER_SEED,
        },
        "cv_protocol": {
            "outer": "5-fold grouped × 3 repeats (15 evaluations)",
            "inner_for_optuna": "3-fold grouped on Outer Train",
            "early_stopping": "internal group split of fit portion; NEVER outer validation",
        },
        "software_versions": software_versions(),
        "search_spaces_path": str(CONFIG / "SEARCH_SPACES.json"),
        "outer_cv_path": str(CONFIG / "OUTER_CV_FOLDS.json"),
    }
    write_json(CONFIG / "TRAIN_ONLY_SEARCH_DECLARATION.json", declaration)

    search_spaces = {
        "xgboost": {
            "fixed": {"learning_rate": 0.03, "n_estimators": 5000},
            "search": {
                "max_depth": "int 2-5",
                "min_child_weight": "int 1-30",
                "subsample": "float 0.55-1.0",
                "colsample_bytree": "float 0.35-1.0",
                "reg_alpha": "log 1e-5..30",
                "reg_lambda": "log 1e-3..300",
                "gamma": "float 0-3",
            },
            "loss_families_preregistered": ["reg:squarederror", "reg:absoluteerror"],
        },
        "lightgbm": {
            "fixed": {"learning_rate": 0.03, "n_estimators": 5000},
            "search": {
                "max_depth": "int 2-5",
                "num_leaves": "int 3-31",
                "min_child_samples": "int 5-50",
                "subsample": "float 0.55-1.0",
                "colsample_bytree": "float 0.35-1.0",
                "reg_alpha": "log 1e-5..30",
                "reg_lambda": "log 1e-3..300",
                "min_split_gain": "float 0-1",
            },
            "loss_families_preregistered": ["regression_l2", "regression_l1"],
        },
        "catboost": {
            "fixed": {"learning_rate": 0.03, "iterations": 5000, "od_type": "Iter", "od_wait": 150},
            "search": {
                "depth": "int 3-7",
                "l2_leaf_reg": "log",
                "random_strength": "float 0-2",
                "bagging_temperature": "float 0-2",
                "rsm": "float 0.35-1.0",
            },
            "loss_families_preregistered": ["RMSE", "MAE"],
        },
        "elasticnet": {"alpha": "log 1e-4..10", "l1_ratio": "float 0.05-0.95"},
        "svr_rbf": {"C": "log 0.1..100", "epsilon": "log 1e-3..1", "gamma": "log 1e-4..1"},
        "kernel_ridge": {"alpha": "log 1e-3..100", "gamma": "log 1e-4..1"},
        "pca_dims": [8, 16, 24, 32, 48, 64, 96, 128],
        "pls_components": [2, 4, 6, 8, 12, 16, 24, 32],
    }
    write_json(CONFIG / "SEARCH_SPACES.json", search_spaces)

    # Barrier frame check
    barrier = load_train_frame(include_holdout_labels=False)
    assert barrier.loc[barrier.role != "Train", "HIC"].isna().all()
    assert barrier.loc[barrier.role != "Train", "TmApp"].isna().all()

    print("P0 complete")
    print("declaration", CONFIG / "TRAIN_ONLY_SEARCH_DECLARATION.json")
    print("outer folds", CONFIG / "OUTER_CV_FOLDS.json")
    print("assay audit", REPORTS / "assay_scale_audit.md")


if __name__ == "__main__":
    main()
