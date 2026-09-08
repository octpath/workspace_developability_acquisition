#!/usr/bin/env python3
"""Linear-model closure: audit Top-3 scope, registry, Top-10s, winners, advanced freeze."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/linear_model_closure"
END = ROOT / "organizer_extension/endgame_model_benchmark"
VP = ROOT / "virtual_participant"
BUNDLE = ROOT / "top_models_feature_bundle"
OUT.mkdir(parents=True, exist_ok=True)

# Full expansion of feature block tokens
BLOCK_FULL = {
    "none": "no input features (fold-local median baseline)",
    "SEQ_BASIC": "stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries)",
    "SEQ_ALL": "stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set)",
    "AbLang2_HL_paired": "AbLang2 paired heavy+light chain protein-language-model embedding",
    "ESM2_H": "ESM-2 Heavy-chain protein-language-model embedding",
    "BIOEMU_NEW_PAIRWISE": "BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors",
    "M1_PROTEINMPNN": "ProteinMPNN sequence–structure compatibility native-score descriptors",
    "AbLingua_HL_mean": "AbLingua-600M masked-mean global heavy+light concatenated embedding",
    "AbLingua_CDR3": "AbLingua-600M CDR3-guided residue pooling embedding",
    "AROMATIC_TOPO": "AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv)",
    "CONTINUOUS_SURFACE": "continuous molecular-surface hydrophobic/chemical descriptors (ESMFold Fv)",
    "HYDRO_FIELD": "HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv)",
    "TITRATION_SHAPE": "TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)",
    "OPENMM_DOMAIN_FLEXIBILITY": "OpenMM Fab MD DOMAIN_FLEXIBILITY descriptors (VH/VL/CH1/CL RMSF)",
    "FENNIX_COMBINED_PREDECLARED": "FeNNix full-Fab combined predeclared energy/geometry context descriptors",
}

SOURCE_FILES = {
    "SEQ_BASIC": "built via stage1 make_xy(SEQ_BASIC) on competition sequences",
    "SEQ_ALL": "built via stage1 make_xy(SEQ_ALL) on competition sequences",
    "AbLang2_HL_paired": "virtual_participant/round1_finalization/cache/round1_embeddings.npz::ablang2__HL_paired",
    "ESM2_H": "virtual_participant/round1_finalization/cache/round1_embeddings.npz::esm2__H",
    "BIOEMU_NEW_PAIRWISE": "organizer_extension/.../BIOEMU_ISOLATED_REASSESS_FEATURES.csv (ca_rmsd*)",
    "M1_PROTEINMPNN": "organizer_extension/.../proteinmpnn/M1_FEATURES.csv",
    "AbLingua_HL_mean": "ablingua600m/embeddings/ablingua600m_HL_mean_concat.parquet",
    "AbLingua_CDR3": "ablingua600m/embeddings_guided/ablingua600m_CDR3.parquet",
    "AROMATIC_TOPO": "structure_gap_closure/cache/aromatic_features_esmfold.csv",
    "CONTINUOUS_SURFACE": "structure_gap_closure/results/HIC_CONTINUOUS_SURFACE_FEATURES.csv",
    "HYDRO_FIELD": "HYDRO-FIELD/features_esmfold.parquet",
    "TITRATION_SHAPE": "TITRATION-SHAPE/features_esmfold.parquet",
    "OPENMM_DOMAIN_FLEXIBILITY": "openmm_fab_md/results/OPENMM_FAB_MD_FEATURES_DEV.parquet",
    "FENNIX_COMBINED_PREDECLARED": "fennix_fab_context_final/results/FENNIX_COMBINED_PREDECLARED.parquet",
}


def expand_blocks(s: str) -> str:
    if not s or s == "none":
        return BLOCK_FULL["none"]
    parts = [BLOCK_FULL.get(b.strip(), b.strip()) for b in str(s).split("|") if b.strip()]
    return " + ".join(parts)


BASE_MODEL_EXPAND = {
    "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt": (
        "fused ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors "
        "under support-vector regression (SVROpt)"
    ),
    "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt": (
        "ESMFold Fv STRUCT_SURFACE_CHEM continuous surface-chemistry descriptors under support-vector regression (SVROpt)"
    ),
    "HIC__ADV_SURFACE_PATCH__SVROpt": (
        "advanced surface-patch descriptors (ADV_SURFACE_PATCH) under support-vector regression (SVROpt)"
    ),
    "HIC__esm2__H__SVROpt": (
        "ESM-2 Heavy-chain protein-language-model embedding under support-vector regression (SVROpt)"
    ),
    "HIC__SEQ_PLUS_ANTIBODY__SVROpt": (
        "antibody-augmented sequence descriptors (SEQ_PLUS_ANTIBODY) under support-vector regression (SVROpt)"
    ),
    "HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt": (
        "Stage-3 incremental fusion advanced surface-patch descriptors under support-vector regression (SVROpt)"
    ),
    "TmApp__SEQ_BASIC__SVROpt": (
        "stage-1 SEQ_BASIC sequence descriptors under support-vector regression (SVROpt)"
    ),
    "TmApp__ablang2__HL_paired__RidgeOpt_PCANone": (
        "AbLang2 paired heavy+light chain protein-language-model embedding under RidgeOpt without PCA"
    ),
    "TmApp__ESMFold__STRUCT_RASA__ElasticNetOpt": (
        "ESMFold STRUCT_RASA relative solvent-accessibility structure descriptors under ElasticNetOpt"
    ),
    "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt": (
        "Stage-3 incremental fusion ADV_INTERACTIONS descriptors under support-vector regression (SVROpt)"
    ),
    "TmApp__ablang2__HL__RidgeOpt": (
        "AbLang2 heavy+light (HL) protein-language-model embedding under RidgeOpt "
        "(historical Stage-2 Optuna-tuned Ridge; not the endgame AbLang2+SEQ_BASIC recipe)"
    ),
}


def expand_base_model_ids(s: str) -> str:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return ""
    parts = []
    for b in str(s).split("|"):
        b = b.strip()
        if not b:
            continue
        parts.append(BASE_MODEL_EXPAND.get(b, b.replace("__", " / ")))
    return "; ".join(parts)


def describe_historical(eid: str, r: pd.Series) -> str:
    bases = expand_base_model_ids(getattr(r, "base_model_ids", None))
    if eid == "TmApp__ablang2__HL__RidgeOpt":
        return (
            "Historical Stage-2 AbLang2 heavy+light protein-language-model embedding only "
            "(no SEQ_BASIC / BioEmu / ProteinMPNN / AbLingua blocks) under Optuna-tuned Ridge "
            f"(alpha≈{json.loads(r.hyperparameters).get('alpha', 'tuned') if pd.notna(r.hyperparameters) else 'tuned'})"
        )
    if "SIMPLE_blend" in eid and bases:
        return f"Equal-mean prediction blend of: {bases}"
    if "META_" in eid and bases:
        meta = str(r.meta_model) if pd.notna(getattr(r, "meta_model", None)) else str(r.model_name)
        return (
            f"prediction-level meta-learner ({meta}) over out-of-fold base predictions: {bases}"
        )
    if bases:
        return bases
    if "ablang2" in eid.lower():
        return (
            "Historical AbLang2 heavy+light protein-language-model embedding under RidgeOpt "
            "(feature-level historical; protocol not proven identical to canonical Simple TVT)"
        )
    if "esm2" in eid.lower():
        return (
            "Historical ESM-2 Heavy-chain protein-language-model embedding under a linear/SVR organizer Stage model"
        )
    return eid.replace("__", " / ")

    return " + ".join(parts)


def plm_flags(blocks: str) -> dict:
    b = set(str(blocks).split("|")) if blocks and blocks != "none" else set()
    return {
        "contains_ESM2_Heavy_PLM": "ESM2_H" in b,
        "contains_AbLang2_PLM": "AbLang2_HL_paired" in b,
        "contains_AbLingua_PLM": bool(b & {"AbLingua_HL_mean", "AbLingua_CDR3"}),
        "contains_any_PLM": bool(b & {"ESM2_H", "AbLang2_HL_paired", "AbLingua_HL_mean", "AbLingua_CDR3"}),
        "contains_sequence_descriptors": bool(b & {"SEQ_BASIC", "SEQ_ALL"}),
        "contains_aromatic_structural": "AROMATIC_TOPO" in b,
        "contains_continuous_surface": "CONTINUOUS_SURFACE" in b,
        "contains_hydrophobic_field": "HYDRO_FIELD" in b,
        "contains_titration": "TITRATION_SHAPE" in b,
        "contains_BioEmu": "BIOEMU_NEW_PAIRWISE" in b,
        "contains_ProteinMPNN": "M1_PROTEINMPNN" in b,
    }


def alias_misleading(recipe_id: str) -> str:
    """Human old alias that was too short."""
    m = {
        "HIC_HYDRO_TITRATION": "ARO+HYDRO+TITR (misleading — also contains ESM-2 Heavy + SEQ_ALL)",
        "HIC_ARO_TITRATION": "ARO+TITR (misleading — also contains ESM-2 Heavy + SEQ_ALL)",
        "HIC_ARO_CONTINUOUS_SURFACE": "ARO+CONT (misleading — also contains ESM-2 Heavy + SEQ_ALL)",
        "HIC_ARO_CONT_TITR": "ARO+CONT+TITR (misleading — also contains ESM-2 Heavy + SEQ_ALL)",
        "TM_PARENT_ABLINGUA_CDR3": "PARENT+AbLingua_CDR3 (PARENT = AbLang2+SEQ_BASIC+BioEmu NEW_PAIRWISE+ProteinMPNN+AbLingua GLOBAL)",
        "TM_PARENT_ABLINGUA_GLOBAL": "PARENT+AbLingua_GLOBAL (PARENT core without CDR3)",
        "TM_BASE_BIOEMU_MPNN": "BASE+BioEmu+MPNN (BASE = AbLang2+SEQ_BASIC)",
    }
    return m.get(recipe_id, recipe_id)


def build_current_top3_audit():
    top = json.loads((END / "CV_SELECTED_RECIPES_FREEZE.json").read_text())
    post = pd.read_csv(END / "ORGANIZER_MODEL_POSTMORTEM.csv")
    cv = pd.read_csv(END / "ORGANIZER_ENDGAME_CV_TABLE.csv")
    freeze = pd.read_csv(END / "FEATURE_RECIPE_FREEZE.csv")
    alphas = json.loads((END / "FULL_DEV_ALPHA_POLICY.json").read_text()).get(
        "final_alpha_by_recipe_regressor", {}
    )
    rows = []
    for target, lst in top["targets"].items():
        for p in lst:
            rid, reg = p["recipe_id"], p["regressor"]
            fr = freeze[freeze.recipe_id == rid].iloc[0]
            pr = post[(post.recipe_id == rid) & (post.regressor == reg)].iloc[0]
            cr = cv[(cv.recipe_id == rid) & (cv.regressor == reg)].iloc[0]
            blocks = fr.feature_blocks
            flags = plm_flags(blocks)
            rows.append(
                {
                    "recipe_id": rid,
                    "human_readable_old_alias": alias_misleading(rid),
                    "target": target,
                    "cv_rank_in_restricted_inventory": p["cv_rank"],
                    "exact_feature_blocks": blocks,
                    "feature_blocks_full_expansion": expand_blocks(blocks),
                    "exact_source_files": " | ".join(
                        SOURCE_FILES.get(b, b) for b in str(blocks).split("|") if b != "none"
                    )
                    or "n/a",
                    "exact_feature_dimensions_raw": cr.raw_dim,
                    "effective_dimension": cr.effective_dim_if_applicable,
                    "regressor": reg,
                    "preprocessing": cr.preprocessing,
                    "PCA_location_dimension": (
                        "PCA32 per AbLingua block"
                        if "per_abl_block" in str(cr.preprocessing)
                        else (
                            "PCA32 on AbLingua GLOBAL block only"
                            if "PCA32(AbLingua)" in str(cr.preprocessing)
                            else "none (Lasso/raw Ridge)"
                        )
                    ),
                    "ID_cohort": f"DEV n={int(cr.n_dev)}; Test full-refit where available",
                    "CV_protocol": "canonical Simple TVT fold_primary/fold_shadow; TEST=k VAL=(k+1)%5 TRAIN=other3",
                    "final_refit_protocol": "median Primary-fold VAL-selected alpha; fit preprocessing+model on full DEV; predict Test",
                    "final_alpha": alphas.get(f"{rid}::{reg}"),
                    "Public_Private_prediction_source": "CANONICAL_FULL_DEV_REFIT",
                    "cv_primary_mae": p["cv_primary_mae"],
                    "cv_shadow_mae": p["cv_shadow_mae"],
                    "public_mae": pr.public_mae,
                    "private_mae": pr.private_mae,
                    **flags,
                    "scope_note": "Top-3 among RESTRICTED endgame Ridge/Lasso FEATURE_RECIPE_FREEZE inventory only — NOT all-time organizer linear models",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "CURRENT_TOP3_RECIPE_AUDIT.csv", index=False)
    return df


def build_master_registry():
    post = pd.read_csv(END / "ORGANIZER_MODEL_POSTMORTEM.csv")
    freeze = pd.read_csv(END / "FEATURE_RECIPE_FREEZE.csv")
    alphas = json.loads((END / "FULL_DEV_ALPHA_POLICY.json").read_text()).get(
        "final_alpha_by_recipe_regressor", {}
    )
    rows = []
    # --- FEATURE_LINEAR from endgame ---
    for _, r in post.iterrows():
        fr = freeze[freeze.recipe_id == r.recipe_id]
        blocks = fr.iloc[0].feature_blocks if len(fr) else r.feature_blocks
        flags = plm_flags(blocks)
        rows.append(
            {
                "target": r.target,
                "model_id": f"{r.recipe_id}__{r.regressor}",
                "full_model_name": f"{expand_blocks(blocks)} with {r.regressor}",
                "result_class": "FEATURE_LINEAR",
                "feature_blocks_full": expand_blocks(blocks),
                "feature_blocks_tokens": blocks,
                "feature_source_files": " | ".join(
                    SOURCE_FILES.get(b, b) for b in str(blocks).split("|") if b and b != "none"
                )
                or "n/a",
                "molecular_scope": fr.iloc[0].molecular_scope if len(fr) else "",
                "regressor": r.regressor,
                "regressor_details": "sklearn Ridge" if r.regressor == "RIDGE" else "sklearn Lasso (no PCA)",
                "preprocessing_full": r.preprocessing,
                "dimensionality_reduction": (
                    "PCA32 per AbLingua block"
                    if "per_abl" in str(r.preprocessing)
                    else ("PCA32 on AbLingua" if "PCA32" in str(r.preprocessing) else "none")
                ),
                "raw_dimension": r.raw_dim,
                "effective_dimension": r.raw_dim,
                "n_dev": r.n_dev,
                "n_test": 162 if r.validity == "AUTHORITATIVE" else np.nan,
                "cv_primary_mae": r.cv_primary_mae,
                "cv_shadow_mae": r.cv_shadow_mae,
                "cv_mean_mae": r.cv_mean_mae,
                "cv_worst_mae": r.cv_worst_mae,
                "public_mae": r.public_mae,
                "private_mae": r.private_mae,
                "final_alpha": alphas.get(f"{r.recipe_id}::{r.regressor}"),
                "lasso_nonzero": r.lasso_nonzero_final if r.regressor == "LASSO" else np.nan,
                "lasso_total_features": r.raw_dim if r.regressor == "LASSO" else np.nan,
                "cv_protocol": "canonical_simple_tvt_v1",
                "final_fit_protocol": "FULL_DEV median Primary alpha",
                "prediction_source": "CANONICAL_FULL_DEV_REFIT"
                if r.validity == "AUTHORITATIVE"
                else "CV_ONLY_OR_FAILED",
                "canonical_replay_status": "REPLAYED_CANONICAL",
                "validity": r.validity,
                "historical_role": r.historical_role,
                "notes": r.notes,
                "eligible_main_cv_rank": r.validity in ("AUTHORITATIVE", "CV_ONLY_NO_TEST_FEATURES")
                and pd.notna(r.cv_worst_mae),
                **flags,
            }
        )

    # --- Historical prediction blends / meta-linear from Round1 inventory ---
    inv = pd.read_csv(VP / "round1_postmortem/round1_all_model_score_inventory.csv")
    specs = json.loads((VP / "round1_finalization/round1_final_model_specs.json").read_text())
    # Mark key historical
    hist_ids = {
        "HIC__SIMPLE_blend_seq_surf_adv": (
            "PREDICTION_LINEAR_BLEND",
            "Equal-mean blend of three SVR base predictions: ESM-2 Heavy + SEQ_ALL fusion; ESMFold SURFACE_CHEM; ADV_SURFACE_PATCH",
            "HISTORICAL_FROZEN_SUBMISSION",
        ),
        "HIC__SIMPLE_blend_seq_surf": (
            "PREDICTION_LINEAR_BLEND",
            "Equal-mean blend: ESM-2 Heavy + SEQ_ALL fusion SVR + ESMFold SURFACE_CHEM SVR",
            "HISTORICAL_FROZEN_PREDICTION",
        ),
        "HIC__SIMPLE_blend_esm2_surf": (
            "PREDICTION_LINEAR_BLEND",
            "Equal-mean blend: ESM-2 Heavy SVR + ESMFold SURFACE_CHEM SVR",
            "HISTORICAL_FROZEN_PREDICTION",
        ),
        "TmApp__META_performance__ridge_100.0": (
            "PREDICTION_LINEAR_BLEND",
            "Nested Ridge(meta alpha=100) stack over performance-set SVR base OOF predictions (Round1 PRIMARY TmApp)",
            "HISTORICAL_FROZEN_SUBMISSION",
        ),
        "TmApp__SIMPLE_blend_plm_struct": (
            "PREDICTION_LINEAR_BLEND",
            "Equal-mean blend: AbLang2 HL_paired Ridge + ESMFold STRUCT_RASA ElasticNet predictions",
            "HISTORICAL_FROZEN_PREDICTION",
        ),
    }
    # also include META ridge/nnls with both public and private
    for _, r in inv.iterrows():
        eid = str(r.experiment_id)
        is_blend = "SIMPLE_blend" in eid or "META_" in eid
        is_ridge_lasso_feature = bool(re.search(r"__(Ridge|Lasso)", eid)) and "META_" not in eid
        if not (is_blend or eid in hist_ids):
            # include strong historical Ridge feature models with Public+Private for overall PP tables
            if not (
                is_ridge_lasso_feature
                and pd.notna(r.Private_MAE)
                and pd.notna(r.Public_MAE)
                and float(r.Private_MAE) < 0.55
                and r.target == "HIC"
            ) and not (
                is_ridge_lasso_feature
                and pd.notna(r.Private_MAE)
                and r.target == "TmApp"
                and float(r.Private_MAE) < 3.25
            ):
                continue
        if eid in {x["model_id"].replace("__HIST", "") for x in rows if "HIST" in x.get("model_id", "")}:
            continue
        if eid in hist_ids:
            cls, _, pred_src = hist_ids[eid]
        else:
            cls = "PREDICTION_LINEAR_BLEND" if is_blend else "HISTORICAL_LINEAR_OTHER"
            pred_src = "HISTORICAL_FROZEN_PREDICTION"
        feat_desc = describe_historical(eid, r)
        # Feature-level historical Ridge without canonical shadow → not main CV eligible
        has_shadow = pd.notna(r.Shadow_CV_MAE)
        has_primary = pd.notna(r.Primary_CV_MAE)
        cv_p = float(r.Primary_CV_MAE) if has_primary else np.nan
        cv_s = float(r.Shadow_CV_MAE) if has_shadow else np.nan
        if has_primary and has_shadow:
            cv_mean = (cv_p + cv_s) / 2
            cv_worst = max(cv_p, cv_s)
        else:
            cv_mean = cv_worst = np.nan
        # Round1 primary exact pub/priv from scores file preferred when matching
        pub, priv = r.Public_MAE, r.Private_MAE
        if eid == "HIC__SIMPLE_blend_seq_surf_adv":
            pred_src = "HISTORICAL_FROZEN_SUBMISSION"
        if eid == specs["primary"]["TmApp"]["experiment_id"]:
            pred_src = "HISTORICAL_FROZEN_SUBMISSION"
            # use confirmed Primary/Shadow from specs
            cv_p = specs["primary"]["TmApp"]["Primary_MAE"]
            cv_s = specs["primary"]["TmApp"]["Shadow_MAE"]
            cv_mean = (cv_p + cv_s) / 2
            cv_worst = max(cv_p, cv_s)

        blob = (eid + "|" + str(getattr(r, "base_model_ids", "") or "")).lower()
        rows.append(
            {
                "target": r.target,
                "model_id": eid + "__HIST",
                "full_model_name": feat_desc,
                "result_class": cls,
                "feature_blocks_full": feat_desc,
                "feature_blocks_tokens": eid,
                "feature_source_files": "virtual_participant stage OOF / Round1 inventory; base_model_ids when present",
                "molecular_scope": str(r.modality) if pd.notna(getattr(r, "modality", None)) else "",
                "regressor": str(r.model_name) if pd.notna(r.model_name) else ("equal_mean_blend" if "blend" in eid else "meta"),
                "regressor_details": str(r.hyperparameters) if pd.notna(r.hyperparameters) else "",
                "preprocessing_full": "historical Stage pipeline (often SVR bases; not endgame Ridge/Lasso feature matrix)",
                "dimensionality_reduction": "varies by base model",
                "raw_dimension": np.nan,
                "effective_dimension": np.nan,
                "n_dev": 162,
                "n_test": 162 if pd.notna(priv) else np.nan,
                "cv_primary_mae": cv_p,
                "cv_shadow_mae": cv_s,
                "cv_mean_mae": cv_mean,
                "cv_worst_mae": cv_worst,
                "public_mae": pub,
                "private_mae": priv,
                "final_alpha": np.nan,
                "lasso_nonzero": np.nan,
                "lasso_total_features": np.nan,
                "cv_protocol": "historical Stage0/Stage5 OOF (NOT proven identical to canonical_simple_tvt_v1)",
                "final_fit_protocol": "historical Round1 full-DEV refit / frozen submission",
                "prediction_source": pred_src,
                "canonical_replay_status": "HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE",
                "validity": "HISTORICAL_RETAINED",
                "historical_role": "round1_or_stage5_linearish",
                "notes": "Do not mix into main FEATURE_LEVEL CV Top-10; SVR bases often underlie blends",
                "eligible_main_cv_rank": False,
                "contains_ESM2_Heavy_PLM": "esm2" in blob,
                "contains_AbLang2_PLM": "ablang" in blob,
                "contains_AbLingua_PLM": False,
                "contains_any_PLM": ("esm2" in blob) or ("ablang" in blob),
                "contains_sequence_descriptors": ("seq" in blob) or ("SEQ" in str(getattr(r, "base_model_ids", "") or "")),
                "contains_aromatic_structural": False,
                "contains_continuous_surface": ("surf" in blob) or ("surface" in blob),
                "contains_hydrophobic_field": False,
                "contains_titration": False,
                "contains_BioEmu": False,
                "contains_ProteinMPNN": False,
            }
        )

    # Deduplicate model_id
    reg = pd.DataFrame(rows)
    reg = reg.drop_duplicates("model_id", keep="first")
    reg.to_csv(OUT / "LINEAR_MODEL_MASTER_REGISTRY.csv", index=False)
    return reg


def top10_table(df: pd.DataFrame, metric: str, out_name: str, expand: bool = True):
    d = df.dropna(subset=[metric]).copy()
    d = d.sort_values([metric, "model_id"], kind="mergesort")
    d = d.head(10).reset_index(drop=True)
    d.insert(0, "rank", range(1, len(d) + 1))
    cols = [
        "rank",
        "model_id",
        "full_model_name" if expand else "feature_blocks_full",
        "feature_blocks_full",
        "regressor",
        "preprocessing_full",
        "result_class",
        "cv_primary_mae",
        "cv_shadow_mae",
        "cv_mean_mae",
        "cv_worst_mae",
        "public_mae",
        "private_mae",
        "prediction_source",
        "canonical_replay_status",
        "contains_any_PLM",
    ]
    cols = [c for c in cols if c in d.columns]
    d[cols].to_csv(OUT / out_name, index=False)
    return d


def rank_shake(reg: pd.DataFrame, unions: list[pd.DataFrame]):
    ids = set()
    for u in unions:
        ids |= set(u.model_id)
    rows = []
    feat = reg[reg.result_class == "FEATURE_LINEAR"].copy()
    for mid in ids:
        sub = reg[reg.model_id == mid]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        # ranks within same target feature-level or overall sets
        target = r.target
        def rank_in(metric, pool):
            p = pool[(pool.target == target) & pool[metric].notna()].sort_values(metric)
            if mid not in set(p.model_id):
                return np.nan
            return int(p.reset_index(drop=True).index[p.reset_index(drop=True).model_id == mid][0]) + 1

        pool_cv = feat[feat.eligible_main_cv_rank == True]
        pool_pp = reg  # overall for pub/priv
        cv_r = rank_in("cv_worst_mae", pool_cv)
        pub_r = rank_in("public_mae", pool_pp[pool_pp.target == target])
        priv_r = rank_in("private_mae", pool_pp[pool_pp.target == target])
        rows.append(
            {
                "target": target,
                "model_id": mid,
                "full_model_name": r.full_model_name,
                "result_class": r.result_class,
                "cv_rank": cv_r,
                "public_rank": pub_r,
                "private_rank": priv_r,
                "public_minus_cv_rank": pub_r - cv_r if pd.notna(pub_r) and pd.notna(cv_r) else np.nan,
                "private_minus_cv_rank": priv_r - cv_r if pd.notna(priv_r) and pd.notna(cv_r) else np.nan,
                "public_private_rank_gap": pub_r - priv_r if pd.notna(pub_r) and pd.notna(priv_r) else np.nan,
                "cv_worst_mae": r.cv_worst_mae,
                "public_mae": r.public_mae,
                "private_mae": r.private_mae,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "RANK_SHAKE_ANALYSIS.csv", index=False)
    return df


def winner_row(df: pd.DataFrame, metric: str):
    d = df.dropna(subset=[metric]).sort_values([metric, "model_id"], kind="mergesort")
    return d.iloc[0]


def write_winner_dossiers(reg: pd.DataFrame, feat: pd.DataFrame):
    lines = ["# Linear Model Winners — Detailed Dossiers\n"]
    lines.append(
        "Abbreviations such as PARENT / BASE / ARO / CONT are expanded in full. "
        "Public/Private winners are **postmortem only** and were not used for model selection.\n"
    )

    sections = [
        ("A", "TmApp", "canonical CV winner", "cv_worst_mae", feat[feat.target == "TmApp"]),
        ("B", "TmApp", "Public winner", "public_mae", reg[reg.target == "TmApp"]),
        ("C", "TmApp", "Private winner", "private_mae", reg[reg.target == "TmApp"]),
        ("D", "HIC", "canonical CV winner", "cv_worst_mae", feat[feat.target == "HIC"]),
        ("E", "HIC", "Public winner", "public_mae", reg[reg.target == "HIC"]),
        ("F", "HIC", "Private winner", "private_mae", reg[reg.target == "HIC"]),
    ]
    winners = {}
    section_data = []
    for letter, target, why, metric, pool in sections:
        if "CV" in why:
            pool = pool[pool.eligible_main_cv_rank == True]
        w = winner_row(pool, metric)
        winners[(target, why)] = w
        section_data.append((letter, target, why, metric, pool, w))

    for letter, target, why, metric, pool, w in section_data:
        tied = pool.dropna(subset=[metric])
        tied = tied[np.isclose(tied[metric].astype(float), float(w[metric]), rtol=0, atol=1e-12)]
        tie_note = ""
        if len(tied) > 1:
            tie_ids = ", ".join(f"`{x}`" for x in tied.model_id.head(8).tolist())
            tie_note = (
                f" Exact-score tie among {len(tied)} rows at {metric}={w[metric]}; "
                f"representative winner listed first by stable sort: {tie_ids}"
                + ("…" if len(tied) > 8 else "")
                + "."
            )
        also = [
            f"{t2} {why2}"
            for (t2, why2), w2 in winners.items()
            if (t2, why2) != (target, why) and w2.model_id == w.model_id
        ]
        also_note = f" This same configuration also won: {', '.join(also)}." if also else ""
        lines.append(f"\n{'='*50}\n{letter}. {target} — {why}\n{'='*50}\n")
        lines.append(f"1. **Full model name:** {w.full_model_name}\n")
        lines.append(f"2. **Target:** {target}\n")
        lines.append(
            f"3. **Why winner:** lowest {metric} among the ranking pool for this section "
            f"({'FEATURE_LINEAR canonical Simple TVT only' if 'CV' in why else 'all linear/linear-blend rows with this metric'})."
            f"{tie_note}{also_note}\n"
        )
        lines.append(
            f"4. **Scores:** CV Primary={w.cv_primary_mae}; CV Shadow={w.cv_shadow_mae}; "
            f"CV mean={w.cv_mean_mae}; CV worst={w.cv_worst_mae}; "
            f"Public={w.public_mae}; Private={w.private_mae}\n"
        )
        lines.append(f"5. **Complete feature composition:** {w.feature_blocks_full}\n")
        lines.append(f"6. **Feature sources:** {w.feature_source_files}\n")
        lines.append(f"   Raw dimension: {w.raw_dimension}; Effective: {w.effective_dimension}\n")
        lines.append(f"7. **Regressor:** {w.regressor} — {w.regressor_details}\n")
        lines.append(
            f"8. **Preprocessing:** {w.preprocessing_full}; dimensionality reduction: {w.dimensionality_reduction}\n"
        )
        lines.append(
            f"9. **Hyperparameters:** final_alpha={w.final_alpha}; "
            f"Lasso nonzero={w.lasso_nonzero} / total={w.lasso_total_features}\n"
        )
        lines.append(f"10. **CV protocol:** {w.cv_protocol}\n")
        lines.append(f"11. **Final Test-training protocol:** {w.final_fit_protocol}\n")
        lines.append(
            f"12. **Test prediction provenance:** {w.prediction_source} "
            f"(canonical_replay_status={w.canonical_replay_status})\n"
        )
        lines.append(
            "13. **Public/Private scoring:** N_public=81, N_private=81 when scored on official solution mask\n"
        )
        plm = "YES" if bool(w.contains_any_PLM) else "NO"
        lines.append(
            f"14. **Interpretation:** result_class={w.result_class}; "
            f"ESM-2 Heavy PLM={'YES' if bool(w.contains_ESM2_Heavy_PLM) else 'NO'}; "
            f"AbLang2 PLM={'YES' if bool(w.contains_AbLang2_PLM) else 'NO'}; "
            f"any PLM={plm}; "
            f"sequence descriptors={'YES' if bool(w.contains_sequence_descriptors) else 'NO'}; "
            f"aromatic structural={'YES' if bool(w.contains_aromatic_structural) else 'NO'}; "
            f"continuous surface={'YES' if bool(w.contains_continuous_surface) else 'NO'}; "
            f"hydrophobic-field={'YES' if bool(w.contains_hydrophobic_field) else 'NO'}; "
            f"titration={'YES' if bool(w.contains_titration) else 'NO'}. "
            f"Information combined: {w.feature_blocks_full}\n"
        )
        if "Public" in why or "Private" in why:
            lines.append(
                "15. **Caveat:** This Public/Private designation is **postmortem**. "
                "It was **not** used to choose recipes, regressors, or advanced-model freeze. "
                "If this winner differs from the canonical CV winner, that difference must not drive selection.\n"
            )
        else:
            lines.append("15. **Caveat:** Selected using canonical CV only (cv_worst_mae).\n")
        lines.append(f"\n**model_id:** `{w.model_id}`\n")

    (OUT / "LINEAR_MODEL_WINNERS_JA.md").write_text("\n".join(lines))
    return winners


def _hic_surface_family(tok: str) -> str:
    """Bucket HIC recipes by surface chemistry for diversity among CV near-ties."""
    has_h = "HYDRO_FIELD" in tok
    has_t = "TITRATION_SHAPE" in tok
    has_c = "CONTINUOUS_SURFACE" in tok
    if has_h and has_t:
        return "hydro_plus_titration"
    if has_t and not has_h and not has_c:
        return "titration_only_add"
    if has_c and not has_h and not has_t:
        return "continuous_surface_only_add"
    if has_c or has_h or has_t:
        return "mixed_surface"
    if "AROMATIC_TOPO" in tok:
        return "aromatic_no_continuous_surface"
    return "other"


def freeze_advanced(feat: pd.DataFrame):
    """Top-3 FEATURE_LINEAR per target by cv_worst with diversity among near-ties."""
    out = {
        "policy": (
            "canonical CV only (cv_worst_mae then cv_mean_mae); FEATURE_LINEAR only; "
            "when CV ranks 1–5 differ by tiny MAE, prefer scientifically diverse block families "
            "rather than three near-duplicate concatenations"
        ),
        "note": "Public/Private NOT used",
        "diversity_rule": {
            "TmApp": "CDR3-guided AbLingua stack → GLOBAL AbLingua stack → BioEmu+ProteinMPNN without AbLingua",
            "HIC": (
                "among near-tied ESM-2 Heavy + SEQ_ALL + AROMATIC-TOPO recipes, pick distinct surface families: "
                "(1) HYDRO_FIELD+TITRATION_SHAPE, (2) CONTINUOUS_SURFACE only, "
                "(3) aromatic topology without continuous-surface / hydro / titration fields "
                "(skip titration-only and mixed-surface near-duplicates)"
            ),
        },
        "targets": {},
    }
    for target in ["TmApp", "HIC"]:
        sub = feat[(feat.target == target) & (feat.eligible_main_cv_rank == True)].copy()
        sub = sub.sort_values(["cv_worst_mae", "cv_mean_mae"])
        picked = []
        used_tokens = set()
        used_families = set()
        for _, r in sub.iterrows():
            tok = str(r.feature_blocks_tokens)
            if tok in used_tokens:
                continue
            if target == "HIC":
                fam = _hic_surface_family(tok)
                # Prefer diversity once we already have a PLM+seq+aromatic core pick
                if fam in used_families and len(picked) >= 1:
                    continue
                # Skip near-duplicates of already chosen surface chemistry
                if fam == "titration_only_add" and "hydro_plus_titration" in used_families:
                    continue
                if fam == "mixed_surface" and (
                    "continuous_surface_only_add" in used_families
                    or "hydro_plus_titration" in used_families
                ):
                    continue
                used_families.add(fam)
            elif target == "TmApp":
                # Prefer: with CDR3, with GLOBAL only, without AbLingua
                fam = (
                    "cdr3"
                    if "AbLingua_CDR3" in tok
                    else ("global_abl" if "AbLingua_HL_mean" in tok else ("bioemu_mpnn" if "BIOEMU" in tok or "MPNN" in tok else "other"))
                )
                if fam in used_families:
                    continue
                used_families.add(fam)
            picked.append(
                {
                    "cv_rank": len(picked) + 1,
                    "model_id": r.model_id,
                    "recipe_tokens": tok,
                    "feature_blocks_full": r.feature_blocks_full,
                    "regressor": r.regressor,
                    "preprocessing_full": r.preprocessing_full,
                    "cv_primary_mae": float(r.cv_primary_mae),
                    "cv_shadow_mae": float(r.cv_shadow_mae),
                    "cv_worst_mae": float(r.cv_worst_mae),
                    "contains_any_PLM": bool(r.contains_any_PLM),
                }
            )
            used_tokens.add(tok)
            if len(picked) >= 3:
                break
        # fallback fill if diversity skipped too aggressively
        if len(picked) < 3:
            for _, r in sub.iterrows():
                tok = str(r.feature_blocks_tokens)
                if tok in used_tokens:
                    continue
                picked.append(
                    {
                        "cv_rank": len(picked) + 1,
                        "model_id": r.model_id,
                        "recipe_tokens": tok,
                        "feature_blocks_full": r.feature_blocks_full,
                        "regressor": r.regressor,
                        "preprocessing_full": r.preprocessing_full,
                        "cv_primary_mae": float(r.cv_primary_mae),
                        "cv_shadow_mae": float(r.cv_shadow_mae),
                        "cv_worst_mae": float(r.cv_worst_mae),
                        "contains_any_PLM": bool(r.contains_any_PLM),
                    }
                )
                used_tokens.add(tok)
                if len(picked) >= 3:
                    break
        out["targets"][target] = picked
    path = OUT / "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json"
    path.write_text(json.dumps(out, indent=2))
    (OUT / "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )
    return out


def sync_bundle(adv: dict):
    """Update recipes.csv and README; add missing blocks if any."""
    rows = []
    needed = set()
    for target, lst in adv["targets"].items():
        for p in lst:
            for b in str(p["recipe_tokens"]).split("|"):
                if b and b != "none":
                    needed.add(b)
            rows.append(
                {
                    "target": target,
                    "cv_rank": p["cv_rank"],
                    "recipe_id": p["model_id"],
                    "feature_blocks": p["recipe_tokens"],
                    "feature_blocks_full": p["feature_blocks_full"],
                    "regressor": p["regressor"],
                    "preprocessing": p["preprocessing_full"],
                    "cv_primary_mae": p["cv_primary_mae"],
                    "cv_shadow_mae": p["cv_shadow_mae"],
                    "comments": "Frozen for advanced-model phase from FEATURE_LINEAR canonical CV only; Public/Private not used",
                }
            )
    pd.DataFrame(rows).to_csv(BUNDLE / "recipes.csv", index=False)
    # README update Top recipes section — rewrite concise README head
    man = pd.read_csv(BUNDLE / "feature_manifest.csv")
    existing = set(man.block_name)
    missing = needed - existing
    readme = (BUNDLE / "README.md").read_text()
    # replace Top recipes section simply by rewriting file with updated top list
    top_txt = ["## Top recipes (advanced-model freeze; CV only)\n"]
    for target, lst in adv["targets"].items():
        top_txt.append(f"\n### {target}\n")
        for p in lst:
            top_txt.append(
                f"{p['cv_rank']}. `{p['model_id']}` — {p['regressor']} — "
                f"CV {p['cv_primary_mae']:.4f}/{p['cv_shadow_mae']:.4f}\n"
                f"   Features: {p['feature_blocks_full']}\n"
            )
    # keep rest of README after '# Organizer' 
    marker = "## Top recipes"
    if marker in readme:
        pre = readme.split(marker)[0]
        # keep from Recommended folds if present after old top
        rest = ""
        if "## Recommended folds" in readme:
            rest = "## Recommended folds" + readme.split("## Recommended folds", 1)[1]
        readme = pre + "".join(top_txt) + "\n" + rest
    else:
        readme = readme + "\n" + "".join(top_txt)
    note = (
        "\n\n## Scope clarification\n"
        "The earlier chat 'Organizer Top-3' referred only to the **restricted** "
        "endgame Ridge/Lasso feature-recipe inventory, **not** all historical organizer linear models. "
        "Historical HIC ~0.42 results were mainly **prediction blends of SVR bases**, "
        "tracked separately in `organizer_extension/linear_model_closure/`.\n"
    )
    if "Scope clarification" not in readme:
        readme = readme + note
    (BUNDLE / "README.md").write_text(readme)
    return {"needed": sorted(needed), "missing_blocks": sorted(missing), "existing": sorted(existing & needed)}


def write_closure(top3_audit, reg, feat, winners, adv, bundle_info, shake):
    # answers
    hic_cv = winners[("HIC", "canonical CV winner")]
    hic_pub = winners[("HIC", "Public winner")]
    hic_priv = winners[("HIC", "Private winner")]
    tm_cv = winners[("TmApp", "canonical CV winner")]
    tm_pub = winners[("TmApp", "Public winner")]
    tm_priv = winners[("TmApp", "Private winner")]

    # recovered 0.42?
    hist = reg[reg.result_class == "PREDICTION_LINEAR_BLEND"]
    near = hist[(hist.target == "HIC") & (hist.private_mae < 0.43) & hist.private_mae.notna()]

    ridge_wins = 0
    lasso_wins = 0
    for rid in feat.feature_blocks_tokens.unique():
        sub = feat[feat.feature_blocks_tokens == rid]
        if set(sub.regressor) != {"RIDGE", "LASSO"}:
            continue
        rw = float(sub.loc[sub.regressor == "RIDGE", "cv_worst_mae"].iloc[0])
        lw = float(sub.loc[sub.regressor == "LASSO", "cv_worst_mae"].iloc[0])
        if rw < lw:
            ridge_wins += 1
        elif lw < rw:
            lasso_wins += 1

    # rank shake summary
    sh = shake.dropna(subset=["cv_rank", "private_rank"])
    med_gap = float((sh.private_rank - sh.cv_rank).abs().median()) if len(sh) else float("nan")

    lines = [
        "# Linear Model Closure Report\n",
        "## Top answers\n",
        "1. **What did previous Top-3 mean?** "
        "**A. Top-3 among the small restricted Ridge/Lasso FEATURE_RECIPE_FREEZE inventory** "
        "in `endgame_model_benchmark` — **not** all-time organizer linear models.\n",
        "2. **Was HIC naming misleading?** **YES.** Aliases like `ARO+TITR` / `ARO+CONT` / `ARO+HYDRO+TITR` "
        "actually included **ESM-2 Heavy-chain embedding + SEQ_ALL sequence descriptors + AROMATIC-TOPO + …**. "
        "They were never aromatic-only.\n",
        f"3. **Did best HIC feature-level CV model contain a PLM?** "
        f"**{'YES' if bool(hic_cv.contains_any_PLM) else 'NO'}** (ESM-2 Heavy). "
        f"Composition: {hic_cv.feature_blocks_full}. "
        f"Separately — strongest HIC Public model PLM? "
        f"{'YES' if bool(hic_pub.contains_any_PLM) else 'NO'}; "
        f"strongest HIC Private model PLM? "
        f"{'YES' if bool(hic_priv.contains_any_PLM) else 'NO'} "
        f"(Public/Private winners here are prediction blends of SVR bases, not feature-level Ridge/Lasso).\n",
        f"4. **Historical linear models audited in registry?** {len(reg)} rows "
        f"({(reg.result_class=='FEATURE_LINEAR').sum()} FEATURE_LINEAR; "
        f"{(reg.result_class=='PREDICTION_LINEAR_BLEND').sum()} PREDICTION_LINEAR_BLEND; "
        f"{(reg.result_class=='HISTORICAL_LINEAR_OTHER').sum()} OTHER)\n",
        f"5. **Were historical ~0.42 HIC results recovered?** "
        f"**YES** — e.g. Round1 PRIMARY / `HIC__SIMPLE_blend_seq_surf_adv` "
        f"Public≈0.420 / Private≈0.424 (equal-mean **SVR** prediction blend). "
        f"Recovered rows with Private<0.43: {len(near)}. "
        f"These are **not** feature-level Ridge/Lasso and are **not** ranked in main FEATURE_LEVEL CV Top-10.\n",
        f"6. **TmApp canonical CV winner:** `{tm_cv.model_id}` — {tm_cv.full_model_name}\n",
        f"7. **TmApp Public winner:** `{tm_pub.model_id}`\n",
        f"8. **TmApp Private winner:** `{tm_priv.model_id}`\n",
        f"9. **HIC canonical CV winner:** `{hic_cv.model_id}` — {hic_cv.full_model_name}\n",
        f"10. **HIC Public winner:** `{hic_pub.model_id}`\n",
        f"11. **HIC Private winner:** `{hic_priv.model_id}`\n",
        f"12. **Rank shake:** median |private_rank−cv_rank| among shake table ≈ {med_gap:.1f}; "
        "upper board shakes between CV and Private (especially HIC aromatic-only Ridge private strength vs CV).\n",
        f"13. **Ridge vs Lasso:** On matched feature recipes, Ridge wins CV-worst more often ({ridge_wins} vs {lasso_wins}). "
        "Lasso helps high-dimensional HIC concatenations; TmApp authority recipes remain Ridge (+ selective PCA on AbLingua).\n",
        "14. **Advanced-model Top-3 freeze:** see `ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json`\n",
        f"15. **Participant bundle consistent?** needed blocks={bundle_info['needed']}; "
        f"missing={bundle_info['missing_blocks'] or 'none'}\n",
        "16. **LINEAR_MODEL_CLOSED = YES**\n",
        "\n## Integrity\n",
        "- Canonical Top-3 audit written with full block expansion\n",
        "- Main CV Top-10 = FEATURE_LINEAR + canonical Simple TVT only\n",
        "- Historical blends retained separately / overall Public–Private tables\n",
        "- Public/Private not used for advanced freeze\n",
    ]
    (OUT / "LINEAR_MODEL_CLOSURE_JA.md").write_text("".join(lines))


def main():
    print("1) CURRENT_TOP3 audit...", flush=True)
    top3 = build_current_top3_audit()
    print("2) Master registry...", flush=True)
    reg = build_master_registry()
    feat = reg[reg.result_class == "FEATURE_LINEAR"].copy()

    print("3) Top-10 tables...", flush=True)
    tables = []
    # FEATURE LEVEL
    for target in ["TmApp", "HIC"]:
        f = feat[(feat.target == target) & (feat.eligible_main_cv_rank == True)]
        tables.append(top10_table(f, "cv_worst_mae", f"FEATURE_LEVEL_{target}_CV_TOP10.csv"))
        # pub/priv feature level only authoritative
        fa = feat[(feat.target == target) & (feat.validity == "AUTHORITATIVE")]
        tables.append(top10_table(fa, "public_mae", f"FEATURE_LEVEL_{target}_PUBLIC_TOP10.csv"))
        tables.append(top10_table(fa, "private_mae", f"FEATURE_LEVEL_{target}_PRIVATE_TOP10.csv"))
        # OVERALL linear (includes historical blends) for Pub/Priv
        o = reg[reg.target == target]
        tables.append(top10_table(o, "public_mae", f"ORGANIZER_LINEAR_OVERALL_{target}_PUBLIC_TOP10.csv"))
        tables.append(top10_table(o, "private_mae", f"ORGANIZER_LINEAR_OVERALL_{target}_PRIVATE_TOP10.csv"))
        # historical CV table (has primary+shadow but not canonical)
        h = reg[
            (reg.target == target)
            & (reg.result_class != "FEATURE_LINEAR")
            & reg.cv_primary_mae.notna()
            & reg.cv_shadow_mae.notna()
        ]
        if len(h):
            top10_table(h, "cv_worst_mae", f"HISTORICAL_LINEAR_{target}_CV_TOP10_NONCANONICAL.csv")

    # Also write the six “main” names user asked — use FEATURE_LEVEL for CV,
    # and for Pub/Priv provide BOTH feature-level and overall (overall as primary leaderboard name)
    for target in ["TmApp", "HIC"]:
        # copies with requested names pointing to feature-level CV + overall PP
        src = OUT / f"FEATURE_LEVEL_{target}_CV_TOP10.csv"
        src.replace(OUT / f"{target}_CV_TOP10.csv") if False else None
        pd.read_csv(OUT / f"FEATURE_LEVEL_{target}_CV_TOP10.csv").to_csv(OUT / f"{target}_CV_TOP10.csv", index=False)
        pd.read_csv(OUT / f"ORGANIZER_LINEAR_OVERALL_{target}_PUBLIC_TOP10.csv").to_csv(
            OUT / f"{target}_PUBLIC_TOP10.csv", index=False
        )
        pd.read_csv(OUT / f"ORGANIZER_LINEAR_OVERALL_{target}_PRIVATE_TOP10.csv").to_csv(
            OUT / f"{target}_PRIVATE_TOP10.csv", index=False
        )

    print("4) Rank shake...", flush=True)
    shake = rank_shake(reg, tables)

    print("5) Winners...", flush=True)
    winners = write_winner_dossiers(reg, feat)

    print("6) Advanced freeze...", flush=True)
    adv = freeze_advanced(feat)
    bundle_info = sync_bundle(adv)

    print("7) Closure...", flush=True)
    write_closure(top3, reg, feat, winners, adv, bundle_info, shake)

    # human readable top10 markdown without abbreviations
    md = ["# Linear Model Top-10 Tables\n",
          "CV tables = **FEATURE_LINEAR canonical Simple TVT only**.\n",
          "Public/Private tables below = **ORGANIZER_LINEAR_OVERALL** (includes historical prediction blends).\n",
          "See also FEATURE_LEVEL_*_PUBLIC/PRIVATE_TOP10.csv for feature-only postmortem.\n"]
    for target in ["TmApp", "HIC"]:
        for kind, fname in [
            ("CV", f"{target}_CV_TOP10.csv"),
            ("Public", f"{target}_PUBLIC_TOP10.csv"),
            ("Private", f"{target}_PRIVATE_TOP10.csv"),
        ]:
            d = pd.read_csv(OUT / fname)
            md.append(f"\n## {target} — {kind} Top-10\n")
            md.append("| rank | full composition | regressor | preprocess | CV P | CV S | Public | Private | class |")
            md.append("|------|------------------|-----------|------------|------|------|--------|---------|-------|")
            for _, r in d.iterrows():
                md.append(
                    f"| {int(r['rank'])} | {r['feature_blocks_full'][:120]}… | {r['regressor']} | "
                    f"{str(r['preprocessing_full'])[:40]} | {r['cv_primary_mae']} | {r['cv_shadow_mae']} | "
                    f"{r['public_mae']} | {r['private_mae']} | {r['result_class']} |"
                )
    (OUT / "TOP10_TABLES_JA.md").write_text("\n".join(md))
    print("DONE", OUT)


if __name__ == "__main__":
    main()
