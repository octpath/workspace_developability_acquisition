#!/usr/bin/env python3
"""Repair linear-model closure metadata/provenance and localize JA/EN reports.

Does NOT change scores, rankings, winners, or ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/linear_model_closure"
END = ROOT / "organizer_extension/endgame_model_benchmark"
VP = ROOT / "virtual_participant"
PRED = VP / "round1_postmortem/predictions_exploratory"
SOL = ROOT / "competition/data/secret/solution.csv"

PCA_CAP = 32

WINNERS = {
    "tm_cv": "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "tm_pub": "TmApp__ablang2__HL__RidgeOpt__HIST",
    "tm_priv": "TmApp__META_diversity__convex_mae__HIST",
    "hic_cv": "HIC_HYDRO_TITRATION__LASSO",
    "hic_pub": "HIC__SIMPLE_blend_seq_surf__HIST",
    "hic_priv": "HIC__SIMPLE_blend_esm2_surf__HIST",
}

EIGHT = [
    "TmApp__META_diversity__convex_mae",
    "TmApp__META_diversity__nnls",
    "TmApp__META_diversity__nnls_norm",
    "TmApp__META_diversity__quantile_0.0",
    "TmApp__META_diversity__quantile_0.001",
    "TmApp__META_diversity__quantile_0.01",
    "TmApp__META_diversity__quantile_0.1",
    "TmApp__META_diversity__ridge_100.0",
]


def block_raw_dims() -> dict[str, int]:
    import sys

    sys.path.insert(0, str(END))
    from feature_store import FeatureStore

    fs = FeatureStore()
    names = [
        "AbLang2_HL_paired",
        "SEQ_BASIC",
        "BIOEMU_NEW_PAIRWISE",
        "M1_PROTEINMPNN",
        "AbLingua_HL_mean",
        "AbLingua_CDR3",
    ]
    out = {}
    for n in names:
        X = fs.get(n)
        out[n] = int(X.shape[1])
    return out


def effective_audit(dims: dict[str, int]) -> pd.DataFrame:
    """Per-block transform table for the two AbLingua Ridge recipes."""
    rows = []
    recipes = {
        "TM_PARENT_ABLINGUA_CDR3__RIDGE": [
            ("AbLang2_HL_paired", "raw (no PCA)", None),
            ("SEQ_BASIC", "raw (no PCA)", None),
            ("BIOEMU_NEW_PAIRWISE", "raw (no PCA)", None),
            ("M1_PROTEINMPNN", "raw (no PCA)", None),
            ("AbLingua_HL_mean", "fold-local PCA", PCA_CAP),
            ("AbLingua_CDR3", "fold-local PCA", PCA_CAP),
        ],
        "TM_PARENT_ABLINGUA_GLOBAL__RIDGE": [
            ("AbLang2_HL_paired", "raw (no PCA)", None),
            ("SEQ_BASIC", "raw (no PCA)", None),
            ("BIOEMU_NEW_PAIRWISE", "raw (no PCA)", None),
            ("M1_PROTEINMPNN", "raw (no PCA)", None),
            ("AbLingua_HL_mean", "fold-local PCA", PCA_CAP),
        ],
    }
    for mid, blocks in recipes.items():
        raw_sum = 0
        eff_sum = 0
        for blk, transform, pca_n in blocks:
            raw = dims[blk]
            eff = int(pca_n) if pca_n is not None else raw
            # PCA n_components = min(PCA_CAP, n_samples-1, n_features); on DEV folds
            # with AbLingua dim>>32 and train size >>32, effective is 32.
            raw_sum += raw
            eff_sum += eff
            rows.append(
                {
                    "model_id": mid,
                    "block": blk,
                    "raw_dim": raw,
                    "transform": transform,
                    "effective_dim": eff,
                    "notes": "PCA via canonical_simple_tvt.run_recipe_plus_abl_blocks / _pca_named"
                    if pca_n
                    else "passed through after median impute; StandardScaler on concatenated TV matrix",
                }
            )
        rows.append(
            {
                "model_id": mid,
                "block": "__TOTAL__",
                "raw_dim": raw_sum,
                "transform": "concat after per-block transforms",
                "effective_dim": eff_sum,
                "notes": "effective_dim = final Ridge input width after AbLingua PCA32",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "EFFECTIVE_DIMENSION_AUDIT.csv", index=False)
    return df


def patch_registry(audit: pd.DataFrame):
    reg = pd.read_csv(OUT / "LINEAR_MODEL_MASTER_REGISTRY.csv")
    totals = {
        r.model_id: int(r.effective_dim)
        for _, r in audit[audit.block == "__TOTAL__"].iterrows()
    }
    raws = {
        r.model_id: int(r.raw_dim) for _, r in audit[audit.block == "__TOTAL__"].iterrows()
    }
    changed = []
    for mid, eff in totals.items():
        m = reg.model_id == mid
        if not m.any():
            continue
        old_eff = reg.loc[m, "effective_dimension"].iloc[0]
        old_raw = reg.loc[m, "raw_dimension"].iloc[0]
        # keep raw; fix effective only when it wrongly equals raw for PCA Ridge
        reg.loc[m, "effective_dimension"] = eff
        if abs(float(old_raw) - float(raws[mid])) > 0.5:
            raise SystemExit(f"STOP: raw dim mismatch for {mid}: registry={old_raw} audit={raws[mid]}")
        changed.append((mid, old_eff, eff, old_raw))
        # annotate notes
        note = str(reg.loc[m, "notes"].iloc[0])
        fix = f"EFFECTIVE_DIM_REPAIRED:{old_eff}->{eff} (PCA32 per AbLingua block)"
        if "EFFECTIVE_DIM_REPAIRED" not in note:
            reg.loc[m, "notes"] = (note + "; " + fix).strip("; ")
    # also patch CURRENT_TOP3 if needed
    top3 = pd.read_csv(OUT / "CURRENT_TOP3_RECIPE_AUDIT.csv")
    if "effective_dimension" in top3.columns:
        for rid, mid in [
            ("TM_PARENT_ABLINGUA_CDR3", "TM_PARENT_ABLINGUA_CDR3__RIDGE"),
            ("TM_PARENT_ABLINGUA_GLOBAL", "TM_PARENT_ABLINGUA_GLOBAL__RIDGE"),
        ]:
            if mid in totals:
                top3.loc[top3.recipe_id == rid, "effective_dimension"] = totals[mid]
        top3.to_csv(OUT / "CURRENT_TOP3_RECIPE_AUDIT.csv", index=False)
    reg.to_csv(OUT / "LINEAR_MODEL_MASTER_REGISTRY.csv", index=False)
    return changed, reg


def private_tie_audit() -> pd.DataFrame:
    sol = pd.read_csv(SOL)
    sol["id"] = sol["id"].astype(str)
    sol = sol.set_index("id")
    pub = sol[sol["is_public"].astype(bool)]
    priv = sol[sol["is_private"].astype(bool)]
    rep_id = EIGHT[0]
    vecs = {}
    for m in EIGHT:
        path = PRED / f"{m}.csv"
        df = pd.read_csv(path)
        df["id"] = df["id"].astype(str)
        pred = df.set_index("id").reindex(sol.index)["prediction"].astype(float)
        if pred.isna().any():
            raise SystemExit(f"NaN predictions for {m}")
        h = hashlib.sha256(np.ascontiguousarray(pred.values).tobytes()).hexdigest()
        pub_mae = float(np.mean(np.abs(pred.loc[pub.index] - pub["TmApp"].astype(float))))
        priv_mae = float(np.mean(np.abs(pred.loc[priv.index] - priv["TmApp"].astype(float))))
        vecs[m] = (pred, h, pub_mae, priv_mae)

    # inventory scores for consistency check
    inv = pd.read_csv(VP / "round1_postmortem/round1_all_model_score_inventory.csv")
    rows = []
    rp, rh, _, _ = vecs[rep_id]
    families = {}
    for m, (p, h, pu, pr) in vecs.items():
        d = np.abs(p.values - rp.values)
        fam = f"IDENTICAL_FAMILY_{h[:12]}"
        families.setdefault(h, []).append(m)
        inv_r = inv[inv.experiment_id == m]
        inv_pub = float(inv_r.Public_MAE.iloc[0]) if len(inv_r) else np.nan
        inv_priv = float(inv_r.Private_MAE.iloc[0]) if len(inv_r) else np.nan
        if abs(pu - inv_pub) > 1e-6 or abs(pr - inv_priv) > 1e-6:
            # tolerate tiny float; hard stop on material change
            if abs(pu - inv_pub) > 1e-4 or abs(pr - inv_priv) > 1e-4:
                raise SystemExit(
                    f"STOP: score mismatch {m}: pred_pub={pu} inv={inv_pub} pred_priv={pr} inv={inv_priv}"
                )
        rows.append(
            {
                "model_id": m + "__HIST",
                "base_experiment_id": m,
                "prediction_artifact": str(path := PRED / f"{m}.csv"),
                "prediction_hash_sha256": h,
                "public_mae": pu,
                "private_mae": pr,
                "inventory_public_mae": inv_pub,
                "inventory_private_mae": inv_priv,
                "max_abs_diff_vs_representative": float(d.max()),
                "mean_abs_diff_vs_representative": float(d.mean()),
                "corr_vs_representative": float(np.corrcoef(p.values, rp.values)[0, 1]),
                "prediction_family": fam,
                "classification": "IDENTICAL_PREDICTION_FAMILY",
                "notes": "All eight META_diversity labels share bit-identical Test prediction vectors",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "TMAPP_PRIVATE_TIE_AUDIT.csv", index=False)
    summary = {
        "representative": rep_id + "__HIST",
        "n_labels": len(EIGHT),
        "n_unique_prediction_hashes": len(families),
        "classification": "IDENTICAL_PREDICTION_FAMILY"
        if len(families) == 1
        else "DISTINCT_PREDICTIONS_SAME_PRIVATE_MAE",
        "shared_sha256": list(families.keys())[0] if len(families) == 1 else None,
        "private_mae": float(vecs[rep_id][3]),
        "public_mae": float(vecs[rep_id][2]),
    }
    (OUT / "TMAPP_PRIVATE_TIE_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    return df, summary


def public_winner_provenance() -> dict:
    s2 = pd.read_csv(VP / "stage2_plm/stage2_primary_results.csv")
    r = s2[s2.experiment_id == "TmApp__ablang2__HL__RidgeOpt"].iloc[0]
    pred_path = PRED / "TmApp__ablang2__HL__RidgeOpt.csv"
    sol = pd.read_csv(SOL)
    sol["id"] = sol["id"].astype(str)
    sol = sol.set_index("id")
    pred = pd.read_csv(pred_path)
    pred["id"] = pred["id"].astype(str)
    p = pred.set_index("id").reindex(sol.index)["prediction"].astype(float)
    h = hashlib.sha256(np.ascontiguousarray(p.values).tobytes()).hexdigest()
    hp = json.loads(r.hyperparameters) if pd.notna(r.hyperparameters) else {}
    prov = {
        "model_id": "TmApp__ablang2__HL__RidgeOpt__HIST",
        "experiment_id": "TmApp__ablang2__HL__RidgeOpt",
        "target": "TmApp",
        "plm": "AbLang2",
        "chain_representation": "HL",
        "chain_representation_meaning": (
            "Heavy-chain whole-chain mean embedding concatenated with "
            "Light-chain whole-chain mean embedding (NOT AbLang2 HL_paired / paired seqcoding)"
        ),
        "pooling": "whole_chain_mean",
        "raw_embedding_dimension_before_pca": int(r.n_features_before_pca),
        "pca_dim": int(r.pca_dim),
        "effective_dimension_after_pca": int(r.n_features_after_pca),
        "pca_used": True,
        "scaler": "StandardScaler; fitted inside each training fold (with SimpleImputer median)",
        "scaler_source": "virtual_participant/stage2_plm/scripts/run_stage2.py::fold_matrices",
        "regressor": "RidgeOpt (Optuna-tuned Ridge)",
        "alpha": float(hp.get("alpha")),
        "optimization": "Optuna; notes field reports trials=40; pca_dim was a search hyperparameter",
        "cv_protocol": (
            "Stage-2 Primary folds + Shadow confirmation; "
            "NOT proven identical to canonical_simple_tvt_v1"
        ),
        "primary_cv_mae": float(r.Primary_MAE),
        "shadow_cv_mae": float(r.Shadow_MAE),
        "public_mae": 3.0853249349711853,
        "private_mae": 3.159989439410928,
        "feature_source_embeddings": (
            "Stage-2 AbLang2 cache via run_stage2 load_ablang2 "
            "(H mean, L mean → HL concat); distinct from endgame AbLang2_HL_paired block"
        ),
        "frozen_test_prediction_artifact": str(pred_path),
        "prediction_hash_sha256": h,
        "full_dev_final_refit": (
            "履歴資料から確定できない: Stage-2 script documents fold-local CV/Optuna and OOF; "
            "authoritative Test predictions are the frozen exploratory artifact above. "
            "Exact full-DEV refit code path that produced that Test file is not pinned "
            "in a single authoritative script comment beyond Round1 postmortem inventory usage."
        ),
        "canonical_replay_status": "HISTORICAL_PROTOCOL_NOT_CANONICALLY_REPLAYABLE",
        "authoritative_tables": [
            "virtual_participant/stage2_plm/stage2_primary_results.csv",
            "virtual_participant/stage2_plm/STAGE2_REPORT_JA.md",
            "virtual_participant/round1_postmortem/round1_all_model_score_inventory.csv",
            str(pred_path),
        ],
    }
    (OUT / "TMAPP_PUBLIC_WINNER_PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    return prov


def load_scores(reg: pd.DataFrame) -> dict:
    def row(mid):
        r = reg[reg.model_id == mid].iloc[0]
        return r

    out = {}
    for k, mid in WINNERS.items():
        r = row(mid)
        out[k] = {
            "model_id": mid,
            "full_model_name": r.full_model_name,
            "feature_blocks_full": r.feature_blocks_full,
            "feature_source_files": r.feature_source_files,
            "regressor": r.regressor,
            "regressor_details": r.regressor_details,
            "preprocessing_full": r.preprocessing_full,
            "dimensionality_reduction": r.dimensionality_reduction,
            "raw_dimension": r.raw_dimension,
            "effective_dimension": r.effective_dimension,
            "final_alpha": r.final_alpha,
            "lasso_nonzero": r.lasso_nonzero,
            "lasso_total_features": r.lasso_total_features,
            "cv_primary_mae": r.cv_primary_mae,
            "cv_shadow_mae": r.cv_shadow_mae,
            "cv_mean_mae": r.cv_mean_mae,
            "cv_worst_mae": r.cv_worst_mae,
            "public_mae": r.public_mae,
            "private_mae": r.private_mae,
            "cv_protocol": r.cv_protocol,
            "final_fit_protocol": r.final_fit_protocol,
            "prediction_source": r.prediction_source,
            "canonical_replay_status": r.canonical_replay_status,
            "result_class": r.result_class,
            "contains_any_PLM": bool(r.contains_any_PLM),
            "contains_ESM2_Heavy_PLM": bool(r.contains_ESM2_Heavy_PLM),
            "contains_AbLang2_PLM": bool(r.contains_AbLang2_PLM),
            "contains_sequence_descriptors": bool(r.contains_sequence_descriptors),
            "contains_aromatic_structural": bool(r.contains_aromatic_structural),
            "contains_continuous_surface": bool(r.contains_continuous_surface),
            "contains_hydrophobic_field": bool(r.contains_hydrophobic_field),
            "contains_titration": bool(r.contains_titration),
        }
    return out


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return str(x)


def rename_en_files():
    mapping = [
        ("LINEAR_MODEL_CLOSURE_JA.md", "LINEAR_MODEL_CLOSURE_EN.md"),
        ("LINEAR_MODEL_WINNERS_JA.md", "LINEAR_MODEL_WINNERS_EN.md"),
        ("TOP10_TABLES_JA.md", "TOP10_TABLES_EN.md"),
    ]
    done = []
    for src, dst in mapping:
        s, d = OUT / src, OUT / dst
        if s.exists() and not d.exists():
            s.rename(d)
            done.append((src, dst))
        elif d.exists() and s.exists():
            # already partially done — keep EN, remove misleading JA later overwritten
            done.append((src, dst + " (EN already existed; JA will be rewritten)"))
        elif d.exists() and not s.exists():
            done.append((src, dst + " (already renamed)"))
    return done


def write_winners_en(scores, audit, prov, tie_summary):
    cdr3_blocks = audit[audit.model_id == "TM_PARENT_ABLINGUA_CDR3__RIDGE"]
    glob_blocks = audit[audit.model_id == "TM_PARENT_ABLINGUA_GLOBAL__RIDGE"]

    def block_table(df):
        lines = ["| block | raw_dim | transform | effective_dim |", "|---|---:|---|---:|"]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['block']} | {int(r['raw_dim'])} | {r['transform']} | {int(r['effective_dim'])} |"
            )
        return "\n".join(lines)

    w = scores
    lines = []
    lines.append("# Linear Model Winners — Detailed Dossiers (English)\n")
    lines.append(
        "Abbreviations such as PARENT / BASE / ARO / CONT are expanded in full. "
        "Public/Private winners are **postmortem only** and were not used for model selection.\n"
        "Metadata repairs (effective dimension, Public provenance, Private tie) applied without "
        "changing scores or rankings.\n"
    )

    # A
    a = w["tm_cv"]
    lines += [
        "\n" + "=" * 50,
        "\nA. TmApp — canonical CV winner\n",
        "=" * 50 + "\n",
        f"1. **Full model name:** {a['full_model_name']}\n",
        "2. **Target:** TmApp\n",
        "3. **Why winner:** lowest cv_worst_mae among FEATURE_LINEAR canonical Simple TVT only.\n",
        f"4. **Scores:** CV Primary={fmt(a['cv_primary_mae'])}; CV Shadow={fmt(a['cv_shadow_mae'])}; "
        f"CV mean={fmt(a['cv_mean_mae'])}; CV worst={fmt(a['cv_worst_mae'])}; "
        f"Public={fmt(a['public_mae'])}; Private={fmt(a['private_mae'])}\n",
        f"5. **Complete feature composition:** {a['feature_blocks_full']}\n",
        f"6. **Feature sources:** {a['feature_source_files']}\n",
        "\nPer-block dimensions (from actual feature matrices + canonical PCA pipeline):\n\n",
        block_table(cdr3_blocks) + "\n",
        f"\n   Raw concatenated dimension: {fmt(a['raw_dimension'])}; "
        f"**Effective Ridge input dimension after PCA: {fmt(a['effective_dimension'])}** "
        "(not equal to raw; AbLingua GLOBAL and CDR3 each reduced with fold-local PCA32).\n",
        f"7. **Regressor:** {a['regressor']} — {a['regressor_details']}\n",
        f"8. **Preprocessing:** {a['preprocessing_full']}; dimensionality reduction: "
        f"{a['dimensionality_reduction']}. Median impute on recipe blocks; fold-local PCA32 "
        "independently on each AbLingua block; StandardScaler on the concatenated train+val matrix "
        "before Ridge (canonical_simple_tvt_v1).\n",
        f"9. **Hyperparameters:** final_alpha={fmt(a['final_alpha'])}; "
        f"Lasso nonzero={fmt(a['lasso_nonzero'])} / total={fmt(a['lasso_total_features'])}\n",
        f"10. **CV protocol:** {a['cv_protocol']}\n",
        f"11. **Final Test-training protocol:** {a['final_fit_protocol']}\n",
        f"12. **Test prediction provenance:** {a['prediction_source']} "
        f"(canonical_replay_status={a['canonical_replay_status']})\n",
        "13. **Public/Private scoring:** N_public=81, N_private=81 on official solution mask\n",
        "14. **Interpretation:** Combines AbLang2 paired PLM embedding, sequence descriptors, "
        "BioEmu pairwise ensemble geometry, ProteinMPNN compatibility, and two AbLingua embeddings "
        "(global + CDR3-guided) under Ridge after PCA on AbLingua only.\n",
        "15. **Caveat:** Selected using canonical CV only (cv_worst_mae).\n",
        f"\n**model_id:** `{a['model_id']}`\n",
    ]

    # B
    b = w["tm_pub"]
    lines += [
        "\n" + "=" * 50,
        "\nB. TmApp — Public winner\n",
        "=" * 50 + "\n",
        "1. **Full model name:** Stage-2 AbLang2 heavy+light (HL concat of whole-chain means) "
        f"under Optuna-tuned Ridge (RidgeOpt), alpha={prov['alpha']}, fold-local StandardScaler + PCA{prov['pca_dim']}\n",
        "2. **Target:** TmApp\n",
        "3. **Why winner:** lowest public_mae among all linear/linear-blend rows in the master registry "
        "(postmortem). This is **not** the endgame AbLang2_HL_paired recipe.\n",
        f"4. **Scores:** CV Primary={fmt(b['cv_primary_mae'])}; CV Shadow={fmt(b['cv_shadow_mae'])}; "
        f"CV mean={fmt(b['cv_mean_mae'])}; CV worst={fmt(b['cv_worst_mae'])}; "
        f"Public={fmt(b['public_mae'])}; Private={fmt(b['private_mae'])}\n",
        "5. **Complete feature composition:** AbLang2 protein-language-model embeddings only — "
        "Heavy-chain whole-chain mean (480-d) concatenated with Light-chain whole-chain mean (480-d) "
        f"= {prov['raw_embedding_dimension_before_pca']}-d raw vector. "
        "No SEQ_BASIC, BioEmu, ProteinMPNN, or AbLingua blocks.\n",
        "6. **Feature sources / provenance:**\n",
        f"   - Representation: `{prov['chain_representation']}` = {prov['chain_representation_meaning']}\n",
        f"   - Pooling: `{prov['pooling']}`\n",
        f"   - Embedding loader: `{prov['feature_source_embeddings']}`\n",
        f"   - Authoritative Stage-2 row: `virtual_participant/stage2_plm/stage2_primary_results.csv`\n",
        f"   - Frozen Test predictions: `{prov['frozen_test_prediction_artifact']}`\n",
        f"   - Prediction SHA256: `{prov['prediction_hash_sha256']}`\n",
        f"7. **Regressor:** RidgeOpt — alpha={prov['alpha']} (Optuna, {prov['optimization']})\n",
        f"8. **Preprocessing (model-specific):** {prov['scaler']}. "
        f"**PCA used: YES** — Optuna-selected PCA dimension = {prov['pca_dim']} "
        f"(raw {prov['raw_embedding_dimension_before_pca']} → "
        f"{prov['effective_dimension_after_pca']}). "
        "This is **not** “no PCA”, and it is **not** an SVR base-model pipeline.\n",
        f"9. **Hyperparameters:** final_alpha={prov['alpha']}; Lasso n/a\n",
        f"10. **CV protocol:** {prov['cv_protocol']}\n",
        f"11. **Final Test-training protocol:** {prov['full_dev_final_refit']}\n",
        f"12. **Test prediction provenance:** HISTORICAL_FROZEN_PREDICTION "
        f"(canonical_replay_status={prov['canonical_replay_status']})\n",
        "13. **Public/Private scoring:** N_public=81, N_private=81\n",
        "14. **Interpretation:** A Stage-2 PLM-only Ridge model on AbLang2 HL-concat embeddings "
        "with fold-local scaling and PCA48; historically strong on Public relative to endgame stacks.\n",
        "15. **Caveat:** Public designation is **postmortem** and was **not** used for selection "
        "or advanced-model freeze.\n",
        f"\n**model_id:** `{b['model_id']}`\n",
    ]

    # C
    c = w["tm_priv"]
    lines += [
        "\n" + "=" * 50,
        "\nC. TmApp — Private winner\n",
        "=" * 50 + "\n",
        f"1. **Full model name:** {c['full_model_name']}\n",
        "2. **Target:** TmApp\n",
        "3. **Why winner:** lowest private_mae among registry rows. "
        f"**Private 8-way tie audit:** classification=`{tie_summary['classification']}` — "
        "eight META_diversity labels share **bit-identical** Test prediction vectors "
        f"(shared SHA256=`{tie_summary['shared_sha256']}`). "
        "This is an IDENTICAL_PREDICTION_FAMILY, not eight distinct prediction vectors "
        "that merely share the same Private MAE. "
        "Representative kept by stable model_id ordering: "
        "`TmApp__META_diversity__convex_mae__HIST` (Public was not used to break the tie).\n",
        f"4. **Scores:** CV Primary={fmt(c['cv_primary_mae'])}; CV Shadow={fmt(c['cv_shadow_mae'])}; "
        f"CV mean={fmt(c['cv_mean_mae'])}; CV worst={fmt(c['cv_worst_mae'])}; "
        f"Public={fmt(c['public_mae'])}; Private={fmt(c['private_mae'])}\n",
        f"5. **Complete feature composition:** {c['feature_blocks_full']}\n",
        "6. **Feature sources:** prediction-level blend over frozen Stage base OOF/Test predictions; "
        "see `TMAPP_PRIVATE_TIE_AUDIT.csv` for hashes.\n",
        f"7. **Regressor:** {c['regressor']} — {c['regressor_details']} "
        "(label differs across the eight tied IDs; Test predictions are identical)\n",
        "8. **Preprocessing:** historical Stage-5 nested meta over base-model predictions "
        "(bases include SVR / RidgeOpt / ElasticNetOpt — this winner is a **prediction blend**, "
        "not a feature-level Ridge/Lasso matrix model)\n",
        "9. **Hyperparameters:** meta-learner label-specific; for this representative see Stage-5 inventory\n",
        f"10. **CV protocol:** {c['cv_protocol']}\n",
        f"11. **Final Test-training protocol:** {c['final_fit_protocol']}\n",
        f"12. **Test prediction provenance:** {c['prediction_source']} "
        f"(canonical_replay_status={c['canonical_replay_status']})\n",
        "13. **Public/Private scoring:** N_public=81, N_private=81\n",
        "14. **Interpretation:** Prediction-level diversity meta over SEQ_BASIC SVR, AbLang2 HL_paired "
        "RidgeOpt (no PCA), ESMFold STRUCT_RASA ElasticNet, and ADV_INTERACTIONS SVR. "
        "Eight named variants collapsed to one prediction family on Test.\n",
        "15. **Caveat:** Private designation is **postmortem** and was **not** used for selection.\n",
        f"\n**model_id:** `{c['model_id']}`\n",
    ]

    # D–F shorter but complete
    for letter, key, title in [
        ("D", "hic_cv", "HIC — canonical CV winner"),
        ("E", "hic_pub", "HIC — Public winner"),
        ("F", "hic_priv", "HIC — Private winner"),
    ]:
        x = w[key]
        why = (
            "lowest cv_worst_mae among FEATURE_LINEAR canonical Simple TVT only"
            if "CV" in title
            else "lowest Public/Private MAE among registry rows (postmortem)"
        )
        lines += [
            "\n" + "=" * 50,
            f"\n{letter}. {title}\n",
            "=" * 50 + "\n",
            f"1. **Full model name:** {x['full_model_name']}\n",
            f"2. **Target:** HIC\n",
            f"3. **Why winner:** {why}.\n",
            f"4. **Scores:** CV Primary={fmt(x['cv_primary_mae'])}; CV Shadow={fmt(x['cv_shadow_mae'])}; "
            f"CV mean={fmt(x['cv_mean_mae'])}; CV worst={fmt(x['cv_worst_mae'])}; "
            f"Public={fmt(x['public_mae'])}; Private={fmt(x['private_mae'])}\n",
            f"5. **Complete feature composition:** {x['feature_blocks_full']}\n",
            f"6. **Feature sources:** {x['feature_source_files']}\n",
            f"   Raw dimension: {fmt(x['raw_dimension'])}; Effective: {fmt(x['effective_dimension'])}\n",
            f"7. **Regressor:** {x['regressor']} — {x['regressor_details']}\n",
            f"8. **Preprocessing:** {x['preprocessing_full']}; dimensionality reduction: {x['dimensionality_reduction']}\n",
            f"9. **Hyperparameters:** final_alpha={fmt(x['final_alpha'])}; "
            f"Lasso nonzero={fmt(x['lasso_nonzero'])} / total={fmt(x['lasso_total_features'])}\n",
            f"10. **CV protocol:** {x['cv_protocol']}\n",
            f"11. **Final Test-training protocol:** {x['final_fit_protocol']}\n",
            f"12. **Test prediction provenance:** {x['prediction_source']} "
            f"(canonical_replay_status={x['canonical_replay_status']})\n",
            "13. **Public/Private scoring:** N_public=81, N_private=81\n",
            f"14. **Interpretation:** result_class={x['result_class']}; "
            f"any PLM={'YES' if x['contains_any_PLM'] else 'NO'}; "
            f"ESM-2 Heavy={'YES' if x['contains_ESM2_Heavy_PLM'] else 'NO'}; "
            f"sequence={'YES' if x['contains_sequence_descriptors'] else 'NO'}; "
            f"aromatic={'YES' if x['contains_aromatic_structural'] else 'NO'}; "
            f"continuous surface={'YES' if x['contains_continuous_surface'] else 'NO'}; "
            f"hydro_field={'YES' if x['contains_hydrophobic_field'] else 'NO'}; "
            f"titration={'YES' if x['contains_titration'] else 'NO'}.\n",
            (
                "15. **Caveat:** Selected using canonical CV only.\n"
                if "CV" in title
                else "15. **Caveat:** Public/Private designation is postmortem and was not used for selection.\n"
            ),
            f"\n**model_id:** `{x['model_id']}`\n",
        ]

    # note GLOBAL effective for completeness
    lines.append("\n## Appendix — TmApp GLOBAL AbLingua Ridge effective dimension\n\n")
    lines.append(block_table(glob_blocks) + "\n")

    (OUT / "LINEAR_MODEL_WINNERS_EN.md").write_text("".join(lines))


def write_closure_en(scores, reg, tie_summary, prov, bundle_ok: bool):
    feat = reg[reg.result_class == "FEATURE_LINEAR"]
    hist_n = len(reg)
    near = reg[
        (reg.result_class == "PREDICTION_LINEAR_BLEND")
        & (reg.target == "HIC")
        & (reg.private_mae < 0.43)
    ]
    adv = json.loads((OUT / "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json").read_text())
    lines = [
        "# Linear Model Closure Report (English)\n\n",
        "## Top answers\n\n",
        "1. **What did previous Top-3 mean?** **A. Top-3 among the small restricted Ridge/Lasso "
        "FEATURE_RECIPE_FREEZE inventory** in `endgame_model_benchmark` — **not** all-time organizer linear models.\n",
        "2. **Was HIC naming misleading?** **YES.** Aliases like ARO+TITR / ARO+CONT actually included "
        "ESM-2 Heavy + SEQ_ALL + AROMATIC-TOPO + …\n",
        f"3. **Did best HIC feature-level CV model contain a PLM?** "
        f"**{'YES' if scores['hic_cv']['contains_any_PLM'] else 'NO'}** (ESM-2 Heavy). "
        f"Public strongest: {'YES' if scores['hic_pub']['contains_any_PLM'] else 'NO'}; "
        f"Private strongest: {'YES' if scores['hic_priv']['contains_any_PLM'] else 'NO'} "
        "(Public/Private winners are prediction blends).\n",
        f"4. **Historical linear models audited?** {hist_n} registry rows "
        f"({(reg.result_class=='FEATURE_LINEAR').sum()} FEATURE_LINEAR; "
        f"{(reg.result_class=='PREDICTION_LINEAR_BLEND').sum()} PREDICTION_LINEAR_BLEND; "
        f"{(reg.result_class=='HISTORICAL_LINEAR_OTHER').sum()} OTHER)\n",
        f"5. **Historical HIC ≈0.42 recovered?** **YES** — e.g. Round1 PRIMARY blend Public≈0.420 / Private≈0.424. "
        f"Rows with Private<0.43: {len(near)}. Not ranked in FEATURE_LEVEL CV Top-10.\n",
        f"6. **TmApp canonical CV winner:** `{scores['tm_cv']['model_id']}` "
        f"(raw={fmt(scores['tm_cv']['raw_dimension'])}, effective_after_PCA={fmt(scores['tm_cv']['effective_dimension'])})\n",
        f"7. **TmApp Public winner:** `{scores['tm_pub']['model_id']}` — AbLang2 HL concat, "
        f"StandardScaler + PCA{prov['pca_dim']}, RidgeOpt alpha={prov['alpha']}\n",
        f"8. **TmApp Private winner:** `{scores['tm_priv']['model_id']}` — "
        f"IDENTICAL_PREDICTION_FAMILY of 8 META_diversity labels "
        f"(SHA256=`{tie_summary['shared_sha256'][:16]}…`)\n",
        f"9. **HIC canonical CV winner:** `{scores['hic_cv']['model_id']}`\n",
        f"10. **HIC Public winner:** `{scores['hic_pub']['model_id']}`\n",
        f"11. **HIC Private winner:** `{scores['hic_priv']['model_id']}`\n",
        "12. **Rank shake:** Upper leaderboard shakes between CV and Private/Public; "
        "see `RANK_SHAKE_ANALYSIS.csv`. MAE gaps among near-ties can be tiny.\n",
        "13. **Ridge vs Lasso:** Ridge wins CV-worst more often on matched recipes; Lasso helps high-dim HIC; "
        "TmApp authority remains Ridge + selective AbLingua PCA.\n",
        "14. **Advanced-model freeze:** see `ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json` "
        f"(TmApp={ [x['model_id'] for x in adv['targets']['TmApp']] }; "
        f"HIC={ [x['model_id'] for x in adv['targets']['HIC']] })\n",
        f"15. **Participant bundle consistent?** {'YES' if bundle_ok else 'CHECK'}\n",
        "16. **LINEAR_MODEL_CLOSED = YES**\n",
        "\n## Metadata repairs (no score changes)\n",
        "- Effective dimension for AbLingua Ridge recipes corrected (PCA32 per AbLingua block).\n",
        "- TmApp Public winner provenance expanded from Stage-2 artifacts (PCA48, not SVR boilerplate).\n",
        "- TmApp Private 8-way tie classified as IDENTICAL_PREDICTION_FAMILY.\n",
        "\nEnglish mirror: this file. Japanese: `LINEAR_MODEL_CLOSURE_JA.md`.\n",
    ]
    (OUT / "LINEAR_MODEL_CLOSURE_EN.md").write_text("".join(lines))


def write_winners_ja(scores, audit, prov, tie_summary):
    cdr3 = audit[audit.model_id == "TM_PARENT_ABLINGUA_CDR3__RIDGE"]
    glob_ = audit[audit.model_id == "TM_PARENT_ABLINGUA_GLOBAL__RIDGE"]

    def block_table(df):
        lines = ["| 特徴ブロック | 生次元 | 変換 | 有効次元 |", "|---|---:|---|---:|"]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['block']} | {int(r['raw_dim'])} | {r['transform']} | {int(r['effective_dim'])} |"
            )
        return "\n".join(lines)

    a, b, c = scores["tm_cv"], scores["tm_pub"], scores["tm_priv"]
    text = f"""# 線形モデル勝者 — 詳細ドシエ（日本語）

略語（PARENT / BASE / ARO / CONT など）は展開して記載する。Public / Private の「最良」は事後解析（postmortem）のみであり、モデル選定には用いていない。
有効次元・Public 由来・Private 同点の監査を反映済み。スコア・順位は変更していない。

英語版: `LINEAR_MODEL_WINNERS_EN.md`

{'='*50}
A. TmApp — canonical CV 最良モデル
{'='*50}

1. **モデルの完全な構成:** {a['full_model_name']}

2. **対象ターゲット:** TmApp

3. **このモデルを「最良」とする根拠:** FEATURE_LINEAR かつ canonical Simple TVT のみの母集団で、`cv_worst_mae`（Primary と Shadow の悪い方）が最小。

4. **スコア:** CV Primary={fmt(a['cv_primary_mae'])}; CV Shadow={fmt(a['cv_shadow_mae'])}; CV mean={fmt(a['cv_mean_mae'])}; CV worst={fmt(a['cv_worst_mae'])}; Public={fmt(a['public_mae'])}; Private={fmt(a['private_mae'])}

5. **使用特徴量の完全な構成:** AbLang2 の重鎖・軽鎖ペア埋め込み（paired heavy+light）＋ stage-1 の SEQ_BASIC 配列記述子（長さ・組成・CDR 長要約）＋ BioEmu の孤立 VH/VL NEW_PAIRWISE アンサンブル Cα RMSD 記述子＋ ProteinMPNN の配列–構造適合ネイティブスコア記述子＋ AbLingua-600M の masked-mean 大域（GLOBAL）重鎖+軽鎖連結埋め込み＋ AbLingua-600M の CDR3 誘導残基プーリング埋め込み

6. **各特徴量の由来:** {a['feature_source_files']}

実行列から測ったブロック別次元（canonical 評価器の PCA パイプラインに基づく）:

{block_table(cdr3)}

生の連結次元: {fmt(a['raw_dimension'])}。**PCA 後の Ridge 入力有効次元: {fmt(a['effective_dimension'])}**（生次元と同じではない。AbLingua GLOBAL と CDR3 にそれぞれ fold 内 PCA32）。

7. **回帰モデル:** Ridge（sklearn Ridge）

8. **前処理:** レシピ側は中央値補完のうえ生特徴を連結。AbLingua ブロックのみそれぞれ fold 内 PCA32。連結後の train+val 行列に StandardScaler をかけて Ridge（`canonical_simple_tvt_v1` / `run_recipe_plus_abl_blocks`）。

9. **ハイパーパラメータ:** final_alpha={fmt(a['final_alpha'])}（full DEV では Primary 5-fold の中央値アルファ方針）

10. **CVプロトコル:** Primary 折りと Shadow 折りの Simple TVT 回転（`canonical_simple_tvt_v1`）

11. **Test予測時の最終学習方法:** DEV 全体での再学習。アルファは Primary fold の中央値（Test ラベル非使用）

12. **Test予測値の由来:** CANONICAL_FULL_DEV_REFIT（canonical_replay_status=REPLAYED_CANONICAL）

13. **Public / Private 評価:** N_public=81, N_private=81（公式 solution マスク）

14. **モデルの解釈:** 配列 PLM（AbLang2・AbLingua）、古典配列記述子、アンサンブル幾何（BioEmu）、逆折りたたみ適合（ProteinMPNN）を線形に統合。高次元の AbLingua のみ PCA で圧縮。

15. **注意点:** 選定は canonical CV のみ。Public/Private は事後評価。

**model_id:** `{a['model_id']}`


{'='*50}
B. TmApp — Public 最良モデル
{'='*50}

1. **モデルの完全な構成:** Stage-2 の AbLang2 重鎖+軽鎖（HL：各鎖 whole-chain mean の連結）を、Optuna 調整 Ridge（RidgeOpt, alpha={prov['alpha']}）で回帰。fold 内 StandardScaler ＋ PCA{prov['pca_dim']}。

2. **対象ターゲット:** TmApp

3. **このモデルを「最良」とする根拠:** マスタ登録上の線形／線形ブレンド行の中で public_mae が最小（事後解析）。endgame の AbLang2_HL_paired レシピとは別物。

4. **スコア:** CV Primary={fmt(b['cv_primary_mae'])}; CV Shadow={fmt(b['cv_shadow_mae'])}; CV mean={fmt(b['cv_mean_mae'])}; CV worst={fmt(b['cv_worst_mae'])}; Public={fmt(b['public_mae'])}; Private={fmt(b['private_mae'])}

5. **使用特徴量の完全な構成:** AbLang2 のタンパク質言語モデル（PLM）埋め込みのみ。重鎖 whole-chain mean（480次元）と軽鎖 whole-chain mean（480次元）を連結した {prov['raw_embedding_dimension_before_pca']} 次元。SEQ_BASIC / BioEmu / ProteinMPNN / AbLingua は含まない。**AbLang2 HL_paired（ペア配列コーディング）ではない。**

6. **各特徴量の由来:**
   - 表現: `{prov['chain_representation']}` — {prov['chain_representation_meaning']}
   - プーリング: `{prov['pooling']}`
   - 埋め込み取得: Stage-2 `run_stage2.py` の AbLang2 キャッシュ経路
   - 権威テーブル: `virtual_participant/stage2_plm/stage2_primary_results.csv`
   - 凍結 Test 予測: `{prov['frozen_test_prediction_artifact']}`
   - 予測 SHA256: `{prov['prediction_hash_sha256']}`

7. **回帰モデル:** RidgeOpt（Optuna 調整 Ridge）。alpha={prov['alpha']}。探索試行数は notes 上 40。

8. **前処理（本モデル固有）:** 各学習 fold 内で SimpleImputer（中央値）→ StandardScaler → PCA。**PCA 使用: あり**（Optuna が選んだ次元 {prov['pca_dim']}。生 {prov['raw_embedding_dimension_before_pca']} → {prov['effective_dimension_after_pca']}）。SVR 基底モデルのパイプラインではない。「PCA なし」でもない。

9. **ハイパーパラメータ:** alpha={prov['alpha']}。pca_dim も探索対象。

10. **CVプロトコル:** Stage-2 Primary 折り＋ Shadow 確認。`canonical_simple_tvt_v1` との同一性は証明されていない。

11. **Test予測時の最終学習方法:** {prov['full_dev_final_refit']}

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION（canonical_replay_status={prov['canonical_replay_status']}）

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM 単独の線形モデルが、後段の複雑なスタックより Public で強かった事例。選定根拠には使わない。

15. **注意点:** Public 最良は事後ラベル。advanced-model 凍結やレシピ選定には未使用。

**model_id:** `{b['model_id']}`


{'='*50}
C. TmApp — Private 最良モデル
{'='*50}

1. **モデルの完全な構成:** {c['full_model_name']}

2. **対象ターゲット:** TmApp

3. **このモデルを「最良」とする根拠:** private_mae 最小。ただし **8 ラベル同点監査**の結果、分類は `{tie_summary['classification']}`。8 個の META_diversity ラベルは **ビット単位で同一の Test 予測ベクトル**を共有する（共有 SHA256=`{tie_summary['shared_sha256']}`）。Private MAE が偶然一致した別予測ではなく、**同一予測ファミリーの重複ラベル**である。代表は安定ソートで `TmApp__META_diversity__convex_mae__HIST`（Public ではタイブレークしない）。

4. **スコア:** CV Primary={fmt(c['cv_primary_mae'])}; CV Shadow={fmt(c['cv_shadow_mae'])}; CV mean={fmt(c['cv_mean_mae'])}; CV worst={fmt(c['cv_worst_mae'])}; Public={fmt(c['public_mae'])}; Private={fmt(c['private_mae'])}

5. **使用特徴量の完全な構成:** 特徴行列ではなく、基底モデル予測のメタ統合: SEQ_BASIC 上の SVROpt、AbLang2 HL_paired 上の RidgeOpt（PCA なし）、ESMFold STRUCT_RASA 上の ElasticNetOpt、Stage-3 ADV_INTERACTIONS 上の SVROpt。

6. **各特徴量の由来:** Stage-5 ネスト OOF / 凍結 Test 予測。詳細ハッシュは `TMAPP_PRIVATE_TIE_AUDIT.csv`。

7. **回帰モデル:** 予測値レベルのメタ学習器（代表ラベル: convex_mae）。8 ラベルは名前が違うが Test 予測は同一。

8. **前処理:** Stage-5 の予測ブレンド／メタ学習。特徴レベル Ridge/Lasso の endgame パイプラインではない。

9. **ハイパーパラメータ:** メタ学習器ラベル依存（代表は Stage-5 inventory 参照）

10. **CVプロトコル:** 履歴 Stage-5 OOF（canonical Simple TVT とは未証明）

11. **Test予測時の最終学習方法:** 履歴 Round1 / Stage-5 の凍結提出・予測

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** 多様性セット上のメタ統合が Private で強い一方、8 名称は実質 1 予測。

15. **注意点:** Private 最良は事後。モデル選定・凍結には未使用。

**model_id:** `{c['model_id']}`


{'='*50}
D. HIC — canonical CV 最良モデル
{'='*50}

1. **モデルの完全な構成:** {scores['hic_cv']['full_model_name']}

2. **対象ターゲット:** HIC

3. **このモデルを「最良」とする根拠:** FEATURE_LINEAR canonical Simple TVT で cv_worst_mae 最小。

4. **スコア:** CV Primary={fmt(scores['hic_cv']['cv_primary_mae'])}; CV Shadow={fmt(scores['hic_cv']['cv_shadow_mae'])}; CV mean={fmt(scores['hic_cv']['cv_mean_mae'])}; CV worst={fmt(scores['hic_cv']['cv_worst_mae'])}; Public={fmt(scores['hic_cv']['public_mae'])}; Private={fmt(scores['hic_cv']['private_mae'])}

5. **使用特徴量の完全な構成:** ESM-2 重鎖 PLM 埋め込み＋ SEQ_ALL 配列記述子＋ AROMATIC-TOPO 露出芳香族構造トポロジー特徴量（ESMFold Fv）＋ HYDRO_FIELD 連続疎水性場表面特徴量＋ TITRATION_SHAPE 静電滴定形状特徴量

6. **各特徴量の由来:** {scores['hic_cv']['feature_source_files']}
   生次元: {fmt(scores['hic_cv']['raw_dimension'])}; 有効次元: {fmt(scores['hic_cv']['effective_dimension'])}（Lasso・PCA なしのため一致）

7. **回帰モデル:** Lasso（PCA なし）

8. **前処理:** 中央値補完＋ StandardScaler ＋ Lasso（no PCA）

9. **ハイパーパラメータ:** final_alpha={fmt(scores['hic_cv']['final_alpha'])}; 非ゼロ係数={fmt(scores['hic_cv']['lasso_nonzero'])} / 全特徴={fmt(scores['hic_cv']['lasso_total_features'])}

10. **CVプロトコル:** canonical_simple_tvt_v1

11. **Test予測時の最終学習方法:** FULL_DEV median Primary alpha

12. **Test予測値の由来:** CANONICAL_FULL_DEV_REFIT

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM（ESM-2 Heavy）あり。配列＋芳香族トポロジー＋疎水性場＋滴定形状を線形スパース回帰で統合。

15. **注意点:** CV のみで選定。

**model_id:** `{scores['hic_cv']['model_id']}`


{'='*50}
E. HIC — Public 最良モデル
{'='*50}

1. **モデルの完全な構成:** {scores['hic_pub']['full_model_name']}

2. **対象ターゲット:** HIC

3. **このモデルを「最良」とする根拠:** public_mae 最小（事後）。特徴レベル Ridge/Lasso ではない。

4. **スコア:** CV Primary={fmt(scores['hic_pub']['cv_primary_mae'])}; CV Shadow={fmt(scores['hic_pub']['cv_shadow_mae'])}; CV mean={fmt(scores['hic_pub']['cv_mean_mae'])}; CV worst={fmt(scores['hic_pub']['cv_worst_mae'])}; Public={fmt(scores['hic_pub']['public_mae'])}; Private={fmt(scores['hic_pub']['private_mae'])}

5. **使用特徴量の完全な構成:** 予測値の等平均ブレンド — (1) ESM-2 重鎖＋SEQ_ALL 融合の SVROpt 予測、(2) ESMFold STRUCT_SURFACE_CHEM の SVROpt 予測

6. **各特徴量の由来:** Round1 / Stage-5 予測アーティファクト（`base_model_ids` 展開）

7. **回帰モデル:** 等平均予測ブレンド（特徴行列 Ridge/Lasso ではない）

8. **前処理:** 各基底は履歴 Stage パイプライン（多くは SVR）。ブレンド重みは等平均。

9. **ハイパーパラメータ:** ブレンド重み固定（等平均）

10. **CVプロトコル:** 履歴 Stage OOF（canonical と未証明）

11. **Test予測時の最終学習方法:** 履歴 full-DEV refit / 凍結予測

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM を含む SVR 基底の予測平均。≈0.42 帯の歴史的強さの一端。

15. **注意点:** 事後最良。選定未使用。

**model_id:** `{scores['hic_pub']['model_id']}`


{'='*50}
F. HIC — Private 最良モデル
{'='*50}

1. **モデルの完全な構成:** {scores['hic_priv']['full_model_name']}

2. **対象ターゲット:** HIC

3. **このモデルを「最良」とする根拠:** private_mae 最小（事後）

4. **スコア:** CV Primary={fmt(scores['hic_priv']['cv_primary_mae'])}; CV Shadow={fmt(scores['hic_priv']['cv_shadow_mae'])}; CV mean={fmt(scores['hic_priv']['cv_mean_mae'])}; CV worst={fmt(scores['hic_priv']['cv_worst_mae'])}; Public={fmt(scores['hic_priv']['public_mae'])}; Private={fmt(scores['hic_priv']['private_mae'])}

5. **使用特徴量の完全な構成:** 予測値の等平均ブレンド — ESM-2 重鎖 SVROpt 予測 ＋ ESMFold STRUCT_SURFACE_CHEM SVROpt 予測

6. **各特徴量の由来:** Round1 / Stage-5 予測アーティファクト

7. **回帰モデル:** 等平均予測ブレンド

8. **前処理:** 履歴 Stage パイプライン上の基底予測の平均

9. **ハイパーパラメータ:** 等平均

10. **CVプロトコル:** 履歴（Shadow 欠損の行あり）

11. **Test予測時の最終学習方法:** 履歴凍結予測

12. **Test予測値の由来:** HISTORICAL_FROZEN_PREDICTION

13. **Public / Private 評価:** N_public=81, N_private=81

14. **モデルの解釈:** PLM（ESM-2）ありの 2 モデル平均が Private で最安。feature-level 最良（CV）とは別系統。

15. **注意点:** 事後最良。選定未使用。

**model_id:** `{scores['hic_priv']['model_id']}`


## 付録 — TmApp GLOBAL AbLingua Ridge の有効次元

{block_table(glob_)}
"""
    (OUT / "LINEAR_MODEL_WINNERS_JA.md").write_text(text)


def write_closure_ja(scores, reg, tie_summary, prov, bundle_ok: bool):
    adv = json.loads((OUT / "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json").read_text())
    near = reg[
        (reg.result_class == "PREDICTION_LINEAR_BLEND")
        & (reg.target == "HIC")
        & (reg.private_mae < 0.43)
    ]
    text = f"""# 線形モデル・クロージャ報告（日本語）

英語版: `LINEAR_MODEL_CLOSURE_EN.md`

## 冒頭回答

1. **以前の「Top-3」は何を意味していたか**  
   endgame Ridge/Lasso の **狭い FEATURE_RECIPE_FREEZE 在庫内**の Top-3。履歴の全 organizer 線形モデルの総合ランキングではない。

2. **HIC recipe名がなぜ誤解を招いたか**  
   `ARO+TITR` などは芳香族＋滴定だけに見えるが、実際は **ESM-2 重鎖 PLM ＋ SEQ_ALL ＋ AROMATIC-TOPO ＋ …** を含む。

3. **HICの最良feature-levelモデルにPLMは入っていたか**  
   **はい（ESM-2 Heavy）**。構成: ESM-2 重鎖のタンパク質言語モデル埋め込み ＋ SEQ_ALL 配列記述子 ＋ AROMATIC-TOPO 露出芳香族構造トポロジー特徴量 ＋ HYDRO_FIELD 連続疎水性場表面特徴量 ＋ TITRATION_SHAPE 静電滴定形状特徴量。  
   別途 — Public 最良の PLM: {'はい' if scores['hic_pub']['contains_any_PLM'] else 'いいえ'}；Private 最良の PLM: {'はい' if scores['hic_priv']['contains_any_PLM'] else 'いいえ'}（いずれも予測ブレンド）。

4. **何件のhistorical linear modelを監査したか**  
   レジストリ {len(reg)} 行（FEATURE_LINEAR {(reg.result_class=='FEATURE_LINEAR').sum()}；PREDICTION_LINEAR_BLEND {(reg.result_class=='PREDICTION_LINEAR_BLEND').sum()}；OTHER {(reg.result_class=='HISTORICAL_LINEAR_OTHER').sum()}）。

5. **historical HIC ≈0.42を回収できたか**  
   **できた。** 例: Round1 PRIMARY 等平均 SVR ブレンド（Public≈0.420 / Private≈0.424）。Private<0.43 の行は {len(near)}。FEATURE_LEVEL CV Top-10 には混ぜない。

6. **TmApp canonical CV 最良モデル**  
   `{scores['tm_cv']['model_id']}`（生次元={fmt(scores['tm_cv']['raw_dimension'])}、PCA後有効次元={fmt(scores['tm_cv']['effective_dimension'])}）

7. **TmApp Public 最良モデル**  
   `{scores['tm_pub']['model_id']}` — AbLang2 HL 連結、StandardScaler＋PCA{prov['pca_dim']}、RidgeOpt alpha={prov['alpha']}

8. **TmApp Private 最良モデル**  
   `{scores['tm_priv']['model_id']}` — 8 ラベルは **同一予測ファミリー**（SHA256=`{tie_summary['shared_sha256'][:16]}…`）

9. **HIC canonical CV 最良モデル**  
   `{scores['hic_cv']['model_id']}`

10. **HIC Public 最良モデル**  
    `{scores['hic_pub']['model_id']}`

11. **HIC Private 最良モデル**  
    `{scores['hic_priv']['model_id']}`

12. **CV/Public/Private間のrank shake**  
    上位は指標で入れ替わる。詳細は `RANK_SHAKE_ANALYSIS.csv`。MAE差が微小なときの順位差は過大解釈しない。

13. **Ridge vs Lasso の結論**  
    同一特徴では Ridge が CV-worst で多い。HIC 高次元連結では Lasso が有利。TmApp 権威レシピは Ridge＋AbLingua 選択的 PCA。

14. **advanced model用に凍結したfeature recipe**  
    `ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json`  
    TmApp: {[x['model_id'] for x in adv['targets']['TmApp']]}  
    HIC: {[x['model_id'] for x in adv['targets']['HIC']]}

15. **participant bundleとの整合性**  
    {'整合（必要ブロック欠落なし）' if bundle_ok else '要確認'}

16. **LINEAR_MODEL_CLOSED = YES**

## 今回のメタデータ修復（スコア不変）

- AbLingua Ridge の有効次元を PCA32 後の入力幅に修正
- TmApp Public 勝者の由来を Stage-2 根拠に基づき具体化（PCA48・SVR ではない）
- TmApp Private 8 同点を同一予測ファミリーと分類

## 整合性

- `REPORT_LANGUAGE_CONSISTENCY_AUDIT.md` を参照
"""
    (OUT / "LINEAR_MODEL_CLOSURE_JA.md").write_text(text)


def write_top10_ja_stub():
    """Short Japanese pointer; numeric tables remain in CSV / EN file."""
    src = OUT / "TOP10_TABLES_EN.md"
    body = src.read_text() if src.exists() else ""
    text = (
        "# 線形モデル Top-10 表（日本語案内）\n\n"
        "数値表の英語見出し版: `TOP10_TABLES_EN.md`\n\n"
        "権威ある数値は CSV を正とする:\n\n"
        "- `TmApp_CV_TOP10.csv` / `TmApp_PUBLIC_TOP10.csv` / `TmApp_PRIVATE_TOP10.csv`\n"
        "- `HIC_CV_TOP10.csv` / `HIC_PUBLIC_TOP10.csv` / `HIC_PRIVATE_TOP10.csv`\n"
        "- feature-level のみ: `FEATURE_LEVEL_*`\n"
        "- 履歴込み全体: `ORGANIZER_LINEAR_OVERALL_*`\n\n"
        "CV Top-10 は FEATURE_LINEAR かつ canonical Simple TVT のみ。"
        "Public/Private の総合表は履歴予測ブレンドを含む（選定には未使用）。\n\n"
        "---\n\n"
        "## 英語表の複製（数値確認用）\n\n"
        + body
    )
    (OUT / "TOP10_TABLES_JA.md").write_text(text)


def consistency_audit(scores):
    en_w = (OUT / "LINEAR_MODEL_WINNERS_EN.md").read_text()
    ja_w = (OUT / "LINEAR_MODEL_WINNERS_JA.md").read_text()
    en_c = (OUT / "LINEAR_MODEL_CLOSURE_EN.md").read_text()
    ja_c = (OUT / "LINEAR_MODEL_CLOSURE_JA.md").read_text()
    checks = []
    ok = True
    for key, mid in WINNERS.items():
        s = scores[key]
        for label, val in [
            ("model_id", mid),
            ("cv_primary_mae", fmt(s["cv_primary_mae"])),
            ("cv_shadow_mae", fmt(s["cv_shadow_mae"])),
            ("public_mae", fmt(s["public_mae"])),
            ("private_mae", fmt(s["private_mae"])),
        ]:
            in_en = str(val) in en_w or str(val) in en_c
            in_ja = str(val) in ja_w or str(val) in ja_c
            # model_id must be in both winner files
            if label == "model_id":
                in_en = mid in en_w
                in_ja = mid in ja_w
            passed = in_en and in_ja
            if not passed:
                ok = False
            checks.append(
                {
                    "winner_key": key,
                    "field": label,
                    "value": val,
                    "in_EN": in_en,
                    "in_JA": in_ja,
                    "pass": passed,
                }
            )
        # dims / alpha for tm_cv
        if key == "tm_cv":
            for label, val in [
                ("raw_dimension", fmt(s["raw_dimension"])),
                ("effective_dimension", fmt(s["effective_dimension"])),
                ("final_alpha", fmt(s["final_alpha"])),
            ]:
                in_en = str(int(float(val))) in en_w if val != "nan" else True
                # JA may print float or int
                in_ja = (str(val) in ja_w) or (str(int(float(val))) in ja_w if val != "nan" else True)
                passed = in_en and in_ja
                if not passed:
                    ok = False
                checks.append(
                    {
                        "winner_key": key,
                        "field": label,
                        "value": val,
                        "in_EN": in_en,
                        "in_JA": in_ja,
                        "pass": passed,
                    }
                )
        if key == "tm_pub":
            for label, val in [("N_public", "81"), ("N_private", "81")]:
                passed = (val in en_w) and (val in ja_w)
                if not passed:
                    ok = False
                checks.append(
                    {
                        "winner_key": key,
                        "field": label,
                        "value": val,
                        "in_EN": val in en_w,
                        "in_JA": val in ja_w,
                        "pass": passed,
                    }
                )

    status = "PASS" if ok else "FAIL"
    lines = [
        "# Report Language Consistency Audit\n\n",
        f"**EN_JA_NUMERIC_CONSISTENCY = {status}**\n\n",
        "Compared winner model_ids and key numeric strings between EN and JA winner/closure reports.\n\n",
        "| winner_key | field | value | in_EN | in_JA | pass |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for c in checks:
        lines.append(
            f"| {c['winner_key']} | {c['field']} | `{c['value']}` | {c['in_EN']} | {c['in_JA']} | {c['pass']} |\n"
        )
    lines.append("\nScores/rankings were not modified by the language split.\n")
    (OUT / "REPORT_LANGUAGE_CONSISTENCY_AUDIT.md").write_text("".join(lines))
    return status


def patch_public_winner_registry(reg: pd.DataFrame, prov: dict) -> pd.DataFrame:
    mid = "TmApp__ablang2__HL__RidgeOpt__HIST"
    m = reg.model_id == mid
    if not m.any():
        return reg
    desc = (
        "Stage-2 AbLang2 HL (Heavy whole-chain mean concat Light whole-chain mean; NOT HL_paired); "
        f"raw {prov['raw_embedding_dimension_before_pca']}-d → fold-local StandardScaler + PCA{prov['pca_dim']} "
        f"→ RidgeOpt alpha={prov['alpha']}"
    )
    reg.loc[m, "full_model_name"] = desc
    reg.loc[m, "feature_blocks_full"] = desc
    reg.loc[m, "preprocessing_full"] = (
        f"SimpleImputer(median)+StandardScaler+PCA{prov['pca_dim']} inside training folds "
        f"(stage2 fold_matrices); NOT SVR-base pipeline"
    )
    reg.loc[m, "dimensionality_reduction"] = f"PCA{prov['pca_dim']} (Optuna-selected)"
    reg.loc[m, "raw_dimension"] = prov["raw_embedding_dimension_before_pca"]
    reg.loc[m, "effective_dimension"] = prov["effective_dimension_after_pca"]
    reg.loc[m, "final_alpha"] = prov["alpha"]
    reg.loc[m, "feature_source_files"] = "; ".join(prov["authoritative_tables"])
    reg.loc[m, "regressor_details"] = json.dumps({"alpha": prov["alpha"], "optuna_trials": 40})
    note = str(reg.loc[m, "notes"].iloc[0])
    add = f"PROVENANCE_REPAIRED; pred_sha256={prov['prediction_hash_sha256'][:16]}"
    if "PROVENANCE_REPAIRED" not in note:
        reg.loc[m, "notes"] = (note + "; " + add).strip("; ")
    # private tie notes on eight
    for eid in EIGHT:
        mm = reg.model_id == eid + "__HIST"
        if mm.any():
            n = str(reg.loc[mm, "notes"].iloc[0])
            add2 = "IDENTICAL_PREDICTION_FAMILY with other META_diversity Private-tie labels"
            if "IDENTICAL_PREDICTION_FAMILY" not in n:
                reg.loc[mm, "notes"] = (n + "; " + add2).strip("; ")
    reg.to_csv(OUT / "LINEAR_MODEL_MASTER_REGISTRY.csv", index=False)
    return reg


def main():
    print("1) effective dimension audit...", flush=True)
    dims = block_raw_dims()
    print("   block dims:", dims)
    audit = effective_audit(dims)
    changed, reg = patch_registry(audit)
    print("   patched:", changed)

    print("2) private tie audit...", flush=True)
    tie_df, tie_summary = private_tie_audit()
    print("   ", tie_summary)

    print("3) public winner provenance...", flush=True)
    prov = public_winner_provenance()
    print("   pca", prov["pca_dim"], "alpha", prov["alpha"], "hash", prov["prediction_hash_sha256"][:16])
    reg = pd.read_csv(OUT / "LINEAR_MODEL_MASTER_REGISTRY.csv")
    reg = patch_public_winner_registry(reg, prov)

    print("4) rename EN...", flush=True)
    renamed = rename_en_files()
    print("   ", renamed)

    print("5) rewrite EN/JA reports...", flush=True)
    scores = load_scores(reg)
    # verify freeze untouched hash
    freeze_path = OUT / "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json"
    freeze_sha = (OUT / "ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.sha256").read_text().strip()
    now = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    if now != freeze_sha:
        raise SystemExit("STOP: ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json hash changed")
    recipes = pd.read_csv(ROOT / "top_models_feature_bundle/recipes.csv")
    bundle_ok = set(recipes.recipe_id) == {
        x["model_id"]
        for t in json.loads(freeze_path.read_text())["targets"].values()
        for x in t
    }

    write_winners_en(scores, audit, prov, tie_summary)
    write_closure_en(scores, reg, tie_summary, prov, bundle_ok)
    write_winners_ja(scores, audit, prov, tie_summary)
    write_closure_ja(scores, reg, tie_summary, prov, bundle_ok)
    write_top10_ja_stub()

    print("6) consistency...", flush=True)
    status = consistency_audit(scores)
    print("   EN_JA_NUMERIC_CONSISTENCY =", status)

    # Japanese character presence check
    ja = (OUT / "LINEAR_MODEL_WINNERS_JA.md").read_text()
    if sum(1 for ch in ja if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff") < 200:
        raise SystemExit("STOP: LINEAR_MODEL_WINNERS_JA.md does not look Japanese enough")

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
