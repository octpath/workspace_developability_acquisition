#!/usr/bin/env python3
"""Phase 2A: audit + backfill historical Transformer experiments into drilldown.

Does NOT retrain. Copies local untracked predictions; scores from authority CSVs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BUNDLE = REPO / "top_models_feature_bundle"
AO = BUNDLE / "advanced_outputs"
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_codes import issue_code, load_codes  # noqa: E402
from _lib import EXPERIMENTS_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical issuance order (approved Plan)
# ---------------------------------------------------------------------------
EXPERIMENTS: list[dict[str, Any]] = [
    # Phase1 sequence TmApp
    dict(experiment_id="TRF_TM_SCRATCH_MIN_CONCAT", source_model_id="TMS1", target="TmApp",
         transformer_type="SCRATCH", plm_source="NONE", annotation_mode="MINIMAL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg",
         content_mode="scratch", input_space="SCRATCH_RESIDUE_SEQUENCE", group="phase1_seq"),
    dict(experiment_id="TRF_TM_SCRATCH_FULL_CONCAT", source_model_id="TMS2", target="TmApp",
         transformer_type="SCRATCH", plm_source="NONE", annotation_mode="FULL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg",
         content_mode="scratch", input_space="SCRATCH_RESIDUE_SEQUENCE", group="phase1_seq"),
    dict(experiment_id="TRF_TM_SCRATCH_FULL_MEAN", source_model_id="TMS3", target="TmApp",
         transformer_type="SCRATCH", plm_source="NONE", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg",
         content_mode="scratch", input_space="SCRATCH_RESIDUE_SEQUENCE", group="phase1_seq"),
    dict(experiment_id="TRF_TM_ABLINGUA_MIN_CONCAT", source_model_id="TMF1", target="TmApp",
         transformer_type="FROZEN_PLM", plm_source="ABLINGUA", annotation_mode="MINIMAL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="phase1_seq"),
    dict(experiment_id="TRF_TM_ABLINGUA_FULL_CONCAT", source_model_id="TMF2", target="TmApp",
         transformer_type="FROZEN_PLM", plm_source="ABLINGUA", annotation_mode="FULL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="phase1_seq"),
    dict(experiment_id="TRF_TM_ABLINGUA_FULL_MEAN", source_model_id="TMF3", target="TmApp",
         transformer_type="FROZEN_PLM", plm_source="ABLINGUA", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="phase1_seq"),
    # Phase1 sequence HIC
    dict(experiment_id="TRF_HIC_SCRATCH_MIN_HONLY", source_model_id="HICS1", target="HIC",
         transformer_type="SCRATCH", plm_source="NONE", annotation_mode="MINIMAL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg",
         content_mode="scratch", input_space="SCRATCH_RESIDUE_SEQUENCE", group="phase1_seq"),
    dict(experiment_id="TRF_HIC_SCRATCH_FULL_HONLY", source_model_id="HICS2", target="HIC",
         transformer_type="SCRATCH", plm_source="NONE", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg",
         content_mode="scratch", input_space="SCRATCH_RESIDUE_SEQUENCE", group="phase1_seq"),
    dict(experiment_id="TRF_HIC_ESM2_MIN_HONLY", source_model_id="HICF1", target="HIC",
         transformer_type="FROZEN_PLM", plm_source="ESM2", annotation_mode="MINIMAL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="phase1_seq"),
    dict(experiment_id="TRF_HIC_ESM2_FULL_HONLY", source_model_id="HICF2", target="HIC",
         transformer_type="FROZEN_PLM", plm_source="ESM2", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="phase1_seq"),
    # Phase1 fusion TmApp scratch
    dict(experiment_id="TRF_TM_SCRATCH_FULL_MEAN_FUS_ABLINGUA_CDR3",
         source_model_id="TMS3__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="NONE", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg", content_mode="scratch",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_PARENT_ABLINGUA_CDR3__RIDGE",
         fusion_feature_set_id="FS_TM_ABLINGUA_CDR3", base_variant="TMS3", group="phase1_fus"),
    dict(experiment_id="TRF_TM_SCRATCH_FULL_MEAN_FUS_ABLINGUA_GLOBAL",
         source_model_id="TMS3__FUSION__TM_PARENT_ABLINGUA_GLOBAL__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="NONE", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg", content_mode="scratch",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
         fusion_feature_set_id="FS_TM_ABLINGUA_GLOBAL", base_variant="TMS3", group="phase1_fus"),
    dict(experiment_id="TRF_TM_SCRATCH_FULL_MEAN_FUS_BIOEMU_MPNN",
         source_model_id="TMS3__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="NONE", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg", content_mode="scratch",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_BASE_BIOEMU_MPNN__RIDGE",
         fusion_feature_set_id="FS_TM_BIOEMU_MPNN", base_variant="TMS3", group="phase1_fus"),
    # Phase1 fusion TmApp AbLingua
    dict(experiment_id="TRF_TM_ABLINGUA_FULL_CONCAT_FUS_ABLINGUA_CDR3",
         source_model_id="TMF2__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="ABLINGUA", annotation_mode="FULL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_PARENT_ABLINGUA_CDR3__RIDGE",
         fusion_feature_set_id="FS_TM_ABLINGUA_CDR3", base_variant="TMF2", group="phase1_fus"),
    dict(experiment_id="TRF_TM_ABLINGUA_FULL_CONCAT_FUS_ABLINGUA_GLOBAL",
         source_model_id="TMF2__FUSION__TM_PARENT_ABLINGUA_GLOBAL__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="ABLINGUA", annotation_mode="FULL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
         fusion_feature_set_id="FS_TM_ABLINGUA_GLOBAL", base_variant="TMF2", group="phase1_fus"),
    dict(experiment_id="TRF_TM_ABLINGUA_FULL_CONCAT_FUS_BIOEMU_MPNN",
         source_model_id="TMF2__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="ABLINGUA", annotation_mode="FULL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_BASE_BIOEMU_MPNN__RIDGE",
         fusion_feature_set_id="FS_TM_BIOEMU_MPNN", base_variant="TMF2", group="phase1_fus"),
    # Phase1 fusion HIC scratch
    dict(experiment_id="TRF_HIC_SCRATCH_FULL_HONLY_FUS_HYDRO_TITRATION",
         source_model_id="HICS2__FUSION__HIC_HYDRO_TITRATION__LASSO", target="HIC",
         transformer_type="FUSION", plm_source="NONE", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg", content_mode="scratch",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="HIC_HYDRO_TITRATION__LASSO",
         fusion_feature_set_id="FS_HIC_HYDRO_TITRATION", base_variant="HICS2", group="phase1_fus"),
    dict(experiment_id="TRF_HIC_SCRATCH_FULL_HONLY_FUS_CONTINUOUS_SURFACE",
         source_model_id="HICS2__FUSION__HIC_ARO_CONTINUOUS_SURFACE__LASSO", target="HIC",
         transformer_type="FUSION", plm_source="NONE", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg", content_mode="scratch",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="HIC_ARO_CONTINUOUS_SURFACE__LASSO",
         fusion_feature_set_id="FS_HIC_CONTINUOUS_SURFACE", base_variant="HICS2", group="phase1_fus"),
    dict(experiment_id="TRF_HIC_SCRATCH_FULL_HONLY_FUS_ESM2_SEQ_ARO",
         source_model_id="HICS2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO", target="HIC",
         transformer_type="FUSION", plm_source="NONE", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg", content_mode="scratch",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="HIC_ESM2_SEQ_AROMATIC__LASSO",
         fusion_feature_set_id="FS_HIC_ESM2_SEQ_AROMATIC", base_variant="HICS2", group="phase1_fus"),
    # Phase1 fusion HIC ESM2
    dict(experiment_id="TRF_HIC_ESM2_FULL_HONLY_FUS_HYDRO_TITRATION",
         source_model_id="HICF2__FUSION__HIC_HYDRO_TITRATION__LASSO", target="HIC",
         transformer_type="FUSION", plm_source="ESM2", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="HIC_HYDRO_TITRATION__LASSO",
         fusion_feature_set_id="FS_HIC_HYDRO_TITRATION", base_variant="HICF2", group="phase1_fus"),
    dict(experiment_id="TRF_HIC_ESM2_FULL_HONLY_FUS_CONTINUOUS_SURFACE",
         source_model_id="HICF2__FUSION__HIC_ARO_CONTINUOUS_SURFACE__LASSO", target="HIC",
         transformer_type="FUSION", plm_source="ESM2", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="HIC_ARO_CONTINUOUS_SURFACE__LASSO",
         fusion_feature_set_id="FS_HIC_CONTINUOUS_SURFACE", base_variant="HICF2", group="phase1_fus"),
    dict(experiment_id="TRF_HIC_ESM2_FULL_HONLY_FUS_ESM2_SEQ_ARO",
         source_model_id="HICF2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO", target="HIC",
         transformer_type="FUSION", plm_source="ESM2", annotation_mode="FULL",
         chain_mode="H_ONLY", merge_mode="h_only", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="HIC_ESM2_SEQ_AROMATIC__LASSO",
         fusion_feature_set_id="FS_HIC_ESM2_SEQ_AROMATIC", base_variant="HICF2", group="phase1_fus"),
    # AbLang2 sequence
    dict(experiment_id="TRF_TM_ABLANG2_MIN_CONCAT", source_model_id="AL2F1_MIN_CONCAT", target="TmApp",
         transformer_type="FROZEN_PLM", plm_source="ABLANG2", annotation_mode="MINIMAL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="ablang2"),
    dict(experiment_id="TRF_TM_ABLANG2_FULL_CONCAT", source_model_id="AL2F2_FULL_CONCAT", target="TmApp",
         transformer_type="FROZEN_PLM", plm_source="ABLANG2", annotation_mode="FULL",
         chain_mode="HL", merge_mode="concat", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="ablang2"),
    dict(experiment_id="TRF_TM_ABLANG2_FULL_MEAN", source_model_id="AL2F3_FULL_MEAN", target="TmApp",
         transformer_type="FROZEN_PLM", plm_source="ABLANG2", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="ablang2"),
    dict(experiment_id="TRF_TM_ABLANG2_FULL_REGION_GATE", source_model_id="AL2F4_FULL_REGION_GATE", target="TmApp",
         transformer_type="FROZEN_PLM", plm_source="ABLANG2", annotation_mode="FULL",
         chain_mode="HL", merge_mode="concat", pooling_mode="region_gate",
         content_mode="frozen", input_space="FROZEN_RESIDUE_EMBEDDING", group="ablang2"),
    # AbLang2 fusion
    dict(experiment_id="TRF_TM_ABLANG2_FULL_MEAN_FUS_ABLINGUA_CDR3",
         source_model_id="AL2F3_FULL_MEAN__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="ABLANG2", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_PARENT_ABLINGUA_CDR3__RIDGE",
         fusion_feature_set_id="FS_TM_ABLINGUA_CDR3", base_variant="AL2F3_FULL_MEAN", group="ablang2"),
    dict(experiment_id="TRF_TM_ABLANG2_FULL_MEAN_FUS_ABLINGUA_GLOBAL",
         source_model_id="AL2F3_FULL_MEAN__FUSION__TM_PARENT_ABLINGUA_GLOBAL__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="ABLANG2", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
         fusion_feature_set_id="FS_TM_ABLINGUA_GLOBAL", base_variant="AL2F3_FULL_MEAN", group="ablang2"),
    dict(experiment_id="TRF_TM_ABLANG2_FULL_MEAN_FUS_BIOEMU_MPNN",
         source_model_id="AL2F3_FULL_MEAN__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE", target="TmApp",
         transformer_type="FUSION", plm_source="ABLANG2", annotation_mode="FULL",
         chain_mode="HL", merge_mode="mean", pooling_mode="reg", content_mode="frozen",
         input_space="RESIDUE_PLUS_FIXED_FEATURES", fusion_recipe="TM_BASE_BIOEMU_MPNN__RIDGE",
         fusion_feature_set_id="FS_TM_BIOEMU_MPNN", base_variant="AL2F3_FULL_MEAN", group="ablang2"),
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_oof(source_id: str, group: str) -> Path:
    import re

    root = AO / "ablang2_followup" if group == "ablang2" else AO / "logs"
    pat = re.compile(rf"^oof_{re.escape(source_id)}_[0-9a-f]+\.npz$")
    hits = sorted(p for p in root.glob(f"oof_{source_id}_*.npz") if pat.match(p.name))
    if len(hits) != 1:
        raise FileNotFoundError(f"oof for {source_id}: {hits}")
    return hits[0]


def find_test(spec: dict) -> Path:
    sid = spec["source_model_id"]
    target = spec["target"]
    group = spec["group"]
    if group == "ablang2":
        if spec["transformer_type"] == "FUSION":
            p = AO / "ablang2_followup" / "predictions" / f"{target}__fusion__{sid}__test.csv"
        else:
            p = AO / "ablang2_followup" / "predictions" / f"{target}__seq__{sid}__test.csv"
    else:
        if spec["transformer_type"] == "FUSION":
            p = AO / "predictions" / f"{target}__fusion__{sid}__test.csv"
        elif spec["transformer_type"] == "SCRATCH":
            p = AO / "predictions" / f"{target}__scratch_transformer__{sid}__test.csv"
        else:
            p = AO / "predictions" / f"{target}__frozen_transformer__{sid}__test.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def selection_policy(spec: dict) -> tuple[str, str]:
    """Evidence-based; do not invent CV_ONLY without sources."""
    mid = spec["source_model_id"]
    if spec["group"] == "ablang2":
        evidence = (
            "ABLANG2_FOLLOWUP_PLAN.json selection.primary_key=cv_worst; "
            "public_private=POSTMORTEM_ONLY; FOLLOWUP_RUN_LOG Public/Private selection forbidden; "
            "MODEL_BENCHMARK_SUMMARY.public_private_role=POSTMORTEM_ONLY"
        )
        return "CV_SELECTED_POSTCOMP_EVALUATED", evidence
    evidence = (
        f"MODEL_BENCHMARK_SUMMARY model_id={mid} public_private_role=POSTMORTEM_ONLY "
        "cv_protocol=canonical_simple_tvt_primary_shadow; "
        "ADVANCED_MODEL_RESULTS notes contain POSTMORTEM ONLY; "
        "run_benchmark.py marks family winners by CV then scores Public/Private postmortem"
    )
    return "CV_SELECTED_POSTCOMP_EVALUATED", evidence


def license_for(spec: dict) -> tuple[str, str]:
    refs: list[str] = []
    statuses: list[str] = []
    # annotations always used
    refs.append("residue_level annotations: IMGT_residue_annotations OK_COMPETITION_DERIVED")
    statuses.append("OK")
    plm = spec["plm_source"]
    if plm == "ABLINGUA":
        refs.append("residue_level/ablingua600m metadata license_status=REVIEW_MODEL_OUTPUT")
        statuses.append("REVIEW")
    elif plm == "ABLANG2":
        refs.append("residue_level/ablang2 metadata license_status=REVIEW_MODEL_OUTPUT")
        statuses.append("REVIEW")
    elif plm == "ESM2":
        refs.append("residue_level/esm2 metadata license_status=REVIEW_MODEL_OUTPUT")
        statuses.append("REVIEW")
    elif plm == "NONE":
        refs.append("scratch AA from competition sequences (OK_COMPETITION_DERIVED)")
        statuses.append("OK")
    fs = spec.get("fusion_feature_set_id")
    if fs:
        # Phase1 Linear/XGB treated all FS_* as REVIEW (PLM/extension mix)
        refs.append(f"fusion fixed branch {fs}: Phase1 FS license aggregate REVIEW")
        statuses.append("REVIEW")
    # conservative aggregate
    order = {"RESTRICTED": 3, "REVIEW": 2, "UNKNOWN": 1, "OK": 0}
    worst = max(statuses, key=lambda s: order.get(s, 1))
    return worst, " | ".join(refs)


def input_asset_ref(spec: dict) -> str:
    plm = spec["plm_source"]
    base = "assets/transformer/residue_asset_manifest.yaml"
    if plm == "NONE":
        return f"{base}#annotations+scratch_sequences"
    if plm == "ABLINGUA":
        return f"{base}#ablingua600m"
    if plm == "ABLANG2":
        return f"{base}#ablang2"
    if plm == "ESM2":
        return f"{base}#esm2"
    return base


def load_scores(mid: str) -> dict[str, float]:
    mbs = pd.read_csv(BUNDLE / "results" / "MODEL_BENCHMARK_SUMMARY.csv")
    row = mbs[mbs["model_id"] == mid]
    if len(row) != 1:
        raise KeyError(f"MBS missing {mid}")
    r = row.iloc[0]
    return {
        "cv_primary_mae": float(r["cv_primary_mae"]),
        "cv_shadow_mae": float(r["cv_shadow_mae"]),
        "cv_mean_mae": float(r["cv_mean_mae"]),
        "cv_worst_mae": float(r["cv_worst_mae"]),
        "public_mae": float(r["public_mae"]),
        "private_mae": float(r["private_mae"]),
        "test_overall_mae": float(r["overall_test_mae"]),
        "score_source": "top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv",
    }


def convert_oof(npz_path: Path, target: str, out_primary: Path, out_shadow: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    primary = np.asarray(z["primary_oof"], float)
    shadow = np.asarray(z["shadow_oof"], float)
    assert len(ids) == len(primary) == len(shadow) == 162
    pd.DataFrame({"id": ids, target: primary}).to_csv(out_primary, index=False)
    pd.DataFrame({"id": ids, target: shadow}).to_csv(out_shadow, index=False)
    return {
        "n_ids": len(ids),
        "primary_sha_src": file_sha256(npz_path),
        "primary_max": float(np.max(np.abs(primary))),
        "shadow_max": float(np.max(np.abs(shadow))),
    }


def copy_test(src: Path, target: str, dest: Path) -> dict:
    df = pd.read_csv(src)
    df["id"] = df["id"].astype(str)
    # normalize column
    cols = [c for c in df.columns if c != "id"]
    if target not in df.columns:
        # often prediction column named target already or 'pred'
        if len(cols) == 1:
            df = df.rename(columns={cols[0]: target})
        else:
            raise ValueError(f"unexpected test columns {df.columns.tolist()} for {src}")
    out = df[["id", target]].copy()
    out.to_csv(dest, index=False)
    return {"n": len(out), "src_sha": file_sha256(src), "dst_sha": file_sha256(dest)}


def write_config(code: str, spec: dict, sel: str, lic: str, lic_ref: str) -> Path:
    ncfg = json.loads((ROOT / "models/antibody_transformer/presets.json").read_text())["neural"]
    pooling_out = "REGION_GATE" if spec["pooling_mode"] == "region_gate" else (
        "MEAN" if spec["merge_mode"] == "mean" else (
            "H_ONLY" if spec["merge_mode"] == "h_only" else "CONCAT"
        )
    )
    cfg = {
        "experiment_code": code,
        "experiment_id": spec["experiment_id"],
        "target": spec["target"],
        "family": "TRANSFORMER",
        "source_model_id": spec["source_model_id"],
        "transformer_type": spec["transformer_type"],
        "input_space": spec["input_space"],
        "input_asset_ref": input_asset_ref(spec),
        "plm_source": spec["plm_source"],
        "annotation_mode": spec["annotation_mode"],
        "chain_mode": spec["chain_mode"],
        "merge_mode": spec["merge_mode"],
        "pooling_mode": pooling_out,
        "content_mode": spec["content_mode"],
        "d_model": ncfg["d_model"],
        "n_layers": ncfg["n_layers"],
        "n_heads": ncfg["n_heads"],
        "ff_dim": ncfg["dim_feedforward"],
        "dropout": ncfg["dropout"],
        "seeds": ncfg["seeds"],
        "optimizer": "AdamW",
        "learning_rate": ncfg["lr"],
        "weight_decay": ncfg["weight_decay"],
        "batch_size": ncfg["batch_size"],
        "max_epochs": ncfg["max_epochs"],
        "patience": ncfg["early_stopping_patience"],
        "gradient_clip": ncfg["gradient_clip_norm"],
        "loss": f"SmoothL1Loss(beta={ncfg['smooth_l1_beta']})",
        "cv_protocol": "canonical_simple_tvt_primary_shadow",
        "selection_policy_at_creation": sel,
        "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
        "representation_status": "HISTORICAL_UNAVAILABLE",
        "representation_aggregation_future": "UNDECIDED",
    }
    if spec.get("fusion_feature_set_id"):
        cfg["fusion_feature_set_id"] = spec["fusion_feature_set_id"]
        cfg["fusion_projection_dim"] = ncfg["fusion_fixed_dim"]
        cfg["fusion_recipe"] = spec.get("fusion_recipe")
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def verify_pred_equality(src_npz: Path, dest_primary: Path, dest_shadow: Path, target: str) -> float:
    z = np.load(src_npz, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    p = pd.read_csv(dest_primary)
    s = pd.read_csv(dest_shadow)
    p = p.set_index("id").loc[ids]
    s = s.set_index("id").loc[ids]
    d1 = float(np.max(np.abs(p[target].to_numpy(float) - np.asarray(z["primary_oof"], float))))
    d2 = float(np.max(np.abs(s[target].to_numpy(float) - np.asarray(z["shadow_oof"], float))))
    return max(d1, d2)


def main() -> int:
    assert len(EXPERIMENTS) == 29, len(EXPERIMENTS)
    mbs = pd.read_csv(BUNDLE / "results" / "MODEL_BENCHMARK_SUMMARY.csv")
    missing = [e["source_model_id"] for e in EXPERIMENTS if e["source_model_id"] not in set(mbs["model_id"])]
    if missing:
        raise SystemExit(f"MBS missing models: {missing}")

    # Extend EXPERIMENTS_COLUMNS dynamically when writing
    extra_cols = [
        "transformer_type",
        "plm_source",
        "annotation_mode",
        "chain_mode",
        "pooling_mode",
        "representation_status",
        "input_space",
        "input_asset_ref",
    ]

    audit_rows = []
    new_exp_rows = []
    codes_existing = set(load_codes()["experiment_id"])

    for spec in EXPERIMENTS:
        mid = spec["source_model_id"]
        target = spec["target"]
        oof_p = find_oof(mid, spec["group"])
        test_p = find_test(spec)
        scores = load_scores(mid)
        sel, sel_ev = selection_policy(spec)
        lic, lic_ref = license_for(spec)

        # issue code if needed
        if spec["experiment_id"] in codes_existing:
            codes = load_codes()
            code = str(codes.set_index("experiment_id").loc[spec["experiment_id"], "experiment_code"])
        else:
            code = issue_code(
                spec["experiment_id"],
                target,
                source_model_id=mid,
                phase="PHASE2A_TRANSFORMER",
                notes=f"historical backfill {mid}",
            )
            codes_existing.add(spec["experiment_id"])

        pred_dir = ROOT / "experiments" / "predictions" / code
        pred_dir.mkdir(parents=True, exist_ok=True)
        op = pred_dir / "oof_primary.csv"
        os_ = pred_dir / "oof_shadow.csv"
        tp = pred_dir / "test.csv"
        oof_meta = convert_oof(oof_p, target, op, os_)
        test_meta = copy_test(test_p, target, tp)
        delta = verify_pred_equality(oof_p, op, os_, target)
        # test equality
        src_t = pd.read_csv(test_p)
        src_t["id"] = src_t["id"].astype(str)
        if target not in src_t.columns:
            cols = [c for c in src_t.columns if c != "id"]
            src_t = src_t.rename(columns={cols[0]: target})
        dst_t = pd.read_csv(tp)
        src_t = src_t.set_index("id").loc[dst_t["id"].astype(str)]
        tdelta = float(np.max(np.abs(src_t[target].to_numpy(float) - dst_t[target].to_numpy(float))))
        # CSV round-trip may introduce ~1e-15 float noise; require exact for practical purposes
        if delta > 1e-12 or tdelta > 1e-12:
            raise SystemExit(f"prediction delta nonzero for {code}: oof={delta} test={tdelta}")

        write_config(code, spec, sel, lic, lic_ref)
        pooling_out = "REGION_GATE" if spec["pooling_mode"] == "region_gate" else (
            "MEAN" if spec["merge_mode"] == "mean" else (
                "H_ONLY" if spec["merge_mode"] == "h_only" else "CONCAT"
            )
        )
        pub, priv = scores["public_mae"], scores["private_mae"]
        row = {c: "" for c in EXPERIMENTS_COLUMNS}
        row.update({
            "experiment_code": code,
            "legacy_experiment_code": "",
            "experiment_id": spec["experiment_id"],
            "target": target,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer" + ("+Fusion" if spec["transformer_type"] == "FUSION" else ""),
            "source_model_id": mid,
            "feature_set_id": spec.get("fusion_feature_set_id", ""),
            "source_recipe_id": spec.get("fusion_recipe", ""),
            "cv_primary_mae": scores["cv_primary_mae"],
            "cv_shadow_mae": scores["cv_shadow_mae"],
            "cv_mean_mae": scores["cv_mean_mae"],
            "cv_worst_mae": scores["cv_worst_mae"],
            "public_mae": pub,
            "private_mae": priv,
            "test_overall_mae": scores["test_overall_mae"],
            "public_private_delta": pub - priv,
            "public_private_gap": abs(pub - priv),
            "cv_protocol": "canonical_simple_tvt_primary_shadow",
            "selection_policy_at_creation": sel,
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "artifact_status": "FULL",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "PASS",
            "config_path": f"experiments/configs/{code}.yaml",
            "feature_path": "",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "n_features": "",
            "feature_space": "",
            "score_source": scores["score_source"],
            "prediction_source": str(oof_p.relative_to(REPO)) + ";" + str(test_p.relative_to(REPO)),
            "feature_source": "",
            "license_status": lic,
            "license_reference": lic_ref,
            "notes": (
                f"Phase2A historical Transformer backfill; representation_status=HISTORICAL_UNAVAILABLE; "
                f"representation_aggregation_future=UNDECIDED; group={spec['group']}"
            ),
        })
        for c, v in {
            "transformer_type": spec["transformer_type"],
            "plm_source": spec["plm_source"],
            "annotation_mode": spec["annotation_mode"],
            "chain_mode": spec["chain_mode"],
            "pooling_mode": pooling_out,
            "representation_status": "HISTORICAL_UNAVAILABLE",
            "input_space": spec["input_space"],
            "input_asset_ref": input_asset_ref(spec),
        }.items():
            row[c] = v
        new_exp_rows.append(row)

        audit_rows.append({
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "source_model_id": mid,
            "target": target,
            "group": spec["group"],
            "transformer_type": spec["transformer_type"],
            "plm_source": spec["plm_source"],
            "score_source": scores["score_source"],
            "source_oof_path": str(oof_p.relative_to(REPO)),
            "source_oof_sha256": oof_meta["primary_sha_src"],
            "source_test_path": str(test_p.relative_to(REPO)),
            "source_test_sha256": test_meta["src_sha"],
            "dest_oof_primary_sha256": file_sha256(op),
            "dest_oof_shadow_sha256": file_sha256(os_),
            "dest_test_sha256": test_meta["dst_sha"],
            "primary_oof_available": "YES",
            "shadow_oof_available": "YES",
            "test_available": "YES",
            "oof_value_max_abs_delta": delta,
            "test_value_max_abs_delta": tdelta,
            "architecture_config_source": "advanced_models/presets.json + presets variants / ablang2 followup plan",
            "selection_policy_at_creation": sel,
            "selection_policy_evidence": sel_ev,
            "license_status": lic,
            "license_evidence": lic_ref,
            "representation_status": "HISTORICAL_UNAVAILABLE",
            "feature_path": "",
            "feature_space": "",
            "input_space": spec["input_space"],
            "input_asset_ref": input_asset_ref(spec),
            "fusion_feature_set_id": spec.get("fusion_feature_set_id", ""),
            "cv_primary_mae": scores["cv_primary_mae"],
            "cv_shadow_mae": scores["cv_shadow_mae"],
            "public_mae": pub,
            "private_mae": priv,
            "test_overall_mae": scores["test_overall_mae"],
        })
        print(f"OK {code} {spec['experiment_id']} delta_oof={delta} delta_test={tdelta}")

    # Merge experiments.csv: keep existing 48 Linear/XGB, append Transformer with new cols
    exp_path = ROOT / "results" / "experiments.csv"
    old = pd.read_csv(exp_path)
    for c in extra_cols:
        if c not in old.columns:
            old[c] = ""
    # drop any prior TRANSFORMER rows (idempotent re-run)
    old = old[old["family"] != "TRANSFORMER"].copy()
    new_df = pd.DataFrame(new_exp_rows)
    for c in old.columns:
        if c not in new_df.columns:
            new_df[c] = ""
    new_df = new_df[old.columns]
    out = pd.concat([old, new_df], ignore_index=True)
    # column order: original + extras at end if not already
    cols = list(dict.fromkeys(list(EXPERIMENTS_COLUMNS) + extra_cols))
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols]
    out.to_csv(exp_path, index=False)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(ROOT / "results" / "TRANSFORMER_BACKFILL_AUDIT.csv", index=False)
    print(f"Wrote {len(new_exp_rows)} transformer rows; total experiments={len(out)}")
    print(out["family"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
