#!/usr/bin/env python3
"""Preregister EXP-H094..H101 STATIC_SAP_KD antibody-level late-fusion batch."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))

from antibody_transformer.static_sap_kd import (  # noqa: E402
    INCLUDE_SELF,
    KD_MAX,
    KD_MIN,
    KYTE_DOOLITTLE,
    N_POINTS_PRIMARY,
    PROBE,
    R_REF,
    R_SENSITIVITY,
    SAP3_COLS,
    SAP9_COLS,
    TIEN_MAXASA,
)

PLATFORM = "DL_FOLDLOCAL_COSINE_V3"
PREREG = ROOT / "results" / "H094_H101_STATIC_SAP_KD_PREREGISTRATION.yaml"

SERIES = [
    {
        "code": "EXP-H094",
        "backbone": "EXP-H071",
        "control_no_aux": "EXP-H071",
        "fusion_bundle_id": "FS_HIC_STATIC_SAP_KD_GLOBAL3",
        "aux_dim": 3,
        "description": "H071 Scratch + STATIC_SAP_KD SAP3 late fusion",
        "experiment_id": "TRF_HIC_SCRATCH_ARCH2_STATIC_SAP_KD_GLOBAL3_V3",
        "input_space": "LATEFUSION_STATIC_SAP_KD_GLOBAL3_SCRATCH_V3",
        "plm_source": "NONE",
        "content_mode": "scratch",
        "representation": "SCRATCH",
        "arch_id": "ARCH-2",
        "merge_mode": "concat",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": True,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": False,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
    {
        "code": "EXP-H095",
        "backbone": "EXP-H071",
        "control_no_aux": "EXP-H071",
        "fusion_bundle_id": "FS_HIC_STATIC_SAP_KD_CHAIN9",
        "aux_dim": 9,
        "description": "H071 Scratch + STATIC_SAP_KD SAP9 late fusion",
        "experiment_id": "TRF_HIC_SCRATCH_ARCH2_STATIC_SAP_KD_CHAIN9_V3",
        "input_space": "LATEFUSION_STATIC_SAP_KD_CHAIN9_SCRATCH_V3",
        "plm_source": "NONE",
        "content_mode": "scratch",
        "representation": "SCRATCH",
        "arch_id": "ARCH-2",
        "merge_mode": "concat",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": True,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": False,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
    {
        "code": "EXP-H096",
        "backbone": "EXP-H071",
        "control_surface": "EXP-H090",
        "fusion_bundle_id": "FS_HIC_SURFACE_PLUS_STATIC_SAP_KD_CHAIN9",
        "aux_dim": 44,
        "description": "H071 Scratch + SURFACE + STATIC_SAP_KD SAP9 late fusion",
        "experiment_id": "TRF_HIC_SCRATCH_ARCH2_SURFACE_STATIC_SAP_KD_CHAIN9_V3",
        "input_space": "LATEFUSION_SURFACE_STATIC_SAP_KD_CHAIN9_SCRATCH_V3",
        "plm_source": "NONE",
        "content_mode": "scratch",
        "representation": "SCRATCH",
        "arch_id": "ARCH-2",
        "merge_mode": "concat",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": True,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": False,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
    {
        "code": "EXP-H097",
        "backbone": "EXP-H071",
        "control_f4": "EXP-H093",
        "fusion_bundle_id": "FS_HIC_F4_PLUS_STATIC_SAP_KD_CHAIN9",
        "aux_dim": 209,
        "description": "H071 Scratch + F4 + STATIC_SAP_KD SAP9 late fusion",
        "experiment_id": "TRF_HIC_SCRATCH_ARCH2_F4_STATIC_SAP_KD_CHAIN9_V3",
        "input_space": "LATEFUSION_F4_STATIC_SAP_KD_CHAIN9_SCRATCH_V3",
        "plm_source": "NONE",
        "content_mode": "scratch",
        "representation": "SCRATCH",
        "arch_id": "ARCH-2",
        "merge_mode": "concat",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": True,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": False,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
    {
        "code": "EXP-H098",
        "backbone": "EXP-H061",
        "control_no_aux": "EXP-H061",
        "fusion_bundle_id": "FS_HIC_STATIC_SAP_KD_GLOBAL3",
        "aux_dim": 3,
        "description": "H061 ESM2 ARCH-4 + STATIC_SAP_KD SAP3 late fusion",
        "experiment_id": "TRF_HIC_ESM2_ARCH4_STATIC_SAP_KD_GLOBAL3_V3",
        "input_space": "LATEFUSION_STATIC_SAP_KD_GLOBAL3_ESM2_V3",
        "plm_source": "ESM2",
        "content_mode": "frozen",
        "representation": "ESM2",
        "arch_id": "ARCH-4",
        "merge_mode": "mean",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": False,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": True,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
    {
        "code": "EXP-H099",
        "backbone": "EXP-H061",
        "control_no_aux": "EXP-H061",
        "fusion_bundle_id": "FS_HIC_STATIC_SAP_KD_CHAIN9",
        "aux_dim": 9,
        "description": "H061 ESM2 ARCH-4 + STATIC_SAP_KD SAP9 late fusion",
        "experiment_id": "TRF_HIC_ESM2_ARCH4_STATIC_SAP_KD_CHAIN9_V3",
        "input_space": "LATEFUSION_STATIC_SAP_KD_CHAIN9_ESM2_V3",
        "plm_source": "ESM2",
        "content_mode": "frozen",
        "representation": "ESM2",
        "arch_id": "ARCH-4",
        "merge_mode": "mean",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": False,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": True,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
    {
        "code": "EXP-H100",
        "backbone": "EXP-H061",
        "control_surface": "EXP-H086",
        "fusion_bundle_id": "FS_HIC_SURFACE_PLUS_STATIC_SAP_KD_CHAIN9",
        "aux_dim": 44,
        "description": "H061 ESM2 ARCH-4 + SURFACE + STATIC_SAP_KD SAP9 late fusion",
        "experiment_id": "TRF_HIC_ESM2_ARCH4_SURFACE_STATIC_SAP_KD_CHAIN9_V3",
        "input_space": "LATEFUSION_SURFACE_STATIC_SAP_KD_CHAIN9_ESM2_V3",
        "plm_source": "ESM2",
        "content_mode": "frozen",
        "representation": "ESM2",
        "arch_id": "ARCH-4",
        "merge_mode": "mean",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": False,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": True,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
    {
        "code": "EXP-H101",
        "backbone": "EXP-H061",
        "control_f4": "EXP-H089",
        "fusion_bundle_id": "FS_HIC_F4_PLUS_STATIC_SAP_KD_CHAIN9",
        "aux_dim": 209,
        "description": "H061 ESM2 ARCH-4 + F4 + STATIC_SAP_KD SAP9 late fusion",
        "experiment_id": "TRF_HIC_ESM2_ARCH4_F4_STATIC_SAP_KD_CHAIN9_V3",
        "input_space": "LATEFUSION_F4_STATIC_SAP_KD_CHAIN9_ESM2_V3",
        "plm_source": "ESM2",
        "content_mode": "frozen",
        "representation": "ESM2",
        "arch_id": "ARCH-4",
        "merge_mode": "mean",
        "transformer_type": "LATE_FUSION",
        "arch": {
            "arch_joint_hl_single_reg": False,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": True,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": False,
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
        },
    },
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_config(spec: dict) -> Path:
    cfg = {
        "experiment_code": spec["code"],
        "experiment_id": spec["experiment_id"],
        "platform_id": PLATFORM,
        "target": "HIC",
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM}::{spec['backbone']}",
        "transformer_type": spec["transformer_type"],
        "input_space": spec["input_space"],
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml",
        "plm_source": spec["plm_source"],
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": spec["merge_mode"],
        "pooling_mode": "REG",
        "content_mode": spec["content_mode"],
        "arch_id": spec["arch_id"],
        "representation": spec["representation"],
        "geometry": False,
        "control_experiment_code": spec["backbone"],
        "fusion_bundle_id": spec["fusion_bundle_id"],
        "fusion_mode": "late_concat_aux32",
        "exclude_global_esm2_h": True,
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "ff_dim": 256,
        "dropout": 0.2,
        "norm_first": True,
        "activation": "gelu",
        "seed": 101,
        "seeds": [101],
        "optimizer": "AdamW",
        "lr_grid": [1.0e-05, 0.0001, 0.001, 0.01],
        "eta_min_frac": 0.01,
        "weight_decay": 0.01,
        "batch_size": 16,
        "max_epochs": 200,
        "min_epochs": 0,
        "patience": 30,
        "scheduler": "explicit_cosine_T_max_200",
        "scheduler_T_max": 200,
        "warmup": None,
        "restart": None,
        "loss": "SmoothL1Loss(beta=0.5)",
        "checkpoint_metric": "validation_MAE",
        "no_full_dev_refit": True,
        "no_nested_cv": True,
        "external_aggregation": ["mean", "median"],
        "representation_status": "NOT_EXPORTED",
        **spec["arch"],
        "description": spec["description"],
        "descriptor_id": "STATIC_SAP_KD",
        "structure_scope": "Fv",
    }
    path = ROOT / "experiments" / "configs" / f"{spec['code']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    meta_path = ROOT / "results" / "STATIC_SAP_KD_EXTRACT_META.json"
    if not meta_path.exists():
        print("ERROR: run extract_static_sap_kd.py first", flush=True)
        return 1
    meta = json.loads(meta_path.read_text())
    if meta.get("sasa_unstable"):
        print("ERROR: SASA unstable — refuse preregistration", flush=True)
        return 2

    for spec in SERIES:
        write_config(spec)

    g3 = ROOT / "experiments/features/static_sap_kd_antibody_global3.parquet"
    c9 = ROOT / "experiments/features/static_sap_kd_antibody_chain9.parquet"
    res = ROOT / "experiments/features/static_sap_kd_residue.parquet"

    doc = {
        "batch_id": "H094_H101_STATIC_SAP_KD",
        "status": "PREREGISTERED_FROZEN",
        "platform_id": PLATFORM,
        "git_rev_at_prereg": git_rev(),
        "descriptor": {
            "name": "STATIC_SAP_KD",
            "full_name": "Static SAP-like Kyte-Doolittle descriptor",
            "prefix": "SSKD_",
            "structure_scope": "Fv",
            "structure_source": "STRUCTURE_INPUT_CROSSWALK_v2.csv esmfold_canonical_path (ESMFold Fv)",
            "do_not_call": "SAP / Chennamsetty MD SAP / Black-Mould STATIC-SAP",
            "formula": "SSKD_i(R)=sum_j I[d(centroid_i,centroid_j)<=R]*KD_norm(j)*clip(SASA_j/Tien_MaxASA_j,0,1)",
            "R_REF_A": R_REF,
            "R_REF_label": meta.get("R_REF_label", "SOURCE_SPECIFIED_FROM_STATIC_SAP_FEATURE_SPEC"),
            "R_SENSITIVITY_A": R_SENSITIVITY,
            "R_SENSITIVITY_note": "QC artifact only; no H094-H101 training",
            "sasa": {
                "engine": "Bio.PDB.SASA.ShrakeRupley",
                "probe_A": PROBE,
                "n_points": N_POINTS_PRIMARY,
                "n_points_label": meta.get("n_points_label", "SOURCE_SPECIFIED"),
                "numerator": "TOTAL_RESIDUE_SASA",
                "maxasa": "Tien2013",
                "tien_maxasa_table": TIEN_MAXASA,
            },
            "hydrophobicity": {
                "scale": "Kyte-Doolittle 1982",
                "normalization": "min-max over 20 AA",
                "KD_min": KD_MIN,
                "KD_max": KD_MAX,
                "table": KYTE_DOOLITTLE,
            },
            "centroid": {
                "definition": "arithmetic mean of non-hydrogen side-chain atoms",
                "excluded_backbone": ["N", "CA", "C", "O", "OXT"],
                "gly_fallback": "CA",
            },
            "include_self": INCLUDE_SELF,
            "no_distance_decay": True,
            "no_plddt_weighting": True,
            "SAP3_cols": SAP3_COLS,
            "SAP9_cols": SAP9_COLS,
            "SAP9_note": "H/L aggregation selects CENTER residues only; neighborhood remains full H+L",
        },
        "artifact_hashes": {
            "residue": _sha_file(res),
            "global3": _sha_file(g3),
            "chain9": _sha_file(c9),
            "extract_meta": meta.get("artifact_hashes"),
        },
        "auxiliary_bundles": {
            "SURFACE": "AROMATIC_TOPO19 + HYDRO_FIELD16 = 35 (unchanged)",
            "F4_H047_AUX_ALL": "SURFACE + SEQ_ALL115 + TITRATION_SHAPE18 + PCA32(FB_ESM2_RASA_CDR3) = 200 effective",
        },
        "late_fusion": "Linear(p,64)->GELU->Dropout(0.2)->Linear(64,32)->GELU; concat with z_DL",
        "preprocessing": "TRAIN-only median impute + StandardScaler; F4 PCA unchanged; SAP block separate",
        "series": SERIES,
        "controls_reused_not_retrained": [
            "EXP-H071",
            "EXP-H090",
            "EXP-H093",
            "EXP-H061",
            "EXP-H086",
            "EXP-H089",
        ],
        "forbidden": [
            "residue-level SAP token injection",
            "SAP-guided attention",
            "charge-SAP",
            "R=10 training",
            "T142",
            "target-driven R / n_points tuning",
        ],
        "statement": "Configs + STATIC_SAP_KD definition frozen before training. No target-driven changes afterward.",
    }
    PREREG.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"Wrote {PREREG}", flush=True)
    print(f"Configs: {[s['code'] for s in SERIES]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
