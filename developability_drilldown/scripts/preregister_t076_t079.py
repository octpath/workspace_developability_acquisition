#!/usr/bin/env python3
"""Preregister EXP-T076..T079 architecture definitions BEFORE any training.

Writes configs + T076_T079_ARCHITECTURE_PREREGISTRATION.yaml.
Does NOT train.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))

from antibody_transformer.protocol_v3 import (  # noqa: E402
    BATCH_SIZE,
    ETA_MIN_FRAC,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    PLATFORM_ID,
    SMOOTH_L1_BETA,
    T_MAX,
    WEIGHT_DECAY,
    coarse_lr_grid,
)
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402

SERIES = [
    {
        "code": "EXP-T076",
        "experiment_id": "TRF_TM_ABLINGUA_FULL_JOINT_SINGLE_REG_V3",
        "input_space": "JOINT_HL_SINGLE_REG_FROZEN_RESIDUE_V3",
        "arch": {
            "joint_hl_single_reg": True,
            "joint_hl_dual_reg": False,
            "joint_hl_chain_specific_dual_reg": False,
            "use_cross_attention_bridge": False,
        },
        "description": "Joint H/L + single REG (T068-like)",
        "attention": "full joint self-attn; H<->L allowed; one REG",
        "readout": "hidden(REG); dim=d_model=128",
        "like": "EXP-T068",
    },
    {
        "code": "EXP-T077",
        "experiment_id": "TRF_TM_ABLINGUA_FULL_JOINT_DUAL_REG_V3",
        "input_space": "JOINT_HL_DUAL_REG_FROZEN_RESIDUE_V3",
        "arch": {
            "joint_hl_single_reg": False,
            "joint_hl_dual_reg": True,
            "joint_hl_chain_specific_dual_reg": False,
            "use_cross_attention_bridge": False,
        },
        "description": "Joint H/L + unrestricted dual REG (T070-like)",
        "attention": "full joint; REG_H/REG_L unrestricted across chains",
        "readout": "concat(REG_H, REG_L); dim=256",
        "like": "EXP-T070",
    },
    {
        "code": "EXP-T078",
        "experiment_id": "TRF_TM_ABLINGUA_FULL_JOINT_CHAIN_SPECIFIC_DUAL_REG_V3",
        "input_space": "JOINT_HL_CHAIN_SPECIFIC_DUAL_REG_FROZEN_RESIDUE_V3",
        "arch": {
            "joint_hl_single_reg": False,
            "joint_hl_dual_reg": False,
            "joint_hl_chain_specific_dual_reg": True,
            "use_cross_attention_bridge": False,
        },
        "description": "Joint H/L + chain-specific dual REG masks (T071-like)",
        "attention": "residue H<->L allowed; REG_H attends H only; REG_L attends L only; cross-REG blocked",
        "readout": "concat(REG_H, REG_L); dim=256",
        "like": "EXP-T071",
    },
    {
        "code": "EXP-T079",
        "experiment_id": "TRF_TM_ABLINGUA_FULL_SEPARATE_CROSSATTN_DUAL_REG_V3",
        "input_space": "SEPARATE_CROSS_ATTENTION_DUAL_REG_FROZEN_RESIDUE_V3",
        "arch": {
            "joint_hl_single_reg": False,
            "joint_hl_dual_reg": False,
            "joint_hl_chain_specific_dual_reg": False,
            "use_cross_attention_bridge": True,
        },
        "description": "Separate H/L + bidirectional residue cross-attn bridge + dual REG (T072-like)",
        "attention": "separate self-attn; residue-only cross-attn with zero-init gates between layer1/2",
        "readout": "concat(REG_H, REG_L); dim=256",
        "like": "EXP-T072",
    },
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def param_count_for(arch: dict) -> dict:
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        **arch,
    )
    n = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    acct = m.param_account() if hasattr(m, "param_account") else {}
    return {"n_trainable": n, "param_account": acct, "repr_dim": int(m.repr_dim)}


def write_config(spec: dict, params: dict) -> Path:
    code = spec["code"]
    cfg = {
        "experiment_code": code,
        "experiment_id": spec["experiment_id"],
        "platform_id": PLATFORM_ID,
        "target": "TmApp",
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM_ID}::EXP-T075",
        "transformer_type": "FROZEN_PLM",
        "input_space": spec["input_space"],
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
        "plm_source": "ABLINGUA",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": "concat",
        "pooling_mode": "REG",
        "content_mode": "frozen",
        "architecture_control": "EXP-T075",
        "scientific_like": spec["like"],
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
        "lr_grid": coarse_lr_grid(),
        "eta_min_frac": ETA_MIN_FRAC,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "scheduler": "explicit_cosine_T_max_200",
        "scheduler_T_max": T_MAX,
        "warmup": None,
        "restart": None,
        "loss": f"SmoothL1Loss(beta={SMOOTH_L1_BETA})",
        "checkpoint_metric": "validation_MAE",
        "no_full_dev_refit": True,
        "no_nested_cv": True,
        "external_aggregation": ["mean", "median"],
        "representation_status": "NOT_EXPORTED",
        "control_experiment_code": "EXP-T075",
        "n_trainable_preregistered": params["n_trainable"],
        "repr_dim": params["repr_dim"],
        "param_account_preregistered": params["param_account"],
        **{f"arch_{k}": v for k, v in spec["arch"].items()},
    }
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    rows = []
    for spec in SERIES:
        params = param_count_for(spec["arch"])
        cfg_path = write_config(spec, params)
        sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        rows.append(
            {
                "experiment_code": spec["code"],
                "experiment_id": spec["experiment_id"],
                "description": spec["description"],
                "scientific_like": spec["like"],
                "architecture": spec["arch"],
                "attention_semantics": spec["attention"],
                "readout": spec["readout"],
                "n_trainable_preregistered": params["n_trainable"],
                "repr_dim": params["repr_dim"],
                "param_account": params["param_account"],
                "platform_id": PLATFORM_ID,
                "config_path": str(cfg_path.relative_to(ROOT)),
                "config_sha256": sha,
            }
        )
        print(spec["code"], "params", params["n_trainable"], "repr", params["repr_dim"], flush=True)

    # T075 baseline params for table
    t075 = param_count_for(
        {
            "joint_hl_single_reg": False,
            "joint_hl_dual_reg": False,
            "joint_hl_chain_specific_dual_reg": False,
            "use_cross_attention_bridge": False,
        }
    )

    doc = {
        "status": "PREREGISTERED_BEFORE_TRAINING",
        "git_rev_at_preregistration": git_rev(),
        "platform_id": PLATFORM_ID,
        "baseline_experiment": "EXP-T075",
        "baseline_n_trainable": t075["n_trainable"],
        "seed": 101,
        "planned_comparisons": [
            {"id": "A", "name": "single_summary_penalty", "contrast": "T076 vs T077"},
            {"id": "B", "name": "full_joint_effect", "contrast": "T077 vs T075"},
            {"id": "C", "name": "reg_specialization", "contrast": "T078 vs T077"},
            {"id": "D", "name": "cross_attention_bridge", "contrast": "T079 vs T075"},
            {"id": "E", "name": "joint_vs_cross_bridge", "contrast": "T079 vs T077/T078"},
        ],
        "primary_comparison_metric": "TEST_mean",
        "robustness_metric": "TEST_worst",
        "external_canonical_aggregation": "Primary mean",
        "experiments": rows,
        "notes": (
            "All four architectures fixed before any result inspection. "
            "No architecture may change after training begins."
        ),
    }
    out = ROOT / "results" / "T076_T079_ARCHITECTURE_PREREGISTRATION.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
