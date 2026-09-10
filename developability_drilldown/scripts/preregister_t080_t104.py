#!/usr/bin/env python3
"""Preregister EXP-T080..T104 architecture × representation sweep BEFORE training.

Writes per-code YAML configs + T080_T104_ARCH_REP_PREREGISTRATION.yaml
+ T075_T079_REUSED.yaml. Does NOT train.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

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
    normalize_arch_flags,
)
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402


def _arch(**flags: Any) -> dict:
    base = {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
    }
    base.update(flags)
    return normalize_arch_flags(base)


ARCH_SPECS: dict[str, dict] = {
    "ARCH-1": {
        "arch": _arch(),
        "attention": "separate H/L self-attn; no H↔L communication; dual REG",
        "hl_base": "separate",
        "comm": "none",
    },
    "ARCH-2": {
        "arch": _arch(joint_hl_single_reg=True),
        "attention": "full joint self-attn; H↔L allowed; one REG",
        "hl_base": "joint",
        "comm": "full joint",
    },
    "ARCH-3": {
        "arch": _arch(joint_hl_dual_reg=True),
        "attention": "full joint; REG_H/REG_L unrestricted across chains",
        "hl_base": "joint",
        "comm": "full joint unrestricted dual REG",
    },
    "ARCH-4": {
        "arch": _arch(joint_hl_chain_specific_dual_reg=True),
        "attention": "residue H↔L allowed; REG_H attends H only; REG_L attends L only; cross-REG blocked",
        "hl_base": "joint",
        "comm": "joint residue; chain-specific REG",
    },
    "ARCH-5": {
        "arch": _arch(use_cross_attention_bridge=True, cross_gate_mode="learned"),
        "attention": "separate self-attn; residue-only cross-attn with learned zero-init gates",
        "hl_base": "separate",
        "comm": "gated cross-attn bridge",
    },
    "ARCH-6": {
        "arch": _arch(use_cross_attention_bridge=True, cross_gate_mode="fixed_one"),
        "attention": "separate self-attn; residue-only cross-attn ungated (fixed g=1 residual)",
        "hl_base": "separate",
        "comm": "ungated cross-attn bridge",
    },
    "ARCH-7": {
        "arch": _arch(use_reg_only_cross_attention=True),
        "attention": "separate encode then REG-only queries attend opposite-chain residues (ungated)",
        "hl_base": "separate",
        "comm": "REG-only cross-attn",
    },
    "ARCH-8": {
        "arch": _arch(use_within_chain_extra_attention=True),
        "attention": "separate self-attn + mid-stack within-chain extra residue attn (ungated H←H / L←L)",
        "hl_base": "separate",
        "comm": "within-chain extra attn",
    },
}


def _readout(arch_id: str, merge_mode: Optional[str]) -> str:
    if arch_id == "ARCH-2":
        return "hidden(REG); dim=d_model=128"
    if merge_mode == "mean":
        return "mean(REG_H, REG_L); dim=128"
    return "concat(REG_H, REG_L); dim=256"


def _mk(
    code: str,
    *,
    arch_id: str,
    representation: str,
    merge_mode: Optional[str],
    experiment_id: str,
    input_space: str,
    description: str,
) -> dict:
    meta = ARCH_SPECS[arch_id]
    content_mode = "frozen" if representation == "ABLINGUA" else "scratch"
    return {
        "code": code,
        "experiment_id": experiment_id,
        "input_space": input_space,
        "representation": representation,
        "arch_id": arch_id,
        "content_mode": content_mode,
        "merge_mode": merge_mode,
        "arch": dict(meta["arch"]),
        "description": description,
        "attention": meta["attention"],
        "readout": _readout(arch_id, merge_mode),
        "hl_base": meta["hl_base"],
        "comm": meta["comm"],
    }


SERIES: list[dict] = [
    # ---- AbLingua T080–T089 ----
    _mk(
        "EXP-T080",
        arch_id="ARCH-1",
        representation="ABLINGUA",
        merge_mode="mean",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_DUAL_MEAN_V3",
        input_space="SEPARATE_DUAL_REG_FROZEN_RESIDUE_MEAN_V3",
        description="AbLingua ARCH-1 separate dual REG MEAN",
    ),
    _mk(
        "EXP-T081",
        arch_id="ARCH-3",
        representation="ABLINGUA",
        merge_mode="mean",
        experiment_id="TRF_TM_ABLINGUA_FULL_JOINT_DUAL_MEAN_V3",
        input_space="JOINT_HL_DUAL_REG_FROZEN_RESIDUE_MEAN_V3",
        description="AbLingua ARCH-3 joint unrestricted dual REG MEAN",
    ),
    _mk(
        "EXP-T082",
        arch_id="ARCH-4",
        representation="ABLINGUA",
        merge_mode="mean",
        experiment_id="TRF_TM_ABLINGUA_FULL_JOINT_CHAIN_SPECIFIC_DUAL_MEAN_V3",
        input_space="JOINT_HL_CHAIN_SPECIFIC_DUAL_REG_FROZEN_RESIDUE_MEAN_V3",
        description="AbLingua ARCH-4 joint chain-specific dual REG MEAN",
    ),
    _mk(
        "EXP-T083",
        arch_id="ARCH-5",
        representation="ABLINGUA",
        merge_mode="mean",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_CROSSATTN_DUAL_MEAN_V3",
        input_space="SEPARATE_CROSS_ATTENTION_DUAL_REG_FROZEN_RESIDUE_MEAN_V3",
        description="AbLingua ARCH-5 gated cross-attn MEAN",
    ),
    _mk(
        "EXP-T084",
        arch_id="ARCH-6",
        representation="ABLINGUA",
        merge_mode="concat",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_CROSSATTN_UNGATED_CONCAT_V3",
        input_space="SEPARATE_CROSS_ATTENTION_UNGATED_FROZEN_RESIDUE_V3",
        description="AbLingua ARCH-6 ungated cross-attn CONCAT",
    ),
    _mk(
        "EXP-T085",
        arch_id="ARCH-6",
        representation="ABLINGUA",
        merge_mode="mean",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_CROSSATTN_UNGATED_MEAN_V3",
        input_space="SEPARATE_CROSS_ATTENTION_UNGATED_FROZEN_RESIDUE_MEAN_V3",
        description="AbLingua ARCH-6 ungated cross-attn MEAN",
    ),
    _mk(
        "EXP-T086",
        arch_id="ARCH-7",
        representation="ABLINGUA",
        merge_mode="concat",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_REG_ONLY_CROSS_CONCAT_V3",
        input_space="SEPARATE_REG_ONLY_CROSS_ATTENTION_FROZEN_RESIDUE_V3",
        description="AbLingua ARCH-7 REG-only cross CONCAT",
    ),
    _mk(
        "EXP-T087",
        arch_id="ARCH-7",
        representation="ABLINGUA",
        merge_mode="mean",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_REG_ONLY_CROSS_MEAN_V3",
        input_space="SEPARATE_REG_ONLY_CROSS_ATTENTION_FROZEN_RESIDUE_MEAN_V3",
        description="AbLingua ARCH-7 REG-only cross MEAN",
    ),
    _mk(
        "EXP-T088",
        arch_id="ARCH-8",
        representation="ABLINGUA",
        merge_mode="concat",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_WITHIN_CHAIN_EXTRA_CONCAT_V3",
        input_space="SEPARATE_WITHIN_CHAIN_EXTRA_ATTENTION_FROZEN_RESIDUE_V3",
        description="AbLingua ARCH-8 within-chain extra attn CONCAT",
    ),
    _mk(
        "EXP-T089",
        arch_id="ARCH-8",
        representation="ABLINGUA",
        merge_mode="mean",
        experiment_id="TRF_TM_ABLINGUA_FULL_SEPARATE_WITHIN_CHAIN_EXTRA_MEAN_V3",
        input_space="SEPARATE_WITHIN_CHAIN_EXTRA_ATTENTION_FROZEN_RESIDUE_MEAN_V3",
        description="AbLingua ARCH-8 within-chain extra attn MEAN",
    ),
    # ---- Scratch T090–T104 ----
    _mk(
        "EXP-T090",
        arch_id="ARCH-1",
        representation="SCRATCH",
        merge_mode="concat",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_DUAL_CONCAT_V3",
        input_space="SEPARATE_DUAL_REG_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-1 separate dual REG CONCAT",
    ),
    _mk(
        "EXP-T091",
        arch_id="ARCH-1",
        representation="SCRATCH",
        merge_mode="mean",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_DUAL_MEAN_V3",
        input_space="SEPARATE_DUAL_REG_SCRATCH_RESIDUE_MEAN_V3",
        description="Scratch ARCH-1 separate dual REG MEAN",
    ),
    _mk(
        "EXP-T092",
        arch_id="ARCH-2",
        representation="SCRATCH",
        merge_mode=None,
        experiment_id="TRF_TM_SCRATCH_FULL_JOINT_SINGLE_REG_V3",
        input_space="JOINT_HL_SINGLE_REG_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-2 joint single REG",
    ),
    _mk(
        "EXP-T093",
        arch_id="ARCH-3",
        representation="SCRATCH",
        merge_mode="concat",
        experiment_id="TRF_TM_SCRATCH_FULL_JOINT_DUAL_CONCAT_V3",
        input_space="JOINT_HL_DUAL_REG_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-3 joint unrestricted dual REG CONCAT",
    ),
    _mk(
        "EXP-T094",
        arch_id="ARCH-3",
        representation="SCRATCH",
        merge_mode="mean",
        experiment_id="TRF_TM_SCRATCH_FULL_JOINT_DUAL_MEAN_V3",
        input_space="JOINT_HL_DUAL_REG_SCRATCH_RESIDUE_MEAN_V3",
        description="Scratch ARCH-3 joint unrestricted dual REG MEAN",
    ),
    _mk(
        "EXP-T095",
        arch_id="ARCH-4",
        representation="SCRATCH",
        merge_mode="concat",
        experiment_id="TRF_TM_SCRATCH_FULL_JOINT_CHAIN_SPECIFIC_DUAL_CONCAT_V3",
        input_space="JOINT_HL_CHAIN_SPECIFIC_DUAL_REG_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-4 joint chain-specific dual REG CONCAT",
    ),
    _mk(
        "EXP-T096",
        arch_id="ARCH-4",
        representation="SCRATCH",
        merge_mode="mean",
        experiment_id="TRF_TM_SCRATCH_FULL_JOINT_CHAIN_SPECIFIC_DUAL_MEAN_V3",
        input_space="JOINT_HL_CHAIN_SPECIFIC_DUAL_REG_SCRATCH_RESIDUE_MEAN_V3",
        description="Scratch ARCH-4 joint chain-specific dual REG MEAN",
    ),
    _mk(
        "EXP-T097",
        arch_id="ARCH-5",
        representation="SCRATCH",
        merge_mode="concat",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_CROSSATTN_CONCAT_V3",
        input_space="SEPARATE_CROSS_ATTENTION_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-5 gated cross-attn CONCAT",
    ),
    _mk(
        "EXP-T098",
        arch_id="ARCH-5",
        representation="SCRATCH",
        merge_mode="mean",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_CROSSATTN_MEAN_V3",
        input_space="SEPARATE_CROSS_ATTENTION_SCRATCH_RESIDUE_MEAN_V3",
        description="Scratch ARCH-5 gated cross-attn MEAN",
    ),
    _mk(
        "EXP-T099",
        arch_id="ARCH-6",
        representation="SCRATCH",
        merge_mode="concat",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_CROSSATTN_UNGATED_CONCAT_V3",
        input_space="SEPARATE_CROSS_ATTENTION_UNGATED_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-6 ungated cross-attn CONCAT",
    ),
    _mk(
        "EXP-T100",
        arch_id="ARCH-6",
        representation="SCRATCH",
        merge_mode="mean",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_CROSSATTN_UNGATED_MEAN_V3",
        input_space="SEPARATE_CROSS_ATTENTION_UNGATED_SCRATCH_RESIDUE_MEAN_V3",
        description="Scratch ARCH-6 ungated cross-attn MEAN",
    ),
    _mk(
        "EXP-T101",
        arch_id="ARCH-7",
        representation="SCRATCH",
        merge_mode="concat",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_REG_ONLY_CROSS_CONCAT_V3",
        input_space="SEPARATE_REG_ONLY_CROSS_ATTENTION_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-7 REG-only cross CONCAT",
    ),
    _mk(
        "EXP-T102",
        arch_id="ARCH-7",
        representation="SCRATCH",
        merge_mode="mean",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_REG_ONLY_CROSS_MEAN_V3",
        input_space="SEPARATE_REG_ONLY_CROSS_ATTENTION_SCRATCH_RESIDUE_MEAN_V3",
        description="Scratch ARCH-7 REG-only cross MEAN",
    ),
    _mk(
        "EXP-T103",
        arch_id="ARCH-8",
        representation="SCRATCH",
        merge_mode="concat",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_WITHIN_CHAIN_EXTRA_CONCAT_V3",
        input_space="SEPARATE_WITHIN_CHAIN_EXTRA_ATTENTION_SCRATCH_RESIDUE_V3",
        description="Scratch ARCH-8 within-chain extra attn CONCAT",
    ),
    _mk(
        "EXP-T104",
        arch_id="ARCH-8",
        representation="SCRATCH",
        merge_mode="mean",
        experiment_id="TRF_TM_SCRATCH_FULL_SEPARATE_WITHIN_CHAIN_EXTRA_MEAN_V3",
        input_space="SEPARATE_WITHIN_CHAIN_EXTRA_ATTENTION_SCRATCH_RESIDUE_MEAN_V3",
        description="Scratch ARCH-8 within-chain extra attn MEAN",
    ),
]

assert len(SERIES) == 25
assert [s["code"] for s in SERIES] == [f"EXP-T{i:03d}" for i in range(80, 105)]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def effective_merge(merge_mode: Optional[str]) -> str:
    """Single-REG readout ignores merge; use concat for model construction."""
    return merge_mode or "concat"


def param_count_for(spec: dict) -> dict:
    content_mode = spec["content_mode"]
    merge = effective_merge(spec["merge_mode"])
    kwargs: dict[str, Any] = dict(
        content_mode=content_mode,
        annotation_mode="full",
        merge_mode=merge,
        chain_mode="HL",
        pooling_mode="reg",
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        **spec["arch"],
    )
    if content_mode == "frozen":
        kwargs["plm_hidden"] = 1280
    else:
        kwargs["plm_hidden"] = 0
    m = AnnotatedTransformer(**kwargs)
    n = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    acct = m.param_account() if hasattr(m, "param_account") else {}
    return {"n_trainable": n, "param_account": acct, "repr_dim": int(m.repr_dim)}


def write_config(spec: dict, params: dict) -> Path:
    code = spec["code"]
    is_ablingua = spec["representation"] == "ABLINGUA"
    cfg = {
        "experiment_code": code,
        "experiment_id": spec["experiment_id"],
        "platform_id": PLATFORM_ID,
        "target": "TmApp",
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM_ID}::EXP-T075",
        "transformer_type": "FROZEN_PLM" if is_ablingua else "SCRATCH",
        "input_space": spec["input_space"],
        "input_asset_ref": (
            "assets/transformer/residue_asset_manifest.yaml#ablingua600m"
            if is_ablingua
            else "assets/transformer/residue_asset_manifest.yaml#annotations+scratch_sequences"
        ),
        "plm_source": "ABLINGUA" if is_ablingua else "NONE",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": spec["merge_mode"],
        "pooling_mode": "REG",
        "content_mode": spec["content_mode"],
        "arch_id": spec["arch_id"],
        "representation": spec["representation"],
        "architecture_control": "EXP-T075",
        "historical_scratch_ref": None if is_ablingua else "EXP-T028",
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


def write_reused() -> Path:
    doc = {
        "status": "REUSED_DO_NOT_RETRAIN",
        "platform_id": PLATFORM_ID,
        "note": (
            "T075–T079 already trained under DL_FOLDLOCAL_COSINE_V3. "
            "They supply AbLingua CONCAT (or single-REG) cells of the ARCH×REP matrix. "
            "Do not retrain or overwrite their prediction/OOF artifacts."
        ),
        "experiments": [
            {
                "experiment_code": "EXP-T075",
                "arch_id": "ARCH-1",
                "representation": "ABLINGUA",
                "merge_mode": "concat",
                "role": "separate dual REG CONCAT baseline",
            },
            {
                "experiment_code": "EXP-T076",
                "arch_id": "ARCH-2",
                "representation": "ABLINGUA",
                "merge_mode": None,
                "role": "joint single REG",
            },
            {
                "experiment_code": "EXP-T077",
                "arch_id": "ARCH-3",
                "representation": "ABLINGUA",
                "merge_mode": "concat",
                "role": "joint unrestricted dual REG CONCAT",
            },
            {
                "experiment_code": "EXP-T078",
                "arch_id": "ARCH-4",
                "representation": "ABLINGUA",
                "merge_mode": "concat",
                "role": "joint chain-specific dual REG CONCAT",
            },
            {
                "experiment_code": "EXP-T079",
                "arch_id": "ARCH-5",
                "representation": "ABLINGUA",
                "merge_mode": "concat",
                "role": "gated cross-attn CONCAT",
            },
        ],
    }
    out = ROOT / "results" / "T075_T079_REUSED.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out


def main() -> int:
    if len(SERIES) != 25:
        raise SystemExit(f"expected 25 SERIES entries, got {len(SERIES)}")

    rows = []
    for spec in SERIES:
        params = param_count_for(spec)
        cfg_path = write_config(spec, params)
        sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        rows.append(
            {
                "experiment_code": spec["code"],
                "experiment_id": spec["experiment_id"],
                "description": spec["description"],
                "arch_id": spec["arch_id"],
                "representation": spec["representation"],
                "content_mode": spec["content_mode"],
                "merge_mode": spec["merge_mode"],
                "input_space": spec["input_space"],
                "architecture": spec["arch"],
                "attention_semantics": spec["attention"],
                "readout": spec["readout"],
                "hl_base": spec["hl_base"],
                "comm": spec["comm"],
                "n_trainable_preregistered": params["n_trainable"],
                "repr_dim": params["repr_dim"],
                "param_account": params["param_account"],
                "platform_id": PLATFORM_ID,
                "config_path": str(cfg_path.relative_to(ROOT)),
                "config_sha256": sha,
            }
        )
        print(
            spec["code"],
            spec["arch_id"],
            spec["representation"],
            "merge",
            spec["merge_mode"],
            "params",
            params["n_trainable"],
            "repr",
            params["repr_dim"],
            flush=True,
        )

    rev = git_rev()
    doc = {
        "status": "PREREGISTERED_BEFORE_TRAINING",
        "git_rev_at_preregistration": rev,
        "platform_id": PLATFORM_ID,
        "seed": 101,
        "n_experiments": len(rows),
        "codes": [r["experiment_code"] for r in rows],
        "reused_ablingua_cells": "results/T075_T079_REUSED.yaml",
        "primary_comparison_metric": "TEST_mean",
        "robustness_metric": "TEST_worst",
        "external_canonical_aggregation": "Primary mean",
        "planned_comparisons": [
            {"id": "MEAN_VS_CONCAT", "name": "mean_vs_concat", "note": "matched ARCH×REP pairs"},
            {"id": "ARCH6_VS_5", "name": "ungated_vs_gated_cross", "contrast": "ARCH-6 vs ARCH-5"},
            {"id": "ARCH6_VS_8", "name": "cross_vs_within", "contrast": "ARCH-6 vs ARCH-8"},
            {"id": "ARCH7_VS_1", "name": "reg_only_vs_separate", "contrast": "ARCH-7 vs ARCH-1"},
            {"id": "SCRATCH_VS_ABLINGUA", "name": "representation", "note": "matched ARCH×merge"},
            {"id": "DoD", "name": "selected_decision_contrasts", "note": "finalize-selected pairs"},
        ],
        "scratch_audit": {
            "historical_semantics": (
                "Historical TMS scratch FULL = learned AA embedding + sequence position "
                "+ chain ID + IMGT + region embeddings, combined by addition into d_model "
                "(no linear projection of the sum). content_mode='scratch' uses aa_emb; "
                "no PLM / plm_proj."
            ),
            "implementation": {
                "model": "models/antibody_transformer/model.py::_residue_stream / _content",
                "vocab": "config.AA_LIST / AA_TO_IDX (pad=0, UNK)",
                "d_model": 128,
                "annotation_mode": "FULL (IMGT + region on)",
                "formula": "x = aa_emb(aa) + pos_emb(pos) + chain_emb[chain] + imgt_emb(imgt) + region_emb(region)",
            },
            "historical_refs": ["EXP-T026", "EXP-T027", "EXP-T028"],
            "note_vs_t028": (
                "EXP-T028 used FULL scratch + mean merge under older protocol; "
                "T090–T104 reuse the same residue semantics under DL_FOLDLOCAL_COSINE_V3."
            ),
            "load_residue_bundle": (
                "Scratch may call load_residue_bundle(..., need_ablingua=False); "
                "AA/IMGT/region tensors suffice. AbDataset ignores PLM when content_mode=scratch."
            ),
        },
        "experiments": rows,
        "notes": (
            "All 25 architectures×representation settings fixed before any result inspection. "
            "T075–T079 are reused and must not be retrained. No architecture may change after "
            "training of T080–T104 begins."
        ),
    }
    out = ROOT / "results" / "T080_T104_ARCH_REP_PREREGISTRATION.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    reused = write_reused()
    print("git_rev_at_preregistration", rev, flush=True)
    print("wrote", out, flush=True)
    print("wrote", reused, flush=True)
    print("SERIES_COUNT", len(SERIES), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
