#!/usr/bin/env python3
"""Preregister EXP-T130..T141 H/L encoder-sharing ablation (TmApp only).

Platform: DL_FOLDLOCAL_COSINE_V3 (frozen). No result-dependent branching.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PLATFORM = "DL_FOLDLOCAL_COSINE_V3"

ARCH1 = {
    "joint_hl_single_reg": False,
    "joint_hl_dual_reg": False,
    "joint_hl_chain_specific_dual_reg": False,
    "use_cross_attention_bridge": False,
    "use_reg_only_cross_attention": False,
    "use_within_chain_extra_attention": False,
    "cross_gate_mode": "learned",
    "use_cross_geometry_bias": False,
    "share_hl_encoder": False,
}
ARCH7 = dict(ARCH1)
ARCH7["use_reg_only_cross_attention"] = True

# (code, control, arch_id, merge, rep, plm, content, transformer_type, input_space_suffix)
SERIES = [
    ("EXP-T130", "EXP-T075", "ARCH-1", "concat", "ABLINGUA", "ABLINGUA", "frozen", "FROZEN_PLM", "SEPARATE_DUAL_REG_UNSHARED_ENCODER_FROZEN_RESIDUE_V3"),
    ("EXP-T131", "EXP-T080", "ARCH-1", "mean", "ABLINGUA", "ABLINGUA", "frozen", "FROZEN_PLM", "SEPARATE_DUAL_REG_UNSHARED_ENCODER_FROZEN_RESIDUE_MEAN_V3"),
    ("EXP-T132", "EXP-T086", "ARCH-7", "concat", "ABLINGUA", "ABLINGUA", "frozen", "FROZEN_PLM", "SEPARATE_REG_ONLY_CROSS_UNSHARED_ENCODER_FROZEN_RESIDUE_V3"),
    ("EXP-T133", "EXP-T087", "ARCH-7", "mean", "ABLINGUA", "ABLINGUA", "frozen", "FROZEN_PLM", "SEPARATE_REG_ONLY_CROSS_UNSHARED_ENCODER_FROZEN_RESIDUE_MEAN_V3"),
    ("EXP-T134", "EXP-T090", "ARCH-1", "concat", "SCRATCH", "NONE", "scratch", "SCRATCH", "SEPARATE_DUAL_REG_UNSHARED_ENCODER_SCRATCH_RESIDUE_V3"),
    ("EXP-T135", "EXP-T091", "ARCH-1", "mean", "SCRATCH", "NONE", "scratch", "SCRATCH", "SEPARATE_DUAL_REG_UNSHARED_ENCODER_SCRATCH_RESIDUE_MEAN_V3"),
    ("EXP-T136", "EXP-T101", "ARCH-7", "concat", "SCRATCH", "NONE", "scratch", "SCRATCH", "SEPARATE_REG_ONLY_CROSS_UNSHARED_ENCODER_SCRATCH_RESIDUE_V3"),
    ("EXP-T137", "EXP-T102", "ARCH-7", "mean", "SCRATCH", "NONE", "scratch", "SCRATCH", "SEPARATE_REG_ONLY_CROSS_UNSHARED_ENCODER_SCRATCH_RESIDUE_MEAN_V3"),
    ("EXP-T138", "EXP-T109", "ARCH-1", "concat", "ABLANG2", "ABLANG2", "frozen", "FROZEN_PLM", "SEPARATE_DUAL_REG_UNSHARED_ENCODER_FROZEN_ABLANG2_RESIDUE_V3"),
    ("EXP-T139", "EXP-T110", "ARCH-1", "mean", "ABLANG2", "ABLANG2", "frozen", "FROZEN_PLM", "SEPARATE_DUAL_REG_UNSHARED_ENCODER_FROZEN_ABLANG2_RESIDUE_MEAN_V3"),
    ("EXP-T140", "EXP-T120", "ARCH-7", "concat", "ABLANG2", "ABLANG2", "frozen", "FROZEN_PLM", "SEPARATE_REG_ONLY_CROSS_UNSHARED_ENCODER_FROZEN_ABLANG2_RESIDUE_V3"),
    ("EXP-T141", "EXP-T121", "ARCH-7", "mean", "ABLANG2", "ABLANG2", "frozen", "FROZEN_PLM", "SEPARATE_REG_ONLY_CROSS_UNSHARED_ENCODER_FROZEN_ABLANG2_RESIDUE_MEAN_V3"),
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def cfg_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def eid_for(code: str, rep: str, arch_id: str, merge: str) -> str:
    m = "MEAN" if merge == "mean" else "CONCAT"
    return f"TRF_TM_{rep}_FULL_{arch_id}_UNSHARED_ENCODER_{m}_V3"


def asset_ref(plm: str) -> str:
    if plm == "ABLANG2":
        return "assets/transformer/residue_asset_manifest.yaml#ablang2"
    if plm == "ABLINGUA":
        return "assets/transformer/residue_asset_manifest.yaml#ablingua"
    return "assets/transformer/residue_asset_manifest.yaml#scratch"


def compute_params(control_code: str, arch: dict, content_mode: str, merge: str, plm: str) -> dict:
    import sys

    sys.path.insert(0, str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from antibody_transformer.data import load_dev_test, load_residue_bundle
    from antibody_transformer.protocol_v3 import build_platform_model

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    need_ablingua = plm.upper() == "ABLINGUA"
    need_ablang2 = plm.upper() == "ABLANG2"
    rb = load_residue_bundle(
        dev, test, need_ablingua=need_ablingua, need_ablang2=need_ablang2, need_esm2=False
    )
    shared_arch = dict(arch)
    shared_arch["share_hl_encoder"] = True
    unshared_arch = dict(arch)
    unshared_arch["share_hl_encoder"] = False
    plm_src = None if content_mode == "scratch" else plm.lower()
    ms = build_platform_model(
        rb, shared_arch, content_mode=content_mode, merge_mode=merge, plm_source=plm_src
    )
    mu = build_platform_model(
        rb, unshared_arch, content_mode=content_mode, merge_mode=merge, plm_source=plm_src
    )
    ps = ms.param_account()
    pu = mu.param_account()
    return {
        "shared_control_params": ps,
        "unshared_params": pu,
        "delta_params": int(pu["total"]) - int(ps["total"]),
        "shared_encoder_params": int(ps["encoder"]),
        "duplicated_encoder_params": int(pu["encoder"]),
        "non_encoder_params": int(pu["total"]) - int(pu["encoder"]),
    }


def write_config(row: tuple, param_info: dict) -> dict:
    code, control, arch_id, merge, rep, plm, content, ttype, inp = row
    arch = ARCH7 if arch_id == "ARCH-7" else ARCH1
    ctl = yaml.safe_load((ROOT / "experiments" / "configs" / f"{control}.yaml").read_text())
    pu = param_info["unshared_params"]
    cfg = {
        "experiment_code": code,
        "experiment_id": eid_for(code, rep, arch_id, merge),
        "platform_id": PLATFORM,
        "target": "TmApp",
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM}::{control}",
        "transformer_type": ttype,
        "input_space": inp,
        "input_asset_ref": asset_ref(plm),
        "plm_source": plm,
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": merge,
        "pooling_mode": "REG",
        "content_mode": content,
        "arch_id": arch_id,
        "representation": rep,
        "geometry": False,
        "encoder_sharing": "UNSHARED",
        "control_experiment_code": control,
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
        "n_trainable_preregistered": int(pu["total"]),
        "repr_dim": 256 if merge == "concat" else 128,
        "param_account_preregistered": {k: int(v) if not isinstance(v, bool) else int(v) for k, v in pu.items() if k != "share_hl_encoder"},
        "param_delta_vs_shared": int(param_info["delta_params"]),
        "shared_control_n_trainable": int(param_info["shared_control_params"]["total"]),
        "arch_joint_hl_single_reg": False,
        "arch_joint_hl_dual_reg": False,
        "arch_joint_hl_chain_specific_dual_reg": False,
        "arch_use_cross_attention_bridge": False,
        "arch_use_reg_only_cross_attention": arch_id == "ARCH-7",
        "arch_use_within_chain_extra_attention": False,
        "arch_cross_gate_mode": "learned",
        "arch_use_cross_geometry_bias": False,
        "arch_share_hl_encoder": False,
        "description": f"{rep} {arch_id} {merge.upper()} UNSHARED encoder (control {control})",
    }
    # Preserve control license-ish fields if present
    for k in ("license_note",):
        if k in ctl:
            cfg[k] = ctl[k]
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return cfg


def main() -> int:
    # Collision check
    for code, *_ in SERIES:
        p = ROOT / "experiments" / "configs" / f"{code}.yaml"
        if p.exists():
            raise SystemExit(f"COLLISION: {code} already exists — STOP")

    records = []
    cache: dict[tuple, dict] = {}
    for row in SERIES:
        code, control, arch_id, merge, rep, plm, content, ttype, inp = row
        arch = ARCH7 if arch_id == "ARCH-7" else ARCH1
        key = (arch_id, content, merge, plm)
        if key not in cache:
            cache[key] = compute_params(control, arch, content, merge, plm)
        info = cache[key]
        cfg = write_config(row, info)
        rec = {
            "code": code,
            "target": "TmApp",
            "base_experiment": control,
            "representation": rep,
            "architecture": arch_id,
            "merge": merge,
            "encoder_sharing": "UNSHARED",
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "FFN": 256,
            "platform_id": PLATFORM,
            "seed": 101,
            "parameter_count": cfg["n_trainable_preregistered"],
            "shared_control_parameter_count": cfg["shared_control_n_trainable"],
            "delta_params": cfg["param_delta_vs_shared"],
            "config_hash": cfg_hash(cfg),
            "planned_contrast": f"{code} vs {control}",
            "description": cfg["description"],
        }
        records.append(rec)

    prereg = {
        "batch_id": "T130_T141_ENCODER_SHARING",
        "platform_id": PLATFORM,
        "scientific_question": "shared vs unshared H/L Transformer encoder weights",
        "git_rev_at_prereg": git_rev(),
        "n_experiments": len(records),
        "experiments": records,
        "isolation": {
            "untied": ["TransformerEncoder stack H", "TransformerEncoder stack L"],
            "remain_shared": [
                "aa_emb",
                "plm_proj",
                "pos_emb",
                "imgt_emb",
                "region_emb",
                "chain_emb",
                "reg_token",
                "cross_attn (ARCH-7)",
                "merge",
                "regression head",
            ],
            "init": "theta_L := exact copy(theta_H) at construction",
        },
        "freeze_statement": "No scientific config may change based on VAL/TEST/Public/Private/Overall after freeze.",
    }
    out = ROOT / "results" / "T130_T141_ENCODER_SHARING_PREREGISTRATION.yaml"
    out.write_text(yaml.safe_dump(prereg, sort_keys=False, allow_unicode=True))
    print(f"wrote {out} n={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
