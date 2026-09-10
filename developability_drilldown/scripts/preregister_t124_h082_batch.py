#!/usr/bin/env python3
"""Preregister EXP-T124..T129 (AbLang2 capacity) + EXP-H082..H093 (H047 late fusion).

Platform: DL_FOLDLOCAL_COSINE_V3 (frozen). No result-dependent branching.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PLATFORM = "DL_FOLDLOCAL_COSINE_V3"

CAPACITY = {
    "BASE": {"d_model": 128, "n_layers": 2, "n_heads": 4, "dim_feedforward": 256, "label": "BASE"},
    "DEEP": {"d_model": 128, "n_layers": 3, "n_heads": 4, "dim_feedforward": 256, "label": "DEEP"},
    "WIDE": {"d_model": 256, "n_layers": 2, "n_heads": 8, "dim_feedforward": 512, "label": "WIDE"},
    "DEEP_WIDE": {
        "d_model": 256,
        "n_layers": 3,
        "n_heads": 8,
        "dim_feedforward": 512,
        "label": "DEEP_WIDE",
    },
}

ARCH3 = {
    "joint_hl_single_reg": False,
    "joint_hl_dual_reg": True,
    "joint_hl_chain_specific_dual_reg": False,
    "use_cross_attention_bridge": False,
    "use_reg_only_cross_attention": False,
    "use_within_chain_extra_attention": False,
    "cross_gate_mode": "learned",
    "use_cross_geometry_bias": False,
}
ARCH7 = {
    "joint_hl_single_reg": False,
    "joint_hl_dual_reg": False,
    "joint_hl_chain_specific_dual_reg": False,
    "use_cross_attention_bridge": False,
    "use_reg_only_cross_attention": True,
    "use_within_chain_extra_attention": False,
    "cross_gate_mode": "learned",
    "use_cross_geometry_bias": False,
}
ARCH_H0 = {k: False for k in ARCH3 if k != "cross_gate_mode"}
ARCH_H0["cross_gate_mode"] = "learned"
ARCH_H0["use_cross_geometry_bias"] = False
ARCH4 = dict(ARCH3)
ARCH4["joint_hl_dual_reg"] = False
ARCH4["joint_hl_chain_specific_dual_reg"] = True
ARCH2 = dict(ARCH3)
ARCH2["joint_hl_dual_reg"] = False
ARCH2["joint_hl_single_reg"] = True

SERIES_TM = [
    {
        "code": "EXP-T124",
        "base": "EXP-T113",
        "arch_id": "ARCH-3",
        "arch": ARCH3,
        "capacity_id": "DEEP",
        "description": "AbLang2 ARCH-3 MEAN DEEP d128 L3",
    },
    {
        "code": "EXP-T125",
        "base": "EXP-T113",
        "arch_id": "ARCH-3",
        "arch": ARCH3,
        "capacity_id": "WIDE",
        "description": "AbLang2 ARCH-3 MEAN WIDE d256 H8",
    },
    {
        "code": "EXP-T126",
        "base": "EXP-T113",
        "arch_id": "ARCH-3",
        "arch": ARCH3,
        "capacity_id": "DEEP_WIDE",
        "description": "AbLang2 ARCH-3 MEAN DEEP_WIDE",
    },
    {
        "code": "EXP-T127",
        "base": "EXP-T121",
        "arch_id": "ARCH-7",
        "arch": ARCH7,
        "capacity_id": "DEEP",
        "description": "AbLang2 ARCH-7 MEAN DEEP d128 L3",
    },
    {
        "code": "EXP-T128",
        "base": "EXP-T121",
        "arch_id": "ARCH-7",
        "arch": ARCH7,
        "capacity_id": "WIDE",
        "description": "AbLang2 ARCH-7 MEAN WIDE d256 H8",
    },
    {
        "code": "EXP-T129",
        "base": "EXP-T121",
        "arch_id": "ARCH-7",
        "arch": ARCH7,
        "capacity_id": "DEEP_WIDE",
        "description": "AbLang2 ARCH-7 MEAN DEEP_WIDE",
    },
]

HIC_BACKBONES = {
    "H054": {
        "base": "EXP-H054",
        "arch_id": "ARCH-H0",
        "arch": ARCH_H0,
        "representation": "ESM2",
        "plm_source": "esm2",
        "content_mode": "frozen",
        "chain_mode": "H_ONLY",
        "merge_mode": "h_only",
    },
    "H061": {
        "base": "EXP-H061",
        "arch_id": "ARCH-4",
        "arch": ARCH4,
        "representation": "ESM2",
        "plm_source": "esm2",
        "content_mode": "frozen",
        "chain_mode": "HL",
        "merge_mode": "mean",
    },
    "H071": {
        "base": "EXP-H071",
        "arch_id": "ARCH-2",
        "arch": ARCH2,
        "representation": "SCRATCH",
        "plm_source": None,
        "content_mode": "scratch",
        "chain_mode": "HL",
                "merge_mode": "concat",  # unused for single-REG; matches AnnotatedTransformer default family
    },
}

BUNDLES = [
    ("F1_SURFACE", "F1"),
    ("F2_SEQUENCE_TITRATION", "F2"),
    ("F3_LOCAL_RASA_CDR3", "F3"),
    ("F4_H047_AUX_ALL", "F4"),
]

SERIES_HIC = []
code_i = 82
for bb_key in ("H054", "H061", "H071"):
    bb = HIC_BACKBONES[bb_key]
    for bundle_id, short in BUNDLES:
        SERIES_HIC.append(
            {
                "code": f"EXP-H{code_i:03d}",
                "base": bb["base"],
                "backbone_key": bb_key,
                "arch_id": bb["arch_id"],
                "arch": bb["arch"],
                "representation": bb["representation"],
                "plm_source": bb["plm_source"],
                "content_mode": bb["content_mode"],
                "chain_mode": bb["chain_mode"],
                "merge_mode": bb["merge_mode"],
                "fusion_bundle_id": bundle_id,
                "fusion_short": short,
                "description": f"{bb['base']} + {short} late fusion",
            }
        )
        code_i += 1

SERIES = SERIES_TM + SERIES_HIC


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_tm_config(spec: dict) -> Path:
    cap = CAPACITY[spec["capacity_id"]]
    code = spec["code"]
    arch = spec["arch"]
    eid = (
        f"TRF_TM_ABLANG2_FULL_{spec['arch_id'].replace('-', '')}_MEAN_"
        f"{spec['capacity_id']}_V3"
    )
    space = (
        f"JOINT_HL_DUAL_REG_FROZEN_ABLANG2_RESIDUE_MEAN_{spec['capacity_id']}_V3"
        if spec["arch_id"] == "ARCH-3"
        else f"SEPARATE_REG_ONLY_CROSS_ATTENTION_FROZEN_ABLANG2_RESIDUE_MEAN_{spec['capacity_id']}_V3"
    )
    cfg = {
        "experiment_code": code,
        "experiment_id": eid,
        "platform_id": PLATFORM,
        "target": "TmApp",
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM}::{spec['base']}",
        "transformer_type": "FROZEN_PLM",
        "input_space": space,
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablang2",
        "plm_source": "ABLANG2",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": "mean",
        "pooling_mode": "REG",
        "content_mode": "frozen",
        "arch_id": spec["arch_id"],
        "representation": "ABLANG2",
        "geometry": False,
        "control_experiment_code": spec["base"],
        "capacity_id": spec["capacity_id"],
        "d_model": cap["d_model"],
        "n_layers": cap["n_layers"],
        "n_heads": cap["n_heads"],
        "ff_dim": cap["dim_feedforward"],
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
        "arch_joint_hl_single_reg": arch["joint_hl_single_reg"],
        "arch_joint_hl_dual_reg": arch["joint_hl_dual_reg"],
        "arch_joint_hl_chain_specific_dual_reg": arch["joint_hl_chain_specific_dual_reg"],
        "arch_use_cross_attention_bridge": arch["use_cross_attention_bridge"],
        "arch_use_reg_only_cross_attention": arch["use_reg_only_cross_attention"],
        "arch_use_within_chain_extra_attention": arch["use_within_chain_extra_attention"],
        "arch_cross_gate_mode": arch["cross_gate_mode"],
        "arch_use_cross_geometry_bias": False,
        "description": spec["description"],
    }
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def write_hic_config(spec: dict) -> Path:
    code = spec["code"]
    arch = spec["arch"]
    plm = spec["plm_source"]
    plm_yaml = {None: "NONE", "esm2": "ESM2"}[plm]
    eid = (
        f"TRF_HIC_{spec['representation']}_{spec['arch_id'].replace('-', '')}_"
        f"{spec['fusion_short']}_LATEFUSION_V3"
    )
    space = (
        f"LATEFUSION_{spec['fusion_bundle_id']}_"
        f"{'HONLY_' if spec['chain_mode']=='H_ONLY' else ''}"
        f"{'SCRATCH' if spec['content_mode']=='scratch' else 'FROZEN_ESM2'}_V3"
    )
    cfg = {
        "experiment_code": code,
        "experiment_id": eid,
        "platform_id": PLATFORM,
        "target": "HIC",
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM}::{spec['base']}",
        "transformer_type": "LATE_FUSION",
        "input_space": space,
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml",
        "plm_source": plm_yaml,
        "annotation_mode": "FULL",
        "chain_mode": spec["chain_mode"],
        "merge_mode": spec["merge_mode"],
        "pooling_mode": "REG" if spec["chain_mode"] == "HL" else "H_ONLY",
        "content_mode": spec["content_mode"],
        "arch_id": spec["arch_id"],
        "representation": spec["representation"],
        "geometry": False,
        "control_experiment_code": spec["base"],
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
        "arch_joint_hl_single_reg": arch["joint_hl_single_reg"],
        "arch_joint_hl_dual_reg": arch["joint_hl_dual_reg"],
        "arch_joint_hl_chain_specific_dual_reg": arch["joint_hl_chain_specific_dual_reg"],
        "arch_use_cross_attention_bridge": arch["use_cross_attention_bridge"],
        "arch_use_reg_only_cross_attention": arch["use_reg_only_cross_attention"],
        "arch_use_within_chain_extra_attention": arch["use_within_chain_extra_attention"],
        "arch_cross_gate_mode": arch["cross_gate_mode"],
        "arch_use_cross_geometry_bias": False,
        "description": spec["description"],
    }
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def write_arch_audit() -> None:
    text = """# T124–T129 architecture audit (ARCH-3 / ARCH-7 depth)

## BASE A — EXP-T113 (ARCH-3)

- Joint H/L Transformer over `[REG_H, H…, REG_L, L…]`
- Unrestricted self-attention across all valid tokens
- Readout: MEAN of REG_H and REG_L
- Encoder: full `nn.TransformerEncoder` with `n_layers` stacked layers

**Depth increase (2→3):** adds one ordinary encoder self-attention layer.
No cross-attention module exists.

## BASE B — EXP-T121 (ARCH-7)

Layer flow (unchanged scientifically):

1. Separate H and L chain embedding (+ REG)
2. Shared-weight `TransformerEncoder` self-attention stack (`n_layers`)
3. **Single** REG-only cross-attention stage:
   - REG_H queries Light residues
   - REG_L queries Heavy residues
4. MEAN merge → head

**Depth increase (2→3):** adds one encoding self-attention layer inside step 2.
Does **not** add a second REG-only cross-attention block.

## Capacity packages

| ID | d_model | layers | heads | FFN | d_head |
|----|--------:|-------:|------:|----:|-------:|
| BASE | 128 | 2 | 4 | 256 | 32 |
| DEEP | 128 | 3 | 4 | 256 | 32 |
| WIDE | 256 | 2 | 8 | 512 | 32 |
| DEEP_WIDE | 256 | 3 | 8 | 512 | 32 |

WIDE is a capacity **package** (d_model+heads+FFN together), not d_model alone.
"""
    (ROOT / "results" / "T124_T129_ARCHITECTURE_AUDIT.md").write_text(text, encoding="utf-8")


def main() -> int:
    # Collision check
    codes_csv = ROOT / "results" / "EXPERIMENT_CODES.csv"
    import pandas as pd

    used = set(pd.read_csv(codes_csv)["experiment_code"])
    need = [s["code"] for s in SERIES]
    coll = [c for c in need if c in used]
    if coll:
        raise SystemExit(f"CODE COLLISION — STOP before training: {coll}")

    cfg_dir = ROOT / "experiments" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_hashes = {}
    for s in SERIES_TM:
        p = write_tm_config(s)
        config_hashes[s["code"]] = file_sha(p)
    for s in SERIES_HIC:
        p = write_hic_config(s)
        config_hashes[s["code"]] = file_sha(p)

    write_arch_audit()

    from antibody_transformer.h047_aux_features import H047AuxFeatureStore

    feat_hashes = {}
    for bid, _ in BUNDLES:
        store = H047AuxFeatureStore(bid)
        feat_hashes[bid] = {
            "artifact_hash": store.artifact_hash,
            "raw_dim": store.raw_dim,
            "effective_dim": store.effective_dim,
            "exclude_esm2_h": True,
        }

    prereg = {
        "status": "FROZEN",
        "platform_id": PLATFORM,
        "git_rev": git_rev(),
        "tm_codes": [s["code"] for s in SERIES_TM],
        "hic_codes": [s["code"] for s in SERIES_HIC],
        "n_experiments": len(SERIES),
        "capacity_packages": CAPACITY,
        "feature_bundles": feat_hashes,
        "config_sha256": config_hashes,
        "planned_contrasts_tm": [
            "T124 vs T113",
            "T125 vs T113",
            "T126 vs T113",
            "T126 vs T125",
            "T126 vs T124",
            "T127 vs T121",
            "T128 vs T121",
            "T129 vs T121",
            "T129 vs T128",
            "T129 vs T127",
            "T124 vs T127",
            "T125 vs T128",
            "T126 vs T129",
        ],
        "planned_contrasts_hic": [
            "F1/F2/F3/F4 vs base per backbone H054/H061/H071",
            "F4 vs F1/F2/F3 per backbone",
        ],
        "experiments": [],
        "statement": (
            "All T124–T129 and H082–H093 configs frozen before training. "
            "No architecture/feature changes based on VAL/TEST/Public/Private/Overall."
        ),
    }
    for s in SERIES_TM:
        cap = CAPACITY[s["capacity_id"]]
        prereg["experiments"].append(
            {
                "code": s["code"],
                "target": "TmApp",
                "base": s["base"],
                "representation": "ABLANG2",
                "architecture": s["arch_id"],
                "merge": "mean",
                "capacity_id": s["capacity_id"],
                "d_model": cap["d_model"],
                "n_layers": cap["n_layers"],
                "n_heads": cap["n_heads"],
                "FFN": cap["dim_feedforward"],
                "seed": 101,
                "platform": PLATFORM,
                "config_sha256": config_hashes[s["code"]],
            }
        )
    for s in SERIES_HIC:
        prereg["experiments"].append(
            {
                "code": s["code"],
                "target": "HIC",
                "base": s["base"],
                "representation": s["representation"],
                "architecture": s["arch_id"],
                "merge": s["merge_mode"],
                "fusion_bundle_id": s["fusion_bundle_id"],
                "aux_effective_dim": feat_hashes[s["fusion_bundle_id"]]["effective_dim"],
                "seed": 101,
                "platform": PLATFORM,
                "config_sha256": config_hashes[s["code"]],
            }
        )

    out = ROOT / "results" / "T124_T129_H082_H093_PREREGISTRATION.yaml"
    out.write_text(yaml.safe_dump(prereg, sort_keys=False), encoding="utf-8")
    print("wrote", out)
    print("configs", len(config_hashes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
