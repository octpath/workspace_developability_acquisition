#!/usr/bin/env python3
"""Preregister EXP-T105..T123 + EXP-H054..H081 under DL_FOLDLOCAL_COSINE_V3.

Writes per-code YAML configs, T105_T123_HIC_TRANSFER_GEOMETRY_PREREGISTRATION.yaml,
and HIC_CODE_ALLOCATION.yaml. Does NOT train.

Code allocation (LOCKED):
  TmApp: EXP-T105 .. EXP-T123  (19)
  HIC:   EXP-H054 .. EXP-H081  (28)  — H048–H075 occupied; +6 shift from plan §22
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
from antibody_transformer.cross_geometry import (  # noqa: E402
    DEFAULT_RBF_CENTERS,
    DEFAULT_RBF_SIGMA,
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
        "use_cross_geometry_bias": False,
    }
    base.update(flags)
    return normalize_arch_flags(base)


ARCH_SPECS: dict[str, dict] = {
    "ARCH-H0": {
        "arch": _arch(),
        "attention": "Heavy-only REG_H + H residues; no Light; no H↔L",
        "hl_base": "H_ONLY",
        "comm": "none",
    },
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
    "ARCH-6G": {
        "arch": _arch(
            use_cross_attention_bridge=True,
            cross_gate_mode="fixed_one",
            use_cross_geometry_bias=True,
        ),
        "attention": "ARCH-6 + H–L Cα RBF distance bias on residue cross-attn (zero-init ≡ ARCH-6)",
        "hl_base": "separate",
        "comm": "ungated cross-attn + geometry RBF bias",
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

PLM_HIDDEN = {
    "ablingua": 1280,
    "ablang2": 480,
    "esm2": 1280,
    None: 0,
}

ASSET_REF = {
    "ablingua": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
    "ablang2": "assets/transformer/residue_asset_manifest.yaml#ablang2",
    "esm2": "assets/transformer/residue_asset_manifest.yaml#esm2",
    None: "assets/transformer/residue_asset_manifest.yaml#annotations+scratch_sequences",
}


def _readout(arch_id: str, merge_mode: Optional[str]) -> str:
    if arch_id in ("ARCH-2", "ARCH-H0"):
        return "hidden(REG_H or REG); dim=d_model=128"
    if merge_mode == "mean":
        return "mean(REG_H, REG_L); dim=128"
    if merge_mode == "h_only":
        return "hidden(REG_H); dim=128"
    return "concat(REG_H, REG_L); dim=256"


def _rep_token(representation: str) -> str:
    return {
        "ABLINGUA": "ABLINGUA",
        "ABLANG2": "ABLANG2",
        "SCRATCH": "SCRATCH",
        "ESM2": "ESM2",
    }[representation]


def _content_plm(representation: str) -> tuple[str, Optional[str]]:
    if representation == "SCRATCH":
        return "scratch", None
    if representation == "ABLINGUA":
        return "frozen", "ablingua"
    if representation == "ABLANG2":
        return "frozen", "ablang2"
    if representation == "ESM2":
        return "frozen", "esm2"
    raise ValueError(representation)


def _input_space(
    arch_id: str,
    representation: str,
    merge_mode: Optional[str],
    *,
    geometry: bool,
) -> str:
    content = "SCRATCH" if representation == "SCRATCH" else "FROZEN"
    rep_tag = {
        "ABLINGUA": "RESIDUE",
        "ABLANG2": "ABLANG2_RESIDUE",
        "ESM2": "ESM2_RESIDUE",
        "SCRATCH": "RESIDUE",
    }[representation]
    mean = "_MEAN" if merge_mode == "mean" else ""
    geom = "_GEOM" if geometry else ""
    if arch_id == "ARCH-H0":
        return f"H_ONLY_REG_{content}_{rep_tag}_V3"
    if arch_id == "ARCH-1":
        return f"SEPARATE_DUAL_REG_{content}_{rep_tag}{mean}_V3"
    if arch_id == "ARCH-2":
        return f"JOINT_HL_SINGLE_REG_{content}_{rep_tag}_V3"
    if arch_id == "ARCH-3":
        return f"JOINT_HL_DUAL_REG_{content}_{rep_tag}{mean}_V3"
    if arch_id == "ARCH-4":
        return f"JOINT_HL_CHAIN_SPECIFIC_DUAL_REG_{content}_{rep_tag}{mean}_V3"
    if arch_id == "ARCH-6":
        return f"SEPARATE_CROSS_ATTENTION_UNGATED_{content}_{rep_tag}{mean}_V3"
    if arch_id == "ARCH-6G":
        return f"SEPARATE_CROSS_ATTENTION_UNGATED{geom}_{content}_{rep_tag}{mean}_V3"
    if arch_id == "ARCH-7":
        return f"SEPARATE_REG_ONLY_CROSS_ATTENTION_{content}_{rep_tag}{mean}_V3"
    if arch_id == "ARCH-8":
        return f"SEPARATE_WITHIN_CHAIN_EXTRA_ATTENTION_{content}_{rep_tag}{mean}_V3"
    raise ValueError(arch_id)


def _experiment_id(
    target: str,
    representation: str,
    arch_id: str,
    merge_mode: Optional[str],
    *,
    geometry: bool,
) -> str:
    tgt = "TM" if target == "TmApp" else "HIC"
    rep = _rep_token(representation)
    merge_tag = {
        None: "",
        "concat": "_CONCAT",
        "mean": "_MEAN",
        "h_only": "_HONLY",
    }[merge_mode]
    arch_map = {
        "ARCH-H0": "HONLY",
        "ARCH-1": "SEPARATE_DUAL",
        "ARCH-2": "JOINT_SINGLE_REG",
        "ARCH-3": "JOINT_DUAL",
        "ARCH-4": "JOINT_CHAIN_SPECIFIC_DUAL",
        "ARCH-6": "SEPARATE_CROSSATTN_UNGATED",
        "ARCH-6G": "SEPARATE_CROSSATTN_GEOM",
        "ARCH-7": "SEPARATE_REG_ONLY_CROSS",
        "ARCH-8": "SEPARATE_WITHIN_CHAIN_EXTRA",
    }
    # ARCH-2 has no merge suffix; ARCH-H0 uses HONLY once
    if arch_id == "ARCH-2":
        merge_tag = ""
    if arch_id == "ARCH-H0":
        merge_tag = ""
    geom = ""  # already in ARCH-6G name
    _ = geometry
    return f"TRF_{tgt}_{rep}_FULL_{arch_map[arch_id]}{merge_tag}_V3{geom}"


def _mk(
    code: str,
    *,
    target: str,
    arch_id: str,
    representation: str,
    merge_mode: Optional[str],
    description: str,
    control_code: Optional[str] = None,
) -> dict:
    meta = ARCH_SPECS[arch_id]
    content_mode, plm_source = _content_plm(representation)
    geometry = arch_id == "ARCH-6G"
    if arch_id == "ARCH-H0":
        chain_mode = "H_ONLY"
        merge_mode = "h_only"
    else:
        chain_mode = "HL"
    eid = _experiment_id(
        target, representation, arch_id, merge_mode, geometry=geometry
    )
    return {
        "code": code,
        "experiment_id": eid,
        "target": target,
        "representation": representation,
        "content_mode": content_mode,
        "plm_source": plm_source,
        "merge_mode": merge_mode,
        "chain_mode": chain_mode,
        "arch_id": arch_id,
        "arch": dict(meta["arch"]),
        "geometry": geometry,
        "input_space": _input_space(
            arch_id, representation, merge_mode, geometry=geometry
        ),
        "description": description,
        "control_code": control_code,
        "attention": meta["attention"],
        "readout": _readout(arch_id, merge_mode),
        "hl_base": meta["hl_base"],
        "comm": meta["comm"],
    }


# ---- TmApp T105–T123 (19) ----
SERIES_TM: list[dict] = [
    _mk(
        "EXP-T105",
        target="TmApp",
        arch_id="ARCH-6G",
        representation="ABLINGUA",
        merge_mode="concat",
        description="AbLingua ARCH-6G geometry cross-attn CONCAT",
        control_code="EXP-T084",
    ),
    _mk(
        "EXP-T106",
        target="TmApp",
        arch_id="ARCH-6G",
        representation="ABLINGUA",
        merge_mode="mean",
        description="AbLingua ARCH-6G geometry cross-attn MEAN",
        control_code="EXP-T085",
    ),
    _mk(
        "EXP-T107",
        target="TmApp",
        arch_id="ARCH-6G",
        representation="SCRATCH",
        merge_mode="concat",
        description="Scratch ARCH-6G geometry cross-attn CONCAT",
        control_code="EXP-T099",
    ),
    _mk(
        "EXP-T108",
        target="TmApp",
        arch_id="ARCH-6G",
        representation="SCRATCH",
        merge_mode="mean",
        description="Scratch ARCH-6G geometry cross-attn MEAN",
        control_code="EXP-T100",
    ),
    # AbLang2 matrix §15
    _mk(
        "EXP-T109",
        target="TmApp",
        arch_id="ARCH-1",
        representation="ABLANG2",
        merge_mode="concat",
        description="AbLang2 ARCH-1 separate dual CONCAT",
    ),
    _mk(
        "EXP-T110",
        target="TmApp",
        arch_id="ARCH-1",
        representation="ABLANG2",
        merge_mode="mean",
        description="AbLang2 ARCH-1 separate dual MEAN",
    ),
    _mk(
        "EXP-T111",
        target="TmApp",
        arch_id="ARCH-2",
        representation="ABLANG2",
        merge_mode=None,
        description="AbLang2 ARCH-2 joint single REG",
    ),
    _mk(
        "EXP-T112",
        target="TmApp",
        arch_id="ARCH-3",
        representation="ABLANG2",
        merge_mode="concat",
        description="AbLang2 ARCH-3 joint unrestricted dual CONCAT",
    ),
    _mk(
        "EXP-T113",
        target="TmApp",
        arch_id="ARCH-3",
        representation="ABLANG2",
        merge_mode="mean",
        description="AbLang2 ARCH-3 joint unrestricted dual MEAN",
    ),
    _mk(
        "EXP-T114",
        target="TmApp",
        arch_id="ARCH-4",
        representation="ABLANG2",
        merge_mode="concat",
        description="AbLang2 ARCH-4 joint chain-specific dual CONCAT",
    ),
    _mk(
        "EXP-T115",
        target="TmApp",
        arch_id="ARCH-4",
        representation="ABLANG2",
        merge_mode="mean",
        description="AbLang2 ARCH-4 joint chain-specific dual MEAN",
    ),
    _mk(
        "EXP-T116",
        target="TmApp",
        arch_id="ARCH-6",
        representation="ABLANG2",
        merge_mode="concat",
        description="AbLang2 ARCH-6 ungated residue cross-attn CONCAT",
    ),
    _mk(
        "EXP-T117",
        target="TmApp",
        arch_id="ARCH-6",
        representation="ABLANG2",
        merge_mode="mean",
        description="AbLang2 ARCH-6 ungated residue cross-attn MEAN",
    ),
    _mk(
        "EXP-T118",
        target="TmApp",
        arch_id="ARCH-6G",
        representation="ABLANG2",
        merge_mode="concat",
        description="AbLang2 ARCH-6G geometry cross-attn CONCAT",
        control_code="EXP-T116",
    ),
    _mk(
        "EXP-T119",
        target="TmApp",
        arch_id="ARCH-6G",
        representation="ABLANG2",
        merge_mode="mean",
        description="AbLang2 ARCH-6G geometry cross-attn MEAN",
        control_code="EXP-T117",
    ),
    _mk(
        "EXP-T120",
        target="TmApp",
        arch_id="ARCH-7",
        representation="ABLANG2",
        merge_mode="concat",
        description="AbLang2 ARCH-7 REG-only cross-attn CONCAT",
    ),
    _mk(
        "EXP-T121",
        target="TmApp",
        arch_id="ARCH-7",
        representation="ABLANG2",
        merge_mode="mean",
        description="AbLang2 ARCH-7 REG-only cross-attn MEAN",
    ),
    _mk(
        "EXP-T122",
        target="TmApp",
        arch_id="ARCH-8",
        representation="ABLANG2",
        merge_mode="concat",
        description="AbLang2 ARCH-8 within-chain extra-attn CONCAT",
    ),
    _mk(
        "EXP-T123",
        target="TmApp",
        arch_id="ARCH-8",
        representation="ABLANG2",
        merge_mode="mean",
        description="AbLang2 ARCH-8 within-chain extra-attn MEAN",
    ),
]

# ---- HIC H054–H081 (28): ESM-2 then Scratch; +6 from plan H048–H075 ----
_HIC_MATRIX = [
    ("ARCH-H0", "h_only", "Heavy-only"),
    ("ARCH-1", "concat", "CONCAT"),
    ("ARCH-1", "mean", "MEAN"),
    ("ARCH-2", None, "single REG"),
    ("ARCH-3", "concat", "CONCAT"),
    ("ARCH-3", "mean", "MEAN"),
    ("ARCH-4", "concat", "CONCAT"),
    ("ARCH-4", "mean", "MEAN"),
    ("ARCH-6", "concat", "CONCAT"),
    ("ARCH-6", "mean", "MEAN"),
    ("ARCH-6G", "concat", "CONCAT"),
    ("ARCH-6G", "mean", "MEAN"),
    ("ARCH-8", "concat", "CONCAT"),
    ("ARCH-8", "mean", "MEAN"),
]

SERIES_HIC: list[dict] = []
# ESM-2: H054–H067
for i, (arch_id, merge, label) in enumerate(_HIC_MATRIX):
    code = f"EXP-H{54 + i:03d}"
    mm = merge if arch_id != "ARCH-H0" else "h_only"
    ctrl = None
    if arch_id == "ARCH-6G":
        # matched ARCH-6 same merge / ESM2
        ctrl = f"EXP-H{54 + i - 2:03d}"
    SERIES_HIC.append(
        _mk(
            code,
            target="HIC",
            arch_id=arch_id,
            representation="ESM2",
            merge_mode=mm if arch_id != "ARCH-2" else None,
            description=f"ESM-2 {arch_id} {label}",
            control_code=ctrl,
        )
    )
# Scratch: H068–H081
for i, (arch_id, merge, label) in enumerate(_HIC_MATRIX):
    code = f"EXP-H{68 + i:03d}"
    mm = merge if arch_id != "ARCH-H0" else "h_only"
    ctrl = None
    if arch_id == "ARCH-6G":
        ctrl = f"EXP-H{68 + i - 2:03d}"
    SERIES_HIC.append(
        _mk(
            code,
            target="HIC",
            arch_id=arch_id,
            representation="SCRATCH",
            merge_mode=mm if arch_id != "ARCH-2" else None,
            description=f"Scratch {arch_id} {label}",
            control_code=ctrl,
        )
    )

SERIES: list[dict] = SERIES_TM + SERIES_HIC

assert len(SERIES_TM) == 19, len(SERIES_TM)
assert len(SERIES_HIC) == 28, len(SERIES_HIC)
assert [s["code"] for s in SERIES_TM] == [f"EXP-T{i:03d}" for i in range(105, 124)]
assert [s["code"] for s in SERIES_HIC] == [f"EXP-H{i:03d}" for i in range(54, 82)]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def effective_merge(merge_mode: Optional[str]) -> str:
    """Single-REG / H_ONLY readout uses concat|h_only for model construction."""
    if merge_mode is None:
        return "concat"
    return merge_mode


def param_count_for(spec: dict) -> dict:
    content_mode = spec["content_mode"]
    merge = effective_merge(spec["merge_mode"])
    chain_mode = spec["chain_mode"]
    if chain_mode == "H_ONLY":
        merge = "h_only"
    plm_h = PLM_HIDDEN[spec["plm_source"]]
    kwargs: dict[str, Any] = dict(
        content_mode=content_mode,
        annotation_mode="full",
        merge_mode=merge,
        chain_mode=chain_mode,
        pooling_mode="reg",
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        plm_hidden=plm_h if content_mode == "frozen" else 0,
        **spec["arch"],
    )
    # Geometry: weight tensors are registered at init; no forward/CA needed for n_trainable.
    m = AnnotatedTransformer(**kwargs)
    n = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    acct = m.param_account() if hasattr(m, "param_account") else {}
    return {"n_trainable": n, "param_account": acct, "repr_dim": int(m.repr_dim)}


def write_config(spec: dict, params: dict) -> Path:
    code = spec["code"]
    plm = spec["plm_source"]
    is_frozen = spec["content_mode"] == "frozen"
    plm_yaml = {None: "NONE", "ablingua": "ABLINGUA", "ablang2": "ABLANG2", "esm2": "ESM2"}[plm]
    cfg = {
        "experiment_code": code,
        "experiment_id": spec["experiment_id"],
        "platform_id": PLATFORM_ID,
        "target": spec["target"],
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM_ID}::EXP-T075"
        if spec["target"] == "TmApp"
        else f"{PLATFORM_ID}::HIC_TRANSFER",
        "transformer_type": "FROZEN_PLM" if is_frozen else "SCRATCH",
        "input_space": spec["input_space"],
        "input_asset_ref": ASSET_REF[plm],
        "plm_source": plm_yaml,
        "annotation_mode": "FULL",
        "chain_mode": spec["chain_mode"],
        "merge_mode": spec["merge_mode"],
        "pooling_mode": "REG" if spec["chain_mode"] == "HL" else "H_ONLY",
        "content_mode": spec["content_mode"],
        "arch_id": spec["arch_id"],
        "representation": spec["representation"],
        "geometry": bool(spec["geometry"]),
        "control_experiment_code": spec.get("control_code"),
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
        "n_trainable_preregistered": params["n_trainable"],
        "repr_dim": params["repr_dim"],
        "param_account_preregistered": params["param_account"],
        **{f"arch_{k}": v for k, v in spec["arch"].items()},
    }
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def write_hic_allocation() -> Path:
    doc = {
        "status": "LOCKED",
        "platform_id": PLATFORM_ID,
        "hic_code_block": "H054-H081",
        "n_codes": 28,
        "reason_H048_H075_unavailable": (
            "EXP-H048..H053 (and adjacent classical/XGB HIC codes through the "
            "previously planned H048–H075 window) are already occupied. "
            "Contiguous free block starts at EXP-H054. Mapping preserves the "
            "exact order from plan §22 with numeric codes shifted +6."
        ),
        "shift_from_plan_section_22": 6,
        "plan_H048_maps_to": "EXP-H054",
        "plan_H075_maps_to": "EXP-H081",
        "esm2": [
            {"code": s["code"], "arch_id": s["arch_id"], "merge_mode": s["merge_mode"], "description": s["description"]}
            for s in SERIES_HIC
            if s["representation"] == "ESM2"
        ],
        "scratch": [
            {"code": s["code"], "arch_id": s["arch_id"], "merge_mode": s["merge_mode"], "description": s["description"]}
            for s in SERIES_HIC
            if s["representation"] == "SCRATCH"
        ],
    }
    out = ROOT / "results" / "HIC_CODE_ALLOCATION.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out


def main() -> int:
    if len(SERIES) != 47:
        raise SystemExit(f"expected 47 SERIES entries, got {len(SERIES)}")

    rows = []
    for spec in SERIES:
        params = param_count_for(spec)
        cfg_path = write_config(spec, params)
        sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        rows.append(
            {
                "experiment_code": spec["code"],
                "experiment_id": spec["experiment_id"],
                "target": spec["target"],
                "description": spec["description"],
                "arch_id": spec["arch_id"],
                "representation": spec["representation"],
                "content_mode": spec["content_mode"],
                "plm_source": spec["plm_source"],
                "merge_mode": spec["merge_mode"],
                "chain_mode": spec["chain_mode"],
                "geometry": spec["geometry"],
                "control_code": spec.get("control_code"),
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
            spec["target"],
            spec["arch_id"],
            spec["representation"],
            "merge",
            spec["merge_mode"],
            "geom",
            spec["geometry"],
            "params",
            params["n_trainable"],
            flush=True,
        )

    rev = git_rev()
    doc = {
        "status": "PREREGISTERED_BEFORE_TRAINING",
        "git_rev_at_preregistration": rev,
        "platform_id": PLATFORM_ID,
        "seed": 101,
        "n_experiments": len(rows),
        "n_tmapp": len(SERIES_TM),
        "n_hic": len(SERIES_HIC),
        "codes_tm": [s["code"] for s in SERIES_TM],
        "codes_hic": [s["code"] for s in SERIES_HIC],
        "hic_code_block": "H054-H081",
        "reason_H048_H075_unavailable": (
            "H048–H075 (plan §22 window) unavailable — codes already issued for "
            "classical/XGB/prior HIC work; contiguous free block is H054–H081 "
            "(same matrix order, +6 numeric shift)."
        ),
        "primary_comparison_metric": "TEST_mean",
        "robustness_metric": "TEST_worst",
        "external_canonical_aggregation": "Primary mean",
        "geometry_rbf_spec": {
            "centers_angstrom": list(DEFAULT_RBF_CENTERS),
            "sigma": DEFAULT_RBF_SIGMA,
            "n_bases": len(DEFAULT_RBF_CENTERS),
            "weight_init": "zeros (ARCH-6G ≡ ARCH-6 at init)",
            "invalid_pair_bias": 0.0,
            "structure_source": "ESMFold Fv Cα via classical_features.ca_cache",
        },
        "ablang2": {
            "PRECONTEXTUALIZED_ACROSS_CHAINS": "NO",
            "evidence": "results/ABLANG2_RESIDUE_REPRESENTATION_AUDIT.md",
            "package": "ablang2==0.2.1",
            "checkpoint": "ablang2-paired",
            "hidden_dim": 480,
            "encoding": "H and L encoded SEPARATELY with empty partner",
        },
        "esm2": {
            "model": "facebook/esm2_t33_650M_UR50D",
            "layer": 33,
            "hidden_dim": 1280,
            "PRECONTEXTUALIZED_ACROSS_CHAINS": "NO",
            "evidence": "results/ESM2_RESIDUE_REPRESENTATION_AUDIT.md",
            "note": (
                "HL HIC experiments require ResidueBundle.esm2_l "
                "(scripts/build_esm2_light_residue_bundle.py). ARCH-H0 uses Heavy only."
            ),
        },
        "scratch_audit": {
            "historical_semantics": (
                "Historical FULL scratch = learned AA embedding + sequence position "
                "+ chain ID + IMGT + region embeddings, combined by addition into d_model "
                "(no linear projection of the sum). content_mode='scratch' uses aa_emb; "
                "no PLM / plm_proj."
            ),
            "implementation": {
                "model": "models/antibody_transformer/model.py::_residue_stream / _content",
                "d_model": 128,
                "annotation_mode": "FULL",
            },
            "historical_refs": ["EXP-T026", "EXP-T027", "EXP-T028", "EXP-T090..T104"],
            "load_residue_bundle": (
                "Scratch may call load_residue_bundle(..., need_ablingua=False); "
                "AA/IMGT/region tensors suffice."
            ),
        },
        "planned_contrasts": {
            "tm_geometry": [
                "T105 vs T084",
                "T106 vs T085",
                "T107 vs T099",
                "T108 vs T100",
                "T118 vs T116",
                "T119 vs T117",
            ],
            "tm_ablang2_merge": [
                "T110 vs T109",
                "T113 vs T112",
                "T115 vs T114",
                "T117 vs T116",
                "T119 vs T118",
                "T121 vs T120",
                "T123 vs T122",
            ],
            "tm_ablang2_hl_interaction": [
                "T111 vs T109",
                "T112/T113 vs T109/T110",
                "T114/T115 vs T112/T113",
                "T116/T117 vs T109/T110",
                "T116/T117 vs T122/T123",
                "T120/T121 vs T109/T110",
            ],
            "hic": [
                "ESM2 vs Scratch matched architecture",
                "Heavy-only vs two-chain",
                "joint vs separate",
                "chain-specific vs unrestricted",
                "cross-attn vs separate",
                "cross-attn vs within-chain",
                "geometry vs non-geometry",
                "MEAN vs CONCAT",
            ],
        },
        "experiments": rows,
        "notes": (
            "All 47 TmApp+HIC settings fixed before any result inspection. "
            "Do not alter architecture after training begins. "
            "Do not print comparative TEST rankings during train phases."
        ),
    }
    out = ROOT / "results" / "T105_T123_HIC_TRANSFER_GEOMETRY_PREREGISTRATION.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    alloc = write_hic_allocation()
    print("git_rev_at_preregistration", rev, flush=True)
    print("wrote", out, flush=True)
    print("wrote", alloc, flush=True)
    print("SERIES_TM_COUNT", len(SERIES_TM), flush=True)
    print("SERIES_HIC_COUNT", len(SERIES_HIC), flush=True)
    print("SERIES_TOTAL", len(SERIES), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
