#!/usr/bin/env python3
"""Preregister EXP-H102..H113 promoted HSP late-fusion batch (confirmatory only)."""
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

PLATFORM = "DL_FOLDLOCAL_COSINE_V3"
PREREG = ROOT / "results" / "H102_H113_HSP_MAINLINE_PREREGISTRATION.yaml"
CANON = ROOT / "results" / "H102_H113_HSP_CANONICAL_FEATURES.json"
SOURCE_COMMIT = "210a270d"

SCRATCH_ARCH = {
    "arch_joint_hl_single_reg": True,
    "arch_joint_hl_dual_reg": False,
    "arch_joint_hl_chain_specific_dual_reg": False,
    "arch_use_cross_attention_bridge": False,
    "arch_use_reg_only_cross_attention": False,
    "arch_use_within_chain_extra_attention": False,
    "arch_cross_gate_mode": "learned",
    "arch_use_cross_geometry_bias": False,
}
ESM2_ARCH = {
    "arch_joint_hl_single_reg": False,
    "arch_joint_hl_dual_reg": False,
    "arch_joint_hl_chain_specific_dual_reg": True,
    "arch_use_cross_attention_bridge": False,
    "arch_use_reg_only_cross_attention": False,
    "arch_use_within_chain_extra_attention": False,
    "arch_cross_gate_mode": "learned",
    "arch_use_cross_geometry_bias": False,
}


def _fam(canon: dict, pid: str) -> dict:
    return canon["families"][pid]


def build_series(canon: dict) -> list[dict]:
    p1, p2, p3 = _fam(canon, "P1"), _fam(canon, "P2"), _fam(canon, "P3")
    specs = [
        # Scratch — HSP alone
        ("EXP-H102", "EXP-H071", None, p1, p1["fs_id"], 3, "Scratch + P1 BM-R5 HSP alone"),
        ("EXP-H103", "EXP-H071", "EXP-H090", p1, p1["surface_fs_id"], 38, "Scratch + SURFACE + P1 BM-R5"),
        ("EXP-H104", "EXP-H071", None, p2, p2["fs_id"], 3, "Scratch + P2 FP-R5 HSP alone"),
        ("EXP-H105", "EXP-H071", "EXP-H090", p2, p2["surface_fs_id"], 38, "Scratch + SURFACE + P2 FP-R5"),
        ("EXP-H106", "EXP-H071", None, p3, p3["fs_id"], 3, "Scratch + P3 EIS-R8 HSP alone"),
        ("EXP-H107", "EXP-H071", "EXP-H090", p3, p3["surface_fs_id"], 38, "Scratch + SURFACE + P3 EIS-R8"),
        # ESM2 — HSP alone / SURFACE+HSP
        ("EXP-H108", "EXP-H061", None, p1, p1["fs_id"], 3, "ESM2 + P1 BM-R5 HSP alone"),
        ("EXP-H109", "EXP-H061", "EXP-H086", p1, p1["surface_fs_id"], 38, "ESM2 + SURFACE + P1 BM-R5"),
        ("EXP-H110", "EXP-H061", None, p2, p2["fs_id"], 3, "ESM2 + P2 FP-R5 HSP alone"),
        ("EXP-H111", "EXP-H061", "EXP-H086", p2, p2["surface_fs_id"], 38, "ESM2 + SURFACE + P2 FP-R5"),
        ("EXP-H112", "EXP-H061", None, p3, p3["fs_id"], 3, "ESM2 + P3 EIS-R8 HSP alone"),
        ("EXP-H113", "EXP-H061", "EXP-H086", p3, p3["surface_fs_id"], 38, "ESM2 + SURFACE + P3 EIS-R8"),
    ]
    out = []
    for code, backbone, surface_ctrl, fam, bid, dim, desc in specs:
        is_esm = backbone == "EXP-H061"
        tag = bid.replace("FS_HIC_", "")
        out.append(
            {
                "code": code,
                "backbone": backbone,
                "control_no_aux": backbone if surface_ctrl is None else None,
                "control_surface": surface_ctrl,
                "fusion_bundle_id": bid,
                "aux_dim": dim,
                "hsp_family_id": fam["family_id"],
                "hsp_bundle_id": "B3",
                "hsp_columns": fam["columns"],
                "source_spec_hash": fam["source_spec_hash"],
                "mainline_feature_hash": fam["mainline_feature_hash"],
                "property_table_hash": fam["property_table_hash"],
                "structure_hash": fam["structure_hash"],
                "promoted_id": next(k for k, v in canon["families"].items() if v["family_id"] == fam["family_id"]),
                "description": desc,
                "experiment_id": (
                    f"TRF_HIC_{'ESM2_ARCH4' if is_esm else 'SCRATCH_ARCH2'}_{tag}_V3"
                ),
                "input_space": (
                    f"LATEFUSION_{tag}_{'ESM2' if is_esm else 'SCRATCH'}_V3"
                ),
                "plm_source": "ESM2" if is_esm else "NONE",
                "content_mode": "frozen" if is_esm else "scratch",
                "representation": "ESM2" if is_esm else "SCRATCH",
                "arch_id": "ARCH-4" if is_esm else "ARCH-2",
                "merge_mode": "mean" if is_esm else "concat",
                "transformer_type": "LATE_FUSION",
                "arch": ESM2_ARCH if is_esm else SCRATCH_ARCH,
            }
        )
    return out


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


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
        "descriptor_id": "HSP_PROMOTED",
        "hsp_family_id": spec["hsp_family_id"],
        "hsp_bundle_id": "B3",
        "structure_scope": "Fv",
        "source_hsp_commit": SOURCE_COMMIT,
        "mainline_feature_hash": spec["mainline_feature_hash"],
    }
    path = ROOT / "experiments" / "configs" / f"{spec['code']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    if not CANON.exists():
        print("ERROR: missing canonical features JSON", flush=True)
        return 1
    canon = json.loads(CANON.read_text())
    series = build_series(canon)
    assert [s["code"] for s in series] == [f"EXP-H{i}" for i in range(102, 114)]

    for spec in series:
        write_config(spec)

    # module-level SERIES for runner import
    global SERIES
    SERIES = series

    doc = {
        "batch_id": "H102_H113_HSP_PROMOTED_MAINLINE",
        "status": "PREREGISTERED_FROZEN",
        "platform_id": PLATFORM,
        "seed": 101,
        "git_rev_at_prereg": git_rev(),
        "source_research_commit": SOURCE_COMMIT,
        "source_track": "feature_research/hic_spatial_hydrophobicity",
        "canonical_features": str(CANON),
        "promotion_audit": str(ROOT / "results" / "H102_H113_HSP_PROMOTION_AUDIT.md"),
        "bundle_id": "B3",
        "late_fusion": "Linear(p,64)->GELU->Dropout(0.2)->Linear(64,32)->GELU; concat with z_DL",
        "preprocessing": "TRAIN-only median impute + StandardScaler per logical block",
        "training_platform": PLATFORM,
        "matched_comparisons": {
            "base_scratch": ["H102-H071", "H104-H071", "H106-H071"],
            "base_esm2": ["H108-H061", "H110-H061", "H112-H061"],
            "replace_surface_scratch": ["H102-H090", "H104-H090", "H106-H090"],
            "replace_surface_esm2": ["H108-H086", "H110-H086", "H112-H086"],
            "increment_scratch_PRIMARY": ["H103-H090", "H105-H090", "H107-H090"],
            "increment_esm2_PRIMARY": ["H109-H086", "H111-H086", "H113-H086"],
        },
        "controls_reused_not_retrained": [
            "EXP-H071",
            "EXP-H090",
            "EXP-H093",
            "EXP-H061",
            "EXP-H086",
            "EXP-H089",
        ],
        "forbidden": [
            "HSP re-optimization",
            "multi-HSP combination",
            "F4+HSP",
            "residue-level HSP",
            "charge/aromatic atlas",
            "T142",
            "H114+",
        ],
        "series": [
            {
                k: v
                for k, v in s.items()
                if k != "arch"
            }
            for s in series
        ],
        "statement": "Configs + promoted HSP B3 columns frozen before training. No result-dependent changes.",
    }
    PREREG.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    # also dump SERIES as importable sidecar
    (ROOT / "results" / "H102_H113_SERIES.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "arch"} for s in series], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {PREREG}", flush=True)
    print(f"Configs: {[s['code'] for s in series]}", flush=True)
    return 0


# Default SERIES for imports before main() — rebuilt from CANON if present
if CANON.exists():
    SERIES = build_series(json.loads(CANON.read_text()))
else:
    SERIES = []


if __name__ == "__main__":
    raise SystemExit(main())
