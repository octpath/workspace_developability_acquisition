#!/usr/bin/env python3
"""Preregister EXP-H114..H125 SOURCE SAP/SCM Transformer fusion ablation."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))

from antibody_transformer.late_fusion import DirectLateFusionModel, LateFusionModel  # noqa: E402
from antibody_transformer.source_sap_scm_aux import (  # noqa: E402
    BUNDLE_TO_SOURCE,
    SourceSapScmAuxFeatureStore,
)
from experiment_codes import next_code  # noqa: E402

PLATFORM = "DL_FOLDLOCAL_COSINE_V3"
PREREG = ROOT / "results" / "H114_H125_SOURCE24_FUSION_PREREGISTRATION.yaml"
SRC_FEAT = REPO / "feature_research" / "hic_sap_scm_source" / "features"
EXP_FEAT = ROOT / "experiments" / "features"
SOURCE_COMMIT = "fb3c4192"

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


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_features() -> dict:
    EXP_FEAT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for bid, name in BUNDLE_TO_SOURCE.items():
        src = SRC_FEAT / name
        dst = EXP_FEAT / name
        shutil.copy2(src, dst)
        df = pd.read_parquet(dst)
        cols = [c for c in df.columns if c != "id"]
        meta[bid] = {
            "parquet": f"experiments/features/{name}",
            "columns": cols,
            "n_dims": len(cols),
            "file_sha256": file_sha(dst),
            "n_antibodies": int(df["id"].nunique()),
        }
        assert meta[bid]["n_antibodies"] == 324
    return meta


def _aux_param_count(p: int, mode: str) -> int:
    """Aux branch + fused head params for a stub backbone repr_dim=256, d_model=128."""
    # Use analytical counts matching DirectLateFusion / LateFusion with d_model=128, repr=256
    # Scratch ARCH-2 concat typically repr_dim = d_model (REG pooling) — compute from real module.
    class _T:
        repr_dim = 128
        d_model = 128

        def forward_repr(self, batch):
            raise NotImplementedError

    import torch

    t = _T()
    if mode == "late_concat_direct":
        m = DirectLateFusionModel(t, p, dropout=0.2)  # type: ignore[arg-type]
    else:
        m = LateFusionModel(t, p, dropout=0.2)  # type: ignore[arg-type]
    # Only fusion-side params (exclude nonexistent backbone)
    return int(sum(x.numel() for x in m.parameters()))


def build_series(feat_meta: dict) -> list[dict]:
    rows = [
        ("EXP-H114", "EXP-H071", "FS_HIC_SOURCE_SAP24", "late_concat_direct", "Scratch + SAP24 DIRECT"),
        ("EXP-H115", "EXP-H071", "FS_HIC_SOURCE_SAP24", "late_concat_aux32", "Scratch + SAP24 AUX32"),
        ("EXP-H116", "EXP-H071", "FS_HIC_SOURCE_SCM24", "late_concat_direct", "Scratch + SCM24 DIRECT"),
        ("EXP-H117", "EXP-H071", "FS_HIC_SOURCE_SCM24", "late_concat_aux32", "Scratch + SCM24 AUX32"),
        ("EXP-H118", "EXP-H071", "FS_HIC_SOURCE_SAP24_SCM24", "late_concat_direct", "Scratch + COMBINED48 DIRECT"),
        ("EXP-H119", "EXP-H071", "FS_HIC_SOURCE_SAP24_SCM24", "late_concat_aux32", "Scratch + COMBINED48 AUX32"),
        ("EXP-H120", "EXP-H061", "FS_HIC_SOURCE_SAP24", "late_concat_direct", "ESM2 + SAP24 DIRECT"),
        ("EXP-H121", "EXP-H061", "FS_HIC_SOURCE_SAP24", "late_concat_aux32", "ESM2 + SAP24 AUX32"),
        ("EXP-H122", "EXP-H061", "FS_HIC_SOURCE_SCM24", "late_concat_direct", "ESM2 + SCM24 DIRECT"),
        ("EXP-H123", "EXP-H061", "FS_HIC_SOURCE_SCM24", "late_concat_aux32", "ESM2 + SCM24 AUX32"),
        ("EXP-H124", "EXP-H061", "FS_HIC_SOURCE_SAP24_SCM24", "late_concat_direct", "ESM2 + COMBINED48 DIRECT"),
        ("EXP-H125", "EXP-H061", "FS_HIC_SOURCE_SAP24_SCM24", "late_concat_aux32", "ESM2 + COMBINED48 AUX32"),
    ]
    out = []
    for code, backbone, bid, fmode, desc in rows:
        is_esm = backbone == "EXP-H061"
        meta = feat_meta[bid]
        dim = meta["n_dims"]
        tag = bid.replace("FS_HIC_", "")
        ftag = "DIRECT" if fmode.endswith("direct") else "AUX32"
        # Approximate fusion-only params (backbone excluded); full n_trainable recorded at train time
        fusion_only = _aux_param_count(dim, fmode)
        out.append(
            {
                "code": code,
                "backbone": backbone,
                "control_experiment_code": backbone,
                "fusion_bundle_id": bid,
                "fusion_mode": fmode,
                "aux_dim": dim,
                "feature_columns": meta["columns"],
                "feature_file_sha256": meta["file_sha256"],
                "feature_parquet": meta["parquet"],
                "fusion_params_approx_excl_backbone": fusion_only,
                "description": desc,
                "experiment_id": (
                    f"TRF_HIC_{'ESM2_ARCH4' if is_esm else 'SCRATCH_ARCH2'}_{tag}_{ftag}_V3"
                ),
                "input_space": (
                    f"LATEFUSION_{tag}_{ftag}_{'ESM2' if is_esm else 'SCRATCH'}_V3"
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
        "fusion_mode": spec["fusion_mode"],
        "aux_dim": spec["aux_dim"],
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
        "descriptor_id": "SOURCE_SAP_SCM",
        "structure_scope": "Fv",
        "source_research_commit": SOURCE_COMMIT,
        "feature_file_sha256": spec["feature_file_sha256"],
        "fusion_params_approx_excl_backbone": spec["fusion_params_approx_excl_backbone"],
    }
    path = ROOT / "experiments" / "configs" / f"{spec['code']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    nxt = next_code("HIC")
    if nxt != "EXP-H114":
        print(f"ERROR: expected next HIC EXP-H114, got {nxt}", flush=True)
        return 1

    feat_meta = install_features()
    # verify stores
    for bid in BUNDLE_TO_SOURCE:
        store = SourceSapScmAuxFeatureStore(bid)
        assert store.effective_dim == feat_meta[bid]["n_dims"]
        feat_meta[bid]["store_artifact_hash"] = store.artifact_hash

    series = build_series(feat_meta)
    assert [s["code"] for s in series] == [f"EXP-H{i}" for i in range(114, 126)]

    for spec in series:
        write_config(spec)

    global SERIES
    SERIES = series

    doc = {
        "batch_id": "H114_H125_SOURCE24_FUSION_ABLATION",
        "status": "PREREGISTERED_FROZEN",
        "platform_id": PLATFORM,
        "seed": 101,
        "git_rev_at_prereg": git_rev(),
        "source_research_commit": SOURCE_COMMIT,
        "next_code_at_prereg": nxt,
        "central_contrasts": {
            "sap24_vs_base": {
                "scratch": ["EXP-H114", "EXP-H115", "vs EXP-H071"],
                "esm2": ["EXP-H120", "EXP-H121", "vs EXP-H061"],
            },
            "fusion_mode": {
                "pairs": [
                    ["EXP-H114", "EXP-H115"],
                    ["EXP-H116", "EXP-H117"],
                    ["EXP-H118", "EXP-H119"],
                    ["EXP-H120", "EXP-H121"],
                    ["EXP-H122", "EXP-H123"],
                    ["EXP-H124", "EXP-H125"],
                ]
            },
            "sap_vs_scm_vs_combined": {
                "direct_scratch": ["EXP-H114", "EXP-H116", "EXP-H118"],
                "aux32_scratch": ["EXP-H115", "EXP-H117", "EXP-H119"],
                "direct_esm2": ["EXP-H120", "EXP-H122", "EXP-H124"],
                "aux32_esm2": ["EXP-H121", "EXP-H123", "EXP-H125"],
            },
        },
        "feature_blocks": feat_meta,
        "experiments": [
            {
                "code": s["code"],
                "backbone": s["backbone"],
                "fusion_bundle_id": s["fusion_bundle_id"],
                "fusion_mode": s["fusion_mode"],
                "aux_dim": s["aux_dim"],
                "feature_file_sha256": s["feature_file_sha256"],
                "feature_columns": s["feature_columns"],
                "fusion_params_approx_excl_backbone": s["fusion_params_approx_excl_backbone"],
                "description": s["description"],
                "experiment_id": s["experiment_id"],
            }
            for s in series
        ],
        "notes": [
            "DIRECT = no aux MLP; AUX32 = historical LateFusionAuxMLP unchanged.",
            "Do not run SURFACE+SOURCE, XGBoost, residue-level, H126, or T142 in this batch.",
        ],
    }
    PREREG.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print("Wrote", PREREG, flush=True)
    print("Installed features:", list(feat_meta), flush=True)
    for s in series:
        print(
            s["code"],
            s["fusion_mode"],
            s["fusion_bundle_id"],
            "aux_dim",
            s["aux_dim"],
            "fusion_params≈",
            s["fusion_params_approx_excl_backbone"],
            flush=True,
        )
    return 0


SERIES: list[dict] = []

if __name__ == "__main__":
    raise SystemExit(main())
