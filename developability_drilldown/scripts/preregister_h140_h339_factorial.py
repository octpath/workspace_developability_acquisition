#!/usr/bin/env python3
"""Formal preregistration: HIC Representation × Topology × Annotation factorial.

Issues EXP-H140..EXP-H339 (200 cells, 0 REUSE), writes configs, freezes manifests.
Does NOT train.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REPORTS = REPO / "reports"
sys_path_scripts = ROOT / "scripts"

import sys

sys.path.insert(0, str(sys_path_scripts))
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402

PRE_FREEZE_MANIFEST_SHA = "ffdf88d6d59cc6af9864fa041c4a057d398f523ce724fb55d392c50282e5f7cf"
EVAL_FREEZE_SHA = "379e0751a93c2af8f6fbfeedaad4d72f3556996b"

# Deterministic factor orders (match human-approved gate manifest)
REP_ORDER = [
    {
        "representation_id": "Scratch",
        "representation": "scratch",
        "plm_family": "scratch",
        "plm_source": None,
        "content_mode": "scratch",
        "subdir": None,
        "raw_dim": 0,
        "representation_context": "LEARNED_AA",
        "paired_match_group": None,
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#annotations+scratch_sequences",
    },
    {
        "representation_id": "AbLingua",
        "representation": "ablingua",
        "plm_family": "ablingua",
        "plm_source": "ablingua",
        "content_mode": "frozen",
        "subdir": "ablingua600m",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
    },
    {
        "representation_id": "AbLang1",
        "representation": "ablang1",
        "plm_family": "ablang1",
        "plm_source": "ablang1",
        "content_mode": "frozen",
        "subdir": "ablang1",
        "raw_dim": 768,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablang1",
    },
    {
        "representation_id": "AbLang2_SEPARATE_CHAIN",
        "representation": "ablang2_paired",
        "plm_family": "ablang2",
        "plm_source": "ablang2",
        "content_mode": "frozen",
        "subdir": "ablang2",
        "raw_dim": 480,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": "ablang2",
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablang2",
    },
    {
        "representation_id": "AbLang2_PAIRED_NATIVE",
        "representation": "ablang2_unpaired",
        "plm_family": "ablang2",
        "plm_source": "ablang2_unpaired",
        "content_mode": "frozen",
        "subdir": "ablang2_unpaired",
        "raw_dim": 480,
        "representation_context": "PAIRED_NATIVE",
        "paired_match_group": "ablang2",
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablang2_unpaired",
    },
    {
        "representation_id": "ESM1b",
        "representation": "esm1b",
        "plm_family": "esm1b",
        "plm_source": "esm1b",
        "content_mode": "frozen",
        "subdir": "esm1b",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#esm1b",
    },
    {
        "representation_id": "ESM2",
        "representation": "esm2",
        "plm_family": "esm2",
        "plm_source": "esm2",
        "content_mode": "frozen",
        "subdir": "esm2",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#esm2",
    },
    {
        "representation_id": "ESMC600M",
        "representation": "esmc600m",
        "plm_family": "esmc600m",
        "plm_source": "esmc600m",
        "content_mode": "frozen",
        "subdir": "esmc600m",
        "raw_dim": 1152,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#esmc600m",
    },
    {
        "representation_id": "CurrAb_SEPARATE_CHAIN",
        "representation": "currab_unpaired",
        "plm_family": "currab",
        "plm_source": "currab_unpaired",
        "content_mode": "frozen",
        "subdir": "currab_unpaired",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": "currab",
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#currab_unpaired",
    },
    {
        "representation_id": "CurrAb_PAIRED_NATIVE",
        "representation": "currab_paired",
        "plm_family": "currab",
        "plm_source": "currab",
        "content_mode": "frozen",
        "subdir": "currab",
        "raw_dim": 1280,
        "representation_context": "PAIRED_NATIVE",
        "paired_match_group": "currab",
        "asset_ref": "assets/transformer/residue_asset_manifest.yaml#currab",
    },
]

TOPO_ORDER = ["SEP", "JOINT", "REG-SEP", "XREG", "FUSE"]
TOPO_LEGACY = {"SEP": "A", "JOINT": "B1", "REG-SEP": "B2", "XREG": "C", "FUSE": "D"}
ANNOT_ORDER = ["BASE", "IMGT", "REGION", "FULL"]

TOPOLOGY_FLAGS = {
    "SEP": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "JOINT": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": True,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "REG-SEP": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": True,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "XREG": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": True,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "FUSE": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": "d3_symmetric_mlp",
    },
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cells() -> list[dict]:
    cells = []
    n = 140
    idx = 0
    for rep in REP_ORDER:
        for topo in TOPO_ORDER:
            for annot in ANNOT_ORDER:
                idx += 1
                code = f"EXP-H{n:03d}"
                n += 1
                legacy = TOPO_LEGACY[topo]
                flags = TOPOLOGY_FLAGS[topo]
                eid = (
                    f"TRF_HIC_{rep['representation'].upper()}_{annot}_{legacy}_MEAN_V3"
                )
                cells.append(
                    {
                        "cell_index": idx,
                        "experiment_code": code,
                        "experiment_id": eid,
                        "representation_id": rep["representation_id"],
                        "representation": rep["representation"],
                        "representation_family": rep["plm_family"],
                        "plm_source": rep["plm_source"] or "",
                        "content_mode": rep["content_mode"],
                        "subdir": rep["subdir"] or "",
                        "raw_dim": rep["raw_dim"],
                        "inference_context": rep["representation_context"],
                        "representation_context": rep["representation_context"],
                        "paired_match_group": rep["paired_match_group"] or "",
                        "topology": topo,
                        "topology_legacy": legacy,
                        "annotation": annot,
                        "annotation_mode": annot.lower(),
                        "target": "HIC",
                        "platform": "DL_FOLDLOCAL_COSINE_V3",
                        "seed": 101,
                        "d_model": 128,
                        "merge_mode": "mean",
                        "pooling_mode": "REG",
                        "share_hl_encoder": True,
                        "pair_interaction_mode": flags["pair_interaction_mode"],
                        "reuse_candidate": False,
                        "reuse_exp_code": "",
                        "reuse_status": "NONE",
                        "embedding_asset_ref": rep["asset_ref"],
                        "config_path": f"experiments/configs/{code}.yaml",
                        "execution_status": "PLANNED",
                        "surface_features_enabled": False,
                        "public_private_dependency": False,
                    }
                )
    if n != 340:
        raise SystemExit(f"expected end n=340, got {n}")
    if len(cells) != 200:
        raise SystemExit(len(cells))
    if cells[0]["experiment_code"] != "EXP-H140" or cells[-1]["experiment_code"] != "EXP-H339":
        raise SystemExit("bad code range")
    return cells


def write_config(cell: dict) -> None:
    flags = TOPOLOGY_FLAGS[cell["topology"]]
    ps = cell["plm_source"]
    ps_out = (ps or "NONE").upper() if cell["content_mode"] == "frozen" else "NONE"
    if ps_out == "ABLINGUA":
        ps_out = "ABLINGUA"
    cfg = {
        "experiment_code": cell["experiment_code"],
        "experiment_id": cell["experiment_id"],
        "platform_id": "DL_FOLDLOCAL_COSINE_V3",
        "target": "HIC",
        "family": "TRANSFORMER",
        "transformer_type": "SCRATCH" if cell["content_mode"] == "scratch" else "FROZEN_PLM",
        "plm_source": ps_out,
        "annotation_mode": cell["annotation_mode"],
        "chain_mode": "HL",
        "merge_mode": "mean",
        "pooling_mode": "REG",
        "content_mode": cell["content_mode"],
        "seed": 101,
        "d_model": 128,
        "share_hl_encoder": True,
        "raw_plm_hidden_dim": int(cell["raw_dim"]) if cell["content_mode"] == "frozen" else 0,
        "representation": cell["representation"],
        "representation_id": cell["representation_id"],
        "representation_context": cell["representation_context"],
        "topology": cell["topology_legacy"],
        "topology_id": cell["topology"],
        "no_full_dev_refit": True,
        "arch_joint_hl_single_reg": False,
        "arch_joint_hl_dual_reg": flags["joint_hl_dual_reg"],
        "arch_joint_hl_chain_specific_dual_reg": flags["joint_hl_chain_specific_dual_reg"],
        "arch_use_cross_attention_bridge": False,
        "arch_use_reg_only_cross_attention": flags["use_reg_only_cross_attention"],
        "arch_use_within_chain_extra_attention": False,
        "arch_cross_gate_mode": "learned",
        "arch_use_cross_geometry_bias": False,
        "arch_share_hl_encoder": True,
        "pair_interaction_mode": flags["pair_interaction_mode"],
        "reg_cross_variant": None,
        "plm_source_canonical": ps or "",
        "input_asset_ref": cell["embedding_asset_ref"],
        "surface_features_enabled": False,
        "batch_id": "H140_H339_REP_TOPO_ANNOT_FACTORIAL",
    }
    (ROOT / "experiments/configs" / f"{cell['experiment_code']}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def issue_all(cells: list[dict]) -> None:
    if next_code("HIC") != "EXP-H140":
        raise SystemExit(f"expected next HIC code EXP-H140, got {next_code('HIC')}")
    codes = load_codes()
    issued = set(codes["experiment_code"].astype(str))
    for cell in cells:
        code = cell["experiment_code"]
        eid = cell["experiment_id"]
        if code in issued:
            # idempotent if already correct
            row = codes[codes.experiment_code == code].iloc[0]
            if str(row.experiment_id) != eid:
                raise SystemExit(f"{code} already issued as {row.experiment_id}")
            continue
        expect = next_code("HIC")
        if expect != code:
            raise SystemExit(f"reserve mismatch want {code} next={expect}")
        got = issue_code(
            eid,
            "HIC",
            source_model_id="HIC_REP_TOPO_ANNOT_FACTORIAL",
            phase="H140_FACTORIAL_PREREG",
            notes=f"{cell['representation']}/{cell['topology']}/{cell['annotation']}",
        )
        if got != code:
            raise SystemExit(got)
        issued.add(code)
        codes = load_codes()
    if next_code("HIC") != "EXP-H340":
        raise SystemExit(f"after issue expected EXP-H340, got {next_code('HIC')}")


def write_manifest(cells: list[dict]) -> tuple[Path, str]:
    inv = {}
    inv_path = REPORTS / "HIC_FACTORIAL_EMBEDDING_INVENTORY.csv"
    if inv_path.exists():
        with inv_path.open() as f:
            for r in csv.DictReader(f):
                inv[r["representation_id"]] = r
    path = REPORTS / "HIC_REP_TOPO_ANNOT_FACTORIAL_CELL_MANIFEST.csv"
    rows = []
    for c in cells:
        er = inv.get(c["representation_id"], {})
        rows.append(
            {
                "cell_index": c["cell_index"],
                "experiment_code": c["experiment_code"],
                "experiment_id": c["experiment_id"],
                "representation_id": c["representation_id"],
                "representation": c["representation"],
                "representation_family": c["representation_family"],
                "inference_context": c["inference_context"],
                "topology": c["topology"],
                "topology_legacy": c["topology_legacy"],
                "annotation": c["annotation"],
                "annotation_mode": c["annotation_mode"],
                "target": "HIC",
                "platform": "DL_FOLDLOCAL_COSINE_V3",
                "seed": 101,
                "d_model": 128,
                "merge_mode": "mean",
                "pooling_mode": "REG",
                "share_hl_encoder": True,
                "pair_interaction_mode": c["pair_interaction_mode"],
                "reuse_candidate": False,
                "reuse_exp_code": "",
                "reuse_status": "NONE",
                "embedding_asset": er.get("path", c["embedding_asset_ref"]),
                "embedding_hash": er.get("embedding_hash", ""),
                "embedding_asset_ref": c["embedding_asset_ref"],
                "config_path": c["config_path"],
                "config_status": "ISSUED",
                "planned_execution_status": "PLANNED",
                "surface_features_enabled": False,
                "public_private_dependency": False,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path, sha256_file(path)


def write_plan_artifacts(cells: list[dict], manifest_sha: str) -> None:
    plan_csv = ROOT / "results/HIC_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv"
    fields = [
        "cell_index",
        "experiment_code",
        "experiment_id",
        "representation",
        "representation_id",
        "plm_family",
        "plm_source",
        "content_mode",
        "subdir",
        "representation_context",
        "paired_match_group",
        "topology",
        "topology_legacy",
        "annotation",
        "annotation_mode",
        "raw_dim",
        "execution_status",
        "config_path",
        "notes",
    ]
    with plan_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow(
                {
                    "cell_index": c["cell_index"],
                    "experiment_code": c["experiment_code"],
                    "experiment_id": c["experiment_id"],
                    "representation": c["representation"],
                    "representation_id": c["representation_id"],
                    "plm_family": c["representation_family"],
                    "plm_source": c["plm_source"],
                    "content_mode": c["content_mode"],
                    "subdir": c["subdir"],
                    "representation_context": c["representation_context"],
                    "paired_match_group": c["paired_match_group"],
                    "topology": c["topology"],
                    "topology_legacy": c["topology_legacy"],
                    "annotation": c["annotation"],
                    "annotation_mode": c["annotation_mode"],
                    "raw_dim": c["raw_dim"],
                    "execution_status": "PLANNED",
                    "config_path": c["config_path"],
                    "notes": "",
                }
            )

    doc = {
        "batch_id": "H140_H339_REP_TOPO_ANNOT_FACTORIAL",
        "status": "PREREGISTERED_FROZEN",
        "git_rev_at_prereg": git_rev(),
        "evaluation_freeze_sha": EVAL_FREEZE_SHA,
        "pre_freeze_manifest_sha256": PRE_FREEZE_MANIFEST_SHA,
        "formal_manifest_sha256": manifest_sha,
        "platform_id": "DL_FOLDLOCAL_COSINE_V3",
        "seed": 101,
        "d_model": 128,
        "merge_mode": "mean",
        "n_cells": 200,
        "n_reuse": 0,
        "n_planned": 200,
        "id_range_new": "EXP-H140..EXP-H339",
        "share_hl_encoder": True,
        "surface_excluded": True,
        "topology_flags": TOPOLOGY_FLAGS,
        "reps": REP_ORDER,
        "principles": [
            "all 200 cells defined before factorial results inspected",
            "no PLM/topology/annotation dropped based on intermediate scores",
            "Public/Private/external results will not drive training or design",
            "no surface/SASA/HSP/physics aux in this batch",
            "TmApp results do not alter HIC cell selection",
            "share_hl_encoder=True explicit on every cell",
        ],
        "cells": cells,
    }
    (ROOT / "results/HIC_REP_TOPO_ANNOT_FACTORIAL_PLAN.yaml").write_text(
        yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"
    )


def write_prereg_md(manifest_sha: str) -> None:
    md = f"""# HIC Representation × Topology × Annotation Factorial — PREREGISTRATION

**STATUS: PREREGISTERED_FROZEN**

| Field | Value |
|-------|--------|
| Evaluation freeze SHA | `{EVAL_FREEZE_SHA}` |
| Git at prereg issuance | `{git_rev()}` |
| Formal manifest | `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_CELL_MANIFEST.csv` |
| **Formal manifest SHA256** | `{manifest_sha}` |
| Pre-freeze manifest SHA256 | `{PRE_FREEZE_MANIFEST_SHA}` |
| Plan | `developability_drilldown/results/HIC_REP_TOPO_ANNOT_FACTORIAL_PLAN.yaml` |
| ID range | **EXP-H140 … EXP-H339** |
| Cells | **200 new** / **0 REUSE** |
| Platform | `DL_FOLDLOCAL_COSINE_V3` / seed `101` / `d_model=128` / mean / REG |
| Surface / physics | **Excluded** |

## Binding commitments

- All 200 cells were defined **before** new factorial HIC results are inspected.
- No PLM, topology, or annotation will be dropped based on intermediate scores.
- Public/Private/external metrics will **not** drive training, HP, representation, topology, annotation, or cell selection.
- No surface / SASA / HSP / physics auxiliary features in this batch.
- TmApp factorial results do **not** alter HIC selection.
- Every config has **`share_hl_encoder=True`** explicitly; topology flags including `pair_interaction_mode` are explicit.
- Scientifically weak MAE is **not** a technical failure; keep the cell.
- Technical failures: repair and re-run the **same** cell. Asset-impossible: **BLOCKED** (no representation substitution).

## Factors

- Representation (10): Scratch; AbLingua; AbLang1; AbLang2 SEPARATE_CHAIN; AbLang2 PAIRED_NATIVE; ESM-1b; ESM-2; ESM-C 600M; CurrAb SEPARATE_CHAIN; CurrAb PAIRED_NATIVE
- Topology (5): SEP / JOINT / REG-SEP / XREG / FUSE (TmApp semantics)
- Annotation (4): BASE / IMGT / REGION / FULL

## Evaluation contract

- **Primary:** `TEST_mean = mean(TEST_P, TEST_S)`
- **Secondary:** `Δ_P` / `Δ_S` sign replication vs baseline; improvement requires both `< 0`
- **Contrasts:** Topology−SEP; Annotation−BASE; PLM−Scratch; PAIRED_NATIVE−SEPARATE_CHAIN (AbLang2, CurrAb)
- **Bootstrap:** antibody-level paired residual; N_BOOT=2000; seed=101; 95% percentile CI; Primary/Shadow separate; no fold-as-iid; no Pub/Priv mix
- **HIGH-tail:** `HIC > 11.5` diagnostic only
- **External:** GEN_0001 after internal freeze only

## Hierarchy note

Main grid uses **10 representation levels**. AbLang2/CurrAb additionally support family×context analysis without collapsing the grid.

## Authoritative files

- This document replaces `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_PREREG_DRAFT.md` as the formal prereg.
- Runner: `developability_drilldown/scripts/run_exp_h140_h339_factorial.py` (full batch requires separate human approval)
"""
    path = REPORTS / "HIC_REP_TOPO_ANNOT_FACTORIAL_PREREG.md"
    path.write_text(md, encoding="utf-8")
    draft = REPORTS / "HIC_REP_TOPO_ANNOT_FACTORIAL_PREREG_DRAFT.md"
    if draft.exists():
        draft.write_text(
            "# SUPERSEDED\n\nAuthoritative formal preregistration:\n"
            "`reports/HIC_REP_TOPO_ANNOT_FACTORIAL_PREREG.md`\n\n"
            f"**STATUS of formal doc: PREREGISTERED_FROZEN** (manifest `{manifest_sha}`)\n",
            encoding="utf-8",
        )


def _input_space(cell: dict) -> str:
    topo = cell["topology_legacy"]
    rep = cell["representation"]
    topo_prefix = {
        "A": "SEPARATE_DUAL_REG",
        "B1": "JOINT_HL_DUAL_REG",
        "B2": "JOINT_HL_CHAIN_SPECIFIC_DUAL_REG",
        "C": "SEPARATE_REG_ONLY_CROSS_ATTENTION",
        "D": "PAIR_D3",
    }[topo]
    if rep == "scratch":
        return "PAIR_D3_SCRATCH_MEAN_V3" if topo == "D" else f"{topo_prefix}_SCRATCH_RESIDUE_MEAN_V3"
    if topo == "D":
        return f"PAIR_D3_FROZEN_{rep.upper()}_MEAN_V3"
    return f"{topo_prefix}_FROZEN_{rep.upper()}_RESIDUE_MEAN_V3"


def register_stubs(cells: list[dict]) -> None:
    """Register PARTIAL / PREREGISTERED rows (configs only; no predictions yet)."""
    exp_path = ROOT / "results/experiments.csv"
    df = pd.read_csv(exp_path)
    existing = set(df["experiment_code"].astype(str))
    new_rows = []
    for c in cells:
        code = c["experiment_code"]
        if code in existing:
            continue
        rec = {col: "" for col in df.columns}
        rec.update(
            {
                "experiment_code": code,
                "experiment_id": c["experiment_id"],
                "target": "HIC",
                "family": "TRANSFORMER",
                "model_type": "TRANSFORMER",
                "source_model_id": "HIC_REP_TOPO_ANNOT_FACTORIAL",
                "cv_protocol": "dl_foldlocal_cosine_v3_oof_test",
                "selection_policy_at_creation": "POSTCOMP_EXPLORATORY",
                "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
                "artifact_status": "PARTIAL",
                "source_reproducible": "YES",
                "drilldown_reproducible": "YES",
                "reproduction_status": "PREREGISTERED",
                "config_path": f"experiments/configs/{code}.yaml",
                "license_status": "OK",
                "notes": f"prereg {c['representation']}/{c['topology']}/{c['annotation']}",
                "transformer_type": "SCRATCH" if c["content_mode"] == "scratch" else "FROZEN_PLM",
                "plm_source": (c["plm_source"] or "NONE").upper() if c["content_mode"] == "frozen" else "NONE",
                "annotation_mode": c["annotation_mode"],
                "chain_mode": "HL",
                "pooling_mode": "REG",
                "representation_status": "NOT_EXPORTED",
                "input_space": _input_space(c),
                "input_asset_ref": c["embedding_asset_ref"],
                "shareability_status": "SHAREABLE_PARTIAL",
                "canonical_benchmark_eligible": "NO",
            }
        )
        new_rows.append(rec)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(exp_path, index=False)

    comp_path = ROOT / "results/EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    comp = pd.read_csv(comp_path)
    cexist = set(comp["experiment_code"].astype(str))
    crows = []
    for c in cells:
        code = c["experiment_code"]
        if code in cexist:
            continue
        crow = {col: "" for col in comp.columns}
        crow.update(
            {
                "experiment_code": code,
                "experiment_id": c["experiment_id"],
                "target": "HIC",
                "family": "TRANSFORMER",
                "config_exists": True,
                "feature_required": False,
                "feature_exists": True,
                "oof_primary_exists": False,
                "oof_shadow_exists": False,
                "test_prediction_exists": False,
                "reproduction_status": "PREREGISTERED",
                "shareability_status": "SHAREABLE_PARTIAL",
                "canonical_benchmark_eligible": "NO",
                "notes": "H140–H339 factorial prereg; awaiting training",
            }
        )
        crows.append(crow)
    if crows:
        comp = pd.concat([comp, pd.DataFrame(crows)], ignore_index=True)
        comp.to_csv(comp_path, index=False)


def validate(cells: list[dict], manifest_sha: str) -> Path:
    codes = [c["experiment_code"] for c in cells]
    assert codes == [f"EXP-H{i:03d}" for i in range(140, 340)]
    assert len(set(codes)) == 200
    combos = {(c["representation"], c["topology"], c["annotation"]) for c in cells}
    assert len(combos) == 200
    assert len({c["representation"] for c in cells}) == 10
    assert len({c["topology"] for c in cells}) == 5
    assert len({c["annotation"] for c in cells}) == 4

    cfg_dir = ROOT / "experiments/configs"
    for c in cells:
        y = yaml.safe_load((cfg_dir / f"{c['experiment_code']}.yaml").read_text())
        assert y["target"] == "HIC"
        assert y["share_hl_encoder"] is True
        assert y["arch_share_hl_encoder"] is True
        assert y["seed"] == 101
        assert y["d_model"] == 128
        assert y["merge_mode"] == "mean"
        assert y["surface_features_enabled"] is False
        flags = TOPOLOGY_FLAGS[c["topology"]]
        assert y["arch_joint_hl_dual_reg"] == flags["joint_hl_dual_reg"]
        assert y["arch_joint_hl_chain_specific_dual_reg"] == flags["joint_hl_chain_specific_dual_reg"]
        assert y["arch_use_reg_only_cross_attention"] == flags["use_reg_only_cross_attention"]
        assert y["pair_interaction_mode"] == flags["pair_interaction_mode"]
        if c["content_mode"] == "frozen":
            assert "residue_asset_manifest.yaml#" in y["input_asset_ref"]
        else:
            assert "scratch" in y["input_asset_ref"]

    reg = load_codes()
    h = [x for x in reg.experiment_code.astype(str) if x.startswith("EXP-H")]
    assert sorted(h, key=lambda x: int(x.split("-H")[1])) == [f"EXP-H{i:03d}" for i in range(1, 340)]
    assert next_code("HIC") == "EXP-H340"

    out = ROOT / "results/HIC_H140_H339_PREREG_VALIDATION.json"
    out.write_text(
        json.dumps(
            {
                "status": "PASS",
                "n_cells": 200,
                "n_configs": 200,
                "id_range": "EXP-H140..EXP-H339",
                "formal_manifest_sha256": manifest_sha,
                "pre_freeze_manifest_sha256": PRE_FREEZE_MANIFEST_SHA,
                "evaluation_freeze_sha": EVAL_FREEZE_SHA,
                "n_reuse": 0,
                "next_free_hic": "EXP-H340",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    cells = build_cells()
    issue_all(cells)
    for c in cells:
        write_config(c)
    register_stubs(cells)
    man_path, man_sha = write_manifest(cells)
    write_plan_artifacts(cells, man_sha)
    write_prereg_md(man_sha)
    val = validate(cells, man_sha)
    print("PREREG_OK", man_path, man_sha, val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
