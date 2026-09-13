#!/usr/bin/env python3
"""Generate T161–T337 Representation × Topology × Annotation factorial preregistration.

200 cells = 10 representations × 5 topologies × 4 annotations.
23 REUSE + 177 PLANNED (EXP-T161..EXP-T337).
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

ANNOT_ORDER = ["FULL", "BASE", "IMGT", "REGION"]
TOPO_ORDER = ["A", "B1", "B2", "C", "D"]

TOPOLOGY_FLAGS = {
    "A": {
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
    "B1": {
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
    "B2": {
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
    "C": {
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
    "D": {
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

REPS = [
    {
        "representation": "scratch",
        "plm_family": "scratch",
        "plm_source": None,
        "content_mode": "scratch",
        "subdir": None,
        "raw_dim": 0,
        "representation_context": "LEARNED_AA",
        "paired_match_group": None,
    },
    {
        "representation": "ablingua",
        "plm_family": "ablingua",
        "plm_source": "ablingua",
        "content_mode": "frozen",
        "subdir": "ablingua600m",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
    },
    {
        "representation": "ablang2_paired",
        "plm_family": "ablang2",
        "plm_source": "ablang2",
        "content_mode": "frozen",
        "subdir": "ablang2",
        "raw_dim": 480,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": "ablang2",
        "notes": (
            "ASSET AUDIT: historical ablang2 bundle uses SEPARATE_CHAIN formats "
            "(<H>| / |<L>) with ablang2-paired checkpoint; name retains historical label."
        ),
    },
    {
        "representation": "ablang2_unpaired",
        "plm_family": "ablang2",
        "plm_source": "ablang2_unpaired",
        "content_mode": "frozen",
        "subdir": "ablang2_unpaired",
        "raw_dim": 480,
        "representation_context": "PAIRED_NATIVE",
        "paired_match_group": "ablang2",
        "notes": (
            "Allocation name ablang2_unpaired holds JOINT paired inference (<H>|<L>) "
            "with the SAME ablang2-paired checkpoint, complementary to historical SEPARATE "
            "ablang2_paired asset, enabling matched inference-context contrast."
        ),
    },
    {
        "representation": "ablang1",
        "plm_family": "ablang1",
        "plm_source": "ablang1",
        "content_mode": "frozen",
        "subdir": "ablang1",
        "raw_dim": 768,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
    },
    {
        "representation": "esm1b",
        "plm_family": "esm1b",
        "plm_source": "esm1b",
        "content_mode": "frozen",
        "subdir": "esm1b",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
    },
    {
        "representation": "esm2",
        "plm_family": "esm2",
        "plm_source": "esm2",
        "content_mode": "frozen",
        "subdir": "esm2",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
    },
    {
        "representation": "esmc600m",
        "plm_family": "esmc600m",
        "plm_source": "esmc600m",
        "content_mode": "frozen",
        "subdir": "esmc600m",
        "raw_dim": 1152,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": None,
    },
    {
        "representation": "currab_paired",
        "plm_family": "currab",
        "plm_source": "currab",
        "content_mode": "frozen",
        "subdir": "currab",
        "raw_dim": 1280,
        "representation_context": "PAIRED_NATIVE",
        "paired_match_group": "currab",
    },
    {
        "representation": "currab_unpaired",
        "plm_family": "currab",
        "plm_source": "currab_unpaired",
        "content_mode": "frozen",
        "subdir": "currab_unpaired",
        "raw_dim": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "paired_match_group": "currab",
    },
]

# (representation, topology, annotation) -> reuse code
REUSE = {
    ("scratch", "A", "FULL"): "EXP-T091",
    ("scratch", "B1", "FULL"): "EXP-T094",
    ("scratch", "B2", "FULL"): "EXP-T096",
    ("scratch", "C", "FULL"): "EXP-T102",
    ("ablingua", "A", "FULL"): "EXP-T080",
    ("ablingua", "B1", "FULL"): "EXP-T081",
    ("ablingua", "B2", "FULL"): "EXP-T082",
    ("ablingua", "C", "FULL"): "EXP-T087",
    ("ablingua", "D", "FULL"): "EXP-T151",
    ("ablang2_paired", "A", "FULL"): "EXP-T110",
    ("ablang2_paired", "B1", "FULL"): "EXP-T113",
    ("ablang2_paired", "B2", "FULL"): "EXP-T115",
    ("ablang2_paired", "C", "FULL"): "EXP-T121",
    ("ablang2_paired", "D", "FULL"): "EXP-T149",
    ("esm2", "A", "FULL"): "EXP-T152",
    ("esm2", "B1", "FULL"): "EXP-T153",
    ("esm2", "B2", "FULL"): "EXP-T154",
    ("esm2", "C", "FULL"): "EXP-T155",
    ("esm2", "D", "FULL"): "EXP-T156",
    ("ablang1", "C", "FULL"): "EXP-T157",
    ("esm1b", "C", "FULL"): "EXP-T158",
    ("esmc600m", "C", "FULL"): "EXP-T159",
    ("currab_paired", "C", "FULL"): "EXP-T160",
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def build_cells() -> list[dict]:
    cells = []
    next_id = 161
    # Deterministic order: representation order in REPS, then ANNOT_ORDER, then TOPO_ORDER
    # but skip REUSE for ID assignment; REUSE cells still appear in plan
    for rep in REPS:
        rname = rep["representation"]
        for annot in ANNOT_ORDER:
            for topo in TOPO_ORDER:
                key = (rname, topo, annot)
                if key in REUSE:
                    code = REUSE[key]
                    status = "REUSE"
                    reuse = code
                else:
                    code = f"EXP-T{next_id:03d}"
                    next_id += 1
                    status = "PLANNED"
                    reuse = ""
                cells.append(
                    {
                        "representation": rname,
                        "plm_family": rep["plm_family"],
                        "plm_source": rep["plm_source"] or "",
                        "content_mode": rep["content_mode"],
                        "subdir": rep["subdir"] or "",
                        "representation_context": rep["representation_context"],
                        "paired_match_group": rep["paired_match_group"] or "",
                        "topology": topo,
                        "annotation": annot,
                        "annotation_mode": annot.lower(),
                        "experiment_code": code,
                        "execution_status": status,
                        "reuse_source": reuse,
                        "raw_dim": rep["raw_dim"],
                        "config_path": f"experiments/configs/{code}.yaml" if status == "PLANNED" else "",
                        "primary_mae": "",
                        "shadow_mae": "",
                        "mean_ps": "",
                        "worst_ps": "",
                        "notes": rep.get("notes", ""),
                    }
                )
    if next_id != 338:
        raise SystemExit(f"expected next_id=338 after allocation, got {next_id}")
    if len(cells) != 200:
        raise SystemExit(len(cells))
    n_reuse = sum(1 for c in cells if c["execution_status"] == "REUSE")
    n_new = sum(1 for c in cells if c["execution_status"] == "PLANNED")
    if n_reuse != 23 or n_new != 177:
        raise SystemExit(f"reuse={n_reuse} new={n_new}")
    # verify T210 / T321
    t210 = next(c for c in cells if c["experiment_code"] == "EXP-T210")
    t321 = next(c for c in cells if c["experiment_code"] == "EXP-T321")
    assert t210["representation"] == "ablang2_unpaired" and t210["topology"] == "C" and t210["annotation"] == "FULL"
    assert t321["representation"] == "currab_unpaired" and t321["topology"] == "C" and t321["annotation"] == "FULL"
    return cells


def main() -> int:
    from experiment_codes import next_code

    if next_code("TmApp") != "EXP-T161":
        raise SystemExit(f"expected EXP-T161 free, got {next_code('TmApp')}")
    cells = build_cells()
    plan_csv = ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv"
    plan_yaml = ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.yaml"
    plan_md = ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_PREREG.md"

    fields = list(cells[0].keys())
    with plan_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cells)

    doc = {
        "batch_id": "T161_T337_REP_TOPO_ANNOT_FACTORIAL",
        "status": "PREREGISTERED_FROZEN",
        "git_rev_at_prereg": git_rev(),
        "platform_id": "DL_FOLDLOCAL_COSINE_V3",
        "seed": 101,
        "n_cells": 200,
        "n_reuse": 23,
        "n_planned": 177,
        "id_range_new": "EXP-T161..EXP-T337",
        "pre_bulk_gate": ["EXP-T210", "EXP-T321"],
        "topology_flags": TOPOLOGY_FLAGS,
        "representations": REPS,
        "principles": [
            "all 200 cells defined before new factorial results inspected",
            "no PLM/topology/annotation dropped based on intermediate scores",
            "Public/Private/external results will not drive training or design",
            "no new topology/annotation microvariant during this batch",
        ],
        "ablang2_asset_audit": {
            "historical_ablang2_bundle": "SEPARATE_CHAIN with ablang2-paired checkpoint",
            "ablang2_unpaired_asset": "JOINT PAIRED_NATIVE complementary contrast",
        },
        "cells": cells,
    }
    plan_yaml.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    md = f"""# TmApp Representation × Topology × Annotation Factorial — PREREGISTRATION

**Status:** PREREGISTERED_FROZEN  
**Git at prereg:** `{git_rev()}`  
**Platform:** `DL_FOLDLOCAL_COSINE_V3` / seed `101`  
**Cells:** 200 (23 REUSE + 177 new `EXP-T161`–`EXP-T337`)

## Binding commitments

- All 200 cells were defined **before** new factorial results are inspected.
- No PLM, topology, or annotation will be dropped based on intermediate scores.
- Public/Private/external metrics will **not** drive training or design changes.
- No new topology/annotation/PLM microvariants during this batch.

## Pre-bulk gate

1. `EXP-T210` — ablang2_unpaired / C / FULL  
2. `EXP-T321` — currab_unpaired / C / FULL  

PASS = extraction QC + train/eval completes + nonzero prediction variance + artifacts.  
MAE quality is **not** a gate.

## AbLang2 asset audit

The historical `residue_level/ablang2/` bundle (factorial label `ablang2_paired`) uses
**SEPARATE_CHAIN** inference formats with the `ablang2-paired` checkpoint.

The factorial allocation name `ablang2_unpaired` stores the complementary **JOINT**
paired inference (`<H>|<L>`) with the **same** checkpoint (`representation_context=PAIRED_NATIVE`)
so matched inference-context contrasts remain scientifically available while preserving
historical REUSE cells that consumed the SEPARATE bundle.

## Files

- `TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv`
- `TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.yaml`
"""
    plan_md.write_text(md, encoding="utf-8")
    print("Wrote", plan_csv, plan_yaml, plan_md)
    print("n_cells", len(cells), "next_after_alloc EXP-T338")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
