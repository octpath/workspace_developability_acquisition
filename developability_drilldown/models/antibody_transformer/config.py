#!/usr/bin/env python3
"""Antibody Transformer constants (behavior-preserving extract from advanced_models)."""
from __future__ import annotations

import json
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
DRILLDOWN_ROOT = MODULE_ROOT.parents[1]
REPO_ROOT = DRILLDOWN_ROOT.parent
BUNDLE_ROOT = REPO_ROOT / "top_models_feature_bundle"
RESIDUE_ROOT = BUNDLE_ROOT / "residue_level"
OUTPUT_ROOT = BUNDLE_ROOT / "advanced_outputs"
PRESETS_PATH = MODULE_ROOT / "presets.json"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {a: i + 1 for i, a in enumerate(AA_LIST)}  # 0 = pad
AA_TO_IDX["X"] = len(AA_LIST) + 1  # unknown
PAD_AA = 0
UNK_AA = AA_TO_IDX["X"]

CHAIN_TO_IDX = {"H": 0, "L": 1}
REGION_ORDER = ["PAD", "FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4", "UNKNOWN"]
REGION_TO_IDX = {r: i for i, r in enumerate(REGION_ORDER)}

TM_RECIPES = [
    "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
    "TM_BASE_BIOEMU_MPNN__RIDGE",
]
HIC_RECIPES = [
    "HIC_HYDRO_TITRATION__LASSO",
    "HIC_ARO_CONTINUOUS_SURFACE__LASSO",
    "HIC_ESM2_SEQ_AROMATIC__LASSO",
]

RECIPE_TO_FEATURE_SET = {
    "TM_PARENT_ABLINGUA_CDR3__RIDGE": "FS_TM_ABLINGUA_CDR3",
    "TM_PARENT_ABLINGUA_GLOBAL__RIDGE": "FS_TM_ABLINGUA_GLOBAL",
    "TM_BASE_BIOEMU_MPNN__RIDGE": "FS_TM_BIOEMU_MPNN",
    "HIC_HYDRO_TITRATION__LASSO": "FS_HIC_HYDRO_TITRATION",
    "HIC_ARO_CONTINUOUS_SURFACE__LASSO": "FS_HIC_CONTINUOUS_SURFACE",
    "HIC_ESM2_SEQ_AROMATIC__LASSO": "FS_HIC_ESM2_SEQ_AROMATIC",
}


def load_presets() -> dict:
    return json.loads(PRESETS_PATH.read_text())


def recipe_ids_for_target(target: str) -> list[str]:
    return list(TM_RECIPES if target == "TmApp" else HIC_RECIPES)
