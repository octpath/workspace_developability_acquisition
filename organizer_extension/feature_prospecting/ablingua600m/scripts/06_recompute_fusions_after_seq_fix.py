#!/usr/bin/env python3
"""Re-evaluate original AbLingua recipe fusions after SEQ_BASIC id-alignment fix."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/ablingua600m"
INTERIM = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"

spec = importlib.util.spec_from_file_location(
    "tvt", INTERIM / "scripts/run_simple_tvt_rescreen.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import runpy

ev = runpy.run_path(str(OUT / "scripts/02_simple_tvt_eval.py"))

ids = [str(x) for x in pd.read_csv(ROOT / "competition/data/distribution/dev.csv")["id"]]
y = mod.load_y("TmApp")
primary = pd.read_csv(PRIMARY)
shadow = pd.read_csv(SHADOW)
primary["id"] = primary.id.astype(str)
shadow["id"] = shadow.id.astype(str)

ab = ev["load_ablang2"]()
seq = ev["load_seq_basic"](ids)
assert seq.isna().sum().sum() == 0
X_pair, m1 = ev["load_pair_m1"]()
HL = ev["load_parquet_X"](OUT / "embeddings/ablingua600m_HL_mean_concat.parquet")
ablang = ab["AbLang2_HL_paired"]
# Use authoritative ablang2_* naming to match load_bases
ablang = ablang.copy()
ablang.columns = [f"ablang2_{i}" for i in range(ablang.shape[1])]
base_ablang_seq = pd.concat([ablang, seq], axis=1, join="inner")
recipe = pd.concat([base_ablang_seq, X_pair, m1], axis=1, join="inner")
recipe = recipe.loc[:, ~recipe.columns.duplicated()]
base_abl_seq = pd.concat([HL.reindex(ids), seq], axis=1, join="inner")
recipe_replace = pd.concat([base_abl_seq, X_pair, m1], axis=1, join="inner")
recipe_replace = recipe_replace.loc[:, ~recipe_replace.columns.duplicated()]

rows = []


def add(rep, fusion, mode, out, parent_note=""):
    rows.append(
        {
            "representation": rep,
            "pooling": "MASKED_MEAN" if "AbLingua" in rep else "AbLang2_default",
            "chains": "HL",
            "fusion_recipe": fusion,
            "downstream_model": "Ridge",
            "dimensionality_mode": "PCA32_struct" if "AbLingua" in fusion or fusion.startswith("ADD") or "AbLang2+AbLingua" in fusion or fusion == "SEQ_BASIC+AbLang2+AbLingua" else ("struct_noPCA" if "REPLACE" in fusion else "PCA32_struct"),
            "primary_mae": None,
            "shadow_mae": None,
            "primary_delta_vs_relevant_baseline": out.get("dP"),
            "shadow_delta_vs_relevant_baseline": out.get("dS"),
            "fixed_alpha_primary_delta": out.get("fP"),
            "fixed_alpha_shadow_delta": out.get("fS"),
            "n_dev": 162,
            "notes": f"RECOMPUTED_AFTER_SEQ_FIX mode={mode}; {parent_note}",
            "_P": out["P"],
            "_S": out["S"],
        }
    )


def eval_incr(base, struct, force_pca):
    free_p = ev["run_incr"](base, struct, y, primary, ids, "FREE_ALPHA", force_pca)
    free_s = ev["run_incr"](base, struct, y, shadow, ids, "FREE_ALPHA", force_pca)
    fix_p = ev["run_incr"](base, struct, y, primary, ids, "BASE_FIXED_ALPHA", force_pca)
    fix_s = ev["run_incr"](base, struct, y, shadow, ids, "BASE_FIXED_ALPHA", force_pca)
    return {
        "FREE": {
            "P": free_p["plus_mae"],
            "S": free_s["plus_mae"],
            "dP": free_p["delta"],
            "dS": free_s["delta"],
            "baseP": free_p["base_mae"],
            "baseS": free_s["base_mae"],
        },
        "FIXED": {
            "P": fix_p["plus_mae"],
            "S": fix_s["plus_mae"],
            "dP": fix_p["delta"],
            "dS": fix_s["delta"],
            "fP": fix_p["delta"],
            "fS": fix_s["delta"],
            "baseP": fix_p["base_mae"],
            "baseS": fix_s["base_mae"],
        },
    }


# ADD
add_res = eval_incr(recipe, HL, True)
print("ADD FREE", add_res["FREE"])
print("ADD FIXED", add_res["FIXED"])

# AbLang2 + AbLingua on SEQ
parent_base = pd.concat([seq, ablang], axis=1, join="inner")
fuse_res = eval_incr(parent_base, HL, True)
print("A2+ABL FREE", fuse_res["FREE"])

# REPLACE absolute comparison
struct_extra = pd.concat([X_pair, m1], axis=1, join="inner")
rep_parent = eval_incr(base_ablang_seq, struct_extra, False)
rep_abl = eval_incr(base_abl_seq, struct_extra, False)
print("REPLACE parent", rep_parent["FREE"]["P"], rep_parent["FREE"]["S"])
print("REPLACE abl", rep_abl["FREE"]["P"], rep_abl["FREE"]["S"])

# SEQ+AbLingua incr
seq_abl = eval_incr(seq, HL, True)
print("SEQ+ABL", seq_abl["FREE"])

out = {
    "AUTHORITATIVE_PARENT_MAE": {
        "Primary": add_res["FREE"]["P"],
        "Shadow": add_res["FREE"]["S"],
    },
    "recipe_only_MAE": {
        "Primary": add_res["FREE"]["baseP"],
        "Shadow": add_res["FREE"]["baseS"],
    },
    "ADD_delta_FREE": {"P": add_res["FREE"]["dP"], "S": add_res["FREE"]["dS"]},
    "ADD_delta_FIXED": {"P": add_res["FIXED"]["dP"], "S": add_res["FIXED"]["dS"]},
    "SEQ_AbLang2_AbLingua_delta_FREE": {
        "P": fuse_res["FREE"]["dP"],
        "S": fuse_res["FREE"]["dS"],
        "plus_P": fuse_res["FREE"]["P"],
        "plus_S": fuse_res["FREE"]["S"],
    },
    "REPLACE_delta_vs_parent_recipe": {
        "P": rep_parent["FREE"]["P"] - rep_abl["FREE"]["P"],
        "S": rep_parent["FREE"]["S"] - rep_abl["FREE"]["S"],
        "parent_P": rep_parent["FREE"]["P"],
        "parent_S": rep_parent["FREE"]["S"],
        "repl_P": rep_abl["FREE"]["P"],
        "repl_S": rep_abl["FREE"]["S"],
    },
    "SEQ_AbLingua_incr": seq_abl["FREE"],
}
(OUT / "results/parent_reconciliation/ORIGINAL_FUSION_RECOMPUTED.json").write_text(
    json.dumps(out, indent=2)
)
print(json.dumps(out, indent=2))
