#!/usr/bin/env python3
"""Simple TVT for thermal microprobe (exact frozen protocol). Labels loaded HERE only."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "fennix_thermal_microprobe"
RES = OUT / "results"
INTERIM = FP / "fennix_fab_context_interim_audit"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
BIO = FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
M1 = FP / "structure_marathon/proteinmpnn/M1_FEATURES.csv"

spec = importlib.util.spec_from_file_location(
    "tvt_rescreen", INTERIM / "scripts/run_simple_tvt_rescreen.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["tvt_rescreen"] = mod
spec.loader.exec_module(mod)


def run_one(base, struct, y, folds, ids, mode):
    common = sorted(
        set(base.index) & set(struct.index) & set(y.index) & set(folds.id.astype(str)) & set(ids)
    )
    cov_ok = True
    for k, tr, va, te in mod.rotation_splits(folds, common):
        ok = len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE
        cov_ok = cov_ok and ok
    if not cov_ok or len(common) < 40:
        return None
    Xb = base.loc[common]
    Xs = struct.loc[common]
    overlap = [c for c in Xs.columns if c in Xb.columns]
    if overlap:
        Xs = Xs.drop(columns=overlap)
    yy = y.loc[common]
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    for k, tr, va, te in mod.rotation_splits(folds, common):
        a_base = mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=False)
        a_plus = (
            a_base
            if mode != "FREE_ALPHA"
            else mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=True)
        )
        if mode != "FREE_ALPHA":
            a_plus = a_base
        pb = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_base, use_struct=False)
        pp = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_plus, use_struct=True)
        base_oof.loc[te] = pb
        plus_oof.loc[te] = pp
    return {
        "N": len(common),
        "struct_dim": int(Xs.shape[1]),
        "base_mae": mod.mae(yy, base_oof),
        "plus_mae": mod.mae(yy, plus_oof),
        "delta": mod.mae(yy, base_oof) - mod.mae(yy, plus_oof),
    }


def verdict(dp, ds):
    if dp > 0 and ds > 0:
        return "COMP_TRY" if min(dp, ds) < 0.02 else "COMP_PRIORITY"
    if dp > 0 or ds > 0:
        return "COMP_MIXED"
    return "COMP_DROP"


def main():
    feat_path = RES / "FENNIX_THERMAL_MICROPROBE_DEV.parquet"
    if not feat_path.exists():
        raise SystemExit("missing DEV feature parquet")
    feat = pd.read_parquet(feat_path).set_index("id")
    freeze = json.loads((OUT / "THERMAL_MICROPROBE_FEATURE_FREEZE.json").read_text())
    cols = freeze["feature_names"]
    X_all = feat[cols].astype(float)
    # HIC-oriented subset
    hic_cols = [c for c in cols if "sasa" in c.lower() or "hydrophobic" in c.lower() or "aromatic" in c.lower()]
    X_hic = X_all[hic_cols]

    bases, _ = mod.load_bases()
    y_tm = mod.load_y("TmApp")
    y_hic = mod.load_y("HIC")
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    usable = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]

    bio = pd.read_csv(BIO)
    pair_cols = [c for c in bio.columns if "ca_rmsd" in c.lower()]
    X_pair = mod.numeric_X(bio[["id"] + pair_cols] if "id" in bio.columns else bio)
    # numeric_X may need id index
    if "id" in bio.columns:
        bp = bio.set_index("id")
        X_pair = bp[pair_cols].apply(pd.to_numeric, errors="coerce")
    m1 = pd.read_csv(M1)
    X_m1 = mod.numeric_X(m1, drop_cols=["status", "extraction_status"])

    rows = []

    def eval_block(name, base, struct, y, target):
        for fold_name, folds in [("Primary", primary), ("Shadow", shadow)]:
            for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
                out = run_one(base, struct, y, folds, usable, mode)
                if out is None:
                    rows.append(
                        {
                            "target": target,
                            "comparison": name,
                            "fold": fold_name,
                            "mode": mode,
                            "status": "SKIP",
                        }
                    )
                    continue
                rows.append(
                    {
                        "target": target,
                        "comparison": name,
                        "fold": fold_name,
                        "mode": mode,
                        "N": out["N"],
                        "struct_dim": out["struct_dim"],
                        "base_mae": out["base_mae"],
                        "cand_mae": out["plus_mae"],
                        "delta": out["delta"],
                        "status": "OK",
                    }
                )

    # TmApp: BASE + microprobe
    eval_block("BASE+THERMAL_MICROPROBE", bases["TmApp_BASE"], X_all, y_tm, "TmApp")

    # competition recipe
    recipe = pd.concat([X_pair, X_m1], axis=1, join="inner")
    recipe = recipe.loc[:, ~recipe.columns.duplicated()]
    base_recipe = pd.concat([bases["TmApp_BASE"], recipe], axis=1, join="inner")
    base_recipe = base_recipe.loc[:, ~base_recipe.columns.duplicated()]
    # Compare recipe vs recipe+microprobe: use recipe as "base" structurally by folding microprobe as struct
    # Exact ask: BASE+BIOEMU_NEW_PAIRWISE+M1 vs same+MICROPROBE
    # Implement as base=TmApp_BASE+pair+m1, struct=microprobe
    eval_block(
        "BASE+BIOEMU_NEW_PAIRWISE+M1+THERMAL_MICROPROBE",
        base_recipe,
        X_all,
        y_tm,
        "TmApp",
    )

    # HIC
    eval_block("HIC_ARO+THERMAL_MICROPROBE_HIC", bases["HIC_BASE_ARO"], X_hic, y_hic, "HIC")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "THERMAL_MICROPROBE_DEV_RESULTS.csv", index=False)

    def get_delta(comp, fold, mode="FREE_ALPHA"):
        sub = df[(df.comparison == comp) & (df.fold == fold) & (df.mode == mode) & (df.status == "OK")]
        if len(sub) == 0:
            return None
        return float(sub.iloc[0]["delta"])

    tm_p = get_delta("BASE+THERMAL_MICROPROBE", "Primary")
    tm_s = get_delta("BASE+THERMAL_MICROPROBE", "Shadow")
    tm_pf = get_delta("BASE+THERMAL_MICROPROBE", "Primary", "BASE_FIXED_ALPHA")
    tm_sf = get_delta("BASE+THERMAL_MICROPROBE", "Shadow", "BASE_FIXED_ALPHA")
    rec_p = get_delta("BASE+BIOEMU_NEW_PAIRWISE+M1+THERMAL_MICROPROBE", "Primary")
    rec_s = get_delta("BASE+BIOEMU_NEW_PAIRWISE+M1+THERMAL_MICROPROBE", "Shadow")
    hic_p = get_delta("HIC_ARO+THERMAL_MICROPROBE_HIC", "Primary")
    hic_s = get_delta("HIC_ARO+THERMAL_MICROPROBE_HIC", "Shadow")

    tm_go = tm_p is not None and tm_s is not None and tm_p > 0 and tm_s > 0
    hic_go = hic_p is not None and hic_s is not None and hic_p > 0 and hic_s > 0
    decision = {
        "tmapp_primary_delta": tm_p,
        "tmapp_shadow_delta": tm_s,
        "tmapp_fixed_primary_delta": tm_pf,
        "tmapp_fixed_shadow_delta": tm_sf,
        "tmapp_verdict": verdict(tm_p or 0, tm_s or 0),
        "recipe_primary_delta": rec_p,
        "recipe_shadow_delta": rec_s,
        "hic_primary_delta": hic_p,
        "hic_shadow_delta": hic_s,
        "hic_verdict": verdict(hic_p or 0, hic_s or 0),
        "GO_TO_TEST_TMAPP": bool(tm_go),
        "GO_TO_TEST_HIC": bool(hic_go) and not tm_go,
        "GO_TO_TEST": bool(tm_go or hic_go),
        "final_gate": "GO_TO_TEST" if (tm_go or hic_go) else "STOP",
    }
    (RES / "DEV_GATE_DECISION.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
