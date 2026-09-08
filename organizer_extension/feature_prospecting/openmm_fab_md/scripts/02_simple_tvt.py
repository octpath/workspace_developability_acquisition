#!/usr/bin/env python3
"""Simple TVT for OpenMM Fab MD features (exact frozen protocol)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "openmm_fab_md"
RES = OUT / "results"
INTERIM = FP / "fennix_fab_context_interim_audit"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
BIO = FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
M1 = FP / "structure_marathon/proteinmpnn/M1_FEATURES.csv"
GAP = FP / "structure_gap_closure"
CONT = GAP / "results/HIC_CONTINUOUS_SURFACE_FEATURES.csv"
TITR = FP / "TITRATION-SHAPE/features_esmfold.parquet"
FREEZE = json.loads((OUT / "OPENMM_MD_FEATURE_FREEZE.json").read_text())
B_BOOT = 10000
RNG = np.random.default_rng(42)

spec = importlib.util.spec_from_file_location("tvt_rescreen", INTERIM / "scripts/run_simple_tvt_rescreen.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["tvt_rescreen"] = mod
spec.loader.exec_module(mod)


def run_one(base, struct, y, folds, ids, mode):
    common = sorted(set(base.index) & set(struct.index) & set(y.index) & set(folds.id.astype(str)) & set(ids))
    cov_ok = True
    for k, tr, va, te in mod.rotation_splits(folds, common):
        ok = len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE
        cov_ok = cov_ok and ok
    if not cov_ok or len(common) < 40:
        return None
    Xb, Xs = base.loc[common], struct.loc[common]
    overlap = [c for c in Xs.columns if c in Xb.columns]
    if overlap:
        Xs = Xs.drop(columns=overlap)
    yy = y.loc[common]
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    for k, tr, va, te in mod.rotation_splits(folds, common):
        a_base = mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=False)
        a_plus = a_base if mode != "FREE_ALPHA" else mod.choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=True)
        if mode != "FREE_ALPHA":
            a_plus = a_base
        base_oof.loc[te] = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_base, use_struct=False)
        plus_oof.loc[te] = mod.predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_plus, use_struct=True)
    return {
        "N": len(common),
        "struct_dim": int(Xs.shape[1]),
        "base_mae": mod.mae(yy, base_oof),
        "plus_mae": mod.mae(yy, plus_oof),
        "delta": mod.mae(yy, base_oof) - mod.mae(yy, plus_oof),
        "y": yy,
        "base_oof": base_oof,
        "plus_oof": plus_oof,
    }


def verdict(dp, ds):
    if dp is None or ds is None:
        return "COMP_DROP"
    if dp > 0 and ds > 0:
        return "COMP_TRY" if min(dp, ds) < 0.02 else "COMP_PRIORITY"
    if dp > 0 or ds > 0:
        return "COMP_MIXED"
    return "COMP_DROP"


def bootstrap(y, b0, p0):
    y = np.asarray(y, float)
    d = np.abs(y - np.asarray(b0, float)) - np.abs(y - np.asarray(p0, float))
    boots = np.array([float(d[RNG.integers(0, len(d), len(d))].mean()) for _ in range(B_BOOT)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float((boots <= 0).mean())


def main():
    feat = pd.read_parquet(RES / "OPENMM_FAB_MD_FEATURES_DEV.parquet").set_index("id")
    cols = FREEZE["feature_names"]
    X_all = feat[cols].astype(float)
    hic_cols = [c for c in cols if "sasa" in c.lower() or "hydrophobic" in c.lower() or "aromatic" in c.lower() or c.startswith("Tyr") or c.startswith("Phe") or c.startswith("Trp") or c.startswith("delta_")]
    X_hic = X_all[hic_cols]

    bases, _ = mod.load_bases()
    y_tm, y_hic = mod.load_y("TmApp"), mod.load_y("HIC")
    primary, shadow = pd.read_csv(PRIMARY), pd.read_csv(SHADOW)
    usable = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]

    bio = pd.read_csv(BIO).set_index("id")
    pair_cols = [c for c in bio.columns if "ca_rmsd" in c.lower()]
    X_pair = bio[pair_cols].apply(pd.to_numeric, errors="coerce")
    m1 = mod.numeric_X(pd.read_csv(M1), drop_cols=["status", "extraction_status"])
    recipe = pd.concat([bases["TmApp_BASE"], X_pair, m1], axis=1, join="inner")
    recipe = recipe.loc[:, ~recipe.columns.duplicated()]

    cont = pd.read_csv(CONT)
    X_cont = mod.numeric_X(cont[["id"] + [c for c in cont.columns if c.startswith("fv_esmfold__")]])
    X_titr = mod.numeric_X(pd.read_parquet(TITR), drop_cols=["extraction_status", "status"])
    hic_strong = pd.concat([bases["HIC_BASE_ARO"], X_cont, X_titr], axis=1, join="inner")
    hic_strong = hic_strong.loc[:, ~hic_strong.columns.duplicated()]

    rows = []
    oofs = {}

    def add(name, base, struct, y, target):
        for fold_name, folds in [("Primary", primary), ("Shadow", shadow)]:
            for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
                out = run_one(base, struct, y, folds, usable, mode)
                if out is None:
                    rows.append({"target": target, "comparison": name, "fold": fold_name, "mode": mode, "status": "SKIP"})
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
                if mode == "FREE_ALPHA":
                    oofs[(name, fold_name)] = out

    add("BASE+OPENMM_MD", bases["TmApp_BASE"], X_all, y_tm, "TmApp")
    add("BASE+BIOEMU_NEW_PAIRWISE+M1+OPENMM_MD", recipe, X_all, y_tm, "TmApp")
    add("HIC_ARO+OPENMM_MD_HIC", bases["HIC_BASE_ARO"], X_hic, y_hic, "HIC")
    add("HIC_ARO+CONT+TITR+OPENMM_MD_HIC", hic_strong, X_hic, y_hic, "HIC")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "OPENMM_MD_DEV_RESULTS.csv", index=False)
    df.to_csv(OUT / "OPENMM_MD_DEV_RESULTS.csv", index=False)

    def dget(comp, fold, mode="FREE_ALPHA"):
        s = df[(df.comparison == comp) & (df.fold == fold) & (df.mode == mode) & (df.status == "OK")]
        return None if s.empty else float(s.iloc[0]["delta"])

    tm_p, tm_s = dget("BASE+OPENMM_MD", "Primary"), dget("BASE+OPENMM_MD", "Shadow")
    tm_pf, tm_sf = dget("BASE+OPENMM_MD", "Primary", "BASE_FIXED_ALPHA"), dget("BASE+OPENMM_MD", "Shadow", "BASE_FIXED_ALPHA")
    rec_p, rec_s = dget("BASE+BIOEMU_NEW_PAIRWISE+M1+OPENMM_MD", "Primary"), dget("BASE+BIOEMU_NEW_PAIRWISE+M1+OPENMM_MD", "Shadow")
    hic_p, hic_s = dget("HIC_ARO+OPENMM_MD_HIC", "Primary"), dget("HIC_ARO+OPENMM_MD_HIC", "Shadow")
    hic2_p, hic2_s = dget("HIC_ARO+CONT+TITR+OPENMM_MD_HIC", "Primary"), dget("HIC_ARO+CONT+TITR+OPENMM_MD_HIC", "Shadow")

    boot = {}
    for key in [("BASE+OPENMM_MD", "Primary"), ("BASE+OPENMM_MD", "Shadow")]:
        if key in oofs and tm_p and tm_s and tm_p > 0 and tm_s > 0:
            out = oofs[key]
            lo, hi, p = bootstrap(out["y"], out["base_oof"], out["plus_oof"])
            boot[f"{key[0]}_{key[1]}"] = {"ci95": [lo, hi], "p_improve": p}

    tm_go = tm_p is not None and tm_s is not None and tm_p > 0 and tm_s > 0
    hic_go = hic_p is not None and hic_s is not None and hic_p > 0 and hic_s > 0
    decision = {
        "tmapp_primary_delta": tm_p,
        "tmapp_shadow_delta": tm_s,
        "tmapp_fixed_primary_delta": tm_pf,
        "tmapp_fixed_shadow_delta": tm_sf,
        "tmapp_verdict": verdict(tm_p, tm_s),
        "recipe_primary_delta": rec_p,
        "recipe_shadow_delta": rec_s,
        "hic_primary_delta": hic_p,
        "hic_shadow_delta": hic_s,
        "hic_cont_titr_primary_delta": hic2_p,
        "hic_cont_titr_shadow_delta": hic2_s,
        "hic_verdict": verdict(hic_p, hic_s),
        "GO_TO_TEST_TMAPP": bool(tm_go),
        "GO_TO_TEST_HIC": bool(hic_go),
        "final_tmapp": "GO_TO_TEST" if tm_go else "STOP",
        "final_hic": "GO_TO_TEST" if hic_go else "STOP",
        "bootstrap": boot,
    }
    (RES / "DEV_GATE_DECISION.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
