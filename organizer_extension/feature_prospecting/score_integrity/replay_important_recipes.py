#!/usr/bin/env python3
"""Replay important organizer recipes through canonical Simple TVT."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "score_integrity"
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)
INTERIM = FP / "fennix_fab_context_interim_audit"
BIO = FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
M1 = FP / "structure_marathon/proteinmpnn/M1_FEATURES.csv"
ARO = FP / "structure_gap_closure/cache/aromatic_features_esmfold.csv"
CONT = FP / "structure_gap_closure/results/HIC_CONTINUOUS_SURFACE_FEATURES.csv"
TITR = FP / "TITRATION-SHAPE/features_esmfold.parquet"
HYDRO = FP / "HYDRO-FIELD/features_esmfold.parquet"
OPENMM = FP / "openmm_fab_md"
ABL = FP / "ablingua600m"
TOL = 1e-6
ROUND_TOL = 5e-4

sys.path.insert(0, str(OUT))
from canonical_simple_tvt import (  # noqa: E402
    CANONICAL_VERSION,
    align_feature_block,
    load_folds,
    run_base_plus_struct,
    run_recipe_plus_abl_blocks,
    run_standalone,
    rotation_splits,
)

spec = importlib.util.spec_from_file_location(
    "tvt", INTERIM / "scripts/run_simple_tvt_rescreen.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["tvt_rescreen"] = mod
spec.loader.exec_module(mod)


def classify(hist_p, hist_s, can_p, can_s, status_hint=None):
    if hist_p is None or hist_s is None:
        return status_hint or "SOURCE_NOT_RECONSTRUCTABLE"
    dp, ds = abs(hist_p - can_p), abs(hist_s - can_s)
    if dp < TOL and ds < TOL:
        return "EXACT_MATCH"
    if dp < ROUND_TOL and ds < ROUND_TOL:
        return "ROUNDING_MATCH"
    if status_hint:
        return status_hint
    return "PROTOCOL_DIFFERENCE_EXPLAINED"


def load_hist_blocks():
    p = INTERIM / "SIMPLE_TVT_ALL_BLOCKS.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        key = (str(r["target"]), str(r["family"]))
        out[key] = (
            float(r["free_primary_combined_mae"]),
            float(r["free_shadow_combined_mae"]),
            float(r["free_primary_delta"]),
            float(r["free_shadow_delta"]),
            float(r["free_primary_base_mae"]),
            float(r["free_shadow_base_mae"]),
        )
    return out


def load_hist_combos():
    p = INTERIM / "SIMPLE_TVT_LIMITED_COMBINATIONS.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        out[(str(r["target"]), str(r["combination"]))] = (
            float(r["free_primary_combined_mae"]),
            float(r["free_shadow_combined_mae"]),
        )
    return out


def fingerprint_block(name, X):
    arr = X.to_numpy(float)
    return {
        "block": name,
        "shape": list(X.shape),
        "matched_ids": int(X.shape[0]),
        "all_nan": bool(np.isnan(arr).all()),
        "finite_frac": float(np.isfinite(arr).mean()),
        "variance": float(np.nanvar(arr)),
        "fingerprint": hashlib.sha256(np.ascontiguousarray(np.nan_to_num(arr)).tobytes()).hexdigest()[:16],
    }


def main():
    ids = [str(x) for x in pd.read_csv(ROOT / "competition/data/distribution/dev.csv")["id"]]
    primary, shadow = load_folds()
    y_tm = mod.load_y("TmApp")
    y_hic = mod.load_y("HIC")
    bases, base_names = mod.load_bases()
    hist = load_hist_blocks()
    hist_combo = load_hist_combos()

    seq_cols = [c for c in bases["TmApp_BASE"].columns if c.startswith("seqB_")]
    seq = align_feature_block(bases["TmApp_BASE"][seq_cols], ids, "SEQ_BASIC")
    ablang = align_feature_block(
        bases["TmApp_BASE"][[c for c in bases["TmApp_BASE"].columns if c.startswith("ablang2_")]],
        ids,
        "AbLang2_HL_paired",
    )
    tm_base = align_feature_block(bases["TmApp_BASE"], ids, "TmApp_BASE")
    hic_base = align_feature_block(bases["HIC_BASE"], ids, "HIC_BASE")
    hic_aro = align_feature_block(bases["HIC_BASE_ARO"], ids, "HIC_BASE_ARO")

    bio = pd.read_csv(BIO)
    bio["id"] = bio["id"].astype(str)
    pair_cols = [c for c in bio.columns if "ca_rmsd" in c.lower()]
    X_pair = align_feature_block(bio.set_index("id")[pair_cols], ids, "BIOEMU_NEW_PAIRWISE")
    m1 = align_feature_block(
        mod.numeric_X(pd.read_csv(M1), drop_cols=["status", "extraction_status"]),
        ids,
        "M1_PROTEINMPNN",
    )
    aro = align_feature_block(mod.numeric_X(pd.read_csv(ARO)), ids, "AROMATIC_TOPO")

    HL = align_feature_block(
        pd.read_parquet(ABL / "embeddings/ablingua600m_HL_mean_concat.parquet").set_index("id"),
        ids,
        "AbLingua_HL_mean",
    )
    cdr3_path = ABL / "embeddings_guided/ablingua600m_CDR3.parquet"
    CDR3 = None
    if cdr3_path.exists():
        CDR3 = align_feature_block(pd.read_parquet(cdr3_path).set_index("id"), ids, "AbLingua_CDR3")

    cont = None
    if CONT.exists():
        cdf = pd.read_csv(CONT)
        cdf["id"] = cdf["id"].astype(str)
        cols = [c for c in cdf.columns if c.startswith("fv_esmfold__")]
        cont = align_feature_block(cdf.set_index("id")[cols], ids, "CONTINUOUS_SURFACE")

    titr = None
    if TITR.exists():
        try:
            titr = align_feature_block(
                mod.numeric_X(pd.read_parquet(TITR), drop_cols=["extraction_status", "status"]),
                ids,
                "TITRATION_SHAPE",
            )
        except Exception as e:
            print("TITR skip", e)

    hydro = None
    if HYDRO.exists():
        try:
            hydro = align_feature_block(
                mod.numeric_X(pd.read_parquet(HYDRO), drop_cols=["extraction_status", "status"]),
                ids,
                "HYDRO_FIELD",
            )
        except Exception as e:
            print("HYDRO skip", e)

    openmm_X = None
    fam_cols: dict[str, list[str]] = {}
    openmm_ids = ids
    omm_feat = OPENMM / "results/OPENMM_FAB_MD_FEATURES_DEV.parquet"
    if omm_feat.exists():
        ox = pd.read_parquet(omm_feat)
        openmm_ids = [str(x) for x in ox["id"]]
        openmm_X = align_feature_block(ox.set_index("id"), openmm_ids, "OPENMM_MD")
        fam_map = OPENMM / "OPENMM_MD_FAMILY_MAP.csv"
        if not fam_map.exists():
            fam_map = OPENMM / "results/OPENMM_MD_FAMILY_MAP.csv"
        if fam_map.exists():
            fm = pd.read_csv(fam_map)
            for fam, g in fm.groupby("ablation_family"):
                fam_cols[str(fam)] = [c for c in g["feature"].astype(str) if c in openmm_X.columns]

    fennix_const = None
    fc = INTERIM / "results/INTERIM_CONSTANT_FEATURES.csv"
    if fc.exists():
        try:
            fennix_const = align_feature_block(
                mod.numeric_X(pd.read_csv(fc)), ids, "INTERIM_CONSTANT", allow_intersection=True
            )
        except Exception as e:
            print("fennix const skip", e)

    block_audits = []
    for name, X in [
        ("SEQ_BASIC", seq),
        ("AbLang2_HL_paired", ablang),
        ("TmApp_BASE", tm_base),
        ("BIOEMU_NEW_PAIRWISE", X_pair),
        ("M1_PROTEINMPNN", m1),
        ("AbLingua_HL_mean", HL),
        ("AROMATIC_TOPO", aro),
        ("HIC_BASE", hic_base),
        ("HIC_BASE_ARO", hic_aro),
    ]:
        block_audits.append(fingerprint_block(name, X))
    if cont is not None:
        block_audits.append(fingerprint_block("CONTINUOUS_SURFACE", cont))
    if titr is not None:
        block_audits.append(fingerprint_block("TITRATION_SHAPE", titr))
    if hydro is not None:
        block_audits.append(fingerprint_block("HYDRO_FIELD", hydro))
    if openmm_X is not None:
        block_audits.append(fingerprint_block("OPENMM_MD", openmm_X))
        for fam, cols in fam_cols.items():
            if cols:
                block_audits.append(fingerprint_block(f"OPENMM_{fam}", openmm_X[cols]))
    if CDR3 is not None:
        block_audits.append(fingerprint_block("AbLingua_CDR3", CDR3))
    if fennix_const is not None:
        block_audits.append(fingerprint_block("INTERIM_CONSTANT", fennix_const))

    def constant_mae(y, folds, id_list):
        common = [i for i in id_list if i in y.index]
        oof = pd.Series(index=common, dtype=float)
        for _, tr, va, te in rotation_splits(folds, common):
            med = float(y.loc[tr + va].median())
            oof.loc[te] = med
        return float(np.mean(np.abs(y.loc[common] - oof))), len(common)

    rows = []

    def add_row(**kw):
        rows.append(kw)
        hp, hs = kw.get("historical_primary"), kw.get("historical_shadow")
        print(
            f"{kw['method_id']}: can={kw.get('canonical_primary'):.4f}/{kw.get('canonical_shadow'):.4f} "
            f"hist={hp}/{hs} -> {kw.get('historical_status')}",
            flush=True,
        )

    def eval_standalone(
        method_id, name, target, X, hist_p=None, hist_s=None, dim_mode="raw", notes="", hint=None, id_list=None
    ):
        id_list = id_list or ids
        y = y_tm if target == "TmApp" else y_hic
        Xa = align_feature_block(X, id_list, method_id, allow_intersection=(set(id_list) != set(ids)))
        use_ids = list(Xa.index)
        rp = run_standalone(Xa, y, primary, use_ids, dim_mode=dim_mode)
        rs = run_standalone(Xa, y, shadow, use_ids, dim_mode=dim_mode)
        status = classify(hist_p, hist_s, rp["mae"], rs["mae"], hint)
        add_row(
            target=target,
            method_id=method_id,
            method_name=name,
            feature_blocks=name,
            molecular_scope="see notes",
            model="Ridge",
            preprocessing=dim_mode,
            n_dev=rp["N"],
            simple_tvt_primary_mae=rp["mae"],
            simple_tvt_shadow_mae=rs["mae"],
            historical_primary_mae=hist_p,
            historical_shadow_mae=hist_s,
            historical_status=status,
            canonical_evaluator_version=CANONICAL_VERSION,
            feature_hashes=rp.get("prediction_hash"),
            prediction_hash=rp["prediction_hash"] + "/" + rs["prediction_hash"],
            validity="AUTHORITATIVE",
            notes=notes,
            canonical_primary=rp["mae"],
            canonical_shadow=rs["mae"],
            historical_primary=hist_p,
            historical_shadow=hist_s,
        )
        return rp, rs

    def eval_incr(
        method_id,
        name,
        target,
        base,
        struct,
        hist_p=None,
        hist_s=None,
        force_pca=None,
        notes="",
        hint=None,
        id_list=None,
        compare_delta=False,
        hist_delta_p=None,
        hist_delta_s=None,
    ):
        id_list = id_list or ids
        y = y_tm if target == "TmApp" else y_hic
        Ba = align_feature_block(base, id_list, method_id + "_base", allow_intersection=True)
        Sa = align_feature_block(struct, list(Ba.index), method_id + "_struct", allow_intersection=True)
        use_ids = [i for i in Ba.index if i in Sa.index]
        Ba, Sa = Ba.loc[use_ids], Sa.loc[use_ids]
        op = run_base_plus_struct(Ba, Sa, y, primary, use_ids, mode="FREE_ALPHA", force_pca_struct=force_pca)
        os_ = run_base_plus_struct(Ba, Sa, y, shadow, use_ids, mode="FREE_ALPHA", force_pca_struct=force_pca)
        if compare_delta:
            can_p, can_s = op["delta"], os_["delta"]
            hp, hs = hist_delta_p, hist_delta_s
        else:
            can_p, can_s = op["plus_mae"], os_["plus_mae"]
            hp, hs = hist_p, hist_s
        status = classify(hp, hs, can_p, can_s, hint)
        add_row(
            target=target,
            method_id=method_id,
            method_name=name,
            feature_blocks=name,
            molecular_scope="see notes",
            model="Ridge",
            preprocessing=f"FREE_ALPHA force_pca={force_pca}",
            n_dev=op["N"],
            simple_tvt_primary_mae=op["plus_mae"],
            simple_tvt_shadow_mae=os_["plus_mae"],
            historical_primary_mae=hist_p if not compare_delta else hp,
            historical_shadow_mae=hist_s if not compare_delta else hs,
            historical_status=status,
            canonical_evaluator_version=CANONICAL_VERSION,
            feature_hashes=op["prediction_hash"],
            prediction_hash=op["prediction_hash"] + "/" + os_["prediction_hash"],
            validity="AUTHORITATIVE",
            notes=notes
            + f"; deltaP={op['delta']:.6f} deltaS={os_['delta']:.6f}; baseP={op['base_mae']:.4f} baseS={os_['base_mae']:.4f}",
            canonical_primary=can_p,
            canonical_shadow=can_s,
            historical_primary=hp,
            historical_shadow=hs,
        )
        return op, os_

    # ---- TmApp ----
    cp, cs = constant_mae(y_tm, primary, ids), constant_mae(y_tm, shadow, ids)
    add_row(
        target="TmApp",
        method_id="TmApp_CONSTANT",
        method_name="constant_median",
        feature_blocks="none",
        molecular_scope="n/a",
        model="median",
        preprocessing="fold_tv_median",
        n_dev=cp[1],
        simple_tvt_primary_mae=cp[0],
        simple_tvt_shadow_mae=cs[0],
        historical_primary_mae=None,
        historical_shadow_mae=None,
        historical_status="SOURCE_NOT_RECONSTRUCTABLE",
        canonical_evaluator_version=CANONICAL_VERSION,
        feature_hashes="",
        prediction_hash="",
        validity="AUTHORITATIVE",
        notes="fold-local median baseline",
        canonical_primary=cp[0],
        canonical_shadow=cs[0],
        historical_primary=None,
        historical_shadow=None,
    )

    eval_standalone("TmApp_SEQ_BASIC", "SEQ_BASIC", "TmApp", seq, notes="canonical SEQ")
    eval_standalone("TmApp_AbLang2_HL_paired", "AbLang2_HL_paired", "TmApp", ablang, notes="from stage2 cache")
    eval_standalone(
        "TmApp_BASE",
        "AbLang2+SEQ_BASIC",
        "TmApp",
        tm_base,
        hist_p=hist.get(("TmApp", "BIOEMU_NEW_PAIRWISE"), (None,) * 6)[4],
        hist_s=hist.get(("TmApp", "BIOEMU_NEW_PAIRWISE"), (None,) * 6)[5],
        notes=base_names["TmApp_BASE"] + "; hist=BASE MAE from ALL_BLOCKS",
    )

    for fam, X, force in [
        ("BIOEMU_NEW_PAIRWISE", X_pair, False),
        ("M1_PROTEINMPNN", m1, False),
    ]:
        h = hist.get(("TmApp", fam))
        eval_incr(
            f"TmApp_BASE+{fam}",
            f"TmApp_BASE+{fam}",
            "TmApp",
            tm_base,
            X,
            hist_p=None if h is None else h[0],
            hist_s=None if h is None else h[1],
            force_pca=force,
            notes="increment on TmApp_BASE; hist=combined MAE ALL_BLOCKS",
            compare_delta=False,
        )

    bm = align_feature_block(pd.concat([X_pair, m1], axis=1), ids, "BIOEMU+M1")
    hc = hist_combo.get(("TmApp", "BIOEMU_NEW_PAIRWISE+M1_PROTEINMPNN"))
    eval_incr(
        "TmApp_BASE+BIOEMU_PAIR+M1",
        "BASE+BIOEMU_NEW_PAIRWISE+M1",
        "TmApp",
        tm_base,
        bm,
        hist_p=None if hc is None else hc[0],
        hist_s=None if hc is None else hc[1],
        force_pca=False,
        notes="current competition recipe core",
    )

    eval_standalone(
        "TmApp_AbLingua_HL_PCA32",
        "AbLingua_HL_mean",
        "TmApp",
        HL,
        hist_p=3.185558892657944,
        hist_s=3.162369431229711,
        dim_mode="PCA32",
        notes="ablingua sprint standalone (no SEQ; not bug-affected)",
    )
    # SEQ+AbLingua corrected should differ from superseded sprint A
    eval_incr(
        "TmApp_SEQ+AbLingua",
        "SEQ_BASIC+AbLingua",
        "TmApp",
        seq,
        HL,
        hist_p=3.185558892657944,
        hist_s=3.162369431229711,
        force_pca=True,
        notes="SEQ+AbLingua; hist=sprintA SUPERSEDED (SEQ NaN => equals standalone)",
        hint="BUG_AFFECTED",
    )

    recipe = align_feature_block(pd.concat([tm_base, X_pair, m1], axis=1), ids, "CURRENT_RECIPE")
    eval_standalone(
        "TmApp_CURRENT_RECIPE",
        "AbLang2+SEQ+BIOEMU_PAIR+M1",
        "TmApp",
        recipe,
        hist_p=2.702681,
        hist_s=2.845937,
        notes="recipe_only CORRECTED",
    )
    eval_incr(
        "TmApp_PARENT_AbLingua_GLOBAL",
        "CURRENT_RECIPE+AbLingua_GLOBAL",
        "TmApp",
        recipe,
        HL,
        hist_p=2.7466001170280285,
        hist_s=2.822725206703513,
        force_pca=True,
        notes="authoritative PARENT after SEQ fix",
    )
    a2seq = align_feature_block(pd.concat([seq, ablang], axis=1), ids, "SEQ+AbLang2")
    eval_incr(
        "TmApp_SEQ_AbLang2_AbLingua",
        "SEQ+AbLang2+AbLingua",
        "TmApp",
        a2seq,
        HL,
        hist_p=2.796211,
        hist_s=2.862282,
        force_pca=True,
        notes="CORRECTED after SEQ fix",
    )

    if CDR3 is not None:
        # Guided protocol: recipe + PCA(GLOBAL) + PCA(CDR3) per-block
        op = run_recipe_plus_abl_blocks(recipe, [HL], [CDR3], y_tm, primary, ids)
        os_ = run_recipe_plus_abl_blocks(recipe, [HL], [CDR3], y_tm, shadow, ids)
        hp, hs = 2.732072376654289, 2.784957206877823
        status = classify(hp, hs, op["plus_mae"], os_["plus_mae"])
        add_row(
            target="TmApp",
            method_id="TmApp_PARENT+AbLingua_CDR3",
            method_name="PARENT+CDR3",
            feature_blocks="recipe+PCA(AbLingua_GLOBAL)+PCA(CDR3)",
            molecular_scope="see notes",
            model="Ridge",
            preprocessing="PCA32_per_abl_block FREE_ALPHA",
            n_dev=op["N"],
            simple_tvt_primary_mae=op["plus_mae"],
            simple_tvt_shadow_mae=os_["plus_mae"],
            historical_primary_mae=hp,
            historical_shadow_mae=hs,
            historical_status=status,
            canonical_evaluator_version=CANONICAL_VERSION,
            feature_hashes=op["prediction_hash"],
            prediction_hash=op["prediction_hash"] + "/" + os_["prediction_hash"],
            validity="AUTHORITATIVE",
            notes=f"guided CDR3 protocol; deltaP={op['delta']:.6f} deltaS={os_['delta']:.6f}; parentP={op['base_mae']:.4f}",
            canonical_primary=op["plus_mae"],
            canonical_shadow=os_["plus_mae"],
            historical_primary=hp,
            historical_shadow=hs,
        )

    if fennix_const is not None:
        h = hist.get(("TmApp", "INTERIM_CONSTANT"))
        # historical N=100 FeNNix cohort — expect protocol difference vs full intersection
        eval_incr(
            "TmApp_BASE+INTERIM_CONSTANT",
            "BASE+INTERIM_CONSTANT",
            "TmApp",
            tm_base,
            fennix_const,
            hist_p=None if h is None else h[0],
            hist_s=None if h is None else h[1],
            force_pca=False,
            notes="interim FeNNix CONSTANT; hist often N=100 → PROTOCOL_DIFFERENCE if N differs",
            hint="PROTOCOL_DIFFERENCE_EXPLAINED",
            id_list=list(fennix_const.index),
        )

    if openmm_X is not None:
        base_o = align_feature_block(tm_base, openmm_ids, "BASE_omm", allow_intersection=True)
        omm_ids = list(base_o.index)
        ommX = align_feature_block(openmm_X, omm_ids, "OPENMM")
        eval_incr(
            "TmApp_BASE+OPENMM_MD",
            "BASE+OPENMM_MD",
            "TmApp",
            base_o,
            ommX,
            hist_p=2.874346,
            hist_s=2.955769,
            force_pca=False,
            notes="openmm whole block; hist=cand_mae OPENMM_MD_DEV_RESULTS",
            id_list=omm_ids,
        )
        for fam in ["GLOBAL_DYNAMICS", "DOMAIN_FLEXIBILITY"]:
            cols = [c for c in fam_cols.get(fam, []) if c in ommX.columns]
            if not cols:
                continue
            # hist from family ablation: base_mae - delta ≈ cand; use summary deltas + known base
            # Prefer exact cand from results if available
            fam_res = OPENMM / "results/OPENMM_MD_FAMILY_ABLATION_RESULTS.csv"
            hp = hs = None
            if fam_res.exists():
                fr = pd.read_csv(fam_res)
                fr["family"] = fr["family"].astype(str)
                fr["comparison"] = fr["comparison"].astype(str)
                fr["mode"] = fr["mode"].astype(str)
                fr["fold"] = fr["fold"].astype(str)
                sub = fr[
                    (fr["family"] == fam)
                    & (fr["comparison"] == "BASE+F")
                    & (fr["mode"] == "FREE_ALPHA")
                ]
                if len(sub):
                    hp = float(sub.loc[sub["fold"] == "Primary", "cand_mae"].iloc[0])
                    hs = float(sub.loc[sub["fold"] == "Shadow", "cand_mae"].iloc[0])
            eval_incr(
                f"TmApp_BASE+OPENMM_{fam}",
                f"BASE+OPENMM_{fam}",
                "TmApp",
                base_o,
                ommX[cols],
                hist_p=hp,
                hist_s=hs,
                force_pca=False,
                notes=f"openmm family {fam}",
                id_list=omm_ids,
            )

    # Historical invalid ADD (registry marker)
    add_row(
        target="TmApp",
        method_id="TmApp_AbLingua_ADD_HISTORICAL_INVALID",
        method_name="ADD_to_current_recipe_sprintA",
        feature_blocks="CURRENT_RECIPE+AbLingua (SEQ NaN bug)",
        molecular_scope="n/a",
        model="Ridge",
        preprocessing="PCA32_struct",
        n_dev=162,
        simple_tvt_primary_mae=2.7466001170280285,
        simple_tvt_shadow_mae=2.822725206703513,
        historical_primary_mae=2.810733334088188,
        historical_shadow_mae=2.990836362735377,
        historical_status="BUG_AFFECTED",
        canonical_evaluator_version=CANONICAL_VERSION,
        feature_hashes="",
        prediction_hash="",
        validity="SUPERSEDED",
        notes="HISTORICAL_RESULT_INVALID due to SEQ_BASIC all-NaN; canonical=corrected PARENT; validity!=AUTHORITATIVE",
        canonical_primary=2.7466001170280285,
        canonical_shadow=2.822725206703513,
        historical_primary=2.810733334088188,
        historical_shadow=2.990836362735377,
    )

    # ---- HIC ----
    ch, csh = constant_mae(y_hic, primary, ids), constant_mae(y_hic, shadow, ids)
    add_row(
        target="HIC",
        method_id="HIC_CONSTANT",
        method_name="constant_median",
        feature_blocks="none",
        molecular_scope="n/a",
        model="median",
        preprocessing="fold_tv_median",
        n_dev=ch[1],
        simple_tvt_primary_mae=ch[0],
        simple_tvt_shadow_mae=csh[0],
        historical_primary_mae=None,
        historical_shadow_mae=None,
        historical_status="SOURCE_NOT_RECONSTRUCTABLE",
        canonical_evaluator_version=CANONICAL_VERSION,
        feature_hashes="",
        prediction_hash="",
        validity="AUTHORITATIVE",
        notes="fold-local median",
        canonical_primary=ch[0],
        canonical_shadow=csh[0],
        historical_primary=None,
        historical_shadow=None,
    )
    eval_standalone(
        "HIC_BASE",
        "ESM2_H+SEQ_ALL",
        "HIC",
        hic_base,
        hist_p=hist.get(("HIC", "AROMATIC_TOPO"), (None,) * 6)[4],
        hist_s=hist.get(("HIC", "AROMATIC_TOPO"), (None,) * 6)[5],
        notes=base_names["HIC_BASE"],
    )
    h_aro = hist.get(("HIC", "AROMATIC_TOPO"))
    eval_standalone(
        "HIC_BASE_ARO",
        "ESM2_H+SEQ_ALL+AROMATIC",
        "HIC",
        hic_aro,
        hist_p=None if h_aro is None else h_aro[0],
        hist_s=None if h_aro is None else h_aro[1],
        notes=base_names["HIC_BASE_ARO"] + "; hist=ARO combined (=BASE_ARO)",
    )
    eval_standalone("HIC_ARO_only", "AROMATIC_TOPO", "HIC", aro, notes="aromatic only")
    if cont is not None:
        h = hist.get(("HIC", "CONTINUOUS_SURFACE"))
        eval_incr(
            "HIC_ARO+CONT",
            "HIC_ARO+CONTINUOUS_SURFACE",
            "HIC",
            hic_aro,
            cont,
            hist_p=None if h is None else h[0],
            hist_s=None if h is None else h[1],
            force_pca=False,
            notes="surface incr",
        )
    if hydro is not None:
        h = hist.get(("HIC", "HYDRO_FIELD"))
        eval_incr(
            "HIC_ARO+HYDRO",
            "HIC_ARO+HYDRO_FIELD",
            "HIC",
            hic_aro,
            hydro,
            hist_p=None if h is None else h[0],
            hist_s=None if h is None else h[1],
            force_pca=False,
            notes="hydro field incr",
        )
    if titr is not None:
        h = hist.get(("HIC", "TITRATION_SHAPE"))
        eval_incr(
            "HIC_ARO+TITR",
            "HIC_ARO+TITRATION_SHAPE",
            "HIC",
            hic_aro,
            titr,
            hist_p=None if h is None else h[0],
            hist_s=None if h is None else h[1],
            force_pca=False,
            notes="titration incr",
        )
    if titr is not None and cont is not None:
        ct = align_feature_block(
            pd.concat([cont, titr], axis=1), ids, "CONT+TITR"
        )
        hc = hist_combo.get(("HIC", "CONTINUOUS_SURFACE+TITRATION_SHAPE"))
        eval_incr(
            "HIC_ARO+CONT+TITR",
            "HIC_ARO+CONT+TITR",
            "HIC",
            hic_aro,
            ct,
            hist_p=None if hc is None else hc[0],
            hist_s=None if hc is None else hc[1],
            force_pca=False,
            notes="strong hic combo",
        )

    # SEQ_BASIC dependent audit
    r_ab = run_standalone(ablang, y_tm, primary, ids, dim_mode="raw")
    r_base = run_standalone(tm_base, y_tm, primary, ids, dim_mode="raw")
    r_seq = run_standalone(seq, y_tm, primary, ids, dim_mode="raw")
    # buggy equal-pred pattern: AbLingua sprintA SEQ+PLM == PLM-only when SEQ NaN
    seq_audit = {
        "SEQ_BASIC_raw_dim": int(seq.shape[1]),
        "SEQ_BASIC_matched_ids": int(seq.shape[0]),
        "SEQ_BASIC_finite_frac_before_impute": float(np.isfinite(seq.to_numpy(float)).mean()),
        "SEQ_BASIC_variance_sum": float(np.nanvar(seq.to_numpy(float), axis=0).sum()),
        "ablang2_mae": r_ab["mae"],
        "base_mae": r_base["mae"],
        "seq_mae": r_seq["mae"],
        "ablang2_vs_base_pred_maxdiff": float(np.max(np.abs(r_ab["oof"] - r_base["oof"]))),
        "SEQ_changes_predictions_vs_AbLang2_alone": float(np.max(np.abs(r_ab["oof"] - r_base["oof"]))) > 1e-8,
        "suspicious_historical_SEQ_noop": [
            "AbLingua Sprint A: SEQ_BASIC+AbLingua PCA32 MAE == AbLingua standalone PCA32 (3.1856/3.1624) → SEQ was all-NaN",
            "AbLingua Sprint A: current_recipe and ADD used NaN SEQ inside AbLang2+SEQ block",
        ],
    }

    df = pd.DataFrame(rows)
    reg_cols = [
        "target",
        "method_id",
        "method_name",
        "feature_blocks",
        "molecular_scope",
        "model",
        "preprocessing",
        "n_dev",
        "simple_tvt_primary_mae",
        "simple_tvt_shadow_mae",
        "historical_primary_mae",
        "historical_shadow_mae",
        "historical_status",
        "canonical_evaluator_version",
        "feature_hashes",
        "prediction_hash",
        "validity",
        "notes",
    ]
    reg = df[reg_cols].copy()
    reg.to_csv(OUT / "ORGANIZER_SCORE_REGISTRY.csv", index=False)
    reg.to_csv(RES / "ORGANIZER_SCORE_REGISTRY.csv", index=False)
    pd.DataFrame(block_audits).to_csv(RES / "BLOCK_FINGERPRINTS.csv", index=False)
    pd.DataFrame(block_audits).to_csv(OUT / "BLOCK_FINGERPRINTS.csv", index=False)
    (RES / "SEQ_BASIC_AUDIT.json").write_text(json.dumps(seq_audit, indent=2))
    (OUT / "SEQ_BASIC_AUDIT.json").write_text(json.dumps(seq_audit, indent=2))

    cmp = df[
        [
            "method_id",
            "target",
            "historical_primary_mae",
            "historical_shadow_mae",
            "simple_tvt_primary_mae",
            "simple_tvt_shadow_mae",
            "historical_status",
            "validity",
            "notes",
        ]
    ].copy()
    cmp["delta_primary"] = cmp["simple_tvt_primary_mae"] - cmp["historical_primary_mae"]
    cmp["delta_shadow"] = cmp["simple_tvt_shadow_mae"] - cmp["historical_shadow_mae"]
    cmp.to_csv(RES / "HISTORICAL_VS_CANONICAL.csv", index=False)
    cmp.to_csv(OUT / "HISTORICAL_VS_CANONICAL.csv", index=False)
    print("Wrote registry", len(reg), "SEQ_changes", seq_audit["SEQ_changes_predictions_vs_AbLang2_alone"])
    print(cmp["historical_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
