#!/usr/bin/env python3
"""Endgame Ridge/Lasso Simple TVT CV → freeze Top-3 → Public/Private postmortem.

Order is enforced:
  1) FEATURE_RECIPE_FREEZE (already hashed)
  2) CV table
  3) CV_SELECTED_RECIPES_FREEZE.json
  4) Public/Private (only after freeze file exists)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/endgame_model_benchmark"
SI = ROOT / "organizer_extension/feature_prospecting/score_integrity"
sys.path.insert(0, str(SI))
sys.path.insert(0, str(OUT))

from canonical_simple_tvt import (  # noqa: E402
    CANONICAL_VERSION,
    FeatureAlignmentError,
    LassoConvergenceError,
    align_feature_block,
    fit_full_dev_lasso,
    impute_apply,
    impute_fit,
    load_folds,
    mae,
    pca_block,
    predict_with_lasso_artifact,
    run_base_plus_struct,
    run_recipe_plus_abl_blocks,
    run_standalone,
    run_standalone_lasso,
    rotation_splits,
)
from feature_store import FeatureStore, load_test_mask, load_test_y, load_y  # noqa: E402

FREEZE_CSV = OUT / "FEATURE_RECIPE_FREEZE.csv"
FREEZE_SHA = OUT / "FEATURE_RECIPE_FREEZE.sha256"
ALPHA_POLICY = OUT / "FULL_DEV_ALPHA_POLICY.json"
CV_TABLE = OUT / "ORGANIZER_ENDGAME_CV_TABLE.csv"
TOP3 = OUT / "CV_SELECTED_RECIPES_FREEZE.json"
POST = OUT / "ORGANIZER_MODEL_POSTMORTEM.csv"
PRED_DIR = OUT / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Recipes that historically use special Ridge PCA protocols
SPECIAL_RIDGE = {
    "TM_PARENT_ABLINGUA_GLOBAL": "recipe_plus_pca_ablingua",
    "TM_PARENT_ABLINGUA_CDR3": "recipe_plus_abl_blocks_pca",
}

INTERSECTION_RECIPES = {"TM_OPENMM_DOMAIN_FLEX", "TM_FENNIX_CONTEXT"}
NO_TEST_FEATURES = {"TM_OPENMM_DOMAIN_FLEX"}  # MD features DEV-only


def parse_blocks(s: str) -> list[str]:
    if not s or s == "none":
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def constant_tvt(y, folds, ids):
    common = [i for i in ids if i in y.index]
    oof = pd.Series(index=common, dtype=float)
    alphas = []
    for _, tr, va, te in rotation_splits(folds, common):
        med = float(y.loc[tr + va].median())
        oof.loc[te] = med
        alphas.append(med)
    return {"N": len(common), "mae": mae(y.loc[common], oof), "oof": oof, "alphas": alphas, "raw_dim": 0}


def recipe_parts(store: FeatureStore, recipe_id: str, blocks: list[str], ids: list[str]):
    allow = recipe_id in INTERSECTION_RECIPES
    if recipe_id == "TM_PARENT_ABLINGUA_GLOBAL":
        core = [b for b in blocks if b != "AbLingua_HL_mean"]
        Xc, _ = store.matrix_for_recipe(core, ids, allow_intersection=False)
        Hl = align_feature_block(store.get("AbLingua_HL_mean"), list(Xc.index), "AbLingua_HL_mean")
        return {"mode": "base_struct", "base": Xc, "struct": Hl, "force_pca": True, "ids": list(Xc.index)}
    if recipe_id == "TM_PARENT_ABLINGUA_CDR3":
        core = [b for b in blocks if b not in ("AbLingua_HL_mean", "AbLingua_CDR3")]
        Xc, _ = store.matrix_for_recipe(core, ids, allow_intersection=False)
        Hl = align_feature_block(store.get("AbLingua_HL_mean"), list(Xc.index), "AbLingua_HL_mean")
        Cd = align_feature_block(store.get("AbLingua_CDR3"), list(Xc.index), "AbLingua_CDR3")
        return {"mode": "abl_blocks", "recipe": Xc, "always": [Hl], "extra": [Cd], "ids": list(Xc.index)}
    X, meta = store.matrix_for_recipe(blocks, ids, allow_intersection=allow)
    return {"mode": "standalone", "X": X, "meta": meta, "ids": list(X.index)}


def eval_ridge(parts, y, folds, ids):
    if parts["mode"] == "standalone":
        if parts["X"].shape[1] == 0:
            return constant_tvt(y, folds, ids) | {"preprocessing": "fold_tv_median", "effective_dim": 0}
        r = run_standalone(parts["X"], y, folds, parts["ids"], dim_mode="raw")
        r["preprocessing"] = "impute+StandardScaler+Ridge(raw)"
        r["effective_dim"] = r["raw_dim"]
        return r
    if parts["mode"] == "base_struct":
        r = run_base_plus_struct(
            parts["base"], parts["struct"], y, folds, parts["ids"], mode="FREE_ALPHA", force_pca_struct=True
        )
        return {
            "N": r["N"],
            "mae": r["plus_mae"],
            "oof": r["plus_oof"],
            "alphas": r["alphas_plus"],
            "raw_dim": r["base_dim"] + r["struct_dim"],
            "effective_dim": r["base_dim"] + 32,
            "preprocessing": "recipe_raw + PCA32(AbLingua) + Ridge FREE_ALPHA",
            "base_mae": r["base_mae"],
        }
    if parts["mode"] == "abl_blocks":
        r = run_recipe_plus_abl_blocks(
            parts["recipe"], parts["always"], parts["extra"], y, folds, parts["ids"], mode="FREE_ALPHA"
        )
        return {
            "N": r["N"],
            "mae": r["plus_mae"],
            "oof": r["plus_oof"],
            "alphas": r["alphas_plus"],
            "raw_dim": parts["recipe"].shape[1] + parts["always"][0].shape[1] + parts["extra"][0].shape[1],
            "effective_dim": parts["recipe"].shape[1] + 64,
            "preprocessing": "recipe_raw + PCA32_per_abl_block + Ridge FREE_ALPHA",
            "base_mae": r["base_mae"],
        }
    raise RuntimeError(parts["mode"])


def eval_lasso(parts, y, folds, ids):
    if parts["mode"] == "standalone":
        X = parts["X"]
        use = parts["ids"]
    elif parts["mode"] == "base_struct":
        X = pd.concat([parts["base"], parts["struct"]], axis=1)
        use = parts["ids"]
    else:
        X = pd.concat([parts["recipe"], parts["always"][0], parts["extra"][0]], axis=1)
        use = parts["ids"]
    if X.shape[1] == 0:
        r = constant_tvt(y, folds, use)
        r.update(
            {
                "preprocessing": "fold_tv_median",
                "effective_dim": 0,
                "alpha_median": np.nan,
                "n_nonzero_median": 0,
                "sparsity_median": np.nan,
                "n_nonzero_min": 0,
                "n_nonzero_max": 0,
                "alphas": [],
                "n_nonzero": [],
            }
        )
        return r
    # No PCA for Lasso
    r = run_standalone_lasso(X, y, folds, use)
    r["effective_dim"] = r["raw_dim"]
    return r


def select_top3(cv: pd.DataFrame) -> dict:
    """Lowest cv_worst_mae, then cv_mean_mae; encourage recipe diversity."""
    out = {"policy": "cv_worst_mae then cv_mean_mae; prefer distinct recipe_id", "targets": {}}
    for target in ["TmApp", "HIC"]:
        sub = cv[cv.target == target].copy()
        sub = sub.sort_values(["cv_worst_mae", "cv_mean_mae", "recipe_id", "regressor"])
        picked = []
        used_recipes = set()
        for _, row in sub.iterrows():
            # Prefer diversity of recipe_id; allow second regressor of same recipe only if <3 and helpful
            key = row.recipe_id
            if key in used_recipes and len([p for p in picked if p["recipe_id"] == key]) >= 1:
                # skip duplicate recipe unless we have <3 and next unique exhausted
                continue
            picked.append(row.to_dict())
            used_recipes.add(key)
            if len(picked) >= 3:
                break
        # If fewer than 3 due to diversity, fill from remaining best
        if len(picked) < 3:
            for _, row in sub.iterrows():
                if any(p["recipe_id"] == row.recipe_id and p["regressor"] == row.regressor for p in picked):
                    continue
                picked.append(row.to_dict())
                if len(picked) >= 3:
                    break
        # serialize
        ser = []
        for i, p in enumerate(picked, 1):
            ser.append(
                {
                    "cv_rank": i,
                    "recipe_id": p["recipe_id"],
                    "regressor": p["regressor"],
                    "preprocessing": p["preprocessing"],
                    "cv_primary_mae": p["cv_primary_mae"],
                    "cv_shadow_mae": p["cv_shadow_mae"],
                    "cv_mean_mae": p["cv_mean_mae"],
                    "cv_worst_mae": p["cv_worst_mae"],
                    "feature_blocks": p["feature_blocks"],
                }
            )
        out["targets"][target] = ser
    return out


def fit_predict_test_ridge(parts, y_dev, alpha, test_ids):
    """Full-DEV Ridge fit with protocol matching CV; predict test."""
    if parts["mode"] == "standalone":
        Xdev = parts["X"]
        if Xdev.shape[1] == 0:
            med = float(y_dev.loc[parts["ids"]].median())
            return pd.Series(med, index=test_ids), {"alpha": None, "n_nonzero": None, "sparsity": None, "raw_dim": 0}
        # Need test matrix same columns — caller must provide X_all
        raise RuntimeError("use fit_predict_matrix_ridge")
    raise RuntimeError("handled below")


def fit_predict_matrix_ridge(X_dev, y_dev, X_te, alpha, preprocessing_tag: str, parts_mode: str, parts=None):
    """Generic: impute+scale(+optional PCA protocol)+Ridge on DEV, predict TE."""
    ids_dev = list(X_dev.index)
    yy = y_dev.loc[ids_dev]
    if X_dev.shape[1] == 0:
        med = float(yy.median())
        return pd.Series(med, index=list(X_te.index)), {
            "alpha": None,
            "n_nonzero": None,
            "sparsity": None,
            "raw_dim": 0,
            "preprocessing": "fold_tv_median",
        }

    if parts_mode == "base_struct":
        # Fit PCA on struct from DEV only
        B = parts["base"].loc[ids_dev]
        S = parts["struct"].loc[ids_dev]
        Bt = parts["base_te"]
        St = parts["struct_te"]
        med_b = impute_fit(B)
        med_s = impute_fit(S)
        Bdv = impute_apply(B, med_b)
        Sdv = impute_apply(S, med_s)
        Bte = impute_apply(Bt, med_b)
        Ste = impute_apply(St, med_s)
        Sdv2, [Ste2], _ = pca_block(Sdv, [Ste])
        Xdv = pd.concat([Bdv, Sdv2], axis=1)
        Xte = pd.concat([Bte, Ste2], axis=1)
        prep = "recipe_raw + PCA32(AbLingua) + Ridge"
    elif parts_mode == "abl_blocks":
        R = parts["recipe"].loc[ids_dev]
        A = parts["always"][0].loc[ids_dev]
        E = parts["extra"][0].loc[ids_dev]
        Rt, At, Et = parts["recipe_te"], parts["always_te"], parts["extra_te"]
        med_r, med_a, med_e = impute_fit(R), impute_fit(A), impute_fit(E)
        Rd, Ad, Ed = impute_apply(R, med_r), impute_apply(A, med_a), impute_apply(E, med_e)
        Rte, Ate, Ete = impute_apply(Rt, med_r), impute_apply(At, med_a), impute_apply(Et, med_e)
        Ad2, [Ate2], _ = pca_block(Ad, [Ate])
        Ed2, [Ete2], _ = pca_block(Ed, [Ete])
        Ad2 = Ad2.rename(columns={c: f"abl0_{c}" for c in Ad2.columns})
        Ate2 = Ate2.rename(columns={c: f"abl0_{c}" for c in Ate2.columns})
        Ed2 = Ed2.rename(columns={c: f"abl1_{c}" for c in Ed2.columns})
        Ete2 = Ete2.rename(columns={c: f"abl1_{c}" for c in Ete2.columns})
        Xdv = pd.concat([Rd, Ad2, Ed2], axis=1)
        Xte = pd.concat([Rte, Ate2, Ete2], axis=1)
        prep = "recipe_raw + PCA32_per_abl_block + Ridge"
    else:
        med = impute_fit(X_dev)
        Xdv = impute_apply(X_dev, med)
        Xte = impute_apply(X_te, med)
        prep = "impute+StandardScaler+Ridge(raw)"

    sc = StandardScaler()
    Z = sc.fit_transform(np.asarray(Xdv, float))
    m = Ridge(alpha=float(alpha), random_state=0)
    m.fit(Z, np.asarray(yy, float))
    pred = pd.Series(m.predict(sc.transform(np.asarray(Xte, float))), index=Xte.index)
    return pred, {"alpha": float(alpha), "n_nonzero": None, "sparsity": None, "raw_dim": int(X_dev.shape[1]), "preprocessing": prep}


def fit_predict_matrix_lasso(X_dev, y_dev, X_te, alpha):
    art = fit_full_dev_lasso(X_dev, y_dev, list(X_dev.index), alpha=float(alpha))
    pred = predict_with_lasso_artifact(art, X_te, list(X_te.index))
    return pred, {
        "alpha": art["alpha"],
        "n_nonzero": art["n_nonzero"],
        "sparsity": art["sparsity"],
        "raw_dim": art["raw_dim"],
        "preprocessing": "impute+StandardScaler+Lasso(no_PCA)",
    }


def score_pub_priv(pred: pd.Series, y_te: pd.Series, mask: pd.DataFrame):
    m = mask.set_index("id")
    common = [i for i in pred.index if i in y_te.index]
    pub = [i for i in common if bool(m.loc[i, "is_public"])]
    priv = [i for i in common if bool(m.loc[i, "is_private"])]
    return {
        "public_mae": mae(y_te.loc[pub], pred.loc[pub]) if pub else np.nan,
        "private_mae": mae(y_te.loc[priv], pred.loc[priv]) if priv else np.nan,
        "test_mean_mae": mae(y_te.loc[common], pred.loc[common]) if common else np.nan,
        "n_public": len(pub),
        "n_private": len(priv),
    }


def main():
    assert FREEZE_CSV.exists(), "FEATURE_RECIPE_FREEZE missing"
    expected = FREEZE_SHA.read_text().split()[0]
    actual = hashlib.sha256(FREEZE_CSV.read_bytes()).hexdigest()
    assert actual == expected, "FEATURE_RECIPE_FREEZE hash mismatch — do not rescore"

    recipes = pd.read_csv(FREEZE_CSV)
    store = FeatureStore()
    primary, shadow = load_folds()
    y_tm, y_hic = load_y("TmApp"), load_y("HIC")
    dev_ids = [str(x) for x in pd.read_csv(ROOT / "competition/data/distribution/dev.csv")["id"]]
    test_ids = [str(x) for x in pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")["id"]]

    # ---------- CV ----------
    rows = []
    cv_detail = {}
    print("=== CV PHASE ===", flush=True)
    for _, rec in recipes.iterrows():
        rid = rec.recipe_id
        target = rec.target
        blocks = parse_blocks(rec.feature_blocks)
        y = y_tm if target == "TmApp" else y_hic
        print(f"CV {rid}...", flush=True)
        try:
            parts = recipe_parts(store, rid, blocks, dev_ids)
        except Exception as e:
            print(f"  SKIP load {e}", flush=True)
            continue
        for regressor, eval_fn in [("RIDGE", eval_ridge), ("LASSO", eval_lasso)]:
            try:
                rp = eval_fn(parts, y, primary, parts["ids"])
                rs = eval_fn(parts, y, shadow, parts["ids"])
            except LassoConvergenceError as e:
                print(f"  {regressor} CONVERGENCE FAIL: {e}", flush=True)
                rows.append(
                    {
                        "target": target,
                        "recipe_id": rid,
                        "recipe_name": rec.description,
                        "feature_blocks": rec.feature_blocks,
                        "regressor": regressor,
                        "preprocessing": "FAILED",
                        "raw_dim": np.nan,
                        "effective_dim_if_applicable": np.nan,
                        "n_dev": len(parts["ids"]),
                        "cv_primary_mae": np.nan,
                        "cv_shadow_mae": np.nan,
                        "cv_mean_mae": np.nan,
                        "cv_worst_mae": np.nan,
                        "lasso_alpha_median": np.nan,
                        "lasso_nonzero_median": np.nan,
                        "lasso_sparsity_median": np.nan,
                        "historical_status": "LASSO_CONVERGENCE_FAIL",
                        "notes": str(e),
                    }
                )
                continue
            except Exception as e:
                print(f"  {regressor} FAIL: {e}", flush=True)
                continue
            prep = rp.get("preprocessing", "")
            row = {
                "target": target,
                "recipe_id": rid,
                "recipe_name": rec.description,
                "feature_blocks": rec.feature_blocks,
                "regressor": regressor,
                "preprocessing": prep,
                "raw_dim": rp.get("raw_dim"),
                "effective_dim_if_applicable": rp.get("effective_dim"),
                "n_dev": rp.get("N"),
                "cv_primary_mae": rp["mae"],
                "cv_shadow_mae": rs["mae"],
                "cv_mean_mae": float(np.mean([rp["mae"], rs["mae"]])),
                "cv_worst_mae": float(max(rp["mae"], rs["mae"])),
                "lasso_alpha_median": rp.get("alpha_median") if regressor == "LASSO" else np.nan,
                "lasso_nonzero_median": rp.get("n_nonzero_median") if regressor == "LASSO" else np.nan,
                "lasso_sparsity_median": rp.get("sparsity_median") if regressor == "LASSO" else np.nan,
                "historical_status": "CV_COMPUTED",
                "notes": rec.notes,
            }
            rows.append(row)
            cv_detail[(rid, regressor)] = {
                "primary_alphas": list(map(float, rp.get("alphas") or [])),
                "shadow_alphas": list(map(float, rs.get("alphas") or [])),
                "parts_mode": parts["mode"],
                "ids": parts["ids"],
                "blocks": blocks,
            }
            print(
                f"  {regressor}: P={rp['mae']:.4f} S={rs['mae']:.4f} worst={row['cv_worst_mae']:.4f}",
                flush=True,
            )

    cv = pd.DataFrame(rows)
    cv.to_csv(CV_TABLE, index=False)
    print("Wrote", CV_TABLE, len(cv), flush=True)

    # ---------- Top-3 freeze BEFORE Public/Private ----------
    top = select_top3(cv)
    top["canonical_evaluator_version"] = CANONICAL_VERSION
    top["feature_recipe_freeze_sha256"] = actual
    top["selection_time_note"] = "Frozen BEFORE Public/Private evaluation"
    # alpha policy freeze
    alpha_policy = {
        "rule": "median of Primary 5-fold VAL-selected alphas from Simple TVT",
        "note": "Documented and written BEFORE Public/Private scoring begins",
        "feature_recipe_freeze_sha256": actual,
    }
    # attach median alphas per (recipe, regressor)
    final_alphas = {}
    for (rid, reg), d in cv_detail.items():
        als = d["primary_alphas"]
        final_alphas[f"{rid}::{reg}"] = float(np.median(als)) if als else None
    alpha_policy["final_alpha_by_recipe_regressor"] = final_alphas
    ALPHA_POLICY.write_text(json.dumps(alpha_policy, indent=2))
    TOP3.write_text(json.dumps(top, indent=2))
    top_sha = hashlib.sha256(TOP3.read_bytes()).hexdigest()
    (OUT / "CV_SELECTED_RECIPES_FREEZE.sha256").write_text(top_sha + "\n")
    print("Froze Top-3", TOP3, top_sha[:16], flush=True)

    # ---------- Public/Private ----------
    assert TOP3.exists(), "Top-3 freeze missing"
    print("=== PUBLIC/PRIVATE PHASE ===", flush=True)
    mask = load_test_mask()
    yte_tm, yte_hic = load_test_y("TmApp"), load_test_y("HIC")

    # postmortem recipes = all CV rows + mark top3
    top_keys = set()
    for t, lst in top["targets"].items():
        for p in lst:
            top_keys.add((t, p["recipe_id"], p["regressor"]))

    post_rows = []
    for _, row in cv.iterrows():
        rid, reg, target = row.recipe_id, row.regressor, row.target
        blocks = parse_blocks(row.feature_blocks)
        y = y_tm if target == "TmApp" else y_hic
        yte = yte_tm if target == "TmApp" else yte_hic
        key = f"{rid}::{reg}"
        alpha = final_alphas.get(key)
        print(f"PP {rid} {reg} alpha={alpha}", flush=True)

        pub = priv = tmean = np.nan
        lasso_nz = lasso_sp = np.nan
        notes = row.notes
        validity = "AUTHORITATIVE"
        prep = row.preprocessing

        if rid in NO_TEST_FEATURES:
            notes = str(notes) + "; Public/Private SKIPPED: Test MD features unavailable"
            validity = "CV_ONLY_NO_TEST_FEATURES"
        elif pd.isna(row.cv_primary_mae):
            validity = "CV_FAILED"
        else:
            try:
                parts_dev = recipe_parts(store, rid, blocks, dev_ids)
                # Build matching test matrices
                if parts_dev["mode"] == "standalone":
                    X_all, _ = store.matrix_for_recipe(
                        blocks, parts_dev["ids"] + [i for i in test_ids], allow_intersection=True
                    )
                    # Restrict: train on parts_dev ids, predict test intersection
                    use_te = [i for i in test_ids if i in X_all.index]
                    X_dev = X_all.loc[parts_dev["ids"]]
                    X_te = X_all.loc[use_te]
                    if reg == "RIDGE":
                        if alpha is None:
                            # constant
                            pred = pd.Series(float(y.loc[parts_dev["ids"]].median()), index=use_te)
                            meta = {"alpha": None, "n_nonzero": None, "sparsity": None, "preprocessing": prep}
                        else:
                            pred, meta = fit_predict_matrix_ridge(
                                X_dev, y, X_te, alpha, prep, "standalone"
                            )
                    else:
                        if alpha is None:
                            pred = pd.Series(float(y.loc[parts_dev["ids"]].median()), index=use_te)
                            meta = {"alpha": None, "n_nonzero": 0, "sparsity": None, "preprocessing": prep}
                        else:
                            pred, meta = fit_predict_matrix_lasso(X_dev, y, X_te, alpha)
                            lasso_nz, lasso_sp = meta["n_nonzero"], meta["sparsity"]
                elif parts_dev["mode"] == "base_struct":
                    core = [b for b in blocks if b != "AbLingua_HL_mean"]
                    Xc_all, _ = store.matrix_for_recipe(core, parts_dev["ids"] + test_ids)
                    Hl_all = align_feature_block(
                        store.get("AbLingua_HL_mean"), list(Xc_all.index), "AbLingua_HL_mean"
                    )
                    use_te = [i for i in test_ids if i in Xc_all.index]
                    parts = {
                        "base": parts_dev["base"],
                        "struct": parts_dev["struct"],
                        "base_te": Xc_all.loc[use_te],
                        "struct_te": Hl_all.loc[use_te],
                    }
                    if reg == "RIDGE":
                        pred, meta = fit_predict_matrix_ridge(
                            parts_dev["base"], y, parts["base_te"], alpha, prep, "base_struct", parts=parts
                        )
                    else:
                        X_dev = pd.concat([parts_dev["base"], parts_dev["struct"]], axis=1)
                        X_te = pd.concat([parts["base_te"], parts["struct_te"]], axis=1)
                        pred, meta = fit_predict_matrix_lasso(X_dev, y, X_te, alpha)
                        lasso_nz, lasso_sp = meta["n_nonzero"], meta["sparsity"]
                else:  # abl_blocks
                    core = [b for b in blocks if b not in ("AbLingua_HL_mean", "AbLingua_CDR3")]
                    Xc_all, _ = store.matrix_for_recipe(core, parts_dev["ids"] + test_ids)
                    Hl_all = align_feature_block(
                        store.get("AbLingua_HL_mean"), list(Xc_all.index), "AbLingua_HL_mean"
                    )
                    Cd_all = align_feature_block(
                        store.get("AbLingua_CDR3"), list(Xc_all.index), "AbLingua_CDR3"
                    )
                    use_te = [i for i in test_ids if i in Xc_all.index]
                    parts = {
                        "recipe": parts_dev["recipe"],
                        "always": parts_dev["always"],
                        "extra": parts_dev["extra"],
                        "recipe_te": Xc_all.loc[use_te],
                        "always_te": Hl_all.loc[use_te],
                        "extra_te": Cd_all.loc[use_te],
                    }
                    if reg == "RIDGE":
                        pred, meta = fit_predict_matrix_ridge(
                            parts_dev["recipe"], y, parts["recipe_te"], alpha, prep, "abl_blocks", parts=parts
                        )
                    else:
                        X_dev = pd.concat(
                            [parts_dev["recipe"], parts_dev["always"][0], parts_dev["extra"][0]], axis=1
                        )
                        X_te = pd.concat(
                            [parts["recipe_te"], parts["always_te"], parts["extra_te"]], axis=1
                        )
                        pred, meta = fit_predict_matrix_lasso(X_dev, y, X_te, alpha)
                        lasso_nz, lasso_sp = meta["n_nonzero"], meta["sparsity"]

                sc = score_pub_priv(pred, yte, mask)
                pub, priv, tmean = sc["public_mae"], sc["private_mae"], sc["test_mean_mae"]
                pred.to_csv(PRED_DIR / f"{rid}__{reg}.csv", header=["prediction"])
                prep = meta.get("preprocessing", prep)
            except Exception as e:
                notes = str(notes) + f"; PP_FAIL: {e}"
                validity = "PP_FAILED"
                print(f"  PP fail: {e}", flush=True)

        post_rows.append(
            {
                "target": target,
                "recipe_id": rid,
                "recipe_name": row.recipe_name,
                "feature_blocks": row.feature_blocks,
                "regressor": reg,
                "preprocessing": prep,
                "n_dev": row.n_dev,
                "raw_dim": row.raw_dim,
                "cv_primary_mae": row.cv_primary_mae,
                "cv_shadow_mae": row.cv_shadow_mae,
                "cv_mean_mae": row.cv_mean_mae,
                "cv_worst_mae": row.cv_worst_mae,
                "public_mae": pub,
                "private_mae": priv,
                "test_mean_mae": tmean,
                "public_minus_cv_mean": pub - row.cv_mean_mae if pd.notna(pub) else np.nan,
                "private_minus_cv_mean": priv - row.cv_mean_mae if pd.notna(priv) else np.nan,
                "lasso_alpha_final": alpha if reg == "LASSO" else np.nan,
                "lasso_nonzero_final": lasso_nz if reg == "LASSO" else np.nan,
                "lasso_sparsity_final": lasso_sp if reg == "LASSO" else np.nan,
                "cv_selected_top3": (target, rid, reg) in top_keys,
                "historical_role": recipes.loc[recipes.recipe_id == rid, "historical_role"].iloc[0],
                "validity": validity,
                "notes": notes,
            }
        )

    post = pd.DataFrame(post_rows)
    post.to_csv(POST, index=False)
    print("Wrote", POST, len(post), flush=True)
    (OUT / "cv_detail_alphas.json").write_text(json.dumps({f"{a}::{b}": c for (a, b), c in cv_detail.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
