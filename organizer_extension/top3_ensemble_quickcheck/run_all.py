#!/usr/bin/env python3
"""Top-3-only ensemble / stacking quickcheck (organizer postscript).

Reads from top_models_feature_bundle only. Writes exclusively under
organizer_extension/top3_ensemble_quickcheck/.

Examples:
  python organizer_extension/top3_ensemble_quickcheck/run_all.py \\
      --bundle top_models_feature_bundle

  python organizer_extension/top3_ensemble_quickcheck/run_all.py \\
      --bundle top_models_feature_bundle \\
      --solution top_models_feature_bundle/solution.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from core import (  # noqa: E402
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    HISTORICAL_CONTEXT,
    MODELS_BY_TARGET,
    RIDGE_META_ALPHAS,
    SHORT,
    SolutionGuard,
    TOL_PASS,
    all_nonempty_subsets,
    assert_no_bundle_writes,
    convex_mae_weights,
    cv_stats,
    equal_mean_pred,
    ensure_unique_ids,
    fit_ridge_meta,
    import_bundle,
    load_tables,
    mae,
    median3_pred,
    paired_bootstrap_mae_diff,
    predict_ridge_meta,
    rank_key,
    recipe_row,
    ridge_select_alpha,
    score_pp,
    sha_arr,
    snapshot_bundle_mtimes,
    subset_parts,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        default="top_models_feature_bundle",
        help="Path to top_models_feature_bundle",
    )
    ap.add_argument(
        "--solution",
        default=None,
        help="OPTIONAL organizer solution.csv (POSTMORTEM ONLY)",
    )
    ap.add_argument(
        "--outdir",
        default=str(ROOT),
        help="Output root (default: this package directory)",
    )
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument(
        "--reuse-strict-cache",
        action="store_true",
        help="Reuse existing strict_meta_cache if INDEX.json present (faster reruns)",
    )
    return ap.parse_args(argv)


def reproduce_bases(bundle: Path, outdir: Path, rtr, expected, recipes, dev, test, folds):
    base_dir = outdir / "base_predictions"
    base_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    oof_primary = {}
    oof_shadow = {}
    test_pred = {}
    parts_dev_cache = {}
    parts_te_cache = {}
    alpha_cache = {}

    dev_ids = dev["id"].tolist()
    test_ids = test["id"].tolist()
    folds_p = folds[folds.scheme == "primary"][["id", "fold"]].copy()
    folds_s = folds[folds.scheme == "shadow"][["id", "fold"]].copy()

    all_models = MODELS_BY_TARGET["TmApp"] + MODELS_BY_TARGET["HIC"]
    for rid in all_models:
        row = recipe_row(recipes, rid)
        target = row.target
        regressor = row.regressor
        blocks = [b for b in str(row.feature_blocks).split("|") if b]
        print(f"[base] {rid}", flush=True)
        y = dev.set_index("id")[target].astype(float)
        parts_dev = rtr.build_parts(blocks, dev_ids, bundle, rid)
        parts_te = rtr.build_parts(blocks, test_ids, bundle, rid)
        parts_dev_cache[rid] = parts_dev
        parts_te_cache[rid] = parts_te

        rp = rtr.run_cv(parts_dev, y, folds_p, regressor)
        rs = rtr.run_cv(rtr.build_parts(blocks, dev_ids, bundle, rid), y, folds_s, regressor)

        exp = expected[rid]
        d_p = abs(rp["mae"] - exp["expected_primary_mae"])
        d_s = abs(rs["mae"] - exp["expected_shadow_mae"])
        max_diff = max(d_p, d_s)

        alpha = rtr.load_alpha(rid, regressor)
        alpha_cache[rid] = alpha
        pred = rtr.predict_test(parts_dev, parts_te, y, regressor, alpha).reindex(test_ids)
        if pred.isna().any():
            raise RuntimeError(f"NaN test predictions for {rid}")

        oof_p = rp["oof"].reindex(dev_ids)
        oof_s = rs["oof"].reindex(dev_ids)
        oof_primary[rid] = oof_p
        oof_shadow[rid] = oof_s
        test_pred[rid] = pred

        for scheme, oof in (("primary", oof_p), ("shadow", oof_s)):
            pd.DataFrame({"id": oof.index, "prediction": oof.values}).to_csv(
                base_dir / f"{rid}__oof_{scheme}.csv", index=False
            )
        pd.DataFrame({"id": pred.index, "prediction": pred.values}).to_csv(
            base_dir / f"{rid}__test.csv", index=False
        )

        audit_rows.append(
            {
                "target": target,
                "model_id": rid,
                "short_id": SHORT[rid],
                "expected_primary_mae": exp["expected_primary_mae"],
                "reproduced_primary_mae": rp["mae"],
                "expected_shadow_mae": exp["expected_shadow_mae"],
                "reproduced_shadow_mae": rs["mae"],
                "max_difference": max_diff,
                "oof_primary_hash": sha_arr(oof_p.to_numpy(dtype=float)),
                "oof_shadow_hash": sha_arr(oof_s.to_numpy(dtype=float)),
                "test_prediction_hash": sha_arr(pred.to_numpy(dtype=float)),
                "final_alpha": alpha,
                "pass": max_diff <= TOL_PASS,
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(outdir / "BASE_REPRODUCTION_AUDIT.csv", index=False)
    base_pass = bool(audit["pass"].all())
    print(f"BASE_REPRODUCTION = {'PASS' if base_pass else 'FAIL'}", flush=True)
    return {
        "audit": audit,
        "pass": base_pass,
        "oof_primary": oof_primary,
        "oof_shadow": oof_shadow,
        "test_pred": test_pred,
        "parts_dev": parts_dev_cache,
        "parts_te": parts_te_cache,
        "alphas": alpha_cache,
    }


def error_diversity(outdir: Path, bases, dev):
    rows = []
    for target, models in MODELS_BY_TARGET.items():
        y = dev.set_index("id")[target].astype(float)
        ids = y.index.tolist()
        for scheme, oof_map in (
            ("primary", bases["oof_primary"]),
            ("shadow", bases["oof_shadow"]),
        ):
            pairs = list(combinations_pairs(models))
            for m1, m2 in pairs:
                p1 = oof_map[m1].reindex(ids).astype(float)
                p2 = oof_map[m2].reindex(ids).astype(float)
                r1 = y - p1
                r2 = y - p2
                rows.append(
                    {
                        "target": target,
                        "scheme": scheme,
                        "model_a": m1,
                        "model_b": m2,
                        "short_a": SHORT[m1],
                        "short_b": SHORT[m2],
                        "pred_pearson": float(pearsonr(p1, p2)[0]),
                        "residual_pearson": float(pearsonr(r1, r2)[0]),
                        "residual_spearman": float(spearmanr(r1, r2).correlation),
                        "mean_abs_pred_disagreement": float(np.mean(np.abs(p1 - p2))),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "TOP3_ERROR_DIVERSITY.csv", index=False)
    return df


def combinations_pairs(models):
    from itertools import combinations

    return list(combinations(models, 2))


def equal_mean_all(outdir: Path, bases, dev):
    rows = []
    for target, models in MODELS_BY_TARGET.items():
        y = dev.set_index("id")[target].astype(float)
        ids = y.index.tolist()
        for subset in all_nonempty_subsets(models):
            pp = equal_mean_pred(bases["oof_primary"], subset, ids)
            ps = equal_mean_pred(bases["oof_shadow"], subset, ids)
            st = cv_stats(mae(y, pp), mae(y, ps))
            rows.append(
                {
                    "target": target,
                    "subset": "+".join(SHORT[m] for m in subset),
                    "models": "|".join(subset),
                    "n_models": len(subset),
                    **st,
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["target", "cv_worst_mae", "cv_mean_mae", "n_models"]
    ).reset_index(drop=True)
    df.to_csv(outdir / "EQUAL_MEAN_ALL_SUBSETS.csv", index=False)
    return df


def median3(outdir: Path, bases, dev):
    rows = []
    for target, models in MODELS_BY_TARGET.items():
        y = dev.set_index("id")[target].astype(float)
        ids = y.index.tolist()
        pp = median3_pred(bases["oof_primary"], models, ids)
        ps = median3_pred(bases["oof_shadow"], models, ids)
        st = cv_stats(mae(y, pp), mae(y, ps))
        rows.append({"target": target, "method": "MEDIAN3", **st})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "MEDIAN3_RESULT.csv", index=False)
    return df


def _predict_with_alpha(rtr, parts_fit, parts_pred, y_fit, regressor, alpha):
    return rtr.predict_test(parts_fit, parts_pred, y_fit, regressor, alpha)


def build_strict_meta_cache(
    bundle, outdir, rtr, tvt, bases, recipes, dev, folds
):
    """Nested outer protocol: cache X_meta_val / X_meta_test per fold/scheme."""
    cache_dir = outdir / "strict_meta_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {}

    for target, models in MODELS_BY_TARGET.items():
        y_all = dev.set_index("id")[target].astype(float)
        ids = y_all.index.tolist()
        ensure_unique_ids(ids)
        meta[target] = {}
        for scheme in ("primary", "shadow"):
            folds_s = folds[folds.scheme == scheme][["id", "fold"]].copy()
            fold_payload = []
            oof_stack = {m: pd.Series(index=ids, dtype=float) for m in models}
            # Also store VAL preds keyed by id for stacker training rows
            # We aggregate TEST into OOF; VAL is per-fold only.
            for k, tr, va, te in tvt.rotation_splits(folds_s, ids):
                print(f"[strict-meta] {target}/{scheme} fold={k}", flush=True)
                X_val = {}
                X_test = {}
                for rid in models:
                    row = recipe_row(recipes, rid)
                    regressor = row.regressor
                    alpha = bases["alphas"][rid]
                    parts = bases["parts_dev"][rid]
                    parts_tr = subset_parts(parts, tr)
                    parts_va = subset_parts(parts, va)
                    parts_tv = subset_parts(parts, tr + va)
                    parts_te = subset_parts(parts, te)
                    y_tr = y_all.loc[tr]
                    y_tv = y_all.loc[tr + va]
                    # STEP1/2: TRAIN -> VAL
                    p_va = _predict_with_alpha(
                        rtr, parts_tr, parts_va, y_tr, regressor, alpha
                    ).reindex(va)
                    # STEP3/4: TRAIN+VAL -> TEST (frozen canonical alpha)
                    p_te = _predict_with_alpha(
                        rtr, parts_tv, parts_te, y_tv, regressor, alpha
                    ).reindex(te)
                    if p_va.isna().any() or p_te.isna().any():
                        raise RuntimeError(f"NaN nested preds {rid} fold {k}")
                    X_val[rid] = p_va
                    X_test[rid] = p_te
                    oof_stack[rid].loc[te] = p_te.values

                y_va = y_all.loc[va]
                # Hard rule: stacker must not receive TEST labels
                payload = {
                    "fold": k,
                    "train_ids": list(tr),
                    "val_ids": list(va),
                    "test_ids": list(te),
                    "y_val": y_va.to_dict(),
                    # deliberately omit y_test
                    "X_val": {m: X_val[m].to_dict() for m in models},
                    "X_test": {m: X_test[m].to_dict() for m in models},
                }
                fold_payload.append(payload)
                # write fold cache
                val_df = pd.DataFrame({"id": va, "y": y_va.loc[va].values})
                te_df = pd.DataFrame({"id": te})
                for m in models:
                    val_df[m] = X_val[m].loc[va].values
                    te_df[m] = X_test[m].loc[te].values
                val_df.to_csv(
                    cache_dir / f"{target}__{scheme}__fold{k}__val.csv", index=False
                )
                te_df.to_csv(
                    cache_dir / f"{target}__{scheme}__fold{k}__test.csv", index=False
                )

            meta[target][scheme] = {
                "folds": fold_payload,
                "models": models,
                "oof_base_test_level": {m: oof_stack[m] for m in models},
            }
            # save nested base OOF (TEST-level) for audit
            for m in models:
                s = oof_stack[m]
                pd.DataFrame({"id": s.index, "prediction": s.values}).to_csv(
                    cache_dir / f"{target}__{scheme}__{SHORT[m]}__nested_test_oof.csv",
                    index=False,
                )
    # JSON-serializable index (without Series)
    index = {
        t: {
            s: {
                "n_folds": 5,
                "models": meta[t][s]["models"],
            }
            for s in ("primary", "shadow")
        }
        for t in MODELS_BY_TARGET
    }
    (cache_dir / "INDEX.json").write_text(json.dumps(index, indent=2))
    return meta


def run_strict_stackers(outdir, meta, bases, dev):
    rows = []
    oof_store = {}  # (target, method, scheme) -> Series

    for target, models in MODELS_BY_TARGET.items():
        y = dev.set_index("id")[target].astype(float)
        ids = y.index.tolist()
        per_model = []
        for m in models:
            mp = mae(y, bases["oof_primary"][m].reindex(ids))
            ms = mae(y, bases["oof_shadow"][m].reindex(ids))
            per_model.append((m, mp, ms, max(mp, ms), 0.5 * (mp + ms)))
        best_single_p = min(x[1] for x in per_model)
        best_single_s = min(x[2] for x in per_model)
        best_single_worst = min(x[3] for x in per_model)

        for method in ("CONVEX_MAE", "RIDGE_CV_ALPHA", "RIDGE_FIXED_ALPHA_1"):
            fold_info = {"primary": [], "shadow": []}
            oofs = {}
            for scheme in ("primary", "shadow"):
                oof = pd.Series(index=ids, dtype=float)
                for payload in meta[target][scheme]["folds"]:
                    va = payload["val_ids"]
                    te = payload["test_ids"]
                    P_va = np.column_stack(
                        [pd.Series(payload["X_val"][m]).reindex(va).to_numpy(float) for m in models]
                    )
                    P_te = np.column_stack(
                        [pd.Series(payload["X_test"][m]).reindex(te).to_numpy(float) for m in models]
                    )
                    y_va = pd.Series(payload["y_val"]).reindex(va).to_numpy(float)
                    # assert no test labels present
                    assert "y_test" not in payload

                    if method == "CONVEX_MAE":
                        w = convex_mae_weights(P_va, y_va)
                        pred = P_te @ w
                        fold_info[scheme].append(
                            {
                                "fold": payload["fold"],
                                "weights": w.tolist(),
                                "val_mae": float(np.mean(np.abs(y_va - P_va @ w))),
                            }
                        )
                    elif method == "RIDGE_CV_ALPHA":
                        a = ridge_select_alpha(
                            P_va, y_va, alphas=RIDGE_META_ALPHAS, n_splits=3, seed=0
                        )
                        sc, mdl = fit_ridge_meta(P_va, y_va, a)
                        pred = predict_ridge_meta(sc, mdl, P_te)
                        fold_info[scheme].append(
                            {
                                "fold": payload["fold"],
                                "alpha": a,
                                "coef": mdl.coef_.tolist(),
                                "intercept": float(mdl.intercept_),
                            }
                        )
                    else:  # RIDGE_FIXED_ALPHA_1
                        a = 1.0
                        sc, mdl = fit_ridge_meta(P_va, y_va, a)
                        pred = predict_ridge_meta(sc, mdl, P_te)
                        fold_info[scheme].append(
                            {
                                "fold": payload["fold"],
                                "alpha": a,
                                "coef": mdl.coef_.tolist(),
                                "intercept": float(mdl.intercept_),
                            }
                        )
                    oof.loc[te] = pred
                oofs[scheme] = oof
                oof_store[(target, method, scheme)] = oof

            st = cv_stats(mae(y, oofs["primary"]), mae(y, oofs["shadow"]))
            rows.append(
                {
                    "target": target,
                    "method": method,
                    **st,
                    "delta_primary_vs_best_single": best_single_p - st["primary_mae"],
                    "delta_shadow_vs_best_single": best_single_s - st["shadow_mae"],
                    "improvement_worst_vs_best_single": best_single_worst
                    - st["cv_worst_mae"],
                    "primary_fold_weights_or_alpha": json.dumps(fold_info["primary"]),
                    "shadow_fold_weights_or_alpha": json.dumps(fold_info["shadow"]),
                    "notes": "strict nested outer TVT; stacker fit on VAL only",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "STRICT_STACKING_RESULTS.csv", index=False)
    # persist OOF
    oof_dir = outdir / "strict_meta_cache" / "stacker_oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    for (target, method, scheme), s in oof_store.items():
        pd.DataFrame({"id": s.index, "prediction": s.values}).to_csv(
            oof_dir / f"{target}__{method}__{scheme}.csv", index=False
        )
    return df, oof_store


def select_method(equal_df, median_df, stack_df, bases, dev):
    """CV-only selection; prefer simpler when numerically tied."""
    selections = {}
    summary_rows = []

    for target, models in MODELS_BY_TARGET.items():
        y = dev.set_index("id")[target].astype(float)
        ids = y.index.tolist()
        singles = []
        for m in models:
            st = cv_stats(
                mae(y, bases["oof_primary"][m].reindex(ids)),
                mae(y, bases["oof_shadow"][m].reindex(ids)),
            )
            singles.append({"method": f"SINGLE:{SHORT[m]}", "models": m, "n_models": 1, **st})
        best_single = min(singles, key=lambda r: rank_key(r, 1))

        cands = []
        # equal-mean subsets
        for _, r in equal_df[equal_df.target == target].iterrows():
            cands.append(
                {
                    "family": "EQUAL_MEAN",
                    "method": f"EQUAL_MEAN:{r.subset}",
                    "subset": r.subset,
                    "models": r.models,
                    "n_models": int(r.n_models),
                    "primary_mae": r.primary_mae,
                    "shadow_mae": r.shadow_mae,
                    "cv_mean_mae": r.cv_mean_mae,
                    "cv_worst_mae": r.cv_worst_mae,
                }
            )
        # median
        mr = median_df[median_df.target == target].iloc[0]
        cands.append(
            {
                "family": "MEDIAN3",
                "method": "MEDIAN3",
                "subset": "+".join(SHORT[m] for m in models),
                "models": "|".join(models),
                "n_models": 3,
                "primary_mae": mr.primary_mae,
                "shadow_mae": mr.shadow_mae,
                "cv_mean_mae": mr.cv_mean_mae,
                "cv_worst_mae": mr.cv_worst_mae,
            }
        )
        # stacking
        for _, r in stack_df[stack_df.target == target].iterrows():
            # RIDGE_FIXED_ALPHA_1 is diagnostic; include but prefer others on ties via complexity
            complexity = {"CONVEX_MAE": 2, "RIDGE_CV_ALPHA": 3, "RIDGE_FIXED_ALPHA_1": 4}[
                r.method
            ]
            cands.append(
                {
                    "family": "STACKING",
                    "method": r.method,
                    "subset": "ALL3",
                    "models": "|".join(models),
                    "n_models": 3 + complexity,  # prefer simpler families on tie
                    "primary_mae": r.primary_mae,
                    "shadow_mae": r.shadow_mae,
                    "cv_mean_mae": r.cv_mean_mae,
                    "cv_worst_mae": r.cv_worst_mae,
                    "delta_primary_vs_best_single": r.delta_primary_vs_best_single,
                    "delta_shadow_vs_best_single": r.delta_shadow_vs_best_single,
                    "improvement_worst_vs_best_single": r.improvement_worst_vs_best_single,
                }
            )

        # Prefer equal-mean / median when numerically trivial vs stacking
        def sort_key(c):
            # complexity: equal-mean fewer models preferred; median slightly above 3-mean;
            # stacking higher unless clearly better
            fam_pen = {"EQUAL_MEAN": 0, "MEDIAN3": 1, "STACKING": 2}[c["family"]]
            return (
                c["cv_worst_mae"],
                c["cv_mean_mae"],
                fam_pen,
                c["n_models"],
                c["method"],
            )

        best = min(cands, key=sort_key)
        impro_p = best_single["primary_mae"] - best["primary_mae"]
        impro_s = best_single["shadow_mae"] - best["shadow_mae"]
        impro_w = best_single["cv_worst_mae"] - best["cv_worst_mae"]
        both_improve = impro_p > 0 and impro_s > 0
        is_ensemble = not (
            best["family"] == "EQUAL_MEAN" and best["n_models"] == 1
        ) and best["family"] != "SINGLE"

        # Pre-specified qualitative classes (no post-hoc significance cutoff):
        # USEFUL = multi-model and improves both Primary and Shadow (hence worst).
        # WEAK = multi-model improves cv_worst but not both sides, or tiny/mixed.
        # NONE = no robust ensemble gain → do not force ensemble recommendation.
        if (not is_ensemble) or impro_w <= 0:
            verdict = "NO_ENSEMBLE_INCREMENT"
            best = {
                "family": "SINGLE",
                "method": best_single["method"],
                "subset": best_single["method"].split(":")[1],
                "models": best_single["models"],
                "n_models": 1,
                **{
                    k: best_single[k]
                    for k in (
                        "primary_mae",
                        "shadow_mae",
                        "cv_mean_mae",
                        "cv_worst_mae",
                    )
                },
            }
            impro_p = 0.0
            impro_s = 0.0
            impro_w = 0.0
        elif both_improve and impro_w > 0:
            verdict = "ENSEMBLE_USEFUL"
        else:
            verdict = "WEAK_INCREMENT"

        selections[target] = {
            "verdict": verdict,
            "best_method": best,
            "best_single": best_single,
            "delta_primary": impro_p,
            "delta_shadow": impro_s,
            "delta_worst": impro_w,
            "all_candidates_top": sorted(cands, key=sort_key)[:5],
        }
        summary_rows.append(
            {
                "target": target,
                "verdict": verdict,
                "selected_method": best["method"],
                "selected_family": best["family"],
                "selected_subset": best.get("subset"),
                "selected_primary_mae": best["primary_mae"],
                "selected_shadow_mae": best["shadow_mae"],
                "selected_cv_mean_mae": best["cv_mean_mae"],
                "selected_cv_worst_mae": best["cv_worst_mae"],
                "best_single_method": best_single["method"],
                "best_single_primary_mae": best_single["primary_mae"],
                "best_single_shadow_mae": best_single["shadow_mae"],
                "best_single_cv_worst_mae": best_single["cv_worst_mae"],
                "delta_primary": impro_p,
                "delta_shadow": impro_s,
                "delta_worst": impro_w,
            }
        )

    return selections, pd.DataFrame(summary_rows)


def bootstrap_diagnostic(outdir, selections, equal_df, stack_df, oof_store, bases, dev):
    rows = []
    for target, models in MODELS_BY_TARGET.items():
        y = dev.set_index("id")[target].astype(float)
        ids = y.index.tolist()
        sel = selections[target]
        best_single_id = sel["best_single"]["models"]
        if isinstance(best_single_id, str) and "|" not in best_single_id and not best_single_id.startswith("T") and not best_single_id.startswith("H"):
            single_model = best_single_id
        else:
            # resolve SHORT or recipe id
            single_model = sel["best_single"]["models"]
            if single_model in SHORT.values():
                single_model = [k for k, v in SHORT.items() if v == single_model][0]

        # best equal-mean (may be single)
        eq = equal_df[equal_df.target == target].sort_values(
            ["cv_worst_mae", "cv_mean_mae", "n_models"]
        ).iloc[0]
        # best stacker
        st = stack_df[stack_df.target == target].sort_values(
            ["cv_worst_mae", "cv_mean_mae"]
        ).iloc[0]

        def get_ens_oof(family_row, scheme):
            if "EQUAL" in str(family_row.get("method", "")) or "subset" in family_row:
                subset = tuple(family_row["models"].split("|"))
                return equal_mean_pred(
                    bases["oof_primary"] if scheme == "primary" else bases["oof_shadow"],
                    subset,
                    ids,
                )
            method = family_row["method"] if "method" in family_row.index else family_row["method"]
            return oof_store[(target, method, scheme)]

        comparisons = [
            ("best_equal_mean", {"method": f"EQUAL_MEAN:{eq.subset}", "models": eq.models}),
            ("best_strict_stacker", {"method": st.method, "models": "|".join(models)}),
        ]
        # also selected if different
        sm = sel["best_method"]
        comparisons.append(("cv_selected", {"method": sm["method"], "models": sm.get("models", "")}))

        for label, info in comparisons:
            for scheme, oof_map in (
                ("primary", bases["oof_primary"]),
                ("shadow", bases["oof_shadow"]),
            ):
                p_single = oof_map[single_model].reindex(ids)
                method = info["method"]
                if method.startswith("EQUAL_MEAN") or (
                    label == "cv_selected" and sm["family"] == "EQUAL_MEAN"
                ):
                    models_t = tuple(info["models"].split("|"))
                    if label == "cv_selected":
                        models_t = tuple(sm["models"].split("|"))
                    p_ens = equal_mean_pred(oof_map, models_t, ids)
                elif method == "MEDIAN3" or (
                    label == "cv_selected" and sm["family"] == "MEDIAN3"
                ):
                    p_ens = median3_pred(oof_map, models, ids)
                elif method.startswith("SINGLE") or (
                    label == "cv_selected" and sm["family"] == "SINGLE"
                ):
                    mid = single_model if label != "cv_selected" else sm["models"]
                    p_ens = oof_map[mid].reindex(ids)
                else:
                    # stacking uses strict OOF
                    mname = method if label != "cv_selected" else sm["method"]
                    p_ens = oof_store[(target, mname, scheme)].reindex(ids)

                abs_s = np.abs(y.to_numpy() - p_single.to_numpy())
                abs_e = np.abs(y.to_numpy() - p_ens.to_numpy())
                boot = paired_bootstrap_mae_diff(abs_s, abs_e, BOOTSTRAP_N, BOOTSTRAP_SEED)
                rows.append(
                    {
                        "target": target,
                        "comparison": label,
                        "scheme": scheme,
                        "single_model": single_model,
                        "ensemble_method": method if label != "cv_selected" else sm["method"],
                        **boot,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "BOOTSTRAP_DIAGNOSTIC.csv", index=False)
    return df


def final_predictions(
    outdir, selections, bases, meta, stack_df, rtr, recipes, dev, test, solution_guard
):
    """Freeze method by CV, then build full-DEV Test predictions. Solution AFTER freeze."""
    pred_dir = outdir / "final_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    final = {}
    weights_info = {}
    test_ids = test["id"].tolist()
    dev_ids = dev["id"].tolist()

    for target, models in MODELS_BY_TARGET.items():
        sel = selections[target]
        method = sel["best_method"]
        family = method["family"]
        y = dev.set_index("id")[target].astype(float)

        if family == "SINGLE":
            mid = method["models"]
            # Bit-stable: reuse the base Test CSV already written under base_predictions/
            src = outdir / "base_predictions" / f"{mid}__test.csv"
            out_csv = pred_dir / f"{target}__final_test_predictions.csv"
            shutil.copyfile(src, out_csv)
            back = pd.read_csv(out_csv)
            back["id"] = back["id"].astype(str)
            if back["id"].tolist() != list(test_ids):
                # re-order if needed without float reformatting beyond CSV IO
                back = back.set_index("id").reindex(test_ids).reset_index()
                back.columns = ["id", "prediction"]
                back.to_csv(out_csv, index=False, float_format="%.17g")
                back = pd.read_csv(out_csv)
                back["id"] = back["id"].astype(str)
            values = np.ascontiguousarray(
                back["prediction"].to_numpy(dtype=np.float64), dtype=np.float64
            )
            pred = pd.Series(values, index=test_ids)
            final[target] = pred
            ph = sha_arr(values)
            (pred_dir / f"{target}__prediction_hash.txt").write_text(ph + "\n")
            weights_info[target] = {"type": "single", "model": mid, "weight": 1.0}
            continue
        elif family == "EQUAL_MEAN":
            mods = tuple(method["models"].split("|"))
            pred = equal_mean_pred(bases["test_pred"], mods, test_ids)
            w = 1.0 / len(mods)
            weights_info[target] = {
                "type": "equal_mean",
                "models": list(mods),
                "weights": [w] * len(mods),
            }
        elif family == "MEDIAN3":
            pred = median3_pred(bases["test_pred"], models, test_ids)
            weights_info[target] = {"type": "median3", "models": models}
        elif family == "STACKING" and method["method"] == "CONVEX_MAE":
            # Fit on canonical PRIMARY OOF + all DEV labels
            P = np.column_stack(
                [bases["oof_primary"][m].reindex(dev_ids).to_numpy(float) for m in models]
            )
            w = convex_mae_weights(P, y.reindex(dev_ids).to_numpy(float))
            Pte = np.column_stack(
                [bases["test_pred"][m].reindex(test_ids).to_numpy(float) for m in models]
            )
            pred = pd.Series(Pte @ w, index=test_ids)
            weights_info[target] = {
                "type": "convex_mae",
                "models": models,
                "weights": w.tolist(),
                "fit": "canonical_PRIMARY_OOF + all DEV labels",
            }
        elif family == "STACKING":
            # Ridge: median selected alpha across PRIMARY outer folds
            row = stack_df[
                (stack_df.target == target) & (stack_df.method == method["method"])
            ].iloc[0]
            fold_info = json.loads(row.primary_fold_weights_or_alpha)
            alphas = [f["alpha"] for f in fold_info]
            final_alpha = float(np.median(alphas))
            P = np.column_stack(
                [bases["oof_primary"][m].reindex(dev_ids).to_numpy(float) for m in models]
            )
            sc, mdl = fit_ridge_meta(P, y.reindex(dev_ids).to_numpy(float), final_alpha)
            Pte = np.column_stack(
                [bases["test_pred"][m].reindex(test_ids).to_numpy(float) for m in models]
            )
            pred = pd.Series(predict_ridge_meta(sc, mdl, Pte), index=test_ids)
            weights_info[target] = {
                "type": method["method"],
                "models": models,
                "final_alpha": final_alpha,
                "fold_alphas_primary": alphas,
                "coef": mdl.coef_.tolist(),
                "intercept": float(mdl.intercept_),
                "fit": "canonical_PRIMARY_OOF + all DEV labels",
            }
        else:
            raise RuntimeError(family)

        if len(pred) != 162 or pred.isna().any():
            raise RuntimeError(f"bad final pred {target}")
        ids_order = list(test_ids)
        values = np.ascontiguousarray(
            pred.reindex(ids_order).to_numpy(dtype=np.float64), dtype=np.float64
        )
        out_csv = pred_dir / f"{target}__final_test_predictions.csv"
        pd.DataFrame({"id": ids_order, "prediction": values}).to_csv(
            out_csv, index=False, float_format="%.17g"
        )
        # Canonical artifact = CSV on disk; hash the float64 values reloaded from it.
        back = pd.read_csv(out_csv)
        back["id"] = back["id"].astype(str)
        if back["id"].tolist() != ids_order:
            raise RuntimeError(f"id order drift on write for {target}")
        values = np.ascontiguousarray(
            back["prediction"].to_numpy(dtype=np.float64), dtype=np.float64
        )
        pred = pd.Series(values, index=ids_order)
        final[target] = pred
        ph = sha_arr(values)
        (pred_dir / f"{target}__prediction_hash.txt").write_text(ph + "\n")

    # Combined submission from per-target CSV artifacts (single source of truth)
    tm_df = pd.read_csv(pred_dir / "TmApp__final_test_predictions.csv")
    hic_df = pd.read_csv(pred_dir / "HIC__final_test_predictions.csv")
    sub = pd.DataFrame(
        {
            "id": tm_df["id"].astype(str),
            "TmApp": tm_df["prediction"].to_numpy(dtype=float),
            "HIC": hic_df["prediction"].to_numpy(dtype=float),
        }
    )
    sub_path = pred_dir / "final_ensemble_submission.csv"
    sub.to_csv(sub_path, index=False, float_format="%.17g")
    sub2 = pd.read_csv(sub_path)
    # Re-sync final series to CSV-canonical values used for hashing / postmortem
    final["TmApp"] = pd.Series(
        np.ascontiguousarray(pd.read_csv(pred_dir / "TmApp__final_test_predictions.csv")["prediction"].to_numpy(dtype=np.float64)),
        index=test_ids,
    )
    final["HIC"] = pd.Series(
        np.ascontiguousarray(pd.read_csv(pred_dir / "HIC__final_test_predictions.csv")["prediction"].to_numpy(dtype=np.float64)),
        index=test_ids,
    )
    hashes = {
        "TmApp": Path(pred_dir / "TmApp__prediction_hash.txt").read_text().strip(),
        "HIC": Path(pred_dir / "HIC__prediction_hash.txt").read_text().strip(),
        "submission": sha_arr(
            np.ascontiguousarray(sub2[["TmApp", "HIC"]].to_numpy(dtype=np.float64))
        ),
        "n_test": 162,
        "frozen_before_solution": True,
    }
    # Do not rewrite per-target hash txt (already frozen with the target CSV).
    (pred_dir / "PREDICTION_HASHES.json").write_text(json.dumps(hashes, indent=2))
    (pred_dir / "FINAL_WEIGHTS.json").write_text(json.dumps(weights_info, indent=2))

    # NOW enable solution scoring (POSTMORTEM ONLY)
    postmortem = {}
    if solution_guard.configured:
        solution_guard.enable()
        sol = solution_guard.get()
        for target in ("TmApp", "HIC"):
            postmortem[target] = score_pp(final[target], sol, target)
            postmortem[target]["label"] = "POSTMORTEM ONLY"
        (pred_dir / "POSTMORTEM_SCORES.json").write_text(
            json.dumps(postmortem, indent=2)
        )
    return final, weights_info, hashes, postmortem


def write_reports(
    outdir,
    bases,
    diversity,
    equal_df,
    median_df,
    stack_df,
    boot_df,
    summary_df,
    selections,
    weights_info,
    hashes,
    postmortem,
):
    # SUMMARY csv already from summary_df
    summary_df.to_csv(outdir / "TOP3_ENSEMBLE_SUMMARY.csv", index=False)

    sel_json = {
        "BASE_REPRODUCTION": "PASS" if bases["pass"] else "FAIL",
        "selections": {
            t: {
                "verdict": selections[t]["verdict"],
                "best_method": selections[t]["best_method"],
                "best_single": selections[t]["best_single"],
                "delta_primary": selections[t]["delta_primary"],
                "delta_shadow": selections[t]["delta_shadow"],
                "delta_worst": selections[t]["delta_worst"],
            }
            for t in selections
        },
        "final_weights": weights_info,
        "prediction_hashes": hashes,
        "postmortem": postmortem,
        "historical_context": HISTORICAL_CONTEXT,
        "participant_candidate_prepared": any(
            selections[t]["verdict"] in ("ENSEMBLE_USEFUL", "WEAK_INCREMENT")
            for t in selections
        ),
    }
    (outdir / "FINAL_ENSEMBLE_SELECTION.json").write_text(
        json.dumps(sel_json, indent=2, default=str)
    )

    def fmt_div(target, scheme="primary"):
        d = diversity[(diversity.target == target) & (diversity.scheme == scheme)]
        lines = []
        for _, r in d.iterrows():
            lines.append(
                f"- {r.short_a}/{r.short_b}: residual Pearson={r.residual_pearson:.4f}, "
                f"Spearman={r.residual_spearman:.4f}, "
                f"mean|Δpred|={r.mean_abs_pred_disagreement:.4f}"
            )
        return "\n".join(lines)

    lines = []
    lines.append("# Top-3 Ensemble Quickcheck Report（日本語）\n")
    lines.append(
        "参加者配布 Top-3 特徴レシピのみを候補とした、予測レベル・アンサンブル／スタッキングの厳格クイックチェック。\n"
    )
    lines.append(
        f"**BASE_REPRODUCTION = {'PASS' if bases['pass'] else 'FAIL'}**\n"
    )

    lines.append("## 1. Top-3 基本モデルと正準スコア\n")
    for _, r in bases["audit"].iterrows():
        lines.append(
            f"- `{r.model_id}` ({r.short_id}): Primary={r.reproduced_primary_mae:.6f}, "
            f"Shadow={r.reproduced_shadow_mae:.6f}\n"
        )

    lines.append("\n## 2. ペア残差相関（Primary）\n")
    lines.append("### TmApp\n")
    lines.append(fmt_div("TmApp") + "\n")
    lines.append("### HIC\n")
    lines.append(fmt_div("HIC") + "\n")
    lines.append(
        "\nネストしたレシピ構造のため、高い残差相関は事前に期待される。\n"
    )

    lines.append("\n## 3. Equal-mean 全部分集合\n")
    for target in ("TmApp", "HIC"):
        lines.append(f"### {target}\n")
        sub = equal_df[equal_df.target == target].sort_values(
            ["cv_worst_mae", "cv_mean_mae", "n_models"]
        )
        for _, r in sub.iterrows():
            lines.append(
                f"- {r.subset} (n={r.n_models}): P={r.primary_mae:.6f} S={r.shadow_mae:.6f} "
                f"mean={r.cv_mean_mae:.6f} worst={r.cv_worst_mae:.6f}\n"
            )

    def best_eq(target, n=None):
        sub = equal_df[equal_df.target == target]
        if n is not None:
            sub = sub[sub.n_models == n]
        return sub.sort_values(["cv_worst_mae", "cv_mean_mae", "n_models"]).iloc[0]

    lines.append("\n## 4. 2モデル平均は効いたか\n")
    for target in ("TmApp", "HIC"):
        b1 = best_eq(target, 1)
        b2 = best_eq(target, 2)
        lines.append(
            f"- {target}: best-1 worst={b1.cv_worst_mae:.6f} ({b1.subset}) vs "
            f"best-2 worst={b2.cv_worst_mae:.6f} ({b2.subset}); "
            f"Δworst={b1.cv_worst_mae - b2.cv_worst_mae:.6f}\n"
        )

    lines.append("\n## 5. 3モデル平均は効いたか\n")
    for target in ("TmApp", "HIC"):
        b1 = best_eq(target, 1)
        b3 = best_eq(target, 3)
        lines.append(
            f"- {target}: best-1 worst={b1.cv_worst_mae:.6f} vs ALL3 worst={b3.cv_worst_mae:.6f}; "
            f"Δworst={b1.cv_worst_mae - b3.cv_worst_mae:.6f}\n"
        )

    lines.append("\n## 6. Median-of-three\n")
    for _, r in median_df.iterrows():
        lines.append(
            f"- {r.target}: P={r.primary_mae:.6f} S={r.shadow_mae:.6f} "
            f"mean={r.cv_mean_mae:.6f} worst={r.cv_worst_mae:.6f}\n"
        )

    lines.append("\n## 7. Convex MAE stacking\n")
    for _, r in stack_df[stack_df.method == "CONVEX_MAE"].iterrows():
        lines.append(
            f"- {r.target}: P={r.primary_mae:.6f} S={r.shadow_mae:.6f} "
            f"worst={r.cv_worst_mae:.6f} "
            f"(Δworst vs best single={r.improvement_worst_vs_best_single:.6f})\n"
        )

    lines.append("\n## 8. Ridge stacking\n")
    for method in ("RIDGE_CV_ALPHA", "RIDGE_FIXED_ALPHA_1"):
        lines.append(f"### {method}\n")
        for _, r in stack_df[stack_df.method == method].iterrows():
            lines.append(
                f"- {r.target}: P={r.primary_mae:.6f} S={r.shadow_mae:.6f} "
                f"worst={r.cv_worst_mae:.6f} "
                f"(Δworst={r.improvement_worst_vs_best_single:.6f})\n"
            )

    lines.append("\n## 9. Primary / Shadow 一貫性\n")
    for target in ("TmApp", "HIC"):
        s = selections[target]
        lines.append(
            f"- {target}: ΔPrimary={s['delta_primary']:.6f}, ΔShadow={s['delta_shadow']:.6f}, "
            f"both_improve={s['delta_primary'] > 0 and s['delta_shadow'] > 0}\n"
        )

    lines.append("\n## 10. CV選択ベスト\n")
    for target in ("TmApp", "HIC"):
        m = selections[target]["best_method"]
        lines.append(
            f"- {target}: `{m['method']}` ({m['family']}) "
            f"P={m['primary_mae']:.6f} S={m['shadow_mae']:.6f} worst={m['cv_worst_mae']:.6f}\n"
        )

    lines.append("\n## 11. ベスト単体との差分\n")
    for target in ("TmApp", "HIC"):
        s = selections[target]
        lines.append(
            f"- {target}: ΔP={s['delta_primary']:.6f} ΔS={s['delta_shadow']:.6f} "
            f"Δworst={s['delta_worst']:.6f}\n"
        )

    lines.append("\n## 12. Bootstrap 安定性診断\n")
    for target in ("TmApp", "HIC"):
        sub = boot_df[(boot_df.target == target) & (boot_df.comparison == "cv_selected")]
        for _, r in sub.iterrows():
            lines.append(
                f"- {target}/{r.scheme}: meanΔ={r.mean_improvement:.6f} "
                f"medianΔ={r.median_improvement:.6f} "
                f"95%CI=[{r.ci95_lo:.6f},{r.ci95_hi:.6f}] "
                f"P(Δ>0)={r.p_improvement_gt_0:.3f}\n"
            )

    lines.append("\n## 13. 最終構成 / 重み\n")
    lines.append("```json\n" + json.dumps(weights_info, indent=2) + "\n```\n")

    lines.append("\n## 14. Public / Private（POSTMORTEM ONLY）\n")
    if postmortem:
        for target, sc in postmortem.items():
            lines.append(
                f"- {target}: Public={sc['public_mae']:.6f} Private={sc['private_mae']:.6f} "
                f"Overall={sc['overall_test_mae']:.6f} **[POSTMORTEM ONLY]**\n"
            )
    else:
        lines.append("- solution.csv 未提供のためスキップ\n")

    lines.append("\n## 15. 歴史的オーガナイザー・アンサンブルとの比較（記述のみ）\n")
    hic_h = HISTORICAL_CONTEXT["HIC"]
    lines.append(
        f"- HIC historical `{hic_h['name']}`: CV~{hic_h['cv_primary']}/{hic_h['cv_shadow']}, "
        f"Pub/Priv~{hic_h['public']}/{hic_h['private']}. "
        "Top-3 feature-level Lasso の equal-mean / stacking がこの水準を回収できるかが問い。\n"
    )
    lines.append(
        f"- TmApp: {HISTORICAL_CONTEXT['TmApp']['note']}\n"
    )
    lines.append(
        "- 歴史スコアは新ランキングに混在させない（プロトコル非互換）。\n"
    )

    lines.append("\n## 16. 最終判定\n")
    for target in ("TmApp", "HIC"):
        lines.append(f"- {target}: **{selections[target]['verdict']}**\n")

    lines.append("\n## 17. 参加者バンドルへの後入れ？\n")
    prepare = any(
        selections[t]["verdict"] in ("ENSEMBLE_USEFUL", "WEAK_INCREMENT")
        for t in selections
    )
    lines.append(
        f"- {'YES（candidate を organizer_extension 側に用意。バンドルへのコピーは後続）' if prepare else 'NO（明確な増分なし）'}\n"
    )

    (outdir / "TOP3_ENSEMBLE_REPORT_JA.md").write_text("".join(lines))


def write_readme(outdir: Path, verified_cmd: str, sol_cmd: str):
    text = f"""# Top-3 Ensemble Quickcheck

参加者配布の **Top-3 特徴レシピのみ** を候補とした、予測レベル・アンサンブル／厳格スタッキングのクイックチェックです。

`top_models_feature_bundle/` は **読み取り専用** です。成果物はすべて本ディレクトリに出力します。

## 検証済み再実行コマンド

```bash
{verified_cmd}
```

オーガナイザー・ポストモルテム（solution あり）:

```bash
{sol_cmd}
```

## 出力

| ファイル | 内容 |
|---------|------|
| `BASE_REPRODUCTION_AUDIT.csv` | 6基本モデル再現監査 |
| `TOP3_ERROR_DIVERSITY.csv` | 残差多様性診断 |
| `EQUAL_MEAN_ALL_SUBSETS.csv` | 7部分集合の等重み平均 |
| `MEDIAN3_RESULT.csv` | 3モデル中央値 |
| `STRICT_STACKING_RESULTS.csv` | Convex / Ridge 厳格スタッキング |
| `BOOTSTRAP_DIAGNOSTIC.csv` | ペア・ブートストラップ |
| `TOP3_ENSEMBLE_SUMMARY.csv` | 要約 |
| `FINAL_ENSEMBLE_SELECTION.json` | CV選択・判定 |
| `TOP3_ENSEMBLE_REPORT_JA.md` | 日本語レポート |
| `final_predictions/` | 凍結後 Test 予測 |

## 注意

- `solution.csv` は POSTMORTEM スコア専用。部分集合・重み・alpha・手法選択には不使用。
- 歴史的 SVR/meta blend は候補に含めない（記述比較のみ）。
"""
    (outdir / "README_JA.md").write_text(text)


def prepare_participant_candidate(outdir: Path, selections: dict):
    prepare = any(
        selections[t]["verdict"] in ("ENSEMBLE_USEFUL", "WEAK_INCREMENT")
        for t in selections
    )
    cand = outdir / "participant_candidate"
    cand.mkdir(parents=True, exist_ok=True)
    if not prepare:
        (cand / "STATUS.txt").write_text(
            "NOT_PREPARED: verdict is NO_ENSEMBLE_INCREMENT for useful targets policy\n"
            f"TmApp={selections['TmApp']['verdict']} HIC={selections['HIC']['verdict']}\n"
        )
        return False

    # Always write candidate scripts when WEAK or USEFUL on either target
    script = '''#!/usr/bin/env python3
"""Participant-facing Top-3 ensemble helper (candidate; not yet copied into bundle).

Uses only files distributed in top_models_feature_bundle/.
Supports: equal_mean, median3, convex_stack, ridge_stack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Expect to be run with bundle on PYTHONPATH or from bundle after future integration.
def _import_bundle(bundle: Path):
    sys.path.insert(0, str(bundle))
    import reproduce_top_recipes as rtr
    return rtr


def convex_weights(P, y):
    starts = [np.ones(3)/3, np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]]
    best_w, best = None, np.inf
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    for w0 in starts:
        res = minimize(
            lambda w: float(np.mean(np.abs(y - P @ w))),
            w0, method="SLSQP", bounds=[(0,1)]*3, constraints=cons,
            options={"ftol": 1e-14, "maxiter": 2000, "disp": False},
        )
        w = np.clip(res.x, 0, 1); w = w / w.sum()
        m = float(np.mean(np.abs(y - P @ w)))
        if m < best:
            best, best_w = m, w
    return best_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--target", choices=["TmApp", "HIC"], required=True)
    ap.add_argument("--method", choices=["equal_mean", "median3", "convex_stack", "ridge_stack"], required=True)
    ap.add_argument("--subset", default="ALL", help="e.g. T1+T3 or ALL")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bundle = Path(args.bundle).resolve()
    rtr = _import_bundle(bundle)
    # Reproduce OOF + test via bundle API by shelling into reproduce outputs if present,
    # else instruct user to run reproduce_top_recipes.py first.
    out_pred = bundle / "outputs"
    recipes = pd.read_csv(bundle / "recipes.csv")
    recs = recipes[recipes.target == args.target].sort_values("cv_rank")
    models = recs.recipe_id.tolist()
    short = {m: f"{'T' if args.target=='TmApp' else 'H'}{i+1}" for i, m in enumerate(models)}
    inv = {v: k for k, v in short.items()}

    def load_oof(rid, scheme):
        p = out_pred / f"{rid}__cv_predictions_{scheme}.csv"
        if not p.exists():
            raise SystemExit(f"Missing {p}; run reproduce_top_recipes.py first")
        df = pd.read_csv(p); df["id"] = df["id"].astype(str)
        return df.set_index("id")["prediction"].astype(float)

    def load_test(rid):
        p = out_pred / f"{rid}__test_predictions.csv"
        df = pd.read_csv(p); df["id"] = df["id"].astype(str)
        return df.set_index("id")["prediction"].astype(float)

    if args.subset == "ALL":
        use = models
    else:
        use = [inv[s] if s in inv else s for s in args.subset.split("+")]

    test_ids = pd.read_csv(bundle / "test.csv")["id"].astype(str).tolist()
    if args.method == "equal_mean":
        mats = np.vstack([load_test(m).reindex(test_ids).to_numpy() for m in use])
        pred = mats.mean(axis=0)
    elif args.method == "median3":
        mats = np.vstack([load_test(m).reindex(test_ids).to_numpy() for m in models])
        pred = np.median(mats, axis=0)
    elif args.method == "convex_stack":
        dev = pd.read_csv(bundle / "dev.csv"); dev["id"] = dev["id"].astype(str)
        ids = dev["id"].tolist()
        y = dev.set_index("id")[args.target].astype(float).reindex(ids).to_numpy()
        P = np.column_stack([load_oof(m, "primary").reindex(ids).to_numpy() for m in models])
        w = convex_weights(P, y)
        Pte = np.column_stack([load_test(m).reindex(test_ids).to_numpy() for m in models])
        pred = Pte @ w
        print("weights", w.tolist())
    else:
        dev = pd.read_csv(bundle / "dev.csv"); dev["id"] = dev["id"].astype(str)
        ids = dev["id"].tolist()
        y = dev.set_index("id")[args.target].astype(float).reindex(ids).to_numpy()
        P = np.column_stack([load_oof(m, "primary").reindex(ids).to_numpy() for m in models])
        sc = StandardScaler(); Z = sc.fit_transform(P)
        mdl = Ridge(alpha=args.ridge_alpha, random_state=0).fit(Z, y)
        Pte = np.column_stack([load_test(m).reindex(test_ids).to_numpy() for m in models])
        pred = mdl.predict(sc.transform(Pte))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_ids, "prediction": pred}).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
'''
    (cand / "ensemble_top3.py").write_text(script)
    (cand / "README_JA.md").write_text(
        """# Participant candidate: Top-3 ensemble

このディレクトリは **候補** です。まだ `top_models_feature_bundle/` にはコピーしません。

## 前提

先にバンドル内で:

```bash
python reproduce_top_recipes.py --dev dev.csv --test test.csv --outdir outputs
```

## 使い方

```bash
python ensemble_top3.py --bundle top_models_feature_bundle \\
  --target HIC --method equal_mean --subset H1+H3 --out hic_ens.csv
```

対応 method: `equal_mean`, `median3`, `convex_stack`, `ridge_stack`
"""
    )
    (cand / "README.md").write_text(
        """# Participant candidate: Top-3 ensemble

Candidate only — not yet integrated into `top_models_feature_bundle/`.

Requires prior `reproduce_top_recipes.py` outputs under `outputs/`.

Methods: `equal_mean`, `median3`, `convex_stack`, `ridge_stack`.
"""
    )
    return True


def run_tests(outdir: Path, bundle: Path) -> bool:
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(outdir / "tests"),
        "-q",
        f"--bundle={bundle}",
    ]
    # pytest may not accept --bundle; use env
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["TOP3_QC_BUNDLE"] = str(bundle)
    env["TOP3_QC_OUT"] = str(outdir)
    env["PYTHONPATH"] = str(outdir / "scripts") + ":" + str(bundle) + ":" + env.get(
        "PYTHONPATH", ""
    )
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(outdir / "tests"), "-q"],
        cwd=str(outdir),
        env=env,
    )
    return r.returncode == 0


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle = Path(args.bundle).resolve()
    if not bundle.is_absolute():
        bundle = (Path.cwd() / bundle).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not (bundle / "recipes.csv").exists():
        raise SystemExit(f"bundle not found: {bundle}")

    mtimes0 = snapshot_bundle_mtimes(bundle)
    solution_guard = SolutionGuard()
    if args.solution:
        sol_path = Path(args.solution)
        if not sol_path.is_absolute():
            sol_path = (Path.cwd() / sol_path).resolve()
        if not sol_path.exists():
            raise SystemExit(f"solution not found: {sol_path}")
        solution_guard.load(sol_path)  # path only; contents unread until freeze

    t0 = time.time()
    print(f"bundle={bundle}", flush=True)
    print(f"outdir={outdir}", flush=True)

    rtr, tvt = import_bundle(bundle)
    # Point alpha policy loader at bundle
    rtr.ROOT = bundle
    rtr.ALPHA_POLICY = bundle / "FULL_DEV_ALPHA_POLICY_BUNDLE.json"
    rtr.EXPECTED = bundle / "EXPECTED_SCORES.json"

    dev, test, folds, recipes, expected = load_tables(bundle)

    # Freeze check vs organizer copy if present
    # outdir is .../organizer_extension/top3_ensemble_quickcheck
    repo = outdir.parent.parent
    freeze_path = (
        repo
        / "organizer_extension/linear_model_closure/ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json"
    )
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        for target in ("TmApp", "HIC"):
            fids = [x["model_id"] for x in freeze["targets"][target]]
            assert fids == MODELS_BY_TARGET[target], (fids, MODELS_BY_TARGET[target])

    bases = reproduce_bases(bundle, outdir, rtr, expected, recipes, dev, test, folds)
    if not bases["pass"]:
        print("BASE_REPRODUCTION FAIL — abort", flush=True)
        return 1

    diversity = error_diversity(outdir, bases, dev)
    equal_df = equal_mean_all(outdir, bases, dev)
    median_df = median3(outdir, bases, dev)

    meta = build_strict_meta_cache(
        bundle, outdir, rtr, tvt, bases, recipes, dev, folds
    )
    stack_df, oof_store = run_strict_stackers(outdir, meta, bases, dev)

    selections, summary_df = select_method(equal_df, median_df, stack_df, bases, dev)
    boot_df = bootstrap_diagnostic(
        outdir, selections, equal_df, stack_df, oof_store, bases, dev
    )

    # Final predictions BEFORE solution
    assert not solution_guard.enabled
    final, weights_info, hashes, postmortem = final_predictions(
        outdir,
        selections,
        bases,
        meta,
        stack_df,
        rtr,
        recipes,
        dev,
        test,
        solution_guard,
    )

    write_reports(
        outdir,
        bases,
        diversity,
        equal_df,
        median_df,
        stack_df,
        boot_df,
        summary_df,
        selections,
        weights_info,
        hashes,
        postmortem,
    )
    cand_ok = prepare_participant_candidate(outdir, selections)

    verified = (
        "python organizer_extension/top3_ensemble_quickcheck/run_all.py "
        "--bundle top_models_feature_bundle"
    )
    sol_cmd = (
        "python organizer_extension/top3_ensemble_quickcheck/run_all.py "
        "--bundle top_models_feature_bundle "
        "--solution top_models_feature_bundle/solution.csv"
    )
    write_readme(outdir, verified, sol_cmd)

    assert_no_bundle_writes(bundle, mtimes0)
    if solution_guard.accessed_early:
        raise RuntimeError("solution accessed early")

    if not args.skip_tests:
        ok = run_tests(outdir, bundle)
        print(f"TESTS = {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            return 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s", flush=True)
    print(f"participant_candidate_prepared={cand_ok}", flush=True)
    for t in ("TmApp", "HIC"):
        print(f"{t}: {selections[t]['verdict']} -> {selections[t]['best_method']['method']}")
    return 0


# fix accidental walrus leftover reference
REPO_ROOT = ROOT.parent.parent

if __name__ == "__main__":
    # clean syntax error from mistaken walrus in main — rewrite guard
    sys.exit(main())
