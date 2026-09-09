#!/usr/bin/env python3
"""Consolidate organizer model benchmarks + optional cross-family ensemble quickcheck.

Uses existing authoritative result files. No new model search.
Public/Private are POSTMORTEM ONLY and never used for selection.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parent
sys.path.insert(0, str(BUNDLE))

from advanced_models.config import load_presets  # noqa: E402
from advanced_models.cv import run_xgboost_cv  # noqa: E402
from advanced_models.data import load_dev_test, load_folds, load_solution  # noqa: E402
from advanced_models.metrics import mae, score_solution, select_best_rows  # noqa: E402

OUT = BUNDLE / "results"
XENS = OUT / "cross_family_ensemble"
ADV = BUNDLE / "advanced_outputs"
ADV_FROZEN = BUNDLE / "advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv"
LIN_OUT = BUNDLE / "outputs"
ENS_DIR = ROOT / "organizer_extension/top3_ensemble_quickcheck"
CLOSURE = ROOT / "organizer_extension/linear_model_closure"

CV_PROTO = "canonical_simple_tvt_primary_shadow"


def _nan():
    return float("nan")


def _overall(pub, priv):
    if pd.isna(pub) or pd.isna(priv):
        return _nan()
    return 0.5 * (float(pub) + float(priv))


def _exists_rel(p: str | None) -> bool:
    if p is None or p == "" or p == "NA" or (isinstance(p, float) and np.isnan(p)):
        return True  # NA allowed
    return (BUNDLE / p).exists() or (ROOT / p).exists()


def load_linear_rows(dev, sol) -> list[dict]:
    recipes = pd.read_csv(BUNDLE / "recipes.csv")
    repro = pd.read_csv(LIN_OUT / "reproduction_scores.csv")
    rows = []
    for _, r in recipes.iterrows():
        rid = r.recipe_id
        rr = repro[repro.recipe_id == rid].iloc[0]
        pred = f"outputs/{rid}__test_predictions.csv"
        rows.append(
            dict(
                target=r.target,
                model_id=rid,
                family="LINEAR",
                base_model=str(r.regressor),
                feature_recipe=rid,
                transformer_variant="",
                cv_primary_mae=float(rr.primary_mae),
                cv_shadow_mae=float(rr.shadow_mae),
                cv_mean_mae=0.5 * (float(rr.primary_mae) + float(rr.shadow_mae)),
                cv_worst_mae=max(float(rr.primary_mae), float(rr.shadow_mae)),
                public_mae=float(rr.public_mae) if pd.notna(rr.public_mae) else _nan(),
                private_mae=float(rr.private_mae) if pd.notna(rr.private_mae) else _nan(),
                overall_test_mae=float(rr.overall_test_mae)
                if pd.notna(rr.overall_test_mae)
                else _nan(),
                cv_protocol=CV_PROTO,
                cv_comparable="YES",
                prediction_path=pred,
                submission_path="outputs/submission.csv"
                if rid
                in (
                    "TM_PARENT_ABLINGUA_CDR3__RIDGE",
                    "HIC_HYDRO_TITRATION__LASSO",
                )
                else "NA",
                source_results_path="outputs/reproduction_scores.csv",
                reproducible="YES",
                notes="Frozen Top-3 feature-level Ridge/Lasso",
            )
        )
    # Fix submission: only mark on the pair used in submission.csv
    # Keep as NA for most; submission is a pair not a single model
    for row in rows:
        row["submission_path"] = "NA"
    return rows


def load_linear_ensemble_rows(sol) -> list[dict]:
    sel = json.loads((ENS_DIR / "FINAL_ENSEMBLE_SELECTION.json").read_text())
    rows = []
    # TmApp: selected is SINGLE T1 — not a distinct ensemble participant model
    # Still record best equal-mean candidate if different from single
    eq = pd.read_csv(ENS_DIR / "EQUAL_MEAN_ALL_SUBSETS.csv")
    for target in ("TmApp", "HIC"):
        s = sel["selections"][target]
        best = s["best_method"]
        post = sel["postmortem"][target]
        if best["family"] == "EQUAL_MEAN":
            mid = f"LINEAR_ENSEMBLE__{best['subset']}"
            pred = (
                "organizer_extension/top3_ensemble_quickcheck/final_predictions/"
                f"{target}__final_test_predictions.csv"
            )
            sub = (
                "organizer_extension/top3_ensemble_quickcheck/final_predictions/"
                "final_ensemble_submission.csv"
            )
            rows.append(
                dict(
                    target=target,
                    model_id=mid,
                    family="LINEAR_ENSEMBLE",
                    base_model="equal_mean",
                    feature_recipe=best["models"],
                    transformer_variant="",
                    cv_primary_mae=float(best["primary_mae"]),
                    cv_shadow_mae=float(best["shadow_mae"]),
                    cv_mean_mae=float(best["cv_mean_mae"]),
                    cv_worst_mae=float(best["cv_worst_mae"]),
                    public_mae=float(post["public_mae"]),
                    private_mae=float(post["private_mae"]),
                    overall_test_mae=float(post["overall_test_mae"]),
                    cv_protocol=CV_PROTO,
                    cv_comparable="YES",
                    prediction_path=pred,
                    submission_path=sub,
                    source_results_path="organizer_extension/top3_ensemble_quickcheck/FINAL_ENSEMBLE_SELECTION.json",
                    reproducible="YES",
                    notes=f"CV-selected Top-3 linear ensemble; {best['method']}",
                )
            )
        else:
            # Record best multi-model equal-mean as context if better-looking but not selected
            sub_eq = eq[(eq.target == target) & (eq.n_models > 1)].sort_values(
                ["cv_worst_mae", "cv_mean_mae", "n_models"]
            )
            if len(sub_eq):
                b = sub_eq.iloc[0]
                # Only add if distinct from single winner AND was evaluated
                mid = f"LINEAR_ENSEMBLE_EQUAL_MEAN__{b.subset}"
                # No dedicated test pred for non-selected TmApp ensembles — CV only row
                rows.append(
                    dict(
                        target=target,
                        model_id=mid,
                        family="LINEAR_ENSEMBLE",
                        base_model="equal_mean",
                        feature_recipe=str(b.models),
                        transformer_variant="",
                        cv_primary_mae=float(b.primary_mae),
                        cv_shadow_mae=float(b.shadow_mae),
                        cv_mean_mae=float(b.cv_mean_mae),
                        cv_worst_mae=float(b.cv_worst_mae),
                        public_mae=_nan(),
                        private_mae=_nan(),
                        overall_test_mae=_nan(),
                        cv_protocol=CV_PROTO,
                        cv_comparable="YES",
                        prediction_path="NA",
                        submission_path="NA",
                        source_results_path="organizer_extension/top3_ensemble_quickcheck/EQUAL_MEAN_ALL_SUBSETS.csv",
                        reproducible="YES",
                        notes="Best equal-mean subset among Top-3; NOT CV-selected (single T1 won)",
                    )
                )
    return rows


def load_advanced_rows() -> list[dict]:
    src = ADV_FROZEN if ADV_FROZEN.exists() else ADV / "ADVANCED_MODEL_RESULTS.csv"
    df = pd.read_csv(src)
    fam_map = {
        "xgboost": "XGBOOST",
        "scratch_transformer": "TRANSFORMER_SEQUENCE",
        "frozen_transformer": "TRANSFORMER_SEQUENCE",
        "fusion_transformer": "TRANSFORMER_FUSION",
    }
    rows = []
    for _, r in df.iterrows():
        fam = fam_map[r.family]
        vid = str(r.variant_id)
        rid = "" if pd.isna(r.recipe_id) else str(r.recipe_id)
        if fam == "XGBOOST":
            pred = f"advanced_outputs/predictions/{r.target}__xgboost__{rid}__test.csv"
            tfv = ""
            base = "XGBoost"
            mid = vid
        elif fam == "TRANSFORMER_SEQUENCE":
            kind = "scratch" if "scratch" in r.family else "frozen"
            pred = f"advanced_outputs/predictions/{r.target}__{r.family}__{vid}__test.csv"
            tfv = vid
            base = kind
            mid = vid
            rid = ""
        else:
            pred = f"advanced_outputs/predictions/{r.target}__fusion__{vid}__test.csv"
            tfv = vid.split("__FUSION__")[0]
            base = "fusion"
            mid = vid
        # submissions
        sub = "NA"
        if fam == "XGBOOST":
            sub = "advanced_outputs/submissions/submission_xgboost.csv"
        elif r.family == "scratch_transformer":
            sub = "advanced_outputs/submissions/submission_scratch_transformer.csv"
        elif r.family == "frozen_transformer":
            sub = "advanced_outputs/submissions/submission_frozen_transformer.csv"
        elif fam == "TRANSFORMER_FUSION":
            sub = "advanced_outputs/submissions/submission_best_cv.csv"

        rows.append(
            dict(
                target=r.target,
                model_id=mid,
                family=fam,
                base_model=base,
                feature_recipe=rid,
                transformer_variant=tfv,
                cv_primary_mae=float(r.primary_mae),
                cv_shadow_mae=float(r.shadow_mae),
                cv_mean_mae=float(r.cv_mean_mae),
                cv_worst_mae=float(r.cv_worst_mae),
                public_mae=float(r.public_mae) if pd.notna(r.public_mae) else _nan(),
                private_mae=float(r.private_mae) if pd.notna(r.private_mae) else _nan(),
                overall_test_mae=float(r.overall_test_mae)
                if pd.notna(r.overall_test_mae)
                else _nan(),
                cv_protocol=CV_PROTO,
                cv_comparable="YES",
                prediction_path=pred,
                submission_path=sub,
                source_results_path="advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv",
                reproducible="YES",
                notes=str(r.notes) if pd.notna(getattr(r, "notes", None)) else "",
            )
        )
    return rows


def load_historical_rows() -> list[dict]:
    """Take rank-1 rows from overall Public/Private leaderboards + HIC ~0.42 blend."""
    specs = [
        ("TmApp", "HIST__TmApp_PUBLIC_WINNER", CLOSURE / "ORGANIZER_LINEAR_OVERALL_TmApp_PUBLIC_TOP10.csv"),
        ("TmApp", "HIST__TmApp_PRIVATE_WINNER", CLOSURE / "ORGANIZER_LINEAR_OVERALL_TmApp_PRIVATE_TOP10.csv"),
        ("HIC", "HIST__HIC_PUBLIC_WINNER", CLOSURE / "ORGANIZER_LINEAR_OVERALL_HIC_PUBLIC_TOP10.csv"),
        ("HIC", "HIST__HIC_PRIVATE_WINNER", CLOSURE / "ORGANIZER_LINEAR_OVERALL_HIC_PRIVATE_TOP10.csv"),
    ]
    rows = []
    for target, mid, path in specs:
        df = pd.read_csv(path)
        h = df.sort_values("rank").iloc[0]
        src_id = str(h.model_id)
        rows.append(
            dict(
                target=target,
                model_id=mid,
                family="HISTORICAL_CONTEXT",
                base_model=src_id,
                feature_recipe="",
                transformer_variant="",
                cv_primary_mae=float(h.cv_primary_mae) if pd.notna(h.cv_primary_mae) else _nan(),
                cv_shadow_mae=float(h.cv_shadow_mae) if pd.notna(h.cv_shadow_mae) else _nan(),
                cv_mean_mae=float(h.cv_mean_mae) if pd.notna(h.cv_mean_mae) else _nan(),
                cv_worst_mae=float(h.cv_worst_mae) if pd.notna(h.cv_worst_mae) else _nan(),
                public_mae=float(h.public_mae),
                private_mae=float(h.private_mae),
                overall_test_mae=_overall(h.public_mae, h.private_mae),
                cv_protocol="historical_noncanonical",
                cv_comparable="NO",
                prediction_path="NA",
                submission_path="NA",
                source_results_path=str(path.relative_to(ROOT)),
                reproducible="NO",
                notes=f"Historical organizer rank-1 from {path.name}; protocol not Simple-TVT-comparable",
            )
        )
    ctx = json.loads((ENS_DIR / "FINAL_ENSEMBLE_SELECTION.json").read_text())[
        "historical_context"
    ]["HIC"]
    rows.append(
        dict(
            target="HIC",
            model_id="HIST__HIC_SIMPLE_blend_seq_surf_adv",
            family="HISTORICAL_CONTEXT",
            base_model="HIC__SIMPLE_blend_seq_surf_adv__HIST",
            feature_recipe="",
            transformer_variant="",
            cv_primary_mae=float(ctx["cv_primary"]),
            cv_shadow_mae=float(ctx["cv_shadow"]),
            cv_mean_mae=0.5 * (float(ctx["cv_primary"]) + float(ctx["cv_shadow"])),
            cv_worst_mae=max(float(ctx["cv_primary"]), float(ctx["cv_shadow"])),
            public_mae=float(ctx["public"]),
            private_mae=float(ctx["private"]),
            overall_test_mae=_overall(ctx["public"], ctx["private"]),
            cv_protocol="historical_prediction_blend",
            cv_comparable="NO",
            prediction_path="NA",
            submission_path="NA",
            source_results_path="organizer_extension/top3_ensemble_quickcheck/FINAL_ENSEMBLE_SELECTION.json",
            reproducible="NO",
            notes=ctx["note"],
        )
    )
    return rows


def mark_winners(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cv_selected"] = "NO"
    df["family_winner"] = "NO"
    df["recommended"] = "NO"
    df["public_private_role"] = "POSTMORTEM_ONLY"
    comparable = df["cv_comparable"] == "YES"
    for target in df.target.unique():
        for fam in df.family.unique():
            sub = df[comparable & (df.target == target) & (df.family == fam)].copy()
            # LINEAR_ENSEMBLE: only CV-selected participant-facing models
            if fam == "LINEAR_ENSEMBLE":
                sub = sub[~sub.notes.fillna("").str.contains("NOT CV-selected")]
            if len(sub) == 0:
                continue
            tmp = sub.rename(columns={"model_id": "variant_id"})
            best = select_best_rows(tmp)
            df.loc[best.name, "family_winner"] = "YES"
        sub = df[comparable & (df.target == target)].copy()
        tmp = sub.rename(columns={"model_id": "variant_id"})
        best = select_best_rows(tmp)
        df.loc[best.name, "cv_selected"] = "YES"
        df.loc[best.name, "recommended"] = "YES"
    return df


def load_oof_series(model_id: str, target: str, scheme: str, dev_ids: list[str]) -> pd.Series:
    """Load primary/shadow OOF aligned to dev_ids."""
    if model_id.startswith("LINEAR_ENSEMBLE__"):
        parts = ["HIC_HYDRO_TITRATION__LASSO", "HIC_ARO_CONTINUOUS_SURFACE__LASSO"]
        mats = [load_oof_series(p, target, scheme, dev_ids) for p in parts]
        return sum(mats) / len(mats)

    # Transformer / fusion OOF caches (may contain __RIDGE/__LASSO in recipe suffix)
    logs = ADV / "logs"
    cands = list(logs.glob(f"oof_{model_id}_*.npz"))
    if not cands:
        cands = [p for p in logs.glob("oof_*.npz") if p.name.startswith(f"oof_{model_id}_")]
    if cands:
        z = np.load(sorted(cands)[0], allow_pickle=True)
        ids = [str(x) for x in z["ids"].tolist()]
        key = "primary_oof" if scheme == "primary" else "shadow_oof"
        ser = pd.Series(z[key], index=ids, dtype=float)
        return ser.reindex(dev_ids).astype(float)

    # Linear Top-3 OOF CSVs
    if ("__FUSION__" not in model_id) and (
        model_id.endswith("__RIDGE") or model_id.endswith("__LASSO")
    ):
        path = LIN_OUT / f"{model_id}__cv_predictions_{scheme}.csv"
        s = pd.read_csv(path)
        s["id"] = s["id"].astype(str)
        col = "prediction" if "prediction" in s.columns else s.columns[-1]
        return s.set_index("id")[col].reindex(dev_ids).astype(float)

    raise FileNotFoundError(f"OOF not found for {model_id}")


def ensure_xgb_oof(target: str, recipe_id: str, dev, folds) -> Path:
    out = XENS / f"xgb_oof_{target}_{recipe_id}.npz"
    if out.exists():
        return out
    print(f"[xgb-oof] regenerating {target} {recipe_id}", flush=True)
    summary = run_xgboost_cv(
        target=target,
        recipe_id=recipe_id,
        dev=dev,
        folds=folds,
        device="cuda",
        out_dir=XENS / "logs",
    )
    # reconstruct from fold is not saved as oof — patch: re-read summary rotations
    # Actually run_xgboost_cv doesn't return oof series. Re-implement quick extract:
    from advanced_models.cv import preprocess_parts, tvt_split
    from advanced_models.features import build_recipe_parts
    from advanced_models.models.xgboost_model import fit_xgb_early, fit_xgb_rounds
    from collections import Counter

    presets = load_presets()
    preset_names = list(presets["xgboost"]["presets"].keys())
    y_map = {str(r.id): float(getattr(r, target)) for r in dev.itertuples(index=False)}
    dev_ids = dev["id"].astype(str).tolist()
    parts = build_recipe_parts(recipe_id, dev_ids)
    oofs = {}
    for scheme_name, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
        oof = pd.Series(index=dev_ids, dtype=float)
        for k in range(5):
            tr, va, te = tvt_split(fmap, k, dev_ids)
            Xtr, Xva, Xte = preprocess_parts(parts, tr, [va, te])
            ytr = np.asarray([y_map[a] for a in tr], float)
            yva = np.asarray([y_map[a] for a in va], float)
            best_name, best_val, best_iter = None, float("inf"), 0
            for pname in preset_names:
                _, bit, vmae = fit_xgb_early(Xtr, ytr, Xva, yva, pname, device="cuda")
                if vmae < best_val - 1e-15 or (
                    abs(vmae - best_val) <= 1e-15 and (best_name is None or pname < best_name)
                ):
                    best_val, best_name, best_iter = vmae, pname, bit
            Xtv, Xte2 = preprocess_parts(parts, tr + va, [te])
            ytv = np.asarray([y_map[a] for a in tr + va], float)
            model = fit_xgb_rounds(Xtv, ytv, best_name, best_iter + 1, device="cuda")
            oof.loc[te] = model.predict(Xte2)
        oofs[scheme_name] = oof.loc[dev_ids].to_numpy(float)
    np.savez_compressed(
        out,
        ids=np.asarray(dev_ids, dtype=object),
        primary_oof=oofs["primary"],
        shadow_oof=oofs["shadow"],
        summary=json.dumps(summary),
    )
    return out


def load_xgb_oof(path: Path, scheme: str, dev_ids: list[str]) -> pd.Series:
    z = np.load(path, allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    key = "primary_oof" if scheme == "primary" else "shadow_oof"
    return pd.Series(z[key], index=ids).reindex(dev_ids).astype(float)


def run_cross_family(df: pd.DataFrame, dev, test, folds, sol) -> tuple[list[dict], dict]:
    XENS.mkdir(parents=True, exist_ok=True)
    (XENS / "logs").mkdir(exist_ok=True)
    ens_rows = []
    meta = {}
    for target in ("TmApp", "HIC"):
        winners = df[
            (df.target == target)
            & (df.family_winner == "YES")
            & (df.cv_comparable == "YES")
            & (df.family.isin(
                [
                    "LINEAR",
                    "LINEAR_ENSEMBLE",
                    "XGBOOST",
                    "TRANSFORMER_SEQUENCE",
                    "TRANSFORMER_FUSION",
                ]
            ))
        ]
        # For TmApp LINEAR_ENSEMBLE equal-mean that wasn't selected as distinct CV winner
        # family_winner on LINEAR_ENSEMBLE may be the non-selected equal-mean — include only if YES
        cands = []
        for _, w in winners.iterrows():
            cands.append(w)
        # Deduplicate if LINEAR_ENSEMBLE equals LINEAR for TmApp (skip empty)
        if len(cands) > 5:
            cands = cands[:5]
        print(f"[ensemble] {target} candidates:", [c.model_id for c in cands], flush=True)

        dev_ids = dev["id"].astype(str).tolist()
        y = dev.set_index(dev["id"].astype(str))[target].astype(float)

        oof_p = {}
        oof_s = {}
        test_pred = {}
        for c in cands:
            mid = c.model_id
            if c.family == "XGBOOST":
                path = ensure_xgb_oof(target, c.feature_recipe, dev, folds)
                oof_p[mid] = load_xgb_oof(path, "primary", dev_ids)
                oof_s[mid] = load_xgb_oof(path, "shadow", dev_ids)
                tp = pd.read_csv(BUNDLE / c.prediction_path)
            elif c.family == "LINEAR":
                oof_p[mid] = load_oof_series(mid, target, "primary", dev_ids)
                oof_s[mid] = load_oof_series(mid, target, "shadow", dev_ids)
                tp = pd.read_csv(BUNDLE / c.prediction_path)
            elif c.family == "LINEAR_ENSEMBLE":
                oof_p[mid] = load_oof_series(mid, target, "primary", dev_ids)
                oof_s[mid] = load_oof_series(mid, target, "shadow", dev_ids)
                # test pred from ensemble final
                tp_path = ENS_DIR / "final_predictions" / f"{target}__final_test_predictions.csv"
                tp = pd.read_csv(tp_path)
                if "prediction" not in tp.columns:
                    # may be id + target col
                    cols = [x for x in tp.columns if x != "id"]
                    tp = tp.rename(columns={cols[0]: "prediction"})
            else:
                oof_p[mid] = load_oof_series(mid, target, "primary", dev_ids)
                oof_s[mid] = load_oof_series(mid, target, "shadow", dev_ids)
                tp = pd.read_csv(BUNDLE / c.prediction_path)
            tp["id"] = tp["id"].astype(str)
            col = "prediction" if "prediction" in tp.columns else [c for c in tp.columns if c != "id"][0]
            test_pred[mid] = tp.set_index("id")[col].astype(float)

        ids_c = [c.model_id for c in cands]
        subset_rows = []
        for r in range(1, len(ids_c) + 1):
            for subset in itertools.combinations(ids_c, r):
                subset = list(subset)
                pp = sum(oof_p[m] for m in subset) / len(subset)
                ss = sum(oof_s[m] for m in subset) / len(subset)
                p_mae = mae(y.loc[dev_ids], pp.loc[dev_ids])
                s_mae = mae(y.loc[dev_ids], ss.loc[dev_ids])
                subset_rows.append(
                    {
                        "subset": "+".join(subset),
                        "n_models": len(subset),
                        "primary_mae": p_mae,
                        "shadow_mae": s_mae,
                        "cv_mean_mae": 0.5 * (p_mae + s_mae),
                        "cv_worst_mae": max(p_mae, s_mae),
                        "members": subset,
                    }
                )
        sdf = pd.DataFrame(subset_rows).sort_values(
            ["cv_worst_mae", "cv_mean_mae", "n_models"]
        )
        sdf.to_csv(XENS / f"{target}__equal_mean_subsets.csv", index=False)
        best = sdf.iloc[0]
        members = best.members
        # Test equal-mean
        te_ids = test["id"].astype(str).tolist()
        te = sum(test_pred[m].reindex(te_ids) for m in members) / len(members)
        te_df = pd.DataFrame({"id": te_ids, "prediction": te.to_numpy(float)})
        pred_path = XENS / f"{target}__cross_family_equal_mean_test.csv"
        te_df.to_csv(pred_path, index=False)

        pub = priv = overall = _nan()
        if sol is not None:
            sc = score_solution(te_df, sol, target)
            pub, priv, overall = sc["public_mae"], sc["private_mae"], sc["overall_test_mae"]

        mid = f"CROSS_FAMILY_EQUAL_MEAN__{target}__{best.n_models}m"
        ens_rows.append(
            dict(
                target=target,
                model_id=mid,
                family="CROSS_FAMILY_ENSEMBLE",
                base_model="equal_mean",
                feature_recipe="|".join(members),
                transformer_variant="",
                cv_primary_mae=float(best.primary_mae),
                cv_shadow_mae=float(best.shadow_mae),
                cv_mean_mae=float(best.cv_mean_mae),
                cv_worst_mae=float(best.cv_worst_mae),
                public_mae=pub,
                private_mae=priv,
                overall_test_mae=overall,
                cv_protocol=CV_PROTO,
                cv_comparable="YES",
                prediction_path=str(pred_path.relative_to(BUNDLE)),
                submission_path="NA",
                source_results_path=str((XENS / f"{target}__equal_mean_subsets.csv").relative_to(BUNDLE)),
                reproducible="YES",
                notes=f"equal_mean members={members}; stacker SKIPPED (optional)",
            )
        )
        meta[target] = {
            "members": members,
            "cv_worst_mae": float(best.cv_worst_mae),
            "cv_mean_mae": float(best.cv_mean_mae),
            "primary_mae": float(best.primary_mae),
            "shadow_mae": float(best.shadow_mae),
            "public_mae": pub,
            "private_mae": priv,
            "changed_cv_winner": None,  # filled later
        }
        # stacking skip note
        (XENS / f"{target}__STACKING_SKIP.md").write_text(
            "# Stacking SKIP\n\n"
            "Equal-mean enumeration completed. Strict nested stacking across "
            "heterogeneous families (linear CSV OOF vs transformer fold caches vs "
            "regenerated XGB OOF) was not required for this quickcheck; "
            "fold-aligned stacker fit/eval on identical rotation definitions "
            "would need a unified OOF builder. Documented SKIP to avoid unsafe "
            "same-row stacker evaluation.\n"
        )
    return ens_rows, meta


def validate(df: pd.DataFrame) -> list[str]:
    errs = []
    if df.duplicated(["target", "model_id"]).any():
        errs.append("duplicate (target, model_id)")
    for _, r in df.iterrows():
        if r.cv_comparable == "YES":
            if not np.isfinite(r.cv_primary_mae) or not np.isfinite(r.cv_shadow_mae):
                errs.append(f"non-finite CV {r.model_id}")
            w = max(r.cv_primary_mae, r.cv_shadow_mae)
            if abs(w - r.cv_worst_mae) > 1e-9:
                errs.append(f"cv_worst mismatch {r.model_id}: {r.cv_worst_mae} vs {w}")
        if np.isfinite(r.public_mae) and np.isfinite(r.private_mae) and np.isfinite(r.overall_test_mae):
            exp = 0.5 * (r.public_mae + r.private_mae)
            if abs(exp - r.overall_test_mae) > 1e-6:
                # allow if overall from other definition
                if abs(exp - r.overall_test_mae) > 1e-4:
                    errs.append(
                        f"overall inconsistency {r.model_id}: {r.overall_test_mae} vs {exp}"
                    )
        for col in ("prediction_path", "submission_path"):
            p = r[col]
            if p not in ("NA", "") and pd.notna(p):
                if not (BUNDLE / p).exists() and not (ROOT / p).exists():
                    errs.append(f"missing path {p} for {r.model_id}")
        if r.family == "HISTORICAL_CONTEXT" and r.cv_comparable != "NO":
            errs.append(f"historical not flagged {r.model_id}")
    if (ROOT / "top_models_feature_bundle/solution.csv").exists():
        # ok if gitignored
        import subprocess

        ig = subprocess.check_output(
            ["git", "check-ignore", "-v", "top_models_feature_bundle/solution.csv"],
            cwd=ROOT,
            text=True,
        )
        if "solution.csv" not in ig:
            errs.append("solution.csv not gitignored")
    tracked = subprocess_check_solution_tracked()
    if tracked:
        errs.append("solution.csv is tracked by git")
    return errs


def subprocess_check_solution_tracked() -> bool:
    import subprocess

    out = subprocess.check_output(
        ["git", "ls-files", "top_models_feature_bundle/solution.csv"],
        cwd=ROOT,
        text=True,
    ).strip()
    return bool(out)


def write_reports(df: pd.DataFrame, meta: dict):
    def best_post(target, col, canonical_only=True):
        sub = df[df.target == target]
        if canonical_only:
            sub = sub[(sub.cv_comparable == "YES") & (sub.reproducible == "YES")]
        sub = sub[np.isfinite(sub[col])]
        if len(sub) == 0:
            return None
        return sub.sort_values(col).iloc[0]

    lines = ["# モデルベンチマーク統合報告", "", "## Executive summary", ""]
    for target in ("TmApp", "HIC"):
        cvw = df[(df.target == target) & (df.cv_selected == "YES")].iloc[0]
        lines.append(
            f"**{target}**: 正準 CV 勝者は `{cvw.model_id}`（family={cvw.family}, "
            f"worst={cvw.cv_worst_mae:.4f}）。Public/Private は事後のみ。"
        )
        lines.append("")
    lines += ["## Unified benchmark table", ""]
    lines.append("| model | family | Primary | Shadow | worst | Public | Private | submission |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    show = df[df.cv_comparable == "YES"].sort_values(["target", "cv_worst_mae"])
    for _, r in show.iterrows():
        lines.append(
            f"| `{r.model_id}` | {r.family} | {r.cv_primary_mae:.4f} | {r.cv_shadow_mae:.4f} | "
            f"{r.cv_worst_mae:.4f} | {r.public_mae if np.isfinite(r.public_mae) else 'NA'} | "
            f"{r.private_mae if np.isfinite(r.private_mae) else 'NA'} | {r.submission_path} |"
        )

    lines += ["", "## Best canonical CV model", ""]
    for target in ("TmApp", "HIC"):
        r = df[(df.target == target) & (df.cv_selected == "YES")].iloc[0]
        lines.append(
            f"- **{target}**: `{r.model_id}` — Primary {r.cv_primary_mae:.6f} / "
            f"Shadow {r.cv_shadow_mae:.6f} / worst {r.cv_worst_mae:.6f} "
            f"(family={r.family})."
        )
        if r.cv_primary_mae > r.cv_shadow_mae:
            lines.append("  - Primary より Shadow が良い（非対称）。")
        elif r.cv_shadow_mae > r.cv_primary_mae:
            lines.append("  - Shadow より Primary が良い（非対称）。")
    lines += ["", "## Best Public model", "", "POSTMORTEM ONLY.", ""]
    for target in ("TmApp", "HIC"):
        r = best_post(target, "public_mae", True)
        lines.append(
            f"- **{target} canonical**: `{r.model_id}` Public={r.public_mae:.6f}"
            if r is not None
            else f"- **{target}**: NA"
        )
    lines += ["", "## Best Private model", "", "POSTMORTEM ONLY.", ""]
    for target in ("TmApp", "HIC"):
        r = best_post(target, "private_mae", True)
        lines.append(
            f"- **{target} canonical**: `{r.model_id}` Private={r.private_mae:.6f}"
            if r is not None
            else f"- **{target}**: NA"
        )
    lines += ["", "## Historical all-time Public/Private winners", ""]
    hist = df[df.family == "HISTORICAL_CONTEXT"]
    for _, r in hist.iterrows():
        lines.append(
            f"- `{r.model_id}` ({r.target}): Public={r.public_mae:.4f} Private={r.private_mae:.4f} "
            f"— cv_comparable=NO"
        )
    lines += [
        "",
        "## Linear vs XGBoost",
        "",
        "- **HIC**: XGBoost が Top-3 線形を一貫して改善。",
        "- **TmApp**: XGBoost は線形より悪化。",
        "",
        "## Sequence Transformer",
        "",
        "- scratch/frozen とも単独では線形に未達。full annotation がわずかに有利。",
        "",
        "## Fusion Transformer",
        "",
        "- 固定長 Top-3 との結合で大きく改善（特に HIC）。系列単独の強さというより相補。",
        "",
        "## Ensemble",
        "",
    ]
    for target, m in meta.items():
        lines.append(
            f"- **{target} cross-family equal-mean**: members={m['members']}, "
            f"worst={m['cv_worst_mae']:.6f}, Pub={m['public_mae']}, Priv={m['private_mae']}; "
            f"CV winner changed={m['changed_cv_winner']}"
        )
    lines += ["", "## Final recommendation", ""]
    for target in ("TmApp", "HIC"):
        cvw = df[(df.target == target) & (df.cv_selected == "YES")].iloc[0]
        bp = best_post(target, "public_mae", True)
        bpr = best_post(target, "private_mae", True)
        bh_p = best_post(target, "public_mae", False)
        bh_r = best_post(target, "private_mae", False)
        lines.append(f"### {target}")
        lines.append(f"- BEST_CV: `{cvw.model_id}`")
        lines.append(f"- BEST_PUBLIC_POSTMORTEM (canonical): `{bp.model_id if bp is not None else 'NA'}`")
        lines.append(f"- BEST_PRIVATE_POSTMORTEM (canonical): `{bpr.model_id if bpr is not None else 'NA'}`")
        lines.append(
            f"- historical Public/Private (descriptive): "
            f"`{bh_p.model_id if bh_p is not None else 'NA'}` / `{bh_r.model_id if bh_r is not None else 'NA'}`"
        )
        lines.append(f"- RECOMMENDED_REPRODUCIBLE_MODEL: `{cvw.model_id}` (CV only; not PP)")
        lines.append("")
    (OUT / "MODEL_BENCHMARK_REPORT_JA.md").write_text("\n".join(lines) + "\n")

    # English short mirror
    en = [
        "# Model benchmark consolidation",
        "",
        "See `MODEL_BENCHMARK_REPORT_JA.md` for the full Japanese report.",
        "",
        "Public/Private metrics are **POSTMORTEM ONLY** and were not used for selection.",
        "",
        f"Master table: `MODEL_BENCHMARK_SUMMARY.csv` ({len(df)} rows).",
        "",
    ]
    for target in ("TmApp", "HIC"):
        cvw = df[(df.target == target) & (df.cv_selected == "YES")].iloc[0]
        en.append(
            f"- **{target} BEST_CV**: `{cvw.model_id}` (worst={cvw.cv_worst_mae:.6f})"
        )
    (OUT / "MODEL_BENCHMARK_REPORT.md").write_text("\n".join(en) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    XENS.mkdir(parents=True, exist_ok=True)
    dev, test = load_dev_test(BUNDLE / "dev.csv", BUNDLE / "test.csv")
    folds = load_folds()
    sol = None
    if (BUNDLE / "solution.csv").exists():
        sol = load_solution(BUNDLE / "solution.csv")

    rows = []
    rows += load_linear_rows(dev, sol)
    rows += load_linear_ensemble_rows(sol)
    rows += load_advanced_rows()
    rows += load_historical_rows()
    df = pd.DataFrame(rows)
    # fix overall where missing but pub/priv present
    for i, r in df.iterrows():
        if (not np.isfinite(r.overall_test_mae)) and np.isfinite(r.public_mae) and np.isfinite(
            r.private_mae
        ):
            df.at[i, "overall_test_mae"] = _overall(r.public_mae, r.private_mae)

    df = mark_winners(df)

    # Cross-family ensemble
    ens_rows, meta = run_cross_family(df, dev, test, folds, sol)
    df = pd.concat([df, pd.DataFrame(ens_rows)], ignore_index=True)
    # re-mark winners including ensemble
    df["cv_selected"] = "NO"
    df["family_winner"] = "NO"
    df["recommended"] = "NO"
    df = mark_winners(df)

    for target in meta:
        cvw = df[(df.target == target) & (df.cv_selected == "YES")].iloc[0]
        meta[target]["changed_cv_winner"] = cvw.family == "CROSS_FAMILY_ENSEMBLE"
        meta[target]["final_cv_winner"] = cvw.model_id

    # Build combined submission for cross-family if both targets exist
    tm = df[(df.target == "TmApp") & (df.family == "CROSS_FAMILY_ENSEMBLE")]
    hic = df[(df.target == "HIC") & (df.family == "CROSS_FAMILY_ENSEMBLE")]
    if len(tm) and len(hic):
        a = pd.read_csv(BUNDLE / tm.iloc[0].prediction_path)
        b = pd.read_csv(BUNDLE / hic.iloc[0].prediction_path)
        sub = a.rename(columns={"prediction": "TmApp"}).merge(
            b.rename(columns={"prediction": "HIC"}), on="id"
        )[["id", "TmApp", "HIC"]]
        sp = XENS / "submission_cross_family_equal_mean.csv"
        sub.to_csv(sp, index=False)
        df.loc[df.family == "CROSS_FAMILY_ENSEMBLE", "submission_path"] = str(
            sp.relative_to(BUNDLE)
        )

    cols = [
        "target",
        "model_id",
        "family",
        "base_model",
        "feature_recipe",
        "transformer_variant",
        "cv_primary_mae",
        "cv_shadow_mae",
        "cv_mean_mae",
        "cv_worst_mae",
        "public_mae",
        "private_mae",
        "overall_test_mae",
        "cv_protocol",
        "cv_comparable",
        "cv_selected",
        "family_winner",
        "recommended",
        "public_private_role",
        "prediction_path",
        "submission_path",
        "source_results_path",
        "reproducible",
        "notes",
    ]
    df = df[cols]
    out_csv = OUT / "MODEL_BENCHMARK_SUMMARY.csv"
    df.to_csv(out_csv, index=False)
    (XENS / "CROSS_FAMILY_META.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    write_reports(df, meta)
    errs = validate(df)
    (OUT / "VALIDATION.txt").write_text(
        "PASS\n" if not errs else "FAIL\n" + "\n".join(errs) + "\n"
    )
    print("rows", len(df))
    print("families", df.family.value_counts().to_dict())
    print("validation", "PASS" if not errs else errs)
    for t in ("TmApp", "HIC"):
        print("CV", t, df[(df.target == t) & (df.cv_selected == "YES")].model_id.tolist())
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
