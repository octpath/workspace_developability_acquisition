#!/usr/bin/env python3
"""Resumable advanced-model benchmark runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advanced_models.config import (
    HIC_RECIPES,
    OUTPUT_ROOT,
    TM_RECIPES,
    load_presets,
)
from advanced_models.cv import (
    full_dev_transformer_predict,
    full_dev_xgb_predict,
    run_transformer_cv,
    run_xgboost_cv,
)
from advanced_models.data import load_dev_test, load_folds, load_residue_bundle, load_solution
from advanced_models.metrics import score_solution, select_best_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        default="all",
        choices=["xgboost", "transformer_sequence", "transformer_fusion", "all"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dev", default="dev.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--solution", default=None)
    p.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    p.add_argument("--quick", action="store_true", help="SMOKE ONLY — not canonical")
    p.add_argument("--write-submissions", action="store_true")
    return p.parse_args(argv)


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    return ROOT / path


def append_result(rows: list, summary: dict, linear_ctx: bool = False):
    presets = load_presets()
    s = dict(summary)
    rid = s.get("recipe_id") or ""
    if rid and rid in presets["linear_baselines"]:
        lin = presets["linear_baselines"][rid]
        s.setdefault("corresponding_linear_primary", lin["primary"])
        s.setdefault("corresponding_linear_shadow", lin["shadow"])
        s.setdefault(
            "delta_primary_vs_linear", lin["primary"] - float(s["primary_mae"])
        )
        s.setdefault(
            "delta_shadow_vs_linear", lin["shadow"] - float(s["shadow_mae"])
        )
    elif linear_ctx:
        # contextual: best linear for target
        target = s["target"]
        keys = TM_RECIPES if target == "TmApp" else HIC_RECIPES
        best = min(keys, key=lambda k: max(presets["linear_baselines"][k]["primary"], presets["linear_baselines"][k]["shadow"]))
        lin = presets["linear_baselines"][best]
        s["corresponding_linear_primary"] = lin["primary"]
        s["corresponding_linear_shadow"] = lin["shadow"]
        s["delta_primary_vs_linear"] = lin["primary"] - float(s["primary_mae"])
        s["delta_shadow_vs_linear"] = lin["shadow"] - float(s["shadow_mae"])
        s["notes"] = (s.get("notes") or "") + f" contextual_linear={best}"
    s.setdefault("public_mae", None)
    s.setdefault("private_mae", None)
    s.setdefault("overall_test_mae", None)
    s.setdefault("selected_by_cv", False)
    rows.append(s)


def stage_xgboost(dev, test, folds, device, out_dir, sol, rows, preds_store):
    for target, recipes in (("TmApp", TM_RECIPES), ("HIC", HIC_RECIPES)):
        for rid in recipes:
            print(f"[xgb] {target} {rid}", flush=True)
            summary = run_xgboost_cv(
                target=target,
                recipe_id=rid,
                dev=dev,
                folds=folds,
                device=device,
                out_dir=out_dir / "logs",
            )
            pred = full_dev_xgb_predict(
                target=target,
                recipe_id=rid,
                final_preset=summary["final_preset"],
                final_n_estimators=summary["final_n_estimators"],
                dev=dev,
                test=test,
                device=device,
            )
            if sol is not None:
                summary.update(score_solution(pred, sol, target))
                summary["notes"] = "POSTMORTEM ONLY"
            append_result(rows, summary)
            preds_store[(target, "xgboost", rid)] = pred
            pred.to_csv(
                out_dir / "predictions" / f"{target}__xgboost__{rid}__test.csv",
                index=False,
            )


def stage_transformer_sequence(
    dev, test, folds, device, out_dir, sol, rows, preds_store, quick
):
    presets = load_presets()
    for target in ("TmApp", "HIC"):
        need_abl = False
        need_esm = False
        # load both if any frozen variant for target
        rb_scratch = load_residue_bundle(dev, test, need_ablingua=False, need_esm2=False)
        rb_frozen = load_residue_bundle(
            dev,
            test,
            need_ablingua=(target == "TmApp"),
            need_esm2=(target == "HIC"),
        )
        chain_mode = "HL" if target == "TmApp" else "H_ONLY"
        for family, content, rb in (
            ("scratch", "scratch", rb_scratch),
            ("frozen", "frozen", rb_frozen),
        ):
            for v in presets["variants"][target][family]:
                print(f"[tf] {v['variant_id']}", flush=True)
                summary = run_transformer_cv(
                    target=target,
                    content_mode=content,
                    annotation_mode=v["annotation_mode"],
                    merge_mode=v["merge"],
                    chain_mode=v["chain_mode"],
                    variant_id=v["variant_id"],
                    dev=dev,
                    rb=rb,
                    folds=folds,
                    device=device,
                    out_dir=out_dir / "logs",
                    quick=quick,
                    recipe_id=None,
                )
                pred = full_dev_transformer_predict(
                    target=target,
                    content_mode=content,
                    annotation_mode=v["annotation_mode"],
                    merge_mode=v["merge"],
                    chain_mode=v["chain_mode"],
                    best_epochs_primary=summary["best_epochs_primary"],
                    dev=dev,
                    test=test,
                    rb=rb,
                    device=device,
                    recipe_id=None,
                    quick=quick,
                )
                if sol is not None:
                    summary.update(score_solution(pred, sol, target))
                    summary["notes"] = (summary.get("notes") or "") + " POSTMORTEM ONLY"
                append_result(rows, summary, linear_ctx=True)
                preds_store[(target, summary["family"], v["variant_id"])] = pred
                pred.to_csv(
                    out_dir
                    / "predictions"
                    / f"{target}__{summary['family']}__{v['variant_id']}__test.csv",
                    index=False,
                )


def _best_seq_variants(df: pd.DataFrame, target: str) -> dict:
    out = {}
    for family in ("scratch_transformer", "frozen_transformer"):
        sub = df[(df.target == target) & (df.family == family) & (df.recipe_id.fillna("") == "")]
        if len(sub) == 0:
            continue
        out[family] = select_best_rows(sub)
    return out


def stage_transformer_fusion(
    dev, test, folds, device, out_dir, sol, rows, preds_store, quick, df_so_far
):
    presets = load_presets()
    for target in ("TmApp", "HIC"):
        winners = _best_seq_variants(df_so_far, target)
        recipes = TM_RECIPES if target == "TmApp" else HIC_RECIPES
        rb = load_residue_bundle(
            dev,
            test,
            need_ablingua=(target == "TmApp"),
            need_esm2=(target == "HIC"),
        )
        # also need scratch without plm
        rb_scratch = load_residue_bundle(dev, test, need_ablingua=False, need_esm2=False)
        for family, content, rb_use in (
            ("scratch_transformer", "scratch", rb_scratch),
            ("frozen_transformer", "frozen", rb),
        ):
            if family not in winners:
                continue
            w = winners[family]
            for rid in recipes:
                vid = f"{w['variant_id']}__FUSION__{rid}"
                print(f"[fusion] {vid}", flush=True)
                summary = run_transformer_cv(
                    target=target,
                    content_mode=content,
                    annotation_mode=w["annotation_mode"],
                    merge_mode=w["merge_mode"],
                    chain_mode=w["chain_mode"],
                    variant_id=vid,
                    dev=dev,
                    rb=rb_use,
                    folds=folds,
                    device=device,
                    out_dir=out_dir / "logs",
                    quick=quick,
                    recipe_id=rid,
                )
                pred = full_dev_transformer_predict(
                    target=target,
                    content_mode=content,
                    annotation_mode=w["annotation_mode"],
                    merge_mode=w["merge_mode"],
                    chain_mode=w["chain_mode"],
                    best_epochs_primary=summary["best_epochs_primary"],
                    dev=dev,
                    test=test,
                    rb=rb_use,
                    device=device,
                    recipe_id=rid,
                    quick=quick,
                )
                if sol is not None:
                    summary.update(score_solution(pred, sol, target))
                    summary["notes"] = (summary.get("notes") or "") + " POSTMORTEM ONLY"
                append_result(rows, summary)
                preds_store[(target, "fusion_transformer", vid)] = pred
                pred.to_csv(
                    out_dir / "predictions" / f"{target}__fusion__{vid}__test.csv",
                    index=False,
                )


def select_and_write(df: pd.DataFrame, preds_store, out_dir: Path, write_submissions: bool):
    selected = {}
    for target in ("TmApp", "HIC"):
        selected[target] = {}
        for family in (
            "xgboost",
            "scratch_transformer",
            "frozen_transformer",
            "fusion_transformer",
        ):
            sub = df[(df.target == target) & (df.family == family)]
            if family == "scratch_transformer" or family == "frozen_transformer":
                sub = sub[sub.recipe_id.fillna("") == ""]
            if len(sub) == 0:
                continue
            best = select_best_rows(sub)
            selected[target][family] = best.to_dict()
            df.loc[best.name, "selected_by_cv"] = True

        # overall best advanced by CV
        sub_all = df[df.target == target]
        if len(sub_all):
            selected[target]["best_overall"] = select_best_rows(sub_all).to_dict()

    (out_dir / "ADVANCED_CV_SELECTED_CONFIGS.json").write_text(
        json.dumps(selected, indent=2, default=str) + "\n"
    )

    if write_submissions:
        sub_dir = out_dir / "submissions"
        sub_dir.mkdir(exist_ok=True)

        def pick_pred(target, family):
            info = selected.get(target, {}).get(family)
            if not info:
                return None
            vid = info["variant_id"]
            rid = info.get("recipe_id") or ""
            # find in preds_store
            for k, v in preds_store.items():
                if k[0] == target and k[1] == family and (vid in k[2] or k[2] == rid or k[2] == vid):
                    return v
            # fallback csv
            matches = list((out_dir / "predictions").glob(f"{target}__{family}*"))
            if family == "fusion_transformer":
                matches = list((out_dir / "predictions").glob(f"{target}__fusion*{vid}*"))
            if matches:
                return pd.read_csv(sorted(matches)[-1])
            return None

        for fam, fname in (
            ("xgboost", "submission_xgboost.csv"),
            ("scratch_transformer", "submission_scratch_transformer.csv"),
            ("frozen_transformer", "submission_frozen_transformer.csv"),
        ):
            tm = pick_pred("TmApp", fam)
            hic = pick_pred("HIC", fam)
            if tm is None or hic is None:
                continue
            m = tm.rename(columns={"prediction": "TmApp"}).merge(
                hic.rename(columns={"prediction": "HIC"}), on="id"
            )
            m = m[["id", "TmApp", "HIC"]]
            assert list(m.columns) == ["id", "TmApp", "HIC"]
            assert len(m) == 162
            m.to_csv(sub_dir / fname, index=False)

        # best_cv overall
        tm = None
        hic = None
        for target, key in (("TmApp", "tm"), ("HIC", "hic")):
            info = selected[target]["best_overall"]
            fam = info["family"]
            pred = pick_pred(target, fam)
            if target == "TmApp":
                tm = pred
            else:
                hic = pred
        if tm is not None and hic is not None:
            m = tm.rename(columns={"prediction": "TmApp"}).merge(
                hic.rename(columns={"prediction": "HIC"}), on="id"
            )[["id", "TmApp", "HIC"]]
            m.to_csv(sub_dir / "submission_best_cv.csv", index=False)
    return selected


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        print("=== SMOKE ONLY — NOT CANONICAL SCORE ===", flush=True)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "submissions").mkdir(exist_ok=True)

    dev, test = load_dev_test(_resolve(args.dev), _resolve(args.test))
    folds = load_folds()
    sol = load_solution(_resolve(args.solution)) if args.solution else None

    results_path = out_dir / "ADVANCED_MODEL_RESULTS.csv"
    if results_path.exists():
        rows = pd.read_csv(results_path).to_dict(orient="records")
    else:
        rows = []
    preds_store = {}

    stages = (
        ["xgboost", "transformer_sequence", "transformer_fusion"]
        if args.stage == "all"
        else [args.stage]
    )

    if "xgboost" in stages:
        stage_xgboost(dev, test, folds, args.device, out_dir, sol, rows, preds_store)
        pd.DataFrame(rows).to_csv(results_path, index=False)

    if "transformer_sequence" in stages:
        stage_transformer_sequence(
            dev, test, folds, args.device, out_dir, sol, rows, preds_store, args.quick
        )
        pd.DataFrame(rows).to_csv(results_path, index=False)

    if "transformer_fusion" in stages:
        df = pd.DataFrame(rows)
        stage_transformer_fusion(
            dev, test, folds, args.device, out_dir, sol, rows, preds_store, args.quick, df
        )
        pd.DataFrame(rows).to_csv(results_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    select_and_write(df, preds_store, out_dir, args.write_submissions or True)
    print(f"Wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
