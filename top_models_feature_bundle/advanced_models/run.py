#!/usr/bin/env python3
"""Single-configuration advanced model runner (bundle-local)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "advanced_models") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advanced_models.config import OUTPUT_ROOT, load_presets, recipe_ids_for_target
from advanced_models.cv import (
    full_dev_transformer_predict,
    full_dev_xgb_predict,
    run_transformer_cv,
    run_xgboost_cv,
)
from advanced_models.data import load_dev_test, load_folds, load_residue_bundle, load_solution
from advanced_models.metrics import score_solution


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run one advanced model configuration")
    p.add_argument("--target", required=True, choices=["TmApp", "HIC"])
    p.add_argument(
        "--model",
        required=True,
        choices=["xgboost", "scratch_transformer", "frozen_transformer", "fusion_transformer"],
    )
    p.add_argument("--recipe", default=None)
    p.add_argument("--annotation-mode", default="full", choices=["minimal", "full"])
    p.add_argument("--merge", default="concat", choices=["concat", "mean", "h_only"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dev", default="dev.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--solution", default=None)
    p.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--variant-id", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle = ROOT
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    dev, test = load_dev_test(bundle / args.dev if not Path(args.dev).is_absolute() else Path(args.dev),
                              bundle / args.test if not Path(args.test).is_absolute() else Path(args.test))
    # resolve relative to CWD if present
    dev_path = Path(args.dev)
    test_path = Path(args.test)
    if not dev_path.exists():
        dev_path = bundle / args.dev
    if not test_path.exists():
        test_path = bundle / args.test
    dev, test = load_dev_test(dev_path, test_path)
    folds = load_folds()

    sol = None
    if args.solution:
        sp = Path(args.solution)
        if not sp.exists():
            sp = bundle / args.solution
        sol = load_solution(sp)

    presets = load_presets()
    chain_mode = "HL" if args.target == "TmApp" else "H_ONLY"
    merge = "h_only" if chain_mode == "H_ONLY" else args.merge

    if args.model == "xgboost":
        recipe = args.recipe or recipe_ids_for_target(args.target)[0]
        summary = run_xgboost_cv(
            target=args.target,
            recipe_id=recipe,
            dev=dev,
            folds=folds,
            device=args.device,
            out_dir=out_dir / "logs",
        )
        pred = full_dev_xgb_predict(
            target=args.target,
            recipe_id=recipe,
            final_preset=summary["final_preset"],
            final_n_estimators=summary["final_n_estimators"],
            dev=dev,
            test=test,
            device=args.device,
        )
    else:
        content = "scratch" if args.model == "scratch_transformer" else "frozen"
        if args.model == "fusion_transformer":
            content = "scratch" if (args.variant_id or "").startswith(("TMS", "HICS")) else "frozen"
        need_abl = content == "frozen" and args.target == "TmApp"
        need_esm = content == "frozen" and args.target == "HIC"
        rb = load_residue_bundle(dev, test, need_ablingua=need_abl, need_esm2=need_esm)
        variant = args.variant_id or f"{args.target}_{args.model}_{args.annotation_mode}_{merge}"
        recipe = args.recipe if args.model == "fusion_transformer" else None
        summary = run_transformer_cv(
            target=args.target,
            content_mode=content,
            annotation_mode=args.annotation_mode,
            merge_mode=merge,
            chain_mode=chain_mode,
            variant_id=variant,
            dev=dev,
            rb=rb,
            folds=folds,
            device=args.device,
            out_dir=out_dir / "logs",
            quick=args.quick,
            recipe_id=recipe,
        )
        pred = full_dev_transformer_predict(
            target=args.target,
            content_mode=content,
            annotation_mode=args.annotation_mode,
            merge_mode=merge,
            chain_mode=chain_mode,
            best_epochs_primary=summary["best_epochs_primary"],
            dev=dev,
            test=test,
            rb=rb,
            device=args.device,
            recipe_id=recipe,
            quick=args.quick,
        )

    pred_path = out_dir / "predictions" / f"{args.target}__{summary['family']}__{summary['variant_id']}__test.csv"
    pred.to_csv(pred_path, index=False)
    if sol is not None:
        pp = score_solution(pred, sol, args.target)
        summary.update(pp)
        summary["notes"] = "POSTMORTEM ONLY public/private"
    else:
        summary["public_mae"] = None
        summary["private_mae"] = None
        summary["overall_test_mae"] = None

    print(json.dumps(summary, indent=2))
    (out_dir / "logs" / f"run_{summary['variant_id']}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
