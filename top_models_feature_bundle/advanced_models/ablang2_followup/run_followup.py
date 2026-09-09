#!/usr/bin/env python3
"""Run pre-registered AbLang2 position-aware Transformer follow-up (TmApp)."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_models.config import BUNDLE_ROOT, TM_RECIPES, load_presets
from advanced_models.cv import (
    full_dev_transformer_predict,
    run_transformer_cv,
)
from advanced_models.data import load_dev_test, load_folds, load_residue_bundle, load_solution
from advanced_models.metrics import cv_mean, cv_worst, mae

FOLLOWUP = Path(__file__).resolve().parent
OUT = BUNDLE_ROOT / "advanced_outputs" / "ablang2_followup"
RESULTS = BUNDLE_ROOT / "results" / "ablang2_followup"
PLAN = json.loads((FOLLOWUP / "ABLANG2_FOLLOWUP_PLAN.json").read_text())


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(**kwargs):
    path = FOLLOWUP / "FOLLOWUP_STATUS.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(kwargs)
    cur["timestamp"] = utcnow()
    path.write_text(json.dumps(cur, indent=2) + "\n")
    log = FOLLOWUP / "FOLLOWUP_RUN_LOG.md"
    line = f"\n## {cur['timestamp']} — {cur.get('stage')}\n\n"
    if cur.get("variant"):
        line += f"- variant={cur['variant']} scheme={cur.get('scheme')} rot={cur.get('rotation')} seed={cur.get('seed')}\n"
    line += f"- completed_units={cur.get('completed_units')}/{cur.get('total_units')}\n"
    with log.open("a") as f:
        f.write(line)


def simpler_rank(v: dict) -> tuple:
    """Lower is simpler."""
    ann = 0 if v["annotation_mode"] == "minimal" else 1
    pool = 0 if v.get("pooling_mode", "reg") == "reg" else 1
    merge = {"mean": 0, "concat": 1}.get(v["merge_mode"], 2)
    return (ann, pool, merge, v["variant_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-sequence", action="store_true")
    ap.add_argument("--skip-fusion", action="store_true")
    ap.add_argument("--skip-final-fit", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_status(stage="SEQUENCE", completed_units=0, total_units=120 if not args.quick else 4)

    dev, test = load_dev_test(BUNDLE_ROOT / "dev.csv", BUNDLE_ROOT / "test.csv")
    folds = load_folds()
    rb = load_residue_bundle(dev, test, need_ablang2=True)
    presets = load_presets()
    linear = presets["linear_baselines"]

    variants = PLAN["variants"]
    seq_summaries = []
    fold_seed_rows = []
    region_rows = []
    runtime_adapt = {"batch_size_reductions": []}

    if not args.skip_sequence:
        for v in variants:
            vid = v["variant_id"]
            write_status(stage="SEQUENCE", variant=vid)
            print(f"=== SEQUENCE {vid} ===", flush=True)
            summary = run_transformer_cv(
                target="TmApp",
                content_mode="frozen",
                annotation_mode=v["annotation_mode"],
                merge_mode=v["merge_mode"],
                chain_mode=v["chain_mode"],
                variant_id=vid,
                dev=dev,
                rb=rb,
                folds=folds,
                device=args.device,
                out_dir=OUT,
                quick=args.quick,
                plm_source="ablang2",
                pooling_mode=v["pooling_mode"],
            )
            seq_summaries.append(summary)
            for bs in summary.get("batch_sizes_used") or []:
                if bs < 16:
                    runtime_adapt["batch_size_reductions"].append(
                        {"variant_id": vid, "batch_size": bs}
                    )
            region_rows.extend(summary.get("region_gate_rows") or [])
            # fold/seed detail from cache files
            ch = summary["config_hash"]
            for scheme in ("primary", "shadow"):
                for seed in ([presets["neural"]["seeds"][0]] if args.quick else presets["neural"]["seeds"]):
                    for k in range(1 if args.quick else 5):
                        fp = OUT / f"fold_{vid}_{scheme}_k{k}_s{seed}_{ch}.npz"
                        if not fp.exists():
                            continue
                        z = np.load(fp, allow_pickle=True)
                        fold_seed_rows.append(
                            {
                                "variant_id": vid,
                                "scheme": scheme,
                                "fold": k,
                                "seed": seed,
                                "best_epoch": int(z["best_epoch"]),
                                "n_test": len(z["ids"]),
                            }
                        )
            write_status(
                stage="SEQUENCE",
                variant=vid,
                completed_units=len(seq_summaries) * (2 if args.quick else 30),
            )

        seq_df = pd.DataFrame(
            [
                {
                    "variant_id": s["variant_id"],
                    "type": "frozen_ablang2_transformer",
                    "content_model": "ablang2-paired_residue",
                    "annotation": s["annotation_mode"],
                    "merge": s["merge_mode"],
                    "pooling": s.get("pooling_mode", "reg"),
                    "primary_mae": s["primary_mae"],
                    "shadow_mae": s["shadow_mae"],
                    "cv_mean_mae": s["cv_mean_mae"],
                    "cv_worst_mae": s["cv_worst_mae"],
                    "seed_disp_primary": s["seed_dispersion_primary"],
                    "seed_disp_shadow": s["seed_dispersion_shadow"],
                    "trainable_params": s["n_trainable_parameters"],
                    "config_hash": s["config_hash"],
                    "notes": "",
                }
                for s in seq_summaries
            ]
        )
        # comparisons vs TMF2 / TMS3 / T1 (from authoritative CSVs when present)
        notes = []
        master = BUNDLE_ROOT / "results" / "MODEL_BENCHMARK_SUMMARY.csv"
        if master.exists():
            m = pd.read_csv(master)
            for mid in (
                "TMF2",
                "TMS3",
                "TM_PARENT_ABLINGUA_CDR3__RIDGE",
            ):
                hit = m[m["model_id"].astype(str).str.contains(mid, regex=False)]
                if len(hit):
                    r = hit.iloc[0]
                    notes.append(
                        f"{mid}: P={r.get('primary_mae', r.get('cv_primary_mae', 'NA'))} "
                        f"S={r.get('shadow_mae', r.get('cv_shadow_mae', 'NA'))}"
                    )
        if notes:
            seq_df.loc[0, "notes"] = " | ".join(notes)
        seq_df.to_csv(RESULTS / "ABLANG2_SEQUENCE_RESULTS.csv", index=False)
        pd.DataFrame(fold_seed_rows).to_csv(
            RESULTS / "ABLANG2_SEQUENCE_FOLD_SEED_RESULTS.csv", index=False
        )

        # hypothesis deltas
        by_id = {r["variant_id"]: r for r in seq_df.to_dict("records")}
        hyp = {
            "AL2F1_vs_AL2F2": {
                "delta_primary": by_id["AL2F1_MIN_CONCAT"]["primary_mae"]
                - by_id["AL2F2_FULL_CONCAT"]["primary_mae"],
                "delta_shadow": by_id["AL2F1_MIN_CONCAT"]["shadow_mae"]
                - by_id["AL2F2_FULL_CONCAT"]["shadow_mae"],
                "delta_worst": by_id["AL2F1_MIN_CONCAT"]["cv_worst_mae"]
                - by_id["AL2F2_FULL_CONCAT"]["cv_worst_mae"],
            },
            "AL2F2_vs_AL2F3": {
                "delta_primary": by_id["AL2F2_FULL_CONCAT"]["primary_mae"]
                - by_id["AL2F3_FULL_MEAN"]["primary_mae"],
                "delta_shadow": by_id["AL2F2_FULL_CONCAT"]["shadow_mae"]
                - by_id["AL2F3_FULL_MEAN"]["shadow_mae"],
                "delta_worst": by_id["AL2F2_FULL_CONCAT"]["cv_worst_mae"]
                - by_id["AL2F3_FULL_MEAN"]["cv_worst_mae"],
            },
            "AL2F2_vs_AL2F4": {
                "delta_primary": by_id["AL2F2_FULL_CONCAT"]["primary_mae"]
                - by_id["AL2F4_FULL_REGION_GATE"]["primary_mae"],
                "delta_shadow": by_id["AL2F2_FULL_CONCAT"]["shadow_mae"]
                - by_id["AL2F4_FULL_REGION_GATE"]["shadow_mae"],
                "delta_worst": by_id["AL2F2_FULL_CONCAT"]["cv_worst_mae"]
                - by_id["AL2F4_FULL_REGION_GATE"]["cv_worst_mae"],
            },
        }
        (RESULTS / "ABLANG2_ANNOTATION_HYPOTHESIS.json").write_text(
            json.dumps(hyp, indent=2) + "\n"
        )

        # select best
        ranked = sorted(
            seq_df.to_dict("records"),
            key=lambda r: (
                r["cv_worst_mae"],
                r["cv_mean_mae"],
                simpler_rank(
                    next(v for v in variants if v["variant_id"] == r["variant_id"])
                ),
            ),
        )
        best = ranked[0]
        sel = {
            "BEST_ABLANG2_TRANSFORMER": best["variant_id"],
            "primary_mae": best["primary_mae"],
            "shadow_mae": best["shadow_mae"],
            "cv_mean_mae": best["cv_mean_mae"],
            "cv_worst_mae": best["cv_worst_mae"],
            "selection_policy": "min(cv_worst) then min(cv_mean) then simpler",
            "timestamp": utcnow(),
            "config_hash": best["config_hash"],
        }
        (RESULTS / "ABLANG2_SEQUENCE_SELECTION.json").write_text(
            json.dumps(sel, indent=2) + "\n"
        )
        write_status(stage="SEQUENCE_SELECTED", variant=best["variant_id"])

        if region_rows:
            rg = pd.DataFrame(region_rows)
            rg.to_csv(RESULTS / "ABLANG2_REGION_GATE_WEIGHTS.csv", index=False)
            summ_rows = []
            for scheme in rg["scheme"].unique():
                sub = rg[rg["scheme"] == scheme]
                for chn in ("H", "L"):
                    for rn in ("FR_ALL", "CDR1", "CDR2", "CDR3"):
                        col = f"{chn}_{rn}"
                        summ_rows.append(
                            {
                                "scheme": scheme,
                                "chain": chn,
                                "region": rn,
                                "mean": float(sub[col].mean()),
                                "std": float(sub[col].std(ddof=0)),
                                "median": float(sub[col].median()),
                                "seed_std": float(sub.groupby("seed")[col].mean().std(ddof=0))
                                if sub["seed"].nunique() > 1
                                else 0.0,
                                "fold_std": float(sub.groupby("fold")[col].mean().std(ddof=0))
                                if sub["fold"].nunique() > 1
                                else 0.0,
                            }
                        )
            pd.DataFrame(summ_rows).to_csv(
                RESULTS / "ABLANG2_REGION_GATE_SUMMARY.csv", index=False
            )
    else:
        sel = json.loads((RESULTS / "ABLANG2_SEQUENCE_SELECTION.json").read_text())
        seq_df = pd.read_csv(RESULTS / "ABLANG2_SEQUENCE_RESULTS.csv")
        best = seq_df[seq_df.variant_id == sel["BEST_ABLANG2_TRANSFORMER"]].iloc[0].to_dict()

    best_vid = sel["BEST_ABLANG2_TRANSFORMER"]
    best_v = next(v for v in variants if v["variant_id"] == best_vid)
    # recover best_epochs from cache
    best_hash = best.get("config_hash") or sel.get("config_hash")
    cache = json.loads((OUT / f"cache_{best_vid}_{best_hash}.json").read_text())
    best_epochs = cache["best_epochs_primary"]

    # Fusion
    fusion_summaries = []
    if not args.skip_fusion:
        write_status(stage="FUSION", fusion_completed_units=0, fusion_total_units=90)
        for recipe in PLAN["fusion_recipes"]:
            fvid = f"{best_vid}__FUSION__{recipe.replace('__RIDGE','').replace('__LASSO','')}"
            # keep recipe id in fusion variant naming consistent with prior suite
            fvid = f"{best_vid}__FUSION__{recipe}"
            print(f"=== FUSION {fvid} ===", flush=True)
            summary = run_transformer_cv(
                target="TmApp",
                content_mode="frozen",
                annotation_mode=best_v["annotation_mode"],
                merge_mode=best_v["merge_mode"],
                chain_mode=best_v["chain_mode"],
                variant_id=fvid,
                dev=dev,
                rb=rb,
                folds=folds,
                device=args.device,
                out_dir=OUT,
                quick=args.quick,
                recipe_id=recipe,
                plm_source="ablang2",
                pooling_mode=best_v["pooling_mode"],
            )
            lin = linear[recipe]
            lin_worst = cv_worst(lin["primary"], lin["shadow"])
            fusion_summaries.append(
                {
                    "transformer_variant": best_vid,
                    "fixed_recipe": recipe,
                    "variant_id": fvid,
                    "primary_mae": summary["primary_mae"],
                    "shadow_mae": summary["shadow_mae"],
                    "cv_mean_mae": summary["cv_mean_mae"],
                    "cv_worst_mae": summary["cv_worst_mae"],
                    "seed_disp_primary": summary["seed_dispersion_primary"],
                    "seed_disp_shadow": summary["seed_dispersion_shadow"],
                    "linear_primary": lin["primary"],
                    "linear_shadow": lin["shadow"],
                    "linear_worst": lin_worst,
                    "delta_primary": lin["primary"] - summary["primary_mae"],
                    "delta_shadow": lin["shadow"] - summary["shadow_mae"],
                    "delta_worst": lin_worst - summary["cv_worst_mae"],
                    "config_hash": summary["config_hash"],
                    "best_epochs_primary": summary["best_epochs_primary"],
                    "n_trainable_parameters": summary["n_trainable_parameters"],
                }
            )
            write_status(
                stage="FUSION",
                variant=fvid,
                fusion_completed_units=len(fusion_summaries) * (2 if args.quick else 30),
            )
        fus_df = pd.DataFrame(fusion_summaries)
        fus_df.to_csv(RESULTS / "ABLANG2_FUSION_RESULTS.csv", index=False)
        # fold seed for fusion omitted detailed rebuild; store summary hashes
        fus_df.to_csv(RESULTS / "ABLANG2_FUSION_FOLD_SEED_RESULTS.csv", index=False)
    else:
        fus_df = pd.read_csv(RESULTS / "ABLANG2_FUSION_RESULTS.csv")
        fusion_summaries = fus_df.to_dict("records")

    # Best fusion by CV
    fus_ranked = sorted(
        fusion_summaries,
        key=lambda r: (r["cv_worst_mae"], r["cv_mean_mae"], r["fixed_recipe"]),
    )
    best_fus = fus_ranked[0]

    # Final fit predictions
    if not args.skip_final_fit:
        write_status(stage="ENSEMBLE")
        pred_dir = OUT / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for s in seq_summaries if not args.skip_sequence else []:
            vid = s["variant_id"]
            vv = next(v for v in variants if v["variant_id"] == vid)
            pred = full_dev_transformer_predict(
                target="TmApp",
                content_mode="frozen",
                annotation_mode=vv["annotation_mode"],
                merge_mode=vv["merge_mode"],
                chain_mode=vv["chain_mode"],
                best_epochs_primary=s["best_epochs_primary"],
                dev=dev,
                test=test,
                rb=rb,
                device=args.device,
                quick=args.quick,
                plm_source="ablang2",
                pooling_mode=vv["pooling_mode"],
            )
            pred.to_csv(pred_dir / f"TmApp__seq__{vid}__test.csv", index=False)
        for fs in fusion_summaries:
            vv = best_v
            pred = full_dev_transformer_predict(
                target="TmApp",
                content_mode="frozen",
                annotation_mode=vv["annotation_mode"],
                merge_mode=vv["merge_mode"],
                chain_mode=vv["chain_mode"],
                best_epochs_primary=fs["best_epochs_primary"]
                if isinstance(fs["best_epochs_primary"], dict)
                else json.loads((OUT / f"cache_{fs['variant_id']}_{fs['config_hash']}.json").read_text())[
                    "best_epochs_primary"
                ],
                dev=dev,
                test=test,
                rb=rb,
                device=args.device,
                recipe_id=fs["fixed_recipe"],
                quick=args.quick,
                plm_source="ablang2",
                pooling_mode=vv["pooling_mode"],
            )
            pred.to_csv(pred_dir / f"TmApp__fusion__{fs['variant_id']}__test.csv", index=False)

    (FOLLOWUP / "runtime_adaptations.json").write_text(
        json.dumps(runtime_adapt, indent=2) + "\n"
    )

    # CV selection freeze (before postmortem)
    final_sel = {
        "best_ablang2_sequence": {
            "id": best_vid,
            "primary": float(best["primary_mae"] if "primary_mae" in best else sel["primary_mae"]),
            "shadow": float(best["shadow_mae"] if "shadow_mae" in best else sel["shadow_mae"]),
            "worst": float(best["cv_worst_mae"] if "cv_worst_mae" in best else sel["cv_worst_mae"]),
            "mean": float(best["cv_mean_mae"] if "cv_mean_mae" in best else sel["cv_mean_mae"]),
        },
        "best_ablang2_fusion": {
            "id": best_fus["variant_id"],
            "fixed_recipe": best_fus["fixed_recipe"],
            "primary": float(best_fus["primary_mae"]),
            "shadow": float(best_fus["shadow_mae"]),
            "worst": float(best_fus["cv_worst_mae"]),
            "mean": float(best_fus["cv_mean_mae"]),
        },
        "selection_policy": "CV only: min cv_worst, then cv_mean, then simpler; Public/Private unused",
        "timestamp": utcnow(),
    }
    (RESULTS / "ABLANG2_FINAL_CV_SELECTION.json").write_text(
        json.dumps(final_sel, indent=2) + "\n"
    )
    write_status(stage="POSTMORTEM")
    print("SEQUENCE+FUSION complete; CV selection frozen.", flush=True)
    print(json.dumps(final_sel, indent=2))


if __name__ == "__main__":
    main()
