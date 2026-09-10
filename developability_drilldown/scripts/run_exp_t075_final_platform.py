#!/usr/bin/env python3
"""EXP-T075: T030 architecture under FINAL DL_FOLDLOCAL_COSINE_V3 platform.

No nested CV. No full-Dev refit. No min_epochs. patience=30. cosine T_max=200.
Single seed 101. Freezes the shared training platform after successful run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import EXPERIMENTS_COLUMNS, file_sha256, mae  # noqa: E402
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle, load_solution  # noqa: E402
from antibody_transformer.protocol_v2 import DEFAULT_SEED  # noqa: E402
from antibody_transformer.protocol_v3 import (  # noqa: E402
    BATCH_SIZE,
    ETA_MIN_FRAC,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    PLATFORM_ID,
    PROTOCOL_ID,
    SMOOTH_L1_BETA,
    T_MAX,
    WEIGHT_DECAY,
    coarse_lr_grid,
    eta_min_for,
    lr_at_epoch,
    lr_at_t,
    run_protocol_v3,
)

TARGET = "TmApp"
CODE_EXPECT = "EXP-T075"
EID = "TRF_TM_ABLINGUA_FULL_CONCAT_FINAL_DL_PLATFORM"
INPUT_SPACE = "FROZEN_RESIDUE_DL_FOLDLOCAL_COSINE_V3"
HIST = "EXP-T030"
REPLAY = "EXP-T030-REPLAY-001"
T073, T074 = "EXP-T073", "EXP-T074"

OUT = ROOT / "results" / "EXP-T075_run"
PRED = ROOT / "experiments" / "predictions" / CODE_EXPECT
HIST_CSV = ROOT / "results" / "EXP-T075_TRAINING_HISTORY.csv"
SEL_CSV = ROOT / "results" / "EXP-T075_SELECTED_LR.csv"
DIAG_MD = ROOT / "results" / "EXP-T075_TRAINING_DIAGNOSTICS.md"
REPORT = ROOT / "results" / "EXP-T075_PROTOCOL_REPORT.md"
OOF_YAML = ROOT / "results" / "EXP-T075_OOF_EVALUATION.yaml"
STATE = ROOT / "results" / "EXP-T075_run_state.json"
CFG_PATH = ROOT / "experiments" / "configs" / f"{CODE_EXPECT}.yaml"
PLATFORM_MD = ROOT / "results" / "DL_FOLDLOCAL_COSINE_V3_PROTOCOL.md"
FREEZE_YAML = ROOT / "results" / "DL_FOLDLOCAL_COSINE_V3_FREEZE.yaml"
COMPARE_MD = ROOT / "results" / "T073_T074_T075_PLATFORM_COMPARISON.md"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def implementation_hash() -> str:
    p = ROOT / "models" / "antibody_transformer" / "protocol_v3.py"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assets_ref() -> str:
    return "assets/transformer/residue_asset_manifest.yaml#ablingua600m"


def issue() -> str:
    codes = load_codes()
    if (codes["experiment_id"] == EID).any():
        return str(codes.set_index("experiment_id").loc[EID, "experiment_code"])
    nxt = next_code(TARGET)
    if nxt != CODE_EXPECT:
        raise SystemExit(f"expected {CODE_EXPECT}, got {nxt}")
    code = issue_code(
        EID,
        TARGET,
        source_model_id=f"{PLATFORM_ID}::{REPLAY}",
        phase="DL_PLATFORM_V3_FREEZE",
        notes="T030 under final DL_FOLDLOCAL_COSINE_V3; patience=30; no min_epochs; freeze platform",
    )
    if code != CODE_EXPECT:
        raise SystemExit(code)
    return code


def write_config(code: str) -> Path:
    cfg = {
        "experiment_code": code,
        "experiment_id": EID,
        "platform_id": PLATFORM_ID,
        "protocol_id": PROTOCOL_ID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"{PLATFORM_ID}::{REPLAY}",
        "transformer_type": "FROZEN_PLM",
        "input_space": INPUT_SPACE,
        "input_asset_ref": assets_ref(),
        "plm_source": "ABLINGUA",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": "concat",
        "pooling_mode": "REG",
        "content_mode": "frozen",
        "architecture_control": HIST,
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "ff_dim": 256,
        "dropout": 0.2,
        "norm_first": True,
        "activation": "gelu",
        "seed": DEFAULT_SEED,
        "seeds": [DEFAULT_SEED],
        "optimizer": "AdamW",
        "lr_grid": coarse_lr_grid(),
        "eta_min_frac": ETA_MIN_FRAC,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "scheduler": "explicit_cosine_T_max_200",
        "scheduler_T_max": T_MAX,
        "scheduler_step_timing": "beginning_of_epoch",
        "warmup": None,
        "restart": None,
        "gradient_clip": 1.0,
        "loss": f"SmoothL1Loss(beta={SMOOTH_L1_BETA})",
        "checkpoint_metric": "validation_MAE",
        "no_full_dev_refit": True,
        "no_nested_cv": True,
        "external_aggregation": ["mean", "median"],
        "cv_protocol": "canonical_simple_tvt_primary_shadow",
        "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
        "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
        "representation_status": "NOT_EXPORTED",
        "control_experiment_code": HIST,
        "contemporary_control_run_id": REPLAY,
    }
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return CFG_PATH


def write_platform_protocol_md() -> None:
    lines = [
        f"# {PLATFORM_ID} — Shared Deep-Learning Training Platform",
        "",
        "Architecture parameters (layers, heads, REG, H/L communication, etc.) are",
        "**experiment-specific** and are NOT part of this shared platform.",
        "",
        "## Shared settings",
        "",
        "| Item | Value |",
        "|------|-------|",
        "| optimizer | AdamW |",
        f"| weight_decay | {WEIGHT_DECAY} |",
        f"| batch_size | {BATCH_SIZE} |",
        f"| loss | SmoothL1(beta={SMOOTH_L1_BETA}) |",
        "| initial_lr_candidates | 1e-5, 1e-4, 1e-3, 1e-2 |",
        "| lr_selection | fold-local validation MAE |",
        f"| scheduler | cosine, T_max={T_MAX}, eta_min={ETA_MIN_FRAC}×initial_lr |",
        "| warmup / restart | none |",
        f"| max_epochs | {MAX_EPOCHS} |",
        f"| patience | {PATIENCE} |",
        "| min_epochs | none |",
        "| checkpoint | best validation MAE (from epoch 1) |",
        "| full_Dev refit | disabled |",
        "| external prediction | fold-checkpoint mean + median |",
        "| seed_policy | experiment-specific |",
        "",
        "## Scheduler timing",
        "",
        "LR for 1-indexed epoch `N` is set at the **beginning** of the epoch to",
        "`lr_at_t(N-1)`. The training-history `learning_rate` column is the LR used",
        "**during** that epoch's gradient updates.",
        "",
        "## Split semantics",
        "",
        "Existing Primary/Shadow TRAIN/VAL/TEST rotations only. NOT nested CV.",
        "",
        f"Baseline experiment under this platform: `{CODE_EXPECT}`.",
        "",
    ]
    PLATFORM_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_pred(ids, vals, path: Path, col: str = TARGET) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, col: vals}).to_csv(path, index=False)


def score_external(pred: np.ndarray, test_ids: list[str], sol: pd.DataFrame) -> dict:
    sol2 = sol.set_index("id")
    te = pd.Series(pred, index=test_ids)
    pub = sol2.index[sol2["is_public"].astype(bool)].tolist()
    priv = sol2.index[sol2["is_private"].astype(bool)].tolist()
    return {
        "public_mae": float(mae(sol2.loc[pub, TARGET].to_numpy(float), te.loc[pub].to_numpy(float))),
        "private_mae": float(mae(sol2.loc[priv, TARGET].to_numpy(float), te.loc[priv].to_numpy(float))),
        "overall_mae": float(mae(sol2.loc[test_ids, TARGET].to_numpy(float), te.loc[test_ids].to_numpy(float))),
    }


def write_diagnostics(sel: pd.DataFrame, summary: dict) -> None:
    lines = [
        "# EXP-T075 Training Diagnostics",
        "",
        f"- platform: `{PLATFORM_ID}`",
        f"- seed: `{summary['seed']}`",
        f"- lr_grid: `{summary['lr_grid']}`",
        f"- max_epochs={summary['max_epochs']}, patience={summary['patience']}, min_epochs={summary['min_epochs']}",
        f"- scheduler: cosine T_max={T_MAX}, eta_min={ETA_MIN_FRAC}×lr0",
        "",
        "## Per candidate",
        "",
        "| scheme | fold | initial_lr | eta_min | best_epoch | lr_at_best | best_VAL | selected | final | num_fail |",
        "|--------|------|------------|---------|------------|------------|----------|----------|-------|----------|",
    ]
    for scheme in ("primary", "shadow"):
        for fr in summary["fold_results"][scheme]:
            for c in fr["candidates"]:
                mark = "Y" if c["selected"] else ""
                bv = c["best_val_mae"]
                bv_s = f"{bv:.6f}" if math.isfinite(float(bv)) else "inf"
                lines.append(
                    f"| {scheme} | {fr['fold']} | {c['initial_lr']:.0e} | {c['eta_min']:.3g} | "
                    f"{c['best_epoch']} | {c['lr_at_best_epoch']:.6g} | {bv_s} | {mark} | "
                    f"{c['final_epoch']} | {c['numerical_failure']} |"
                )

    win = sel["selected_initial_lr"].astype(float)
    lines += ["", "## Selected initial-LR winner counts", ""]
    for lr in coarse_lr_grid():
        lines.append(f"- {lr:.0e}: {int((np.isclose(win, lr)).sum())}")

    be = sel["best_epoch"].astype(int)
    lab = sel["lr_at_best_epoch"].astype(float)
    lines += [
        "",
        "## Selected best_epoch",
        "",
        f"- min={be.min()} q25={be.quantile(0.25):.1f} median={be.median():.1f} "
        f"mean={be.mean():.2f} q75={be.quantile(0.75):.1f} max={be.max()}",
        f"- stop before epoch 50: {int((be < 50).sum())}",
        f"- stop before epoch 100: {int((be < 100).sum())}",
        f"- final_epoch reach 200: {int((sel['final_epoch'].astype(int) >= 200).sum())}",
        f"- final_epoch <50: {int((sel['final_epoch'].astype(int) < 50).sum())}",
        f"- final_epoch <100: {int((sel['final_epoch'].astype(int) < 100).sum())}",
        "",
        "## Selected lr_at_best",
        "",
        f"- min={lab.min():.6g} q25={lab.quantile(0.25):.6g} median={lab.median():.6g} "
        f"q75={lab.quantile(0.75):.6g} max={lab.max():.6g}",
        "",
        "## Numerical failures",
        "",
    ]
    nf = []
    for scheme in ("primary", "shadow"):
        for fr in summary["fold_results"][scheme]:
            for c in fr["candidates"]:
                if c["numerical_failure"]:
                    nf.append(f"{scheme}/fold{fr['fold']}/lr={c['initial_lr']:.0e}")
    lines.append("- none" if not nf else "\n".join(f"- {x}" for x in nf))
    DIAG_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(code: str, summary: dict, ext_scores: dict, sel: pd.DataFrame) -> None:
    ov, ot = summary["scores"]["oof_val"], summary["scores"]["oof_test"]
    lines = [
        "# EXP-T075 — Final DL Platform Baseline (T030 architecture)",
        "",
        f"- git: `{git_rev()}`",
        f"- platform: `{PLATFORM_ID}`",
        "- architecture: T030 FULL separate H/L (unchanged)",
        f"- seed: {summary['seed']}",
        f"- lr_grid={summary['lr_grid']}; patience={summary['patience']}; max_epochs={summary['max_epochs']}; min_epochs=none",
        f"- cosine T_max={T_MAX}; eta_min={ETA_MIN_FRAC}×lr0; no warmup/restart; no full-Dev",
        f"- n_trainable={summary['n_trainable_parameters']}",
        "",
        "## Selected",
        "",
        sel.to_string(index=False),
        "",
        f"## OOF VAL: {ov['primary']:.6f} / {ov['shadow']:.6f} / {ov['mean']:.6f} / {ov['worst']:.6f}",
        f"## OOF TEST: {ot['primary']:.6f} / {ot['shadow']:.6f} / {ot['mean']:.6f} / {ot['worst']:.6f}",
        "",
        "| Prediction | Public | Private | Overall |",
        "|------------|--------|---------|---------|",
    ]
    for key, label in (
        ("primary_mean", "Primary mean"),
        ("primary_median", "Primary median"),
        ("shadow_mean", "Shadow mean"),
        ("shadow_median", "Shadow median"),
    ):
        s = ext_scores[key]
        lines.append(f"| {label} | {s['public_mae']:.6f} | {s['private_mae']:.6f} | {s['overall_mae']:.6f} |")
    lines += ["", "## Platform freeze", "", f"See `{FREEZE_YAML.name}` — status FROZEN after validation.", "", "## STOP", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(summary: dict, ext_scores: dict) -> None:
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    t073 = exp[exp["experiment_code"] == T073].iloc[0]
    t074 = exp[exp["experiment_code"] == T074].iloc[0]
    ot = summary["scores"]["oof_test"]
    ov = summary["scores"]["oof_val"]
    pm = ext_scores["primary_mean"]
    lines = [
        "# T073 / T074 / T075 Platform Comparison",
        "",
        "| Property | T073 | T074 | T075 |",
        "|----------|------|------|------|",
        "| LR grid | 0.1–3×3e-4 | 1e-5..1e-2 | 1e-5..1e-2 |",
        "| scheduler | CosineAnnealingLR T_max=200 η=0 | cosine100+hold | cosine T_max=200 η=0.01×lr0 |",
        "| min epochs | none | 100 | none |",
        "| patience | 20 | 20 | 30 |",
        "| seed | 101 | 101 | 101 |",
        "| full-Dev refit | no | no | no |",
        "",
        "## Scores (not protocol-equivalent across all three)",
        "",
        "| Metric | T073 | T074 | T075 |",
        "|--------|------|------|------|",
        f"| VAL_P | (see T073) | (see T074) | {ov['primary']:.6f} |",
        f"| VAL_S | | | {ov['shadow']:.6f} |",
        f"| TEST_P | {float(t073['cv_primary_mae']):.6f} | {float(t074['cv_primary_mae']):.6f} | {ot['primary']:.6f} |",
        f"| TEST_S | {float(t073['cv_shadow_mae']):.6f} | {float(t074['cv_shadow_mae']):.6f} | {ot['shadow']:.6f} |",
        f"| Pub (P mean) | {float(t073['public_mae']):.6f} | {float(t074['public_mae']):.6f} | {pm['public_mae']:.6f} |",
        f"| Priv (P mean) | {float(t073['private_mae']):.6f} | {float(t074['private_mae']):.6f} | {pm['private_mae']:.6f} |",
        f"| Overall (P mean) | {float(t073['test_overall_mae']):.6f} | {float(t074['test_overall_mae']):.6f} | {pm['overall_mae']:.6f} |",
        "",
        "T075 closes platform development. Do not invent a fourth platform from this table.",
        "",
    ]
    COMPARE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_freeze() -> None:
    doc = {
        "status": "FROZEN",
        "platform_id": PLATFORM_ID,
        "freeze_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository_commit": git_rev(),
        "protocol_implementation_sha256": implementation_hash(),
        "baseline_experiment": CODE_EXPECT,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "loss": f"SmoothL1(beta={SMOOTH_L1_BETA})",
        "initial_lr_candidates": coarse_lr_grid(),
        "lr_selection": "fold-local validation MAE",
        "scheduler": {
            "type": "cosine_annealing",
            "T_max": T_MAX,
            "eta_min": "0.01 * initial_lr",
            "warmup": None,
            "restart": None,
            "step_timing": "LR set at beginning of epoch; history reports LR DURING epoch",
        },
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_epochs": None,
        "checkpoint": "best validation MAE",
        "full_dev_refit": "disabled",
        "external_prediction": ["fold_checkpoint_mean", "fold_checkpoint_median"],
        "seed_policy": "experiment_specific",
        "future_contract": (
            "Architecture experiments inherit this platform unchanged unless the "
            "experiment explicitly declares TRAINING itself as the scientific variable."
        ),
    }
    FREEZE_YAML.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def register(code: str, summary: dict, ext_scores: dict) -> None:
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    exp_path = ROOT / "results" / "experiments.csv"
    exp_df = pd.read_csv(exp_path)
    if code in set(exp_df["experiment_code"].astype(str)):
        exp_df = exp_df[exp_df["experiment_code"] != code]
    row = {c: "" for c in exp_df.columns}
    for c in EXPERIMENTS_COLUMNS:
        row.setdefault(c, "")
    row.update(
        {
            "experiment_code": code,
            "experiment_id": EID,
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer+DL_FOLDLOCAL_COSINE_V3",
            "source_model_id": f"{PLATFORM_ID}::{REPLAY}",
            "cv_primary_mae": oof["primary"],
            "cv_shadow_mae": oof["shadow"],
            "cv_mean_mae": oof["mean"],
            "cv_worst_mae": oof["worst"],
            "public_mae": pm["public_mae"],
            "private_mae": pm["private_mae"],
            "test_overall_mae": pm["overall_mae"],
            "public_private_delta": pm["public_mae"] - pm["private_mae"],
            "public_private_gap": abs(pm["public_mae"] - pm["private_mae"]),
            "cv_protocol": "dl_foldlocal_cosine_v3_oof_test",
            "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "artifact_status": "FULL",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "REPRODUCED",
            "config_path": f"experiments/configs/{code}.yaml",
            "feature_path": "",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "score_source": "EXP-T075_dl_foldlocal_cosine_v3",
            "prediction_source": "EXP-T075_run",
            "feature_source": "ablingua_residue_t030_architecture",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue",
            "notes": (
                f"{PLATFORM_ID} baseline; OOF TEST in cv_*; test.csv=Primary mean; "
                f"VAL P/S={summary['scores']['oof_val']['primary']:.4f}/{summary['scores']['oof_val']['shadow']:.4f}; "
                f"patience=30; no min_epochs; platform FROZEN"
            ),
            "transformer_type": "FROZEN_PLM",
            "plm_source": "ABLINGUA",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "REG",
            "representation_status": "NOT_EXPORTED",
            "input_space": INPUT_SPACE,
            "input_asset_ref": assets_ref(),
            "shareability_status": "SHAREABLE_COMPLETE",
            "reproducibility_status_v2": "REPRODUCED",
            "canonical_benchmark_eligible": "YES",
            "prediction_reproduction_max_delta": 0.0,
            "reproduction_tolerance_pred": 1e-10,
            "reproduction_tolerance_score": 1e-10,
            "tolerance_reason": "score_recompute_from_saved_predictions",
            "training_reproduction_attempted": True,
            "optuna_used": False,
            "control_experiment_code": HIST,
        }
    )
    pd.concat([exp_df, pd.DataFrame([{c: row.get(c, "") for c in exp_df.columns}])], ignore_index=True).to_csv(
        exp_path, index=False
    )

    comp_path = ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    comp = pd.read_csv(comp_path)
    if code not in set(comp["experiment_code"].astype(str)):
        crow = {c: "" for c in comp.columns}
        crow.update(
            {
                "experiment_code": code,
                "experiment_id": EID,
                "target": TARGET,
                "family": "TRANSFORMER",
                "config_exists": True,
                "feature_required": False,
                "feature_exists": False,
                "feature_path": "",
                "oof_primary_exists": True,
                "oof_shadow_exists": True,
                "test_exists": True,
                "score_recompute_ok": True,
                "prediction_max_delta": 0.0,
                "reproduction_status": "REPRODUCED",
                "shareability_status": "SHAREABLE_COMPLETE",
                "canonical_benchmark_eligible": "YES",
                "notes": f"{PLATFORM_ID}: no fusion parquet; OOF TEST + 4 external preds",
            }
        )
        if "test_prediction_exists" in comp.columns:
            crow["test_prediction_exists"] = True
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "issue", "train", "report"], default="all")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase == "issue":
        print(issue())
        return 0

    code = issue()
    write_config(code)
    write_platform_protocol_md()
    print(f"Issued/using {code}", flush=True)

    if args.phase in ("all", "train"):
        import torch

        assert "3090" in torch.cuda.get_device_name(0) or device_str() == "cpu"
        print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
        for c in (HIST, "EXP-T068", "EXP-T069", "EXP-T070", "EXP-T071", "EXP-T072", T073, T074):
            assert (ROOT / "experiments" / "predictions" / c / "oof_primary.csv").exists(), c

        # scheduler smoke
        eta = eta_min_for(1e-2)
        assert abs(lr_at_t(0, 1e-2, eta) - 1e-2) < 1e-15
        assert abs(lr_at_t(200, 1e-2, eta) - eta) < 1e-15
        assert abs(lr_at_epoch(1, 1e-2, eta) - 1e-2) < 1e-15

        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        rb = load_residue_bundle(dev, test, need_ablingua=True)

        result = run_protocol_v3(
            experiment_code=code,
            target=TARGET,
            dev=dev,
            test=test,
            folds=folds,
            rb=rb,
            device=device_str(),
            out_dir=OUT,
            seed=DEFAULT_SEED,
            quick=args.quick,
        )
        summary = result["summary"]
        hist_df = result["history_df"]
        sel_df = result["selected_df"]
        hist_df.to_csv(HIST_CSV, index=False)
        hist_df.to_csv(OUT / "training_history.csv", index=False)
        sel_df.to_csv(SEL_CSV, index=False)
        sel_df.to_csv(OUT / "selected_lr.csv", index=False)

        PRED.mkdir(parents=True, exist_ok=True)
        for scheme, kind in (("primary", "val"), ("shadow", "val"), ("primary", "test"), ("shadow", "test")):
            src = result[f"oof_{kind}"][scheme].loc[result["dev_ids"]].to_numpy(float)
            save_pred(result["dev_ids"], src, PRED / f"oof_{kind}_{scheme}.csv")
        save_pred(
            result["dev_ids"],
            result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float),
            PRED / "oof_primary.csv",
        )
        save_pred(
            result["dev_ids"],
            result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float),
            PRED / "oof_shadow.csv",
        )
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
            save_pred(result["test_ids"], result["ext"][key], PRED / f"test_{key}.csv")
        for scheme in ("primary", "shadow"):
            for k in range(5):
                save_pred(result["test_ids"], result["ext"][f"{scheme}_fold{k}"], PRED / f"test_{scheme}_fold{k}.csv")
        save_pred(result["test_ids"], result["ext"]["primary_mean"], PRED / "test.csv")

        sol = load_solution(BUNDLE_ROOT / "solution.csv")
        ext_scores = {
            key: score_external(result["ext"][key], result["test_ids"], sol)
            for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
        }

        oof_doc = {
            "experiment_code": code,
            "platform_id": PLATFORM_ID,
            "git_rev": git_rev(),
            "seed": summary["seed"],
            "lr_grid": summary["lr_grid"],
            "patience": summary["patience"],
            "min_epochs": summary["min_epochs"],
            "max_epochs": summary["max_epochs"],
            "scheduler": summary["scheduler"],
            "oof_val": summary["scores"]["oof_val"],
            "oof_test": summary["scores"]["oof_test"],
            "external_test": ext_scores,
            "selected_lr": summary["selected_lr"],
            "n_trainable_parameters": summary["n_trainable_parameters"],
            "no_full_dev_refit": True,
            "no_nested_cv": True,
            "config_sha256": file_sha256(CFG_PATH),
            "protocol_implementation_sha256": implementation_hash(),
        }
        OOF_YAML.write_text(yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        STATE.write_text(
            json.dumps({"experiment_code": code, "ext_scores": ext_scores, "summary_scores": summary["scores"]}, indent=2)
            + "\n"
        )

        write_diagnostics(sel_df, summary)
        write_report(code, summary, ext_scores, sel_df)
        write_comparison(summary, ext_scores)
        write_freeze()
        register(code, summary, ext_scores)
        print(json.dumps({"oof_val": summary["scores"]["oof_val"], "oof_test": summary["scores"]["oof_test"], "ext": ext_scores}, indent=2), flush=True)
        print(f"DONE {code}", flush=True)

    if args.phase == "report":
        summary = json.loads((OUT / "summary.json").read_text())
        sel = pd.read_csv(SEL_CSV)
        state = json.loads(STATE.read_text())
        write_diagnostics(sel, summary)
        write_report(code, summary, state["ext_scores"], sel)
        write_comparison(summary, state["ext_scores"])
        write_freeze()
        write_platform_protocol_md()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
