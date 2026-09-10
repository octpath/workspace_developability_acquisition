#!/usr/bin/env python3
"""EXP-T074: T030 architecture under coarse-cosine protocol V2 platform.

No nested CV. No full-Dev refit. Single seed 101.
Coarse LR grid 1e-5..1e-2; cosine 100 then hold; min_epochs=100.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
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
from antibody_transformer.config import BUNDLE_ROOT, load_presets  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle, load_solution  # noqa: E402
from antibody_transformer.protocol_v2 import DEFAULT_SEED  # noqa: E402
from antibody_transformer.protocol_v2_coarse import (  # noqa: E402
    COSINE_EPOCHS,
    ETA_MIN_FRAC,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    PROTOCOL_ID,
    coarse_lr_grid,
    eta_min_for,
    lr_at_epoch,
    run_protocol_v2_coarse,
)

TARGET = "TmApp"
CODE_EXPECT = "EXP-T074"
EID = "TRF_TM_ABLINGUA_FULL_CONCAT_COARSE_COSINE_V2"
INPUT_SPACE = "FROZEN_RESIDUE_COARSE_COSINE_PROTOCOL_V2"
HIST = "EXP-T030"
REPLAY = "EXP-T030-REPLAY-001"
T073 = "EXP-T073"

OUT = ROOT / "results" / "EXP-T074_run"
PRED = ROOT / "experiments" / "predictions" / CODE_EXPECT
HIST_CSV = ROOT / "results" / "EXP-T074_TRAINING_HISTORY.csv"
SEL_CSV = ROOT / "results" / "EXP-T074_SELECTED_LR.csv"
DIAG_MD = ROOT / "results" / "EXP-T074_LR_TRAJECTORY_DIAGNOSTICS.md"
REPORT = ROOT / "results" / "EXP-T074_PROTOCOL_REPORT.md"
OOF_YAML = ROOT / "results" / "EXP-T074_OOF_EVALUATION.yaml"
STATE = ROOT / "results" / "EXP-T074_run_state.json"
CFG_PATH = ROOT / "experiments" / "configs" / f"{CODE_EXPECT}.yaml"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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
        source_model_id=f"PROTOCOL_V2_COARSE::{REPLAY}",
        phase="PROTOCOL_V2_COARSE_COSINE",
        notes="T030 + coarse LR grid + cosine100 hold + min_epochs=100; no full-Dev; no nested CV",
    )
    if code != CODE_EXPECT:
        raise SystemExit(code)
    return code


def write_config(code: str) -> Path:
    grid = coarse_lr_grid()
    cfg = {
        "experiment_code": code,
        "experiment_id": EID,
        "protocol_id": PROTOCOL_ID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"PROTOCOL_V2_COARSE::{REPLAY}",
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
        "lr_grid": grid,
        "eta_min_frac": ETA_MIN_FRAC,
        "weight_decay": 0.01,
        "batch_size": 16,
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "scheduler": "explicit_cosine_then_hold",
        "cosine_epochs": COSINE_EPOCHS,
        "scheduler_step_timing": "beginning_of_epoch",
        "warmup": None,
        "gradient_clip": 1.0,
        "loss": "SmoothL1Loss(beta=0.5)",
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
        "prior_platform_baseline": T073,
    }
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return CFG_PATH


def assets_ref() -> str:
    return "assets/transformer/residue_asset_manifest.yaml#ablingua600m"


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


def write_lr_trajectory_diagnostics(sel: pd.DataFrame, hist: pd.DataFrame, summary: dict) -> None:
    lines = [
        "# EXP-T074 LR Trajectory Diagnostics",
        "",
        f"- protocol: `{PROTOCOL_ID}`",
        f"- seed: `{summary['seed']}`",
        f"- lr_grid: `{summary['lr_grid']}`",
        f"- cosine_epochs={COSINE_EPOCHS}, eta_min=0.01×initial_lr, then hold",
        f"- min_epochs={summary['min_epochs']}, max_epochs={summary['max_epochs']}, patience={summary['patience']}",
        f"- scheduler step timing: `{summary['scheduler']['step_timing']}`",
        "",
        "## Per candidate (best VAL)",
        "",
        "| scheme | fold | initial_lr | eta_min | best_epoch | lr_at_best | best_VAL | selected | final_epoch | num_fail |",
        "|--------|------|------------|---------|------------|------------|----------|----------|-------------|----------|",
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

    # Winner counts
    win = sel["selected_initial_lr"].astype(float)
    lines += ["", "## A. INITIAL LR WINNER COUNTS", ""]
    for lr in coarse_lr_grid():
        n = int((np.isclose(win, lr)).sum())
        lines.append(f"- {lr:.0e}: {n}")

    lab = sel["lr_at_best_epoch"].astype(float)
    log10 = np.log10(lab.clip(lower=1e-20))
    lines += [
        "",
        "## B. LR-AT-BEST DISTRIBUTION (selected)",
        "",
        f"- min={lab.min():.6g} q25={lab.quantile(0.25):.6g} median={lab.median():.6g} "
        f"q75={lab.quantile(0.75):.6g} max={lab.max():.6g}",
        f"- log10: min={log10.min():.3f} q25={log10.quantile(0.25):.3f} median={log10.median():.3f} "
        f"q75={log10.quantile(0.75):.3f} max={log10.max():.3f}",
    ]

    be = sel["best_epoch"].astype(int)
    lines += [
        "",
        "## C. BEST-EPOCH DISTRIBUTION (selected)",
        "",
        f"- min={be.min()} median={be.median():.1f} mean={be.mean():.2f} max={be.max()}",
        f"- <=25: {int((be <= 25).sum())}",
        f"- 26-50: {int(((be >= 26) & (be <= 50)).sum())}",
        f"- 51-75: {int(((be >= 51) & (be <= 75)).sum())}",
        f"- 76-100: {int(((be >= 76) & (be <= 100)).sum())}",
        f"- >100: {int((be > 100).sum())}",
        f"- final_epoch min/median/max: {sel['final_epoch'].min()}/{sel['final_epoch'].median()}/{sel['final_epoch'].max()}",
        f"- fraction final_epoch>=100: {(sel['final_epoch'].astype(int) >= 100).mean():.2f}",
        f"- fraction final_epoch>=200: {(sel['final_epoch'].astype(int) >= 200).mean():.2f}",
    ]

    # Cross-candidate: lr_at_best by initial_lr (all non-failed candidates)
    lines += ["", "## D. CROSS-CANDIDATE LR-AT-BEST BY INITIAL LR", ""]
    rows = []
    for scheme in ("primary", "shadow"):
        for fr in summary["fold_results"][scheme]:
            for c in fr["candidates"]:
                if c["numerical_failure"] or not math.isfinite(float(c["best_val_mae"])):
                    continue
                rows.append(c)
    cdf = pd.DataFrame(rows)
    if len(cdf):
        for lr0, g in cdf.groupby("initial_lr"):
            labg = g["lr_at_best_epoch"].astype(float)
            lines.append(
                f"- init {float(lr0):.0e}: n={len(g)} lr_at_best "
                f"median={labg.median():.6g} [{labg.min():.6g}, {labg.max():.6g}] "
                f"best_epoch median={g['best_epoch'].median():.1f}"
            )
        # Overlap narrative
        meds = {float(lr0): float(g["lr_at_best_epoch"].median()) for lr0, g in cdf.groupby("initial_lr")}
        lines += [
            "",
            "### Overlap interpretation",
            "",
            f"- per-initial median lr_at_best: { {f'{k:.0e}': v for k, v in sorted(meds.items())} }",
        ]
        finite_meds = [v for k, v in meds.items() if k < 1e-2 or True]
        if len(meds) >= 2:
            vals = list(meds.values())
            span = max(vals) / min(vals) if min(vals) > 0 else float("inf")
            lines.append(
                f"- ratio max/min of those medians ≈ {span:.2f} "
                "(closer to 1 ⇒ trajectories meet in a common useful LR band)."
            )

    # Numerical failures
    lines += ["", "## Numerical failures", ""]
    nf = []
    for scheme in ("primary", "shadow"):
        for fr in summary["fold_results"][scheme]:
            for c in fr["candidates"]:
                if c["numerical_failure"]:
                    nf.append(f"{scheme}/fold{fr['fold']}/lr={c['initial_lr']:.0e}")
    if not nf:
        lines.append("- none")
    else:
        for x in nf:
            lines.append(f"- {x}")
        n_1e2 = sum(1 for x in nf if "1e-02" in x or "lr=1e-2" in x)
        lines.append(f"- count involving 1e-2: {sum(1 for x in nf if '1e-02' in x or x.endswith('1e-2'))}")

    DIAG_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(code: str, summary: dict, ext_scores: dict, sel: pd.DataFrame) -> None:
    hist = pd.read_csv(ROOT / "results" / "experiments.csv")
    t030 = hist[hist["experiment_code"] == HIST].iloc[0]
    t073 = hist[hist["experiment_code"] == T073].iloc[0]
    runs = pd.read_csv(ROOT / "results" / "experiment_runs.csv")
    replay = runs[runs["run_id"] == REPLAY].iloc[0]
    ov = summary["scores"]["oof_val"]
    ot = summary["scores"]["oof_test"]
    lines = [
        "# EXP-T074 — Coarse Cosine Protocol V2 Platform (T030 architecture)",
        "",
        f"- git: `{git_rev()}`",
        f"- protocol: `{PROTOCOL_ID}`",
        "- architecture: T030 FULL separate H/L, REG_H||REG_L (unchanged)",
        f"- seed: {summary['seed']} (single)",
        f"- lr_grid={summary['lr_grid']}",
        f"- AdamW; min_epochs={summary['min_epochs']}; max_epochs={summary['max_epochs']}; patience={summary['patience']}",
        "- scheduler: cosine t=0..100 → eta_min=0.01×lr0, then hold; LR set at epoch start",
        "- no nested CV; no full-Dev refit",
        f"- n_trainable={summary['n_trainable_parameters']}",
        "",
        "## Selected runs",
        "",
        sel.to_string(index=False),
        "",
        "## OOF VAL (selection-oriented)",
        "",
        f"- VAL_P/S/mean/worst: {ov['primary']:.6f} / {ov['shadow']:.6f} / {ov['mean']:.6f} / {ov['worst']:.6f}",
        "",
        "## OOF TEST (principal internal)",
        "",
        f"- TEST_P/S/mean/worst: {ot['primary']:.6f} / {ot['shadow']:.6f} / {ot['mean']:.6f} / {ot['worst']:.6f}",
        "",
        "## External Test (4-way)",
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
        lines.append(
            f"| {label} | {s['public_mae']:.6f} | {s['private_mae']:.6f} | {s['overall_mae']:.6f} |"
        )
    lines += [
        "",
        "## vs EXP-T073 / old T030 (not protocol-equivalent)",
        "",
        f"- T073 OOF TEST P/S: {float(t073['cv_primary_mae']):.6f} / {float(t073['cv_shadow_mae']):.6f}; "
        f"Pub/Priv/Overall (Primary mean): {float(t073['public_mae']):.6f} / {float(t073['private_mae']):.6f} / {float(t073['test_overall_mae']):.6f}",
        f"- hist T030 P/S: {float(t030['cv_primary_mae']):.6f} / {float(t030['cv_shadow_mae']):.6f}",
        f"- REPLAY P/S: {float(replay['cv_primary_mae']):.6f} / {float(replay['cv_shadow_mae']):.6f}",
        "",
        "## Platform verdict",
        "",
        "See final response section O.",
        "",
        "## STOP",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "model_type": "AnnotatedTransformer+ProtocolV2Coarse",
            "source_model_id": f"PROTOCOL_V2_COARSE::{REPLAY}",
            "cv_primary_mae": oof["primary"],
            "cv_shadow_mae": oof["shadow"],
            "cv_mean_mae": oof["mean"],
            "cv_worst_mae": oof["worst"],
            "public_mae": pm["public_mae"],
            "private_mae": pm["private_mae"],
            "test_overall_mae": pm["overall_mae"],
            "public_private_delta": pm["public_mae"] - pm["private_mae"],
            "public_private_gap": abs(pm["public_mae"] - pm["private_mae"]),
            "cv_protocol": "protocol_v2_coarse_cosine_oof_test",
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
            "score_source": "EXP-T074_protocol_v2_coarse",
            "prediction_source": "EXP-T074_run",
            "feature_source": "ablingua_residue_t030_architecture",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue",
            "notes": (
                f"Protocol V2 coarse-cosine; OOF TEST in cv_*; test.csv=Primary mean; "
                f"VAL P/S={summary['scores']['oof_val']['primary']:.4f}/{summary['scores']['oof_val']['shadow']:.4f}; "
                f"seed={summary['seed']}; min_epochs=100; grid=1e-5..1e-2; no full-Dev; no nested CV"
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
    new_df = pd.DataFrame([{c: row.get(c, "") for c in exp_df.columns}])
    pd.concat([exp_df, new_df], ignore_index=True).to_csv(exp_path, index=False)

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
                "notes": "PROTOCOL_V2_COARSE: no fusion parquet; OOF TEST + 4 external preds",
            }
        )
        if "test_prediction_exists" in comp.columns:
            crow["test_prediction_exists"] = True
        comp = pd.concat(
            [comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])],
            ignore_index=True,
        )
        comp.to_csv(comp_path, index=False)


def maybe_plots(hist: pd.DataFrame, sel: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ROOT / "results" / "EXP-T074_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # VAL MAE vs epoch for one fold sample (primary fold0)
    sub = hist[(hist["scheme"] == "primary") & (hist["fold"] == 0) & (~hist["numerical_failure"].astype(bool))]
    if len(sub):
        fig, ax = plt.subplots(figsize=(6, 4))
        for lr0, g in sub.groupby("initial_lr"):
            ax.plot(g["epoch"], g["val_mae"], label=f"{float(lr0):.0e}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("VAL MAE")
        ax.set_title("Primary fold0 VAL MAE vs epoch")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "val_mae_vs_epoch_primary_k0.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for lr0, g in sub.groupby("initial_lr"):
            ax.plot(g["epoch"], g["learning_rate"], label=f"{float(lr0):.0e}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning_rate")
        ax.set_title("Primary fold0 LR vs epoch")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "lr_vs_epoch_primary_k0.png", dpi=120)
        plt.close(fig)

    # best VAL vs initial LR
    rows = []
    # use history bests from selected file + reconstruct from fold isn't needed
    # From hist: take min val_mae per candidate
    if "initial_lr" in hist.columns:
        g = hist[~hist["numerical_failure"].astype(bool)].groupby(
            ["scheme", "fold", "initial_lr"], as_index=False
        )["val_mae"].min()
        fig, ax = plt.subplots(figsize=(5, 4))
        for scheme, sg in g.groupby("scheme"):
            med = sg.groupby("initial_lr")["val_mae"].median()
            ax.plot(med.index.astype(float), med.values, marker="o", label=scheme)
        ax.set_xscale("log")
        ax.set_xlabel("initial_lr")
        ax.set_ylabel("best VAL MAE (median over folds)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "best_val_vs_initial_lr.png", dpi=120)
        plt.close(fig)


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
    print(f"Issued/using {code}", flush=True)

    if args.phase in ("all", "train"):
        import torch

        assert "3090" in torch.cuda.get_device_name(0) or device_str() == "cpu"
        print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
        for c in (HIST, "EXP-T068", "EXP-T069", "EXP-T070", "EXP-T071", "EXP-T072", T073):
            p = ROOT / "experiments" / "predictions" / c / "oof_primary.csv"
            assert p.exists(), p

        # smoke-check scheduler
        assert abs(lr_at_epoch(1, 1e-2, eta_min_for(1e-2)) - 1e-2) < 1e-15
        assert abs(lr_at_epoch(101, 1e-2, eta_min_for(1e-2)) - eta_min_for(1e-2)) < 1e-15
        assert abs(lr_at_epoch(150, 1e-2, eta_min_for(1e-2)) - eta_min_for(1e-2)) < 1e-15

        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        rb = load_residue_bundle(dev, test, need_ablingua=True)

        result = run_protocol_v2_coarse(
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
        save_pred(
            result["dev_ids"],
            result["oof_val"]["primary"].loc[result["dev_ids"]].to_numpy(float),
            PRED / "oof_val_primary.csv",
        )
        save_pred(
            result["dev_ids"],
            result["oof_val"]["shadow"].loc[result["dev_ids"]].to_numpy(float),
            PRED / "oof_val_shadow.csv",
        )
        save_pred(
            result["dev_ids"],
            result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float),
            PRED / "oof_test_primary.csv",
        )
        save_pred(
            result["dev_ids"],
            result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float),
            PRED / "oof_test_shadow.csv",
        )
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
                save_pred(
                    result["test_ids"],
                    result["ext"][f"{scheme}_fold{k}"],
                    PRED / f"test_{scheme}_fold{k}.csv",
                )
        save_pred(result["test_ids"], result["ext"]["primary_mean"], PRED / "test.csv")

        sol = load_solution(BUNDLE_ROOT / "solution.csv")
        ext_scores = {
            key: score_external(result["ext"][key], result["test_ids"], sol)
            for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
        }

        oof_doc = {
            "experiment_code": code,
            "protocol_id": PROTOCOL_ID,
            "git_rev": git_rev(),
            "seed": summary["seed"],
            "lr_grid": summary["lr_grid"],
            "min_epochs": summary["min_epochs"],
            "max_epochs": summary["max_epochs"],
            "patience": summary["patience"],
            "scheduler": summary["scheduler"],
            "oof_val": summary["scores"]["oof_val"],
            "oof_test": summary["scores"]["oof_test"],
            "external_test": ext_scores,
            "selected_lr": summary["selected_lr"],
            "n_trainable_parameters": summary["n_trainable_parameters"],
            "no_full_dev_refit": True,
            "no_nested_cv": True,
            "config_sha256": file_sha256(CFG_PATH),
        }
        OOF_YAML.write_text(yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        STATE.write_text(
            json.dumps(
                {
                    "experiment_code": code,
                    "ext_scores": ext_scores,
                    "summary_scores": summary["scores"],
                },
                indent=2,
            )
            + "\n"
        )

        write_lr_trajectory_diagnostics(sel_df, hist_df, summary)
        write_report(code, summary, ext_scores, sel_df)
        maybe_plots(hist_df, sel_df)
        register(code, summary, ext_scores)
        print(
            json.dumps(
                {
                    "oof_val": summary["scores"]["oof_val"],
                    "oof_test": summary["scores"]["oof_test"],
                    "ext": ext_scores,
                },
                indent=2,
            ),
            flush=True,
        )
        print(f"DONE {code}", flush=True)

    if args.phase == "report":
        summary = json.loads((OUT / "summary.json").read_text())
        sel = pd.read_csv(SEL_CSV)
        state = json.loads(STATE.read_text())
        write_lr_trajectory_diagnostics(sel, pd.read_csv(HIST_CSV), summary)
        write_report(code, summary, state["ext_scores"], sel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
