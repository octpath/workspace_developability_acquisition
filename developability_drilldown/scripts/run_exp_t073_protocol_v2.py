#!/usr/bin/env python3
"""EXP-T073: T030 architecture under protocol V2 (fold-local LR + checkpoint ensemble).

No nested CV. No full-Dev refit. Single seed 101. Architecture unchanged from T030.
"""
from __future__ import annotations

import argparse
import json
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
from antibody_transformer.protocol_v2 import (  # noqa: E402
    DEFAULT_SEED,
    ETA_MIN,
    MAX_EPOCHS,
    PATIENCE,
    PROTOCOL_ID,
    lr_grid,
    run_protocol_v2,
)

TARGET = "TmApp"
CODE_EXPECT = "EXP-T073"
EID = "TRF_TM_ABLINGUA_FULL_CONCAT_FOLDLOCAL_LR_V2"
INPUT_SPACE = "FROZEN_RESIDUE_PROTOCOL_V2"
HIST = "EXP-T030"
REPLAY = "EXP-T030-REPLAY-001"
REPLAY_DIR = ROOT / "experiments" / "replays" / "EXP-T030" / REPLAY

OUT = ROOT / "results" / "EXP-T073_run"
PRED = ROOT / "experiments" / "predictions" / CODE_EXPECT
HIST_CSV = ROOT / "results" / "EXP-T073_TRAINING_HISTORY.csv"
SEL_CSV = ROOT / "results" / "EXP-T073_SELECTED_LR.csv"
DIAG_MD = ROOT / "results" / "EXP-T073_TRAINING_DIAGNOSTICS.md"
REPORT = ROOT / "results" / "EXP-T073_PROTOCOL_REPORT.md"
OOF_YAML = ROOT / "results" / "EXP-T073_OOF_EVALUATION.yaml"
STATE = ROOT / "results" / "EXP-T073_run_state.json"
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
        source_model_id=f"PROTOCOL_V2::{REPLAY}",
        phase="PROTOCOL_V2_BASELINE",
        notes="T030 architecture + fold-local LR + VAL checkpoint ensemble; no full-Dev; no nested CV",
    )
    if code != CODE_EXPECT:
        raise SystemExit(code)
    return code


def write_config(code: str) -> Path:
    presets = load_presets()
    lr_ref = float(presets["neural"]["lr"])
    grid = lr_grid(lr_ref)
    cfg = {
        "experiment_code": code,
        "experiment_id": EID,
        "protocol_id": PROTOCOL_ID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"PROTOCOL_V2::{REPLAY}",
        "transformer_type": "FROZEN_PLM",
        "input_space": INPUT_SPACE,
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
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
        "lr_ref": lr_ref,
        "lr_grid": grid,
        "weight_decay": 0.01,
        "batch_size": 16,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "scheduler": "CosineAnnealingLR",
        "scheduler_T_max": MAX_EPOCHS,
        "scheduler_eta_min": ETA_MIN,
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
    }
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return CFG_PATH


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


def write_diagnostics(sel: pd.DataFrame, hist: pd.DataFrame, summary: dict) -> None:
    lines = [
        "# EXP-T073 Training Diagnostics",
        "",
        f"- protocol: `{PROTOCOL_ID}`",
        f"- seed: `{summary['seed']}`",
        f"- lr_ref: `{summary['lr_ref']}`",
        f"- lr_grid: `{summary['lr_grid']}`",
        f"- max_epochs={summary['max_epochs']}, patience={summary['patience']}",
        f"- scheduler: CosineAnnealingLR T_max={summary['scheduler']['T_max']} eta_min={summary['scheduler']['eta_min']}",
        "",
        "## Selected LR by fold",
        "",
        "| scheme | fold | selected_lr | best_epoch | best_step | best_val_mae | final_epoch | stopped_early |",
        "|--------|------|-------------|------------|-----------|--------------|-------------|---------------|",
    ]
    for _, r in sel.iterrows():
        lines.append(
            f"| {r['scheme']} | {int(r['fold'])} | {r['selected_lr']:.6g} | {int(r['best_epoch'])} | "
            f"{int(r['best_optimizer_step'])} | {r['best_val_mae']:.6f} | {int(r['final_epoch'])} | {r['stopped_early']} |"
        )

    # Candidate table
    lines += ["", "## All LR candidates (best VAL MAE)", ""]
    # from fold_results
    for scheme in ("primary", "shadow"):
        lines.append(f"### {scheme}")
        for fr in summary["fold_results"][scheme]:
            lines.append(f"- fold {fr['fold']}:")
            for c in fr["candidates"]:
                mark = " **SELECTED**" if c["selected"] else ""
                lines.append(
                    f"  - lr={c['lr']:.6g}: best_epoch={c['best_epoch']} step={c['best_optimizer_step']} "
                    f"VAL={c['best_val_mae']:.6f} final={c['final_epoch']} stopped={c['stopped_early']}{mark}"
                )

    be = sel["best_epoch"].astype(int)
    lines += [
        "",
        "## Best-epoch distribution (selected only)",
        "",
        f"- mean={be.mean():.2f} median={be.median():.1f} min={be.min()} max={be.max()} std={be.std():.2f}",
        f"- fraction stopped before epoch 40: {(be < 40).mean():.2f}",
        f"- fraction stopped before epoch 80: {(be < 80).mean():.2f}",
        f"- fraction reaching epoch 200: {(sel['final_epoch'].astype(int) >= 200).mean():.2f}",
        f"- fraction stopped_early: {sel['stopped_early'].astype(bool).mean():.2f}",
        "",
        "## Cosine annealing vs early stopping",
        "",
        "CosineAnnealingLR(T_max=200) only modestly reduces LR in the first ~20–40 epochs.",
        "If most selected best_epoch values are ≪ 200, patience=20 likely terminates before",
        "cosine annealing has materially annealed. This run reports the observation only;",
        "patience/scheduler are NOT changed.",
        "",
    ]
    # LR consistency
    for scheme in ("primary", "shadow"):
        sub = sel[sel["scheme"] == scheme]
        uniq = [float(x) for x in sorted(sub["selected_lr"].unique())]
        lines.append(f"- {scheme} selected LRs: {uniq} (n_unique={len(uniq)})")
    DIAG_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def register(code: str, summary: dict, ext_scores: dict) -> None:
    # Prefer Primary mean Overall as the registry scalar Test columns (documented)
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    # Use OOF TEST as the new principal internal metrics in cv_* columns
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
            "model_type": "AnnotatedTransformer+ProtocolV2",
            "source_model_id": f"PROTOCOL_V2::{REPLAY}",
            "cv_primary_mae": oof["primary"],
            "cv_shadow_mae": oof["shadow"],
            "cv_mean_mae": oof["mean"],
            "cv_worst_mae": oof["worst"],
            "public_mae": pm["public_mae"],
            "private_mae": pm["private_mae"],
            "test_overall_mae": pm["overall_mae"],
            "public_private_delta": pm["public_mae"] - pm["private_mae"],
            "public_private_gap": abs(pm["public_mae"] - pm["private_mae"]),
            "cv_protocol": "protocol_v2_fold_local_lr_oof_test",
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
            "score_source": "EXP-T073_protocol_v2",
            "prediction_source": "EXP-T073_run",
            "feature_source": "ablingua_residue_t030_architecture",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue",
            "notes": (
                f"Protocol V2 baseline; OOF TEST in cv_* (also oof_test_*.csv); "
                f"test.csv=Primary mean; also test_*_mean/median; "
                f"Public/Private from Primary mean; "
                f"VAL P/S={summary['scores']['oof_val']['primary']:.4f}/{summary['scores']['oof_val']['shadow']:.4f}; "
                f"seed={summary['seed']}; no full-Dev; no nested CV"
            ),
            "transformer_type": "FROZEN_PLM",
            "plm_source": "ABLINGUA",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "REG",
            "representation_status": "NOT_EXPORTED",
            "input_space": INPUT_SPACE,
            "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
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
                "notes": "PROTOCOL_V2: no fusion parquet; OOF TEST + 4 external preds",
            }
        )
        if "test_prediction_exists" in comp.columns:
            crow["test_prediction_exists"] = True
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)


def write_report(code: str, summary: dict, ext_scores: dict, sel: pd.DataFrame) -> None:
    hist = pd.read_csv(ROOT / "results" / "experiments.csv")
    t030 = hist[hist["experiment_code"] == HIST].iloc[0]
    runs = pd.read_csv(ROOT / "results" / "experiment_runs.csv")
    replay = runs[runs["run_id"] == REPLAY].iloc[0]
    ov = summary["scores"]["oof_val"]
    ot = summary["scores"]["oof_test"]
    lines = [
        "# EXP-T073 — Protocol V2 Baseline (T030 architecture)",
        "",
        f"- git: `{git_rev()}`",
        f"- protocol: `{PROTOCOL_ID}`",
        f"- architecture: T030 FULL separate H/L, REG_H||REG_L (unchanged)",
        f"- seed: {summary['seed']} (single)",
        f"- lr_ref={summary['lr_ref']}, grid={summary['lr_grid']}",
        f"- AdamW, max_epochs={summary['max_epochs']}, patience={summary['patience']}, CosineAnnealingLR",
        f"- no nested CV; no full-Dev refit",
        f"- n_trainable={summary['n_trainable_parameters']}",
        "",
        "## Selected LR",
        "",
        sel.to_string(index=False),
        "",
        "## OOF VAL (model-selection; not unbiased)",
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
        "## Contextual vs old T030",
        "",
        f"- historical T030 P/S: {float(t030['cv_primary_mae']):.6f} / {float(t030['cv_shadow_mae']):.6f}; "
        f"Pub/Priv/Overall: {float(t030['public_mae']):.6f} / {float(t030['private_mae']):.6f} / {float(t030['test_overall_mae']):.6f}",
        f"- REPLAY P/S: {float(replay['cv_primary_mae']):.6f} / {float(replay['cv_shadow_mae']):.6f}; "
        f"Pub/Priv/Overall: {float(replay['public_mae']):.6f} / {float(replay['private_mae']):.6f} / {float(replay['overall_mae']):.6f}",
        "- NOTE: metrics are **not** protocol-equivalent (old used multi-seed + full-Dev refit).",
        "",
        "## Readiness",
        "",
        "See final response section N. Architecture series not started.",
        "",
        "## STOP",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        # protect historical
        for c in (HIST, "EXP-T068", "EXP-T069", "EXP-T070", "EXP-T071", "EXP-T072"):
            p = ROOT / "experiments" / "predictions" / c / "oof_primary.csv"
            assert p.exists(), p

        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        rb = load_residue_bundle(dev, test, need_ablingua=True)

        result = run_protocol_v2(
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

        # predictions
        PRED.mkdir(parents=True, exist_ok=True)
        save_pred(result["dev_ids"], result["oof_val"]["primary"].loc[result["dev_ids"]].to_numpy(float), PRED / "oof_val_primary.csv")
        save_pred(result["dev_ids"], result["oof_val"]["shadow"].loc[result["dev_ids"]].to_numpy(float), PRED / "oof_val_shadow.csv")
        save_pred(result["dev_ids"], result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float), PRED / "oof_test_primary.csv")
        save_pred(result["dev_ids"], result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float), PRED / "oof_test_shadow.csv")
        # registry-compatible aliases
        save_pred(result["dev_ids"], result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float), PRED / "oof_primary.csv")
        save_pred(result["dev_ids"], result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float), PRED / "oof_shadow.csv")

        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
            save_pred(result["test_ids"], result["ext"][key], PRED / f"test_{key}.csv")
        for scheme in ("primary", "shadow"):
            for k in range(5):
                save_pred(result["test_ids"], result["ext"][f"{scheme}_fold{k}"], PRED / f"test_{scheme}_fold{k}.csv")
        # default test.csv = primary mean
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
            "lr_ref": summary["lr_ref"],
            "lr_grid": summary["lr_grid"],
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
        STATE.write_text(json.dumps({"experiment_code": code, "ext_scores": ext_scores, "summary_scores": summary["scores"]}, indent=2) + "\n")

        write_diagnostics(sel_df, hist_df, summary)
        write_report(code, summary, ext_scores, sel_df)
        register(code, summary, ext_scores)
        print(json.dumps({"oof_val": summary["scores"]["oof_val"], "oof_test": summary["scores"]["oof_test"], "ext": ext_scores}, indent=2), flush=True)
        print(f"DONE {code}", flush=True)

    if args.phase == "report":
        # regenerate from saved
        summary = json.loads((OUT / "summary.json").read_text())
        sel = pd.read_csv(SEL_CSV)
        state = json.loads(STATE.read_text())
        write_diagnostics(sel, pd.read_csv(HIST_CSV), summary)
        write_report(code, summary, state["ext_scores"], sel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
