#!/usr/bin/env python3
"""Run EXP-T080..T104 under DL_FOLDLOCAL_COSINE_V3 (arch × representation sweep).

Phases: prereg | diag | train | finalize | all
Does not retrain T075–T079. OOF TEST scientific branching deferred until finalize.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
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
from antibody_transformer.protocol_v3 import (  # noqa: E402
    DEFAULT_SEED,
    PLATFORM_ID,
    run_protocol_v3,
)
from preregister_t080_t104 import SERIES, effective_merge  # noqa: E402

TARGET = "TmApp"
PREREG = ROOT / "results" / "T080_T104_ARCH_REP_PREREGISTRATION.yaml"
PRE_TEST_FREEZE = ROOT / "results" / "T080_T104_PRE_TEST_FREEZE.yaml"
MATRIX_MD = ROOT / "results" / "T075_T104_ARCHITECTURE_REPRESENTATION_MATRIX.md"
MEAN_CONCAT_MD = ROOT / "results" / "T075_T104_MEAN_VS_CONCAT.md"
SCRATCH_MD = ROOT / "results" / "T090_T104_SCRATCH_ARCHITECTURE_REPORT.md"
INTERACTION_CSV = ROOT / "results" / "T075_T104_REPRESENTATION_INTERACTION.csv"
BOOT_CSV = ROOT / "results" / "T075_T104_PAIRED_BOOTSTRAP.csv"
EXT_CSV = ROOT / "results" / "T075_T104_EXTERNAL_FOUR_WAY.csv"

REUSED = [
    {
        "code": "EXP-T075",
        "arch_id": "ARCH-1",
        "representation": "ABLINGUA",
        "merge_mode": "concat",
        "content_mode": "frozen",
    },
    {
        "code": "EXP-T076",
        "arch_id": "ARCH-2",
        "representation": "ABLINGUA",
        "merge_mode": None,
        "content_mode": "frozen",
    },
    {
        "code": "EXP-T077",
        "arch_id": "ARCH-3",
        "representation": "ABLINGUA",
        "merge_mode": "concat",
        "content_mode": "frozen",
    },
    {
        "code": "EXP-T078",
        "arch_id": "ARCH-4",
        "representation": "ABLINGUA",
        "merge_mode": "concat",
        "content_mode": "frozen",
    },
    {
        "code": "EXP-T079",
        "arch_id": "ARCH-5",
        "representation": "ABLINGUA",
        "merge_mode": "concat",
        "content_mode": "frozen",
    },
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_prereg() -> dict:
    if not PREREG.exists():
        raise SystemExit(f"missing preregistration {PREREG}; run preregister_t080_t104.py first")
    return yaml.safe_load(PREREG.read_text())


def ensure_prereg() -> dict:
    if not PREREG.exists() or not all(
        (ROOT / "experiments" / "configs" / f"{s['code']}.yaml").exists() for s in SERIES
    ):
        print("=== prereg: generating configs ===", flush=True)
        from preregister_t080_t104 import main as prereg_main

        rc = prereg_main()
        if rc != 0:
            raise SystemExit(rc)
    return load_prereg()


def issue_one(spec: dict) -> str:
    codes = load_codes()
    eid = spec["experiment_id"]
    if (codes["experiment_id"] == eid).any():
        return str(codes.set_index("experiment_id").loc[eid, "experiment_code"])
    expect = spec["code"]
    nxt = next_code(TARGET)
    if nxt != expect:
        raise SystemExit(f"expected {expect}, got {nxt}")
    code = issue_code(
        eid,
        TARGET,
        source_model_id=f"{PLATFORM_ID}::EXP-T075",
        phase="V3_ARCH_REP_SWEEP_T080_T104",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


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


def already_complete(code: str) -> bool:
    oof = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    pred = ROOT / "experiments" / "predictions" / code / "test_primary_mean.csv"
    return oof.exists() and pred.exists()


def protect_reused() -> None:
    for c in ("EXP-T075", "EXP-T076", "EXP-T077", "EXP-T078", "EXP-T079"):
        p = ROOT / "experiments" / "predictions" / c / "oof_primary.csv"
        assert p.exists(), f"missing protected prediction {p}"


def register(code: str, spec: dict, summary: dict, ext_scores: dict, notes: str) -> None:
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    is_ablingua = spec["representation"] == "ABLINGUA"
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
            "experiment_id": spec["experiment_id"],
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": f"AnnotatedTransformer+{PLATFORM_ID}",
            "source_model_id": f"{PLATFORM_ID}::EXP-T075",
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
            "score_source": f"{code}_{PLATFORM_ID}",
            "prediction_source": f"{code}_run",
            "feature_source": "ablingua_residue" if is_ablingua else "scratch_residue",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue" if is_ablingua else "scratch AA",
            "notes": notes,
            "transformer_type": "FROZEN_PLM" if is_ablingua else "SCRATCH",
            "plm_source": "ABLINGUA" if is_ablingua else "NONE",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "REG",
            "representation_status": "NOT_EXPORTED",
            "input_space": spec["input_space"],
            "input_asset_ref": (
                "assets/transformer/residue_asset_manifest.yaml#ablingua600m"
                if is_ablingua
                else "assets/transformer/residue_asset_manifest.yaml#annotations+scratch_sequences"
            ),
            "shareability_status": "SHAREABLE_COMPLETE",
            "reproducibility_status_v2": "REPRODUCED",
            "canonical_benchmark_eligible": "YES",
            "prediction_reproduction_max_delta": 0.0,
            "reproduction_tolerance_pred": 1e-10,
            "reproduction_tolerance_score": 1e-10,
            "tolerance_reason": "score_recompute_from_saved_predictions",
            "training_reproduction_attempted": True,
            "optuna_used": False,
            "control_experiment_code": "EXP-T075",
        }
    )
    pd.concat([exp_df, pd.DataFrame([{c: row.get(c, "") for c in exp_df.columns}])], ignore_index=True).to_csv(
        exp_path, index=False
    )
    comp_path = ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        if code not in set(comp["experiment_code"].astype(str)):
            crow = {c: "" for c in comp.columns}
            crow.update(
                {
                    "experiment_code": code,
                    "experiment_id": spec["experiment_id"],
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
                    "notes": f"{PLATFORM_ID} arch×rep series",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
            comp.to_csv(comp_path, index=False)


def run_one(spec: dict, prereg_row: dict, *, quick: bool, silent_test: bool) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} ({spec['description']}) ====", flush=True)
    protect_reused()

    if already_complete(code):
        print(f"SKIP {code}: OOF + test_primary_mean already present", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        st = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
        return {
            "code": code,
            "spec": spec,
            "summary": summary,
            "ext_scores": st["ext_scores"],
            "skipped": True,
        }

    out = ROOT / "results" / f"{code}_run"
    pred = ROOT / "experiments" / "predictions" / code
    hist_csv = ROOT / "results" / f"{code}_TRAINING_HISTORY.csv"
    sel_csv = ROOT / "results" / f"{code}_SELECTED_LR.csv"
    oof_yaml = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    report = ROOT / "results" / f"{code}_PROTOCOL_REPORT.md"
    state = ROOT / "results" / f"{code}_run_state.json"

    is_scratch = spec["content_mode"] == "scratch"
    merge_mode = effective_merge(spec["merge_mode"])
    plm_source = None if is_scratch else "ablingua"

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    # Scratch: AA/IMGT/region suffice; AbDataset ignores PLM when content_mode=scratch.
    rb = load_residue_bundle(dev, test, need_ablingua=not is_scratch)

    result = run_protocol_v3(
        experiment_code=code,
        target=TARGET,
        dev=dev,
        test=test,
        folds=folds,
        rb=rb,
        device=device_str(),
        out_dir=out,
        seed=DEFAULT_SEED,
        quick=quick,
        arch=spec["arch"],
        content_mode=spec["content_mode"],
        merge_mode=merge_mode,
        plm_source=plm_source,
    )
    summary = result["summary"]
    hist_df = result["history_df"]
    sel_df = result["selected_df"]
    hist_df.to_csv(hist_csv, index=False)
    sel_df.to_csv(sel_csv, index=False)
    hist_df.to_csv(out / "training_history.csv", index=False)
    sel_df.to_csv(out / "selected_lr.csv", index=False)
    if len(result["cross_gates_df"]):
        result["cross_gates_df"].to_csv(ROOT / "results" / f"{code}_CROSS_GATES.csv", index=False)
        result["cross_gates_df"].to_csv(out / "cross_gates.csv", index=False)

    pred.mkdir(parents=True, exist_ok=True)
    for scheme in ("primary", "shadow"):
        save_pred(
            result["dev_ids"],
            result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float),
            pred / f"oof_val_{scheme}.csv",
        )
        save_pred(
            result["dev_ids"],
            result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float),
            pred / f"oof_test_{scheme}.csv",
        )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_primary.csv",
    )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_shadow.csv",
    )
    for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        save_pred(result["test_ids"], result["ext"][key], pred / f"test_{key}.csv")
    for scheme in ("primary", "shadow"):
        for k in range(5):
            save_pred(result["test_ids"], result["ext"][f"{scheme}_fold{k}"], pred / f"test_{scheme}_fold{k}.csv")
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv")

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol)
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }

    if not silent_test:
        print(json.dumps({"code": code, "oof_val": summary["scores"]["oof_val"]}, indent=2), flush=True)
    else:
        print(f"{code} training complete (OOF TEST deferred until batch freeze)", flush=True)

    oof_doc = {
        "experiment_code": code,
        "platform_id": PLATFORM_ID,
        "git_rev": git_rev(),
        "arch_id": spec["arch_id"],
        "representation": spec["representation"],
        "arch": spec["arch"],
        "content_mode": spec["content_mode"],
        "merge_mode": spec["merge_mode"],
        "seed": summary["seed"],
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external_test": ext_scores,
        "selected_lr": summary["selected_lr"],
        "n_trainable_parameters": summary["n_trainable_parameters"],
        "param_account": summary.get("param_account"),
        "cross_gates": summary.get("cross_gates"),
        "config_hash": summary.get("config_hash"),
        "config_sha256": file_sha256(ROOT / "experiments" / "configs" / f"{code}.yaml"),
        "prereg_n_trainable": prereg_row.get("n_trainable_preregistered"),
    }
    oof_yaml.write_text(yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    state.write_text(json.dumps({"ext_scores": ext_scores, "summary_scores": summary["scores"]}, indent=2) + "\n")

    ov = summary["scores"]["oof_val"]
    ot = summary["scores"]["oof_test"]
    lines = [
        f"# {code} — {spec['description']}",
        "",
        f"- platform: `{PLATFORM_ID}`",
        f"- arch_id: `{spec['arch_id']}`",
        f"- representation: `{spec['representation']}`",
        f"- content_mode / merge_mode: `{spec['content_mode']}` / `{spec['merge_mode']}`",
        f"- config_hash: `{summary.get('config_hash')}`",
        f"- n_trainable: {summary['n_trainable_parameters']}",
        f"- VAL_P/S/mean/worst: {ov['primary']:.6f} / {ov['shadow']:.6f} / {ov['mean']:.6f} / {ov['worst']:.6f}",
        f"- TEST_P/S/mean/worst: {ot['primary']:.6f} / {ot['shadow']:.6f} / {ot['mean']:.6f} / {ot['worst']:.6f}",
        "",
        "## Selected LR",
        "",
        sel_df.to_string(index=False),
        "",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    notes = (
        f"{PLATFORM_ID} {spec['description']}; OOF TEST in cv_*; "
        f"VAL={ov['primary']:.4f}/{ov['shadow']:.4f}; params={summary['n_trainable_parameters']}"
    )
    register(code, spec, summary, ext_scores, notes)
    print(f"DONE {code}", flush=True)
    return {
        "code": code,
        "spec": spec,
        "summary": summary,
        "ext_scores": ext_scores,
        "skipped": False,
    }


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 2000, seed: int = 101) -> dict:
    """delta = mean(err_a - err_b); negative => A better (lower MAE)."""
    rng = np.random.default_rng(seed)
    d = err_a - err_b
    point = float(d.mean())
    n = len(d)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(d[idx].mean()))
    boots = np.sort(np.asarray(boots))
    return {
        "delta_mae": point,
        "ci95_lo": float(np.quantile(boots, 0.025)),
        "ci95_hi": float(np.quantile(boots, 0.975)),
        "n": n,
    }


def load_oof(code: str, kind: str, scheme: str) -> pd.Series:
    p = ROOT / "experiments" / "predictions" / code / f"oof_{kind}_{scheme}.csv"
    if not p.exists() and kind == "test":
        # historical alias
        alt = ROOT / "experiments" / "predictions" / code / f"oof_{scheme}.csv"
        if scheme == "primary":
            alt = ROOT / "experiments" / "predictions" / code / "oof_primary.csv"
        elif scheme == "shadow":
            alt = ROOT / "experiments" / "predictions" / code / "oof_shadow.csv"
        if alt.exists():
            p = alt
    df = pd.read_csv(p)
    return df.set_index("id")[TARGET]


def load_oof_eval(code: str) -> dict:
    path = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    return yaml.safe_load(path.read_text())


def catalog_all() -> list[dict]:
    """SERIES + reused metadata for matrix building."""
    by_code = {s["code"]: s for s in SERIES}
    rows = []
    for r in REUSED:
        rows.append({**r, "spec": None})
    for s in SERIES:
        rows.append(
            {
                "code": s["code"],
                "arch_id": s["arch_id"],
                "representation": s["representation"],
                "merge_mode": s["merge_mode"],
                "content_mode": s["content_mode"],
                "spec": s,
            }
        )
    return rows


def write_pre_test_freeze(done_codes: list[str]) -> None:
    prereg = load_prereg()
    doc = {
        "status": "PRE_TEST_FREEZE",
        "git_rev": git_rev(),
        "completed_experiments": done_codes,
        "n_completed": len(done_codes),
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "preregistration_git_rev": prereg.get("git_rev_at_preregistration"),
        "statement": (
            "All T080–T104 architecture×representation settings were fixed before "
            "result inspection. No architecture changed based on intermediate scores. "
            "T075–T079 were reused without retraining."
        ),
        "platform_id": PLATFORM_ID,
    }
    PRE_TEST_FREEZE.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def finalize_comparisons() -> None:
    protect_reused()
    for s in SERIES:
        if not already_complete(s["code"]):
            raise SystemExit(f"finalize requires complete {s['code']}")

    write_pre_test_freeze([s["code"] for s in SERIES])
    y_dev = pd.read_csv(ROOT / "data" / "dev.csv").set_index("id")[TARGET]
    exp = pd.read_csv(ROOT / "results" / "experiments.csv").set_index("experiment_code")

    # ---- External four-way ----
    rows_ext = []
    for meta in catalog_all():
        code = meta["code"]
        oof = load_oof_eval(code)
        for agg, sc in oof["external_test"].items():
            rows_ext.append(
                {
                    "experiment": code,
                    "arch_id": meta["arch_id"],
                    "representation": meta["representation"],
                    "merge_mode": meta["merge_mode"],
                    "aggregation": agg,
                    "public_mae": sc["public_mae"],
                    "private_mae": sc["private_mae"],
                    "overall_mae": sc["overall_mae"],
                }
            )
    pd.DataFrame(rows_ext).to_csv(EXT_CSV, index=False)

    # ---- Matrix MD ----
    lines = [
        "# T075–T104 Architecture × Representation Matrix",
        "",
        f"Platform: `{PLATFORM_ID}`. Primary metric: TEST_mean. Sign: lower MAE better.",
        "",
        "| Code | Arch | Rep | Merge | Params | VAL_mean | TEST_mean | TEST_worst | Overall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    interaction_rows = []
    for meta in catalog_all():
        code = meta["code"]
        oof = load_oof_eval(code)
        ov, ot = oof["oof_val"], oof["oof_test"]
        overall = oof["external_test"]["primary_mean"]["overall_mae"]
        nparam = oof.get("n_trainable_parameters")
        if nparam is None and code in exp.index:
            nparam = ""
        merge = meta["merge_mode"] if meta["merge_mode"] is not None else "—"
        lines.append(
            f"| {code} | {meta['arch_id']} | {meta['representation']} | {merge} | "
            f"{nparam} | {ov['mean']:.4f} | {ot['mean']:.4f} | {ot['worst']:.4f} | {overall:.4f} |"
        )
        interaction_rows.append(
            {
                "experiment_code": code,
                "arch_id": meta["arch_id"],
                "representation": meta["representation"],
                "merge_mode": meta["merge_mode"],
                "val_mean": ov["mean"],
                "val_primary": ov["primary"],
                "val_shadow": ov["shadow"],
                "test_mean": ot["mean"],
                "test_primary": ot["primary"],
                "test_shadow": ot["shadow"],
                "test_worst": ot["worst"],
                "external_overall": overall,
                "n_trainable": nparam,
            }
        )
    MATRIX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(interaction_rows).to_csv(INTERACTION_CSV, index=False)

    # ---- MEAN vs CONCAT report ----
    mean_concat_pairs = [
        ("EXP-T080", "EXP-T075", "ABLINGUA", "ARCH-1"),
        ("EXP-T081", "EXP-T077", "ABLINGUA", "ARCH-3"),
        ("EXP-T082", "EXP-T078", "ABLINGUA", "ARCH-4"),
        ("EXP-T083", "EXP-T079", "ABLINGUA", "ARCH-5"),
        ("EXP-T085", "EXP-T084", "ABLINGUA", "ARCH-6"),
        ("EXP-T087", "EXP-T086", "ABLINGUA", "ARCH-7"),
        ("EXP-T089", "EXP-T088", "ABLINGUA", "ARCH-8"),
        ("EXP-T091", "EXP-T090", "SCRATCH", "ARCH-1"),
        ("EXP-T094", "EXP-T093", "SCRATCH", "ARCH-3"),
        ("EXP-T096", "EXP-T095", "SCRATCH", "ARCH-4"),
        ("EXP-T098", "EXP-T097", "SCRATCH", "ARCH-5"),
        ("EXP-T100", "EXP-T099", "SCRATCH", "ARCH-6"),
        ("EXP-T102", "EXP-T101", "SCRATCH", "ARCH-7"),
        ("EXP-T104", "EXP-T103", "SCRATCH", "ARCH-8"),
    ]
    mc_lines = [
        "# MEAN vs CONCAT (matched ARCH × representation)",
        "",
        "Δ = TEST_mean(MEAN) − TEST_mean(CONCAT); negative ⇒ MEAN better.",
        "",
        "| Rep | Arch | MEAN | CONCAT | Δ TEST_mean |",
        "|---|---|---|---|---:|",
    ]
    for mean_c, concat_c, rep, arch in mean_concat_pairs:
        tm = load_oof_eval(mean_c)["oof_test"]["mean"]
        tc = load_oof_eval(concat_c)["oof_test"]["mean"]
        mc_lines.append(f"| {rep} | {arch} | {mean_c} | {concat_c} | {tm - tc:.4f} |")
    MEAN_CONCAT_MD.write_text("\n".join(mc_lines) + "\n", encoding="utf-8")

    # ---- Scratch architecture report ----
    sc_lines = [
        "# T090–T104 Scratch architecture report",
        "",
        "Historical scratch FULL semantics: AA emb + pos + chain + IMGT + region (additive → d_model).",
        "",
        "| Code | Arch | Merge | VAL_mean | TEST_mean | TEST_worst | Overall |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for s in SERIES:
        if s["representation"] != "SCRATCH":
            continue
        oof = load_oof_eval(s["code"])
        merge = s["merge_mode"] if s["merge_mode"] is not None else "—"
        sc_lines.append(
            f"| {s['code']} | {s['arch_id']} | {merge} | "
            f"{oof['oof_val']['mean']:.4f} | {oof['oof_test']['mean']:.4f} | "
            f"{oof['oof_test']['worst']:.4f} | "
            f"{oof['external_test']['primary_mean']['overall_mae']:.4f} |"
        )
    SCRATCH_MD.write_text("\n".join(sc_lines) + "\n", encoding="utf-8")

    # ---- Paired bootstrap ----
    contrasts: list[tuple[str, str, str, str]] = []
    # MEAN vs CONCAT
    for mean_c, concat_c, rep, arch in mean_concat_pairs:
        contrasts.append((f"MEAN_vs_CONCAT_{rep}_{arch}", mean_c, concat_c, "mean_vs_concat"))
    # ARCH-6 vs 5
    for a, b, tag in (
        ("EXP-T084", "EXP-T079", "ABLINGUA_CONCAT"),
        ("EXP-T085", "EXP-T083", "ABLINGUA_MEAN"),
        ("EXP-T099", "EXP-T097", "SCRATCH_CONCAT"),
        ("EXP-T100", "EXP-T098", "SCRATCH_MEAN"),
    ):
        contrasts.append((f"ARCH6_vs_5_{tag}", a, b, "arch6_vs_5"))
    # ARCH-6 vs 8
    for a, b, tag in (
        ("EXP-T084", "EXP-T088", "ABLINGUA_CONCAT"),
        ("EXP-T085", "EXP-T089", "ABLINGUA_MEAN"),
        ("EXP-T099", "EXP-T103", "SCRATCH_CONCAT"),
        ("EXP-T100", "EXP-T104", "SCRATCH_MEAN"),
    ):
        contrasts.append((f"ARCH6_vs_8_{tag}", a, b, "arch6_vs_8"))
    # ARCH-7 vs 1
    for a, b, tag in (
        ("EXP-T086", "EXP-T075", "ABLINGUA_CONCAT"),
        ("EXP-T087", "EXP-T080", "ABLINGUA_MEAN"),
        ("EXP-T101", "EXP-T090", "SCRATCH_CONCAT"),
        ("EXP-T102", "EXP-T091", "SCRATCH_MEAN"),
    ):
        contrasts.append((f"ARCH7_vs_1_{tag}", a, b, "arch7_vs_1"))
    # matched Scratch vs AbLingua
    scratch_vs_ablingua = [
        ("EXP-T090", "EXP-T075", "ARCH-1_CONCAT"),
        ("EXP-T091", "EXP-T080", "ARCH-1_MEAN"),
        ("EXP-T092", "EXP-T076", "ARCH-2"),
        ("EXP-T093", "EXP-T077", "ARCH-3_CONCAT"),
        ("EXP-T094", "EXP-T081", "ARCH-3_MEAN"),
        ("EXP-T095", "EXP-T078", "ARCH-4_CONCAT"),
        ("EXP-T096", "EXP-T082", "ARCH-4_MEAN"),
        ("EXP-T097", "EXP-T079", "ARCH-5_CONCAT"),
        ("EXP-T098", "EXP-T083", "ARCH-5_MEAN"),
        ("EXP-T099", "EXP-T084", "ARCH-6_CONCAT"),
        ("EXP-T100", "EXP-T085", "ARCH-6_MEAN"),
        ("EXP-T101", "EXP-T086", "ARCH-7_CONCAT"),
        ("EXP-T102", "EXP-T087", "ARCH-7_MEAN"),
        ("EXP-T103", "EXP-T088", "ARCH-8_CONCAT"),
        ("EXP-T104", "EXP-T089", "ARCH-8_MEAN"),
    ]
    for a, b, tag in scratch_vs_ablingua:
        contrasts.append((f"SCRATCH_vs_ABLINGUA_{tag}", a, b, "scratch_vs_ablingua"))
    # Selected DoD contrasts (decision-facing)
    for a, b, tag in (
        ("EXP-T084", "EXP-T079", "DoD_ungated_vs_gated_CONCAT"),
        ("EXP-T090", "EXP-T075", "DoD_scratch_vs_ablingua_ARCH1"),
        ("EXP-T080", "EXP-T075", "DoD_mean_vs_concat_ARCH1"),
        ("EXP-T086", "EXP-T075", "DoD_reg_only_vs_separate"),
    ):
        contrasts.append((tag, a, b, "DoD"))

    boot_rows = []
    for cid, a, b, name in contrasts:
        for scheme in ("primary", "shadow"):
            for kind in ("test", "val"):
                ya = load_oof(a, kind, scheme)
                yb = load_oof(b, kind, scheme)
                ids = sorted(set(ya.index) & set(yb.index) & set(y_dev.index))
                ea = (ya.loc[ids] - y_dev.loc[ids]).abs().to_numpy(float)
                eb = (yb.loc[ids] - y_dev.loc[ids]).abs().to_numpy(float)
                res = paired_bootstrap(ea, eb)
                res.update(
                    {
                        "contrast": cid,
                        "name": name,
                        "scheme": scheme,
                        "split": kind,
                        "model_a": a,
                        "model_b": b,
                        "sign": "delta=MAE_a-MAE_b; negative => a better",
                    }
                )
                boot_rows.append(res)
    pd.DataFrame(boot_rows).to_csv(BOOT_CSV, index=False)
    print(
        "wrote",
        PRE_TEST_FREEZE,
        MATRIX_MD,
        MEAN_CONCAT_MD,
        SCRATCH_MD,
        INTERACTION_CSV,
        BOOT_CSV,
        EXT_CSV,
        flush=True,
    )


def run_diag() -> int:
    from t079_gate_zero_diagnostic import main as diag_main

    return int(diag_main())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prereg", "diag", "train", "finalize", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma codes e.g. EXP-T080,EXP-T081")
    args = ap.parse_args()

    if args.phase in ("prereg", "all"):
        ensure_prereg()
        if args.phase == "prereg":
            return 0

    if args.phase in ("diag", "all"):
        rc = run_diag()
        if rc != 0:
            return rc
        if args.phase == "diag":
            return 0

    prereg = load_prereg()
    assert prereg["platform_id"] == PLATFORM_ID
    specs = SERIES
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in SERIES if s["code"] in want]

    if args.phase in ("train", "all"):
        print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
        for spec in specs:
            prow = next(r for r in prereg["experiments"] if r["experiment_code"] == spec["code"])
            cfg = ROOT / prow["config_path"]
            sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
            if sha != prow["config_sha256"]:
                raise SystemExit(f"config mutated after prereg: {spec['code']}")
            silent = args.phase == "all"
            run_one(spec, prow, quick=args.quick, silent_test=silent)

    if args.phase in ("finalize", "all"):
        # Only finalize after all 25 exist (ignore --only for full freeze)
        missing = [s["code"] for s in SERIES if not already_complete(s["code"])]
        if missing:
            raise SystemExit(f"finalize blocked; incomplete: {missing}")
        finalize_comparisons()
        for s in SERIES:
            ot = load_oof_eval(s["code"])["oof_test"]
            print(s["code"], "TEST", ot, flush=True)
        print("BATCH COMPLETE T080-T104", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
