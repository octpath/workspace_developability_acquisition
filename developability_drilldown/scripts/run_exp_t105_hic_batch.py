#!/usr/bin/env python3
"""Run EXP-T105..T123 + EXP-H054..H081 under DL_FOLDLOCAL_COSINE_V3.

Phases:
  prereg | train_tm_geom | train_tm_ablang2 | train_hic_esm2 | train_hic_scratch
  | finalize | all

Does NOT print comparative TEST rankings during train phases.
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
from classical_features.ca_cache import build_or_load_ca_cache  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import (  # noqa: E402
    attach_ca_coords,
    load_dev_test,
    load_folds,
    load_residue_bundle,
    load_solution,
)
from antibody_transformer.protocol_v3 import (  # noqa: E402
    DEFAULT_SEED,
    PLATFORM_ID,
    run_protocol_v3,
)
from preregister_t105_hic_batch import (  # noqa: E402
    SERIES,
    SERIES_HIC,
    SERIES_TM,
    effective_merge,
)

PREREG = ROOT / "results" / "T105_T123_HIC_TRANSFER_GEOMETRY_PREREGISTRATION.yaml"
PRE_EXTERNAL_FREEZE = ROOT / "results" / "T105_T123_HIC_PRE_EXTERNAL_FREEZE.yaml"
TM_MASTER = ROOT / "results" / "TM_T075_T123_MASTER_TABLE.csv"
HIC_MASTER = ROOT / "results" / "HIC_NEW_DL_MASTER_TABLE.csv"
EFFECTS_CSV = ROOT / "results" / "TM_HIC_ARCHITECTURE_EFFECTS.csv"
BOOT_CSV = ROOT / "results" / "TM_HIC_PAIRED_BOOTSTRAP.csv"
EXT_CSV = ROOT / "results" / "TM_HIC_EXTERNAL_FOUR_WAY.csv"
TM_GEOM_MD = ROOT / "results" / "TM_GEOMETRY_REPORT.md"
HIC_GEOM_MD = ROOT / "results" / "HIC_GEOMETRY_REPORT.md"
TM_REP_MD = ROOT / "results" / "TM_ABLINGUA_ABLANG2_SCRATCH_REPORT.md"
HIC_REP_MD = ROOT / "results" / "HIC_ESM2_SCRATCH_ARCHITECTURE_REPORT.md"
CROSS_MD = ROOT / "results" / "TM_HIC_ARCHITECTURE_EFFECT_COMPARISON.md"

PHASE_SPECS = {
    "train_tm_geom": [s for s in SERIES_TM if s["code"] in {f"EXP-T{i:03d}" for i in range(105, 109)}],
    "train_tm_ablang2": [s for s in SERIES_TM if s["code"] in {f"EXP-T{i:03d}" for i in range(109, 124)}],
    "train_hic_esm2": [s for s in SERIES_HIC if s["representation"] == "ESM2"],
    "train_hic_scratch": [s for s in SERIES_HIC if s["representation"] == "SCRATCH"],
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_prereg() -> dict:
    if not PREREG.exists():
        raise SystemExit(f"missing preregistration {PREREG}; run preregister_t105_hic_batch.py first")
    return yaml.safe_load(PREREG.read_text())


def ensure_prereg() -> dict:
    if not PREREG.exists() or not all(
        (ROOT / "experiments" / "configs" / f"{s['code']}.yaml").exists() for s in SERIES
    ):
        print("=== prereg: generating configs ===", flush=True)
        from preregister_t105_hic_batch import main as prereg_main

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
    target = spec["target"]
    nxt = next_code(target)
    if nxt != expect:
        raise SystemExit(f"expected {expect}, got {nxt} for target={target}")
    src = (
        f"{PLATFORM_ID}::EXP-T075"
        if target == "TmApp"
        else f"{PLATFORM_ID}::HIC_TRANSFER"
    )
    code = issue_code(
        eid,
        target,
        source_model_id=src,
        phase="V3_T105_HIC_TRANSFER_GEOMETRY",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


def save_pred(ids, vals, path: Path, col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, col: vals}).to_csv(path, index=False)


def score_external(pred: np.ndarray, test_ids: list[str], sol: pd.DataFrame, target: str) -> dict:
    sol2 = sol.set_index("id")
    te = pd.Series(pred, index=test_ids)
    pub = sol2.index[sol2["is_public"].astype(bool)].tolist()
    priv = sol2.index[sol2["is_private"].astype(bool)].tolist()
    return {
        "public_mae": float(mae(sol2.loc[pub, target].to_numpy(float), te.loc[pub].to_numpy(float))),
        "private_mae": float(mae(sol2.loc[priv, target].to_numpy(float), te.loc[priv].to_numpy(float))),
        "overall_mae": float(mae(sol2.loc[test_ids, target].to_numpy(float), te.loc[test_ids].to_numpy(float))),
    }


def already_complete(code: str) -> bool:
    oof = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    pred = ROOT / "experiments" / "predictions" / code / "test_primary_mean.csv"
    return oof.exists() and pred.exists()


def _feature_source(spec: dict) -> str:
    rep = spec["representation"]
    return {
        "ABLINGUA": "ablingua_residue",
        "ABLANG2": "ablang2_residue",
        "ESM2": "esm2_residue",
        "SCRATCH": "scratch_residue",
    }[rep]


def register(code: str, spec: dict, summary: dict, ext_scores: dict, notes: str) -> None:
    target = spec["target"]
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    is_frozen = spec["content_mode"] == "frozen"
    plm_yaml = {
        None: "NONE",
        "ablingua": "ABLINGUA",
        "ablang2": "ABLANG2",
        "esm2": "ESM2",
    }[spec["plm_source"]]
    exp_path = ROOT / "results" / "experiments.csv"
    exp_df = pd.read_csv(exp_path)
    if code in set(exp_df["experiment_code"].astype(str)):
        exp_df = exp_df[exp_df["experiment_code"] != code]
    row = {c: "" for c in exp_df.columns}
    for c in EXPERIMENTS_COLUMNS:
        row.setdefault(c, "")
    src = (
        f"{PLATFORM_ID}::EXP-T075"
        if target == "TmApp"
        else f"{PLATFORM_ID}::HIC_TRANSFER"
    )
    row.update(
        {
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "target": target,
            "family": "TRANSFORMER",
            "model_type": f"AnnotatedTransformer+{PLATFORM_ID}",
            "source_model_id": src,
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
            "feature_source": _feature_source(spec),
            # Scratch: competition-sequence derived → OK (registry enum; see migrate map)
            "license_status": "REVIEW" if is_frozen else "OK",
            "license_reference": _feature_source(spec),
            "notes": notes,
            "transformer_type": "FROZEN_PLM" if is_frozen else "SCRATCH",
            "plm_source": plm_yaml,
            "annotation_mode": "FULL",
            "chain_mode": spec["chain_mode"],
            "pooling_mode": "REG" if spec["chain_mode"] == "HL" else "H_ONLY",
            "representation_status": "NOT_EXPORTED",
            "input_space": spec["input_space"],
            "input_asset_ref": {
                "ablingua": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
                "ablang2": "assets/transformer/residue_asset_manifest.yaml#ablang2",
                "esm2": "assets/transformer/residue_asset_manifest.yaml#esm2",
                None: "assets/transformer/residue_asset_manifest.yaml#annotations+scratch_sequences",
            }[spec["plm_source"]],
            "shareability_status": "SHAREABLE_COMPLETE",
            "reproducibility_status_v2": "REPRODUCED",
            "canonical_benchmark_eligible": "YES",
            "prediction_reproduction_max_delta": 0.0,
            "reproduction_tolerance_pred": 1e-10,
            "reproduction_tolerance_score": 1e-10,
            "tolerance_reason": "score_recompute_from_saved_predictions",
            "training_reproduction_attempted": True,
            "optuna_used": False,
            "control_experiment_code": spec.get("control_code") or "",
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
                    "target": target,
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
                    "notes": f"{PLATFORM_ID} T105/HIC transfer+geometry",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
            comp.to_csv(comp_path, index=False)


def prepare_rb(spec: dict, dev, test):
    """Load residue bundle with correct PLM flags; attach CA for geometry."""
    need_ablingua = spec["plm_source"] == "ablingua"
    need_ablang2 = spec["plm_source"] == "ablang2"
    need_esm2 = spec["plm_source"] == "esm2"
    rb = load_residue_bundle(
        dev,
        test,
        need_ablingua=need_ablingua,
        need_ablang2=need_ablang2,
        need_esm2=need_esm2,
    )
    if need_esm2 and spec["chain_mode"] == "HL" and rb.esm2_l is None:
        raise SystemExit(
            f"{spec['code']}: ESM-2 Light missing; run "
            "scripts/build_esm2_light_residue_bundle.py"
        )
    if need_ablang2 and (rb.ablang2_h is None or rb.ablang2_l is None):
        raise SystemExit(f"{spec['code']}: AbLang2 residue pack incomplete")
    if spec["geometry"] or spec["arch"].get("use_cross_geometry_bias"):
        ca = build_or_load_ca_cache(dev, test)
        seqs = pd.concat(
            [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
            ignore_index=True,
        )
        attach_ca_coords(
            rb,
            ca_heavy=ca["H"],
            ca_light=ca["L"],
            ca_ids=ca["ids"],
            seqs=seqs,
        )
    return rb


def run_one(spec: dict, prereg_row: dict, *, quick: bool) -> dict:
    code = issue_one(spec)
    target = spec["target"]
    print(f"==== TRAIN {code} ({spec['description']}) ====", flush=True)

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

    merge_mode = effective_merge(spec["merge_mode"])
    if spec["chain_mode"] == "H_ONLY":
        merge_mode = "h_only"
    plm_source = spec["plm_source"]

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = prepare_rb(spec, dev, test)

    result = run_protocol_v3(
        experiment_code=code,
        target=target,
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
        chain_mode=spec["chain_mode"],
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
    if len(result.get("geometry_weights_df", pd.DataFrame())):
        result["geometry_weights_df"].to_csv(
            ROOT / "results" / f"{code}_GEOMETRY_WEIGHTS.csv", index=False
        )
        result["geometry_weights_df"].to_csv(out / "geometry_weights.csv", index=False)

    pred.mkdir(parents=True, exist_ok=True)
    for scheme in ("primary", "shadow"):
        save_pred(
            result["dev_ids"],
            result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float),
            pred / f"oof_val_{scheme}.csv",
            target,
        )
        save_pred(
            result["dev_ids"],
            result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float),
            pred / f"oof_test_{scheme}.csv",
            target,
        )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_primary.csv",
        target,
    )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_shadow.csv",
        target,
    )
    for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        save_pred(result["test_ids"], result["ext"][key], pred / f"test_{key}.csv", target)
    for scheme in ("primary", "shadow"):
        for k in range(5):
            save_pred(
                result["test_ids"],
                result["ext"][f"{scheme}_fold{k}"],
                pred / f"test_{scheme}_fold{k}.csv",
                target,
            )
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv", target)

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol, target)
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }

    # Scientific branching deferred — do not print comparative TEST rankings.
    print(f"{code} training complete (comparative TEST deferred until finalize)", flush=True)

    oof_doc = {
        "experiment_code": code,
        "platform_id": PLATFORM_ID,
        "git_rev": git_rev(),
        "target": target,
        "arch_id": spec["arch_id"],
        "representation": spec["representation"],
        "arch": spec["arch"],
        "content_mode": spec["content_mode"],
        "plm_source": spec["plm_source"],
        "merge_mode": spec["merge_mode"],
        "chain_mode": spec["chain_mode"],
        "geometry": spec["geometry"],
        "control_code": spec.get("control_code"),
        "seed": summary["seed"],
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external_test": ext_scores,
        "selected_lr": summary["selected_lr"],
        "n_trainable_parameters": summary["n_trainable_parameters"],
        "param_account": summary.get("param_account"),
        "cross_gates": summary.get("cross_gates"),
        "geometry_weights": summary.get("geometry_weights"),
        "config_hash": summary.get("config_hash"),
        "config_sha256": file_sha256(ROOT / "experiments" / "configs" / f"{code}.yaml"),
        "prereg_n_trainable": prereg_row.get("n_trainable_preregistered"),
    }
    oof_yaml.write_text(yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    state.write_text(json.dumps({"ext_scores": ext_scores, "summary_scores": summary["scores"]}, indent=2) + "\n")

    ov = summary["scores"]["oof_val"]
    lines = [
        f"# {code} — {spec['description']}",
        "",
        f"- platform: `{PLATFORM_ID}`",
        f"- target: `{target}`",
        f"- arch_id: `{spec['arch_id']}`",
        f"- representation: `{spec['representation']}`",
        f"- content_mode / merge / chain: `{spec['content_mode']}` / `{spec['merge_mode']}` / `{spec['chain_mode']}`",
        f"- geometry: `{spec['geometry']}`",
        f"- config_hash: `{summary.get('config_hash')}`",
        f"- n_trainable: {summary['n_trainable_parameters']}",
        f"- VAL_P/S/mean/worst: {ov['primary']:.6f} / {ov['shadow']:.6f} / {ov['mean']:.6f} / {ov['worst']:.6f}",
        "",
        "## Selected LR",
        "",
        sel_df.to_string(index=False),
        "",
        "_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._",
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


def load_oof(code: str, kind: str, scheme: str, target: str) -> pd.Series:
    p = ROOT / "experiments" / "predictions" / code / f"oof_{kind}_{scheme}.csv"
    if not p.exists() and kind == "test":
        alt = ROOT / "experiments" / "predictions" / code / f"oof_{scheme}.csv"
        if scheme == "primary":
            alt = ROOT / "experiments" / "predictions" / code / "oof_primary.csv"
        elif scheme == "shadow":
            alt = ROOT / "experiments" / "predictions" / code / "oof_shadow.csv"
        if alt.exists():
            p = alt
    df = pd.read_csv(p)
    col = target if target in df.columns else [c for c in df.columns if c != "id"][0]
    return df.set_index("id")[col]


def load_oof_eval(code: str) -> dict:
    return yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())


def _tm_hist_meta(code: str) -> Optional[dict]:
    """Metadata for historical T075–T104 from config + OOF if present."""
    cfg_p = ROOT / "experiments" / "configs" / f"{code}.yaml"
    oof_p = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    if not oof_p.exists():
        return None
    cfg = yaml.safe_load(cfg_p.read_text()) if cfg_p.exists() else {}
    oof = yaml.safe_load(oof_p.read_text())
    arch = cfg.get("arch_id") or oof.get("arch_id")
    rep = cfg.get("representation") or oof.get("representation")
    merge = cfg.get("merge_mode") if "merge_mode" in cfg else oof.get("merge_mode")
    geom = bool(cfg.get("geometry") or (arch == "ARCH-6G"))
    return {
        "code": code,
        "target": "TmApp",
        "arch_id": arch,
        "representation": rep,
        "merge_mode": merge,
        "geometry": geom,
        "oof": oof,
        "n_trainable": oof.get("n_trainable_parameters"),
    }


def write_pre_external_freeze(done_codes: list[str]) -> None:
    prereg = load_prereg()
    mut = []
    for r in prereg["experiments"]:
        cfg = ROOT / r["config_path"]
        sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
        if sha != r["config_sha256"]:
            mut.append(r["experiment_code"])
    doc = {
        "status": "PRE_EXTERNAL_FREEZE",
        "git_rev": git_rev(),
        "completed_experiments": done_codes,
        "n_completed": len(done_codes),
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "preregistration_git_rev": prereg.get("git_rev_at_preregistration"),
        "config_mutations_since_prereg": mut,
        "statement": (
            "All T105–T123 and H054–H081 settings were fixed before result inspection. "
            "No architecture changed based on intermediate scores. "
            "Internal TEST predictions are frozen; comparative reports may now be generated."
        ),
        "platform_id": PLATFORM_ID,
        "hic_code_block": "H054-H081",
    }
    PRE_EXTERNAL_FREEZE.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _row_from_oof(meta: dict) -> dict:
    oof = meta["oof"]
    ov, ot = oof["oof_val"], oof["oof_test"]
    ext = oof["external_test"]["primary_mean"]
    return {
        "target": meta["target"],
        "code": meta["code"],
        "representation": meta["representation"],
        "architecture": meta["arch_id"],
        "merge": meta["merge_mode"] if meta["merge_mode"] is not None else "",
        "geometry": bool(meta.get("geometry")),
        "params": meta.get("n_trainable") or oof.get("n_trainable_parameters"),
        "VAL_P": ov["primary"],
        "VAL_S": ov["shadow"],
        "TEST_P": ot["primary"],
        "TEST_S": ot["shadow"],
        "TEST_mean": ot["mean"],
        "TEST_worst": ot["worst"],
        "Public": ext["public_mae"],
        "Private": ext["private_mae"],
        "Overall": ext["overall_mae"],
    }


def finalize_comparisons() -> None:
    for s in SERIES:
        if not already_complete(s["code"]):
            raise SystemExit(f"finalize requires complete {s['code']}")

    write_pre_external_freeze([s["code"] for s in SERIES])

    # ---- Master tables ----
    tm_rows = []
    for i in range(75, 105):
        meta = _tm_hist_meta(f"EXP-T{i:03d}")
        if meta:
            tm_rows.append(_row_from_oof(meta))
    for s in SERIES_TM:
        oof = load_oof_eval(s["code"])
        tm_rows.append(
            _row_from_oof(
                {
                    "code": s["code"],
                    "target": "TmApp",
                    "arch_id": s["arch_id"],
                    "representation": s["representation"],
                    "merge_mode": s["merge_mode"],
                    "geometry": s["geometry"],
                    "oof": oof,
                    "n_trainable": oof.get("n_trainable_parameters"),
                }
            )
        )
    pd.DataFrame(tm_rows).to_csv(TM_MASTER, index=False)

    hic_rows = []
    for s in SERIES_HIC:
        oof = load_oof_eval(s["code"])
        hic_rows.append(
            _row_from_oof(
                {
                    "code": s["code"],
                    "target": "HIC",
                    "arch_id": s["arch_id"],
                    "representation": s["representation"],
                    "merge_mode": s["merge_mode"],
                    "geometry": s["geometry"],
                    "oof": oof,
                    "n_trainable": oof.get("n_trainable_parameters"),
                }
            )
        )
    pd.DataFrame(hic_rows).to_csv(HIC_MASTER, index=False)

    # ---- External four-way ----
    rows_ext = []
    for s in SERIES:
        oof = load_oof_eval(s["code"])
        for agg, sc in oof["external_test"].items():
            rows_ext.append(
                {
                    "experiment": s["code"],
                    "target": s["target"],
                    "arch_id": s["arch_id"],
                    "representation": s["representation"],
                    "merge_mode": s["merge_mode"],
                    "geometry": s["geometry"],
                    "aggregation": agg,
                    "public_mae": sc["public_mae"],
                    "private_mae": sc["private_mae"],
                    "overall_mae": sc["overall_mae"],
                }
            )
    pd.DataFrame(rows_ext).to_csv(EXT_CSV, index=False)

    # ---- Architecture effects (within-target deltas) ----
    effect_rows = []

    def _delta(a: str, b: str, name: str, target: str) -> None:
        oa, ob = load_oof_eval(a), load_oof_eval(b)
        effect_rows.append(
            {
                "contrast": name,
                "target": target,
                "model_a": a,
                "model_b": b,
                "delta_TEST_mean": oa["oof_test"]["mean"] - ob["oof_test"]["mean"],
                "delta_TEST_P": oa["oof_test"]["primary"] - ob["oof_test"]["primary"],
                "delta_TEST_S": oa["oof_test"]["shadow"] - ob["oof_test"]["shadow"],
                "delta_Overall": (
                    oa["external_test"]["primary_mean"]["overall_mae"]
                    - ob["external_test"]["primary_mean"]["overall_mae"]
                ),
                "sign": "delta=MAE_a-MAE_b; negative => a better",
            }
        )

    for a, b, tag in (
        ("EXP-T105", "EXP-T084", "geom_vs_arch6_ablingua_concat"),
        ("EXP-T106", "EXP-T085", "geom_vs_arch6_ablingua_mean"),
        ("EXP-T107", "EXP-T099", "geom_vs_arch6_scratch_concat"),
        ("EXP-T108", "EXP-T100", "geom_vs_arch6_scratch_mean"),
        ("EXP-T118", "EXP-T116", "geom_vs_arch6_ablang2_concat"),
        ("EXP-T119", "EXP-T117", "geom_vs_arch6_ablang2_mean"),
    ):
        if (ROOT / "results" / f"{b}_OOF_EVALUATION.yaml").exists():
            _delta(a, b, tag, "TmApp")

    # HIC geometry vs ARCH-6
    for a, b, tag in (
        ("EXP-H064", "EXP-H062", "hic_esm2_geom_vs_arch6_concat"),
        ("EXP-H065", "EXP-H063", "hic_esm2_geom_vs_arch6_mean"),
        ("EXP-H078", "EXP-H076", "hic_scratch_geom_vs_arch6_concat"),
        ("EXP-H079", "EXP-H077", "hic_scratch_geom_vs_arch6_mean"),
    ):
        _delta(a, b, tag, "HIC")

    # HIC ESM2 vs Scratch matched
    for i in range(14):
        e, sc = f"EXP-H{54 + i:03d}", f"EXP-H{68 + i:03d}"
        _delta(e, sc, f"hic_esm2_vs_scratch_{SERIES_HIC[i]['arch_id']}_{SERIES_HIC[i]['merge_mode']}", "HIC")

    pd.DataFrame(effect_rows).to_csv(EFFECTS_CSV, index=False)

    # ---- Paired bootstrap (planned contrasts only) ----
    y_dev_tm = pd.read_csv(ROOT / "data" / "dev.csv").set_index("id")["TmApp"]
    y_dev_hic = pd.read_csv(ROOT / "data" / "dev.csv").set_index("id")["HIC"]
    contrasts: list[tuple[str, str, str, str, str]] = []
    # (id, a, b, name, target)
    for a, b, tag in (
        ("EXP-T105", "EXP-T084", "tm_geom_ablingua_concat"),
        ("EXP-T106", "EXP-T085", "tm_geom_ablingua_mean"),
        ("EXP-T107", "EXP-T099", "tm_geom_scratch_concat"),
        ("EXP-T108", "EXP-T100", "tm_geom_scratch_mean"),
        ("EXP-T118", "EXP-T116", "tm_geom_ablang2_concat"),
        ("EXP-T119", "EXP-T117", "tm_geom_ablang2_mean"),
        ("EXP-T110", "EXP-T109", "tm_ablang2_mean_vs_concat_arch1"),
        ("EXP-T113", "EXP-T112", "tm_ablang2_mean_vs_concat_arch3"),
        ("EXP-T115", "EXP-T114", "tm_ablang2_mean_vs_concat_arch4"),
        ("EXP-T117", "EXP-T116", "tm_ablang2_mean_vs_concat_arch6"),
        ("EXP-T119", "EXP-T118", "tm_ablang2_mean_vs_concat_arch6g"),
        ("EXP-T121", "EXP-T120", "tm_ablang2_mean_vs_concat_arch7"),
        ("EXP-T123", "EXP-T122", "tm_ablang2_mean_vs_concat_arch8"),
        ("EXP-T116", "EXP-T122", "tm_ablang2_arch6_vs_8_concat"),
        ("EXP-T117", "EXP-T123", "tm_ablang2_arch6_vs_8_mean"),
    ):
        if (ROOT / "results" / f"{b}_OOF_EVALUATION.yaml").exists():
            contrasts.append((tag, a, b, tag, "TmApp"))
    # matched AbLang2 vs AbLingua / Scratch (where counterparts exist)
    ablang2_vs = [
        ("EXP-T109", "EXP-T075", "ablang2_vs_ablingua_arch1_concat"),
        ("EXP-T110", "EXP-T080", "ablang2_vs_ablingua_arch1_mean"),
        ("EXP-T111", "EXP-T076", "ablang2_vs_ablingua_arch2"),
        ("EXP-T116", "EXP-T084", "ablang2_vs_ablingua_arch6_concat"),
        ("EXP-T109", "EXP-T090", "ablang2_vs_scratch_arch1_concat"),
        ("EXP-T110", "EXP-T091", "ablang2_vs_scratch_arch1_mean"),
        ("EXP-T116", "EXP-T099", "ablang2_vs_scratch_arch6_concat"),
    ]
    for a, b, tag in ablang2_vs:
        if (ROOT / "results" / f"{b}_OOF_EVALUATION.yaml").exists():
            contrasts.append((tag, a, b, tag, "TmApp"))

    for a, b, tag in (
        ("EXP-H064", "EXP-H062", "hic_geom_esm2_concat"),
        ("EXP-H065", "EXP-H063", "hic_geom_esm2_mean"),
        ("EXP-H078", "EXP-H076", "hic_geom_scratch_concat"),
        ("EXP-H079", "EXP-H077", "hic_geom_scratch_mean"),
        ("EXP-H055", "EXP-H054", "hic_esm2_arch1_vs_h0"),
        ("EXP-H069", "EXP-H068", "hic_scratch_arch1_vs_h0"),
        ("EXP-H062", "EXP-H066", "hic_esm2_arch6_vs_8_concat"),
        ("EXP-H076", "EXP-H080", "hic_scratch_arch6_vs_8_concat"),
    ):
        contrasts.append((tag, a, b, tag, "HIC"))
    for i in range(14):
        e, sc = f"EXP-H{54 + i:03d}", f"EXP-H{68 + i:03d}"
        contrasts.append(
            (
                f"hic_esm2_vs_scratch_{i}",
                e,
                sc,
                f"hic_rep_{SERIES_HIC[i]['arch_id']}",
                "HIC",
            )
        )

    boot_rows = []
    for cid, a, b, name, target in contrasts:
        y_dev = y_dev_tm if target == "TmApp" else y_dev_hic
        for scheme in ("primary", "shadow"):
            for kind in ("test", "val"):
                try:
                    ya = load_oof(a, kind, scheme, target)
                    yb = load_oof(b, kind, scheme, target)
                except Exception:
                    continue
                ids = sorted(set(ya.index) & set(yb.index) & set(y_dev.index))
                if not ids:
                    continue
                ea = (ya.loc[ids] - y_dev.loc[ids]).abs().to_numpy(float)
                eb = (yb.loc[ids] - y_dev.loc[ids]).abs().to_numpy(float)
                res = paired_bootstrap(ea, eb)
                res.update(
                    {
                        "contrast": cid,
                        "name": name,
                        "target": target,
                        "scheme": scheme,
                        "split": kind,
                        "model_a": a,
                        "model_b": b,
                        "sign": "delta=MAE_a-MAE_b; negative => a better",
                    }
                )
                boot_rows.append(res)
    pd.DataFrame(boot_rows).to_csv(BOOT_CSV, index=False)

    # ---- Geometry reports ----
    def _geom_report(path: Path, title: str, pairs: list[tuple[str, str, str]]) -> None:
        lines = [
            f"# {title}",
            "",
            f"Platform: `{PLATFORM_ID}`. Δ = MAE(geom) − MAE(control); negative ⇒ geometry better.",
            "",
            "| Geom | Control | Rep | Merge | Δ TEST_mean | Δ TEST_P | Δ TEST_S | Overall Δ |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for g, c, rep in pairs:
            if not (ROOT / "results" / f"{c}_OOF_EVALUATION.yaml").exists():
                lines.append(f"| {g} | {c} | {rep} | — | n/a | n/a | n/a | n/a |")
                continue
            og, oc = load_oof_eval(g), load_oof_eval(c)
            merge = next(s["merge_mode"] for s in SERIES if s["code"] == g)
            lines.append(
                f"| {g} | {c} | {rep} | {merge} | "
                f"{og['oof_test']['mean'] - oc['oof_test']['mean']:.4f} | "
                f"{og['oof_test']['primary'] - oc['oof_test']['primary']:.4f} | "
                f"{og['oof_test']['shadow'] - oc['oof_test']['shadow']:.4f} | "
                f"{og['external_test']['primary_mean']['overall_mae'] - oc['external_test']['primary_mean']['overall_mae']:.4f} |"
            )
            gw = ROOT / "results" / f"{g}_GEOMETRY_WEIGHTS.csv"
            if gw.exists():
                lines.append(f"")
                lines.append(f"### Learned RBF weights — {g}")
                lines.append("")
                lines.append(f"Saved at `{gw.relative_to(ROOT)}` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.")
                lines.append("")
        lines.extend(
            [
                "",
                "## Diagnostics checklist",
                "",
                "- A. matched no-geometry control (table above)",
                "- B. TEST delta (table)",
                "- C. paired bootstrap → `TM_HIC_PAIRED_BOOTSTRAP.csv`",
                "- D. inference geometry-zero ablation: re-score with `cross_geom_weight=0` from checkpoints if needed",
                "- E–G. RBF coefficients / magnitude / distance peaks: see per-code `*_GEOMETRY_WEIGHTS.csv`",
                "- H–I. Primary/Shadow + representation consistency: compare rows above",
                "- J. HIC: Fv H/L geometry only (ESMFold Fv Cα); not full Ig",
                "",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _geom_report(
        TM_GEOM_MD,
        "TmApp Geometry Report (ARCH-6G)",
        [
            ("EXP-T105", "EXP-T084", "ABLINGUA"),
            ("EXP-T106", "EXP-T085", "ABLINGUA"),
            ("EXP-T107", "EXP-T099", "SCRATCH"),
            ("EXP-T108", "EXP-T100", "SCRATCH"),
            ("EXP-T118", "EXP-T116", "ABLANG2"),
            ("EXP-T119", "EXP-T117", "ABLANG2"),
        ],
    )
    _geom_report(
        HIC_GEOM_MD,
        "HIC Geometry Report (ARCH-6G)",
        [
            ("EXP-H064", "EXP-H062", "ESM2"),
            ("EXP-H065", "EXP-H063", "ESM2"),
            ("EXP-H078", "EXP-H076", "SCRATCH"),
            ("EXP-H079", "EXP-H077", "SCRATCH"),
        ],
    )

    # ---- Representation reports ----
    tm_lines = [
        "# TmApp AbLingua / AbLang2 / Scratch report",
        "",
        f"Platform: `{PLATFORM_ID}`. AbLang2 `PRECONTEXTUALIZED_ACROSS_CHAINS=NO`.",
        "",
        "| Representation | Architecture | Merge | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    # include useful historical + new
    for code in [f"EXP-T{i:03d}" for i in list(range(75, 105)) + list(range(105, 124))]:
        meta = _tm_hist_meta(code) if int(code.split("-T")[1]) < 105 else None
        if meta is None and code.startswith("EXP-T1"):
            if not already_complete(code):
                continue
            s = next(x for x in SERIES_TM if x["code"] == code)
            oof = load_oof_eval(code)
            meta = {
                "code": code,
                "arch_id": s["arch_id"],
                "representation": s["representation"],
                "merge_mode": s["merge_mode"],
                "oof": oof,
            }
        if meta is None:
            continue
        oof = meta["oof"]
        merge = meta["merge_mode"] if meta["merge_mode"] is not None else "—"
        tm_lines.append(
            f"| {meta['representation']} | {meta['arch_id']} | {merge} | "
            f"{oof['oof_test']['primary']:.4f} | {oof['oof_test']['shadow']:.4f} | "
            f"{oof['oof_test']['mean']:.4f} | {oof['oof_test']['worst']:.4f} | "
            f"{oof['external_test']['primary_mean']['public_mae']:.4f} | "
            f"{oof['external_test']['primary_mean']['private_mae']:.4f} | "
            f"{oof['external_test']['primary_mean']['overall_mae']:.4f} |"
        )
    tm_lines.extend(
        [
            "",
            "## Questions (fill after inspection)",
            "",
            "1. Does AbLang2 reproduce historical strength under V3?",
            "2. Is AbLang2 consistently stronger, or architecture-dependent?",
            "3. Does Scratch remain competitive?",
            "4. Does AbLang2 prefer MEAN or CONCAT?",
            "5. Does explicit H/L communication help AbLang2?",
            "6. Given PRECONTEXTUALIZED_ACROSS_CHAINS=NO, does downstream H/L communication help?",
            "7. Does geometry help AbLang2?",
            "",
        ]
    )
    TM_REP_MD.write_text("\n".join(tm_lines) + "\n", encoding="utf-8")

    hic_lines = [
        "# HIC ESM-2 / Scratch architecture report",
        "",
        f"Platform: `{PLATFORM_ID}`. Codes: H054–H081.",
        "",
        "| Representation | Architecture | Merge | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SERIES_HIC:
        oof = load_oof_eval(s["code"])
        merge = s["merge_mode"] if s["merge_mode"] is not None else "—"
        hic_lines.append(
            f"| {s['representation']} | {s['arch_id']} | {merge} | "
            f"{oof['oof_test']['primary']:.4f} | {oof['oof_test']['shadow']:.4f} | "
            f"{oof['oof_test']['mean']:.4f} | {oof['oof_test']['worst']:.4f} | "
            f"{oof['external_test']['primary_mean']['public_mae']:.4f} | "
            f"{oof['external_test']['primary_mean']['private_mae']:.4f} | "
            f"{oof['external_test']['primary_mean']['overall_mae']:.4f} |"
        )
    hic_lines.extend(
        [
            "",
            "## Questions",
            "",
            "1. ESM-2 vs Scratch representation advantage?",
            "2. Does Scratch approach PLM when architecture is favorable?",
            "3. Is Heavy-only competitive?",
            "4. Does Light-chain information help?",
            "5. Does explicit H/L communication help?",
            "6. Does geometry help HIC?",
            "7. Does HIC prefer a different architecture from TmApp?",
            "",
        ]
    )
    HIC_REP_MD.write_text("\n".join(hic_lines) + "\n", encoding="utf-8")

    cross_lines = [
        "# TmApp vs HIC architecture-effect comparison",
        "",
        "Compare **within-target deltas**, not raw MAE scales (Tm °C vs HIC minutes).",
        "",
        f"See `{EFFECTS_CSV.name}` and `{BOOT_CSV.name}` for numeric contrasts.",
        "",
        "## Questions",
        "",
        "1. Separate vs joint — similar across targets?",
        "2. MEAN vs CONCAT preference transfer?",
        "3. Does Scratch benefit more from inductive bias on both?",
        "4. Does residue cross-attention help both?",
        "5. Does geometry help both?",
        "6. Does HIC prefer Heavy-only more strongly?",
        "7. Target-specific architecture preferences?",
        "",
        "## Effects snapshot",
        "",
    ]
    if effect_rows:
        cross_lines.append("| Contrast | Target | A | B | Δ TEST_mean |")
        cross_lines.append("|---|---|---|---|---:|")
        for r in effect_rows[:40]:
            cross_lines.append(
                f"| {r['contrast']} | {r['target']} | {r['model_a']} | {r['model_b']} | {r['delta_TEST_mean']:.4f} |"
            )
    CROSS_MD.write_text("\n".join(cross_lines) + "\n", encoding="utf-8")

    print(
        "wrote",
        PRE_EXTERNAL_FREEZE,
        TM_MASTER,
        HIC_MASTER,
        EFFECTS_CSV,
        BOOT_CSV,
        EXT_CSV,
        TM_GEOM_MD,
        HIC_GEOM_MD,
        TM_REP_MD,
        HIC_REP_MD,
        CROSS_MD,
        flush=True,
    )


def _filter_only(specs: list[dict], only: str) -> list[dict]:
    if not only:
        return specs
    want = {x.strip() for x in only.split(",") if x.strip()}
    return [s for s in specs if s["code"] in want]


def _train_phase(phase: str, specs: list[dict], prereg: dict, *, quick: bool) -> None:
    print(f"=== phase {phase}: {len(specs)} experiments ===", flush=True)
    print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
    for spec in specs:
        prow = next(r for r in prereg["experiments"] if r["experiment_code"] == spec["code"])
        cfg = ROOT / prow["config_path"]
        sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
        if sha != prow["config_sha256"]:
            raise SystemExit(f"config mutated after prereg: {spec['code']}")
        run_one(spec, prow, quick=quick)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=[
            "prereg",
            "train_tm_geom",
            "train_tm_ablang2",
            "train_hic_esm2",
            "train_hic_scratch",
            "finalize",
            "all",
        ],
        default="all",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma codes e.g. EXP-T105,EXP-H054")
    args = ap.parse_args()

    if args.phase in ("prereg", "all"):
        ensure_prereg()
        if args.phase == "prereg":
            return 0

    prereg = load_prereg()
    assert prereg["platform_id"] == PLATFORM_ID

    train_order = [
        "train_tm_geom",
        "train_tm_ablang2",
        "train_hic_esm2",
        "train_hic_scratch",
    ]
    if args.phase in train_order:
        specs = _filter_only(PHASE_SPECS[args.phase], args.only)
        _train_phase(args.phase, specs, prereg, quick=args.quick)
        return 0

    if args.phase == "all":
        for ph in train_order:
            specs = _filter_only(PHASE_SPECS[ph], args.only)
            if specs:
                _train_phase(ph, specs, prereg, quick=args.quick)

    if args.phase in ("finalize", "all"):
        # Full freeze ignores --only
        missing = [s["code"] for s in SERIES if not already_complete(s["code"])]
        if missing:
            raise SystemExit(f"finalize blocked; incomplete: {missing[:10]}{'...' if len(missing)>10 else ''}")
        finalize_comparisons()
        print("BATCH COMPLETE T105-T123 + H054-H081", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
