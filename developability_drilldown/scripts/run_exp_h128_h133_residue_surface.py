#!/usr/bin/env python3
"""EXP-H128–H133: residue-level F1_SURFACE fusion (additive / gated / surface-aware pool).

Phases: prereg | train | analyze | all
Schema frozen in results/H128_H133_RESIDUE_SURFACE_SCHEMA.yaml before training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import EXPERIMENTS_COLUMNS, FEATURE_SPACE, mae  # noqa: E402
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle, load_solution  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID, run_protocol_v3  # noqa: E402
from antibody_transformer.residue_f1_surface import (  # noqa: E402
    CHANNELS,
    COMPACT_PQ,
    P,
    attach_residue_surface_compact,
    load_schema,
    schema_hash,
)

PREREG = ROOT / "results" / "H128_H133_RESIDUE_SURFACE_PREREGISTRATION.yaml"
SCHEMA = ROOT / "results" / "H128_H133_RESIDUE_SURFACE_SCHEMA.yaml"
SEED = DEFAULT_SEED

SERIES = [
    {
        "code": "EXP-H128",
        "backbone": "EXP-H071",
        "control": "EXP-H071",
        "ab_surface": "EXP-H090",
        "mode": "additive",
        "content_mode": "scratch",
        "plm_source": None,
        "arch": {"joint_hl_single_reg": True},
        "merge_mode": "concat",
        "experiment_id": "TRF_HIC_SCRATCH_RESIDUE_F1_ADDITIVE_V3",
        "description": "H071 + residue F1 SURFACE additive",
    },
    {
        "code": "EXP-H129",
        "backbone": "EXP-H071",
        "control": "EXP-H071",
        "ab_surface": "EXP-H090",
        "mode": "gated",
        "content_mode": "scratch",
        "plm_source": None,
        "arch": {"joint_hl_single_reg": True},
        "merge_mode": "concat",
        "experiment_id": "TRF_HIC_SCRATCH_RESIDUE_F1_GATED_V3",
        "description": "H071 + residue F1 SURFACE gated",
    },
    {
        "code": "EXP-H130",
        "backbone": "EXP-H071",
        "control": "EXP-H071",
        "ab_surface": "EXP-H090",
        "mode": "surface_aware_pooling",
        "content_mode": "scratch",
        "plm_source": None,
        "arch": {"joint_hl_single_reg": True},
        "merge_mode": "concat",
        "experiment_id": "TRF_HIC_SCRATCH_RESIDUE_F1_POOL_V3",
        "description": "H071 + residue F1 SURFACE-aware pooling",
    },
    {
        "code": "EXP-H131",
        "backbone": "EXP-H061",
        "control": "EXP-H061",
        "ab_surface": "EXP-H086",
        "mode": "additive",
        "content_mode": "frozen",
        "plm_source": "esm2",
        "arch": {"joint_hl_chain_specific_dual_reg": True},
        "merge_mode": "mean",
        "experiment_id": "TRF_HIC_ESM2_RESIDUE_F1_ADDITIVE_V3",
        "description": "H061 + residue F1 SURFACE additive",
    },
    {
        "code": "EXP-H132",
        "backbone": "EXP-H061",
        "control": "EXP-H061",
        "ab_surface": "EXP-H086",
        "mode": "gated",
        "content_mode": "frozen",
        "plm_source": "esm2",
        "arch": {"joint_hl_chain_specific_dual_reg": True},
        "merge_mode": "mean",
        "experiment_id": "TRF_HIC_ESM2_RESIDUE_F1_GATED_V3",
        "description": "H061 + residue F1 SURFACE gated",
    },
    {
        "code": "EXP-H133",
        "backbone": "EXP-H061",
        "control": "EXP-H061",
        "ab_surface": "EXP-H086",
        "mode": "surface_aware_pooling",
        "content_mode": "frozen",
        "plm_source": "esm2",
        "arch": {"joint_hl_chain_specific_dual_reg": True},
        "merge_mode": "mean",
        "experiment_id": "TRF_HIC_ESM2_RESIDUE_F1_POOL_V3",
        "description": "H061 + residue F1 SURFACE-aware pooling",
    },
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def save_pred(ids, vals, path: Path, col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, col: vals}).to_csv(path, index=False)


def score_external(pred, test_ids, sol, target="HIC"):
    sol2 = sol.set_index("id")
    te = pd.Series(pred, index=test_ids)
    pub = sol2.index[sol2["is_public"].astype(bool)].tolist()
    priv = sol2.index[sol2["is_private"].astype(bool)].tolist()
    return {
        "public_mae": float(mae(sol2.loc[pub, target].to_numpy(float), te.loc[pub].to_numpy(float))),
        "private_mae": float(mae(sol2.loc[priv, target].to_numpy(float), te.loc[priv].to_numpy(float))),
        "overall_mae": float(mae(sol2.loc[test_ids, target].to_numpy(float), te.loc[test_ids].to_numpy(float))),
    }


def write_prereg() -> None:
    nxt = next_code("HIC")
    if nxt != "EXP-H128":
        raise SystemExit(f"expected next HIC EXP-H128, got {nxt}")
    schema = load_schema()
    if not SCHEMA.exists():
        raise SystemExit("schema must be frozen before prereg")
    doc = {
        "batch_id": "H128_H133_RESIDUE_F1_SURFACE_FUSION",
        "status": "PREREGISTERED_FROZEN",
        "git_rev_at_prereg": git_rev(),
        "schema_path": str(SCHEMA.relative_to(ROOT)),
        "schema_hash": schema_hash(),
        "schema_content_sha256": schema["compact_artifact"]["content_sha256"],
        "p": P,
        "channels": CHANNELS,
        "platform_id": PLATFORM_ID,
        "seed": SEED,
        "controls": {
            "sequence_only": ["EXP-H071", "EXP-H061"],
            "antibody_level_F1": ["EXP-H090", "EXP-H086"],
        },
        "experiments": [
            {
                "code": s["code"],
                "backbone": s["backbone"],
                "mode": s["mode"],
                "experiment_id": s["experiment_id"],
                "antibody_surface_control": s["ab_surface"],
            }
            for s in SERIES
        ],
        "do_not": ["SAP24", "SCM24", "cross_attention", "H134", "TEST_tuning"],
    }
    for s in SERIES:
        cfg = {
            "experiment_code": s["code"],
            "experiment_id": s["experiment_id"],
            "platform_id": PLATFORM_ID,
            "target": "HIC",
            "family": "TRANSFORMER",
            "control_experiment_code": s["control"],
            "antibody_surface_control": s["ab_surface"],
            "residue_surface_mode": s["mode"],
            "residue_surface_dim": P,
            "residue_surface_schema_id": "FS_HIC_RESIDUE_F1_SURFACE_COMPACT10",
            "residue_surface_schema_hash": schema_hash(),
            "content_mode": s["content_mode"],
            "plm_source": s["plm_source"] or "NONE",
            "fusion_bundle_id": None,
            "no_antibody_level_f1_concat": True,
            "seed": SEED,
            "description": s["description"],
            **{f"arch_{k}" if not k.startswith("joint") and not k.startswith("use") and not k.startswith("cross") and not k.startswith("share") else k: v for k, v in s["arch"].items()},
        }
        # flatten arch flags like other configs
        for k, v in s["arch"].items():
            cfg[k if k.startswith("joint") or k.startswith("use") or k.startswith("cross") or k.startswith("share") else f"arch_{k}"] = v
        cfg["merge_mode"] = s["merge_mode"]
        cfg["chain_mode"] = "HL"
        cfg["pooling_mode"] = "REG"
        cfg["no_full_dev_refit"] = True
        (ROOT / "experiments" / "configs" / f"{s['code']}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    PREREG.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print("Wrote", PREREG, flush=True)


def issue_one(spec: dict) -> str:
    codes = load_codes()
    eid = spec["experiment_id"]
    if (codes["experiment_id"] == eid).any():
        return str(codes.set_index("experiment_id").loc[eid, "experiment_code"])
    expect = spec["code"]
    if next_code("HIC") != expect:
        raise SystemExit(f"expected {expect}, got {next_code('HIC')}")
    code = issue_code(
        eid,
        "HIC",
        source_model_id=f"RESIDUE_F1_{spec['mode'].upper()}",
        phase="H128_RESIDUE_F1_SURFACE",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


def already_complete(code: str) -> bool:
    return (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").exists() and (
        ROOT / "experiments" / "predictions" / code / "test_primary_mean.csv"
    ).exists()


def register(code: str, spec: dict, summary: dict, ext_scores: dict, n_params: int) -> None:
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    exp_path = ROOT / "results" / "experiments.csv"
    df = pd.read_csv(exp_path)
    df = df[df["experiment_code"] != code]
    row = {c: "" for c in df.columns}
    for c in EXPERIMENTS_COLUMNS:
        row.setdefault(c, "")
    row.update(
        {
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "target": "HIC",
            "family": "TRANSFORMER",
            "model_type": "TRANSFORMER",
            "feature_set_id": "FS_HIC_RESIDUE_F1_SURFACE_COMPACT10",
            "feature_space": FEATURE_SPACE,
            "n_features": P,
            "config_path": f"experiments/configs/{code}.yaml",
            "artifact_status": "FULL",
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
            "license_status": "OK",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "REPRODUCED",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "shareability_status": "SHAREABLE_COMPLETE",
            "canonical_benchmark_eligible": "YES",
            "notes": spec["description"],
            "n_trainable": n_params,
            "control_experiment_code": spec["control"],
            "transformer_type": "RESIDUE_SURFACE_FUSION",
        }
    )
    pd.concat([df, pd.DataFrame([{c: row.get(c, "") for c in df.columns}])], ignore_index=True).to_csv(
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
                    "target": "HIC",
                    "family": "TRANSFORMER",
                    "config_exists": True,
                    "feature_required": False,
                    "feature_exists": True,
                    "oof_primary_exists": True,
                    "oof_shadow_exists": True,
                    "test_exists": True,
                    "score_recompute_ok": True,
                    "reproduction_status": "REPRODUCED",
                    "shareability_status": "SHAREABLE_COMPLETE",
                    "canonical_benchmark_eligible": "YES",
                    "notes": "residue F1 SURFACE fusion",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
            comp.to_csv(comp_path, index=False)


def run_one(spec: dict, *, quick: bool, rb, compact, folds, dev, test) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} mode={spec['mode']} ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        st = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
        return {"code": code, "summary": summary, "ext_scores": st["ext_scores"], "skipped": True}

    out = ROOT / "results" / f"{code}_run"
    result = run_protocol_v3(
        experiment_code=code,
        target="HIC",
        dev=dev,
        test=test,
        folds=folds,
        rb=rb,
        device=device_str(),
        out_dir=out,
        seed=SEED,
        quick=quick,
        arch=spec["arch"],
        content_mode=spec["content_mode"],
        merge_mode=spec["merge_mode"],
        plm_source=spec["plm_source"],
        chain_mode="HL",
        residue_surface_mode=spec["mode"],
        residue_surface_dim=P,
        residue_surface_compact=compact,
    )
    summary = result["summary"]
    sel = result["selected_df"]
    sel.to_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv", index=False)

    pred = ROOT / "experiments" / "predictions" / code
    pred.mkdir(parents=True, exist_ok=True)
    save_pred(result["dev_ids"], result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_primary.csv", "HIC")
    save_pred(result["dev_ids"], result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_shadow.csv", "HIC")
    for scheme in ("primary", "shadow"):
        save_pred(result["dev_ids"], result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_val_{scheme}.csv", "HIC")
        save_pred(result["dev_ids"], result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_test_{scheme}.csv", "HIC")
    for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        save_pred(result["test_ids"], result["ext"][key], pred / f"test_{key}.csv", "HIC")
    for scheme in ("primary", "shadow"):
        for k in range(5):
            save_pred(result["test_ids"], result["ext"][f"{scheme}_fold{k}"], pred / f"test_{scheme}_fold{k}.csv", "HIC")
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv", "HIC")

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol)
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }
    oof_doc = {
        "experiment_code": code,
        "residue_surface_mode": spec["mode"],
        "schema_hash": schema_hash(),
        "p": P,
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external": ext_scores,
        "n_trainable": summary.get("n_trainable"),
        "no_full_dev_refit": True,
        "selected": sel.to_dict(orient="records"),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8")
    (ROOT / "results" / f"{code}_run_state.json").write_text(json.dumps({"ext_scores": ext_scores}, indent=2))
    register(code, spec, summary, ext_scores, int(summary.get("n_trainable") or 0))
    print(f"DONE {code} TEST_mean={summary['scores']['oof_test']['mean']:.6f}", flush=True)
    return {"code": code, "summary": summary, "ext_scores": ext_scores, "skipped": False}


def write_report() -> None:
    exp = pd.read_csv(ROOT / "results" / "experiments.csv").set_index("experiment_code")
    controls = {
        "EXP-H071": float(exp.loc["EXP-H071", "cv_mean_mae"]),
        "EXP-H061": float(exp.loc["EXP-H061", "cv_mean_mae"]),
        "EXP-H090": float(exp.loc["EXP-H090", "cv_mean_mae"]),
        "EXP-H086": float(exp.loc["EXP-H086", "cv_mean_mae"]),
    }
    lines = [
        "# HIC Residue-Level F1_SURFACE Fusion (H128–H133)",
        "",
        f"Schema: compact p={P} (`FS_HIC_RESIDUE_F1_SURFACE_COMPACT10`). Schema hash `{schema_hash()[:16]}…`",
        "",
        "Channels: " + ", ".join(CHANNELS),
        "",
        "## Summary",
        "",
        "| Backbone | Mode | Code | TEST_P | TEST_S | TEST_mean | Δseq | ΔAb-SURFACE | Public | Private | Overall |",
        "|----------|------|------|-------:|-------:|----------:|-----:|------------:|-------:|--------:|--------:|",
    ]
    for s in SERIES:
        code = s["code"]
        oof = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
        ot = oof["oof_test"]
        ext = oof["external"]["primary_mean"]
        base = controls[s["control"]]
        absurf = controls[s["ab_surface"]]
        lines.append(
            f"| {s['backbone']} | {s['mode']} | {code} | {ot['primary']:.4f} | {ot['shadow']:.4f} | "
            f"{ot['mean']:.4f} | {ot['mean']-base:+.4f} | {ot['mean']-absurf:+.4f} | "
            f"{ext['public_mae']:.4f} | {ext['private_mae']:.4f} | {ext['overall_mae']:.4f} |"
        )
    lines += [
        "",
        "## Controls",
        "",
        f"- H071={controls['EXP-H071']:.4f}, H090={controls['EXP-H090']:.4f}",
        f"- H061={controls['EXP-H061']:.4f}, H086={controls['EXP-H086']:.4f}",
        "",
        "Diagnostics (bootstrap / zero-SURFACE / residue-shuffle) written by analyze phase when available.",
        "",
        "STOP after H133. Next HIC = EXP-H134 (DO NOT RUN).",
    ]
    (ROOT / "results" / "HIC_RESIDUE_F1_SURFACE_FUSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    freeze = {
        "status": "INTERNAL_COMPLETE",
        "codes": [s["code"] for s in SERIES],
        "schema_hash": schema_hash(),
        "git_rev": git_rev(),
        "controls": controls,
    }
    (ROOT / "results" / "H128_H133_PRE_EXTERNAL_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "train", "analyze", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.phase in ("prereg", "all"):
        write_prereg()
    if args.phase in ("train", "all"):
        if not PREREG.exists():
            write_prereg()
        compact = pd.read_parquet(COMPACT_PQ)
        assert list(compact.columns[-P:]) == CHANNELS or all(c in compact.columns for c in CHANNELS)
        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        need_esm2 = any(s["plm_source"] == "esm2" for s in SERIES)
        rb = load_residue_bundle(dev, test, need_esm2=need_esm2)
        attach_residue_surface_compact(rb, compact)
        for spec in SERIES:
            run_one(spec, quick=args.quick, rb=rb, compact=compact, folds=folds, dev=dev, test=test)
    if args.phase in ("analyze", "all"):
        write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
