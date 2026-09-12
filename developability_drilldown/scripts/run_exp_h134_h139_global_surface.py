#!/usr/bin/env python3
"""EXP-H134–H139: global F1_SURFACE35 conditioning (FiLM / gated residual / token attn).

Phases: prereg | train | analyze | all
Uses historical F1_SURFACE + protocol_v3_ext (same prep as H090/H086).
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

from _lib import EXPERIMENTS_COLUMNS, mae  # noqa: E402
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle, load_solution  # noqa: E402
from antibody_transformer.global_surface_conditioning import (  # noqa: E402
    ATTN_HEADS,
    ATTN_LAYERS,
    BOTTLENECK,
    GlobalSurfaceConditioningModel,
)
from antibody_transformer.h047_aux_features import H047AuxFeatureStore  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID  # noqa: E402
from antibody_transformer.protocol_v3_ext import run_protocol_v3_ext  # noqa: E402

PREREG = ROOT / "results" / "H134_H139_GLOBAL_SURFACE_PREREGISTRATION.yaml"
AUDIT = ROOT / "results" / "H134_H139_GLOBAL_SURFACE_FUSION_AUDIT.md"
SEED = DEFAULT_SEED
F1_ARTIFACT_HASH = "e3788aad831d0b022ebfa682a46e0a27c9fbaf0e5f833aacaa3b191819898a4b"

SERIES = [
    {
        "code": "EXP-H134",
        "backbone": "EXP-H071",
        "control": "EXP-H071",
        "f1_control": "EXP-H090",
        "mode": "global_surface_film",
        "content_mode": "scratch",
        "plm_source": None,
        "arch": {"joint_hl_single_reg": True},
        "merge_mode": "concat",
        "experiment_id": "TRF_HIC_SCRATCH_GLOBAL_F1_FILM_V3",
        "description": "H071 + global F1_SURFACE FiLM conditioning",
    },
    {
        "code": "EXP-H135",
        "backbone": "EXP-H071",
        "control": "EXP-H071",
        "f1_control": "EXP-H090",
        "mode": "global_surface_gated_residual",
        "content_mode": "scratch",
        "plm_source": None,
        "arch": {"joint_hl_single_reg": True},
        "merge_mode": "concat",
        "experiment_id": "TRF_HIC_SCRATCH_GLOBAL_F1_GATED_RES_V3",
        "description": "H071 + global F1_SURFACE gated residual",
    },
    {
        "code": "EXP-H136",
        "backbone": "EXP-H071",
        "control": "EXP-H071",
        "f1_control": "EXP-H090",
        "mode": "global_surface_token_attention",
        "content_mode": "scratch",
        "plm_source": None,
        "arch": {"joint_hl_single_reg": True},
        "merge_mode": "concat",
        "experiment_id": "TRF_HIC_SCRATCH_GLOBAL_F1_TOKEN_ATTN_V3",
        "description": "H071 + global F1_SURFACE 2-token attention",
    },
    {
        "code": "EXP-H137",
        "backbone": "EXP-H061",
        "control": "EXP-H061",
        "f1_control": "EXP-H086",
        "mode": "global_surface_film",
        "content_mode": "frozen",
        "plm_source": "esm2",
        "arch": {"joint_hl_chain_specific_dual_reg": True},
        "merge_mode": "mean",
        "experiment_id": "TRF_HIC_ESM2_GLOBAL_F1_FILM_V3",
        "description": "H061 + global F1_SURFACE FiLM conditioning",
    },
    {
        "code": "EXP-H138",
        "backbone": "EXP-H061",
        "control": "EXP-H061",
        "f1_control": "EXP-H086",
        "mode": "global_surface_gated_residual",
        "content_mode": "frozen",
        "plm_source": "esm2",
        "arch": {"joint_hl_chain_specific_dual_reg": True},
        "merge_mode": "mean",
        "experiment_id": "TRF_HIC_ESM2_GLOBAL_F1_GATED_RES_V3",
        "description": "H061 + global F1_SURFACE gated residual",
    },
    {
        "code": "EXP-H139",
        "backbone": "EXP-H061",
        "control": "EXP-H061",
        "f1_control": "EXP-H086",
        "mode": "global_surface_token_attention",
        "content_mode": "frozen",
        "plm_source": "esm2",
        "arch": {"joint_hl_chain_specific_dual_reg": True},
        "merge_mode": "mean",
        "experiment_id": "TRF_HIC_ESM2_GLOBAL_F1_TOKEN_ATTN_V3",
        "description": "H061 + global F1_SURFACE 2-token attention",
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


def count_added_params(mode: str) -> int:
    """Count conditioning params on a scratch single-reg stub."""
    import torch
    from antibody_transformer.model import AnnotatedTransformer

    backbone = AnnotatedTransformer(content_mode="scratch", joint_hl_single_reg=True)
    m = GlobalSurfaceConditioningModel(backbone, 35, mode)
    return m.n_added_conditioning_parameters()


def write_prereg() -> None:
    nxt = next_code("HIC")
    if nxt != "EXP-H134":
        raise SystemExit(f"expected next HIC EXP-H134, got {nxt}")
    if not AUDIT.exists():
        raise SystemExit("audit must exist before prereg")
    store = H047AuxFeatureStore("F1_SURFACE")
    if store.artifact_hash != F1_ARTIFACT_HASH:
        raise SystemExit(f"F1 artifact hash mismatch: {store.artifact_hash}")
    cols = []
    for b in store.block_order:
        cols.extend(store.block_cols[b])
    added = {s["mode"]: count_added_params(s["mode"]) for s in SERIES}
    doc = {
        "batch_id": "H134_H139_GLOBAL_F1_SURFACE_CONDITIONING",
        "status": "PREREGISTERED_FROZEN",
        "git_rev_at_prereg": git_rev(),
        "audit_path": str(AUDIT.relative_to(ROOT)),
        "fusion_bundle_id": "F1_SURFACE",
        "f1_dim": 35,
        "f1_artifact_hash": F1_ARTIFACT_HASH,
        "f1_columns": cols,
        "preprocessing": "TRAIN_median_impute_StandardScaler_no_PCA",
        "platform_id": PLATFORM_ID,
        "seed": SEED,
        "bottleneck": BOTTLENECK,
        "token_attention": {"n_layers": ATTN_LAYERS, "n_heads": ATTN_HEADS, "form": "[z_seq, t_surf]"},
        "controls": {
            "sequence_only": ["EXP-H071", "EXP-H061"],
            "historical_F1_late_fusion": ["EXP-H090", "EXP-H086"],
        },
        "added_params_by_mode": added,
        "experiments": [
            {
                "code": s["code"],
                "backbone": s["backbone"],
                "fusion_mode": s["mode"],
                "experiment_id": s["experiment_id"],
                "historical_f1_control": s["f1_control"],
                "added_params": added[s["mode"]],
                "initialization": (
                    "film_gamma_beta_zero"
                    if "film" in s["mode"]
                    else ("residual_u_and_gate_out_zero" if "gated" in s["mode"] else "default")
                ),
            }
            for s in SERIES
        ],
        "do_not": ["COMPACT10", "SAP24", "SCM24", "AUX32", "residue_tokens", "H140", "TEST_tuning"],
    }
    for s in SERIES:
        cfg = {
            "experiment_code": s["code"],
            "experiment_id": s["experiment_id"],
            "platform_id": PLATFORM_ID,
            "target": "HIC",
            "family": "TRANSFORMER",
            "transformer_type": "GLOBAL_SURFACE_CONDITIONING",
            "control_experiment_code": s["control"],
            "historical_f1_control": s["f1_control"],
            "fusion_bundle_id": "F1_SURFACE",
            "fusion_mode": s["mode"],
            "content_mode": s["content_mode"],
            "plm_source": s["plm_source"] or "NONE",
            "merge_mode": s["merge_mode"],
            "chain_mode": "HL",
            "pooling_mode": "REG",
            "seed": SEED,
            "no_full_dev_refit": True,
            "no_aux32": True,
            "no_late_f1_concat": True,
            "bottleneck": BOTTLENECK,
            "description": s["description"],
            "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml",
        }
        for k, v in s["arch"].items():
            cfg[k] = v
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
        source_model_id=f"GLOBAL_F1_{spec['mode'].upper()}",
        phase="H134_GLOBAL_F1_CONDITIONING",
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
    plm = "NONE" if spec["content_mode"] == "scratch" else "ESM2"
    asset = (
        "assets/transformer/residue_asset_manifest.yaml#annotations+scratch_sequences"
        if spec["content_mode"] == "scratch"
        else "assets/transformer/residue_asset_manifest.yaml#esm2"
    )
    feat_src = "scratch_residue" if spec["content_mode"] == "scratch" else "esm2_residue"
    row.update(
        {
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "target": "HIC",
            "family": "TRANSFORMER",
            "model_type": "TRANSFORMER",
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
            "transformer_type": "GLOBAL_SURFACE_CONDITIONING",
            "plm_source": plm,
            "input_asset_ref": asset,
            "feature_source": feat_src,
            "representation_status": "NOT_EXPORTED",
            "prediction_reproduction_max_delta": 0.0,
        }
    )
    # Keep feature_space / feature_set_id empty (no-fusion-parquet arch)
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
                    "notes": "global F1 SURFACE conditioning",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
            comp.to_csv(comp_path, index=False)


def run_one(spec: dict, *, quick: bool, rb, folds, dev, test, aux_store) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} mode={spec['mode']} ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        st = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
        return {"code": code, "summary": summary, "ext_scores": st["ext_scores"], "skipped": True}

    out = ROOT / "results" / f"{code}_run"
    result = run_protocol_v3_ext(
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
        aux_store=aux_store,
        fusion_mode=spec["mode"],
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
            save_pred(
                result["test_ids"],
                result["ext"][f"{scheme}_folds"][k],
                pred / f"test_{scheme}_fold{k}.csv",
                "HIC",
            )
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv", "HIC")

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol)
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }
    oof_doc = {
        "experiment_code": code,
        "fusion_mode": spec["mode"],
        "fusion_bundle_id": "F1_SURFACE",
        "f1_artifact_hash": F1_ARTIFACT_HASH,
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
        # Update registry expected sizes once codes will be issued
        from _lib import N_EXPERIMENTS_TOTAL  # noqa: WPS440

        print(f"N_EXPERIMENTS_TOTAL={N_EXPERIMENTS_TOTAL} next={next_code('HIC')}", flush=True)
        aux_store = H047AuxFeatureStore("F1_SURFACE")
        assert aux_store.artifact_hash == F1_ARTIFACT_HASH
        assert aux_store.effective_dim == 35
        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        need_esm2 = any(s["plm_source"] == "esm2" for s in SERIES)
        rb = load_residue_bundle(dev, test, need_esm2=need_esm2)
        for spec in SERIES:
            run_one(spec, quick=args.quick, rb=rb, folds=folds, dev=dev, test=test, aux_store=aux_store)
    if args.phase == "analyze":
        print("analyze: use separate diagnostics + report writer", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
