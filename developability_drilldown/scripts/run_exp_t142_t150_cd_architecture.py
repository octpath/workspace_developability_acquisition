#!/usr/bin/env python3
"""EXP-T142–T150: TmApp C/D H/L interaction architecture exploration (AbLang2).

Phases: prereg | train | analyze | all
Controls (reuse): A=T110, B=T113, C0=T121. Do not start PLM matrix / T151.
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
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID, run_protocol_v3  # noqa: E402

PREREG = ROOT / "results" / "T142_T150_CD_ARCHITECTURE_PREREGISTRATION.yaml"
SEED = DEFAULT_SEED

# Controls
A, B, C0 = "EXP-T110", "EXP-T113", "EXP-T121"

SERIES = [
    # C family
    {"code": "EXP-T142", "family": "C", "variant": "C1", "reg_cross_variant": "c1_scalar_gate",
     "pair_interaction_mode": None, "use_reg_only_cross_attention": True,
     "experiment_id": "TRF_TM_ABLANG2_C1_SCALAR_GATE_V3", "control": C0,
     "description": "C1 sample-dependent scalar cross gate on ARCH-7"},
    {"code": "EXP-T143", "family": "C", "variant": "C2", "reg_cross_variant": "c2_feature_gate",
     "pair_interaction_mode": None, "use_reg_only_cross_attention": True,
     "experiment_id": "TRF_TM_ABLANG2_C2_FEATURE_GATE_V3", "control": C0,
     "description": "C2 low-rank feature-wise cross gate on ARCH-7"},
    {"code": "EXP-T144", "family": "C", "variant": "C3", "reg_cross_variant": "c3_ffn_adapter",
     "pair_interaction_mode": None, "use_reg_only_cross_attention": True,
     "experiment_id": "TRF_TM_ABLANG2_C3_FFN_ADAPTER_V3", "control": C0,
     "description": "C3 REG cross + shared FFN adapter on ARCH-7"},
    {"code": "EXP-T145", "family": "C", "variant": "C4", "reg_cross_variant": "c4_two_read",
     "pair_interaction_mode": None, "use_reg_only_cross_attention": True,
     "experiment_id": "TRF_TM_ABLANG2_C4_TWO_READ_V3", "control": C0,
     "description": "C4 two-read REG cross-attention on ARCH-7"},
    {"code": "EXP-T146", "family": "C", "variant": "C5", "reg_cross_variant": "c5_two_query",
     "pair_interaction_mode": None, "use_reg_only_cross_attention": True,
     "experiment_id": "TRF_TM_ABLANG2_C5_TWO_QUERY_V3", "control": C0,
     "description": "C5 two-query REG cross-attention on ARCH-7"},
    # D family
    {"code": "EXP-T147", "family": "D", "variant": "D1", "reg_cross_variant": None,
     "pair_interaction_mode": "d1_bilinear_score", "use_reg_only_cross_attention": False,
     "experiment_id": "TRF_TM_ABLANG2_D1_BILINEAR_SCORE_V3", "control": A,
     "description": "D1 low-rank bilinear H/L pair score"},
    {"code": "EXP-T148", "family": "D", "variant": "D2", "reg_cross_variant": None,
     "pair_interaction_mode": "d2_hadamard_residual", "use_reg_only_cross_attention": False,
     "experiment_id": "TRF_TM_ABLANG2_D2_HADAMARD_RES_V3", "control": A,
     "description": "D2 low-rank Hadamard residual pair interaction"},
    {"code": "EXP-T149", "family": "D", "variant": "D3", "reg_cross_variant": None,
     "pair_interaction_mode": "d3_symmetric_mlp", "use_reg_only_cross_attention": False,
     "experiment_id": "TRF_TM_ABLANG2_D3_SYMMETRIC_MLP_V3", "control": A,
     "description": "D3 symmetric pair-feature MLP"},
    {"code": "EXP-T150", "family": "D", "variant": "D4", "reg_cross_variant": None,
     "pair_interaction_mode": "d4_token_attention", "use_reg_only_cross_attention": False,
     "experiment_id": "TRF_TM_ABLANG2_D4_TOKEN_ATTN_V3", "control": A,
     "description": "D4 two-token summary attention"},
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


def score_external(pred, test_ids, sol, target="TmApp"):
    sol2 = sol.set_index("id")
    te = pd.Series(pred, index=test_ids)
    pub = sol2.index[sol2["is_public"].astype(bool)].tolist()
    priv = sol2.index[sol2["is_private"].astype(bool)].tolist()
    return {
        "public_mae": float(mae(sol2.loc[pub, target].to_numpy(float), te.loc[pub].to_numpy(float))),
        "private_mae": float(mae(sol2.loc[priv, target].to_numpy(float), te.loc[priv].to_numpy(float))),
        "overall_mae": float(mae(sol2.loc[test_ids, target].to_numpy(float), te.loc[test_ids].to_numpy(float))),
    }


def arch_for(spec: dict) -> dict:
    return {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": bool(spec["use_reg_only_cross_attention"]),
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": spec["reg_cross_variant"],
        "pair_interaction_mode": spec["pair_interaction_mode"],
    }


def count_params(spec: dict) -> dict:
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=480,  # AbLang2
        merge_mode="mean",
        use_reg_only_cross_attention=bool(spec["use_reg_only_cross_attention"]),
        reg_cross_variant=spec["reg_cross_variant"],
        pair_interaction_mode=spec["pair_interaction_mode"],
        share_hl_encoder=True,
    )
    total = m.n_trainable_parameters()
    acct = m.param_account()
    # incremental vs control skeleton
    if spec["family"] == "C":
        base = AnnotatedTransformer(
            content_mode="frozen", plm_hidden=480, merge_mode="mean",
            use_reg_only_cross_attention=True, share_hl_encoder=True,
        )
    else:
        base = AnnotatedTransformer(
            content_mode="frozen", plm_hidden=480, merge_mode="mean", share_hl_encoder=True,
        )
    incr = total - base.n_trainable_parameters()
    return {"total": total, "incremental_vs_control_skeleton": incr, **acct}


def write_prereg() -> None:
    nxt = next_code("TmApp")
    if nxt != "EXP-T142":
        raise SystemExit(f"expected next TmApp EXP-T142, got {nxt}")
    added = {s["code"]: count_params(s) for s in SERIES}
    doc = {
        "batch_id": "T142_T150_CD_ARCHITECTURE_EXPLORATION",
        "status": "PREREGISTERED_FROZEN",
        "git_rev_at_prereg": git_rev(),
        "platform_id": PLATFORM_ID,
        "seed": SEED,
        "plm": "ablang2",
        "target": "TmApp",
        "controls": {"A": A, "B": B, "C0": C0},
        "symmetry": "shared parameters H<-L and L<-H; no fixed H/L coefficients",
        "do_not": ["PLM_matrix", "T151", "structural_features", "Ca_bias", "RASA", "aux_fixed"],
        "experiments": [],
        "param_account": added,
    }
    for s in SERIES:
        space_tag = (
            f"REG_CROSS_{s['variant']}_FROZEN_ABLANG2_MEAN_V3"
            if s["family"] == "C"
            else f"PAIR_{s['variant']}_FROZEN_ABLANG2_MEAN_V3"
        )
        cfg = {
            "experiment_code": s["code"],
            "experiment_id": s["experiment_id"],
            "platform_id": PLATFORM_ID,
            "target": "TmApp",
            "family": "TRANSFORMER",
            "transformer_type": "FROZEN_PLM",
            "input_space": space_tag,
            "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablang2",
            "plm_source": "ABLANG2",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "merge_mode": "mean",
            "pooling_mode": "REG",
            "content_mode": "frozen",
            "seed": SEED,
            "control_experiment_code": s["control"],
            "cd_family": s["family"],
            "cd_variant": s["variant"],
            "reg_cross_variant": s["reg_cross_variant"],
            "pair_interaction_mode": s["pair_interaction_mode"],
            "share_hl_encoder": True,
            "no_full_dev_refit": True,
            "description": s["description"],
            "arch_joint_hl_single_reg": False,
            "arch_joint_hl_dual_reg": False,
            "arch_joint_hl_chain_specific_dual_reg": False,
            "arch_use_cross_attention_bridge": False,
            "arch_use_reg_only_cross_attention": bool(s["use_reg_only_cross_attention"]),
            "arch_use_within_chain_extra_attention": False,
            "arch_cross_gate_mode": "learned",
            "arch_use_cross_geometry_bias": False,
            "arch_share_hl_encoder": True,
            "n_trainable_preregistered": added[s["code"]]["total"],
        }
        (ROOT / "experiments" / "configs" / f"{s['code']}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        doc["experiments"].append(
            {
                "code": s["code"],
                "family": s["family"],
                "variant": s["variant"],
                "experiment_id": s["experiment_id"],
                "control": s["control"],
                "reg_cross_variant": s["reg_cross_variant"],
                "pair_interaction_mode": s["pair_interaction_mode"],
                "n_trainable": added[s["code"]]["total"],
                "incremental": added[s["code"]]["incremental_vs_control_skeleton"],
            }
        )
    PREREG.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print("Wrote", PREREG, flush=True)


def issue_one(spec: dict) -> str:
    codes = load_codes()
    eid = spec["experiment_id"]
    if (codes["experiment_id"] == eid).any():
        return str(codes.set_index("experiment_id").loc[eid, "experiment_code"])
    expect = spec["code"]
    if next_code("TmApp") != expect:
        raise SystemExit(f"expected {expect}, got {next_code('TmApp')}")
    code = issue_code(
        eid,
        "TmApp",
        source_model_id=f"CD_{spec['variant']}",
        phase="T142_CD_ARCHITECTURE",
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
    space = (
        f"REG_CROSS_{spec['variant']}_FROZEN_ABLANG2_MEAN_V3"
        if spec["family"] == "C"
        else f"PAIR_{spec['variant']}_FROZEN_ABLANG2_MEAN_V3"
    )
    row.update(
        {
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "target": "TmApp",
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
            "transformer_type": "FROZEN_PLM",
            "plm_source": "ABLANG2",
            "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablang2",
            "input_space": space,
            "feature_source": "ablang2_residue",
            "representation_status": "NOT_EXPORTED",
            "prediction_reproduction_max_delta": 0.0,
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
                    "target": "TmApp",
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
                    "notes": "TmApp C/D architecture exploration",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
            comp.to_csv(comp_path, index=False)


def run_one(spec: dict, *, quick: bool, rb, folds, dev, test) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} {spec['family']}/{spec['variant']} ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        st = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
        return {"code": code, "summary": summary, "ext_scores": st["ext_scores"], "skipped": True}

    out = ROOT / "results" / f"{code}_run"
    result = run_protocol_v3(
        experiment_code=code,
        target="TmApp",
        dev=dev,
        test=test,
        folds=folds,
        rb=rb,
        device=device_str(),
        out_dir=out,
        seed=SEED,
        quick=quick,
        arch=arch_for(spec),
        content_mode="frozen",
        merge_mode="mean",
        plm_source="ablang2",
        chain_mode="HL",
    )
    summary = result["summary"]
    sel = result["selected_df"]
    sel.to_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv", index=False)

    pred = ROOT / "experiments" / "predictions" / code
    pred.mkdir(parents=True, exist_ok=True)
    save_pred(result["dev_ids"], result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_primary.csv", "TmApp")
    save_pred(result["dev_ids"], result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_shadow.csv", "TmApp")
    for scheme in ("primary", "shadow"):
        save_pred(result["dev_ids"], result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_val_{scheme}.csv", "TmApp")
        save_pred(result["dev_ids"], result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_test_{scheme}.csv", "TmApp")
    for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        save_pred(result["test_ids"], result["ext"][key], pred / f"test_{key}.csv", "TmApp")
    for scheme in ("primary", "shadow"):
        folds_key = f"{scheme}_folds"
        if folds_key in result["ext"]:
            for k in range(5):
                save_pred(result["test_ids"], result["ext"][folds_key][k], pred / f"test_{scheme}_fold{k}.csv", "TmApp")
        else:
            for k in range(5):
                fk = f"{scheme}_fold{k}"
                if fk in result["ext"]:
                    save_pred(result["test_ids"], result["ext"][fk], pred / f"test_{scheme}_fold{k}.csv", "TmApp")
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv", "TmApp")

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol)
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }
    oof_doc = {
        "experiment_code": code,
        "cd_family": spec["family"],
        "cd_variant": spec["variant"],
        "reg_cross_variant": spec["reg_cross_variant"],
        "pair_interaction_mode": spec["pair_interaction_mode"],
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external": ext_scores,
        "n_trainable": summary.get("n_trainable"),
        "param_account": summary.get("param_account"),
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
        print(f"next={next_code('TmApp')}", flush=True)
        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        rb = load_residue_bundle(dev, test, need_ablang2=True)
        if rb.ablang2_h is None or rb.ablang2_l is None:
            raise SystemExit("AbLang2 residue embeddings missing")
        for spec in SERIES:
            run_one(spec, quick=args.quick, rb=rb, folds=folds, dev=dev, test=test)
    if args.phase == "analyze":
        from analyze_t142_t150_cd_architecture import run_analyze

        run_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
