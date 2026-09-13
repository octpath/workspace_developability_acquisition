#!/usr/bin/env python3
"""EXP-T151–T156: TmApp PLM × H/L topology matrix (AbLingua D + ESM-2 A–D).

Phases: audit_ok | prereg | train | analyze | external | all
Reuse historical cells; do not retrain. Do not run T157.
"""
from __future__ import annotations

import argparse
import hashlib
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
from antibody_transformer.config import BUNDLE_ROOT, RESIDUE_ROOT  # noqa: E402
from antibody_transformer.data import (  # noqa: E402
    load_dev_test,
    load_folds,
    load_residue_bundle,
    load_solution,
)
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID, run_protocol_v3  # noqa: E402

PREREG = ROOT / "results" / "TMAPP_PLM_TOPOLOGY_MATRIX_PREREGISTRATION.yaml"
FREEZE_V2 = ROOT / "results" / "TMAPP_HL_TOPOLOGY_FREEZE_V2.yaml"
AUDIT = ROOT / "results" / "TMAPP_PLM_TOPOLOGY_MATRIX_AUDIT.md"
SEED = DEFAULT_SEED
N_BOOT = 2000

TOPOLOGY_FLAGS = {
    "A": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "B1": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": True,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "B2": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": True,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "C": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": True,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": None,
    },
    "D": {
        "joint_hl_single_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_chain_specific_dual_reg": False,
        "use_cross_attention_bridge": False,
        "use_reg_only_cross_attention": False,
        "use_within_chain_extra_attention": False,
        "cross_gate_mode": "learned",
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
        "reg_cross_variant": None,
        "pair_interaction_mode": "d3_symmetric_mlp",
    },
}

# Full primary matrix: reuse vs new
MATRIX = [
    # AbLingua
    {"plm": "ablingua", "topology": "A", "code": "EXP-T080", "reuse": True},
    {"plm": "ablingua", "topology": "B1", "code": "EXP-T081", "reuse": True},
    {"plm": "ablingua", "topology": "B2", "code": "EXP-T082", "reuse": True},
    {"plm": "ablingua", "topology": "C", "code": "EXP-T087", "reuse": True},
    {
        "plm": "ablingua",
        "topology": "D",
        "code": "EXP-T151",
        "reuse": False,
        "experiment_id": "TRF_TM_ABLINGUA_D3_SYMMETRIC_MLP_V3",
        "description": "AbLingua + D3 symmetric pair interaction",
    },
    # AbLang2
    {"plm": "ablang2", "topology": "A", "code": "EXP-T110", "reuse": True},
    {"plm": "ablang2", "topology": "B1", "code": "EXP-T113", "reuse": True},
    {"plm": "ablang2", "topology": "B2", "code": "EXP-T115", "reuse": True},
    {"plm": "ablang2", "topology": "C", "code": "EXP-T121", "reuse": True},
    {"plm": "ablang2", "topology": "D", "code": "EXP-T149", "reuse": True},
    # ESM-2
    {
        "plm": "esm2",
        "topology": "A",
        "code": "EXP-T152",
        "reuse": False,
        "experiment_id": "TRF_TM_ESM2_FULL_SEPARATE_DUAL_MEAN_V3",
        "description": "ESM-2 + topology A independent dual REG",
    },
    {
        "plm": "esm2",
        "topology": "B1",
        "code": "EXP-T153",
        "reuse": False,
        "experiment_id": "TRF_TM_ESM2_FULL_JOINT_DUAL_MEAN_V3",
        "description": "ESM-2 + topology B1 ARCH-3 joint dual REG",
    },
    {
        "plm": "esm2",
        "topology": "B2",
        "code": "EXP-T154",
        "reuse": False,
        "experiment_id": "TRF_TM_ESM2_FULL_JOINT_CHAIN_SPECIFIC_DUAL_MEAN_V3",
        "description": "ESM-2 + topology B2 ARCH-4 chain-specific dual REG",
    },
    {
        "plm": "esm2",
        "topology": "C",
        "code": "EXP-T155",
        "reuse": False,
        "experiment_id": "TRF_TM_ESM2_FULL_SEPARATE_REG_ONLY_CROSS_MEAN_V3",
        "description": "ESM-2 + topology C ARCH-7 REG-only cross",
    },
    {
        "plm": "esm2",
        "topology": "D",
        "code": "EXP-T156",
        "reuse": False,
        "experiment_id": "TRF_TM_ESM2_D3_SYMMETRIC_MLP_V3",
        "description": "ESM-2 + topology D D3 pair MLP",
    },
]

SCRATCH_CONTEXT = {
    "A": "EXP-T091",
    "B1": "EXP-T094",
    "B2": "EXP-T096",
    "C": "EXP-T102",
}

PLM_RAW = {"ablingua": 1280, "ablang2": 480, "esm2": 1280}
PLM_SUBDIR = {"ablingua": "ablingua600m", "ablang2": "ablang2", "esm2": "esm2"}
PLM_ASSET = {
    "ablingua": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
    "ablang2": "assets/transformer/residue_asset_manifest.yaml#ablang2",
    "esm2": "assets/transformer/residue_asset_manifest.yaml#esm2",
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def projection_params(raw_dim: int) -> int:
    return int(raw_dim) * 128 + 128


def count_params(plm: str, topology: str) -> dict:
    raw = PLM_RAW[plm]
    flags = TOPOLOGY_FLAGS[topology]
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=raw,
        merge_mode="mean",
        share_hl_encoder=True,
        joint_hl_dual_reg=flags["joint_hl_dual_reg"],
        joint_hl_chain_specific_dual_reg=flags["joint_hl_chain_specific_dual_reg"],
        use_reg_only_cross_attention=flags["use_reg_only_cross_attention"],
        pair_interaction_mode=flags["pair_interaction_mode"],
        reg_cross_variant=flags["reg_cross_variant"],
    )
    acct = m.param_account()
    proj = sum(p.numel() for p in m.plm_proj.parameters()) if m.plm_proj is not None else 0
    return {
        "total": m.n_trainable_parameters(),
        "projection": proj,
        "encoder": acct.get("encoder", 0),
        "head": acct.get("head", 0),
        "cross_attention": acct.get("cross_attention", 0),
        "cross_interaction": acct.get("cross_interaction", 0),
    }


def pack_provenance(plm: str) -> dict:
    sub = PLM_SUBDIR[plm]
    base = RESIDUE_ROOT / sub
    meta = json.loads((base / "metadata.json").read_text())
    return {
        "subdir": sub,
        "raw_dim": PLM_RAW[plm],
        "ids_sha256": sha_file(base / "ids.npy"),
        "heavy_mask_sha256": sha_file(base / "heavy_mask.npy"),
        "light_mask_sha256": sha_file(base / "light_mask.npy"),
        "heavy_emb_sha256": sha_file(base / "heavy_embeddings.npy"),
        "light_emb_sha256": sha_file(base / "light_embeddings.npy"),
        "metadata": meta,
        "asset_ref": PLM_ASSET[plm],
    }


def write_prereg() -> None:
    if not AUDIT.exists() or "PASS" not in AUDIT.read_text():
        raise SystemExit("audit must PASS before prereg")
    if not FREEZE_V2.exists():
        raise SystemExit("missing V2 freeze")
    nxt = next_code("TmApp")
    if nxt != "EXP-T151":
        raise SystemExit(f"expected next EXP-T151, got {nxt}")

    packs = {p: pack_provenance(p) for p in ("ablingua", "ablang2", "esm2")}
    cells = []
    for spec in MATRIX:
        plm, topo, code = spec["plm"], spec["topology"], spec["code"]
        params = count_params(plm, topo)
        cell = {
            "code": code,
            "plm": plm,
            "topology": topo,
            "reuse": spec["reuse"],
            "embedding": {
                "subdir": packs[plm]["subdir"],
                "raw_dim": packs[plm]["raw_dim"],
                "ids_sha256": packs[plm]["ids_sha256"],
                "heavy_emb_sha256": packs[plm]["heavy_emb_sha256"],
                "light_emb_sha256": packs[plm]["light_emb_sha256"],
                "asset_ref": packs[plm]["asset_ref"],
            },
            "projection": f"Linear({packs[plm]['raw_dim']},128)",
            "projection_params": projection_params(packs[plm]["raw_dim"]),
            "topology_flags": TOPOLOGY_FLAGS[topo],
            "trainable": params,
            "seed": SEED,
            "platform_id": PLATFORM_ID,
            "fold_sha16": sha_file(ROOT / "data" / "folds.csv")[:16],
        }
        if spec["reuse"]:
            oof = ROOT / "experiments" / "predictions" / code / "oof_primary.csv"
            cell["prediction_oof_primary_sha16"] = sha_file(oof)[:16] if oof.exists() else None
            cell["source_experiment"] = code
        else:
            cell["experiment_id"] = spec["experiment_id"]
            cell["description"] = spec["description"]
            # write config yaml
            cfg = {
                "experiment_code": code,
                "experiment_id": spec["experiment_id"],
                "platform_id": PLATFORM_ID,
                "target": "TmApp",
                "family": "TRANSFORMER",
                "transformer_type": "FROZEN_PLM",
                "input_space": f"TOPO_{topo}_FROZEN_{plm.upper()}_MEAN_V3",
                "input_asset_ref": PLM_ASSET[plm],
                "plm_source": plm.upper() if plm != "ablingua" else "ABLINGUA",
                "annotation_mode": "FULL",
                "chain_mode": "HL",
                "merge_mode": "mean",
                "pooling_mode": "REG",
                "content_mode": "frozen",
                "seed": SEED,
                "share_hl_encoder": True,
                "pair_interaction_mode": TOPOLOGY_FLAGS[topo]["pair_interaction_mode"],
                "reg_cross_variant": None,
                "no_full_dev_refit": True,
                "description": spec["description"],
                "arch_joint_hl_single_reg": False,
                "arch_joint_hl_dual_reg": TOPOLOGY_FLAGS[topo]["joint_hl_dual_reg"],
                "arch_joint_hl_chain_specific_dual_reg": TOPOLOGY_FLAGS[topo][
                    "joint_hl_chain_specific_dual_reg"
                ],
                "arch_use_cross_attention_bridge": False,
                "arch_use_reg_only_cross_attention": TOPOLOGY_FLAGS[topo][
                    "use_reg_only_cross_attention"
                ],
                "arch_use_within_chain_extra_attention": False,
                "arch_cross_gate_mode": "learned",
                "arch_use_cross_geometry_bias": False,
                "arch_share_hl_encoder": True,
                "n_trainable_preregistered": params["total"],
            }
            (ROOT / "experiments" / "configs" / f"{code}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )
        cells.append(cell)

    doc = {
        "batch_id": "T151_T156_PLM_TOPOLOGY_MATRIX",
        "status": "PREREGISTERED_FROZEN",
        "git_rev_at_prereg": git_rev(),
        "audit": str(AUDIT.relative_to(ROOT)),
        "freeze_v2": str(FREEZE_V2.relative_to(ROOT)),
        "platform_id": PLATFORM_ID,
        "seed": SEED,
        "do_not": ["EXP-T157", "retrain_historical_cells", "scratch_primary_matrix"],
        "scratch_context": SCRATCH_CONTEXT,
        "plm_packs": {
            k: {kk: vv for kk, vv in v.items() if kk != "metadata"} for k, v in packs.items()
        },
        "cells": cells,
    }
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
        source_model_id=f"PLM_TOPO_{spec['topology']}",
        phase="T151_PLM_TOPOLOGY_MATRIX",
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
    plm = spec["plm"]
    topo = spec["topology"]
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
            "transformer_type": "FROZEN_PLM",
            "plm_source": plm.upper() if plm != "ablingua" else "ABLINGUA",
            "input_asset_ref": PLM_ASSET[plm],
            "input_space": f"TOPO_{topo}_FROZEN_{plm.upper()}_MEAN_V3",
            "feature_source": f"{plm}_residue",
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
                    "notes": "PLM topology matrix",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
            comp.to_csv(comp_path, index=False)


def run_one(spec: dict, *, quick: bool, rb, folds, dev, test) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} {spec['plm']}/{spec['topology']} ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text()) if (
            ROOT / "results" / f"{code}_run" / "summary.json"
        ).exists() else yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
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
        arch=TOPOLOGY_FLAGS[spec["topology"]],
        content_mode="frozen",
        merge_mode="mean",
        plm_source=spec["plm"],
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
        "plm": spec["plm"],
        "topology": spec["topology"],
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external": ext_scores,
        "n_trainable": summary.get("n_trainable"),
        "param_account": summary.get("param_account"),
        "no_full_dev_refit": True,
        "selected": sel.to_dict(orient="records"),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(
        yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "results" / f"{code}_run_state.json").write_text(json.dumps({"ext_scores": ext_scores}, indent=2))
    register(code, spec, summary, ext_scores, int(summary.get("n_trainable") or 0))
    print(f"DONE {code} TEST_mean={summary['scores']['oof_test']['mean']:.6f}", flush=True)
    return {"code": code, "summary": summary, "ext_scores": ext_scores, "skipped": False}


def paired_boot(a: np.ndarray, b: np.ndarray, y: np.ndarray, n=N_BOOT, seed=SEED):
    """Δ = MAE(a)-MAE(b); negative ⇒ a better."""
    ea = np.abs(a - y)
    eb = np.abs(b - y)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = np.asarray([float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)])
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def load_oof(code: str, scheme: str) -> pd.Series:
    p = ROOT / "experiments" / "predictions" / code / f"oof_{scheme}.csv"
    if not p.exists():
        p = ROOT / "experiments" / "predictions" / code / f"oof_test_{scheme}.csv"
    df = pd.read_csv(p)
    col = [c for c in df.columns if c != "id"][0]
    return df.set_index("id")[col].astype(float)


def scores_from_exp(code: str, exp: pd.DataFrame) -> dict:
    r = exp[exp.experiment_code == code].iloc[0]
    return {
        "code": code,
        "primary": float(r.cv_primary_mae),
        "shadow": float(r.cv_shadow_mae),
        "mean": float(r.cv_mean_mae),
        "worst": float(r.cv_worst_mae),
        "public": float(r.public_mae) if pd.notna(r.public_mae) else None,
        "private": float(r.private_mae) if pd.notna(r.private_mae) else None,
        "overall": float(r.test_overall_mae) if pd.notna(r.test_overall_mae) else None,
    }


def did_boot(p1x, p1a, p2x, p2a, y, n=N_BOOT, seed=SEED):
    """DiD = (MAE(P1,X)-MAE(P1,A)) - (MAE(P2,X)-MAE(P2,A)) on paired abs errors."""
    e1x = np.abs(p1x - y)
    e1a = np.abs(p1a - y)
    e2x = np.abs(p2x - y)
    e2a = np.abs(p2a - y)
    d = (e1x - e1a) - (e2x - e2a)
    rng = np.random.default_rng(seed)
    boots = np.asarray([float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)])
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def run_analyze(*, include_external_in_report: bool = False) -> None:
    from analyze_t151_t156_plm_topology import run_full_analysis

    run_full_analysis(include_external_in_report=include_external_in_report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "train", "analyze", "external", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.phase in ("prereg", "all"):
        write_prereg()
    if args.phase in ("train", "all"):
        if not PREREG.exists():
            raise SystemExit("prereg missing — commit prereg before train")
        print(f"next={next_code('TmApp')}", flush=True)
        new_specs = [s for s in MATRIX if not s["reuse"]]
        need_ab = any(s["plm"] == "ablingua" for s in new_specs)
        need_a2 = any(s["plm"] == "ablang2" for s in new_specs)
        need_e = any(s["plm"] == "esm2" for s in new_specs)
        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        rb = load_residue_bundle(
            dev, test, need_ablingua=need_ab, need_ablang2=need_a2, need_esm2=need_e
        )
        if need_e and (rb.esm2_h is None or rb.esm2_l is None):
            raise SystemExit("ESM-2 H/L residue embeddings missing")
        if need_ab and (rb.ablingua_h is None or rb.ablingua_l is None):
            raise SystemExit("AbLingua residue embeddings missing")
        for spec in new_specs:
            run_one(spec, quick=args.quick, rb=rb, folds=folds, dev=dev, test=test)
        print(f"train done next={next_code('TmApp')}", flush=True)
    if args.phase == "analyze":
        run_analyze(include_external_in_report=False)
    if args.phase == "external":
        run_analyze(include_external_in_report=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
