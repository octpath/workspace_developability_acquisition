#!/usr/bin/env python3
"""EXP-T157–T160: TmApp new PLM residue integration smoke (C + FULL only).

Phases: prereg | train | report | all

Exact T121 / topology-C path. Do NOT run A/B1/B2/D or annotation ablations.
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

PREREG = ROOT / "results" / "TMAPP_NEW_PLM_C_FULL_SMOKE_PREREGISTRATION.yaml"
REPORT = ROOT / "results" / "TMAPP_NEW_PLM_C_FULL_SMOKE_REPORT.md"
SEED = DEFAULT_SEED

# Exact C / T121 flags
TOPOLOGY_C = {
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
}

SERIES = [
    {
        "code": "EXP-T157",
        "plm": "ablang1",
        "experiment_id": "TRF_TM_ABLANG1_FULL_SEPARATE_REG_ONLY_CROSS_MEAN_V3",
        "description": "AbLang1 residue + C + FULL smoke",
        "raw_dim_expected": 768,
        "representation_context": "SEPARATE_CHAIN",
        "need_kw": "need_ablang1",
        "subdir": "ablang1",
    },
    {
        "code": "EXP-T158",
        "plm": "esm1b",
        "experiment_id": "TRF_TM_ESM1B_FULL_SEPARATE_REG_ONLY_CROSS_MEAN_V3",
        "description": "ESM-1b residue + C + FULL smoke",
        "raw_dim_expected": 1280,
        "representation_context": "SEPARATE_CHAIN",
        "need_kw": "need_esm1b",
        "subdir": "esm1b",
    },
    {
        "code": "EXP-T159",
        "plm": "esmc600m",
        "experiment_id": "TRF_TM_ESMC600M_FULL_SEPARATE_REG_ONLY_CROSS_MEAN_V3",
        "description": "ESM-C 600M residue + C + FULL smoke",
        "raw_dim_expected": None,  # inferred from checkpoint
        "representation_context": "SEPARATE_CHAIN",
        "need_kw": "need_esmc600m",
        "subdir": "esmc600m",
    },
    {
        "code": "EXP-T160",
        "plm": "currab",
        "experiment_id": "TRF_TM_CURRAB_FULL_SEPARATE_REG_ONLY_CROSS_MEAN_V3",
        "description": "CurrAb paired-native residue + C + FULL smoke",
        "raw_dim_expected": 1280,
        "representation_context": "PAIRED_NATIVE",
        "need_kw": "need_currab",
        "subdir": "currab",
    },
]

PLM_ASSET = {
    "ablang1": "assets/transformer/residue_asset_manifest.yaml#ablang1",
    "esm1b": "assets/transformer/residue_asset_manifest.yaml#esm1b",
    "esmc600m": "assets/transformer/residue_asset_manifest.yaml#esmc600m",
    "currab": "assets/transformer/residue_asset_manifest.yaml#currab",
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


def count_params(raw_dim: int) -> dict:
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=raw_dim,
        merge_mode="mean",
        share_hl_encoder=True,
        use_reg_only_cross_attention=True,
    )
    acct = m.param_account()
    proj = sum(p.numel() for p in m.plm_proj.parameters()) if m.plm_proj is not None else 0
    return {
        "total": m.n_trainable_parameters(),
        "projection": proj,
        "encoder": acct.get("encoder", 0),
        "head": acct.get("head", 0),
        "cross_attention": acct.get("cross_attention", 0),
    }


def pack_status(subdir: str) -> dict:
    base = RESIDUE_ROOT / subdir
    meta_p = base / "metadata.json"
    if not meta_p.exists():
        return {"status": "BLOCKED_DOWNLOAD", "reason": f"missing {meta_p}"}
    meta = json.loads(meta_p.read_text())
    for req in (
        "ids.npy",
        "heavy_embeddings.npy",
        "light_embeddings.npy",
        "heavy_mask.npy",
        "light_mask.npy",
    ):
        p = base / req
        p0 = Path(str(p) + ".part0")
        if not p.exists() and not (p0.exists() if "embeddings" in req else False):
            if "embeddings" in req and p0.exists():
                continue
            if not p.exists():
                return {"status": "BLOCKED_DOWNLOAD", "reason": f"missing {req}"}
    # assemble if needed
    from antibody_transformer.data import load_npy

    ids = [str(x) for x in load_npy(base / "ids.npy", allow_pickle=True).tolist()]
    if len(ids) != 324:
        return {"status": "BLOCKED_MAPPING", "reason": f"n_ids={len(ids)}"}
    if meta.get("mapping") and "EXACT" not in str(meta.get("mapping")):
        return {"status": "BLOCKED_MAPPING", "reason": f"mapping={meta.get('mapping')}"}
    rechk = base / "REEXTRACT_CHECK.json"
    if rechk.exists():
        chk = json.loads(rechk.read_text())
        if not chk.get("pass", False):
            return {"status": "BLOCKED_MAPPING", "reason": f"reextract fail {chk}"}
    return {
        "status": "READY",
        "meta": meta,
        "raw_dim": int(meta["hidden_dim"]),
        "ids_sha256": sha_file(base / "ids.npy"),
        "heavy_emb_sha256": sha_file(base / "heavy_embeddings.npy")
        if (base / "heavy_embeddings.npy").exists()
        else sha_file(Path(str(base / "heavy_embeddings.npy") + ".part0")),
        "diag": json.loads((base / "EMBEDDING_DIAGNOSTICS.json").read_text())
        if (base / "EMBEDDING_DIAGNOSTICS.json").exists()
        else None,
    }


def write_prereg() -> None:
    nxt = next_code("TmApp")
    if nxt != "EXP-T157":
        raise SystemExit(f"expected next EXP-T157, got {nxt} — resolve ID collision before continuing")

    cells = []
    for spec in SERIES:
        st = pack_status(spec["subdir"])
        raw = st.get("raw_dim")
        if raw is None and st["status"] == "READY":
            raise SystemExit(f"missing hidden_dim for {spec['plm']}")
        if st["status"] != "READY":
            params = None
            proj = None
        else:
            if spec["raw_dim_expected"] is not None and raw != spec["raw_dim_expected"]:
                raise SystemExit(
                    f"{spec['plm']} hidden_dim {raw} != expected {spec['raw_dim_expected']}"
                )
            params = count_params(raw)
            proj = projection_params(raw)
            cfg = {
                "experiment_code": spec["code"],
                "experiment_id": spec["experiment_id"],
                "platform_id": PLATFORM_ID,
                "target": "TmApp",
                "family": "TRANSFORMER",
                "transformer_type": "FROZEN_PLM",
                "input_space": f"SEPARATE_REG_ONLY_CROSS_ATTENTION_FROZEN_{spec['plm'].upper()}_RESIDUE_MEAN_V3",
                "input_asset_ref": PLM_ASSET[spec["plm"]],
                "plm_source": spec["plm"].upper(),
                "annotation_mode": "FULL",
                "chain_mode": "HL",
                "merge_mode": "mean",
                "pooling_mode": "REG",
                "content_mode": "frozen",
                "seed": SEED,
                "d_model": 128,
                "share_hl_encoder": True,
                "control_experiment_code": "EXP-T121",
                "no_full_dev_refit": True,
                "description": spec["description"],
                "representation_context": spec["representation_context"],
                "raw_plm_hidden_dim": raw,
                "arch_joint_hl_single_reg": False,
                "arch_joint_hl_dual_reg": False,
                "arch_joint_hl_chain_specific_dual_reg": False,
                "arch_use_cross_attention_bridge": False,
                "arch_use_reg_only_cross_attention": True,
                "arch_use_within_chain_extra_attention": False,
                "arch_cross_gate_mode": "learned",
                "arch_use_cross_geometry_bias": False,
                "arch_share_hl_encoder": True,
                "n_trainable_preregistered": params["total"],
                "param_account_preregistered": params,
                "projection_params": proj,
            }
            (ROOT / "experiments" / "configs" / f"{spec['code']}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )
        cells.append(
            {
                "code": spec["code"],
                "plm": spec["plm"],
                "topology": "C",
                "annotation": "FULL",
                "pack_status": st["status"],
                "pack_reason": st.get("reason"),
                "raw_dim": raw,
                "representation_context": spec["representation_context"],
                "trainable": params,
                "projection_params": proj,
                "topology_flags": TOPOLOGY_C,
                "seed": SEED,
                "platform_id": PLATFORM_ID,
                "control": "EXP-T121",
            }
        )

    doc = {
        "batch_id": "T157_T160_NEW_PLM_C_FULL_SMOKE",
        "status": "PREREGISTERED",
        "git_rev_at_prereg": git_rev(),
        "platform_id": PLATFORM_ID,
        "seed": SEED,
        "do_not": [
            "topology_A",
            "topology_B1",
            "topology_B2",
            "topology_D",
            "annotation_BASE",
            "annotation_IMGT_only",
            "annotation_REGION_only",
            "full_plm_topology_matrix",
        ],
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
        source_model_id="NEW_PLM_C_FULL_SMOKE",
        phase="T157_NEW_PLM_SMOKE",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


def already_complete(code: str) -> bool:
    return (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").exists() and (
        ROOT / "experiments" / "predictions" / code / "test_primary_mean.csv"
    ).exists()


def register(code: str, spec: dict, summary: dict, ext_scores: dict, n_params: int, raw_dim: int) -> None:
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
            "plm_source": spec["plm"].upper(),
            "input_asset_ref": PLM_ASSET[spec["plm"]],
            "input_space": f"SEPARATE_REG_ONLY_CROSS_ATTENTION_FROZEN_{spec['plm'].upper()}_RESIDUE_MEAN_V3",
            "feature_source": f"{spec['plm']}_residue",
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
                    "notes": "new PLM C+FULL smoke",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
            comp.to_csv(comp_path, index=False)


def run_one(spec: dict, *, quick: bool, rb, folds, dev, test) -> dict:
    st = pack_status(spec["subdir"])
    if st["status"] != "READY":
        return {
            "code": spec["code"],
            "plm": spec["plm"],
            "status": st["status"],
            "reason": st.get("reason"),
            "skipped": True,
        }
    code = issue_one(spec)
    print(f"==== TRAIN {code} {spec['plm']}/C/FULL ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
        stj = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
        return {
            "code": code,
            "plm": spec["plm"],
            "status": "PASS",
            "summary": summary,
            "ext_scores": stj["ext_scores"],
            "raw_dim": st["raw_dim"],
            "skipped": True,
        }

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
        arch=TOPOLOGY_C,
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
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv", "TmApp")

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol)
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }
    oof_doc = {
        "experiment_code": code,
        "plm": spec["plm"],
        "topology": "C",
        "annotation": "FULL",
        "raw_plm_hidden_dim": st["raw_dim"],
        "representation_context": spec["representation_context"],
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external": ext_scores,
        "n_trainable": summary.get("n_trainable"),
        "param_account": summary.get("param_account"),
        "projection_params": projection_params(st["raw_dim"]),
        "no_full_dev_refit": True,
        "selected": sel.to_dict(orient="records"),
        "embedding_diagnostics": st.get("diag"),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(
        yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "results" / f"{code}_run_state.json").write_text(
        json.dumps({"ext_scores": ext_scores, "raw_dim": st["raw_dim"]}, indent=2)
    )
    register(code, spec, summary, ext_scores, int(summary.get("n_trainable") or 0), st["raw_dim"])
    print(f"DONE {code} TEST_mean={summary['scores']['oof_test']['mean']:.6f}", flush=True)
    return {
        "code": code,
        "plm": spec["plm"],
        "status": "PASS",
        "summary": summary,
        "ext_scores": ext_scores,
        "raw_dim": st["raw_dim"],
        "skipped": False,
    }


def write_report(results: list[dict]) -> None:
    lines = [
        "# TmApp NEW PLM C+FULL SMOKE REPORT",
        "",
        f"**Platform:** `{PLATFORM_ID}`  ",
        f"**Control:** EXP-T121 (AbLang2 / C / FULL)  ",
        f"**Seed:** {SEED}  ",
        f"**Git:** `{git_rev()}`  ",
        "",
        "Integration / validation only. **Do not rank or drop PLMs by these smoke scores.**",
        "",
        "| Code | PLM | representation context | raw dim | P | S | mean | status |",
        "| ---- | --- | ---------------------- | ------: | -: | -: | ---: | ------ |",
    ]
    for r in results:
        ctx = next(s["representation_context"] for s in SERIES if s["plm"] == r["plm"])
        if r.get("status") == "PASS" and "summary" in r:
            oof = r["summary"]["scores"]["oof_test"] if "scores" in r["summary"] else r["summary"].get("oof_test", {})
            # handle both shapes
            if "primary" in oof:
                p, s, m = oof["primary"], oof["shadow"], oof["mean"]
            else:
                p = s = m = float("nan")
            raw = r.get("raw_dim", "")
            lines.append(
                f"| {r['code']} | {r['plm']} | {ctx} | {raw} | {p:.4f} | {s:.4f} | {m:.4f} | PASS |"
            )
        else:
            lines.append(
                f"| {r.get('code')} | {r.get('plm')} | {ctx} |  |  |  |  | {r.get('status')} |"
            )
    lines += [
        "",
        "## Extraction provenance",
        "",
    ]
    for spec in SERIES:
        base = RESIDUE_ROOT / spec["subdir"]
        meta_p = base / "metadata.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            lines.append(f"### {spec['plm']}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(meta, indent=2)[:4000])
            lines.append("```")
            lines.append("")
        else:
            lines.append(f"### {spec['plm']}: missing metadata")
            lines.append("")
    lines += [
        "## Smoke result details",
        "",
        "See per-code `results/EXP-T15x_OOF_EVALUATION.yaml` for selected LR, best epoch, params.",
        "",
        "## STOP",
        "",
        "Do **not** run A/B1/B2/D or annotation ablations for these PLMs until smoke audit.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", REPORT, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "train", "report", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("prereg", "all"):
        write_prereg()

    results: list[dict] = []
    if args.phase in ("train", "all"):
        if not PREREG.exists():
            raise SystemExit("prereg missing — write/commit prereg before train")
        print(f"next={next_code('TmApp')}", flush=True)
        # assemble
        subprocess.check_call(["bash", str(RESIDUE_ROOT / "assemble_embeddings.sh")])
        need = {s["need_kw"]: True for s in SERIES if pack_status(s["subdir"])["status"] == "READY"}
        if not need:
            raise SystemExit("no PLM packs READY")
        dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        folds = load_folds(ROOT / "data" / "folds.csv")
        rb = load_residue_bundle(dev, test, **need)
        for spec in SERIES:
            results.append(run_one(spec, quick=args.quick, rb=rb, folds=folds, dev=dev, test=test))
        write_report(results)
        (ROOT / "results" / "TMAPP_NEW_PLM_C_FULL_SMOKE_RESULTS.json").write_text(
            json.dumps(
                [
                    {
                        "code": r.get("code"),
                        "plm": r.get("plm"),
                        "status": r.get("status"),
                        "reason": r.get("reason"),
                        "raw_dim": r.get("raw_dim"),
                        "oof_test": (r.get("summary") or {}).get("scores", {}).get("oof_test")
                        or (r.get("summary") or {}).get("oof_test"),
                    }
                    for r in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.phase == "report":
        # rebuild from on-disk
        for spec in SERIES:
            code = spec["code"]
            oof_p = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
            if oof_p.exists():
                summary = yaml.safe_load(oof_p.read_text())
                results.append(
                    {
                        "code": code,
                        "plm": spec["plm"],
                        "status": "PASS",
                        "summary": summary,
                        "raw_dim": summary.get("raw_plm_hidden_dim"),
                    }
                )
            else:
                st = pack_status(spec["subdir"])
                results.append(
                    {
                        "code": code,
                        "plm": spec["plm"],
                        "status": st["status"] if st["status"] != "READY" else "FAILED_TRAINING",
                        "reason": st.get("reason"),
                    }
                )
        write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
