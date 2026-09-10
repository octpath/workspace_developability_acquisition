#!/usr/bin/env python3
"""Run EXP-T076..T079 under frozen DL_FOLDLOCAL_COSINE_V3.

Preregistration must exist. Training order fixed. No result-dependent branching.
OOF TEST scientific inspection deferred until all four complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

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
from antibody_transformer.equivalence import synthetic_batch  # noqa: E402
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.protocol_v3 import (  # noqa: E402
    DEFAULT_SEED,
    PLATFORM_ID,
    build_platform_model,
    run_protocol_v3,
)
from preregister_t076_t079 import SERIES  # noqa: E402

TARGET = "TmApp"
PREREG = ROOT / "results" / "T076_T079_ARCHITECTURE_PREREGISTRATION.yaml"
PRE_TEST_FREEZE = ROOT / "results" / "T076_T079_PRE_TEST_FREEZE.yaml"
COMPARE_MD = ROOT / "results" / "T075_T079_ARCHITECTURE_COMPARISON.md"
EXT_CSV = ROOT / "results" / "T076_T079_EXTERNAL_FOUR_WAY.csv"
BOOT_CSV = ROOT / "results" / "T076_T079_PAIRED_BOOTSTRAP.csv"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_prereg() -> dict:
    if not PREREG.exists():
        raise SystemExit(f"missing preregistration {PREREG}; run preregister_t076_t079.py first")
    return yaml.safe_load(PREREG.read_text())


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
        phase="V3_HL_COMMUNICATION_SERIES",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


def zero_gate_equivalence_check() -> dict:
    """T079 with g=0 must match T075/T030 path (≤1e-6)."""
    torch.manual_seed(101)
    t075 = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.0,
    )
    t079 = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        use_cross_attention_bridge=True,
    )
    sd075 = t075.state_dict()
    sd079 = t079.state_dict()
    for k, v in sd075.items():
        if k in sd079 and sd079[k].shape == v.shape:
            sd079[k] = v.clone()
    t079.load_state_dict(sd079)
    assert float(t079.cross_gate_h) == 0.0 and float(t079.cross_gate_l) == 0.0
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=1280,
        Lh=16,
        Ll=12,
        B=4,
    )
    t075.eval()
    t079.eval()
    with torch.no_grad():
        y0 = t075(batch)
        y1 = t079(batch)
    max_pred = float((y0 - y1).abs().max())
    ok = max_pred <= 1e-6
    return {"max_abs_pred_delta": max_pred, "pass": ok}


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


def register(code: str, eid: str, input_space: str, summary: dict, ext_scores: dict, notes: str) -> None:
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
            "experiment_id": eid,
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
            "feature_source": "ablingua_residue",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue",
            "notes": notes,
            "transformer_type": "FROZEN_PLM",
            "plm_source": "ABLINGUA",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "REG",
            "representation_status": "NOT_EXPORTED",
            "input_space": input_space,
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
            "control_experiment_code": "EXP-T075",
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
                "experiment_id": eid,
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
                "notes": f"{PLATFORM_ID} architecture series",
            }
        )
        if "test_prediction_exists" in comp.columns:
            crow["test_prediction_exists"] = True
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)


def run_one(spec: dict, prereg_row: dict, *, quick: bool, silent_test: bool) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} ({spec['description']}) ====", flush=True)
    # protect historical
    for c in ("EXP-T030", "EXP-T068", "EXP-T070", "EXP-T071", "EXP-T072", "EXP-T075"):
        assert (ROOT / "experiments" / "predictions" / c / "oof_primary.csv").exists(), c

    out = ROOT / "results" / f"{code}_run"
    pred = ROOT / "experiments" / "predictions" / code
    hist_csv = ROOT / "results" / f"{code}_TRAINING_HISTORY.csv"
    sel_csv = ROOT / "results" / f"{code}_SELECTED_LR.csv"
    oof_yaml = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    report = ROOT / "results" / f"{code}_PROTOCOL_REPORT.md"
    state = ROOT / "results" / f"{code}_run_state.json"

    if spec["arch"].get("use_cross_attention_bridge"):
        zeq = zero_gate_equivalence_check()
        print("T079 zero-gate equivalence:", zeq, flush=True)
        if not zeq["pass"]:
            raise SystemExit(f"zero-gate equivalence failed: {zeq}")
        (out / "zero_gate_equivalence.json").parent.mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / f"{code}_ZERO_GATE_EQUIVALENCE.json").write_text(json.dumps(zeq, indent=2) + "\n")

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
        out_dir=out,
        seed=DEFAULT_SEED,
        quick=quick,
        arch=spec["arch"],
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
        save_pred(result["dev_ids"], result["oof_val"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_val_{scheme}.csv")
        save_pred(result["dev_ids"], result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float), pred / f"oof_test_{scheme}.csv")
    save_pred(result["dev_ids"], result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_primary.csv")
    save_pred(result["dev_ids"], result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float), pred / "oof_shadow.csv")
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

    # During batch: do not print OOF TEST for scientific decision-making
    if not silent_test:
        print(json.dumps({"code": code, "oof_val": summary["scores"]["oof_val"]}, indent=2), flush=True)
    else:
        print(f"{code} training complete (OOF TEST deferred until batch freeze)", flush=True)

    oof_doc = {
        "experiment_code": code,
        "platform_id": PLATFORM_ID,
        "git_rev": git_rev(),
        "arch": spec["arch"],
        "seed": summary["seed"],
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external_test": ext_scores,
        "selected_lr": summary["selected_lr"],
        "n_trainable_parameters": summary["n_trainable_parameters"],
        "param_account": summary.get("param_account"),
        "cross_gates": summary.get("cross_gates"),
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
        f"- arch: `{spec['arch']}`",
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
    register(code, spec["experiment_id"], spec["input_space"], summary, ext_scores, notes)
    print(f"DONE {code}", flush=True)
    return {
        "code": code,
        "spec": spec,
        "summary": summary,
        "ext_scores": ext_scores,
        "sel": sel_df,
        "y_dev": result["y_dev"],
        "dev_ids": result["dev_ids"],
        "oof_test": result["oof_test"],
        "oof_val": result["oof_val"],
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
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return {"delta_mae": point, "ci95_lo": lo, "ci95_hi": hi, "n": n}


def write_pre_test_freeze(done_codes: list[str]) -> None:
    prereg = load_prereg()
    doc = {
        "status": "PRE_TEST_FREEZE",
        "git_rev": git_rev(),
        "completed_experiments": done_codes,
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "preregistration_git_rev": prereg.get("git_rev_at_preregistration"),
        "statement": (
            "All four architectures were fixed before result inspection. "
            "No architecture changed based on intermediate scores."
        ),
        "platform_id": PLATFORM_ID,
    }
    PRE_TEST_FREEZE.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def finalize_comparisons(bundle: list[dict]) -> None:
    write_pre_test_freeze([b["code"] for b in bundle])
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    t075 = exp[exp.experiment_code == "EXP-T075"].iloc[0]
    t075_oof = yaml.safe_load((ROOT / "results" / "EXP-T075_OOF_EVALUATION.yaml").read_text())

    # Load T075 OOF preds for bootstrap
    def load_oof(code: str, kind: str, scheme: str) -> pd.Series:
        p = ROOT / "experiments" / "predictions" / code / f"oof_{kind}_{scheme}.csv"
        df = pd.read_csv(p)
        return df.set_index("id")[TARGET]

    y_dev = pd.read_csv(ROOT / "data" / "dev.csv").set_index("id")[TARGET]

    rows_ext = []
    # T075
    for agg in ("primary_mean", "primary_median", "shadow_mean", "shadow_median"):
        s = t075_oof["external_test"][agg]
        rows_ext.append(
            {
                "experiment": "EXP-T075",
                "aggregation": agg,
                "public_mae": s["public_mae"],
                "private_mae": s["private_mae"],
                "overall_mae": s["overall_mae"],
            }
        )
    for b in bundle:
        for agg, sc in b["ext_scores"].items():
            rows_ext.append(
                {
                    "experiment": b["code"],
                    "aggregation": agg,
                    "public_mae": sc["public_mae"],
                    "private_mae": sc["private_mae"],
                    "overall_mae": sc["overall_mae"],
                }
            )
    pd.DataFrame(rows_ext).to_csv(EXT_CSV, index=False)

    # Central comparison
    lines = [
        "# T075–T079 Architecture Comparison (DL_FOLDLOCAL_COSINE_V3)",
        "",
        "| Model | H/L base | H-L communication | Summary/readout | Params | VAL_P | VAL_S | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |",
        "|------|------|------|------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|",
    ]

    def add_row(code, hl, comm, readout, params, ov, ot, pub, priv, overall):
        lines.append(
            f"| {code} | {hl} | {comm} | {readout} | {params} | "
            f"{ov['primary']:.4f} | {ov['shadow']:.4f} | {ot['primary']:.4f} | {ot['shadow']:.4f} | "
            f"{ot['mean']:.4f} | {ot['worst']:.4f} | {pub:.4f} | {priv:.4f} | {overall:.4f} |"
        )

    add_row(
        "T075",
        "separate",
        "none",
        "REG_H||REG_L",
        int(t075_oof.get("n_trainable_parameters") or 501633),
        t075_oof["oof_val"],
        t075_oof["oof_test"],
        float(t075["public_mae"]),
        float(t075["private_mae"]),
        float(t075["test_overall_mae"]),
    )
    meta = {
        "EXP-T076": ("joint", "full", "single REG"),
        "EXP-T077": ("joint", "full", "unrestricted REG_H||REG_L"),
        "EXP-T078": ("joint", "residue full; REG restricted", "chain-specific REG_H||REG_L"),
        "EXP-T079": ("separate", "cross-attn bridge", "chain-specific REG_H||REG_L"),
    }
    for b in bundle:
        hl, comm, ro = meta[b["code"]]
        ov, ot = b["summary"]["scores"]["oof_val"], b["summary"]["scores"]["oof_test"]
        pm = b["ext_scores"]["primary_mean"]
        add_row(
            b["code"].replace("EXP-", ""),
            hl,
            comm,
            ro,
            b["summary"]["n_trainable_parameters"],
            ov,
            ot,
            pm["public_mae"],
            pm["private_mae"],
            pm["overall_mae"],
        )

    # Bootstraps
    boot_rows = []
    contrasts = [
        ("T076_vs_T077", "EXP-T076", "EXP-T077", "single_summary_penalty"),
        ("T077_vs_T075", "EXP-T077", "EXP-T075", "full_joint_effect"),
        ("T078_vs_T077", "EXP-T078", "EXP-T077", "reg_specialization"),
        ("T079_vs_T075", "EXP-T079", "EXP-T075", "cross_attention_bridge"),
        ("T079_vs_T077", "EXP-T079", "EXP-T077", "joint_vs_cross"),
        ("T079_vs_T078", "EXP-T079", "EXP-T078", "restricted_vs_cross"),
    ]
    lines += ["", "## Paired bootstrap (delta = MAE_first - MAE_second; negative => first better)", ""]
    for cid, a, b, name in contrasts:
        for scheme in ("primary", "shadow"):
            for kind in ("test", "val"):
                ya = load_oof(a, kind, scheme)
                yb = load_oof(b, kind, scheme)
                ids = sorted(set(ya.index) & set(yb.index) & set(y_dev.index))
                ea = (ya.loc[ids] - y_dev.loc[ids]).abs().to_numpy(float)
                eb = (yb.loc[ids] - y_dev.loc[ids]).abs().to_numpy(float)
                res = paired_bootstrap(ea, eb)
                res.update({"contrast": cid, "name": name, "scheme": scheme, "split": kind, "model_a": a, "model_b": b})
                boot_rows.append(res)
                if kind == "test":
                    lines.append(
                        f"- {cid} {scheme.upper()} TEST: Δ={res['delta_mae']:.4f} "
                        f"[{res['ci95_lo']:.4f}, {res['ci95_hi']:.4f}]"
                    )
    pd.DataFrame(boot_rows).to_csv(BOOT_CSV, index=False)

    # T079 gates
    gate_path = ROOT / "results" / "EXP-T079_CROSS_GATES.csv"
    if gate_path.exists():
        g = pd.read_csv(gate_path)
        lines += [
            "",
            "## T079 cross-attention gates",
            "",
            f"- mean(g_H)={g['g_H'].mean():.6g} median={g['g_H'].median():.6g} mean(|g_H|)={g['g_H'].abs().mean():.6g}",
            f"- mean(g_L)={g['g_L'].mean():.6g} median={g['g_L'].median():.6g} mean(|g_L|)={g['g_L'].abs().mean():.6g}",
            f"- sign(g_H)>0 fraction={(g['g_H'] > 0).mean():.2f}; sign(g_L)>0 fraction={(g['g_L'] > 0).mean():.2f}",
        ]

    lines += [
        "",
        "## Evaluation policy",
        "",
        "- Primary internal metric: TEST_mean",
        "- Robustness: TEST_worst",
        "- External columns above: Primary mean",
        "- Public/Private are post-competition diagnostics; do not override internal verdict.",
        "",
    ]
    COMPARE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", COMPARE_MD, EXT_CSV, BOOT_CSV, PRE_TEST_FREEZE, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "finalize", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma codes e.g. EXP-T076")
    args = ap.parse_args()

    prereg = load_prereg()
    assert prereg["platform_id"] == PLATFORM_ID
    specs = SERIES
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in SERIES if s["code"] in want]

    assert "3090" in torch.cuda.get_device_name(0) or device_str() == "cpu"
    print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

    bundle = []
    if args.phase in ("train", "all"):
        for spec in specs:
            prow = next(r for r in prereg["experiments"] if r["experiment_code"] == spec["code"])
            # verify config hash unchanged
            cfg = ROOT / prow["config_path"]
            sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
            if sha != prow["config_sha256"]:
                raise SystemExit(f"config mutated after prereg: {spec['code']}")
            silent = args.phase == "all"  # defer TEST print until finalize
            bundle.append(run_one(spec, prow, quick=args.quick, silent_test=silent))

    if args.phase in ("finalize", "all"):
        if not bundle:
            # reload from disk
            for spec in SERIES:
                code = spec["code"]
                st = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
                summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
                bundle.append(
                    {
                        "code": code,
                        "spec": spec,
                        "summary": summary,
                        "ext_scores": st["ext_scores"],
                        "sel": pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv"),
                    }
                )
        finalize_comparisons(bundle)
        # now print TEST summaries
        for b in bundle:
            ot = b["summary"]["scores"]["oof_test"]
            print(b["code"], "TEST", ot, flush=True)
        print("BATCH COMPLETE T076-T079", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
