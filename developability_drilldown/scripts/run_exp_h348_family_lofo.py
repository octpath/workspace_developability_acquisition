#!/usr/bin/env python3
"""EXP-H348+: F1_SURFACE physical-family LOFO (FULL_MINUS_F × 8 families × 2 reps).

FULL baselines reused: EXP-H341 (AbLang1), EXP-H343 (AbLingua). No FULL retrain.
Phases: verify | smoke | issue | train | all
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

from _lib import EXPERIMENTS_COLUMNS  # noqa: E402
from experiment_codes import issue_code, load_codes  # noqa: E402
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID  # noqa: E402
from antibody_transformer.protocol_v3_ext import run_protocol_v3_ext  # noqa: E402
from antibody_transformer.sham_f1_aux import (  # noqa: E402
    TOTAL_DIM,
    make_family_lofo_aux,
    make_surface_prospective_aux,
    BUNDLE_REAL,
)

PREREG_SHA = "e45d9ea8c4720ac92388f7219e7c0d6a0b92cf3f"
FREEZE_V3_SHA = "1bfdf9107f55d14c58117aaa7f64e255dc387ed9"
ARO_HYDRO_SHA = "ebf2fda9615c12f037914bc5b605a6ec6ba2eabc"
FULL_BASELINES = {"ablang1": "EXP-H341", "ablingua": "EXP-H343"}
PARENTS = {"ablang1": "EXP-H187", "ablingua": "EXP-H167"}
ASSET = {
    "ablang1": "assets/transformer/residue_asset_manifest.yaml#ablang1",
    "ablingua": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
}
TAX_PATH = ROOT / "results/HIC_SURFACE_FAMILY_TAXONOMY_FROZEN.json"
MAP_PATH = ROOT / "results/H348_FAMILY_LOFO_SERIES.json"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def require_cuda() -> None:
    import torch

    if ".venv_b1" not in str(Path(sys.executable).resolve()) and ".venv_b1" not in str(Path(torch.__file__).resolve()):
        raise SystemExit(f"must use .venv_b1 (got {sys.executable})")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")


def save_pred(ids, vals, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": list(ids), "y_pred": np.asarray(vals, float)}).to_csv(path, index=False)


def load_tax() -> dict:
    return json.loads(TAX_PATH.read_text())


def build_series(tax: dict) -> list[dict]:
    series = []
    for fam in tax["family_order"]:
        idxs = tax["family_indices_0based"][fam]
        for rep, parent in PARENTS.items():
            series.append(
                {
                    "key": f"{rep}__{fam}",
                    "parent": parent,
                    "full_baseline": FULL_BASELINES[rep],
                    "representation": rep,
                    "family_id": fam,
                    "zero_indices": idxs,
                    "arm": f"FULL_MINUS_{fam}",
                    "experiment_id": f"TRF_HIC_{rep.upper()}_JOINT_FULL_MINUS_{fam}_V3",
                    "description": f"Family LOFO {rep} JOINT FULL minus {fam}",
                }
            )
    return series


def verify() -> dict:
    out = {"ok": True, "details": []}
    tax = load_tax()
    cols = tax["canonical_columns"]
    if len(cols) != 35:
        out["ok"] = False
    flat = []
    for fam in tax["family_order"]:
        flat.extend(tax["family_indices_0based"][fam])
    if sorted(flat) != list(range(35)):
        out["ok"] = False
        out["index_error"] = flat
    for rep, code in FULL_BASELINES.items():
        cfg = ROOT / "experiments/configs" / f"{code}.yaml"
        pred = ROOT / "experiments/predictions" / code / "test.csv"
        oof = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
        ok = cfg.exists() and pred.exists() and oof.exists()
        out["details"].append({"baseline": code, "rep": rep, "ok": ok})
        if not ok:
            out["ok"] = False
    for rep, code in PARENTS.items():
        cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
        checks = {
            "representation": cfg.get("representation") == rep,
            "topology_id_JOINT": cfg.get("topology_id") == "JOINT",
            "annotation_full": str(cfg.get("annotation_mode")).lower() == "full",
            "dual_reg": cfg.get("arch_joint_hl_dual_reg") is True,
            "seed_101": int(cfg.get("seed", -1)) == 101,
        }
        ok = all(checks.values())
        out["details"].append({"parent": code, "checks": checks, "ok": ok})
        if not ok:
            out["ok"] = False
    return out


def config_from_parent(parent_code: str, spec: dict, code: str) -> dict:
    parent = yaml.safe_load((ROOT / "experiments/configs" / f"{parent_code}.yaml").read_text())
    cfg = dict(parent)
    cfg.update(
        {
            "experiment_code": code,
            "experiment_id": spec["experiment_id"],
            "transformer_type": "LATE_FUSION",
            "fusion_bundle_id": f"FULL35_MINUS_{spec['family_id']}",
            "fusion_mode": "late_concat_aux32",
            "aux_dim": TOTAL_DIM,
            "surface_arm": spec["arm"],
            "surface_features_enabled": True,
            "family_id": spec["family_id"],
            "family_zero_indices": list(spec["zero_indices"]),
            "mask_stage": "post_standardscaler_model_input",
            "control_experiment_code": parent_code,
            "full_baseline_code": spec["full_baseline"],
            "batch_id": "H348_FAMILY_LOFO",
            "prereg_sha": PREREG_SHA,
            "freeze_v3_sha": FREEZE_V3_SHA,
            "aro_hydro_sha": ARO_HYDRO_SHA,
            "description": spec["description"],
            "input_asset_ref": ASSET[spec["representation"]],
            "no_external_scoring": False,
            "embargo_public_private_test": False,
        }
    )
    return cfg


def intended_config_diffs(parent: dict, child: dict) -> list[str]:
    ignore = {
        "experiment_code",
        "experiment_id",
        "description",
        "transformer_type",
        "fusion_bundle_id",
        "fusion_mode",
        "aux_dim",
        "surface_arm",
        "surface_features_enabled",
        "family_id",
        "family_zero_indices",
        "mask_stage",
        "control_experiment_code",
        "full_baseline_code",
        "batch_id",
        "prereg_sha",
        "freeze_v3_sha",
        "aro_hydro_sha",
        "no_external_scoring",
        "embargo_public_private_test",
        "input_space",
        "input_asset_ref",
    }
    diffs = []
    for k in sorted(set(parent) | set(child)):
        if k in ignore:
            continue
        if parent.get(k) != child.get(k):
            diffs.append(k)
    return diffs


def write_cfg(code: str, cfg: dict) -> None:
    path = ROOT / "experiments/configs" / f"{code}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def issue_series() -> list[dict]:
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    tax = load_tax()
    issued = []
    for spec in build_series(tax):
        codes = load_codes()
        if spec["experiment_id"] in set(codes["experiment_id"].astype(str)):
            code = codes.loc[codes.experiment_id == spec["experiment_id"], "experiment_code"].iloc[0]
        else:
            code = issue_code(target="HIC", experiment_id=spec["experiment_id"], notes=spec["description"])
        row = {**spec, "experiment_code": code}
        cfg = config_from_parent(spec["parent"], spec, code)
        parent = yaml.safe_load((ROOT / "experiments/configs" / f"{spec['parent']}.yaml").read_text())
        extra = intended_config_diffs(parent, cfg)
        if extra:
            raise SystemExit(f"unexpected config diffs for {code}: {extra}")
        write_cfg(code, cfg)
        issued.append(row)
    MAP_PATH.write_text(json.dumps(issued, indent=2) + "\n")
    return issued


def already_complete(code: str) -> bool:
    oof = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
    pred = ROOT / "experiments/predictions" / code
    needed = [oof, pred / "oof_primary.csv", pred / "oof_shadow.csv", pred / "test.csv"]
    if not all(p.exists() for p in needed):
        return False
    doc = yaml.safe_load(oof.read_text())
    return not (doc.get("quick") or doc.get("technical_smoke"))


def register_internal(code: str, spec: dict, summary: dict) -> None:
    exp_path = ROOT / "results/experiments.csv"
    exp = pd.read_csv(exp_path)
    row = {c: "" for c in EXPERIMENTS_COLUMNS if c in exp.columns}
    oof = summary["scores"]["oof_test"]
    vals = {
        "experiment_code": code,
        "experiment_id": spec["experiment_id"],
        "target": "HIC",
        "family": "TRANSFORMER",
        "notes": spec["description"],
        "cv_primary_mae": oof["primary"],
        "cv_shadow_mae": oof["shadow"],
        "cv_mean_mae": oof["mean"],
        "cv_worst_mae": oof.get("worst", max(oof["primary"], oof["shadow"])),
        "public_mae": np.nan,
        "private_mae": np.nan,
        "test_overall_mae": np.nan,
        "status": "INTERNAL_COMPLETE",
    }
    for k, v in vals.items():
        if k in exp.columns:
            row[k] = v
    if code in set(exp.experiment_code.astype(str)):
        idx = exp.index[exp.experiment_code == code][0]
        for k, v in vals.items():
            if k in exp.columns:
                exp.at[idx, k] = v
    else:
        exp = pd.concat([exp, pd.DataFrame([row])], ignore_index=True)
    exp.to_csv(exp_path, index=False)


def smoke() -> None:
    tax = load_tax()
    full = make_surface_prospective_aux(BUNDLE_REAL)
    ids = full.ids[:48]
    prep = full.fit(ids)
    Xf = full.transform(prep, ids)
    for fam, idxs in tax["family_indices_0based"].items():
        store = make_family_lofo_aux(fam, idxs)
        Xm = store.transform(prep, ids)
        assert np.allclose(Xm[:, idxs], 0.0)
        keep = [i for i in range(TOTAL_DIM) if i not in set(idxs)]
        assert np.allclose(Xm[:, keep], Xf[:, keep])
        print(f"SMOKE_OK {fam} zero={idxs}", flush=True)
    print("SMOKE_FAMILY_LOFO_OK", flush=True)


def run_one(spec: dict, *, quick: bool) -> dict:
    code = spec["experiment_code"]
    print(f"==== TRAIN {code} ({spec['description']}) ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        return {"code": code, "skipped": True, "summary": summary}

    cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
    out = ROOT / "results" / f"{code}_run"
    pred = ROOT / "experiments/predictions" / code
    aux_store = make_family_lofo_aux(spec["family_id"], spec["zero_indices"])
    arch = {
        "joint_hl_single_reg": bool(cfg["arch_joint_hl_single_reg"]),
        "joint_hl_dual_reg": bool(cfg["arch_joint_hl_dual_reg"]),
        "joint_hl_chain_specific_dual_reg": bool(cfg["arch_joint_hl_chain_specific_dual_reg"]),
        "use_cross_attention_bridge": bool(cfg["arch_use_cross_attention_bridge"]),
        "use_reg_only_cross_attention": bool(cfg["arch_use_reg_only_cross_attention"]),
        "use_within_chain_extra_attention": bool(cfg["arch_use_within_chain_extra_attention"]),
        "cross_gate_mode": cfg.get("arch_cross_gate_mode", "learned"),
        "use_cross_geometry_bias": False,
        "share_hl_encoder": True,
    }
    plm = cfg["representation"]
    merge = cfg["merge_mode"]
    dev, test = load_dev_test(ROOT / "data/dev.csv", ROOT / "data/test.csv")
    folds = load_folds(ROOT / "data/folds.csv")
    need = {"need_ablang1": plm == "ablang1", "need_ablingua": plm == "ablingua"}
    rb = load_residue_bundle(dev, test, **need)

    result = run_protocol_v3_ext(
        experiment_code=code,
        target="HIC",
        dev=dev,
        test=test,
        folds=folds,
        rb=rb,
        device=device_str(),
        out_dir=out,
        seed=DEFAULT_SEED,
        quick=quick,
        arch=arch,
        content_mode="frozen",
        merge_mode=merge,
        plm_source=plm,
        chain_mode=cfg.get("chain_mode", "HL"),
        capacity=None,
        aux_store=aux_store,
        fusion_mode="late_concat_aux32",
    )
    summary = result["summary"]
    result["history_df"].to_csv(ROOT / "results" / f"{code}_TRAINING_HISTORY.csv", index=False)
    result["selected_df"].to_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv", index=False)

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
            save_pred(result["test_ids"], result["ext"][f"{scheme}_folds"][k], pred / f"test_{scheme}_fold{k}.csv")
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv")

    oof_doc = {
        "experiment_code": code,
        "platform_id": PLATFORM_ID,
        "config_hash": summary["config_hash"],
        "fusion_bundle_id": cfg["fusion_bundle_id"],
        "family_id": spec["family_id"],
        "zero_indices": spec["zero_indices"],
        "mask_stage": "post_standardscaler_model_input",
        "full_baseline": spec["full_baseline"],
        "prereg_sha": PREREG_SHA,
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external_scored": False,
        "n_trainable": summary["n_trainable"],
        "quick": bool(quick),
        "code_sha": git_rev(),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8")
    (ROOT / "results" / f"{code}_run_state.json").write_text(
        json.dumps({"config_hash": summary["config_hash"], "n_trainable": summary["n_trainable"]}, indent=2) + "\n"
    )
    if not quick:
        register_internal(code, spec, summary)
    print(f"DONE {code} n_trainable={summary['n_trainable']}", flush=True)
    return {"code": code, "skipped": False, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["verify", "smoke", "issue", "train", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("verify", "all"):
        v = verify()
        (ROOT / "results/H348_FAMILY_LOFO_VERIFY.json").write_text(json.dumps(v, indent=2) + "\n")
        print("VERIFY", v["ok"], flush=True)
        if not v["ok"]:
            raise SystemExit("VERIFY_FAILED")

    if args.phase in ("smoke", "all"):
        smoke()

    if args.phase in ("issue", "train", "all"):
        series = issue_series()
        print("N_ARMS", len(series), flush=True)
        print(json.dumps([{"code": s["experiment_code"], "key": s["key"]} for s in series], indent=2), flush=True)

    if args.phase in ("train", "all"):
        require_cuda()
        series = issue_series()
        for spec in series:
            run_one(spec, quick=args.quick)
        # n_trainable equality vs FULL baselines within representation
        for rep, base in FULL_BASELINES.items():
            base_nt = yaml.safe_load((ROOT / "results" / f"{base}_OOF_EVALUATION.yaml").read_text()).get("n_trainable")
            for s in series:
                if s["representation"] != rep:
                    continue
                if not already_complete(s["experiment_code"]):
                    continue
                nt = yaml.safe_load((ROOT / "results" / f"{s['experiment_code']}_OOF_EVALUATION.yaml").read_text()).get(
                    "n_trainable"
                )
                if base_nt is not None and nt is not None and nt != base_nt:
                    raise SystemExit(f"n_trainable mismatch {s['experiment_code']}: {nt} vs {base} {base_nt}")
        print("TRAIN_PHASE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
