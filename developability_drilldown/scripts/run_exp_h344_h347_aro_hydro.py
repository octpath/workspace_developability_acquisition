#!/usr/bin/env python3
"""EXP-H344–H347: F1_SURFACE ARO_ONLY35 / HYDRO_ONLY35 block ablation.

Reuses H340–H343 for SHAM35 / FULL35. External scoring authorized after train.
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
    ARO_DIM,
    BUNDLE_ARO_ONLY,
    BUNDLE_HYDRO_ONLY,
    BUNDLE_REAL,
    BUNDLE_SHAM,
    HYDRO_DIM,
    TOTAL_DIM,
    make_surface_prospective_aux,
)

PREREG_SHA = "40f8ac0a7cf9caff59cbae913f99638d8cc93c4b"
FREEZE_V3_SHA = "1bfdf9107f55d14c58117aaa7f64e255dc387ed9"
REUSED = {
    "ablang1_SHAM35": "EXP-H340",
    "ablang1_FULL35": "EXP-H341",
    "ablingua_SHAM35": "EXP-H342",
    "ablingua_FULL35": "EXP-H343",
}
PARENTS = {"ablang1": "EXP-H187", "ablingua": "EXP-H167"}
SERIES = [
    {
        "key": "ablang1_aro",
        "parent": "EXP-H187",
        "representation": "ablang1",
        "arm": "ARO_ONLY35",
        "fusion_bundle_id": BUNDLE_ARO_ONLY,
        "experiment_id": "TRF_HIC_ABLANG1_JOINT_FULL_ARO_ONLY35_V3",
        "description": "ARO/HYDRO decomp AbLang1 JOINT FULL + ARO_ONLY35",
    },
    {
        "key": "ablang1_hydro",
        "parent": "EXP-H187",
        "representation": "ablang1",
        "arm": "HYDRO_ONLY35",
        "fusion_bundle_id": BUNDLE_HYDRO_ONLY,
        "experiment_id": "TRF_HIC_ABLANG1_JOINT_FULL_HYDRO_ONLY35_V3",
        "description": "ARO/HYDRO decomp AbLang1 JOINT FULL + HYDRO_ONLY35",
    },
    {
        "key": "ablingua_aro",
        "parent": "EXP-H167",
        "representation": "ablingua",
        "arm": "ARO_ONLY35",
        "fusion_bundle_id": BUNDLE_ARO_ONLY,
        "experiment_id": "TRF_HIC_ABLINGUA_JOINT_FULL_ARO_ONLY35_V3",
        "description": "ARO/HYDRO decomp AbLingua JOINT FULL + ARO_ONLY35",
    },
    {
        "key": "ablingua_hydro",
        "parent": "EXP-H167",
        "representation": "ablingua",
        "arm": "HYDRO_ONLY35",
        "fusion_bundle_id": BUNDLE_HYDRO_ONLY,
        "experiment_id": "TRF_HIC_ABLINGUA_JOINT_FULL_HYDRO_ONLY35_V3",
        "description": "ARO/HYDRO decomp AbLingua JOINT FULL + HYDRO_ONLY35",
    },
]
ASSET = {
    "ablang1": "assets/transformer/residue_asset_manifest.yaml#ablang1",
    "ablingua": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
}
MAP_PATH = ROOT / "results/H344_H347_ARO_HYDRO_SERIES.json"


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


def verify_reused_and_parents() -> dict:
    out = {"ok": True, "details": [], "reused": []}
    for key, code in REUSED.items():
        cfg_path = ROOT / "experiments/configs" / f"{code}.yaml"
        oof = ROOT / "results" / f"{code}_OOF_EVALUATION.yaml"
        pred = ROOT / "experiments/predictions" / code / "test.csv"
        ok = cfg_path.exists() and oof.exists() and pred.exists()
        out["reused"].append({"key": key, "code": code, "ok": ok})
        if not ok:
            out["ok"] = False
    for rep, code in PARENTS.items():
        cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
        checks = {
            "representation": cfg.get("representation") == rep,
            "topology_id_JOINT": cfg.get("topology_id") == "JOINT",
            "topology_B1": cfg.get("topology") == "B1",
            "annotation_full": str(cfg.get("annotation_mode")).lower() == "full",
            "dual_reg": cfg.get("arch_joint_hl_dual_reg") is True,
            "share_hl": cfg.get("arch_share_hl_encoder") is True or cfg.get("share_hl_encoder") is True,
            "pooling_REG": cfg.get("pooling_mode") == "REG",
            "merge_mean": cfg.get("merge_mode") == "mean",
            "seed_101": int(cfg.get("seed", -1)) == 101,
            "frozen": cfg.get("content_mode") == "frozen",
        }
        ok = all(checks.values())
        out["details"].append({"parent": code, "representation": rep, "checks": checks, "ok": ok})
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
            "fusion_bundle_id": spec["fusion_bundle_id"],
            "fusion_mode": "late_concat_aux32",
            "aux_dim": TOTAL_DIM,
            "surface_arm": spec["arm"],
            "surface_features_enabled": True,
            "control_experiment_code": parent_code,
            "batch_id": "H344_H347_ARO_HYDRO_DECOMPOSITION",
            "prereg_sha": PREREG_SHA,
            "freeze_v3_sha": FREEZE_V3_SHA,
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
        "control_experiment_code",
        "batch_id",
        "prereg_sha",
        "freeze_v3_sha",
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
    issued = []
    for spec in SERIES:
        codes = load_codes()
        if spec["experiment_id"] in set(codes["experiment_id"].astype(str)):
            code = codes.loc[codes.experiment_id == spec["experiment_id"], "experiment_code"].iloc[0]
        else:
            code = issue_code(
                target="HIC",
                experiment_id=spec["experiment_id"],
                notes=spec["description"],
            )
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
    if doc.get("quick") or doc.get("technical_smoke"):
        return False
    return True


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


def smoke_blocks() -> None:
    ids_probe = None
    for bid, expect_zero in [
        (BUNDLE_SHAM, {"AROMATIC_TOPO", "HYDRO_FIELD"}),
        (BUNDLE_ARO_ONLY, {"HYDRO_FIELD"}),
        (BUNDLE_HYDRO_ONLY, {"AROMATIC_TOPO"}),
        (BUNDLE_REAL, set()),
    ]:
        store = make_surface_prospective_aux(bid)
        if ids_probe is None:
            ids_probe = store.ids[:48]
        prep = store.fit(ids_probe)
        X = store.transform(prep, ids_probe)
        assert X.shape == (len(ids_probe), TOTAL_DIM), (bid, X.shape)
        aro = X[:, :ARO_DIM]
        hydro = X[:, ARO_DIM:]
        if "AROMATIC_TOPO" in expect_zero:
            assert np.allclose(aro, 0.0), bid
        else:
            assert not np.allclose(aro, 0.0), bid
        if "HYDRO_FIELD" in expect_zero:
            assert np.allclose(hydro, 0.0), bid
        else:
            assert not np.allclose(hydro, 0.0), bid
        print(f"SMOKE_OK {bid} dim={TOTAL_DIM} aro_nonzero={not np.allclose(aro,0)} hydro_nonzero={not np.allclose(hydro,0)}", flush=True)
    print("SMOKE_ARO_HYDRO_OK", flush=True)


def run_one(spec: dict, *, quick: bool) -> dict:
    code = spec["experiment_code"]
    print(f"==== TRAIN {code} ({spec['description']}) ====", flush=True)
    if already_complete(code) and not quick:
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        return {"code": code, "spec": spec, "summary": summary, "skipped": True}

    cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
    out = ROOT / "results" / f"{code}_run"
    pred = ROOT / "experiments/predictions" / code
    aux_store = make_surface_prospective_aux(cfg["fusion_bundle_id"])
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
    hist_df = result["history_df"]
    sel_df = result["selected_df"]
    hist_df.to_csv(ROOT / "results" / f"{code}_TRAINING_HISTORY.csv", index=False)
    sel_df.to_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv", index=False)

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
        "surface_arm": spec["arm"],
        "parent": spec["parent"],
        "prereg_sha": PREREG_SHA,
        "freeze_v3_sha": FREEZE_V3_SHA,
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external_scored": False,
        "n_trainable": summary["n_trainable"],
        "quick": bool(quick),
        "code_sha": git_rev(),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(
        yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "results" / f"{code}_run_state.json").write_text(
        json.dumps({"config_hash": summary["config_hash"], "n_trainable": summary["n_trainable"]}, indent=2) + "\n"
    )
    if not quick:
        register_internal(code, spec, summary)
    print(f"DONE {code} n_trainable={summary['n_trainable']}", flush=True)
    return {"code": code, "spec": spec, "summary": summary, "skipped": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["verify", "smoke", "issue", "train", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("verify", "all"):
        v = verify_reused_and_parents()
        (ROOT / "results/H344_H347_PARENT_VERIFY.json").write_text(json.dumps(v, indent=2) + "\n")
        print("VERIFY", v["ok"], flush=True)
        if not v["ok"]:
            raise SystemExit("VERIFY_FAILED")

    if args.phase in ("smoke", "all"):
        smoke_blocks()

    if args.phase in ("issue", "train", "all"):
        series = issue_series()
        print(
            "SERIES",
            json.dumps([{k: s[k] for k in ("experiment_code", "key", "arm")} for s in series]),
            flush=True,
        )

    if args.phase in ("train", "all"):
        require_cuda()
        series = issue_series()
        for spec in series:
            run_one(spec, quick=args.quick)
        # param equality audit vs H340/H341
        nt = {}
        for code in ["EXP-H340", "EXP-H341"] + [s["experiment_code"] for s in series if s["representation"] == "ablang1"]:
            doc = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
            nt[code] = doc.get("n_trainable")
        print("N_TRAINABLE_AUDIT", nt, flush=True)
        vals = [v for v in nt.values() if v is not None]
        if vals and len(set(vals)) != 1:
            raise SystemExit(f"n_trainable mismatch: {nt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
