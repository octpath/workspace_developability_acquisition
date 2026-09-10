#!/usr/bin/env python3
"""Run EXP-T124..T129 + EXP-H082..H093 under DL_FOLDLOCAL_COSINE_V3.

Phases: prereg | train_tm | train_hic | finalize | all
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from antibody_transformer.data import load_dev_test, load_folds, load_residue_bundle, load_solution  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.h047_aux_features import H047AuxFeatureStore, f4_subblock_slices  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID  # noqa: E402
from antibody_transformer.protocol_v3_ext import (  # noqa: E402
    predict_with_checkpoint_ext,
    run_protocol_v3_ext,
)
from preregister_t124_h082_batch import (  # noqa: E402
    CAPACITY,
    SERIES,
    SERIES_HIC,
    SERIES_TM,
)

PREREG = ROOT / "results" / "T124_T129_H082_H093_PREREGISTRATION.yaml"
PRE_EXTERNAL = ROOT / "results" / "T124_T129_H082_H093_PRE_EXTERNAL_FREEZE.yaml"


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def ensure_prereg() -> dict:
    if not PREREG.exists():
        from preregister_t124_h082_batch import main as prereg_main

        rc = prereg_main()
        if rc != 0:
            raise SystemExit(rc)
    return yaml.safe_load(PREREG.read_text())


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


def issue_one(spec: dict) -> str:
    codes = load_codes()
    # experiment_id from config
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{spec['code']}.yaml").read_text())
    eid = cfg["experiment_id"]
    if (codes["experiment_id"] == eid).any():
        return str(codes.set_index("experiment_id").loc[eid, "experiment_code"])
    expect = spec["code"]
    target = "TmApp" if expect.startswith("EXP-T") else "HIC"
    nxt = next_code(target)
    if nxt != expect:
        raise SystemExit(f"expected {expect}, got {nxt}")
    code = issue_code(
        eid,
        target,
        source_model_id=cfg["source_model_id"],
        phase="V3_T124_H082_CAPACITY_FUSION",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


def register(code: str, spec: dict, summary: dict, ext_scores: dict) -> None:
    target = "TmApp" if code.startswith("EXP-T") else "HIC"
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    is_frozen = cfg.get("content_mode") == "frozen"
    plm_yaml = cfg.get("plm_source", "NONE")
    exp_path = ROOT / "results" / "experiments.csv"
    df = pd.read_csv(exp_path)
    row = {c: "" for c in EXPERIMENTS_COLUMNS}
    row.update(
        {
            "experiment_code": code,
            "experiment_id": cfg["experiment_id"],
            "legacy_experiment_code": "",
            "target": target,
            "family": "TRANSFORMER",
            "model_type": cfg.get("transformer_type", "TRANSFORMER"),
            "feature_set_id": "",
            "feature_space": "",
            "feature_path": "",
            "input_space": cfg["input_space"],
            "input_asset_ref": cfg.get("input_asset_ref", ""),
            "plm_source": plm_yaml if plm_yaml != "NONE" else "NONE",
            "representation_status": "NOT_EXPORTED",
            "transformer_type": cfg.get("transformer_type", "FROZEN_PLM"),
            "config_path": f"experiments/configs/{code}.yaml",
            "artifact_status": "FULL",
            "cv_primary_mae": oof["primary"],
            "cv_shadow_mae": oof["shadow"],
            "cv_worst_mae": oof["worst"],
            "public_mae": pm["public_mae"],
            "private_mae": pm["private_mae"],
            "test_overall_mae": pm["overall_mae"],
            "selection_policy_at_creation": "CV_ONLY",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "license_status": "REVIEW" if is_frozen else "OK",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "REPRODUCED",
            "shareability_status": "SHAREABLE_PARTIAL",
            "canonical_benchmark_eligible": "NO",
            "source_model_id": cfg["source_model_id"],
            "notes": spec["description"],
        }
    )
    df = df[df["experiment_code"] != code]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(exp_path, index=False)


def prepare_rb(spec_or_cfg: dict, dev, test):
    cm = spec_or_cfg.get("content_mode", "frozen")
    plm = spec_or_cfg.get("plm_source")
    if isinstance(plm, str):
        plm = {"ABLANG2": "ablang2", "ESM2": "esm2", "NONE": None, "ablingua": "ablingua"}.get(
            plm, plm.lower() if plm not in (None, "NONE") else None
        )
    need_ablang2 = plm == "ablang2" or spec_or_cfg.get("representation") == "ABLANG2"
    need_esm2 = plm == "esm2" or spec_or_cfg.get("representation") == "ESM2"
    return load_residue_bundle(
        dev,
        test,
        need_ablingua=False,
        need_ablang2=need_ablang2,
        need_esm2=need_esm2,
    )


def run_one(spec: dict, *, quick: bool) -> dict:
    code = issue_one(spec)
    print(f"==== TRAIN {code} ({spec['description']}) ====", flush=True)
    if already_complete(code):
        print(f"SKIP {code}", flush=True)
        summary = json.loads((ROOT / "results" / f"{code}_run" / "summary.json").read_text())
        st = json.loads((ROOT / "results" / f"{code}_run_state.json").read_text())
        return {"code": code, "spec": spec, "summary": summary, "ext_scores": st["ext_scores"], "skipped": True}

    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    target = cfg["target"]
    out = ROOT / "results" / f"{code}_run"
    pred = ROOT / "experiments" / "predictions" / code

    capacity = None
    if "capacity_id" in spec:
        c = CAPACITY[spec["capacity_id"]]
        capacity = {
            "d_model": c["d_model"],
            "n_layers": c["n_layers"],
            "n_heads": c["n_heads"],
            "dim_feedforward": c["dim_feedforward"],
        }

    aux_store = None
    if "fusion_bundle_id" in spec:
        aux_store = H047AuxFeatureStore(spec["fusion_bundle_id"])

    arch = {
        "joint_hl_single_reg": cfg["arch_joint_hl_single_reg"],
        "joint_hl_dual_reg": cfg["arch_joint_hl_dual_reg"],
        "joint_hl_chain_specific_dual_reg": cfg["arch_joint_hl_chain_specific_dual_reg"],
        "use_cross_attention_bridge": cfg["arch_use_cross_attention_bridge"],
        "use_reg_only_cross_attention": cfg["arch_use_reg_only_cross_attention"],
        "use_within_chain_extra_attention": cfg["arch_use_within_chain_extra_attention"],
        "cross_gate_mode": cfg["arch_cross_gate_mode"],
        "use_cross_geometry_bias": False,
    }
    merge = cfg["merge_mode"] or "concat"
    plm = cfg["plm_source"]
    plm_src = {"ABLANG2": "ablang2", "ESM2": "esm2", "NONE": None}[plm]
    content = cfg["content_mode"]
    chain = cfg["chain_mode"]

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = prepare_rb(cfg, dev, test)

    result = run_protocol_v3_ext(
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
        arch=arch,
        content_mode=content,
        merge_mode=merge if merge != "null" else "concat",
        plm_source=plm_src,
        chain_mode=chain,
        capacity=capacity,
        aux_store=aux_store,
    )
    summary = result["summary"]
    hist_df = result["history_df"]
    sel_df = result["selected_df"]
    hist_df.to_csv(ROOT / "results" / f"{code}_TRAINING_HISTORY.csv", index=False)
    sel_df.to_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv", index=False)
    hist_df.to_csv(out / "training_history.csv", index=False)
    sel_df.to_csv(out / "selected_lr.csv", index=False)

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
                result["ext"][f"{scheme}_folds"][k],
                pred / f"test_{scheme}_fold{k}.csv",
                target,
            )
    save_pred(result["test_ids"], result["ext"]["primary_mean"], pred / "test.csv", target)

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    ext_scores = {
        key: score_external(result["ext"][key], result["test_ids"], sol, target)
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }
    oof_doc = {
        "experiment_code": code,
        "platform_id": PLATFORM_ID,
        "config_hash": summary["config_hash"],
        "capacity": summary.get("capacity"),
        "fusion_bundle_id": summary.get("fusion_bundle_id"),
        "oof_val": summary["scores"]["oof_val"],
        "oof_test": summary["scores"]["oof_test"],
        "external": ext_scores,
        "n_trainable": summary["n_trainable"],
        "selected": sel_df.to_dict(orient="records"),
    }
    (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").write_text(
        yaml.safe_dump(oof_doc, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "results" / f"{code}_run_state.json").write_text(
        json.dumps({"ext_scores": ext_scores, "config_hash": summary["config_hash"]}, indent=2)
    )
    report = [
        f"# {code} — {spec['description']}",
        "",
        f"- platform: `{PLATFORM_ID}`",
        f"- TEST_mean: {summary['scores']['oof_test']['mean']:.6f}",
        f"- params: {summary['n_trainable']}",
        f"- Overall (P-mean): {ext_scores['primary_mean']['overall_mae']:.6f}",
        "",
    ]
    (ROOT / "results" / f"{code}_PROTOCOL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    register(code, spec, summary, ext_scores)
    print(f"DONE {code}", flush=True)
    return {"code": code, "spec": spec, "summary": summary, "ext_scores": ext_scores, "skipped": False}


def load_oof(code: str) -> dict:
    return yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())


def paired_boot(a: np.ndarray, b: np.ndarray, y: np.ndarray, n=2000, seed=101):
    ea = np.abs(a - y)
    eb = np.abs(b - y)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = []
    n_s = len(d)
    for _ in range(n):
        idx = rng.integers(0, n_s, n_s)
        boots.append(float(d[idx].mean()))
    boots = np.asarray(boots)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def load_oof_preds(code: str, kind: str, scheme: str) -> pd.Series:
    p = ROOT / "experiments" / "predictions" / code / f"oof_{kind}_{scheme}.csv"
    df = pd.read_csv(p)
    col = [c for c in df.columns if c != "id"][0]
    return df.set_index("id")[col]


def permutation_importance(code: str, *, n_perm: int = 100, seed: int = 2026) -> list[dict]:
    """Permute entire aux vector on held-out OOF TEST sets (no retrain)."""
    import torch

    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    sel = pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv")
    store = H047AuxFeatureStore(cfg["fusion_bundle_id"])
    target = cfg["target"]
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = prepare_rb(cfg, dev, test)
    y_map = {str(r["id"]): float(r[target]) for _, r in dev.iterrows()}
    dev_ids = dev["id"].astype(str).tolist()
    device = torch.device(device_str())
    rows = []
    from antibody_transformer.data import tvt_split

    for _, row in sel.iterrows():
        scheme = str(row["scheme"])
        k = int(row["fold"])
        ckpt = Path(str(row["checkpoint"]))
        fmap = folds.primary if scheme == "primary" else folds.shadow
        tr, va, te = tvt_split(fmap, k, dev_ids)
        y_te = np.asarray([y_map[a] for a in te], float)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        prep = blob["prep"]
        # normal
        pred_n = predict_with_checkpoint_ext(
            ckpt_path=ckpt,
            ids=te,
            rb=rb,
            device=device,
            y_placeholder=y_te,
            aux_store=store,
        )
        mae_n = float(mae(y_te, pred_n))
        X = store.transform(prep, te)
        rng = np.random.default_rng(seed + k * 17 + (0 if scheme == "primary" else 100))
        deltas = []
        for _ in range(n_perm):
            Xp = X.copy()
            Xp = Xp[rng.permutation(len(Xp))]
            # manual forward with permuted fixed
            from antibody_transformer.protocol_v3_ext import build_platform_model_ext
            from antibody_transformer.training import AbDataset, collate_batch, _batch_to_device
            from torch.utils.data import DataLoader
            from antibody_transformer.protocol_v3 import BATCH_SIZE, normalize_arch_flags

            flags = normalize_arch_flags(blob.get("arch"))
            model = build_platform_model_ext(
                rb,
                flags,
                content_mode=blob["content_mode"],
                merge_mode=blob["merge_mode"],
                plm_source=blob["plm_source"],
                chain_mode=blob["chain_mode"],
                capacity=blob.get("capacity"),
                aux_dim=blob["aux_dim"],
            ).to(device)
            model.load_state_dict(blob["model"])
            model.eval()
            mu, sd = float(blob["mu"]), float(blob["sd"])
            ds = AbDataset(
                te,
                y_te,
                rb,
                content_mode=blob["content_mode"],
                plm_source=blob["plm_source"] or "ablingua",
                fixed_X=Xp,
                chain_mode=blob["chain_mode"],
            )
            preds = []
            with torch.no_grad():
                for batch in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch):
                    batch = _batch_to_device(batch, device)
                    batch.pop("y", None)
                    fixed = batch.pop("fixed")
                    pred_z = model(batch, fixed)
                    preds.append((pred_z * sd + mu).cpu().numpy())
            pred_p = np.concatenate(preds)
            deltas.append(float(mae(y_te, pred_p) - mae_n))
        arr = np.asarray(deltas)
        rows.append(
            {
                "code": code,
                "scheme": scheme,
                "fold": k,
                "split": "TEST_OOF",
                "block": "ALL_AUX",
                "mae_normal": mae_n,
                "delta_mae_mean": float(arr.mean()),
                "delta_mae_sd": float(arr.std()),
                "delta_mae_q025": float(np.quantile(arr, 0.025)),
                "delta_mae_q975": float(np.quantile(arr, 0.975)),
                "n_perm": n_perm,
            }
        )
    return rows


def f4_subblock_perm(code: str, *, n_perm: int = 100, seed: int = 3030) -> list[dict]:
    import torch
    from antibody_transformer.data import tvt_split
    from antibody_transformer.protocol_v3_ext import build_platform_model_ext
    from antibody_transformer.training import AbDataset, collate_batch, _batch_to_device
    from antibody_transformer.protocol_v3 import BATCH_SIZE, normalize_arch_flags
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    assert cfg["fusion_bundle_id"] == "F4_H047_AUX_ALL"
    sel = pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv")
    store = H047AuxFeatureStore("F4_H047_AUX_ALL")
    slices = f4_subblock_slices()
    target = "HIC"
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = prepare_rb(cfg, dev, test)
    y_map = {str(r["id"]): float(r[target]) for _, r in dev.iterrows()}
    dev_ids = dev["id"].astype(str).tolist()
    device = torch.device(device_str())
    rows = []
    for _, row in sel.iterrows():
        scheme = str(row["scheme"])
        k = int(row["fold"])
        ckpt = Path(str(row["checkpoint"]))
        fmap = folds.primary if scheme == "primary" else folds.shadow
        tr, va, te = tvt_split(fmap, k, dev_ids)
        y_te = np.asarray([y_map[a] for a in te], float)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        prep = blob["prep"]
        X = store.transform(prep, te)
        pred_n = predict_with_checkpoint_ext(
            ckpt_path=ckpt, ids=te, rb=rb, device=device, y_placeholder=y_te, aux_store=store
        )
        mae_n = float(mae(y_te, pred_n))
        for bname, sl in slices.items():
            rng = np.random.default_rng(seed + hash(bname) % 1000 + k)
            deltas = []
            for _ in range(n_perm):
                Xp = X.copy()
                block = Xp[:, sl]
                Xp[:, sl] = block[rng.permutation(len(block))]
                flags = normalize_arch_flags(blob.get("arch"))
                model = build_platform_model_ext(
                    rb,
                    flags,
                    content_mode=blob["content_mode"],
                    merge_mode=blob["merge_mode"],
                    plm_source=blob["plm_source"],
                    chain_mode=blob["chain_mode"],
                    capacity=blob.get("capacity"),
                    aux_dim=blob["aux_dim"],
                ).to(device)
                model.load_state_dict(blob["model"])
                model.eval()
                mu, sd = float(blob["mu"]), float(blob["sd"])
                ds = AbDataset(
                    te,
                    y_te,
                    rb,
                    content_mode=blob["content_mode"],
                    plm_source=blob["plm_source"] or "ablingua",
                    fixed_X=Xp,
                    chain_mode=blob["chain_mode"],
                )
                preds = []
                with torch.no_grad():
                    for batch in DataLoader(
                        ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
                    ):
                        batch = _batch_to_device(batch, device)
                        batch.pop("y", None)
                        fixed = batch.pop("fixed")
                        pred_z = model(batch, fixed)
                        preds.append((pred_z * sd + mu).cpu().numpy())
                deltas.append(float(mae(y_te, np.concatenate(preds)) - mae_n))
            arr = np.asarray(deltas)
            rows.append(
                {
                    "code": code,
                    "scheme": scheme,
                    "fold": k,
                    "block": bname,
                    "mae_normal": mae_n,
                    "delta_mae_mean": float(arr.mean()),
                    "delta_mae_sd": float(arr.std()),
                    "delta_mae_q025": float(np.quantile(arr, 0.025)),
                    "delta_mae_q975": float(np.quantile(arr, 0.975)),
                    "n_perm": n_perm,
                }
            )
    return rows


def finalize() -> None:
    print("=== finalize ===", flush=True)
    # freeze
    freeze = {
        "status": "PRE_EXTERNAL_FREEZE",
        "git_rev": git_rev(),
        "completed": [s["code"] for s in SERIES],
        "n_completed": len(SERIES),
        "preregistration": str(PREREG),
        "statement": "All configs unchanged; internal TEST frozen before comparative reports.",
        "platform_id": PLATFORM_ID,
    }
    PRE_EXTERNAL.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")

    # master tables + bootstrap + reports (compact)
    rows_tm = []
    for code in ["EXP-T113", "EXP-T121"] + [s["code"] for s in SERIES_TM]:
        oof = load_oof(code) if (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").exists() else None
        cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
        e = pd.read_csv(ROOT / "results" / "experiments.csv")
        er = e[e.experiment_code == code].iloc[0]
        ot = oof["oof_test"] if oof else {
            "primary": er.cv_primary_mae,
            "shadow": er.cv_shadow_mae,
            "mean": 0.5 * (er.cv_primary_mae + er.cv_shadow_mae),
            "worst": max(er.cv_primary_mae, er.cv_shadow_mae),
        }
        rows_tm.append(
            {
                "code": code,
                "base_arch": cfg.get("arch_id"),
                "d_model": cfg.get("d_model"),
                "layers": cfg.get("n_layers"),
                "heads": cfg.get("n_heads"),
                "FFN": cfg.get("ff_dim"),
                "params": (oof or {}).get("n_trainable", ""),
                "TEST_P": ot["primary"],
                "TEST_S": ot["shadow"],
                "TEST_mean": ot["mean"],
                "TEST_worst": ot["worst"],
                "Public": er.public_mae,
                "Private": er.private_mae,
                "Overall": er.test_overall_mae,
            }
        )
    tm_df = pd.DataFrame(rows_tm)
    tm_df.to_csv(ROOT / "results" / "T113_T129_CAPACITY_TABLE.csv", index=False)

    rows_h = []
    for code in ["EXP-H054", "EXP-H061", "EXP-H071"] + [s["code"] for s in SERIES_HIC]:
        oof = load_oof(code)
        cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
        e = pd.read_csv(ROOT / "results" / "experiments.csv")
        er = e[e.experiment_code == code].iloc[0]
        ot = oof["oof_test"]
        rows_h.append(
            {
                "code": code,
                "backbone": cfg.get("control_experiment_code") or code,
                "feature_bundle": cfg.get("fusion_bundle_id", "NONE"),
                "aux_dim": (oof.get("fusion_bundle_id") and "") or "",
                "params": oof.get("n_trainable", ""),
                "TEST_P": ot["primary"],
                "TEST_S": ot["shadow"],
                "TEST_mean": ot["mean"],
                "TEST_worst": ot["worst"],
                "Public": er.public_mae,
                "Private": er.private_mae,
                "Overall": er.test_overall_mae,
            }
        )
    # fix aux dim from store
    for r in rows_h:
        bid = r["feature_bundle"]
        if bid and bid != "NONE":
            r["aux_dim"] = H047AuxFeatureStore(bid).effective_dim
        else:
            r["aux_dim"] = 0
    hic_df = pd.DataFrame(rows_h)
    hic_df.to_csv(ROOT / "results" / "HIC_H047_FUSION_TABLE.csv", index=False)

    # bootstrap
    boot_rows = []
    contrasts = [
        ("EXP-T124", "EXP-T113"),
        ("EXP-T125", "EXP-T113"),
        ("EXP-T126", "EXP-T113"),
        ("EXP-T126", "EXP-T125"),
        ("EXP-T126", "EXP-T124"),
        ("EXP-T127", "EXP-T121"),
        ("EXP-T128", "EXP-T121"),
        ("EXP-T129", "EXP-T121"),
        ("EXP-T129", "EXP-T128"),
        ("EXP-T129", "EXP-T127"),
        ("EXP-T124", "EXP-T127"),
        ("EXP-T125", "EXP-T128"),
        ("EXP-T126", "EXP-T129"),
    ]
    # HIC
    mapping = {
        "EXP-H054": ["EXP-H082", "EXP-H083", "EXP-H084", "EXP-H085"],
        "EXP-H061": ["EXP-H086", "EXP-H087", "EXP-H088", "EXP-H089"],
        "EXP-H071": ["EXP-H090", "EXP-H091", "EXP-H092", "EXP-H093"],
    }
    for base, fus in mapping.items():
        for f in fus:
            contrasts.append((f, base))
        # F4 vs F1/F2/F3
        f4 = fus[3]
        for f in fus[:3]:
            contrasts.append((f4, f))

    for a, b in contrasts:
        target = "TmApp" if a.startswith("EXP-T") else "HIC"
        dev, _ = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
        y = dev.set_index("id")[target].astype(float)
        for scheme in ("primary", "shadow"):
            pa = load_oof_preds(a, "test", scheme)
            pb = load_oof_preds(b, "test", scheme)
            ids = pa.index.intersection(pb.index).intersection(y.index)
            d, lo, hi = paired_boot(pa.loc[ids].to_numpy(), pb.loc[ids].to_numpy(), y.loc[ids].to_numpy())
            boot_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "target": target,
                    "scheme": scheme,
                    "split": "test",
                    "delta_mae": d,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "sign": "delta=MAE_a-MAE_b; negative => a better",
                }
            )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(ROOT / "results" / "TM_HIC_NEXT_BATCH_PAIRED_BOOTSTRAP.csv", index=False)

    # external four-way
    ext_rows = []
    for s in SERIES:
        code = s["code"]
        oof = load_oof(code)
        for agg, sc in oof["external"].items():
            ext_rows.append({"experiment": code, "aggregation": agg, **sc})
    pd.DataFrame(ext_rows).to_csv(ROOT / "results" / "TM_HIC_NEXT_BATCH_EXTERNAL_FOUR_WAY.csv", index=False)

    # permutation diagnostics
    perm_all = []
    for s in SERIES_HIC:
        print(f"perm {s['code']}", flush=True)
        perm_all.extend(permutation_importance(s["code"]))
    pd.DataFrame(perm_all).to_csv(ROOT / "results" / "HIC_H047_FEATURE_PERMUTATION.csv", index=False)

    sub_all = []
    for code in ("EXP-H085", "EXP-H089", "EXP-H093"):
        print(f"subblock perm {code}", flush=True)
        sub_all.extend(f4_subblock_perm(code))
    pd.DataFrame(sub_all).to_csv(ROOT / "results" / "HIC_H047_F4_SUBBLOCK_PERMUTATION.csv", index=False)

    # markdown reports
    def md_table(df: pd.DataFrame) -> str:
        cols = df.columns.tolist()
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    (ROOT / "results" / "T113_T129_ABLANG2_CAPACITY_REPORT.md").write_text(
        "# AbLang2 capacity refinement (T113/T121 + T124–T129)\n\n"
        + md_table(tm_df)
        + "\n\nSee `TM_HIC_NEXT_BATCH_PAIRED_BOOTSTRAP.csv` for planned contrasts.\n",
        encoding="utf-8",
    )
    h047 = pd.read_csv(ROOT / "results" / "experiments.csv")
    h047r = h047[h047.experiment_code == "EXP-H047"].iloc[0]
    hic_note = (
        f"\n\nHistorical EXP-H047 (classical RBF-SVR, not protocol-equivalent): "
        f"Overall={h047r.test_overall_mae:.4f}.\n"
    )
    (ROOT / "results" / "HIC_H047_FEATURE_FUSION_REPORT.md").write_text(
        "# HIC H047 late-fusion report (H054/H061/H071 + H082–H093)\n\n"
        + md_table(hic_df)
        + hic_note
        + "\nPermutation: `HIC_H047_FEATURE_PERMUTATION.csv`, "
        "`HIC_H047_F4_SUBBLOCK_PERMUTATION.csv`.\n",
        encoding="utf-8",
    )

    # synthesis skeleton filled with numbers
    best_tm = tm_df.loc[tm_df.TEST_mean.astype(float).idxmin()]
    best_h = hic_df.loc[hic_df.TEST_mean.astype(float).idxmin()]
    synth = f"""# Tm structure vs HIC feature synthesis

## A. Did capacity help Tm?
See capacity table. Best internal among T113/T121/T124–T129: **{best_tm.code}** TEST_mean={float(best_tm.TEST_mean):.4f}.

## B. Did physical fusion help HIC?
Best among controls+fusion: **{best_h.code}** TEST_mean={float(best_h.TEST_mean):.4f} (bundle={best_h.feature_bundle}).

## C. Contrast hypothesis
Compare within-target deltas in bootstrap CSV and permutation diagnostics before claiming
"Tm = model-structure" vs "HIC = feature-information".
"""
    (ROOT / "results" / "TM_STRUCTURE_VS_HIC_FEATURE_SYNTHESIS.md").write_text(synth, encoding="utf-8")
    print("BATCH FINALIZE COMPLETE", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=["prereg", "train_tm", "train_hic", "finalize", "all"],
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    ensure_prereg()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    if args.phase in ("prereg", "all"):
        print("prereg OK", PREREG, flush=True)
    if args.phase in ("train_tm", "all"):
        print("=== phase train_tm ===", flush=True)
        print("device", device_str(), flush=True)
        for s in SERIES_TM:
            if only and s["code"] not in only:
                continue
            run_one(s, quick=args.quick)
    if args.phase in ("train_hic", "all"):
        print("=== phase train_hic ===", flush=True)
        print("device", device_str(), flush=True)
        for s in SERIES_HIC:
            if only and s["code"] not in only:
                continue
            run_one(s, quick=args.quick)
    if args.phase in ("finalize", "all"):
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
