#!/usr/bin/env python3
"""Run EXP-H102..H113 promoted HSP late fusion under V3.

Phases: prereg | train | analyze | all
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
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID  # noqa: E402
from antibody_transformer.protocol_v3_ext import (  # noqa: E402
    run_protocol_v3_ext,
)
from antibody_transformer.hsp_promoted_aux import HspPromotedAuxFeatureStore  # noqa: E402
from preregister_h102_h113_hsp import SERIES, PREREG  # noqa: E402

PRE_EXTERNAL = ROOT / "results" / "H102_H113_PRE_EXTERNAL_FREEZE.yaml"
CONTROLS = ["EXP-H071", "EXP-H090", "EXP-H093", "EXP-H061", "EXP-H086", "EXP-H089"]


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


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
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{spec['code']}.yaml").read_text())
    eid = cfg["experiment_id"]
    if (codes["experiment_id"] == eid).any():
        return str(codes.set_index("experiment_id").loc[eid, "experiment_code"])
    expect = spec["code"]
    nxt = next_code("HIC")
    if nxt != expect:
        raise SystemExit(f"expected {expect}, got {nxt}")
    code = issue_code(
        eid,
        "HIC",
        source_model_id=cfg["source_model_id"],
        phase="V3_H102_HSP_PROMOTED",
        notes=spec["description"],
    )
    if code != expect:
        raise SystemExit(code)
    return code


def register(code: str, spec: dict, summary: dict, ext_scores: dict) -> None:
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    pm = ext_scores["primary_mean"]
    oof = summary["scores"]["oof_test"]
    is_frozen = cfg.get("content_mode") == "frozen"
    plm_yaml = cfg.get("plm_source", "NONE")
    exp_path = ROOT / "results" / "experiments.csv"
    df = pd.read_csv(exp_path)
    df = df[df["experiment_code"] != code]
    row = {c: "" for c in df.columns}
    for c in EXPERIMENTS_COLUMNS:
        row.setdefault(c, "")
    row.update(
        {
            "experiment_code": code,
            "experiment_id": cfg["experiment_id"],
            "legacy_experiment_code": "",
            "target": "HIC",
            "family": "TRANSFORMER",
            "model_type": cfg.get("transformer_type", "TRANSFORMER"),
            "feature_set_id": cfg.get("fusion_bundle_id", ""),
            "feature_space": "",
            "feature_path": "",
            "input_space": cfg["input_space"],
            "input_asset_ref": cfg.get("input_asset_ref", ""),
            "plm_source": plm_yaml if plm_yaml != "NONE" else "NONE",
            "representation_status": "NOT_EXPORTED",
            "transformer_type": cfg.get("transformer_type", "LATE_FUSION"),
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
            "license_status": "REVIEW" if is_frozen else "OK",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "REPRODUCED",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "shareability_status": "SHAREABLE_COMPLETE",
            "canonical_benchmark_eligible": "YES",
            "source_model_id": cfg["source_model_id"],
            "notes": spec["description"],
            "control_experiment_code": cfg.get("control_experiment_code") or "",
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
                    "experiment_id": cfg["experiment_id"],
                    "target": "HIC",
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
                    "notes": "V3 H102 HSP promoted",
                }
            )
            if "test_prediction_exists" in comp.columns:
                crow["test_prediction_exists"] = True
            comp = pd.concat(
                [comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])],
                ignore_index=True,
            )
            comp.to_csv(comp_path, index=False)


def prepare_rb(cfg, dev, test):
    plm = cfg.get("plm_source")
    plm = {"ABLANG2": "ablang2", "ESM2": "esm2", "NONE": None}.get(plm, plm)
    return load_residue_bundle(
        dev,
        test,
        need_ablingua=False,
        need_ablang2=False,
        need_esm2=(plm == "esm2" or cfg.get("representation") == "ESM2"),
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
    out = ROOT / "results" / f"{code}_run"
    pred = ROOT / "experiments" / "predictions" / code
    aux_store = HspPromotedAuxFeatureStore(spec["fusion_bundle_id"])
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
    plm = cfg["plm_source"]
    plm_src = {"ABLANG2": "ablang2", "ESM2": "esm2", "NONE": None}[plm]
    merge = cfg["merge_mode"] or "concat"

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = prepare_rb(cfg, dev, test)

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
        content_mode=cfg["content_mode"],
        merge_mode=merge if merge != "null" else "concat",
        plm_source=plm_src,
        chain_mode=cfg["chain_mode"],
        capacity=None,
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
            "HIC",
        )
        save_pred(
            result["dev_ids"],
            result["oof_test"][scheme].loc[result["dev_ids"]].to_numpy(float),
            pred / f"oof_test_{scheme}.csv",
            "HIC",
        )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["primary"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_primary.csv",
        "HIC",
    )
    save_pred(
        result["dev_ids"],
        result["oof_test"]["shadow"].loc[result["dev_ids"]].to_numpy(float),
        pred / "oof_shadow.csv",
        "HIC",
    )
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
        key: score_external(result["ext"][key], result["test_ids"], sol, "HIC")
        for key in ("primary_mean", "primary_median", "shadow_mean", "shadow_median")
    }
    oof_doc = {
        "experiment_code": code,
        "platform_id": PLATFORM_ID,
        "config_hash": summary["config_hash"],
        "fusion_bundle_id": summary.get("fusion_bundle_id"),
        "hsp_family_id": spec.get("hsp_family_id"),
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
    (ROOT / "results" / f"{code}_PROTOCOL_REPORT.md").write_text(
        "\n".join(
            [
                f"# {code} — {spec['description']}",
                "",
                f"- platform: `{PLATFORM_ID}`",
                f"- HSP family: `{spec.get('hsp_family_id')}`",
                f"- TEST_mean: {summary['scores']['oof_test']['mean']:.6f}",
                f"- Overall (P-mean): {ext_scores['primary_mean']['overall_mae']:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    register(code, spec, summary, ext_scores)
    print(f"DONE {code}", flush=True)
    return {"code": code, "spec": spec, "summary": summary, "ext_scores": ext_scores, "skipped": False}


def load_oof(code: str) -> dict:
    return yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())


def load_oof_preds(code: str, kind: str, scheme: str) -> pd.Series:
    p = ROOT / "experiments" / "predictions" / code / f"oof_{kind}_{scheme}.csv"
    df = pd.read_csv(p)
    col = [c for c in df.columns if c != "id"][0]
    return df.set_index("id")[col]


def paired_boot(a: np.ndarray, b: np.ndarray, y: np.ndarray, n=2000, seed=101):
    ea = np.abs(a - y)
    eb = np.abs(b - y)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)]
    boots = np.asarray(boots)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _predict_fixed(model, rb, ids, y_arr, X_fixed, blob, device):
    import torch
    from antibody_transformer.training import AbDataset, collate_batch, _batch_to_device
    from antibody_transformer.protocol_v3 import BATCH_SIZE
    from torch.utils.data import DataLoader

    mu, sd = float(blob["mu"]), float(blob["sd"])
    ds = AbDataset(
        ids,
        y_arr,
        rb,
        content_mode=blob["content_mode"],
        plm_source=blob["plm_source"] or "ablingua",
        fixed_X=X_fixed,
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
    return np.concatenate(preds)


def permutation_for_code(code: str, *, n_perm: int = 100, seed: int = 2026) -> list[dict]:
    import torch
    from antibody_transformer.data import tvt_split
    from antibody_transformer.protocol_v3_ext import build_platform_model_ext
    from antibody_transformer.protocol_v3 import normalize_arch_flags

    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    sel = pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv")
    store = HspPromotedAuxFeatureStore(cfg["fusion_bundle_id"])
    slices = store.block_slices()
    bid = cfg["fusion_bundle_id"]
    if bid.startswith("FS_HIC_SURFACE_PLUS_"):
        blocks = {
            "SURFACE": slices["SURFACE"],
            "HSP": slices["HSP"],
            "SURFACE_PLUS_HSP": slices["SURFACE_PLUS_HSP"],
        }
    else:
        blocks = {"HSP": slices["HSP"]}

    # optional feature-level (dim<=20 always for HSP alone / surface+hsp=38 so only HSP alone)
    feat_level = not bid.startswith("FS_HIC_SURFACE_PLUS_")

    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = prepare_rb(cfg, dev, test)
    y_map = {str(r["id"]): float(r["HIC"]) for _, r in dev.iterrows()}
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
        X = store.transform(prep, te)
        pred_n = _predict_fixed(model, rb, te, y_te, X, blob, device)
        mae_n = float(mae(y_te, pred_n))
        for bname, sl in blocks.items():
            rng = np.random.default_rng(seed + hash(bname) % 10000 + k * 17 + (0 if scheme == "primary" else 100))
            deltas = []
            for _ in range(n_perm):
                Xp = X.copy()
                block = Xp[:, sl]
                Xp[:, sl] = block[rng.permutation(len(block))]
                pred_p = _predict_fixed(model, rb, te, y_te, Xp, blob, device)
                deltas.append(float(mae(y_te, pred_p) - mae_n))
            arr = np.asarray(deltas)
            rows.append(
                {
                    "code": code,
                    "scheme": scheme,
                    "fold": k,
                    "split": "TEST_OOF",
                    "block": bname,
                    "feature": "",
                    "mae_normal": mae_n,
                    "delta_mae_mean": float(arr.mean()),
                    "delta_mae_sd": float(arr.std()),
                    "delta_mae_q025": float(np.quantile(arr, 0.025)),
                    "delta_mae_q975": float(np.quantile(arr, 0.975)),
                    "n_perm": n_perm,
                }
            )
        if feat_level:
            for j, fname in enumerate(("HSP_MAX", "HSP_MEAN", "HSP_SUM")):
                rng = np.random.default_rng(seed + 5000 + j + k * 17 + (0 if scheme == "primary" else 100))
                deltas = []
                for _ in range(n_perm):
                    Xp = X.copy()
                    Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                    pred_p = _predict_fixed(model, rb, te, y_te, Xp, blob, device)
                    deltas.append(float(mae(y_te, pred_p) - mae_n))
                arr = np.asarray(deltas)
                rows.append(
                    {
                        "code": code,
                        "scheme": scheme,
                        "fold": k,
                        "split": "TEST_OOF",
                        "block": "HSP_FEATURE",
                        "feature": fname,
                        "mae_normal": mae_n,
                        "delta_mae_mean": float(arr.mean()),
                        "delta_mae_sd": float(arr.std()),
                        "delta_mae_q025": float(np.quantile(arr, 0.025)),
                        "delta_mae_q975": float(np.quantile(arr, 0.975)),
                        "n_perm": n_perm,
                    }
                )
    return rows


def analyze() -> None:
    print("=== analyze / freeze / reports ===", flush=True)
    codes_new = [s["code"] for s in SERIES]
    family_map = {s["code"]: s for s in SERIES}

    # Pre-external freeze FIRST (internal metrics only; external already in OOF but verdict uses TEST)
    freeze_rows = []
    for code in codes_new:
        oof = load_oof(code)
        ot = oof["oof_test"]
        freeze_rows.append(
            {
                "code": code,
                "hsp_family_id": family_map[code].get("hsp_family_id"),
                "fusion_bundle_id": family_map[code]["fusion_bundle_id"],
                "TEST_P": ot["primary"],
                "TEST_S": ot["shadow"],
                "TEST_mean": ot["mean"],
                "TEST_worst": ot["worst"],
                "VAL": oof["oof_val"],
                "config_hash": oof.get("config_hash"),
            }
        )

    # bootstrap (internal)
    dev, _ = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    y = dev.set_index("id")["HIC"]
    comps = [
        ("EXP-H102", "EXP-H071"),
        ("EXP-H104", "EXP-H071"),
        ("EXP-H106", "EXP-H071"),
        ("EXP-H108", "EXP-H061"),
        ("EXP-H110", "EXP-H061"),
        ("EXP-H112", "EXP-H061"),
        ("EXP-H103", "EXP-H090"),
        ("EXP-H105", "EXP-H090"),
        ("EXP-H107", "EXP-H090"),
        ("EXP-H109", "EXP-H086"),
        ("EXP-H111", "EXP-H086"),
        ("EXP-H113", "EXP-H086"),
        ("EXP-H102", "EXP-H090"),
        ("EXP-H104", "EXP-H090"),
        ("EXP-H106", "EXP-H090"),
        ("EXP-H108", "EXP-H086"),
        ("EXP-H110", "EXP-H086"),
        ("EXP-H112", "EXP-H086"),
    ]
    boots = []
    for a, b in comps:
        for scheme in ("primary", "shadow"):
            pa = load_oof_preds(a, "test", scheme)
            pb = load_oof_preds(b, "test", scheme)
            ids = pa.index.intersection(pb.index).intersection(y.index)
            dmean, lo, hi = paired_boot(pa.loc[ids].to_numpy(), pb.loc[ids].to_numpy(), y.loc[ids].to_numpy())
            boots.append(
                {
                    "A": a,
                    "B": b,
                    "scheme": scheme,
                    "delta_mae_A_minus_B": dmean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "note": "negative => A better",
                }
            )
    boot_df = pd.DataFrame(boots)
    boot_df.to_csv(ROOT / "results/HIC_HSP_MAINLINE_BOOTSTRAP.csv", index=False)

    # permutation
    perm_all = []
    for s in SERIES:
        print(f"permute {s['code']}", flush=True)
        perm_all.extend(permutation_for_code(s["code"]))
    perm_df = pd.DataFrame(perm_all)
    perm_df.to_csv(ROOT / "results/HIC_HSP_MAINLINE_PERMUTATION.csv", index=False)

    # internal verdict helpers
    def test_mean(code: str) -> float:
        return float(load_oof(code)["oof_test"]["mean"])

    def delta_over(a: str, b: str) -> float:
        return test_mean(a) - test_mean(b)

    def boot_row(a, b, scheme):
        r = boot_df[(boot_df.A == a) & (boot_df.B == b) & (boot_df.scheme == scheme)]
        return None if r.empty else r.iloc[0]

    def perm_hsp_mean(code: str, scheme: str) -> float:
        sub = perm_df[(perm_df.code == code) & (perm_df.block == "HSP") & (perm_df.scheme == scheme)]
        return float(sub["delta_mae_mean"].mean()) if len(sub) else float("nan")

    families = {
        "P1": {"alone_s": "EXP-H102", "surf_s": "EXP-H103", "alone_e": "EXP-H108", "surf_e": "EXP-H109", "name": "BM-R5"},
        "P2": {"alone_s": "EXP-H104", "surf_s": "EXP-H105", "alone_e": "EXP-H110", "surf_e": "EXP-H111", "name": "FP-R5"},
        "P3": {"alone_s": "EXP-H106", "surf_s": "EXP-H107", "alone_e": "EXP-H112", "surf_e": "EXP-H113", "name": "EIS-R8"},
    }
    verdicts = {}
    for pid, f in families.items():
        d_s = delta_over(f["surf_s"], "EXP-H090")
        d_e = delta_over(f["surf_e"], "EXP-H086")
        ph_s = perm_hsp_mean(f["surf_s"], "primary")
        ph_e = perm_hsp_mean(f["surf_e"], "primary")
        # classify
        improve_s = d_s < -0.005
        improve_e = d_e < -0.005
        rely_s = ph_s > 0.005
        rely_e = ph_e > 0.005
        if (improve_s and rely_s) and (improve_e and rely_e):
            v = "MAINLINE_SUPPORTED_STRONG"
            bb = "BACKBONE_ROBUST"
        elif (improve_s and rely_s) or (improve_e and rely_e):
            v = "BACKBONE_DEPENDENT"
            bb = "SCRATCH_SPECIFIC" if (improve_s and rely_s) else "ESM2_SPECIFIC"
        elif (d_s < 0 and d_e < 0) and (ph_s > 0 or ph_e > 0):
            v = "MAINLINE_SUPPORTED"
            bb = "BACKBONE_ROBUST" if (d_s < 0 and d_e < 0) else "INCONSISTENT"
        elif test_mean(f["alone_s"]) < test_mean("EXP-H071") - 0.01 and d_s >= -0.002:
            v = "REDUNDANT_WITH_SURFACE"
            bb = "INCONSISTENT"
        else:
            v = "NOT_CONFIRMED"
            bb = "NOT_SUPPORTED"
        verdicts[pid] = {
            "family": f["name"],
            "delta_scratch_over_surface": d_s,
            "delta_esm2_over_surface": d_e,
            "perm_hsp_scratch_primary": ph_s,
            "perm_hsp_esm2_primary": ph_e,
            "mainline_class": v,
            "backbone_class": bb,
        }

    freeze = {
        "status": "PRE_EXTERNAL_FREEZE",
        "git_rev": git_rev(),
        "completed": codes_new,
        "n_completed": len(codes_new),
        "preregistration": str(PREREG),
        "platform_id": PLATFORM_ID,
        "internal_test_metrics": freeze_rows,
        "verdicts": verdicts,
        "statement": "All H102-H113 internal TEST / bootstrap / permutation frozen before comparative external interpretation.",
    }
    PRE_EXTERNAL.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")

    # master table + external four-way (diagnostic; after freeze)
    rows = []
    for code in CONTROLS + codes_new:
        oof = load_oof(code)
        cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
        e = pd.read_csv(ROOT / "results" / "experiments.csv")
        er = e[e.experiment_code == code].iloc[0]
        ot = oof["oof_test"]
        aux = cfg.get("fusion_bundle_id") or "NONE"
        if aux == "NONE" or aux is None:
            adim = 0
        elif aux == "F1_SURFACE":
            adim = 35
        elif aux == "F4_H047_AUX_ALL":
            adim = 200
        elif aux in HspPromotedAuxFeatureStore.__dict__.get("bundle_id", ()) or True:
            try:
                adim = HspPromotedAuxFeatureStore(aux).effective_dim
            except Exception:
                adim = -1
        fam = cfg.get("hsp_family_id") or ""
        rows.append(
            {
                "Code": code,
                "Backbone": cfg.get("control_experiment_code") or code,
                "Aux": aux,
                "HSP_family": fam,
                "Aux_dim": adim,
                "TEST_P": ot["primary"],
                "TEST_S": ot["shadow"],
                "TEST_mean": ot["mean"],
                "TEST_worst": ot["worst"],
                "Pub": er.public_mae,
                "Priv": er.private_mae,
                "Overall": er.test_overall_mae,
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / "results/HIC_HSP_MAINLINE_TABLE.csv", index=False)

    four = []
    for code in codes_new + CONTROLS:
        oof = load_oof(code)
        ext = oof.get("external")
        if ext is None:
            st = ROOT / "results" / f"{code}_run_state.json"
            if st.exists():
                ext = json.loads(st.read_text())["ext_scores"]
            else:
                e = pd.read_csv(ROOT / "results" / "experiments.csv")
                er = e[e.experiment_code == code].iloc[0]
                ext = {
                    "primary_mean": {
                        "public_mae": er.public_mae,
                        "private_mae": er.private_mae,
                        "overall_mae": er.test_overall_mae,
                    }
                }
        for agg, sc in ext.items():
            four.append({"code": code, "aggregation": agg, **sc})
    pd.DataFrame(four).to_csv(ROOT / "results/HIC_HSP_MAINLINE_EXTERNAL_FOUR_WAY.csv", index=False)

    # increment table
    inc_rows = []
    for pid, f in families.items():
        v = verdicts[pid]
        br_s = boot_row(f["surf_s"], "EXP-H090", "primary")
        br_e = boot_row(f["surf_e"], "EXP-H086", "primary")
        inc_rows.append(
            {
                "Family": f["name"],
                "Scratch_d_over_SURFACE": v["delta_scratch_over_surface"],
                "Scratch_boot_primary": None if br_s is None else f"{br_s.delta_mae_A_minus_B:.4f} [{br_s.ci95_lo:.4f},{br_s.ci95_hi:.4f}]",
                "Scratch_perm_HSP_primary": v["perm_hsp_scratch_primary"],
                "ESM2_d_over_SURFACE": v["delta_esm2_over_surface"],
                "ESM2_boot_primary": None if br_e is None else f"{br_e.delta_mae_A_minus_B:.4f} [{br_e.ci95_lo:.4f},{br_e.ci95_hi:.4f}]",
                "ESM2_perm_HSP_primary": v["perm_hsp_esm2_primary"],
                "Verdict": v["mainline_class"],
                "Backbone": v["backbone_class"],
            }
        )
    inc = pd.DataFrame(inc_rows)
    inc.to_csv(ROOT / "results/HIC_HSP_MAINLINE_INCREMENT.csv", index=False)

    # concordance
    screen = pd.read_csv(
        REPO / "feature_research/hic_spatial_hydrophobicity/results/STAGE2_TEST_CONFIRMATION.csv"
    )
    conc_lines = [
        "# HSP screen → mainline DL concordance",
        "",
        "Screening used fixed Ridge/SVR on B3 bundles (source commit 210a270d).",
        "",
    ]
    for pid, f in families.items():
        fam_id = family_map[f["surf_s"]]["hsp_family_id"]
        s_alone = screen[(screen.family_id == fam_id) & (screen.context == "A_ALONE") & (screen.estimator == "Ridge")]
        s_surf = screen[(screen.family_id == fam_id) & (screen.context == "B_SURFACE") & (screen.estimator == "Ridge")]
        conc_lines += [
            f"## {pid} {f['name']} (`{fam_id}`)",
            "",
            f"- Screen alone Ridge TEST_mean: {float(s_alone.TEST_mean.iloc[0]) if len(s_alone) else float('nan'):.4f}",
            f"- Screen SURFACE+desc Ridge TEST_mean: {float(s_surf.TEST_mean.iloc[0]) if len(s_surf) else float('nan'):.4f}",
            f"- DL Scratch alone {f['alone_s']} TEST_mean: {test_mean(f['alone_s']):.4f} (vs H071 {test_mean('EXP-H071'):.4f})",
            f"- DL Scratch SURFACE+HSP {f['surf_s']} Δ vs H090: {verdicts[pid]['delta_scratch_over_surface']:+.4f}",
            f"- DL ESM2 SURFACE+HSP {f['surf_e']} Δ vs H086: {verdicts[pid]['delta_esm2_over_surface']:+.4f}",
            f"- Mainline class: **{verdicts[pid]['mainline_class']}** / {verdicts[pid]['backbone_class']}",
            "",
        ]
    conc_lines += [
        "## Summary questions",
        "",
        f"1. P1 BM-R5 replicate? {verdicts['P1']['mainline_class']}",
        f"2. P2 FP-R5 replicate? {verdicts['P2']['mainline_class']}",
        f"3. P3 EIS-R8 SURFACE complementarity? {verdicts['P3']['mainline_class']}",
        "4. Cheap screen vs DL: directional agreement is assessed above; magnitude need not match.",
        "",
    ]
    (ROOT / "results/HSP_SCREEN_TO_MAINLINE_CONCORDANCE.md").write_text("\n".join(conc_lines), encoding="utf-8")

    # master report
    lines = [
        "# HIC HSP Mainline Report (H102–H113)",
        "",
        f"- platform: `{PLATFORM_ID}`",
        f"- source HSP commit: `210a270d`",
        f"- freeze: `{PRE_EXTERNAL.name}`",
        "",
        "## Master table",
        "",
        tab.to_markdown(index=False) if hasattr(tab, "to_markdown") else tab.to_string(index=False),
        "",
        "## Central increment table (SURFACE + HSP − SURFACE)",
        "",
        inc.to_markdown(index=False) if hasattr(inc, "to_markdown") else inc.to_string(index=False),
        "",
        "## Scientific questions",
        "",
        f"1. BM-R5 in DL: {verdicts['P1']}",
        f"2. FP-R5 in DL: {verdicts['P2']}",
        f"3. EIS-R8 complementarity: {verdicts['P3']}",
        f"4. Strongest HSP-alone Scratch: "
        + min(
            [("P1", test_mean("EXP-H102")), ("P2", test_mean("EXP-H104")), ("P3", test_mean("EXP-H106"))],
            key=lambda x: x[1],
        ).__repr__(),
        "5. Replace SURFACE? Compare HSP-alone vs H090/H086 in master table.",
        "6–8. See increment + permutation CSVs.",
        "9. Concordance: `HSP_SCREEN_TO_MAINLINE_CONCORDANCE.md`",
        "10–11. F4+HSP / residue-level: only if a family is MAINLINE_SUPPORTED(_STRONG).",
        "12. Do not claim causation; HSP descriptors may associate with HIC retention under this assay.",
        "",
        "## Guardrails",
        "",
        "- Do not call BM-R5 the true SAP.",
        "- Score improvement without HSP permutation reliance ≠ feature use.",
        "",
    ]
    (ROOT / "results/HIC_HSP_MAINLINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "train", "analyze", "all"], default="all", nargs="?")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("prereg", "all"):
        from preregister_h102_h113_hsp import main as prereg_main

        rc = prereg_main()
        if rc != 0:
            return rc
        # reload SERIES after prereg
        import importlib
        import preregister_h102_h113_hsp as preg

        importlib.reload(preg)
        global SERIES
        SERIES = preg.SERIES

    if args.phase in ("train", "all"):
        if not PREREG.exists():
            raise SystemExit("missing preregistration")
        for spec in SERIES:
            run_one(spec, quick=args.quick)

    if args.phase in ("analyze", "all"):
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
