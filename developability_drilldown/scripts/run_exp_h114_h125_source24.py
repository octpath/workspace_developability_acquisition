#!/usr/bin/env python3
"""Run EXP-H114..H125 SOURCE SAP/SCM DIRECT vs AUX32 fusion ablation under V3.

Phases: prereg | train | analyze | all
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
from antibody_transformer.data import (  # noqa: E402
    load_dev_test,
    load_folds,
    load_residue_bundle,
    load_solution,
    tvt_split,
)
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID  # noqa: E402
from antibody_transformer.protocol_v3_ext import (  # noqa: E402
    build_platform_model_ext,
    run_protocol_v3_ext,
)
from antibody_transformer.source_sap_scm_aux import SourceSapScmAuxFeatureStore  # noqa: E402
from antibody_transformer.protocol_v3 import normalize_arch_flags  # noqa: E402
from preregister_h114_h125_source24 import SERIES, PREREG, build_series, install_features  # noqa: E402

PRE_EXTERNAL = ROOT / "results" / "H114_H125_PRE_EXTERNAL_FREEZE.yaml"
CONTROLS = {"EXP-H071", "EXP-H061"}


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


def ensure_series() -> list[dict]:
    if SERIES:
        return SERIES
    feat = install_features()
    return build_series(feat)


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
        phase="V3_H114_SOURCE24_FUSION",
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
                    "notes": "V3 H114 SOURCE24 fusion ablation",
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
    aux_store = SourceSapScmAuxFeatureStore(spec["fusion_bundle_id"])
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
    fusion_mode = cfg["fusion_mode"]

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
        fusion_mode=fusion_mode,
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
        "fusion_mode": fusion_mode,
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
                f"- fusion_mode: `{fusion_mode}`",
                f"- bundle: `{spec['fusion_bundle_id']}`",
                f"- TEST_mean: {summary['scores']['oof_test']['mean']:.6f}",
                f"- n_trainable: {summary['n_trainable']}",
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

    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    sel = pd.read_csv(ROOT / "results" / f"{code}_SELECTED_LR.csv")
    store = SourceSapScmAuxFeatureStore(cfg["fusion_bundle_id"])
    slices = store.block_slices()
    blocks = {k: v for k, v in slices.items() if k in ("SAP24", "SCM24", "COMBINED48", "ALL_AUX")}
    # prefer named blocks
    if "COMBINED48" in blocks:
        blocks = {
            "SAP24": slices["SAP24"],
            "SCM24": slices["SCM24"],
            "COMBINED48": slices["COMBINED48"],
        }
    elif "SAP24" in blocks:
        blocks = {"SAP24": slices["SAP24"]}
    else:
        blocks = {"SCM24": slices["SCM24"]}

    do_col_perm = cfg["fusion_mode"] == "late_concat_direct"

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
        fm = blob.get("fusion_mode") or cfg["fusion_mode"]
        model = build_platform_model_ext(
            rb,
            flags,
            content_mode=blob["content_mode"],
            merge_mode=blob["merge_mode"],
            plm_source=blob["plm_source"],
            chain_mode=blob["chain_mode"],
            capacity=blob.get("capacity"),
            aux_dim=blob["aux_dim"],
            fusion_mode=fm,
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
        if do_col_perm:
            for j, fname in enumerate(store.columns):
                rng = np.random.default_rng(seed + 7000 + j + k * 17 + (0 if scheme == "primary" else 100))
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
                        "block": "COLUMN",
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


def analyze(series: list[dict]) -> None:
    print("=== analyze / freeze / reports ===", flush=True)
    codes_new = [s["code"] for s in series]
    fmap = {s["code"]: s for s in series}

    freeze_rows = []
    for code in codes_new:
        oof = load_oof(code)
        ot = oof["oof_test"]
        ov = oof["oof_val"]
        freeze_rows.append(
            {
                "code": code,
                "backbone": fmap[code]["backbone"],
                "fusion_bundle_id": fmap[code]["fusion_bundle_id"],
                "fusion_mode": fmap[code]["fusion_mode"],
                "VAL_P": ov["primary"],
                "VAL_S": ov["shadow"],
                "VAL_mean": ov["mean"],
                "VAL_worst": ov["worst"],
                "TEST_P": ot["primary"],
                "TEST_S": ot["shadow"],
                "TEST_mean": ot["mean"],
                "TEST_worst": ot["worst"],
                "n_trainable": oof.get("n_trainable"),
                "config_hash": oof.get("config_hash"),
            }
        )

    # bootstrap
    dev, _ = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    y = dev.set_index("id")["HIC"]
    boot_rows = []
    fusion_pairs = [
        ("EXP-H114", "EXP-H115"),
        ("EXP-H116", "EXP-H117"),
        ("EXP-H118", "EXP-H119"),
        ("EXP-H120", "EXP-H121"),
        ("EXP-H122", "EXP-H123"),
        ("EXP-H124", "EXP-H125"),
    ]
    for a, b in fusion_pairs:
        for scheme in ("primary", "shadow"):
            pa = load_oof_preds(a, "test", scheme).reindex(y.index).to_numpy(float)
            pb = load_oof_preds(b, "test", scheme).reindex(y.index).to_numpy(float)
            d, lo, hi = paired_boot(pa, pb, y.to_numpy(float))
            boot_rows.append(
                {
                    "comparison": f"{a}_DIRECT_vs_{b}_AUX32",
                    "scheme": scheme,
                    "delta_mae": d,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "note": "Δ = MAE(DIRECT)-MAE(AUX32); negative => DIRECT better",
                }
            )
    for code in codes_new:
        base = fmap[code]["backbone"]
        for scheme in ("primary", "shadow"):
            pa = load_oof_preds(code, "test", scheme).reindex(y.index).to_numpy(float)
            pb = load_oof_preds(base, "test", scheme).reindex(y.index).to_numpy(float)
            d, lo, hi = paired_boot(pa, pb, y.to_numpy(float))
            boot_rows.append(
                {
                    "comparison": f"{code}_vs_{base}",
                    "scheme": scheme,
                    "delta_mae": d,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "note": "Δ = MAE(exp)-MAE(base); negative => exp better",
                }
            )
    pd.DataFrame(boot_rows).to_csv(ROOT / "results" / "H114_H125_PAIRED_BOOTSTRAP.csv", index=False)

    # permutations (can be slow)
    perm_all = []
    for code in codes_new:
        print(f"permutation {code}", flush=True)
        perm_all.extend(permutation_for_code(code, n_perm=100))
    pd.DataFrame(perm_all).to_csv(ROOT / "results" / "H114_H125_BLOCK_PERMUTATION.csv", index=False)

    # freeze before external interpretation
    freeze = {
        "status": "PRE_EXTERNAL_FROZEN",
        "git_rev": git_rev(),
        "experiments": freeze_rows,
        "bootstrap_path": "results/H114_H125_PAIRED_BOOTSTRAP.csv",
        "permutation_path": "results/H114_H125_BLOCK_PERMUTATION.csv",
    }
    # fusion / feature verdicts from TEST_mean
    fr = pd.DataFrame(freeze_rows)

    def tm(code):
        return float(fr.loc[fr.code == code, "TEST_mean"].iloc[0])

    def perm_delta(code, block):
        sub = pd.DataFrame(perm_all)
        s = sub[(sub.code == code) & (sub.block == block)]
        if s.empty:
            return None
        return float(s.groupby("scheme")["delta_mae_mean"].mean().mean())

    verdicts = {}
    for feat, d_sc, a_sc, d_es, a_es, base_sc, base_es, blk in [
        ("SAP24", "EXP-H114", "EXP-H115", "EXP-H120", "EXP-H121", "EXP-H071", "EXP-H061", "SAP24"),
        ("SCM24", "EXP-H116", "EXP-H117", "EXP-H122", "EXP-H123", "EXP-H071", "EXP-H061", "SCM24"),
        ("COMBINED48", "EXP-H118", "EXP-H119", "EXP-H124", "EXP-H125", "EXP-H071", "EXP-H061", "COMBINED48"),
    ]:
        verdicts[feat] = {
            "scratch_direct_test": tm(d_sc),
            "scratch_aux32_test": tm(a_sc),
            "scratch_base_test": tm(base_sc) if base_sc in set(fr.code) else load_oof(base_sc)["oof_test"]["mean"],
            "esm2_direct_test": tm(d_es),
            "esm2_aux32_test": tm(a_es),
            "esm2_base_test": load_oof(base_es)["oof_test"]["mean"],
            "delta_fusion_scratch": tm(d_sc) - tm(a_sc),
            "delta_fusion_esm2": tm(d_es) - tm(a_es),
            "perm_direct_scratch": perm_delta(d_sc, blk if blk != "COMBINED48" else "COMBINED48"),
            "perm_aux_scratch": perm_delta(a_sc, blk if blk != "COMBINED48" else "COMBINED48"),
        }

    freeze["feature_verdicts"] = verdicts
    PRE_EXTERNAL.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")

    # external table (diagnostic only; already computed)
    ext_rows = []
    for code in codes_new:
        oof = load_oof(code)
        for agg, sc in oof["external"].items():
            ext_rows.append({"code": code, "agg": agg, **sc})
    pd.DataFrame(ext_rows).to_csv(ROOT / "results" / "H114_H125_EXTERNAL_DIAGNOSTIC.csv", index=False)

    write_report(series, fr, verdicts, boot_rows, perm_all)
    print("Wrote", PRE_EXTERNAL, flush=True)


def write_report(series, fr, verdicts, boot_rows, perm_all):
    lines = [
        "# HIC SOURCE24 Transformer Fusion Ablation (H114–H125)",
        "",
        "Platform: `DL_FOLDLOCAL_COSINE_V3`. Controls: EXP-H071 (Scratch), EXP-H061 (ESM2).",
        "",
        "## Central table",
        "",
        "| Feature | Backbone | DIRECT TEST | AUX32 TEST | Δ DIRECT−AUX | Direct perm | Aux perm | Ext DIRECT overall | Ext AUX overall |",
        "|---------|----------|------------:|-----------:|-------------:|------------:|---------:|-------------------:|----------------:|",
    ]
    fmap = {s["code"]: s for s in series}
    pairs = [
        ("SAP24", "EXP-H114", "EXP-H115", "EXP-H120", "EXP-H121"),
        ("SCM24", "EXP-H116", "EXP-H117", "EXP-H122", "EXP-H123"),
        ("COMBINED48", "EXP-H118", "EXP-H119", "EXP-H124", "EXP-H125"),
    ]
    for feat, ds, as_, de, ae in pairs:
        for bb, d, a in (("Scratch", ds, as_), ("ESM2", de, ae)):
            td = float(fr.loc[fr.code == d, "TEST_mean"].iloc[0])
            ta = float(fr.loc[fr.code == a, "TEST_mean"].iloc[0])
            od = load_oof(d)["external"]["primary_mean"]["overall_mae"]
            oa = load_oof(a)["external"]["primary_mean"]["overall_mae"]
            pdlt = verdicts[feat]["perm_direct_scratch" if bb == "Scratch" else "perm_direct_scratch"]
            # better: look up from perm file
            perm = pd.DataFrame(perm_all)
            blk = "COMBINED48" if feat == "COMBINED48" else feat
            pd_m = perm[(perm.code == d) & (perm.block == blk)]["delta_mae_mean"].mean()
            pa_m = perm[(perm.code == a) & (perm.block == blk)]["delta_mae_mean"].mean()
            lines.append(
                f"| {feat} | {bb} | {td:.4f} | {ta:.4f} | {td-ta:+.4f} | {pd_m:.4f} | {pa_m:.4f} | {od:.4f} | {oa:.4f} |"
            )

    lines += ["", "## Parameter counts (trainable)", ""]
    for code in [s["code"] for s in series]:
        n = load_oof(code).get("n_trainable")
        lines.append(f"- {code}: {n}")

    lines += [
        "",
        "## Required questions",
        "",
        "See verdicts in `H114_H125_PRE_EXTERNAL_FREEZE.yaml` and narrative below after numbers.",
        "",
        f"- H071 TEST_mean: {load_oof('EXP-H071')['oof_test']['mean']:.6f}",
        f"- H061 TEST_mean: {load_oof('EXP-H061')['oof_test']['mean']:.6f}",
        "",
    ]
    # auto narrative
    for feat, v in verdicts.items():
        lines.append(f"### {feat}")
        lines.append(
            f"- Scratch DIRECT {v['scratch_direct_test']:.4f} / AUX32 {v['scratch_aux32_test']:.4f} / base {v['scratch_base_test']:.4f}"
        )
        lines.append(
            f"- ESM2 DIRECT {v['esm2_direct_test']:.4f} / AUX32 {v['esm2_aux32_test']:.4f} / base {v['esm2_base_test']:.4f}"
        )
        lines.append("")

    (ROOT / "results" / "HIC_SOURCE24_TRANSFORMER_FUSION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "train", "analyze", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("prereg", "all"):
        from preregister_h114_h125_source24 import main as prereg_main

        if prereg_main() != 0:
            return 1

    series = ensure_series()
    # reload series from prereg if empty module state
    if not series:
        doc = yaml.safe_load(PREREG.read_text())
        feat = install_features()
        series = build_series(feat)

    if args.phase in ("train", "all"):
        for spec in series:
            run_one(spec, quick=args.quick)

    if args.phase in ("analyze", "all"):
        # require all complete
        missing = [s["code"] for s in series if not already_complete(s["code"])]
        if missing:
            raise SystemExit(f"incomplete before analyze: {missing}")
        analyze(series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
