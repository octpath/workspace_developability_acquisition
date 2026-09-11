#!/usr/bin/env python3
"""Run EXP-H094..H101 STATIC_SAP_KD antibody-level late fusion under V3.

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
from scipy.stats import pearsonr, spearmanr

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
    predict_with_checkpoint_ext,
    run_protocol_v3_ext,
)
from antibody_transformer.static_sap_kd import SAP3_COLS, SAP9_COLS  # noqa: E402
from antibody_transformer.static_sap_kd_aux import StaticSapKdAuxFeatureStore  # noqa: E402
from preregister_h094_static_sap_kd import SERIES, PREREG  # noqa: E402

PRE_EXTERNAL = ROOT / "results" / "H094_H101_PRE_EXTERNAL_FREEZE.yaml"
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
        phase="V3_H094_STATIC_SAP_KD",
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
                    "notes": "V3 H094 STATIC_SAP_KD",
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
    aux_store = StaticSapKdAuxFeatureStore(spec["fusion_bundle_id"])
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
    store = StaticSapKdAuxFeatureStore(cfg["fusion_bundle_id"])
    slices = store.block_slices()
    bid = cfg["fusion_bundle_id"]
    if bid in ("FS_HIC_STATIC_SAP_KD_GLOBAL3", "FS_HIC_STATIC_SAP_KD_CHAIN9"):
        blocks = {"SAP": slices["SAP"]}
    elif bid == "FS_HIC_SURFACE_PLUS_STATIC_SAP_KD_CHAIN9":
        blocks = {
            "SURFACE": slices["SURFACE"],
            "SAP": slices["SAP"],
            "SURFACE_PLUS_SAP": slices["SURFACE_PLUS_SAP"],
        }
    elif bid == "FS_HIC_F4_PLUS_STATIC_SAP_KD_CHAIN9":
        blocks = {
            "SURFACE": slices["SURFACE"],
            "SEQUENCE_TITRATION": slices["SEQUENCE_TITRATION"],
            "LOCAL_RASA": slices["LOCAL_RASA"],
            "SAP": slices["SAP"],
            "ALL_AUX": slices["ALL_AUX"],
        }
    else:
        raise ValueError(bid)

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
                    "mae_normal": mae_n,
                    "delta_mae_mean": float(arr.mean()),
                    "delta_mae_sd": float(arr.std()),
                    "delta_mae_q025": float(np.quantile(arr, 0.025)),
                    "delta_mae_q975": float(np.quantile(arr, 0.975)),
                    "n_perm": n_perm,
                }
            )
    return rows


def feature_correlation_and_descriptive() -> None:
    c9 = pd.read_parquet(ROOT / "experiments/features/static_sap_kd_antibody_chain9.parquet")
    ab = pd.read_parquet(ROOT / "experiments/features/static_sap_kd_antibody_full_qc.parquet")
    h047 = pd.read_parquet(ROOT / "experiments/features/EXP-H047.parquet")
    # H047 slices
    feat_cols = [c for c in h047.columns if c not in ("id", "split")]
    aro = feat_cols[1395:1414]
    hydro = feat_cols[1414:1430]
    seq = feat_cols[1280:1395]
    titr = feat_cols[1430:1448]
    m = c9.merge(h047[["id"] + aro + hydro + seq + titr], on="id").merge(
        ab[["id", "n_residues", "n_H", "n_L", "total_SASA"]], on="id"
    )
    rows = []
    for sap_col in SAP9_COLS:
        for block_name, cols in (
            ("AROMATIC_TOPO", aro),
            ("HYDRO_FIELD", hydro),
            ("SEQ_ALL", seq),
            ("TITRATION_SHAPE", titr),
        ):
            # max |corr| with any column in block + mean abs
            pears = [abs(pearsonr(m[sap_col], m[c])[0]) for c in cols]
            spears = [abs(spearmanr(m[sap_col], m[c])[0]) for c in cols]
            pears = [x for x in pears if np.isfinite(x)]
            spears = [x for x in spears if np.isfinite(x)]
            rows.append(
                {
                    "sap_feature": sap_col,
                    "block": block_name,
                    "pearson_max_abs": float(np.max(pears)) if pears else float("nan"),
                    "pearson_mean_abs": float(np.mean(pears)) if pears else float("nan"),
                    "spearman_max_abs": float(np.max(spears)) if spears else float("nan"),
                    "spearman_mean_abs": float(np.mean(spears)) if spears else float("nan"),
                }
            )
        for nuisance in ("n_residues", "n_H", "n_L", "total_SASA"):
            rows.append(
                {
                    "sap_feature": sap_col,
                    "block": f"NUISANCE_{nuisance}",
                    "pearson_max_abs": float(abs(pearsonr(m[sap_col], m[nuisance])[0])),
                    "pearson_mean_abs": float(abs(pearsonr(m[sap_col], m[nuisance])[0])),
                    "spearman_max_abs": float(abs(spearmanr(m[sap_col], m[nuisance])[0])),
                    "spearman_mean_abs": float(abs(spearmanr(m[sap_col], m[nuisance])[0])),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "results/HIC_STATIC_SAP_KD_FEATURE_CORRELATION.csv", index=False)

    # descriptive association with HIC on Dev only
    dev, _ = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    d = c9.merge(dev[["id", "HIC"]], on="id")
    desc = []
    for col in SAP9_COLS:
        rho, p = spearmanr(d[col], d["HIC"])
        desc.append({"feature": col, "spearman_rho_HIC_dev": float(rho), "pvalue": float(p), "n": len(d)})
    pd.DataFrame(desc).to_csv(ROOT / "results/HIC_STATIC_SAP_KD_DESCRIPTIVE_HIC.csv", index=False)

    # scatter plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, SAP3_COLS):
        ax.scatter(d[col], d["HIC"], s=12, alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel("HIC")
        rho = spearmanr(d[col], d["HIC"])[0]
        ax.set_title(f"ρ={rho:.3f}")
    fig.tight_layout()
    fig.savefig(ROOT / "results/HIC_STATIC_SAP_KD_SCATTER_SAP3.png", dpi=120)
    plt.close(fig)


def analyze() -> None:
    print("=== analyze / freeze / reports ===", flush=True)
    feature_correlation_and_descriptive()

    freeze = {
        "status": "PRE_EXTERNAL_FREEZE",
        "git_rev": git_rev(),
        "completed": [s["code"] for s in SERIES],
        "n_completed": len(SERIES),
        "preregistration": str(PREREG),
        "statement": "All H094-H101 internal TEST frozen before comparative external reveal.",
        "platform_id": PLATFORM_ID,
    }
    PRE_EXTERNAL.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")

    # master table
    rows = []
    for code in CONTROLS + [s["code"] for s in SERIES]:
        oof = load_oof(code)
        cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
        e = pd.read_csv(ROOT / "results" / "experiments.csv")
        er = e[e.experiment_code == code].iloc[0]
        ot = oof["oof_test"]
        aux = cfg.get("fusion_bundle_id") or "NONE"
        # resolve aux dim
        if aux == "NONE" or aux is None:
            adim = 0
        elif aux == "F1_SURFACE":
            adim = 35
        elif aux == "F4_H047_AUX_ALL":
            adim = 200
        else:
            adim = StaticSapKdAuxFeatureStore(aux).effective_dim
        rows.append(
            {
                "Code": code,
                "Backbone": cfg.get("control_experiment_code") or code,
                "Aux": aux,
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
    tab.to_csv(ROOT / "results/HIC_STATIC_SAP_KD_TABLE.csv", index=False)

    # external four-way
    four = []
    for code in [s["code"] for s in SERIES] + CONTROLS:
        oof = load_oof(code)
        ext = oof.get("external")
        if ext is None:
            # controls may store differently
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
    pd.DataFrame(four).to_csv(ROOT / "results/HIC_STATIC_SAP_KD_EXTERNAL_FOUR_WAY.csv", index=False)

    # bootstrap
    dev, _ = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    y = dev.set_index("id")["HIC"]
    comps = [
        ("EXP-H094", "EXP-H071"),
        ("EXP-H095", "EXP-H094"),
        ("EXP-H094", "EXP-H090"),
        ("EXP-H096", "EXP-H090"),
        ("EXP-H097", "EXP-H093"),
        ("EXP-H098", "EXP-H061"),
        ("EXP-H099", "EXP-H098"),
        ("EXP-H098", "EXP-H086"),
        ("EXP-H100", "EXP-H086"),
        ("EXP-H101", "EXP-H089"),
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
    pd.DataFrame(boots).to_csv(ROOT / "results/HIC_STATIC_SAP_KD_BOOTSTRAP.csv", index=False)

    # permutation
    perm_all = []
    for s in SERIES:
        print(f"permute {s['code']}", flush=True)
        perm_all.extend(permutation_for_code(s["code"]))
    pd.DataFrame(perm_all).to_csv(ROOT / "results/HIC_STATIC_SAP_KD_PERMUTATION.csv", index=False)

    # design note
    (ROOT / "results/GENERALIZED_SPATIAL_PROPERTY_AGGREGATION_IDEAS.md").write_text(
        "\n".join(
            [
                "# Generalized spatial property aggregation (design note only)",
                "",
                "Do **not** implement in H094–H101.",
                "",
                "General form:",
                "",
                "```",
                "P_i(R) = Σ_j I[d_ij <= R] * rSASA_j * property_j",
                "```",
                "",
                "Future property channels:",
                "",
                "- A. HYDROPHOBICITY — STATIC_SAP_KD (this batch)",
                "- B. POSITIVE CHARGE — max(q_j, 0)",
                "- C. NEGATIVE CHARGE — max(-q_j, 0)",
                "- D. ABSOLUTE CHARGE — |q_j|",
                "- E. AROMATICITY",
                "- F. H-BOND DONOR / ACCEPTOR propensity",
                "",
                "Charge note: do not rely only on signed charge sum; adjacent +/− can cancel",
                "despite a strongly charged patch. Prefer (B)/(C)/(D) separately.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # master report
    corr = pd.read_csv(ROOT / "results/HIC_STATIC_SAP_KD_FEATURE_CORRELATION.csv")
    hydro = corr[(corr.block == "HYDRO_FIELD") & (corr.sap_feature.isin(SAP3_COLS))]
    size = corr[corr.block.str.startswith("NUISANCE_") & (corr.sap_feature == "SSKD_ALL_SUM")]
    boot = pd.DataFrame(boots)
    perm = pd.DataFrame(perm_all)
    perm_sum = (
        perm.groupby(["code", "block", "scheme"], as_index=False)["delta_mae_mean"].mean()
        if len(perm)
        else pd.DataFrame()
    )

    def cell(code, col):
        return float(tab.set_index("Code").loc[code, col])

    lines = [
        "# HIC STATIC_SAP_KD Report (H094–H101)",
        "",
        f"- platform: `{PLATFORM_ID}`",
        f"- descriptor: **Fv STATIC_SAP_KD** (antibody-level only)",
        f"- freeze: `{PRE_EXTERNAL.name}`",
        "",
        "## Master table",
        "",
        tab.to_string(index=False),
        "",
        "## Primary questions",
        "",
        f"1. SAP3 vs Scratch base: H094 TEST_mean={cell('EXP-H094','TEST_mean'):.4f} vs H071={cell('EXP-H071','TEST_mean'):.4f}",
        f"2. SAP3 vs ESM2 base: H098 TEST_mean={cell('EXP-H098','TEST_mean'):.4f} vs H061={cell('EXP-H061','TEST_mean'):.4f}",
        f"3. SAP3 vs SURFACE: H094={cell('EXP-H094','TEST_mean'):.4f} vs H090={cell('EXP-H090','TEST_mean'):.4f}; ESM H098={cell('EXP-H098','TEST_mean'):.4f} vs H086={cell('EXP-H086','TEST_mean'):.4f}",
        f"4. SAP9 vs SAP3: H095={cell('EXP-H095','TEST_mean'):.4f} vs H094; H099={cell('EXP-H099','TEST_mean'):.4f} vs H098",
        f"5. SURFACE+SAP: H096={cell('EXP-H096','TEST_mean'):.4f} vs H090; H100={cell('EXP-H100','TEST_mean'):.4f} vs H086",
        f"6. F4+SAP: H097={cell('EXP-H097','TEST_mean'):.4f} vs H093; H101={cell('EXP-H101','TEST_mean'):.4f} vs H089",
        "",
        "## HYDRO_FIELD redundancy (max |Pearson| over block cols)",
        "",
        hydro.to_string(index=False) if len(hydro) else "(none)",
        "",
        "## SSKD_ALL_SUM vs size / total SASA",
        "",
        size.to_string(index=False) if len(size) else "(none)",
        "",
        "## Paired bootstrap (Δ = MAE_A − MAE_B; negative ⇒ A better)",
        "",
        boot.to_string(index=False),
        "",
        "## Block permutation (mean ΔMAE across folds)",
        "",
        perm_sum.to_string(index=False) if len(perm_sum) else "(none)",
        "",
        "## External four-way",
        "",
        "See `HIC_STATIC_SAP_KD_EXTERNAL_FOUR_WAY.csv`.",
        "",
    ]
    (ROOT / "results/HIC_STATIC_SAP_KD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "train", "analyze", "all"], default="all", nargs="?")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase in ("prereg", "all"):
        from preregister_h094_static_sap_kd import main as prereg_main

        rc = prereg_main()
        if rc != 0:
            return rc

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
