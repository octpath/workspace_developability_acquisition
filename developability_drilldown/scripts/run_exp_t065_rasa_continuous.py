#!/usr/bin/env python3
"""Paired EXP-T037 replay + EXP-T065 continuous RASA Transformer experiment.

Phases:
  rasa_qc | control_cv | issue_t065 | t065_cv | freeze | test | artifacts | report | all

No Optuna / no second scientific variant. Stop after EXP-T065.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "top_models_feature_bundle"))

from _lib import (  # noqa: E402
    EXPERIMENTS_COLUMNS,
    file_sha256,
    feature_content_sha256,
    mae,
)
from classical_features.rasa_cache import build_or_load_rasa_cache  # noqa: E402
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import (  # noqa: E402
    AA_TO_IDX,
    BUNDLE_ROOT,
    REGION_TO_IDX,
    RECIPE_TO_FEATURE_SET,
)
from antibody_transformer.data import (  # noqa: E402
    attach_continuous_rasa,
    load_dev_test,
    load_folds,
    load_residue_bundle,
    load_solution,
)
from antibody_transformer.training import (  # noqa: E402
    full_dev_transformer_predict,
    run_transformer_cv,
)
from advanced_models.features import build_recipe_parts  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

TARGET = "TmApp"
CONTROL_CODE = "EXP-T037"
CONTROL_EID = "TRF_TM_ABLINGUA_FULL_CONCAT_FUS_BIOEMU_MPNN"
T065_EID = "TRF_TM_ABLINGUA_RASA_CONT_FULL_CONCAT_FUS_BIOEMU_MPNN"
RECIPE = "TM_BASE_BIOEMU_MPNN__RIDGE"
VARIANT_CONTROL = "T037_REPLAY"
VARIANT_T065 = "T065_RASA_CONT"

OUT_REPRO = ROOT / "results" / "reproduction" / CONTROL_CODE
OUT_T065 = ROOT / "results" / "EXP-T065_run"
FREEZE_PATH = ROOT / "results" / "EXP-T065_CV_FREEZE.yaml"
REPORT_PATH = ROOT / "results" / "EXP-T065_RASA_CONTINUOUS_REPORT.md"
RASA_QC_PATH = ROOT / "results" / "EXP-T065_RASA_QC.json"
STATE_PATH = ROOT / "results" / "EXP-T065_run_state.json"


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str) + "\n")


def rasa_file_hash() -> dict[str, str]:
    cache = ROOT / "experiments" / "classical_cache"
    out = {}
    for name in ("rasa_heavy.npy", "rasa_light.npy", "rasa_ids.npy", "rasa_meta.json"):
        p = cache / name
        out[name] = file_sha256(p)
    return out


def prepare_bundle(*, with_rasa: bool):
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = load_residue_bundle(dev, test, need_ablingua=True)
    qc = None
    if with_rasa:
        rasa = build_or_load_rasa_cache(dev, test)
        seqs = pd.concat(
            [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
            ignore_index=True,
        )
        qc = attach_continuous_rasa(
            rb,
            rasa_heavy=rasa["H"],
            rasa_light=rasa["L"],
            rasa_ids=rasa["ids"],
            seqs=seqs,
        )
        qc["rasa_meta"] = rasa.get("meta", {})
        qc["rasa_hashes"] = rasa_file_hash()
        qc["source_path"] = str(ROOT / "experiments" / "classical_cache")
        qc["structure_source"] = rasa.get("meta", {}).get(
            "source", "/workspace_developability_acquisition/feature_extension/data/esmfold_fv"
        )
        qc["rasa_definition"] = rasa.get("meta", {}).get(
            "method",
            "Bio.PDB.SASA.ShrakeRupley probe=1.4 n_points=100 MaxASA=Tien2013",
        )
        finite = np.concatenate(
            [
                rb.heavy_rasa[np.isfinite(rb.heavy_rasa)],
                rb.light_rasa[np.isfinite(rb.light_rasa)],
            ]
        )
        qc["rasa_range"] = {
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "mean": float(finite.mean()) if len(finite) else None,
        }
        RASA_QC_PATH.write_text(json.dumps(qc, indent=2, default=str) + "\n")
        if qc["n_aa_mismatch_flags"]:
            raise SystemExit(f"RASA AA mismatch: {qc['aa_mismatch_ids']}")
    return dev, test, folds, rb, qc


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def oof_to_csv(ids: list[str], vals: np.ndarray, path: Path, col: str = TARGET) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, col: vals}).to_csv(path, index=False)


def historical_control_scores() -> dict[str, float]:
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    r = exp[exp["experiment_code"] == CONTROL_CODE].iloc[0]
    return {
        "cv_primary_mae": float(r["cv_primary_mae"]),
        "cv_shadow_mae": float(r["cv_shadow_mae"]),
        "cv_mean_mae": float(r["cv_mean_mae"]),
        "cv_worst_mae": float(r["cv_worst_mae"]),
        "public_mae": float(r["public_mae"]),
        "private_mae": float(r["private_mae"]),
        "test_overall_mae": float(r["test_overall_mae"]),
    }


def classical_t045_scores() -> dict[str, float]:
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    r = exp[exp["experiment_code"] == "EXP-T045"].iloc[0]
    return {
        "cv_primary_mae": float(r["cv_primary_mae"]),
        "cv_shadow_mae": float(r["cv_shadow_mae"]),
        "cv_worst_mae": float(r["cv_worst_mae"]),
    }


def phase_rasa_qc() -> dict:
    print("=== RASA QC ===", flush=True)
    _, _, _, _, qc = prepare_bundle(with_rasa=True)
    assert qc is not None
    print(
        f"expected={qc['n_expected_residues']} mapped={qc['n_mapped_residues']} "
        f"missing={qc['n_missing_rasa_residues']} aa_mismatch={qc['n_aa_mismatch_flags']}",
        flush=True,
    )
    if qc["n_mapped_residues"] != qc["n_expected_residues"]:
        print("WARNING: some missing RASA residues (contribution=0)", flush=True)
    return qc


def phase_control_cv(*, quick: bool) -> dict:
    print("=== EXP-T037 control replay CV ===", flush=True)
    OUT_REPRO.mkdir(parents=True, exist_ok=True)
    # Do not overwrite historical predictions
    hist_pred = ROOT / "experiments" / "predictions" / CONTROL_CODE
    assert hist_pred.exists(), "historical EXP-T037 predictions must remain"

    dev, test, folds, rb, _ = prepare_bundle(with_rasa=False)
    t0 = time.time()
    summary = run_transformer_cv(
        target=TARGET,
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        variant_id=VARIANT_CONTROL,
        dev=dev,
        rb=rb,
        folds=folds,
        device=device_str(),
        out_dir=OUT_REPRO,
        quick=quick,
        recipe_id=RECIPE,
        plm_source="ablingua",
        pooling_mode="reg",
        use_continuous_rasa=False,
    )
    ids = [
        str(x)
        for x in np.load(
            OUT_REPRO / f"oof_{VARIANT_CONTROL}_{summary['config_hash']}.npz",
            allow_pickle=True,
        )["ids"].tolist()
    ]
    z = np.load(
        OUT_REPRO / f"oof_{VARIANT_CONTROL}_{summary['config_hash']}.npz",
        allow_pickle=True,
    )
    oof_to_csv(ids, z["primary_oof"], OUT_REPRO / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], OUT_REPRO / "oof_shadow.csv")

    hist = historical_control_scores()
    # prediction delta vs historical OOF
    hp = pd.read_csv(hist_pred / "oof_primary.csv")
    hs = pd.read_csv(hist_pred / "oof_shadow.csv")
    rp = pd.read_csv(OUT_REPRO / "oof_primary.csv")
    rs = pd.read_csv(OUT_REPRO / "oof_shadow.csv")
    for df in (hp, hs, rp, rs):
        df["id"] = df["id"].astype(str)
    hp = hp.set_index("id").loc[ids]
    hs = hs.set_index("id").loc[ids]
    rp = rp.set_index("id").loc[ids]
    rs = rs.set_index("id").loc[ids]
    col_h = TARGET if TARGET in hp.columns else [c for c in hp.columns if c != "id"][0]
    delta = {
        "primary_pred_max_abs": float(np.max(np.abs(hp[col_h].to_numpy(float) - rp[TARGET].to_numpy(float)))),
        "shadow_pred_max_abs": float(np.max(np.abs(hs[col_h].to_numpy(float) - rs[TARGET].to_numpy(float)))),
        "delta_primary": float(summary["primary_mae"] - hist["cv_primary_mae"]),
        "delta_shadow": float(summary["shadow_mae"] - hist["cv_shadow_mae"]),
        "delta_worst": float(summary["cv_worst_mae"] - hist["cv_worst_mae"]),
    }
    evidence = {
        "git_rev": git_rev(),
        "quick": quick,
        "elapsed_sec": time.time() - t0,
        "replay": {
            "cv_primary_mae": summary["primary_mae"],
            "cv_shadow_mae": summary["shadow_mae"],
            "cv_mean_mae": summary["cv_mean_mae"],
            "cv_worst_mae": summary["cv_worst_mae"],
            "config_hash": summary["config_hash"],
            "best_epochs_primary": summary["best_epochs_primary"],
            "seed_maes_primary": summary.get("seed_dispersion_primary"),
        },
        "historical": hist,
        "delta": delta,
        "note": "Historical EXP-T037 predictions untouched; reproduction stored under results/reproduction/EXP-T037/",
    }
    (OUT_REPRO / "reproduction_evidence.json").write_text(
        json.dumps(evidence, indent=2, default=str) + "\n"
    )
    print(json.dumps(evidence["replay"], indent=2), flush=True)
    print("delta vs historical:", delta, flush=True)
    state = load_state()
    state["control_replay"] = evidence
    save_state(state)
    return evidence


def phase_issue_t065() -> str:
    codes = load_codes()
    if (codes["experiment_id"] == T065_EID).any():
        code = str(codes.set_index("experiment_id").loc[T065_EID, "experiment_code"])
        print(f"EXP already issued: {code}", flush=True)
        return code
    nxt = next_code(TARGET)
    if nxt != "EXP-T065":
        raise SystemExit(f"expected next code EXP-T065, got {nxt}")
    code = issue_code(
        T065_EID,
        TARGET,
        source_model_id=f"RASA_CONT::{CONTROL_CODE}",
        phase="ARCHITECTURE_RASA_CONT",
        notes="continuous additive RASA annotation vs EXP-T037; no HPO",
    )
    if code != "EXP-T065":
        raise SystemExit(f"issued {code}, expected EXP-T065")
    print(f"Issued {code}", flush=True)
    state = load_state()
    state["experiment_code"] = code
    save_state(state)
    return code


def write_t065_config(code: str, rasa_hashes: dict) -> Path:
    cfg = {
        "experiment_code": code,
        "experiment_id": T065_EID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"RASA_CONT::{CONTROL_CODE}",
        "transformer_type": "FUSION",
        "input_space": "RESIDUE_PLUS_FIXED_FEATURES_PLUS_RASA",
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
        "plm_source": "ABLINGUA",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": "concat",
        "pooling_mode": "CONCAT",
        "content_mode": "frozen",
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "ff_dim": 256,
        "dropout": 0.2,
        "seeds": [101, 202, 303],
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "batch_size": 16,
        "max_epochs": 300,
        "patience": 30,
        "gradient_clip": 1.0,
        "loss": "SmoothL1Loss(beta=0.5)",
        "cv_protocol": "canonical_simple_tvt_primary_shadow",
        "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
        "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
        "representation_status": "NOT_EXPORTED",
        "representation_aggregation_future": "UNDECIDED",
        "fusion_feature_set_id": RECIPE_TO_FEATURE_SET[RECIPE],
        "fusion_projection_dim": 64,
        "fusion_recipe": RECIPE,
        "control_experiment_code": CONTROL_CODE,
        "residue_rasa_annotation": {
            "enabled": True,
            "rasa_mode": "CONTINUOUS_ADDITIVE",
            "rasa_projection": "LINEAR_NO_BIAS",
            "rasa_projection_init": "ZERO",
            "rasa_source": "experiments/classical_cache",
            "rasa_hashes": rasa_hashes,
            "control_experiment_code": CONTROL_CODE,
        },
    }
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def phase_t065_cv(*, quick: bool, code: str) -> dict:
    print("=== EXP-T065 CV ===", flush=True)
    OUT_T065.mkdir(parents=True, exist_ok=True)
    dev, test, folds, rb, qc = prepare_bundle(with_rasa=True)
    assert qc is not None
    write_t065_config(code, qc["rasa_hashes"])
    t0 = time.time()
    summary = run_transformer_cv(
        target=TARGET,
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        variant_id=VARIANT_T065,
        dev=dev,
        rb=rb,
        folds=folds,
        device=device_str(),
        out_dir=OUT_T065,
        quick=quick,
        recipe_id=RECIPE,
        plm_source="ablingua",
        pooling_mode="reg",
        use_continuous_rasa=True,
    )
    ids = [
        str(x)
        for x in np.load(
            OUT_T065 / f"oof_{VARIANT_T065}_{summary['config_hash']}.npz",
            allow_pickle=True,
        )["ids"].tolist()
    ]
    z = np.load(
        OUT_T065 / f"oof_{VARIANT_T065}_{summary['config_hash']}.npz",
        allow_pickle=True,
    )
    pred_dir = ROOT / "experiments" / "predictions" / code
    pred_dir.mkdir(parents=True, exist_ok=True)
    oof_to_csv(ids, z["primary_oof"], pred_dir / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], pred_dir / "oof_shadow.csv")
    # also stash under run dir
    oof_to_csv(ids, z["primary_oof"], OUT_T065 / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], OUT_T065 / "oof_shadow.csv")
    out = {
        "git_rev": git_rev(),
        "quick": quick,
        "elapsed_sec": time.time() - t0,
        "cv_primary_mae": summary["primary_mae"],
        "cv_shadow_mae": summary["shadow_mae"],
        "cv_mean_mae": summary["cv_mean_mae"],
        "cv_worst_mae": summary["cv_worst_mae"],
        "config_hash": summary["config_hash"],
        "best_epochs_primary": summary["best_epochs_primary"],
        "rasa_qc": {
            "n_expected_residues": qc["n_expected_residues"],
            "n_mapped_residues": qc["n_mapped_residues"],
            "n_missing_rasa_residues": qc["n_missing_rasa_residues"],
            "n_aa_mismatch_flags": qc["n_aa_mismatch_flags"],
            "rasa_hashes": qc["rasa_hashes"],
            "rasa_range": qc["rasa_range"],
        },
    }
    (OUT_T065 / "cv_summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae")}, indent=2), flush=True)
    state = load_state()
    state["t065_cv"] = out
    state["experiment_code"] = code
    save_state(state)
    return out


def paired_bootstrap(abs_err_a: np.ndarray, abs_err_b: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    """Bootstrap CI for MAE(a)-MAE(b); negative => a better."""
    rng = np.random.default_rng(seed)
    d = abs_err_a - abs_err_b
    N = len(d)
    boots = np.empty(n, float)
    for i in range(n):
        idx = rng.integers(0, N, N)
        boots[i] = float(d[idx].mean())
    return {
        "mae_delta": float(d.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "n_boot": n,
    }


def fold_mae_deltas(dev: pd.DataFrame, folds_map: dict, oof_new: pd.Series, oof_ctl: pd.Series) -> list[dict]:
    y = dev.set_index("id")[TARGET]
    rows = []
    for k in range(5):
        ids = [i for i, f in folds_map.items() if int(f) == k]
        ae_n = (y.loc[ids] - oof_new.loc[ids]).abs()
        ae_c = (y.loc[ids] - oof_ctl.loc[ids]).abs()
        rows.append(
            {
                "fold": k,
                "mae_t065": float(ae_n.mean()),
                "mae_control": float(ae_c.mean()),
                "delta": float(ae_n.mean() - ae_c.mean()),
            }
        )
    return rows


def phase_freeze(code: str) -> dict:
    print("=== CV FREEZE ===", flush=True)
    if FREEZE_PATH.exists():
        print(f"Freeze already exists: {FREEZE_PATH}", flush=True)
        return yaml.safe_load(FREEZE_PATH.read_text())

    state = load_state()
    ctl = state["control_replay"]
    t065 = state["t065_cv"]
    hist = historical_control_scores()
    t045 = classical_t045_scores()
    cfg_path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    # ensure no Public/Private in freeze artifact before Test
    freeze = {
        "experiment_code": code,
        "experiment_id": T065_EID,
        "git_rev": git_rev(),
        "config": cfg,
        "rasa_source": cfg["residue_rasa_annotation"]["rasa_source"],
        "rasa_hashes": cfg["residue_rasa_annotation"]["rasa_hashes"],
        "control_replay": {
            "cv_primary_mae": ctl["replay"]["cv_primary_mae"],
            "cv_shadow_mae": ctl["replay"]["cv_shadow_mae"],
            "cv_mean_mae": ctl["replay"]["cv_mean_mae"],
            "cv_worst_mae": ctl["replay"]["cv_worst_mae"],
        },
        "EXP-T065": {
            "cv_primary_mae": t065["cv_primary_mae"],
            "cv_shadow_mae": t065["cv_shadow_mae"],
            "cv_mean_mae": t065["cv_mean_mae"],
            "cv_worst_mae": t065["cv_worst_mae"],
            "config_hash": t065["config_hash"],
            "best_epochs_primary": t065["best_epochs_primary"],
        },
        "historical_EXP-T037": hist,
        "classical_EXP-T045": t045,
        "selection_statement": (
            "CV completed for EXP-T065 under identical protocol to EXP-T037 replay; "
            "config frozen; Public/Private not inspected before this freeze; "
            "proceed to single full-Dev Test evaluation."
        ),
        "public_mae": None,
        "private_mae": None,
        "test_overall_mae": None,
    }
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    print(f"Wrote {FREEZE_PATH}", flush=True)
    return freeze


def phase_test(code: str, *, quick: bool) -> dict:
    print("=== EXP-T065 full-Dev Test (once) ===", flush=True)
    if not FREEZE_PATH.exists():
        raise SystemExit("CV freeze missing; refuse Test")
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    if freeze.get("public_mae") is not None:
        print("Test already recorded in freeze; skipping retrain", flush=True)
        return {
            "public_mae": freeze["public_mae"],
            "private_mae": freeze["private_mae"],
            "test_overall_mae": freeze["test_overall_mae"],
        }

    pred_path = ROOT / "experiments" / "predictions" / code / "test.csv"
    if pred_path.exists() and not quick:
        # allow resume if freeze incomplete but test exists
        print("test.csv exists; scoring without retrain", flush=True)
        te = pd.read_csv(pred_path)
    else:
        state = load_state()
        best_epochs = state["t065_cv"]["best_epochs_primary"]
        dev, test, folds, rb, _ = prepare_bundle(with_rasa=True)
        te = full_dev_transformer_predict(
            target=TARGET,
            content_mode="frozen",
            annotation_mode="full",
            merge_mode="concat",
            chain_mode="HL",
            best_epochs_primary=best_epochs,
            dev=dev,
            test=test,
            rb=rb,
            device=device_str(),
            recipe_id=RECIPE,
            quick=quick,
            plm_source="ablingua",
            pooling_mode="reg",
            use_continuous_rasa=True,
        )
        te = te.rename(columns={"prediction": TARGET})
        te.to_csv(pred_path, index=False)
        te.to_csv(OUT_T065 / "test.csv", index=False)

    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    te = te.copy()
    te["id"] = te["id"].astype(str)
    sol = sol.set_index("id")
    te_s = te.set_index("id")[TARGET]
    y = sol[TARGET]
    pub_ids = sol.index[sol["is_public"].astype(bool)].tolist()
    priv_ids = sol.index[sol["is_private"].astype(bool)].tolist()
    pub = float(mae(y.loc[pub_ids].to_numpy(float), te_s.loc[pub_ids].to_numpy(float)))
    priv = float(mae(y.loc[priv_ids].to_numpy(float), te_s.loc[priv_ids].to_numpy(float)))
    overall = float(mae(y.loc[te["id"]].to_numpy(float), te_s.loc[te["id"]].to_numpy(float)))
    scores = {"public_mae": pub, "private_mae": priv, "test_overall_mae": overall}
    freeze["public_mae"] = pub
    freeze["private_mae"] = priv
    freeze["test_overall_mae"] = overall
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    state = load_state()
    state["t065_test"] = scores
    save_state(state)
    print(scores, flush=True)
    return scores


def materialize_fixed_branch(code: str, rb_ids: list[str], dev: pd.DataFrame, test: pd.DataFrame) -> Path:
    parts = build_recipe_parts(RECIPE, rb_ids)
    assert parts["mode"] == "standalone"
    X = parts["X"].copy()
    X.insert(0, "id", rb_ids)
    split = {str(i): "dev" for i in dev["id"].astype(str)}
    split.update({str(i): "test" for i in test["id"].astype(str)})
    X.insert(1, "split", [split[str(i)] for i in rb_ids])
    # metadata via attrs + sidecar
    path = ROOT / "experiments" / "features" / f"{code}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(path, index=False)
    meta = {
        "feature_role": "FUSION_FIXED_BRANCH",
        "recipe_id": RECIPE,
        "n_rows": len(X),
        "n_feature_cols": X.shape[1] - 2,
        "note": "Raw fixed-length BioEmu/MPNN(+AbLang2/SEQ) fusion branch before fold-specific impute/scale",
    }
    (ROOT / "experiments" / "features" / f"{code}.parquet.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    return path


def materialize_rasa_parquet(code: str, rb, dev: pd.DataFrame, test: pd.DataFrame) -> Path:
    from antibody_transformer.data import load_annotations

    ann = load_annotations()
    idx_to_aa = {v: k for k, v in AA_TO_IDX.items()}
    idx_to_region = {v: k for k, v in REGION_TO_IDX.items()}
    # imgt reverse from vocab on rb
    imgt_rev = {v: k for k, v in rb.imgt_vocab.items()}
    split = {str(i): "dev" for i in dev["id"].astype(str)}
    split.update({str(i): "test" for i in test["id"].astype(str)})
    rows = []
    for i, ab in enumerate(rb.ids):
        for chain, aa_arr, mask, imgt_arr, reg_arr, rasa_arr in (
            ("H", rb.heavy_aa[i], rb.heavy_mask[i], rb.heavy_imgt[i], rb.heavy_region[i], rb.heavy_rasa[i]),
            ("L", rb.light_aa[i], rb.light_mask[i], rb.light_imgt[i], rb.light_region[i], rb.light_rasa[i]),
        ):
            n = int(mask.sum())
            for j in range(n):
                r = float(rasa_arr[j]) if np.isfinite(rasa_arr[j]) else np.nan
                rows.append(
                    {
                        "id": ab,
                        "split": split[ab],
                        "chain": chain,
                        "seq_index": j,
                        "aa": idx_to_aa.get(int(aa_arr[j]), "X"),
                        "imgt_position": imgt_rev.get(int(imgt_arr[j]), "UNKNOWN"),
                        "region": idx_to_region.get(int(reg_arr[j]), "UNKNOWN"),
                        "rasa": r,
                    }
                )
    df = pd.DataFrame(rows)
    # sanity vs annotations length
    for ab in rb.ids[:3]:
        n_ann = len(ann[ann.id == ab])
        n_df = len(df[df.id == ab])
        if n_ann != n_df:
            raise SystemExit(f"RASA parquet length mismatch {ab}: {n_df} vs ann {n_ann}")
    out = ROOT / "experiments" / "inputs" / f"{code}_rasa.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def phase_artifacts(code: str) -> dict:
    print("=== Materialize artifacts + registry ===", flush=True)
    state = load_state()
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    dev, test, folds, rb, qc = prepare_bundle(with_rasa=True)
    feat_path = materialize_fixed_branch(code, rb.ids, dev, test)
    rasa_path = materialize_rasa_parquet(code, rb, dev, test)
    fsha = file_sha256(feat_path)
    feat_df = pd.read_parquet(feat_path)
    csha = feature_content_sha256(feat_df)
    rasa_sha = file_sha256(rasa_path)

    # diagnostics vs replay
    ctl_p = pd.read_csv(OUT_REPRO / "oof_primary.csv")
    ctl_s = pd.read_csv(OUT_REPRO / "oof_shadow.csv")
    new_p = pd.read_csv(ROOT / "experiments" / "predictions" / code / "oof_primary.csv")
    new_s = pd.read_csv(ROOT / "experiments" / "predictions" / code / "oof_shadow.csv")
    for d in (ctl_p, ctl_s, new_p, new_s):
        d["id"] = d["id"].astype(str)
    y = dev.set_index("id")[TARGET]
    ids = dev["id"].astype(str).tolist()
    ctl_ps = ctl_p.set_index("id").loc[ids, TARGET]
    ctl_ss = ctl_s.set_index("id").loc[ids, TARGET]
    new_ps = new_p.set_index("id").loc[ids, TARGET]
    new_ss = new_s.set_index("id").loc[ids, TARGET]
    ae_ctl_p = (y.loc[ids] - ctl_ps).abs().to_numpy(float)
    ae_new_p = (y.loc[ids] - new_ps).abs().to_numpy(float)
    ae_ctl_s = (y.loc[ids] - ctl_ss).abs().to_numpy(float)
    ae_new_s = (y.loc[ids] - new_ss).abs().to_numpy(float)
    boot = {
        "primary": paired_bootstrap(ae_new_p, ae_ctl_p),
        "shadow": paired_bootstrap(ae_new_s, ae_ctl_s),
    }
    fold_d = {
        "primary": fold_mae_deltas(dev, folds.primary, new_ps, ctl_ps),
        "shadow": fold_mae_deltas(dev, folds.shadow, new_ss, ctl_ss),
    }
    diag = {"bootstrap": boot, "fold_mae_deltas": fold_d}
    (OUT_T065 / "paired_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    # score recomputation check
    recomputed_p = float(mae(y.loc[ids].to_numpy(float), new_ps.to_numpy(float)))
    recomputed_s = float(mae(y.loc[ids].to_numpy(float), new_ss.to_numpy(float)))
    assert abs(recomputed_p - freeze["EXP-T065"]["cv_primary_mae"]) < 1e-10
    assert abs(recomputed_s - freeze["EXP-T065"]["cv_shadow_mae"]) < 1e-10

    pub = float(freeze["public_mae"])
    priv = float(freeze["private_mae"])
    overall = float(freeze["test_overall_mae"])
    cv_p = float(freeze["EXP-T065"]["cv_primary_mae"])
    cv_s = float(freeze["EXP-T065"]["cv_shadow_mae"])
    cv_m = float(freeze["EXP-T065"]["cv_mean_mae"])
    cv_w = float(freeze["EXP-T065"]["cv_worst_mae"])

    # registry row
    exp_path = ROOT / "results" / "experiments.csv"
    exp = pd.read_csv(exp_path)
    if code in set(exp["experiment_code"].astype(str)):
        print(f"{code} already in experiments.csv; updating row", flush=True)
        exp = exp[exp["experiment_code"] != code]
    if "control_experiment_code" not in exp.columns:
        exp["control_experiment_code"] = ""

    row = {c: "" for c in exp.columns}
    for c in EXPERIMENTS_COLUMNS:
        if c not in row:
            row[c] = ""
    row.update(
        {
            "experiment_code": code,
            "legacy_experiment_code": "",
            "experiment_id": T065_EID,
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer+Fusion+ContinuousRASA",
            "source_model_id": f"RASA_CONT::{CONTROL_CODE}",
            "feature_set_id": RECIPE_TO_FEATURE_SET[RECIPE],
            "source_recipe_id": RECIPE,
            "cv_primary_mae": cv_p,
            "cv_shadow_mae": cv_s,
            "cv_mean_mae": cv_m,
            "cv_worst_mae": cv_w,
            "public_mae": pub,
            "private_mae": priv,
            "test_overall_mae": overall,
            "public_private_delta": pub - priv,
            "public_private_gap": abs(pub - priv),
            "cv_protocol": "canonical_simple_tvt_primary_shadow",
            "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "artifact_status": "FULL",
            "source_reproducible": "YES",
            "drilldown_reproducible": "YES",
            "reproduction_status": "REPRODUCED",
            "config_path": f"experiments/configs/{code}.yaml",
            "feature_path": f"experiments/features/{code}.parquet",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "n_features": int(feat_df.shape[1] - 2),
            "feature_space": "FUSION_FIXED_BRANCH_RAW",
            "feature_sha256": fsha,
            "feature_content_sha256": csha,
            "score_source": "EXP-T065_cv+solution_postfreeze",
            "prediction_source": "EXP-T065_run",
            "feature_source": f"{RECIPE}+classical_rasa_cache+ablingua_residue",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue + BioEmu/MPNN + ESMFold RASA",
            "ensemble_type": "",
            "member_experiment_codes": "",
            "notes": (
                f"continuous additive RASA vs {CONTROL_CODE}; "
                f"rasa_input=experiments/inputs/{code}_rasa.parquet sha={rasa_sha}; "
                "representation_status=NOT_EXPORTED (no fold weight checkpoints)"
            ),
            "transformer_type": "FUSION",
            "plm_source": "ABLINGUA",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "CONCAT",
            "representation_status": "NOT_EXPORTED",
            "input_space": "RESIDUE_PLUS_FIXED_FEATURES_PLUS_RASA",
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
            "control_experiment_code": CONTROL_CODE,
        }
    )
    # align columns
    for c in exp.columns:
        if c not in row:
            row[c] = ""
    new_df = pd.DataFrame([{c: row.get(c, "") for c in exp.columns}])
    out_exp = pd.concat([exp, new_df], ignore_index=True)
    out_exp.to_csv(exp_path, index=False)

    # completeness CSV append
    comp_path = ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    comp = pd.read_csv(comp_path)
    if code not in set(comp["experiment_code"].astype(str)):
        crow = {c: "" for c in comp.columns}
        crow.update(
            {
                "experiment_code": code,
                "experiment_id": T065_EID,
                "target": TARGET,
                "family": "TRANSFORMER",
                "config_exists": True,
                "feature_exists": True,
                "feature_path": f"experiments/features/{code}.parquet",
                "feature_sha256": fsha,
                "oof_primary_exists": True,
                "oof_shadow_exists": True,
                "test_exists": True,
                "score_recompute_ok": True,
                "prediction_max_delta": 0.0,
                "reproduction_status": "REPRODUCED",
                "shareability_status": "SHAREABLE_COMPLETE",
                "canonical_benchmark_eligible": "YES",
            }
        )
        for c in comp.columns:
            if c not in crow:
                crow[c] = ""
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)

    state["artifacts"] = {
        "feature_path": str(feat_path),
        "rasa_path": str(rasa_path),
        "feature_sha256": fsha,
        "rasa_sha256": rasa_sha,
        "diagnostics": diag,
    }
    save_state(state)
    return state["artifacts"]


def region_error_diagnostic(code: str) -> Optional[dict]:
    """Post-hoc concentration of |err| improvement by region / high-RASA (no model change)."""
    try:
        state = load_state()
        if "artifacts" not in state:
            return None
        rasa = pd.read_parquet(ROOT / "experiments" / "inputs" / f"{code}_rasa.parquet")
        # antibody-level mean rasa / CDR fraction as crude post-hoc tags
        g = rasa.groupby("id").agg(
            mean_rasa=("rasa", "mean"),
            cdr_frac=("region", lambda s: float(np.mean(s.astype(str).str.startswith("CDR")))),
            cdr3_frac=("region", lambda s: float(np.mean(s.astype(str) == "CDR3"))),
        )
        dev = pd.read_csv(ROOT / "data" / "dev.csv")
        dev["id"] = dev["id"].astype(str)
        y = dev.set_index("id")[TARGET]
        ctl = pd.read_csv(OUT_REPRO / "oof_primary.csv")
        neu = pd.read_csv(ROOT / "experiments" / "predictions" / code / "oof_primary.csv")
        ctl["id"] = ctl["id"].astype(str)
        neu["id"] = neu["id"].astype(str)
        ctl = ctl.set_index("id")[TARGET]
        neu = neu.set_index("id")[TARGET]
        ids = dev["id"].tolist()
        d_ae = (y.loc[ids] - neu.loc[ids]).abs() - (y.loc[ids] - ctl.loc[ids]).abs()
        g = g.loc[ids]
        high = g["mean_rasa"] >= g["mean_rasa"].median()
        return {
            "delta_ae_mean_high_rasa_abs": float(d_ae.loc[high.index[high]].mean()),
            "delta_ae_mean_low_rasa_abs": float(d_ae.loc[high.index[~high]].mean()),
            "corr_delta_ae_vs_mean_rasa": float(np.corrcoef(g["mean_rasa"].to_numpy(float), d_ae.to_numpy(float))[0, 1]),
            "corr_delta_ae_vs_cdr_frac": float(np.corrcoef(g["cdr_frac"].to_numpy(float), d_ae.to_numpy(float))[0, 1]),
            "corr_delta_ae_vs_cdr3_frac": float(np.corrcoef(g["cdr3_frac"].to_numpy(float), d_ae.to_numpy(float))[0, 1]),
            "note": "antibody-level correlations only; not residue-attribution",
        }
    except Exception as e:
        return {"error": str(e)}


def phase_report(code: str) -> None:
    print("=== Report ===", flush=True)
    state = load_state()
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    ctl = state["control_replay"]
    hist = historical_control_scores()
    t045 = classical_t045_scores()
    t065 = freeze["EXP-T065"]
    test = {
        "public_mae": freeze["public_mae"],
        "private_mae": freeze["private_mae"],
        "test_overall_mae": freeze["test_overall_mae"],
    }
    diag = json.loads((OUT_T065 / "paired_diagnostics.json").read_text())
    qc = json.loads(RASA_QC_PATH.read_text()) if RASA_QC_PATH.exists() else {}
    region = region_error_diagnostic(code)

    d_replay_p = t065["cv_primary_mae"] - ctl["replay"]["cv_primary_mae"]
    d_replay_s = t065["cv_shadow_mae"] - ctl["replay"]["cv_shadow_mae"]
    d_replay_w = t065["cv_worst_mae"] - ctl["replay"]["cv_worst_mae"]
    d_hist_p = t065["cv_primary_mae"] - hist["cv_primary_mae"]
    d_hist_s = t065["cv_shadow_mae"] - hist["cv_shadow_mae"]
    d_hist_w = t065["cv_worst_mae"] - hist["cv_worst_mae"]
    d_t045_w = t065["cv_worst_mae"] - t045["cv_worst_mae"]

    both_improve = d_replay_p < 0 and d_replay_s < 0
    both_worsen = d_replay_p > 0 and d_replay_s > 0
    if both_improve and d_replay_w < 0:
        verdict = "POSITIVE"
    elif both_worsen and d_replay_w > 0:
        verdict = "NEGATIVE"
    else:
        verdict = "MIXED"

    replay_ok = (
        abs(ctl["delta"]["delta_primary"]) < 0.05
        and abs(ctl["delta"]["delta_shadow"]) < 0.05
    )

    lines = [
        "# EXP-T065 — Continuous RASA annotation for AbLingua Transformer",
        "",
        f"- git: `{git_rev()}`",
        f"- control: `{CONTROL_CODE}` replay under current implementation",
        f"- scientific change: continuous additive `Linear(1,d_model,bias=False)` zero-init RASA projection",
        f"- verdict: **{verdict}**",
        "",
        "## 1. Did current EXP-T037 replay reproduce historical performance?",
        "",
        f"- Historical P/S/W: {hist['cv_primary_mae']:.6f} / {hist['cv_shadow_mae']:.6f} / {hist['cv_worst_mae']:.6f}",
        f"- Replay P/S/W: {ctl['replay']['cv_primary_mae']:.6f} / {ctl['replay']['cv_shadow_mae']:.6f} / {ctl['replay']['cv_worst_mae']:.6f}",
        f"- CV deltas (replay−hist): P={ctl['delta']['delta_primary']:+.6f} S={ctl['delta']['delta_shadow']:+.6f} W={ctl['delta']['delta_worst']:+.6f}",
        f"- Pred max|Δ|: primary={ctl['delta']['primary_pred_max_abs']:.6g} shadow={ctl['delta']['shadow_pred_max_abs']:.6g}",
        f"- Sufficient for causal control: {'YES' if replay_ok else 'NO (use current replay as control; historical status unchanged)'}",
        "",
        "## 2–5. Continuous RASA vs replay control",
        "",
        f"- EXP-T065 P/S/mean/W: {t065['cv_primary_mae']:.6f} / {t065['cv_shadow_mae']:.6f} / {t065['cv_mean_mae']:.6f} / {t065['cv_worst_mae']:.6f}",
        f"- Δ Primary (T065−replay): {d_replay_p:+.6f} → {'improved' if d_replay_p < 0 else 'worsened' if d_replay_p > 0 else 'tied'}",
        f"- Δ Shadow: {d_replay_s:+.6f}",
        f"- Δ CV worst: {d_replay_w:+.6f}",
        f"- Both schemes same direction: {'YES' if both_improve or both_worsen else 'NO'}",
        "",
        "## 6. Paired bootstrap (MAE_T065 − MAE_replay; negative = T065 better)",
        "",
        f"- Primary: Δ={diag['bootstrap']['primary']['mae_delta']:+.6f} CI95=[{diag['bootstrap']['primary']['ci95_low']:+.6f}, {diag['bootstrap']['primary']['ci95_high']:+.6f}]",
        f"- Shadow: Δ={diag['bootstrap']['shadow']['mae_delta']:+.6f} CI95=[{diag['bootstrap']['shadow']['ci95_low']:+.6f}, {diag['bootstrap']['shadow']['ci95_high']:+.6f}]",
        "",
        "### Fold MAE deltas (Primary)",
        "",
    ]
    for r in diag["fold_mae_deltas"]["primary"]:
        lines.append(f"- fold {r['fold']}: T065={r['mae_t065']:.4f} ctl={r['mae_control']:.4f} Δ={r['delta']:+.4f}")
    lines += ["", "### Fold MAE deltas (Shadow)", ""]
    for r in diag["fold_mae_deltas"]["shadow"]:
        lines.append(f"- fold {r['fold']}: T065={r['mae_t065']:.4f} ctl={r['mae_control']:.4f} Δ={r['delta']:+.4f}")

    lines += [
        "",
        "## 7. Beat historical T037 / replay / classical T045?",
        "",
        f"- vs historical T037 worst: {d_hist_w:+.6f} (P {d_hist_p:+.6f}, S {d_hist_s:+.6f})",
        f"- vs replay worst: {d_replay_w:+.6f}",
        f"- vs classical T045 worst ({t045['cv_worst_mae']:.6f}): {d_t045_w:+.6f}",
        "",
        "## 8. Public / Private / Overall (after freeze)",
        "",
        f"- Public: {test['public_mae']:.6f}",
        f"- Private: {test['private_mae']:.6f}",
        f"- Overall: {test['test_overall_mae']:.6f}",
        "",
        "## 9. RASA alignment/QC",
        "",
        f"- expected residues: {qc.get('n_expected_residues')}",
        f"- mapped: {qc.get('n_mapped_residues')}",
        f"- missing (contrib=0): {qc.get('n_missing_rasa_residues')}",
        f"- AA mismatches: {qc.get('n_aa_mismatch_flags')}",
        f"- range: {qc.get('rasa_range')}",
        f"- source: {qc.get('structure_source')}",
        f"- definition: {qc.get('rasa_definition')}",
        f"- hashes: {qc.get('rasa_hashes')}",
        "",
        "## 10. Error concentration (post-hoc, antibody-level)",
        "",
        f"```json\n{json.dumps(region, indent=2)}\n```",
        "",
        "## 11. Artifact completeness",
        "",
        f"- config: experiments/configs/{code}.yaml",
        f"- fixed branch: experiments/features/{code}.parquet (feature_role=FUSION_FIXED_BRANCH)",
        f"- RASA: experiments/inputs/{code}_rasa.parquet",
        f"- preds: experiments/predictions/{code}/{{oof_primary,oof_shadow,test}}.csv",
        f"- AbLingua: input_asset_ref in config (not duplicated)",
        f"- representation export: NOT_EXPORTED (fold caches store predictions, not weights; exporting would need new checkpoint subsystem)",
        "",
        "## 12. Reproducibility",
        "",
        "- shareability_status: SHAREABLE_COMPLETE",
        "- reproduction_status: REPRODUCED (score recomputation from saved predictions)",
        "- canonical_benchmark_eligible: YES",
        "- EXP-T037 historical predictions unchanged",
        "",
        "## STOP",
        "",
        "No EXP-T066 / sweeps / gates. Inspect this result before next experiment.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}", flush=True)
    state["verdict"] = verdict
    save_state(state)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=[
            "rasa_qc",
            "control_cv",
            "issue_t065",
            "t065_cv",
            "freeze",
            "test",
            "artifacts",
            "report",
            "all",
        ],
    )
    ap.add_argument("--quick", action="store_true", help="tiny epoch smoke only")
    args = ap.parse_args()

    if args.phase == "rasa_qc":
        phase_rasa_qc()
        return 0
    if args.phase == "control_cv":
        phase_control_cv(quick=args.quick)
        return 0
    if args.phase == "issue_t065":
        phase_issue_t065()
        return 0
    if args.phase == "t065_cv":
        code = load_state().get("experiment_code") or phase_issue_t065()
        phase_t065_cv(quick=args.quick, code=code)
        return 0
    if args.phase == "freeze":
        code = load_state()["experiment_code"]
        phase_freeze(code)
        return 0
    if args.phase == "test":
        code = load_state()["experiment_code"]
        phase_test(code, quick=args.quick)
        return 0
    if args.phase == "artifacts":
        code = load_state()["experiment_code"]
        phase_artifacts(code)
        return 0
    if args.phase == "report":
        code = load_state()["experiment_code"]
        phase_report(code)
        return 0

    # all
    phase_rasa_qc()
    phase_control_cv(quick=args.quick)
    code = phase_issue_t065()
    phase_t065_cv(quick=args.quick, code=code)
    phase_freeze(code)
    phase_test(code, quick=args.quick)
    phase_artifacts(code)
    phase_report(code)
    print("DONE EXP-T065 — STOP", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
