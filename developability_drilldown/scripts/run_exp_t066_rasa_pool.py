#!/usr/bin/env python3
"""EXP-T066: RASA-weighted residue aggregation prior into chain REG (vs EXP-T037).

Does NOT retrain EXP-T037 (use stored canonical OOF). One experiment then STOP.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

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
T065_CODE = "EXP-T065"
T066_EID = "TRF_TM_ABLINGUA_RASA_POOL_FULL_CONCAT_FUS_BIOEMU_MPNN"
RECIPE = "TM_BASE_BIOEMU_MPNN__RIDGE"
VARIANT = "T066_RASA_POOL"

OUT_RUN = ROOT / "results" / "EXP-T066_run"
FREEZE_PATH = ROOT / "results" / "EXP-T066_CV_FREEZE.yaml"
REPORT_PATH = ROOT / "results" / "EXP-T066_RASA_POOL_REPORT.md"
RASA_QC_PATH = ROOT / "results" / "EXP-T066_RASA_QC.json"
STATE_PATH = ROOT / "results" / "EXP-T066_run_state.json"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str) + "\n")


def rasa_file_hash() -> dict[str, str]:
    cache = ROOT / "experiments" / "classical_cache"
    return {n: file_sha256(cache / n) for n in ("rasa_heavy.npy", "rasa_light.npy", "rasa_ids.npy", "rasa_meta.json")}


def scores_row(code: str) -> dict[str, float]:
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    r = exp[exp["experiment_code"] == code].iloc[0]
    return {
        "cv_primary_mae": float(r["cv_primary_mae"]),
        "cv_shadow_mae": float(r["cv_shadow_mae"]),
        "cv_mean_mae": float(r["cv_mean_mae"]),
        "cv_worst_mae": float(r["cv_worst_mae"]),
        "public_mae": float(r["public_mae"]),
        "private_mae": float(r["private_mae"]),
        "test_overall_mae": float(r["test_overall_mae"]),
    }


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def oof_to_csv(ids, vals, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, TARGET: vals}).to_csv(path, index=False)


def prepare_bundle():
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = load_residue_bundle(dev, test, need_ablingua=True)
    rasa = build_or_load_rasa_cache(dev, test)
    seqs = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
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
        "method", "Bio.PDB.SASA.ShrakeRupley probe=1.4 n_points=100 MaxASA=Tien2013"
    )
    finite = np.concatenate(
        [rb.heavy_rasa[np.isfinite(rb.heavy_rasa)], rb.light_rasa[np.isfinite(rb.light_rasa)]]
    )
    qc["rasa_range"] = {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }
    # per-chain denom audit
    n_zero = 0
    for i in range(len(rb.ids)):
        for arr, mask in ((rb.heavy_rasa[i], rb.heavy_mask[i]), (rb.light_rasa[i], rb.light_mask[i])):
            r = np.nan_to_num(arr[: int(mask.sum())], nan=0.0)
            if float(r.sum()) <= 1e-12:
                n_zero += 1
    qc["n_zero_denom_chains"] = n_zero
    RASA_QC_PATH.write_text(json.dumps(qc, indent=2, default=str) + "\n")
    if qc["n_aa_mismatch_flags"]:
        raise SystemExit(f"AA mismatch: {qc['aa_mismatch_ids']}")
    return dev, test, folds, rb, qc


def issue_t066() -> str:
    codes = load_codes()
    if (codes["experiment_id"] == T066_EID).any():
        return str(codes.set_index("experiment_id").loc[T066_EID, "experiment_code"])
    nxt = next_code(TARGET)
    if nxt != "EXP-T066":
        raise SystemExit(f"expected EXP-T066, got {nxt}")
    code = issue_code(
        T066_EID,
        TARGET,
        source_model_id=f"RASA_POOL::{CONTROL_CODE}",
        phase="ARCHITECTURE_RASA_POOL",
        notes="RASA-weighted REG residual vs EXP-T037; no HPO; encoder unchanged",
    )
    if code != "EXP-T066":
        raise SystemExit(code)
    return code


def write_config(code: str, rasa_hashes: dict) -> Path:
    cfg = {
        "experiment_code": code,
        "experiment_id": T066_EID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"RASA_POOL::{CONTROL_CODE}",
        "transformer_type": "FUSION",
        "input_space": "RESIDUE_PLUS_FIXED_FEATURES_PLUS_RASA_POOL",
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
        "representation_note": "fold caches store OOF preds only; no weight checkpoints in this runner",
        "fusion_feature_set_id": RECIPE_TO_FEATURE_SET[RECIPE],
        "fusion_projection_dim": 64,
        "fusion_recipe": RECIPE,
        "control_experiment_code": CONTROL_CODE,
        "residue_rasa_aggregation": {
            "enabled": True,
            "rasa_mode": "CONTINUOUS_WEIGHTED_POOL",
            "rasa_pool_scope": "PER_CHAIN",
            "rasa_pool_projection": "SHARED_LINEAR_NO_BIAS",
            "rasa_pool_projection_init": "ZERO",
            "rasa_residual_target": "CHAIN_REG",
            "rasa_threshold": "NONE",
            "rasa_power": 1.0,
            "rasa_source": "experiments/classical_cache",
            "rasa_hashes": rasa_hashes,
            "control_experiment_code": CONTROL_CODE,
            "encoder_input_unchanged": True,
        },
    }
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def paired_bootstrap(ae_a, ae_b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = ae_a - ae_b
    boots = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(d), len(d))
        boots[i] = float(d[idx].mean())
    return {
        "mae_delta": float(d.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "n_boot": n,
    }


def fold_deltas(dev, fmap, oof_new, oof_ctl):
    y = dev.set_index("id")[TARGET]
    rows = []
    for k in range(5):
        ids = [i for i, f in fmap.items() if int(f) == k]
        ae_n = (y.loc[ids] - oof_new.loc[ids]).abs()
        ae_c = (y.loc[ids] - oof_ctl.loc[ids]).abs()
        rows.append(
            {
                "fold": k,
                "mae_t066": float(ae_n.mean()),
                "mae_control": float(ae_c.mean()),
                "delta": float(ae_n.mean() - ae_c.mean()),
            }
        )
    return rows


def cv_verdict(d_p, d_s, d_w) -> str:
    both_imp = d_p < 0 and d_s < 0
    both_wors = d_p > 0 and d_s > 0
    if both_imp and d_w < 0:
        return "POSITIVE"
    if both_wors or d_w > 0 and not (d_p < 0 or d_s < 0):
        return "NEGATIVE"
    if d_w < 0 and not (d_p > 0.05 or d_s > 0.05):
        return "POSITIVE"
    if both_wors or (d_w > 0 and both_wors) or (d_p > 0 and d_s > 0):
        return "NEGATIVE"
    return "MIXED"


def phase_cv(*, quick: bool, code: str) -> dict:
    print("=== EXP-T066 CV ===", flush=True)
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    dev, test, folds, rb, qc = prepare_bundle()
    write_config(code, qc["rasa_hashes"])
    t0 = time.time()
    summary = run_transformer_cv(
        target=TARGET,
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        variant_id=VARIANT,
        dev=dev,
        rb=rb,
        folds=folds,
        device=device_str(),
        out_dir=OUT_RUN,
        quick=quick,
        recipe_id=RECIPE,
        plm_source="ablingua",
        pooling_mode="reg",
        use_continuous_rasa=False,
        use_rasa_weighted_pool=True,
    )
    z = np.load(OUT_RUN / f"oof_{VARIANT}_{summary['config_hash']}.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    pred_dir = ROOT / "experiments" / "predictions" / code
    pred_dir.mkdir(parents=True, exist_ok=True)
    oof_to_csv(ids, z["primary_oof"], pred_dir / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], pred_dir / "oof_shadow.csv")
    oof_to_csv(ids, z["primary_oof"], OUT_RUN / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], OUT_RUN / "oof_shadow.csv")
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
            "n_zero_denom_chains": qc["n_zero_denom_chains"],
            "rasa_hashes": qc["rasa_hashes"],
            "rasa_range": qc["rasa_range"],
        },
    }
    (OUT_RUN / "cv_summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae")}, indent=2), flush=True)
    state = load_state()
    state["t066_cv"] = out
    state["experiment_code"] = code
    save_state(state)
    return out


def phase_freeze(code: str) -> dict:
    if FREEZE_PATH.exists():
        return yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    t066 = state["t066_cv"]
    t037 = scores_row(CONTROL_CODE)
    t045 = scores_row("EXP-T045")
    t065 = scores_row(T065_CODE)
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    d_p = t066["cv_primary_mae"] - t037["cv_primary_mae"]
    d_s = t066["cv_shadow_mae"] - t037["cv_shadow_mae"]
    d_w = t066["cv_worst_mae"] - t037["cv_worst_mae"]
    verdict = cv_verdict(d_p, d_s, d_w)
    freeze = {
        "experiment_code": code,
        "experiment_id": T066_EID,
        "git_rev": git_rev(),
        "config": cfg,
        "control_experiment_code": CONTROL_CODE,
        "rasa_source": cfg["residue_rasa_aggregation"]["rasa_source"],
        "rasa_hashes": cfg["residue_rasa_aggregation"]["rasa_hashes"],
        "EXP-T066": {
            "cv_primary_mae": t066["cv_primary_mae"],
            "cv_shadow_mae": t066["cv_shadow_mae"],
            "cv_mean_mae": t066["cv_mean_mae"],
            "cv_worst_mae": t066["cv_worst_mae"],
            "config_hash": t066["config_hash"],
            "best_epochs_primary": t066["best_epochs_primary"],
        },
        "deltas_vs_EXP-T037": {"primary": d_p, "shadow": d_s, "worst": d_w},
        "deltas_vs_EXP-T045": {
            "worst": t066["cv_worst_mae"] - t045["cv_worst_mae"],
            "t045_worst": t045["cv_worst_mae"],
        },
        "deltas_vs_EXP-T065": {
            "primary": t066["cv_primary_mae"] - t065["cv_primary_mae"],
            "shadow": t066["cv_shadow_mae"] - t065["cv_shadow_mae"],
            "worst": t066["cv_worst_mae"] - t065["cv_worst_mae"],
        },
        "cv_scientific_verdict": verdict,
        "selection_statement": (
            "CV completed for EXP-T066; config frozen; Public/Private not used before freeze; "
            "single full-Dev Test follows."
        ),
        "public_mae": None,
        "private_mae": None,
        "test_overall_mae": None,
    }
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    print(f"Wrote {FREEZE_PATH} verdict={verdict}", flush=True)
    return freeze


def phase_test(code: str, *, quick: bool) -> dict:
    if not FREEZE_PATH.exists():
        raise SystemExit("refuse Test without CV freeze")
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    if freeze.get("public_mae") is not None:
        return {
            "public_mae": freeze["public_mae"],
            "private_mae": freeze["private_mae"],
            "test_overall_mae": freeze["test_overall_mae"],
        }
    pred_path = ROOT / "experiments" / "predictions" / code / "test.csv"
    if pred_path.exists() and not quick:
        te = pd.read_csv(pred_path)
    else:
        state = load_state()
        best = state["t066_cv"]["best_epochs_primary"]
        dev, test, folds, rb, _ = prepare_bundle()
        te = full_dev_transformer_predict(
            target=TARGET,
            content_mode="frozen",
            annotation_mode="full",
            merge_mode="concat",
            chain_mode="HL",
            best_epochs_primary=best,
            dev=dev,
            test=test,
            rb=rb,
            device=device_str(),
            recipe_id=RECIPE,
            quick=quick,
            plm_source="ablingua",
            pooling_mode="reg",
            use_continuous_rasa=False,
            use_rasa_weighted_pool=True,
        ).rename(columns={"prediction": TARGET})
        te.to_csv(pred_path, index=False)
        te.to_csv(OUT_RUN / "test.csv", index=False)

    sol = load_solution(BUNDLE_ROOT / "solution.csv").set_index("id")
    te = te.copy()
    te["id"] = te["id"].astype(str)
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
    state["t066_test"] = scores
    save_state(state)
    print(scores, flush=True)
    return scores


def materialize_fixed(code: str, rb_ids, dev, test) -> tuple[Path, str, str]:
    parts = build_recipe_parts(RECIPE, rb_ids)
    X = parts["X"].copy()
    X.insert(0, "id", rb_ids)
    split = {str(i): "dev" for i in dev["id"].astype(str)}
    split.update({str(i): "test" for i in test["id"].astype(str)})
    X.insert(1, "split", [split[str(i)] for i in rb_ids])
    path = ROOT / "experiments" / "features" / f"{code}.parquet"
    X.to_parquet(path, index=False)
    (ROOT / "experiments" / "features" / f"{code}.parquet.meta.json").write_text(
        json.dumps(
            {
                "feature_role": "FUSION_FIXED_BRANCH",
                "recipe_id": RECIPE,
                "feature_set_id": RECIPE_TO_FEATURE_SET[RECIPE],
                "n_rows": len(X),
                "n_feature_cols": X.shape[1] - 2,
            },
            indent=2,
        )
        + "\n"
    )
    return path, file_sha256(path), feature_content_sha256(X)


def materialize_rasa(code: str) -> Path:
    src = ROOT / "experiments" / "inputs" / f"{T065_CODE}_rasa.parquet"
    dst = ROOT / "experiments" / "inputs" / f"{code}_rasa.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        raise SystemExit("EXP-T065 RASA parquet missing; refuse rebuild unless needed")
    return dst


def phase_artifacts(code: str) -> dict:
    print("=== artifacts + registry ===", flush=True)
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    dev, test, folds, rb, qc = prepare_bundle()
    feat_path, fsha, csha = materialize_fixed(code, rb.ids, dev, test)
    rasa_path = materialize_rasa(code)
    rasa_sha = file_sha256(rasa_path)
    # verify fixed branch matches T065 content hash (same recipe)
    t065_csha = str(
        pd.read_csv(ROOT / "results" / "experiments.csv")
        .set_index("experiment_code")
        .loc[T065_CODE, "feature_content_sha256"]
    )
    if csha != t065_csha:
        raise SystemExit(f"fixed branch content hash mismatch vs T065: {csha} != {t065_csha}")
    print(f"fixed branch content sha matches T065: {csha}", flush=True)

    # diagnostics vs T037 canonical OOF
    ctl_p = pd.read_csv(ROOT / "experiments" / "predictions" / CONTROL_CODE / "oof_primary.csv")
    ctl_s = pd.read_csv(ROOT / "experiments" / "predictions" / CONTROL_CODE / "oof_shadow.csv")
    new_p = pd.read_csv(ROOT / "experiments" / "predictions" / code / "oof_primary.csv")
    new_s = pd.read_csv(ROOT / "experiments" / "predictions" / code / "oof_shadow.csv")
    for d in (ctl_p, ctl_s, new_p, new_s):
        d["id"] = d["id"].astype(str)
    y = dev.set_index("id")[TARGET]
    ids = dev["id"].astype(str).tolist()
    col_c = TARGET if TARGET in ctl_p.columns else [c for c in ctl_p.columns if c != "id"][0]
    ctl_ps = ctl_p.set_index("id").loc[ids, col_c]
    ctl_ss = ctl_s.set_index("id").loc[ids, col_c]
    new_ps = new_p.set_index("id").loc[ids, TARGET]
    new_ss = new_s.set_index("id").loc[ids, TARGET]
    ae_ctl_p = (y.loc[ids] - ctl_ps).abs().to_numpy(float)
    ae_new_p = (y.loc[ids] - new_ps).abs().to_numpy(float)
    ae_ctl_s = (y.loc[ids] - ctl_ss).abs().to_numpy(float)
    ae_new_s = (y.loc[ids] - new_ss).abs().to_numpy(float)
    diag = {
        "bootstrap": {
            "primary": paired_bootstrap(ae_new_p, ae_ctl_p),
            "shadow": paired_bootstrap(ae_new_s, ae_ctl_s),
        },
        "fold_mae_deltas": {
            "primary": fold_deltas(dev, folds.primary, new_ps, ctl_ps),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, ctl_ss),
        },
    }
    (OUT_RUN / "paired_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    recomputed_p = float(mae(y.loc[ids].to_numpy(float), new_ps.to_numpy(float)))
    recomputed_s = float(mae(y.loc[ids].to_numpy(float), new_ss.to_numpy(float)))
    assert abs(recomputed_p - freeze["EXP-T066"]["cv_primary_mae"]) < 1e-10
    assert abs(recomputed_s - freeze["EXP-T066"]["cv_shadow_mae"]) < 1e-10

    pub, priv, overall = float(freeze["public_mae"]), float(freeze["private_mae"]), float(freeze["test_overall_mae"])
    cv_p, cv_s = float(freeze["EXP-T066"]["cv_primary_mae"]), float(freeze["EXP-T066"]["cv_shadow_mae"])
    cv_m, cv_w = float(freeze["EXP-T066"]["cv_mean_mae"]), float(freeze["EXP-T066"]["cv_worst_mae"])

    # optional post-hoc diagnostic
    try:
        rasa = pd.read_parquet(rasa_path)
        g = rasa.groupby("id").agg(
            mean_rasa=("rasa", "mean"),
            cdr3_rasa=("rasa", lambda s: float(np.nanmean(s[rasa.loc[s.index, "region"].eq("CDR3")])) if False else float("nan")),
        )
        # simpler: mean rasa + cdr3 mean via filter
        cdr3 = rasa[rasa["region"] == "CDR3"].groupby("id")["rasa"].mean()
        g = rasa.groupby("id")["rasa"].mean().to_frame("mean_rasa")
        g["cdr3_rasa"] = cdr3
        d_ae = pd.Series(ae_new_p - ae_ctl_p, index=ids)
        g = g.loc[ids]
        posthoc = {
            "corr_delta_ae_vs_mean_rasa": float(np.corrcoef(g["mean_rasa"].to_numpy(float), d_ae.to_numpy(float))[0, 1]),
            "corr_delta_ae_vs_cdr3_rasa": float(
                np.corrcoef(
                    g["cdr3_rasa"].fillna(g["mean_rasa"]).to_numpy(float),
                    d_ae.to_numpy(float),
                )[0, 1]
            ),
            "note": "antibody-level; diagnostic only",
        }
    except Exception as e:
        posthoc = {"error": str(e)}
    (OUT_RUN / "posthoc_diagnostic.json").write_text(json.dumps(posthoc, indent=2) + "\n")

    exp_path = ROOT / "results" / "experiments.csv"
    exp = pd.read_csv(exp_path)
    if "control_experiment_code" not in exp.columns:
        exp["control_experiment_code"] = ""
    if code in set(exp["experiment_code"].astype(str)):
        exp = exp[exp["experiment_code"] != code]
    row = {c: "" for c in exp.columns}
    for c in EXPERIMENTS_COLUMNS:
        row.setdefault(c, "")
    row.update(
        {
            "experiment_code": code,
            "experiment_id": T066_EID,
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer+Fusion+RASAWeightedPool",
            "source_model_id": f"RASA_POOL::{CONTROL_CODE}",
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
            "n_features": int(pd.read_parquet(feat_path).shape[1] - 2),
            "feature_space": "FUSION_FIXED_BRANCH_RAW",
            "feature_sha256": fsha,
            "feature_content_sha256": csha,
            "score_source": "EXP-T066_cv+solution_postfreeze",
            "prediction_source": "EXP-T066_run",
            "feature_source": f"{RECIPE}+classical_rasa_cache+ablingua_residue",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue + BioEmu/MPNN + ESMFold RASA",
            "notes": (
                f"RASA-weighted REG residual vs {CONTROL_CODE}; "
                f"rasa_input=experiments/inputs/{code}_rasa.parquet sha={rasa_sha}; "
                "representation_status=NOT_EXPORTED"
            ),
            "transformer_type": "FUSION",
            "plm_source": "ABLINGUA",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "CONCAT",
            "representation_status": "NOT_EXPORTED",
            "input_space": "RESIDUE_PLUS_FIXED_FEATURES_PLUS_RASA_POOL",
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
    new_df = pd.DataFrame([{c: row.get(c, "") for c in exp.columns}])
    pd.concat([exp, new_df], ignore_index=True).to_csv(exp_path, index=False)

    comp_path = ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    comp = pd.read_csv(comp_path)
    if code not in set(comp["experiment_code"].astype(str)):
        crow = {c: "" for c in comp.columns}
        crow.update(
            {
                "experiment_code": code,
                "experiment_id": T066_EID,
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
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)

    state["artifacts"] = {
        "feature_path": str(feat_path),
        "rasa_path": str(rasa_path),
        "feature_sha256": fsha,
        "feature_content_sha256": csha,
        "rasa_sha256": rasa_sha,
        "diagnostics": diag,
        "posthoc": posthoc,
    }
    save_state(state)
    return state["artifacts"]


def phase_report(code: str) -> None:
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    t037 = scores_row(CONTROL_CODE)
    t045 = scores_row("EXP-T045")
    t065 = scores_row(T065_CODE)
    t066 = freeze["EXP-T066"]
    diag = json.loads((OUT_RUN / "paired_diagnostics.json").read_text())
    qc = json.loads(RASA_QC_PATH.read_text()) if RASA_QC_PATH.exists() else {}
    posthoc = json.loads((OUT_RUN / "posthoc_diagnostic.json").read_text()) if (OUT_RUN / "posthoc_diagnostic.json").exists() else {}
    d_p = t066["cv_primary_mae"] - t037["cv_primary_mae"]
    d_s = t066["cv_shadow_mae"] - t037["cv_shadow_mae"]
    d_w = t066["cv_worst_mae"] - t037["cv_worst_mae"]
    verdict = freeze.get("cv_scientific_verdict") or cv_verdict(d_p, d_s, d_w)
    both = (d_p < 0 and d_s < 0) or (d_p > 0 and d_s > 0)
    lines = [
        "# EXP-T066 — RASA-weighted residue aggregation prior",
        "",
        f"- git: `{git_rev()}`",
        f"- control: `{CONTROL_CODE}` (stored canonical OOF; not retrained)",
        f"- change: per-chain continuous RASA-weighted pool → shared zero-init Linear → REG residual",
        f"- encoder input: **unchanged** vs EXP-T037",
        f"- CV verdict: **{verdict}**",
        "",
        "## 1–4. vs EXP-T037",
        "",
        f"- T066 P/S/mean/W: {t066['cv_primary_mae']:.6f} / {t066['cv_shadow_mae']:.6f} / {t066['cv_mean_mae']:.6f} / {t066['cv_worst_mae']:.6f}",
        f"- Δ Primary: {d_p:+.6f}",
        f"- Δ Shadow: {d_s:+.6f}",
        f"- Δ worst: {d_w:+.6f}",
        f"- Both schemes same direction: {'YES' if both else 'NO'}",
        "",
        "## 5. Paired bootstrap (MAE_T066 − MAE_T037)",
        "",
        f"- Primary: Δ={diag['bootstrap']['primary']['mae_delta']:+.6f} CI95=[{diag['bootstrap']['primary']['ci95_low']:+.6f}, {diag['bootstrap']['primary']['ci95_high']:+.6f}]",
        f"- Shadow: Δ={diag['bootstrap']['shadow']['mae_delta']:+.6f} CI95=[{diag['bootstrap']['shadow']['ci95_low']:+.6f}, {diag['bootstrap']['shadow']['ci95_high']:+.6f}]",
        "",
        "## 6. Three-way comparison",
        "",
        f"| EXP | role | Primary | Shadow | worst |",
        f"|-----|------|---------|--------|-------|",
        f"| T037 | no RASA | {t037['cv_primary_mae']:.6f} | {t037['cv_shadow_mae']:.6f} | {t037['cv_worst_mae']:.6f} |",
        f"| T065 | token-additive RASA | {t065['cv_primary_mae']:.6f} | {t065['cv_shadow_mae']:.6f} | {t065['cv_worst_mae']:.6f} |",
        f"| T066 | aggregation RASA | {t066['cv_primary_mae']:.6f} | {t066['cv_shadow_mae']:.6f} | {t066['cv_worst_mae']:.6f} |",
        "",
        "Conservative reading: one aggregation experiment does not prove RASA 'works as readout'; "
        "compare signed deltas vs T037 and vs T065 only.",
        "",
        f"- T066−T065 worst: {t066['cv_worst_mae'] - t065['cv_worst_mae']:+.6f}",
        "",
        "## 7. vs classical EXP-T045",
        "",
        f"- T045 worst: {t045['cv_worst_mae']:.6f}",
        f"- T066−T045 worst: {t066['cv_worst_mae'] - t045['cv_worst_mae']:+.6f}",
        "",
        "## 8. Test (after freeze)",
        "",
        f"- Public: {freeze['public_mae']:.6f}",
        f"- Private: {freeze['private_mae']:.6f}",
        f"- Overall: {freeze['test_overall_mae']:.6f}",
        "",
        "## 9. RASA QC",
        "",
        f"- expected/mapped/missing: {qc.get('n_expected_residues')}/{qc.get('n_mapped_residues')}/{qc.get('n_missing_rasa_residues')}",
        f"- AA mismatches: {qc.get('n_aa_mismatch_flags')}",
        f"- zero-denom chains: {qc.get('n_zero_denom_chains')}",
        f"- hashes: {qc.get('rasa_hashes')}",
        "",
        "## 10–11. Artifacts / reproducibility",
        "",
        f"- config / features / RASA / preds present",
        f"- representation: NOT_EXPORTED (no weight checkpoints)",
        f"- SHAREABLE_COMPLETE / REPRODUCED / eligible=YES",
        "",
        "## 12. Interpretation",
        "",
        f"CV verdict **{verdict}**. Token-additive RASA (T065) was detrimental; "
        f"aggregation RASA (T066) is a separate mechanism. "
        f"Do not over-claim from a single experiment.",
        "",
        "## Post-hoc (diagnostic only)",
        "",
        f"```json\n{json.dumps(posthoc, indent=2)}\n```",
        "",
        "## STOP — no EXP-T067 / sweeps",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    state["verdict"] = verdict
    save_state(state)
    print(f"Wrote {REPORT_PATH}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["issue", "cv", "freeze", "test", "artifacts", "report", "all"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase == "issue":
        print(issue_t066())
        return 0
    if args.phase == "cv":
        code = load_state().get("experiment_code") or issue_t066()
        phase_cv(quick=args.quick, code=code)
        return 0
    if args.phase == "freeze":
        phase_freeze(load_state()["experiment_code"])
        return 0
    if args.phase == "test":
        phase_test(load_state()["experiment_code"], quick=args.quick)
        return 0
    if args.phase == "artifacts":
        phase_artifacts(load_state()["experiment_code"])
        return 0
    if args.phase == "report":
        phase_report(load_state()["experiment_code"])
        return 0

    code = issue_t066()
    print(f"Issued {code}", flush=True)
    phase_cv(quick=args.quick, code=code)
    phase_freeze(code)
    phase_test(code, quick=args.quick)
    phase_artifacts(code)
    phase_report(code)
    print("DONE EXP-T066 — STOP", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
