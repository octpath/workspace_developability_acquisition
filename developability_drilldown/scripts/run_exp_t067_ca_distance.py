#!/usr/bin/env python3
"""EXP-T067: Learnable Cα-distance attention bias (vs EXP-T037).

Does NOT retrain EXP-T037 (use stored canonical OOF). One experiment then STOP.
"""
from __future__ import annotations

import argparse
import json
import os
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
from classical_features.ca_cache import build_or_load_ca_cache  # noqa: E402
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import (  # noqa: E402
    AA_TO_IDX,
    BUNDLE_ROOT,
    REGION_TO_IDX,
    RECIPE_TO_FEATURE_SET,
)
from antibody_transformer.data import (  # noqa: E402
    attach_ca_coords,
    load_annotations,
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
T045_CODE = "EXP-T045"
T065_CODE = "EXP-T065"
T066_CODE = "EXP-T066"
T067_EID = "TRF_TM_ABLINGUA_CA_DISTANCE_FULL_CONCAT_FUS_BIOEMU_MPNN"
RECIPE = "TM_BASE_BIOEMU_MPNN__RIDGE"
VARIANT = "T067_CA_DIST"

OUT_RUN = ROOT / "results" / "EXP-T067_run"
FREEZE_PATH = ROOT / "results" / "EXP-T067_CV_FREEZE.yaml"
REPORT_PATH = ROOT / "results" / "EXP-T067_CA_DISTANCE_REPORT.md"
CA_QC_PATH = ROOT / "results" / "EXP-T067_CA_QC.json"
DIST_PARAMS_PATH = ROOT / "results" / "EXP-T067_DISTANCE_PARAMETERS.csv"
STATE_PATH = ROOT / "results" / "EXP-T067_run_state.json"
DIAG_PATH = OUT_RUN / "distance_usage_diagnostic.json"


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


def ca_file_hash() -> dict[str, str]:
    cache = ROOT / "experiments" / "classical_cache"
    return {
        n: file_sha256(cache / n)
        for n in ("ca_heavy.npy", "ca_light.npy", "ca_ids.npy", "ca_meta.json")
        if (cache / n).exists()
    }


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
    ca = build_or_load_ca_cache(dev, test)
    seqs = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
    qc = attach_ca_coords(
        rb,
        ca_heavy=ca["H"],
        ca_light=ca["L"],
        ca_ids=ca["ids"],
        seqs=seqs,
    )
    qc["ca_meta"] = ca.get("meta", {})
    qc["ca_hashes"] = ca_file_hash()
    qc["source_path"] = str(ROOT / "experiments" / "classical_cache")
    qc["structure_source"] = ca.get("meta", {}).get(
        "source", "/workspace_developability_acquisition/feature_extension/data/esmfold_fv"
    )
    qc["ca_definition"] = ca.get("meta", {}).get(
        "method", "Bio.PDB CA coordinates; chain map H/A L/B then sequence match"
    )
    qc["units"] = ca.get("meta", {}).get("units", "angstrom")
    finite = []
    for arr, mask in ((rb.heavy_ca, rb.heavy_mask), (rb.light_ca, rb.light_mask)):
        for i in range(len(rb.ids)):
            n = int(mask[i].sum())
            xyz = arr[i, :n]
            ok = np.isfinite(xyz).all(axis=-1)
            if ok.any():
                finite.append(xyz[ok])
    if finite:
        all_xyz = np.concatenate(finite, axis=0)
        qc["ca_coord_range"] = {
            "min": all_xyz.min(axis=0).tolist(),
            "max": all_xyz.max(axis=0).tolist(),
            "mean": all_xyz.mean(axis=0).tolist(),
        }
    else:
        qc["ca_coord_range"] = None
    CA_QC_PATH.write_text(json.dumps(qc, indent=2, default=str) + "\n")
    if qc["n_aa_mismatch_flags"]:
        raise SystemExit(f"AA mismatch: {qc['aa_mismatch_ids']}")
    return dev, test, folds, rb, qc


def issue_t067() -> str:
    codes = load_codes()
    if (codes["experiment_id"] == T067_EID).any():
        return str(codes.set_index("experiment_id").loc[T067_EID, "experiment_code"])
    nxt = next_code(TARGET)
    if nxt != "EXP-T067":
        raise SystemExit(f"expected EXP-T067, got {nxt}")
    code = issue_code(
        T067_EID,
        TARGET,
        source_model_id=f"CA_DISTANCE::{CONTROL_CODE}",
        phase="ARCHITECTURE_CA_DISTANCE",
        notes="Cα-distance attention bias vs EXP-T037; no HPO; encoder topology unchanged (H/L separate)",
    )
    if code != "EXP-T067":
        raise SystemExit(code)
    return code


def write_config(code: str, ca_hashes: dict) -> Path:
    cfg = {
        "experiment_code": code,
        "experiment_id": T067_EID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"CA_DISTANCE::{CONTROL_CODE}",
        "transformer_type": "FUSION",
        "input_space": "RESIDUE_PLUS_FIXED_FEATURES_PLUS_CA_DISTANCE",
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
        "representation_note": (
            "NOT_EXPORTED; full-Dev weight checkpoints may be saved under "
            "results/EXP-T067_run/checkpoints/ if size permits"
        ),
        "fusion_feature_set_id": RECIPE_TO_FEATURE_SET[RECIPE],
        "fusion_projection_dim": 64,
        "fusion_recipe": RECIPE,
        "control_experiment_code": CONTROL_CODE,
        "pairwise_geometry": {
            "enabled": True,
            "geometry_source": "CA_COORDINATES",
            "distance_metric": "EUCLIDEAN_ANGSTROM",
            "distance_kernel": "EXPONENTIAL",
            "distance_kernel_count_per_head": 1,
            "distance_amplitude_init": "ZERO",
            "distance_scale": "LEARNABLE_PER_HEAD",
            "distance_scale_init": "TRAIN_MEDIAN_PAIR_DISTANCE",
            "distance_kernel_shared_across_layers": True,
            "cross_chain_distance": False,
            "self_pair_bias": False,
            "ca_source": "experiments/classical_cache",
            "ca_hashes": ca_hashes,
            "control_experiment_code": CONTROL_CODE,
            "encoder_topology_unchanged": True,
            "within_chain_only": True,
        },
    }
    path = ROOT / "experiments" / "configs" / f"{code}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "mae_t067": float(ae_n.mean()),
                "mae_control": float(ae_c.mean()),
                "delta": float(ae_n.mean() - ae_c.mean()),
            }
        )
    return rows


def cv_verdict(d_p: float, d_s: float, d_w: float) -> str:
    """delta = T067 - T037 (negative = improvement)."""
    both_imp = d_p < 0 and d_s < 0
    both_wors = d_p > 0 and d_s > 0
    if both_imp and d_w < 0:
        return "POSITIVE"
    # weaker positive: worst improves without major opposing-split deterioration
    if d_w < 0 and not (d_p > 0.05 or d_s > 0.05):
        return "POSITIVE"
    if both_wors or (d_w > 0 and not (d_p < 0 or d_s < 0)):
        return "NEGATIVE"
    if (d_p < 0) != (d_s < 0):
        return "MIXED"
    return "MIXED"


def write_distance_params_csv(rows: list[dict]) -> Path:
    DIST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DIST_PARAMS_PATH, index=False)
    return DIST_PARAMS_PATH


def distance_param_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_rows": 0}
    return {
        "n_rows": int(len(df)),
        "mean_abs_a_h": float(df["a_h"].abs().mean()),
        "median_abs_a_h": float(df["a_h"].abs().median()),
        "mean_a_h": float(df["a_h"].mean()),
        "median_ell_h_angstrom": float(df["ell_h_angstrom"].median()),
        "mean_ell_h_angstrom": float(df["ell_h_angstrom"].mean()),
        "median_initial_ell": float(df["initial_ell_angstrom"].median())
        if "initial_ell_angstrom" in df.columns and df["initial_ell_angstrom"].notna().any()
        else None,
        "by_head": {
            int(h): {
                "mean_abs_a": float(g["a_h"].abs().mean()),
                "median_ell": float(g["ell_h_angstrom"].median()),
            }
            for h, g in df.groupby("head")
        },
    }


def collect_within_chain_pair_stats(rb) -> dict:
    """Data-driven Cα pair distances + sequence-separation categories (diagnostic)."""
    dist_all: list[np.ndarray] = []
    neighbor_d: list[float] = []
    local_d: list[float] = []
    nonlocal_d: list[float] = []
    for i in range(len(rb.ids)):
        for ca, mask in ((rb.heavy_ca[i], rb.heavy_mask[i]), (rb.light_ca[i], rb.light_mask[i])):
            n = int(mask.sum())
            if n < 2:
                continue
            xyz = ca[:n]
            ok = np.isfinite(xyz).all(axis=1)
            if int(ok.sum()) < 2:
                continue
            # Keep original indices for |i-j|; only use finite residues
            idx = np.where(ok)[0]
            xyz_ok = xyz[idx]
            d = np.linalg.norm(xyz_ok[:, None, :] - xyz_ok[None, :, :], axis=-1)
            m = len(idx)
            iu, ju = np.triu_indices(m, k=1)
            dists = d[iu, ju].astype(np.float64)
            dist_all.append(dists)
            sep = np.abs(idx[iu].astype(np.int64) - idx[ju].astype(np.int64))
            for dij, s in zip(dists, sep):
                if s == 1:
                    neighbor_d.append(float(dij))
                elif 1 < s <= 4:
                    local_d.append(float(dij))
                else:
                    nonlocal_d.append(float(dij))
    if not dist_all:
        raise RuntimeError("no valid within-chain Cα pairs for diagnostic")
    all_d = np.concatenate(dist_all)
    qs = {f"q{p}": float(np.quantile(all_d, p / 100.0)) for p in (10, 25, 50, 75, 90)}
    return {
        "n_pairs": int(all_d.size),
        "distance_quantiles_angstrom": qs,
        "pair_category_counts": {
            "sequence_neighbor_|i-j|=1": len(neighbor_d),
            "local_1<|i-j|<=4": len(local_d),
            "nonlocal_|i-j|>4": len(nonlocal_d),
        },
        "pair_category_mean_distance_angstrom": {
            "sequence_neighbor_|i-j|=1": float(np.mean(neighbor_d)) if neighbor_d else None,
            "local_1<|i-j|<=4": float(np.mean(local_d)) if local_d else None,
            "nonlocal_|i-j|>4": float(np.mean(nonlocal_d)) if nonlocal_d else None,
        },
    }


def write_distance_usage_diagnostic(rb, dist_df: pd.DataFrame) -> dict:
    """Post-hoc geometry usage diagnostic (after freeze; no selection)."""
    pair_stats = collect_within_chain_pair_stats(rb)
    qs = pair_stats["distance_quantiles_angstrom"]
    cv_df = dist_df[dist_df["cv_scheme"].isin(["primary", "shadow"])].copy() if not dist_df.empty else dist_df
    mean_abs_a = float(cv_df["a_h"].abs().mean()) if len(cv_df) else float("nan")
    median_ell = float(cv_df["ell_h_angstrom"].median()) if len(cv_df) else float("nan")

    # Bias curve at data-driven quantiles using mean |a| and median ell (summary)
    bias_curve_summary = {
        qk: float(mean_abs_a * np.exp(-qv / median_ell)) if np.isfinite(mean_abs_a) and median_ell > 0 else None
        for qk, qv in qs.items()
    }

    # Per-head curves (mean |a_h|, median ell_h across fold/seed for that head)
    per_head = {}
    if len(cv_df):
        for h, g in cv_df.groupby("head"):
            ah = float(g["a_h"].abs().mean())
            ell = float(g["ell_h_angstrom"].median())
            per_head[int(h)] = {
                "mean_abs_a_h": ah,
                "median_ell_h_angstrom": ell,
                "bias_at_quantiles": {
                    qk: float(ah * np.exp(-qv / ell)) if ell > 0 else None for qk, qv in qs.items()
                },
            }

    # Category bias using mean category distances with summary a/ell
    cat_means = pair_stats["pair_category_mean_distance_angstrom"]
    category_bias = {}
    for name, dmean in cat_means.items():
        if dmean is None or not (np.isfinite(mean_abs_a) and median_ell > 0):
            category_bias[name] = None
        else:
            category_bias[name] = float(mean_abs_a * np.exp(-float(dmean) / median_ell))

    # Apparent dominance by category mean bias magnitude
    ranked = sorted(
        ((k, abs(v) if v is not None else -1.0) for k, v in category_bias.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    dominant = ranked[0][0] if ranked and ranked[0][1] >= 0 else None

    out = {
        "note": (
            "Diagnostic only; not used for selection. "
            "Within-chain pairs only (no H-L). "
            "Bias kernel: a*exp(-d/ell); summary uses mean |a_h| and median ell."
        ),
        "summary_params": {
            "mean_abs_a_h": mean_abs_a,
            "median_ell_h_angstrom": median_ell,
            "n_param_rows_used": int(len(cv_df)),
        },
        "pair_stats": pair_stats,
        "bias_curve_at_distance_quantiles": bias_curve_summary,
        "per_head": per_head,
        "category_mean_bias": category_bias,
        "apparent_geometry_use_dominant_category": dominant,
        "cross_chain_hl": False,
    }
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    DIAG_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def phase_cv(*, quick: bool, code: str) -> dict:
    print("=== EXP-T067 CV ===", flush=True)
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    dev, test, folds, rb, qc = prepare_bundle()
    write_config(code, qc["ca_hashes"])
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
        use_rasa_weighted_pool=False,
        use_ca_distance_bias=True,
    )
    z = np.load(OUT_RUN / f"oof_{VARIANT}_{summary['config_hash']}.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    pred_dir = ROOT / "experiments" / "predictions" / code
    pred_dir.mkdir(parents=True, exist_ok=True)
    oof_to_csv(ids, z["primary_oof"], pred_dir / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], pred_dir / "oof_shadow.csv")
    oof_to_csv(ids, z["primary_oof"], OUT_RUN / "oof_primary.csv")
    oof_to_csv(ids, z["shadow_oof"], OUT_RUN / "oof_shadow.csv")

    dist_rows = list(summary.get("distance_param_rows") or [])
    write_distance_params_csv(dist_rows)

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
        "n_distance_param_rows": len(dist_rows),
        "distance_param_summary": distance_param_summary(pd.DataFrame(dist_rows)),
        "ca_qc": {
            "n_expected_residues": qc["n_expected_residues"],
            "n_mapped_ca_residues": qc["n_mapped_ca_residues"],
            "n_missing_ca_residues": qc["n_missing_ca_residues"],
            "n_aa_mismatch_flags": qc["n_aa_mismatch_flags"],
            "ca_hashes": qc["ca_hashes"],
            "structure_source": qc["structure_source"],
            "ca_coord_range": qc.get("ca_coord_range"),
        },
    }
    (OUT_RUN / "cv_summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {k: out[k] for k in ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae")},
            indent=2,
        ),
        flush=True,
    )
    state = load_state()
    state["t067_cv"] = out
    state["experiment_code"] = code
    state["distance_param_rows_cv"] = dist_rows
    save_state(state)
    return out


def phase_freeze(code: str) -> dict:
    if FREEZE_PATH.exists():
        return yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    t067 = state["t067_cv"]
    t037 = scores_row(CONTROL_CODE)
    t045 = scores_row(T045_CODE)
    t065 = scores_row(T065_CODE)
    t066 = scores_row(T066_CODE)
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    d_p = t067["cv_primary_mae"] - t037["cv_primary_mae"]
    d_s = t067["cv_shadow_mae"] - t037["cv_shadow_mae"]
    d_w = t067["cv_worst_mae"] - t037["cv_worst_mae"]
    verdict = cv_verdict(d_p, d_s, d_w)

    # Post-hoc distance diagnostic (after CV; before Test)
    _, _, _, rb, qc = prepare_bundle()
    dist_df = pd.read_csv(DIST_PARAMS_PATH) if DIST_PARAMS_PATH.exists() else pd.DataFrame()
    diag = write_distance_usage_diagnostic(rb, dist_df)

    freeze = {
        "experiment_code": code,
        "experiment_id": T067_EID,
        "git_rev": git_rev(),
        "config": cfg,
        "control_experiment_code": CONTROL_CODE,
        "ca_source": cfg["pairwise_geometry"]["ca_source"],
        "ca_hashes": cfg["pairwise_geometry"]["ca_hashes"],
        "structure_source": qc.get("structure_source"),
        "EXP-T067": {
            "cv_primary_mae": t067["cv_primary_mae"],
            "cv_shadow_mae": t067["cv_shadow_mae"],
            "cv_mean_mae": t067["cv_mean_mae"],
            "cv_worst_mae": t067["cv_worst_mae"],
            "config_hash": t067["config_hash"],
            "best_epochs_primary": t067["best_epochs_primary"],
            "distance_param_summary": t067.get("distance_param_summary"),
        },
        "deltas_vs_EXP-T037": {"primary": d_p, "shadow": d_s, "worst": d_w},
        "deltas_vs_EXP-T045": {
            "worst": t067["cv_worst_mae"] - t045["cv_worst_mae"],
            "t045_worst": t045["cv_worst_mae"],
        },
        "deltas_vs_EXP-T065": {
            "primary": t067["cv_primary_mae"] - t065["cv_primary_mae"],
            "shadow": t067["cv_shadow_mae"] - t065["cv_shadow_mae"],
            "worst": t067["cv_worst_mae"] - t065["cv_worst_mae"],
        },
        "deltas_vs_EXP-T066": {
            "primary": t067["cv_primary_mae"] - t066["cv_primary_mae"],
            "shadow": t067["cv_shadow_mae"] - t066["cv_shadow_mae"],
            "worst": t067["cv_worst_mae"] - t066["cv_worst_mae"],
        },
        "cv_scientific_verdict": verdict,
        "distance_usage_diagnostic_path": str(DIAG_PATH.relative_to(ROOT)),
        "distance_usage_summary": {
            "mean_abs_a_h": diag["summary_params"]["mean_abs_a_h"],
            "median_ell_h_angstrom": diag["summary_params"]["median_ell_h_angstrom"],
            "dominant_category": diag.get("apparent_geometry_use_dominant_category"),
        },
        "selection_statement": (
            "CV completed for EXP-T067; config frozen; Public/Private not used before freeze; "
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
    ckpt_dir = OUT_RUN / "checkpoints"
    fulldev_rows: list[dict] = []
    if pred_path.exists() and not quick:
        te = pd.read_csv(pred_path)
        # recover fulldev distance rows if previously written
        fd_csv = ckpt_dir / "distance_params_fulldev.csv"
        if fd_csv.exists():
            fulldev_rows = pd.read_csv(fd_csv).to_dict(orient="records")
    else:
        state = load_state()
        best = state["t067_cv"]["best_epochs_primary"]
        dev, test, folds, rb, _ = prepare_bundle()
        te_df = full_dev_transformer_predict(
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
            use_rasa_weighted_pool=False,
            use_ca_distance_bias=True,
            checkpoint_dir=ckpt_dir,
        )
        te = te_df.rename(columns={"prediction": TARGET})
        te.to_csv(pred_path, index=False)
        te.to_csv(OUT_RUN / "test.csv", index=False)
        fulldev_rows = list(getattr(te_df, "attrs", {}).get("distance_param_rows") or [])
        if not fulldev_rows:
            fd_csv = ckpt_dir / "distance_params_fulldev.csv"
            if fd_csv.exists():
                fulldev_rows = pd.read_csv(fd_csv).to_dict(orient="records")

    # Merge CV + fulldev distance parameters
    state = load_state()
    cv_rows = list(state.get("distance_param_rows_cv") or [])
    if not cv_rows and DIST_PARAMS_PATH.exists():
        prev = pd.read_csv(DIST_PARAMS_PATH)
        cv_rows = prev[prev["cv_scheme"].isin(["primary", "shadow"])].to_dict(orient="records")
    all_rows = cv_rows + fulldev_rows
    write_distance_params_csv(all_rows)
    state["distance_param_rows_fulldev"] = fulldev_rows

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
    ckpts = sorted(ckpt_dir.glob("fulldev_seed*.pt")) if ckpt_dir.exists() else []
    freeze["checkpoints"] = {
        "dir": str(ckpt_dir.relative_to(ROOT)) if ckpt_dir.exists() else None,
        "n_fulldev": len(ckpts),
        "paths": [str(p.relative_to(ROOT)) for p in ckpts],
        "total_bytes": int(sum(p.stat().st_size for p in ckpts)) if ckpts else 0,
    }
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    state["t067_test"] = scores
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def materialize_ca_parquet(code: str, rb, dev: pd.DataFrame, test: pd.DataFrame) -> Path:
    """Long-form Cα coords; annotations from rb arrays (+ length check vs annotations.parquet)."""
    ann = load_annotations()
    idx_to_aa = {v: k for k, v in AA_TO_IDX.items()}
    idx_to_region = {v: k for k, v in REGION_TO_IDX.items()}
    imgt_rev = {v: k for k, v in rb.imgt_vocab.items()}
    split = {str(i): "dev" for i in dev["id"].astype(str)}
    split.update({str(i): "test" for i in test["id"].astype(str)})
    rows = []
    for i, ab in enumerate(rb.ids):
        for chain, aa_arr, mask, imgt_arr, reg_arr, ca_arr in (
            ("H", rb.heavy_aa[i], rb.heavy_mask[i], rb.heavy_imgt[i], rb.heavy_region[i], rb.heavy_ca[i]),
            ("L", rb.light_aa[i], rb.light_mask[i], rb.light_imgt[i], rb.light_region[i], rb.light_ca[i]),
        ):
            n = int(mask.sum())
            for j in range(n):
                xyz = ca_arr[j]
                finite = bool(np.isfinite(xyz).all())
                rows.append(
                    {
                        "id": ab,
                        "split": split[ab],
                        "chain": chain,
                        "seq_index": j,
                        "aa": idx_to_aa.get(int(aa_arr[j]), "X"),
                        "imgt_position": imgt_rev.get(int(imgt_arr[j]), "UNKNOWN"),
                        "region": idx_to_region.get(int(reg_arr[j]), "UNKNOWN"),
                        "ca_x": float(xyz[0]) if finite else np.nan,
                        "ca_y": float(xyz[1]) if finite else np.nan,
                        "ca_z": float(xyz[2]) if finite else np.nan,
                    }
                )
    df = pd.DataFrame(rows)

    # Join / sanity vs residue_level/annotations.parquet
    ann = ann.copy()
    ann["id"] = ann["id"].astype(str)
    for ab in rb.ids[:5]:
        n_ann = len(ann[ann.id == ab])
        n_df = len(df[df.id == ab])
        if n_ann != n_df:
            raise SystemExit(f"CA parquet length mismatch {ab}: {n_df} vs ann {n_ann}")
    # Optional column-level join check on aa/region for a sample
    sample_ids = rb.ids[:3]
    for ab in sample_ids:
        left = df[df.id == ab].sort_values(["chain", "seq_index"]).reset_index(drop=True)
        right = ann[ann.id == ab].sort_values(["chain", "seq_index"]).reset_index(drop=True)
        if not (left["aa"].to_numpy() == right["aa"].to_numpy()).all():
            raise SystemExit(f"CA parquet AA mismatch vs annotations for {ab}")

    out = ROOT / "experiments" / "inputs" / f"{code}_ca.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "coordinate_source": "experiments/classical_cache + ESMFold Fv CA",
        "n_rows": len(df),
        "n_missing_ca": int(df["ca_x"].isna().sum()),
        "schema": list(df.columns),
        "content_sha256_note": "file sha recorded in artifacts; pairwise distances deterministic from coords",
    }
    (ROOT / "experiments" / "inputs" / f"{code}_ca.parquet.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    return out


def phase_artifacts(code: str) -> dict:
    print("=== artifacts + registry ===", flush=True)
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    dev, test, folds, rb, qc = prepare_bundle()
    feat_path, fsha, csha = materialize_fixed(code, rb.ids, dev, test)
    ca_path = materialize_ca_parquet(code, rb, dev, test)
    ca_sha = file_sha256(ca_path)

    exp = pd.read_csv(ROOT / "results" / "experiments.csv").set_index("experiment_code")
    t065_csha = str(exp.loc[T065_CODE, "feature_content_sha256"])
    t066_csha = str(exp.loc[T066_CODE, "feature_content_sha256"])
    if csha != t065_csha:
        raise SystemExit(f"fixed branch content hash mismatch vs T065: {csha} != {t065_csha}")
    if csha != t066_csha:
        raise SystemExit(f"fixed branch content hash mismatch vs T066: {csha} != {t066_csha}")
    print(f"fixed branch content sha matches T065/T066: {csha}", flush=True)

    # Refresh distance diagnostic with final params CSV (CV + fulldev)
    dist_df = pd.read_csv(DIST_PARAMS_PATH) if DIST_PARAMS_PATH.exists() else pd.DataFrame()
    diag_usage = write_distance_usage_diagnostic(rb, dist_df)

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
    assert abs(recomputed_p - freeze["EXP-T067"]["cv_primary_mae"]) < 1e-10
    assert abs(recomputed_s - freeze["EXP-T067"]["cv_shadow_mae"]) < 1e-10

    pub = float(freeze["public_mae"])
    priv = float(freeze["private_mae"])
    overall = float(freeze["test_overall_mae"])
    cv_p = float(freeze["EXP-T067"]["cv_primary_mae"])
    cv_s = float(freeze["EXP-T067"]["cv_shadow_mae"])
    cv_m = float(freeze["EXP-T067"]["cv_mean_mae"])
    cv_w = float(freeze["EXP-T067"]["cv_worst_mae"])

    ckpt_dir = OUT_RUN / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("fulldev_seed*.pt")) if ckpt_dir.exists() else []
    rep_note = (
        "representation_status=NOT_EXPORTED; "
        f"checkpoints under results/EXP-T067_run/checkpoints/ n={len(ckpts)}"
        if ckpts
        else "representation_status=NOT_EXPORTED; no checkpoints saved"
    )

    exp_path = ROOT / "results" / "experiments.csv"
    exp_df = pd.read_csv(exp_path)
    if "control_experiment_code" not in exp_df.columns:
        exp_df["control_experiment_code"] = ""
    if code in set(exp_df["experiment_code"].astype(str)):
        exp_df = exp_df[exp_df["experiment_code"] != code]
    row = {c: "" for c in exp_df.columns}
    for c in EXPERIMENTS_COLUMNS:
        row.setdefault(c, "")
    row.update(
        {
            "experiment_code": code,
            "experiment_id": T067_EID,
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer+Fusion+CADistanceBias",
            "source_model_id": f"CA_DISTANCE::{CONTROL_CODE}",
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
            "score_source": "EXP-T067_cv+solution_postfreeze",
            "prediction_source": "EXP-T067_run",
            "feature_source": f"{RECIPE}+classical_ca_cache+ablingua_residue",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue + BioEmu/MPNN + ESMFold CA",
            "notes": (
                f"Cα-distance attention bias vs {CONTROL_CODE}; "
                f"ca_input=experiments/inputs/{code}_ca.parquet sha={ca_sha}; "
                f"{rep_note}"
            ),
            "transformer_type": "FUSION",
            "plm_source": "ABLINGUA",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "CONCAT",
            "representation_status": "NOT_EXPORTED",
            "input_space": "RESIDUE_PLUS_FIXED_FEATURES_PLUS_CA_DISTANCE",
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
    new_df = pd.DataFrame([{c: row.get(c, "") for c in exp_df.columns}])
    pd.concat([exp_df, new_df], ignore_index=True).to_csv(exp_path, index=False)

    comp_path = ROOT / "results" / "EXPERIMENT_ARTIFACT_COMPLETENESS.csv"
    comp = pd.read_csv(comp_path)
    if code not in set(comp["experiment_code"].astype(str)):
        crow = {c: "" for c in comp.columns}
        crow.update(
            {
                "experiment_code": code,
                "experiment_id": T067_EID,
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
        "ca_path": str(ca_path),
        "feature_sha256": fsha,
        "feature_content_sha256": csha,
        "ca_sha256": ca_sha,
        "diagnostics": diag,
        "distance_usage": diag_usage,
        "checkpoints": [str(p) for p in ckpts],
    }
    save_state(state)
    return state["artifacts"]


def phase_report(code: str) -> None:
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    t037 = scores_row(CONTROL_CODE)
    t045 = scores_row(T045_CODE)
    t065 = scores_row(T065_CODE)
    t066 = scores_row(T066_CODE)
    t067 = freeze["EXP-T067"]
    diag = json.loads((OUT_RUN / "paired_diagnostics.json").read_text())
    qc = json.loads(CA_QC_PATH.read_text()) if CA_QC_PATH.exists() else {}
    usage = json.loads(DIAG_PATH.read_text()) if DIAG_PATH.exists() else {}
    dist_sum = t067.get("distance_param_summary") or {}
    d_p = t067["cv_primary_mae"] - t037["cv_primary_mae"]
    d_s = t067["cv_shadow_mae"] - t037["cv_shadow_mae"]
    d_w = t067["cv_worst_mae"] - t037["cv_worst_mae"]
    verdict = freeze.get("cv_scientific_verdict") or cv_verdict(d_p, d_s, d_w)
    both = (d_p < 0 and d_s < 0) or (d_p > 0 and d_s > 0)
    ckpts = freeze.get("checkpoints") or {}
    lines = [
        "# EXP-T067 — Learnable Cα-distance attention bias",
        "",
        f"- git: `{git_rev()}`",
        f"- control: `{CONTROL_CODE}` (stored canonical OOF; not retrained)",
        f"- change: per-head additive attention bias `a_h * exp(-d_ij / ell_h)` from within-chain Cα distances",
        f"- topology: H and L encoded separately (no H–L cross-chain distance)",
        f"- CV verdict: **{verdict}**",
        "",
        "## 1–4. vs EXP-T037",
        "",
        f"- T067 P/S/mean/W: {t067['cv_primary_mae']:.6f} / {t067['cv_shadow_mae']:.6f} / {t067['cv_mean_mae']:.6f} / {t067['cv_worst_mae']:.6f}",
        f"- Δ Primary: {d_p:+.6f}",
        f"- Δ Shadow: {d_s:+.6f}",
        f"- Δ worst: {d_w:+.6f}",
        f"- Both schemes same direction: {'YES' if both else 'NO'}",
        "",
        "## 5. Paired bootstrap (MAE_T067 − MAE_T037)",
        "",
        f"- Primary: Δ={diag['bootstrap']['primary']['mae_delta']:+.6f} CI95=[{diag['bootstrap']['primary']['ci95_low']:+.6f}, {diag['bootstrap']['primary']['ci95_high']:+.6f}]",
        f"- Shadow: Δ={diag['bootstrap']['shadow']['mae_delta']:+.6f} CI95=[{diag['bootstrap']['shadow']['ci95_low']:+.6f}, {diag['bootstrap']['shadow']['ci95_high']:+.6f}]",
        "",
        "## 6. Classical ceiling EXP-T045",
        "",
        f"- T045 worst: {t045['cv_worst_mae']:.6f}",
        f"- T067−T045 worst: {t067['cv_worst_mae'] - t045['cv_worst_mae']:+.6f}",
        "",
        "## 7–8. Learned distance parameters",
        "",
        f"- mean |a_h|: {dist_sum.get('mean_abs_a_h')}",
        f"- median ell_h (Å): {dist_sum.get('median_ell_h_angstrom')}",
        f"- by head: `{json.dumps(dist_sum.get('by_head', {}), default=str)}`",
        f"- full table: `results/EXP-T067_DISTANCE_PARAMETERS.csv`",
        "",
        "## 9. Consistency / four-way comparison",
        "",
        f"| EXP | role | Primary | Shadow | worst |",
        f"|-----|------|---------|--------|-------|",
        f"| T037 | no geometry | {t037['cv_primary_mae']:.6f} | {t037['cv_shadow_mae']:.6f} | {t037['cv_worst_mae']:.6f} |",
        f"| T065 | token-additive RASA | {t065['cv_primary_mae']:.6f} | {t065['cv_shadow_mae']:.6f} | {t065['cv_worst_mae']:.6f} |",
        f"| T066 | aggregation RASA | {t066['cv_primary_mae']:.6f} | {t066['cv_shadow_mae']:.6f} | {t066['cv_worst_mae']:.6f} |",
        f"| T067 | Cα distance bias | {t067['cv_primary_mae']:.6f} | {t067['cv_shadow_mae']:.6f} | {t067['cv_worst_mae']:.6f} |",
        "",
        "## 10–11. Data-driven bias curves / neighbor–local–nonlocal",
        "",
        f"- diagnostic: `{DIAG_PATH.relative_to(ROOT)}`",
        f"- distance quantiles: `{json.dumps(usage.get('pair_stats', {}).get('distance_quantiles_angstrom', {}), default=str)}`",
        f"- bias at quantiles (summary): `{json.dumps(usage.get('bias_curve_at_distance_quantiles', {}), default=str)}`",
        f"- category mean bias: `{json.dumps(usage.get('category_mean_bias', {}), default=str)}`",
        f"- apparent dominant category: `{usage.get('apparent_geometry_use_dominant_category')}`",
        "",
        "T067 cannot assess H–L cross-chain geometry.",
        "",
        "## Test (after freeze)",
        "",
        f"- Public: {freeze['public_mae']:.6f}",
        f"- Private: {freeze['private_mae']:.6f}",
        f"- Overall: {freeze['test_overall_mae']:.6f}",
        "",
        "## Cα QC",
        "",
        f"- expected/mapped/missing: {qc.get('n_expected_residues')}/{qc.get('n_mapped_ca_residues')}/{qc.get('n_missing_ca_residues')}",
        f"- AA mismatches: {qc.get('n_aa_mismatch_flags')}",
        f"- structure_source: {qc.get('structure_source')}",
        f"- hashes: {qc.get('ca_hashes')}",
        "",
        "## Artifacts / reproducibility",
        "",
        f"- config / features / CA parquet / preds present",
        f"- representation: NOT_EXPORTED",
        f"- checkpoints: n={ckpts.get('n_fulldev', 0)} under `results/EXP-T067_run/checkpoints/` "
        f"(total_bytes={ckpts.get('total_bytes', 0)})",
        f"- SHAREABLE_COMPLETE / REPRODUCED / eligible=YES",
        "",
        "## Interpretation",
        "",
        f"CV verdict **{verdict}**. Explicit pairwise Cα geometry is a different mechanism from RASA "
        f"(T065/T066). Do not over-claim from a single experiment; within-chain only.",
        "",
        "## STOP — no EXP-T068 / sweeps",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    state["verdict"] = verdict
    save_state(state)
    print(f"Wrote {REPORT_PATH}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=["issue", "cv", "freeze", "test", "artifacts", "report", "all"],
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.phase == "issue":
        print(issue_t067())
        return 0
    if args.phase == "cv":
        code = load_state().get("experiment_code") or issue_t067()
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

    code = issue_t067()
    print(f"Issued {code}", flush=True)
    phase_cv(quick=args.quick, code=code)
    phase_freeze(code)
    phase_test(code, quick=args.quick)
    phase_artifacts(code)
    phase_report(code)
    print("DONE EXP-T067 — STOP", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
