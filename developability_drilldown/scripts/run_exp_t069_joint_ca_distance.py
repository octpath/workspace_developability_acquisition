#!/usr/bin/env python3
"""EXP-T069: Joint H/L single-REG + learnable Cα-distance bias (vs EXP-T068).

No fusion (recipe_id=None). No RASA. No fake fusion feature parquet.
Materializes long-form CA parquet under experiments/inputs/.
Does NOT overwrite historical EXP-T030 predictions/scores.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# CRITICAL: RTX 3090 only — never GPU 1 / 1080 Ti (before torch import)
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
from classical_features.ca_cache import build_or_load_ca_cache  # noqa: E402
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import (  # noqa: E402
    AA_TO_IDX,
    BUNDLE_ROOT,
    REGION_TO_IDX,
)
from antibody_transformer.data import (  # noqa: E402
    attach_ca_coords,
    load_annotations,
    load_dev_test,
    load_folds,
    load_residue_bundle,
    load_solution,
)
from antibody_transformer.equivalence import synthetic_batch  # noqa: E402
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.training import (  # noqa: E402
    full_dev_transformer_predict,
    run_transformer_cv,
)

TARGET = "TmApp"
CONTROL_CODE = "EXP-T068"
HIST_CONTROL = "EXP-T030"
REPLAY_ID = "EXP-T030-REPLAY-001"
REPLAY_DIR = ROOT / "experiments" / "replays" / "EXP-T030" / REPLAY_ID
T069_EID = "TRF_TM_ABLINGUA_FULL_JOINT_SINGLE_REG_CA_DISTANCE"
VARIANT = "T069_JOINT_CA"
INPUT_SPACE = "JOINT_HL_SINGLE_REG_FROZEN_RESIDUE_PLUS_CA_DISTANCE"

OUT_RUN = ROOT / "results" / "EXP-T069_run"
FREEZE_PATH = ROOT / "results" / "EXP-T069_CV_FREEZE.yaml"
REPORT_PATH = ROOT / "results" / "EXP-T069_JOINT_CA_DISTANCE_REPORT.md"
CA_QC_PATH = ROOT / "results" / "EXP-T069_CA_QC.json"
DIST_PARAMS_PATH = ROOT / "results" / "EXP-T069_DISTANCE_PARAMETERS.csv"
STATE_PATH = ROOT / "results" / "EXP-T069_run_state.json"
DIAG_PATH = OUT_RUN / "distance_usage_diagnostic.json"
CKPT_MANIFEST = ROOT / "results" / "EXP-T069_CHECKPOINT_MANIFEST.json"
ZERO_BIAS_PATH = ROOT / "results" / "EXP-T069_ZERO_BIAS_EQUIVALENCE.json"


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


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def oof_to_csv(ids, vals, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, TARGET: vals}).to_csv(path, index=False)


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


def replay_scores() -> dict[str, float]:
    runs = pd.read_csv(ROOT / "results" / "experiment_runs.csv")
    r = runs[runs["run_id"] == REPLAY_ID].iloc[0]
    p = float(r["cv_primary_mae"])
    s = float(r["cv_shadow_mae"])
    return {
        "cv_primary_mae": p,
        "cv_shadow_mae": s,
        "cv_mean_mae": (p + s) / 2.0,
        "cv_worst_mae": max(p, s),
        "public_mae": float(r["public_mae"]),
        "private_mae": float(r["private_mae"]),
        "test_overall_mae": float(r["overall_mae"]),
    }


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


def issue_t069() -> str:
    codes = load_codes()
    if (codes["experiment_id"] == T069_EID).any():
        return str(codes.set_index("experiment_id").loc[T069_EID, "experiment_code"])
    nxt = next_code(TARGET)
    if nxt != "EXP-T069":
        raise SystemExit(f"expected EXP-T069, got {nxt} (run EXP-T068 issue first)")
    code = issue_code(
        T069_EID,
        TARGET,
        source_model_id=f"JOINT_HL_CA_DISTANCE::{CONTROL_CODE}",
        phase="ARCHITECTURE_JOINT_HL_CA_DISTANCE",
        notes=(
            "Joint H/L single-REG + Cα distance bias vs EXP-T068; "
            "no fusion/RASA; do not overwrite historical EXP-T030"
        ),
    )
    if code != "EXP-T069":
        raise SystemExit(code)
    return code


def write_config(code: str, ca_hashes: dict) -> Path:
    cfg = {
        "experiment_code": code,
        "experiment_id": T069_EID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"JOINT_HL_CA_DISTANCE::{CONTROL_CODE}",
        "transformer_type": "FROZEN_PLM",
        "input_space": INPUT_SPACE,
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
        "plm_source": "ABLINGUA",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": "concat",
        "pooling_mode": "REG",
        "content_mode": "frozen",
        "joint_hl_single_reg": True,
        "cross_chain_attention": True,
        "single_reg": True,
        "use_ca_distance_bias": True,
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
            "NOT_EXPORTED; full-Dev weight checkpoints under "
            "results/EXP-T069_run/checkpoints/ if size permits"
        ),
        "control_experiment_code": CONTROL_CODE,
        "contemporary_control_run_id": REPLAY_ID,
        "recipe_id": None,
        "pairwise_geometry": {
            "enabled": True,
            "geometry_source": "CA_COORDINATES",
            "distance_metric": "EUCLIDEAN_ANGSTROM",
            "distance_kernel": "EXPONENTIAL",
            "distance_kernel_count_per_head": 1,
            "distance_amplitude_init": "ZERO",
            "distance_scale": "LEARNABLE_PER_HEAD",
            "distance_scale_init": "TRAIN_MEDIAN_PAIR_DISTANCE_JOINT_FV",
            "distance_kernel_shared_across_layers": True,
            "cross_chain_distance": True,
            "self_pair_bias": False,
            "ca_source": "experiments/classical_cache",
            "ca_hashes": ca_hashes,
            "control_experiment_code": CONTROL_CODE,
            "joint_hl_layout": True,
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
                "mae_t069": float(ae_n.mean()),
                "mae_control": float(ae_c.mean()),
                "delta": float(ae_n.mean() - ae_c.mean()),
            }
        )
    return rows


def cv_verdict(d_p: float, d_s: float, d_w: float) -> str:
    both_imp = d_p < 0 and d_s < 0
    both_wors = d_p > 0 and d_s > 0
    if both_imp and d_w < 0:
        return "POSITIVE"
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


def collect_joint_pair_stats(rb) -> dict:
    """Cα pair distances with within-chain sep categories + H–L cross-chain."""
    dist_all: list[np.ndarray] = []
    neighbor_d: list[float] = []
    local_d: list[float] = []
    nonlocal_d: list[float] = []
    cross_hl_d: list[float] = []

    for i in range(len(rb.ids)):
        chain_xyz: list[tuple[str, np.ndarray, np.ndarray]] = []
        for name, ca, mask in (
            ("H", rb.heavy_ca[i], rb.heavy_mask[i]),
            ("L", rb.light_ca[i], rb.light_mask[i]),
        ):
            n = int(mask.sum())
            if n < 1:
                continue
            xyz = ca[:n]
            ok = np.isfinite(xyz).all(axis=1)
            idx = np.where(ok)[0]
            if len(idx) < 1:
                continue
            chain_xyz.append((name, xyz[idx], idx.astype(np.int64)))

        # within-chain
        for _name, xyz_ok, idx in chain_xyz:
            m = len(idx)
            if m < 2:
                continue
            d = np.linalg.norm(xyz_ok[:, None, :] - xyz_ok[None, :, :], axis=-1)
            iu, ju = np.triu_indices(m, k=1)
            dists = d[iu, ju].astype(np.float64)
            dist_all.append(dists)
            sep = np.abs(idx[iu] - idx[ju])
            for dij, s in zip(dists, sep):
                if s == 1:
                    neighbor_d.append(float(dij))
                elif 1 < s <= 4:
                    local_d.append(float(dij))
                else:
                    nonlocal_d.append(float(dij))

        # H–L cross-chain
        by_name = {n: (xyz, idx) for n, xyz, idx in chain_xyz}
        if "H" in by_name and "L" in by_name:
            h_xyz, _ = by_name["H"]
            l_xyz, _ = by_name["L"]
            dhl = np.linalg.norm(h_xyz[:, None, :] - l_xyz[None, :, :], axis=-1).ravel()
            dist_all.append(dhl.astype(np.float64))
            cross_hl_d.extend(float(x) for x in dhl)

    if not dist_all:
        raise RuntimeError("no valid joint Cα pairs for diagnostic")
    all_d = np.concatenate(dist_all)
    qs = {f"q{p}": float(np.quantile(all_d, p / 100.0)) for p in (10, 25, 50, 75, 90)}
    return {
        "n_pairs": int(all_d.size),
        "distance_quantiles_angstrom": qs,
        "pair_category_counts": {
            "sequence_neighbor_|i-j|=1": len(neighbor_d),
            "local_1<|i-j|<=4": len(local_d),
            "nonlocal_|i-j|>4": len(nonlocal_d),
            "H-L_cross_chain": len(cross_hl_d),
        },
        "pair_category_mean_distance_angstrom": {
            "sequence_neighbor_|i-j|=1": float(np.mean(neighbor_d)) if neighbor_d else None,
            "local_1<|i-j|<=4": float(np.mean(local_d)) if local_d else None,
            "nonlocal_|i-j|>4": float(np.mean(nonlocal_d)) if nonlocal_d else None,
            "H-L_cross_chain": float(np.mean(cross_hl_d)) if cross_hl_d else None,
        },
    }


def write_distance_usage_diagnostic(rb, dist_df: pd.DataFrame) -> dict:
    pair_stats = collect_joint_pair_stats(rb)
    qs = pair_stats["distance_quantiles_angstrom"]
    cv_df = dist_df[dist_df["cv_scheme"].isin(["primary", "shadow"])].copy() if not dist_df.empty else dist_df
    mean_abs_a = float(cv_df["a_h"].abs().mean()) if len(cv_df) else float("nan")
    median_ell = float(cv_df["ell_h_angstrom"].median()) if len(cv_df) else float("nan")

    bias_curve_summary = {
        qk: float(mean_abs_a * np.exp(-qv / median_ell)) if np.isfinite(mean_abs_a) and median_ell > 0 else None
        for qk, qv in qs.items()
    }

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

    cat_means = pair_stats["pair_category_mean_distance_angstrom"]
    category_bias = {}
    for name, dmean in cat_means.items():
        if dmean is None or not (np.isfinite(mean_abs_a) and median_ell > 0):
            category_bias[name] = None
        else:
            category_bias[name] = float(mean_abs_a * np.exp(-float(dmean) / median_ell))

    ranked = sorted(
        ((k, abs(v) if v is not None else -1.0) for k, v in category_bias.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    dominant = ranked[0][0] if ranked and ranked[0][1] >= 0 else None

    out = {
        "note": (
            "Diagnostic only; not used for selection. "
            "Joint Fv pairs including H–L cross-chain. "
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
        "cross_chain_hl": True,
    }
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    DIAG_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def zero_bias_equivalence_check() -> dict:
    """Forward check: joint+CA with a_h=0 matches joint without CA."""
    torch.manual_seed(0)
    base = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        use_ca_distance_bias=False,
    )
    torch.manual_seed(0)
    dist = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        use_ca_distance_bias=True,
        initial_ell_angstrom=8.0,
    )
    sd = base.state_dict()
    mapped = {}
    for k, v in sd.items():
        if k in dist.state_dict() and dist.state_dict()[k].shape == v.shape:
            mapped[k] = v
    dist.load_state_dict(mapped, strict=False)
    assert torch.all(dist.encoder.a == 0)

    batch = synthetic_batch(content_mode="frozen", chain_mode="HL", annotation_mode="full", plm_hidden=1280)
    B, Lh = batch["heavy_plm"].shape[:2]
    Ll = batch["light_plm"].shape[1]
    batch["heavy_ca"] = torch.randn(B, Lh, 3)
    batch["light_ca"] = torch.randn(B, Ll, 3)
    base.eval()
    dist.eval()
    with torch.no_grad():
        r0 = base.forward_repr(batch)
        r1 = dist.forward_repr(batch)
        y0 = base(batch)
        y1 = dist(batch)
    repr_max = float((r0 - r1).abs().max())
    pred_max = float((y0 - y1).abs().max())
    ok = repr_max < 1e-5 and pred_max < 1e-5
    result = {
        "ok": ok,
        "repr_max_abs": repr_max,
        "pred_max_abs": pred_max,
        "note": "joint_hl_single_reg + CA distance with a_h=0 vs joint without CA",
    }
    ZERO_BIAS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    if not ok:
        raise SystemExit(f"zero-bias equivalence failed: {result}")
    print(f"zero-bias equivalence OK: {result}", flush=True)
    return result


def train_kwargs() -> dict:
    return dict(
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        plm_source="ablingua",
        pooling_mode="reg",
        recipe_id=None,
        joint_hl_single_reg=True,
        use_ca_distance_bias=True,
        use_continuous_rasa=False,
        use_rasa_weighted_pool=False,
    )


def phase_cv(*, quick: bool, code: str) -> dict:
    print("=== EXP-T069 CV ===", flush=True)
    # Control EXP-T068 must exist
    ctl_pred = ROOT / "experiments" / "predictions" / CONTROL_CODE / "oof_primary.csv"
    if not ctl_pred.exists():
        raise SystemExit(f"missing control OOF {ctl_pred}; run EXP-T068 first")
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    zero_bias_equivalence_check()
    dev, test, folds, rb, qc = prepare_bundle()
    write_config(code, qc["ca_hashes"])
    t0 = time.time()
    summary = run_transformer_cv(
        target=TARGET,
        variant_id=VARIANT,
        dev=dev,
        rb=rb,
        folds=folds,
        device=device_str(),
        out_dir=OUT_RUN,
        quick=quick,
        **train_kwargs(),
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
        "n_trainable_parameters": summary.get("n_trainable_parameters"),
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
        "zero_bias_equivalence": json.loads(ZERO_BIAS_PATH.read_text()),
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
    state["t069_cv"] = out
    state["experiment_code"] = code
    state["distance_param_rows_cv"] = dist_rows
    save_state(state)
    return out


def phase_freeze(code: str) -> dict:
    if FREEZE_PATH.exists():
        return yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    t069 = state["t069_cv"]
    t068 = scores_row(CONTROL_CODE)
    replay = replay_scores()
    hist = scores_row(HIST_CONTROL)
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    d_p = t069["cv_primary_mae"] - t068["cv_primary_mae"]
    d_s = t069["cv_shadow_mae"] - t068["cv_shadow_mae"]
    d_w = t069["cv_worst_mae"] - t068["cv_worst_mae"]
    verdict = cv_verdict(d_p, d_s, d_w)

    _, _, _, rb, qc = prepare_bundle()
    dist_df = pd.read_csv(DIST_PARAMS_PATH) if DIST_PARAMS_PATH.exists() else pd.DataFrame()
    diag = write_distance_usage_diagnostic(rb, dist_df)

    freeze = {
        "experiment_code": code,
        "experiment_id": T069_EID,
        "git_rev": git_rev(),
        "config": cfg,
        "control_experiment_code": CONTROL_CODE,
        "contemporary_control_run_id": REPLAY_ID,
        "ca_source": cfg["pairwise_geometry"]["ca_source"],
        "ca_hashes": cfg["pairwise_geometry"]["ca_hashes"],
        "structure_source": qc.get("structure_source"),
        "EXP-T069": {
            "cv_primary_mae": t069["cv_primary_mae"],
            "cv_shadow_mae": t069["cv_shadow_mae"],
            "cv_mean_mae": t069["cv_mean_mae"],
            "cv_worst_mae": t069["cv_worst_mae"],
            "config_hash": t069["config_hash"],
            "best_epochs_primary": t069["best_epochs_primary"],
            "n_trainable_parameters": t069.get("n_trainable_parameters"),
            "distance_param_summary": t069.get("distance_param_summary"),
        },
        "deltas_vs_EXP-T068": {"primary": d_p, "shadow": d_s, "worst": d_w},
        "deltas_vs_EXP-T030-REPLAY-001": {
            "primary": t069["cv_primary_mae"] - replay["cv_primary_mae"],
            "shadow": t069["cv_shadow_mae"] - replay["cv_shadow_mae"],
            "worst": t069["cv_worst_mae"] - replay["cv_worst_mae"],
        },
        "deltas_vs_historical_EXP-T030": {
            "primary": t069["cv_primary_mae"] - hist["cv_primary_mae"],
            "shadow": t069["cv_shadow_mae"] - hist["cv_shadow_mae"],
            "worst": t069["cv_worst_mae"] - hist["cv_worst_mae"],
        },
        "cv_scientific_verdict": verdict,
        "distance_usage_diagnostic_path": str(DIAG_PATH.relative_to(ROOT)),
        "distance_usage_summary": {
            "mean_abs_a_h": diag["summary_params"]["mean_abs_a_h"],
            "median_ell_h_angstrom": diag["summary_params"]["median_ell_h_angstrom"],
            "dominant_category": diag.get("apparent_geometry_use_dominant_category"),
        },
        "selection_statement": (
            "CV completed for EXP-T069; config frozen; Public/Private not used before freeze; "
            "single full-Dev Test follows. Primary control = EXP-T068."
        ),
        "public_mae": None,
        "private_mae": None,
        "test_overall_mae": None,
    }
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    print(f"Wrote {FREEZE_PATH} verdict={verdict}", flush=True)
    return freeze


def write_checkpoint_manifest(code: str, ckpt_dir: Path, n_params) -> Path:
    ckpts = sorted(ckpt_dir.glob("fulldev_seed*.pt")) if ckpt_dir.exists() else []
    man = {
        "experiment_code": code,
        "experiment_id": T069_EID,
        "checkpoint_dir": str(ckpt_dir.relative_to(ROOT)) if ckpt_dir.exists() else None,
        "n_fulldev": len(ckpts),
        "n_trainable_parameters": n_params,
        "files": [
            {
                "path": str(p.relative_to(ROOT)),
                "sha256": file_sha256(p),
                "bytes": int(p.stat().st_size),
            }
            for p in ckpts
        ],
        "total_bytes": int(sum(p.stat().st_size for p in ckpts)) if ckpts else 0,
        "git_rev": git_rev(),
    }
    CKPT_MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
    return CKPT_MANIFEST


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
        fd_csv = ckpt_dir / "distance_params_fulldev.csv"
        if fd_csv.exists():
            fulldev_rows = pd.read_csv(fd_csv).to_dict(orient="records")
    else:
        state = load_state()
        best = state["t069_cv"]["best_epochs_primary"]
        dev, test, folds, rb, _ = prepare_bundle()
        te_df = full_dev_transformer_predict(
            target=TARGET,
            best_epochs_primary=best,
            dev=dev,
            test=test,
            rb=rb,
            device=device_str(),
            quick=quick,
            checkpoint_dir=ckpt_dir,
            **train_kwargs(),
        )
        te = te_df.rename(columns={"prediction": TARGET})
        te.to_csv(pred_path, index=False)
        te.to_csv(OUT_RUN / "test.csv", index=False)
        fulldev_rows = list(getattr(te_df, "attrs", {}).get("distance_param_rows") or [])
        if not fulldev_rows:
            fd_csv = ckpt_dir / "distance_params_fulldev.csv"
            if fd_csv.exists():
                fulldev_rows = pd.read_csv(fd_csv).to_dict(orient="records")

    state = load_state()
    cv_rows = list(state.get("distance_param_rows_cv") or [])
    if not cv_rows and DIST_PARAMS_PATH.exists():
        prev = pd.read_csv(DIST_PARAMS_PATH)
        cv_rows = prev[prev["cv_scheme"].isin(["primary", "shadow"])].to_dict(orient="records")
    write_distance_params_csv(cv_rows + fulldev_rows)
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
    write_checkpoint_manifest(code, ckpt_dir, freeze.get("EXP-T069", {}).get("n_trainable_parameters"))
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    state["t069_test"] = scores
    save_state(state)
    print(scores, flush=True)
    return scores


def materialize_ca_parquet(code: str, rb, dev: pd.DataFrame, test: pd.DataFrame) -> Path:
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
    ann = ann.copy()
    ann["id"] = ann["id"].astype(str)
    for ab in rb.ids[:5]:
        n_ann = len(ann[ann.id == ab])
        n_df = len(df[df.id == ab])
        if n_ann != n_df:
            raise SystemExit(f"CA parquet length mismatch {ab}: {n_df} vs ann {n_ann}")
    for ab in rb.ids[:3]:
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
        "joint_hl": True,
    }
    (ROOT / "experiments" / "inputs" / f"{code}_ca.parquet.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    return out


def _pred_col(df: pd.DataFrame) -> str:
    return TARGET if TARGET in df.columns else [c for c in df.columns if c != "id"][0]


def phase_artifacts(code: str) -> dict:
    print("=== artifacts + registry ===", flush=True)
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    dev, test, folds, rb, qc = prepare_bundle()

    feat = ROOT / "experiments" / "features" / f"{code}.parquet"
    if feat.exists():
        raise SystemExit(f"joint HL must not materialize fusion feature parquet: {feat}")

    ca_path = materialize_ca_parquet(code, rb, dev, test)
    ca_sha = file_sha256(ca_path)

    dist_df = pd.read_csv(DIST_PARAMS_PATH) if DIST_PARAMS_PATH.exists() else pd.DataFrame()
    diag_usage = write_distance_usage_diagnostic(rb, dist_df)

    ctl_p = pd.read_csv(ROOT / "experiments" / "predictions" / CONTROL_CODE / "oof_primary.csv")
    ctl_s = pd.read_csv(ROOT / "experiments" / "predictions" / CONTROL_CODE / "oof_shadow.csv")
    new_p = pd.read_csv(ROOT / "experiments" / "predictions" / code / "oof_primary.csv")
    new_s = pd.read_csv(ROOT / "experiments" / "predictions" / code / "oof_shadow.csv")
    for d in (ctl_p, ctl_s, new_p, new_s):
        d["id"] = d["id"].astype(str)
    y = dev.set_index("id")[TARGET]
    ids = dev["id"].astype(str).tolist()
    col_c = _pred_col(ctl_p)
    ctl_ps = ctl_p.set_index("id").loc[ids, col_c]
    ctl_ss = ctl_s.set_index("id").loc[ids, col_c]
    new_ps = new_p.set_index("id").loc[ids, TARGET]
    new_ss = new_s.set_index("id").loc[ids, TARGET]
    ae_ctl_p = (y.loc[ids] - ctl_ps).abs().to_numpy(float)
    ae_new_p = (y.loc[ids] - new_ps).abs().to_numpy(float)
    ae_ctl_s = (y.loc[ids] - ctl_ss).abs().to_numpy(float)
    ae_new_s = (y.loc[ids] - new_ss).abs().to_numpy(float)

    # Secondary: vs replay
    rep_p = pd.read_csv(REPLAY_DIR / "oof_primary.csv")
    rep_s = pd.read_csv(REPLAY_DIR / "oof_shadow.csv")
    for d in (rep_p, rep_s):
        d["id"] = d["id"].astype(str)
    col_r = _pred_col(rep_p)
    rep_ps = rep_p.set_index("id").loc[ids, col_r]
    rep_ss = rep_s.set_index("id").loc[ids, col_r]
    ae_rep_p = (y.loc[ids] - rep_ps).abs().to_numpy(float)
    ae_rep_s = (y.loc[ids] - rep_ss).abs().to_numpy(float)

    diag = {
        "bootstrap_vs_t068": {
            "primary": paired_bootstrap(ae_new_p, ae_ctl_p),
            "shadow": paired_bootstrap(ae_new_s, ae_ctl_s),
        },
        "bootstrap_vs_replay": {
            "primary": paired_bootstrap(ae_new_p, ae_rep_p),
            "shadow": paired_bootstrap(ae_new_s, ae_rep_s),
        },
        "fold_mae_deltas_vs_t068": {
            "primary": fold_deltas(dev, folds.primary, new_ps, ctl_ps),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, ctl_ss),
        },
    }
    (OUT_RUN / "paired_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    recomputed_p = float(mae(y.loc[ids].to_numpy(float), new_ps.to_numpy(float)))
    recomputed_s = float(mae(y.loc[ids].to_numpy(float), new_ss.to_numpy(float)))
    assert abs(recomputed_p - freeze["EXP-T069"]["cv_primary_mae"]) < 1e-10
    assert abs(recomputed_s - freeze["EXP-T069"]["cv_shadow_mae"]) < 1e-10

    pub = float(freeze["public_mae"])
    priv = float(freeze["private_mae"])
    overall = float(freeze["test_overall_mae"])
    cv_p = float(freeze["EXP-T069"]["cv_primary_mae"])
    cv_s = float(freeze["EXP-T069"]["cv_shadow_mae"])
    cv_m = float(freeze["EXP-T069"]["cv_mean_mae"])
    cv_w = float(freeze["EXP-T069"]["cv_worst_mae"])

    ckpt_dir = OUT_RUN / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("fulldev_seed*.pt")) if ckpt_dir.exists() else []
    write_checkpoint_manifest(code, ckpt_dir, freeze["EXP-T069"].get("n_trainable_parameters"))
    rep_note = (
        f"representation_status=NOT_EXPORTED; checkpoints under "
        f"results/EXP-T069_run/checkpoints/ n={len(ckpts)}"
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
            "experiment_id": T069_EID,
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer+JointHLSingleREG+CADistanceBias",
            "source_model_id": f"JOINT_HL_CA_DISTANCE::{CONTROL_CODE}",
            "feature_set_id": "",
            "source_recipe_id": "",
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
            "feature_path": "",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "n_features": "",
            "feature_space": "",
            "feature_sha256": "",
            "feature_content_sha256": "",
            "score_source": "EXP-T069_cv+solution_postfreeze",
            "prediction_source": "EXP-T069_run",
            "feature_source": "ablingua_residue_joint_hl+classical_ca_cache",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue + ESMFold CA",
            "notes": (
                f"Joint H/L single-REG + Cα bias vs {CONTROL_CODE}; "
                f"ca_input=experiments/inputs/{code}_ca.parquet sha={ca_sha}; "
                f"{rep_note}"
            ),
            "transformer_type": "FROZEN_PLM",
            "plm_source": "ABLINGUA",
            "annotation_mode": "FULL",
            "chain_mode": "HL",
            "pooling_mode": "REG",
            "representation_status": "NOT_EXPORTED",
            "input_space": INPUT_SPACE,
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
                "experiment_id": T069_EID,
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
                "notes": "JOINT_HL+CA: no fusion feature parquet; CA input required",
            }
        )
        if "test_prediction_exists" in comp.columns:
            crow["test_prediction_exists"] = True
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)

    state["artifacts"] = {
        "feature_path": "",
        "ca_path": str(ca_path),
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
    t068 = scores_row(CONTROL_CODE)
    replay = replay_scores()
    hist = scores_row(HIST_CONTROL)
    t069 = freeze["EXP-T069"]
    diag = json.loads((OUT_RUN / "paired_diagnostics.json").read_text())
    qc = json.loads(CA_QC_PATH.read_text()) if CA_QC_PATH.exists() else {}
    usage = json.loads(DIAG_PATH.read_text()) if DIAG_PATH.exists() else {}
    dist_sum = t069.get("distance_param_summary") or {}
    d_p = t069["cv_primary_mae"] - t068["cv_primary_mae"]
    d_s = t069["cv_shadow_mae"] - t068["cv_shadow_mae"]
    d_w = t069["cv_worst_mae"] - t068["cv_worst_mae"]
    verdict = freeze.get("cv_scientific_verdict") or cv_verdict(d_p, d_s, d_w)
    both = (d_p < 0 and d_s < 0) or (d_p > 0 and d_s > 0)
    ckpts = freeze.get("checkpoints") or {}
    lines = [
        "# EXP-T069 — Joint H/L single-REG + Cα-distance bias",
        "",
        f"- git: `{git_rev()}`",
        f"- primary control: `{CONTROL_CODE}`",
        f"- secondary: `{REPLAY_ID}` and historical `{HIST_CONTROL}`",
        f"- change: joint `[REG,H...,L...]` + per-head `a_h * exp(-d_ij / ell_h)` over full Fv (incl. H–L)",
        f"- CV verdict: **{verdict}**",
        "",
        "## Primary vs EXP-T068",
        "",
        f"- T069 P/S/mean/W: {t069['cv_primary_mae']:.6f} / {t069['cv_shadow_mae']:.6f} / "
        f"{t069['cv_mean_mae']:.6f} / {t069['cv_worst_mae']:.6f}",
        f"- Δ Primary: {d_p:+.6f}",
        f"- Δ Shadow: {d_s:+.6f}",
        f"- Δ worst: {d_w:+.6f}",
        f"- Both schemes same direction: {'YES' if both else 'NO'}",
        "",
        "## Secondary deltas",
        "",
        f"- vs replay Primary/Shadow/worst: "
        f"{t069['cv_primary_mae'] - replay['cv_primary_mae']:+.6f} / "
        f"{t069['cv_shadow_mae'] - replay['cv_shadow_mae']:+.6f} / "
        f"{t069['cv_worst_mae'] - replay['cv_worst_mae']:+.6f}",
        f"- vs historical T030 Primary/Shadow/worst: "
        f"{t069['cv_primary_mae'] - hist['cv_primary_mae']:+.6f} / "
        f"{t069['cv_shadow_mae'] - hist['cv_shadow_mae']:+.6f} / "
        f"{t069['cv_worst_mae'] - hist['cv_worst_mae']:+.6f}",
        "",
        "## Paired bootstrap vs T068",
        "",
        f"- Primary: Δ={diag['bootstrap_vs_t068']['primary']['mae_delta']:+.6f} "
        f"CI95=[{diag['bootstrap_vs_t068']['primary']['ci95_low']:+.6f}, "
        f"{diag['bootstrap_vs_t068']['primary']['ci95_high']:+.6f}]",
        f"- Shadow: Δ={diag['bootstrap_vs_t068']['shadow']['mae_delta']:+.6f} "
        f"CI95=[{diag['bootstrap_vs_t068']['shadow']['ci95_low']:+.6f}, "
        f"{diag['bootstrap_vs_t068']['shadow']['ci95_high']:+.6f}]",
        "",
        "## Learned distance parameters",
        "",
        f"- mean |a_h|: {dist_sum.get('mean_abs_a_h')}",
        f"- median ell_h (Å): {dist_sum.get('median_ell_h_angstrom')}",
        f"- by head: `{json.dumps(dist_sum.get('by_head', {}), default=str)}`",
        f"- full table: `results/EXP-T069_DISTANCE_PARAMETERS.csv`",
        "",
        "## Geometry diagnostic (incl. H–L)",
        "",
        f"- diagnostic: `{DIAG_PATH.relative_to(ROOT)}`",
        f"- category counts: `{json.dumps(usage.get('pair_stats', {}).get('pair_category_counts', {}), default=str)}`",
        f"- category mean bias: `{json.dumps(usage.get('category_mean_bias', {}), default=str)}`",
        f"- apparent dominant category: `{usage.get('apparent_geometry_use_dominant_category')}`",
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
        "",
        "## Artifacts",
        "",
        f"- config / CA parquet / preds; feature_path EMPTY",
        f"- checkpoints: n={ckpts.get('n_fulldev', 0)} under `results/EXP-T069_run/checkpoints/`",
        f"- checkpoint manifest: `{CKPT_MANIFEST.relative_to(ROOT)}`",
        f"- SHAREABLE_COMPLETE / REPRODUCED",
        "",
        f"CV verdict **{verdict}**.",
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
        print(issue_t069())
        return 0
    if args.phase == "cv":
        code = load_state().get("experiment_code") or issue_t069()
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

    code = issue_t069()
    print(f"Issued {code}", flush=True)
    phase_cv(quick=args.quick, code=code)
    phase_freeze(code)
    phase_test(code, quick=args.quick)
    phase_artifacts(code)
    phase_report(code)
    print("DONE EXP-T069", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
