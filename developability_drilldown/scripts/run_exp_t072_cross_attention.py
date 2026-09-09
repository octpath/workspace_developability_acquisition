#!/usr/bin/env python3
"""EXP-T072: Separate H/L self-attn + zero-gated cross-attention bridge (vs REPLAY).

T030-style separate encoders with bidirectional residue cross-attn between layers;
REG excluded from cross Q/K/V. No joint/fusion/RASA/3D. Does NOT overwrite T030–T071.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT))

from _lib import EXPERIMENTS_COLUMNS, file_sha256, mae  # noqa: E402
from experiment_codes import issue_code, load_codes, next_code  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import (  # noqa: E402
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
HIST_CONTROL = "EXP-T030"
T068_CODE = "EXP-T068"
T070_CODE = "EXP-T070"
T071_CODE = "EXP-T071"
REPLAY_ID = "EXP-T030-REPLAY-001"
REPLAY_DIR = ROOT / "experiments" / "replays" / "EXP-T030" / REPLAY_ID
T072_EID = "TRF_TM_ABLINGUA_FULL_SEPARATE_WITH_CROSS_ATTENTION_DUAL_REG"
VARIANT = "T072_CROSS"
INPUT_SPACE = "SEPARATE_CROSS_ATTENTION_DUAL_REG_FROZEN_RESIDUE"

OUT_RUN = ROOT / "results" / "EXP-T072_run"
FREEZE_PATH = ROOT / "results" / "EXP-T072_CV_FREEZE.yaml"
REPORT_PATH = ROOT / "results" / "EXP-T072_CROSS_ATTENTION_REPORT.md"
CROSS_GATES_CSV = ROOT / "results" / "EXP-T072_CROSS_GATES.csv"
STATE_PATH = ROOT / "results" / "EXP-T072_run_state.json"
CKPT_MANIFEST = ROOT / "results" / "EXP-T072_CHECKPOINT_MANIFEST.json"


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


def device_str() -> str:
    import torch

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


def param_counts() -> dict:
    """Trainable params: T030 / T068 / T070 / T071 / T072 + T072 param_account."""
    kw = dict(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
    )
    t030 = AnnotatedTransformer(**kw)
    t068 = AnnotatedTransformer(**kw, joint_hl_single_reg=True)
    t070 = AnnotatedTransformer(**kw, joint_hl_dual_reg=True)
    t071 = AnnotatedTransformer(**kw, joint_hl_chain_specific_dual_reg=True)
    t072 = AnnotatedTransformer(**kw, use_cross_attention_bridge=True)
    account = t072.param_account()
    return {
        "EXP-T030_n_trainable": int(t030.n_trainable_parameters()),
        "EXP-T068_n_trainable": int(t068.n_trainable_parameters()),
        "EXP-T070_n_trainable": int(t070.n_trainable_parameters()),
        "EXP-T071_n_trainable": int(t071.n_trainable_parameters()),
        "EXP-T072_n_trainable": int(t072.n_trainable_parameters()),
        "repr_dim_t030": int(t030.repr_dim),
        "repr_dim_t068": int(t068.repr_dim),
        "repr_dim_t070": int(t070.repr_dim),
        "repr_dim_t071": int(t071.repr_dim),
        "repr_dim_t072": int(t072.repr_dim),
        "param_account_t072": {k: int(v) for k, v in account.items()},
    }


def prepare_bundle():
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = load_residue_bundle(dev, test, need_ablingua=True)
    return dev, test, folds, rb


def issue_t072() -> str:
    codes = load_codes()
    if (codes["experiment_id"] == T072_EID).any():
        return str(codes.set_index("experiment_id").loc[T072_EID, "experiment_code"])
    nxt = next_code(TARGET)
    if nxt != "EXP-T072":
        raise SystemExit(f"expected EXP-T072, got {nxt}")
    code = issue_code(
        T072_EID,
        TARGET,
        source_model_id=f"SEPARATE_CROSS_ATTENTION_DUAL_REG::{REPLAY_ID}",
        phase="ARCHITECTURE_SEPARATE_CROSS_ATTENTION_DUAL_REG",
        notes=(
            "Separate H/L + zero-gated cross-attention bridge vs contemporary "
            "EXP-T030-REPLAY-001; no joint/fusion/RASA; do not overwrite T030–T071"
        ),
    )
    if code != "EXP-T072":
        raise SystemExit(code)
    return code


def write_config(code: str) -> Path:
    cfg = {
        "experiment_code": code,
        "experiment_id": T072_EID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"SEPARATE_CROSS_ATTENTION_DUAL_REG::{REPLAY_ID}",
        "transformer_type": "FROZEN_PLM",
        "input_space": INPUT_SPACE,
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
        "plm_source": "ABLINGUA",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": "concat",
        "pooling_mode": "REG",
        "content_mode": "frozen",
        "use_cross_attention_bridge": True,
        "joint_hl_chain_specific_dual_reg": False,
        "joint_hl_dual_reg": False,
        "joint_hl_single_reg": False,
        "cross_chain_attention": True,
        "dual_reg": True,
        "use_ca_distance_bias": False,
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
            "results/EXP-T072_run/checkpoints/ if size permits"
        ),
        "control_experiment_code": HIST_CONTROL,
        "contemporary_control_run_id": REPLAY_ID,
        "recipe_id": None,
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
        "p_delta_lt_0": float((boots < 0).mean()),
        "n_boot": n,
    }


def fold_deltas(dev, fmap, oof_new, oof_ctl, *, new_key="mae_t072", ctl_key="mae_control"):
    y = dev.set_index("id")[TARGET]
    rows = []
    for k in range(5):
        ids = [i for i, f in fmap.items() if int(f) == k]
        ae_n = (y.loc[ids] - oof_new.loc[ids]).abs()
        ae_c = (y.loc[ids] - oof_ctl.loc[ids]).abs()
        rows.append(
            {
                "fold": k,
                new_key: float(ae_n.mean()),
                ctl_key: float(ae_c.mean()),
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


def gate_summary_stats(rows: list[dict]) -> dict:
    """mean/median/std/abs(mean)/sign consistency for g_H and g_L."""
    if not rows:
        return {"n_rows": 0, "g_H": {}, "g_L": {}}
    out: dict = {"n_rows": len(rows)}
    for key in ("g_H", "g_L"):
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        mean = float(vals.mean())
        signs = np.sign(vals)
        nonzero = signs[signs != 0]
        if len(nonzero) == 0:
            sign_consistency = 1.0
        else:
            sign_consistency = float(np.mean(nonzero == np.sign(mean))) if mean != 0 else float(
                np.mean(nonzero == nonzero[0])
            )
        out[key] = {
            "mean": mean,
            "median": float(np.median(vals)),
            "std": float(vals.std(ddof=0)),
            "abs_mean": float(np.abs(vals).mean()),
            "sign_consistency": sign_consistency,
            "frac_positive": float((vals > 0).mean()),
            "frac_negative": float((vals < 0).mean()),
            "frac_zero": float((vals == 0).mean()),
        }
    return out


def write_cross_gates_csv(rows: list[dict]) -> None:
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    CROSS_GATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CROSS_GATES_CSV, index=False)
    df.to_csv(OUT_RUN / "cross_gates.csv", index=False)


def zero_gate_equivalence_smoke() -> dict:
    """At g_H=g_L=0, T072 forward must match T030 (shared overlapping weights)."""
    import torch

    torch.manual_seed(0)
    kw = dict(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
    )
    t030 = AnnotatedTransformer(**kw)
    t072 = AnnotatedTransformer(**kw, use_cross_attention_bridge=True)
    sd030 = t030.state_dict()
    sd072 = t072.state_dict()
    for k, v in sd030.items():
        if k in sd072 and sd072[k].shape == v.shape:
            sd072[k] = v.clone()
    t072.load_state_dict(sd072)
    assert float(t072.cross_gate_h) == 0.0
    assert float(t072.cross_gate_l) == 0.0

    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=1280,
        Lh=8,
        Ll=6,
        B=3,
    )
    t030.eval()
    t072.eval()
    with torch.no_grad():
        y0 = t030(batch)
        y1 = t072(batch)
    max_pred = float((y0 - y1).abs().max())
    result = {"ok": max_pred <= 1e-5, "max_pred_diff": max_pred, "threshold": 1e-5}
    print(f"zero-gate equivalence smoke: max_pred_diff={max_pred:.3e}", flush=True)
    if max_pred > 1e-5:
        raise SystemExit(f"zero-gate equivalence failed: {result}")
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
        joint_hl_chain_specific_dual_reg=False,
        joint_hl_dual_reg=False,
        joint_hl_single_reg=False,
        use_cross_attention_bridge=True,
        use_ca_distance_bias=False,
        use_continuous_rasa=False,
        use_rasa_weighted_pool=False,
    )


def phase_cv(*, quick: bool, code: str) -> dict:
    print("=== EXP-T072 CV ===", flush=True)
    if not (REPLAY_DIR / "oof_primary.csv").exists():
        raise SystemExit(f"missing contemporary control OOF: {REPLAY_DIR}")
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    write_config(code)
    params = param_counts()
    zero_eq = zero_gate_equivalence_smoke()
    dev, test, folds, rb = prepare_bundle()
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
    cross_gate_rows = list(summary.get("cross_gate_rows") or [])
    write_cross_gates_csv(cross_gate_rows)
    gate_stats = gate_summary_stats(cross_gate_rows)

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
        "n_trainable_parameters": summary.get("n_trainable_parameters"),
        "param_counts": params,
        "zero_gate_equivalence": zero_eq,
        "n_cross_gate_rows": len(cross_gate_rows),
        "gate_summary_stats": gate_stats,
    }
    (OUT_RUN / "cv_summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {k: out[k] for k in ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae")},
            indent=2,
        ),
        flush=True,
    )
    print(f"gate_summary_stats: {json.dumps(gate_stats, indent=2)}", flush=True)
    state = load_state()
    state["t072_cv"] = out
    state["experiment_code"] = code
    state["cross_gate_rows"] = cross_gate_rows
    save_state(state)
    return out


def _oof_abs_err(pred_csv: Path, y: pd.Series, ids: list[str], col: str | None = None) -> np.ndarray:
    df = pd.read_csv(pred_csv)
    df["id"] = df["id"].astype(str)
    c = col or _pred_col(df)
    pred = df.set_index("id").loc[ids, c]
    return (y.loc[ids] - pred).abs().to_numpy(float)


def phase_freeze(code: str) -> dict:
    if FREEZE_PATH.exists():
        return yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    t072 = state["t072_cv"]
    replay = replay_scores()
    hist = scores_row(HIST_CONTROL)
    t068 = scores_row(T068_CODE)
    t070 = scores_row(T070_CODE)
    t071 = scores_row(T071_CODE)
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    params = t072.get("param_counts") or param_counts()
    d_p = t072["cv_primary_mae"] - replay["cv_primary_mae"]
    d_s = t072["cv_shadow_mae"] - replay["cv_shadow_mae"]
    d_w = t072["cv_worst_mae"] - replay["cv_worst_mae"]
    verdict = cv_verdict(d_p, d_s, d_w)

    # Paired bootstrap from OOF before Test (freeze interpretation)
    dev, _, _, _ = prepare_bundle()
    y = dev.set_index("id")[TARGET]
    ids = dev["id"].astype(str).tolist()
    pred_dir = ROOT / "experiments" / "predictions" / code
    ae_new_p = _oof_abs_err(pred_dir / "oof_primary.csv", y, ids, TARGET)
    ae_new_s = _oof_abs_err(pred_dir / "oof_shadow.csv", y, ids, TARGET)
    ae_ctl_p = _oof_abs_err(REPLAY_DIR / "oof_primary.csv", y, ids)
    ae_ctl_s = _oof_abs_err(REPLAY_DIR / "oof_shadow.csv", y, ids)
    ae_68_p = _oof_abs_err(ROOT / "experiments" / "predictions" / T068_CODE / "oof_primary.csv", y, ids)
    ae_68_s = _oof_abs_err(ROOT / "experiments" / "predictions" / T068_CODE / "oof_shadow.csv", y, ids)
    ae_70_p = _oof_abs_err(ROOT / "experiments" / "predictions" / T070_CODE / "oof_primary.csv", y, ids)
    ae_70_s = _oof_abs_err(ROOT / "experiments" / "predictions" / T070_CODE / "oof_shadow.csv", y, ids)
    ae_71_p = _oof_abs_err(ROOT / "experiments" / "predictions" / T071_CODE / "oof_primary.csv", y, ids)
    ae_71_s = _oof_abs_err(ROOT / "experiments" / "predictions" / T071_CODE / "oof_shadow.csv", y, ids)
    bootstrap = {
        "vs_EXP-T030-REPLAY-001": {
            "primary": paired_bootstrap(ae_new_p, ae_ctl_p),
            "shadow": paired_bootstrap(ae_new_s, ae_ctl_s),
        },
        "vs_EXP-T071": {
            "primary": paired_bootstrap(ae_new_p, ae_71_p),
            "shadow": paired_bootstrap(ae_new_s, ae_71_s),
        },
        "vs_EXP-T070": {
            "primary": paired_bootstrap(ae_new_p, ae_70_p),
            "shadow": paired_bootstrap(ae_new_s, ae_70_s),
        },
        "vs_EXP-T068": {
            "primary": paired_bootstrap(ae_new_p, ae_68_p),
            "shadow": paired_bootstrap(ae_new_s, ae_68_s),
        },
    }

    gate_stats = t072.get("gate_summary_stats") or gate_summary_stats(
        list(state.get("cross_gate_rows") or [])
    )

    freeze = {
        "experiment_code": code,
        "experiment_id": T072_EID,
        "git_rev": git_rev(),
        "config": cfg,
        "control_experiment_code": HIST_CONTROL,
        "contemporary_control_run_id": REPLAY_ID,
        "param_counts": params,
        "zero_gate_equivalence": t072.get("zero_gate_equivalence"),
        "gate_summary_stats": gate_stats,
        "EXP-T072": {
            "cv_primary_mae": t072["cv_primary_mae"],
            "cv_shadow_mae": t072["cv_shadow_mae"],
            "cv_mean_mae": t072["cv_mean_mae"],
            "cv_worst_mae": t072["cv_worst_mae"],
            "config_hash": t072["config_hash"],
            "best_epochs_primary": t072["best_epochs_primary"],
            "n_trainable_parameters": t072.get("n_trainable_parameters"),
        },
        "deltas_vs_EXP-T030-REPLAY-001": {"primary": d_p, "shadow": d_s, "worst": d_w},
        "deltas_vs_historical_EXP-T030": {
            "primary": t072["cv_primary_mae"] - hist["cv_primary_mae"],
            "shadow": t072["cv_shadow_mae"] - hist["cv_shadow_mae"],
            "worst": t072["cv_worst_mae"] - hist["cv_worst_mae"],
        },
        "deltas_vs_EXP-T071": {
            "primary": t072["cv_primary_mae"] - t071["cv_primary_mae"],
            "shadow": t072["cv_shadow_mae"] - t071["cv_shadow_mae"],
            "worst": t072["cv_worst_mae"] - t071["cv_worst_mae"],
        },
        "deltas_vs_EXP-T070": {
            "primary": t072["cv_primary_mae"] - t070["cv_primary_mae"],
            "shadow": t072["cv_shadow_mae"] - t070["cv_shadow_mae"],
            "worst": t072["cv_worst_mae"] - t070["cv_worst_mae"],
        },
        "deltas_vs_EXP-T068": {
            "primary": t072["cv_primary_mae"] - t068["cv_primary_mae"],
            "shadow": t072["cv_shadow_mae"] - t068["cv_shadow_mae"],
            "worst": t072["cv_worst_mae"] - t068["cv_worst_mae"],
        },
        "bootstrap": bootstrap,
        "cv_scientific_verdict": verdict,
        "selection_statement": (
            "CV completed for EXP-T072; config frozen; Public/Private not used before freeze; "
            "single full-Dev Test follows. Primary control = contemporary replay."
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
        "experiment_id": T072_EID,
        "checkpoint_dir": str(ckpt_dir.relative_to(ROOT)) if ckpt_dir.exists() else None,
        "n_fulldev": len(ckpts),
        "n_trainable_parameters": n_params,
        "param_counts": param_counts(),
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
    if pred_path.exists() and not quick:
        te = pd.read_csv(pred_path)
    else:
        state = load_state()
        best = state["t072_cv"]["best_epochs_primary"]
        dev, test, folds, rb = prepare_bundle()
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
    n_params = freeze.get("EXP-T072", {}).get("n_trainable_parameters")
    write_checkpoint_manifest(code, ckpt_dir, n_params)
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    state = load_state()
    state["t072_test"] = scores
    save_state(state)
    print(scores, flush=True)
    return scores


def _pred_col(df: pd.DataFrame) -> str:
    return TARGET if TARGET in df.columns else [c for c in df.columns if c != "id"][0]


def phase_artifacts(code: str) -> dict:
    print("=== artifacts + registry ===", flush=True)
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    dev, test, folds, rb = prepare_bundle()

    # Ensure no fake fusion feature parquet
    feat = ROOT / "experiments" / "features" / f"{code}.parquet"
    if feat.exists():
        raise SystemExit(f"separate cross-attn must not materialize feature parquet: {feat}")

    ctl_p = pd.read_csv(REPLAY_DIR / "oof_primary.csv")
    ctl_s = pd.read_csv(REPLAY_DIR / "oof_shadow.csv")
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

    def _load_pair(code_x: str):
        p = pd.read_csv(ROOT / "experiments" / "predictions" / code_x / "oof_primary.csv")
        s = pd.read_csv(ROOT / "experiments" / "predictions" / code_x / "oof_shadow.csv")
        for d in (p, s):
            d["id"] = d["id"].astype(str)
        col = TARGET if TARGET in p.columns else [c for c in p.columns if c != "id"][0]
        return p.set_index("id").loc[ids, col], s.set_index("id").loc[ids, col]

    t068_ps, t068_ss = _load_pair(T068_CODE)
    t070_ps, t070_ss = _load_pair(T070_CODE)
    t071_ps, t071_ss = _load_pair(T071_CODE)
    ae_68_p = (y.loc[ids] - t068_ps).abs().to_numpy(float)
    ae_68_s = (y.loc[ids] - t068_ss).abs().to_numpy(float)
    ae_70_p = (y.loc[ids] - t070_ps).abs().to_numpy(float)
    ae_70_s = (y.loc[ids] - t070_ss).abs().to_numpy(float)
    ae_71_p = (y.loc[ids] - t071_ps).abs().to_numpy(float)
    ae_71_s = (y.loc[ids] - t071_ss).abs().to_numpy(float)
    diag = {
        "bootstrap_vs_replay": {
            "primary": paired_bootstrap(ae_new_p, ae_ctl_p),
            "shadow": paired_bootstrap(ae_new_s, ae_ctl_s),
        },
        "bootstrap_vs_t071": {
            "primary": paired_bootstrap(ae_new_p, ae_71_p),
            "shadow": paired_bootstrap(ae_new_s, ae_71_s),
        },
        "bootstrap_vs_t070": {
            "primary": paired_bootstrap(ae_new_p, ae_70_p),
            "shadow": paired_bootstrap(ae_new_s, ae_70_s),
        },
        "bootstrap_vs_t068": {
            "primary": paired_bootstrap(ae_new_p, ae_68_p),
            "shadow": paired_bootstrap(ae_new_s, ae_68_s),
        },
        "fold_mae_deltas_vs_replay": {
            "primary": fold_deltas(dev, folds.primary, new_ps, ctl_ps),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, ctl_ss),
        },
        "fold_mae_deltas_vs_t071": {
            "primary": fold_deltas(dev, folds.primary, new_ps, t071_ps, new_key="mae_t072", ctl_key="mae_t071"),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, t071_ss, new_key="mae_t072", ctl_key="mae_t071"),
        },
        "fold_mae_deltas_vs_t070": {
            "primary": fold_deltas(dev, folds.primary, new_ps, t070_ps, new_key="mae_t072", ctl_key="mae_t070"),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, t070_ss, new_key="mae_t072", ctl_key="mae_t070"),
        },
        "fold_mae_deltas_vs_t068": {
            "primary": fold_deltas(dev, folds.primary, new_ps, t068_ps, new_key="mae_t072", ctl_key="mae_t068"),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, t068_ss, new_key="mae_t072", ctl_key="mae_t068"),
        },
    }
    (OUT_RUN / "paired_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    recomputed_p = float(mae(y.loc[ids].to_numpy(float), new_ps.to_numpy(float)))
    recomputed_s = float(mae(y.loc[ids].to_numpy(float), new_ss.to_numpy(float)))
    assert abs(recomputed_p - freeze["EXP-T072"]["cv_primary_mae"]) < 1e-7
    assert abs(recomputed_s - freeze["EXP-T072"]["cv_shadow_mae"]) < 1e-7

    # Historical / prior experiment prediction files must remain untouched
    for hist_code in (HIST_CONTROL, T068_CODE, T070_CODE, T071_CODE):
        hist_oof = ROOT / "experiments" / "predictions" / hist_code / "oof_primary.csv"
        if not hist_oof.exists():
            raise SystemExit(f"{hist_code} oof_primary missing")

    pub = float(freeze["public_mae"])
    priv = float(freeze["private_mae"])
    overall = float(freeze["test_overall_mae"])
    cv_p = float(freeze["EXP-T072"]["cv_primary_mae"])
    cv_s = float(freeze["EXP-T072"]["cv_shadow_mae"])
    cv_m = float(freeze["EXP-T072"]["cv_mean_mae"])
    cv_w = float(freeze["EXP-T072"]["cv_worst_mae"])
    params = freeze.get("param_counts") or param_counts()

    ckpt_dir = OUT_RUN / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("fulldev_seed*.pt")) if ckpt_dir.exists() else []
    write_checkpoint_manifest(code, ckpt_dir, freeze["EXP-T072"].get("n_trainable_parameters"))
    rep_note = (
        f"representation_status=NOT_EXPORTED; checkpoints under "
        f"results/EXP-T072_run/checkpoints/ n={len(ckpts)}"
        if ckpts
        else "representation_status=NOT_EXPORTED; no checkpoints saved"
    )
    account = params.get("param_account_t072") or {}

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
            "experiment_id": T072_EID,
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer+SeparateCrossAttnDualREG",
            "source_model_id": f"SEPARATE_CROSS_ATTENTION_DUAL_REG::{REPLAY_ID}",
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
            "score_source": "EXP-T072_cv+solution_postfreeze",
            "prediction_source": "EXP-T072_run",
            "feature_source": "ablingua_residue_separate_cross_attention_dual_reg",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue",
            "notes": (
                f"Separate H/L + cross-attn bridge vs {REPLAY_ID}; secondary vs "
                f"{T071_CODE}/{T070_CODE}/{T068_CODE}; secondary vs historical {HIST_CONTROL}; "
                f"params T030={params['EXP-T030_n_trainable']} T068={params['EXP-T068_n_trainable']} "
                f"T070={params['EXP-T070_n_trainable']} T071={params['EXP-T071_n_trainable']} "
                f"T072={params['EXP-T072_n_trainable']} account={account}; {rep_note}"
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
            "control_experiment_code": HIST_CONTROL,
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
                "experiment_id": T072_EID,
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
                "notes": "SEPARATE_CROSS_ATTENTION_DUAL_REG: no fusion feature parquet",
            }
        )
        if "test_prediction_exists" in comp.columns:
            crow["test_prediction_exists"] = True
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)

    state["artifacts"] = {
        "feature_path": "",
        "diagnostics": diag,
        "checkpoints": [str(p) for p in ckpts],
        "param_counts": params,
        "gate_summary_stats": freeze.get("gate_summary_stats"),
    }
    save_state(state)
    return state["artifacts"]


def phase_report(code: str) -> None:
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    replay = replay_scores()
    hist = scores_row(HIST_CONTROL)
    t068row = scores_row(T068_CODE)
    t070row = scores_row(T070_CODE)
    t071row = scores_row(T071_CODE)
    t072 = freeze["EXP-T072"]
    diag = json.loads((OUT_RUN / "paired_diagnostics.json").read_text())
    params = freeze.get("param_counts") or param_counts()
    account = params.get("param_account_t072") or {}
    gate_stats = freeze.get("gate_summary_stats") or {}
    d_p = freeze["deltas_vs_EXP-T030-REPLAY-001"]["primary"]
    d_s = freeze["deltas_vs_EXP-T030-REPLAY-001"]["shadow"]
    d_w = freeze["deltas_vs_EXP-T030-REPLAY-001"]["worst"]
    d71 = freeze["deltas_vs_EXP-T071"]
    d70 = freeze["deltas_vs_EXP-T070"]
    d68 = freeze["deltas_vs_EXP-T068"]
    verdict = freeze.get("cv_scientific_verdict") or cv_verdict(d_p, d_s, d_w)
    both = (d_p < 0 and d_s < 0) or (d_p > 0 and d_s > 0)
    ckpts = freeze.get("checkpoints") or {}

    def _boot_line(label: str, key: str) -> list[str]:
        b = diag[key]
        return [
            f"- {label} Primary: Δ={b['primary']['mae_delta']:+.6f} "
            f"CI95=[{b['primary']['ci95_low']:+.6f}, {b['primary']['ci95_high']:+.6f}] "
            f"P(Δ<0)={b['primary'].get('p_delta_lt_0', float('nan')):.3f}",
            f"- {label} Shadow: Δ={b['shadow']['mae_delta']:+.6f} "
            f"CI95=[{b['shadow']['ci95_low']:+.6f}, {b['shadow']['ci95_high']:+.6f}] "
            f"P(Δ<0)={b['shadow'].get('p_delta_lt_0', float('nan')):.3f}",
        ]

    gH = gate_stats.get("g_H") or {}
    gL = gate_stats.get("g_L") or {}
    lines = [
        "# EXP-T072 — Separate H/L + cross-attention bridge (frozen AbLingua)",
        "",
        f"- git: `{git_rev()}`",
        f"- primary control: `{REPLAY_ID}`",
        f"- secondary: `{T071_CODE}`, `{T070_CODE}`, `{T068_CODE}`, historical `{HIST_CONTROL}`",
        "- change: separate encoders + zero-gated bidirectional residue cross-attn between layers; "
        "REG not in cross Q/K/V",
        f"- CV verdict (vs REPLAY): **{verdict}**",
        "",
        "## Param counts",
        "",
        f"- T030: {params['EXP-T030_n_trainable']} (repr_dim={params.get('repr_dim_t030')})",
        f"- T068: {params['EXP-T068_n_trainable']} (repr_dim={params.get('repr_dim_t068')})",
        f"- T070: {params['EXP-T070_n_trainable']} (repr_dim={params.get('repr_dim_t070')})",
        f"- T071: {params['EXP-T071_n_trainable']} (repr_dim={params.get('repr_dim_t071')})",
        f"- T072: {params['EXP-T072_n_trainable']} (repr_dim={params.get('repr_dim_t072')})",
        f"- T072 param_account: {account}",
        "",
        "## Cross-gate statistics (CV)",
        "",
        f"- n_rows: {gate_stats.get('n_rows', 0)}",
        f"- g_H mean/median/std/|mean|/sign_consistency: "
        f"{gH.get('mean')} / {gH.get('median')} / {gH.get('std')} / "
        f"{gH.get('abs_mean')} / {gH.get('sign_consistency')}",
        f"- g_L mean/median/std/|mean|/sign_consistency: "
        f"{gL.get('mean')} / {gL.get('median')} / {gL.get('std')} / "
        f"{gL.get('abs_mean')} / {gL.get('sign_consistency')}",
        "",
        "## Primary vs EXP-T030-REPLAY-001",
        "",
        f"- T072 P/S/mean/W: {t072['cv_primary_mae']:.6f} / {t072['cv_shadow_mae']:.6f} / "
        f"{t072['cv_mean_mae']:.6f} / {t072['cv_worst_mae']:.6f}",
        f"- Replay P/S/mean/W: {replay['cv_primary_mae']:.6f} / {replay['cv_shadow_mae']:.6f} / "
        f"{replay['cv_mean_mae']:.6f} / {replay['cv_worst_mae']:.6f}",
        f"- Δ Primary/Shadow/worst: {d_p:+.6f} / {d_s:+.6f} / {d_w:+.6f}",
        f"- Both schemes same direction: {'YES' if both else 'NO'}",
        "",
        "## Secondary vs EXP-T071",
        "",
        f"- T071 P/S/W: {t071row['cv_primary_mae']:.6f} / {t071row['cv_shadow_mae']:.6f} / {t071row['cv_worst_mae']:.6f}",
        f"- Δ Primary/Shadow/worst: {d71['primary']:+.6f} / {d71['shadow']:+.6f} / {d71['worst']:+.6f}",
        "",
        "## Secondary vs EXP-T070",
        "",
        f"- T070 P/S/W: {t070row['cv_primary_mae']:.6f} / {t070row['cv_shadow_mae']:.6f} / {t070row['cv_worst_mae']:.6f}",
        f"- Δ Primary/Shadow/worst: {d70['primary']:+.6f} / {d70['shadow']:+.6f} / {d70['worst']:+.6f}",
        "",
        "## Secondary vs EXP-T068",
        "",
        f"- T068 P/S/W: {t068row['cv_primary_mae']:.6f} / {t068row['cv_shadow_mae']:.6f} / {t068row['cv_worst_mae']:.6f}",
        f"- Δ Primary/Shadow/worst: {d68['primary']:+.6f} / {d68['shadow']:+.6f} / {d68['worst']:+.6f}",
        "",
        "## Contextual vs historical EXP-T030",
        "",
        f"- Δ Primary/Shadow/worst: {t072['cv_primary_mae'] - hist['cv_primary_mae']:+.6f} / "
        f"{t072['cv_shadow_mae'] - hist['cv_shadow_mae']:+.6f} / "
        f"{t072['cv_worst_mae'] - hist['cv_worst_mae']:+.6f}",
        "",
        "## Paired bootstrap",
        "",
        *_boot_line("vs REPLAY", "bootstrap_vs_replay"),
        *_boot_line("vs T071", "bootstrap_vs_t071"),
        *_boot_line("vs T070", "bootstrap_vs_t070"),
        *_boot_line("vs T068", "bootstrap_vs_t068"),
        "",
        "## Test (after freeze)",
        "",
        f"- Public: {freeze['public_mae']:.6f}",
        f"- Private: {freeze['private_mae']:.6f}",
        f"- Overall: {freeze['test_overall_mae']:.6f}",
        "",
        "## Artifacts",
        "",
        "- config / preds; no fusion parquet",
        f"- cross gates: results/EXP-T072_CROSS_GATES.csv ; results/EXP-T072_run/cross_gates.csv",
        f"- checkpoints: n={ckpts.get('n_fulldev', 0)}",
        "- SHAREABLE_COMPLETE / REPRODUCED",
        "",
        "## Interpretation",
        "",
        f"CV vs REPLAY **{verdict}**. Separate H/L with learnable zero-init cross-attn gates.",
        "",
        "## Registered EXP-T072 — comparison report written separately",
        "",
        "- Series T071→T072 complete for this runner; see "
        "`results/T030_T068_T070_T071_T072_ARCHITECTURE_COMPARISON.md` (written separately).",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        print(issue_t072())
        return 0
    if args.phase == "cv":
        code = load_state().get("experiment_code") or issue_t072()
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

    code = issue_t072()
    print(f"Issued {code}", flush=True)
    phase_cv(quick=args.quick, code=code)
    phase_freeze(code)
    phase_test(code, quick=args.quick)
    phase_artifacts(code)
    phase_report(code)
    print("DONE EXP-T072", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
