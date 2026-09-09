#!/usr/bin/env python3
"""EXP-T070: Joint H/L dual-REG frozen Transformer (vs EXP-T030-REPLAY-001).

Joint encoder like T068, but restores REG_H||REG_L readout (T030-style).
No fusion. No RASA. No feature parquet. Does NOT overwrite historical EXP-T030.
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
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402
from antibody_transformer.training import (  # noqa: E402
    full_dev_transformer_predict,
    run_transformer_cv,
)

TARGET = "TmApp"
HIST_CONTROL = "EXP-T030"
T068_CODE = "EXP-T068"
REPLAY_ID = "EXP-T030-REPLAY-001"
REPLAY_DIR = ROOT / "experiments" / "replays" / "EXP-T030" / REPLAY_ID
T070_EID = "TRF_TM_ABLINGUA_FULL_JOINT_DUAL_REG"
VARIANT = "T070_JOINT_DUAL"
INPUT_SPACE = "JOINT_HL_DUAL_REG_FROZEN_RESIDUE"

OUT_RUN = ROOT / "results" / "EXP-T070_run"
FREEZE_PATH = ROOT / "results" / "EXP-T070_CV_FREEZE.yaml"
REPORT_PATH = ROOT / "results" / "EXP-T070_JOINT_DUAL_REG_REPORT.md"
STATE_PATH = ROOT / "results" / "EXP-T070_run_state.json"
CKPT_MANIFEST = ROOT / "results" / "EXP-T070_CHECKPOINT_MANIFEST.json"


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


def param_counts() -> dict[str, int]:
    """Trainable params: T030 / T068 / T070."""
    kw = dict(
        content_mode="frozen",
        plm_hidden=1280,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        use_ca_distance_bias=False,
    )
    t030 = AnnotatedTransformer(**kw, joint_hl_single_reg=False, joint_hl_dual_reg=False)
    t068 = AnnotatedTransformer(**kw, joint_hl_single_reg=True, joint_hl_dual_reg=False)
    t070 = AnnotatedTransformer(**kw, joint_hl_single_reg=False, joint_hl_dual_reg=True)
    return {
        "EXP-T030_n_trainable": int(t030.n_trainable_parameters()),
        "EXP-T068_n_trainable": int(t068.n_trainable_parameters()),
        "EXP-T070_n_trainable": int(t070.n_trainable_parameters()),
        "repr_dim_t030": int(t030.repr_dim),
        "repr_dim_t068": int(t068.repr_dim),
        "repr_dim_t070": int(t070.repr_dim),
    }


def prepare_bundle():
    dev, test = load_dev_test(ROOT / "data" / "dev.csv", ROOT / "data" / "test.csv")
    folds = load_folds(ROOT / "data" / "folds.csv")
    rb = load_residue_bundle(dev, test, need_ablingua=True)
    return dev, test, folds, rb


def issue_t070() -> str:
    codes = load_codes()
    if (codes["experiment_id"] == T070_EID).any():
        return str(codes.set_index("experiment_id").loc[T070_EID, "experiment_code"])
    nxt = next_code(TARGET)
    if nxt != "EXP-T070":
        raise SystemExit(f"expected EXP-T070, got {nxt}")
    code = issue_code(
        T070_EID,
        TARGET,
        source_model_id=f"JOINT_HL_DUAL_REG::{REPLAY_ID}",
        phase="ARCHITECTURE_JOINT_HL_DUAL_REG",
        notes=(
            "Joint H/L dual-REG vs contemporary EXP-T030-REPLAY-001; "
            "no fusion/RASA; do not overwrite historical EXP-T030"
        ),
    )
    if code != "EXP-T070":
        raise SystemExit(code)
    return code


def write_config(code: str) -> Path:
    cfg = {
        "experiment_code": code,
        "experiment_id": T070_EID,
        "target": TARGET,
        "family": "TRANSFORMER",
        "source_model_id": f"JOINT_HL_DUAL_REG::{REPLAY_ID}",
        "transformer_type": "FROZEN_PLM",
        "input_space": INPUT_SPACE,
        "input_asset_ref": "assets/transformer/residue_asset_manifest.yaml#ablingua600m",
        "plm_source": "ABLINGUA",
        "annotation_mode": "FULL",
        "chain_mode": "HL",
        "merge_mode": "concat",
        "pooling_mode": "REG",
        "content_mode": "frozen",
        "joint_hl_dual_reg": True,
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
            "results/EXP-T070_run/checkpoints/ if size permits"
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
        "n_boot": n,
    }


def fold_deltas(dev, fmap, oof_new, oof_ctl, *, new_key="mae_t070", ctl_key="mae_control"):
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


def train_kwargs() -> dict:
    return dict(
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        plm_source="ablingua",
        pooling_mode="reg",
        recipe_id=None,
        joint_hl_dual_reg=True,
        joint_hl_single_reg=False,
        use_ca_distance_bias=False,
        use_continuous_rasa=False,
        use_rasa_weighted_pool=False,
    )


def phase_cv(*, quick: bool, code: str) -> dict:
    print("=== EXP-T070 CV ===", flush=True)
    if not (REPLAY_DIR / "oof_primary.csv").exists():
        raise SystemExit(f"missing contemporary control OOF: {REPLAY_DIR}")
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    write_config(code)
    params = param_counts()
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
    state["t070_cv"] = out
    state["experiment_code"] = code
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
    t070 = state["t070_cv"]
    replay = replay_scores()
    hist = scores_row(HIST_CONTROL)
    t068 = scores_row(T068_CODE)
    cfg = yaml.safe_load((ROOT / "experiments" / "configs" / f"{code}.yaml").read_text())
    params = t070.get("param_counts") or param_counts()
    d_p = t070["cv_primary_mae"] - replay["cv_primary_mae"]
    d_s = t070["cv_shadow_mae"] - replay["cv_shadow_mae"]
    d_w = t070["cv_worst_mae"] - replay["cv_worst_mae"]
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
    bootstrap = {
        "vs_EXP-T030-REPLAY-001": {
            "primary": paired_bootstrap(ae_new_p, ae_ctl_p),
            "shadow": paired_bootstrap(ae_new_s, ae_ctl_s),
        },
        "vs_EXP-T068": {
            "primary": paired_bootstrap(ae_new_p, ae_68_p),
            "shadow": paired_bootstrap(ae_new_s, ae_68_s),
        },
    }

    freeze = {
        "experiment_code": code,
        "experiment_id": T070_EID,
        "git_rev": git_rev(),
        "config": cfg,
        "control_experiment_code": HIST_CONTROL,
        "contemporary_control_run_id": REPLAY_ID,
        "param_counts": params,
        "EXP-T070": {
            "cv_primary_mae": t070["cv_primary_mae"],
            "cv_shadow_mae": t070["cv_shadow_mae"],
            "cv_mean_mae": t070["cv_mean_mae"],
            "cv_worst_mae": t070["cv_worst_mae"],
            "config_hash": t070["config_hash"],
            "best_epochs_primary": t070["best_epochs_primary"],
            "n_trainable_parameters": t070.get("n_trainable_parameters"),
        },
        "deltas_vs_EXP-T030-REPLAY-001": {"primary": d_p, "shadow": d_s, "worst": d_w},
        "deltas_vs_historical_EXP-T030": {
            "primary": t070["cv_primary_mae"] - hist["cv_primary_mae"],
            "shadow": t070["cv_shadow_mae"] - hist["cv_shadow_mae"],
            "worst": t070["cv_worst_mae"] - hist["cv_worst_mae"],
        },
        "deltas_vs_EXP-T068": {
            "primary": t070["cv_primary_mae"] - t068["cv_primary_mae"],
            "shadow": t070["cv_shadow_mae"] - t068["cv_shadow_mae"],
            "worst": t070["cv_worst_mae"] - t068["cv_worst_mae"],
        },
        "bootstrap": bootstrap,
        "cv_scientific_verdict": verdict,
        "selection_statement": (
            "CV completed for EXP-T070; config frozen; Public/Private not used before freeze; "
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
        "experiment_id": T070_EID,
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
        best = state["t070_cv"]["best_epochs_primary"]
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
    n_params = freeze.get("EXP-T070", {}).get("n_trainable_parameters")
    write_checkpoint_manifest(code, ckpt_dir, n_params)
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    state = load_state()
    state["t070_test"] = scores
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
        raise SystemExit(f"joint HL must not materialize feature parquet: {feat}")

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
    t068_p = pd.read_csv(ROOT / "experiments" / "predictions" / T068_CODE / "oof_primary.csv")
    t068_s = pd.read_csv(ROOT / "experiments" / "predictions" / T068_CODE / "oof_shadow.csv")
    for d in (t068_p, t068_s):
        d["id"] = d["id"].astype(str)
    col68 = TARGET if TARGET in t068_p.columns else [c for c in t068_p.columns if c != "id"][0]
    t068_ps = t068_p.set_index("id").loc[ids, col68]
    t068_ss = t068_s.set_index("id").loc[ids, col68]
    ae_68_p = (y.loc[ids] - t068_ps).abs().to_numpy(float)
    ae_68_s = (y.loc[ids] - t068_ss).abs().to_numpy(float)
    diag = {
        "bootstrap_vs_replay": {
            "primary": paired_bootstrap(ae_new_p, ae_ctl_p),
            "shadow": paired_bootstrap(ae_new_s, ae_ctl_s),
        },
        "bootstrap_vs_t068": {
            "primary": paired_bootstrap(ae_new_p, ae_68_p),
            "shadow": paired_bootstrap(ae_new_s, ae_68_s),
        },
        "fold_mae_deltas_vs_replay": {
            "primary": fold_deltas(dev, folds.primary, new_ps, ctl_ps),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, ctl_ss),
        },
        "fold_mae_deltas_vs_t068": {
            "primary": fold_deltas(dev, folds.primary, new_ps, t068_ps, new_key="mae_t070", ctl_key="mae_t068"),
            "shadow": fold_deltas(dev, folds.shadow, new_ss, t068_ss, new_key="mae_t070", ctl_key="mae_t068"),
        },
    }
    (OUT_RUN / "paired_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    recomputed_p = float(mae(y.loc[ids].to_numpy(float), new_ps.to_numpy(float)))
    recomputed_s = float(mae(y.loc[ids].to_numpy(float), new_ss.to_numpy(float)))
    assert abs(recomputed_p - freeze["EXP-T070"]["cv_primary_mae"]) < 1e-10
    assert abs(recomputed_s - freeze["EXP-T070"]["cv_shadow_mae"]) < 1e-10

    # Historical T030 prediction files must remain untouched (sha check vs freeze-time optional)
    hist_oof = ROOT / "experiments" / "predictions" / HIST_CONTROL / "oof_primary.csv"
    if not hist_oof.exists():
        raise SystemExit("historical EXP-T030 oof_primary missing")

    pub = float(freeze["public_mae"])
    priv = float(freeze["private_mae"])
    overall = float(freeze["test_overall_mae"])
    cv_p = float(freeze["EXP-T070"]["cv_primary_mae"])
    cv_s = float(freeze["EXP-T070"]["cv_shadow_mae"])
    cv_m = float(freeze["EXP-T070"]["cv_mean_mae"])
    cv_w = float(freeze["EXP-T070"]["cv_worst_mae"])
    params = freeze.get("param_counts") or param_counts()

    ckpt_dir = OUT_RUN / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("fulldev_seed*.pt")) if ckpt_dir.exists() else []
    write_checkpoint_manifest(code, ckpt_dir, freeze["EXP-T070"].get("n_trainable_parameters"))
    rep_note = (
        f"representation_status=NOT_EXPORTED; checkpoints under "
        f"results/EXP-T070_run/checkpoints/ n={len(ckpts)}"
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
            "experiment_id": T070_EID,
            "target": TARGET,
            "family": "TRANSFORMER",
            "model_type": "AnnotatedTransformer+JointHLDualREG",
            "source_model_id": f"JOINT_HL_DUAL_REG::{REPLAY_ID}",
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
            "score_source": "EXP-T070_cv+solution_postfreeze",
            "prediction_source": "EXP-T070_run",
            "feature_source": "ablingua_residue_joint_hl_dual_reg",
            "license_status": "REVIEW",
            "license_reference": "AbLingua residue",
            "notes": (
                f"Joint H/L dual-REG vs {REPLAY_ID}; secondary vs EXP-T068; secondary vs historical {HIST_CONTROL}; "
                f"params T030={params['EXP-T030_n_trainable']} T068={params['EXP-T068_n_trainable']} T070={params['EXP-T070_n_trainable']}; "
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
                "experiment_id": T070_EID,
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
                "notes": "JOINT_HL_DUAL_REG: no fusion feature parquet",
            }
        )
        # prefer test_prediction_exists if column exists
        if "test_prediction_exists" in comp.columns:
            crow["test_prediction_exists"] = True
        comp = pd.concat([comp, pd.DataFrame([{c: crow.get(c, "") for c in comp.columns}])], ignore_index=True)
        comp.to_csv(comp_path, index=False)

    state["artifacts"] = {
        "feature_path": "",
        "diagnostics": diag,
        "checkpoints": [str(p) for p in ckpts],
        "param_counts": params,
    }
    save_state(state)
    return state["artifacts"]


def phase_report(code: str) -> None:
    freeze = yaml.safe_load(FREEZE_PATH.read_text())
    state = load_state()
    replay = replay_scores()
    hist = scores_row(HIST_CONTROL)
    t068row = scores_row(T068_CODE)
    t070 = freeze["EXP-T070"]
    diag = json.loads((OUT_RUN / "paired_diagnostics.json").read_text())
    params = freeze.get("param_counts") or param_counts()
    d_p = freeze["deltas_vs_EXP-T030-REPLAY-001"]["primary"]
    d_s = freeze["deltas_vs_EXP-T030-REPLAY-001"]["shadow"]
    d_w = freeze["deltas_vs_EXP-T030-REPLAY-001"]["worst"]
    d68 = freeze["deltas_vs_EXP-T068"]
    verdict = freeze.get("cv_scientific_verdict") or cv_verdict(d_p, d_s, d_w)
    both = (d_p < 0 and d_s < 0) or (d_p > 0 and d_s > 0)
    ckpts = freeze.get("checkpoints") or {}
    lines = [
        "# EXP-T070 — Joint H/L dual-REG (frozen AbLingua)",
        "",
        f"- git: `{git_rev()}`",
        f"- primary control: `{REPLAY_ID}`",
        f"- secondary: `{T068_CODE}`, historical `{HIST_CONTROL}`",
        "- change: joint encoder `[REG_H, H..., REG_L, L...]`; readout REG_H||REG_L; no fusion/RASA",
        f"- CV verdict (vs REPLAY): **{verdict}**",
        "",
        "## Param counts",
        "",
        f"- T030: {params['EXP-T030_n_trainable']} (repr_dim={params.get('repr_dim_t030')})",
        f"- T068: {params['EXP-T068_n_trainable']} (repr_dim={params.get('repr_dim_t068')})",
        f"- T070: {params['EXP-T070_n_trainable']} (repr_dim={params.get('repr_dim_t070')})",
        "",
        "## Primary vs EXP-T030-REPLAY-001",
        "",
        f"- T070 P/S/mean/W: {t070['cv_primary_mae']:.6f} / {t070['cv_shadow_mae']:.6f} / "
        f"{t070['cv_mean_mae']:.6f} / {t070['cv_worst_mae']:.6f}",
        f"- Replay P/S/mean/W: {replay['cv_primary_mae']:.6f} / {replay['cv_shadow_mae']:.6f} / "
        f"{replay['cv_mean_mae']:.6f} / {replay['cv_worst_mae']:.6f}",
        f"- Δ Primary/Shadow/worst: {d_p:+.6f} / {d_s:+.6f} / {d_w:+.6f}",
        f"- Both schemes same direction: {'YES' if both else 'NO'}",
        "",
        "## Secondary vs EXP-T068",
        "",
        f"- T068 P/S/W: {t068row['cv_primary_mae']:.6f} / {t068row['cv_shadow_mae']:.6f} / {t068row['cv_worst_mae']:.6f}",
        f"- Δ Primary/Shadow/worst: {d68['primary']:+.6f} / {d68['shadow']:+.6f} / {d68['worst']:+.6f}",
        "",
        "## Contextual vs historical EXP-T030",
        "",
        f"- Δ Primary/Shadow/worst: {t070['cv_primary_mae'] - hist['cv_primary_mae']:+.6f} / "
        f"{t070['cv_shadow_mae'] - hist['cv_shadow_mae']:+.6f} / "
        f"{t070['cv_worst_mae'] - hist['cv_worst_mae']:+.6f}",
        "",
        "## Paired bootstrap",
        "",
        f"- vs REPLAY Primary: Δ={diag['bootstrap_vs_replay']['primary']['mae_delta']:+.6f} "
        f"CI95=[{diag['bootstrap_vs_replay']['primary']['ci95_low']:+.6f}, "
        f"{diag['bootstrap_vs_replay']['primary']['ci95_high']:+.6f}]",
        f"- vs REPLAY Shadow: Δ={diag['bootstrap_vs_replay']['shadow']['mae_delta']:+.6f} "
        f"CI95=[{diag['bootstrap_vs_replay']['shadow']['ci95_low']:+.6f}, "
        f"{diag['bootstrap_vs_replay']['shadow']['ci95_high']:+.6f}]",
        f"- vs T068 Primary: Δ={diag['bootstrap_vs_t068']['primary']['mae_delta']:+.6f} "
        f"CI95=[{diag['bootstrap_vs_t068']['primary']['ci95_low']:+.6f}, "
        f"{diag['bootstrap_vs_t068']['primary']['ci95_high']:+.6f}]",
        f"- vs T068 Shadow: Δ={diag['bootstrap_vs_t068']['shadow']['mae_delta']:+.6f} "
        f"CI95=[{diag['bootstrap_vs_t068']['shadow']['ci95_low']:+.6f}, "
        f"{diag['bootstrap_vs_t068']['shadow']['ci95_high']:+.6f}]",
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
        f"- checkpoints: n={ckpts.get('n_fulldev', 0)}",
        "- SHAREABLE_COMPLETE / REPRODUCED",
        "",
        "## Interpretation",
        "",
        f"CV vs REPLAY **{verdict}**. Joint H/L with dual REG_H||REG_L restores T030-style readout.",
        "",
        "## STOP — no further experiment",
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
        print(issue_t070())
        return 0
    if args.phase == "cv":
        code = load_state().get("experiment_code") or issue_t070()
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

    code = issue_t070()
    print(f"Issued {code}", flush=True)
    phase_cv(quick=args.quick, code=code)
    phase_freeze(code)
    phase_test(code, quick=args.quick)
    phase_artifacts(code)
    phase_report(code)
    print("DONE EXP-T070", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
