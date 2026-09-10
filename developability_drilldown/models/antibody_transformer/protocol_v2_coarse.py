#!/usr/bin/env python3
"""DL protocol V2 coarse cosine platform (EXP-T074).

Fold-local LR selection + VAL checkpoint ensemble; NO full-Dev; NOT nested CV.

Differs from T073 (protocol_v2.py):
  - coarse initial LR grid: 1e-5 / 1e-4 / 1e-3 / 1e-2
  - cosine decay over t=0..100 to eta_min=0.01*lr0, then hold
  - min_epochs=100 before early-stop allowed
  - numerical-failure abort for non-finite loss/preds
"""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_presets
from .data import FoldMaps, ResidueBundle, tvt_split
from .protocol_v2 import (  # noqa: F401 — re-export DEFAULT_SEED for tests/runners
    DEFAULT_SEED,
    build_t030_model,
    predict_with_checkpoint,
    select_lr,
    state_dict_sha256,
)
from .training import AbDataset, _batch_to_device, _y_stats, collate_batch, mae

PROTOCOL_ID = "DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2_COARSE_COSINE"
COARSE_LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
MAX_EPOCHS = 200
MIN_EPOCHS = 100
PATIENCE = 20
COSINE_EPOCHS = 100  # t in [0, COSINE_EPOCHS] inclusive
ETA_MIN_FRAC = 0.01


def coarse_lr_grid() -> list[float]:
    return list(COARSE_LR_GRID)


def eta_min_for(lr0: float) -> float:
    return ETA_MIN_FRAC * float(lr0)


def lr_at_epoch(epoch: int, lr0: float, eta_min: float, cosine_epochs: int = COSINE_EPOCHS) -> float:
    """LR used at the *start* of 1-indexed ``epoch`` (no post-epoch step rise).

    t = epoch - 1. Cosine for 0 <= t <= cosine_epochs; hold eta_min for t > cosine_epochs.
    """
    t = int(epoch) - 1
    if t >= cosine_epochs:
        return float(eta_min)
    return float(eta_min + 0.5 * (lr0 - eta_min) * (1.0 + math.cos(math.pi * t / cosine_epochs)))


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_nonfinite(*vals: float) -> bool:
    for v in vals:
        if v is None or not math.isfinite(float(v)):
            return True
    return False


def train_one_candidate_coarse(
    *,
    train_ids: list[str],
    val_ids: list[str],
    y_map: dict[str, float],
    rb: ResidueBundle,
    device: torch.device,
    seed: int,
    lr0: float,
    init_state: dict,
    init_hash: str,
    ckpt_path: Path,
    experiment_code: str,
    scheme: str,
    fold: int,
    max_epochs: int = MAX_EPOCHS,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    quick: bool = False,
) -> dict[str, Any]:
    """TRAIN-only; VAL-MAE checkpoint; min_epochs before early stop; cosine then hold."""
    presets = load_presets()
    ncfg = presets["neural"]
    if quick:
        max_epochs = min(8, max_epochs)
        min_epochs = min(3, min_epochs)
        patience = min(2, patience)
    bs = int(ncfg["batch_size"])
    wd = float(ncfg["weight_decay"])
    clip = float(ncfg["gradient_clip_norm"])
    beta = float(ncfg["smooth_l1_beta"])
    eta_min = eta_min_for(lr0)

    y_tr = np.asarray([y_map[a] for a in train_ids], float)
    y_va = np.asarray([y_map[a] for a in val_ids], float)
    mu, sd = _y_stats(y_tr)

    model = build_t030_model(rb).to(device)
    model.load_state_dict(deepcopy(init_state))
    assert state_dict_sha256(model) == init_hash

    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss(beta=beta)
    g = torch.Generator()
    g.manual_seed(seed)

    def make_loader(ids, y_arr, shuffle):
        ds = AbDataset(ids, y_arr, rb, content_mode="frozen", plm_source="ablingua")
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_batch,
            generator=g if shuffle else None,
        )

    history_rows: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    best_step = 0
    lr_at_best = float(lr0)
    epochs_since = 0
    global_step = 0
    stopped_early = False
    numerical_failure = False
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        cur_lr = lr_at_epoch(epoch, lr0, eta_min)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        model.train()
        tr_loader = make_loader(train_ids, y_tr, True)
        epoch_losses = []
        for batch in tr_loader:
            batch = _batch_to_device(batch, device)
            y = batch.pop("y")
            batch.pop("fixed", None)
            opt.zero_grad(set_to_none=True)
            pred_z = model(batch)
            y_z = (y - mu) / sd
            loss = loss_fn(pred_z, y_z)
            if not torch.isfinite(loss).all() or not torch.isfinite(pred_z).all():
                numerical_failure = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
        if numerical_failure:
            history_rows.append(
                {
                    "experiment_code": experiment_code,
                    "scheme": scheme,
                    "fold": fold,
                    "seed": seed,
                    "initial_lr": lr0,
                    "eta_min": eta_min,
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "train_loss": float("nan"),
                    "val_loss": float("nan"),
                    "val_mae": float("nan"),
                    "learning_rate": cur_lr,
                    "best_val_mae_so_far": best_val if math.isfinite(best_val) else float("nan"),
                    "best_epoch_so_far": best_epoch,
                    "epochs_since_improvement": epochs_since,
                    "is_best": False,
                    "early_stop_allowed": epoch >= min_epochs,
                    "stopped_early": True,
                    "numerical_failure": True,
                }
            )
            break

        model.eval()
        va_loader = make_loader(val_ids, y_va, False)
        va_preds = []
        va_losses = []
        with torch.no_grad():
            for batch in va_loader:
                batch = _batch_to_device(batch, device)
                y = batch.pop("y")
                batch.pop("fixed", None)
                pred_z = model(batch)
                if not torch.isfinite(pred_z).all():
                    numerical_failure = True
                    break
                y_z = (y - mu) / sd
                va_losses.append(float(loss_fn(pred_z, y_z).cpu()))
                va_preds.append((pred_z * sd + mu).cpu().numpy())
        if numerical_failure:
            history_rows.append(
                {
                    "experiment_code": experiment_code,
                    "scheme": scheme,
                    "fold": fold,
                    "seed": seed,
                    "initial_lr": lr0,
                    "eta_min": eta_min,
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                    "val_loss": float("nan"),
                    "val_mae": float("nan"),
                    "learning_rate": cur_lr,
                    "best_val_mae_so_far": best_val if math.isfinite(best_val) else float("nan"),
                    "best_epoch_so_far": best_epoch,
                    "epochs_since_improvement": epochs_since,
                    "is_best": False,
                    "early_stop_allowed": epoch >= min_epochs,
                    "stopped_early": True,
                    "numerical_failure": True,
                }
            )
            break

        va_pred = np.concatenate(va_preds) if va_preds else np.array([])
        if len(va_pred) == 0 or not np.isfinite(va_pred).all():
            numerical_failure = True
            history_rows.append(
                {
                    "experiment_code": experiment_code,
                    "scheme": scheme,
                    "fold": fold,
                    "seed": seed,
                    "initial_lr": lr0,
                    "eta_min": eta_min,
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                    "val_loss": float("nan"),
                    "val_mae": float("nan"),
                    "learning_rate": cur_lr,
                    "best_val_mae_so_far": best_val if math.isfinite(best_val) else float("nan"),
                    "best_epoch_so_far": best_epoch,
                    "epochs_since_improvement": epochs_since,
                    "is_best": False,
                    "early_stop_allowed": epoch >= min_epochs,
                    "stopped_early": True,
                    "numerical_failure": True,
                }
            )
            break

        val_mae = float(mae(y_va, va_pred))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        if _is_nonfinite(train_loss, val_loss, val_mae):
            numerical_failure = True
            history_rows.append(
                {
                    "experiment_code": experiment_code,
                    "scheme": scheme,
                    "fold": fold,
                    "seed": seed,
                    "initial_lr": lr0,
                    "eta_min": eta_min,
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "learning_rate": cur_lr,
                    "best_val_mae_so_far": best_val if math.isfinite(best_val) else float("nan"),
                    "best_epoch_so_far": best_epoch,
                    "epochs_since_improvement": epochs_since,
                    "is_best": False,
                    "early_stop_allowed": epoch >= min_epochs,
                    "stopped_early": True,
                    "numerical_failure": True,
                }
            )
            break

        is_best = val_mae < best_val - 1e-12
        if is_best:
            best_val = val_mae
            best_epoch = epoch
            best_step = global_step
            lr_at_best = cur_lr
            epochs_since = 0
            torch.save(
                {
                    "model": deepcopy(model.state_dict()),
                    "mu": mu,
                    "sd": sd,
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "initial_lr": lr0,
                    "eta_min": eta_min,
                    "lr_at_best": lr_at_best,
                    "val_mae": val_mae,
                    "init_hash": init_hash,
                    "seed": seed,
                },
                ckpt_path,
            )
        else:
            epochs_since += 1

        early_ok = epoch >= min_epochs
        history_rows.append(
            {
                "experiment_code": experiment_code,
                "scheme": scheme,
                "fold": fold,
                "seed": seed,
                "initial_lr": lr0,
                "eta_min": eta_min,
                "epoch": epoch,
                "optimizer_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "learning_rate": cur_lr,
                "best_val_mae_so_far": best_val,
                "best_epoch_so_far": best_epoch,
                "epochs_since_improvement": epochs_since,
                "is_best": bool(is_best),
                "early_stop_allowed": early_ok,
                "stopped_early": False,
                "numerical_failure": False,
            }
        )

        # No CosineAnnealingLR.step() — LR set explicitly at epoch start (documented).
        if early_ok and epochs_since >= patience:
            stopped_early = True
            history_rows[-1]["stopped_early"] = True
            break

    if numerical_failure and not math.isfinite(best_val):
        best_val = float("inf")

    return {
        "lr": lr0,
        "initial_lr": lr0,
        "eta_min": eta_min,
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "best_optimizer_step": best_step,
        "lr_at_best_epoch": lr_at_best,
        "final_epoch": history_rows[-1]["epoch"] if history_rows else 0,
        "stopped_early": stopped_early or numerical_failure,
        "numerical_failure": numerical_failure,
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else "",
        "init_hash": init_hash,
        "history": history_rows,
        "mu": mu,
        "sd": sd,
        "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def run_protocol_v2_coarse(
    *,
    experiment_code: str,
    target: str,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    folds: FoldMaps,
    rb: ResidueBundle,
    device: str,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    quick: bool = False,
) -> dict[str, Any]:
    lrs = coarse_lr_grid()
    if quick:
        lrs = [1e-3]

    device_t = torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    y_map = {str(r["id"]): float(r[target]) for _, r in dev.iterrows()}
    dev_ids = dev["id"].astype(str).tolist()
    test_ids = test["id"].astype(str).tolist()

    all_history: list[dict] = []
    selected_rows: list[dict] = []
    fold_results: dict[str, list[dict]] = {"primary": [], "shadow": []}

    oof_val = {
        "primary": pd.Series(np.nan, index=dev_ids, dtype=float),
        "shadow": pd.Series(np.nan, index=dev_ids, dtype=float),
    }
    oof_test = {
        "primary": pd.Series(np.nan, index=dev_ids, dtype=float),
        "shadow": pd.Series(np.nan, index=dev_ids, dtype=float),
    }
    ext_fold_preds = {
        "primary": np.zeros((5, len(test_ids)), dtype=float),
        "shadow": np.zeros((5, len(test_ids)), dtype=float),
    }

    for scheme_name, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
        for k in range(5):
            tr, va, te = tvt_split(fmap, k, dev_ids)
            assert set(te).isdisjoint(tr) and set(te).isdisjoint(va)
            assert set(va).isdisjoint(tr)

            # Canonical init once per fold
            _set_seed(seed)
            model0 = build_t030_model(rb)
            init_state = deepcopy(model0.state_dict())
            init_hash = state_dict_sha256(model0)

            cand_summaries = []
            init_hashes = []
            for lr0 in lrs:
                ckpt = ckpt_root / f"{scheme_name}_k{k}_lr{lr0:.0e}_seed{seed}.pt"
                print(
                    f"=== {experiment_code} {scheme_name} fold={k} lr0={lr0:.0e} ===",
                    flush=True,
                )
                summary = train_one_candidate_coarse(
                    train_ids=tr,
                    val_ids=va,
                    y_map=y_map,
                    rb=rb,
                    device=device_t,
                    seed=seed,
                    lr0=lr0,
                    init_state=init_state,
                    init_hash=init_hash,
                    ckpt_path=ckpt,
                    experiment_code=experiment_code,
                    scheme=scheme_name,
                    fold=k,
                    quick=quick,
                )
                init_hashes.append(summary["init_hash"])
                cand_summaries.append(summary)
                all_history.extend(summary["history"])

            if len(set(init_hashes)) != 1:
                raise RuntimeError(
                    f"init hash mismatch within {scheme_name} fold {k}: {init_hashes}"
                )

            # Prefer finite candidates; select_lr on those with finite best_val
            finite = [c for c in cand_summaries if math.isfinite(c["best_val_mae"]) and c["checkpoint"]]
            if not finite:
                raise RuntimeError(f"all LR candidates failed {scheme_name} fold {k}")
            selected = select_lr(finite)
            selected_rows.append(
                {
                    "scheme": scheme_name,
                    "fold": k,
                    "selected_initial_lr": selected["initial_lr"],
                    "eta_min": selected["eta_min"],
                    "best_epoch": selected["best_epoch"],
                    "best_optimizer_step": selected["best_optimizer_step"],
                    "lr_at_best_epoch": selected["lr_at_best_epoch"],
                    "best_val_mae": selected["best_val_mae"],
                    "final_epoch": selected["final_epoch"],
                    "stopped_early": selected["stopped_early"],
                    "numerical_failure": selected["numerical_failure"],
                    "checkpoint": selected["checkpoint"],
                    "init_hash": selected["init_hash"],
                    "seed": seed,
                }
            )

            ckpt_path = Path(selected["checkpoint"])
            y_va = np.asarray([y_map[a] for a in va], float)
            y_te = np.asarray([y_map[a] for a in te], float)
            pred_va = predict_with_checkpoint(
                ckpt_path=ckpt_path, ids=va, rb=rb, device=device_t, y_placeholder=y_va
            )
            pred_te = predict_with_checkpoint(
                ckpt_path=ckpt_path, ids=te, rb=rb, device=device_t, y_placeholder=y_te
            )
            pred_ext = predict_with_checkpoint(
                ckpt_path=ckpt_path, ids=test_ids, rb=rb, device=device_t
            )

            oof_val[scheme_name].loc[va] = pred_va
            oof_test[scheme_name].loc[te] = pred_te
            ext_fold_preds[scheme_name][k] = pred_ext

            fold_results[scheme_name].append(
                {
                    "fold": k,
                    "selected": selected_rows[-1],
                    "val_mae_recheck": float(mae(y_va, pred_va)),
                    "test_mae": float(mae(y_te, pred_te)),
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "candidates": [
                        {
                            "initial_lr": c["initial_lr"],
                            "eta_min": c["eta_min"],
                            "best_val_mae": c["best_val_mae"],
                            "best_epoch": c["best_epoch"],
                            "best_optimizer_step": c["best_optimizer_step"],
                            "lr_at_best_epoch": c["lr_at_best_epoch"],
                            "final_epoch": c["final_epoch"],
                            "stopped_early": c["stopped_early"],
                            "numerical_failure": c["numerical_failure"],
                            "selected": abs(c["initial_lr"] - selected["initial_lr"]) < 1e-15,
                        }
                        for c in cand_summaries
                    ],
                }
            )

    y_dev = np.asarray([y_map[a] for a in dev_ids], float)
    for scheme_name in ("primary", "shadow"):
        assert not oof_val[scheme_name].isna().any()
        assert not oof_test[scheme_name].isna().any()
    val_p = float(mae(y_dev, oof_val["primary"].loc[dev_ids].to_numpy(float)))
    val_s = float(mae(y_dev, oof_val["shadow"].loc[dev_ids].to_numpy(float)))
    test_p = float(mae(y_dev, oof_test["primary"].loc[dev_ids].to_numpy(float)))
    test_s = float(mae(y_dev, oof_test["shadow"].loc[dev_ids].to_numpy(float)))
    scores = {
        "oof_val": {
            "primary": val_p,
            "shadow": val_s,
            "mean": (val_p + val_s) / 2.0,
            "worst": max(val_p, val_s),
        },
        "oof_test": {
            "primary": test_p,
            "shadow": test_s,
            "mean": (test_p + test_s) / 2.0,
            "worst": max(test_p, test_s),
        },
    }

    ext = {}
    for scheme_name in ("primary", "shadow"):
        mat = ext_fold_preds[scheme_name]
        ext[f"{scheme_name}_mean"] = mat.mean(axis=0)
        ext[f"{scheme_name}_median"] = np.median(mat, axis=0)
        for k in range(5):
            ext[f"{scheme_name}_fold{k}"] = mat[k]

    hist_df = pd.DataFrame(all_history)
    sel_df = pd.DataFrame(selected_rows)
    model0 = build_t030_model(rb)
    n_params = int(sum(p.numel() for p in model0.parameters() if p.requires_grad))
    presets = load_presets()

    summary = {
        "protocol_id": PROTOCOL_ID,
        "experiment_code": experiment_code,
        "seed": seed,
        "lr_grid": lrs,
        "max_epochs": MAX_EPOCHS if not quick else min(8, MAX_EPOCHS),
        "min_epochs": MIN_EPOCHS if not quick else min(3, MIN_EPOCHS),
        "patience": PATIENCE if not quick else min(2, PATIENCE),
        "scheduler": {
            "name": "explicit_cosine_then_hold",
            "cosine_epochs": COSINE_EPOCHS,
            "eta_min_frac": ETA_MIN_FRAC,
            "step_timing": "LR set at beginning of each epoch; no post-epoch CosineAnnealingLR.step()",
        },
        "optimizer": "AdamW",
        "weight_decay": float(presets["neural"]["weight_decay"]),
        "batch_size": int(presets["neural"]["batch_size"]),
        "loss": f"SmoothL1Loss(beta={presets['neural']['smooth_l1_beta']})",
        "gradient_clip_norm": float(presets["neural"]["gradient_clip_norm"]),
        "n_trainable_parameters": n_params,
        "architecture": "T030_FULL_CONCAT_SEPARATE_HL",
        "no_full_dev_refit": True,
        "no_nested_cv": True,
        "scores": scores,
        "selected_lr": selected_rows,
        "fold_results": fold_results,
        "quick": quick,
    }

    return {
        "summary": summary,
        "history_df": hist_df,
        "selected_df": sel_df,
        "oof_val": oof_val,
        "oof_test": oof_test,
        "ext": ext,
        "test_ids": test_ids,
        "dev_ids": dev_ids,
        "y_dev": y_dev,
    }
