#!/usr/bin/env python3
"""DL protocol V2: fold-local LR selection + VAL checkpoint ensemble (no full-Dev).

NOT nested CV. Uses existing Primary/Shadow TRAIN/VAL/TEST rotations only.
"""
from __future__ import annotations

import hashlib
import json
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
from .training import (
    AbDataset,
    _batch_to_device,
    _y_stats,
    build_transformer,
    collate_batch,
    mae,
    resolve_plm_source,
)

PROTOCOL_ID = "DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2"
DEFAULT_SEED = 101
MAX_EPOCHS = 200
PATIENCE = 20
ETA_MIN = 0.0


def lr_grid(lr_ref: float) -> list[float]:
    return [0.1 * lr_ref, 0.3 * lr_ref, 1.0 * lr_ref, 3.0 * lr_ref]


def state_dict_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items(), key=lambda x: x[0]):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_t030_model(rb: ResidueBundle) -> nn.Module:
    """T030 scientific architecture only (no joint/cross/RASA/CA/fusion)."""
    presets = load_presets()
    return build_transformer(
        content_mode="frozen",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        rb=rb,
        presets=presets,
        plm_source="ablingua",
        pooling_mode="reg",
        use_continuous_rasa=False,
        use_rasa_weighted_pool=False,
        use_ca_distance_bias=False,
        joint_hl_single_reg=False,
        joint_hl_dual_reg=False,
        joint_hl_chain_specific_dual_reg=False,
        use_cross_attention_bridge=False,
    )


def train_one_candidate(
    *,
    train_ids: list[str],
    val_ids: list[str],
    y_map: dict[str, float],
    rb: ResidueBundle,
    device: torch.device,
    seed: int,
    lr: float,
    ckpt_path: Path,
    experiment_code: str,
    scheme: str,
    fold: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    quick: bool = False,
) -> dict[str, Any]:
    """Train on TRAIN only; select best VAL-MAE checkpoint. No TV retrain."""
    presets = load_presets()
    ncfg = presets["neural"]
    if quick:
        max_epochs = min(5, max_epochs)
        patience = min(2, patience)
    bs = int(ncfg["batch_size"])
    wd = float(ncfg["weight_decay"])
    clip = float(ncfg["gradient_clip_norm"])
    beta = float(ncfg["smooth_l1_beta"])

    y_tr = np.asarray([y_map[a] for a in train_ids], float)
    y_va = np.asarray([y_map[a] for a in val_ids], float)
    mu, sd = _y_stats(y_tr)

    _set_seed(seed)
    model = build_t030_model(rb).to(device)
    init_hash = state_dict_sha256(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=ETA_MIN)
    loss_fn = nn.SmoothL1Loss(beta=beta)

    def make_loader(ids, y_arr, shuffle):
        ds = AbDataset(ids, y_arr, rb, content_mode="frozen", plm_source="ablingua")
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_batch)

    history_rows: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    best_step = 0
    epochs_since = 0
    global_step = 0
    stopped_early = False
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))

        # VAL
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
                y_z = (y - mu) / sd
                va_losses.append(float(loss_fn(pred_z, y_z).cpu()))
                va_preds.append((pred_z * sd + mu).cpu().numpy())
        va_pred = np.concatenate(va_preds) if va_preds else np.array([])
        val_mae = float(mae(y_va, va_pred)) if len(va_pred) else float("inf")
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        cur_lr = float(opt.param_groups[0]["lr"])

        is_best = val_mae < best_val - 1e-12
        if is_best:
            best_val = val_mae
            best_epoch = epoch
            best_step = global_step
            epochs_since = 0
            torch.save(
                {
                    "model": deepcopy(model.state_dict()),
                    "mu": mu,
                    "sd": sd,
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "lr": lr,
                    "val_mae": val_mae,
                    "init_hash": init_hash,
                    "seed": seed,
                },
                ckpt_path,
            )
        else:
            epochs_since += 1

        history_rows.append(
            {
                "experiment_code": experiment_code,
                "scheme": scheme,
                "fold": fold,
                "seed": seed,
                "lr_candidate": lr,
                "epoch": epoch,
                "optimizer_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "learning_rate": cur_lr,
                "best_val_mae_so_far": best_val,
                "epochs_since_improvement": epochs_since,
                "is_best": bool(is_best),
                "stopped_early": False,
            }
        )

        sched.step()

        if epochs_since >= patience:
            stopped_early = True
            history_rows[-1]["stopped_early"] = True
            break

    return {
        "lr": lr,
        "best_val_mae": best_val,
        "best_epoch": best_epoch,
        "best_optimizer_step": best_step,
        "final_epoch": history_rows[-1]["epoch"] if history_rows else 0,
        "stopped_early": stopped_early,
        "checkpoint": str(ckpt_path),
        "init_hash": init_hash,
        "history": history_rows,
        "mu": mu,
        "sd": sd,
        "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


@torch.no_grad()
def predict_with_checkpoint(
    *,
    ckpt_path: Path,
    ids: list[str],
    rb: ResidueBundle,
    device: torch.device,
    y_placeholder: Optional[np.ndarray] = None,
) -> np.ndarray:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_t030_model(rb).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    mu = float(blob["mu"])
    sd = float(blob["sd"])
    y_arr = y_placeholder if y_placeholder is not None else np.zeros(len(ids), dtype=float)
    ds = AbDataset(ids, y_arr, rb, content_mode="frozen", plm_source="ablingua")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_batch)
    preds = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        batch.pop("y", None)
        batch.pop("fixed", None)
        pred_z = model(batch)
        preds.append((pred_z * sd + mu).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def select_lr(candidates: list[dict], tol: float = 1e-12) -> dict:
    """Argmin best_val_mae; tie → smaller lr."""
    best = None
    for c in candidates:
        if best is None:
            best = c
            continue
        if c["best_val_mae"] < best["best_val_mae"] - tol:
            best = c
        elif abs(c["best_val_mae"] - best["best_val_mae"]) <= tol and c["lr"] < best["lr"]:
            best = c
    assert best is not None
    return best


def run_protocol_v2(
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
    """Full Primary+Shadow fold-local LR protocol. No nested CV. No full-Dev."""
    presets = load_presets()
    lr_ref = float(presets["neural"]["lr"])
    lrs = lr_grid(lr_ref)
    if quick:
        lrs = [lr_ref]  # single candidate for smoke

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

    oof_val = {"primary": pd.Series(np.nan, index=dev_ids, dtype=float), "shadow": pd.Series(np.nan, index=dev_ids, dtype=float)}
    oof_test = {"primary": pd.Series(np.nan, index=dev_ids, dtype=float), "shadow": pd.Series(np.nan, index=dev_ids, dtype=float)}
    ext_fold_preds = {
        "primary": np.zeros((5, len(test_ids)), dtype=float),
        "shadow": np.zeros((5, len(test_ids)), dtype=float),
    }

    for scheme_name, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
        for k in range(5):
            tr, va, te = tvt_split(fmap, k, dev_ids)
            # QC: TEST not in train/val
            assert set(te).isdisjoint(tr) and set(te).isdisjoint(va)
            assert set(va).isdisjoint(tr)

            cand_summaries = []
            init_hashes = []
            for lr in lrs:
                ckpt = ckpt_root / f"{scheme_name}_k{k}_lr{lr:.6g}_seed{seed}.pt"
                print(
                    f"=== {experiment_code} {scheme_name} fold={k} lr={lr:.6g} ===",
                    flush=True,
                )
                summary = train_one_candidate(
                    train_ids=tr,
                    val_ids=va,
                    y_map=y_map,
                    rb=rb,
                    device=device_t,
                    seed=seed,
                    lr=lr,
                    ckpt_path=ckpt,
                    experiment_code=experiment_code,
                    scheme=scheme_name,
                    fold=k,
                    quick=quick,
                )
                init_hashes.append(summary["init_hash"])
                cand_summaries.append(summary)
                all_history.extend(summary["history"])

            # equivalent init within fold
            if len(set(init_hashes)) != 1:
                raise RuntimeError(
                    f"init hash mismatch within {scheme_name} fold {k}: {init_hashes}"
                )

            selected = select_lr(cand_summaries)
            selected_rows.append(
                {
                    "scheme": scheme_name,
                    "fold": k,
                    "selected_lr": selected["lr"],
                    "best_epoch": selected["best_epoch"],
                    "best_optimizer_step": selected["best_optimizer_step"],
                    "best_val_mae": selected["best_val_mae"],
                    "final_epoch": selected["final_epoch"],
                    "stopped_early": selected["stopped_early"],
                    "checkpoint": selected["checkpoint"],
                    "init_hash": selected["init_hash"],
                    "seed": seed,
                }
            )

            ckpt_path = Path(selected["checkpoint"])
            # VAL / fold TEST / external Test with restored checkpoint
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
                            "lr": c["lr"],
                            "best_val_mae": c["best_val_mae"],
                            "best_epoch": c["best_epoch"],
                            "best_optimizer_step": c["best_optimizer_step"],
                            "final_epoch": c["final_epoch"],
                            "stopped_early": c["stopped_early"],
                            "selected": abs(c["lr"] - selected["lr"]) < 1e-15,
                        }
                        for c in cand_summaries
                    ],
                }
            )

    # OOF scores
    y_dev = np.asarray([y_map[a] for a in dev_ids], float)
    scores = {}
    for scheme_name in ("primary", "shadow"):
        assert not oof_val[scheme_name].isna().any(), f"OOF VAL gaps {scheme_name}"
        assert not oof_test[scheme_name].isna().any(), f"OOF TEST gaps {scheme_name}"
        # each id once
        assert len(oof_val[scheme_name]) == 162
    val_p = float(mae(y_dev, oof_val["primary"].loc[dev_ids].to_numpy(float)))
    val_s = float(mae(y_dev, oof_val["shadow"].loc[dev_ids].to_numpy(float)))
    test_p = float(mae(y_dev, oof_test["primary"].loc[dev_ids].to_numpy(float)))
    test_s = float(mae(y_dev, oof_test["shadow"].loc[dev_ids].to_numpy(float)))
    scores["oof_val"] = {
        "primary": val_p,
        "shadow": val_s,
        "mean": (val_p + val_s) / 2.0,
        "worst": max(val_p, val_s),
    }
    scores["oof_test"] = {
        "primary": test_p,
        "shadow": test_s,
        "mean": (test_p + test_s) / 2.0,
        "worst": max(test_p, test_s),
    }

    # External aggregates
    ext = {}
    for scheme_name in ("primary", "shadow"):
        mat = ext_fold_preds[scheme_name]  # [5, N]
        ext[f"{scheme_name}_mean"] = mat.mean(axis=0)
        ext[f"{scheme_name}_median"] = np.median(mat, axis=0)
        for k in range(5):
            ext[f"{scheme_name}_fold{k}"] = mat[k]

    # Save history / selected
    hist_df = pd.DataFrame(all_history)
    sel_df = pd.DataFrame(selected_rows)

    model0 = build_t030_model(rb)
    n_params = int(sum(p.numel() for p in model0.parameters() if p.requires_grad))

    summary = {
        "protocol_id": PROTOCOL_ID,
        "experiment_code": experiment_code,
        "seed": seed,
        "lr_ref": lr_ref,
        "lr_grid": lrs,
        "max_epochs": MAX_EPOCHS if not quick else min(5, MAX_EPOCHS),
        "patience": PATIENCE if not quick else min(2, PATIENCE),
        "scheduler": {"name": "CosineAnnealingLR", "T_max": MAX_EPOCHS, "eta_min": ETA_MIN},
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
