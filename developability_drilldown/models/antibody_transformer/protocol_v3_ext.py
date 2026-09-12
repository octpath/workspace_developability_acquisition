#!/usr/bin/env python3
"""V3 protocol extensions: capacity overrides + H047 late fusion.

Keeps DL_FOLDLOCAL_COSINE_V3 optimizer/schedule unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import FoldMaps, ResidueBundle, tvt_split
from .h047_aux_features import FoldPreprocessor, H047AuxFeatureStore
from .global_surface_conditioning import FUSION_MODES as GLOBAL_SURFACE_MODES
from .global_surface_conditioning import GlobalSurfaceConditioningModel
from .late_fusion import DirectLateFusionModel, LateFusionModel
from .metrics import mae
from .protocol_v3 import (
    BATCH_SIZE,
    DEFAULT_SEED,
    MAX_EPOCHS,
    PATIENCE,
    PLATFORM_ID,
    SMOOTH_L1_BETA,
    WEIGHT_DECAY,
    _atomic_write_json,
    _dataset_plm_source,
    _manifest_path,
    _set_seed,
    _try_load_complete_manifest,
    coarse_lr_grid,
    eta_min_for,
    lr_at_epoch,
    normalize_arch_flags,
)
from .protocol_v2 import select_lr, state_dict_sha256
from .training import AbDataset, _batch_to_device, _y_stats, build_transformer, collate_batch
from .config import load_presets


def normalize_capacity(capacity: Optional[dict]) -> Optional[dict]:
    if not capacity:
        return None
    return {
        "d_model": int(capacity["d_model"]),
        "n_layers": int(capacity["n_layers"]),
        "n_heads": int(capacity["n_heads"]),
        "dim_feedforward": int(
            capacity.get("dim_feedforward", capacity.get("ff_dim"))
        ),
    }


def candidate_config_hash_ext(
    arch: Optional[dict],
    content_mode: str,
    merge_mode: str,
    plm_source: Optional[str],
    chain_mode: str = "HL",
    capacity: Optional[dict] = None,
    fusion_bundle_id: Optional[str] = None,
    feature_artifact_hash: Optional[str] = None,
    fusion_mode: Optional[str] = None,
) -> str:
    flags = normalize_arch_flags(arch)
    fm = fusion_mode or ("late_concat_aux32" if fusion_bundle_id else None)
    if fm == "late_concat_aux32":
        aux_mlp = "Linear64_GELU_Drop_Linear32_GELU"
    elif fm == "late_concat_direct":
        aux_mlp = None
    elif fm in GLOBAL_SURFACE_MODES:
        aux_mlp = f"global_surface_conditioning:{fm}:rank16"
    else:
        aux_mlp = None
    payload = {
        "arch": flags,
        "content_mode": content_mode,
        "merge_mode": merge_mode,
        "plm_source": plm_source,
        "chain_mode": chain_mode,
        "platform_id": PLATFORM_ID,
        "capacity": normalize_capacity(capacity),
        "fusion_bundle_id": fusion_bundle_id,
        "feature_artifact_hash": feature_artifact_hash,
        "late_fusion": bool(fusion_bundle_id),
        "fusion_mode": fm,
        "aux_mlp": aux_mlp if fusion_bundle_id else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def build_platform_model_ext(
    rb: ResidueBundle,
    arch: Optional[dict] = None,
    *,
    content_mode: str = "frozen",
    merge_mode: str = "concat",
    plm_source: Optional[str] = "ablingua",
    annotation_mode: str = "full",
    chain_mode: str = "HL",
    capacity: Optional[dict] = None,
    aux_dim: Optional[int] = None,
    fusion_mode: str = "late_concat_aux32",
) -> nn.Module:
    flags = normalize_arch_flags(arch)
    presets = load_presets()
    plm = None if content_mode == "scratch" else (plm_source or "ablingua")
    cm = "H_ONLY" if chain_mode == "H_ONLY" else "HL"
    mm = "h_only" if cm == "H_ONLY" else merge_mode
    backbone = build_transformer(
        content_mode=content_mode,
        annotation_mode=annotation_mode,
        merge_mode=mm,
        chain_mode=cm,
        rb=rb,
        presets=presets,
        plm_source=plm or "ablingua",
        pooling_mode="reg",
        capacity=normalize_capacity(capacity),
        **flags,
    )
    if aux_dim is None:
        return backbone
    drop = float(presets["neural"]["dropout"])
    if fusion_mode == "late_concat_direct":
        return DirectLateFusionModel(backbone, int(aux_dim), dropout=drop)
    if fusion_mode == "late_concat_aux32":
        return LateFusionModel(backbone, int(aux_dim), dropout=drop)
    if fusion_mode in GLOBAL_SURFACE_MODES:
        return GlobalSurfaceConditioningModel(
            backbone, int(aux_dim), fusion_mode, dropout=drop
        )
    raise ValueError(f"unknown fusion_mode: {fusion_mode}")


def train_one_candidate_ext(
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
    patience: int = PATIENCE,
    quick: bool = False,
    arch: Optional[dict] = None,
    content_mode: str = "frozen",
    merge_mode: str = "concat",
    plm_source: Optional[str] = "ablingua",
    chain_mode: str = "HL",
    capacity: Optional[dict] = None,
    X_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    fusion_bundle_id: Optional[str] = None,
    feature_artifact_hash: Optional[str] = None,
    prep: Optional[FoldPreprocessor] = None,
    fusion_mode: str = "late_concat_aux32",
) -> dict[str, Any]:
    presets = load_presets()
    ncfg = presets["neural"]
    flags = normalize_arch_flags(arch)
    aux_dim = None if X_train is None else int(X_train.shape[1])
    cfg_hash = candidate_config_hash_ext(
        flags,
        content_mode,
        merge_mode,
        plm_source,
        chain_mode=chain_mode,
        capacity=capacity,
        fusion_bundle_id=fusion_bundle_id,
        feature_artifact_hash=feature_artifact_hash,
        fusion_mode=fusion_mode if aux_dim is not None else None,
    )
    resumed = _try_load_complete_manifest(
        ckpt_path=ckpt_path,
        experiment_code=experiment_code,
        scheme=scheme,
        fold=fold,
        seed=seed,
        lr0=lr0,
        config_hash=cfg_hash,
    )
    if resumed is not None:
        print(
            f"=== resume skip {experiment_code} {scheme} fold={fold} lr0={lr0:.0e} ===",
            flush=True,
        )
        return resumed

    if quick:
        max_epochs = min(8, max_epochs)
        patience = min(3, patience)
    bs = int(BATCH_SIZE)
    wd = float(WEIGHT_DECAY)
    clip = float(ncfg["gradient_clip_norm"])
    beta = float(SMOOTH_L1_BETA)
    eta_min = eta_min_for(lr0)
    ds_plm = _dataset_plm_source(content_mode, plm_source)

    y_tr = np.asarray([y_map[a] for a in train_ids], float)
    y_va = np.asarray([y_map[a] for a in val_ids], float)
    mu, sd = _y_stats(y_tr)

    model = build_platform_model_ext(
        rb,
        flags,
        content_mode=content_mode,
        merge_mode=merge_mode,
        plm_source=plm_source,
        chain_mode=chain_mode,
        capacity=capacity,
        aux_dim=aux_dim,
        fusion_mode=fusion_mode,
    ).to(device)
    model.load_state_dict(deepcopy(init_state))
    assert state_dict_sha256(model) == init_hash

    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss(beta=beta)
    g = torch.Generator()
    g.manual_seed(seed)

    def make_loader(ids, y_arr, X_fixed, shuffle):
        ds = AbDataset(
            ids,
            y_arr,
            rb,
            content_mode=content_mode,
            plm_source=ds_plm,
            fixed_X=X_fixed,
            chain_mode=chain_mode,
        )
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
    use_fusion = X_train is not None

    for epoch in range(1, max_epochs + 1):
        cur_lr = lr_at_epoch(epoch, lr0, eta_min)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        model.train()
        tr_loader = make_loader(train_ids, y_tr, X_train, True)
        epoch_losses = []
        for batch in tr_loader:
            batch = _batch_to_device(batch, device)
            y = batch.pop("y")
            fixed = batch.pop("fixed", None)
            opt.zero_grad(set_to_none=True)
            if use_fusion:
                pred_z = model(batch, fixed)
            else:
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
            break

        model.eval()
        with torch.no_grad():
            va_loader = make_loader(val_ids, y_va, X_val, False)
            preds, ys = [], []
            for batch in va_loader:
                batch = _batch_to_device(batch, device)
                y = batch.pop("y")
                fixed = batch.pop("fixed", None)
                if use_fusion:
                    pred_z = model(batch, fixed)
                else:
                    pred_z = model(batch)
                preds.append((pred_z * sd + mu).cpu().numpy())
                ys.append(y.cpu().numpy())
            pred_va = np.concatenate(preds)
            y_va_np = np.concatenate(ys)
            val_mae = float(mae(y_va_np, pred_va))

        history_rows.append(
            {
                "scheme": scheme,
                "fold": fold,
                "seed": seed,
                "initial_lr": lr0,
                "epoch": epoch,
                "optimizer_step": global_step,
                "learning_rate": cur_lr,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "val_mae": val_mae,
            }
        )
        if val_mae < best_val - 1e-12:
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
                    "arch": flags,
                    "content_mode": content_mode,
                    "merge_mode": merge_mode,
                    "plm_source": plm_source,
                    "chain_mode": chain_mode,
                    "capacity": normalize_capacity(capacity),
                    "fusion_bundle_id": fusion_bundle_id,
                    "feature_artifact_hash": feature_artifact_hash,
                    "aux_dim": aux_dim,
                    "fusion_mode": fusion_mode,
                    "prep": prep,
                    "platform_id": PLATFORM_ID,
                    "init_hash": init_hash,
                    "config_hash": cfg_hash,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val,
                    "initial_lr": lr0,
                    "seed": seed,
                },
                ckpt_path,
            )
        else:
            epochs_since += 1
            if epochs_since >= patience:
                stopped_early = True
                break

    summary = {
        "initial_lr": lr0,
        "eta_min": eta_min,
        "best_epoch": best_epoch,
        "best_optimizer_step": best_step,
        "lr_at_best_epoch": lr_at_best,
        "best_val_mae": best_val if math.isfinite(best_val) else float("inf"),
        "final_epoch": history_rows[-1]["epoch"] if history_rows else 0,
        "stopped_early": stopped_early,
        "numerical_failure": numerical_failure,
        "checkpoint": str(ckpt_path) if ckpt_path.exists() and math.isfinite(best_val) else "",
        "init_hash": init_hash,
        "history": history_rows,
        "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "arch": flags,
        "content_mode": content_mode,
        "merge_mode": merge_mode,
        "plm_source": plm_source,
        "chain_mode": chain_mode,
        "capacity": normalize_capacity(capacity),
        "fusion_bundle_id": fusion_bundle_id,
        "platform_id": PLATFORM_ID,
        "config_hash": cfg_hash,
        "resumed": False,
    }
    if ckpt_path.exists() and math.isfinite(float(best_val)):
        _atomic_write_json(
            _manifest_path(ckpt_path),
            {
                "status": "complete",
                "experiment_code": experiment_code,
                "scheme": scheme,
                "fold": fold,
                "seed": seed,
                "initial_lr": lr0,
                "config_hash": cfg_hash,
                "best_val_mae": best_val,
                "platform_id": PLATFORM_ID,
                "fusion_bundle_id": fusion_bundle_id,
                "summary": summary,
            },
        )
    return summary


@torch.no_grad()
def predict_with_checkpoint_ext(
    *,
    ckpt_path: Path,
    ids: list[str],
    rb: ResidueBundle,
    device: torch.device,
    y_placeholder: Optional[np.ndarray] = None,
    arch: Optional[dict] = None,
    content_mode: str = "frozen",
    merge_mode: str = "concat",
    plm_source: Optional[str] = "ablingua",
    chain_mode: str = "HL",
    capacity: Optional[dict] = None,
    aux_store: Optional[H047AuxFeatureStore] = None,
    fusion_mode: Optional[str] = None,
) -> np.ndarray:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flags = normalize_arch_flags(blob.get("arch") or arch)
    cm = blob.get("content_mode", content_mode)
    mm = blob.get("merge_mode", merge_mode)
    ps = blob.get("plm_source", plm_source)
    chm = blob.get("chain_mode", chain_mode)
    cap = blob.get("capacity") or capacity
    aux_dim = blob.get("aux_dim")
    fm = blob.get("fusion_mode") or fusion_mode or "late_concat_aux32"
    prep: Optional[FoldPreprocessor] = blob.get("prep")
    model = build_platform_model_ext(
        rb,
        flags,
        content_mode=cm,
        merge_mode=mm,
        plm_source=ps,
        chain_mode=chm,
        capacity=cap,
        aux_dim=aux_dim,
        fusion_mode=fm,
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    mu = float(blob["mu"])
    sd = float(blob["sd"])
    y_arr = y_placeholder if y_placeholder is not None else np.zeros(len(ids), dtype=float)
    X_fixed = None
    if aux_dim is not None:
        if aux_store is None or prep is None:
            raise RuntimeError("late-fusion predict requires aux_store + prep in checkpoint")
        X_fixed = aux_store.transform(prep, ids)
    ds = AbDataset(
        ids,
        y_arr,
        rb,
        content_mode=cm,
        plm_source=_dataset_plm_source(cm, ps),
        fixed_X=X_fixed,
        chain_mode=chm,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    preds = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        batch.pop("y", None)
        fixed = batch.pop("fixed", None)
        if aux_dim is not None:
            pred_z = model(batch, fixed)
        else:
            pred_z = model(batch)
        preds.append((pred_z * sd + mu).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def run_protocol_v3_ext(
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
    arch: Optional[dict] = None,
    content_mode: str = "frozen",
    merge_mode: str = "concat",
    plm_source: Optional[str] = "ablingua",
    chain_mode: str = "HL",
    capacity: Optional[dict] = None,
    aux_store: Optional[H047AuxFeatureStore] = None,
    fusion_mode: str = "late_concat_aux32",
) -> dict[str, Any]:
    flags = normalize_arch_flags(arch)
    fusion_bundle_id = aux_store.bundle_id if aux_store is not None else None
    feature_artifact_hash = aux_store.artifact_hash if aux_store is not None else None
    fm = fusion_mode if aux_store is not None else None
    cfg_hash = candidate_config_hash_ext(
        flags,
        content_mode,
        merge_mode,
        plm_source,
        chain_mode=chain_mode,
        capacity=capacity,
        fusion_bundle_id=fusion_bundle_id,
        feature_artifact_hash=feature_artifact_hash,
        fusion_mode=fm,
    )
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
    fold_prep_rows: list[dict] = []
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
            prep = None
            X_tr = X_va = None
            aux_dim = None
            if aux_store is not None:
                prep = aux_store.fit(tr)
                X_tr = aux_store.transform(prep, tr)
                X_va = aux_store.transform(prep, va)
                aux_dim = int(prep.effective_dim)
                fold_prep_rows.append(
                    {
                        "scheme": scheme_name,
                        "fold": k,
                        "bundle": fusion_bundle_id,
                        "prep_hash": prep.hash,
                        "effective_dim": aux_dim,
                        "n_train": len(tr),
                    }
                )

            _set_seed(seed)
            model0 = build_platform_model_ext(
                rb,
                flags,
                content_mode=content_mode,
                merge_mode=merge_mode,
                plm_source=plm_source,
                chain_mode=chain_mode,
                capacity=capacity,
                aux_dim=aux_dim,
                fusion_mode=fusion_mode if aux_dim is not None else "late_concat_aux32",
            )
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
                summary = train_one_candidate_ext(
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
                    arch=flags,
                    content_mode=content_mode,
                    merge_mode=merge_mode,
                    plm_source=plm_source,
                    chain_mode=chain_mode,
                    capacity=capacity,
                    X_train=X_tr,
                    X_val=X_va,
                    fusion_bundle_id=fusion_bundle_id,
                    feature_artifact_hash=feature_artifact_hash,
                    prep=prep,
                    fusion_mode=fusion_mode if aux_dim is not None else "late_concat_aux32",
                )
                init_hashes.append(summary["init_hash"])
                cand_summaries.append(summary)
                all_history.extend(summary["history"])

            if len(set(init_hashes)) != 1:
                raise RuntimeError(
                    f"init hash mismatch within {scheme_name} fold {k}: {init_hashes}"
                )
            finite = [
                c for c in cand_summaries if math.isfinite(c["best_val_mae"]) and c["checkpoint"]
            ]
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
                    "config_hash": cfg_hash,
                    "fusion_bundle_id": fusion_bundle_id,
                }
            )

            ckpt_path = Path(selected["checkpoint"])
            y_va = np.asarray([y_map[a] for a in va], float)
            y_te = np.asarray([y_map[a] for a in te], float)
            pred_va = predict_with_checkpoint_ext(
                ckpt_path=ckpt_path,
                ids=va,
                rb=rb,
                device=device_t,
                y_placeholder=y_va,
                arch=flags,
                content_mode=content_mode,
                merge_mode=merge_mode,
                plm_source=plm_source,
                chain_mode=chain_mode,
                capacity=capacity,
                aux_store=aux_store,
            )
            pred_te = predict_with_checkpoint_ext(
                ckpt_path=ckpt_path,
                ids=te,
                rb=rb,
                device=device_t,
                y_placeholder=y_te,
                arch=flags,
                content_mode=content_mode,
                merge_mode=merge_mode,
                plm_source=plm_source,
                chain_mode=chain_mode,
                capacity=capacity,
                aux_store=aux_store,
            )
            pred_ext = predict_with_checkpoint_ext(
                ckpt_path=ckpt_path,
                ids=test_ids,
                rb=rb,
                device=device_t,
                arch=flags,
                content_mode=content_mode,
                merge_mode=merge_mode,
                plm_source=plm_source,
                chain_mode=chain_mode,
                capacity=capacity,
                aux_store=aux_store,
            )
            oof_val[scheme_name].loc[va] = pred_va
            oof_test[scheme_name].loc[te] = pred_te
            ext_fold_preds[scheme_name][k] = pred_ext

    y_dev = np.asarray([y_map[a] for a in dev_ids], float)
    scores = {
        "oof_val": {
            "primary": float(mae(y_dev, oof_val["primary"].to_numpy(float))),
            "shadow": float(mae(y_dev, oof_val["shadow"].to_numpy(float))),
        },
        "oof_test": {
            "primary": float(mae(y_dev, oof_test["primary"].to_numpy(float))),
            "shadow": float(mae(y_dev, oof_test["shadow"].to_numpy(float))),
        },
    }
    for split in ("oof_val", "oof_test"):
        p, s = scores[split]["primary"], scores[split]["shadow"]
        scores[split]["mean"] = 0.5 * (p + s)
        scores[split]["worst"] = max(p, s)

    ext = {
        "primary_mean": ext_fold_preds["primary"].mean(axis=0),
        "primary_median": np.median(ext_fold_preds["primary"], axis=0),
        "shadow_mean": ext_fold_preds["shadow"].mean(axis=0),
        "shadow_median": np.median(ext_fold_preds["shadow"], axis=0),
        "primary_folds": ext_fold_preds["primary"],
        "shadow_folds": ext_fold_preds["shadow"],
    }

    # param account from a fresh model
    probe = build_platform_model_ext(
        rb,
        flags,
        content_mode=content_mode,
        merge_mode=merge_mode,
        plm_source=plm_source,
        chain_mode=chain_mode,
        capacity=capacity,
        aux_dim=(aux_store.effective_dim if aux_store is not None else None),
        fusion_mode=fusion_mode if aux_store is not None else "late_concat_aux32",
    )
    n_trainable = int(sum(p.numel() for p in probe.parameters() if p.requires_grad))
    is_fusion = isinstance(probe, (LateFusionModel, DirectLateFusionModel, GlobalSurfaceConditioningModel))
    account = (
        probe.transformer.param_account()
        if is_fusion
        else probe.param_account()
    )
    if is_fusion:
        account = dict(account)
        account["late_fusion_aux_head"] = n_trainable - int(account.get("total", 0))
        account["total"] = n_trainable
        account["fusion_mode"] = getattr(probe, "fusion_mode", fusion_mode)
        if isinstance(probe, GlobalSurfaceConditioningModel):
            account["global_surface_conditioning"] = int(probe.n_added_conditioning_parameters())

    summary = {
        "experiment_code": experiment_code,
        "platform_id": PLATFORM_ID,
        "config_hash": cfg_hash,
        "scores": scores,
        "n_trainable": n_trainable,
        "param_account": account,
        "capacity": normalize_capacity(capacity),
        "fusion_bundle_id": fusion_bundle_id,
        "feature_artifact_hash": feature_artifact_hash,
        "fusion_mode": fm,
        "seed": seed,
        "arch": flags,
        "content_mode": content_mode,
        "merge_mode": merge_mode,
        "plm_source": plm_source,
        "chain_mode": chain_mode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if fold_prep_rows:
        pd.DataFrame(fold_prep_rows).to_csv(out_dir / "feature_preprocess_manifest.csv", index=False)

    return {
        "summary": summary,
        "history_df": pd.DataFrame(all_history),
        "selected_df": pd.DataFrame(selected_rows),
        "oof_val": oof_val,
        "oof_test": oof_test,
        "ext": ext,
        "test_ids": test_ids,
        "dev_ids": dev_ids,
        "y_dev": y_dev,
        "cross_gates_df": pd.DataFrame(),
        "geometry_weights_df": pd.DataFrame(),
        "resumed_experiment": False,
    }
