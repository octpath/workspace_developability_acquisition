#!/usr/bin/env python3
"""DL_FOLDLOCAL_COSINE_V3 — final fold-local LR + cosine platform (EXP-T075).

NOT nested CV. NO full-Dev refit. NO minimum-epoch constraint.

Architecture settings are experiment-specific; this module only encodes the
shared training platform (optimizer, LR grid, cosine schedule, early stop).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
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
from .protocol_v2 import (  # noqa: F401
    DEFAULT_SEED,
    select_lr,
    state_dict_sha256,
)
from .training import AbDataset, _batch_to_device, _y_stats, build_transformer, collate_batch, mae

PLATFORM_ID = "DL_FOLDLOCAL_COSINE_V3"
PROTOCOL_ID = PLATFORM_ID  # alias
COARSE_LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
MAX_EPOCHS = 200
MIN_EPOCHS = 0  # none — ordinary patience from epoch 1
PATIENCE = 30
T_MAX = 200
ETA_MIN_FRAC = 0.01
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
SMOOTH_L1_BETA = 0.5

# Experiment-specific architecture flags (NOT platform settings).
# Defaults = EXP-T075 / T030 scientific architecture.
# use_cross_geometry_bias is a *modifier* on the cross-attention bridge (ARCH-6G),
# not a 7th mutually exclusive arch switch. HIC batch codes: EXP-H054..EXP-H081.
ARCH_BOOL_KEYS = (
    "joint_hl_single_reg",
    "joint_hl_dual_reg",
    "joint_hl_chain_specific_dual_reg",
    "use_cross_attention_bridge",
    "use_reg_only_cross_attention",
    "use_within_chain_extra_attention",
)
ARCH_T030 = {
    "joint_hl_single_reg": False,
    "joint_hl_dual_reg": False,
    "joint_hl_chain_specific_dual_reg": False,
    "use_cross_attention_bridge": False,
    "use_reg_only_cross_attention": False,
    "use_within_chain_extra_attention": False,
    "cross_gate_mode": "learned",
    "use_cross_geometry_bias": False,
    "share_hl_encoder": True,
}


def normalize_arch_flags(arch: Optional[dict] = None) -> dict:
    out = dict(ARCH_T030)
    if arch:
        for k in ARCH_T030:
            if k not in arch:
                continue
            if k == "cross_gate_mode":
                out[k] = str(arch[k])
            else:
                out[k] = bool(arch[k])
    if out["cross_gate_mode"] not in ("learned", "fixed_one"):
        raise ValueError(
            f"cross_gate_mode must be 'learned' or 'fixed_one', got {out['cross_gate_mode']!r}"
        )
    n_true = sum(1 for k in ARCH_BOOL_KEYS if out[k])
    if n_true > 1:
        raise ValueError(f"mutually exclusive arch flags: {out}")
    if out["use_cross_geometry_bias"] and not out["use_cross_attention_bridge"]:
        raise ValueError("use_cross_geometry_bias requires use_cross_attention_bridge=True")
    return out


def candidate_config_hash(
    arch: Optional[dict],
    content_mode: str,
    merge_mode: str,
    plm_source: Optional[str],
    chain_mode: str = "HL",
) -> str:
    """Stable hash of architecture + content settings under V3 platform."""
    flags = normalize_arch_flags(arch)
    payload = {
        "arch": flags,
        "content_mode": content_mode,
        "merge_mode": merge_mode,
        "plm_source": plm_source,
        "chain_mode": chain_mode,
        "platform_id": PLATFORM_ID,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _manifest_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_suffix(".manifest.json")


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _try_load_complete_manifest(
    *,
    ckpt_path: Path,
    experiment_code: str,
    scheme: str,
    fold: int,
    seed: int,
    lr0: float,
    config_hash: str,
) -> Optional[dict[str, Any]]:
    """Return summary dict if a matching complete sidecar manifest exists."""
    man_path = _manifest_path(ckpt_path)
    if not ckpt_path.exists() or not man_path.exists():
        return None
    try:
        man = json.loads(man_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if man.get("status") != "complete":
        return None
    if man.get("experiment_code") != experiment_code:
        return None
    if man.get("scheme") != scheme:
        return None
    if int(man.get("fold", -1)) != int(fold):
        return None
    if int(man.get("seed", -1)) != int(seed):
        return None
    if abs(float(man.get("initial_lr", float("nan"))) - float(lr0)) > 1e-15:
        return None
    if man.get("config_hash") != config_hash:
        return None
    best = man.get("best_val_mae")
    if best is None or not math.isfinite(float(best)):
        return None
    summary = man.get("summary")
    if not isinstance(summary, dict):
        return None
    # Ensure checkpoint path is current
    out = dict(summary)
    out["checkpoint"] = str(ckpt_path)
    out["resumed"] = True
    return out


def build_platform_model(
    rb: ResidueBundle,
    arch: Optional[dict] = None,
    *,
    content_mode: str = "frozen",
    merge_mode: str = "concat",
    plm_source: Optional[str] = "ablingua",
    annotation_mode: str = "full",
    chain_mode: str = "HL",
    residue_surface_mode: Optional[str] = None,
    residue_surface_dim: int = 0,
) -> nn.Module:
    """Build model under V3 platform; architecture flags are experiment-specific."""
    flags = normalize_arch_flags(arch)
    presets = load_presets()
    plm = None if content_mode == "scratch" else (plm_source or "ablingua")
    cm = "H_ONLY" if chain_mode == "H_ONLY" else "HL"
    mm = "h_only" if cm == "H_ONLY" else merge_mode
    return build_transformer(
        content_mode=content_mode,
        annotation_mode=annotation_mode,
        merge_mode=mm,
        chain_mode=cm,
        rb=rb,
        presets=presets,
        plm_source=plm or "ablingua",
        pooling_mode="reg",
        use_continuous_rasa=False,
        use_rasa_weighted_pool=False,
        use_ca_distance_bias=False,
        residue_surface_mode=residue_surface_mode,
        residue_surface_dim=int(residue_surface_dim or 0),
        **flags,
    )


def build_t030_model(rb: ResidueBundle) -> nn.Module:
    """Backward-compatible T030/T075 builder."""
    return build_platform_model(rb, ARCH_T030)


def coarse_lr_grid() -> list[float]:
    return list(COARSE_LR_GRID)


def eta_min_for(lr0: float) -> float:
    return ETA_MIN_FRAC * float(lr0)


def lr_at_t(t: int, lr0: float, eta_min: float, t_max: int = T_MAX) -> float:
    """Cosine LR at 0-indexed schedule index ``t`` (used DURING that step).

    lr(t) = eta_min + 0.5*(lr0-eta_min)*(1+cos(pi*t/T_max)) for 0 <= t <= T_max;
    for t > T_max return eta_min (should not occur in normal training).
    """
    t = int(t)
    if t <= 0:
        return float(lr0)
    if t >= t_max:
        return float(eta_min)
    return float(eta_min + 0.5 * (lr0 - eta_min) * (1.0 + math.cos(math.pi * t / t_max)))


def lr_at_epoch(epoch: int, lr0: float, eta_min: float, t_max: int = T_MAX) -> float:
    """LR used for gradient updates in 1-indexed ``epoch``.

    Scheduler timing: LR is set at the *beginning* of each epoch to lr_at_t(epoch-1).
    The CSV ``learning_rate`` column reports this value (LR DURING the epoch).
    There is no post-epoch CosineAnnealingLR.step() that would advance LR after logging.
    """
    return lr_at_t(int(epoch) - 1, lr0, eta_min, t_max=t_max)


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


def _dataset_plm_source(content_mode: str, plm_source: Optional[str]) -> str:
    if content_mode == "scratch":
        return plm_source or "ablingua"  # ignored by AbDataset when scratch
    return plm_source or "ablingua"


def train_one_candidate_v3(
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
    residue_surface_mode: Optional[str] = None,
    residue_surface_dim: int = 0,
    surface_prep=None,
) -> dict[str, Any]:
    """TRAIN-only; VAL-MAE checkpoint; ordinary patience (no min_epochs)."""
    presets = load_presets()
    ncfg = presets["neural"]
    flags = normalize_arch_flags(arch)
    cfg_hash = candidate_config_hash(
        flags, content_mode, merge_mode, plm_source, chain_mode=chain_mode
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
    need_ca = bool(flags.get("use_cross_geometry_bias"))

    y_tr = np.asarray([y_map[a] for a in train_ids], float)
    y_va = np.asarray([y_map[a] for a in val_ids], float)
    mu, sd = _y_stats(y_tr)

    model = build_platform_model(
        rb,
        flags,
        content_mode=content_mode,
        merge_mode=merge_mode,
        plm_source=plm_source,
        chain_mode=chain_mode,
        residue_surface_mode=residue_surface_mode,
        residue_surface_dim=residue_surface_dim,
    ).to(device)
    model.load_state_dict(deepcopy(init_state))
    assert state_dict_sha256(model) == init_hash

    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss(beta=beta)
    g = torch.Generator()
    g.manual_seed(seed)

    def make_loader(ids, y_arr, shuffle):
        ds = AbDataset(
            ids,
            y_arr,
            rb,
            content_mode=content_mode,
            plm_source=ds_plm,
            use_cross_geometry_bias=need_ca,
            chain_mode=chain_mode,
            residue_surface_mode=residue_surface_mode,
            surface_prep=surface_prep,
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
                    "platform_id": PLATFORM_ID,
                    "arch": flags,
                    "content_mode": content_mode,
                    "merge_mode": merge_mode,
                    "plm_source": plm_source,
                    "chain_mode": chain_mode,
                    "residue_surface_mode": residue_surface_mode,
                    "residue_surface_dim": residue_surface_dim,
                    "surface_prep": None if surface_prep is None else surface_prep.to_dict(),
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
                "stopped_early": False,
                "numerical_failure": False,
            }
        )

        if epochs_since >= patience:
            stopped_early = True
            history_rows[-1]["stopped_early"] = True
            break

    if numerical_failure and not math.isfinite(best_val):
        best_val = float("inf")

    summary = {
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
        "arch": flags,
        "content_mode": content_mode,
        "merge_mode": merge_mode,
        "plm_source": plm_source,
        "chain_mode": chain_mode,
        "platform_id": PLATFORM_ID,
        "config_hash": cfg_hash,
        "resumed": False,
    }

    # Write resume sidecar only after a successful candidate with a finite best + ckpt.
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
                "summary": summary,
            },
        )
    return summary


@torch.no_grad()
def predict_with_checkpoint(
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
    residue_surface_mode: Optional[str] = None,
    residue_surface_dim: int = 0,
    surface_prep=None,
) -> np.ndarray:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flags = normalize_arch_flags(blob.get("arch") or arch)
    cm = blob.get("content_mode", content_mode)
    mm = blob.get("merge_mode", merge_mode)
    ps = blob.get("plm_source", plm_source)
    chm = blob.get("chain_mode", chain_mode)
    rs_mode = blob.get("residue_surface_mode", residue_surface_mode)
    rs_dim = int(blob.get("residue_surface_dim", residue_surface_dim) or 0)
    prep = surface_prep
    if prep is None and blob.get("surface_prep"):
        from .residue_f1_surface import FoldSurfacePrep

        prep = FoldSurfacePrep.from_dict(blob["surface_prep"])
    model = build_platform_model(
        rb,
        flags,
        content_mode=cm,
        merge_mode=mm,
        plm_source=ps,
        chain_mode=chm,
        residue_surface_mode=rs_mode,
        residue_surface_dim=rs_dim,
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    mu = float(blob["mu"])
    sd = float(blob["sd"])
    y_arr = y_placeholder if y_placeholder is not None else np.zeros(len(ids), dtype=float)
    ds = AbDataset(
        ids,
        y_arr,
        rb,
        content_mode=cm,
        plm_source=_dataset_plm_source(cm, ps),
        use_cross_geometry_bias=bool(flags.get("use_cross_geometry_bias")),
        chain_mode=chm,
        residue_surface_mode=rs_mode,
        surface_prep=prep,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    preds = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        batch.pop("y", None)
        batch.pop("fixed", None)
        pred_z = model(batch)
        preds.append((pred_z * sd + mu).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def _experiment_summary_complete(
    out_dir: Path,
    *,
    config_hash: str,
    lrs: list[float],
    seed: int,
) -> bool:
    """Optional early skip: summary.json matches hash and all 10 selected ckpts exist."""
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        doc = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if doc.get("config_hash") != config_hash:
        return False
    selected = doc.get("selected_lr") or []
    if len(selected) != 10:
        return False
    for row in selected:
        ckpt = row.get("checkpoint")
        if not ckpt or not Path(ckpt).exists():
            return False
    return True


def run_protocol_v3(
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
    residue_surface_mode: Optional[str] = None,
    residue_surface_dim: int = 0,
    residue_surface_compact: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    flags = normalize_arch_flags(arch)
    cfg_hash = candidate_config_hash(
        flags, content_mode, merge_mode, plm_source, chain_mode=chain_mode
    )
    if residue_surface_mode:
        cfg_hash = hashlib.sha256(
            f"{cfg_hash}|rs:{residue_surface_mode}|p:{residue_surface_dim}".encode()
        ).hexdigest()
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

    if _experiment_summary_complete(out_dir, config_hash=cfg_hash, lrs=lrs, seed=seed):
        print(
            f"=== experiment resume: re-infer from selected ckpts {experiment_code} ===",
            flush=True,
        )
        doc = json.loads((out_dir / "summary.json").read_text())
        selected_rows = list(doc.get("selected_lr") or [])
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
        for row in selected_rows:
            scheme_name = str(row["scheme"])
            k = int(row["fold"])
            ckpt_path = Path(row["checkpoint"])
            tr, va, te = tvt_split(
                folds.primary if scheme_name == "primary" else folds.shadow, k, dev_ids
            )
            y_va = np.asarray([y_map[a] for a in va], float)
            y_te = np.asarray([y_map[a] for a in te], float)
            pred_va = predict_with_checkpoint(
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
                residue_surface_mode=residue_surface_mode,
                residue_surface_dim=residue_surface_dim,
            )
            pred_te = predict_with_checkpoint(
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
                residue_surface_mode=residue_surface_mode,
                residue_surface_dim=residue_surface_dim,
            )
            pred_ext = predict_with_checkpoint(
                ckpt_path=ckpt_path,
                ids=test_ids,
                rb=rb,
                device=device_t,
                arch=flags,
                content_mode=content_mode,
                merge_mode=merge_mode,
                plm_source=plm_source,
                chain_mode=chain_mode,
                residue_surface_mode=residue_surface_mode,
                residue_surface_dim=residue_surface_dim,
            )
            oof_val[scheme_name].loc[va] = pred_va
            oof_test[scheme_name].loc[te] = pred_te
            ext_fold_preds[scheme_name][k] = pred_ext
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
        doc["scores"] = scores
        ext = {}
        for scheme_name in ("primary", "shadow"):
            mat = ext_fold_preds[scheme_name]
            ext[f"{scheme_name}_mean"] = mat.mean(axis=0)
            ext[f"{scheme_name}_median"] = np.median(mat, axis=0)
            for k in range(5):
                ext[f"{scheme_name}_fold{k}"] = mat[k]
        hist_path = out_dir / "training_history.csv"
        hist_df = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
        return {
            "summary": doc,
            "history_df": hist_df,
            "selected_df": pd.DataFrame(selected_rows),
            "oof_val": oof_val,
            "oof_test": oof_test,
            "ext": ext,
            "test_ids": test_ids,
            "dev_ids": dev_ids,
            "y_dev": y_dev,
            "cross_gates_df": pd.DataFrame(doc.get("cross_gates") or []),
            "geometry_weights_df": pd.DataFrame(doc.get("geometry_weights") or []),
            "resumed_experiment": True,
        }

    all_history: list[dict] = []
    selected_rows: list[dict] = []
    fold_results: dict[str, list[dict]] = {"primary": [], "shadow": []}
    cross_gates_rows: list[dict] = []
    geometry_weight_rows: list[dict] = []

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

            _set_seed(seed)
            fold_prep = None
            if residue_surface_mode is not None:
                from .residue_f1_surface import fit_fold_surface_prep

                if residue_surface_compact is None:
                    raise ValueError("residue_surface_compact required")
                fold_prep = fit_fold_surface_prep(residue_surface_compact, tr)
            model0 = build_platform_model(
                rb,
                flags,
                content_mode=content_mode,
                merge_mode=merge_mode,
                plm_source=plm_source,
                chain_mode=chain_mode,
                residue_surface_mode=residue_surface_mode,
                residue_surface_dim=residue_surface_dim,
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
                summary = train_one_candidate_v3(
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
                    residue_surface_mode=residue_surface_mode,
                    residue_surface_dim=residue_surface_dim,
                    surface_prep=fold_prep,
                )
                init_hashes.append(summary["init_hash"])
                cand_summaries.append(summary)
                all_history.extend(summary["history"])

            if len(set(init_hashes)) != 1:
                raise RuntimeError(
                    f"init hash mismatch within {scheme_name} fold {k}: {init_hashes}"
                )

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
                residue_surface_mode=residue_surface_mode,
                residue_surface_dim=residue_surface_dim,
            )
            pred_te = predict_with_checkpoint(
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
                residue_surface_mode=residue_surface_mode,
                residue_surface_dim=residue_surface_dim,
            )
            pred_ext = predict_with_checkpoint(
                ckpt_path=ckpt_path,
                ids=test_ids,
                rb=rb,
                device=device_t,
                arch=flags,
                content_mode=content_mode,
                merge_mode=merge_mode,
                plm_source=plm_source,
                chain_mode=chain_mode,
                residue_surface_mode=residue_surface_mode,
                residue_surface_dim=residue_surface_dim,
            )

            oof_val[scheme_name].loc[va] = pred_va
            oof_test[scheme_name].loc[te] = pred_te
            ext_fold_preds[scheme_name][k] = pred_ext

            if (
                flags.get("use_cross_attention_bridge")
                and flags.get("cross_gate_mode") == "learned"
            ):
                blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                mtmp = build_platform_model(
                    rb,
                    flags,
                    content_mode=content_mode,
                    merge_mode=merge_mode,
                    plm_source=plm_source,
                    chain_mode=chain_mode,
                    residue_surface_mode=residue_surface_mode,
                    residue_surface_dim=residue_surface_dim,
                )
                mtmp.load_state_dict(blob["model"])
                gates = mtmp.cross_gate_values()
                cross_gates_rows.append(
                    {
                        "scheme": scheme_name,
                        "fold": k,
                        "g_H": gates["g_H"],
                        "g_L": gates["g_L"],
                        "best_epoch": selected["best_epoch"],
                        "selected_initial_lr": selected["initial_lr"],
                        "lr_at_best_epoch": selected["lr_at_best_epoch"],
                    }
                )

            if flags.get("use_cross_geometry_bias"):
                blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                mtmp = build_platform_model(
                    rb,
                    flags,
                    content_mode=content_mode,
                    merge_mode=merge_mode,
                    plm_source=plm_source,
                    chain_mode=chain_mode,
                    residue_surface_mode=residue_surface_mode,
                    residue_surface_dim=residue_surface_dim,
                )
                mtmp.load_state_dict(blob["model"])
                w = mtmp.cross_geometry_weight_values().numpy()
                row = {
                    "scheme": scheme_name,
                    "fold": k,
                    "best_epoch": selected["best_epoch"],
                    "selected_initial_lr": selected["initial_lr"],
                    "lr_at_best_epoch": selected["lr_at_best_epoch"],
                }
                for hi in range(w.shape[0]):
                    for bi in range(w.shape[1]):
                        row[f"w_h{hi}_b{bi}"] = float(w[hi, bi])
                geometry_weight_rows.append(row)

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
    model0 = build_platform_model(
        rb,
        flags,
        content_mode=content_mode,
        merge_mode=merge_mode,
        plm_source=plm_source,
        chain_mode=chain_mode,
        residue_surface_mode=residue_surface_mode,
        residue_surface_dim=residue_surface_dim,
    )
    n_params = int(sum(p.numel() for p in model0.parameters() if p.requires_grad))
    param_account = model0.param_account() if hasattr(model0, "param_account") else {}

    summary = {
        "platform_id": PLATFORM_ID,
        "protocol_id": PROTOCOL_ID,
        "experiment_code": experiment_code,
        "seed": seed,
        "arch": flags,
        "content_mode": content_mode,
        "merge_mode": merge_mode,
        "plm_source": plm_source,
        "chain_mode": chain_mode,
        "config_hash": cfg_hash,
        "lr_grid": lrs,
        "max_epochs": MAX_EPOCHS if not quick else min(8, MAX_EPOCHS),
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE if not quick else min(3, PATIENCE),
        "scheduler": {
            "name": "explicit_cosine_T_max_200",
            "T_max": T_MAX,
            "eta_min_frac": ETA_MIN_FRAC,
            "step_timing": (
                "LR set at beginning of each epoch to lr_at_t(epoch-1); "
                "CSV learning_rate = LR DURING that epoch; no post-epoch step"
            ),
            "warmup": None,
            "restart": None,
        },
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "loss": f"SmoothL1Loss(beta={SMOOTH_L1_BETA})",
        "gradient_clip_norm": float(load_presets()["neural"]["gradient_clip_norm"]),
        "n_trainable_parameters": n_params,
        "param_account": param_account,
        "architecture": "experiment_specific_under_V3",
        "no_full_dev_refit": True,
        "no_nested_cv": True,
        "scores": scores,
        "selected_lr": selected_rows,
        "fold_results": fold_results,
        "cross_gates": cross_gates_rows,
        "geometry_weights": geometry_weight_rows,
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
        "cross_gates_df": pd.DataFrame(cross_gates_rows),
        "geometry_weights_df": pd.DataFrame(geometry_weight_rows),
    }
