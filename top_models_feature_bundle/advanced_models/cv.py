#!/usr/bin/env python3
"""TVT CV loops for XGBoost and annotation-aware Transformers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import BUNDLE_ROOT, load_presets
from .data import FoldMaps, ResidueBundle, load_folds, tvt_split
from .features import build_recipe_parts, preprocess_parts
from .metrics import cv_mean, cv_worst, mae, seed_dispersion
from .models.annotated_transformer import AnnotatedTransformer
from .models.feature_fusion import FeatureFusionModel
from .models.xgboost_model import fit_xgb_early, fit_xgb_rounds


class SolutionGuard:
    def __init__(self):
        self.accessed = False

    def touch(self):
        self.accessed = True
        raise RuntimeError("solution accessed during CV/selection")


def config_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _y_stats(y: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(y))
    sd = float(np.std(y))
    if sd < 1e-8:
        sd = 1.0
    return mu, sd


class AbDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        y: Optional[np.ndarray],
        rb: ResidueBundle,
        *,
        content_mode: str,
        plm_source: str = "ablingua",  # ablingua | esm2
        fixed_X: Optional[np.ndarray] = None,
    ):
        self.ids = ids
        self.y = y
        self.rb = rb
        self.content_mode = content_mode
        self.plm_source = plm_source
        self.fixed_X = fixed_X
        self.idxs = [rb.id_to_idx[a] for a in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        j = self.idxs[i]
        item = {
            "heavy_aa": self.rb.heavy_aa[j],
            "light_aa": self.rb.light_aa[j],
            "heavy_mask": self.rb.heavy_mask[j],
            "light_mask": self.rb.light_mask[j],
            "heavy_pos": self.rb.heavy_pos[j],
            "light_pos": self.rb.light_pos[j],
            "heavy_imgt": self.rb.heavy_imgt[j],
            "light_imgt": self.rb.light_imgt[j],
            "heavy_region": self.rb.heavy_region[j],
            "light_region": self.rb.light_region[j],
        }
        if self.content_mode == "frozen":
            if self.plm_source == "ablingua":
                item["heavy_plm"] = self.rb.ablingua_h[j]
                item["light_plm"] = self.rb.ablingua_l[j]
            else:
                item["heavy_plm"] = self.rb.esm2_h[j]
        if self.fixed_X is not None:
            item["fixed"] = self.fixed_X[i].astype(np.float32)
        if self.y is not None:
            item["y"] = float(self.y[i])
        return item


def collate_batch(items: list[dict]) -> dict:
    out: dict[str, Any] = {}
    keys = [k for k in items[0].keys() if k != "y"]
    for k in keys:
        arr = np.stack([it[k] for it in items], axis=0)
        if arr.dtype == np.bool_ or arr.dtype == bool:
            out[k] = torch.as_tensor(arr, dtype=torch.bool)
        elif np.issubdtype(arr.dtype, np.floating):
            out[k] = torch.as_tensor(arr, dtype=torch.float32)
        else:
            out[k] = torch.as_tensor(arr, dtype=torch.long)
    if "y" in items[0]:
        out["y"] = torch.as_tensor([it["y"] for it in items], dtype=torch.float32)
    return out

def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }


def build_transformer(
    *,
    content_mode: str,
    annotation_mode: str,
    merge_mode: str,
    chain_mode: str,
    rb: ResidueBundle,
    presets: dict,
) -> AnnotatedTransformer:
    ncfg = presets["neural"]
    if content_mode == "frozen":
        plm_h = rb.esm2_hidden if chain_mode == "H_ONLY" else rb.ablingua_hidden
        if plm_h <= 0:
            raise ValueError("PLM hidden dim not loaded in ResidueBundle")
    else:
        plm_h = 0
    return AnnotatedTransformer(
        content_mode=content_mode,
        plm_hidden=plm_h,
        n_aa=22,
        max_seq_pos=max(rb.heavy_aa.shape[1], rb.light_aa.shape[1]) + 2,
        n_imgt=max(rb.imgt_vocab.values()) + 1,
        n_region=9,
        annotation_mode=annotation_mode,
        merge_mode=merge_mode if chain_mode != "H_ONLY" else "h_only",
        chain_mode=chain_mode,
        d_model=ncfg["d_model"],
        n_heads=ncfg["n_heads"],
        n_layers=ncfg["n_layers"],
        dim_feedforward=ncfg["dim_feedforward"],
        dropout=ncfg["dropout"],
        norm_first=ncfg["norm_first"],
    )


def train_transformer_seed(
    *,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    y_map: dict[str, float],
    rb: ResidueBundle,
    content_mode: str,
    annotation_mode: str,
    merge_mode: str,
    chain_mode: str,
    seed: int,
    device: torch.device,
    quick: bool = False,
    fixed_parts: Optional[dict] = None,
) -> dict:
    presets = load_presets()
    ncfg = presets["neural"]
    max_epochs = 5 if quick else int(ncfg["max_epochs"])
    patience = 2 if quick else int(ncfg["early_stopping_patience"])
    bs = int(ncfg["batch_size"])

    y_tr = np.asarray([y_map[a] for a in train_ids], float)
    y_va = np.asarray([y_map[a] for a in val_ids], float)
    mu, sd = _y_stats(y_tr)
    plm_source = "esm2" if (content_mode == "frozen" and chain_mode == "H_ONLY") else "ablingua"

    Xtr = Xva = Xte = None
    fixed_dim = 0
    if fixed_parts is not None:
        Xtr, Xva, Xte = preprocess_parts(fixed_parts, train_ids, [val_ids, test_ids])
        fixed_dim = int(Xtr.shape[1])

    def make_loader(ids, y_arr, X_fixed, shuffle):
        ds = AbDataset(
            ids,
            y_arr,
            rb,
            content_mode=content_mode,
            plm_source=plm_source,
            fixed_X=X_fixed,
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_batch)

    torch.manual_seed(seed)
    np.random.seed(seed)
    core = build_transformer(
        content_mode=content_mode,
        annotation_mode=annotation_mode,
        merge_mode=merge_mode,
        chain_mode=chain_mode,
        rb=rb,
        presets=presets,
    )
    if fixed_parts is not None:
        model: nn.Module = FeatureFusionModel(
            core,
            fixed_dim,
            fixed_proj_dim=ncfg["fusion_fixed_dim"],
            head_hidden=ncfg["fusion_head_hidden"],
            dropout=ncfg["dropout"],
        )
    else:
        model = core
    model.to(device)
    n_params = model.n_trainable_parameters()
    if n_params > 2_000_000:
        print(f"WARNING: trainable params {n_params} > 2M", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"]
    )
    loss_fn = nn.SmoothL1Loss(beta=float(ncfg["smooth_l1_beta"]))

    def run_epoch(loader, y_mu, y_sd, train: bool, model_ref):
        model_ref.train(train)
        preds = []
        for batch in loader:
            batch = _batch_to_device(batch, device)
            y = batch.pop("y")
            fixed = batch.pop("fixed", None)
            y_std = (y - y_mu) / y_sd
            if train:
                opt.zero_grad(set_to_none=True)
            if fixed is None:
                out = model_ref(batch)
            else:
                out = model_ref(batch, fixed)
            loss = loss_fn(out, y_std)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model_ref.parameters(), ncfg["gradient_clip_norm"])
                opt.step()
            with torch.no_grad():
                preds.append((out * y_sd + y_mu).detach().cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])

    best_epoch = 1
    best_val = float("inf")
    stale = 0
    for epoch in range(1, max_epochs + 1):
        tr_loader = make_loader(train_ids, y_tr, Xtr, True)
        run_epoch(tr_loader, mu, sd, True, model)
        va_loader = make_loader(val_ids, y_va, Xva, False)
        va_pred = run_epoch(va_loader, mu, sd, False, model)
        v_mae = mae(y_va, va_pred)
        if v_mae + 1e-12 < best_val:
            best_val = v_mae
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # STEP B: reinit, train TRAIN+VAL for exactly best_epoch
    tv_ids = train_ids + val_ids
    y_tv = np.asarray([y_map[a] for a in tv_ids], float)
    mu2, sd2 = _y_stats(y_tv)
    Xtv = Xte2 = None
    if fixed_parts is not None:
        Xtv, Xte2 = preprocess_parts(fixed_parts, tv_ids, [test_ids])

    torch.manual_seed(seed)
    np.random.seed(seed)
    core2 = build_transformer(
        content_mode=content_mode,
        annotation_mode=annotation_mode,
        merge_mode=merge_mode,
        chain_mode=chain_mode,
        rb=rb,
        presets=presets,
    )
    if fixed_parts is not None:
        model2: nn.Module = FeatureFusionModel(
            core2,
            int(Xtv.shape[1]),
            fixed_proj_dim=ncfg["fusion_fixed_dim"],
            head_hidden=ncfg["fusion_head_hidden"],
            dropout=ncfg["dropout"],
        )
    else:
        model2 = core2
    model2.to(device)
    opt = torch.optim.AdamW(
        model2.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"]
    )
    for epoch in range(1, best_epoch + 1):
        tv_loader = make_loader(tv_ids, y_tv, Xtv, True)
        run_epoch(tv_loader, mu2, sd2, True, model2)

    y_te = np.asarray([y_map[a] for a in test_ids], float)
    te_loader = make_loader(test_ids, y_te, Xte2, False)
    te_pred = run_epoch(te_loader, mu2, sd2, False, model2)
    return {
        "best_epoch": best_epoch,
        "val_mae_phase_a": best_val,
        "test_pred": te_pred,
        "test_ids": test_ids,
        "test_mae": mae(y_te, te_pred),
        "n_trainable_parameters": n_params,
        "mu_tv": mu2,
        "sd_tv": sd2,
    }


def run_transformer_cv(
    *,
    target: str,
    content_mode: str,
    annotation_mode: str,
    merge_mode: str,
    chain_mode: str,
    variant_id: str,
    dev: pd.DataFrame,
    rb: ResidueBundle,
    folds: FoldMaps,
    device: str,
    out_dir: Path,
    quick: bool = False,
    recipe_id: Optional[str] = None,
) -> dict:
    """Primary+Shadow multi-seed ensemble OOF."""
    device_t = torch.device(device if device != "cuda" else "cuda:0")
    if device == "cpu":
        device_t = torch.device("cpu")
    presets = load_presets()
    seeds = [presets["neural"]["seeds"][0]] if quick else list(presets["neural"]["seeds"])
    y_map = {str(r.id): float(getattr(r, target)) for r in dev.itertuples(index=False)}
    dev_ids = dev["id"].astype(str).tolist()
    fixed_parts = None
    if recipe_id:
        fixed_parts = build_recipe_parts(recipe_id, rb.ids)

    cfg = {
        "target": target,
        "variant_id": variant_id,
        "content_mode": content_mode,
        "annotation_mode": annotation_mode,
        "merge_mode": merge_mode,
        "chain_mode": chain_mode,
        "recipe_id": recipe_id,
        "seeds": seeds,
        "quick": quick,
    }
    ch = config_hash(cfg)
    cache_path = out_dir / f"cache_{variant_id}_{ch}.json"
    oof_path = out_dir / f"oof_{variant_id}_{ch}.npz"

    results_scheme = {}
    best_epochs_primary = {s: [] for s in seeds}

    for scheme_name, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
        oof = pd.Series(0.0, index=dev_ids, dtype=float)
        seed_maes = []
        for seed in seeds:
            seed_oof = pd.Series(index=dev_ids, dtype=float)
            for k in range(5):
                cache_fold = out_dir / f"fold_{variant_id}_{scheme_name}_k{k}_s{seed}_{ch}.npz"
                if cache_fold.exists() and not quick:
                    z = np.load(cache_fold, allow_pickle=True)
                    pred = z["pred"]
                    tids = [str(x) for x in z["ids"].tolist()]
                    be = int(z["best_epoch"])
                else:
                    tr, va, te = tvt_split(fmap, k, dev_ids)
                    out = train_transformer_seed(
                        train_ids=tr,
                        val_ids=va,
                        test_ids=te,
                        y_map=y_map,
                        rb=rb,
                        content_mode=content_mode,
                        annotation_mode=annotation_mode,
                        merge_mode=merge_mode,
                        chain_mode=chain_mode,
                        seed=seed,
                        device=device_t,
                        quick=quick,
                        fixed_parts=fixed_parts,
                    )
                    pred = out["test_pred"]
                    tids = out["test_ids"]
                    be = out["best_epoch"]
                    n_params = out["n_trainable_parameters"]
                    np.savez_compressed(
                        cache_fold,
                        pred=pred,
                        ids=np.asarray(tids, dtype=object),
                        best_epoch=be,
                        n_params=out["n_trainable_parameters"],
                    )
                seed_oof.loc[tids] = pred
                if scheme_name == "primary":
                    best_epochs_primary[seed].append(be)
            sm = mae(
                np.asarray([y_map[a] for a in dev_ids], float),
                seed_oof.loc[dev_ids].to_numpy(float),
            )
            seed_maes.append(sm)
            oof = oof + seed_oof / len(seeds)
        results_scheme[scheme_name] = {
            "mae": mae(
                np.asarray([y_map[a] for a in dev_ids], float),
                oof.loc[dev_ids].to_numpy(float),
            ),
            "oof": oof,
            "seed_dispersion": seed_dispersion(seed_maes),
            "seed_maes": seed_maes,
        }

    # parameter count from one fold file
    n_params = 0
    for p in out_dir.glob(f"fold_{variant_id}_primary_k0_s{seeds[0]}_{ch}.npz"):
        n_params = int(np.load(p)["n_params"])
        break

    primary = results_scheme["primary"]["mae"]
    shadow = results_scheme["shadow"]["mae"]
    summary = {
        "target": target,
        "family": (
            "fusion_transformer"
            if recipe_id
            else ("scratch_transformer" if content_mode == "scratch" else "frozen_transformer")
        ),
        "variant_id": variant_id,
        "recipe_id": recipe_id or "",
        "annotation_mode": annotation_mode,
        "merge_mode": merge_mode,
        "chain_mode": chain_mode,
        "primary_mae": primary,
        "shadow_mae": shadow,
        "cv_mean_mae": cv_mean(primary, shadow),
        "cv_worst_mae": cv_worst(primary, shadow),
        "seed_dispersion_primary": results_scheme["primary"]["seed_dispersion"],
        "seed_dispersion_shadow": results_scheme["shadow"]["seed_dispersion"],
        "n_trainable_parameters": n_params,
        "config_hash": ch,
        "best_epochs_primary": {str(k): v for k, v in best_epochs_primary.items()},
        "quick": quick,
    }
    cache_path.write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        oof_path,
        primary_oof=results_scheme["primary"]["oof"].loc[dev_ids].to_numpy(float),
        shadow_oof=results_scheme["shadow"]["oof"].loc[dev_ids].to_numpy(float),
        ids=np.asarray(dev_ids, dtype=object),
    )
    return summary


def run_xgboost_cv(
    *,
    target: str,
    recipe_id: str,
    dev: pd.DataFrame,
    folds: FoldMaps,
    device: str,
    out_dir: Path,
) -> dict:
    presets = load_presets()
    preset_names = list(presets["xgboost"]["presets"].keys())
    y_map = {str(r.id): float(getattr(r, target)) for r in dev.itertuples(index=False)}
    dev_ids = dev["id"].astype(str).tolist()
    # Feature parts over all bundle ids (dev+test) for consistent columns; CV uses DEV only
    # Build parts on DEV ids only is fine
    parts = build_recipe_parts(recipe_id, dev_ids)
    cfg = {"target": target, "recipe_id": recipe_id, "presets": preset_names}
    ch = config_hash(cfg)

    results_scheme = {}
    primary_preset_votes = []
    primary_best_iters = []

    for scheme_name, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
        oof = pd.Series(index=dev_ids, dtype=float)
        chosen = []
        for k in range(5):
            tr, va, te = tvt_split(fmap, k, dev_ids)
            Xtr, Xva, Xte = preprocess_parts(parts, tr, [va, te])
            ytr = np.asarray([y_map[a] for a in tr], float)
            yva = np.asarray([y_map[a] for a in va], float)
            yte = np.asarray([y_map[a] for a in te], float)

            best_name, best_val, best_iter = None, float("inf"), 0
            for pname in preset_names:
                _, bit, vmae = fit_xgb_early(Xtr, ytr, Xva, yva, pname, device=device)
                if vmae < best_val - 1e-15 or (
                    abs(vmae - best_val) <= 1e-15 and (best_name is None or pname < best_name)
                ):
                    best_val = vmae
                    best_name = pname
                    best_iter = bit
            # refit TRAIN+VAL
            Xtv, Xte2 = preprocess_parts(parts, tr + va, [te])
            ytv = np.asarray([y_map[a] for a in tr + va], float)
            model = fit_xgb_rounds(
                Xtv, ytv, best_name, n_estimators=best_iter + 1, device=device
            )
            pred = model.predict(Xte2)
            oof.loc[te] = pred
            chosen.append({"preset": best_name, "best_iteration": best_iter, "val_mae": best_val})
            if scheme_name == "primary":
                primary_preset_votes.append(best_name)
                primary_best_iters.append(best_iter)
        results_scheme[scheme_name] = {
            "mae": mae(
                np.asarray([y_map[a] for a in dev_ids], float),
                oof.loc[dev_ids].to_numpy(float),
            ),
            "oof": oof,
            "rotations": chosen,
        }

    # Full-DEV policy: majority/lexicographic preset by Primary VAL; median best_iteration
    from collections import Counter

    cnt = Counter(primary_preset_votes)
    final_preset = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
    # median best_iteration among rotations that chose final_preset; else overall median
    iters = [
        primary_best_iters[i]
        for i, p in enumerate(primary_preset_votes)
        if p == final_preset
    ]
    if not iters:
        iters = primary_best_iters
    final_rounds = int(np.median(iters)) + 1

    primary = results_scheme["primary"]["mae"]
    shadow = results_scheme["shadow"]["mae"]
    lin = load_presets()["linear_baselines"][recipe_id]
    summary = {
        "target": target,
        "family": "xgboost",
        "variant_id": f"XGB__{recipe_id}",
        "recipe_id": recipe_id,
        "annotation_mode": "",
        "merge_mode": "",
        "chain_mode": "",
        "primary_mae": primary,
        "shadow_mae": shadow,
        "cv_mean_mae": cv_mean(primary, shadow),
        "cv_worst_mae": cv_worst(primary, shadow),
        "seed_dispersion_primary": 0.0,
        "seed_dispersion_shadow": 0.0,
        "n_trainable_parameters": 0,
        "config_hash": ch,
        "final_preset": final_preset,
        "final_n_estimators": final_rounds,
        "corresponding_linear_primary": lin["primary"],
        "corresponding_linear_shadow": lin["shadow"],
        "delta_primary_vs_linear": lin["primary"] - primary,
        "delta_shadow_vs_linear": lin["shadow"] - shadow,
        "primary_rotations": results_scheme["primary"]["rotations"],
    }
    (out_dir / f"xgb_{recipe_id}_{ch}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def full_dev_xgb_predict(
    *,
    target: str,
    recipe_id: str,
    final_preset: str,
    final_n_estimators: int,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    all_ids = (
        pd.concat([dev[["id"]], test[["id"]]], ignore_index=True)["id"].astype(str).tolist()
    )
    parts = build_recipe_parts(recipe_id, all_ids)
    Xdev, Xte = preprocess_parts(
        parts, dev["id"].astype(str).tolist(), [test["id"].astype(str).tolist()]
    )
    y = dev[target].to_numpy(float)
    model = fit_xgb_rounds(Xdev, y, final_preset, final_n_estimators, device=device)
    pred = model.predict(Xte)
    return pd.DataFrame({"id": test["id"].astype(str), "prediction": pred})


def full_dev_transformer_predict(
    *,
    target: str,
    content_mode: str,
    annotation_mode: str,
    merge_mode: str,
    chain_mode: str,
    best_epochs_primary: dict,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    rb: ResidueBundle,
    device: str,
    recipe_id: Optional[str] = None,
    quick: bool = False,
) -> pd.DataFrame:
    """Median Primary best_epoch per seed → train full DEV → average Test preds."""
    presets = load_presets()
    seeds = [presets["neural"]["seeds"][0]] if quick else list(presets["neural"]["seeds"])
    device_t = torch.device("cpu" if device == "cpu" else "cuda:0")
    ncfg = presets["neural"]
    plm_source = "esm2" if (content_mode == "frozen" and chain_mode == "H_ONLY") else "ablingua"
    y_map = {str(r.id): float(getattr(r, target)) for r in dev.itertuples(index=False)}
    # dummy y for test
    for a in test["id"].astype(str):
        y_map.setdefault(a, 0.0)
    dev_ids = dev["id"].astype(str).tolist()
    test_ids = test["id"].astype(str).tolist()
    y_dev = np.asarray([y_map[a] for a in dev_ids], float)
    mu, sd = _y_stats(y_dev)
    fixed_parts = build_recipe_parts(recipe_id, rb.ids) if recipe_id else None
    Xdev = Xte = None
    if fixed_parts is not None:
        Xdev, Xte = preprocess_parts(fixed_parts, dev_ids, [test_ids])

    preds = []
    for seed in seeds:
        epochs_list = best_epochs_primary.get(str(seed)) or best_epochs_primary.get(seed) or [30]
        n_epochs = int(np.median(epochs_list))
        if quick:
            n_epochs = min(n_epochs, 5)
        n_epochs = max(n_epochs, 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        core = build_transformer(
            content_mode=content_mode,
            annotation_mode=annotation_mode,
            merge_mode=merge_mode,
            chain_mode=chain_mode,
            rb=rb,
            presets=presets,
        )
        if fixed_parts is not None:
            model: nn.Module = FeatureFusionModel(
                core,
                int(Xdev.shape[1]),
                fixed_proj_dim=ncfg["fusion_fixed_dim"],
                head_hidden=ncfg["fusion_head_hidden"],
                dropout=ncfg["dropout"],
            )
        else:
            model = core
        model.to(device_t)
        opt = torch.optim.AdamW(
            model.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"]
        )
        loss_fn = nn.SmoothL1Loss(beta=float(ncfg["smooth_l1_beta"]))
        bs = int(ncfg["batch_size"])

        def make_loader(ids, y_arr, X_fixed, shuffle):
            ds = AbDataset(
                ids,
                y_arr,
                rb,
                content_mode=content_mode,
                plm_source=plm_source,
                fixed_X=X_fixed,
            )
            return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_batch)

        for _ in range(n_epochs):
            loader = make_loader(dev_ids, y_dev, Xdev, True)
            model.train(True)
            for batch in loader:
                batch = _batch_to_device(batch, device_t)
                y = batch.pop("y")
                fixed = batch.pop("fixed", None)
                y_std = (y - mu) / sd
                opt.zero_grad(set_to_none=True)
                out = model(batch) if fixed is None else model(batch, fixed)
                loss = loss_fn(out, y_std)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ncfg["gradient_clip_norm"])
                opt.step()

        te_y = np.zeros(len(test_ids), dtype=float)
        te_loader = make_loader(test_ids, te_y, Xte, False)
        model.eval()
        outs = []
        with torch.no_grad():
            for batch in te_loader:
                batch = _batch_to_device(batch, device_t)
                batch.pop("y", None)
                fixed = batch.pop("fixed", None)
                out = model(batch) if fixed is None else model(batch, fixed)
                outs.append((out * sd + mu).cpu().numpy())
        preds.append(np.concatenate(outs))
    ens = np.mean(np.stack(preds, axis=0), axis=0)
    return pd.DataFrame({"id": test_ids, "prediction": ens})
