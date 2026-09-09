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
from .metrics import cv_mean, cv_worst, mae, seed_dispersion
from .model import AnnotatedTransformer
from .fusion import FeatureFusionModel

# Fold-local fixed-feature prep remains in immutable bundle (behavior preservation).
import sys

_BUNDLE_ADV = str(BUNDLE_ROOT)
if _BUNDLE_ADV not in sys.path:
    sys.path.insert(0, _BUNDLE_ADV)
_DRILLDOWN = str(Path(__file__).resolve().parents[2])
if _DRILLDOWN not in sys.path:
    sys.path.insert(0, _DRILLDOWN)
from advanced_models.features import build_recipe_parts, preprocess_parts  # noqa: E402


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
        plm_source: str = "ablingua",  # ablingua | esm2 | ablang2
        fixed_X: Optional[np.ndarray] = None,
        use_continuous_rasa: bool = False,
        use_rasa_weighted_pool: bool = False,
        use_ca_distance_bias: bool = False,
    joint_hl_single_reg: bool = False,
    ):
        self.ids = ids
        self.y = y
        self.rb = rb
        self.content_mode = content_mode
        self.plm_source = plm_source
        self.fixed_X = fixed_X
        self.use_continuous_rasa = bool(use_continuous_rasa)
        self.use_rasa_weighted_pool = bool(use_rasa_weighted_pool)
        self.use_ca_distance_bias = bool(use_ca_distance_bias)
        self.need_rasa = self.use_continuous_rasa or self.use_rasa_weighted_pool
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
        if self.need_rasa:
            if self.rb.heavy_rasa is None or self.rb.light_rasa is None:
                raise RuntimeError("RASA requested but ResidueBundle lacks rasa arrays")
            item["heavy_rasa"] = self.rb.heavy_rasa[j]
            item["light_rasa"] = self.rb.light_rasa[j]
        if self.use_ca_distance_bias:
            if self.rb.heavy_ca is None or self.rb.light_ca is None:
                raise RuntimeError("CA distance bias requested but ResidueBundle lacks CA arrays")
            item["heavy_ca"] = self.rb.heavy_ca[j]
            item["light_ca"] = self.rb.light_ca[j]
        if self.content_mode == "frozen":
            if self.plm_source == "ablingua":
                item["heavy_plm"] = self.rb.ablingua_h[j]
                item["light_plm"] = self.rb.ablingua_l[j]
            elif self.plm_source == "ablang2":
                item["heavy_plm"] = self.rb.ablang2_h[j]
                item["light_plm"] = self.rb.ablang2_l[j]
            elif self.plm_source == "esm2":
                item["heavy_plm"] = self.rb.esm2_h[j]
            else:
                raise ValueError(self.plm_source)
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


def resolve_plm_source(content_mode: str, chain_mode: str, plm_source: Optional[str]) -> str:
    if content_mode != "frozen":
        return plm_source or "ablingua"
    if plm_source:
        return plm_source
    return "esm2" if chain_mode == "H_ONLY" else "ablingua"


def plm_hidden_for(rb: ResidueBundle, plm_source: str) -> int:
    if plm_source == "ablingua":
        return rb.ablingua_hidden
    if plm_source == "ablang2":
        return rb.ablang2_hidden
    if plm_source == "esm2":
        return rb.esm2_hidden
    raise ValueError(plm_source)


def build_transformer(
    *,
    content_mode: str,
    annotation_mode: str,
    merge_mode: str,
    chain_mode: str,
    rb: ResidueBundle,
    presets: dict,
    plm_source: str = "ablingua",
    pooling_mode: str = "reg",
    use_continuous_rasa: bool = False,
    use_rasa_weighted_pool: bool = False,
    use_ca_distance_bias: bool = False,
    joint_hl_single_reg: bool = False,
    initial_ell_angstrom: Optional[float] = None,
) -> AnnotatedTransformer:
    ncfg = presets["neural"]
    if content_mode == "frozen":
        plm_h = plm_hidden_for(rb, plm_source)
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
        pooling_mode=pooling_mode,
        d_model=ncfg["d_model"],
        n_heads=ncfg["n_heads"],
        n_layers=ncfg["n_layers"],
        dim_feedforward=ncfg["dim_feedforward"],
        dropout=ncfg["dropout"],
        norm_first=ncfg["norm_first"],
        use_continuous_rasa=use_continuous_rasa,
        use_rasa_weighted_pool=use_rasa_weighted_pool,
        use_ca_distance_bias=use_ca_distance_bias,
        joint_hl_single_reg=joint_hl_single_reg,
        initial_ell_angstrom=initial_ell_angstrom,
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
    plm_source: Optional[str] = None,
    pooling_mode: str = "reg",
    batch_size: Optional[int] = None,
    use_continuous_rasa: bool = False,
    use_rasa_weighted_pool: bool = False,
    use_ca_distance_bias: bool = False,
    joint_hl_single_reg: bool = False,
) -> dict:
    presets = load_presets()
    ncfg = presets["neural"]
    max_epochs = 5 if quick else int(ncfg["max_epochs"])
    patience = 2 if quick else int(ncfg["early_stopping_patience"])
    bs = int(batch_size or ncfg["batch_size"])
    plm_source = resolve_plm_source(content_mode, chain_mode, plm_source)

    y_tr = np.asarray([y_map[a] for a in train_ids], float)
    y_va = np.asarray([y_map[a] for a in val_ids], float)
    mu, sd = _y_stats(y_tr)

    initial_ell = None
    if use_ca_distance_bias:
        from classical_features.ca_cache import (
            train_median_pair_distance,
            train_median_pair_distance_joint_fv,
        )

        if rb.heavy_ca is None or rb.light_ca is None:
            raise RuntimeError("CA distance bias requires attached CA coords")
        train_idxs = [rb.id_to_idx[a] for a in train_ids]
        if joint_hl_single_reg:
            initial_ell = train_median_pair_distance_joint_fv(
                rb.heavy_ca, rb.light_ca, rb.heavy_mask, rb.light_mask, train_idxs
            )
        else:
            initial_ell = train_median_pair_distance(
                rb.heavy_ca, rb.light_ca, rb.heavy_mask, rb.light_mask, train_idxs
            )

    Xtr = Xva = Xte = None
    fixed_dim = 0
    if fixed_parts is not None:
        Xtr, Xva, Xte = preprocess_parts(fixed_parts, train_ids, [val_ids, test_ids])
        fixed_dim = int(Xtr.shape[1])

    def make_loader(ids, y_arr, X_fixed, shuffle, batch_sz):
        ds = AbDataset(
            ids,
            y_arr,
            rb,
            content_mode=content_mode,
            plm_source=plm_source,
            fixed_X=X_fixed,
            use_continuous_rasa=use_continuous_rasa,
            use_rasa_weighted_pool=use_rasa_weighted_pool,
            use_ca_distance_bias=use_ca_distance_bias,
            joint_hl_single_reg=joint_hl_single_reg,
        )
        return DataLoader(ds, batch_size=batch_sz, shuffle=shuffle, collate_fn=collate_batch)

    def build_model():
        core = build_transformer(
            content_mode=content_mode,
            annotation_mode=annotation_mode,
            merge_mode=merge_mode,
            chain_mode=chain_mode,
            rb=rb,
            presets=presets,
            plm_source=plm_source,
            pooling_mode=pooling_mode,
            use_continuous_rasa=use_continuous_rasa,
            use_rasa_weighted_pool=use_rasa_weighted_pool,
            use_ca_distance_bias=use_ca_distance_bias,
            joint_hl_single_reg=joint_hl_single_reg,
            initial_ell_angstrom=initial_ell,
        )
        if fixed_parts is not None:
            return FeatureFusionModel(
                core,
                fixed_dim,
                fixed_proj_dim=ncfg["fusion_fixed_dim"],
                head_hidden=ncfg["fusion_head_hidden"],
                dropout=ncfg["dropout"],
            )
        return core

    def run_epoch(loader, y_mu, y_sd, train: bool, model_ref, opt_ref):
        model_ref.train(train)
        preds = []
        for batch in loader:
            batch = _batch_to_device(batch, device)
            y = batch.pop("y")
            fixed = batch.pop("fixed", None)
            y_std = (y - y_mu) / y_sd
            if train:
                opt_ref.zero_grad(set_to_none=True)
            if fixed is None:
                out = model_ref(batch)
            else:
                out = model_ref(batch, fixed)
            loss = loss_fn(out, y_std)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model_ref.parameters(), ncfg["gradient_clip_norm"])
                opt_ref.step()
            with torch.no_grad():
                preds.append((out * y_sd + y_mu).detach().cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])

    oom_fallbacks = [bs]
    for cand in (8, 4):
        if cand < bs and cand not in oom_fallbacks:
            oom_fallbacks.append(cand)

    last_err: Optional[BaseException] = None
    for try_bs in oom_fallbacks:
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model()
            model.to(device)
            n_params = model.n_trainable_parameters()
            if n_params > 2_000_000:
                print(f"WARNING: trainable params {n_params} > 2M", flush=True)
            opt = torch.optim.AdamW(
                model.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"]
            )
            loss_fn = nn.SmoothL1Loss(beta=float(ncfg["smooth_l1_beta"]))

            best_epoch = 1
            best_val = float("inf")
            stale = 0
            for epoch in range(1, max_epochs + 1):
                tr_loader = make_loader(train_ids, y_tr, Xtr, True, try_bs)
                run_epoch(tr_loader, mu, sd, True, model, opt)
                va_loader = make_loader(val_ids, y_va, Xva, False, try_bs)
                va_pred = run_epoch(va_loader, mu, sd, False, model, opt)
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
                fixed_dim = int(Xtv.shape[1])

            torch.manual_seed(seed)
            np.random.seed(seed)
            model2 = build_model() if fixed_parts is None else None
            if fixed_parts is not None:
                core2 = build_transformer(
                    content_mode=content_mode,
                    annotation_mode=annotation_mode,
                    merge_mode=merge_mode,
                    chain_mode=chain_mode,
                    rb=rb,
                    presets=presets,
                    plm_source=plm_source,
                    pooling_mode=pooling_mode,
                    use_continuous_rasa=use_continuous_rasa,
                    use_rasa_weighted_pool=use_rasa_weighted_pool,
                    use_ca_distance_bias=use_ca_distance_bias,
                    joint_hl_single_reg=joint_hl_single_reg,
                    initial_ell_angstrom=initial_ell,
                )
                model2 = FeatureFusionModel(
                    core2,
                    int(Xtv.shape[1]),
                    fixed_proj_dim=ncfg["fusion_fixed_dim"],
                    head_hidden=ncfg["fusion_head_hidden"],
                    dropout=ncfg["dropout"],
                )
            else:
                model2 = build_model()
            model2.to(device)
            opt2 = torch.optim.AdamW(
                model2.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"]
            )
            for epoch in range(1, best_epoch + 1):
                tv_loader = make_loader(tv_ids, y_tv, Xtv, True, try_bs)
                run_epoch(tv_loader, mu2, sd2, True, model2, opt2)

            y_te = np.asarray([y_map[a] for a in test_ids], float)
            te_loader = make_loader(test_ids, y_te, Xte2, False, try_bs)
            te_pred = run_epoch(te_loader, mu2, sd2, False, model2, opt2)

            region_weights = None
            core_ref = model2.transformer if isinstance(model2, FeatureFusionModel) else model2
            if getattr(core_ref, "pooling_mode", "reg") == "region_gate":
                w = core_ref.region_gate_weights()
                region_weights = {
                    "H": {n: float(w["H"][i].detach().cpu()) for i, n in enumerate(["FR_ALL", "CDR1", "CDR2", "CDR3"])},
                    "L": {n: float(w["L"][i].detach().cpu()) for i, n in enumerate(["FR_ALL", "CDR1", "CDR2", "CDR3"])},
                }

            distance_params = None
            if use_ca_distance_bias and getattr(core_ref, "use_ca_distance_bias", False):
                enc = core_ref.encoder
                a = enc.a.detach().cpu().numpy()
                ell = enc.length_scales().detach().cpu().numpy()
                distance_params = {
                    "initial_ell_angstrom": float(initial_ell) if initial_ell is not None else None,
                    "a": a.tolist(),
                    "ell_angstrom": ell.tolist(),
                }

            return {
                "best_epoch": best_epoch,
                "val_mae_phase_a": best_val,
                "test_pred": te_pred,
                "test_ids": test_ids,
                "test_mae": mae(y_te, te_pred),
                "n_trainable_parameters": n_params,
                "mu_tv": mu2,
                "sd_tv": sd2,
                "batch_size_used": try_bs,
                "region_gate_weights": region_weights,
                "distance_params": distance_params,
            }
        except RuntimeError as e:
            last_err = e
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"OOM/runtime at batch_size={try_bs}: {e}; trying smaller", flush=True)
                continue
            raise
    raise RuntimeError(f"train_transformer_seed failed after OOM fallbacks: {last_err}")


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
    plm_source: Optional[str] = None,
    pooling_mode: str = "reg",
    use_continuous_rasa: bool = False,
    use_rasa_weighted_pool: bool = False,
    use_ca_distance_bias: bool = False,
    joint_hl_single_reg: bool = False,
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
    plm_source = resolve_plm_source(content_mode, chain_mode, plm_source)

    cfg = {
        "target": target,
        "variant_id": variant_id,
        "content_mode": content_mode,
        "annotation_mode": annotation_mode,
        "merge_mode": merge_mode,
        "chain_mode": chain_mode,
        "pooling_mode": pooling_mode,
        "plm_source": plm_source,
        "recipe_id": recipe_id,
        "seeds": seeds,
        "quick": quick,
        "use_continuous_rasa": bool(use_continuous_rasa),
        "use_rasa_weighted_pool": bool(use_rasa_weighted_pool),
        "use_ca_distance_bias": bool(use_ca_distance_bias),
        "joint_hl_single_reg": bool(joint_hl_single_reg),
    }
    ch = config_hash(cfg)
    cache_path = out_dir / f"cache_{variant_id}_{ch}.json"
    oof_path = out_dir / f"oof_{variant_id}_{ch}.npz"

    results_scheme = {}
    best_epochs_primary = {s: [] for s in seeds}
    region_gate_rows: list[dict] = []
    batch_sizes_used: list[int] = []
    distance_param_rows: list[dict] = []

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
                    if "distance_params" in z.files and z["distance_params"].item() is not None:
                        dp = z["distance_params"].item()
                        for hi, (ah, eh) in enumerate(zip(dp["a"], dp["ell_angstrom"])):
                            distance_param_rows.append(
                                {
                                    "cv_scheme": scheme_name,
                                    "fold": k,
                                    "seed": seed,
                                    "head": hi,
                                    "a_h": float(ah),
                                    "ell_h_angstrom": float(eh),
                                    "initial_ell_angstrom": dp.get("initial_ell_angstrom"),
                                    "best_epoch": be,
                                }
                            )
                    if "region_gate_weights" in z.files:
                        rg = z["region_gate_weights"].item()
                        if rg is not None:
                            region_gate_rows.append(
                                {
                                    "variant_id": variant_id,
                                    "scheme": scheme_name,
                                    "fold": k,
                                    "seed": seed,
                                    **{
                                        f"{chn}_{rn}": rg[chn][rn]
                                        for chn in ("H", "L")
                                        for rn in ("FR_ALL", "CDR1", "CDR2", "CDR3")
                                    },
                                }
                            )
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
                        plm_source=plm_source,
                        pooling_mode=pooling_mode,
                        use_continuous_rasa=use_continuous_rasa,
                        use_rasa_weighted_pool=use_rasa_weighted_pool,
                        use_ca_distance_bias=use_ca_distance_bias,
            joint_hl_single_reg=joint_hl_single_reg,
                    )
                    pred = out["test_pred"]
                    tids = out["test_ids"]
                    be = out["best_epoch"]
                    batch_sizes_used.append(int(out.get("batch_size_used", 16)))
                    rg = out.get("region_gate_weights")
                    dp = out.get("distance_params")
                    np.savez_compressed(
                        cache_fold,
                        pred=pred,
                        ids=np.asarray(tids, dtype=object),
                        best_epoch=be,
                        n_params=out["n_trainable_parameters"],
                        region_gate_weights=rg,
                        batch_size_used=out.get("batch_size_used", 16),
                        distance_params=dp,
                    )
                    if dp is not None:
                        for hi, (ah, eh) in enumerate(zip(dp["a"], dp["ell_angstrom"])):
                            distance_param_rows.append(
                                {
                                    "cv_scheme": scheme_name,
                                    "fold": k,
                                    "seed": seed,
                                    "head": hi,
                                    "a_h": float(ah),
                                    "ell_h_angstrom": float(eh),
                                    "initial_ell_angstrom": dp.get("initial_ell_angstrom"),
                                    "best_epoch": be,
                                }
                            )
                    if rg is not None:
                        region_gate_rows.append(
                            {
                                "variant_id": variant_id,
                                "scheme": scheme_name,
                                "fold": k,
                                "seed": seed,
                                **{
                                    f"{chn}_{rn}": rg[chn][rn]
                                    for chn in ("H", "L")
                                    for rn in ("FR_ALL", "CDR1", "CDR2", "CDR3")
                                },
                            }
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
        "pooling_mode": pooling_mode,
        "plm_source": plm_source,
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
        "batch_sizes_used": batch_sizes_used,
        "region_gate_rows": region_gate_rows,
        "distance_param_rows": distance_param_rows,
    }
    cache_path.write_text(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("region_gate_rows", "distance_param_rows")},
            indent=2,
        )
        + "\n"
    )
    # keep region rows separately to avoid huge JSON duplication in cache; still return them
    np.savez_compressed(
        oof_path,
        primary_oof=results_scheme["primary"]["oof"].loc[dev_ids].to_numpy(float),
        shadow_oof=results_scheme["shadow"]["oof"].loc[dev_ids].to_numpy(float),
        ids=np.asarray(dev_ids, dtype=object),
    )
    return summary


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
    plm_source: Optional[str] = None,
    pooling_mode: str = "reg",
    use_continuous_rasa: bool = False,
    use_rasa_weighted_pool: bool = False,
    use_ca_distance_bias: bool = False,
    joint_hl_single_reg: bool = False,
    checkpoint_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Median Primary best_epoch per seed → train full DEV → average Test preds."""
    presets = load_presets()
    seeds = [presets["neural"]["seeds"][0]] if quick else list(presets["neural"]["seeds"])
    device_t = torch.device("cpu" if device == "cpu" else "cuda:0")
    ncfg = presets["neural"]
    plm_source = resolve_plm_source(content_mode, chain_mode, plm_source)
    y_map = {str(r.id): float(getattr(r, target)) for r in dev.itertuples(index=False)}
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

    initial_ell = None
    if use_ca_distance_bias:
        from classical_features.ca_cache import (
            train_median_pair_distance,
            train_median_pair_distance_joint_fv,
        )

        if rb.heavy_ca is None or rb.light_ca is None:
            raise RuntimeError("CA distance bias requires attached CA coords")
        train_idxs = [rb.id_to_idx[a] for a in dev_ids]
        if joint_hl_single_reg:
            initial_ell = train_median_pair_distance_joint_fv(
                rb.heavy_ca, rb.light_ca, rb.heavy_mask, rb.light_mask, train_idxs
            )
        else:
            initial_ell = train_median_pair_distance(
                rb.heavy_ca, rb.light_ca, rb.heavy_mask, rb.light_mask, train_idxs
            )

    preds = []
    fulldev_distance_rows: list[dict] = []
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            plm_source=plm_source,
            pooling_mode=pooling_mode,
            use_continuous_rasa=use_continuous_rasa,
            use_rasa_weighted_pool=use_rasa_weighted_pool,
            use_ca_distance_bias=use_ca_distance_bias,
            joint_hl_single_reg=joint_hl_single_reg,
            initial_ell_angstrom=initial_ell,
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
                use_continuous_rasa=use_continuous_rasa,
                use_rasa_weighted_pool=use_rasa_weighted_pool,
                use_ca_distance_bias=use_ca_distance_bias,
            joint_hl_single_reg=joint_hl_single_reg,
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

        if use_ca_distance_bias:
            core_ref = model.transformer if isinstance(model, FeatureFusionModel) else model
            enc = core_ref.encoder
            a = enc.a.detach().cpu().numpy()
            ell = enc.length_scales().detach().cpu().numpy()
            for hi, (ah, eh) in enumerate(zip(a, ell)):
                fulldev_distance_rows.append(
                    {
                        "cv_scheme": "full_dev",
                        "fold": -1,
                        "seed": seed,
                        "head": hi,
                        "a_h": float(ah),
                        "ell_h_angstrom": float(eh),
                        "initial_ell_angstrom": float(initial_ell) if initial_ell is not None else None,
                        "best_epoch": n_epochs,
                    }
                )
            if checkpoint_dir is not None:
                ckpt = checkpoint_dir / f"fulldev_seed{seed}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "seed": seed,
                        "n_epochs": n_epochs,
                        "initial_ell_angstrom": initial_ell,
                        "mu": mu,
                        "sd": sd,
                    },
                    ckpt,
                )

    if checkpoint_dir is not None and fulldev_distance_rows:
        pd.DataFrame(fulldev_distance_rows).to_csv(
            checkpoint_dir / "distance_params_fulldev.csv", index=False
        )

    ens = np.mean(np.stack(preds, axis=0), axis=0)
    out_df = pd.DataFrame({"id": test_ids, "prediction": ens})
    out_df.attrs["distance_param_rows"] = fulldev_distance_rows
    out_df.attrs["initial_ell_angstrom"] = initial_ell
    return out_df
