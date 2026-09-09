#!/usr/bin/env python3
"""Old (bundle) vs new (drilldown) numerical equivalence helpers.

Unit tests only — not historical training reruns.
Future representation seed aggregation is UNDECIDED; do not assume seed-mean.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import BUNDLE_ROOT, REGION_TO_IDX, load_presets
from .fusion import FeatureFusionModel as NewFusion
from .model import AnnotatedTransformer as NewTransformer

# Immutable old implementation
_ADV = str(BUNDLE_ROOT)
if _ADV not in sys.path:
    sys.path.insert(0, _ADV)
from advanced_models.models.annotated_transformer import (  # noqa: E402
    AnnotatedTransformer as OldTransformer,
)
from advanced_models.models.feature_fusion import (  # noqa: E402
    FeatureFusionModel as OldFusion,
)


def _kw_for_variant(
    *,
    content_mode: str,
    annotation_mode: str,
    merge_mode: str,
    chain_mode: str,
    pooling_mode: str,
    plm_hidden: int,
) -> dict[str, Any]:
    ncfg = load_presets()["neural"]
    return dict(
        content_mode=content_mode,
        plm_hidden=plm_hidden if content_mode == "frozen" else 0,
        n_aa=22,
        max_seq_pos=160,
        n_imgt=64,
        n_region=len(REGION_TO_IDX),
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
    )


def make_pair(
    *,
    content_mode: str = "scratch",
    annotation_mode: str = "full",
    merge_mode: str = "concat",
    chain_mode: str = "HL",
    pooling_mode: str = "reg",
    plm_hidden: int = 1280,
    fusion: bool = False,
    fixed_dim: int = 32,
    seed: int = 0,
) -> tuple[nn.Module, nn.Module]:
    kw = _kw_for_variant(
        content_mode=content_mode,
        annotation_mode=annotation_mode,
        merge_mode=merge_mode,
        chain_mode=chain_mode,
        pooling_mode=pooling_mode,
        plm_hidden=plm_hidden,
    )
    torch.manual_seed(seed)
    old_core = OldTransformer(**kw)
    new_core = NewTransformer(**kw)
    new_core.load_state_dict(copy.deepcopy(old_core.state_dict()))
    ncfg = load_presets()["neural"]
    if fusion:
        torch.manual_seed(seed + 1)
        old = OldFusion(
            old_core,
            fixed_dim,
            fixed_proj_dim=ncfg["fusion_fixed_dim"],
            head_hidden=ncfg["fusion_head_hidden"],
            dropout=ncfg["dropout"],
        )
        new = NewFusion(
            new_core,
            fixed_dim,
            fixed_proj_dim=ncfg["fusion_fixed_dim"],
            head_hidden=ncfg["fusion_head_hidden"],
            dropout=ncfg["dropout"],
        )
        new.load_state_dict(copy.deepcopy(old.state_dict()))
        return old, new
    return old_core, new_core


def synthetic_batch(
    *,
    B: int = 4,
    Lh: int = 24,
    Ll: int = 20,
    content_mode: str = "scratch",
    chain_mode: str = "HL",
    annotation_mode: str = "full",
    plm_hidden: int = 1280,
    seed: int = 123,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    device = device or torch.device("cpu")
    g = torch.Generator().manual_seed(seed)
    heavy_mask = torch.ones(B, Lh, dtype=torch.bool)
    light_mask = torch.ones(B, Ll, dtype=torch.bool)
    heavy_mask[:, -3:] = False
    light_mask[:, -2:] = False
    batch: dict[str, torch.Tensor] = {
        "heavy_mask": heavy_mask,
        "light_mask": light_mask,
        "heavy_pos": torch.arange(1, Lh + 1).view(1, -1).expand(B, -1).clamp(max=160),
        "light_pos": torch.arange(1, Ll + 1).view(1, -1).expand(B, -1).clamp(max=160),
    }
    if content_mode == "scratch":
        batch["heavy_aa"] = torch.randint(1, 21, (B, Lh), generator=g)
        batch["light_aa"] = torch.randint(1, 21, (B, Ll), generator=g)
    else:
        batch["heavy_plm"] = torch.randn(B, Lh, plm_hidden, generator=g)
        batch["light_plm"] = torch.randn(B, Ll, plm_hidden, generator=g)
    if annotation_mode == "full":
        batch["heavy_imgt"] = torch.randint(1, 40, (B, Lh), generator=g)
        batch["light_imgt"] = torch.randint(1, 40, (B, Ll), generator=g)
        # region indices 1..7 (non-PAD)
        batch["heavy_region"] = torch.randint(1, 8, (B, Lh), generator=g)
        batch["light_region"] = torch.randint(1, 8, (B, Ll), generator=g)
    if chain_mode == "H_ONLY":
        # keep light tensors for shape compatibility; model should ignore
        pass
    return {k: v.to(device) for k, v in batch.items()}


def compare_forward(
    old: nn.Module,
    new: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    fixed: torch.Tensor | None = None,
) -> dict[str, float]:
    old.eval()
    new.eval()
    with torch.no_grad():
        if fixed is None:
            old_core = old
            new_core = new
            r0 = old_core.forward_repr(batch)
            r1 = new_core.forward_repr(batch)
            p0 = old_core(batch)
            p1 = new_core(batch)
        else:
            r0 = old.transformer.forward_repr(batch)
            r1 = new.transformer.forward_repr(batch)
            p0 = old(batch, fixed)
            p1 = new(batch, fixed)
    return {
        "repr_max_abs": float((r0 - r1).abs().max().item()),
        "pred_max_abs": float((p0 - p1).abs().max().item()),
    }


def compare_one_step(
    old: nn.Module,
    new: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    fixed: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    seed: int = 7,
) -> dict[str, float]:
    """Same RNG / state / batch → one AdamW + SmoothL1 step; compare loss/grad/params."""
    ncfg = load_presets()["neural"]
    if y is None:
        y = torch.randn(batch["heavy_mask"].shape[0])

    def _clone_model(m: nn.Module) -> nn.Module:
        m2 = copy.deepcopy(m)
        return m2

    old_m = _clone_model(old)
    new_m = _clone_model(new)
    # ensure identical initial state
    new_m.load_state_dict(copy.deepcopy(old_m.state_dict()))

    torch.manual_seed(seed)
    opt_o = torch.optim.AdamW(old_m.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"])
    torch.manual_seed(seed)
    opt_n = torch.optim.AdamW(new_m.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"])
    loss_fn = nn.SmoothL1Loss(beta=float(ncfg["smooth_l1_beta"]))

    old_m.train()
    new_m.train()
    opt_o.zero_grad(set_to_none=True)
    opt_n.zero_grad(set_to_none=True)

    # Identical dropout masks: reset RNG before each forward
    torch.manual_seed(seed + 100)
    if fixed is None:
        po = old_m(batch)
    else:
        po = old_m(batch, fixed)
    torch.manual_seed(seed + 100)
    if fixed is None:
        pn = new_m(batch)
    else:
        pn = new_m(batch, fixed)

    pred_delta = float((po - pn).abs().max().item())
    lo = loss_fn(po, y)
    ln = loss_fn(pn, y)
    lo.backward()
    ln.backward()

    grad_deltas = []
    for (n0, p0), (n1, p1) in zip(old_m.named_parameters(), new_m.named_parameters()):
        assert n0 == n1
        g0 = p0.grad if p0.grad is not None else torch.zeros_like(p0)
        g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
        grad_deltas.append(float((g0 - g1).abs().max().item()))

    nn.utils.clip_grad_norm_(old_m.parameters(), ncfg["gradient_clip_norm"])
    nn.utils.clip_grad_norm_(new_m.parameters(), ncfg["gradient_clip_norm"])
    opt_o.step()
    opt_n.step()

    param_deltas = []
    for (n0, p0), (n1, p1) in zip(old_m.named_parameters(), new_m.named_parameters()):
        param_deltas.append(float((p0.detach() - p1.detach()).abs().max().item()))

    return {
        "initial_pred_max_abs": pred_delta,
        "loss_abs_delta": float(abs(lo.item() - ln.item())),
        "grad_max_abs": max(grad_deltas) if grad_deltas else 0.0,
        "param_max_abs": max(param_deltas) if param_deltas else 0.0,
    }
