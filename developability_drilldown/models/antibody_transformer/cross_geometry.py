#!/usr/bin/env python3
"""H–L Cα distance RBF bias for residue cross-attention (ARCH-6G).

HIC experiment-code allocation for the T105+/HIC batch that consumes this
geometry path (do not reuse; H048–H053 already occupied):

    EXP-H054 .. EXP-H081  (28 codes)
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

# Exact RBF centers (Angstrom) for ARCH-6G cross-geometry bias.
DEFAULT_RBF_CENTERS: list[float] = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
DEFAULT_RBF_SIGMA: float = 2.5
N_RBF = len(DEFAULT_RBF_CENTERS)


def build_hl_distance(
    ca_h: torch.Tensor,
    ca_l: torch.Tensor,
    mask_h: torch.Tensor,
    mask_l: torch.Tensor,
) -> torch.Tensor:
    """Pairwise H–L Cα Euclidean distances.

    Args:
        ca_h: [B, Lh, 3] Cα coords (Å); missing → NaN
        ca_l: [B, Ll, 3]
        mask_h: [B, Lh] True = real residue
        mask_l: [B, Ll]

    Returns:
        d: [B, Lh, Ll] distances; NaN for invalid pairs (pad or missing CA).
    """
    # [B, Lh, 1, 3] - [B, 1, Ll, 3] → [B, Lh, Ll, 3]
    diff = ca_h.unsqueeze(2) - ca_l.unsqueeze(1)
    d = torch.linalg.vector_norm(diff, dim=-1)
    ca_h_ok = torch.isfinite(ca_h).all(dim=-1) & mask_h
    ca_l_ok = torch.isfinite(ca_l).all(dim=-1) & mask_l
    pair_ok = ca_h_ok.unsqueeze(2) & ca_l_ok.unsqueeze(1)
    return torch.where(pair_ok, d, torch.full_like(d, float("nan")))


def rbf_bias(
    d: torch.Tensor,
    weight: torch.Tensor,
    centers: Optional[Union[Sequence[float], torch.Tensor]] = None,
    sigma: float = DEFAULT_RBF_SIGMA,
) -> torch.Tensor:
    """Per-head RBF distance bias for H→L cross-attention.

    Args:
        d: [B, Lh, Ll] distances; NaN marks invalid pairs
        weight: [n_heads, 8] (or [n_heads, n_bases]) learnable RBF weights
        centers: length-8 centers in Å (default DEFAULT_RBF_CENTERS)
        sigma: RBF width (default 2.5)

    Returns:
        bias: [B, n_heads, Lh, Ll]; invalid pairs are exactly 0.
    """
    if centers is None:
        centers_t = torch.tensor(
            DEFAULT_RBF_CENTERS, device=d.device, dtype=d.dtype
        )
    elif isinstance(centers, torch.Tensor):
        centers_t = centers.to(device=d.device, dtype=d.dtype)
    else:
        centers_t = torch.tensor(list(centers), device=d.device, dtype=d.dtype)

    valid = torch.isfinite(d)
    d_safe = torch.where(valid, d, torch.zeros_like(d))
    # [B, Lh, Ll, n_bases]
    phi = torch.exp(
        -((d_safe.unsqueeze(-1) - centers_t.view(1, 1, 1, -1)) ** 2)
        / (2.0 * float(sigma) ** 2)
    )
    # weight: [H, K] → bias [B, H, Lh, Ll]
    # einsum: bijk,hk -> bhij
    bias = torch.einsum("bijk,hk->bhij", phi, weight.to(dtype=d.dtype))
    return bias * valid.unsqueeze(1).to(dtype=bias.dtype)
