#!/usr/bin/env python3
"""Distance-biased TransformerEncoder (shared per-head Cα exponential bias)."""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inverse(y: float) -> float:
    y = max(float(y), 1e-6)
    return float(math.log(math.expm1(y)))


class SharedDistanceBiasTransformerEncoder(nn.Module):
    """Stack of TransformerEncoderLayer with shared per-head distance bias.

    Bias (per head h): a_h * exp(-d_ij / ell_h), added via float attn_mask.
    Parameters {a_h, ell_h} are shared across all layers.
    Sequence layout: [REG, residue_0, ..., residue_{L-1}].
    """

    def __init__(
        self,
        encoder_layer: nn.TransformerEncoderLayer,
        num_layers: int,
        n_heads: int,
        *,
        tiny: float = 1e-4,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.tiny = float(tiny)
        self.a = nn.Parameter(torch.zeros(n_heads))
        self.raw_ell = nn.Parameter(torch.zeros(n_heads))
        self._initial_ell: Optional[float] = None

    def length_scales(self) -> torch.Tensor:
        return F.softplus(self.raw_ell) + self.tiny

    def set_initial_ell(self, ell_angstrom: float) -> None:
        target = max(float(ell_angstrom) - self.tiny, 1e-3)
        inv = softplus_inverse(target)
        with torch.no_grad():
            self.raw_ell.fill_(inv)
        self._initial_ell = float(ell_angstrom)

    def build_attn_bias(
        self,
        residue_coords: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build additive attn bias [B*H, 1+L, 1+L].

        residue_coords: [B, L, 3]
        residue_mask: [B, L] True = real residue
        REG (index 0) and padded residues get zero bias.
        Diagonal (i==j) among residues also zero.
        """
        B, L, _ = residue_coords.shape
        H = self.n_heads
        device = residue_coords.device
        dtype = residue_coords.dtype

        # Pairwise Euclidean distances among residues
        diff = residue_coords.unsqueeze(2) - residue_coords.unsqueeze(1)  # [B,L,L,3]
        d = torch.linalg.vector_norm(diff, dim=-1)  # [B,L,L]
        pair_ok = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)  # [B,L,L]
        d = torch.where(pair_ok, d, torch.zeros_like(d))

        ell = self.length_scales().to(dtype=dtype)  # [H]
        a = self.a.to(dtype=dtype)
        # [B, H, L, L]
        bias_res = a.view(1, H, 1, 1) * torch.exp(
            -d.unsqueeze(1) / ell.view(1, H, 1, 1)
        )
        bias_res = bias_res * pair_ok.unsqueeze(1).to(dtype=dtype)
        # zero self pairs
        eye = torch.eye(L, device=device, dtype=torch.bool)
        bias_res = bias_res.masked_fill(eye.view(1, 1, L, L), 0.0)

        # Pad REG at front: zeros for any pair involving REG
        full = torch.zeros(B, H, 1 + L, 1 + L, device=device, dtype=dtype)
        full[:, :, 1:, 1:] = bias_res
        return full.reshape(B * H, 1 + L, 1 + L)

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        residue_coords: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if residue_coords is None or residue_mask is None:
            raise ValueError("distance-biased encoder requires residue_coords and residue_mask")
        attn_bias = self.build_attn_bias(residue_coords, residue_mask)
        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=attn_bias,
                src_key_padding_mask=src_key_padding_mask,
            )
        return output
