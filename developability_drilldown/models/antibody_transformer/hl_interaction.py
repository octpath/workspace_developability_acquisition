#!/usr/bin/env python3
"""Shared H/L interaction modules for TmApp C/D architecture exploration (T142–T150).

All modules are directionally symmetric: same parameters for H←L and L←H (C)
or shared projections for pair features (D).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REG_CROSS_VARIANTS = (
    "c1_scalar_gate",
    "c2_feature_gate",
    "c3_ffn_adapter",
    "c4_two_read",
    "c5_two_query",
)

PAIR_INTERACTION_MODES = (
    "d1_bilinear_score",
    "d2_hadamard_residual",
    "d3_symmetric_mlp",
    "d4_token_attention",
)

RANK = 16


class SharedScalarCrossGate(nn.Module):
    """C1: g = 2*sigmoid(MLP(concat(q,d,q*d))); zero-init → g=1."""

    def __init__(self, d_model: int, bottleneck: int = RANK):
        super().__init__()
        self.force_identity = False  # ablation: g=1
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_d = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, reg: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # reg, delta: [B,1,D] or [B,D]
        squeeze = reg.dim() == 3
        q = self.ln_q(reg)
        d = self.ln_d(delta)
        f = torch.cat([q, d, q * d], dim=-1)
        a = self.mlp(f)
        g = 2.0 * torch.sigmoid(a)
        if self.force_identity:
            g = torch.ones_like(g)
        out = reg + g * delta
        return out, g.squeeze(-1) if squeeze else g.squeeze(-1)


class SharedFeatureCrossGate(nn.Module):
    """C2: feature-wise g ∈ R^d; zero-init → g=1."""

    def __init__(self, d_model: int, bottleneck: int = RANK):
        super().__init__()
        self.force_identity = False  # ablation: g=1
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_d = nn.LayerNorm(d_model)
        self.proj = nn.Linear(2 * d_model, bottleneck)
        self.out = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, reg: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.ln_q(reg)
        d = self.ln_d(delta)
        h = F.gelu(self.proj(torch.cat([q, d], dim=-1)))
        g = 2.0 * torch.sigmoid(self.out(h))
        if self.force_identity:
            g = torch.ones_like(g)
        return reg + g * delta, g


class SharedRegFFNAdapter(nn.Module):
    """C3: REG' = (REG+delta) + FFN; zero-init final → identity."""

    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.force_identity = False  # ablation: zero adapter
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if self.force_identity:
            return r
        a = F.gelu(self.fc1(self.ln(r)))
        a = self.fc2(a)
        return r + a


class SharedSecondReadAlpha(nn.Module):
    """C4: alpha = tanh(MLP(concat(LN(REG1), LN(delta2)))); zero-init → 0."""

    def __init__(self, d_model: int, bottleneck: int = RANK):
        super().__init__()
        self.force_identity = False  # ablation: alpha=0
        self.ln_r = nn.LayerNorm(d_model)
        self.ln_d = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, reg1: torch.Tensor, delta2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.mlp(torch.cat([self.ln_r(reg1), self.ln_d(delta2)], dim=-1))
        alpha = torch.tanh(a)
        if self.force_identity:
            alpha = torch.zeros_like(alpha)
        return reg1 + alpha * delta2, alpha.squeeze(-1)


class SharedTwoQueryCombiner(nn.Module):
    """C5: combine two cross deltas with shared REG-conditioned scores."""

    def __init__(self, d_model: int, bottleneck: int = RANK):
        super().__init__()
        self.force_identity = False  # ablation: single-query (slot0 only, weight=[1,0])
        self.slot = nn.Parameter(torch.zeros(2, d_model))
        # tiny symmetric perturbation
        nn.init.normal_(self.slot, mean=0.0, std=1e-3)
        self.ln_reg = nn.LayerNorm(d_model)
        self.ln_delta = nn.LayerNorm(d_model)
        self.w_reg = nn.Linear(d_model, bottleneck, bias=False)
        self.w_delta = nn.Linear(d_model, bottleneck, bias=False)
        self.v = nn.Linear(bottleneck, 1, bias=False)
        nn.init.zeros_(self.v.weight)

    def queries(self, reg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # reg: [B,1,D]
        if self.force_identity:
            return reg, reg
        q1 = reg + self.slot[0].view(1, 1, -1)
        q2 = reg + self.slot[1].view(1, 1, -1)
        return q1, q2

    def combine(
        self, reg: torch.Tensor, delta1: torch.Tensor, delta2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.force_identity:
            # collapse to single-query C0-equivalent: use delta1 only
            w = torch.zeros(reg.shape[0], 2, device=reg.device, dtype=reg.dtype)
            w[:, 0] = 1.0
            return reg + delta1, w
        r = self.ln_reg(reg)
        d1 = self.ln_delta(delta1)
        d2 = self.ln_delta(delta2)
        s1 = self.v(torch.tanh(self.w_reg(r) + self.w_delta(d1)))
        s2 = self.v(torch.tanh(self.w_reg(r) + self.w_delta(d2)))
        w = torch.softmax(torch.cat([s1, s2], dim=-1), dim=-1)  # [B,1,2]
        delta = w[..., 0:1] * delta1 + w[..., 1:2] * delta2
        return reg + delta, w.squeeze(1)


class PairBilinearScore(nn.Module):
    """D1: y = head(m) + w^T (U z_H ⊙ U z_L); w zero-init."""

    def __init__(self, d_model: int, rank: int = RANK):
        super().__init__()
        self.force_identity = False  # ablation: pair score off
        self.ln = nn.LayerNorm(d_model)
        self.U = nn.Linear(d_model, rank, bias=False)
        self.w_pair = nn.Parameter(torch.zeros(rank))

    def pair_score(self, z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        if self.force_identity:
            return torch.zeros(z_h.shape[0], device=z_h.device, dtype=z_h.dtype)
        u_h = self.U(self.ln(z_h))
        u_l = self.U(self.ln(z_l))
        p = u_h * u_l
        return (p * self.w_pair).sum(dim=-1)

    def mean_repr(self, z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        return 0.5 * (z_h + z_l)


class PairHadamardResidual(nn.Module):
    """D2: z = m + W(U z_H ⊙ U z_L); W zero-init."""

    def __init__(self, d_model: int, rank: int = RANK):
        super().__init__()
        self.force_identity = False  # ablation: pair residual off
        self.ln = nn.LayerNorm(d_model)
        self.U = nn.Linear(d_model, rank, bias=False)
        self.W = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.W.weight)

    def forward(self, z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        m = 0.5 * (z_h + z_l)
        if self.force_identity:
            return m
        p = self.U(self.ln(z_h)) * self.U(self.ln(z_l))
        return m + self.W(p)


class PairSymmetricMLP(nn.Module):
    """D3: swap-invariant pair MLP residual; final Linear zero-init."""

    def __init__(self, d_model: int, rank: int = RANK):
        super().__init__()
        self.force_identity = False  # ablation: pair residual off
        self.ln = nn.LayerNorm(d_model)
        self.U = nn.Linear(d_model, rank, bias=False)
        self.fc1 = nn.Linear(3 * rank, rank)
        self.fc2 = nn.Linear(rank, d_model)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        m = 0.5 * (z_h + z_l)
        if self.force_identity:
            return m
        u_h = self.U(self.ln(z_h))
        u_l = self.U(self.ln(z_l))
        s = u_h + u_l
        d = (u_h - u_l).abs()
        p = u_h * u_l
        h = F.gelu(self.fc1(torch.cat([s, d, p], dim=-1)))
        delta = self.fc2(h)
        return m + delta


class PairTokenAttention(nn.Module):
    """D4: one shared MHA over [z_H, z_L]; residual + LN; mean merge."""

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.force_identity = False  # ablation: skip attention, mean only
        if d_model % int(n_heads) != 0:
            n_heads = 1
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=int(n_heads), dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        # near-identity: zero out output projection of MHA
        if hasattr(self.mha, "out_proj"):
            nn.init.zeros_(self.mha.out_proj.weight)
            nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([z_h, z_l], dim=1)  # [B,2,D]
        if self.force_identity:
            # pair-off: keep LN residual path without attention (matches zero out_proj init)
            out = self.norm(tokens)
            return 0.5 * (out[:, 0] + out[:, 1])
        attn_out, _ = self.mha(tokens, tokens, tokens, need_weights=False)
        out = self.norm(tokens + attn_out)
        return 0.5 * (out[:, 0] + out[:, 1])
