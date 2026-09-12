#!/usr/bin/env python3
"""Global antibody-level F1_SURFACE conditioning of Transformer z_seq (H134–H139).

Modes (fixed bottleneck=16; no AUX32; no late F1 concat):
  global_surface_film
  global_surface_gated_residual
  global_surface_token_attention
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import AnnotatedTransformer

BOTTLENECK = 16
ATTN_HEADS = 2
ATTN_LAYERS = 1

FUSION_MODES = (
    "global_surface_film",
    "global_surface_gated_residual",
    "global_surface_token_attention",
)


class GlobalSurfaceConditioningModel(nn.Module):
    """Wrap backbone.forward_repr; condition with F1_SURFACE35; use backbone.head."""

    def __init__(
        self,
        transformer: AnnotatedTransformer,
        aux_dim: int,
        fusion_mode: str,
        *,
        bottleneck: int = BOTTLENECK,
        n_heads: int = ATTN_HEADS,
        dropout: float = 0.2,
    ):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion_mode: {fusion_mode}")
        if int(aux_dim) != 35:
            raise ValueError(f"H134–H139 require F1_SURFACE35, got aux_dim={aux_dim}")
        self.transformer = transformer
        self.fusion_mode = fusion_mode
        self.aux_dim = int(aux_dim)
        self.bottleneck = int(bottleneck)
        d = int(transformer.repr_dim)
        self.repr_dim = d
        r = self.bottleneck

        self.surf_to_q = None
        self.film_gamma = None
        self.film_beta = None
        self.residual_u = None
        self.gate_z = None
        self.gate_x = None
        self.gate_out = None
        self.surf_token = None
        self.token_mha = None
        self.token_norm = None
        self.token_attn = None

        if fusion_mode == "global_surface_film":
            self.surf_to_q = nn.Sequential(nn.Linear(self.aux_dim, r), nn.GELU())
            self.film_gamma = nn.Linear(r, d)
            self.film_beta = nn.Linear(r, d)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
        elif fusion_mode == "global_surface_gated_residual":
            self.surf_to_q = nn.Sequential(nn.Linear(self.aux_dim, r), nn.GELU())
            self.residual_u = nn.Linear(r, d)
            self.gate_z = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, r))
            self.gate_x = nn.Linear(self.aux_dim, r)
            self.gate_out = nn.Linear(r, 1)
            nn.init.zeros_(self.residual_u.weight)
            nn.init.zeros_(self.residual_u.bias)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)
        else:
            self.surf_token = nn.Linear(self.aux_dim, d)
            self.token_mha = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=int(n_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            self.token_norm = nn.LayerNorm(d)
            # Keep attribute name for diagnostics/tests that expect a single interaction layer.
            self.token_attn = self.token_mha


    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_added_conditioning_parameters(self) -> int:
        backbone = {id(p) for p in self.transformer.parameters()}
        return sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in backbone)

    def condition(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        if self.fusion_mode == "global_surface_film":
            q = self.surf_to_q(x)
            gamma = self.film_gamma(q)
            beta = self.film_beta(q)
            return (1.0 + gamma) * z + beta
        if self.fusion_mode == "global_surface_gated_residual":
            q = self.surf_to_q(x)
            u = self.residual_u(q)
            interaction = self.gate_z(z) * self.gate_x(x)
            g = torch.sigmoid(self.gate_out(interaction))
            return z + g * u
        t_surf = self.surf_token(x)
        tokens = torch.stack([z, t_surf], dim=1)
        attn_out, _ = self.token_mha(tokens, tokens, tokens, need_weights=False)
        out = self.token_norm(tokens + attn_out)
        return out[:, 0]

    def forward_repr(self, batch: dict, fixed: torch.Tensor) -> torch.Tensor:
        z = self.transformer.forward_repr(batch)
        return self.condition(z, fixed)

    def forward(self, batch: dict, fixed: torch.Tensor) -> torch.Tensor:
        return self.transformer.head(self.forward_repr(batch, fixed)).squeeze(-1)

    @torch.no_grad()
    def film_stats(self, z: torch.Tensor, x: torch.Tensor) -> dict[str, float]:
        x = torch.nan_to_num(x, nan=0.0)
        q = self.surf_to_q(x)
        gamma = self.film_gamma(q)
        beta = self.film_beta(q)
        zp = (1.0 + gamma) * z + beta
        cos = torch.nn.functional.cosine_similarity(z, zp, dim=-1)
        rel = (zp - z).norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-8)
        return {
            "gamma_mean": float(gamma.mean()),
            "gamma_sd": float(gamma.std()),
            "gamma_abs_median": float(gamma.abs().median()),
            "beta_norm_mean": float(beta.norm(dim=-1).mean()),
            "cosine_z_zp_mean": float(cos.mean()),
            "rel_l2_mean": float(rel.mean()),
        }

    @torch.no_grad()
    def gate_values(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        interaction = self.gate_z(z) * self.gate_x(x)
        return torch.sigmoid(self.gate_out(interaction)).squeeze(-1)

    @torch.no_grad()
    def token_attention_weights(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        t_surf = self.surf_token(x)
        tokens = torch.stack([z, t_surf], dim=1)
        _, weights = self.token_mha(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        return weights
