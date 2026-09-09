#!/usr/bin/env python3
"""Fusion of Transformer representation with fixed-length Top-3 features."""
from __future__ import annotations

import torch
import torch.nn as nn

from .annotated_transformer import AnnotatedTransformer


class FeatureFusionModel(nn.Module):
    def __init__(
        self,
        transformer: AnnotatedTransformer,
        fixed_dim: int,
        *,
        fixed_proj_dim: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.transformer = transformer
        self.fixed_mlp = nn.Sequential(
            nn.Linear(fixed_dim, fixed_proj_dim),
            nn.LayerNorm(fixed_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        in_dim = transformer.repr_dim + fixed_proj_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: dict, fixed: torch.Tensor) -> torch.Tensor:
        tr = self.transformer.forward_repr(batch)
        fr = self.fixed_mlp(fixed)
        return self.head(torch.cat([tr, fr], dim=-1)).squeeze(-1)
