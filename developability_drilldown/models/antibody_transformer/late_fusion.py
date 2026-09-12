#!/usr/bin/env python3
"""Late fusion of antibody latent z_DL with a small auxiliary-feature MLP.

Spec (T124/H082 batch):
    x_aux (already TRAIN-standardized / PCA'd)
      -> Linear(p, 64) -> GELU -> Dropout(0.2) -> Linear(64, 32) -> GELU
      -> z_aux ∈ R^32
    z = concat(z_DL, z_aux)
    -> same head family as backbone (Linear(d,d)->GELU->Dropout->Linear(d,1)
       with d = z_DL.dim + 32, hidden = backbone.d_model when available)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import AnnotatedTransformer


class LateFusionAuxMLP(nn.Module):
    def __init__(self, in_dim: int, *, hidden: int = 64, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(out_dim)),
            nn.GELU(),
        )
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionModel(nn.Module):
    """Wrap backbone.forward_repr with aux MLP + regression head."""

    fusion_mode = "late_concat_aux32"

    def __init__(
        self,
        transformer: AnnotatedTransformer,
        aux_dim: int,
        *,
        aux_hidden: int = 64,
        aux_out: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.transformer = transformer
        self.aux_mlp = LateFusionAuxMLP(
            aux_dim, hidden=aux_hidden, out_dim=aux_out, dropout=dropout
        )
        head_hidden = int(getattr(transformer, "d_model", transformer.repr_dim))
        in_dim = int(transformer.repr_dim) + int(aux_out)
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.repr_dim = in_dim
        self.aux_dim = int(aux_dim)

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_repr(self, batch: dict, fixed: torch.Tensor) -> torch.Tensor:
        z_dl = self.transformer.forward_repr(batch)
        z_aux = self.aux_mlp(fixed)
        return torch.cat([z_dl, z_aux], dim=-1)

    def forward(self, batch: dict, fixed: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_repr(batch, fixed)).squeeze(-1)


class DirectLateFusionModel(nn.Module):
    """Late fusion with genuine direct concat of standardized aux features.

    No Linear/LayerNorm/GELU/Dropout on the auxiliary branch before concat.
    Head: Linear(repr_dim + p, d_model) -> GELU -> Dropout -> Linear(d_model, 1).
    """

    fusion_mode = "late_concat_direct"

    def __init__(
        self,
        transformer: AnnotatedTransformer,
        aux_dim: int,
        *,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.transformer = transformer
        self.aux_mlp = None  # explicit: no aux encoder
        head_hidden = int(getattr(transformer, "d_model", transformer.repr_dim))
        in_dim = int(transformer.repr_dim) + int(aux_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.repr_dim = in_dim
        self.aux_dim = int(aux_dim)

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_repr(self, batch: dict, fixed: torch.Tensor) -> torch.Tensor:
        z_dl = self.transformer.forward_repr(batch)
        # Direct: pass TRAIN-standardized features unchanged into concat.
        return torch.cat([z_dl, fixed], dim=-1)

    def forward(self, batch: dict, fixed: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_repr(batch, fixed)).squeeze(-1)
