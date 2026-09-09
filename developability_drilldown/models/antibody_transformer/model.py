#!/usr/bin/env python3
"""Small annotation-aware Transformer (2 layers, shared H/L encoder)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import REGION_TO_IDX

# Region-gate groups (global learned weights; descriptive only)
FR_REGION_IDX = [
    REGION_TO_IDX["FR1"],
    REGION_TO_IDX["FR2"],
    REGION_TO_IDX["FR3"],
    REGION_TO_IDX["FR4"],
]
CDR_REGION_IDX = {
    "CDR1": REGION_TO_IDX["CDR1"],
    "CDR2": REGION_TO_IDX["CDR2"],
    "CDR3": REGION_TO_IDX["CDR3"],
}
REGION_GATE_NAMES = ["FR_ALL", "CDR1", "CDR2", "CDR3"]


class AnnotatedTransformer(nn.Module):
    def __init__(
        self,
        *,
        content_mode: str,  # scratch | frozen
        plm_hidden: int = 0,
        n_aa: int = 22,  # pad + 20 + unk
        max_seq_pos: int = 160,
        n_imgt: int = 64,
        n_region: int = len(REGION_TO_IDX),
        annotation_mode: str = "full",  # minimal | full
        merge_mode: str = "concat",  # concat | mean | h_only
        chain_mode: str = "HL",  # HL | H_ONLY
        pooling_mode: str = "reg",  # reg | region_gate
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.20,
        norm_first: bool = True,
    ):
        super().__init__()
        if n_layers != 2:
            raise ValueError("n_layers is frozen at 2")
        if pooling_mode not in ("reg", "region_gate"):
            raise ValueError(pooling_mode)
        self.content_mode = content_mode
        self.annotation_mode = annotation_mode
        self.merge_mode = merge_mode
        self.chain_mode = chain_mode
        self.pooling_mode = pooling_mode
        self.d_model = d_model

        if content_mode == "scratch":
            self.aa_emb = nn.Embedding(n_aa, d_model, padding_idx=0)
            self.plm_proj = None
        elif content_mode == "frozen":
            if plm_hidden <= 0:
                raise ValueError("plm_hidden required for frozen mode")
            self.aa_emb = None
            self.plm_proj = nn.Linear(plm_hidden, d_model)
        else:
            raise ValueError(content_mode)

        self.pos_emb = nn.Embedding(max_seq_pos + 1, d_model, padding_idx=0)
        self.chain_emb = nn.Embedding(2, d_model)  # H=0, L=1
        self.use_imgt = annotation_mode == "full"
        self.use_region = annotation_mode == "full"
        if self.use_imgt:
            self.imgt_emb = nn.Embedding(n_imgt, d_model, padding_idx=0)
        if self.use_region:
            self.region_emb = nn.Embedding(n_region, d_model, padding_idx=0)

        # REG tokens: index 0 = REG_H, 1 = REG_L (learned)
        self.reg_token = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.reg_token, std=0.02)

        # Global (not sample-dependent) region-gate logits per chain
        if pooling_mode == "region_gate":
            self.region_gate_logits = nn.Parameter(torch.zeros(2, 4))
            nn.init.zeros_(self.region_gate_logits)
        else:
            self.register_parameter("region_gate_logits", None)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        out_dim = d_model if merge_mode in ("mean", "h_only") else 2 * d_model
        self.head = nn.Sequential(
            nn.Linear(out_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.repr_dim = out_dim

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def region_gate_weights(self) -> dict[str, torch.Tensor]:
        """Return softmax weights [4] per chain (descriptive aggregation weights)."""
        if self.region_gate_logits is None:
            raise RuntimeError("pooling_mode is not region_gate")
        w = F.softmax(self.region_gate_logits, dim=-1)
        return {"H": w[0], "L": w[1]}

    def _content(
        self,
        aa: Optional[torch.Tensor],
        plm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.content_mode == "scratch":
            return self.aa_emb(aa)
        return self.plm_proj(plm)

    def _pool_region_gate(
        self,
        h_res: torch.Tensor,
        mask: torch.Tensor,
        region: torch.Tensor,
        chain_idx: int,
    ) -> torch.Tensor:
        """Masked means over FR_ALL/CDR1/CDR2/CDR3 then global learned mix."""
        B, L, D = h_res.shape
        device = h_res.device
        group_masks = []
        fr = torch.zeros(B, L, dtype=torch.bool, device=device)
        for ri in FR_REGION_IDX:
            fr = fr | (region == ri)
        fr = fr & mask
        group_masks.append(fr)
        for name in ("CDR1", "CDR2", "CDR3"):
            group_masks.append((region == CDR_REGION_IDX[name]) & mask)

        means = []
        present = []
        for gm in group_masks:
            denom = gm.float().sum(dim=1)
            present.append(denom > 0)
            summed = (h_res * gm.unsqueeze(-1).float()).sum(dim=1)
            means.append(
                torch.where(
                    denom.unsqueeze(-1) > 0,
                    summed / denom.unsqueeze(-1).clamp_min(1.0),
                    summed,
                )
            )
        stacked = torch.stack(means, dim=1)  # [B,4,D]
        present_t = torch.stack(present, dim=1).float()  # [B,4]
        logits = self.region_gate_logits[chain_idx].view(1, 4).expand(B, 4)
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(present_t > 0, logits, torch.full_like(logits, neg_inf))
        all_miss = present_t.sum(dim=1) == 0
        if all_miss.any():
            masked_logits = torch.where(
                all_miss.unsqueeze(-1), torch.zeros_like(masked_logits), masked_logits
            )
        w = F.softmax(masked_logits, dim=-1)
        return (stacked * w.unsqueeze(-1)).sum(dim=1)

    def encode_chain(
        self,
        *,
        chain_idx: int,
        aa: Optional[torch.Tensor],
        plm: Optional[torch.Tensor],
        mask: torch.Tensor,
        pos: torch.Tensor,
        imgt: Optional[torch.Tensor],
        region: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return chain representation. mask: [B,L] True=valid residue."""
        B, L = mask.shape
        content = self._content(aa, plm)
        x = content + self.pos_emb(pos) + self.chain_emb.weight[chain_idx]
        if self.use_imgt and imgt is not None:
            x = x + self.imgt_emb(imgt)
        if self.use_region and region is not None:
            x = x + self.region_emb(region)

        reg = self.reg_token[chain_idx].view(1, 1, -1).expand(B, 1, -1)
        reg = reg + self.chain_emb.weight[chain_idx]
        x = torch.cat([reg, x], dim=1)

        pad = torch.zeros(B, 1 + L, dtype=torch.bool, device=mask.device)
        pad[:, 1:] = ~mask
        h = self.encoder(self.dropout(x), src_key_padding_mask=pad)
        if self.pooling_mode == "reg":
            return h[:, 0]
        if region is None:
            raise ValueError("region required for region_gate pooling")
        return self._pool_region_gate(h[:, 1:], mask, region, chain_idx)

    def forward_repr(self, batch: dict) -> torch.Tensor:
        h_h = self.encode_chain(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            mask=batch["heavy_mask"],
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
        )
        if self.chain_mode == "H_ONLY" or self.merge_mode == "h_only":
            return h_h
        h_l = self.encode_chain(
            chain_idx=1,
            aa=batch.get("light_aa"),
            plm=batch.get("light_plm"),
            mask=batch["light_mask"],
            pos=batch["light_pos"],
            imgt=batch.get("light_imgt"),
            region=batch.get("light_region"),
        )
        if self.merge_mode == "concat":
            return torch.cat([h_h, h_l], dim=-1)
        if self.merge_mode == "mean":
            return 0.5 * (h_h + h_l)
        raise ValueError(self.merge_mode)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.forward_repr(batch)).squeeze(-1)
