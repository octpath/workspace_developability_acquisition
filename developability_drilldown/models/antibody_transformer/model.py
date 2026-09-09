#!/usr/bin/env python3
"""Small annotation-aware Transformer (2 layers, shared H/L encoder)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import REGION_TO_IDX
from .distance_bias import SharedDistanceBiasTransformerEncoder

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
        use_continuous_rasa: bool = False,
        use_rasa_weighted_pool: bool = False,
        use_ca_distance_bias: bool = False,
        joint_hl_single_reg: bool = False,
        joint_hl_dual_reg: bool = False,
        initial_ell_angstrom: Optional[float] = None,
    ):
        super().__init__()
        if n_layers != 2:
            raise ValueError("n_layers is frozen at 2")
        if pooling_mode not in ("reg", "region_gate"):
            raise ValueError(pooling_mode)
        if use_continuous_rasa and use_rasa_weighted_pool:
            raise ValueError("use continuous RASA annotation XOR rasa-weighted pool, not both")
        if use_ca_distance_bias and (use_continuous_rasa or use_rasa_weighted_pool):
            raise ValueError("CA distance bias is mutually exclusive with RASA modes in this experiment series")
        if joint_hl_single_reg and joint_hl_dual_reg:
            raise ValueError("joint_hl_single_reg XOR joint_hl_dual_reg")
        if (joint_hl_single_reg or joint_hl_dual_reg) and (use_continuous_rasa or use_rasa_weighted_pool):
            raise ValueError("joint H/L modes are mutually exclusive with RASA modes in this series")
        if (joint_hl_single_reg or joint_hl_dual_reg) and pooling_mode != "reg":
            raise ValueError("joint H/L modes require pooling_mode='reg'")
        if (joint_hl_single_reg or joint_hl_dual_reg) and chain_mode != "HL":
            raise ValueError("joint H/L modes require chain_mode='HL'")
        if joint_hl_dual_reg and use_ca_distance_bias:
            raise ValueError("joint_hl_dual_reg does not support CA distance bias in EXP-T070")
        self.content_mode = content_mode
        self.annotation_mode = annotation_mode
        self.merge_mode = merge_mode
        self.chain_mode = chain_mode
        self.pooling_mode = pooling_mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_continuous_rasa = bool(use_continuous_rasa)
        self.use_rasa_weighted_pool = bool(use_rasa_weighted_pool)
        self.use_ca_distance_bias = bool(use_ca_distance_bias)
        self.joint_hl_single_reg = bool(joint_hl_single_reg)
        self.joint_hl_dual_reg = bool(joint_hl_dual_reg)
        self.rasa_pool_eps = 1e-12
        self.rasa_pool_zero_denom_count = 0
        self.distance_kernel_shared_across_layers = True

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

        # Continuous additive RASA annotation: x += Linear(1, d_model, bias=False)(r)
        # Zero-init => contribution is identically 0 at initialization (matches control).
        if self.use_continuous_rasa:
            self.rasa_proj = nn.Linear(1, d_model, bias=False)
            nn.init.zeros_(self.rasa_proj.weight)
        else:
            self.rasa_proj = None

        # Shared RASA-weighted residue pool -> REG residual (EXP-T066).
        # Zero-init => REG' == REG at initialization (matches control).
        if self.use_rasa_weighted_pool:
            if pooling_mode != "reg":
                raise ValueError("rasa_weighted_pool requires pooling_mode='reg'")
            self.rasa_pool_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.rasa_pool_proj.weight)
        else:
            self.rasa_pool_proj = None

        # REG tokens:
        # - separate H/L encoding / joint dual-REG: index 0 = REG_H, 1 = REG_L
        # - joint H/L single-REG: one antibody-level REG (neither H nor L)
        if self.joint_hl_single_reg:
            self.register_parameter("reg_token", None)
            self.single_reg_token = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.single_reg_token, std=0.02)
        else:
            self.reg_token = nn.Parameter(torch.zeros(2, d_model))
            nn.init.normal_(self.reg_token, std=0.02)
            self.register_parameter("single_reg_token", None)

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
        if self.use_ca_distance_bias:
            self.encoder = SharedDistanceBiasTransformerEncoder(
                enc_layer, num_layers=n_layers, n_heads=n_heads
            )
            if initial_ell_angstrom is not None:
                self.encoder.set_initial_ell(float(initial_ell_angstrom))
        else:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        if self.joint_hl_single_reg:
            out_dim = d_model
        else:
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

    def rasa_weighted_residue_pool(
        self,
        h_res: torch.Tensor,
        mask: torch.Tensor,
        rasa: torch.Tensor,
    ) -> torch.Tensor:
        """Continuous RASA-weighted mean of residue hidden states (real AA only).

        rasa_pool = sum_i r_i h_i / sum_i r_i  over mask==True residues.
        If sum_i r_i <= eps for a chain, return zeros and count the event.
        """
        r = torch.nan_to_num(rasa, nan=0.0) * mask.float()
        denom = r.sum(dim=1)  # [B]
        weighted = (h_res * r.unsqueeze(-1)).sum(dim=1)  # [B, D]
        ok = denom > self.rasa_pool_eps
        n_bad = int((~ok).sum().item())
        if n_bad:
            self.rasa_pool_zero_denom_count += n_bad
        denom_safe = torch.where(ok, denom, torch.ones_like(denom))
        pool = weighted / denom_safe.unsqueeze(-1)
        return torch.where(ok.unsqueeze(-1), pool, torch.zeros_like(pool))

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
        rasa: Optional[torch.Tensor] = None,
        ca_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return chain representation. mask: [B,L] True=valid residue."""
        B, L = mask.shape
        content = self._content(aa, plm)
        x = content + self.pos_emb(pos) + self.chain_emb.weight[chain_idx]
        if self.use_imgt and imgt is not None:
            x = x + self.imgt_emb(imgt)
        if self.use_region and region is not None:
            x = x + self.region_emb(region)
        if self.use_continuous_rasa and self.rasa_proj is not None:
            if rasa is None:
                raise ValueError("continuous RASA enabled but rasa tensor missing")
            # Missing/unresolved -> 0 contribution (explicit); pad already 0.
            r = torch.nan_to_num(rasa, nan=0.0).unsqueeze(-1)  # [B,L,1]
            x = x + self.rasa_proj(r)

        reg = self.reg_token[chain_idx].view(1, 1, -1).expand(B, 1, -1)
        reg = reg + self.chain_emb.weight[chain_idx]
        x = torch.cat([reg, x], dim=1)

        pad = torch.zeros(B, 1 + L, dtype=torch.bool, device=mask.device)
        pad[:, 1:] = ~mask
        if self.use_ca_distance_bias:
            if ca_coords is None:
                raise ValueError("CA distance bias enabled but ca_coords missing")
            # Missing CA -> nan; treat as non-contributing residue in pair mask
            coords = torch.nan_to_num(ca_coords, nan=0.0)
            ca_ok = torch.isfinite(ca_coords).all(dim=-1) & mask
            h = self.encoder(
                self.dropout(x),
                src_key_padding_mask=pad,
                residue_coords=coords,
                residue_mask=ca_ok,
            )
        else:
            h = self.encoder(self.dropout(x), src_key_padding_mask=pad)
        if self.pooling_mode == "reg":
            reg_out = h[:, 0]
            if self.use_rasa_weighted_pool and self.rasa_pool_proj is not None:
                if rasa is None:
                    raise ValueError("rasa-weighted pool enabled but rasa tensor missing")
                # Pool over residue states only (exclude REG / pad); shared W_pool for H/L.
                rasa_pool = self.rasa_weighted_residue_pool(h[:, 1:], mask, rasa)
                reg_out = reg_out + self.rasa_pool_proj(rasa_pool)
            return reg_out
        if region is None:
            raise ValueError("region required for region_gate pooling")
        return self._pool_region_gate(h[:, 1:], mask, region, chain_idx)

    def _residue_stream(
        self,
        *,
        chain_idx: int,
        aa: Optional[torch.Tensor],
        plm: Optional[torch.Tensor],
        pos: torch.Tensor,
        imgt: Optional[torch.Tensor],
        region: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-residue embeddings with T030 FULL annotations (no REG)."""
        content = self._content(aa, plm)
        x = content + self.pos_emb(pos) + self.chain_emb.weight[chain_idx]
        if self.use_imgt and imgt is not None:
            x = x + self.imgt_emb(imgt)
        if self.use_region and region is not None:
            x = x + self.region_emb(region)
        return x

    def encode_joint_hl(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """Single-encoder joint H+L sequence: [REG, H..., L...]; return REG hidden."""
        if not self.joint_hl_single_reg:
            raise RuntimeError("encode_joint_hl requires joint_hl_single_reg=True")
        mh = batch["heavy_mask"]
        ml = batch["light_mask"]
        B = mh.shape[0]
        Lh = mh.shape[1]
        Ll = ml.shape[1]
        x_h = self._residue_stream(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
        )
        x_l = self._residue_stream(
            chain_idx=1,
            aa=batch.get("light_aa"),
            plm=batch.get("light_plm"),
            pos=batch["light_pos"],
            imgt=batch.get("light_imgt"),
            region=batch.get("light_region"),
        )
        # Single REG: neither H nor L — no chain / IMGT / region / position embedding.
        reg = self.single_reg_token.view(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([reg, x_h, x_l], dim=1)

        pad = torch.zeros(B, 1 + Lh + Ll, dtype=torch.bool, device=mh.device)
        pad[:, 1 : 1 + Lh] = ~mh
        pad[:, 1 + Lh :] = ~ml

        if self.use_ca_distance_bias:
            ca_h = batch.get("heavy_ca")
            ca_l = batch.get("light_ca")
            if ca_h is None or ca_l is None:
                raise ValueError("CA distance bias enabled but CA coords missing")
            coords = torch.cat(
                [torch.nan_to_num(ca_h, nan=0.0), torch.nan_to_num(ca_l, nan=0.0)],
                dim=1,
            )
            ca_ok = torch.cat(
                [
                    torch.isfinite(ca_h).all(dim=-1) & mh,
                    torch.isfinite(ca_l).all(dim=-1) & ml,
                ],
                dim=1,
            )
            h = self.encoder(
                self.dropout(x),
                src_key_padding_mask=pad,
                residue_coords=coords,
                residue_mask=ca_ok,
            )
        else:
            h = self.encoder(self.dropout(x), src_key_padding_mask=pad)
        return h[:, 0]

    def encode_joint_hl_dual_reg(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """Joint H+L with dual REG slots: [REG_H, H..., REG_L, L...]; return REG_H||REG_L.

        Residue chain identity: H residues get chain_emb[0], L get chain_emb[1].
        REG chain semantics match T030 separate encoding: REG_H += chain_emb[0],
        REG_L += chain_emb[1] (historical T030 assigned chain emb to REG tokens).
        REG tokens do not receive IMGT/region/position embeddings.
        """
        if not self.joint_hl_dual_reg:
            raise RuntimeError("encode_joint_hl_dual_reg requires joint_hl_dual_reg=True")
        mh = batch["heavy_mask"]
        ml = batch["light_mask"]
        B = mh.shape[0]
        Lh = mh.shape[1]
        Ll = ml.shape[1]
        x_h = self._residue_stream(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
        )
        x_l = self._residue_stream(
            chain_idx=1,
            aa=batch.get("light_aa"),
            plm=batch.get("light_plm"),
            pos=batch["light_pos"],
            imgt=batch.get("light_imgt"),
            region=batch.get("light_region"),
        )
        reg_h = self.reg_token[0].view(1, 1, -1).expand(B, 1, -1) + self.chain_emb.weight[0]
        reg_l = self.reg_token[1].view(1, 1, -1).expand(B, 1, -1) + self.chain_emb.weight[1]
        # Layout: [REG_H], H_1..H_n, [REG_L], L_1..L_m
        x = torch.cat([reg_h, x_h, reg_l, x_l], dim=1)

        pad = torch.zeros(B, 2 + Lh + Ll, dtype=torch.bool, device=mh.device)
        pad[:, 1 : 1 + Lh] = ~mh
        # REG_L at index 1+Lh is never padding
        pad[:, 2 + Lh :] = ~ml

        h = self.encoder(self.dropout(x), src_key_padding_mask=pad)
        z_h = h[:, 0]
        z_l = h[:, 1 + Lh]
        return torch.cat([z_h, z_l], dim=-1)

    def forward_repr(self, batch: dict) -> torch.Tensor:
        if self.joint_hl_dual_reg:
            return self.encode_joint_hl_dual_reg(batch)
        if self.joint_hl_single_reg:
            return self.encode_joint_hl(batch)
        h_h = self.encode_chain(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            mask=batch["heavy_mask"],
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
            rasa=batch.get("heavy_rasa"),
            ca_coords=batch.get("heavy_ca"),
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
            rasa=batch.get("light_rasa"),
            ca_coords=batch.get("light_ca"),
        )
        if self.merge_mode == "concat":
            return torch.cat([h_h, h_l], dim=-1)
        if self.merge_mode == "mean":
            return 0.5 * (h_h + h_l)
        raise ValueError(self.merge_mode)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.forward_repr(batch)).squeeze(-1)
