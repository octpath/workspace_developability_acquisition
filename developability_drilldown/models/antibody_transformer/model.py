#!/usr/bin/env python3
"""Small annotation-aware Transformer (2 layers, shared H/L encoder)."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import REGION_TO_IDX
from .cross_geometry import (
    DEFAULT_RBF_CENTERS,
    DEFAULT_RBF_SIGMA,
    N_RBF,
    build_hl_distance,
    rbf_bias,
)
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
        joint_hl_chain_specific_dual_reg: bool = False,
        use_cross_attention_bridge: bool = False,
        use_reg_only_cross_attention: bool = False,
        use_within_chain_extra_attention: bool = False,
        cross_gate_mode: str = "learned",  # "learned" | "fixed_one"
        use_cross_geometry_bias: bool = False,
        cross_geometry_rbf_centers: Optional[Sequence[float]] = None,
        cross_geometry_rbf_sigma: float = DEFAULT_RBF_SIGMA,
        initial_ell_angstrom: Optional[float] = None,
    ):
        super().__init__()
        # Default platform depth is 2. Capacity refinements (T124–T129) may use 3
        # only for architectures that stack a full TransformerEncoder (not mid-bridge).
        if int(n_layers) not in (2, 3):
            raise ValueError(f"n_layers must be 2 or 3, got {n_layers}")
        if pooling_mode not in ("reg", "region_gate"):
            raise ValueError(pooling_mode)
        if use_continuous_rasa and use_rasa_weighted_pool:
            raise ValueError("use continuous RASA annotation XOR rasa-weighted pool, not both")
        if use_ca_distance_bias and (use_continuous_rasa or use_rasa_weighted_pool):
            raise ValueError("CA distance bias is mutually exclusive with RASA modes in this experiment series")
        if cross_gate_mode not in ("learned", "fixed_one"):
            raise ValueError(f"cross_gate_mode must be 'learned' or 'fixed_one', got {cross_gate_mode!r}")
        if use_cross_geometry_bias and not use_cross_attention_bridge:
            raise ValueError("use_cross_geometry_bias requires use_cross_attention_bridge=True")
        if use_cross_geometry_bias and use_ca_distance_bias:
            raise ValueError("use_cross_geometry_bias is mutually exclusive with encoder use_ca_distance_bias")
        arch_switches = (
            int(joint_hl_single_reg)
            + int(joint_hl_dual_reg)
            + int(joint_hl_chain_specific_dual_reg)
            + int(use_cross_attention_bridge)
            + int(use_reg_only_cross_attention)
            + int(use_within_chain_extra_attention)
        )
        if arch_switches > 1:
            raise ValueError(
                "at most one of joint_hl_single_reg / joint_hl_dual_reg / "
                "joint_hl_chain_specific_dual_reg / use_cross_attention_bridge / "
                "use_reg_only_cross_attention / use_within_chain_extra_attention"
            )
        special_arch = arch_switches > 0
        if special_arch and (use_continuous_rasa or use_rasa_weighted_pool):
            raise ValueError("joint/cross modes are mutually exclusive with RASA modes in this series")
        if special_arch and pooling_mode != "reg":
            raise ValueError("joint/cross modes require pooling_mode='reg'")
        if special_arch and chain_mode != "HL":
            raise ValueError("joint/cross modes require chain_mode='HL'")
        if (joint_hl_dual_reg or joint_hl_chain_specific_dual_reg) and use_ca_distance_bias:
            raise ValueError("joint dual-REG modes do not support CA distance bias")
        if (
            use_cross_attention_bridge
            or use_reg_only_cross_attention
            or use_within_chain_extra_attention
        ) and use_ca_distance_bias:
            raise ValueError("cross-attention variants do not support CA distance bias")
        # Mid-bridge ARCH-6/8 (and gated bridge) hard-code layers[0]/layers[1].
        if int(n_layers) != 2 and (
            use_cross_attention_bridge or use_within_chain_extra_attention
        ):
            raise ValueError(
                "use_cross_attention_bridge / use_within_chain_extra_attention require n_layers=2"
            )
        self.content_mode = content_mode
        self.annotation_mode = annotation_mode
        self.merge_mode = merge_mode
        self.chain_mode = chain_mode
        self.pooling_mode = pooling_mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.use_continuous_rasa = bool(use_continuous_rasa)
        self.use_rasa_weighted_pool = bool(use_rasa_weighted_pool)
        self.use_ca_distance_bias = bool(use_ca_distance_bias)
        self.joint_hl_single_reg = bool(joint_hl_single_reg)
        self.joint_hl_dual_reg = bool(joint_hl_dual_reg)
        self.joint_hl_chain_specific_dual_reg = bool(joint_hl_chain_specific_dual_reg)
        self.use_cross_attention_bridge = bool(use_cross_attention_bridge)
        self.use_reg_only_cross_attention = bool(use_reg_only_cross_attention)
        self.use_within_chain_extra_attention = bool(use_within_chain_extra_attention)
        self.cross_gate_mode = str(cross_gate_mode)
        self.use_cross_geometry_bias = bool(use_cross_geometry_bias)
        centers = (
            list(cross_geometry_rbf_centers)
            if cross_geometry_rbf_centers is not None
            else list(DEFAULT_RBF_CENTERS)
        )
        if len(centers) != N_RBF:
            raise ValueError(f"cross_geometry_rbf_centers must have length {N_RBF}, got {len(centers)}")
        self.cross_geometry_rbf_centers = centers
        self.cross_geometry_rbf_sigma = float(cross_geometry_rbf_sigma)
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
        # - separate H/L encoding / joint dual-REG / chain-specific dual-REG: index 0 = REG_H, 1 = REG_L
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

        # Shared MHA for residue bridge / within-chain extra / REG-only cross paths.
        needs_cross_attn = (
            self.use_cross_attention_bridge
            or self.use_reg_only_cross_attention
            or self.use_within_chain_extra_attention
        )
        if needs_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.cross_attn = None
        # Trainable gates only for learned cross-attention bridge (EXP-T072 default).
        if self.use_cross_attention_bridge and self.cross_gate_mode == "learned":
            self.cross_gate_h = nn.Parameter(torch.zeros(()))
            self.cross_gate_l = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("cross_gate_h", None)
            self.register_parameter("cross_gate_l", None)

        # ARCH-6G: per-head RBF weights on H↔L Cα distances (zero-init ≡ ARCH-6).
        if self.use_cross_geometry_bias:
            self.cross_geom_weight = nn.Parameter(torch.zeros(n_heads, N_RBF))
            self.register_buffer(
                "cross_geom_centers",
                torch.tensor(self.cross_geometry_rbf_centers, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_parameter("cross_geom_weight", None)
            self.register_buffer("cross_geom_centers", None, persistent=False)

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

    def cross_gate_values(self) -> dict[str, float]:
        if not (
            self.use_cross_attention_bridge
            and self.cross_gate_mode == "learned"
            and self.cross_gate_h is not None
            and self.cross_gate_l is not None
        ):
            raise RuntimeError("cross gates only exist with use_cross_attention_bridge and cross_gate_mode='learned'")
        return {
            "g_H": float(self.cross_gate_h.detach().cpu()),
            "g_L": float(self.cross_gate_l.detach().cpu()),
        }

    def cross_geometry_weight_values(self) -> torch.Tensor:
        """Return a detached copy of per-head RBF weights [n_heads, 8]."""
        if self.cross_geom_weight is None:
            raise RuntimeError("cross geometry weights only exist with use_cross_geometry_bias=True")
        return self.cross_geom_weight.detach().cpu().clone()

    def _merge_dual_reg(self, z_h: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        if self.merge_mode == "concat":
            return torch.cat([z_h, z_l], dim=-1)
        if self.merge_mode == "mean":
            return 0.5 * (z_h + z_l)
        raise ValueError(self.merge_mode)

    @staticmethod
    def build_chain_specific_reg_attn_mask(Lh: int, Ll: int, *, device=None) -> torch.Tensor:
        """Bool attn mask [T,T]: True = blocked (PyTorch MHA convention).

        Layout: [REG_H]=0, H=1..Lh, [REG_L]=1+Lh, L=2+Lh..T-1
        """
        T = 2 + Lh + Ll
        mask = torch.zeros(T, T, dtype=torch.bool, device=device)
        reg_h = 0
        reg_l = 1 + Lh
        # REG_H: only REG_H + Heavy
        mask[reg_h, reg_l] = True
        if Ll:
            mask[reg_h, 2 + Lh : T] = True
        # REG_L: only REG_L + Light
        mask[reg_l, reg_h] = True
        if Lh:
            mask[reg_l, 1 : 1 + Lh] = True
        # Heavy residues: cannot attend REG_L
        if Lh:
            mask[1 : 1 + Lh, reg_l] = True
        # Light residues: cannot attend REG_H
        if Ll:
            mask[2 + Lh : T, reg_h] = True
        return mask

    def param_account(self) -> dict[str, int]:
        """Trainable param breakdown for reporting."""
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        cross = (
            sum(p.numel() for p in self.cross_attn.parameters() if p.requires_grad)
            if self.cross_attn is not None
            else 0
        )
        gates = 0
        if self.cross_gate_h is not None:
            gates += int(self.cross_gate_h.numel())
        if self.cross_gate_l is not None:
            gates += int(self.cross_gate_l.numel())
        geometry = (
            int(self.cross_geom_weight.numel())
            if self.cross_geom_weight is not None and self.cross_geom_weight.requires_grad
            else 0
        )
        total = self.n_trainable_parameters()
        return {
            "encoder": enc,
            "cross_attention": cross,
            "gates": gates,
            "geometry": geometry,
            "head": head,
            "other": total - enc - cross - gates - geometry - head,
            "total": total,
        }

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
        *,
        chain_specific_reg: bool = False,
    ) -> torch.Tensor:
        """Joint H+L with dual REG slots: [REG_H, H..., REG_L, L...]; return REG_H||REG_L.

        Residue chain identity: H residues get chain_emb[0], L get chain_emb[1].
        REG chain semantics match T030 separate encoding: REG_H += chain_emb[0],
        REG_L += chain_emb[1] (historical T030 assigned chain emb to REG tokens).
        REG tokens do not receive IMGT/region/position embeddings.

        If chain_specific_reg=True (EXP-T071): apply directional attention mask so
        REG_H only queries Heavy(+self), REG_L only Light(+self); residues may still
        attend cross-chain but not the other chain's REG.
        """
        if not (self.joint_hl_dual_reg or self.joint_hl_chain_specific_dual_reg):
            raise RuntimeError("encode_joint_hl_dual_reg requires a dual-REG joint mode")
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

        attn_mask = None
        if chain_specific_reg or self.joint_hl_chain_specific_dual_reg:
            attn_mask = self.build_chain_specific_reg_attn_mask(Lh, Ll, device=mh.device)

        h = self.encoder(
            self.dropout(x),
            mask=attn_mask,
            src_key_padding_mask=pad,
        )
        z_h = h[:, 0]
        z_l = h[:, 1 + Lh]
        return self._merge_dual_reg(z_h, z_l)

    def _embed_chain_with_reg(
        self,
        *,
        chain_idx: int,
        aa: Optional[torch.Tensor],
        plm: Optional[torch.Tensor],
        mask: torch.Tensor,
        pos: torch.Tensor,
        imgt: Optional[torch.Tensor],
        region: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build [REG, residues] embeddings and padding mask (True=pad)."""
        B, L = mask.shape
        x_res = self._residue_stream(
            chain_idx=chain_idx, aa=aa, plm=plm, pos=pos, imgt=imgt, region=region
        )
        reg = self.reg_token[chain_idx].view(1, 1, -1).expand(B, 1, -1)
        reg = reg + self.chain_emb.weight[chain_idx]
        x = torch.cat([reg, x_res], dim=1)
        pad = torch.zeros(B, 1 + L, dtype=torch.bool, device=mask.device)
        pad[:, 1:] = ~mask
        return x, pad

    def encode_separate_with_cross_attn(self, batch: dict) -> torch.Tensor:
        """Separate self-attn with mid-stack residue attention (cross-chain or within-chain).

        Layer1 self-attn (shared) on each chain independently → residue MHA
        (REG excluded from Q/K/V) → Layer2 self-attn → dual REG merge.

        - use_cross_attention_bridge: H↔L cross with learned gates or fixed_one residual
        - use_within_chain_extra_attention: H←H / L←L ungated residual
        """
        if not (self.use_cross_attention_bridge or self.use_within_chain_extra_attention):
            raise RuntimeError(
                "encode_separate_with_cross_attn requires use_cross_attention_bridge "
                "or use_within_chain_extra_attention"
            )
        if self.use_ca_distance_bias:
            raise RuntimeError("CA bias incompatible with cross-attention bridge")
        layers = self.encoder.layers
        if len(layers) != 2:
            raise RuntimeError("expected exactly 2 encoder layers")

        mh = batch["heavy_mask"]
        ml = batch["light_mask"]
        x_h, pad_h = self._embed_chain_with_reg(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            mask=mh,
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
        )
        x_l, pad_l = self._embed_chain_with_reg(
            chain_idx=1,
            aa=batch.get("light_aa"),
            plm=batch.get("light_plm"),
            mask=ml,
            pos=batch["light_pos"],
            imgt=batch.get("light_imgt"),
            region=batch.get("light_region"),
        )
        # Match T030: dropout once on input then layer stack
        h1_h = layers[0](self.dropout(x_h), src_key_padding_mask=pad_h)
        h1_l = layers[0](self.dropout(x_l), src_key_padding_mask=pad_l)

        # Residue-only attention (exclude REG at index 0)
        h_res = h1_h[:, 1:]
        l_res = h1_l[:, 1:]
        if self.use_within_chain_extra_attention:
            cross_h, _ = self.cross_attn(
                h_res, h_res, h_res, key_padding_mask=~mh, need_weights=False
            )
            cross_l, _ = self.cross_attn(
                l_res, l_res, l_res, key_padding_mask=~ml, need_weights=False
            )
            h_res = h_res + cross_h
            l_res = l_res + cross_l
        else:
            attn_mask_hl = None
            attn_mask_lh = None
            if self.use_cross_geometry_bias:
                ca_h = batch.get("heavy_ca")
                ca_l = batch.get("light_ca")
                if ca_h is None or ca_l is None:
                    raise ValueError(
                        "use_cross_geometry_bias requires batch heavy_ca and light_ca"
                    )
                d_hl = build_hl_distance(ca_h, ca_l, mh, ml)
                bias_hl = rbf_bias(
                    d_hl,
                    self.cross_geom_weight,
                    centers=self.cross_geom_centers,
                    sigma=self.cross_geometry_rbf_sigma,
                )  # [B, H, Lh, Ll]
                B = bias_hl.shape[0]
                # batch_first MultiheadAttention: float attn_mask [B*n_heads, Lq, Lk]
                attn_mask_hl = bias_hl.reshape(B * self.n_heads, mh.shape[1], ml.shape[1])
                attn_mask_lh = bias_hl.transpose(-2, -1).reshape(
                    B * self.n_heads, ml.shape[1], mh.shape[1]
                )
            # key_padding_mask True = ignore; residue mask True=valid → invert
            hl_kwargs = dict(key_padding_mask=~ml, need_weights=False)
            lh_kwargs = dict(key_padding_mask=~mh, need_weights=False)
            if attn_mask_hl is not None:
                hl_kwargs["attn_mask"] = attn_mask_hl
            if attn_mask_lh is not None:
                lh_kwargs["attn_mask"] = attn_mask_lh
            cross_h, _ = self.cross_attn(h_res, l_res, l_res, **hl_kwargs)
            cross_l, _ = self.cross_attn(l_res, h_res, h_res, **lh_kwargs)
            if self.cross_gate_mode == "learned":
                h_res = h_res + self.cross_gate_h * cross_h
                l_res = l_res + self.cross_gate_l * cross_l
            else:
                # fixed_one: ungated residual (g ≡ 1)
                h_res = h_res + cross_h
                l_res = l_res + cross_l
        # REG tokens unchanged at bridge
        h2_in_h = torch.cat([h1_h[:, :1], h_res], dim=1)
        h2_in_l = torch.cat([h1_l[:, :1], l_res], dim=1)

        h2_h = layers[1](h2_in_h, src_key_padding_mask=pad_h)
        h2_l = layers[1](h2_in_l, src_key_padding_mask=pad_l)
        return self._merge_dual_reg(h2_h[:, 0], h2_l[:, 0])

    def encode_reg_only_cross_attention(self, batch: dict) -> torch.Tensor:
        """Full separate 2-layer encode, then REG-only cross-chain attention (ungated).

        REG_H queries all Light residues; REG_L queries all Heavy residues.
        Residues are unmodified after the cross path; return merged REG' pair.
        """
        if not self.use_reg_only_cross_attention:
            raise RuntimeError("encode_reg_only_cross_attention requires use_reg_only_cross_attention")
        if self.use_ca_distance_bias:
            raise RuntimeError("CA bias incompatible with REG-only cross-attention")
        if self.cross_attn is None:
            raise RuntimeError("cross_attn module missing")

        mh = batch["heavy_mask"]
        ml = batch["light_mask"]
        x_h, pad_h = self._embed_chain_with_reg(
            chain_idx=0,
            aa=batch.get("heavy_aa"),
            plm=batch.get("heavy_plm"),
            mask=mh,
            pos=batch["heavy_pos"],
            imgt=batch.get("heavy_imgt"),
            region=batch.get("heavy_region"),
        )
        x_l, pad_l = self._embed_chain_with_reg(
            chain_idx=1,
            aa=batch.get("light_aa"),
            plm=batch.get("light_plm"),
            mask=ml,
            pos=batch["light_pos"],
            imgt=batch.get("light_imgt"),
            region=batch.get("light_region"),
        )
        # Full shared encoder per chain (same as encode_chain path, keep all hidden)
        h_h = self.encoder(self.dropout(x_h), src_key_padding_mask=pad_h)
        h_l = self.encoder(self.dropout(x_l), src_key_padding_mask=pad_l)

        reg_h = h_h[:, :1]
        reg_l = h_l[:, :1]
        h_res = h_h[:, 1:]
        l_res = h_l[:, 1:]
        delta_h, _ = self.cross_attn(
            reg_h, l_res, l_res, key_padding_mask=~ml, need_weights=False
        )
        delta_l, _ = self.cross_attn(
            reg_l, h_res, h_res, key_padding_mask=~mh, need_weights=False
        )
        # Ungated residual on REG only; residues unused after this
        reg_h = reg_h + delta_h
        reg_l = reg_l + delta_l
        return self._merge_dual_reg(reg_h.squeeze(1), reg_l.squeeze(1))

    def forward_repr(self, batch: dict) -> torch.Tensor:
        if self.joint_hl_chain_specific_dual_reg:
            return self.encode_joint_hl_dual_reg(batch, chain_specific_reg=True)
        if self.joint_hl_dual_reg:
            return self.encode_joint_hl_dual_reg(batch, chain_specific_reg=False)
        if self.joint_hl_single_reg:
            return self.encode_joint_hl(batch)
        if self.use_reg_only_cross_attention:
            return self.encode_reg_only_cross_attention(batch)
        if self.use_cross_attention_bridge or self.use_within_chain_extra_attention:
            return self.encode_separate_with_cross_attn(batch)
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
        return self._merge_dual_reg(h_h, h_l)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.forward_repr(batch)).squeeze(-1)
