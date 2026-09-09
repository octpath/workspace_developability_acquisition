# T071 / T072 Pre-implementation Audit

Codes confirmed free: **EXP-T071**, **EXP-T072** (next unused = EXP-T071).

## Assumptions vs code

| Claim | Status | Evidence |
|-------|--------|----------|
| T030: separate H/L encoder calls, shared weights | **TRUE** | `forward_repr` → `encode_chain` twice; one `self.encoder` |
| T030: learned REG_H / REG_L | **TRUE** | `reg_token` shape `(2, d_model)` |
| T030: REG_H sees only Heavy; REG_L only Light | **TRUE** | each `encode_chain` sequence is `[REG_i, residues_i]` only |
| T030: readout = concat(REG_H, REG_L) | **TRUE** | `merge_mode=concat` → `2*d_model` |
| T030: chain ID in FULL annotations | **TRUE** | `chain_emb` H=0 / L=1 on residues; REG also `+ chain_emb[i]` |
| T068: joint encoder, one REG | **TRUE** | `encode_joint_hl` → `[REG, H..., L...]`, `single_reg_token` |
| T070: one joint encoder, two REG, concat readout | **TRUE** | `encode_joint_hl_dual_reg` |
| T070: REG_H/REG_L unrestricted H+L access | **TRUE** | encoder called with `src_key_padding_mask` only; **no** H–L / REG block mask |

## Verdict

All material assumptions hold → proceed to implement T071 then T072.

## Planned deltas

- **T071**: same token layout as T070 + directional bool attention mask enforcing chain-specific REG queries; residues may still attend cross-chain.
- **T072**: keep separate H/L sequences (T030); insert zero-gated bidirectional residue cross-attention between self-attn layer 1 and 2; REG not in cross Q/K/V.
