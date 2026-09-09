# EXP-T070 pre-run architecture audit

## A–G checklist

| ID | Claim | Status |
|----|-------|--------|
| A | T030 uses separate H/L encoder calls | **TRUE** — `encode_chain` twice in `forward_repr` |
| B | T030 uses REG_H and REG_L | **TRUE** — `reg_token` shape `(2, d_model)` |
| C | T030 readout is REG_H \|\| REG_L | **TRUE** — `merge_mode=concat` → `2*d_model` |
| D | chain ID present in FULL residue annotation | **TRUE** — `chain_emb` H=0 / L=1 on residues |
| E | T068 uses one joint encoder and single REG | **TRUE** — `encode_joint_hl` + `single_reg_token` |
| F | T070 differs from T030 primarily by joint H/L attention | **TRUE** — one encoder over `[REG_H,H...,REG_L,L...]` |
| G | T070 differs from T068 primarily by restoring dual REG | **TRUE** — REG_H\|\|REG_L vs single REG |

## Chain-ID policy (documented)

### T030 (separate)
- Residue tokens: `+ chain_emb[chain_idx]`
- REG tokens: `reg_token[i] + chain_emb[i]` (REG_H gets H emb, REG_L gets L emb)

### T068 (joint single-REG)
- Residue tokens: same chain_emb as T030
- Single REG: **no** chain_emb (neither H nor L)

### T070 (joint dual-REG)
- Residue tokens: same chain_emb as T030 (H/L retained)
- REG_H / REG_L: **match T030** — `+ chain_emb[0/1]` (historical T030 assigned chain emb to REG)
- REG tokens still receive **no** IMGT / region / position embeddings

## Token layout T070

`[REG_H], H_1..H_n, [REG_L], L_1..L_m`

No SEP. No H–L attention block mask. Per-chain sequence positions restart (T030 semantics).

## Verdict

All A–G true → proceed to training.
