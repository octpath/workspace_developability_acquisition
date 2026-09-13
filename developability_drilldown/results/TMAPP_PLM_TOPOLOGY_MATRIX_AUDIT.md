# TmApp PLM × H/L Topology Matrix — Pre-Training Audit

**Status:** PASS — safe to preregister and train T151–T156.  
**Date:** 2026-09-13  
**Prior freeze (unchanged provenance):** `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.*`

## 5.1 Historical architecture equivalence

| Cell | Code | PLM | Topology flags | merge | content | seed | platform |
|------|------|-----|----------------|-------|---------|------|----------|
| AbLingua A | EXP-T080 | ABLINGUA | separate dual REG (no cross) | mean | frozen | 101 | DL_FOLDLOCAL_COSINE_V3 |
| AbLingua B1 | EXP-T081 | ABLINGUA | `joint_hl_dual_reg=True` (ARCH-3) | mean | frozen | 101 | V3 |
| AbLingua B2 | EXP-T082 | ABLINGUA | `joint_hl_chain_specific_dual_reg=True` (ARCH-4) | mean | frozen | 101 | V3 |
| AbLingua C | EXP-T087 | ABLINGUA | `use_reg_only_cross_attention=True` (ARCH-7 / C0) | mean | frozen | 101 | V3 |
| AbLang2 A | EXP-T110 | ABLANG2 | separate dual REG | mean | frozen | 101 | V3 |
| AbLang2 B1 | EXP-T113 | ABLANG2 | ARCH-3 | mean | frozen | 101 | V3 |
| AbLang2 B2 | EXP-T115 | ABLANG2 | ARCH-4 | mean | frozen | 101 | V3 |
| AbLang2 C | EXP-T121 | ABLANG2 | ARCH-7 C0 | mean | frozen | 101 | V3 |
| AbLang2 D | EXP-T149 | ABLANG2 | `pair_interaction_mode=d3_symmetric_mlp` | mean | frozen | 101 | V3 |

FULL annotations; `share_hl_encoder` default True (explicit on T149).  
`reg_cross_variant` absent/null on C cells → C0 ungated.  
No silent retrain of these cells.

Scratch context (not primary matrix): T091=A, T094=B1, T096=B2, T102=C.

## 5.2 PLM residue-alignment audit

Canonical antibody set: **324** (162 Dev + 162 Test).  
Shared `ids.npy` SHA256 prefix: `ce62b155c7355352…` across AbLingua / AbLang2 / ESM-2.  
Shared heavy/light mask hashes identical across all three packs.

| Check | AbLingua | AbLang2 | ESM-2 |
|-------|----------|---------|-------|
| n_ids = 324 | OK | OK | OK |
| missing/extra IDs vs Dev∪Test | 0/0 | 0/0 | 0/0 |
| H/L mask sum vs annotation mask | 0 mismatches | 0 mismatches | 0 mismatches |
| max_H=140 / max_L=120 vs arrays | OK | OK | OK |
| pad positions mean‖emb‖ | — | 0.0 | 0.0 |
| ESM-2 L materialized | — | — | OK `(324,120,1280)` |

No BOS/EOS leakage into residue slots (pad embeddings zero; mask lengths match AA).  
No silent truncation. **STOP not triggered.**

## 5.3 Projection semantics

Downstream `d_model=128`. Trainable `plm_proj: Linear(raw_dim → 128)`.

| PLM | raw_dim | projection params | model artifact |
|-----|---------|-------------------|----------------|
| AbLingua | 1280 | 163,968 | IDEA-AI4S/AbLingua (`ablingua600m/`) |
| AbLang2 | 480 | 61,568 | ablang2-paired |
| ESM-2 | 1280 | 163,968 | facebook/esm2_t33_650M_UR50D |

Do not add extra bottlenecks to equalize PLM parameter counts.

## Topology trainable totals (frozen PLM)

| Topology | AbLingua/ESM-2 | AbLang2 | interaction extras |
|----------|----------------|---------|--------------------|
| A/B1/B2 | 476,033 | 373,633 | — |
| C | 542,081 | 439,681 | cross-attn 66,048 |
| D (D3) | 481,297 | 378,897 | pair MLP 5,264 |

## Fold / protocol

- folds.csv SHA16: `09e0df54d280b4fe`
- seed: 101  
- platform: `DL_FOLDLOCAL_COSINE_V3`

## Verdict

Audit **PASS**. Proceed to V2 topology freeze + preregistration commit, then train T151–T156 only.
