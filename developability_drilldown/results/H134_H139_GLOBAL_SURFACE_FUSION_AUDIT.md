# H134–H139 Global F1_SURFACE Fusion Audit

**Status:** COMPLETE before H134 training  
**Scope:** Exact historical H090/H086 late fusion; do not redefine it.

## Historical F1_SURFACE35

| Item | Value |
|------|-------|
| Bundle | `F1_SURFACE` |
| Composition | ARO19 + HYDRO16 = **35D** |
| Source | `experiments/features/EXP-H047.parquet` |
| Loader | `H047AuxFeatureStore("F1_SURFACE")` |
| Columns | H047 feat indices 1395:1430 (`aro_*` then `hydro_*` / field cols) |
| artifact_hash | `e3788aad831d0b022ebfa682a46e0a27c9fbaf0e5f833aacaa3b191819898a4b` |
| parquet content SHA256 | `79348fa8e9b17344463a677fd99d5976c51c5cce02f0d9c7386f7074fc529225` |

## Preprocessing (H090/H086)

```
TRAIN-only median imputation
  → StandardScaler (TRAIN fit)
  → no PCA for F1
```

Applied fold-locally via `FoldPreprocessor`. VAL/TEST/external use frozen TRAIN stats only.

## Historical late fusion architecture

| Item | H090 | H086 |
|------|------|------|
| Backbone | EXP-H071 Scratch ARCH-2 | EXP-H061 ESM2 ARCH-4 |
| Sequence path | `joint_hl_single_reg` | `joint_hl_chain_specific_dual_reg` + **mean** merge |
| `z_seq` dim | 128 | 128 |
| Fusion mode | **`late_concat_aux32`** | **`late_concat_aux32`** |
| Aux encoder | `LateFusionAuxMLP`: Linear(35,64)→GELU→Drop0.2→Linear(64,32)→GELU | same |
| Fusion | `concat(z_seq, z_aux)` → **160D** | same |
| Head | **New** Linear(160,128)→GELU→Drop→Linear(128,1) (backbone.head unused) | same |
| Platform | `DL_FOLDLOCAL_COSINE_V3` | same |
| Seed | 101 | 101 |
| n_trainable | 349090 | 510370 |
| TEST_mean | **0.471512** | **0.486326** |

Controls (not retrained): H071 TEST_mean ≈ 0.501688; H061 ≈ 0.506795.

## Sequence representation entering fusion

Both use `transformer.forward_repr(batch)` → `z_seq ∈ R^128`.

| Backbone | Independent `z_H` / `z_L` exposed? |
|----------|-------------------------------------|
| H071 | **No** (single REG) |
| H061 | **No** for fusion API (merged mean before head) |

**Decision for Mode C:** use two-token form `[z_seq, t_surf]` for **both** backbones. Do not manufacture H/L tokens.

## What H134–H139 change

Condition **after** canonical `z_seq`, **before** the **canonical backbone regression head** (dim unchanged).

No AUX32 branch. No late F1 concat after conditioning. No residue COMPACT10 / SAP / SCM.

## Token form decision (prereg)

```
global_surface_token_attention: [z_seq, t_surf]  # 2 tokens, n_heads=2, n_layers=1
```

Token attention uses a single MultiheadAttention + residual LayerNorm over 2 tokens (not a full TransformerEncoderLayer FFN) to keep capacity modest while satisfying n_layers=1, n_heads=2.

| Mode | Added params |
|------|-------------:|
| FiLM | 4928 |
| gated residual | 5665 |
| token attention | 70912 |

Flag: token-attention adds more capacity than FiLM/gated (~70k); still no architecture search.
