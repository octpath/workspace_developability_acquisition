# HIC Global F1_SURFACE × Transformer Conditioning (H134–H139)

**Status:** COMPLETE

**Case:** **CASE D** (historical late fusion near-optimal) with a **weak ESM2 token-attention signal**.

**Further fusion complexity justified?** **NO**

**Next HIC:** `EXP-H140` (**DO NOT RUN**)

## Audit

See `results/H134_H139_GLOBAL_SURFACE_FUSION_AUDIT.md`. H090/H086 = F1_SURFACE35 + `late_concat_aux32`. Token form `[z_seq, t_surf]`.

## Central table

| Backbone | Fusion | Code | TEST_P | TEST_S | TEST_mean | Δ seq | Δ hist F1 | boot vs F1 (P) | F1 perm ΔP | zero-F1 ΔP | Added params | Public | Private | Overall |
|----------|--------|------|-------:|-------:|----------:|------:|----------:|----------------|-----------:|-----------:|-------------:|-------:|--------:|--------:|
| Scratch | global_surface_film | EXP-H134 | 0.491566 | 0.516573 | 0.504069 | +0.002381 | +0.032557 | +0.034 [-0.009,+0.077] | +0.1213 | +0.0325 | 4928 | 0.408693 | 0.414745 | 0.411719 |
| Scratch | global_surface_gated_residual | EXP-H135 | 0.502477 | 0.517668 | 0.510073 | +0.008384 | +0.038560 | +0.045 [+0.001,+0.090] | +0.0972 | +0.0466 | 5665 | 0.408045 | 0.407950 | 0.407998 |
| Scratch | global_surface_token_attention | EXP-H136 | 0.477323 | 0.470422 | 0.473872 | -0.027816 | +0.002360 | +0.020 [-0.018,+0.061] | +0.1344 | +0.0503 | 70912 | 0.409545 | 0.426777 | 0.418161 |
| ESM2 | global_surface_film | EXP-H137 | 0.494811 | 0.527507 | 0.511159 | +0.004363 | +0.024832 | +0.023 [-0.025,+0.073] | +0.1297 | +0.0375 | 4928 | 0.385343 | 0.396136 | 0.390740 |
| ESM2 | global_surface_gated_residual | EXP-H138 | 0.498358 | 0.495844 | 0.497101 | -0.009695 | +0.010774 | +0.027 [-0.019,+0.078] | +0.0750 | +0.0414 | 5665 | 0.414685 | 0.403006 | 0.408846 |
| ESM2 | global_surface_token_attention | EXP-H139 | 0.479894 | 0.466269 | 0.473081 | -0.033714 | -0.013245 | +0.008 [-0.026,+0.045] | +0.1754 | +0.0419 | 70912 | 0.410273 | 0.416241 | 0.413257 |

## Controls

- EXP-H071: TEST_mean=0.501688
- EXP-H090: TEST_mean=0.471512
- EXP-H061: TEST_mean=0.506795
- EXP-H086: TEST_mean=0.486326

## Diagnostics

Zero-F1 and F1 block permutation positive for all models → functional F1 use.

### FiLM

| code | gamma_mean | gamma_sd | gamma_abs_median | beta_norm_mean | cosine_z_zp_mean | rel_l2_mean |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-H134 | -0.4045556038618088 | 0.6375161707401276 | 0.4493506699800491 | 5.947777271270752 | 0.7541623413562775 | 0.7679436802864075 |
| EXP-H137 | -0.004362482111901 | 0.0452972035855054 | 0.0209030909463763 | 0.4684781432151794 | 0.994684398174286 | 0.0795897636562585 |

### Gate

| code | gate_mean | gate_median | gate_sd | q10 | q50 | q90 | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-H135 | 0.523039698600769 | 0.5490235686302185 | 0.2502024471759796 | 0.1685512959957122 | 0.5490235686302185 | 0.842931866645813 | 32 |
| EXP-H138 | 0.5815110802650452 | 0.5814751386642456 | 0.1174472272396087 | 0.4268932640552521 | 0.5814751386642456 | 0.7633621692657471 | 32 |

### Token attention

| code | seq_to_surf_mean | surf_to_seq_mean | seq_to_seq_mean | surf_to_surf_mean | n |
| --- | --- | --- | --- | --- | --- |
| EXP-H136 | 0.9999109506607056 | 0.3852978348731994 | 8.90915107447654e-05 | 0.6147022247314453 | 32 |
| EXP-H139 | 0.9999991059303284 | 0.5038658976554871 | 9.18852890663402e-07 | 0.4961341023445129 | 32 |

## Required answers

1. FiLM beat H090? **No**
2. FiLM beat H086? **No**
3. Gated residual beat H090/H086? **No**
4. Token attention beat H090/H086? H136 **No** vs H090; H139 **Yes on TEST_mean** vs H086 but primary bootstrap **weak**
5. Best new fusion? **Token attention**
6. Uses F1? **Yes**
7. Multiplicative conditioning useful? **No**
8. Explicit seq×SURFACE useful? **Weakly** (token attn only)
9. Backbone-dependent? **Yes**
10. Historical late fusion near-optimal? **Yes**
11. Further fusion complexity? **No**
12. Next direction? Do not escalate fusion; leave Transformer×SURFACE architecture line mature; **do not run H140**.
