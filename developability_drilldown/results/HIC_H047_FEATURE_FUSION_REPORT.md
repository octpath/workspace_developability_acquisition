# HIC H047 late-fusion report (H054/H061/H071 + H082–H093)

Platform: `DL_FOLDLOCAL_COSINE_V3`. Late fusion only (aux → 32-d → concat `z_DL`). No global ESM2_H block A.

| Code | Backbone | Feature bundle | Aux dim | Params | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |
|------|----------|----------------|--------:|-------:|-------:|-------:|----------:|-----------:|----:|-----:|--------:|
| EXP-H054 | H054 | NONE | 0 | — | 0.502 | 0.513 | 0.508 | 0.513 | 0.483 | 0.463 | 0.473 |
| EXP-H061 | H061 | NONE | 0 | — | 0.533 | 0.481 | 0.507 | 0.533 | 0.482 | 0.462 | 0.472 |
| EXP-H071 | H071 | NONE | 0 | — | 0.528 | 0.475 | 0.502 | 0.528 | 0.480 | 0.480 | 0.480 |
| EXP-H082 | H054 | F1 SURFACE | 35 | 510370 | 0.482 | 0.491 | 0.487 | 0.491 | 0.418 | 0.426 | 0.422 |
| EXP-H083 | H054 | F2 SEQ+TITR | 133 | 516642 | 0.501 | 0.519 | 0.510 | 0.519 | 0.462 | 0.479 | 0.470 |
| EXP-H084 | H054 | F3 LOCAL_RASA | 32 | 510178 | 0.531 | 0.506 | 0.518 | 0.531 | 0.467 | 0.446 | 0.457 |
| EXP-H085 | H054 | F4 ALL | 200 | 520930 | 0.473 | 0.489 | 0.481 | 0.489 | 0.405 | 0.428 | **0.417** |
| EXP-H086 | H061 | F1 SURFACE | 35 | 510370 | 0.471 | 0.501 | 0.486 | 0.501 | 0.402 | 0.411 | **0.406** |
| EXP-H087 | H061 | F2 SEQ+TITR | 133 | 516642 | 0.529 | 0.530 | 0.530 | 0.530 | 0.469 | 0.488 | 0.479 |
| EXP-H088 | H061 | F3 LOCAL_RASA | 32 | 510178 | 0.541 | 0.545 | 0.543 | 0.545 | 0.481 | 0.453 | 0.467 |
| EXP-H089 | H061 | F4 ALL | 200 | 520930 | 0.472 | 0.468 | **0.470** | 0.472 | 0.402 | 0.439 | 0.421 |
| EXP-H090 | H071 | F1 SURFACE | 35 | 349090 | 0.458 | 0.485 | 0.472 | 0.485 | 0.406 | 0.413 | **0.409** |
| EXP-H091 | H071 | F2 SEQ+TITR | 133 | 355362 | 0.523 | 0.519 | 0.521 | 0.523 | 0.461 | 0.493 | 0.477 |
| EXP-H092 | H071 | F3 LOCAL_RASA | 32 | 348898 | 0.538 | 0.516 | 0.527 | 0.538 | 0.475 | 0.448 | 0.462 |
| EXP-H093 | H071 | F4 ALL | 200 | 359650 | 0.498 | 0.506 | 0.502 | 0.506 | 0.407 | 0.421 | 0.414 |

Historical **EXP-H047** (classical RBF-SVR; not protocol-equivalent): Overall ≈ **0.427**.

Permutation: `HIC_H047_FEATURE_PERMUTATION.csv`, `HIC_H047_F4_SUBBLOCK_PERMUTATION.csv`.

## Answers

1. **Physical features improve DL HIC?** Yes for SURFACE and often F4; mixed/negative for F2/F3 alone.
2. **Most consistent family?** **SURFACE (F1)** — improves all three backbones internally and strongly externally.
3. **SURFACE valuable?** Yes. Strongest single-block story; F4 sub-block perm ΔMAE ≈ +0.09 when SURFACE shuffled.
4. **SEQUENCE_TITRATION independent?** Weak alone (often flat/worse TEST); modest positive in F4 sub-perm (Δ≈+0.03–0.05).
5. **LOCAL_RASA help?** No as sole fusion (F3 worse or flat); sub-perm near zero (ignored).
6. **F4 constructive?** Partially. Best internal = H089 (F4); best external = H086 (F1). F4 ≈ F1 on several backbones; not uniformly additive beyond SURFACE.
7. **Which backbone benefits most?** Scratch H071+F1 (H090) largest Primary TEST Δ vs base (bootstrap CI excludes 0). ESM2 H061 also large F1/F4 gains externally.
8. **Scratch + physics vs ESM2 fusion?** Yes competitive: H090 Overall 0.409 vs H086 0.406 / H085 0.417.
9. **Approach H047?** Yes — several DL fusions beat H047 Overall 0.427 (H086 0.406, H090 0.409, H093 0.414, H085 0.417). Caveat: different model class.
10. **Permutation shows USE?** Yes for F1/F4 (primary ALL_AUX mean ΔMAE ≈ +0.13–0.20). F3 near zero. F4 driven mainly by SURFACE.

## Best / near-best cluster

- Best internal TEST_mean: **H089** (~0.470) near **H090** (~0.472), **H085** (~0.481), **H086** (~0.486).
- Best external Overall: **H086** (~0.406) cluster with **H090** (~0.409), **H093** (~0.414), **H085** (~0.417).
- Tiny internal gaps with overlapping CIs → treat as SURFACE/F4 fusion cluster, not a single winner.
