# Phase 3 — Tail vs Center Performance

## Region MAE (NESTED_STACK_NNLS vs median, CV)

| Region | N | Model MAE | Median MAE | Gain (median − model) |
|--------|---|-----------|------------|------------------------|
| central_80 | 129 | 0.341 | 0.302 | **−0.039** (worse) |
| upper_20 | 33 | 0.965 | 1.624 | **+0.659** |
| upper_10 | 17 | 1.442 | 2.241 | **+0.799** |
| upper_5 | 9 | 1.902 | 2.711 | **+0.809** |

Overall CV MAE gain (~0.05) is **largely from the upper tail**, not the dense center.
In the central 80%, the nested stack is slightly worse than a constant median.

## Tail bias
High-HIC antibodies are **systematically underpredicted**:
- upper_10: mean(pred−true) = −1.442 (MAE = 1.442 → essentially always under)
- upper_5: mean(pred−true) = −1.902

## Error burden (NESTED_STACK_NNLS, CV)
- upper 5% → ~22.6% of total abs error
- upper 10% → ~32.4%
- upper 20% → ~42.1%
- worst 10 antibodies → ~24.6%

## Phase-3 answers
1. **Advanced MAE gain is largely tail-driven** (center slightly worse than median; large MAE reduction on high-HIC vs constant median).
2. **Yes — absolute high-HIC errors remain much worse** (upper_10 MAE ≈ 1.44 vs central_80 ≈ 0.34).
3. **Yes — systematic underprediction** of the upper tail.
4. **Yes on MAE vs median** in the upper tail (large gain), but residual magnitude and bias remain severe.
5. Developability-risk utility is **partial**: models move predictions off the global median toward higher values, but still compress extremes.
6. **Yes — overall MAE can hide** that center behavior is near-median while remaining risk-relevant error lives in a thin, underpredicted tail.
