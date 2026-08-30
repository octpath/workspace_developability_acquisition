# Phase 2 — Prediction Shrinkage / Center Behavior

## Constant baselines
Train-median constant: CV MAE=0.5171, RMSE=0.8541 (Pearson undefined).

## Best advanced model on CV
- Model: **NESTED_STACK_NNLS**
- MAE=0.4667 vs median 0.5171 → gain=0.0504 (9.8%)
- pred_SD/obs_SD=0.561; pred_IQR/obs_IQR=0.696
- Pearson=0.564; Spearman=0.574

## Artificial shrinkage (Train/CV OOF)
Affine: median + α (p − median). Pearson invariant for α>0.
- ESMFN_STRUCTURE_ElasticNet: MAE-optimal α=0.5, MAE=0.4581
- FUSION_ESM2_ESMFN_ElasticNet: MAE-optimal α=0.5, MAE=0.4573
- NESTED_STACK_NNLS: MAE-optimal α=0.75, MAE=0.4499
- PLM_ESM2_PCA64_SVR: MAE-optimal α=0.5, MAE=0.4592
- SEQ_SIMPLE_Ridge: MAE-optimal α=0.5, MAE=0.4857

## Phase-2 answers
1. Median baseline is **strong** (CV MAE≈0.517).
2. Best advanced gain is **modest** (~0.050 MAE / ~10%).
3. **Yes — predictions under-dispersed** (pred_SD/obs_SD≈0.56).
4. **Yes — α≈0.75 typically improves MAE** vs α=1.
5. **Yes — MAE rewards center compression.**
6. Ordering signal remains (Pearson/Spearman clearly > 0).
7. Continuous HIC is **more than pure median prediction**, but only modestly under MAE.
