# HIC scientific analysis

Question: is HIC driven by local exposed hydrophobic patches rather than bulk sequence hydrophobicity?

## Univariate Spearman vs `hic_rt_min` (n=348)

```
x                                    spearman  pearson
H_CDR3_len                            0.279    0.278
ABB_h3_hydrophobic_sasa_exposed       0.187    0.251
B_all_CDR_gravy                       0.170    0.174
ABB_BSA                               0.151    0.149
ABB_cdr_hydrophobic_sasa_exposed      0.149    0.228
ABB_Fv_rasa_w_hydrophobicity_sum      0.148    0.129
A2_HL_gravy                           0.135    0.122
ABB_largest_hydrophobic_patch_sasa    0.121    0.179
ABB_Fv_sasa_hydrophobic               0.121    0.118
B_H_CDR3_gravy                        0.120    0.098
```

## Controlled modeling evidence

- BIO_SHORTCUT CV ≪ SEQ_SIMPLE (canonical ~0.24 vs ~0.47) — not a germline-dominated label.
- S2 (absolute surface physchem) > S1 (RASA-only) on nested ablation.
- Native ESMFold surface/all families lead CV and often Private.
- Residual: ESMFN predicts SEQ_SIMPLE / BIO residuals at Spearman ~0.41–0.45.

## Verdict

**Partially supported.** Bulk GRAVY is weak; CDR/H3 length and exposed hydrophobic surface features are stronger univariate correlates; multivariate structure/surface stacks beat sequence gravy and BIO shortcuts. No single patch feature is sufficient alone — exposure × chemistry ensembles matter.
