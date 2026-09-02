# VOID-EXPLICIT_v1

Paper: [https://doi.org/10.1186/s12859-021-04519-4](https://doi.org/10.1186/s12859-021-04519-4)  
Repository: https://github.com/LBC-LNBio/pyKVFinder

See master: [PHYSICAL_BATCH1_REPORT_JA.md](../PHYSICAL_BATCH1_REPORT_JA.md)

## Bottom line

| Target | Role | Empirical verdict (ESMFold) |
|--------|------|-----------------------------|
| TmApp | PRIMARY | **MIXED** |
| HIC | SECONDARY_CROSS_ENDPOINT_AUDIT | **NO_EVIDENCE_IN_CURRENT_DATA** |

Robustness: **FRAGILE** {'esmfold_vs_abodybuilder2': 0.34376396813793464, 'esmfold_vs_boltz2': 0.20352928153035318, 'abodybuilder2_vs_boltz2': 0.2464419342281942}

Standalone/incremental (TmApp): WEAK / NO_INCREMENT

MAE CV/Public/Private: 3.4102 / 3.7669 / 3.8100  
ΔMAE: +0.0195 / +0.0087 / +0.0284
