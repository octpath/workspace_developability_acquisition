# AROMATIC-TOPO_v1

Paper: [https://doi.org/10.1080/19420862.2020.1743053](https://doi.org/10.1080/19420862.2020.1743053)  
Repository: N/A

See master: [PHYSICAL_BATCH1_REPORT_JA.md](../PHYSICAL_BATCH1_REPORT_JA.md)

## Bottom line

| Target | Role | Empirical verdict (ESMFold) |
|--------|------|-----------------------------|
| HIC | PRIMARY | **PROMISING_BUT_REDUNDANT** |
| TmApp | SECONDARY_CROSS_ENDPOINT_AUDIT | **MIXED** |

Robustness: **MODERATE** {'esmfold_vs_abodybuilder2': 0.682417859631482, 'esmfold_vs_boltz2': 0.6410950916436607, 'abodybuilder2_vs_boltz2': 0.6479227141078466}

Standalone/incremental (HIC): REPRODUCIBLE / NO_INCREMENT

MAE CV/Public/Private: 0.5038 / 0.4919 / 0.4661  
ΔMAE: +0.0238 / +0.0230 / +0.0065
