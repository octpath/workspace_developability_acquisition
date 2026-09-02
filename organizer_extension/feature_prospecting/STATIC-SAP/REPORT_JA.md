# STATIC-SAP_v1

Paper: [https://doi.org/10.1073/pnas.0904191106](https://doi.org/10.1073/pnas.0904191106)  
Repository: N/A

See master: [PHYSICAL_BATCH1_REPORT_JA.md](../PHYSICAL_BATCH1_REPORT_JA.md)

## Bottom line

| Target | Role | Empirical verdict (ESMFold) |
|--------|------|-----------------------------|
| HIC | PRIMARY | **NO_EVIDENCE_IN_CURRENT_DATA** |
| TmApp | SECONDARY_CROSS_ENDPOINT_AUDIT | **MIXED** |

Robustness: **MODERATE** {'esmfold_vs_abodybuilder2': 0.741882024749678, 'esmfold_vs_boltz2': 0.7323650251823628, 'abodybuilder2_vs_boltz2': 0.7378498809100851}

Standalone/incremental (HIC): TEST_ONLY_POSTHOC / NO_INCREMENT

MAE CV/Public/Private: 0.5236 / 0.5673 / 0.4750  
ΔMAE: +0.0241 / +0.0491 / +0.0039

**STATIC approximation:** not identical to canonical dynamic SAP.
