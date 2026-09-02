# POLAR-SAT_v1

Paper: [https://doi.org/10.1371/journal.pcbi.1008061](https://doi.org/10.1371/journal.pcbi.1008061)  
Repository: https://github.com/Electrostatics/pdb2pqr

See master: [PHYSICAL_BATCH1_REPORT_JA.md](../PHYSICAL_BATCH1_REPORT_JA.md)

## Bottom line

| Target | Role | Empirical verdict (ESMFold) |
|--------|------|-----------------------------|
| TmApp | PRIMARY | **PROMISING_BUT_REDUNDANT** |
| HIC | SECONDARY_CROSS_ENDPOINT_AUDIT | **NO_EVIDENCE_IN_CURRENT_DATA** |

Robustness: **FRAGILE** {'esmfold_vs_abodybuilder2': 0.31091068073119194, 'esmfold_vs_boltz2': 0.5007599189603426, 'abodybuilder2_vs_boltz2': 0.42976105087698563}

Standalone/incremental (TmApp): REPRODUCIBLE / NO_INCREMENT

MAE CV/Public/Private: 3.2351 / 3.7573 / 3.4320  
ΔMAE: +0.0261 / +0.0752 / -0.1081
