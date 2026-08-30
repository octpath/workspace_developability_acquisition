# Final GEN_0001 Pearson Sanity Audit

Split:
    GEN_0001_B_20271100

Seed:
    20271100

Public ID hash:
    2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376

Private ID hash:
    f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0

Public N:
    81

Private N:
    81

Status:
    DIAGNOSTIC ONLY

Primary competition metric:
    MAE

Pearson is used here only as:
    secondary split/model-behavior sanity check

Sanity:
    Public∩Private empty: True
    |Public∪Private|=162: True
    TmApp/HIC identical Public/Private IDs: True
    HIC MEDIUM pub/priv: 3/3
    HIC HIGH pub/priv: 4/3
    CV Pearson: frozen OOF values (max |Δ| vs recomputed = 0.00e+00)
    Public/Private Pearson: frozen full-Test preds + GEN_0001 masks
    No retrain / no retune / no split change


## TmApp Pearson 3×3

| Selected by | Model | CV Pearson | Public Pearson | Private Pearson |
|---|---|---:|---:|---:|
| CV-best | NESTED_STACK_MEAN | 0.5770 | 0.4033 | 0.5268 |
| Public-best | NESTED_STACK_MEAN | 0.5770 | 0.4033 | 0.5268 |
| Private-best | PLM_ABLANG2_PCA32_SVR | 0.5336 | 0.3770 | 0.5521 |

CV-best == Public-best? **True**
CV-best == Private-best? **False**
Public-best == Private-best? **False**

Public-selection Private Pearson loss:
    0.0252

CV-selection Private Pearson loss:
    0.0252

Model-rank transfer:
    CV→Public: 0.8000
    Public→Private: 0.8000
    CV→Private: 0.6000


## HIC Pearson 3×3

| Selected by | Model | CV Pearson | Public Pearson | Private Pearson |
|---|---|---:|---:|---:|
| CV-best | NESTED_STACK_NNLS | 0.5641 | 0.5146 | 0.5453 |
| Public-best | FUSION_ESM2_ESMFN_ElasticNet | 0.4644 | 0.5697 | 0.5082 |
| Private-best | ESMFN_STRUCTURE_ElasticNet | 0.5202 | 0.4914 | 0.6103 |

CV-best == Public-best? **False**
CV-best == Private-best? **False**
Public-best == Private-best? **False**

Public-selection Private Pearson loss:
    0.1021

CV-selection Private Pearson loss:
    0.0649

Model-rank transfer:
    CV→Public: 0.3000
    Public→Private: 0.6000
    CV→Private: 0.5000


## Model-rank transfer summary

| Target | CV→Public model-rank rho | Public→Private model-rank rho | CV→Private model-rank rho |
|---|---:|---:|---:|
| TmApp | 0.8000 | 0.8000 | 0.6000 |
| HIC | 0.3000 | 0.6000 | 0.5000 |


## Compare with MAE result (GEN_0001)

Frozen MAE benchmark (primary):

| Target | Public-selection Private MAE regret | CV-selection Private MAE regret |
|---|---:|---:|
| TmApp | 0.0000 °C | 0.0689 °C |
| HIC | 0.0630 min | 0.0231 min |

Pearson diagnostic (this audit):

| Target | Public-selection Private Pearson loss | CV-selection Private Pearson loss |
|---|---:|---:|
| TmApp | 0.0252 | 0.0252 |
| HIC | 0.1021 | 0.0649 |

Diagnostic Q&A:

1. Does Pearson tell a broadly compatible story with MAE?
    TmApp: YES (broadly) — CV-best is NESTED_STACK_MEAN under both metrics. MAE Public-best==Private-best=PLM_ABLANG2_PCA32_SVR; Pearson Public-best=NESTED_STACK_MEAN while Private-best=PLM_ABLANG2_PCA32_SVR. Public Pearson therefore prefers the CV winner, not the MAE Public winner — a small metric swap, not a split failure. Public-selection Pearson loss remains small (0.0252).
    HIC: YES (directionally) — both metrics show Public-best ≠ Private-best. MAE Public-best=FUSION_ESM2_ESMFN_ElasticNet / Private-best=ESMFN_STRUCTURE_ElasticNet; Pearson winners match that same pair. CV-best stays NESTED_STACK_NNLS under both.

2. Does Public Pearson winner remain reasonably competitive on Private?
    TmApp: YES — Public-selection loss=0.0252 (NESTED Private=0.5268 vs PLM best=0.5521).
    HIC: MIXED — Public-selection loss=0.1021; Public-best Private Pearson=0.5082 vs Private-best=0.6103.

3. Does CV Pearson winner remain reasonably competitive on Private?
    TmApp: YES — CV-selection loss=0.0252 (same as Public-selection here; CV==Public winner).
    HIC: YES — CV-selection loss=0.0649 (NESTED Private=0.5453 vs best=0.6103).

4. Is there any severe Public/Private reversal?
    TmApp: NO — Public→Private model-rank ρ=0.8000; Public≠Private winner but loss is tiny.
    HIC: MILD — Public→Private model-rank ρ=0.6000; winners differ but transfer ρ is not inverted.

5. Is HIC Pearson materially less stable than TmApp Pearson?
    YES — TmApp Public→Private ρ=0.8000 with Public-selection loss=0.0252; HIC Public→Private ρ=0.6000 with Public-selection loss=0.1021. HIC CV→Public ρ=0.3000 vs TmApp 0.8000.

6. Does HIC show evidence consistent with previously known high-tail sensitivity of Pearson?
    YES — HIGH band remains asymmetric (4/3). Structure/fusion-class models swap Public vs Private Pearson leadership (Public favors FUSION_ESM2_ESMFN_ElasticNet; Private favors ESMFN_STRUCTURE_ElasticNet), consistent with Pearson overweighting high-tail residuals relative to MAE.

7. Is there any Pearson-specific pathology serious enough to question GEN_0001_B_20271100?
    NO — discrepancies are modest secondary-metric noise. Primary metric remains MAE; TmApp Pearson is clean; HIC Pearson fragility is known and not newly catastrophic.


## Optional comparison vs CAND_12528 (authoritative frozen B6 scores)

Source: `gate_b6_split_search/metrics/per_model_candidate_scores.csv` (CAND_12528 rows only; not reconstructed).

| Target | Split | CV→Public ρ | Public→Private ρ | CV→Private ρ | Public-selection Private Pearson loss |
|---|---|---:|---:|---:|---:|
| TmApp | CAND_12528 | 0.8000 | 0.8000 | 0.6000 | 0.0831 |
| TmApp | GEN_0001_B_20271100 | 0.8000 | 0.8000 | 0.6000 | 0.0252 |
| HIC | CAND_12528 | -0.1000 | 0.6000 | 0.7000 | 0.0369 |
| HIC | GEN_0001_B_20271100 | 0.3000 | 0.6000 | 0.5000 | 0.1021 |

CAND_12528 Pearson winners (reference):
    TmApp: CV=NESTED_STACK_MEAN; Public=NESTED_STACK_MEAN; Private=PLM_ABLANG2_PCA32_SVR
    HIC: CV=NESTED_STACK_NNLS; Public=ESMFN_STRUCTURE_ElasticNet; Private=NESTED_STACK_NNLS


## Final diagnostic conclusion

Compatible with MAE result:
    YES (broadly) — TmApp: small Public-winner swap (NESTED vs PLM) with tiny Private Pearson loss; HIC: same Public≠Private pair as MAE. No new failure mode.

Any severe Pearson-specific pathology:
    NO

HIC Pearson fragility visible:
    YES (expected; high-tail / band asymmetry; Public→Private transfer weaker than TmApp)

Does this change the production-split recommendation:
    NO CHANGE

Note:
    Diagnostic only. Do not redesign or reselect the split from Pearson.
    Production manifests unchanged.


## Appendix — full GEN_0001 Pearson scores

### TmApp
| model | CV Pearson | Public Pearson | Private Pearson | public_rank | private_rank |
|---|---:|---:|---:|---:|---:|
| NESTED_STACK_MEAN | 0.5770 | 0.4033 | 0.5268 | 1 | 2 |
| PLM_ABLANG2_PCA32_SVR | 0.5336 | 0.3770 | 0.5521 | 2 | 1 |
| SEQ_SIMPLE_Ridge | 0.3151 | 0.2748 | 0.4317 | 3 | 3 |
| BIO_Ridge | 0.4399 | 0.2441 | 0.3652 | 4 | 4 |
| CONST_MEDIAN | -0.3741 | nan | nan | 5 | 5 |

### HIC
| model | CV Pearson | Public Pearson | Private Pearson | public_rank | private_rank |
|---|---:|---:|---:|---:|---:|
| FUSION_ESM2_ESMFN_ElasticNet | 0.4644 | 0.5697 | 0.5082 | 1 | 3 |
| NESTED_STACK_NNLS | 0.5641 | 0.5146 | 0.5453 | 2 | 2 |
| ESMFN_STRUCTURE_ElasticNet | 0.5202 | 0.4914 | 0.6103 | 3 | 1 |
| PLM_ESM2_PCA64_SVR | 0.5290 | 0.4228 | 0.3605 | 4 | 4 |
| SEQ_SIMPLE_Ridge | 0.3834 | 0.3386 | 0.3304 | 5 | 5 |
| CONST_MEDIAN | -0.0693 | nan | nan | 6 | 6 |
