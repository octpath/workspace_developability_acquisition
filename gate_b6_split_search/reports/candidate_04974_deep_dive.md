# Deep dive — CAND_04974

Priority score (lower better): **-5.623**

## Transfer (recomputed)
- TmApp MAE PP ρ=0.8000; CV→Pub=0.7000; CV→Priv=0.8000
- HIC MAE PP ρ=0.7714; CV→Pub=0.7143; CV→Priv=0.2571
- HIC Pearson PP ρ=0.7000; CV→Pub=0.7000

## Public-selection regret
- TmApp: winner=TmApp__NESTED_STACK_MEAN, Private rank=2.0, regret=0.2158 °C
- HIC: winner=HIC__NESTED_STACK_NNLS, Private rank=3.0, regret=0.0235

## Strategies
| target   | strategy               | chosen_model             |   private_rank |   private_mae |    regret |
|:---------|:-----------------------|:-------------------------|---------------:|--------------:|----------:|
| TmApp    | A_public_only          | TmApp__NESTED_STACK_MEAN |              2 |      3.54262  | 0.215781  |
| TmApp    | B_cv_only              | TmApp__NESTED_STACK_MEAN |              2 |      3.54262  | 0.215781  |
| TmApp    | C_cv_among_public_top2 | TmApp__NESTED_STACK_MEAN |              2 |      3.54262  | 0.215781  |
| TmApp    | D_consensus            | TmApp__NESTED_STACK_MEAN |              2 |      3.54262  | 0.215781  |
| HIC      | A_public_only          | HIC__NESTED_STACK_NNLS   |              3 |      0.495434 | 0.0234589 |
| HIC      | B_cv_only              | HIC__NESTED_STACK_NNLS   |              3 |      0.495434 | 0.0234589 |
| HIC      | C_cv_among_public_top2 | HIC__NESTED_STACK_NNLS   |              3 |      0.495434 | 0.0234589 |
| HIC      | D_consensus            | HIC__NESTED_STACK_NNLS   |              3 |      0.495434 | 0.0234589 |

## Pairwise MAE inversions
| candidate_id   | target   | metric   |   n_inversions |   n_pairs |   fraction_inverted |   effect_size_weighted_burden |
|:---------------|:---------|:---------|---------------:|----------:|--------------------:|------------------------------:|
| CAND_04974     | TmApp    | MAE      |              2 |        10 |                 0.2 |                      0.312753 |
| CAND_04974     | HIC      | MAE      |              3 |        15 |                 0.2 |                      0.130328 |

## Local sensitivity
| candidate_id   | target   | metric            |   n_perturbations |    median |        iqr |       p05 |     p_lt0 |
|:---------------|:---------|:------------------|------------------:|----------:|-----------:|----------:|----------:|
| CAND_04974     | TmApp    | mae_pp            |               300 | 0.8       | 0.2        | 0.2       | 0.0166667 |
| CAND_04974     | TmApp    | pear_pp           |               300 | 0.8       | 0          | 0.6       | 0.0166667 |
| CAND_04974     | TmApp    | pub_winner_regret |               300 | 0.204171  | 0.211793   | 0         | 0         |
| CAND_04974     | HIC      | mae_pp            |               300 | 0.771429  | 0.0571429  | 0.485714  | 0         |
| CAND_04974     | HIC      | pear_pp           |               300 | 0.6       | 0.1        | 0.5       | 0         |
| CAND_04974     | HIC      | pub_winner_regret |               300 | 0.0246429 | 0.00674479 | 0.0160037 | 0         |

## Ratings
- TmApp Public feedback quality: EXCELLENT
- TmApp Public-selection regret: ACCEPTABLE
- TmApp bootstrap robustness: ACCEPTABLE
- HIC Public feedback quality: EXCELLENT
- HIC Public-selection regret: GOOD
- HIC bootstrap robustness: GOOD
- CV→Private consistency (TmApp MAE): EXCELLENT
- CV→Private consistency (HIC MAE): ACCEPTABLE
- pre-model statistical balance: EXCELLENT
- local mask stability (TmApp): EXCELLENT
- local mask stability (HIC): EXCELLENT

## Candidate-specific answers
1. Weak HIC CV→Private (ρ=0.2571) — practical Public-winner regret is 0.0235; small/harmless.
2. Effect-size-weighted inversion burden (HIC MAE)=0.1303 over 3/15 inversions.
3. Trusting HIC Public → regret=0.0235.
4. Public top2∩Private top2=1; top3∩top3=3.
5. Local TmApp MAE-PP median=0.8000, P(ρ<0)=0.0167.
