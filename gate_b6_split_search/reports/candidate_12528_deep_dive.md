# Deep dive — CAND_12528

Priority score (lower better): **-7.187**

## Transfer (recomputed)
- TmApp MAE PP ρ=0.8000; CV→Pub=0.7000; CV→Priv=0.8000
- HIC MAE PP ρ=0.7714; CV→Pub=0.4857; CV→Priv=0.6000
- HIC Pearson PP ρ=0.6000; CV→Pub=-0.1000

## Public-selection regret
- TmApp: winner=TmApp__NESTED_STACK_MEAN, Private rank=2.0, regret=0.1862 °C
- HIC: winner=HIC__ESMFN_STRUCTURE_ElasticNet, Private rank=3.0, regret=0.0155

## Strategies
| target   | strategy               | chosen_model                    |   private_rank |   private_mae |    regret |
|:---------|:-----------------------|:--------------------------------|---------------:|--------------:|----------:|
| TmApp    | A_public_only          | TmApp__NESTED_STACK_MEAN        |              2 |      3.43021  | 0.186239  |
| TmApp    | B_cv_only              | TmApp__NESTED_STACK_MEAN        |              2 |      3.43021  | 0.186239  |
| TmApp    | C_cv_among_public_top2 | TmApp__NESTED_STACK_MEAN        |              2 |      3.43021  | 0.186239  |
| TmApp    | D_consensus            | TmApp__NESTED_STACK_MEAN        |              2 |      3.43021  | 0.186239  |
| HIC      | A_public_only          | HIC__ESMFN_STRUCTURE_ElasticNet |              3 |      0.446758 | 0.0154982 |
| HIC      | B_cv_only              | HIC__NESTED_STACK_NNLS          |              1 |      0.43126  | 0         |
| HIC      | C_cv_among_public_top2 | HIC__NESTED_STACK_NNLS          |              1 |      0.43126  | 0         |
| HIC      | D_consensus            | HIC__NESTED_STACK_NNLS          |              1 |      0.43126  | 0         |

## Pairwise MAE inversions
| candidate_id   | target   | metric   |   n_inversions |   n_pairs |   fraction_inverted |   effect_size_weighted_burden |
|:---------------|:---------|:---------|---------------:|----------:|--------------------:|------------------------------:|
| CAND_12528     | TmApp    | MAE      |              2 |        10 |                 0.2 |                      0.260442 |
| CAND_12528     | HIC      | MAE      |              3 |        15 |                 0.2 |                      0.030533 |

## Local sensitivity
| candidate_id   | target   | metric            |   n_perturbations |    median |        iqr |        p05 |     p_lt0 |
|:---------------|:---------|:------------------|------------------:|----------:|-----------:|-----------:|----------:|
| CAND_12528     | TmApp    | mae_pp            |               300 | 0.8       | 0.2        | 0.2        | 0.0233333 |
| CAND_12528     | TmApp    | pear_pp           |               300 | 0.8       | 0          | 0.6        | 0         |
| CAND_12528     | TmApp    | pub_winner_regret |               300 | 0.17959   | 0.217392   | 0          | 0         |
| CAND_12528     | HIC      | mae_pp            |               300 | 0.771429  | 0.0571429  | 0.485714   | 0         |
| CAND_12528     | HIC      | pear_pp           |               300 | 0.6       | 0.1        | 0.5        | 0         |
| CAND_12528     | HIC      | pub_winner_regret |               300 | 0.0177296 | 0.00888995 | 0.00837518 | 0         |

## Ratings
- TmApp Public feedback quality: EXCELLENT
- TmApp Public-selection regret: ACCEPTABLE
- TmApp bootstrap robustness: ACCEPTABLE
- HIC Public feedback quality: EXCELLENT
- HIC Public-selection regret: EXCELLENT
- HIC bootstrap robustness: GOOD
- CV→Private consistency (TmApp MAE): EXCELLENT
- CV→Private consistency (HIC MAE): GOOD
- pre-model statistical balance: EXCELLENT
- local mask stability (TmApp): EXCELLENT
- local mask stability (HIC): EXCELLENT

## Candidate-specific answers
1. Stronger HIC CV→Private (ρ=0.6000) vs Public-winner regret=0.0155.
2. Negative HIC Pearson CV→Public (ρ=-0.1000) — MAE CV→Public remains 0.4857.
3. Primary participant metric is MAE; negative Pearson transfer is secondary.
4. TmApp regret=0.1862; HIC regret=0.0155.
5. Local TmApp MAE-PP median=0.8000, P(ρ<0)=0.0233.
