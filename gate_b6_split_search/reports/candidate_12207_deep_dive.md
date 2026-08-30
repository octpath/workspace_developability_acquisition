# Deep dive — CAND_12207

Priority score (lower better): **2.910**

## Transfer (recomputed)
- TmApp MAE PP ρ=0.8000; CV→Pub=0.7000; CV→Priv=0.8000
- HIC MAE PP ρ=0.4857; CV→Pub=0.3714; CV→Priv=0.5429
- HIC Pearson PP ρ=0.6000; CV→Pub=0.3000

## Public-selection regret
- TmApp: winner=TmApp__NESTED_STACK_MEAN, Private rank=2.0, regret=0.2006 °C
- HIC: winner=HIC__FUSION_ESM2_ESMFN_ElasticNet, Private rank=4.0, regret=0.0246

## Strategies
| target   | strategy               | chosen_model                      |   private_rank |   private_mae |     regret |
|:---------|:-----------------------|:----------------------------------|---------------:|--------------:|-----------:|
| TmApp    | A_public_only          | TmApp__NESTED_STACK_MEAN          |              2 |      3.44022  | 0.20057    |
| TmApp    | B_cv_only              | TmApp__NESTED_STACK_MEAN          |              2 |      3.44022  | 0.20057    |
| TmApp    | C_cv_among_public_top2 | TmApp__NESTED_STACK_MEAN          |              2 |      3.44022  | 0.20057    |
| TmApp    | D_consensus            | TmApp__NESTED_STACK_MEAN          |              2 |      3.44022  | 0.20057    |
| HIC      | A_public_only          | HIC__FUSION_ESM2_ESMFN_ElasticNet |              4 |      0.491593 | 0.0246277  |
| HIC      | B_cv_only              | HIC__NESTED_STACK_NNLS            |              2 |      0.472902 | 0.00593678 |
| HIC      | C_cv_among_public_top2 | HIC__NESTED_STACK_NNLS            |              2 |      0.472902 | 0.00593678 |
| HIC      | D_consensus            | HIC__NESTED_STACK_NNLS            |              2 |      0.472902 | 0.00593678 |

## Pairwise MAE inversions
| candidate_id   | target   | metric   |   n_inversions |   n_pairs |   fraction_inverted |   effect_size_weighted_burden |
|:---------------|:---------|:---------|---------------:|----------:|--------------------:|------------------------------:|
| CAND_12207     | TmApp    | MAE      |              2 |        10 |            0.2      |                      0.213161 |
| CAND_12207     | HIC      | MAE      |              5 |        15 |            0.333333 |                      0.144002 |

## Local sensitivity
| candidate_id   | target   | metric            |   n_perturbations |    median |       iqr |      p05 |      p_lt0 |
|:---------------|:---------|:------------------|------------------:|----------:|----------:|---------:|-----------:|
| CAND_12207     | TmApp    | mae_pp            |               300 | 0.8       | 0.2       | 0.5      | 0.00333333 |
| CAND_12207     | TmApp    | pear_pp           |               300 | 0.8       | 0.2       | 0.6      | 0.00333333 |
| CAND_12207     | TmApp    | pub_winner_regret |               300 | 0.193812  | 0.229882  | 0        | 0          |
| CAND_12207     | HIC      | mae_pp            |               300 | 0.571429  | 0.285714  | 0.311429 | 0          |
| CAND_12207     | HIC      | pear_pp           |               300 | 0.6       | 0         | 0.6      | 0          |
| CAND_12207     | HIC      | pub_winner_regret |               300 | 0.0212067 | 0.0193124 | 0        | 0          |

## Ratings
- TmApp Public feedback quality: EXCELLENT
- TmApp Public-selection regret: ACCEPTABLE
- TmApp bootstrap robustness: GOOD
- HIC Public feedback quality: CONCERNING
- HIC Public-selection regret: GOOD
- HIC bootstrap robustness: GOOD
- CV→Private consistency (TmApp MAE): EXCELLENT
- CV→Private consistency (HIC MAE): GOOD
- pre-model statistical balance: EXCELLENT
- local mask stability (TmApp): EXCELLENT
- local mask stability (HIC): GOOD

## Candidate-specific answers
1. Compromise check: TmApp PP=0.8000, HIC PP=0.4857 (B6 floor fail if <0.50).
2. HIC PP≈0.4857 with Public-winner regret=0.0246.
3. Regret vs rank: TmApp regret=0.2006, HIC regret=0.0246.
4. Pre-model imbalance sum=1.5836.
5. Local TmApp MAE-PP median=0.8000, P(ρ<0)=0.0033.
