# Gate B6 — Common Public/Private Production Split Search FINAL

Overall decision:

**SELECT_COMMON_PRODUCTION_SPLIT** (`CAND_04974`)

Current split:
    TmApp MAE Public→Private: 0.2000
    TmApp Pearson Public→Private: 0.4000
    HIC MAE Public→Private: 0.7714
    HIC Pearson Public→Private: 0.6000

Selected production split:
    TmApp MAE CV→Public: 0.7000
    TmApp MAE Public→Private: 0.8000
    TmApp MAE CV→Private: 0.8000

    TmApp Pearson CV→Public: 0.8000
    TmApp Pearson Public→Private: 0.8000
    TmApp Pearson CV→Private: 0.6000

    HIC MAE CV→Public: 0.7143
    HIC MAE Public→Private: 0.7714
    HIC MAE CV→Private: 0.2571

    HIC Pearson CV→Public: 0.7000
    HIC Pearson Public→Private: 0.7000
    HIC Pearson CV→Private: 0.2000

TmApp shortlist size: 8
HIC safety pass count: 5
Model-set robustness: LOFO replay retained near-top ranks
Pre-model balance: survivors=2001/20001; selected balance=4.072046058032389
Common split requirement satisfied: YES
Production split freeze recommendation: YES — freeze under frozen/

## Comparison table

| candidate_id           | is_selected   |   tm_mae_cv_public |   tm_mae_public_private |   tm_mae_cv_private |   tm_pear_cv_public |   tm_pear_public_private |   tm_pear_cv_private |   hic_mae_cv_public |   hic_mae_public_private |   hic_mae_cv_private |   hic_pear_cv_public |   hic_pear_public_private |   hic_pear_cv_private |   balance_score |
|:-----------------------|:--------------|-------------------:|------------------------:|--------------------:|--------------------:|-------------------------:|---------------------:|--------------------:|-------------------------:|---------------------:|---------------------:|--------------------------:|----------------------:|----------------:|
| CURRENT_BASELINE_SPLIT | False         |                0   |                     0.2 |                 0.6 |                 0   |                      0.4 |                  0.8 |            0.257143 |                0.771429  |            0.485714  |                  0.3 |                       0.6 |                   0.5 |         6.35987 |
| CAND_11283             | False         |                0.8 |                     0.8 |                 0.7 |                 0.8 |                      1   |                  0.8 |            0.257143 |                0.657143  |            0.6       |                 -0.1 |                       0.6 |                   0.3 |         4.30563 |
| CAND_12207             | False         |                0.7 |                     0.8 |                 0.8 |                 0.8 |                      0.8 |                  0.6 |            0.371429 |                0.485714  |            0.542857  |                  0.3 |                       0.6 |                   0.5 |         3.95375 |
| CAND_04974             | True          |                0.7 |                     0.8 |                 0.8 |                 0.8 |                      0.8 |                  0.6 |            0.714286 |                0.771429  |            0.257143  |                  0.7 |                       0.7 |                   0.2 |         4.07205 |
| CAND_01965             | False         |                0.7 |                     0.8 |                 0.8 |                 0.8 |                      0.8 |                  0.6 |            0.142857 |                0.0857143 |            0.0857143 |                 -0.1 |                       0.8 |                   0.5 |         4.08737 |
| CAND_00144             | False         |                0.7 |                     0.8 |                 0.8 |                 0.8 |                      0.8 |                  0.6 |            0.485714 |                0.828571  |            0.485714  |                  0.6 |                       0.5 |                  -0.1 |         4.15714 |
| CAND_12528             | False         |                0.7 |                     0.8 |                 0.8 |                 0.8 |                      0.8 |                  0.6 |            0.485714 |                0.771429  |            0.6       |                 -0.1 |                       0.6 |                   0.7 |         4.16941 |
| CAND_07084             | False         |                0.8 |                     0.8 |                 0.7 |                 0.6 |                      0.8 |                  0.8 |            0.485714 |                0.771429  |            0.257143  |                 -0.1 |                       0.6 |                   0.7 |         4.20885 |
| CAND_13110             | False         |                0.8 |                     0.8 |                 0.7 |                 0.6 |                      0.8 |                  0.8 |            0.142857 |                0.771429  |            0.6       |                 -0.1 |                       0.8 |                   0.5 |         4.29132 |

## Interpretation boundary

This mask is a **production leaderboard partition** selected with frozen organizer models. It is **not** an untouched scientific holdout. For scientific organizer conclusions, **Train grouped CV** remains the primary model-development evidence; **Private** remains the competition final ranking set.

## Required questions

1. Raw candidate 81/81 partitions generated: **20001** (including CURRENT_BASELINE_SPLIT).
2. Survived pre-model balance filtering: **2001** (incl. baseline).
3. Entered TmApp leaderboard scoring: **2001**.
4. TmApp shortlist size: **8**.
5. TmApp Public→Private MAE model-rank Δ: **+0.6000** (0.200 → 0.800).
6. Pearson-based Public→Private Δ: **+0.4000** (0.400 → 0.800).
7. CV→Public MAE meaningfully positive? **YES** (ρ=0.700; baseline 0.000).
8. CV→Private remained strong? **YES** (ρ=0.800).
9. Bootstrap robustness: median Public→Private MAE ρ=0.400, p05=-0.500, P(ρ<0)=0.221.
10. Family removal robustness: worst LOFO Public→Private=0.600; replay=[{'left_out_family': 'ABLANG2_NONLINEAR', 'selected_rank_among_shortlist': 3, 'shortlist_n': 8, 'selected_still_topk3': True}, {'left_out_family': 'BIO', 'selected_rank_among_shortlist': 2, 'shortlist_n': 8, 'selected_still_topk3': True}, {'left_out_family': 'CONST', 'selected_rank_among_shortlist': 3, 'shortlist_n': 8, 'selected_still_topk3': True}, {'left_out_family': 'NESTED_ENSEMBLE', 'selected_rank_among_shortlist': 3, 'shortlist_n': 8, 'selected_still_topk3': True}, {'left_out_family': 'SEQ_SIMPLE', 'selected_rank_among_shortlist': 3, 'shortlist_n': 8, 'selected_still_topk3': True}].
11. Biologically/statistically balanced? balance_score=4.072 (baseline 6.360; lower better).
12. Variables improved most vs old Pub/Priv: cluster_size: Δ|imb|=+0.412, A2_HL_pI: Δ|imb|=+0.194, A2_HL_charge_ph7: Δ|imb|=+0.164, vl_family: Δ|imb|=+0.034, HIC: Δ|imb|=+0.011, A2_HL_gravy: Δ|imb|=+0.006.
13. Shortlisted candidates passing HIC safety: **5 / 8**.
14. HIC degradation vs current: MAE Public→Private Δ=+0.0000; Pearson Δ=+0.1000; class=HIC_EXCELLENT.
15. HIC still acceptable participant feedback? **YES** (HIC_EXCELLENT).
16. Final common split satisfies both tracks? **YES**.
17. Dependent on one organizer family? **No strong evidence of single-family dependence**.
18. Robust region vs lucky isolates? mean pairwise Public Jaccard among shortlist=0.332 (broad region).
19. Should new mask replace old Public/Private? **YES — recommended for human approval**.
20. After replacement, permanently freeze? **YES** — do not change based on further model results.

_Elapsed: 476.4s_
