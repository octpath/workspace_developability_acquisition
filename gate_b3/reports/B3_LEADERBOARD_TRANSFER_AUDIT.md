# B3 Leaderboard Transfer Audit

**Scope:** frozen B3 finalists only. No retraining, no split/model changes.

**Split:** Train=162, Public=81, Private=81 (immutable).

**Caveat:** In the one-shot Public/Private file, HIC `RESID_*`, `ENSEMBLE_NNLS`, and `ENSEMBLE_RANKMEAN` share identical Public/Private scores (proxy fit in `05_finalists_oneshot.py`). TmApp residual/ensemble similarly share scores. Primary correlations use the full frozen finalist rows; a deduped sensitivity line is also reported.


---

# HIC

Frozen finalists N = **12**

## Rank-transfer correlations

- CV → Public ρ = **0.120**
- Public → Private ρ = **-0.113**
- CV → Private ρ = **0.430**

- CV → Public Kendall τ = **0.109**
- Public → Private Kendall τ = **-0.048**
- CV → Private Kendall τ = **0.326**

Deduped by identical Public/Private score tuples (n=10; audit caveat for proxied ensemble/residual oneshot scores):
- CV → Public ρ = 0.345; Public → Private ρ = 0.127; CV → Private ρ = 0.261

Excluding proxied `ENSEMBLE*` / `RESID_*` rows (n=9):
- CV → Public ρ = **0.300**; Public → Private ρ = **0.333**; CV → Private ρ = **0.300**

## Winner transfer

- **CV winner:** `ESMFN_STRUCTURE/ElasticNet` (structure) → Public rank **4.0**, Private rank **1.0**
- **Public winner:** `SEQ_SIMPLE/Ridge_grid` (simple_sequence) → CV rank **11.0**, Private rank **7.0**
- **Private winner:** `ESMFN_STRUCTURE/ElasticNet` (structure) → CV rank **1.0**, Public rank **4.0**

## Top-k overlap

- Top-k CV vs Public (k=3): overlap=0/3, Jaccard=0.000
- Top-k Public vs Private (k=3): overlap=0/3, Jaccard=0.000
- Top-k CV vs Private (k=3): overlap=1/3, Jaccard=0.200
- Top-k CV vs Public (k=2): overlap=0/2, Jaccard=0.000
- Top-k Public vs Private (k=2): overlap=0/2, Jaccard=0.000
- Top-k CV vs Private (k=2): overlap=1/2, Jaccard=0.333

## Table sorted by CV rank

| CV_rank | Public_rank | Private_rank | model_id | model_family | CV_mean_spearman | Public_spearman | Private_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 4.0 | 1.0 | ESMFN_STRUCTURE/ElasticNet | structure | 0.580 | 0.476 | 0.542 |
| 2.0 | 9.0 | 4.0 | ENSEMBLE_NNLS/nonneg_ridge | ensemble | 0.571 | 0.313 | 0.370 |
| 3.0 | 9.0 | 4.0 | ENSEMBLE_RANKMEAN/rank_mean | top_fill | 0.550 | 0.313 | 0.370 |
| 4.0 | 3.0 | 9.0 | PLM_ESM1B_PCA64/Ridge | plm_latent | 0.517 | 0.481 | 0.261 |
| 5.0 | 6.0 | 10.0 | FUSION_ESM2_IMGT/ElasticNet | fusion | 0.516 | 0.381 | 0.252 |
| 6.0 | 2.0 | 6.0 | PLM_ESM1B/ElasticNet | plm_linear | 0.508 | 0.482 | 0.335 |
| 7.0 | 5.0 | 8.0 | PLM_ESM2_rank/Ridge_grid | rank_oriented | 0.473 | 0.422 | 0.271 |
| 8.0 | 9.0 | 4.0 | RESID_PLM_ESM2__ESMFN_STRUCTURE/Ridge_stack | residual | 0.450 | 0.313 | 0.370 |
| 9.0 | 11.0 | 12.0 | IMGT_POS_HL/ElasticNet | imgt_positional | 0.442 | 0.226 | 0.132 |
| 10.0 | 12.0 | 2.0 | GERMLINE_REL/ElasticNet | bio_germline | 0.403 | 0.206 | 0.474 |
| 11.0 | 1.0 | 7.0 | SEQ_SIMPLE/Ridge_grid | simple_sequence | 0.336 | 0.536 | 0.294 |
| 12.0 | 7.0 | 11.0 | NGRAM_HL_3/Ridge_grid | ngram | 0.336 | 0.334 | 0.236 |

## Table sorted by Public rank

| CV_rank | Public_rank | Private_rank | model_id | model_family | CV_mean_spearman | Public_spearman | Private_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11.0 | 1.0 | 7.0 | SEQ_SIMPLE/Ridge_grid | simple_sequence | 0.336 | 0.536 | 0.294 |
| 6.0 | 2.0 | 6.0 | PLM_ESM1B/ElasticNet | plm_linear | 0.508 | 0.482 | 0.335 |
| 4.0 | 3.0 | 9.0 | PLM_ESM1B_PCA64/Ridge | plm_latent | 0.517 | 0.481 | 0.261 |
| 1.0 | 4.0 | 1.0 | ESMFN_STRUCTURE/ElasticNet | structure | 0.580 | 0.476 | 0.542 |
| 7.0 | 5.0 | 8.0 | PLM_ESM2_rank/Ridge_grid | rank_oriented | 0.473 | 0.422 | 0.271 |
| 5.0 | 6.0 | 10.0 | FUSION_ESM2_IMGT/ElasticNet | fusion | 0.516 | 0.381 | 0.252 |
| 12.0 | 7.0 | 11.0 | NGRAM_HL_3/Ridge_grid | ngram | 0.336 | 0.334 | 0.236 |
| 8.0 | 9.0 | 4.0 | RESID_PLM_ESM2__ESMFN_STRUCTURE/Ridge_stack | residual | 0.450 | 0.313 | 0.370 |
| 2.0 | 9.0 | 4.0 | ENSEMBLE_NNLS/nonneg_ridge | ensemble | 0.571 | 0.313 | 0.370 |
| 3.0 | 9.0 | 4.0 | ENSEMBLE_RANKMEAN/rank_mean | top_fill | 0.550 | 0.313 | 0.370 |
| 9.0 | 11.0 | 12.0 | IMGT_POS_HL/ElasticNet | imgt_positional | 0.442 | 0.226 | 0.132 |
| 10.0 | 12.0 | 2.0 | GERMLINE_REL/ElasticNet | bio_germline | 0.403 | 0.206 | 0.474 |

## Bootstrap CI widths (existing; N=81)

- Public Spearman 95% CI width: mean=0.397, median=0.425
- Private Spearman 95% CI width: mean=0.391, median=0.398

| model_id | role | point | CI_lo | CI_hi | width |
|---|---|---:|---:|---:|---:|
| `ESMFN_STRUCTURE/ElasticNet` | Public | 0.476 | 0.292 | 0.640 | 0.347 |
| `ESMFN_STRUCTURE/ElasticNet` | Private | 0.542 | 0.369 | 0.697 | 0.328 |
| `SEQ_SIMPLE/Ridge_grid` | Public | 0.536 | 0.359 | 0.672 | 0.313 |
| `SEQ_SIMPLE/Ridge_grid` | Private | 0.294 | 0.093 | 0.487 | 0.394 |
| `ESMFN_STRUCTURE/ElasticNet` | Public | 0.476 | 0.292 | 0.640 | 0.347 |
| `ESMFN_STRUCTURE/ElasticNet` | Private | 0.542 | 0.369 | 0.697 | 0.328 |

## Interpretation

- Is CV predictive of Private? **Yes** (ρ=0.430 full; 0.300 excluding proxy ensembles)
- Is Public predictive of Private? **Weakly after removing proxy-tied rows** (full ρ=-0.113; clean ρ=0.333). Public alone is a noisy selector (SEQ_SIMPLE wins Public, ranks 7th Private).
- Is Public noisy enough that strong models may be misranked? **Yes** (mean Public CI width=0.397; Top-3 Public∩Private = 0)
- Does frozen 81/81 Public/Private still look acceptable? **Yes** — CV winner = Private winner (`ESMFN_STRUCTURE`); CI width ~0.4 implies rank noise is expected at N=81, but the design remains usable if organizers emphasize uncertainty / do not treat Public #1 as definitive.

**Verdict: `LEADERBOARD_BEHAVIOR_NOISY_BUT_ACCEPTABLE`**

Frozen finalists N = **11**

## Rank-transfer correlations

- CV → Public ρ = **-0.292**
- Public → Private ρ = **-0.633**
- CV → Private ρ = **0.749**

- CV → Public Kendall τ = **-0.093**
- Public → Private Kendall τ = **-0.434**
- CV → Private Kendall τ = **0.574**

Deduped by identical Public/Private score tuples (n=9; audit caveat for proxied ensemble/residual oneshot scores):
- CV → Public ρ = -0.300; Public → Private ρ = -0.633; CV → Private ρ = 0.717

Excluding proxied `ENSEMBLE*` / `RESID_*` rows (n=9):
- CV → Public ρ = **-0.393**; Public → Private ρ = **-0.580**; CV → Private ρ = **0.778**

## Winner transfer

- **CV winner:** `PLM_ABLANG2_PCA64/SVR_RBF` (plm_latent) → Public rank **7.5**, Private rank **2.5**
- **Public winner:** `RESID_BIO_SHORTCUT__PLM_ESM2/Ridge_stack` (residual) → CV rank **6.0**, Private rank **7.5**
- **Private winner:** `PLM_ABLANG2/ElasticNet` (plm_linear) → CV rank **3.0**, Public rank **9.0**

## Top-k overlap

- Top-k CV vs Public (k=3): overlap=0/3, Jaccard=0.000
- Top-k Public vs Private (k=3): overlap=0/3, Jaccard=0.000
- Top-k CV vs Private (k=3): overlap=3/3, Jaccard=1.000
- Top-k CV vs Public (k=2): overlap=0/2, Jaccard=0.000
- Top-k Public vs Private (k=2): overlap=0/2, Jaccard=0.000
- Top-k CV vs Private (k=2): overlap=1/2, Jaccard=0.333

## Table sorted by CV rank

| CV_rank | Public_rank | Private_rank | model_id | model_family | CV_mean_spearman | Public_spearman | Private_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 7.5 | 2.5 | PLM_ABLANG2_PCA64/SVR_RBF | plm_latent | 0.480 | 0.385 | 0.576 |
| 2.0 | 7.5 | 2.5 | PLM_ABLANG2_PCA32/SVR_RBF | top_fill | 0.477 | 0.385 | 0.576 |
| 3.0 | 9.0 | 1.0 | PLM_ABLANG2/ElasticNet | plm_linear | 0.476 | 0.357 | 0.603 |
| 4.0 | 11.0 | 5.0 | IMGT_POS_HL/Ridge_grid | imgt_positional | 0.456 | 0.117 | 0.419 |
| 5.0 | 1.5 | 7.5 | ENSEMBLE_NNLS/nonneg_ridge | ensemble | 0.446 | 0.493 | 0.373 |
| 6.0 | 1.5 | 7.5 | RESID_BIO_SHORTCUT__PLM_ESM2/Ridge_stack | residual | 0.391 | 0.493 | 0.373 |
| 7.0 | 3.0 | 10.0 | SEQ_CDR_gauss/Ridge_grid | rank_oriented | 0.387 | 0.462 | 0.314 |
| 8.0 | 6.0 | 4.0 | FUSION_ESM2_IMGT/Ridge_grid | fusion | 0.362 | 0.392 | 0.467 |
| 9.0 | 10.0 | 6.0 | NGRAM_HL_3/Ridge_grid | ngram | 0.360 | 0.282 | 0.413 |
| 10.0 | 4.0 | 11.0 | ESMFN_STRUCTURE/Ridge_grid | structure | 0.357 | 0.448 | 0.244 |
| 11.0 | 5.0 | 9.0 | SEQ_SIMPLE/ElasticNet | simple_sequence | 0.311 | 0.430 | 0.319 |

## Table sorted by Public rank

| CV_rank | Public_rank | Private_rank | model_id | model_family | CV_mean_spearman | Public_spearman | Private_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6.0 | 1.5 | 7.5 | RESID_BIO_SHORTCUT__PLM_ESM2/Ridge_stack | residual | 0.391 | 0.493 | 0.373 |
| 5.0 | 1.5 | 7.5 | ENSEMBLE_NNLS/nonneg_ridge | ensemble | 0.446 | 0.493 | 0.373 |
| 7.0 | 3.0 | 10.0 | SEQ_CDR_gauss/Ridge_grid | rank_oriented | 0.387 | 0.462 | 0.314 |
| 10.0 | 4.0 | 11.0 | ESMFN_STRUCTURE/Ridge_grid | structure | 0.357 | 0.448 | 0.244 |
| 11.0 | 5.0 | 9.0 | SEQ_SIMPLE/ElasticNet | simple_sequence | 0.311 | 0.430 | 0.319 |
| 8.0 | 6.0 | 4.0 | FUSION_ESM2_IMGT/Ridge_grid | fusion | 0.362 | 0.392 | 0.467 |
| 2.0 | 7.5 | 2.5 | PLM_ABLANG2_PCA32/SVR_RBF | top_fill | 0.477 | 0.385 | 0.576 |
| 1.0 | 7.5 | 2.5 | PLM_ABLANG2_PCA64/SVR_RBF | plm_latent | 0.480 | 0.385 | 0.576 |
| 3.0 | 9.0 | 1.0 | PLM_ABLANG2/ElasticNet | plm_linear | 0.476 | 0.357 | 0.603 |
| 9.0 | 10.0 | 6.0 | NGRAM_HL_3/Ridge_grid | ngram | 0.360 | 0.282 | 0.413 |
| 4.0 | 11.0 | 5.0 | IMGT_POS_HL/Ridge_grid | imgt_positional | 0.456 | 0.117 | 0.419 |

## Bootstrap CI widths (existing; N=81)

- Public Spearman 95% CI width: mean=0.378, median=0.364
- Private Spearman 95% CI width: mean=0.350, median=0.372

| model_id | role | point | CI_lo | CI_hi | width |
|---|---|---:|---:|---:|---:|
| `PLM_ABLANG2_PCA64/SVR_RBF` | Public | 0.385 | 0.170 | 0.560 | 0.390 |
| `PLM_ABLANG2_PCA64/SVR_RBF` | Private | 0.576 | 0.415 | 0.696 | 0.281 |
| `RESID_BIO_SHORTCUT__PLM_ESM2/Ridge_stack` | Public | 0.493 | 0.301 | 0.661 | 0.360 |
| `RESID_BIO_SHORTCUT__PLM_ESM2/Ridge_stack` | Private | 0.373 | 0.168 | 0.540 | 0.372 |
| `PLM_ABLANG2/ElasticNet` | Public | 0.357 | 0.152 | 0.515 | 0.364 |
| `PLM_ABLANG2/ElasticNet` | Private | 0.603 | 0.448 | 0.721 | 0.273 |

## Interpretation

- Is CV predictive of Private? **Yes — strongly** (ρ=0.749 full; 0.778 clean). Top-3 CV vs Private overlap = 3/3.
- Is Public predictive of Private? **No — Public is anti-correlated with Private** (ρ=-0.633 full; -0.580 clean). Public #1 family does not match Private winners (AbLang2).
- Is Public noisy enough that strong models may be misranked? **Yes, and more than noise:** ranking by Public actively reverses Private order among finalists (mean Public CI width=0.378).
- Does frozen 81/81 Public/Private still look acceptable? **Split sizes yes; Public-as-proxy-for-Private no.** Do **not** redesign membership from this audit, but treat the TmApp Public board as a noisy / potentially misleading live ranking and rely on Private + Train-CV for organizer conclusions.

**Verdict: `LEADERBOARD_BEHAVIOR_CONCERNING`** (specifically Public→Private rank transfer; CV→Private remains healthy)

---

## Summary verdicts

- **HIC:** `LEADERBOARD_BEHAVIOR_NOISY_BUT_ACCEPTABLE`
- **TmApp:** `LEADERBOARD_BEHAVIOR_CONCERNING`

No split redesign recommended from this audit alone. For TmApp, communicate Public uncertainty; do not use Public ranks to pick the “true” winner.
