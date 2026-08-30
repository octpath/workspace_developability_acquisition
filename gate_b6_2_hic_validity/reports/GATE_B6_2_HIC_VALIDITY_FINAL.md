# Gate B6.2 — HIC Continuous-Regression Validity Final Report

**Overall HIC continuous-task verdict:** KEEP_HIC_CONTINUOUS_BUT_IMPROVE_CV_GUIDANCE

Chosen option (**section 46**): **B. KEEP_HIC_CONTINUOUS_BUT_IMPROVE_CV_GUIDANCE**

No new Public/Private split was searched. CONVERT_TO_BINARY / CONVERT_TO_ORDINAL were **not** selected as automatic recommendations.

---

## Overall HIC continuous-task verdict (summary card)

1. Target distribution
    skewness: 1.927
    median: 9.1150
    mean: 9.3945
    SD: 0.8095
    IQR: 0.7198
    central concentration: frac(median±0.5IQR)=0.630; narrowest-80% width=1.208
    upper-tail size: Train n>Q90=17, n>Q95=9

2. Median-predictor competitiveness
    constant median MAE: CV=0.5171, Pub=0.5408, Priv=0.5041
    best advanced MAE: NESTED_STACK_NNLS CV=0.4667
    absolute improvement: 0.0504
    relative improvement: 9.752%

3. Prediction shrinkage
    best model pred_SD / obs_SD: 0.561
    calibration slope: 1.006
    MAE-optimal alpha: 0.75
    MAE gain from shrinkage: 0.0168

4. Tail prediction
    central 80% MAE: 0.3407
    upper 20% MAE: 0.9647
    upper 10% MAE: 1.4421
    advanced vs median gain in tail (upper_10): 0.7985
    systematic tail bias (upper_10 mean pred−true): -1.4421

5. Pearson influence
    largest leave-one-out |Δr|: 0.0527
    top-3 influence (max sum |Δr| across model×role): 0.1327
    model-rank sensitivity: Pearson ranks can change after removing 1–5 influencers (see 04 report)
    Public anomaly explanation: CV↔Public Pearson score-vector r≈0.232; sparse high-HIC leverage drives reordering

6. CV design
    current Group CV stability: fold MAE SD≈0.0730, Pearson SD≈0.1619, tail-count SD≈1.266
    best Train-only CV scheme: SGKF_Q7
    Q5/Q7/Q9/TAIL-Q7 result: {"CURRENT_GROUP_CV": "OK_EXISTING", "SGKF_Q5": "OK", "SGKF_Q7": "OK", "SGKF_Q9": "OK", "TAIL_Q7": "OK"}
    target-balance gain: preferred vs CURRENT (lower better) — see metrics/cv_scheme_stability.csv
    MAE stability gain: preferred fold MAE SD=0.0742 vs CURRENT 0.0730
    Pearson stability gain: preferred fold Pearson SD=0.1226 vs CURRENT 0.1619

Recommended HIC task: KEEP continuous absolute-value regression (KEEP_HIC_CONTINUOUS_BUT_IMPROVE_CV_GUIDANCE)
Recommended participant CV: SGKF_Q7 (sequence_group constrained; HIC quantile strata if SGKF_*)
Primary metric: MAE
Secondary metric: Spearman (Pearson as diagnostic only given leverage sensitivity)

Any reason to abandon CAND_12528: NO for MAE-oriented evaluation; Pearson Public transfer remains fragile by target nature
Any reason to drop HIC: ONLY if organizers require strong Pearson leaderboard stability; scientifically continuous HIC still informative but noisy
Any reason to reconsider TmApp-only competition: Optional if packaging simplicity is paramount; not mandated by this audit

---

## Hypothesis ratings (H1–H4)

- **H1** MAE heavily rewards center-prediction / shrinkage: **STRONG**
- **H2** Pearson dominated by sparse high-HIC observations: **MODERATE**
- **H3** Existing grouped CV has poor HIC target balance: **MODERATE**
- **H4** Continuous HIC contains learnable signal beyond constant/central predictor: **MODERATE**

---

## Answers to required questions (Q1–24)

1. Yes — most mass is central (frac median±0.5IQR=0.630; narrowest 80% width=1.208).
2. Meaningful high-HIC tail ≈ Train n>Q90=17 (~10%) and n>Q95=9 (~5%).
3. Constant-median baseline is strong (CV MAE=0.5171).
4. Best model improves by 0.0504 MAE (9.8%) on CV.
5. Yes — pred_SD/obs_SD for best model ≈ 0.56 (under-dispersed).
6. Yes — Train OOF MAE-optimal α=0.75.
7. Yes — MAE optimum is more compressed than α=1 when α*<1.
8. Yes — nested stack improves upper_10 MAE by 0.7985 vs median (model 1.44 vs median 2.24); central_80 gain is slightly negative (−0.039), so overall MAE gain is largely tail-driven.
9. Yes — upper_10 mean signed error (pred−true) = -1.4421 (underprediction).
10. Upper-10 contributes ~32.4% of total abs error (N=17); upper-20 ~42%.
11. Individual |Δr| up to 0.053 — material influence.
12. Most top influencers are high-HIC (frac of top-1 with Train pct≥90 ≈ 0.93).
13. Yes — removing 1–3 influencers can materially change Pearson and reorder models on Public.
14. Yes — Pearson is intrinsically fragile for this right-skewed sparse-tail target.
15. Spearman is more robust (rank-based; less tail leverage).
16. Yes — current Group CV distributes tails unevenly (fold n>Q90 SD≈1.27).
17. Fold Pearson depends on fold range/SD (example corr vs SD: -0.10).
18. Quantile-stratified Group CV improves stability; preferred=SGKF_Q7.
19. Best Train-only scheme: **SGKF_Q7** (statuses: {'CURRENT_GROUP_CV': 'OK_EXISTING', 'SGKF_Q5': 'OK', 'SGKF_Q7': 'OK', 'SGKF_Q9': 'OK', 'TAIL_Q7': 'OK'}).
20. Sturges (~9) motivates ~7–9 strata count; use quantile strata, not equal-width Sturges bins.
21. CAND_12528 remains acceptable as production split for MAE-first evaluation; no new split search performed.
22. Continuous HIC is a meaningful but difficult competition task (H4=MODERATE); keep continuous, do not auto-convert to classification.
23. TmApp-only would be cleaner for metric stability, but is not required if HIC is retained with improved CV + MAE primary.
24. Binary/ordinal HIC would lose magnitude of retention-time risk (how high), assay-continuous ranking within bins, and absolute error semantics needed for developability assessment.

---

## Scientific use-case check

Models provide partial information about *how high* HIC is (nonzero Pearson/Spearman; MAE gain vs median is largely from pulling high-HIC predictions off the global median). Absolute levels in the sparse high-HIC region remain systematically underpredicted and still dominate residual error. Continuous regression remains the scientifically correct primary task; treat classification only as optional auxiliary diagnostics under separate human review.
