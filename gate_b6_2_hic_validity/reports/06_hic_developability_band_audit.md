# Gate B6.3 — HIC Developability-Band Utility Audit

HIC continuous-task developability verdict:

**KEEP_HIC_CONTINUOUS_WITH_CAVEATS**

Literature-informed bands:
- LOW: HIC < 10.5 min
- MEDIUM: 10.5 ≤ HIC ≤ 11.5 min
- HIGH: HIC > 11.5 min

(These are Jain/Shehata assay-context interpretation aids — **not** universal clinical cutoffs and **not** replacements for continuous HIC RT. Diagnostic only.)

True-band counts:
- Train: LOW=144, MEDIUM=12, HIGH=6
- Public: LOW=73, MEDIUM=5, HIGH=3
- Private: LOW=76, MEDIUM=1, HIGH=4

NESTED_STACK_NNLS:
- overall MAE: Train OOF=0.4667, Public=0.5135, Private=0.4313
- HIGH-band MAE: Train=2.0172, Public=3.0715, Private=1.6053
- true HIGH:
  - Train: 6
  - Public: 3
  - Private: 4
- HIGH→LOW:
  - Train: 4 (rate=0.667)
  - Public: 3 (rate=1.000)
  - Private: 2 (rate=0.500)
- HIGH predicted MEDIUM/HIGH:
  - Train: 0.333
  - Public: 0.000
  - Private: 0.500
- MEDIUM+HIGH predicted LOW:
  - Train: 0.833
  - Public: 1.000
  - Private: 0.600

CONST_MEDIAN comparison:
- CONST never predicts ≥10.5 (Train median ≈ 9.115) → elevated recognition = 0 on all roles.
- NESTED elevated (≥10.5) recognition: CV=0.167, Pub=0.000, Priv=0.400.
- Sequence models do move *some* elevated antibodies above 10.5 vs a constant median, but most elevated antibodies remain predicted LOW, especially on Public.

Best advanced model for HIGH recognition: **ESMFN_STRUCTURE_ElasticNet** (CV MEDIUM|HIGH rate=0.500)

Best advanced model for HIGH-band MAE: **PLM_ESM2_PCA64_SVR** (CV HIGH MAE=1.972)

Does Public Pearson weakness correspond to dangerous HIGH→LOW errors:
**Yes, partly.** Public is not only Pearson-fragile: Nested HIGH→LOW rate=1.00 and elevated≥10.5 recognition=0.00 on Public (worse than Private). Sparse HIGH still makes rates noisy.

Does continuous HIC provide useful developability information:
**Partially yes vs CONST_MEDIAN** (nonzero elevated recognition on CV/Private; continuous ranking signal), but **developability-band recognition remains weak** — most true MEDIUM/HIGH stay predicted LOW, and HIGH-band bias is strongly negative.

Does CAND_12528 remain acceptable:
**YES** for MAE-first continuous evaluation. Sparse HIGH is a target-property issue, not a reason to reopen split search from this band audit alone.

Final recommendation:
**KEEP_HIC_CONTINUOUS_WITH_CAVEATS** — keep continuous HIC as the competition task; use 10.5/11.5 only as interpretation / organizer diagnostics; do **not** auto-convert to classification.

---

## Band sample-size caveat

MEDIUM/HIGH are sparse. Rates involving HIGH are descriptive and unstable.

## Statistical tail vs literature bands

Train n>Q90=17 vs n≥10.5=18 (AND=17, XOR=1); n>Q95=9 vs n>11.5=6 (AND=6, XOR=3). Q90=10.5372 ≈ 10.5; Q95=11.1356 < 11.5 so literature HIGH is stricter than statistical top 5%.

## Model comparison (Train OOF)

| model | MAE | Spearman | HIGH→LOW | HIGH≥10.5 | elev≥10.5 | HIGH MAE | HIGH bias |
|-------|-----|----------|----------|-----------|-----------|----------|-----------|
| NESTED_STACK_NNLS | 0.467 | 0.574 | 0.67 | 0.33 | 0.17 | 2.02 | -2.02 |
| PLM_ESM2_PCA64_SVR | 0.490 | 0.542 | 0.67 | 0.33 | 0.17 | 1.97 | -1.97 |
| ESMFN_STRUCTURE_ElasticNet | 0.497 | 0.523 | 0.50 | 0.50 | 0.28 | 2.04 | -2.04 |
| FUSION_ESM2_ESMFN_ElasticNet | 0.521 | 0.496 | 0.83 | 0.17 | 0.17 | 2.08 | -2.08 |
| SEQ_SIMPLE_Ridge | 0.565 | 0.363 | 0.83 | 0.17 | 0.17 | 2.23 | -2.23 |

CONST_MEDIAN CV: MAE=0.518, elev recognition=0, HIGH→LOW rate=1.00 (always LOW).

## MAE ranking vs developability ranking

See `metrics/hic_mae_vs_developability_ranks.csv`. Overall-MAE winners need not be the safest HIGH→LOW avoiders.

## Public vs Private (advanced)

| model | Pub HIGH→LOW | Priv HIGH→LOW | Pub elev≥10.5 | Priv elev≥10.5 | Pub HIGH MAE | Priv HIGH MAE |
|-------|--------------|---------------|---------------|----------------|--------------|---------------|
| SEQ_SIMPLE_Ridge | 1.00 | 1.00 | 0.00 | 0.00 | 2.92 | 1.99 |
| PLM_ESM2_PCA64_SVR | 1.00 | 0.75 | 0.00 | 0.20 | 3.33 | 1.58 |
| ESMFN_STRUCTURE_ElasticNet | 1.00 | 0.50 | 0.00 | 0.40 | 2.80 | 1.63 |
| FUSION_ESM2_ESMFN_ElasticNet | 0.67 | 0.75 | 0.12 | 0.20 | 2.82 | 1.53 |
| NESTED_STACK_NNLS | 1.00 | 0.50 | 0.00 | 0.40 | 3.07 | 1.61 |

## Consensus severe failures (true HIGH & ≥3 advanced models predict LOW)

n=10 — IDs: ADI-45486, ADI-47126, ADI-45498, ADI-45499, ADI-47163, ADI-45433, ADI-47265, ADI-47162, ADI-47161, ADI-47160

Full table: `metrics/hic_consensus_failures.csv` (includes VH/VL family when available). No new feature discovery was run.

## Answers (Q1–17)

1. Counts as in header (Train/Public/Private LOW/MEDIUM/HIGH).
2. Train Q90≈10.537 ≈ literature 10.5 (MEDIUM+HIGH gateway); Train Q95≈11.136 < 11.5 → literature HIGH is a stricter subset of the statistical upper tail.
3. true HIGH: Train=6, Public=3, Private=4.
4. NESTED HIGH→LOW counts: Train=4, Public=3, Private=2.
5. NESTED HIGH→MEDIUM|HIGH rates: Train=0.333, Public=0.000, Private=0.500.
6. MEDIUM+HIGH→LOW rates: Train=0.833, Public=1.000, Private=0.600.
7. Yes — NESTED (and other advanced models) beat CONST_MEDIAN on elevated recognition (CONST=0 by construction).
8. Best HIGH recognition (CV): ESMFN_STRUCTURE_ElasticNet.
9. Lowest HIGH→LOW (CV): ESMFN_STRUCTURE_ElasticNet.
10. Lowest HIGH-band MAE (CV): PLM_ESM2_PCA64_SVR.
11. Same as overall MAE winner? overall=NESTED_STACK_NNLS; HIGH-recog=ESMFN_STRUCTURE_ElasticNet; HIGH-MAE=PLM_ESM2_PCA64_SVR; HIGH→LOW=ESMFN_STRUCTURE_ElasticNet — not fully aligned.
12. Yes — HIGH-band mean bias (pred−true) remains strongly negative even among models that sometimes clear 10.5.
13. Public shows worse HIGH→LOW / elevated recognition than Private — not only Pearson fragility.
14. CAND_12528 remains acceptable; no new split search indicated by this band audit.
15. Yes — continuous models provide elevated-HIC directional signal beyond the median predictor, with residual undershoot.
16. Yes — categorical conversion would discard magnitude (e.g. 12.0→11.4 vs 14.0→10.6 both collapse as HIGH→MEDIUM).
17. Yes — keep HIC as a **continuous** competition track; do not auto-convert to binary/ordinal.

## Plots / metrics

- `plots/hic_band_audit/nested_stack_cv_public_private_bands.png`
- `plots/hic_band_audit/nested_stack_cv_public_private_bands_labeled.png`
- `plots/hic_band_audit/esmfold_structure_cv_public_private_bands.png`
- `plots/hic_band_audit/esm2_svr_cv_public_private_bands.png`
- `plots/hic_band_audit/fusion_cv_public_private_bands.png`
- `plots/hic_band_audit/nested_stack_band_confusion.png`
- `metrics/hic_band_counts.csv`
- `metrics/hic_band_tail_overlap.csv`
- `metrics/hic_band_confusion.csv`
- `metrics/hic_band_errors.csv`
- `metrics/hic_developability_model_comparison.csv`
- `metrics/hic_elevated_antibody_predictions.csv`
- `metrics/hic_consensus_failures.csv`
