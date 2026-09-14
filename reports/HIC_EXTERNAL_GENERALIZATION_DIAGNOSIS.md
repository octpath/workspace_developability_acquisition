# HIC External Generalization Diagnosis

**STATUS: EXTERNAL_DIAGNOSIS_COMPLETE_NOT_YET_SCIENTIFICALLY_FROZEN**

**PUBLIC_PRIVATE_AUTHORIZED_AFTER_INTERNAL_FREEZE**

## Provenance

| Field | Value |
|-------|--------|
| Internal factorial freeze | `cca9bd5016ecd14f06727f285bd26bf58fc46647` |
| Scientific conclusion freeze (DO NOT EDIT) | `559980e37576eaffaeba9603cee591544e52d991` / `reports/HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md` |
| Diagnosis start HEAD | `559980e37576eaffaeba9603cee591544e52d991` |
| Evaluation asset | `top_models_feature_bundle/solution.csv` via `_lib.load_solution` + `_lib.mae` |
| Split | solution `is_public`/`is_private` (81/81); overall = MAE on all 162 test ids |
| New training | **None** |
| Predictions | existing `experiments/predictions/EXP-Hxxx/test.csv` only |
| experiments.csv overwritten | **No** |

### Evidence provenance

- **Prospective relative to this factorial freeze:** EXP-H140–H339 (internal conclusions frozen before Pub/Priv).
- **Retrospective historical evidence:** H001–H139 and other prior Test-scored rows (Test may already have informed project history). Not a pristine holdout.

## 1. Pipeline validation

**PASS** — max |Δ| vs registry on anchors = 2.989e-08 (tol=1e-06).

Anchors: EXP-H030, EXP-H047, EXP-H107, EXP-H109, EXP-H113, EXP-H137

See `reports/HIC_EXTERNAL_PIPELINE_VALIDATION.csv`.

## 2. Factorial 200 external overview

- External best cell: **EXP-H185** (AbLang1 × JOINT × IMGT) Test Overall = **0.441834** (Public=0.429583, Private=0.454084)
- External mean / median Test Overall: 0.480282 / 0.480943
- Best average representation (Test Overall mean): **AbLang1**
- EXP-H266 external: Test=0.466027, rank=26/200 (internal rank 1)

CSV: `reports/HIC_FACTORIAL_EXTERNAL_DIAGNOSIS.csv`

## 3. Internal → external rank transfer

- Pearson(cv_mean, test) = **0.1150**
- Spearman = **0.1268**
- Kendall τ = **0.0850**
- mean |rank change| = 61.89
- Top10 both = []
- median (Test − CV mean) = -0.0503
- mean |P−S| = 0.0269; mean |Pub−Priv| = 0.0173

## 4. Representation external validation

| Rep | int mean | ext mean | int rank | ext rank | Δrank | ext best |
|-----|----------|----------|----------|----------|-------|----------|
| AbLang1 | 0.5337 | 0.4723 | 6 | 1 | -5 | 0.4418 |
| AbLang2 SEPARATE | 0.5427 | 0.4758 | 9 | 2 | -7 | 0.4479 |
| AbLingua | 0.5180 | 0.4762 | 1 | 3 | +2 | 0.4521 |
| ESM-2 | 0.5235 | 0.4773 | 2 | 4 | +2 | 0.4660 |
| CurrAb PAIRED | 0.5414 | 0.4774 | 8 | 5 | -3 | 0.4574 |
| ESM-1b | 0.5332 | 0.4783 | 5 | 6 | +1 | 0.4576 |
| AbLang2 PAIRED | 0.5447 | 0.4804 | 10 | 7 | -3 | 0.4561 |
| ESM-C | 0.5255 | 0.4826 | 4 | 8 | +4 | 0.4619 |
| CurrAb SEPARATE | 0.5392 | 0.4900 | 7 | 9 | +2 | 0.4635 |
| Scratch | 0.5255 | 0.4925 | 3 | 10 | +7 | 0.4673 |

- **R1** AbLingua best average: **WEAKENED**
- **R2** ESM-2 best-cell ≠ average-best: **INCONCLUSIVE** (ext best cell family=ablang1; esm2 ext avg rank=4)
- **R3** several Ab-PLMs worse than Scratch on average: **CONTRADICTED** (0/5 Ab-PLMs worse than Scratch externally)

## 5. Topology (matched Rep×Annot, Test Overall)

| Contrast | mean Δ | median Δ | improve frac |
|----------|--------|----------|--------------|
| JOINT-SEP | -0.00439 | -0.00354 | 0.600 |
| REG-SEP-SEP | 0.00177 | 0.00185 | 0.475 |
| XREG-SEP | 0.00082 | 0.00164 | 0.450 |
| FUSE-SEP | 0.00277 | 0.00564 | 0.400 |

**Internal claim (small effects / no strong winner): SUPPORTED**

## 6. Annotation (matched Rep×Topo)

| Contrast | mean Δ | median Δ | improve frac |
|----------|--------|----------|--------------|
| IMGT-BASE | 0.00308 | 0.00612 | 0.340 |
| REGION-BASE | -0.00104 | -0.00302 | 0.580 |
| FULL-BASE | -0.00395 | -0.00075 | 0.520 |

**Internal claim (FULL slight+/IMGT slight− / rep-dependent): SUPPORTED**

## 7. Scratch vs PLM (external)

| Rep | ΔCV vs Scratch | ΔTest vs Scratch | ext improve frac |
|-----|----------------|------------------|------------------|
| AbLang1 | +0.0082 | -0.0203 | 1.00 |
| AbLang2 SEPARATE | +0.0172 | -0.0167 | 1.00 |
| AbLingua | -0.0075 | -0.0163 | 0.90 |
| ESM-2 | -0.0020 | -0.0152 | 0.85 |
| CurrAb PAIRED | +0.0160 | -0.0151 | 0.85 |
| ESM-1b | +0.0077 | -0.0142 | 0.85 |
| AbLang2 PAIRED | +0.0192 | -0.0121 | 0.70 |
| ESM-C | +0.0001 | -0.0099 | 0.65 |
| CurrAb SEPARATE | +0.0138 | -0.0025 | 0.50 |

**Scratch competitiveness claim: CONTRADICTED**

## 8. Context (external additive; internal freeze not rewritten)

- **ablang2** PAIRED−SEPARATE Test: mean Δ=+0.00463, improve frac=0.40 (n=20). external additive only; does not rewrite internal no-robust-advantage conclusion
- **currab** PAIRED−SEPARATE Test: mean Δ=-0.01262, improve frac=0.70 (n=20). external additive only; does not rewrite internal no-robust-advantage conclusion

## 9. HIGH-tail (Test, HIC>11.5)

- n_high=[7]
- MAE_high mean/min/max = 2.8741 / 2.3066 / 3.1737
- mean signed error = -2.8741
- best/worst HIGH cells: EXP-H185 / EXP-H277
- **SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE** — verdict vs internal: **SUPPORTED**

## 10. All-history comparison (retrospective + prospective)

- Test-scored HIC rows in combined view: 339
- Best all-history Test Overall: **EXP-H137** (TRF_HIC_ESM2_GLOBAL_F1_FILM_V3) = 0.390740 [provenance=retrospective_historical, surfaceish=True]
- Best balanced robust CV+Test candidate (class A, min CV+Test): **EXP-H107** CV=0.4476 Test=0.3936 [provenance=retrospective_historical, surfaceish=True]
- Class counts: A=4, B=7, C=8

Artifacts: `HIC_ALL_HISTORY_TEST_TOP30.csv`, `HIC_ALL_HISTORY_CV_TOP30.csv`, `HIC_ALL_HISTORY_PARETO_FRONT.csv`, `HIC_ALL_HISTORY_CANDIDATE_CLASSES.csv`.

## 11. Surface hypothesis (careful)

- Factorial sequence-only best Test = 0.4418; mean = 0.4803
- Historical surfaceish best Test = 0.3907 (n=68)
- Surface stronger on Test vs factorial best by ≳0.02: **True**
- Top surface Test cells also stronger on CV than top factorial Test cells: **True**

Interpretation: if surfaceish models dominate Test, evidence is consistent with **external data being substantially more consistent with models containing explicit surface/physicochemical information** — not a causal proof from this diagnosis alone. Much of that surface evidence is **retrospective** relative to prior Test access.

## 12. Claim status summary (vs internal scientific freeze)

- `R1_AbLingua_best_average`: **WEAKENED**
- `R2_ESM2_best_cell_not_avg`: **INCONCLUSIVE**
- `R3_several_AbPLM_worse_than_Scratch`: **CONTRADICTED**
- `topology_effects_small_no_strong_winner`: **SUPPORTED**
- `annotation_small_rep_dependent`: **SUPPORTED**
- `scratch_surprisingly_competitive`: **CONTRADICTED**
- `ablang2_no_robust_context_advantage_internal`: **NOT_REWRITTEN_external_additive_only**
- `currab_no_robust_context_advantage_internal`: **NOT_REWRITTEN_external_additive_only**
- `high_tail_unresolved`: **SUPPORTED**
- `sequence_only_plateau_near_0_50`: **WEAKENED**

## 13. Explicit answers

1. H266 external strength: rank 26/200, Test=0.4660.
2. AbLingua average-best: WEAKENED.
3. ESM-2 pattern: INCONCLUSIVE.
4. Scratch strong: CONTRADICTED.
5. Topology small: SUPPORTED.
6. Annotation pattern: SUPPORTED.
7. HIGH-tail: SUPPORTED.
8. Sequence-only plateau: WEAKENED (best Test=0.4418, median=0.4809).
9. Surface clearer externally than factorial seq-only: True.
10. That gap also CV-supported (top cells): True.

## 14. Final diagnosis summary (not a scientific freeze)

1. Evaluation pipeline validation: **PASS** (max |Δ|=2.989e-08).
2. Factorial 200-cell external best: **EXP-H185** (AbLang1 × JOINT × IMGT) Test=0.441834.
3. Factorial external best average representation: **AbLang1**.
4. Internal→external rank correlation: Pearson=0.115, Spearman=0.127, Kendall=0.085; Top10∩=[].
5. Topology conclusion: **SUPPORTED**.
6. Annotation conclusion: **SUPPORTED**.
7. Scratch conclusion: **CONTRADICTED**.
8. AbLang2 context: external PAIRED−SEPARATE mean Δ=+0.0046 (improve frac=0.40); internal no-robust-advantage **not rewritten**.
9. CurrAb context: external PAIRED−SEPARATE mean Δ=-0.0126 (improve frac=0.70); internal no-robust-advantage **not rewritten**.
10. HIGH-tail conclusion: **SUPPORTED** (SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE).
11. Sequence-only plateau conclusion: **WEAKENED** (best=0.4418, median=0.4809; historical surface best≈0.3907).
12. Best all-history external model: **EXP-H137** Test=0.390740 (retrospective_historical).
13. Best robust CV+external candidate: **EXP-H107** CV=0.4476 Test=0.3936.
14. Surface hypothesis status: external data are substantially more consistent with models containing explicit surface/physicochemical information (retrospective + matched CV support for top surface cells); **not** a causal proof.
15. Internal Freeze claim outcomes:
    - `R1_AbLingua_best_average` → **WEAKENED**
    - `R2_ESM2_best_cell_not_avg` → **INCONCLUSIVE**
    - `R3_several_AbPLM_worse_than_Scratch` → **CONTRADICTED**
    - `topology_effects_small_no_strong_winner` → **SUPPORTED**
    - `annotation_small_rep_dependent` → **SUPPORTED**
    - `scratch_surprisingly_competitive` → **CONTRADICTED**
    - `ablang2_no_robust_context_advantage_internal` → **NOT_REWRITTEN_external_additive_only**
    - `currab_no_robust_context_advantage_internal` → **NOT_REWRITTEN_external_additive_only**
    - `high_tail_unresolved` → **SUPPORTED**
    - `sequence_only_plateau_near_0_50` → **WEAKENED**
16. Unresolved: surface causality vs confounders; calibration of factorial cells; Pub/Priv outlier mechanisms; Freeze v2 synthesis pending human+ChatGPT review.

## 15. Unresolved (detail)

- Causal role of surface features vs confounders in historical surface experiments
- Whether any factorial cell would become competitive after calibration (not tested)
- Private-specific failures within Pub/Priv gap outliers
- Final Freeze v2 synthesis (awaits human + ChatGPT review)

## 16. Non-actions

- No new scientific freeze created
- Internal scientific freeze file untouched
- No `experiments.csv` external overwrite for H140–H339

