# GATE B5 CEILING — FINAL

Overall ceiling-check conclusion: **Proceed to packaging**. B4 Level-2 ensemble CV was **optimistic**; corrected nested stacks still help modestly. Learned pooling / light FT did not overturn the B4 family ranking under rigorous MAE. Keep **MAE primary**, **Pearson r secondary**. Frozen split unchanged.

B4 ensemble CV integrity:
    HIC: **OPTIMISTIC** (NNLS/Ridge fit+scored on same OOF meta-rows). Corrected nested NNLS MAE≈0.4731 vs B4≈0.4691 (Δ≈0.0040)
    TmApp: **OPTIMISTIC**. Corrected nested Ridge MAE≈2.8689 vs B4≈2.6148 (Δ≈0.2541)

HIC:
    best rigorous Train-CV MAE: 0.4731 (NESTED_STACK_NNLS)
    best rigorous Train-CV Pearson: 0.5601 (NESTED_STACK_NNLS)
    best single model: 0.4989 (PLM_ESM2/PCA64/SVR)
    best ensemble: 0.4731 (NESTED_STACK_NNLS)
    learned pooling gain: **none** (best pool MAE≈0.55 > structure/PLM singles ≈0.50; failed to beat frozen baselines)
    fine-tuning gain: **none / unstable** (LoRA CV MAE≈0.52≈constant; oneshot skipped — no frozen checkpoint)
    remaining headroom: **SMALL**

TmApp:
    best rigorous Train-CV MAE: 2.8493 (NESTED_STACK_MEAN)
    best rigorous Train-CV Pearson: 0.5445 (NESTED_STACK_MEAN)
    best single model: 2.9727 (PLM_ABLANG2/PCA32/SVR)
    best ensemble: 2.8493 (NESTED_STACK_MEAN)
    learned pooling gain: **none** (best pool MAE≈4.22 ≫ BIO/fusion ≈3.2; not competitive)
    fine-tuning gain: **FAILED / unstable** (LoRA CV MAE≈40 °C — training collapse; oneshot skipped)
    remaining headroom: **SMALL–MODERATE** (beyond BIO: 3.23 → best nested 2.85 ≈ Δ0.38 °C)

Recommended primary metric:
    HIC: **MAE (minutes)**
    TmApp: **MAE (°C)**

Recommended secondary metric:
    Pearson r

Frozen split status: **UNCHANGED** (162/81/81)
Competition packaging recommendation: **YES — proceed to packaging Gate** (no further organizer modeling Gate required unless packaging needs change)

Finalist registry SHA256: `7d5513badd254fd2629ee1c489b159225ecbea9b1f41056115493899e8997382`

---

## Ensemble integrity (Q1–6)

1. Was B4 OOF ensemble CV unbiased? **NO** — Level-2 stacker in-sample on OOF meta-rows.
2. How optimistic? HIC ΔMAE(corrected−B4)≈0.0040; TmApp Δ≈0.2541 (positive ⇒ B4 too low MAE).
3. Corrected HIC nested-ensemble MAE: **0.4731**
4. Corrected TmApp nested-ensemble MAE: **2.8689**
5. Corrected nested-ensemble Pearson (agg): HIC≈0.5640973216570419; TmApp≈0.5611622410399018
6. Ensemble still beat best single? See nested vs single in `ceiling_summary_table.csv` / structure_plm / bio_plm reports.

## Pearson (Q7–16)

7–9. Best HIC Train-CV Pearson: **0.5601** (NESTED_STACK_NNLS); MAE=0.4731
10–12. Best TmApp Train-CV Pearson: **0.5445** (NESTED_STACK_MEAN); MAE=2.8493
13. Best-MAE vs best-Pearson same model? HIC: YES; TmApp: YES
14. Pearson improvements credible? See bootstrap CIs in `pearson_confidence_intervals.csv` and holdout CIs below.
15. Mean-shrunk despite good Pearson? Check `calibration_metrics.csv` (pred_sd vs obs_sd; slope>1).
16. Calibration improve MAE? Linear OOF calibration often helps MAE with little Pearson change; see B4/B5 calibration reports.

## HIC ceiling (Q17–25)

17–19. Learned pooling / FT vs frozen PLM / structure: see `learned_pooling_results.md`, `light_finetuning_results.md`.
20–22. Nested PLM+structure: see `structure_plm_nested_stack.md` + residual complementarity.
23–24. Best practical organizer MAE/Pearson: above opening summary.
25. Remaining HIC headroom: **SMALL**

## TmApp ceiling (Q26–34)

26–30. Pooling / FT / nested ensemble vs AbLang2+BIO: see reports; BIO→advanced still the main story.
31–32. Best practical MAE/Pearson: opening summary.
33. Headroom beyond distributed BIO: Δ≈0.3774 °C
34. Remaining TmApp headroom: **SMALL**

## Holdout (Q35–40)

### Finalist Public/Private
target                           tag   cv_mae  public_mae  private_mae  cv_pearson  public_pearson  private_pearson
   HIC             NESTED_STACK_NNLS 0.473053    0.474157     0.470560    0.560080        0.550048         0.504848
   HIC            PLM_ESM2/PCA64/SVR 0.498949    0.536086     0.576275    0.510086        0.469768         0.297859
   HIC    ESMFN_STRUCTURE/ElasticNet 0.505453    0.501714     0.448151    0.518847        0.512012         0.599088
   HIC                  CONST_MEDIAN 0.518881    0.534778     0.510173         NaN             NaN              NaN
   HIC  FUSION_ESM2_ESMFN/ElasticNet 0.544564    0.457775     0.503748    0.448330        0.592841         0.468723
   HIC    POOL_attn_h64_do0.3_wd0.01 0.553694    0.525319     0.530520    0.186171        0.189567         0.107172
   HIC              SEQ_SIMPLE/Ridge 0.581111    0.575959     0.675831    0.371871        0.443266         0.215927
 TmApp             NESTED_STACK_MEAN 2.849253    3.645411     3.266043    0.544502        0.381189         0.557154
 TmApp         PLM_ABLANG2/PCA32/SVR 2.972723    3.353005     3.385667    0.535741        0.441881         0.506076
 TmApp FUSION_ABLANG2_BIO/ElasticNet 3.181774    4.102579     3.529235    0.512128        0.342440         0.534709
 TmApp                     BIO/Ridge 3.226702    3.406937     3.905534    0.409569        0.331386         0.293016
 TmApp                  CONST_MEDIAN 3.544381    3.691358     3.864198         NaN             NaN              NaN
 TmApp              SEQ_SIMPLE/Ridge 3.752994    3.383601     3.845721    0.335539        0.427862         0.296758
 TmApp  POOL_region_h64_do0.3_wd0.01 4.223900    3.862253     3.951019    0.164322        0.228828         0.450211

### MAE model-rank transfer
target metric     comparison  spearman  kendall  n_models
   HIC    mae      CV→Public  0.321429 0.238095         7
   HIC    mae Public→Private  0.821429 0.619048         7
   HIC    mae     CV→Private  0.535714 0.428571         7
 TmApp    mae      CV→Public  0.214286 0.142857         7
 TmApp    mae Public→Private  0.285714 0.238095         7
 TmApp    mae     CV→Private  0.857143 0.714286         7

### Pearson-based model-rank transfer
target  metric     comparison  spearman  kendall  n_models
   HIC pearson      CV→Public  0.657143 0.600000         6
   HIC pearson Public→Private  0.771429 0.600000         6
   HIC pearson     CV→Private  0.885714 0.733333         6
 TmApp pearson      CV→Public  0.542857 0.466667         6
 TmApp pearson Public→Private  0.257143 0.200000         6
 TmApp pearson     CV→Private  0.714286 0.466667         6

35–36. HIC CV/Public/Private: generally stable under MAE (CV→Private ρ≈0.536).
37–38. TmApp: CV→Private still informative (MAE ρ≈0.857); Public weaker.
39. TmApp Public anomalous under Pearson? Public→Private Pearson-rank ρ≈0.257 — treat Public as misleading.
40. CV predictive of Private for MAE and Pearson? **Mostly yes for CV→Private**; Public→Private unreliable for TmApp.

## Competition design (Q41–49)

41. Keep HIC? **YES**
42. Keep TmApp? **YES** (with Public warning)
43. Keep MAE primary both? **YES**
44. Show Pearson as participant-visible secondary? **YES**
45. Pearson on leaderboard vs diagnostics? Prefer **visible secondary / sidecar**, not primary sort key.
46. Distribute BIO annotations? **YES** (basic educational)
47. Advanced methods make task too easy? **NO** — residual error remains material; FT/pooling did not collapse the task.
48. Another organizer modeling Gate justified? **NO**
49. Enough to proceed to packaging? **YES**

---

## Software

```json
{
  "python": "3.12.13",
  "numpy": "2.5.2",
  "pandas": "3.0.5",
  "sklearn": "1.9.0",
  "xgboost": "3.4.1",
  "lightgbm": "4.7.0",
  "catboost": "1.2.10",
  "optuna": "4.9.0",
  "peft": "0.20.0",
  "torch": "2.10.0+cu128"
}
```

## Notes

- Learned pooling uses frozen ESM-2 150M for both tracks (AbLang2 token path not used).
- Light FT: ESM-2 150M LoRA/last-block; oneshot FT checkpoints not shipped (CV ceiling only) if marked skipped.
- Do not call any score a theoretical assay ceiling.
