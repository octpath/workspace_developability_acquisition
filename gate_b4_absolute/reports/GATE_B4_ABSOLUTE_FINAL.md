# GATE B4 ABSOLUTE — FINAL

Overall recommendation: **KEEP BOTH TRACKS** under absolute assay-value prediction; primary leaderboard metric = **MAE** in native units; Pearson/Spearman/RMSE/R²/calibration remain secondary. Frozen B3 split **unchanged**.

Primary competition metric recommendation:
    HIC: **MAE (minutes)**
    TmApp: **MAE (°C)**

Frozen split status: **UNCHANGED** (Train/Public/Private = 162/81/81; B3 ID hashes verified in `TRAIN_ONLY_SEARCH_DECLARATION.json`)

HIC best organizer-observed Train-CV:
    MAE: 0.4691 (ENSEMBLE_NNLS/oof_nnls)
    RMSE: 0.6807
    Pearson: 0.5479
    Spearman: 0.5268
    (best non-ensemble single: 0.4781 = ESMFN_STRUCTURE/ElasticNet)

TmApp best organizer-observed Train-CV:
    MAE: 2.6148 (ENSEMBLE_RIDGE/oof_ridge)
    RMSE: 3.4576
    Pearson: 0.5956
    Spearman: 0.5933
    (best non-ensemble single: 2.8172 = FUSION_ABLANG2_BIO/ElasticNet)

HIC beginner→advanced MAE gain: CONST_MEDIAN 0.5174 → best 0.4691 (Δ=0.0483); SEQ_SIMPLE 0.5225
TmApp beginner→advanced MAE gain: CONST_MEDIAN 3.4360 → best 2.6148 (Δ=0.8213); BIO 3.1295

HIC CV→Private model-rank transfer under MAE: Spearman=0.817 (Public→Private=0.900)
TmApp CV→Private model-rank transfer under MAE: Spearman=0.720 (Public→Private=0.028)

Any reason to change frozen split: **NO**
Any reason to drop HIC: **NO**
Any reason to drop TmApp: **NO** — keep with strong Public-LB warning (B3.2 adverse board; MAE Public→Private ρ=0.028)

---

## Metric questions (1–7)

1. Absolute-value prediction scientifically meaningful for HIC? **YES** — single Shehata study, HIC retention time in minutes, common protocol/scale (`reports/assay_scale_audit.md`).
2. Absolute-value meaningful for TmApp? **YES** — DSF/TmApp in °C on the same study panel; half-degree discretization does not void absolute error.
3. MAE remain recommended primary for HIC? **YES**.
4. MAE remain recommended primary for TmApp? **YES**.
5. Pearson/Spearman better as secondary diagnostics? **YES**.
6. Does MAE improve leaderboard stability vs Spearman? **Partially / mixed.** Under MAE, HIC CV→Private ρ≈0.817 and Public→Private ρ≈0.900 look usable. TmApp CV→Private ρ≈0.709 is good, but Public→Private ρ≈0.018 remains weaker than CV→Private — do **not** choose MAE because Public looks prettier; choose MAE for scientific task fit. Ranking concordance: HIC: Spearman(MAE-rank, Spearman-rank)≈0.750 (higher=more agreement); TmApp: Spearman(MAE-rank, Spearman-rank)≈0.879 (higher=more agreement).
7. Heavy TmApp tie structure still problematic under MAE? **Ties remain** (half-degree reporting), but MAE in °C stays interpretable; ties hurt rank metrics more than absolute error.

## HIC questions (8–18)

8. Constant-median MAE: **0.5174 min**
9. SEQ_SIMPLE MAE: **0.5225**
10. Best PLM MAE: **0.4785** (PLM_ESM2_PCA64/SVR); linear PLM_ESM2/ElasticNet **0.5120**
11. Best structure MAE: **0.4781** (ESMFN_STRUCTURE/ElasticNet, nested)
12. Best GBDT MAE: **0.4851** (FUSION_ESM2_ESMFN/XGB_L1, nested; lr=0.03 fixed)
13. Best fusion MAE: **0.4892** (FUSION_ESM2_ESMFN/ElasticNet nested); screening favored fusion slightly
14. Best ensemble MAE: **0.4691** (ENSEMBLE_NNLS OOF)
15. Best optional learned/fine-tuned PLM MAE: **NOT RUN** (P4/P5)
16. Reproducible MAE improvement simple→advanced: SEQ_SIMPLE 0.5225 → ensemble/best 0.4691 (Δ≈0.0534); vs structure single Δ≈0.0444
17. Structure beyond PLM under MAE? **YES** — nested ESMFN 0.4781 < PLM_ESM2 linear 0.5120; fusion/ensemble adds small further gain.
18. HIC still a strong competition task? **YES** — clear headroom above constant/SEQ, structure signal, stable MAE transfer.

## TmApp questions (19–32)

19. Constant-median MAE: **3.4360 °C**
20. SEQ_SIMPLE MAE: **3.2650**
21. BIO MAE: **3.1295**
22. IMGT/germline MAE: IMGT_POS_HL **3.2275**; GERMLINE_REL **3.1605**
23. Best frozen PLM MAE: **3.0035** (AbLang2/ElasticNet)
24. Best nonlinear PLM MAE: **2.9995** (AbLang2 PCA32/SVR)
25. Best GBDT MAE: **3.1932** (FUSION_ABLANG2_BIO/LGB_L1) — **does not beat** linear fusion after nested
26. Best fusion MAE: **2.8172** (FUSION_ABLANG2_BIO/ElasticNet) — strongest single family
27. Best ensemble MAE: **2.6148** (ENSEMBLE_RIDGE OOF)
28. Best optional learned/fine-tuned PLM MAE: **NOT RUN**
29. MAE improvement beyond BIO: BIO 3.1295 → fusion 2.8172 (Δ≈0.3123); → ensemble 2.6148 (Δ≈0.5147)
30. Meaningfully learnable beyond basic BIO? **YES, moderately** — AbLang2+BIO fusion improves ~0.2–0.5 °C MAE over BIO; not a huge leap, still shortcut-aware.
31. Frozen Public anomaly persist under MAE? **Partially.** CV→Private transfer under MAE is solid (ρ≈0.709); Public→Private weaker (ρ≈0.018). Do not chase Public.
32. TmApp still a strong competition task? **YES, with caveats** — learnable; BIO/germline shortcuts matter; warn on Public.

## Optuna / GBDT (33–41)

33. For every GBDT family, learning_rate was fixed: **YES**
34. Exact fixed learning rate: **0.03**
35. n_estimators/iterations NOT tuned by Optuna: **YES** (ceiling 5000 + early stopping)
36. Effective boosting-round distribution (nested): median≈84, mean≈149, max≈625 (n=30)
37. How often was 5000-round ceiling reached? **0%** of nested GBDT outer folds (expect ≈0)
38. Outer Validation NEVER used for early stopping: **YES** (internal ~18% group split only; then refit with median rounds)
39. Optuna trial counts: Stage1 ElasticNet≈40, PCA-SVR≈30, GBDT≈40; nested inner≈20–25/outer-fold; SQLite under `optuna/*.db`
40. Hyperparameters that mattered: tree depth/leaves, min_child_*, subsample/colsample, reg_alpha/lambda (see trials CSVs); loss L1 vs L2 compared as separate pipelines
41. Did GBDT beat simpler regularized models after nested? **Generally NO** — HIC XGB_L1 ≈0.497 vs ESMFN ElasticNet ≈0.476; TmApp LGB/XGB worse than FUSION ElasticNet ≈2.89

## Competition design (42–50)

42. Recommended leaderboard metric for HIC: **MAE (minutes)**
43. Recommended leaderboard metric for TmApp: **MAE (°C)**
44. Same metric both tracks? **YES (MAE)**; units differ; no combined cross-track score
45. Participant-visible secondary metrics: RMSE, Pearson r, Spearman ρ, R²; optional calibration slope/intercept
46. Should basic BIO annotations be distributed? **YES**
47. Which: VH/VL germline family, κ/λ, CDR lengths, germline identity/distance, basic IMGT region labels — **not** donor / B-cell subset
48. Two final submissions per team? Optional; default one file `id,TmApp,HIC`
49. Special Public-LB warning for TmApp under MAE? **YES — strongly** (B3.2 + B4 MAE Public→Private weaker than CV→Private)
50. Competition ready for packaging after this Gate? **YES** (split frozen; absolute-value objective validated; packaging Gate next)

---

## One-shot Public/Private (frozen finalists)

See `metrics/finalist_public_private.csv`. Highlights (lower MAE better):

### HIC
                            tag        model   cv_mae  public_mae  private_mae  public_spearman  private_spearman
                  ENSEMBLE_NNLS     oof_nnls 0.469072    0.468358     0.442942         0.531206          0.541246
              CAL_ENSEMBLE_NNLS   linear_oof 0.482434    0.472736     0.452015         0.531206          0.541246
              FUSION_ESM2_ESMFN       XGB_L1 0.496847    0.465654     0.469764         0.510914          0.547017
                ESMFN_STRUCTURE   ElasticNet 0.475754    0.488010     0.494793         0.500965          0.531806
                 PLM_ESM2_PCA64      SVR_RBF 0.478858    0.482343     0.500925         0.545874          0.431629
                   CONST_MEDIAN     constant 0.517388    0.534778     0.510173              NaN               NaN
                       PLM_ESM2   ElasticNet 0.524919    0.506040     0.526657         0.461172          0.383623
RESID_ESMFN_STRUCTURE__PLM_ESM2 OOF_residual 0.709243    0.647060     0.658730         0.423942          0.264373
                     SEQ_SIMPLE   Ridge_grid 0.520200    0.575959     0.675831         0.535599          0.294456

### TmApp
                   tag        model   cv_mae  public_mae  private_mae  public_spearman  private_spearman
     CAL_ENSEMBLE_NNLS   linear_oof 2.684785    3.398112     3.224328         0.402894          0.546887
           PLM_ABLANG2   ElasticNet 3.084598    3.580384     3.228803         0.391940          0.598475
    FUSION_ABLANG2_BIO   ElasticNet 2.893104    3.495218     3.229976         0.414550          0.591739
        ENSEMBLE_RIDGE    oof_ridge 2.614769    3.414256     3.231587         0.402747          0.544954
     PLM_ABLANG2_PCA32      SVR_RBF 3.002878    3.272414     3.404303         0.456549          0.517720
RESID_BIO__PLM_ABLANG2 OOF_residual 3.259949    4.147601     3.584453         0.390040          0.508928
    FUSION_ABLANG2_BIO       LGB_L1 3.248686    3.481521     3.587973         0.355096          0.379919
          GERMLINE_REL   ElasticNet 3.193401    3.296433     3.637413         0.479046          0.376981
           IMGT_POS_HL   Ridge_grid 3.227537    4.018881     3.785208         0.117461          0.418748
            SEQ_SIMPLE   Ridge_grid 3.265012    3.383601     3.845721         0.427732          0.319166
          CONST_MEDIAN     constant 3.436048    3.691358     3.864198              NaN               NaN
                   BIO   Ridge_grid 3.128306    3.406937     3.905534         0.293494          0.308171

### Transfer
target     comparison  spearman  kendall  top3_overlap  n_models
   HIC      CV→Public  0.716667 0.500000             1         9
   HIC Public→Private  0.900000 0.722222             3         9
   HIC     CV→Private  0.816667 0.666667             1         9
 TmApp      CV→Public  0.335664 0.242424             0        12
 TmApp Public→Private  0.027972 0.000000             0        12
 TmApp     CV→Private  0.720280 0.515152             2        12

---

## Protocol notes

- Primary Optuna objective: **minimize MAE**
- GBDT: `learning_rate=0.03` fixed; `n_estimators/iterations=5000` ceiling; `early_stopping_rounds/od_wait=150`
- Nested GBDT confirmation used **repeat-0 only (5 folds)** for cost; linear/kernel used full 5×3=15
- Learned pooling / LoRA: **NOT RUN**
- Assay noise ceiling: **NO EMPIRICAL ASSAY NOISE CEILING AVAILABLE**
- HIC is **not** an aggregation assay; RT is protocol-dependent

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
  "optuna": "4.9.0"
}
```

## Finalist registry

Path: `config/FINAL_ABSOLUTE_VALUE_FINALISTS.json`  
SHA256: `2bc0963082c69ace8473539382d8e8c0c37658c5bb439cb878072f3c7a31bbe7`
