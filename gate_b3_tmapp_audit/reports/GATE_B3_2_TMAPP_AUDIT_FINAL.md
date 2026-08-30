# GATE B3.2 — TmApp Public Leaderboard Anomaly Audit (FINAL)

```text
Primary explanation:     SUBPOPULATION_MIX_EFFECT + FINITE_SAMPLE_RANK_NOISE
                         (this Public draw is an adverse / atypical board for AbLang2-class models)
Secondary explanation:   MODEL_INSTABILITY across Public vs Private for PLM vs simple/structure families;
                         mild FEATURE_TARGET_ASSOCIATION_SHIFT (some sign flips)
Implementation integrity: PASS (anomaly reproduced on CLEAN CORE; no ID/label/join bug)
Observed Public→Private ρ (clean recomputed):  -0.433
Expected pseudo-board distribution (Train OOF, ~81/81 groups):
    median ≈ +0.707 ; p5 ≈ +0.17 ; p1 ≈ -0.09 ; min ≈ -0.67
Percentile of observed ρ:  ≈ 0.08%  (frac ≤ observed ≈ 0.0008)
                              frac ≤ -0.58 ≈ 0.0002
Split integrity:         KEEP FROZEN (no integrity-based reason to unfreeze)
Recommended action:      Keep TmApp track + frozen split; treat Public as noisy/misleading;
                         emphasize Train-CV (and Private for organizers); warn participants
```

---

## Integrity

1. **ID/prediction/label alignment error?** **No.** Unique IDs; 162/81/81; hidden labels join by `id` exactly match population TmApp.
2. **Scorer error?** **No** for direct id-joined Spearman recomputation.
3. **Proxy-prediction contamination?** **Yes, but quarantined.** `RESID_*` / `ENSEMBLE_*` in B3 oneshot used a `PLM_ESM2` proxy and shared identical Public/Private scores. Excluded from CLEAN CORE.
4. **Do clean independently predicted finalists reproduce the anomaly?** **Yes.**

CLEAN CORE (n=9, genuine Train→Test one-shot fits, no retuning):

| Transfer | Spearman ρ | Kendall τ |
|---|---:|---:|
| CV → Public | **-0.100** | 0.000 |
| Public → Private | **-0.433** | -0.278 |
| CV → Private | **+0.783** | +0.611 |

---

## Statistical behavior

5. **Is Public TmApp unusually narrow/tied?** **Not unusually vs Private.**  
   Unique values: Train 39 / Public 33 / Private 35. Tie fractions high everywhere (~0.83–0.94). KS Public vs Private ≈ 0.06 (p≈1.0). Wasserstein Pub–Priv ≈ 0.53 °C. **Not a Public-only range/tie story.**

6. **How unusual is ρ≈−0.58 / observed −0.43 under pseudo-81/81 boards?** **Extremely unusual.**  
   Unmatched Train-OOF simulations (n=5000): median **+0.71**; frac≤−0.58 = **0.02%**; frac≤observed(−0.43) = **0.08%**. Matched simulations similar.  
   → The real Public/Private pair is an **outlier partition** for model-rank transfer, not “typical N=81 noise.”

7. **Bootstrapped uncertainty of Public→Private model-rank ρ (resample within fixed roles)?**  
   median **−0.18**; 95% CI **[−0.73, +0.55]**; **P(ρ<0)=0.65**. Observed −0.43 is inside this CI.  
   → Given *this* Public and Private composition, anti-correlation is a **stable property of the boards**, not a one-antibody fluke.

8. **Spearman-specific?** **Mostly correlation/ranking, not absolute error.**  
   Public→Private model-rank: Spearman **−0.43**, Pearson **−0.50**, MAE **+0.25**, RMSE **+0.08**.  
   Models that look better on Public by Spearman/Pearson tend to look worse on Private; MAE ranking is only weakly aligned.

---

## Biology / population

9. **Distinguishable Public vs Private from target-independent features?** **Mildly.**  
   Membership ROC-AUC: SEQ_SIMPLE **0.60**, PLM-PCA **0.53**, BIO_basic **0.44**. Detectable sequence-composition shift, not a sharp separable batch.

10. **Largest covariate differences?**  
    - `cluster_size` SMD Pub−Priv ≈ **−0.49** (Public enriched for larger sequence clusters)  
    - `vl_family` JS ≈ **0.10**  
    - charge / pI SMD ≈ **−0.27**  
    - nearest-Train VH identity ≈ similar (~0.72)  
    - donor JS ≈ **0** (no donor separation)

11. **Feature→TmApp associations differ?** **Partially.**  
    Association-vector Spearman: Train↔Public **0.79**, Train↔Private **0.79**, Public↔Private **0.84**.  
    Germline-distance stays negative in all roles. Some flips: `vh_len`, `H_CDR3_len` (Train + / Public −). Charge/pI associations stronger in Public than Private.

12. **Donor/B-cell/germline subgroup driving?**  
    No donor imbalance. B-cell JS small (~0.01). Germline-distance strata show AbLang2 still weaker on Public than Private where n≥15 (see `subgroup_scores.csv`). Not a single obvious stratum monopolizing the effect; **cluster-size / sequence-neighborhood mix** is the clearest structural difference.

---

## Models

13. **Why AbLang2 poor Public / strong Private?**  
    Recomputed: Public ρ≈**0.36**, Private ρ≈**0.60**. Absolute-error and rank-error concentrate on a subset of Public antibodies (see `ablang2_failure_analysis.md` / `ablang2_public_errors.csv`). Private retains a germline-distance–aligned ranking that matches Train CV; Public’s winning models are simpler/structure/gauss-CDR families that do **not** carry that same Private ranking.

14. **Why Public-favored models reverse on Private?**  
    Public top models (e.g. `SEQ_CDR_gauss`, `ESMFN_STRUCTURE`, `SEQ_SIMPLE`) have middling/low Private ranks; AbLang2 / IMGT sit mid/low on Public but high on Private. Prediction–prediction correlations differ by board (`pred_corr_public.csv` vs `pred_corr_private.csv`): Public ranking is not just tiny-score jitter among identical predictors.

15. **Few antibodies responsible?** **Partially, not exclusively.**  
    Max |Δ Public→Private ρ| from deleting one Public antibody ≈ **0.22** (still negative after deletion). Removing top 1/3/5 influencers does not restore a healthy positive transfer alone.

16. **Removing influencers change conclusion?** **No** — anti-correlation softens but the qualitative story (CV↔Private good; Public misleading) remains.

---

## Competition answers

17. **Keep TmApp track?** **Yes.** CV→Private ρ≈0.78 shows a real, learnable signal; Private ranking is credible.
18. **Keep frozen split?** **Yes.** No mapping/leakage/batch-separation bug. Unfreezing for aesthetics is not justified.
19. **Warn that Public is noisy?** **Strongly.** For TmApp, Public model ranks are **anti-aligned** with Private and with CV. Participants should prioritize robust Train CV / multi-seed validation over chasing Public.
20. **Before multi-hour headroom Gate?**  
    - Do **not** change the split.  
    - Document Public caveat in organizer notes.  
    - Consider: tighter submission limits and/or allow **two nominated** final submissions.  
    - Keep distributing BIO features (legitimate; Train/Private associations consistent) with clear baseline disclosure.  
    - Optional later: organizer-only monitoring of Public vs CV disagreement — not a participant feature.

---

## Cause classification

| Cause | Assigned? | Evidence strength |
|---|---|---|
| IMPLEMENTATION_ERROR | **No** | Integrity PASS; clean-core reproduces anomaly |
| FINITE_SAMPLE_RANK_NOISE | **Yes (secondary)** | N=81 CI wide; bootstrap P(ρ<0)=0.65 on this board |
| TARGET_RANGE_OR_TIE_EFFECT | **Weak / no** | Public not uniquely tied/narrow vs Private |
| COVARIATE_SHIFT | **Mild yes** | SEQ AUC 0.60; cluster-size SMD −0.49 |
| SUBPOPULATION_MIX_EFFECT | **Yes (primary)** | Pseudo-boards rarely this anti-correlated; AbLang2 Public/Private split |
| FEATURE_TARGET_ASSOCIATION_SHIFT | **Mild yes** | Some sign flips; overall assoc vectors still ~0.8 |
| FEW_INFLUENTIAL_SAMPLES | **Contributing** | Max LOO Δρ≈0.22; not sole driver |
| MODEL_INSTABILITY | **Yes** | AbLang2 vs simple/structure rank reversal |
| UNRESOLVED | — | Mechanism localized enough to act |

**HIC control:** stored cleanish Public→Private ρ≈**+0.33**; pseudo median ≈0.70; frac≤−0.58≈1%. TmApp’s observed anti-correlation is **more extreme** than HIC’s real transfer and rarer under null partitions.

---

## Bottom line

Train-CV **does** generalize to Private for TmApp. The Public board on this frozen split is an **adverse, atypical ranking environment**—not a broken pipeline. Preserve the split; compete on TmApp with eyes open about Public.
