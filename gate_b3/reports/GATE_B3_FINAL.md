# GATE B3 FINAL — Split Freeze + Modeling Headroom

**Overall recommendation: `PROCEED_TO_FINAL_PACKAGING`**

| Track | Label |
|---|---|
| HIC | `READY_FOR_COMPETITION` |
| TmApp | `READY_BUT_SHORTCUT_AWARE` |

Independent leaderboards; common Train/Public/Private IDs; one submission file `(id, TmApp, HIC)`.

---

## Split

1. **Counts:** Train/Public/Private = **162 / 81 / 81**
2. **Public ≈ Private:** yes — difference = **0** (exact equality)
3. **Sequence-cluster leakage:** **zero** (90% paired-identity groups intact)
4. **HIC/TmApp balance:** selected by pre-model Wasserstein/quantile/family JS score (seed=20274057, pool=2500)
5. **Safe to freeze permanently:** **YES** — marked immutable except genuine leakage bugs

```
FINAL SPLIT FROZEN
DO NOT CHANGE BASED ON MODEL PERFORMANCE
```

Population N=324, sha256=`085b2cdb25e3d0e2…`

---

## HIC

6. **Best Train-CV:** `ESMFN_STRUCTURE` / `ElasticNet` ρ≈**0.580**
7. **Best Public (finalists):** `SEQ_SIMPLE` ρ≈**0.536**
8. **Best Private (finalists):** `ESMFN_STRUCTURE` ρ≈**0.542**
9. **Robust family:** native ESMFold structure (surface/SASA/physchem/patches) ± ensembles
10. **Simple→advanced gap:** SEQ_SIMPLE ~0.376 → best ~0.580 (**~0.20**)
11. **Structure beyond PLM:** **YES** — ESMFN 0.580 > best raw PLM ESM1B 0.517
12. **Rank-oriented learning:** mild (ESM2_rank ~0.473); not primary lever
13. **Ensemble:** **YES** — NNLS/rank-mean ~0.55–0.57 on Train OOF

---

## TmApp

14. **Best Train-CV:** `PLM_ABLANG2_PCA64` / `SVR_RBF` ρ≈**0.480**
15. **Best Public (finalists):** `ENSEMBLE_NNLS` ρ≈**0.493**
16. **Best Private (finalists):** `PLM_ABLANG2` ρ≈**0.603**
17. **BIO/germline explains:** BIO_SHORTCUT ~0.431 (large share of signal)
18. **PLM beyond BIO:** AbLang2 ~0.480 − BIO ≈ **+0.049**
19. **Structure:** **little/no** — ESMFN ~0.357 < BIO/PLM
20. **Rank-oriented:** modest (gauss SEQ_CDR); representation > transform
21. **Ensemble:** limited beyond best PLM (~0.45)

---

## New methods

22. **IMGT positional one-hot:** helpful (HIC~0.442, TmApp~0.456)
23. **Germline-relative:** useful for TmApp (~0.418); secondary for HIC
24. **N-grams:** fair IS baseline (~0.34–0.36); not organizer ceiling
25. **PCA/PLS:** **YES** — especially AbLang2 PCA+SVR on TmApp; ESM1B PCA on HIC
26. **RBF SVR / KRR:** helpful on PCA-reduced PLM (TmApp leader)
27. **Target rank/Gauss:** small effect
28. **Pairwise RankSVM:** not completed as full branch; rank transforms already tested
29. **VH vs VL:** HL ≫ H or L alone (IMGT ablation)
30. **CDR vs FR:** germline region gravies encoded; CDR-aware SEQ_CDR > SEQ_SIMPLE
31. **Multitask:** exploratory MultiTaskEN logged; not superior to single-task leaders
32. **Fine-tuning:** not run (optional P5)

---

## Competition

33. **HIC strong track?** **YES** — structure headroom, weak BIO shortcut, clear science story
34. **TmApp strong track?** **YES**, with BIO transparency
35. **Too easy?** HIC **no** (large advanced−simple gap). TmApp **borderline** if BIO is “free” — mitigate with published BIO baseline
36. **Public LB acceptable?** HIC finalist CV→Public rank ρ is noisy (small finalist set / N=81); still usable with bootstrap CIs. Prefer shadow/bootstrap communication to participants.
37. **BEGINNER_BASELINE:** `SEQ_SIMPLE + Ridge` (composition/ProtParam)
38. **ADVANCED_REFERENCE_BASELINE:** HIC `ESMFN_STRUCTURE + ElasticNet`; TmApp `AbLang2 + Ridge` (or PCA+SVR)
39. **Blockers to packaging?** **None critical.** Split frozen; staging files ready; scorer implemented. Do not change split after these results.

---

## Design reminder

- Two **independent** leaderboards (Spearman per track)
- Common molecule IDs and split
- Participant files: `frozen/participant_staging/{train,test,sample_submission}.csv`
- Hidden labels: `frozen/organizer/test_labels_hidden.csv`

## Artifacts

- `config/FINAL_SPLIT_MANIFEST.json`
- `config/FINALIST_REGISTRY.json`
- `config/TRAIN_CV_FOLDS.json`
- `metrics/all_train_cv_results.csv`
- `metrics/finalist_results.csv`
- `metrics/public_private_results.csv`
- `metrics/bootstrap_results.csv`
- `scripts/score_submission.py`

Optional GPU pooling/fine-tuning deferred; not required to proceed.
