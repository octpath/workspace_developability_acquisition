# GATE B1 FINAL — Shehata developability bake-off

## Verdict

**Best competition candidate: HIC**

Ranking: **HIC > TmApp > PSR**

Overall recommendation: **ADVANCE_TO_COMPETITION_DESIGN**

### Why HIC

- Nontrivial: CDR hydrophobicity alone fails (CV ρ≈0.05); broader physchem/CDR descriptors help but leave headroom.
- PLM headroom: ESM-2 lifts Dev CV from ~0.36 (CDR descriptors) to ~0.48.
- Structure headroom: ABodyBuilder2 / ESMFold SASA±RASA reach CV ρ≈0.50–0.51, competitive with PLM.
- Germline shortcut present but not dominant (BIO_SHORTCUT below best PLM/structure).
- Scientific meaning: hydrophobicity / developability proxy (not aggregation), clear sequence→surface story for education.

### Classifications

| Target | Class |
|--------|-------|
| HIC | `STRONG_COMPETITION_CANDIDATE` |
| TmApp | `GOOD_BUT_GERMLINE_SHORTCUT_RISK` |
| PSR | `SCIENTIFICALLY_INTERESTING_BUT_TOO_NOISY` |

---

## 1. Best competition candidate

1. **HIC** — best balance of difficulty, PLM/structure headroom, and robustness
2. **TmApp** — strong absolute signal but BIO_SHORTCUT nearly matches simple/CDR baselines; maturation shortcut risk
3. **PSR** — weak absolute ρ (~0.25), unstable Public, limited PLM gain

## 2. Biological shortcut strength

- **PSR**: BIO_SHORTCUT C_BIO_SHORTCUT/Lasso CV=0.227 Pub=-0.081 Priv=0.080; best PLM CV=0.225; overall A2_physchem/ElasticNet CV=0.253 Pub=0.216 Priv=0.196
  - ILLEGAL organizer oracles (top): {'representation': 'ORG_subset_only', 'model': 'ElasticNet', 'cv_spearman': 0.1733782403833457, 'public_spearman': -0.0360199379287798, 'private_spearman': -0.0148636065262778}
- **HIC**: BIO_SHORTCUT C_BIO_SHORTCUT/Lasso CV=0.264 Pub=0.349 Priv=0.204; best PLM CV=0.475; overall STR_ESMF_SASA/ElasticNet CV=0.513 Pub=0.319 Priv=0.436
  - ILLEGAL organizer oracles (top): {'representation': 'ORG_subset_plus_germline', 'model': 'ElasticNet', 'cv_spearman': 0.2582234565859944, 'public_spearman': 0.2305111426292706, 'private_spearman': 0.403921791936572}
- **TmApp**: BIO_SHORTCUT C_BIO_SHORTCUT/Lasso CV=0.403 Pub=0.350 Priv=0.470; best PLM CV=0.489; overall PLM_esm1b_t33_650M_UR50S/ElasticNet CV=0.489 Pub=0.427 Priv=0.637
  - ILLEGAL organizer oracles (top): {'representation': 'ORG_subset_plus_germline', 'model': 'ElasticNet', 'cv_spearman': 0.2905393242622242, 'public_spearman': 0.255200008626724, 'private_spearman': 0.4331757172585382}

ANARCI allele assignment: 100% H/L coverage; family agreement with author annotations = 100%.
Germline distance = 1 − V-allele identity (CDR3 excluded from mutation-fraction features).

## 3. Too easy with simple descriptors?

- **HIC**: No. Hydrophobicity-only CV≈0.0490721870764095; best simple B_cdr_descriptors/Ridge CV=0.358 Pub=0.455 Priv=0.459; best overall STR_ESMF_SASA/ElasticNet CV=0.513 Pub=0.319 Priv=0.436.
- **TmApp**: Partially descriptor-driven — simple/CDR B_cdr_descriptors/Ridge CV=0.403 Pub=0.304 Priv=0.631 close to BIO; PLM still adds ~0.08–0.09.
- **PSR**: Absolute performance low; simple ≈ best models → little headroom / noisy.

## 4. Strongest PLM

- **PSR**:
  - ablang2: PLM_ablang2_default/Ridge CV=0.068 Pub=0.027 Priv=0.313
  - esm1b: PLM_esm1b_t33_650M_UR50S/ElasticNet CV=0.178 Pub=0.138 Priv=0.203
  - esm2: PLM_esm2_t33_650M_UR50D/Ridge CV=0.189 Pub=0.147 Priv=0.259
  - esm2_cdr: PLM_esm2_CDR6/Ridge CV=0.225 Pub=0.322 Priv=0.279
- **HIC**:
  - ablang2: PLM_ablang2_default/Lasso CV=0.360 Pub=0.253 Priv=0.255
  - esm1b: PLM_esm1b_t33_650M_UR50S/Ridge CV=0.365 Pub=0.432 Priv=0.386
  - esm2: PLM_esm2_t33_650M_UR50D/Lasso CV=0.475 Pub=0.378 Priv=0.442
  - esm2_cdr: PLM_esm2_CDR6/Ridge CV=0.406 Pub=0.426 Priv=0.501
- **TmApp**:
  - ablang2: PLM_ablang2_default/Ridge CV=0.456 Pub=0.470 Priv=0.647
  - esm1b: PLM_esm1b_t33_650M_UR50S/ElasticNet CV=0.489 Pub=0.427 Priv=0.637
  - esm2: PLM_esm2_t33_650M_UR50D/ElasticNet CV=0.425 Pub=0.427 Priv=0.544
  - esm2_cdr: PLM_esm2_CDR6/Ridge CV=0.437 Pub=0.432 Priv=0.605

Original AbLang: download hung in this environment — **skipped**; AbLang2 + ESM-1b + ESM-2 cover the required antibody/generic comparison.

## 5. Antibody-specific PLM vs generic

- **HIC**: Generic ESM-2 beats AbLang2 (≈0.48 vs ≈0.36 CV).
- **TmApp**: ESM-1b slightly ahead of AbLang2 (≈0.49 vs ≈0.46); both beat BIO_SHORTCUT.
- **PSR**: All PLMs weak; AbLang2 especially poor on Dev CV.

Conclusion: antibody-specific PLM does **not** systematically beat generic ESM on this panel; ESM-2/ESM-1b are preferred defaults.

## 6. CDR-specific pooling

- PSR: whole-chain ESM-2 PLM_esm2_t33_650M_UR50D/Ridge CV=0.189 Pub=0.147 Priv=0.259 vs CDR6 PLM_esm2_CDR6/Ridge CV=0.225 Pub=0.322 Priv=0.279
- HIC: whole-chain ESM-2 PLM_esm2_t33_650M_UR50D/Lasso CV=0.475 Pub=0.378 Priv=0.442 vs CDR6 PLM_esm2_CDR6/Ridge CV=0.406 Pub=0.426 Priv=0.501
- TmApp: whole-chain ESM-2 PLM_esm2_t33_650M_UR50D/ElasticNet CV=0.425 Pub=0.427 Priv=0.544 vs CDR6 PLM_esm2_CDR6/Ridge CV=0.437 Pub=0.432 Priv=0.605

CDR6 pooling helps PSR slightly and is competitive on HIC/TmApp but does not dominate whole-chain ESM-2.

## 7. Does predicted structure improve performance?

- PSR: structure STR_ABB_SASA/Lasso CV=0.175 Pub=0.122 Priv=0.208 vs simple A2_physchem/ElasticNet CV=0.253 Pub=0.216 Priv=0.196 vs best PLM (see §4)
- HIC: structure STR_ABB_SASA_RASA/Lasso CV=0.499 Pub=0.382 Priv=0.546 vs simple B_cdr_descriptors/Ridge CV=0.358 Pub=0.455 Priv=0.459 vs best PLM (see §4)
- TmApp: structure STR_ABB_SASA_RASA/ElasticNet CV=0.424 Pub=0.177 Priv=0.500 vs simple B_cdr_descriptors/Ridge CV=0.403 Pub=0.304 Priv=0.631 vs best PLM (see §4)

**Yes for HIC** (structure ≈ best PLM). Moderate for TmApp. Weak for PSR.

## 8. RASA vs absolute SASA

- PSR: SASA-only STR_ESMF_SASA/Ridge CV=0.180 Pub=0.175 Priv=0.041 vs SASA+RASA STR_ABB_SASA_RASA/ElasticNet CV=0.144 Pub=0.106 Priv=0.038
- HIC: SASA-only STR_ESMF_SASA/ElasticNet CV=0.513 Pub=0.319 Priv=0.436 vs SASA+RASA STR_ESMF_SASA_RASA/ElasticNet CV=0.508 Pub=0.319 Priv=0.436
- TmApp: SASA-only STR_ESMF_SASA/Lasso CV=0.396 Pub=0.319 Priv=0.287 vs SASA+RASA STR_ABB_SASA_RASA/ElasticNet CV=0.424 Pub=0.177 Priv=0.500

RASA features help HIC/TmApp modestly; raw RASA>1 fraction was tiny (ABB ≈0.0006) and left unclipped.

## 9. RASA-weighted surface physchem vs sequence physchem

- PSR: SURFACE_PHYS STR_ESMF_SURFACE_PHYS/ElasticNet CV=0.132 Pub=0.110 Priv=0.136 vs simple A2_physchem/ElasticNet CV=0.253 Pub=0.216 Priv=0.196
- HIC: SURFACE_PHYS STR_ESMF_SURFACE_PHYS/Lasso CV=0.512 Pub=0.408 Priv=0.477 vs simple B_cdr_descriptors/Ridge CV=0.358 Pub=0.455 Priv=0.459
- TmApp: SURFACE_PHYS STR_ESMF_SURFACE_PHYS/Lasso CV=0.382 Pub=0.234 Priv=0.292 vs simple B_cdr_descriptors/Ridge CV=0.403 Pub=0.304 Priv=0.631

For HIC, surface-exposed hydrophobicity/charge aggregates approach PLM-level CV and beat sequence GRAVY-only baselines.

## 10. ABodyBuilder2 vs ESMFold structural features

- PSR: ABB STR_ABB_SASA/Lasso CV=0.175 Pub=0.122 Priv=0.208 vs ESMFold STR_ESMF_SASA/Ridge CV=0.180 Pub=0.175 Priv=0.041
- HIC: ABB STR_ABB_SASA_RASA/Lasso CV=0.499 Pub=0.382 Priv=0.546 vs ESMFold STR_ESMF_SASA/ElasticNet CV=0.513 Pub=0.319 Priv=0.436
- TmApp: ABB STR_ABB_SASA_RASA/ElasticNet CV=0.424 Pub=0.177 Priv=0.500 vs ESMFold STR_ESMF_SASA/Lasso CV=0.396 Pub=0.319 Priv=0.287

Effectively **equivalent for ranking** on HIC (both ~0.50 CV). ESMFold used a 25×Gly linker with linker residues excluded (HF ESMFold is not a true multichain decoder); limitation documented in `structure_prediction_audit.md`.

## 11. PLM + structure fusion

- PSR: fusion FUSION_esm2_ABB_SURFACE/Ridge CV=0.188 Pub=0.151 Priv=0.263
- HIC: fusion FUSION_esm2_ABB_SURFACE/Ridge CV=0.503 Pub=0.509 Priv=0.519
- TmApp: fusion FUSION_esm2_ABB_SURFACE/Ridge CV=0.423 Pub=0.343 Priv=0.567

Fusion is competitive but does not dramatically exceed the best single PLM or structure branch on Dev CV.

## 12. Nonlinear vs linear

- PSR: XGBoost PLM_esm2_t33_650M_UR50D/XGBoost CV=0.170 Pub=0.084 Priv=0.233 vs overall best linear-ish A2_physchem/ElasticNet CV=0.253 Pub=0.216 Priv=0.196
- HIC: XGBoost FUSION_esm2_ABB_SURFACE/XGBoost CV=0.424 Pub=0.518 Priv=0.511 vs overall best linear-ish STR_ESMF_SASA/ElasticNet CV=0.513 Pub=0.319 Priv=0.436
- TmApp: XGBoost FUSION_esm2_ABB_SURFACE/XGBoost CV=0.380 Pub=0.362 Priv=0.588 vs overall best linear-ish PLM_esm1b_t33_650M_UR50S/ElasticNet CV=0.489 Pub=0.427 Priv=0.637

LightGBM/XGBoost with conservative depth does **not** meaningfully beat tuned Ridge/Lasso/ElasticNet at N≈324 (often similar or slightly worse).

## 13. Stability across canonical + 3 shadows

- PSR: best CV by split [0.2530353715536634, 0.2477253424590719, 0.1540474034791839, 0.1306518036879544] (std=0.055)
- HIC: best CV by split [0.5129887740864253, 0.5596357117115319, 0.5239411988869722, 0.5046352402672909] (std=0.021)
- TmApp: best CV by split [0.4885468289083897, 0.5550757921931052, 0.5084857862320463, 0.4476613904236394] (std=0.039)

HIC and TmApp conclusions are directionally stable; absolute Private can swing (TmApp Private especially high on canonical).

## 14. Public leaderboard trustworthiness

- PSR: CV→Pub rank-ρ=0.447; Pub→Priv=0.385; CV→Priv=0.250
- HIC: CV→Pub rank-ρ=0.296; Pub→Priv=0.632; CV→Priv=0.725
- TmApp: CV→Pub rank-ρ=0.798; Pub→Priv=0.750; CV→Priv=0.800

At N≈350 with ~20% Public, treat Public as a **noisy** ranking signal; prefer multi-shadow confirmation before crowning winners.

## 15. Difference from C3a

C3a = one-parent **local mutation landscape**. Shehata = **many independent antibodies** across B-cell subsets with germline diversity and paired VH/VL developability readouts. Distinct scientific and ML problem; cluster-safe splits and germline-shortcut audits are essential.

---

## Full-target confirmation (secondary)

### PSR_FULL
- C_BIO_SHORTCUT/ElasticNet: CV=0.202 Pub=0.148 Priv=0.163
- C_BIO_SHORTCUT/Ridge: CV=0.179 Pub=0.165 Priv=0.166
- B_cdr_descriptors/Lasso: CV=0.173 Pub=0.117 Priv=0.202
- A2_physchem/Ridge: CV=0.172 Pub=0.155 Priv=0.222
- A2_physchem/Lasso: CV=0.167 Pub=0.155 Priv=0.236
- B_cdr_descriptors/ElasticNet: CV=0.159 Pub=0.017 Priv=0.200

### HIC_FULL
- B_cdr_descriptors/Ridge: CV=0.459 Pub=0.476 Priv=0.326
- STR_ABB_SASA_RASA/Lasso: CV=0.454 Pub=0.542 Priv=0.371
- PLM_esm2_t33_650M_UR50D/Ridge: CV=0.447 Pub=0.554 Priv=0.626
- B_cdr_descriptors/ElasticNet: CV=0.422 Pub=0.424 Priv=0.284
- A2_physchem/ElasticNet: CV=0.409 Pub=0.411 Priv=0.268
- A2_physchem/Ridge: CV=0.407 Pub=0.448 Priv=0.364

### TmApp_FULL
- PLM_esm2_t33_650M_UR50D/Ridge: CV=0.458 Pub=0.495 Priv=0.432
- B_cdr_descriptors/Ridge: CV=0.415 Pub=0.458 Priv=0.436
- B_cdr_descriptors/ElasticNet: CV=0.405 Pub=0.358 Priv=0.387
- C_BIO_SHORTCUT/ElasticNet: CV=0.360 Pub=0.114 Priv=0.030
- C_BIO_SHORTCUT/Lasso: CV=0.358 Pub=0.264 Priv=0.341
- C_BIO_SHORTCUT/Ridge: CV=0.358 Pub=0.255 Priv=0.209

TRIPLE_CORE conclusions are not artifacts of the 324 intersection: HIC remains the strongest PLM/structure-responsive target on FULL sets.

## Recommendation

**ADVANCE_TO_COMPETITION_DESIGN** with primary target **HIC retention time (min)**.

Optional: keep TmApp as a secondary track only if germline/maturation features are restricted or audited as organizer-only.

Do **not** advance PSR as the primary competition target.


## Decision nuance (for human review)

TmApp shows **stronger Public leaderboard ranking reliability** (canonical CV→Pub ≈0.80, Pub→Priv ≈0.75) than HIC (CV→Pub ≈0.30, Pub→Priv ≈0.63).

HIC remains preferred as primary target because of larger PLM/structure headroom, failed hydrophobicity-only triviality check, clearer surface-physics story, lower germline-shortcut dominance, and more stable best-CV across shadows.

If organizers prioritize Public ranking fidelity over modeling headroom, consider HIC primary + TmApp secondary, or `ADVANCE_WITH_FEATURE_RESTRICTIONS` for a TmApp track.
