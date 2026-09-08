# ENDGAME MODEL BENCHMARK AUDIT
## Top answers
1. **Recipes evaluated?** 21 recipes × regressors → **42** CV rows
2. **Ridge vs Lasso (cv_worst)?** Ridge 12 / Lasso 7 / Tie 2
3. **Lasso help mainly on high-dim concat?** **YES for HIC** (ESM2+… Lasso dominates). TmApp AbLingua concat still prefers Ridge+PCA protocol.
4. **Best TmApp CV?** `TM_PARENT_ABLINGUA_CDR3` + RIDGE (P=2.7321/S=2.7850)
5. **Best HIC CV?** `HIC_HYDRO_TITRATION` + LASSO (P=0.4872/S=0.4834)
6. **TmApp Top-3:** TM_PARENT_ABLINGUA_CDR3/RIDGE, TM_PARENT_ABLINGUA_GLOBAL/RIDGE, TM_BASE_BIOEMU_MPNN/RIDGE
7. **HIC Top-3:** HIC_HYDRO_TITRATION/LASSO, HIC_ARO_TITRATION/LASSO, HIC_ARO_CONTINUOUS_SURFACE/LASSO
8. **Public/Private (Top-3):**
   - TmApp TM_PARENT_ABLINGUA_CDR3/RIDGE: Pub=3.1185 Priv=3.2899
   - TmApp TM_PARENT_ABLINGUA_GLOBAL/RIDGE: Pub=3.1069 Priv=3.3264
   - TmApp TM_BASE_BIOEMU_MPNN/RIDGE: Pub=3.1199 Priv=3.4327
   - HIC HIC_HYDRO_TITRATION/LASSO: Pub=0.4702 Priv=0.4642
   - HIC HIC_ARO_TITRATION/LASSO: Pub=0.4799 Priv=0.4757
   - HIC HIC_ARO_CONTINUOUS_SURFACE/LASSO: Pub=0.4799 Priv=0.4757
9. **CV winner fail badly on Private?** TmApp Top-1 CDR3 Private≈3.290 vs CV mean≈2.759 (gap≈+0.53), similar band to other PLM stacks. HIC Top-1 Private **beats** CV.
10. **Weaker CV but strong Private?** HIC Lasso surface recipes generalize well. TmApp: PARENT GLOBAL has best Public among Top-3; CDR3 best CV-worst and best Private among Top-3.
11. **Bundle complete for Top-3?** YES — union of required blocks exported.
12. **License-blocked?** No hard omit; AbLang2/AbLingua/ESM2 marked REVIEW_MODEL_OUTPUT; FeNNix not in Top-3.
13. **Validation/smoke?** `validate_bundle.py` → VALIDATION OK

## Process notes
- FEATURE_RECIPE_FREEZE hashed before scoring
- FULL_DEV_ALPHA_POLICY = median Primary fold alphas; frozen before PP
- OpenMM DOMAIN_FLEX: CV only (no Test MD features)
- AROMATIC `aro_error` all-NaN column dropped in feature_store
