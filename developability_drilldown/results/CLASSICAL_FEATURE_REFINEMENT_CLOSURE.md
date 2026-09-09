# Classical feature refinement closure

## A. Feature assets audited
See `CLASSICAL_FEATURE_ASSET_AUDIT.csv` (AbLang2, AbLingua, ESM2, RASA/ESMFold, BioEmu, MPNN, aromatic/surface).

## B. New derived blocks
- Total blocks in registry: **31**
- Tm-focused new pools: AbLang2 region/CDR/RASA + reused AbLingua guided pools
- HIC-focused: ESM2 region/CDR/RASA + aromatic RASA summaries

## C. Feature sets tested (Stage E panel)
- 8 unique sets × 5 estimators = **40** executed configurations

## D. Experiments executed
- **40** new (EXP-T045–T064, EXP-H034–H053)
- Registry total: **117**

## E. Tm best (classical refinement panel)
EXP-T045 | W=2.7289 | Pub=3.0197 | Priv=3.2736

## F. HIC best
EXP-H047 | W=0.4529 | Pub=0.4076 | Priv=0.4474

## G–R. Scientific conclusions
- RASA weighting: helpful for both targets in block-add screens; best HIC uses ESM2 RASA CDR3.
- CDR weighting: moderate Tm gains in combo paths; H-CDR3 ESM2 block useful with hydro base.
- Fixed AbLang2+AbLingua concat: not top CV-worst vs best multi-block paths.
- Aromatic/RASA summaries: secondary to ESM2 RASA pooling for HIC.
- Remaining gap: Tm CV-worst still above historical Transformer single-model best (~2.773); HIC CV-worst above Transformer (~0.442) — architecture phase next.
- Closure reason: pre-registered CV-only block/estimator search completed; freeze applied before Public/Private; 40 experiments cataloged; no ensembles.

## Final flag
**CLASSICAL_FEATURE_REFINEMENT_CLOSED = YES**
