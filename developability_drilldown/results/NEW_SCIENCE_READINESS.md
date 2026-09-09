# New science readiness freeze

## 1. Current registry counts

- Total: **77**
- By family: {'LINEAR': 42, 'TRANSFORMER': 29, 'XGBOOST': 6}
- By target: {'TmApp': 44, 'HIC': 33}

## 2. Historical single-model completeness

- Authority sources audited: LINEAR_MODEL_MASTER_REGISTRY, MODEL_BENCHMARK_SUMMARY, ADVANCED_MODEL_RESULTS,
  round1_all_model_score_inventory, stage5 blends, cross-family / top3 ensemble CSVs, gate reports (inventory).
- Completeness audit rows: **441**
- Classification: `{'NONCOMPARABLE': 304, 'ALREADY_REGISTERED': 77, 'ENSEMBLE_EXCLUDED': 55, 'DUPLICATE_ALIAS': 5}`

## 3. Missing comparable single models found

- TmApp: **0**
- HIC: **0**

## 4. Backfilled

- FULL / PARTIAL / SCORE_ONLY: **0 / 0 / 0** (no eligible comparable gaps)
- EXP codes issued this phase: **none** (next remains EXP-T045 / EXP-H034)

## 5. Unresolved INCONSISTENT

- Count: **0**

## 6. Historical best comparable singles

See `HISTORICAL_SINGLE_MODEL_BESTS.md`.

## 7. Historical best ensembles (not registered)

- HIC Private ceiling among inventoried ensembles ≈ **0.418** (`HIST__HIC_PRIVATE_WINNER` / SIMPLE_blend_esm2_surf).
- TmApp follow-up cross-family equal-mean Private ≈ **3.146**.

## 8. Next experiment codes

- TmApp: **EXP-T045**
- HIC: **EXP-H034**
- Multi: **EXP-M001** (reserved, unused)

## 9. Phase 2B readiness

Comparable historical single-model catalog is complete relative to frozen Simple-TVT authorities.
Noncomparable Stage/SVR/structure historical singles remain documented in the audit only.
Ensembles inventoried separately; not mixed into single-model registry.

## Final flag

**NEW_SCIENCE_READY = YES**

Conditions met: audit complete; eligible comparable missing models addressed (none);
no INCONSISTENT blockers; existing 77 preserved; ensembles excluded from registry;
validation/tests expected PASS.


## 10. Classical feature refinement (Phase 2B)

- Derived blocks: region/CDR/RASA/aromatic pooling from existing residue + structure assets
- CV-only selection through Stage E; freeze: `CLASSICAL_REFINEMENT_FREEZE.yaml`
- New experiments: **40** (no prediction ensembles)
- Existing **77** preserved (score delta 0)

### Flags
- **HISTORICAL_COMPLETENESS = PASS**
- **CLASSICAL_FEATURE_REFINEMENT_CLOSED = YES**
- **ENSEMBLE_REFINEMENT = NOT_STARTED**
- **NEW_ARCHITECTURE_READY = YES**

## 11. Artifact completeness / reproduction

- See `EXPERIMENT_ARTIFACT_COMPLETENESS.csv` and `EXPERIMENT_REPRODUCTION_AND_ARTIFACT_COMPLETION.md`
- **ARTIFACT_COMPLETENESS_AUDITED = YES**
- **CANONICAL_RESULTS_VERIFIED = YES** (eligible experiments only)
- **NEW_ARCHITECTURE_READY = YES**
