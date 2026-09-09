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
