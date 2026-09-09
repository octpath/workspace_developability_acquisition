# Developability Drilldown

Post-competition research catalog for antibody developability models.

This is **not** a V2 of `top_models_feature_bundle/`. The old bundle remains immutable.

## Experiment codes (canonical)

| Namespace | Meaning | Examples |
|-----------|---------|----------|
| **EXP-Txxx** | TmApp **single-target** experiment | `EXP-T001`, `EXP-T002`, … |
| **EXP-Hxxx** | HIC **single-target** experiment | `EXP-H001`, `EXP-H022`, … |
| **EXP-Mxxx** | Reserved for a **jointly trained** multi-target experiment | next: `EXP-M001` (none yet) |

Rules:

- T / H / M mark **prediction scope**, not model family
- Do **not** encode LINEAR / XGBOOST / TRANSFORMER into the code
- Each namespace is **append-only**; codes are never reused or renumbered
- Descriptive `experiment_id` remains (e.g. `LIN_TM_ABLINGUA_CDR3_RIDGE`)
- Flat legacy codes `EXP001`–`EXP048` are preserved in `LEGACY_EXPERIMENT_CODE_MAP.csv` / `legacy_experiment_code`

Display: `EXP-T001 — LIN_TM_ABLINGUA_CDR3_RIDGE`

### What EXP-M is **not**

Composing an independent Tm prediction with an independent HIC prediction is a **submission**, not an EXP-M experiment.

`EXP-M` is only for **one joint model/config/training procedure** that predicts multiple targets together. This competition has separate TmApp/HIC metrics and no combined score; EXP-M remains reserved until a real joint model exists.

## Terminology

| Term | Meaning |
|------|---------|
| **Experiment** | One configuration (single-target T/H today; joint M later) |
| **Prediction** | Experiment output artifact (single-target: `id,TmApp` or `id,HIC`). Never called a submission |
| **Submission** | Materialized competition file under `submissions/`: `id,TmApp,HIC` |

Even if an EXP-M model later emits `id,TmApp,HIC`, that stored experiment output is still a **multi-target prediction** until explicitly composed/copied as a submission.

## Artifact paths

```text
experiments/configs/EXP-T001.yaml
experiments/features/EXP-T001.parquet
experiments/predictions/EXP-T001/{oof_primary,oof_shadow,test}.csv
submissions/sub__EXP-T001__EXP-H001.csv   # T first, H second
```

## Compose

```bash
python scripts/compose_submission.py --tm EXP-T001 --hic EXP-H001
# also: descriptive experiment_id; legacy EXP001 warns and resolves
```

## Validate

```bash
python scripts/validate_repository.py
pytest tests/
```

## Next phase

Transformer backfill will issue new append-only `EXP-T` / `EXP-H` codes (or `EXP-M` only for genuine joint models).
