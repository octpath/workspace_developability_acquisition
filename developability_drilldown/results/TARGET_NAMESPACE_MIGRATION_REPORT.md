# Target-namespace migration report

**Baseline:** `486c9e37` (permanent flat EXP001–EXP048)  
**Script:** `scripts/migrate_target_namespace.py`

## Motivation

Flat `EXP001`… codes hid the target. Conversations, submissions, and notebooks benefit from seeing scope in the code itself, without encoding model family.

## Old → new policy

| Old | New |
|-----|-----|
| `EXP001`…`EXP048` (flat) | `EXP-Txxx` (TmApp), `EXP-Hxxx` (HIC) |
| (none) | `EXP-Mxxx` **reserved** for jointly trained multi-target experiments |

- T/H/M = prediction scope only
- Family stays in metadata (`LINEAR`, `XGBOOST`, …)
- Within each target, **legacy numeric order is preserved** (no score/family re-sort)
- Existing codes never renumbered again; append-only per namespace

## Exact counts

| Namespace | Count | Range | Next |
|-----------|------:|-------|------|
| TmApp (T) | **25** | EXP-T001 … EXP-T025 | EXP-T026 |
| HIC (H) | **23** | EXP-H001 … EXP-H023 | EXP-H024 |
| MULTI (M) | **0** | — | **EXP-M001** (reserved) |

## Why M ≠ submission

A submission pairs two independent single-target predictions (`sub__EXP-T…__EXP-H….csv`).

`EXP-M` means **one joint training/config** that predicts multiple targets. Do not invent fake combined CV/Public/Private scores; extend the score schema only when a real joint model appears.

## Legacy mapping

Authority: `results/LEGACY_EXPERIMENT_CODE_MAP.csv` (48/48)  
Also mirrored as `legacy_experiment_code` on each `experiments.csv` / `EXPERIMENT_CODES.csv` row.

## Preservation

| Check | Result |
|-------|--------|
| Scores (join by `experiment_id`) | max abs delta **0.0** |
| Predictions (FULL 12) | max abs delta **0.0**; file SHA256 unchanged |
| Feature parquet | file SHA256 + content SHA256 unchanged (rename-only) |
| Shared feature_set hashes | PASS |

## Path / submission migration

FULL artifacts moved to `EXP-Txxx` / `EXP-Hxxx` paths.  
Examples regenerated as `sub__EXP-T001__EXP-H001.csv` (T then H only).

## Validation / tests

`validate_repository.py` → PASS  
`pytest tests/` → run at commit time
