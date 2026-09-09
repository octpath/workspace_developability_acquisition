# AbLang2 follow-up run log

## 2026-09-09 — PLAN_FROZEN

- Baseline HEAD: `3bff7a5c`
- Plan files written under `advanced_models/ablang2_followup/`
- Experiment matrix locked; Public/Private selection forbidden

## QC PASS

- 324/324 H/L, hidden=480, EXACT_AA_1TO1, finite 100%

## 2026-09-09T02:33:16Z — SEQUENCE

- completed_units=0/120

## 2026-09-09T02:33:16Z — SEQUENCE

- variant=AL2F1_MIN_CONCAT scheme=None rot=None seed=None
- completed_units=0/120

## 2026-09-09T02:37:33Z — SEQUENCE

- variant=AL2F1_MIN_CONCAT scheme=None rot=None seed=None
- completed_units=30/120

## 2026-09-09T02:37:33Z — SEQUENCE

- variant=AL2F2_FULL_CONCAT scheme=None rot=None seed=None
- completed_units=30/120

## 2026-09-09T02:42:03Z — SEQUENCE

- variant=AL2F2_FULL_CONCAT scheme=None rot=None seed=None
- completed_units=60/120

## 2026-09-09T02:42:03Z — SEQUENCE

- variant=AL2F3_FULL_MEAN scheme=None rot=None seed=None
- completed_units=60/120

## 2026-09-09T02:46:19Z — SEQUENCE

- variant=AL2F3_FULL_MEAN scheme=None rot=None seed=None
- completed_units=90/120

## 2026-09-09T02:46:19Z — SEQUENCE

- variant=AL2F4_FULL_REGION_GATE scheme=None rot=None seed=None
- completed_units=90/120

## 2026-09-09T02:50:21Z — SEQUENCE

- variant=AL2F4_FULL_REGION_GATE scheme=None rot=None seed=None
- completed_units=120/120

## 2026-09-09T02:50:21Z — SEQUENCE_SELECTED

- variant=AL2F3_FULL_MEAN scheme=None rot=None seed=None
- completed_units=120/120

## 2026-09-09T02:50:21Z — FUSION

- variant=AL2F3_FULL_MEAN scheme=None rot=None seed=None
- completed_units=120/120

## 2026-09-09T02:55:53Z — FUSION

- variant=AL2F3_FULL_MEAN__FUSION__TM_PARENT_ABLINGUA_CDR3__RIDGE scheme=None rot=None seed=None
- completed_units=120/120

## 2026-09-09T03:01:23Z — FUSION

- variant=AL2F3_FULL_MEAN__FUSION__TM_PARENT_ABLINGUA_GLOBAL__RIDGE scheme=None rot=None seed=None
- completed_units=120/120

## 2026-09-09T03:05:18Z — FUSION

- variant=AL2F3_FULL_MEAN__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE scheme=None rot=None seed=None
- completed_units=120/120

## 2026-09-09T03:05:18Z — ENSEMBLE

- variant=AL2F3_FULL_MEAN__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE scheme=None rot=None seed=None
- completed_units=120/120

## 2026-09-09T03:06:26Z — POSTMORTEM

- variant=AL2F3_FULL_MEAN__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE scheme=None rot=None seed=None
- completed_units=120/120

## COMPLETE

- Sequence+fusion+ensemble+postmortem+benchmark+validation done
