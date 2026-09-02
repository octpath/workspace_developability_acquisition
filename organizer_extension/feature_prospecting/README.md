# Organizer Extension — Feature Prospecting

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`

Round1 Participant とは独立した Organizer exploratory workspace。

## Evidence boundary

ORGANIZER-EXPLORATORY。Public/Private は post-hoc replication のみ。unseen-test / prospective evidence ではない。

## Gates

| Gate | State |
|------|-------|
| 1 | Contract frozen |
| 1.1 | Median baseline, nested Ridge, residual Ridge, crosswalk v1 |
| **1.2** | Residual leakage fix, 5-level priors, **Boltz-2 third generator**, crosswalk v2 |

## Key docs

- [ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md](ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md)
- [SIGNAL_CLASSIFICATION_SPEC.md](SIGNAL_CLASSIFICATION_SPEC.md)
- [STRUCTURE_ROBUSTNESS_SPEC.md](STRUCTURE_ROBUSTNESS_SPEC.md)
- [GATE1_2_STRUCTURE_SOURCE_EXTENSION_REPORT_JA.md](GATE1_2_STRUCTURE_SOURCE_EXTENSION_REPORT_JA.md)
- [STRUCTURE_INPUT_CROSSWALK_v2.csv](STRUCTURE_INPUT_CROSSWALK_v2.csv)
- [FEATURE_RELEVANCE_SUMMARY.csv](FEATURE_RELEVANCE_SUMMARY.csv)
- Boltz-2: `structure_sources/boltz2_fv_standard_v1/`

## Structure generators

1. ESMFold（primary）
2. ABodyBuilder2
3. Boltz-2 `BOLTZ2_FV_STANDARD_v1`（third technical replicate; not ESMFold replacement）

## Next

Human review 後に first-wave feature family を開始。Gate1.2 では feature family 実験を自動開始しない。
