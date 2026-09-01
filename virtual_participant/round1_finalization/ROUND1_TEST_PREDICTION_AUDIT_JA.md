# Round 1 Test Prediction Sanity Audit

**状態:** `ROUND1_SUBMISSION_FROZEN_BEFORE_REVEAL`（本 audit 作成時点）  
**タイムスタンプ:** 2026-09-01T21:09:12.870096+00:00

## ID checks

- [x] 162/162 IDs
- [x] no duplicate
- [x] train/test disjoint
- [x] TmApp finite
- [x] HIC finite
- [x] no NaN

## TmApp prediction (Test)

| stat | value |
|---|---:|
| min | 63.7659 |
| max | 76.6791 |
| mean | 69.4326 |
| median | 69.1787 |
| sd | 2.5672 |
| q05 | 65.4747 |
| q95 | 73.7201 |

## HIC prediction (Test)

| stat | value |
|---|---:|
| min | 8.5534 |
| max | 10.2731 |
| mean | 9.1830 |
| median | 9.1463 |
| sd | 0.2855 |
| q05 | 8.8045 |
| q95 | 9.6576 |

## Dev vs Test distribution comparison

| target | Dev mean | Test pred mean | mean shift | Dev SD | Test pred SD | SD ratio |
|---|---:|---:|---:|---:|---:|---:|
| TmApp | 69.8951 | 69.4326 | -0.4625 | 4.5024 | 2.5672 | 0.5702 |
| HIC | 9.3945 | 9.1830 | -0.2116 | 0.8095 | 0.2855 | 0.3526 |

## Notes

- distribution shift を観測しても model 変更は行わない（Round 1 protocol）。
- 本 audit 時点で organizer secret / Public / Private label は未参照。
- sanity audit: **PASS**
