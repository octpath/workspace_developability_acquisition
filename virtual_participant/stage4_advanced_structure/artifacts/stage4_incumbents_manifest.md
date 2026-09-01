# Stage 4 — Sequence/Structure incumbents freeze (start of Stage4)

Values reconfirmed from Stage2/2b/3 JSON and OOF (no re-training).

## TmApp

| track | source | Primary | Shadow |
|---|---|---:|---:|
| PLM-only | Stage2b AbLang2 HL_paired raw RidgeOpt | 2.8634 | 2.9803 |
| sequence overall | Stage2 AbLang2 + SEQ_BASIC | 2.7756 | 2.8316 |
| Stage3 provisional overall | sequence overall + RASA | ≈2.7539 | ≈2.8202 |

## HIC

| track | source | Primary | Shadow |
|---|---|---:|---:|
| PLM-only | Stage2 ESM-2 Heavy SVR | 0.4552 | 0.4529 |
| sequence overall | Stage2 ESM-2 Heavy + SEQ_ALL | 0.4485 | 0.4510 |
| Stage3 provisional overall | sequence overall + SURFACE_ALL | ≈0.4405 | ≈0.4384 |

Stage4 compares advanced-feature-only vs incumbent + advanced feature.
Stage3 artifacts are not overwritten.
