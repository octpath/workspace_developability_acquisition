# T073 / T074 / T075 Platform Comparison

| Property | T073 | T074 | T075 |
|----------|------|------|------|
| LR grid | 0.1–3×3e-4 | 1e-5..1e-2 | 1e-5..1e-2 |
| scheduler | CosineAnnealingLR T_max=200 η=0 | cosine100+hold | cosine T_max=200 η=0.01×lr0 |
| min epochs | none | 100 | none |
| patience | 20 | 20 | 30 |
| seed | 101 | 101 | 101 |
| full-Dev refit | no | no | no |

## Scores (not strictly protocol-equivalent)

| Metric | T073 | T074 | T075 |
|--------|------|------|------|
| VAL_P | 2.973623 | 2.979829 | 2.972246 |
| VAL_S | 3.081734 | 3.097439 | 3.120248 |
| TEST_P | 3.361936 | 3.286656 | 3.512144 |
| TEST_S | 3.144960 | 3.426023 | 3.291950 |
| Pub (P mean) | 3.541398 | 3.484075 | 3.478785 |
| Priv (P mean) | 3.300341 | 3.274552 | 3.287402 |
| Overall (P mean) | 3.420870 | 3.379313 | 3.383093 |

## Selected initial-LR (approx)

- T073: often 9e-4 (upper end of narrow grid)
- T074: 9×1e-3, 1×1e-4
- T075: 6×1e-4, 4×1e-3

T075 closes platform development. Do not invent a fourth platform from this table.

