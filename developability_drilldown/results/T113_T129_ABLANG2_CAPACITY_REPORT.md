# AbLang2 capacity refinement (T113/T121 + T124–T129)

Platform: `DL_FOLDLOCAL_COSINE_V3` (frozen). Geometry closed. MEAN readout only.

| Code | Base arch | d_model | layers | heads | FFN | Params | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |
|------|-----------|--------:|-------:|------:|----:|-------:|-------:|-------:|----------:|-----------:|----:|-----:|--------:|
| EXP-T113 | ARCH-3 | 128 | 2 | 4 | 256 | — | 2.995 | 3.283 | **3.139** | 3.283 | 3.105 | 2.999 | **3.052** |
| EXP-T121 | ARCH-7 | 128 | 2 | 4 | 256 | — | 3.023 | 3.252 | **3.138** | 3.252 | 3.247 | 3.194 | 3.221 |
| EXP-T124 | ARCH-3 | 128 | 3 | 4 | 256 | 515329 | 3.058 | 3.396 | 3.227 | 3.396 | 3.110 | 3.116 | 3.113 |
| EXP-T125 | ARCH-3 | 256 | 2 | 8 | 512 | 1322753 | 3.156 | 3.240 | 3.198 | 3.240 | 3.255 | 3.060 | 3.158 |
| EXP-T126 | ARCH-3 | 256 | 3 | 8 | 512 | 1849857 | 3.192 | 3.290 | 3.241 | 3.290 | 3.176 | 3.057 | 3.116 |
| EXP-T127 | ARCH-7 | 128 | 3 | 4 | 256 | 581377 | 3.088 | 3.349 | 3.218 | 3.349 | 3.268 | 3.136 | 3.202 |
| EXP-T128 | ARCH-7 | 256 | 2 | 8 | 512 | 1585921 | 3.144 | 3.210 | 3.177 | 3.210 | 3.193 | 3.088 | 3.141 |
| EXP-T129 | ARCH-7 | 256 | 3 | 8 | 512 | 2113025 | 3.160 | 3.279 | 3.220 | 3.279 | 3.251 | 3.091 | 3.171 |

Paired bootstrap: `TM_HIC_NEXT_BATCH_PAIRED_BOOTSTRAP.csv`. ARCH-7 depth semantics: `T124_T129_ARCHITECTURE_AUDIT.md`.

## Answers

1. **Depth 2→3 help ARCH-3?** No. T124 TEST_mean 3.227 vs T113 3.139 (Δ≈+0.088; Primary CI overlaps 0).
2. **Depth 2→3 help ARCH-7?** No. T127 3.218 vs T121 3.138 (Δ≈+0.081).
3. **WIDE package help?** No. T125/T128 both worse than baselines; best new T128 still 3.177 > 3.138.
4. **DEEP_WIDE better alone?** No. T126/T129 worst or near-worst in each family; T126 Primary vs T113 Δ≈+0.197 with CI excluding 0 (harmful).
5. **Arch preference change with capacity?** No clear flip. Matched-capacity ARCH-3 vs ARCH-7 deltas small with overlapping CIs. At baseline, T113 wins external Overall; T121 slightly better internal TEST_mean.
6. **Original d128/L2 sufficient?** Yes. Capacity did not improve internal or (for T113) external.
7. **Over-capacity?** Yes, directionally: all six capacity variants worsen TEST_mean; larger models trend worse.

## Best / near-best cluster

- Best internal: **T121** ≈ **T113** (TEST_mean ~3.138–3.139; overlapping).
- Best external Overall: **T113** (~3.052).
- Near-best new: T128 (~3.177) — still behind baselines.
- STOP structural expansion (no L4+, d512, ensembles).
