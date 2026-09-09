# T030 / T068 / T070 / T071 / T072 Architecture Comparison

Primary control: **EXP-T030-REPLAY-001**  
Codes: EXP-T071, EXP-T072 registered. Historical EXP-T030 immutable.

## Central table

| Model | H/L self-attn | H–L communication | Summary | P | S | mean | worst | Pub | Priv | Overall |
|------|------|------|------|------|------|------|------|------|------|------|
| T030 replay | separate | none | hard REG_H \|\| REG_L | 3.233 | 3.202 | 3.217 | 3.233 | 3.596 | 3.256 | 3.426 |
| T068 | joint | full | single REG | 3.539 | 3.306 | 3.422 | 3.539 | 3.588 | 3.148 | 3.368 |
| T070 | joint | full | unrestricted dual REG | 3.329 | 3.187 | 3.258 | 3.329 | 3.441 | 3.085 | 3.263 |
| T071 | joint | full residues / restricted REG | chain-specific dual REG | 3.312 | 3.218 | 3.265 | 3.312 | 3.610 | 3.381 | 3.496 |
| T072 | separate | explicit cross-attn bridge | chain-specific dual REG | 3.267 | 3.252 | 3.259 | 3.267 | 3.663 | 3.306 | 3.485 |

## Parameter counts

| Model | n_trainable | notes |
|------|-------------|-------|
| T030 | 492417 | repr 256 |
| T068 | 475905 | repr 128 |
| T070 | 492417 | repr 256 |
| T071 | 492417 | same as T070 (+mask only) |
| T072 | 558467 | +cross MHA 66048 + gates 2 |

T072 breakdown: encoder 264960 / cross 66048 / gates 2 / head 33025 / other 194432.

## Scientific verdicts

### A. Single-REG penalty (T068 vs T070)

T070 recovers a large part of the T068 degradation (P −0.210, S −0.119 vs T068).  
**Dual-summary capacity matters.**

### B. REG specialization (T070 vs T071)

T071 vs T070: P −0.017 / S +0.031 (MIXED). Bootstrap CIs include 0.  
**Hard REG chain specialization does not clearly help** beyond dual capacity; may be slightly restrictive on Shadow.

### C. Communication architecture (T030 vs T071 vs T072)

- Fully joint (T071): worse than REPLAY on both schemes (NEGATIVE).
- Separate + zero-gated cross-attn (T072): also worse than REPLAY on both (NEGATIVE; ΔP +0.034, ΔS +0.050).
- T072 Primary better than T071 (−0.045) but Shadow worse (+0.034).

**Under this protocol, neither joint self-attn nor a small learned cross-attn bridge beats separate-chain T030.**  
Separate-chain dual-REG remains the strongest matched architecture.

### Learned gates (T072)

n=30 fold×seed×scheme rows:  
g_H mean≈0.0021 (sign consistency 0.60); g_L mean≈0.0012 (sign consistency 0.50).  
Gates leave zero but stay small with unstable sign → pathway is **weak / noisy**, not a stable cross-chain driver.

## Geometry readiness (DO NOT RUN)

**Cleanest future host: EXP-T072-style separate + cross-attention bridge.**  
Geometry can modify only `CrossAttn` scores:

`score(i,j) = q_i·k_j/√d + geometry_bias(d_ij)`  

without disturbing per-chain self-attn or REG specialization.  
Do **not** implement or issue a geometry experiment now.

## Artifacts

- `results/EXP-T071_CV_FREEZE.yaml`, `EXP-T071_CHAIN_SPECIFIC_REG_REPORT.md`
- `results/EXP-T072_CV_FREEZE.yaml`, `EXP-T072_CROSS_ATTENTION_REPORT.md`, `EXP-T072_CROSS_GATES.csv`
- Predictions under `experiments/predictions/EXP-T071|072/`
- Shareability: SHAREABLE_COMPLETE / REPRODUCED / canonical_benchmark_eligible=YES

## STOP

No 3D / RASA / fusion / PLM change / architecture sweep.
