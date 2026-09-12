# HIC Residue-Level SURFACE Fusion Report (H128–H133)

**Status: STOPPED — no training**

Audit: `results/H128_H133_RESIDUE_SURFACE_AUDIT.md`  
Prereg/gate: `results/H128_H133_RESIDUE_SURFACE_PREREGISTRATION.yaml`

## Why training did not run

Historical SURFACE used in HIC Transformer late fusion is **`F1_SURFACE` = antibody-level 35D**. There is no frozen residue-aligned SURFACE table that maps 1:1 onto Transformer tokens. Batch rules forbid inventing residue descriptors or reconstructing them from antibody aggregates. Therefore EXP-H128–H133 were **not issued and not trained**. Next HIC remains **EXP-H134 is not reached; next remains EXP-H128**.

## Distinction that remains valid (historical evidence)

| Level | Experiments | TEST_mean |
|-------|-------------|----------:|
| Sequence-only Scratch | H071 | 0.502 |
| Sequence-only ESM2 | H061 | 0.507 |
| Antibody-level SURFACE late fusion Scratch | **H090** | **0.472** |
| Antibody-level SURFACE late fusion ESM2 | **H086** | **0.486** |

Antibody-level SURFACE already helps. Residue-correspondence incremental value is **untested** because the prerequisite residue source is missing.

## Central metrics table

| Backbone | Mode | TEST_P | TEST_S | TEST_mean | Δbase | bootstrap CI | SURFACE-zero Δ | residue-shuffle Δ | Public | Private | Overall |
|----------|------|--------|--------|-----------|-------|--------------|----------------|-------------------|--------|---------|---------|
| — | — | — | — | — | — | — | — | — | — | — | **NOT RUN** |

## Required questions

1. Does residue-level SURFACE improve H071? **UNKNOWN / NOT TESTABLE** (no residue F1 source)
2. Does residue-level SURFACE improve H061? **UNKNOWN / NOT TESTABLE**
3. Is residue-level fusion better than historical antibody-level late fusion? **UNKNOWN / NOT TESTABLE**
4. Does correct residue-to-SURFACE alignment matter? **UNKNOWN / NOT TESTABLE**
5. Is additive, gated, or SURFACE-aware pooling preferable? **N/A**
6. Is the effect backbone-dependent? **N/A**
7. Does the model functionally rely on SURFACE? **N/A for residue modes**; antibody-level H090/H086 historically yes
8. Does SURFACE primarily modify residue representations or residue importance? **N/A**
9. Is cross-attention scientifically justified next? **NO** — simpler aligned fusion was never executable
10. If yes, what interaction should cross-attention represent? **N/A**

## Unlock path (future batch only)

Materialize+freeze residue channels that definitionally underlie F1 (and/or a specified residue reduction of HYDRO), with token mapping audit, then redesign H128+. Do not escalate to cross-attention until that exists.

**STOP.**
