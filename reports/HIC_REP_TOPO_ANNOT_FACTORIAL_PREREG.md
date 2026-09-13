# HIC Representation × Topology × Annotation Factorial — PREREGISTRATION

**STATUS: PREREGISTERED_FROZEN**

| Field | Value |
|-------|--------|
| Evaluation freeze SHA | `379e0751a93c2af8f6fbfeedaad4d72f3556996b` |
| Git at prereg issuance | `1dba2c330abda7caa94919de2bf2f62a5de4588c` |
| Formal manifest | `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_CELL_MANIFEST.csv` |
| **Formal manifest SHA256** | `b0790d0609da3beaa22c7aaa63be10835ce0833247575c5ee039467d64e5b6c3` |
| Pre-freeze manifest SHA256 | `ffdf88d6d59cc6af9864fa041c4a057d398f523ce724fb55d392c50282e5f7cf` |
| Plan | `developability_drilldown/results/HIC_REP_TOPO_ANNOT_FACTORIAL_PLAN.yaml` |
| ID range | **EXP-H140 … EXP-H339** |
| Cells | **200 new** / **0 REUSE** |
| Platform | `DL_FOLDLOCAL_COSINE_V3` / seed `101` / `d_model=128` / mean / REG |
| Surface / physics | **Excluded** |

## Binding commitments

- All 200 cells were defined **before** new factorial HIC results are inspected.
- No PLM, topology, or annotation will be dropped based on intermediate scores.
- Public/Private/external metrics will **not** drive training, HP, representation, topology, annotation, or cell selection.
- No surface / SASA / HSP / physics auxiliary features in this batch.
- TmApp factorial results do **not** alter HIC selection.
- Every config has **`share_hl_encoder=True`** explicitly; topology flags including `pair_interaction_mode` are explicit.
- Scientifically weak MAE is **not** a technical failure; keep the cell.
- Technical failures: repair and re-run the **same** cell. Asset-impossible: **BLOCKED** (no representation substitution).

## Factors

- Representation (10): Scratch; AbLingua; AbLang1; AbLang2 SEPARATE_CHAIN; AbLang2 PAIRED_NATIVE; ESM-1b; ESM-2; ESM-C 600M; CurrAb SEPARATE_CHAIN; CurrAb PAIRED_NATIVE
- Topology (5): SEP / JOINT / REG-SEP / XREG / FUSE (TmApp semantics)
- Annotation (4): BASE / IMGT / REGION / FULL

## Evaluation contract

- **Primary:** `TEST_mean = mean(TEST_P, TEST_S)`
- **Secondary:** `Δ_P` / `Δ_S` sign replication vs baseline; improvement requires both `< 0`
- **Contrasts:** Topology−SEP; Annotation−BASE; PLM−Scratch; PAIRED_NATIVE−SEPARATE_CHAIN (AbLang2, CurrAb)
- **Bootstrap:** antibody-level paired residual; N_BOOT=2000; seed=101; 95% percentile CI; Primary/Shadow separate; no fold-as-iid; no Pub/Priv mix
- **HIGH-tail:** `HIC > 11.5` diagnostic only
- **External:** GEN_0001 after internal freeze only

## Hierarchy note

Main grid uses **10 representation levels**. AbLang2/CurrAb additionally support family×context analysis without collapsing the grid.

## Authoritative files

- This document replaces `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_PREREG_DRAFT.md` as the formal prereg.
- Runner: `developability_drilldown/scripts/run_exp_h140_h339_factorial.py` (full batch requires separate human approval)
