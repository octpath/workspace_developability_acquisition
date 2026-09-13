# TmApp Representation × Topology × Annotation Factorial — PREREGISTRATION

**Status:** PREREGISTERED_FROZEN  
**Git at prereg:** `6ac0e169f90547d4dda7aa970f24a768cdce8581`  
**Platform:** `DL_FOLDLOCAL_COSINE_V3` / seed `101`  
**Cells:** 200 (23 REUSE + 177 new `EXP-T161`–`EXP-T337`)

## Binding commitments

- All 200 cells were defined **before** new factorial results are inspected.
- No PLM, topology, or annotation will be dropped based on intermediate scores.
- Public/Private/external metrics will **not** drive training or design changes.
- No new topology/annotation/PLM microvariants during this batch.

## Pre-bulk gate

1. `EXP-T210` — ablang2_unpaired / C / FULL  
2. `EXP-T321` — currab_unpaired / C / FULL  

PASS = extraction QC + train/eval completes + nonzero prediction variance + artifacts.  
MAE quality is **not** a gate.

## AbLang2 asset audit

The historical `residue_level/ablang2/` bundle (factorial label `ablang2_paired`) uses
**SEPARATE_CHAIN** inference formats with the `ablang2-paired` checkpoint.

The factorial allocation name `ablang2_unpaired` stores the complementary **JOINT**
paired inference (`<H>|<L>`) with the **same** checkpoint (`representation_context=PAIRED_NATIVE`)
so matched inference-context contrasts remain scientifically available while preserving
historical REUSE cells that consumed the SEPARATE bundle.

## Files

- `TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv`
- `TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.yaml`
