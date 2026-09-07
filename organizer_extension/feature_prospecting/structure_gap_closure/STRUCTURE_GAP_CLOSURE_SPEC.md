# Structure Gap Closure — Feature Spec (FROZEN, target-blind)

**Frozen UTC:** 2026-09-07T09:10:00Z  
**Workspace:** `organizer_extension/feature_prospecting/structure_gap_closure/`  
**CPU budget:** ≤8 cores (affinity 16–23); FeNNix (0–15) untouched.

## Purpose

Close remaining structure-information gaps without repeating SASA/AROMATIC-TOPO/SAP/FeNNix-v2/inverse-folding families already tested.

## Primary targets

- **TmApp** (Fab-measured) → PRIMARY structural scope = **full Fab**
- **HIC** (IgG-measured) → PRIMARY practical scope = **Fv**; Fab = SECONDARY context only

## Structure sources (no regeneration)

| Source | Path pattern | N |
|--------|--------------|---|
| Raw ESMFold Fab | `fab_reconstruction/structures/esmfold_fab/{id}.pdb` | 324 |
| Prepared Fab | `fennix_fab_context/cache/prepared_fab/{id}_prepared.pdb` (read-only) | 323 |
| Fv ESMFold / ABB2 / Boltz2 | `STRUCTURE_INPUT_CROSSWALK_v2.csv` | 324 |

**ADI-47265:** TECHNICAL_SKIP for prepared-only features; evaluate shared 323 subset and state N.

## Domain mapping (Fab)

From `SHEHATA_RECONSTRUCTED_FAB.csv`:

- **VH** = heavy residues `[0, VH_len_used)`
- **CH1** = heavy residues `[VH_len_used, end)`
- **VL** = light residues `[0, VL_len_used)`
- **CL** = light residues `[VL_len_used, end)`
- CDR/framework: `cdr_sequence_index_imgt.csv` (Fv indices)

Interfaces: VH–VL, VH–CH1, VL–CL, CH1–CL.

## Task A classification

See `SURFACE_PATCH_IMPLEMENTATION_AUDIT.md`:

**`RESIDUE_GRAPH_ONLY_OR_INCOMPLETE`** → Task B runs.

## Task B — Continuous molecular-surface HIC features

### PRIMARY_SURFACE_DEF (frozen)

1. FreeSASA **Lee–Richards** atom SASA, probe 1.4 Å (heavy atoms).
2. Exterior surface sampling: Fibonacci directions on atoms with SASA ≥ 0.5 Å²; retain points outside other probe-expanded spheres (same construction as project `hydro_surface.py`).
3. Each sample point inherits parent-residue hydrophobicity and CDR/FW/aromatic flags.
4. Hydrophobic surface mask: point with residue scale value **> scale_threshold** (per scale below).
5. Patches = connected components of hydrophobic sample points with **link distance 2.0 Å** (surface-point adjacency; **not** MSMS triangular mesh faces).
6. Patch **area** = sum of sample-point area weights (Å²).
7. Patch **perimeter** = sum of lengths of adjacency edges connecting a hydrophobic point to a non-hydrophobic / boundary neighbor (sampled-surface perimeter proxy).
8. **Compactness** = `perimeter^2 / (4 * π * area)` (1 ≈ circle).

**Honest scope:** continuous SAS-sampled hydrophobic patches with geometric area/perimeter — distinct from residue-CA graphs; **not** classical MSMS SES triangulation.

### Hydrophobicity scales (predeclared; evaluate ALL)

| id | scale | hydrophobic iff |
|----|-------|-----------------|
| `KD` | Kyte–Doolittle | KD > 0 |
| `FP` | Fauchère–Pliska π | π > 0.5 |
| `BM` | Black & Mould (Gly-centered SAP) | BM_SAP > 0 |

Max 3 scales; no target-based selection.

### Features (per scale × scope)

`hydro_area_total`, `patch_area_max`, `patch_area_2nd`, `patch_area_max_over_total`, `n_patches`, `fragmentation` (= n_patches / max(1, hydro_area_total/100)), `perimeter_max`, `compactness_max`, `aro_area_in_max`, `aro_frac_max`, `tyr_area_in_max`, `phe_area_in_max`, `trp_area_in_max`, `cdr_frac_max`, `fw_frac_max`.

Scopes:

- **Fv PRIMARY:** isolated Fv structures (ESMFold, ABB2, Boltz2)
- **Fab SECONDARY:** raw ESMFold Fab; variable-region-only surface in Fab context
- **ΔPATCH_CONTEXT** = feature(variable region in Fab) − feature(isolated Fv ESMFold)

## Task C — Packing / cavity (TmApp, Fab PRIMARY)

### Cavity definition (frozen)

Voxel grid spacing **1.0 Å**. Heavy-atom occupancy with VdW radii. Exterior flood-fill from grid boundary through empty voxels. Remaining empty connected components = **internal cavities** (packing voids), not ligand pockets. Ignore cavities with volume < **15 Å³**.

Descriptors: total/largest/count/volume_fraction for whole Fab and VH/VL/CH1/CL.

### Packing density (frozen)

For each residue with RASA < 0.20 (buried): count heavy atoms of other residues within **4.5 Å** of any heavy atom of the residue (sidechain neighbor density). Summaries mean/median/q10 for whole Fab, VH, VL, CH1, CL, CDR, FW.

## Task D — Buried unsatisfied polar / H-bond network

**PRIMARY structure:** prepared Fab (323) when hydrogens present; else heavy-atom geometric fallback documented in QC.

Buried: residue RASA < 0.20 (FreeSASA/Bio.PDB Shrake–Rupley).

H-bond (heavy-atom fallback if no H): donor N/O … acceptor O/N distance ≤ **3.5 Å**.

Salt bridge: Asp/Glu O … Arg/Lys/His N within **4.0 Å**.

Features: buried_unsat_{donor,acceptor,total,per_100res} ± per domain; network counts; interface H-bond/salt-bridge for four Fab interfaces.

## Task E — Fab domain-interface quality

For VH–VL, VH–CH1, VL–CL, CH1–CL:

- buried surface area (ΔSASA complex vs partners)
- interface residue count (CA–CA ≤ 8 Å cross-domain)
- contact density, hydrophobic/polar buried area
- H-bond / salt-bridge counts

**Shape complementarity:** `SHAPE_COMPLEMENTARITY_TECHNICAL_BLOCK` (no invented SC).

Secondary: elbow angle (VH–CH1 / VL–CL axis angle proxy).

## Raw vs prepared sensitivity

For families computable on both raw and prepared Fab: Spearman, median NAD, class `PREP_ROBUST|PREP_SENSITIVE|PREP_DEPENDENT`.

## Evaluation (after freeze)

Primary/Shadow CV; Ridge alphas `[0.1,1,10,100]`; MAE primary.

TmApp: standalone / vs AbLang2 / vs incumbent `TmApp__META_performance__ridge_100.0` / residual.

HIC: standalone / vs ESM2-H / vs ESM2-H+AROMATIC-TOPO / vs `HIC__SIMPLE_blend_seq_surf_adv` / residual.

Bootstrap B=10000 only if Primary+Shadow ΔMAE < 0.

## Optional Task F

GearNet / frozen geometric encoder: only after A–E; 45 min dep timebox; else `DEFERRED_TECHNICAL`.

## Out of scope

OpenMM, second FeNNix, MD, modifying `fennix_fab_context/` run outputs, target-driven feature selection.
