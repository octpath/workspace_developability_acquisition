# FeNNix Fab Context — SPEC (frozen before TmApp scoring)

**State:** `FENNIX_FAB_CONTEXT_SPEC_FROZEN`  
**Construct:** `RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB` (not exact experimental Fab)

## Goal

Decompose whether FeNNix-v2 curvature features gain TmApp-relevant information when moving from isolated Fv to reconstructed full Fab — without comparing raw total energies across different atom counts.

## Conditions

| ID | Structure | Prep |
|----|-----------|------|
| **A** ISO_FV | Isolated Fv ESMFold | Reuse FeNNix-v2 esmfold features when compatible |
| **B** FABGEOM_FV_RELAXED | VH+VL cut from Fab ESMFold | Independent R1 |
| **C** FULL_FAB | Full Fab | Gate F0 disulfide prep + R1 |
| **M** MATCHED_FV | VH+VL from **relaxed C** | Delete CH1/CL; **no** re-relax |

## Primary contrasts

- `DELTA_GEOM = feat(B) − feat(A)` — structure-prediction context
- `DELTA_ENV = feat(C_var) − feat(M_var)` — constant-domain environment at fixed Fv coords
- `PREP_RELAX_SENSITIVITY` = M vs B — control only

## Physics (identical to FeNNix-v2)

- φ/ψ ±2°, χ1 ±5°
- \(K_E = [E(+δ)+E(−δ)−2E_0]/δ_{\mathrm{rad}}^2\)
- R1: backbone heavy fixed; FIRE fmax=0.10 eV/Å; max 200 steps

## Gate F0

OpenMM + PDBFixer; explicit topology-based disulfides (κ/λ audited separately); restrained minimization; QC distances / clashes / domain RMSD.

## Evaluation

TmApp only; Primary + Shadow CV; standalone Ridge; incremental vs AbLang2; incremental vs frozen incumbent `TmApp__META_performance__ridge_100.0`; residual; bootstrap B=10000.


## Protocol freeze (post CPU/CUDA audit)

- Decision: `CPU_CUDA_EQUIVALENT_REUSE_113_CUDA_REMAINING_ON_CPU`
- Curvature Spearman=0.9839, median NAD=0.0189
- Reuse existing CUDA preparations; remaining Abs use OpenMM CPU small batches.
