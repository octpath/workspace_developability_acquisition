# FeNNix Fab Micro-Dynamics — Feature SPEC (pre-target freeze)

**Status:** SPEC drafted target-blind. **Full-cohort generation aborted** (`MICRODYNAMICS_TOO_SLOW`).  
This document freezes the *intended* low-dimensional feature set so it cannot be redesigned after seeing labels.

## Scientific interpretation

Finite-temperature **dynamical perturbation descriptors** on reconstructed/prepared Fab  
(VH+CH1 / VL+CL), **not** rigorous solution MD, unfolding, or direct Tm simulation.

## Input structure

- Cohort: `fennix_fab_context_final/results/FINAL_FENNIX_COHORT.csv` (N=323; ADI-47265 excluded)
- Coordinates: `fennix_fab_context/cache/r1/{id}_C_r1.pdb` (FeNNix R1-relaxed full Fab)
- Do not regenerate structures / OpenMM prep

## Dynamics protocol (ONE protocol only)

| Parameter | Value |
|-----------|-------|
| Engine | FeNNol `fennol.md` Langevin (LGV) NVT |
| Model | `foundation_stability_v2/cache/fennix-bio1S.fnx` |
| Device | `cuda:0` (RTX 3090) with working JAX CUDA libs |
| T | 300 K |
| dt | 1 fs (0.001 ps) |
| Thermostat | LGV, γ = 1 / ps |
| Equilibration | 1 ps (1000 steps) |
| Production | 5 ps (5000 steps); fallback 2 ps if timing gate fails |
| Seeds | 1 (seed=42) for first pass |

## Native contacts

- Defined on starting `C_r1` heavy-atom pairs between domains
- Cutoff: **4.5 Å** (frozen; same as Fab-context contact convention)
- Retention = fraction of native pairs still within cutoff at a frame

## Feature families (low-dimensional)

### DYN_GLOBAL

| feature | definition |
|---------|------------|
| `bb_rmsd_mean` | mean backbone (N,CA,C,O) RMSD vs t0 over production |
| `bb_rmsd_max` | max backbone RMSD over production |
| `rg_mean` | mean radius of gyration (all atoms) |
| `rg_sd` | SD of Rg |
| `epot_mean` | mean potential energy (production) |
| `epot_sd` | SD of Epot |
| `epot_drift` | (Epot_last − Epot_first) / production_time |

### DYN_DOMAIN_RMSF

Per domain {VH, VL, CH1, CL} backbone RMSF mean (optional q90 if finite):

- `rmsf_bb_mean_VH`, `rmsf_bb_mean_VL`, `rmsf_bb_mean_CH1`, `rmsf_bb_mean_CL`
- optional: `rmsf_bb_q90_*`

### DYN_INTERFACE

Native-contact retention mean / SD / min for:

- VH–VL, VH–CH1, VL–CL, CH1–CL

### DYN_EXPOSURE (HIC-oriented; sparse frame subsample)

If cheap (Bio.PDB SASA on ≤10 frames):

- aromatic SASA mean / SD / q90
- Tyr / Phe / Trp SASA mean
- hydrophobic SASA mean / SD / q90

Largest hydrophobic patch: **SKIP** if continuous-surface mesh is slow.

## Non-goals

- No T / timestep / contact-cutoff sweeps guided by CV
- No second seed over full cohort unless later subset audit
- No Test/Public/Private for feature design
