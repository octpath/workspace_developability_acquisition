# BioEmu Isolated-Domain Physical-Ensemble Reassessment — Spec

**SPEC_FROZEN_BEFORE_TARGET_SCORING = true**  
**Evidence:** ORGANIZER-EXPLORATORY  
**Target:** TmApp only  
**Parent (read-only):** `foundation_stability_v2/`  
**Do not:** VL+CL / VH+CH1 context sampling, AbLingua, disable filter, TmApp-tuned N

---

## Motivation

v2 reported requested/raw N=16 as usable ensemble size. Reconciliation showed features used **filtered XTC** with median **~5 physical frames** (~31% official-mask pass). Prior “N=16 convergence” is invalid as a physical-ensemble claim.

## Protocol

| Item | Choice |
|------|--------|
| Model | `bioemu-v1.2` (same env/checkpoint as v2) |
| Chains | Isolated VH, Isolated VL |
| Physical definition | Frames in `samples.xtc` after `filter_samples=True` / official `filter_unphysical_traj` |
| Pilot | 12 Abs from v2 `BIOEMU_CONVERGENCE_ANTIBODIES.csv` (target-blind) |
| Pilot target | ≥ **64 physical** frames / chain |
| Steering | **Off** (characterize default sampler) |
| Convergence Nphys | 4, 8, 16, 32, 64 with **20 deterministic MC subsets** (not only prefix) |
| Freeze rule | Smallest Nphys with median Spearman vs N64 ≥ 0.95 AND median NAD ≤ 10% on major families; also report MC noise |
| Full cohort | After freeze; reuse existing physical frames; stop at frozen Nphys |
| Descriptors | Reproduce v2 families: pairwise / flexibility / Rg / contact (+ SS if robust) |
| Eval | Same AbLang2 + nested Ridge + CV/Shadow as v2; OLD vs NEW mandatory |

## Forbidden

TmApp-based N choice; filter off; context-chain resampling; rewriting v2 artifacts in place (correction note only).
