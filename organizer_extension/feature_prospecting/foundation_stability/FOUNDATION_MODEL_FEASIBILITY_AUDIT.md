# FOUNDATION_MODEL_FEASIBILITY_AUDIT

**Branch:** `organizer_extension/feature_prospecting/foundation_stability/`  
**Scope mismatch (all models):** TmApp measured on **Fab**; inputs are **Fv** predicted structures. Do not fabricate CH1/CL.  
**Evidence boundary:** ORGANIZER-EXPLORATORY  
**Date:** 2026-09-03

Absolute total energies are **not** assumed comparable across antibodies. Primary features use **perturbation responses / ensemble aggregates**.

---

## Summary

| Model | Classification | Proceed? |
|-------|----------------|----------|
| FeNNix-Bio1 (via FeNNol) | **READY_WITH_LIMITATIONS** | Yes (academic ASL weights) |
| LiTEN-FF | **PILOT_ONLY** (post-install) | Energy/forces OK on Fv (CPU); GPU OOM at Fv; full Dev extract not completed |
| BioEmu | **PILOT_ONLY** (post-pilot) | Chainwise VH pilot OK; full Dev×32×VH+VL not completed (compute) |
| CGSchNet / mlcg | **BLOCKED** | No clean two-chain Ab Fv CG mapping without inventing topology |

Secondary (UMA/OrbMol/MACE): **not started** unless first-wave blocked.

---

## 1. FeNNix-Bio1

| Item | Record |
|------|--------|
| Model/version | FeNNix-Bio1 **S** (`fennix-bio1S.fnx`) primary; M optional if VRAM allows |
| Paper | ChemRxiv / FeNNix-Bio1 (SPICE2-extended biomolecular MLIP) |
| Code repo | https://github.com/FeNNol-tools/FeNNol — **LGPLv3** |
| Weights | https://github.com/FeNNol-tools/FeNNol-PMC / HF fennol-tools/FeNNix-Bio1 — **ASL (Academic Software License)** |
| Academic/commercial | Weights: **academic non-commercial only**; commercial needs separate license |
| Domain | Organic / biological systems (SPICE2-like); energy & forces |
| Elements | Bio-organic set (H,C,N,O,S,P,halogens, metals in training — verify at runtime for Ab atoms) |
| Protein support | Yes (intended use) |
| Multichain | Atomistic system = all atoms in one calculation; VH+VL Fv OK as one molecular system |
| Max practical size | Fv ~200–250 residues / ~3500–4500 atoms: expect GPU seconds–tens of seconds per energy+force eval on RTX 3090 |
| Solvent | Vacuum / implicit as model-trained (no explicit water unless user adds waters) |
| H / protonation | PDB typically needs **hydrogens**; use fixed OpenBabel/PDB2PQR or ASE neighbor H-add with frozen protocol |
| PDB direct | Heavy-atom PDB → H-add → species+coords arrays |
| Energy comparability | Absolute E **not** comparable across Abs; use ΔE / forces under fixed perturbations |
| Classification | **READY_WITH_LIMITATIONS** (ASL; vacuum; Fab/Fv mismatch) |

---

## 2. LiTEN-FF

| Item | Record |
|------|--------|
| Model | LiTEN-FF foundation biomolecular FF (nablaDFT pretrain + SPICE finetune) |
| Paper | Nat Commun / arXiv:2507.00884 |
| Code | https://github.com/lingcon01/LiTEN-FF — **MIT**; LiTEN parent MIT; LiTENexus Apache-2.0 |
| Weights | Zenodo / repo-distributed Conda package (per upstream README) |
| Domain | Vacuum & solvated biomolecular; energy/forces/opt/FES claimed |
| Multichain | Atomistic — Fv as one system OK if install succeeds |
| Size | Claims speedup vs MACE-OFF on ~1000 atoms; Fv-scale should be feasible |
| Restrictions | Open MIT for code; confirm weight redistribution terms at download |
| Classification | **READY_WITH_LIMITATIONS** pending successful install of Conda/Zenodo package |

---

## 3. BioEmu (Microsoft)

| Item | Record |
|------|--------|
| Model | `bioemu-v1.1` (Science paper default) |
| Repo | https://github.com/microsoft/bioemu — pip `bioemu` / `bioemu[cuda]` |
| License | Code Apache-2.0 components + model cards on HF `microsoft/bioemu` (verify HF license at download) |
| Domain | **Protein monomers** → equilibrium ensemble from sequence |
| Multichain | **Officially unsupported.** Linker-scFv “trick” documented as poor; **BLOCK** as primary Fv protocol |
| Valid protocol here | **Chainwise VH and VL** separate sampling (32 conformers each if practical); aggregate VH/VL descriptors; **explicitly no interface dynamics** |
| H / PDB | Sequence-only input; structures are generated (not from ESMFold PDB) |
| Runtime | ~minutes per chain on 3090 for 32 samples (length ~110–130) |
| Classification | **READY_WITH_LIMITATIONS** (chainwise-only scientific framing) |

---

## 4. CGSchNet / mlcg (Nat. Chem. 2025)

| Item | Record |
|------|--------|
| Model | Transferable CGSchNet (5-bead/residue) |
| Ecosystem | https://github.com/ClementiGroup/mlcg (MIT); third-party FlashMD HF `pingzhili/cg-schnet` |
| Multichain / Fv | Training emphasized domains + peptide dimers; **two-chain antibody Fv applicability not guaranteed** |
| Risk | Requires CG mapping + priors; scientifically dubious if we invent Ab-specific CG topology |
| Classification | **PILOT_ONLY** — proceed only if official CG mapping for proteins applies cleanly to Fv without custom linker/topology invention; else **BLOCKED** |

---

## Shared limitations (all)

1. Fab experimental TmApp vs Fv structures.  
2. Predicted-structure (ESMFold/ABB2) geometry quality can dominate “stability” proxies.  
3. Absolute energies not cross-molecule comparable.  
4. No fine-tuning; no TmApp-driven perturbation tuning.

---

## Proceed order

1. FeNNix-Bio1S pilot (3 Abs)  
2. LiTEN-FF pilot  
3. BioEmu chainwise pilot  
4. CGSchNet only if mapping is clean  

Secondary UMA/OrbMol/MACE: **deferred**.

---

## Post-run status (after Gate F1 / extract)

| Model | Final class | Notes |
|-------|-------------|-------|
| FeNNix-Bio1S | **READY_WITH_LIMITATIONS** → full cohort scored | 324×2 SUCCESS; JAX CPU |
| LiTEN-FF | **PILOT_ONLY** | 3×2 feature SUCCESS on CPU; GPU OOM; no Dev TmApp scoring |
| BioEmu | **PILOT_ONLY** | Chainwise VH pilot only; no Dev scoring |
| CGSchNet | **BLOCKED** | Mapping unsupported without invention |

