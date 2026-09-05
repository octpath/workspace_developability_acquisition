# Foundation Stability V2 — Target-Blind Spec

**SPEC_FROZEN_BEFORE_TARGET_SCORING = true**  
**Evidence:** ORGANIZER-EXPLORATORY  
**Target:** TmApp only (HIC not optimized)  
**Scope:** Fab experimental TmApp vs Fv (FeNNix) or isolated VH/VL (BioEmu)  
**PLM reference (frozen):** `ablang2__HL_paired` → fold-local StandardScaler → PCA32 → Ridge α∈{0.1,1,10,100} nested  
**Folds:** `cv_primary.csv` / `cv_shadow.csv`  
**Do not:** AbLingua, other PLMs, Optuna, Public/Private protocol tuning

---

## Branch A — FeNNix relaxed curvature

| Item | Frozen choice |
|------|----------------|
| Model | FeNNix-Bio1S (`fennix-bio1S.fnx`, same as v1) |
| Structures | ESMFold + ABodyBuilder2 Fv |
| Prep | Reuse v1 `pdb2pqr --ff=AMBER --keep-chain` after audit |
| R0 | Prepared, no relax (diagnostic) |
| R1 (PRIMARY) | Side-chain + H free; fix N,CA,C,O; FIRE; fmax=0.10 eV/Å; max_steps=200 |
| R2 | Not primary (no unconstrained whole-protein min) |
| Primary perturbation | Symmetric torsional: φ/ψ ±2°; χ1 ±5° |
| Sites | ~8 FW + ~8 CDR + ≤4 HCDR3 backbone; ≤12 χ1; deterministic hash |
| Feature | \(K_E=(E_++E_--2E_0)/\delta^2\) aggregates ≤25 dims |
| Units | eV / rad² |

## Branch B — BioEmu-v1.2 chainwise

| Item | Frozen choice |
|------|----------------|
| Model | `bioemu-v1.2` (fallback: newest accessible official, document) |
| Protocol | VH and VL **separately**; no linker-scFv primary |
| Convergence Abs | 12 target-blind (length / germline span, not TmApp) |
| N candidates | 16 / 32 / 64; rule: median Spearman vs N=64 ≥0.95 AND median NAD ≤10% → smallest such N; else 64 |
| Cohort | All 324 VH + 324 VL at frozen N |
| Features | ≤30 dims (spread, RMSF, Rg, contacts, SS if robust, CDR/FW region) |
| ΔG proxy | Only if official bioemu-benchmarks transferable; else NOT_SCIENTIFICALLY_JUSTIFIED |

## Evaluation (after freeze)

Standalone / Incremental (concat vs AbLang2) / Residual (leakage-safe); B=10000 paired bootstrap; Public/Private post-reveal only.
