# BioEmu Constant-Domain Context — Target-Blind Spec

**SPEC_FROZEN_BEFORE_TARGET_SCORING = true**  
**Evidence:** ORGANIZER-EXPLORATORY  
**Target:** TmApp only (HIC not optimized)  
**Parent (read-only):** `foundation_stability_v2/` (isolated VH/VL BioEmu-v1.2)  
**Sequences/structures (read-only):** `fab_reconstruction/`

---

## Biological framing (frozen)

| Arm | Input | Label | Role |
|-----|-------|-------|------|
| LIGHT PRIMARY | `VL+CL` (κ→Cκ / λ→Cλ) | genuine monomeric light chain | PRIMARY constant-context test |
| HEAVY SECONDARY | `VH+CH1(+EPKSC stub)` | `UNPAIRED_HEAVY_CONTEXT` | diagnostic only — **not** assembled Fab |

- BioEmu = **monomer** sampler. No native two-chain Fab. No linker-scFv primary.
- Unpaired CH1 folding is biologically coupled to CL; VH+CH1 ≠ Fab heavy dynamics.

## Model / sampling

- Checkpoint: `bioemu-v1.2` (same env as v2)
- Filter: physicality filter **ON**
- Convergence Abs: 12 target-blind (κ/λ, length, germline; **no** TmApp/HIC/errors)
- N candidates: 16 / 32 / 64 separately for LIGHT_CHAIN and UNPAIRED_HEAVY
- Rule: smallest N with median Spearman vs N=64 ≥ 0.95 AND median NAD ≤ 10%; else 64
- MSA: local single-sequence a3m of full reconstructed chain (document); ColabFold only if needed

## Alignment frames (both required)

- `LOCAL_VARIABLE_ALIGNMENT` — align on VL (or VH) CA only, then measure V internals
- `WHOLE_CHAIN_ALIGNMENT` — align full VL+CL / VH+CH1

## Feature families (frozen before scoring)

**LIGHT (≤30 dims combined; ≤15 each):**  
`L1_FULL_LIGHT`, `L2_VL_IN_CONTEXT`, `L3_CONTEXT_DELTA`, `L4_VL_CL_COUPLING`, `L5_ESMFOLD_NATIVE_LIKENESS`

**HEAVY secondary:**  
`H1_UNPAIRED_VH_CONTEXT_DELTA`, `H2_UNPAIRED_CH1_NATIVE_LIKENESS`, `H3_UNPAIRED_VH_CH1_COUPLING`

## Evaluation (after freeze)

Standalone / Incremental (vs AbLang2; vs isolated BioEmu) / Residual; B=10000 paired bootstrap; Public/Private post-reveal only.

## Forbidden

AbLingua, other PLMs, Optuna, fine-tune, disable filtering for quotas, pseudo-scFv primary, call VH+CH1 “Fab”, call ESMFold-Q “true folded fraction”, Public/Private tuning.

## Gate B0.5 (mandatory before full cohort)

See `BIOEMU_LOW_PASS_DIAGNOSTIC.md`.

**Decision:** `BIOEMU_EXTENDED_CHAIN_UNRESOLVED`  
→ **Do NOT run full 324 LIGHT** until a human-approved protocol change.  
→ Do NOT treat ~5–10% physical survivors as an unbiased equilibrium ensemble.  
→ Physical filter stays ON. Official `physical_steering.yaml` was tested (pass↑, ensemble validity unresolved).
