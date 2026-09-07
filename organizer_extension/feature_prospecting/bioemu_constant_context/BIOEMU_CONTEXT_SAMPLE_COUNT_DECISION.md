# BioEmu context sample-count decision

**SPEC_FROZEN_BEFORE_TARGET_SCORING**

| Arm | Frozen N |
|-----|----------|
| LIGHT_CHAIN (VL+CL) | **16** |
| UNPAIRED_HEAVY (VH+CH1) | **16** |

Rule: smallest of {16,32} with median Spearman vs N=64 ≥ 0.95 and median NAD ≤ 0.10; else 64.

See `BIOEMU_CONTEXT_SAMPLE_CONVERGENCE.csv`.

Model: bioemu-v1.2; MSA: singleseq a3m of reconstructed Fab chain; filter_samples=ON.
