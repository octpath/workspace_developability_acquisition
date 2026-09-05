# BioEmu MSA fallback (target-blind)

For Abs without precomputed ColabFold embeds, a3m was replaced by **single-sequence** MSA to make full-cohort N=16 sampling computationally feasible.

Convergence Abs (12) used Boltz-derived MSA (depth ≤256) and remain the basis for N freeze.

Primary scientific framing: ensemble descriptors from bioemu-v1.2; MSA depth heterogeneity documented.
