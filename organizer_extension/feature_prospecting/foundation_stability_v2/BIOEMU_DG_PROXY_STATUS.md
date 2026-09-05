# BioEmu folding-free-energy-like proxy

**Decision: BIOEMU_DG_PROXY_NOT_SCIENTIFICALLY_JUSTIFIED** for isolated antibody VH/VL in this study.

## Reasons (target-blind)

1. Official `bioemu-benchmarks` / `folding_free_energies` protocols are built around proteins with validated folding ΔG experimental sets and defined folded/unfolded references — not antibody variable domains.
2. Our BioEmu protocol is **chainwise monomer VH or VL**. Fab TmApp is a different physical object (multi-domain, interface, constant regions).
3. Using ESMFold/ABB2 chain structures as “native” would make any ΔG-like score **structure-reference-dependent** and confounded with generator artifacts — the failure mode v2 is trying to escape.
4. Therefore primary BioEmu evidence remains **ensemble descriptors only**. No forced ΔG formula.

