# BioEmu Constant-Context Validity Audit

**Evidence:** ORGANIZER-EXPLORATORY  
**Updated after:** Gate B0.5 (`BIOEMU_LOW_PASS_DIAGNOSTIC.md`)

---

## Classifications（final for this gate）

| Arm | Class |
|-----|-------|
| LIGHT (VL+CL) | **`TECHNICALLY_UNSTABLE`** for equilibrium-ensemble claims — sampling runs, but physical-pass ≈9% with strong interdomain-clash selection bias. See Gate B0.5 decision **`BIOEMU_EXTENDED_CHAIN_UNRESOLVED`**. |
| HEAVY (VH+CH1) | **`SECONDARY_DIAGNOSTIC`** — not promoted; unpaired CH1 biology + likely similar/worse multi-domain filter issues. Full HEAVY cohort not justified while LIGHT unresolved. |

---

## Checklist answers

1. **VL+CL technically/structurally coherent?**  
   Partially: jobs complete; filtered survivors exist. **Not** coherent as a representative physical ensemble (Gate B0.5).

2. **Does CL alter VL ensemble stats?**  
   Not scientifically scoreable yet — pass-biased survivors cannot support target scoring.

3. **VH+CH1 CH1 instability?**  
   Pilot npz=64 completed earlier; physical keep similarly low. Not expanded pending LIGHT resolution.

4. **Heavy diagnostic-only?**  
   **Yes** (design + validity).

5. **Rejection rates acceptable?**  
   **No** for primary TmApp feature use (~90% reject; mostly interdomain clash).

---

## Gate B0.5 one-line

Low pass is **multi-domain-specific**, **clash-dominated**, **selection-biased**; official physical steering helps pass rate but does not resolve ensemble validity → **do not run full 324 LIGHT**.
