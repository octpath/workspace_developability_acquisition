# ESMFOLD FAB PILOT REPORT

**Verdict:** `PASS_FULL_COHORT`  
**N:** 12  
**Success (length-matched):** 12

## Summary stats (successful)

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| global_pLDDT | 85.9 | 82.8 | 87.8 |
| VH_pLDDT | 87.0 | 81.7 | 90.1 |
| VL_pLDDT | 87.2 | 82.7 | 90.4 |
| CH1_pLDDT | 83.2 | 81.8 | 84.8 |
| CL_pLDDT | 86.1 | 85.1 | 87.3 |
| CH1_CL_contacts_8A | 37.9 | 32 | 46 |
| VH_VL_contacts_8A | 40.8 | 32 | 52 |
| H-L SS distance (Å) | 2.86 | 2.32 | 3.34 |
| severe_clash_count | 0.0 | 0 | 0 |

## Architecture flags

       id architecture_flags  com_distance_A  Fv_const_com_distance_A
ADI-47109                          17.608907                34.440369
ADI-47068                          17.621153                35.287408
ADI-45497                          17.613654                35.619040
ADI-47189                          17.677108                35.444310
ADI-47280                          17.628165                35.157794
ADI-45447                          17.593564                34.951150
ADI-45405                          17.659099                35.380606
ADI-47092                          17.624448                36.329061
ADI-47153                          17.898389                35.256624
ADI-45428                          17.904279                35.690080
ADI-45389                          17.918751                34.484614
ADI-47196                          17.447585                35.826153

## Interpretation

- Structures are **reconstructed experimental-like Fabs**, not exact Shehata experimental sequences.
- ESMFold does not enforce disulfides; H–L SG distances are QC only.
- Verdict rule is target-blind (geometry/confidence only).
