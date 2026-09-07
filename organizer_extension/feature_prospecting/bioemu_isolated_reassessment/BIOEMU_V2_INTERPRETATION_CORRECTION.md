# BioEmu-v1.2 (foundation_stability_v2) interpretation correction

**Does not rewrite prior artifacts.** Historical CSVs/reports in `foundation_stability_v2/` remain as-is.

## Previous wording (incorrect as physical-ensemble claim)

Prior reporting treated **requested / raw NPZ N = 16** as if it were the usable ensemble size, and stated that **N=16 had “converged”** for isolated VH/VL descriptors.

## What was actually used

| Quantity | Meaning |
|----------|---------|
| v2 `sample_log.valid` / “648/648” | Unfiltered NPZ presence (`count_samples_in_output_dir`) |
| Feature trajectories | Official `filter_samples=True` → `samples.xtc` |
| Median physical frames (cohort) | **~5** (pass rate median **~0.31**) |

Therefore:

- **“N=16 converged” does NOT establish convergence of the physical filtered ensemble.**
- Predictive results in v2 remain historically valid as: *prediction using descriptors computed from whatever filtered frames existed after conversion*.
- Ensemble-estimation precision was **weaker than stated**; contact occupancy especially is coarse at ~5 frames.

## Reassessment response

See `bioemu_isolated_reassessment/`:

- Pilot oversample to **≥64 physical frames**
- Target-blind MC convergence → frozen **Nphys = 8** (VH and VL)
- Full-cohort resample to Nphys≥8 and re-score TmApp under frozen Primary/Shadow CV

Final state target: `ORGANIZER_TMAPP_BIOEMU_ISOLATED_REASSESS_COMPLETE`
