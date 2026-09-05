# Fv vs Fab structure comparison (full cohort)

**N:** 324 / 324  
**Target-blind.** Reconstructed experimental-like Fab vs existing ESMFold Fv.

## Distributions (Å unless noted)

| Metric | Median | IQR | q90 | Max |
|--------|--------|-----|-----|-----|
| VH_backbone_RMSD | 0.587 | 0.381 | 1.294 | 3.549 |
| VL_backbone_RMSD | 0.271 | 0.166 | 0.569 | 1.234 |
| combined_Fv_backbone_RMSD | 0.576 | 0.292 | 1.091 | 24.725 |
| all_CDR_backbone_RMSD | 0.758 | 0.809 | 1.997 | 28.302 |
| HCDR3_backbone_RMSD | 0.499 | 1.252 | 2.403 | 4.596 |
| VH_VL_orientation_COM_diff_A | 0.501 | 0.279 | 0.870 | 57.369 |
| VL_RMSD_after_VH_align_A | 0.768 | 0.339 | 1.163 | 61.207 |
| VH_VL_interface_contact_delta | 0.000 | 3.000 | 3.000 | 14.000 |
| exposed_aromatic_SASA_delta | -30.546 | 79.000 | 41.464 | 595.516 |

## Scientific question

Does adding CH1/CL materially change the predicted Fv geometry?

**Answer:** **No (bulk).** Adding CH1/CL does not materially rewrite predicted Fv backbone geometry for the cohort: combined Fv CA RMSD median **0.58 Å**, IQR 0.29, q90 **1.09 Å**. VH/VL orientation COM shift after VH-align median **0.50 Å**. Interface contact Δ median **0.0**. Exposed aromatic SASA Δ (Fab Fv-portion − Fv) median **-30.5 Å²** (constant domains can bury some aromatics). One extreme outlier (ADI-47265, combined RMSD 24.7 Å) should be treated as QC failure, not biology.

## Notes

- Backbone RMSDs use CA atoms; joint Fv alignment for combined RMSD.
- Orientation metric uses BioPython Superimposer convention (`coord @ rot + tran`) after VH-only alignment.
- `exposed_aromatic_SASA_*`: Bio.PDB ShrakeRupley probe=1.4, n_points=100 (gate_b2 frozen params). Fab counts only VH/VL residues after SASA on the full Fab complex.
- Outliers (top combined RMSD): ADI-47265=24.72Å, ADI-47221=2.75Å, ADI-45469=2.35Å, ADI-47223=2.33Å, ADI-47125=2.26Å.
