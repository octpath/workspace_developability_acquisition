# HIC STATIC_SAP_KD Report (H094–H101)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- descriptor: **Fv STATIC_SAP_KD** (antibody-level only; not MD Chennamsetty SAP)
- freeze: `H094_H101_PRE_EXTERNAL_FREEZE.yaml`
- prereg: `H094_H101_STATIC_SAP_KD_PREREGISTRATION.yaml`

## Master table

| Code | Backbone | Aux | Aux dim | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |
|------|----------|-----|--------:|-------:|-------:|----------:|-----------:|----:|-----:|--------:|
| EXP-H071 | H071 Scratch | NONE | 0 | 0.5284 | 0.4749 | 0.5017 | 0.5284 | 0.4796 | 0.4800 | 0.4798 |
| EXP-H090 | H071 | SURFACE | 35 | 0.4576 | 0.4855 | 0.4715 | 0.4855 | 0.4059 | 0.4129 | 0.4094 |
| EXP-H093 | H071 | F4 | 200 | 0.4984 | 0.5059 | 0.5021 | 0.5059 | 0.4070 | 0.4215 | 0.4142 |
| EXP-H094 | H071 | SAP3 | 3 | 0.5202 | 0.5005 | 0.5104 | 0.5202 | 0.4910 | 0.4893 | 0.4902 |
| EXP-H095 | H071 | SAP9 | 9 | 0.5212 | 0.5288 | 0.5250 | 0.5288 | 0.5220 | 0.4660 | 0.4940 |
| EXP-H096 | H071 | SURFACE+SAP9 | 44 | 0.4806 | 0.4807 | 0.4807 | 0.4807 | 0.4163 | 0.4233 | 0.4198 |
| EXP-H097 | H071 | F4+SAP9 | 209 | 0.4759 | 0.4933 | 0.4846 | 0.4933 | 0.4071 | 0.4269 | 0.4170 |
| EXP-H061 | H061 ESM2 | NONE | 0 | 0.5331 | 0.4805 | 0.5068 | 0.5331 | 0.4816 | 0.4615 | 0.4716 |
| EXP-H086 | H061 | SURFACE | 35 | 0.4714 | 0.5012 | 0.4863 | 0.5012 | 0.4018 | 0.4109 | 0.4063 |
| EXP-H089 | H061 | F4 | 200 | 0.4717 | 0.4678 | 0.4698 | 0.4717 | 0.4018 | 0.4392 | 0.4205 |
| EXP-H098 | H061 | SAP3 | 3 | 0.5491 | 0.5041 | 0.5266 | 0.5491 | 0.4954 | 0.4513 | 0.4734 |
| EXP-H099 | H061 | SAP9 | 9 | 0.5520 | 0.5074 | 0.5297 | 0.5520 | 0.5311 | 0.4661 | 0.4986 |
| EXP-H100 | H061 | SURFACE+SAP9 | 44 | 0.4747 | 0.4740 | 0.4743 | 0.4747 | 0.4134 | 0.4050 | 0.4092 |
| EXP-H101 | H061 | F4+SAP9 | 209 | 0.4682 | 0.4898 | 0.4790 | 0.4898 | 0.4163 | 0.4265 | 0.4214 |

## Explicit answers

1. **SAP3 improve Scratch?** No. H094 TEST_mean 0.510 > H071 0.502 (worse). Overall also worse (0.490 vs 0.480).
2. **SAP3 improve ESM-2?** No. H098 0.527 > H061 0.507 (worse).
3. **SAP3 vs SURFACE?** Much worse. Scratch 0.510 vs 0.472; ESM 0.527 vs 0.486. Primary bootstrap H094−H090 CI excludes 0 on TEST_P (SAP worse).
4. **SAP9 vs SAP3?** No improvement (H095/H099 both worse than SAP3).
5. **SAP on top of SURFACE?** Not complementary by the batch criterion. Scratch H096 (0.481) worse than H090 (0.472). ESM H100 TEST_mean improves vs H086 (0.474 vs 0.486) but Overall does not (0.409 vs 0.406); Primary direction mixed.
6. **SAP on top of F4?** Mixed. Scratch H097 improves TEST vs H093 (0.485 vs 0.502) with both-scheme bootstrap direction negative (favor H097) but CIs include 0. ESM H101 does not beat H089 (0.479 vs 0.470).
7. **Permutation — does the model use SAP?** Mildly yes when alone (ΔMAE ~0.01–0.02). With SURFACE present, SURFACE ΔMAE (~0.12–0.14) dominates; SAP ΔMAE remains small (~0.01–0.02). Under F4, SAP ΔMAE ≈ 0.00–0.01 (near ignored).
8. **Redundant with HYDRO_FIELD?** Not strongly. Max |Pearson| ≈ 0.47 (MEAN/SUM); MAX only ≈ 0.17.
9. **SSKD_SUM size/SASA proxy?** Partial, not dominant. |ρ| vs n_residues ≈ 0.37; vs total_SASA ≈ 0.19.
10. **More valuable for Scratch than ESM-2?** Neither benefits from SAP-alone; no clear Scratch advantage.
11. **Reported “pseudo-SAP highly informative for HIC” reproduce?** **No** under this Fv STATIC_SAP_KD + antibody-level late-fusion protocol. SURFACE remains the strong physical signal.
12. **Justify residue-level SAP next?** **No** as a priority. Antibody-level signal is weak/non-reproducing; residue injection is not justified by this batch. Optional later only if a different definition (e.g. MD SAP, Fab scope) is motivated independently.

## Verdict

**not supported** (for the reported claim that antibody-level pseudo-SAP is highly informative for HIC here).

SURFACE remains preferred. STATIC_SAP_KD is not a SURFACE replacement; complementarity evidence is weak/inconsistent.

## Definition (frozen)

`SSKD_i(R) = Σ_j I[d(centroid_i,centroid_j) ≤ R] · KD_norm(j) · clip(SASA_j / Tien_MaxASA_j, 0, 1)`

- structure: ESMFold **Fv** (`STRUCTURE_INPUT_CROSSWALK_v2` `esmfold_canonical_path`)
- R_REF = 5.0 Å (SOURCE_SPECIFIED from STATIC-SAP FEATURE_SPEC `radii_A[0]`)
- Shrake–Rupley probe=1.4, n_points=100 (SOURCE_SPECIFIED)
- side-chain centroid; Gly→CA; include self; no decay / pLDDT

## Artifacts

- features: `static_sap_kd_residue.parquet`, `..._antibody_global3.parquet`, `..._antibody_chain9.parquet`
- QC / corr / boot / perm / external CSVs under `results/`
- design note only: `GENERALIZED_SPATIAL_PROPERTY_AGGREGATION_IDEAS.md`

## Next HIC code

`EXP-H102` — **DO NOT RUN** in this batch.
