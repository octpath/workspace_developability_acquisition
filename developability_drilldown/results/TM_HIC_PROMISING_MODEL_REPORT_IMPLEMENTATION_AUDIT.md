# TM_HIC_PROMISING_MODEL_REPORT — Implementation Audit

**Status:** COMPLETE (FINAL issued)  
**HEAD:** `df1da85339a67599018ed0a1a902afc8fe56b2c2`  
**Draft audited:** `TM_HIC_PROMISING_MODEL_REPORT_DRAFT.md` (Draft 0.9)  
**Output:** `results/TM_HIC_PROMISING_MODEL_REPORT_FINAL.md`  
**Scope:** documentation fact-check only (no training / no new EXP / no feature regen)

---

## Checked configs

- `experiments/configs/EXP-T113.yaml`
- `experiments/configs/EXP-T121.yaml`
- `experiments/configs/EXP-H061.yaml`
- `experiments/configs/EXP-H086.yaml`
- `experiments/configs/EXP-H090.yaml`
- `experiments/configs/EXP-H085.yaml`
- `experiments/configs/EXP-H089.yaml`
- `experiments/features/EXP-H047.parquet` (column slices)
- `results/experiments.csv`

## Checked source paths

| Topic | Path | Function / notes |
|---|---|---|
| AROMATIC_TOPO | `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py` | `aromatic_features` L58–111 |
| SASA/RASA/patch consts | `.../common/structure_utils.py` | PROBE/N_POINTS/RASA/PATCH_R/LOCAL_R/MAX_ASA/AROMATIC |
| ARO SPEC | `.../AROMATIC-TOPO/FEATURE_SPEC.json` | 15 canonical + 4 QC |
| HYDRO_FIELD | `.../common/hydro_surface.py` | `build_surface_and_field`, `hydro_summaries`, `connected_components` |
| HYDRO SPEC | `.../HYDRO-FIELD/FEATURE_SPEC.json` | 14 canonical |
| phi_finite_frac | `.../scripts/extract_hydro_copatch_batch2.py` | APBS QC colocated |
| ARCH-3/4/6/7 | `developability_drilldown/models/antibody_transformer/model.py` | encode_* / mask builders |
| Late fusion | `.../late_fusion.py` | `LateFusionAuxMLP`, `LateFusionModel` |
| H047 aux | `.../h047_aux_features.py` | F1–F4 slices + TRAIN-only preprocess |
| AbLang2 extract | `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py` | paired empty-partner |
| AbLang2 meta | `top_models_feature_bundle/residue_level/ablang2/metadata.json` | ablang2==0.2.1, dim 480 |
| ARCH-4 mask test | `tests/test_chain_specific_dual_reg.py` | True=blocked convention |

---

## AROMATIC_TOPO — 19/19

Exact columns listed in FINAL §10.8. Parquet `aro_*` set matches generator output.

**Local 10 Å neighbor set:** exposed aromatic only (draft was uncertain; now fixed).

**CDR fraction:** numerator = CDR∩exposed-arom SASA; denominator = all exposed-arom SASA.

**Empty / singleton:** empty → 0; singleton is a size-1 patch counted in `aromatic_patch_count`.

## HYDRO_FIELD — 16/16

Exact columns in FINAL §11.8. Draft field equation / FreeSASA / Fibonacci / Fauchère / q0.80 / link 2.0 match code.

**QC boundary:** #1–14 canonical; #15 `n_surface_points`; #16 `phi_finite_frac` (APBS; not hydrophobicity).

**Largest patch:** max vertex-count CC (not max area), then area fraction of that CC.

**SPEC wording mismatch (not draft):** FEATURE_SPEC text about equal area weights vs code using FreeSASA-derived `areas` — **code is authoritative**; FINAL follows code.

## ARCH-3 — exact

Padding-only unrestricted confirmed (`attn_mask=None`, `src_key_padding_mask` only).  
Pre-LN, MEAN, head layers as FINAL §4.2 table.

## ARCH-7 — exact

Draft conceptual residual matched; details filled:

- cross-attn **after** full shared encoder
- **shared** `self.cross_attn` for both directions
- **no** LN / FFN / gate after cross
- REG-only query; opposite-chain residues only as K/V

## ARCH-4 — exact

4×4 matrix in FINAL §14. Mask polarity verified against PyTorch (True=blocked).

## AbLang2 residue — exact

Distinct from historical seqcoding. Empty partner per chain. Raw 480 → Linear 128.

## Late fusion — exact

Aux MLP has **no LayerNorm** (draft prereg suggested possible LN; production has none).  
F1 order ARO→HYDRO. F4: impute→PCA(FB)→scale per block; no global PCA; no ESM2_H.

## Score registry consistency

V3 Transformer `cv_*` = OOF TEST (`cv_protocol=dl_foldlocal_cosine_v3_oof_test`).  
H047 uses classical protocol (`canonical_simple_tvt_primary_shadow`).

| Code | cv_P | cv_S | cv_mean | Public | Private | Overall |
|---|---:|---:|---:|---:|---:|---:|
| EXP-T113 | 2.995055 | 3.283071 | 3.139063 | 3.105442 | 2.999122 | 3.052282 |
| EXP-T121 | 3.023221 | 3.252085 | 3.137653 | 3.247434 | 3.193637 | 3.220536 |
| EXP-T096 | 3.252413 | 3.263832 | 3.258122 | 3.618574 | 3.460298 | 3.539436 |
| EXP-H047 | 0.452878 | 0.443979 | 0.448429 | 0.407578 | 0.447367 | 0.427473 |
| EXP-H054 | 0.502151 | 0.512973 | 0.507562 | 0.483359 | 0.462787 | 0.473073 |
| EXP-H061 | 0.533068 | 0.480523 | 0.506795 | 0.481635 | 0.461520 | 0.471577 |
| EXP-H071 | 0.528430 | 0.474947 | 0.501688 | 0.479571 | 0.480027 | 0.479799 |
| EXP-H085 | 0.472874 | 0.488744 | 0.480809 | 0.404951 | 0.428063 | 0.416507 |
| EXP-H086 | 0.471437 | 0.501216 | 0.486326 | 0.401757 | 0.410931 | 0.406344 |
| EXP-H089 | 0.471707 | 0.467812 | 0.469759 | 0.401818 | 0.439192 | 0.420505 |
| EXP-H090 | 0.457571 | 0.485454 | 0.471512 | 0.405876 | 0.412875 | 0.409375 |

Draft rounded tables match registry within rounding (no material numeric mismatch).

---

## Mismatches corrected in FINAL

| Draft statement | Actual implementation | Evidence | Material? |
|---|---|---|---|
| 10 Å local SASA neighbor set unspecified | **exposed aromatic only** | `aromatic_features` L82–89 | Yes (definition) |
| ARCH-7 LN/FFN/share unspecified | shared MHA; **no** LN/FFN after cross; ungated residual | `model.py` L820–829 | Yes (equation) |
| possible LayerNorm in aux branch | **no** LayerNorm in `LateFusionAuxMLP` | `late_fusion.py` L23–29 | Yes (architecture) |
| AbLang2 API / seqcoding ambiguity | residue AA-only 480-d ≠ seqcoding | metadata + extract script | Yes (provenance) |
| ARCH-4 mask not tabulated | 4×4 matrix filled | `build_chain_specific_dual_reg_attn_mask` | Documentation |
| YAML vs run param counts | run = YAML +9216 (dynamic emb sizes) | T113/T121 summary.json | Minor |

Scientific narrative (Tm vs HIC levers; SURFACE value; capacity failure) left unchanged.

---

## Unresolved items

**None.** FINAL issued (not REVIEW_DRAFT_1).

Optional non-blocking notes (not unresolved for report claims):

- YAML `n_trainable_preregistered` vs run counts differ by embedding vocabulary length (+9216).
- HYDRO FEATURE_SPEC prose on area weighting differs from code; FINAL follows code.
