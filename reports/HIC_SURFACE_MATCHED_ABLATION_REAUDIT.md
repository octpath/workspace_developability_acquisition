# HIC SURFACE Matched Ablation Re-Audit (H102–H113)

**STATUS: SURFACE_MATCHED_ABLATION_REAUDIT_COMPLETE**

| Field | Value |
|-------|--------|
| Diagnosis HEAD | `4a2568b454e0b039a5566576b50768e7c0595938` |
| New training | **None** |
| Contrast definition | `SURFACE+HSP − HSP-only` (Δ < 0 = SURFACE improves) |
| Pairs | 6 (Scratch×3 + ESM2×3) |
| Bootstrap | antibody-level paired, N=10000, seed=101 |
| HIGH-tail | HIC > 11.5 only |
| Provenance | **retrospective_historical** (not pristine holdout) |

## B1. Config equality audit

YAML configs for each pair differ **only** in identity fields and `fusion_bundle_id` / `input_space`
(HSP-only vs SURFACE+HSP). Shared across pairs: platform `DL_FOLDLOCAL_COSINE_V3`, seed 101,
AdamW + lr_grid, SmoothL1, patience 30, max_epochs 200, annotation FULL, pooling REG,
`late_concat_aux32`, same backbone within representation (Scratch←H071 / ESM2←H061),
same HSP family within each pair.

**Inevitable non-YAML differences (treatment-induced):**
- aux_dim **3 → 38** (F1_SURFACE 35D + HSP3)
- AuxMLP first-layer parameter count scales with aux_dim
- Fold-wise selected LR / early-stop epoch trajectories (same protocol, different features)
- `config_hash` / `init_hash`

| pair | match_class | notes |
|------|-------------|-------|
| Scratch_P1_BM_R5 | **NEAR_MATCH_WITH_DIFFERENCES** | EXP-H102→EXP-H103; HSP=P1 |
| Scratch_P2_FP_R5 | **NEAR_MATCH_WITH_DIFFERENCES** | EXP-H104→EXP-H105; HSP=P2 |
| Scratch_P3_EIS_R8 | **NEAR_MATCH_WITH_DIFFERENCES** | EXP-H106→EXP-H107; HSP=P3 |
| ESM2_P1_BM_R5 | **NEAR_MATCH_WITH_DIFFERENCES** | EXP-H108→EXP-H109; HSP=P1 |
| ESM2_P2_FP_R5 | **NEAR_MATCH_WITH_DIFFERENCES** | EXP-H110→EXP-H111; HSP=P2 |
| ESM2_P3_EIS_R8 | **NEAR_MATCH_WITH_DIFFERENCES** | EXP-H112→EXP-H113; HSP=P3 |

**Validity counts:** STRICT_MATCH=0, NEAR_MATCH_WITH_DIFFERENCES=6, NOT_VALID_MATCH=0.

All 6 pairs are retained as **valid near-matches** for SURFACE addition on fixed HSP.
They are **not** pure ‘no-physics vs SURFACE’ contrasts: the no-surface arm already carries HSP3.

## B2. SURFACE feature provenance

In H102–H113, `SURFACE` = canonical **F1_SURFACE** (35D) loaded via `H047AuxFeatureStore`,
concatenated with HSP3 in `HspPromotedAuxFeatureStore`
(`developability_drilldown/models/antibody_transformer/hsp_promoted_aux.py`).

| Item | Value |
|------|-------|
| Composition | ARO19 + HYDRO16 = **35D** |
| Source matrix | `experiments/features/EXP-H047.parquet` slices (ARO‖HYDRO) |
| Standalone | `top_models_feature_bundle/data/aromatic_topo.parquet` + `hydro_field.parquet` |
| Structure | ESMFold Fv (`STRUCTURE_INPUT_CROSSWALK_v2` → `esmfold_canonical_path`) |
| Aromatic | exposed FWY counts/SASA/RASA, CDR exposure, aromatic patches (RASA≥0.20 / 0.50) |
| Hydrophobic | FreeSASA Lee–Richards surface-field summaries (Fauchère–Pliska H(s)) |
| SASA/RASA | Yes (Shrake–Rupley residue SASA/RASA; FreeSASA surface points) |
| Antibody scope | Fv (H+L) |
| Preprocessing | **TRAIN-fold-only** median impute + StandardScaler per logical block |
| Fold-local | Scalers fit on train fold ids only (`FoldPreprocessor`) |
| Target leakage in features | Features are structure/sequence physicochemical — **no HIC label in feature construction** |
| Audit twin | `results/F1_SURFACE_RESIDUE_RECONSTRUCTION_AUDIT.md` (PASS-EXACT) |

HSP arm (held fixed within pair): antibody-level B3 aggregations (ALL_FV MAX/MEAN/SUM),
3D from promoted spatial-hydrophobicity families (source commit `210a270d`).

## B3. Matched effect table (Δ = surface − no_surface)

| pair | ΔCV_P | ΔCV_S | ΔCV_mean | ΔPub | ΔPriv | ΔTest | Δ|Pub−Priv| |
|------|-------|-------|----------|------|-------|-------|-------------|
| Scratch_P1_BM_R5 | -0.0368 | +0.0277 | -0.0045 | -0.0737 | -0.0133 | -0.0435 | -0.0376 |
| Scratch_P2_FP_R5 | -0.0419 | -0.0092 | -0.0255 | -0.0260 | -0.0322 | -0.0291 | -0.0062 |
| Scratch_P3_EIS_R8 | -0.0213 | -0.0368 | -0.0290 | -0.0982 | -0.0508 | -0.0745 | -0.0383 |
| ESM2_P1_BM_R5 | +0.0002 | +0.0009 | +0.0005 | -0.0517 | -0.0238 | -0.0378 | -0.0218 |
| ESM2_P2_FP_R5 | +0.0043 | +0.0307 | +0.0175 | -0.0090 | -0.0120 | -0.0105 | -0.0030 |
| ESM2_P3_EIS_R8 | -0.0228 | +0.0186 | -0.0021 | -0.0716 | -0.0326 | -0.0521 | -0.0098 |

CSV: `reports/HIC_SURFACE_MATCHED_ABLATION_PAIRS.csv`

## B4. Aggregate effects (valid near-matches = 6)

### CV mean
- mean Δ = **-0.007197**, median = **-0.003325**, improve = **4/6**
- Scratch: mean Δ=-0.019693, improve=3/3
- ESM2: mean Δ=+0.005298, improve=1/3

### Test Overall (retrospective)
- mean Δ = **-0.041242**, median = **-0.040641**, improve = **6/6**
- Scratch: mean Δ=-0.049026, improve=3/3
- ESM2: mean Δ=-0.033457, improve=3/3

Registry recomputation check: max abs registry−recompute on Test = 2.776e-16.

## B5. Antibody-level paired bootstrap

N_BOOT=10000, seed=101. Metric = mean(AE_surface − AE_no_surface).

| pair | Primary mean [CI] | Shadow mean [CI] | Test mean [CI] (retrospective) |
|------|-------------------|------------------|--------------------------------|
| Scratch_P1_BM_R5 | -0.0368 [-0.0992,+0.0248] | +0.0277 [-0.0304,+0.0893] | -0.0435 [-0.0892,+0.0007] |
| Scratch_P2_FP_R5 | -0.0419 [-0.1037,+0.0183] | -0.0092 [-0.0827,+0.0640] | -0.0291 [-0.0777,+0.0173] |
| Scratch_P3_EIS_R8 | -0.0213 [-0.0720,+0.0297] | -0.0368 [-0.0916,+0.0192] | -0.0745 [-0.1197,-0.0319] |
| ESM2_P1_BM_R5 | +0.0002 [-0.0609,+0.0625] | +0.0009 [-0.0631,+0.0661] | -0.0378 [-0.0840,+0.0088] |
| ESM2_P2_FP_R5 | +0.0043 [-0.0516,+0.0580] | +0.0307 [-0.0329,+0.0991] | -0.0105 [-0.0496,+0.0283] |
| ESM2_P3_EIS_R8 | -0.0228 [-0.0746,+0.0280] | +0.0186 [-0.0431,+0.0818] | -0.0521 [-0.0993,-0.0075] |

CSV: `reports/HIC_SURFACE_MATCHED_ABLATION_BOOTSTRAP.csv`

## B6. HIGH-tail rescue (HIC > 11.5)

- n_high on Test = **7** (small; no statistical claim)
- mean ΔMAE_high = **-0.5168**
- mean Δ signed(pred−true)_high = **+0.5544** (positive ⇒ less underprediction)
- mean ΔMAE_nonHIGH = **-0.0198**

**Answer:** HIGH-tail appears to receive a **larger** MAE reduction than non-HIGH on average, but n_high is tiny — diagnostic only.

## B7. Improvement consistency

- Test improve: **6/6**
- Public **and** Private improve: **6/6**
- Primary **and** Shadow improve: **2/6**
- Promoted families with Test improve on **both** Scratch and ESM2: **3/3**

**Surface evidence classification:** `PARTIAL_MATCHED_SUPPORT`

## B8. Historical-selection contamination

### HSP family selection (P1/P2/P3)
- Source: `feature_research/hic_spatial_hydrophobicity/results/PROMOTION_RECOMMENDATION.md`
- Explicitly cites **alone TEST** / external diagnostics when classifying promote candidates
  (e.g. BM-R5 “Best alone TEST (~0.483)”).
- Therefore P1/P2/P3 **family choice is retrospective / Test-informed** relative to a pristine holdout.
- Stage screens also recorded Public/Private as post-hoc diagnostics (`run_screen.py`).

### SURFACE (F1_SURFACE) family
- Predates H102–H113 as the H047 / H090 / H086 late-fusion SURFACE block.
- Historical project already used Test rankings for surfaceish models (retrospective).
- Feature construction itself does not include the HIC label; contamination risk is **selection**, not label leakage into X.

### Matched ablation strength vs holdout strength
- **Internal matched comparison strength:** high for protocol near-match (SURFACE add-on with HSP fixed).
- **Holdout evidence strength:** **not** a fully unused holdout — families and historical SURFACE lineage
  were selected with Test-aware processes. Treat external deltas as **retrospective association**.

## B9 / C. Next-experiment decision

### `C2` — `PROSPECTIVE_SURFACE_REPLICATION_RECOMMENDED`

Rationale: Test/Public/Private association for adding F1_SURFACE on top of fixed HSP is strong (6/6),
but CV Primary+Shadow consistency is weak (2/6), and family/SURFACE selection is historically contaminated.
This supports a **strong association, not causality**, and justifies a **small prospective replication**
before freezing SURFACE causality.

### Minimal prospective design (proposal only — do not train yet)

| Factor | Choice | Reason |
|--------|--------|--------|
| Representations (≤2) | **AbLang1**, **AbLingua** | External factorial best-average; internal factorial best-average |
| Optional control | Scratch | Isolates SURFACE add-on without PLM confound; only if budget allows |
| Topology | **JOINT** (fixed) | External factorial modest JOINT−SEP mean ≈ −0.004; freeze says topology secondary — do not re-search |
| Annotation | **FULL** (fixed) | External FULL−BASE modestly favorable; freeze says annotation secondary |
| Treatment | sequence-only vs +F1_SURFACE (± optional fixed HSP EIS-R8 from H107 lineage) | Purpose = SURFACE addition reproducibility |
| Platform | DL_FOLDLOCAL_COSINE_V3, seed 101 | Match historical protocol |
| Selection | **CV-only promote**; Public/Private embargo until prereg unlock | Prospective relative to this decision |
| Forbidden | new topology/annotation grids; Optuna; multi-HSP search | — |

## Non-actions

- No new training executed in this reaudit
- Did not edit `HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md` or `HIC_EXTERNAL_GENERALIZATION_DIAGNOSIS.md`

