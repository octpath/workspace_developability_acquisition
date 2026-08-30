# GATE B2 FINAL — HIC vs TmApp Shootout

**Recommendation: `HIC_PRIMARY_TMAPP_SECONDARY`**

Secondary option if organizers want maximum educational breadth: `DUAL_TARGET_COMPETITION`
(equal mean Spearman), with a mandatory BIO_SHORTCUT audit track on TmApp.

---

## Decision rationale

| Criterion | HIC | TmApp |
|---|---|---|
| Mean best CV Spearman (6 splits) | **0.567** | 0.499 |
| Fair SEQ_SIMPLE (canonical) | 0.474 | 0.467 |
| BIO_SHORTCUT (canonical) | **0.239** (weak — good) | **0.505** (strong — risk) |
| PLM − BIO (mean) | **+0.262** | +0.144 |
| Best structure − PLM | ESMFold **+0.033** | ESMFold **−0.201** |
| Public→Private model-rank ρ (mean) | 0.591 | **0.725** |
| Private #1 family (canonical+shadows) | **native ESMFold / fusion** | **PLM** |
| Overlap Spearman(HIC, TmApp) | 0.118 (n=324) — nearly independent |

**Why not pure TmApp primary:** absolute PLM scores are competitive and leaderboard fidelity is better, but BIO_SHORTCUT nearly matches the best PLM on Dev CV. That recreates the B1 germline-shortcut risk for a primary competition target.

**Why not pure dual as default:** both targets are scientifically interesting and weakly correlated, but equal weighting would let a BIO-heavy TmApp track dominate educational narrative unless BIO is tightly controlled. Prefer HIC as primary (surface/physchem + structure story) with TmApp as secondary / optional dual arm.

**Why not DO_NOT_ADVANCE / NEEDS_SPLIT_REDESIGN:** redesigned pre-model splits improved ranking fidelity vs B1 (HIC Pub→Priv mean ρ≈0.59; TmApp ≈0.73). Residual bootstrap CIs on Public/Private remain wide (≈0.38–0.40 Spearman width) — expected at N≈70 — but usable.

---

## Scope completed

- Targets: **HIC + TmApp only** (PSR not primary).
- Native ESMFold: `facebookresearch/esm` 2.0.1, `f"{VH}:{VL}"` → infer (370/370 OK).
- ABodyBuilder2: reused B1 structures.
- Surface features: ShrakeRupley probe_radius=**1.4**, n_points=**100**, MaxASA Tien2013, patch RASA≥**0.20**, CA contact **8 Å**.
- Splits: pre-model frozen `canonical` + `shadow_1…5` (see `split_design.md`).
- Modeling: nested GroupKFold; linear (+ RF/ET/XGB on selected structure/PLM); rank-average + Dev-OOF blend ensembles.
- **No** participant distribution files created.

---

## Required comparisons (A–J)

### A — Native `VH:VL` ESMFold vs B1 HF/linker?
**Yes, distinct branch — but geometry is close.**  
B1 used HuggingFace `EsmForProteinFolding` + manual Gly25 linker + chain rewrite. Native uses colon multimer API with internal linker / `residue_index_offset=512`.  
Validation: 15/15 length-matched distinct chains; COM ~22 Å.  
Fv CA RMSD B1-HF vs native ≈ **0.29 Å** (n=100); ABB vs native ≈ **1.11 Å**.  
Feature-level |Δ| SASA/BSA still material (Fv SASA |Δ| mean ~479 Å²; BSA |Δ| ~147) with Spearman 0.58–0.85 — enough to matter for downstream models.

### B — Does native ESMFold improve prediction?
**HIC: yes (modest).** Mean ESMN−PLM CV Δ = **+0.033**; native ESMFold families often Private #1.  
**TmApp: no.** Mean ESMN−PLM = **−0.201**; PLM dominates.

### C — ABB vs native ESMFold structure features?
**HIC: native ESMFold slightly better** (ABB−ESMN = −0.029).  
**TmApp: ABB slightly less bad** (+0.030) but both trail PLM.

### D — Does RASA itself help?
**Not clearly for HIC.** S1−S0 = **−0.014**.  
**Mild yes for TmApp** (+0.040), still far below PLM.  
Do **not** claim RASA useful as a general rule for HIC.

### E — Surface physicochemical features?
**Mild yes.** S2−S0: HIC **+0.011**, TmApp **+0.066**.  
RASA-weighted physchem alone (S3) **hurts** HIC (−0.104). Absolute exposed physchem > RASA reweighting for HIC.

### F — Surface patches?
Patches alone are **weak absolute** predictors (best patch CV ~0.16–0.20) but are part of the useful ALL/SURFACE stacks for HIC. Residual analysis: ESMFN predicts SEQ_SIMPLE residuals at Spearman **0.45** on HIC — complementary surface geometry signal exists.

### G — Structure beyond PLM?
**HIC: yes.** Fusion−PLM **+0.035**; Dev-OOF blend puts **~51% weight on ESMFN**.  
**TmApp: no.** Fusion−PLM **−0.047**; blend weight on ESMFN **0**.

### H — PLM beyond BIO_SHORTCUT?
**HIC: clearly yes** (+0.262). BIO is a poor HIC baseline.  
**TmApp: yes but smaller** (+0.144); BIO already strong.

### I — More modeling headroom?
**HIC** — larger overall−simple gap and structure/PLM complementarity; BIO does not saturate the target.

### J — More reliable leaderboard?
**TmApp** — Pub→Priv rank Spearman mean **0.725** (HIC **0.591**).  
HIC canonical still shows weaker CV→Public rank correlation (0.32) than shadows (up to 0.84); use multi-shadow reporting for winner claims.

---

## Competition format options

| Option | Verdict |
|---|---|
| A — HIC only | Strong scientifically; loses TmApp’s LB-fidelity educational contrast |
| B — TmApp only | Good LB behavior; **BIO shortcut risk** for primary |
| C — Dual (mean Spearman) | Attractive (ρ_targets=0.12; different modalities); needs BIO controls |

**Chosen default:** `HIC_PRIMARY_TMAPP_SECONDARY`  
**Optional:** dual track scored as `mean(Spearman_HIC, Spearman_TmApp)` with published BIO_SHORTCUT baselines.

---

## Scientific snapshots

### HIC
Univariate |ρ| are modest; H3 length and exposed H3 hydrophobicity outrank bulk GRAVY. Controlled ablations favor **absolute surface physchem / ESMFold surface stacks** over RASA-only or BIO shortcuts. Central hypothesis (exposed hydrophobic surface vs bulk gravy) is **partially supported** — structure/surface beats BIO and adds residual signal beyond SEQ_SIMPLE, but no single patch feature dominates.

### TmApp
Germline-distance correlates (~−0.22 to −0.25). PLM captures most available signal; structure adds little. Residual PLM on BIO residuals ~0.24 — some residual sequence signal, but competition design must expose BIO baselines.

### Ensembles
Rank-mean and Dev-OOF linear blends help HIC Public (~0.56–0.58). TmApp Private blend (~0.24) warns against overtrusting Dev-tuned blends on TmApp Private.

---

## Bootstrap / uncertainty

- Finalist Public/Private Spearman **95% CI width ≈ 0.38–0.40** (N_public/private ~60–90).
- Pairwise Private win probabilities in `bootstrap_results.csv`.
- Winner stability: HIC Private #1 almost always an **ESMF_NATIVE_*** / fusion family; TmApp almost always **PLM_***.

---

## Language / constraints respected

- ESMFold supports multimer inputs through colon-separated chains and chain-aware inference machinery (not AlphaFold-Multimer).
- HIC is **not** called an aggregation measurement.
- Germline distance is **not** a perfect SHM count.
- RASA not claimed useful for HIC absent ablation support.
- Organizer-only donor / B-cell subset unused in participant models.
- PSR not reintroduced as primary.

---

## Artifacts

```text
gate_b2/reports/GATE_B2_FINAL.md          (this file)
gate_b2/reports/esmfold_b1_implementation_audit.md
gate_b2/reports/esmfold_native_multimer_audit.md
gate_b2/reports/structure_comparison.md
gate_b2/reports/surface_feature_ablation.md
gate_b2/reports/split_design.md
gate_b2/reports/leaderboard_uncertainty.md
gate_b2/reports/hic_scientific_analysis.md
gate_b2/reports/tmapp_scientific_analysis.md
gate_b2/reports/target_comparison.md
gate_b2/metrics/all_results.csv
gate_b2/metrics/split_results.csv
gate_b2/metrics/bootstrap_results.csv
gate_b2/metrics/paired_deltas.csv
gate_b2/config/split_manifest.json
gate_b2/config/pipeline_registry.json
```

---

## Stop condition

Native ESMFold rerun ✓ · surface/patch analysis ✓ · split redesign ✓ · canonical+5 shadows ✓ · leaderboard uncertainty ✓ · HIC vs TmApp vs dual decision ✓  

Participant distribution files: **not created** (per Gate instructions).
