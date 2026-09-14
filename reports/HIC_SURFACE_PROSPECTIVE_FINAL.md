# HIC SURFACE Prospective Replication — Final (External Complete)

**STATUS: `HIC_SURFACE_PROSPECTIVE_EXTERNAL_COMPLETE`**

| Field | Value |
|-------|--------|
| Prereg SHA | `993b8a9e268f65ed8d203949c34cbe3ef87446e6` |
| Internal freeze SHA | `f53039ffe606a2ca685fe64efaaeafb9c3993c6b` |
| External diagnosis HEAD (parent) | `241452d4300110541c16169fcb7929a1fc425d87` |
| Retraining | **None** |
| Predictions | existing H340–H343 `test.csv` only |
| Pipeline validation | **PASS** (max |Δ|=2.989e-08, tol=1e-06) |

## Absolute freeze compliance

- No retraining / config / seed / representation / topology / annotation / feature changes
- No new experiments, calibration, ensembles, or post-hoc HP selection
- All four arms scored (no selection after internal results)

## External scores

| Code | Rep | Arm | Public | Private | Test Overall | |P−Priv| |
|------|-----|-----|--------|---------|--------------|----------|
| EXP-H340 | ablang1 | SHAM35 | 0.465721 | 0.470414 | 0.468068 | 0.004693 |
| EXP-H341 | ablang1 | REAL_F1_SURFACE35 | 0.372349 | 0.407956 | 0.390153 | 0.035607 |
| EXP-H342 | ablingua | SHAM35 | 0.445832 | 0.480763 | 0.463297 | 0.034931 |
| EXP-H343 | ablingua | REAL_F1_SURFACE35 | 0.376478 | 0.429763 | 0.403121 | 0.053285 |

## PRIMARY contrasts (REAL − SHAM)

| Rep | ΔPublic | ΔPrivate | ΔTest |
|-----|---------|----------|-------|
| ablang1 | -0.093372 | -0.062458 | -0.077915 |
| ablingua | -0.069354 | -0.050999 | -0.060177 |

- Test improve: **2/2**
- Public+Private both improve: **2/2**

**Preregistered external verdict:** `PROSPECTIVE_SURFACE_EXTERNAL_STRONG`

## Prediction-level paired bootstrap (Test; secondary)

N_BOOT=10000, seed=101. Metric = mean(AE_REAL − AE_SHAM).

| Rep | mean | median | 95% CI | frac improve |
|-----|------|--------|--------|--------------|
| ablang1 | -0.077915 | -0.038871 | [-0.128451, -0.031497] | 0.593 |
| ablingua | -0.060177 | -0.036617 | [-0.103564, -0.019374] | 0.586 |

Bootstrap does **not** rewrite the preregistered external verdict.

## HIGH-tail external diagnostic (HIC > 11.5; not primary)

- n_high = **7**
- ablang1: ΔMAE_high=-0.9317, Δsigned=+0.9326, ΔMAE_nonHIGH=-0.0394
- ablingua: ΔMAE_high=-0.7643, Δsigned=+0.8088, ΔMAE_nonHIGH=-0.0284
- mean ΔMAE_high = -0.8480; mean ΔMAE_nonHIGH = -0.0339
- HIGH-tail vs retrospective rescue: **REPLICATED**

Retrospective H102–H113 mean ΔMAE_high ≈ −0.52 (large rescue). Prospective HIGH rescue is smaller / mixed relative to that magnitude; treat as secondary.

## Retrospective vs prospective consistency

| Aspect | Retrospective H102–H113 | Prospective H340–H343 |
|--------|-------------------------|------------------------|
| mean ΔTest | -0.0412 | -0.0690 |
| median ΔTest | -0.0406 | -0.0690 |
| Test improve | 6/6 | 2/2 |
| Pub+Priv both | 6/6 | 2/2 |
| Control | HSP-only vs SURFACE+HSP | **SHAM35 vs REAL F1** (param-matched) |
| HSP | Present (Test-informed families) | **Absent** |
| Selection | Historical / Test-informed | **Preregistered; Pub/Priv unseen until internal freeze** |

Direction: **consistent** (SURFACE improves Test in both eras).
Effect size: prospective ΔTest is **comparable or larger** than retrospective mean ΔTest under a stricter SHAM control.
Representation consistency: both AbLang1 and AbLingua improve externally.
HIGH-tail: retrospective large rescue; prospective status = **REPLICATED**.

## Freeze v2 surface claim status

Freeze v2 file is **not edited**. Relative to its wording that explicit surface/physicochemical information is the leading remaining explanatory hypothesis (causality not established):

**`STRENGTHENED`**

## Claim language (allowed)

> In two preregistered representation contexts, explicit antibody-specific surface physicochemical information provided reproducible incremental predictive value over a parameter- and architecture-matched sham auxiliary branch, with the effect observed on previously unconsulted Public and Private evaluation subsets.

> This strengthens the hypothesis that surface physicochemical information captures predictive signal missing from sequence-only representations.

**Not claimed:** biological causality of exposed aromatics / surface chemistry for HIC retention.

## Final synthesis

1. Prereg SHA: `993b8a9e268f65ed8d203949c34cbe3ef87446e6`
2. Internal freeze SHA: `f53039ffe606a2ca685fe64efaaeafb9c3993c6b`
3. Pipeline validation: **PASS**
4. H340–H343 scores: see table above / `HIC_SURFACE_PROSPECTIVE_EXTERNAL_SCORES.csv`
5/6. ablang1: ΔPub=-0.093372, ΔPriv=-0.062458, ΔTest=-0.077915
5/6. ablingua: ΔPub=-0.069354, ΔPriv=-0.050999, ΔTest=-0.060177
7. Test improve count: **2/2**
8. Public+Private both-improve count: **2/2**
9. Paired bootstrap: see table / CSV
10. HIGH-tail diagnostic: **REPLICATED**
11. Retrospective vs prospective: directionally consistent; prospective evidence higher quality
12. Internal verdict: `INTERNAL_SURFACE_REPLICATION_DIRECTIONAL`
13. External verdict: `PROSPECTIVE_SURFACE_EXTERNAL_STRONG`
14. Combined surface evidence: `PROSPECTIVE_REPLICATION_STRONG_WITH_DIRECTIONAL_INTERNAL`
15. Freeze v2 surface claim status: **STRENGTHENED**
16. Further surface replication scientifically necessary: **NO**

## Final scientific decision

### `SURFACE_INCREMENTAL_VALUE_PROSPECTIVELY_REPLICATED`

Additional model exploration *for the purpose of confirming surface incremental value* is not scientifically required.

## Artifacts

- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_SCORES.csv`
- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_CONTRASTS.csv`
- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_BOOTSTRAP.csv`
- `reports/HIC_SURFACE_PROSPECTIVE_EXTERNAL_HIGHTAIL.csv`
- `reports/HIC_SURFACE_PROSPECTIVE_PIPELINE_VALIDATION.csv`

`reports/HIC_SCIENTIFIC_FREEZE_V2.md` was **not** modified.

