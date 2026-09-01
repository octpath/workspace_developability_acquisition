# Round 1 Pre-Test Lock

**状態:** `ROUND1_PRETEST_LOCKED`  
**タイムスタンプ:** 2026-09-01T21:05:49.507186+00:00  
**git commit:** `3cca698aed8b36171e66b0aec129dc6d2fcb8f17`

## 目的

Test prediction 作成前に、Round 1 として評価するモデルを完全 freeze する。Public / Private 結果を見てから final model を選ぶことを禁止する。

## PRIMARY — ROUND1_PRIMARY

| Target | experiment_id | Primary CV | Shadow CV | status |
|---|---|---:|---:|---|
| TmApp | `TmApp__META_performance__ridge_100.0` | 2.7135 | 2.7591 | STAGE5_SHADOW_CONFIRMED |
| HIC | `HIC__SIMPLE_blend_seq_surf_adv` | 0.4252 | 0.4321 | STAGE5_SHADOW_CONFIRMED |

### TmApp stack recipe (frozen)

- Base models: `TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt`, `TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt`, `TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt`, `TmApp__ADV_TMAPP_ALL__SVROpt`
- Meta: Ridge α=100
- Final Test: full Dev 162 cross-fitted OOF → meta fit → full Dev base refit → Test predict

### HIC blend recipe (frozen)

- Base models: `HIC__FUSION__esm2__H__SEQ_ALL__SVROpt`, `HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt`, `HIC__ADV_SURFACE_PATCH__SVROpt`
- Weights: 1/3, 1/3, 1/3 (equal mean, no refit)

## SECONDARY — pre-registered diagnostics

| Target | role | experiment_id |
|---|---|---|
| TmApp | conservative | `TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt` |
| TmApp | plm_seq | `TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt` |
| HIC | conservative | `HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt` |
| HIC | plm_only | `HIC__esm2__H__SVROpt` |

**SECONDARY は reveal 後 postmortem 用。primary 選択には使用しない。**

## SHA-256 (pre-lock)

```
8fce3dc8b46566c1a69fe649be632554212e184077575f2c14b1ef34437d9cf7  virtual_participant/round1_finalization/round1_final_model_specs.json
b6a6aeb468a0d93cdcfcb787ef4fe763b8950398f81902bcb306c478709e5b6d  virtual_participant/round1_finalization/round1_model_shortlist.csv
23643677938dbbfaf81ea269027eeb3bcbb47af18585fc7b210896440870f8e5  virtual_participant/stage5_integration/STAGE5_REPORT_JA.md
```

**Public / Private / organizer secret は本 lock 時点で未参照。**
