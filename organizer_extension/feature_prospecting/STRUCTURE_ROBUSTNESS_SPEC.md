# Structure Robustness Specification

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`

ESMFold（primary）・ABodyBuilder2・**Boltz-2**（third technical replicate）における feature 一致度の凍結仕様。

---

## 1. 目的

1. **target を見る前**に structure 間 consistency を評価する
2. 低一致を自動 discard しない
3. generator-specific signal を別途分類する
4. **median / consensus feature はまだ作らない**（別 pre-specified version のみ）

---

## 2. Canonical structure inputs

| Generator | Role | Crosswalk columns |
|-----------|------|-------------------|
| ESMFold | Primary | `esmfold_canonical_path` |
| ABodyBuilder2 | Technical replicate | `abodybuilder2_path` |
| Boltz-2 | Third technical replicate (`BOLTZ2_FV_STANDARD_v1`) | `boltz2_native_cif_path` / `boltz2_pdb_path` |

**Source of truth:** [STRUCTURE_INPUT_CROSSWALK_v2.csv](STRUCTURE_INPUT_CROSSWALK_v2.csv)

Legacy v1（ESMFold+ABB2 only）は保持するが、Gate1.2 以降の family は **v2** を参照。

Boltz-2 は ESMFold replacement ではない。独立性のため **no explicit templates**、**use_potentials=false**、**diffusion_samples=1**。

---

## 3. Pairwise consistency（3 generators）

Pairs:

1. ESMFold vs ABodyBuilder2
2. ESMFold vs Boltz-2
3. ABodyBuilder2 vs Boltz-2

Per semantic feature × pair: Pearson, Spearman, top20 Jaccard, bottom20 Jaccard

Family summary:

- `median_spearman` per pair
- `minimum_pairwise_median_spearman`（**primary classification input**）
- `median_of_pairwise_median_spearman`

Pairwise 値は必ず REPORT に併記（one-number に潰さない）。

---

## 4. Robustness classification（3-generator）

| Class | Criterion |
|-------|-----------|
| `ROBUST` | all 3 pairwise family median Spearman ≥ 0.8 |
| `MODERATE` | all ≥ 0.5 かつ少なくとも 1 pair < 0.8 |
| `FRAGILE` | 少なくとも 1 pairwise median Spearman < 0.5 |

2-generator のみの family は Gate1 thresholds（median Spearman）を継続使用可。

**FRAGILE でも自動除外しない。**

---

## 5. Generator-specific signal labels（future target evaluation）

`THREE_GENERATORS_SIGNAL` | `ESMFOLD_ABB2_SIGNAL` | `ESMFOLD_BOLTZ2_SIGNAL` | `ABB2_BOLTZ2_SIGNAL` | `ESMFOLD_ONLY_SIGNAL` | `ABB2_ONLY_SIGNAL` | `BOLTZ2_ONLY_SIGNAL` | `STRUCTURE_DEPENDENT_MIXED` | `NO_SIGNAL`

Gate1.2 では target signal 未評価 → `NOT_RUN`。

---

## 6. Timing / missing / confounds

- robustness は target 前
- missing / FAIL mapping: silent skip 禁止
- confound: pLDDT / Boltz confidence / clash / SASA 等を記録（target 相関はしない）

---

## 7. Future protocols（NOT RUN in Gate1.2）

- `BOLTZ2_NOMSA_v1`
- `BOLTZ2_POTENTIALS_v1`
- `BOLTZ2_MULTISAMPLE_v1`

---

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`
