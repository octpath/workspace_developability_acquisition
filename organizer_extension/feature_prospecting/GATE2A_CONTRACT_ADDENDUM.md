# Gate2A Contract Addendum

**親状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`（historical freeze — 直接書き換え禁止）

**本 addendum の適用:** Gate2A 以降の feature-family experiments の source of truth。

---

## 1. Structure sources

```json
["esmfold", "abodybuilder2", "boltz2"]
```

| Key | Crosswalk path column |
|-----|------------------------|
| `esmfold` | `esmfold_canonical_path` |
| `abodybuilder2` | `abodybuilder2_path` |
| `boltz2` | `boltz2_pdb_path`（native mmCIF は `boltz2_native_cif_path`） |

**Crosswalk:** `STRUCTURE_INPUT_CROSSWALK_v2.csv` のみ。独自 path discovery 禁止。

---

## 2. Per-family feature outputs

```
features_esmfold.parquet
features_abodybuilder2.parquet
features_boltz2.parquet
```

---

## 3. Generator-specific signal labels

`THREE_GENERATORS_SIGNAL` | `ESMFOLD_ABB2_SIGNAL` | `ESMFOLD_BOLTZ2_SIGNAL` | `ABB2_BOLTZ2_SIGNAL` | `ESMFOLD_ONLY_SIGNAL` | `ABB2_ONLY_SIGNAL` | `BOLTZ2_ONLY_SIGNAL` | `STRUCTURE_DEPENDENT_MIXED` | `NO_SIGNAL`

Structure robustness と target signal は別軸。

---

## 4. Target-blind structure QC（Gate2A 追加）

全 generator 共通（ANM exclusion には使わない）:

- `n_residue`, `n_CA`, `missing_CA`, `nan_coords`
- `severe_clash_count`, `severe_clash_per_100_residues`

**Severe clash（凍結）:** 同一 model 内で、シーケンス近傍 `|resseq_i - resseq_j| ≤ 1`（同 chain）を除いた **Cα–Cα 距離 < 2.0 Å** のペア数。

ANM は Cα coarse-grained のため、atomistic clash は ANM の exclusion criterion にしない（structure-quality covariate）。

---

## 5. Boltz MSA provenance

既存 `msa/*/msa_meta.json` から recover 可能な範囲で manifest 追記。再 prediction 不要。不可なら `NOT_RECOVERABLE`。Gate2A を block しない。

---

## 6. Canonical evaluation（再掲）

- Baseline: `TRAIN_MEDIAN_BASELINE`
- Standalone: exact nested Ridge α∈`[0.1,1,10,100]`
- Incremental PRIMARY: Gate1.2 leakage-safe `CANONICAL_RESIDUAL_RIDGE`
- Primary CV: `opt_joint_group_k5_s42`
