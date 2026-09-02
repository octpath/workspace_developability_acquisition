# Structure Robustness Specification

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_1_CONTRACT_FROZEN`

ESMFold（primary）と ABodyBuilder2（technical replicate / ablation）の二重構造における feature 一致度の凍結仕様。主契約 [ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md](ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md) の補足。

---

## 1. 目的

Predicted PDB-derived physics は structure generator artifact に敏感な可能性がある。各 feature family について:

1. **target を見る前**に structure 間 consistency を評価する
2. 低一致を自動 discard 理由にしない（scientific finding になり得る）
3. generator-specific signal を別途分類する

---

## 2. Canonical structure input（Gate1.1）

**唯一の path / ID mapping ソース:**

[STRUCTURE_INPUT_CROSSWALK.csv](STRUCTURE_INPUT_CROSSWALK.csv)

| 項目 | 値 |
|------|-----|
| Scope | competition N=324 |
| Version | `v1`（`FEATURE_SPEC.json` の `structure_crosswalk_version`） |
| Audit | [STRUCTURE_INPUT_CROSSWALK_AUDIT.json](STRUCTURE_INPUT_CROSSWALK_AUDIT.json) |

各 future family は **独自 path discovery を行わない**。  
`esmfold_canonical_path` / `abodybuilder2_path` を crosswalk から読む。

Silent fallback 禁止。`canonical_mapping_status` が `PASS` / `PASS_EXPLAINED` 以外の ID は family 実装時に明示扱いを `FEATURE_SPEC.json` で決める。

### 2.1 Gate1 audit discrepancies（解決方針）

| Issue | Resolution |
|-------|------------|
| ESMFold path 二系統 | crosswalk が `esmfold_antibody_id_path` + `esmfold_pair_hash_path` + `esmfold_canonical_path` を記録。v1 では antibody-id path を canonical（全 324 存在・配列一致） |
| ABB2 cache superset | crosswalk は competition 324 のみ。cache 全体の余剰ファイルは無視 |

Round1 inventory 参照: `virtual_participant/round1_pdb_feature_inventory/ROUND1_PDB_FEATURE_INVENTORY_JA.md` §3

---

## 3. 評価タイミング

```
Resolve paths via STRUCTURE_INPUT_CROSSWALK.csv
        ↓
Feature extraction (ESMFold + ABB2)
        ↓
Structure robustness metrics   ← target labels を使わない
        ↓
Target association / modeling
```

`structure_robustness_evaluated_before_target=true` を `FEATURE_SPEC.json` に記録。

---

## 4. Per-feature metrics

各 semantic feature について、ESMFold 値 `x_i` と ABB2 値 `y_i`:

| Metric | Definition |
|--------|------------|
| Pearson(x, y) | 線形一致 |
| Spearman(x, y) | 順位一致 |
| Rank consistency | fraction of pairs where sign(x_i - x_j) == sign(y_i - y_j) |
| Sign agreement | fraction where sign(x_i) == sign(y_i)（0 中心特徴のみ） |
| Top-quantile overlap | 上位 20% ID の Jaccard |
| Bottom-quantile overlap | 下位 20% ID の Jaccard |

`structure_robustness.csv` 列:

```
semantic_feature,pearson,spearman,rank_consistency,sign_agreement,top20_jaccard,bottom20_jaccard,n_valid,missing_esmfold,missing_abb2,notes
```

---

## 5. Family-level summary

| Metric | Definition |
|--------|------------|
| `median_spearman` | semantic features の Spearman 中央値 |
| `iqr_spearman` | IQR |
| `frac_rho_gt_0.8` | Spearman > 0.8 の割合 |
| `frac_rho_gt_0.5` | Spearman > 0.5 の割合 |
| `low_consistency_features` | Spearman < 0.5 のリスト |

任意: ICC(2,1) / concordance correlation（`FEATURE_SPEC.json` で pre-specify）。

---

## 6. Robustness classification（凍結目安）

| Class | Criterion |
|-------|-----------|
| `ROBUST` | median Spearman ≥ 0.8 |
| `MODERATE` | 0.5 ≤ median Spearman < 0.8 |
| `FRAGILE` | median Spearman < 0.5 |

**FRAGILE でも自動除外しない。** semantic meaning を併記。

---

## 7. Generator-specific target association

| Class | 条件 |
|-------|------|
| `BOTH_STRUCTURES_SIGNAL` | ESMFold standalone ≥ WEAK かつ ABB2 standalone ≥ WEAK、同符号 |
| `ESMFOLD_ONLY_SIGNAL` | ESMFold のみ |
| `ABB2_ONLY_SIGNAL` | ABB2 のみ |
| `STRUCTURE_DEPENDENT_MIXED` | 両方 signal だが direction 不一致、または association が大きく異なる |
| `NO_SIGNAL` | 両方 NO_SIGNAL |

Standalone class: [SIGNAL_CLASSIFICATION_SPEC.md](SIGNAL_CLASSIFICATION_SPEC.md)

---

## 8. Structure-sensitive vs backbone-scale（期待 prior）

| 比較的 robust と期待 | 比較的 fragile と期待 |
|---------------------|----------------------|
| ANM / GNM / ENCoM | pKa / protonation |
| VH-VL orientation | cavity / void |
| 3Di / coarse embedding | OpenMM strain |
| | canonical SAP / hydrophobic field |
| | electrostatic × hydrophobic co-patch |

事前期待と実測の乖離も REPORT に記載。実装 priority と混同しない。

---

## 9. Missing structure handling

| 状況 | 処理 |
|------|------|
| crosswalk で片方 missing | `n_valid` 記録; available pairs のみ |
| 両方 missing | 除外し N を manifest に記録 |
| `FAIL_*` mapping status | 黙って skip せず registry / REPORT に記録 |

---

## 10. Confound recording

- mean pLDDT（ESMFold）
- ABB2 confidence proxy（利用可能なら）
- missing residue count, clash / severe contact, total SASA

---

## 11. REPORT 必須セクション

[FAMILY_REPORT_TEMPLATE.md](FAMILY_REPORT_TEMPLATE.md) §5:

- family-level summary
- low-consistency list
- robustness class + semantic interpretation
- generator-specific signal
- 低一致を棄却理由としなかった旨

---

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_1_CONTRACT_FROZEN`
