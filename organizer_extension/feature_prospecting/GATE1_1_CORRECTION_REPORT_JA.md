# Gate 1.1 Correction Report

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_1_CONTRACT_FROZEN`

Gate1（`ORGANIZER_FEATURE_PROSPECTING_GATE1_CONTRACT_FROZEN`）を baseline として承認したうえで、feature family 実行前の限定修正を行った。

**禁止事項の遵守:** 新規 feature extraction / model training / Public-Private score 探索 / Round1 変更 / ANM・pKa・SAP 実験開始 — いずれも未実施。

---

## 1. 修正概要

| 項目 | Gate1 | Gate1.1 |
|------|-------|---------|
| MAE naive baseline | train-mean | **TRAIN_MEDIAN_BASELINE** |
| Ridge alpha | fold-internal（曖昧） | **exact nested Primary folds** + tie→larger alpha |
| Incremental PRIMARY | family-specific fusion | **CANONICAL_RESIDUAL_RIDGE**（共通） |
| WEAK/MIXED | neutral_band 未設定 | **exactly 2/3 + bootstrap CI contains 0** |
| Structure paths | family ごと再実装リスク | **STRUCTURE_INPUT_CROSSWALK.csv** |
| Relevance UX | 統計ラベル中心 | mechanistic prior + empirical verdict + summary table |

---

## 2. Baseline correction

MAE の constant-optimal predictor は **median**。

- CV OOF: outer-training median → held-out へ定数予測
- Public / Private / All Test: full Dev N=162 median

保存キー: `baseline_MAE_CV` / `Public` / `Private` / `AllTest`、`baseline_type=TRAIN_MEDIAN_BASELINE`

Round1 artifact は変更していない。

---

## 3. Nested Ridge exact protocol

1. Outer: `opt_joint_group_k5_s42`
2. Inner: remaining 4 Primary fold IDs（新しい random KFold 禁止）
3. Grid: `[0.1, 1.0, 10.0, 100.0]`
4. Select: min mean inner MAE; **tie → larger alpha**
5. Outer-training refit → held-out 予測
6. Test: full Dev 5-fold で alpha 選択 → full Dev refit
7. Preprocessing は各 training 分割内 fit

---

## 4. Canonical residual fusion

```
p_candidate = p_ref + Ridge(X → residual)
residual = y - p_ref
```

- CV: Round1 frozen **OOF** reference; outer-training のみ fit; meta in-sample 禁止
- Test: Dev OOF residual で fit; Test = Round1 Test reference + residual correction
- Family-specific concat/stacking は `SECONDARY_FUSION` のみ

---

## 5. Split reproducibility rule

- `REPRODUCIBLE_INCREMENT`: 3 split 全て `delta < 0`（effect 極小でも directionally reproducible; PRACTICALLY_LARGE は付与しない）
- `WEAK_OR_MIXED_INCREMENT`: exactly 2/3 で `delta < 0`、残り 1 の bootstrap 95% CI が 0 を含む
- `DIRECTION_REVERSAL`: 2/3 改善だが残り 1 の CI lower bound > 0
- 固定 MAE threshold は導入しない

---

## 6. Structure crosswalk audit

| Metric | Value |
|--------|-------|
| File | `STRUCTURE_INPUT_CROSSWALK.csv` |
| N | 324（DEV 162 / PUBLIC 81 / PRIVATE 81） |
| ESMFold exists | 324 |
| ABB2 exists | 324 |
| Sequence match (H/L × both generators) | **EXACT_MATCH × 全列** |
| `canonical_mapping_status` | **PASS × 324** |
| Canonical ESMFold path | `esmfold_native/{id}.pdb`（antibody-id; hash path も併記） |
| Builder | `scripts/build_structure_input_crosswalk.py` |
| Audit JSON | `STRUCTURE_INPUT_CROSSWALK_AUDIT.json` |

Silent fallback なし。今後の `FEATURE_SPEC.json` は `structure_crosswalk_version=v1` と `crosswalk_hash` を参照。

---

## 7. Freeze hashes

詳細は `GATE1_FREEZE_MANIFEST.json`（`repository_git_head` / `git_status_short` / per-file sha256）。

旧 Gate1 hash の偽装は行っていない。state は `ORGANIZER_FEATURE_PROSPECTING_GATE1_1_CONTRACT_FROZEN`。

---

## 8. Pre-experiment relevance map

**Source note:** repository 内に独立 DeepResearch レポートファイルは見つからなかった。Gate1 handoff prompt §F（DeepResearch 要約）および `ROUND1_PDB_FEATURE_INVENTORY_JA.md` §22 handoff を根拠に mechanistic prior を freeze。empirical 列は全て `NOT_RUN`。

| Family | TmApp relevance | HIC relevance | Why |
|--------|-----------------|---------------|-----|
| ANM-SPECTRUM | LIKELY_RELEVANT | UNLIKELY_PRIMARY | 集団運動↔Tm; HIC は表面疎水性中心 |
| VHL-ANGLE | LIKELY_RELEVANT | RELATED_BUT_INDIRECT | Fv 配向↔安定性; HIC は間接 |
| PKA-SHIFT | LIKELY_RELEVANT | PLAUSIBLY_RELEVANT | ΔpKa/電荷ひずみ↔安定性; HIC は極性経由で弱い可能性 |
| 3DI-FROZEN | PLAUSIBLY_RELEVANT | PLAUSIBLY_RELEVANT | 学習構造トークン（機構不透明） |
| AROMATIC-TOPO | RELATED_BUT_INDIRECT | LIKELY_RELEVANT | 芳香環表面↔HIC |
| STATIC-SAP | RELATED_BUT_INDIRECT | LIKELY_RELEVANT | 古典的 HIC/凝集指標 |
| VOID-EXPLICIT | LIKELY_RELEVANT | UNLIKELY_PRIMARY | packing defect↔Tm |
| POLAR-SAT | LIKELY_RELEVANT | RELATED_BUT_INDIRECT | buried polar↔安定性 |
| OPENMM-STRAIN | PLAUSIBLY_RELEVANT | NO_CLEAR_MECHANISTIC_LINK | ひずみ↔Tm はあり得るが脆弱; HIC 無関係に近い |
| ENCOM-CHEM | LIKELY_RELEVANT | UNLIKELY_PRIMARY | ANM 系の化学拡張 |
| INTERFACE-ENERGY | LIKELY_RELEVANT | RELATED_BUT_INDIRECT | VH-VL interface↔Fab 安定性 |
| HYDRO-FIELD | RELATED_BUT_INDIRECT | LIKELY_RELEVANT | 連続疎水場↔HIC |
| ELEC-HYDRO-COPATCH | RELATED_BUT_INDIRECT | LIKELY_RELEVANT | 静電×疎水パッチ↔HIC |
| GEARNET-FROZEN | PLAUSIBLY_RELEVANT | PLAUSIBLY_RELEVANT | 幾何 GNN（機構不透明） |
| SURFACE-DL | RELATED_BUT_INDIRECT | PLAUSIBLY_RELEVANT | 表面 DL; ライセンス/依存注意 |
| SHORT-ENSEMBLE | PLAUSIBLY_RELEVANT | RELATED_BUT_INDIRECT | 短 MD; コスト高 |
| MLFF | PLAUSIBLY_RELEVANT | NO_CLEAR_MECHANISTIC_LINK | 予測 PDB 上のエネルギーは speculative |

実装 priority（first/second wave）と mechanistic relevance は別軸（例: OPENMM-STRAIN は TmApp PLAUSIBLY でも fragility のため ANM より実装優先度低）。

---

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_1_CONTRACT_FROZEN`
