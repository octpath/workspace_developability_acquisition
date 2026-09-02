# Signal Classification Specification

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`

本書は Organizer Feature Prospecting における signal 判定の凍結仕様である。主契約 [ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md](ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md) の補足。

Gate1.1: TRAIN_MEDIAN_BASELINE、exact nested Ridge、CANONICAL_RESIDUAL_RIDGE、WEAK/MIXED bootstrap CI、empirical verdict。  
Gate1.2: **outer residual CV leakage 修正**、mechanistic relevance **5-level numeric display**。

---

## 1. 判定軸の独立性

各 feature family について、以下は**必ず別々に**記録・分類する。

| 軸 | 問い | 主な入力 |
|----|------|----------|
| Physical association | pre-specified summary と target / residual の関連 | univariate / multivariate pre-spec |
| Standalone signal | feature-only canonical Ridge の予測力 | MAE vs TRAIN_MEDIAN_BASELINE, correlation, pred SD |
| Incremental signal | incumbent residual への補完（PRIMARY = CANONICAL_RESIDUAL_RIDGE） | delta_MAE, bootstrap |
| Split reproducibility | CV / Public / Private の direction 一貫性 | delta 符号 + bootstrap CI |
| Structure robustness | ESMFold vs ABB2 一致 | Spearman 等 |
| Generator-specific | どちらの structure で signal か | per-generator metrics |
| Mechanistic prior / empirical verdict | 「結局 TmApp/HIC に効きそうか」 | pre/post labels |

**Signal ≠ MAE improvement only.** 上記を総合して final statement を書く。

---

## 2. TRAIN_MEDIAN_BASELINE（naive MAE baseline）

MAE の constant-optimal predictor は **median** である。Gate1 の `train-mean predictor` は **廃止**。

### 2.0 Frozen baseline name

`TRAIN_MEDIAN_BASELINE`

### 2.0.1 Primary CV OOF

Outer held-out fold `f` について:

1. Outer-training antibodies（残 4 Primary folds）の target **median** を計算
2. Held-out fold 全例へその定数を予測

### 2.0.2 Public / Private / All Test

Full Dev N=162 の target **median** を constant prediction として使用。

### 2.0.3 必須保存

- `baseline_MAE_CV`
- `baseline_MAE_Public`
- `baseline_MAE_Private`
- `baseline_MAE_AllTest`
- `baseline_type` = `TRAIN_MEDIAN_BASELINE`

Standalone の「MAE beats naive baseline」は全て `TRAIN_MEDIAN_BASELINE` を意味する。

Round1 既存 artifact は変更しない。baseline は Organizer Extension 側で再計算して保存する。

---

## 3. Exact nested Ridge alpha selection（canonical screening）

Alpha grid（凍結）: `[0.1, 1.0, 10.0, 100.0]`

### 3.1 Primary CV OOF

Outer folds: frozen `opt_joint_group_k5_s42`（変更禁止）

Outer held-out fold = `f`、outer training = 残り 4 Primary folds。

**Inner CV:** 残り 4 Primary fold IDs をそのまま inner folds として使用。**新しい random KFold を作らない。**

各 alpha について inner validation MAE の **mean** を計算。

**Select:** smallest mean inner MAE  
**Tie:** **larger alpha** を選択（stronger regularization、deterministic）

選択 alpha で outer training 全体を refit → outer held-out を予測。

### 3.2 Public / Private / All Test

Full Dev 上で frozen Primary 5 folds を用いて alpha selection（同上: 5-fold mean MAE、tie → larger alpha）。

選択 alpha で full Dev refit → Test prediction。

### 3.3 Preprocessing nesting

imputation / standardization / PCA はそれぞれ **その時点の training 分割内のみ** fit:

- Outer OOF: outer-training 内 fit（inner CV 内でも各 inner-training で fit）
- Test: full Dev で fit

Public/Private 全体で scaler/PCA を fit しない。

---

## 4. Standalone signal classification

対象: feature-only **canonical Ridge**（§3 exact nested procedure）

### 4.1 入力 metrics

| Metric | Splits |
|--------|--------|
| MAE | CV OOF, Public, Private, All Test |
| Pearson(pred, true) | 同上 |
| Spearman(pred, true) | 同上 |
| pred SD, pred range | 同上 |
| pred SD / true SD | 同上 |
| vs TRAIN_MEDIAN_BASELINE | 同上 |

### 4.2 Standalone クラス

| Class | Operational definition |
|-------|------------------------|
| `REPRODUCIBLE` | CV / Public / Private 全てで (a) MAE < TRAIN_MEDIAN_BASELINE MAE、かつ (b) Pearson または Spearman が同符号で > 0、かつ (c) いずれかの split で \|r\| ≥ 0.15 |
| `WEAK` | 上記のうち 2 split でのみ (a)(b) を満たす、または \|r\| が全 split < 0.15 だが MAE が 3 split 全てで baseline を下回る |
| `CV_ONLY` | CV のみ (a)(b) を満たし Public / Private で再現しない |
| `TEST_ONLY_POSTHOC` | CV で満たさず Public / Private のみ — **探索記録のみ** |
| `NO_SIGNAL` | 全 split で association なし、または direction 反転 |

**注:** \|r\| ≥ 0.15 は探索的閾値。formal inference ではない。

### 4.3 MODEL_DEPENDENT_SIGNAL

canonical Ridge で `NO_SIGNAL` または `WEAK` だが、pre-specified secondary（linear SVR / RBF-SVR）でのみ `REPRODUCIBLE` に該当する場合。主結果とは分離。

---

## 5. Canonical incremental model: CANONICAL_RESIDUAL_RIDGE

**PRIMARY incremental evaluation（全 family 共通）。**

Family-specific concat / stacking / SVR fusion は `SECONDARY_FUSION` として許可するが、**canonical incremental classification は必ず CANONICAL_RESIDUAL_RIDGE を主結果とする。**

### 5.1 Definition

```
p_candidate_i = p_ref_baseline_i + r_hat_i
```

Candidate features `X` から Ridge で residual correction `r_hat_i` を予測。Alpha grid: `[0.1, 1.0, 10.0, 100.0]`（tie → larger alpha）。

### 5.2 Outer-CV procedure（Gate1.2 — leakage-safe）

Outer fold `f`:

- `outer_test` = Primary fold `f`
- `outer_train` = remaining 4 Primary folds

#### Reference prediction for `outer_test`

Frozen Round1 reference **OOF** prediction for fold `f` は、fold `f` を除く 4 folds で学習されているため、**outer_test baseline** として利用してよい。

#### Residual targets for `outer_train`（CRITICAL）

**Global frozen Round1 OOF residual を outer_train residual target に使わない**（outer-heldout fold `f` の labels が global OOF の training に含まれる meta-level contamination を避ける）。

代わりに:

1. `outer_train` **のみ** を使い、frozen Round1 reference **recipe** を cross-fit し直す
2. 各 sample `i ∈ outer_train` について、`i` 自身を reference training から除外した  
   `reference_crossfit_prediction_i` を生成する
3. **outer_test fold `f` の labels / samples を reference cross-fitting に一切使わない**
4. Residual target:

```
residual_i = y_i - reference_crossfit_prediction_i
```

5. `outer_train` 内の frozen Primary fold IDs で residual Ridge alpha を選択（新 random split 禁止）
6. `outer_train` 全体で residual Ridge を fit
7. outer-heldout features → residual correction
8. Final outer candidate:

```
p_candidate = frozen Round1 reference OOF prediction  +  predicted residual correction
```

**Meta-level in-sample evaluation 禁止。**

### 5.3 Test residual modeling

Public / Private / AllTest:

- Dev 全 162 の frozen Round1 reference **OOF** から residual targets を作ってよい
- residual Ridge を full Dev で fit（Primary 5-fold alpha selection）
- Test labels は training/tuning に使わない

```
p_candidate_test = frozen Round1 Test prediction + residual prediction
```

### 5.4 Historical frozen OOF shortcut

outer_train residual targets に global frozen OOF を使う方式は **PRIMARY canonical incremental として禁止**。

必要なら `NON_NESTED_DIAGNOSTIC_ONLY` として secondary diagnostic にできるが、原則実行不要。

### 5.5 Delta

```
delta_MAE_s = MAE(candidate)_s - MAE(reference)_s
```

delta < 0 → improvement。

---

## 6. Incremental signal classification（Gate1.1）

固定 MAE threshold（neutral_band）は **導入しない**。

| Class | Operational definition |
|-------|------------------------|
| `REPRODUCIBLE_INCREMENT` | `delta_CV < 0` AND `delta_Public < 0` AND `delta_Private < 0` |
| `WEAK_OR_MIXED_INCREMENT` | 3 split 中 **exactly 2** で `delta < 0`、かつ残り 1 split の paired-bootstrap **95% CI が 0 を含む** |
| `DIRECTION_REVERSAL` | 3 split 中 2 で改善だが、残り 1 split で bootstrap 95% CI **lower bound > 0**（明確な deterioration） |
| `CV_ONLY_INCREMENT` | CV 改善だが Public / Private で再現しない |
| `TEST_ONLY_POSTHOC_INCREMENT` | CV 改善なし、Public / Private 側のみ改善 |
| `NO_INCREMENT` | consistent improvement direction なし |
| `HARMFUL` | 2 split 以上で `delta > 0` かつ当該 split の bootstrap 95% CI lower bound > 0 |

### 6.1 Effect size 解釈の禁止事項

3 split 全て `delta < 0` なら effect size が極小でも **directionally** は `REPRODUCIBLE_INCREMENT`（= directionally reproducible）。

REPORT では必ず absolute ΔMAE / bootstrap CI / P(improve) を併記し、`PRACTICALLY_LARGE` 等の意味を勝手に付与しない。

### 6.2 Paired bootstrap

各 split:

- `bootstrap_mean_delta`, `bootstrap_median_delta`
- `bootstrap_ci95_low`, `bootstrap_ci95_high`
- `P(delta_MAE < 0)`

Formal hypothesis test ではない。effect-direction stability diagnostic。

---

## 7. Split reproducibility classification（統合）

| Class | 条件 |
|-------|------|
| `REPRODUCIBLE_SIGNAL` | CV / Public / Private で improvement direction 一致 |
| `WEAK_OR_MIXED_SIGNAL` | exactly 2/3 改善、残り 1 の bootstrap CI が 0 を含む |
| `DIRECTION_REVERSAL` | 2/3 改善だが残り 1 が明確 deterioration（CI lower > 0） |
| `CV_ONLY_SIGNAL` | CV 改善、Public / Private 非再現 |
| `TEST_ONLY_POSTHOC_SIGNAL` | CV なし、Test のみ — Participant evidence 不可 |
| `NO_REPRODUCIBLE_SIGNAL` | consistent direction なし |

---

## 8. Empirical verdict mapping（post-experiment）

機械的な最低限 mapping（human が REPORT で nuance を補足してよいが、registry ラベルは以下に従う）:

| Standalone | Incremental | Empirical verdict |
|------------|-------------|-------------------|
| REPRODUCIBLE | REPRODUCIBLE_INCREMENT | `PROMISING` |
| REPRODUCIBLE | NO_INCREMENT | `PROMISING_BUT_REDUNDANT` |
| WEAK または NO_SIGNAL | REPRODUCIBLE_INCREMENT | `COMPLEMENTARY` |
| （split / generator / model で方向不一致） | | `MIXED` |
| NO_SIGNAL | NO_INCREMENT | `NO_EVIDENCE_IN_CURRENT_DATA` |

追加規則:

- mechanistic prior が `UNLIKELY_PRIMARY` または `NO_CLEAR_MECHANISTIC_LINK`、かつ empirical が NO_SIGNAL / NO_INCREMENT → `UNLIKELY_RELEVANT_UNDER_CURRENT_SETUP`

**禁止:** `IRRELEVANT` / `UNRELATED` の普遍断定。negative は必ず `IN_CURRENT_DATA` / `UNDER_CURRENT_SETUP` に限定。

Allowed empirical labels:

`PROMISING` | `PROMISING_BUT_REDUNDANT` | `COMPLEMENTARY` | `MIXED` | `NO_EVIDENCE_IN_CURRENT_DATA` | `UNLIKELY_RELEVANT_UNDER_CURRENT_SETUP` | `NOT_RUN`

---

## 9. Mechanistic prior（pre-experiment）— 5-level numeric scale（Gate1.2）

Target labels を見る前に endpoint ごとに freeze。  
数字が大きいほど endpoint との mechanistic relevance が高い。

| score | display | Meaning |
|------:|---------|---------|
| 5 | `5_LIKELY_RELEVANT` | 比較的直接的で自然な関係 |
| 4 | `4_PLAUSIBLY_RELEVANT` | 十分関係し得るが directness が一段弱い |
| 3 | `3_RELATED_BUT_INDIRECT` | 関連するが mechanistic distance あり |
| 2 | `2_UNLIKELY_PRIMARY` | 主因とは考えにくい（secondary はあり得る） |
| 1 | `1_NO_CLEAR_MECHANISTIC_LINK` | 明瞭な mechanistic link を認めない |

CSV は target ごとに `score` / `label` / `display` を保存。human-readable report は **display** を primary。

**empirical verdict には numeric score を付けない**（PROMISING 等は categorical）。

これは予測性能の予想ではない。実装 priority と混同しない。

---

## 10. Residual association（univariate / family summary）

```
r_i = y_i - pred_reference_i
```

pre-specified feature summary と r の Pearson / Spearman。many-feature fishing 禁止。

| Label | 条件 |
|-------|------|
| `RESIDUAL_ASSOCIATION_PRESENT` | CV OOF residual で \|Spearman\| ≥ 0.15 かつ Public / Private 同符号 |
| `RESIDUAL_ASSOCIATION_WEAK` | 1–2 split のみ |
| `NO_RESIDUAL_ASSOCIATION` | なし |

---

## 11. Final family statement テンプレート

```
<FAMILY_ID>_<version>

Mechanistic prior (TmApp / HIC): ...
Empirical verdict (TmApp / HIC): ...

Physical association:     PRESENT | WEAK | ABSENT
Standalone signal:        ...
Incremental signal:       ... (CANONICAL_RESIDUAL_RIDGE)
Structure robustness:     ROBUST | MODERATE | FRAGILE
Generator signal:         BOTH | ESMFOLD_ONLY | ABB2_ONLY | MIXED | NONE
Split reproducibility:    ...

Final:
    <one-line>

Evidence limitations:
    ORGANIZER-EXPLORATORY; Public/Private revealed; not prospective evidence
```

---

## 12. Score registry 列との対応

[FEATURE_PROSPECTING_SCORE_REGISTRY.csv](FEATURE_PROSPECTING_SCORE_REGISTRY.csv):

| Column | Content |
|--------|---------|
| `baseline_type` | `TRAIN_MEDIAN_BASELINE` |
| `canonical_incremental_model` | `CANONICAL_RESIDUAL_RIDGE` |
| `residual_model_alpha` | selected alpha |
| `standalone_signal` | §4 |
| `incremental_signal` | §6 |
| `mechanistic_prior` / `evidence_grade` / `empirical_verdict` | §8–9 |
| bootstrap CI columns | §6.2 |
| `structure_robustness` | STRUCTURE_ROBUSTNESS_SPEC |
| `split_reproducibility` | §7 |

---

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`
