# HIC Evaluation Protocol Freeze

**Type:** pre-factorial / pre-experiment evaluation freeze（新規 split 探索・学習なし）  
**Companions:** `reports/HIC_PROTOCOL_MATRIX.csv`, `reports/HIC_CLAIM_VALIDATION.md`, `reports/HIC_CURRENT_STATE_AUDIT.md`

---

## 0. Freeze marker

| Field | Value |
|-------|--------|
| **STATUS** | **`FROZEN FOR FUTURE HIC EXPERIMENTS`** |
| Freeze date | **2026-09-13** |
| Pre-freeze HEAD SHA | **`2f1e178dedb424f1da60fbd16b8b41e4f35f10ee`** |
| Freeze kind | Pre-factorial evidence + evaluation contract |
| New training / feature / embedding / structure / split search / factorial | **Forbidden until human review of this freeze** |

### Future evaluation contract（要約）

| Layer | Contract |
|-------|----------|
| **Primary** | Future HIC **V3** experiments: **`TEST_mean`** only for selection / ranking / factorial cells |
| **Secondary** | Primary vs Shadow **directional replication** of intervention ΔMAE（ranking metric ではない） |
| **Diagnostic** | HIGH-tail (`HIC > 11.5 min`) + calibration；selection 禁止 |
| **External** | GEN_0001 Public/Private **after** internal selection only |

```
HIC_EVALUATION_PROTOCOL_FROZEN
STATUS: FROZEN FOR FUTURE HIC EXPERIMENTS
freeze_date: 2026-09-13
pre_freeze_HEAD: 2f1e178dedb424f1da60fbd16b8b41e4f35f10ee
primary_V3: TEST_mean
legacy_classical_metric: cv_worst_mae (read-only; not for future V3/factorial ranking)
secondary: Primary/Shadow directional ΔMAE replication
external_diagnostic: GEN_0001_B_20271100 Public/Private post-selection only
high_tail_diagnostic: HIC > 11.5 min
no_new_split_search: true
```

---

## 1. Historical protocol map

| Era | Split / CV | 用途 |
|-----|------------|------|
| B6 search | CAND_04974 → bake-off **CAND_12528** | 共通 production 候補 |
| B6.2 / B6.4 / B7 VC / TrustCV / B7.2 | **CAND_12528** | 妥当性・HIGH・仮想コンペ・Trust |
| B7.3 | **GEN_0001_B_20271100** 生成・推奨 | model-blind principled |
| Competition package | **GEN_0001** | 本番 Pub/Priv |
| Drilldown classical | `canonical_simple_tvt_primary_shadow` + GEN_0001 external | EXP-H001–H053 等（**legacy**） |
| Drilldown V3 | `dl_foldlocal_cosine_v3_oof_test` + post-hoc GEN_0001 score | EXP-H054–H139 |

確認: `top_models_feature_bundle/solution.csv` の `is_public`/`is_private` は `competition/organizer/SPLIT_MANIFEST.json` の GEN_0001 と **完全一致**。CAND_12528 との Public 重複は **35/81** のみ → 数値非互換。

---

## 2. Split comparison

| identifier | train n | val/test n | generation | seed | seq similarity | target balance | HIGH coverage | external used? | 今後の役割 |
|------------|--------:|-----------:|------------|------|----------------|----------------|---------------|----------------|------------|
| **GEN_0001_B_20271100** | 162 Dev | Pub81/Priv81 | B7.3 model-blind SA | 20271100 | atomic groups | q-balance + HIGH∈{3,4} | HIGH **4/3** | 設計時 model-blind；スコアは後で | **External diagnostic の正** |
| CAND_12528 | 162 | 81/81 | B6.1 bake-off | （候補ID） | group-aware | HIC safety floor | HIGH 3/4 | モデル rank で選定 | **protocol incompatible（歴史のみ）** |
| CAND_04974 | 162 | 81/81 | B6 common | — | — | — | — | あり | 歴史のみ |
| Dev/Test role_map | 162/162 | — | B3 freeze | — | group 非重複 | — | 母集団全体 | なし | 人口定義 |

**新規により良い split の探索は行わない。**

---

## 3. Future primary evaluation（契約）

### 3.1 V3 / future DL / factorial（唯一の primary）

今後の HIC **V3 系**実験の primary evaluation は:

# **`TEST_mean`**

定義（既存 V3 と同一）:

`TEST_mean = mean(TEST_P, TEST_S)`

ここで `TEST_P` / `TEST_S` は `DL_FOLDLOCAL_COSINE_V3` の Dev OOF MAE（Primary / Shadow schemes）。

| 用途 | `TEST_mean` を使うか |
|------|---------------------:|
| future candidate selection | **YES** |
| V3 model ranking | **YES** |
| factorial cell ranking | **YES** |
| hyperparameter selection（内部） | **YES**（Pub/Priv は使わない） |

任意 tie-break（primary を置き換えない）: `TEST_worst = max(TEST_P, TEST_S)` をより悪い方として参照してよい。

選択時に **GEN_0001 Public/Private を見ない。**

### 3.2 `cv_worst_mae` は legacy read-only

`cv_worst_mae`（`canonical_simple_tvt_primary_shadow` の Primary/Shadow 最悪 MAE）は:

- **historical classical experiments（概ね EXP-H001–H053）を読むための legacy metric**
- **future candidate selection に使わない**
- **V3 model ranking に使わない**
- **factorial cell ranking に使わない**

古典系 `cv_worst_mae` と V3 `TEST_mean` を **直接比較・混在 ranking しない。**

古典実験を再解釈するときは canonical 族内のみで語る。

---

## 4. Historical comparability（契約）

| Family / source | vs future V3 `TEST_mean` | Rule |
|-----------------|-------------------------:|------|
| **EXP-H054–H139** V3 | **直接比較可能**（protocol compatibility が確認できる範囲） | 同一 `dl_foldlocal_cosine_v3_oof_test` の `TEST_mean` |
| Classical EXP-H001–H053 | **NO** | legacy `canonical_simple_*` |
| Gate / models on **CAND_12528** | **NO** | protocol incompatible |
| **Organizer prospecting**（AROMATIC 等） | **NO** | protocol incompatible |
| Gate B7.3 / competition Pub/Priv on GEN_0001 | External 同士のみ | 内部 `TEST_mean` とは混在 ranking しない |

**禁止:** MAE 数値が近いことだけを根拠に、異なる protocol 間を直接 ranking すること。

**過剰一般化の禁止:** H054–H139 が比較可能でも、(a) annotation/seed/HP/platform が変わった将来セル、(b) 非 V3 プロトコル、(c) Pub/Priv Overall との混同、まで自動互換とはみなさない。互換は **同一 V3 OOF 定義の `TEST_mean`** に限定する。

---

## 5. Secondary evaluation（契約）

Secondary の目的は **安定性確認**であり、Primary を置き換える ranking metric **ではない。**

### 5.1 Directional replication（定義）

同一 baseline に対する候補の介入効果:

`ΔMAE = candidate_MAE − baseline_MAE`

（MAE は低いほど良い → **改善は `ΔMAE < 0`**）

**Directionally replicated** とは、原則として:

1. Primary scheme と Shadow scheme で **ΔMAE の符号が一致**する  
2. improvement を主張する場合は **双方で `ΔMAE < 0`**

V3 では scheme MAE は `TEST_P` / `TEST_S` を用いる:

- `Δ_P = candidate_TEST_P − baseline_TEST_P`
- `Δ_S = candidate_TEST_S − baseline_TEST_S`

Primary 集約 `TEST_mean` で候補を選んだあと、Secondary として `Δ_P`/`Δ_S` の符号一致を確認する。

### 5.2 Repository alignment

既存 V3 資産は Primary/Shadow の対を標準で保持する（`protocol_v3` / `experiments.csv` の cv_primary・cv_shadow または TEST_P/TEST_S）。本定義はその対を **介入 Δ の符号再現**に使うものであり、Trust-CV の「不一致時は Public 優先」とは異なる。Trust-CV FOLLOW_PUBLIC は **採用しない**（`HIC_CLAIM_VALIDATION.md` で Tier3 降格）。

Secondary 不一致時: Primary の微小差だけで強い勝敗を主張しない。Secondary 数値で Primary ranking を上書きしない。

分散の参考（ranking ではない）: `|TEST_P − TEST_S|` を報告してよい。

---

## 6. HIGH-tail diagnostic（契約）

| Field | Freeze |
|-------|--------|
| Definition | **`HIC > 11.5 min`**（B6.2 / B6.4 / PROVENANCE） |
| Role | **failure-mode diagnostic only** |
| Required | `n`, `MAE`, **mean signed error (pred−true)**, observed range, predicted range |
| Optional | HIGH→LOW；Spearman（n 小のため参考） |

**HIGH-tail metric を使わない用途:**

- candidate selection  
- hyperparameter selection  
- early stopping  
- representation / topology / annotation selection  
- factorial cell selection  

---

## 7. Public / Private external diagnostic（契約）

| Field | Freeze |
|-------|--------|
| Split | **`GEN_0001_B_20271100` only** |
| When | **internal model selection（Primary ± Secondary）完了後** |
| Metrics | Public MAE, Private MAE；任意 Overall=mean、Spearman |

**使用禁止（internal selection を含む）:**

- model selection  
- hyperparameter tuning  
- representation selection  
- topology selection  
- annotation selection  
- factorial cell selection  
- early stopping  

external score を見た**後**に設計・解釈した解析は、必ず **post-hoc diagnostic** と明記する。  
CAND_12528 の Pub/Priv は歴史引用のみ（比較表に混ぜない）。

---

## 8. CV / metric roles（混同禁止）

| Role | Metric / protocol | Selection? |
|------|-------------------|------------|
| Primary | V3 **`TEST_mean`** | **YES** |
| Legacy read | classical **`cv_worst_mae`** | **NO**（future V3/factorial） |
| Secondary | Primary/Shadow **ΔMAE 符号再現** | **NO**（ranking 置換禁止） |
| Diagnostic | HIGH-tail；calibration slope/intercept；pred range | **NO** |
| External diagnostic | GEN_0001 Pub/Priv | **NO**（事後のみ） |
| Fragile | Pearson | **NO** |

---

## 9. Score comparability map

| Experiment family | Primary compatible? | Direct `TEST_mean` ranking w/ future V3? | Reason |
|-------------------|--------------------:|------------------------------------------:|--------|
| H054–H081 V3 arch | YES | **YES**（互換確認範囲） | 同一 V3 OOF |
| H082–H093 SURFACE fusion | YES | **YES** | 同一 V3 |
| H094–H139 later V3 | YES | **YES** | 同一 V3 |
| H001–H053 classical | legacy | **NO** | canonical_simple |
| Gate on CAND_12528 | NO | **NO** | incompatible |
| Organizer prospecting | NO | **NO** | incompatible |
| GEN_0001 Pub/Priv tables | external | **NO** vs internal `TEST_mean` | 役割が違う |

---

## 10. Claim scope reminder（Established vs NOT）

詳細は `reports/HIC_CLAIM_VALIDATION.md`。Freeze 文書上の言語規則:

### Established observations（書いてよい）

- HIGH-tail で error が大きい  
- HIGH-tail に systematic underprediction がある  
- prediction range compression がある  
- V3 architecture は **tested configuration 内**で plateau を示した  
- SURFACE auxiliary が **tested V3 configurations** で改善を与えた  

### NOT established（Established / 一般因果として書かない）

- HIGH-tail failure の**唯一**の原因は data scarcity  
- representation limitation は存在しない  
- V3 architecture に**一般的な**性能上限がある  
- aromatic chemistry 自体が HIC の因果ドライバー  
- surface representation が**全** HIC prediction に一般的に優越する  

---

## 11. Frozen operating rules（checklist）

1. Future V3 / factorial: rank by **`TEST_mean`**.  
2. Do not use **`cv_worst_mae`** for future V3/factorial selection or cross-protocol ranking.  
3. Secondary = Primary/Shadow **ΔMAE sign replication**; not a replacement ranker.  
4. Record HIGH-tail (`>11.5`) diagnostics every run; never select on them.  
5. Score GEN_0001 Pub/Priv only after internal freeze; never select on them.  
6. Do not mix CAND_12528 / organizer / classical scores into V3 `TEST_mean` leaderboards.  
7. Do not explore new splits under this freeze.  
8. Keep claim language within Established observations above.

### Metric checklist

**Selection（V3）:** `TEST_mean`（+ optional `TEST_worst` tie-break）；Spearman は同スキームの参考可（Pub/Priv 不可）。  

**Diagnostic（必須記録・非選択）:** HIGH n/MAE/signed error/ranges；calibration slope/intercept。  

**External（事後）:** Public MAE, Private MAE。  

**任意:** Pearson, RMSE, HIGH→LOW, Overall。
