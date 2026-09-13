# HIC Claim Validation

**Type:** claim re-validation only（新規学習・特徴量・split探索なし）  
**HEAD (start):** `2f1e178dedb424f1da60fbd16b8b41e4f35f10ee`  
**Inputs:** `reports/HIC_EVIDENCE_MATRIX.csv`, `reports/HIC_CURRENT_STATE_AUDIT.md`, 一次 metrics/reports  
**Output companion:** `reports/HIC_EVIDENCE_MATRIX_VALIDATED.csv`

---

## 1. Executive summary

元の Evidence Matrix の **Tier1=12 / Tier2=12** を一次ソースまで遡って再評価した。

| 結果 | 内容 |
|------|------|
| Tier1 → Tier1 | **12 / 12**（ただし複数を **スコープ縮小**；一般化読みは別 claim として Tier3/4 へ分離） |
| Tier2 → Tier2 | **9 / 12** |
| Tier2 → Tier3 | **3**（非対称仮説・Trust-CV 将来適用・organizer BEST_CV） |
| 新規 split claims | HIGH-tail / architecture ceiling / SURFACE 一般化 / 因果 などを分離 |

最重要の是正:

1. **HIGH-tail の「現象」と「原因説明」を分離** — 過小予測・大誤差・レンジ圧縮は Tier1；「データ不足が唯一の原因」「表現不足ではない」は **未確立**。
2. **V3 ≈0.50 は `OBSERVED_PLATEAU_UNDER_TESTED_CONFIGURATION`** — `GENERAL_ARCHITECTURE_CEILING` ではない。
3. **SURFACE late fusion は「V3 で試した最有力レバー」** — 「一般に優れた HIC representation」ではない。
4. **露出芳香族の因果主張は Established にしない**（repository 自身が非因果）。

---

## 2. Original Tier 1 / 2 claims

元ファイル `HIC_EVIDENCE_MATRIX.csv` より（要約）:

### Tier 1（12）

1. 連続回帰を維持（二値化しない）
2. HIGH-tail 系統的過小予測 / regression-to-center
3. Pearson 脆弱 → MAE primary
4. 本番 split = GEN_0001（≠ CAND_12528）
5. プロトコル横断スコア非互換
6. SURFACE late fusion が V3 DL を改善
7. LOCAL_RASA 単独 fusion は無効
8. V3 architecture が TEST_mean≈0.50 で頭打ち
9. STATIC_SAP_KD NOT_SUPPORTED
10. SOURCE24 / SOURCE_SAP24 NOT_SUPPORTED
11. Residue F1 は Ab-level F1 に劣る
12. TmApp factorial は HIC 未実施

### Tier 2（12）

1. Geometry ARCH-6G 非勝利  
2. HIC=物理特徴 / Tm=表現・位相 の非対称  
3. HSP BM/EIS 加法  
4. 露出芳香族幾何が配列 YWF を超える  
5. ESMFN aromatic SASA ↔ HIGH 関連  
6. H047 古典 champion  
7. Global F1 conditioning は late fusion より正当化されない  
8. Trust-CV: Public ≥ CV（CAND_12528）  
9. Virtual competition: Pub 過学習限定；overall≠HIGH 改善  
10. Organizer BEST_CV equal-mean  
11. Generic hydro / OpenMM 等は Round1 surface 超の増分なし  
12. HIGH learnability は n≈13 で data-limited  

---

## 3. Claim-by-claim validation

略号: **R**=replication, **C**=comparability, **E**=external class, **VT**=validated tier。

### Tier1 originals

| # | Claim | R | C | E | VT | 判定要点 |
|---|-------|---|---|---|----|----------|
| 1 | Keep continuous | 複数ゲート | 定義は全HIC共通 | MIXED | **T1** | 決定事項として維持 |
| 2 | HIGH underpred / RTC | 複数モデル×CV/Pub/Priv | **CAND_12528 数値**；帯定義は可搬 | MIXED | **T1** | 現象としては強い。原因は分離（§4） |
| 3 | Pearson fragile | 2 splits | 方針は可搬 | INTERNAL | **T1** | B6.2 + B7.3 |
| 4 | GEN_0001 production | provenance | 定義 | INTERNAL | **T1** | bundle `solution.csv` の Pub/Priv が GEN_0001 と **81/81 一致**；CAND とは Pub 重複 **35/81** |
| 5 | Non-interchangeable scores | protocol inventory | meta | INTERNAL | **T1** | 維持 |
| 6 | SURFACE improves V3 DL | 3 backbones + perm | **V3 のみ** | MIXED | **T1** | vs matched seq backbone。一般化は別 claim（downgrade） |
| 7 | LOCAL_RASA alone no | 3 backbones | V3 | MIXED | **T1** | 維持 |
| 8 | V3 plateau ~0.50 | **28 cells** | annot=FULL, seed=101, ESM2\|Scratch, LR grid 固定 | INTERNAL | **T1（再スコープ）** | `OBSERVED_PLATEAU…`。一般 ceiling は Tier4 新 claim |
| 9 | STATIC_SAP_KD no | 8 cells | V3 | MIXED | **T1** | 定義限定の否定結果 |
| 10 | SOURCE24 no | multi-cell | V3 | MIXED | **T1** | 維持 |
| 11 | Residue F1 < Ab F1 | 6 cells | V3 | MIXED | **T1** | CASE G |
| 12 | No HIC factorial | registry absence | N/A | INTERNAL | **T1** | 維持 |

### Tier2 originals

| # | Claim | VT | 要点 |
|---|-------|----|------|
| 1 | Geometry not win | **T2** | 限定セル + ablation |
| 2 | Asymmetric HIC vs Tm | **T3 ↓** | 単一批 synthesis；解釈 |
| 3 | HSP BM/EIS additive | **T2** | 支持だが bootstrap CI が 0 を含むことが多い |
| 4 | Geometry > seq YWF | **T2** | POST_PRIVATE；Round1 には redundant |
| 5 | ESMFN aromatic↔HIGH | **T2** | n=13 関連のみ |
| 6 | H047 champion | **T2** | canonical_simple 限定 |
| 7 | Global F1 not justified | **T2** | CASE D；内部優先なら妥当 |
| 8 | Trust-CV Public≥CV | **T3 ↓** | **CAND_12528 限定**；GEN_0001 へ非転用 |
| 9 | VC limited overfit; MAE≠HIGH | **T2** | split-scoped + HIGH 教訓は定性可搬 |
| 10 | Organizer BEST_CV | **T3 ↓** | postmortem 工学記録 |
| 11 | Generic hydro no Δ | **T2** | POST_PRIVATE 否定結果 |
| 12 | Data-limited by n≈13 | **T2** | **INTERPRETATION**；唯一原因ではない |

---

## 4. Observation vs interpretation separation

### HIGH-tail（重点）

| ID | Claim | Type | Validated |
|----|-------|------|-----------|
| H1 | HIGH で absolute error が大きい | OBSERVATION | **Tier1** |
| H2 | HIGH が systematic underprediction | OBSERVATION | **Tier1** |
| H3 | prediction dynamic range が圧縮（under-dispersion） | OBSERVATION | **Tier1** |
| H4 | HIGH failure が data-limited（寄与要因） | INTERPRETATION | **Tier2** |
| H4′ | HIGH failure の**唯一**原因がデータ不足 | CAUSAL | **Tier4** |
| H5 | HIGH failure は representation-limited **ではない** | CAUSAL | **Tier4（非支持）** |

根拠: B6.2 band/shrinkage；B6.4 は scarcity **STRONG** と同時に *partially learnable*・*incomplete surface/physchem* も記載。H1–H3 から H4′/H5 は導けない。

### V3 architecture

| Claim | Type | Validated |
|-------|------|-----------|
| 試験構成下で TEST_mean∈[0.5017,0.5567] の plateau | OBSERVATION | **Tier1** |
| 表現・annotation・HP 非依存の一般 architecture ceiling | INTERPRETATION | **Tier4** |

設定事実（configs）: `annotation_mode=FULL`；`seed=101`；`plm_source∈{ESM2,NONE}`；`lr_grid=[1e-5,1e-4,1e-3,1e-2]`。

### SURFACE

| Claim | Type | Validated |
|-------|------|-----------|
| AROMATIC_TOPO+HYDRO_FIELD late fusion が seq-only V3 backbone より改善 | OBSERVATION | **Tier1** |
| SURFACE が一般に優れた HIC representation | INTERPRETATION | **Tier3** |

### Aromatic / geometry

| Claim | Type | Validated |
|-------|------|-----------|
| seq YWF count に信号 | OBSERVATION | 弱い単独（audit で seq-only は劣る） |
| exposed aromatic / AROMATIC-TOPO が予測信号 | OBSERVATION | **Tier2** |
| structure-derived exposure が配列を超える | OBSERVATION | **Tier2**（GEOMETRY_ADDS_SIGNAL） |
| spatial topology が必要 | 未分離十分 | **未決 / Suggestive** |
| hydrophobic field 一般が増分 | OBSERVATION(neg) | **Tier2**（増分なし） |
| aromatic chemistry が因果 | CAUSAL | **Tier4** |

---

## 5. Re-tiered claims

集計（validated ファイル全体、split 含む）:

| Tier | Count |
|------|------:|
| Tier1 Established | 14 |
| Tier2 Supported | 9 |
| Tier3 Suggestive | 7 |
| Tier4 Unresolved/contradictory | 7 |

元 Tier1/2 に限定した遷移:

- Tier1: **12 → 12**（Tier1 のまま；うち V3/SURFACE は文言スコープ縮小）
- Tier2: **12 → 9 Tier2 + 3 Tier3**

詳細は `HIC_EVIDENCE_MATRIX_VALIDATED.csv`。

---

## 6. Important downgrades / upgrades

### Downgrades（元 Tier2 → Tier3）— 3件

1. **HIC vs TmApp 非対称仮説** — 単一 synthesis；factorial 欠如。  
2. **Trust-CV「Public≥CV」** — CAND_12528 の歴史的観察に降格；将来 primary 規則にしない。  
3. **Organizer BEST_CV equal-mean** — postmortem 工学値；科学的確立から外す。

### Overclaim demotions（新 claim / 読みの Tier4・Tier3）

4. **GENERAL_ARCHITECTURE_CEILING** → Tier4（観測 plateau は Tier1 維持）。  
5. **SURFACE = generally superior representation** → Tier3。  
6. **HIGH の唯一原因=データ不足** → Tier4。  
7. **HIGH は representation-limited ではない** → Tier4（非支持）。

### Upgrades

なし（元 Tier3/4 を Tier1/2 へ上げたものなし）。

---

## 7. Remaining uncertainty

1. GEN_0001 上での Trust-CV / VC 相当の再検証は **未実施**（新規実験禁止のため本タスクでも未実施）。  
2. HIGH-tail が loss・表現・標本のどれに最も応答するかは **未介入**。  
3. AbLang2 等を含む HIC factorial は **未実施** → architecture/representation 一般論は未決。  
4. HSP 加法の bootstrap CI はしばしば 0 を含む → Tier2 のまま。  
5. GEOMETRY_ADDS_SIGNAL は post-private；Round1 増分とは緊張関係。  
6. V3 の seed は実質単一（101）— multi-seed 安定性は未検証。

**次の HIC 実験は、本 validation と protocol freeze の人間確認後まで起動しない。**
