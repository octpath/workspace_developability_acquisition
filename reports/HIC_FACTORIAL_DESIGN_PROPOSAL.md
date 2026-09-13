# HIC Factorial Design Proposal

**STATUS:** DESIGN REVISED — awaiting human approval of preregistration draft  
**Not** formally preregistered; **not** executable  
**Evaluation freeze SHA:** `379e0751a93c2af8f6fbfeedaad4d72f3556996b`  
**Evaluation contract:** `reports/HIC_EVALUATION_PROTOCOL_FREEZE.md`  
**Companion prereg draft:** `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_PREREG_DRAFT.md`  
**Scope:** Representation × Topology × Annotation only — **no** surface / SASA / HSP / late-fusion physics

禁止（本段階）: training、GPU、embedding 新規生成、EXP-H 正式発行、registry 変更、Public/Private 評価、prereg FROZEN 化、実行スクリプト起動。

---

## 0. FINAL HUMAN DECISION

# **FINAL HUMAN DECISION = DESIGN A — Full factorial**

| Field | Decision |
|-------|----------|
| Representation | **10**（TmApp 同一；AbLingua / AbLang1 **含む**） |
| Topology | **5**（SEP / JOINT / REG-SEP / XREG / FUSE） |
| Annotation | **4**（BASE / IMGT / REGION / FULL） |
| **Total cells** | **200** |
| Reuse | **0–6**（checksum 後；現時点 confirmed = 未確定） |
| New training | **194–200** |

**Design B（8×5×4）の事前 representation 削減推奨は撤回する。**  
Design B / C は比較履歴として残すが、最終採用ではない。

### Human rationale（レビュー決定）

* AbLang1 除外でも削減は 20 cells のみ；AbLingua+AbLang1 でも 40 cells のみ  
* 目的は最高性能探索だけでなく **TmApp と HIC の representation 依存性比較**  
* 10×5×4 により TmApp との **cell-definition-level parity が最大**  
* Rep×Topo / Rep×Annot / Topo×Annot を **全 representation** で解釈可能  
* 中途半端な事前 representation selection を避ける  
* Compute: total-cell ratio vs TmApp = **200/200 = 1.00×**；new-cell ratio（reuse=6 仮定）= **194/177 ≈ 1.10×**（許容）

---

## 1. Scientific objective

TmApp では Representation × Topology × Annotation 完全要因計画（200 cell）により主効果と interaction を分離した。

HIC では同等 factorial は **未実施**。既存 V3（H054–H139）は annotation=`FULL` 偏り、ARCH 語彙 ≠ factorial 5-topo、性能レバーの多くが surface physics（本格子の外）。

本実験の問い:

> surface を混ぜずに、HIC の sequence / PLM pipeline で Representation / Topology / Annotation のどこに再現可能な headroom があるか。  
> 併せて、同一 10×5×4 定義で **TmApp vs HIC の因子依存性を比較**できるか。

---

## 2. Relevant TmApp factorial findings

（前回ドラフト §2 を維持。要約のみ）

* 10×5×4=200；REUSE 23 + new 177；FAILED=0  
* Topology は representation 依存；Annotation は万能でない  
* PLM 推論文脈 ≠ 下流 topology  
* 定義: `terminology.yaml` / TmApp factorial freeze  

---

## 3. HIC historical constraints

* Factorial 未実施；V3 arch は FULL + ESM2|Scratch 中心  
* XREG/FUSE 未実施（factorial 語彙）  
* H082+ physics aux は本格子から除外  
* Evaluation: Primary=`TEST_mean`；Pub/Priv selection 禁止  

---

## 4. Candidate factor levels — FINAL（Design A）

### 4.1 Representation — 10（TmApp 同一；metadata context が正）

| # | Display (analysis) | Internal / asset notes | `representation_context` |
|--:|--------------------|------------------------|--------------------------|
| 1 | Scratch | learned AA；no frozen PLM | LEARNED_AA |
| 2 | AbLingua | `ablingua` / ablingua600m | SEPARATE_CHAIN |
| 3 | AbLang1 | `ablang1` | SEPARATE_CHAIN |
| 4 | AbLang2（鎖別推論） | historical alloc `ablang2_paired` → asset `ablang2` | **SEPARATE_CHAIN** |
| 5 | AbLang2（H/Lペア推論） | historical alloc `ablang2_unpaired` | **PAIRED_NATIVE** |
| 6 | ESM-1b | `esm1b` | SEPARATE_CHAIN |
| 7 | ESM-2 | `esm2` | SEPARATE_CHAIN |
| 8 | ESM-C 600M | `esmc600m` | SEPARATE_CHAIN |
| 9 | CurrAb（鎖別推論） | `currab_unpaired` | **SEPARATE_CHAIN** |
| 10 | CurrAb（H/Lペア推論） | `currab` / `currab_paired` | **PAIRED_NATIVE** |

**表示は allocation 名ではなく metadata 上の実 context を正とする**（TmApp technical report と同じ）。

### 4.2 Hierarchical representation interpretation（解析用；格子は 10 levels）

**Family（8）:** Scratch; AbLingua; AbLang1; AbLang2; ESM-1b; ESM-2; ESM-C; CurrAb  

**Inference context（該当 family）:** SEPARATE_CHAIN; PAIRED_NATIVE  

主格子は TmApp 整合のため **10 representation levels**。AbLang2 / CurrAb は追加で **family × context** contrast を事前定義（prereg draft §7–9）。

### 4.3 Topology — 5

SEP / JOINT / REG-SEP / XREG / FUSE — TmApp `terminology.yaml` 正本。

### 4.4 Annotation — 4

BASE / IMGT / REGION / FULL — 位置・鎖 ID は全条件オン（TmApp 同一定義）。

---

## 5. Existing reusable cells

### Provisional candidates（max 6）

H056 (ESM-2×SEP×FULL), H059 (ESM-2×JOINT×FULL), H061 (ESM-2×REG-SEP×FULL),  
H070 (Scratch×SEP×FULL), H073 (Scratch×JOINT×FULL), H075 (Scratch×REG-SEP×FULL)

### Pre-gate validation snapshot（本改訂時）

| Check | Status |
|-------|--------|
| target=HIC, platform V3, seed 101, merge=mean, annot=FULL | OK（yaml / OOF yaml） |
| topology arch flags SEP/JOINT/REG-SEP | OK |
| OOF Primary/Shadow artifacts | OK（`oof_test_*.csv` 存在） |
| `share_hl_encoder` in yaml / run_state | **不明**（未記録；code default True） |
| `pair_interaction_mode` explicit | **不明/未記録**（SEP/JOINT/REG-SEP では null 想定） |
| embedding content hash pinned in cell record | **要 gate** |

**規則:** 1 項目でも不一致または不明なら REUSE しない → 現時点 **confirmed REUSE = 0**；gate 通過後のみ 1–6。  
したがって計画レンジ: **reuse 0–6；new 194–200**。

H082–H139 および ARCH-H0/2/6/6G/8/concat は sequence-only factorial に **NON-reusable**。

---

## 6. Design A — Full（ADOPTED）

**10 × 5 × 4 = 200**

| | |
|--|--:|
| Total | 200 |
| Reused | 0–6 |
| New | 194–200 |

全 main effects；全 2-way；10-level 格子内 3-way；TmApp parity 最大。

---

## 7. Design B — Reduced（not adopted；history）

8×5×4=160（AbLingua/AbLang1 除外）。**人間決定により不採用。** 事前除外推奨は撤回。

---

## 8. Design C — Staged（not adopted）

FULL 先行 → annot 展開。selection bias と Annot interaction 欠損のため不採用。

---

## 9. Comparison（updated）

| Design | New (approx) | Reused | Total | Main effects | 2-way | Bias risk | Compute notes | Status |
|--------|-------------:|-------:|------:|--------------|-------|-----------|---------------|--------|
| **A Full** | 194–200 | 0–6 | **200** | All 10×5×4 | Complete | Low | total **1.00×** TmApp；new≈**1.10×** if reuse=6 | **ADOPTED** |
| B Reduced | 154–160 | 0–6 | 160 | 8×5×4 | Complete in 8 | Low | new≈0.87× TmApp-177 | Rejected by human |
| C Staged | ~94+ | 0–6 | ~100 | Annot incomplete | Annot 2-way lost | **High** | ~0.5× A | Rejected |

### Compute denominators（混同禁止）

| Ratio | Definition | Value |
|-------|------------|------:|
| **Total-cell ratio vs TmApp** | `200 / 200` | **1.00×** |
| **New-cell ratio vs TmApp new**（reuse=6） | `194 / 177` | **≈1.10×** |
| **New-cell ratio**（reuse=0） | `200 / 177` | **≈1.13×** |

---

## 10. Recommended design

# **Design A — Full factorial（10 × 5 × 4 = 200）**

**FINAL HUMAN DECISION。**

### Factor levels

- **R (10):** Scratch; AbLingua; AbLang1; AbLang2 SEPARATE_CHAIN; AbLang2 PAIRED_NATIVE; ESM-1b; ESM-2; ESM-C 600M; CurrAb SEPARATE_CHAIN; CurrAb PAIRED_NATIVE  
- **T (5):** SEP; JOINT; REG-SEP; XREG; FUSE  
- **A (4):** BASE; IMGT; REGION; FULL  

### Cells

| | Count |
|--|------:|
| Total | **200** |
| Reusable | **0–6** |
| New | **194–200** |

### Predefined primary contrasts（詳細は prereg draft）

* Topology − SEP（同一 Rep×Annot）  
* Annotation − BASE（同一 Rep×Topo）  
* PLM − Scratch（同一 Topo×Annot；absolute ranking と分離）  
* PAIRED_NATIVE − SEPARATE_CHAIN（AbLang2 / CurrAb；同一 Topo×Annot）  

### Interactions

* Rep × Topo；Rep × Annot；Topo × Annot（主）  
* AbLang2/CurrAb: Family×Context×Topo / ×Annot（探索的；10-level 解析と混同しない）  

### Largest remaining compromise

Surface/physics を意図的に除外（次段）。REUSE が 0 なら compute 上振れ（≈1.13× new）。

### Surface

本設計に **一切含めない**。

---

## 11. Evaluation contract

変更なし（freeze）:

| Layer | Rule |
|-------|------|
| Primary | `TEST_mean = mean(TEST_P, TEST_S)` |
| Secondary | `Δ_P`/`Δ_S` 符号一致；改善は原則双方 `<0`；ranking 置換禁止 |
| HIGH-tail | `HIC > 11.5`；記録のみ；selection 禁止 |
| External | GEN_0001 Pub/Priv；internal freeze 後のみ；selection/解釈書き換え禁止 |

---

## 12. Analysis plan

* **Level 1:** factor-pattern conclusions  
* **Level 2:** predefined paired contrasts + bootstrap  
* **Level 3:** individual best cell（記述のみ；主要結論にしない）  
* Bootstrap: TmApp と同型の **antibody-level paired residual bootstrap**（N_BOOT=2000, seed=101）を HIC OOF Primary/Shadow に適用（prereg draft で根拠明記）  
* Cross-target TmApp vs HIC: **HIC internal freeze 後**；TmApp 結果で HIC cell selection しない  

---

## 13. Technical gates

1. REUSE checksum  
2. Representation asset inventory / coverage  
3. Annotation wiring smoke  
4. Topology wiring smoke  
5. Primary/Shadow prediction smoke  
6. Artifact completeness / nonzero variance  
7. Analysis pipeline dry-run  

MAE で pass/fail しない。科学的 cell 削除禁止。

---

## 14. Expected compute

| Item | Count / ratio |
|------|----------------|
| Design A total | 200 |
| TmApp total | 200 → **total-cell ratio 1.00×** |
| TmApp new | 177 |
| HIC new if reuse=6 | 194 → **new-cell ratio ≈1.10×** |
| HIC new if reuse=0 | 200 → **new-cell ratio ≈1.13×** |

---

## 15. Risks / confounders

| Risk | Mitigation |
|------|------------|
| REUSE share_hl 不明 | 不明なら new；gate 必須 |
| AbLang2/CurrAb naming | context metadata 正；display 規則固定 |
| Surface 混同 | 格子から除外 |
| External / HIGH-tail 汚染 | freeze 契約 |
| 10-level vs family/context 混同 | 階層解析を別セクション |
| TmApp 結果で HIC 選択 | 明示禁止 |

---

## 16. Items before formal preregistration / execution

1. 人間が prereg **draft** を承認  
2. REUSE checksum 完了 → reuse 0–6 確定  
3. Embedding inventory + hashes  
4. Exact 200-cell plan CSV/YAML  
5. EXP-H code 範囲の **予約/発行**（承認後のみ）  
6. Formal prereg commit（FROZEN）— **未実施**  
7. Gates → training  

---

## End matter

| Item | Value |
|------|--------|
| **FINAL design** | **Design A — Full 10×5×4** |
| Total cells | **200** |
| Reuse | **0–6** |
| New | **194–200** |
| Evaluation freeze | `379e0751…` |
| Prereg draft | `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_PREREG_DRAFT.md` |

**ここで停止。人間承認まで formal prereg freeze / ID 発行 / training に進まない。**
