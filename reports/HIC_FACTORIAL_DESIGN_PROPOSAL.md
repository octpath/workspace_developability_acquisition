# HIC Factorial Design Proposal

**STATUS:** DESIGN / PREREGISTRATION DRAFT ONLY — **not** preregistered; **not** executable  
**HEAD at draft:** `379e0751a93c2af8f6fbfeedaad4d72f3556996b`  
**Evaluation contract:** `reports/HIC_EVALUATION_PROTOCOL_FREEZE.md`（`FROZEN FOR FUTURE HIC EXPERIMENTS`）  
**Scope:** Representation × Topology × Annotation only — **no** surface / SASA / HSP / late-fusion physics

禁止（本ドラフト段階）: training、GPU job、embedding 新規生成、feature 新規生成、experiment ID 正式発行、registry 登録、Public/Private 評価、cell 実行。

---

## 1. Scientific objective

TmApp では Representation × Topology × Annotation の完全要因計画（200 cell）により、主効果と interaction を系統分離した（`TMAPP_REP_TOPO_ANNOT_FACTORIAL_*`；technical report freeze 済み）。

HIC では同等 factorial は **未実施**。既存 V3（H054–H139）は:

- sequence-only arch 探索でも **annotation=`FULL` 固定**が中心  
- topology 語彙は ARCH-H0/1/2/3/4/6/6G/8 であり、factorial の **SEP/JOINT/REG-SEP/XREG/FUSE と 1:1 ではない**  
- 性能改善の多くは **SURFACE / HSP 等の物理 aux**（本 factorial の対象外）

本設計の問い:

> surface-specific information を混ぜずに、HIC 予測の sequence / PLM pipeline において  
> Representation / Topology / Annotation のどこに **再現可能な headroom** があるか。

目的は TmApp の再演ではなく、**主効果と重要 2-way（必要なら 3-way）interaction を識別できる必要十分設計**である。

---

## 2. Relevant TmApp factorial findings

出典: `developability_drilldown/results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_REPORT.md`、`TMAPP_REP_TOPO_ANNOT_FACTORIAL_PREREG.md`、`technical_report/tmapp_factorial/terminology.yaml`、`TMAPP_TECHNICAL_REPORT.md`。

### 2.1 Grid

| Factor | Levels | n |
|--------|--------|--:|
| Representation | Scratch; AbLingua; AbLang1; AbLang2（鎖別）; AbLang2（H/Lペア）; ESM-1b; ESM-2; ESM-C 600M; CurrAb（鎖別）; CurrAb（H/Lペア） | 10 |
| Topology | SEP, JOINT, REG-SEP, XREG, FUSE | 5 |
| Annotation | BASE, IMGT, REGION, FULL | 4 |
| **Total** | | **200** |

- **REUSE 23** + **new 177**（EXP-T161–T337）  
- FAILED/BLOCKED = 0  
- Platform: `DL_FOLDLOCAL_COSINE_V3`, seed **101**, `d_model=128`, merge **mean**, fold-local LR on VAL, no full-Dev refit  
- Gate: T210 / T321（技術成立のみ；MAE は gate でない）

### 2.2 Topology definitions（推測禁止・terminology 原文要約）

| ID | Display | 定義要点 |
|----|---------|----------|
| **SEP** | Separate H/L | H/L 独立処理；最終 merge 前に下流 H/L 通信なし（legacy A / ARCH-1 mean） |
| **JOINT** | Full Joint | H/L 残基 + REG_H/REG_L の無制限共同 Transformer（legacy B1 / ARCH-3） |
| **REG-SEP** | Joint Residues, Separate REGs | 残基は共同、REG は鎖固有（legacy B2 / ARCH-4） |
| **XREG** | Cross-REG Read | 鎖別符号化後、REG_H↔Light 残基 / REG_L↔Heavy 残基（legacy C / ARCH-7）；残基同士の直接 cross-attend はこの段ではない |
| **FUSE** | REG Fusion | 鎖別符号化後、凍結 D3 で REG_H↔REG_L（legacy D） |

**REG** = 下流 Transformer が処理した残基から情報を集約する学習された鎖レベル・トークン（「summary」と呼称しない）。

### 2.3 Annotation definitions

位置・鎖 ID 埋め込みは全条件でオン。その上で:

| ID | IMGT | CDR/FR |
|----|------|--------|
| BASE | off | off |
| IMGT | on | off |
| REGION | off | on |
| FULL | on | on |

### 2.4 Strongest findings（HIC 設計への含意）

1. **Topology 効果は representation 依存**（単一の勝ち topo なし）。→ HIC でも topo を削ると interaction を失う。  
2. **Annotation は万能改善ではない**（REGION/FULL が常に良いわけではない）。→ FULL 固定の HIC 歴史は annotation 主効果を未評価。  
3. **PLM 推論文脈（鎖別 vs H/Lペア）≠ 下流 topology**。→ AbLang2 / CurrAb の context 対は科学的に残す価値が高い。  
4. **Scratch は mismatched PLM セルと競合し得る**。→ Scratch 対照は必須。  
5. TmApp 最良クラスタは AbLang2 × XREG × REGION 付近。ESM-2 は FUSE 寄り、ESM-C は JOINT+BASE 寄りなど **家族差が大きい**。

### 2.5 Practically weak / mismatched（文書上）

- ESM-C + FULL では interaction Δ が非負寄り（SEP 選好）など、**annot×topo の食い違い**が明示。  
- AbLingua / AbLang1 は AbLang2 より絶対性能で劣位寄り（再演必須ではない候補）。  
- 「redundant cell」公式ラベルは無し；FAILED も無し。

### 2.6 Compute

壁時計 GPU 時間の単一公式値は factorial レポートに未記載。規模感は **新規 177 cell ≈ 本 HIC 設計の上限参照**。

---

## 3. HIC historical constraints

| Constraint | Evidence |
|------------|----------|
| Factorial 未実施 | `HIC_EVIDENCE_MATRIX*`；EXP-H は H139 まで |
| Annotation ≈ FULL only（V3 arch） | H054–H081 configs / master table |
| Rep ≈ ESM2 \| Scratch only（V3 arch） | 同上；AbLang2 等は HIC V3 factorial 語彙では未交差 |
| ARCH ≠ factorial 5-topo | H0/2/6/6G/8/concat は SEP…FUSE 外；**XREG(ARCH-7)/FUSE(D3) は HIC 未実施** |
| Physics aux が強いレバー | H082+ SURFACE/HSP 等 → **本 factorial から除外**（次段 physics） |
| Evaluation freeze | Primary=`TEST_mean`；Pub/Priv は selection 禁止 |

**Bridge 価値:** ESM2/Scratch × SEP/JOINT/REG-SEP × FULL の一部は、後述の exact REUSE 候補として歴史と接続できる。XREG/FUSE と非 FULL annot は **新規が必須**。

---

## 4. Candidate factor levels

### 4.1 Representation（提案セット）

| Include? | Level | 理由 |
|----------|-------|------|
| **YES** | Scratch | 対照；TmApp でも必須 |
| **YES** | AbLang2（鎖別推論） | TmApp 最強族；asset 既存（`ablang2` / SEPARATE_CHAIN） |
| **YES** | AbLang2（H/Lペア推論） | 推論文脈対比；asset 既存（`ablang2_unpaired`） |
| **YES** | CurrAb（鎖別） | 同上・別 PLM；asset 既存 |
| **YES** | CurrAb（H/Lペア） | 同上 |
| **YES** | ESM-2 | HIC 歴史の主 PLM；REUSE bridge |
| **YES** | ESM-1b | TmApp で ESM-2 と異なる topo 選好；asset 既存 |
| **YES** | ESM-C 600M | TmApp で特異パターン（JOINT/SEP）；asset 既存 |
| Design A only | AbLingua | TmApp 劣位寄り；HIC 固有理由弱 |
| Design A only | AbLang1 | 同上（AbLang2 が後継） |

**新規 PLM 追加はしない。** embedding は TmApp factorial 時点で disk 上に存在（`residue_level/*/metadata.json`）。HIC 人口 N=324 は同一 competition population 前提（TmApp と同 bundle 系）。length limitation / context mismatch は TmApp prereg の AbLang2 audit を継承（allocation 名と context の反転に注意）。

### 4.2 Topology（全5を推奨）

SEP / JOINT / REG-SEP / XREG / FUSE — 定義は §2.2。  
HIC では XREG/FUSE が未踏のため、**削減すると最大の科学的損失**になる。

### 4.3 Annotation（全4を推奨）

BASE / IMGT / REGION / FULL。  
HIC V3 が FULL 偏りのため、**annotation 主効果と Rep×Annot / Topo×Annot interaction が本実験の中核的新規情報**。

科学的意味:

- **Rep × Annot:** PLM が既に位置/領域構造を内在化しているか、明示 annot が必要か（TmApp で非一様）。  
- **Topo × Annot:** 領域埋め込みが特定の H/L 通信パターンと噛み合うか（TmApp で REGION が一部 PLM で害）。

---

## 5. Existing reusable cells

**Reuse 規則（厳格）:** target / representation / embedding / topology flags / annotation / regressor（V3 transformer） / split・seed / training / evaluation が **完全一致**。「ほぼ同じ」は不可。

### 5.1 Exact-match candidates（最大 6）

| H code | Factorial cell | 根拠（要約） |
|--------|----------------|--------------|
| **EXP-H056** | ESM-2 × **SEP** × FULL | ARCH-1, mean, arch flags SEP 相当, seed 101, V3, HIC |
| **EXP-H059** | ESM-2 × **JOINT** × FULL | ARCH-3, `arch_joint_hl_dual_reg=true` |
| **EXP-H061** | ESM-2 × **REG-SEP** × FULL | ARCH-4, chain-specific dual REG |
| **EXP-H070** | Scratch × **SEP** × FULL | ARCH-1 mean Scratch |
| **EXP-H073** | Scratch × **JOINT** × FULL | ARCH-3 |
| **EXP-H075** | Scratch × **REG-SEP** × FULL | ARCH-4 |

`share_hl_encoder` は HIC yaml に明示がないが training デフォルト **True**（`training.py`）；TmApp factorial も True。arch_* フラグ・merge=mean・d_model=128・seed=101・platform は TmApp REUSE セルと同型。

**Preregistration 前の必須確認:** 各 REUSE 候補について `share_hl_encoder` / `pair_interaction_mode` / embedding asset hash を checklist で署名し、不一致なら **再学習（REUSE 取り下げ）**。

### 5.2 Non-reusable（明示）

| 帯 | 理由 |
|----|------|
| H054/H068（H-only） | SEP…FUSE 外 |
| H055 等 concat | factorial は mean のみ |
| H057/H071（ARCH-2 single REG） | ≠ JOINT dual-REG |
| H062–H067, H076–H081（ARCH-6/6G/8） | ≠ XREG/FUSE |
| **全 H082–H139** | SURFACE/HSP/SAP/residue/global conditioning 等 — sequence-only factorial 外 |
| 任意の CAND_12528 gate スコア | protocol incompatible |

→ **信頼できる REUSE 上限 = 6**（全 design 共通）。残りは新規。

---

## 6. Design A — Full

**10 × 5 × 4 = 200 cells**（TmApp 同一因子集合）

| | |
|--|--:|
| Reused (max) | 6 |
| New | 194 |
| Total | 200 |

- 全 main effects + 全 2-way + 全 3-way（因子集合内）を識別可能  
- TmApp との **セル定義レベル比較可能性が最大**  
- AbLingua/AbLang1 を含むため compute 最大  
- HIC 固有の新規情報に対し、劣位寄り 2 PLM のフル交差は **限界情報が薄い**可能性

---

## 7. Design B — Reduced（推奨候補）

**8 × 5 × 4 = 160 cells**

**Representation（8）:** Scratch; AbLang2 鎖別; AbLang2 H/Lペア; ESM-1b; ESM-2; ESM-C 600M; CurrAb 鎖別; CurrAb H/Lペア  

**除外:** AbLingua, AbLang1（TmApp で AbLang2 に対し実質的に劣位；HIC に先行投資なし；抗体 PLM 問いは AbLang2 context 対でカバー）

| | |
|--|--:|
| Reused (max) | 6 |
| New | 154 |
| Total | 160 |

- **保持:** 全 Topology × 全 Annotation の完全交差（HIC で最も欠ける次元）  
- **保持:** 推論文脈対（AbLang2, CurrAb）と一般 PLM 多様性（ESM 系）+ Scratch  
- **失うもの:** AbLingua/AbLang1 の HIC 主効果・interaction（TmApp 比較の完全対称性）  
- **失わないもの:** 残 8×5×4 内の 2-way / 3-way interaction 識別能力  

除外は **中間スコアによる cell 削除ではない**（設計時点の事前削減）。

---

## 8. Design C — Staged

**Stage 1:** 8 × 5 × **FULL only** = **40** cells（REUSE ≤6 → new ≥34）  
**Stage 2:** Stage1 の有望 Representation（例: 上位 4）× 5 topo × {BASE, IMGT, REGION} = **60** new  
**Total executed（例）:** ~100（選択ルール次第）

| 利点 | 欠点（重大） |
|------|----------------|
| compute 削減 | Stage2 に入る rep の選択が **selection bias** |
| FULL 下の Rep×Topo は見える | **Annot 主効果・Rep×Annot・Topo×Annot が全因子で識別不能** |
| | 「有望」定義に TEST_mean を使うと、annot 展開前に情報を消費 |

→ interaction 識別という目的に対し **構造的に不足**。compute 節約だけの staged は非推奨。

---

## 9. Comparison

| Design | New cells | Reused | Total | Main effects | 2-way interactions | Bias risk | Compute | Interpretability |
|--------|----------:|-------:|------:|--------------|--------------------|-----------|---------|------------------|
| **A Full** | 194 | ≤6 | 200 | All 10×5×4 factors | Complete within 10 reps | Low（事前固定） | ≈ TmApp new（177）超 | Max TmApp parity |
| **B Reduced** | 154 | ≤6 | 160 | 8×5×4 complete | Complete within 8 reps | Low（事前除外のみ） | ≈ **0.87×** Design A；≈ **0.87×** TmApp-200 | HIC 目的に十分；2 PLM 欠 |
| **C Staged** | ~94+ | ≤6 | ~100 | Annot 不完全 | Annot 関連 2-way 欠損 | **High** | ≈ **0.5×** A | Interaction 解釈が歪む |

追加評価:

| 軸 | A | B | C |
|----|---|---|---|
| 科学的情報量（HIC headroom） | 高 | **高（核心を保持）** | 中〜低 |
| selection bias | 低 | 低 | **高** |
| TmApp 比較可能性 | **最大** | 高（8 共通 rep） | 部分 |
| HIC 特有問い（annot/XREG/FUSE） | 充足 | **充足** | 部分 |
| compute 効率 | 低 | **中** | 高だが情報損失 |

---

## 10. Recommended design

# **Design B — Reduced factorial（8 × 5 × 4 = 160）**

### 推奨理由

1. HIC の空白は **Annotation 横断**と **XREG/FUSE** と **抗体 PLM（AbLang2/CurrAb）** であり、これらを落とさない。  
2. AbLingua/AbLang1 のフル交差は TmApp 再演色が強く、HIC headroom 問いへの追加情報は限定的（事前除外；中間結果での削除ではない）。  
3. Design C は interaction を構造的に失うため不採用。  
4. Design A は比較可能性最大だが、**必要十分を超える compute** になりやすい。人間が「TmApp 完全対称が必須」と判断した場合のみ A に切り替え。

### Factor levels（推奨）

- **R (8):** Scratch; AbLang2（鎖別）; AbLang2（H/Lペア）; ESM-1b; ESM-2; ESM-C 600M; CurrAb（鎖別）; CurrAb（H/Lペア）  
- **T (5):** SEP; JOINT; REG-SEP; XREG; FUSE  
- **A (4):** BASE; IMGT; REGION; FULL  

### Cells

| | Count |
|--|------:|
| Total | **160** |
| Reusable (max exact) | **6**（H056/H059/H061/H070/H073/H075） |
| New | **154**（REUSE 確認後に確定） |

### Measurable interactions（主要）

- Rep × Topo（TmApp の中核再現；HIC で未測）  
- Rep × Annot / Topo × Annot（HIC FULL 偏りを是正）  
- AbLang2 / CurrAb の **inference context × Topo**（3-way は探索的に報告可）  
- Scratch vs PLM の topo 依存（surface なし）

### Largest scientific compromise

AbLingua / AbLang1 を入れない → それら PLM の HIC 性能と interaction は **本バッチでは不明のまま**（必要なら後続の小さい補完格子で追加可）。

### Surface

本設計に **一切含めない**。physics は別 prereg。

---

## 11. Evaluation contract（全 cell 共通）

`reports/HIC_EVALUATION_PROTOCOL_FREEZE.md` に従う。

| Layer | Rule |
|-------|------|
| **Primary** | V3 **`TEST_mean`**（candidate / cell ranking / factorial ranking） |
| **Secondary** | 同一 baseline に対し `ΔMAE = candidate − baseline` の Primary/Shadow **符号一致**（改善は双方 `<0`）。ranking 置換禁止 |
| **Diagnostic** | `HIC > 11.5 min`：n, MAE, mean signed error, obs/pred range。**selection 禁止** |
| **External** | GEN_0001 Public/Private は **internal freeze 後のみ**。設計・cell selection・HP・rep/topo/annot 選択に使用禁止 |
| Legacy | `cv_worst_mae` は classical 読取専用；本 factorial に使わない |

---

## 12. Analysis plan（実行前に固定）

TmApp 分析パッケージの再利用:

- `technical_report/tmapp_factorial/scripts/build_report_package.py`  
- master CSV / topology Δ vs SEP / annotation Δ vs BASE / bootstrap tables / DiD notes  
- terminology（SEP…FUSE, BASE…FULL, AbLang2 表記規則）

| Analysis | Plan |
|----------|------|
| Cell-level | 全 160 の `TEST_P/S`, `TEST_mean`, `TEST_worst`, \|P−S\| |
| Main effect | 周辺平均（Rep / Topo / Annot）；基準 SEP および BASE |
| Pairwise contrast | 同一他因子固定のペア；**paired bootstrap on OOF**（TmApp と同様） |
| Interaction | Rep×Topo, Rep×Annot, Topo×Annot の Δ ヒート／表；3-way は探索的 |
| DiD | 多重比較 DiD は **Primary と Shadow が一致しない限り探索的**（TmApp technical report 方針） |
| Ranking stability | mean vs worst；Secondary 方向再現 |
| Uncertainty | 主要 contrast に paired bootstrap；CI が 0 を含む場合は強い主張を避ける |
| Factor vs cell | 結論は **パターン（因子レベル）** を優先；単一セル優勝は記述的 |
| External | internal freeze 後にのみ算出し診断；結論の書き換え禁止 |
| HIGH-tail | 全 cell diagnostic 記録；因子選択に使わない |

---

## 13. Technical gates

Gate は **技術的成立のみ**。MAE / 科学的都合での cell 削除は禁止。

提案ゲート（TmApp 類似）:

1. **REUSE checksum gate:** H056/H059/H061/H070/H073/H075 の config・asset・metric 再読取が計画セルと exact match。不一致 → その cell は new に降格。  
2. **Embedding load gate:** 8 representation の HIC 用 residue bundle が読めること（新規抽出が必要なら **別タスクとして人間承認**；本ドラフトでは実行しない）。  
3. **Annotation wiring gate:** BASE/IMGT/REGION/FULL がモデルに正しく載る smoke（1 rep × 1 topo × 4 annot または最小）。  
4. **Topology smoke:** 同一 rep で SEP と XREG または FUSE が学習完了し予測分散 >0。  
5. **Checkpoint compatibility:** V3 protocol・seed 101・mean merge・REG pooling。

PASS 基準案（TmApp PREREG 踏襲）: extraction/load QC + train/eval 完了 + nonzero prediction variance + artifacts。**MAE 品質は gate でない。**

---

## 14. Expected compute

| Reference | Cells new-ish | Relative |
|-----------|---------------|----------|
| TmApp factorial | 177 new (+23 REUSE) | 1.0×（参照） |
| Design A | ~194 new | ≈ **1.1×** TmApp new |
| **Design B（推奨）** | ~154 new | ≈ **0.87×** TmApp new；≈ **0.8×** Design A |
| Design C | ~94+ new | ≈ **0.5×** A（情報損失大） |

壁時計はハードウェア依存のため未推定。cell 単価は TmApp V3 と同オーダー想定。

---

## 15. Risks / confounders

| Risk | Mitigation |
|------|------------|
| REUSE の隠れ不一致（share_hl 明示欠落等） | prereg 前 checksum；疑義は再学習 |
| AbLang2 allocation 名と context の混同 | TmApp と同じ display 規則を強制 |
| Surface 歴史との混同 | 本格子に物理 aux を入れない；比較表で分離 |
| External 覗き見 | freeze 契約；分析パイプラインで後段のみ |
| HIGH-tail で格子を歪める | diagnostic only |
| Design B の 2 PLM 除外 | 妥協として明記；必要なら補完格子 |
| ESM-2 HIC 歴史 plateau の過剰一般化 | 本 factorial は annot/topo/他 PLM を変えるため別主張 |

---

## 16. Exact preregistration items needed before execution

人間承認後に初めて実施。本ドラフトでは **ID 発行しない**。

1. Final design lock（B 確定 or A への切替決定）  
2. Exact cell table CSV/YAML（rep × topo × annot × reuse_code）  
3. REUSE checksum report（6 候補の pass/fail）  
4. Embedding inventory（path, hash, context, coverage N=324）  
5. Next free EXP-H code range reservation（発行は承認後）  
6. Platform freeze echo: `DL_FOLDLOCAL_COSINE_V3`, seed 101, mean, REG, d_model=128, LR grid  
7. Gate list + PASS 定義（MAE 除外）  
8. Analysis plan hash（本 §12 の凍結）  
9. Evaluation contract citation（`HIC_EVALUATION_PROTOCOL_FREEZE.md` + commit）  
10. Binding commitments: 中間スコアで PLM/topo/annot を落とさない；Pub/Priv で設計変更しない；surface を混ぜない  
11. Pre-reg git SHA + plan file SHA256  
12. No-training attestation until prereg commit

---

## End matter — draft summary for reviewers

| Item | Value |
|------|--------|
| **Recommended design** | **Design B — Reduced 8×5×4** |
| Factor levels | R=8（Scratch, AbLang2×2, ESM-1b, ESM-2, ESM-C, CurrAb×2）; T=5; A=4 |
| Total cells | **160** |
| Reusable cells | **≤6**（H056/H059/H061/H070/H073/H075） |
| New cells | **≥154** |
| Major interactions measurable | Rep×Topo, Rep×Annot, Topo×Annot；context×Topo（探索的） |
| Largest compromise | AbLingua/AbLang1 除外 |
| Compute vs TmApp factorial | ≈ **0.87×** new-cell count（Design A ≈1.1×） |
| HEAD SHA | `379e0751a93c2af8f6fbfeedaad4d72f3556996b` |

**人間が本設計を承認するまで preregistration / training を開始しない。**
