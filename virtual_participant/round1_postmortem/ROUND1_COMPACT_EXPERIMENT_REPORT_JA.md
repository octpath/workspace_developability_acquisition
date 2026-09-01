# Virtual Participant Round 1 — 実験結果サマリー

**状態:** `ROUND1_COMPACT_EXPERIMENT_REPORT_FINAL_FROZEN`

---

## 1. このレポートについて

Round 1 で実施した代表的な実験を、**Target 別**に時系列で整理したコンパクトなサマリーである。同一 Stage 内の細かなハイパーパラメータ違いは代表モデルに集約し、読者が TmApp / HIC それぞれの探索経路を上から追える構成にした。スコアの **CV** は Primary CV MAE、**Public / Private** は organizer reveal 後の Test 評価（各 N=81）である。**MAE は低いほど良い。** Public / Private 列には、reveal 前に hash freeze 済みの **pre-reveal frozen** モデルと、reveal 後に同一 frozen experiment specification を Test へ適用して生成した **postmortem exploratory** モデルの双方が含まれる（Round 1 PRIMARY の選択は reveal 前に freeze 済み）。Stage 1 の古典特徴 screening など、Test prediction を個別生成していない行は Public / Private を「—」とした（CV のみ参照）。

---

## 2. TmApp

### 2.1 全体像

TmApp では、まず AA 組成・charge・疎水性などの単純配列特徴と CDR 長・germline 情報で median baseline（CV 3.44）から MAE ≈ 3.10 まで改善した。PLM 段階では抗体特化の **AbLang2（Heavy/Light paired）** が generic ESM 系を上回り、**SEQ_BASIC との fusion** で CV 2.78 まで大きく伸びた。Stage 2b では PCA を外した raw embedding + 強い Ridge も有効な選択肢だった。

構造特徴 **単独** では PLM に及ばなかったが、**RASA や packing を sequence / PLM incumbent に追加** すると小さな改善が得られ、構造は主予測源より補助情報として機能した。Stage 4 の salt bridge / contact / packing 等の interaction 特徴にもさらに小さな gain があった。最終的には 4 種類の base model prediction を **nested Ridge stacking（α=100）** で統合し、Round 1 pre-registered PRIMARY となった（CV 2.71）。Test では絶対 MAE は CV より約 +0.5 悪化したが、model ranking はある程度 generalize した。

### 2.2 TmApp 実験一覧

| Stage | 工夫・実験 | representative model | CV | Public | Private |
|:---:|---|---:|---:|---:|---:|
| Baseline | Dev 中央値を全抗体へ定数予測 | Median | 3.437 | 3.784 | 3.772 |
| 1 | AA 組成・charge・疎水性等の単純配列特徴 + RBF-SVR | `TmApp__SEQ_BASIC__SVROpt` | 3.101 | 3.587 | 3.352 |
| 1 | CDR 長を追加（非線形） | `TmApp__ANN_CDR_LENGTH__SVROpt` | 3.104 | — | — |
| 1 | germline family / identity を追加 | `TmApp__ANN_GERMLINE__Ridge` | 3.182 | — | — |
| 1 | 配列特徴を統合（SEQ_ALL + LGBM） | `TmApp__SEQ_ALL__LGBMOpt` | 3.106 | 3.680 | 3.485 |
| 2 | ESM-2 whole-chain embedding | `TmApp__esm2__HL__SVROpt` | 3.115 | 3.597 | 3.272 |
| 2 | ESM-1b Heavy/Light embedding | `TmApp__esm1b__HL__SVROpt` | 3.105 | 3.510 | 3.219 |
| 2 | AbLang2 Heavy/Light paired（PLM 単独） | `TmApp__ablang2__HL_paired__SVROpt` | 2.923 | 3.489 | 3.080 |
| 2 | AbLang2 paired + SEQ_BASIC fusion | `TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt` | 2.776 | 3.239 | 3.263 |
| 2b | PCA なし raw AbLang2 + 強い Ridge | `TmApp__ablang2__HL_paired__RidgeOpt_PCANone` | 2.863 | 3.647 | 3.372 |
| 3 | RASA 由来構造特徴（structure-only） | `TmApp__ESMFold__STRUCT_RASA__ElasticNetOpt` | 3.241 | 3.753 | 3.473 |
| 3 | sequence incumbent + RASA fusion | `TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt` | 2.754 | 3.346 | 3.205 |
| 3 | packing 特徴を PLM 系へ追加 | `TmApp__FUSION_OVERALL__ESMFold__STRUCT_PACKING__SVROpt` | 2.777 | 3.316 | 3.129 |
| 3 | 表面・幾何を束ねた STRUCT_ALL fusion | `TmApp__FUSION_OVERALL__ESMFold__STRUCT_ALL__SVROpt` | 2.839 | 3.451 | 3.177 |
| 4 | salt bridge / contact / packing 等 interaction | `TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt` | 2.743 | 3.332 | 3.161 |
| 4 | inverse folding 単独（ESM-IF） | `TmApp__ADV_INVFOLD__ElasticNetOpt` | 3.338 | 3.751 | 3.754 |
| 4 | TmApp 向け advanced feature 束（単独） | `TmApp__ADV_TMAPP_ALL__SVROpt` | 3.224 | 3.661 | 3.733 |
| 5 | base prediction の単純平均 blend | `TmApp__SIMPLE_blend_plm_struct` | 2.841 | 3.450 | 3.161 |
| 5 | diversity 重視 Ridge meta stack | `TmApp__META_diversity__ridge_100.0` | 2.743 | 3.346 | 3.150 |
| 5 | **nested Ridge stack（Round1 PRIMARY）** | `TmApp__META_performance__ridge_100.0` | **2.714** | **3.231** | **3.212** |

### 2.3 TmApp で特に効いた工夫

#### Stage 1

単純な sequence physicochemical feature（SEQ_BASIC）だけでも median baseline から約 10% 改善した。CDR 長（ANN_CDR_LENGTH）は SEQ_BASIC とほぼ同程度（CV 3.104 vs 3.101）、germline identity にも一定の予測 signal があり（CV 3.182）、RBF-SVR で非線形に扱うことで classical feature の性能が伸びた。ただし上位候補間の CV 差は 0.005 °C 程度と極小で、Primary 順位そのものは過信しない。

#### Stage 2

最も大きな改善は **AbLang2** だった。Heavy/Light 両鎖の antibody-specific embedding（CV 2.92）が generic ESM-2（3.12）より良く、さらに **SEQ_BASIC を連結して SVR** に入れる fusion で CV 2.78 まで改善した。TmApp では H+L 両鎖情報が重要で、Light-only は一貫して弱かった。

#### Stage 2b

PCA は常に必要ではなかった。AbLang2 paired embedding では **raw embedding + 強い Ridge 正則化**（CV 2.86）が、PCA 版と同等以上の性能を示し、Stage 3 以降の PLM-only incumbent 候補になった。

#### Stage 3

structure-only（CV ≈ 3.24）は Stage 2 fusion（2.78）に及ばなかった。一方、**RASA や packing を sequence / PLM incumbent に追加** すると CV 2.75–2.78 台で小さな改善が得られ、構造情報は主予測源より **補助情報** として有効だった。

#### Stage 4

salt bridge、contact density、packing、interface 等をまとめた **ADV_INTERACTIONS** を Stage 3 incumbent に fusion すると CV 2.74 まで改善した。inverse folding 単独（CV 3.34）は弱く、interaction 系の幾何記述の方が TmApp には適合した。

#### Stage 5

単純平均 blend（CV 2.84）は incumbent 単体より悪化したが、**異なる base model family の OOF prediction を nested Ridge stack** で統合すると CV 2.71 となり、pre-registered PRIMARY になった。Test でも secondary conservative（Private 3.16）に近いが、CV 選択ルール上 stack が最終提出モデルとなった。

### 2.4 TmApp 最終モデル

| Final model | CV | Public | Private | All Test |
|---|:---:|:---:|:---:|:---:|
| **Ridge stack α=100** (`TmApp__META_performance__ridge_100.0`) | **2.7135** | **3.2307** | **3.2116** | **3.2211** |

Base models: ADV_INTERACTIONS fusion、overall+RASA fusion、AbLang2+SEQ_BASIC fusion、ADV_TMAPP_ALL。full Dev 162 で cross-fitted OOF → meta fit → Test predict。

CV より Test 絶対 MAE は約 **+0.51** 悪化するが、これは PRIMARY 単体に限らず **モデル群全体の dataset-wide shift**（median gap ≈ +0.34）でもある。pre-registered secondary（Stage 4 conservative incumbent）の Private MAE 3.16 は PRIMARY 3.21 より僅かに良いが、CV 順位では stack が上。model ranking は Private と r≈0.72 である程度 generalize した。

---

## 3. HIC

### 3.1 全体像

HIC では classical sequence feature から始め、**RBF-SVR による非線形モデリング**で median baseline（CV 0.518）から CV 0.47 台まで改善した。PLM では **ESM-2 Heavy** が AbLang2 より強く、**SEQ_ALL fusion** で CV 0.449 まで伸びた。

構造段階では単純な total SASA より、**溶媒へ露出した疎水性・芳香族・荷電残基**を表す **SURFACE_CHEM**（CV 0.440）が最も効いた。Stage 4 では generic APBS / PROPKA electrostatics は追加 gain が小さく、**spatial hydrophobic / aromatic patch** に限定的な改善があった。最終的には **ESM-2 sequence model、SURFACE_CHEM model、advanced surface-patch model の 1/3 ずつ単純平均**（CV 0.425）が Round 1 PRIMARY となり、CV / Public / Private が非常によく一致した。

### 3.2 HIC 実験一覧

| Stage | 工夫・実験 | representative model | CV | Public | Private |
|:---:|---|---:|---:|---:|---:|
| Baseline | Dev 中央値を全抗体へ定数予測 | Median | 0.518 | 0.535 | 0.510 |
| 1 | sequence physicochemical features + SVR | `HIC__SEQ_ALL__SVROpt` | 0.477 | 0.477 | 0.477 |
| 1 | Heavy-only 配列特徴（線形） | `HIC__SEQ_HEAVY__Lasso` | 0.528 | — | — |
| 1 | Light-only 配列特徴（線形） | `HIC__SEQ_LIGHT__Lasso` | 0.595 | — | — |
| 1 | antibody-aware features を追加 | `HIC__SEQ_PLUS_ANTIBODY__SVROpt` | 0.473 | 0.485 | 0.463 |
| 2 | ESM-1b Heavy + classical fusion | `HIC__FUSION__esm1b__H__SEQ_PLUS_ANTIBODY__SVROpt` | 0.462 | 0.459 | 0.478 |
| 2 | ESM-2 Heavy（PLM 単独） | `HIC__esm2__H__SVROpt` | 0.455 | 0.424 | 0.459 |
| 2 | AbLang2 Heavy + classical fusion | `HIC__FUSION__ablang2__H__SEQ_PLUS_ANTIBODY__SVROpt` | 0.476 | 0.481 | 0.467 |
| 2 | ESM-2 Heavy + SEQ_ALL fusion | `HIC__FUSION__esm2__H__SEQ_ALL__SVROpt` | 0.449 | 0.422 | 0.490 |
| 2b | raw ESM-2 Heavy + SVR（PCA なし） | `HIC__esm2__H__SVROpt_PCANone` | 0.453 | 0.431 | 0.456 |
| 3 | total SASA（structure-only） | `HIC__ESMFold__STRUCT_SASA__ElasticNetOpt` | 0.502 | — | — |
| 3 | RASA のみ fusion | `HIC__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt` | 0.455 | 0.580 | 0.554 |
| 3 | **exposed hydrophobic/aromatic/charged SASA** | `HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt` | 0.440 | 0.483 | 0.438 |
| 3 | simple surface patch fusion | `HIC__FUSION_OVERALL__ESMFold__STRUCT_PATCH__SVROpt` | 0.452 | 0.581 | 0.554 |
| 3 | sequence incumbent + SURFACE_ALL | `HIC__FUSION_OVERALL__ESMFold__STRUCT_SURFACE_ALL__SVROpt` | 0.441 | 0.438 | 0.444 |
| 4 | PROPKA / charge features（単独） | `HIC__ADV_PROPKA__ElasticNetOpt` | 0.581 | 0.608 | 0.581 |
| 4 | APBS electrostatic potential（単独） | `HIC__ADV_ELECTROSTATICS__ElasticNetOpt` | 0.581 | 0.608 | 0.581 |
| 4 | advanced spatial hydrophobic/aromatic patch（advanced-only） | `HIC__ADV_SURFACE_PATCH__SVROpt` | 0.472 | 0.471 | 0.453 |
| 4 | Stage3 incumbent + ADV_SURFACE_PATCH fusion | `HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt` | 0.437 | 0.440 | 0.439 |
| 5 | learned NNLS / Ridge meta stack | `HIC__META_performance__nnls_norm` | 0.432 | 0.514 | 0.501 |
| 5 | diversity meta stack | `HIC__META_diversity__nnls_norm` | 0.427 | 0.520 | 0.487 |
| 5 | **3-model equal-weight blend（Round1 PRIMARY）** | `HIC__SIMPLE_blend_seq_surf_adv` | **0.425** | **0.420** | **0.424** |

### 3.3 HIC で特に効いた工夫

#### Stage 1

Heavy 側の情報が Light-only より明確に強かった（線形比較: Heavy Lasso CV 0.528 vs Light 0.595）。sequence composition / physicochemical feature に antibody-aware feature を加え、**RBF-SVR** を使うことで baseline を改善した。annotation 追加の incremental value は SEQ_ALL との差 0.003 min 程度と小さく、Shadow では SEQ_COMBINED が最良だった。

#### Stage 2

generic PLM の中では **ESM-2 Heavy**（CV 0.455）が AbLang2（0.476）より強かった。TmApp とは異なり AbLang2 が最良ではなく、**ESM-2 Heavy + SEQ_ALL fusion**（CV 0.449）が Stage 2 overall incumbent となった。

#### Stage 3

HIC で最も重要な構造上の知見。**単純な total SASA（CV 0.50）より、「どの種類の残基が表面に露出しているか」を表す SURFACE_CHEM（CV 0.440）が強かった。** 具体的には exposed hydrophobic / aromatic / positive / negative SASA と CDR / HCDR3-aware 集約である。RASA そのものを予測量として使う fusion は Test で大きく悪化（Public 0.58）し、**RASA を surface chemistry 抽出の入力に使う** 方が重要だった。

#### Stage 4

generic fixed-condition **APBS / PROPKA electrostatics**（CV ≈ 0.58）は incumbent に届かず、明確な追加 gain を示さなかった。一方、**local aromatic/hydrophobic exposure + spatial neighborhood** を使った advanced surface patch（CV 0.437）には小さいが再現した追加 gain があった。

#### Stage 5

learned NNLS stack（CV 0.432）より、**ESM-2+SEQ_ALL / SURFACE_CHEM / ADV_SURFACE_PATCH の 3 モデル equal-weight mean**（CV 0.425）の方が Test でも良かった。small-N では weight を学習するより **低自由度 ensemble** が安定した可能性がある。

### 3.4 HIC 最終モデル

| Final model | CV | Public | Private | All Test |
|---|:---:|:---:|:---:|:---:|
| **3-model equal mean** (`HIC__SIMPLE_blend_seq_surf_adv`) | **0.4252** | **0.4204** | **0.4239** | **0.4222** |

各 base model は full Dev 162 で refit し、ensemble weight は学習せず 1/3 ずつに固定。Base: `HIC__FUSION__esm2__H__SEQ_ALL__SVROpt`、`HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt`、`HIC__ADV_SURFACE_PATCH__SVROpt`。

CV / Public / Private が非常によく一致し（Primary–Private r≈0.77、median gap≈0）、overall prediction は良好に generalize した。ただし **high-HIC tail（≥10.54 min, N=13）** では Dev/Test とも系統的 underprediction（13/13、slope≈0.20）が残り、Round 2 ではこの tail regime が主要課題である。

---

## 4. TmApp と HIC を比べて分かったこと

| 観点 | TmApp | HIC |
|---|---|---|
| 強かった PLM | AbLang2（paired H+L） | ESM-2 Heavy |
| H/L の重要性 | H+L 両鎖が重要 | Heavy 優位 |
| structure-only | 弱い（CV ≈ 3.24） | SURFACE_CHEM は強い（CV 0.440） |
| 有効な構造情報 | packing / RASA を incumbent に追加 | exposed surface chemistry / spatial patch |
| advanced physics | interaction が小改善 | electrostatics 弱、spatial patch 小改善 |
| 最終統合 | learned Ridge stack（α=100） | equal-weight 3-model mean |
| CV→Test | Test で絶対 MAE 悪化（≈+0.5） | ほぼ一致（≈±0.003） |
| 主な残課題 | generalization gap / calibration | high-HIC tail shrinkage |

両 target とも structure information は有用だったが、**有効な構造情報の種類が異なった**。TmApp では internal packing / stability 寄り、HIC では solvent-exposed surface chemistry 寄りの特徴がより有効だった。

---

## 5. Round 1 まとめ

TmApp では、抗体特化 PLM である AbLang2 を中心に sequence feature と structure-derived packing / RASA 情報を重ね、最後に **prediction-level Ridge stacking** することで性能を伸ばした。HIC では ESM-2 Heavy に加え、予測構造上で **表面へ露出した疎水性・芳香族残基を明示的に記述** した SURFACE_CHEM が大きな改善につながった。

最終モデルは TmApp では learned Ridge stack、HIC では low-complexity equal-weight ensemble となった。TmApp は CV 順位の generalization はある程度保たれる一方、絶対 MAE の CV→Test gap が dataset-wide に存在する。HIC は CV と Test がよく整合するが、高 HIC 域の underprediction が残る。

詳細な model-level 相関解析は `ROUND1_MODEL_SCORE_CORRELATION_ANALYSIS_JA.md`、提出スコアと tail 診断は `ROUND1_RESULTS_JA.md` を参照。

---

**状態:** `ROUND1_COMPACT_EXPERIMENT_REPORT_FINAL_FROZEN`
