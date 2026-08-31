# 予測スコアの科学的解釈

English: [SCORE_INTERPRETATION_TECHNICAL.md](SCORE_INTERPRETATION_TECHNICAL.md)

コンペ: **Antibody Developability — TmApp & HIC**  
対象読者: シニアサイエンティスト / プロジェクトリーダー / 部門マネージャ / 科学レビュー / 監査・ガバナンス  
数値ソース: `SCORE_INTERPRETATION_STATS.json`  
位置づけ: ドキュメントのみ（採点・リリースデータは変更しない）

---

## 組織としての立場（要約段落）

本コンペでは、**local cross-validation** における **TmApp MAE おおよそ 2.8–3.0 °C** および **HIC MAE おおよそ 0.45–0.48 min** を、**強い benchmark 級の予測**として解釈すべきである。これらの水準は、定数 baseline（Train-CV 中央値 baseline）に対して意味のある配列→物性 signal を示す（それぞれ MAE 約 **21.6%** / **10.0%** 削減）。おおよそ **2.5 °C** および **0.40 min** を下回るスコアは、現行 organizer benchmark を明確に超える改善とみなせる。これらの目安は **コンペ相対（competition-relative）** であり、普遍的な developability 閾値ではない。関連する DSF / HIC アッセイの公表再現性データは、分析測定が現行 ML 誤差より実質的にタイトに再現し得ることを示しており、現状モデルは **早期スクリーニング / 優先順位付けの補助** として見るべきであり、実験的キャラクタリゼーションの置き換えではない。

---

## 1. 目的

本文書は、繰り返し生じるレビュー上の問いに答える。

> *この*コンペで目指すべき予測スコアは何か。そしてそのスコアに、安全に付与できる科学的意味は何か。

次の2つのものさしを、単一の「有用 / 臨床 / 製造」スケールに潰してはならない。

1. **コンペ相対パフォーマンス** — 本 Shehata 由来データセット上での baseline・organizer benchmark に対する改善。  
2. **科学 / アッセイ相対解釈** — 予測誤差が、関連アッセイで報告される technical variation と qualitatively どう比較できるか。

本文中の機械可読数値は `SCORE_INTERPRETATION_STATS.json` に凍結されている。

---

## 2. エグゼクティブ結論

1. **強いコンペ相対結果**は、**local CV** 上でおおよそ **TmApp ~2.8–3.0 °C** および **HIC ~0.47 min** である。  
2. これらの誤差は、関連する DSF / HIC 文脈で報告される technical variation より **数倍大きい** ままである。  
3. したがって **意味のある配列→物性予測** は示すが、モデルを DSF / HIC の代替と記述する根拠には **ならない**。  
4. **~2.5 °C**（TmApp）または **~0.40 min**（HIC）を下回るスコアは、現行 organizer benchmark を超える明確な前進である。  
5. **~1 °C**（TmApp）または **~0.1 min**（HIC）に近づくスコアは、関連実験で報告される再現性の *スケール* に近づくため科学的に注目に値する。ただしアッセイ等価性にはなお **直接的な実験検証** が必要である。  
6. 参加者向けガイダンスは **local CV** の目安を主とすべきである。Public N=81 はノイズが大きく、Public MAE 帯を local-CV 帯と同一視してはならない。

---

## 3. コンペターゲットと評価指標

| Track | Target | 測定内容 | 単位 | 主評価指標 |
|---|---|---|---|---|
| 1 | `TmApp` | **Fab** の **DSF** による apparent melting temperature | °C | MAE（小さいほど良い） |
| 2 | `HIC` | **IgG** の hydrophobic interaction chromatography retention time | min | MAE（小さいほど良い） |

母集団: **N=324**（両ラベル）。Dev **162**、Test **162**、Public/Private **81/81**、split `GEN_0001_B_20271100`。  
独立したリーダーボードが2つ。**合算スコアなし**。タイ政策: Private MAE が完全一致 → 同率。

原研究: Shehata et al., *Cell Reports* (2019), DOI `10.1016/j.celrep.2019.08.056`（CC BY 4.0 VoR 由来パッケージ）。

---

## 4. 実証的なコンペベンチマーク

### 4.1 Local CV（主たる目安のものさし）

凍結 organizer Train-CV / 集約 OOF 成果物（B5 calibration + GEN_0001 再スコア CV）より:

| Target | 定数 / 中央値 baseline MAE | 強い organizer 水準 | およその MAE skill |
|---|---:|---:|---:|
| TmApp | ≈ **3.543 °C** | ≈ **2.78–2.95 °C**（最良 nested ≈ **2.778 °C**） | 誤差削減 ≈ **21.6%** |
| HIC | ≈ **0.518 min** | ≈ **0.45–0.48 min**（CV最良 ≈ **0.467 min**） | 誤差削減 ≈ **10.0%** |

**MAE skill** = `1 − MAE_model / MAE_baseline`。  
skill 0 = 定数 baseline から改善なし。0.10 = MAE が 10% 低い。0.20 = MAE が 20% 低い。

skill が測るのは **コンペ相対の予測改善** である。臨床成功、developability 確率、アッセイ置換、製造成功を測るものでは **ない**。

### 4.2 Final-split の例（転送診断のみ）

本番 split GEN_0001 上（organizer モデル再スコア。参加者目標ではない）:

| Target | 設定 | Public MAE | Private MAE |
|---|---|---:|---:|
| TmApp | 中央値 baseline | ≈ 3.784 | ≈ 3.772 |
| TmApp | 強い凍結モデル | ≈ 3.5–3.6 | ≈ 3.2–3.3 |
| HIC | 中央値 baseline | ≈ 0.535 | ≈ 0.510 |
| HIC | CV最良凍結モデル | ≈ 0.488 | ≈ 0.457 |

local-CV の目安と Public リーダーボード閾値を **混ぜてはならない**。TmApp の local CV ~2.8–3.0 は Public では mid-3s 付近に見えることがある。比較は同種同士で行う。

---

## 5. 実際のターゲット分布統計（N=324）

凍結コンペラベル（`final_population.csv`）から直接計算。baseline MAE からの推定ではない。

| 統計量 | TmApp (°C) | HIC (min) |
|---|---:|---:|
| Mean | 69.89 | 9.38 |
| SD | **4.69** | **0.83** |
| Median | 70.0 | 9.10 |
| IQR | **6.0** | **0.70** |
| MAD（中央値まわり） | **3.0** | **0.27** |
| P10–P90 | 64.5–75.5 | 8.75–10.40 |
| Min–Max | 52.5–83.5 | 8.47–13.86 |

HIC は **右歪み** で **疎な high tail** を持つ。SD だけでは、稀な HIGH-band 抗体のスクリーニング重要性を過小評価しうる。

記述的正規化（普遍的解釈ではない）:

| Landmark | TmApp MAE/SD | TmApp MAE/IQR | HIC MAE/SD | HIC MAE/IQR |
|---|---:|---:|---:|---:|
| Local-CV baseline | ≈ 0.76 | ≈ 0.59 | ≈ 0.62 | ≈ 0.74 |
| Local-CV best | ≈ 0.59 | ≈ 0.46 | ≈ 0.56 | ≈ 0.66 |

---

## 6. コンペ相対パフォーマンス尺度

*本*データセットと現行 organizer benchmark に対する **local CV** のおよその目安（「~ / おおよそ / だいたい」を用いる）:

### TmApp MAE (°C)

| 帯 | およその MAE |
|---|---|
| baseline 級 | ~3.5 以上（同等またはそれより悪い） |
| 明確な予測 signal | ~3.3 未満 |
| 強い organizer-benchmark 級 | **~2.8–3.0** |
| 現行 benchmark に対して例外的 | ~2.5 未満 |
| 非常に高い予測精度 | ~2.0 未満 |

### HIC MAE (min)

| 帯 | およその MAE |
|---|---|
| baseline 級 | ~0.52 |
| 明確な予測 signal | ~0.50 未満 |
| 強い organizer-benchmark 級 | **~0.45–0.48** |
| 現行 benchmark に対して例外的 | ~0.40 未満 |
| 非常に高い予測精度 | ~0.30 未満 |

これらは **経験的なコンペ landmarks** であり、自然な科学的カットオフでも産業上の受入基準でもない。

---

## 7. 科学的エビデンス階層

| Grade | 意味 |
|---|---|
| **A** | Shehata 直接、または極めて近いプロトコルで対応が明確 |
| **B** | 近縁の抗体アッセイ（同種の測定） |
| **C** | 一般的な方法論エビデンス |
| **Inference** | organizer の合成 / コンペ相対解釈 |

**重要な欠落:** Shehata 固有の technical repeatability は、利用可能なパッケージ証拠中の反復統計からは **確立されていない**。関連アッセイの数値を「Shehata noise floor」として書き換えてはならない。

---

## 8. TmApp アッセイ解釈

TmApp は、研究条件下で **Fab** 断片に対する **differential scanning fluorimetry（DSF）** により測られた **見かけの（apparent）** 熱転移 / melting temperature である。

- TmApp が高いほど、一般に **そのアッセイ条件下** での熱 / 構造安定性が高いことを示す。  
- TmApp は **アッセイ依存** であり、普遍的な熱力学定数ではない。  
- **直接の凝集測定ではない**。  
- それ単独で developability や製造成功を決定しない。

---

## 9. TmApp 再現性エビデンス

文書化の推奨文言:

> 関連する抗体 / 治療用タンパクの DSF 研究では、プロトコルと反復設計に応じて、technical variation がおおよそ **サブ °C から約 1 °C** のスケールで報告される。ただし **Shehata アッセイ固有の再現性は、利用可能な証拠からは直接確立されていない。**

例（Grade B/C — Shehata の floor ではない）:

- 治療用タンパクの **nanoDSF** 文献では、一部の comparability 設定で Fab ドメイン再現性が **~0.2 °C** オーダーと報告される（関連方法論）。  
- より広い DSF プロトコル文献では、複数日 / 装置 / プレート位置効果が一部設計で **~0.5–1 °C** に達し得るとされる。

**書いてはならない:** 「Shehata assay noise floor = 0.2–0.5 °C」。

---

## 10. HIC アッセイ解釈

HIC retention time は、特定プロトコル下でのクロマトグラフィ固定相との **実効的な疎水性相互作用** を反映する。

保持が長いことは、より強い実効疎水性や、self-association / 非特異的相互作用に関連する developability リスクの議論と結び付けられることがある。しかし:

- HIC は **凝集アッセイではない**。  
- HIC が高いからといって、抗体が必ず凝集するわけでは **ない**。  
- 絶対的な保持時間は **プロトコル依存** である。

---

## 11. HIC 再現性エビデンス

文書化上、最も防御可能な比較:

- Jain et al., *Bioinformatics* (2017), DOI `10.1093/bioinformatics/btx519`  
- Adimab HIC セットアップで定期測定された reference IgG1（adalimumab 可変領域コントロール）: **127** 回測定で **8.6 ± 0.12 min**。

**~0.12 min** は **有用な近縁プロトコル変動スケール**（Grade A/B）として扱う。普遍的な HIC noise floor でも、直接証明された Shehata 反復統計でも **ない**。

高度に最適化された固定系 HIC の注入間精度は、他ラボ文脈ではさらに小さくなり得る。再現性は系・カラム・プロトコルに強く依存する。この対比は、単一数値を「唯一の」noise floor と呼ぶことへの警戒を強める。

---

## 12. なぜアッセイ再現性 ≠ ML 誤差フロアなのか

分析再現性は、制御条件下で **同一** の実験手順が測定をどれだけタイトに再現するかを記述する。

ML MAE は、**異なる抗体** にわたる **配列→物性予測器** の平均絶対誤差を記述する。

たとえ ML MAE が関連アッセイの再現性スケールに近づいても:

- 自動的に Shehata アッセイ等価にはならない。  
- 同じデルタでの正しい pairwise 順位付けを保証しない。  
- 前向き検証なしに実験キャラクタリゼーション置換を正当化しない。

---

## 13. Baseline に対する MAE skill

凍結 local-CV 値を用いる:

| Target | Baseline | Best | Skill | 誤差削減 |
|---|---:|---:|---:|---:|
| TmApp | 3.543 °C | 2.778 °C | ≈ 0.216 | ≈ **21.6%** |
| HIC | 0.518 min | 0.467 min | ≈ 0.100 | ≈ **10.0%** |

解釈: 強い organizer モデルは本物の signal を抽出しており、現行の特徴量 / モデル族では相対的な余白（headroom）は TmApp の方が HIC より大きい。

---

## 14. MAE と順位 / 判別

- **MAE**: アッセイ値そのものの忠実度（コンペ主指標）。  
- **Spearman**: 順位一致性（診断用）。  

コンペ指標は **MAE** のまま。変更しない。

スクリーニング有用性の内部科学議論では、有用な診断として次がある:

- high-tail recall / top-*k* enrichment（特に HIC HIGH band）、  
- 選択したデルタでの pairwise ranking accuracy、  
- calibration / 残差構造。

### MAE は「分解能」ではない

`TmApp MAE = 2 °C` は、2 °C 差の全ペアを確実に分離できることを **意味しない**。  
`HIC MAE = 0.3 min` は、0.3 min 差の全ペアを確実に順位付けできることを **意味しない**。

ペア判別は、誤差分布・バイアス・calibration・予測誤差相関・真の分離幅に依存する。

---

## 15. コンペ相対スコア帯

§6 を参照。参加者ガイダンスは **local CV** を引用すべきである。Public リーダーボード比較は Public baseline / Public ピアを用いる（N=81。順位は揺れうる）。

---

## 16. スコア landmarks の科学的解釈

### TmApp

| およその MAE | コンペ上の読み | 科学的注意 |
|---|---|---|
| ~3.5–3.8 °C | baseline 級 | 定数予測からの改善は小さい / 限定的 |
| ~3.0 °C | 強いコンペ水準 | 意味のある配列→TmApp signal。関連 DSF の technical variation よりなお数倍大きい。**広い**安定性変動は捉えうる — 「3 °C 差がすべて分解される」ではない。アッセイ置換でもない |
| <2.5 °C | organizer benchmark に対して例外的 | 主要な抗体間ばらつきより誤差が実質的に小さい — 自動的に製造 / 臨床有用ではない |
| <2.0 °C | 本データでは非常に高い予測精度 | 関連 DSF 再現性 *スケール* への接近として科学的に興味深い — なおアッセイ置換には不十分 |
| ~1.0 °C | 一部関連 DSF 再現性のオーダーに接近 | **検証済み DSF 置換ではない**。**Shehata noise floor ではない**。DSF と実験的に互換でもない |

### HIC

| およその MAE | コンペ上の読み | 科学的注意 |
|---|---|---|
| ~0.52 min | baseline 級 | 定数水準 |
| ~0.47 min | 強いコンペ水準 | 本物の配列→HIC signal（CV 誤差削減 ~10%）。近縁プロトコル reference 変動 ~0.12 min よりなお数倍大きい — 無情報ではないが、アッセイ級精度でもない |
| <0.40 min | organizer benchmark に対して例外的 | 現行実証 organizer モデルを超える明確な前進 |
| <0.30 min | 本データでは非常に高い予測精度 | 近縁プロトコル分析変動と同程度の広いオーダーに近づく — なお 0.3 min の pairwise 分解能ではない |
| ~0.1 min | 近縁 reference-control 変動のオーダー | **証明済みアッセイ置換ではない**。**普遍的 noise floor ではない**。最終 developability 判断には不十分 |

---

## 17. 支持される主張

- Local-CV MAE ~2.8–3.0 °C（TmApp）および ~0.45–0.48 min（HIC）は、**本コンペの baseline と organizer benchmark に対して強い**。  
- これらの水準は、配列からの **再現可能な予測 signal** を示す。  
- 関連 DSF / HIC 文献は、現行 ML 誤差より **実質的にタイトな** 分析変動を報告する。  
- Public N=81 は小さく、リーダーボード順位は揺れうる。  
- HIC ラベル分布は歪みと疎な high tail を持ち、スクリーニング有用性と関連する。

---

## 18. 支持されない主張

- これらの MAE 帯が **産業上の受入閾値** である。  
- 強い benchmark 水準のモデルが DSF や HIC を **置き換える**。  
- モデルが developability・製造準備・臨床有用性・「安全な抗体」を認定する。  
- 引用した関連アッセイ SD が **Shehata noise floor** である。  
- MAE が同じ数値デルタでの pairwise 分解能に等しい。  
- Public MAE 帯が local-CV 帯に等しい。

---

## 19. エビデンス台帳

| 主張 | 出典 | アッセイ / 試料 | 数値観察 | Grade | Shehata への関連 | 解釈 |
|---|---|---|---|---|---|---|
| TmApp は Fab DSF apparent Tm | Shehata et al. 2019; packaging PROVENANCE | Fab DSF | mmc2 の TmApp (°C) | A | 直接 | アッセイ定義 |
| HIC は IgG retention time | Shehata et al. 2019; PROVENANCE | IgG HIC | mmc2 の分 | A | 直接 | アッセイ定義 |
| Local-CV TmApp baseline ≈ 3.543 °C | B5 `calibration_analysis.md` CONST_MEDIAN | Train OOF | MAE 3.54321 | Inference | コンペ母集団 | baseline landmark |
| Local-CV TmApp best ≈ 2.778 °C | B5 calibration / GEN_0001 3×3 | Train OOF | MAE 2.778 | Inference | コンペ母集団 | 強い benchmark |
| Local-CV HIC baseline ≈ 0.518 min | B5 calibration CONST_MEDIAN | Train OOF | MAE 0.518465 | Inference | コンペ母集団 | baseline landmark |
| Local-CV HIC best ≈ 0.467 min | GEN_0001 3×3 CV-best | Train CV | MAE 0.4667 | Inference | コンペ母集団 | 強い benchmark |
| TmApp N=324 SD ≈ 4.69 °C | `final_population.csv` | コンペラベル | SD 4.691 | A | 直接ラベル | 生物学的ばらつき |
| HIC N=324 SD ≈ 0.83 min; 歪みあり | `final_population.csv` | コンペラベル | SD 0.832; skew > 2 | A | 直接ラベル | 生物学的ばらつき / 尾部 |
| 関連 nanoDSF Fab 再現性 ~0.2 °C | 治療用タンパク nanoDSF comparability 文献 | nanoDSF Fab/domain Tm | 再現性 ~0.2 °C | B | 関連方法。Shehata プロトコルではない | 定性的なアッセイタイトネス尺度 |
| 複数日 / プロトコル DSF 変動 ~0.5–1 °C | 一般 DSF 方法論文献 | DSF プロトコル | クラス効果で最大 ~1 °C | C | 方法クラス | floor 過大主張への警戒 |
| 近縁プロトコル HIC reference 8.6 ± 0.12 min（n=127） | Jain et al. 2017 Bioinformatics | Adimab HIC reference IgG1 | 8.6 ± 0.12 min | A/B | 近縁の Adimab HIC 系 | 最も近い HIC 変動スケール |
| 強い ≠ アッセイ置換 | organizer 合成 | — | — | Inference | 方針 | 必須の注意 |

---

## 20. 主要参考文献

1. Shehata L, et al. (2019). Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability. *Cell Reports* 28:3300–3308.e4. DOI: [10.1016/j.celrep.2019.08.056](https://doi.org/10.1016/j.celrep.2019.08.056).  
2. Jain T, et al. (2017). Prediction of delayed retention of antibodies in hydrophobic interaction chromatography from sequence using machine learning. *Bioinformatics* 33:3758–3766. DOI: [10.1093/bioinformatics/btx519](https://doi.org/10.1093/bioinformatics/btx519).（Reference-control HIC 8.6 ± 0.12 min, n=127.）  
3. Jain T, et al. (2017). Biophysical properties of the clinical-stage antibody landscape. *PNAS* 114:944–949. DOI: [10.1073/pnas.1616408114](https://doi.org/10.1073/pnas.1616408114).（Developability アッセイ景観の文脈。）  
4. 治療用タンパク **nanoDSF** の comparability / 再現性文献で、Fab ドメイン再現性が ~0.2 °C オーダーと報告されるもの（Grade B。プロトコルは Shehata DSF と異なる）。  
5. プレート内 / 複数日 / 装置寄与がサブ °C〜約 1 °C に達し得ることを記す一般 DSF プロトコル文献（Grade C）。  
6. Organizer 凍結成果物: `gate_b5_ceiling/reports/calibration_analysis.md`, `gate_b5_ceiling/reports/GATE_B5_CEILING_FINAL.md`, `gate_b7_3_principled_split/reports/07_FINAL_GEN0001_3X3_BENCHMARK.md`, `competition/organizer/SCORE_INTERPRETATION_STATS.json`.

---

## 付録 — レビューア Q&A（品質テスト）

**Q1. なぜ TmApp 3.0 °C が「強い」のか？**  
organizer の local-CV benchmark（~2.8–3.0）近傍であり、中央値 baseline ~3.54 °C を十分下回る（MAE 削減 ~15–20%+）からである。すなわち明確な **コンペ相対** signal であり、3 °C が産業受入基準だからではない。

**Q2. なぜ HIC 0.47 min が「強い」のか？**  
同じ論理: organizer local-CV 最良（~0.467）近傍で、baseline ~0.518 より約 10% 良い。

**Q3. これらは産業上の受入閾値か？**  
**いいえ。** コンペ相対 landmarks のみ。

**Q4. モデルはアッセイと同精度か？**  
**いいえ。** 関連アッセイ再現性は現行 ML MAE より実質的にタイトである。

**Q5. なぜ 0.1 min を HIC noise floor と呼べないのか？**  
~0.1–0.12 min は関連 Adimab HIC の **近縁 reference-control 変動スケール** であり、証明された Shehata 反復 floor でも普遍値でもない。

**Q6. MAE 0.3 は、0.3 min 離れた2抗体を区別できる意味か？**  
**いいえ。** MAE は平均絶対誤差であり、pairwise 分解能ではない。

**Q7. 実験再現性比較を支える証拠は何か？**  
関連 DSF 文献（サブ °C〜約 1 °C）と Jain 2017 HIC reference-control（±0.12 min）。エビデンス台帳参照。Shehata 固有の反復はパッケージ証拠に無い。

**Q8. 直接測定と organizer 推論の境界は？**  
直接: Shehata アッセイ定義と N=324 ラベル分布。推論: スコア帯、skill、organizer CV 相対の「強い / 例外的」という言語。
