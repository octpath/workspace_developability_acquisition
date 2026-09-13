# TmApp 技術レポート

**VH/VL 配列 → TmApp 予測**  
表現 × H/L トポロジー × アノテーションの要因計画

**対象範囲.** 本レポートは **TmApp 予測のみ**を対象とする。予測ターゲットは VH/VL 配列からの TmApp である。ここで得たトポロジーやアノテーションに関する結論を、HIC・粘度・凝集・抗体 developability 全般へ自動的に外挿してはならない。それらの評価は別途必要である。

**用語.** 下流の H/L 設計は **SEP**（Separate H/L）、**JOINT**（Full Joint）、**REG-SEP**（Joint Residues, Separate REGs）、**XREG**（Cross-REG Read）、**FUSE**（REG Fusion）と呼ぶ。旧ラベル A/B1/B2/C/D は再現性メモと対応表のみに用いる。**REG** とは、下流 Transformer が処理した残基から情報を集約する **学習された鎖レベル・トークン**である。以降は REG / REG_H / REG_L を用いる。

図表は `developability_drilldown/technical_report/tmapp_factorial/` にある。数値は凍結済み 200 セル内部行列（`tmapp_factorial_human_master.csv` および bootstrap / DiD 表）に基づく。MAE は小さいほど良い。負の ΔMAE は、指定した基準に対する改善を意味する。

---

## 1. 導入

### 1.1 抗体構造と VH/VL 入力

抗体は重鎖と軽鎖のペアからなる。Fab アームでは、重鎖・軽鎖の可変領域（VH と VL）が対になった可変部を形成する。抗体の差異がすべて VH/VL に閉じるわけではないが、**本予測課題の入力は VH/VL 配列である**。各鎖の配列特性が重要であることに加え、二鎖の関係も VH/VL 条件付き回帰では重要になり得る。

### 1.2 なぜ Heavy と Light を明示的に扱うか

VH/VL → TmApp 予測器は、少なくとも次の設計選択を要する。

1. 残基レベルで Heavy / Light をどう表現するか  
2. 下流で二鎖間の情報交換を行うか、どこで行うか  
3. 抗体特異的な位置・領域アノテーションを明示的に与えるか  

これらが、共有学習プロトコル下での Transformer H/L トポロジー比較を動機づける。

### 1.3 Transformer の H/L 設計空間

凍結した下流トポロジーは次の 5 つである（Figure 1）。

| ID | 名称 | 概念的役割 |
| --- | --- | --- |
| **SEP** | Separate H/L | 最終マージ前に明示的な下流 H/L 通信なし |
| **JOINT** | Full Joint | H 残基・L 残基・REG_H・REG_L の無制限な共同処理 |
| **REG-SEP** | Joint Residues, Separate REGs | 生物学的残基は共同通信、REG_H / REG_L は鎖固有 |
| **XREG** | Cross-REG Read | 鎖ごと符号化の後、REG_H が Light 残基を、REG_L が Heavy 残基を読む |
| **FUSE** | REG Fusion | 鎖ごと処理の後、凍結 D3 モジュールで REG_H ↔ REG_L が相互作用 |

### 1.4 残基表現が重要である理由

残基入力は、Scratch の学習 AA 埋め込み、一般タンパク質 PLM、抗体志向 PLM、さらに同一チェックポイント内でのペア／鎖別推論コンテキストから得られうる。したがって、下流で有用な H/L 帰納バイアスは、残基表現に既に何が含まれているかに依存し得る。いずれかの PLM が「構造情報を含む」と事前に仮定しない。

### 1.5 明示的抗体アノテーション

すべての条件で配列位置埋め込みと鎖 ID 埋め込みを用いる。その上で BASE を基準に次を比較する。

- **BASE** — IMGT 位置なし、CDR/FR 領域なし  
- **IMGT**（IMGT Position）— IMGT 位置埋め込みあり  
- **REGION**（CDR/FR Region）— 領域埋め込みあり  
- **FULL**（IMGT + Region）— 両方  

問いは、「事前学習残基表現を使うときでも、明示的な抗体特異アノテーションはなお役立つか」である。

### 1.6 主たる実験問い

凍結した下流プラットフォーム上で、要因計画

\[
\text{Representation} \times \text{H/L topology} \times \text{Annotation}
\]

が **TmApp** 予測においてどのように相互作用するかを検証する。

---

## 2. 方法（要約）

### 2.1 課題とデータ

**予測課題.** ペアとなった **VH/VL** アミノ酸配列（`heavy`, `light`）から、連続値 **TmApp** を予測する。

**一次文献.** ラベルと配列は Shehata et al. (2019), *Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability*（*Cell Reports*；参考文献参照）に由来する。リポジトリのアッセイ定義（`gate_b1/reports/assay_definitions.md`）では、**TmApp** を著者報告の **見かけの熱転移／立体構造安定性**指標とし、単位は **°C**（補足表列 **TmApp (°C)**；内部列 `tm_app_C`）と定義する。TmApp が高いほど見かけの熱安定性が高い。この凍結アッセイ記述は、「Fab 断片 + DSF」といった操作手順文字列を別途記録していない。したがって本レポートは、当該メタデータにない測定詳細を追加せず、リポジトリの文言に従う。

**コホート規模.** 本 drilldown／コンペティション用コホートは、TmApp と HIC の両方を持つ抗体 **ちょうど 324 件**である（Dev 162 + Test 162 の一意 ID；残基アセットも `n_ids = 324`）。Shehata 補足における TmApp 完全例数は同アッセイ定義で **N = 346** と大きく、本モデル表は凍結された **324 ID 交差**であり、TmApp 完全補足全体ではない。

**評価.** 内部評価は、これらの VH/VL 表に対する既存の TmApp Primary / Shadow OOF プロトコル（凍結 drilldown と同じ fold）を用いる。報告指標は Primary MAE、Shadow MAE、mean(P,S)、worst(P,S)、|P−S|（TmApp スケール上の MAE、単位 °C）である。

### 2.2 表現（10）

| 表示名 | 役割 | Context |
| --- | --- | --- |
| Scratch | 学習 AA 埋め込み | LEARNED_AA |
| AbLingua | 抗体 PLM | SEPARATE_CHAIN |
| AbLang1 | 抗体 PLM | SEPARATE_CHAIN |
| AbLang2（鎖別推論） | 抗体 PLM | SEPARATE_CHAIN |
| AbLang2（H/Lペア推論） | 抗体 PLM | PAIRED_NATIVE |
| ESM-1b / ESM-2 / ESM-C 600M | 一般タンパク質 PLM | SEPARATE_CHAIN |
| CurrAb（鎖別推論） | 抗体 PLM | SEPARATE_CHAIN |
| CurrAb（H/Lペア推論） | 抗体 PLM | PAIRED_NATIVE |

AbLang2 の表示名は歴史的 allocation 名ではなく `representation_context` に従う（`ablang2_paired` は鎖別推論、`ablang2_unpaired` は同一チェックポイントでの H/L ペア推論）。CurrAb のペア／鎖別比較は同一 revision を用いる。定義の詳細は Table 1。

### 2.3 下流プラットフォーム（凍結）

新規セルはすべて `DL_FOLDLOCAL_COSINE_V3`、seed 101、`d_model=128`、MEAN merge、VAL のみでの fold 局所 LR 選択、full-Dev 再学習なし。行列実行中にトポロジーやアノテーションを調整しない。preregister 済みの歴史的 FULL セルは再利用した。プロトコルと登録の詳細は要因計画の preregistration および internal freeze にある。

### 2.4 対比の定義

- トポロジー効果: \(\Delta = \mathrm{MAE}(\text{topology}) - \mathrm{MAE}(\mathrm{SEP})\)（負 = SEP より改善）  
- アノテーション効果: \(\Delta = \mathrm{MAE}(\text{annotation}) - \mathrm{MAE}(\mathrm{BASE})\)（負 = BASE より改善）  
- マッチしたコンテキスト: \(\Delta = \mathrm{MAE}(\text{H/Lペア推論}) - \mathrm{MAE}(\text{鎖別推論})\)（負 = ペア推論が良い）  

選択した対比には OOF 予測の paired bootstrap を用いる。多重比較の DiD は、Primary と Shadow が一致しない限り探索的とする。

---

## 3. 結果

### 3.1 全体像

200 セルのうち、内部で先頭に来る構成は AbLang2 系である（Figure 2；Table 5）。mean(P,S) 最良は **AbLang2（鎖別推論） / XREG / REGION**（EXP-T205；mean≈2.994；P≈3.005、S≈2.984）。対応する AbLang2（H/Lペア推論）の最良も **XREG / REGION**（EXP-T225；mean≈3.002）である。絶対性能の 4×5 ヒートマップは表現ごとに大きく異なり（Figure 3）、全入力で共有される単一の「勝ちパターン」はない。

表現ごとの最良セル（Table 4）:

| 表現 | 最良トポロジー | 最良アノテーション | mean(P,S) |
| --- | --- | ---: |
| Scratch | REG-SEP | FULL | 3.258 |
| AbLingua | FUSE | IMGT | 3.245 |
| AbLang1 | JOINT | BASE | 3.298 |
| AbLang2（鎖別推論） | XREG | REGION | 2.994 |
| AbLang2（H/Lペア推論） | XREG | REGION | 3.002 |
| ESM-1b | XREG | BASE | 3.208 |
| ESM-2 | FUSE | FULL | 3.332 |
| ESM-C 600M | JOINT | BASE | 3.123 |
| CurrAb（鎖別推論） | XREG | REGION | 3.248 |
| CurrAb（H/Lペア推論） | XREG | FULL | 3.250 |

わずかな MAE 差は、不確かさの評価なしに決定的とはみなさない（§3.6）。

### 3.2 Finding 1 — H/L トポロジー要求は表現依存である

**明示的な下流 H/L 通信の予測上の価値は、残基表現に強く依存した。**

最良ラベルだけでも一致しない（Scratch → REG-SEP；AbLang2 → XREG；ESM-C 600M → JOINT；ESM-2 → FUSE；ESM-1b → XREG）。さらに重要には、SEP に対するトポロジー ΔMAE（Figure 4）の符号パターンが異なる。

- **Scratch, FULL:** JOINT −0.185、REG-SEP −0.224、XREG −0.131、FUSE −0.043 — SEP に対し相互作用トポロジーが大きく改善  
- **AbLang2（鎖別推論）, FULL:** JOINT −0.103、XREG −0.104；REG-SEP と FUSE は改善しない  
- **ESM-2, FULL:** 4 相互作用トポロジーすべてが SEP より改善し、FUSE −0.194 は大きい側  
- **ESM-C 600M, FULL:** 相互作用 Δ はすべて **正**（FULL 下では SEP が有利）；当該表現の最良セルは別セルの JOINT+BASE  

アノテーション平均の SEP 相対 Δ でも系統は分かれる（例: ESM-1b/ESM-2 は XREG が有利寄り、ESM-C は相互作用 Δ が非負寄り、AbLang2 コンテキストは平均的に XREG 有利）。頑健性を強調するときは、単独セルより Primary/Shadow 一致と bootstrap のトポロジー利得表を優先する。

### 3.3 Finding 2 — 明示的抗体アノテーションも文脈依存である

**抗体特異的な位置・領域アノテーションは、万能の改善ではなく、表現およびトポロジーに依存する帰納バイアスとして働いた。**

Figure 5 は IMGT / REGION / FULL の BASE 相対 ΔMAE が、行（表現）と列（トポロジー）の両方で符号を変えることを示す。REGION/FULL は **普遍的改善ではない**。

**XREG** 下の鋭い例: REGION 対 BASE は AbLang2（鎖別推論）で mean(P,S) が約 −0.15、AbLang2（H/Lペア推論）で約 −0.32 改善する一方、同じトポロジーでは Scratch（+0.12）、ESM-1b（+0.17）、ESM-C（+0.16）を悪化させる。強い一般タンパク質セルのいくつかは **BASE** を用いる（ESM-1b の XREG+BASE；ESM-C の JOINT+BASE）。AbLang1 の最良も JOINT+BASE である。

弱い IMGT や FULL 効果を「PLM がすでに IMGT を知っている」証拠とは解釈しない。本アブレーションは下流モデルへ渡す情報に関するものである。

### 3.4 Finding 3 — PLM のペア推論と下流 H/L 相互作用は等価ではない

**PLM 推論時の H/L コンテキストを変えると下流の誤差面は変化したが、下流 H/L 相互作用との関係は PLM 間で異なった。**

同一チェックポイントの PAIR−SEPARATE ヒートマップ（Figure 6；同一カラースケール）では次のとおりである。

- **CurrAb:** 平均 PAIR−SEPARATE ≈ −0.07（平均では H/L ペア推論が良い）。最良の鎖別／ペアセルの絶対 MAE は近い（Table 4: 3.248 vs 3.250）。  
- **AbLang2:** 平均 PAIR−SEPARATE ≈ +0.04（本行列では平均的に鎖別推論が良い）。アノテーション×トポロジー面はより混在するが、両コンテキストともピークは **XREG+REGION**。  

したがって、平均的なコンテキスト優位と最良セルの接近は両立し得る。pairing × topology の交互作用（three-way 表）は探索的であり、下流 XREG（や他トポロジー）が PLM 内部ペア文脈の字義どおりの代替である、という主張を支持しない。

### 3.5 科学的対照としての Scratch

Scratch は弱いベースラインにすぎない存在ではない。事前学習残基がないとき、予測器のトポロジー／アノテーション依存は大きい。FULL 下で SEP から REG-SEP へ移ると mean(P,S) は約 3.483 から約 3.258 へ改善する（EXP-T091 → EXP-T096）。Primary と Shadow はともに約 3.25–3.26 で |P−S| は非常に小さい。

適切な REG-SEP+FULL 帰納バイアスを与えると、Scratch はトポロジー／アノテーションが合っていない複数の PLM セルと競争可能になる。好ましい読みは次である。

> 事前学習残基表現は、下流帰納バイアスの必要性を単に消したのではなく、どの下流バイアスが有用かを変えた。

### 3.6 頑健性と不確かさ

Figure 7 は全 200 セルの Primary 対 Shadow を示す。多くは対角線付近だが、|P−S| が大きい少数（mean が強い ESM-C の JOINT+BASE や ESM-2 の FUSE+FULL を含む）もある。したがって worst(P,S) 順位は mean のみの順位と異なり得る（Table 5b）。マッチした paired-minus-separate 対比の bootstrap CI はしばしば 0 を含み、Figure 6 の星印は記述的マーカーであって二項の有意／非有意図ではない。外部 Public/Private は internal freeze 後のみ算出し診断用である（内部結論の書き換えや再設計の契機にしてはならない）。

### 3.7 二次的観察（記述的）

- **ESM-C 600M** は JOINT+BASE（mean≈3.123）で強くなり得る一方、狭い XREG+FULL スモーク読み（≈3.345）では弱い — 単一スモークセルは完全要因計画の代替にならない。  
- 本 TmApp 行列では **ESM-1b** の最良 mean（3.208）が **ESM-2**（3.332）を上回り、単純な世代序列はない。  
- 抗体志向モデルも単調階層をなさない。AbLang2 は先頭だが、AbLang1／AbLingua／CurrAb の最良がすべての一般タンパク質構成を一様に上回るわけではない。

---

## 4. 限界

- エンドポイントは **TmApp のみ**であり、HIC や developability 全般への主張はない。  
- 凍結プラットフォームと preregister 済み設計空間であり、あらゆるアーキテクチャ上の大域最適性は主張しない。  
- OOF Primary/Shadow は内部評価；外部診断は事後的・探索的である。  
- 多重性: 表現×トポロジー×アノテーションの対比は多く、単独の CI 除外は決定的でない。  
- AbLang2 の人間向け表示は、歴史的 allocation 名が推論コンテキストと逆転しているため `representation_context` に従う必要がある。  
- 予測的帰納バイアスの証拠は、PLM 内部機構・REG の幾何・物理的 H/L 界面に関する機序証拠ではない。

---

## 5. 結論

共有下流プロトコル下の VH/VL → **TmApp** について:

1. **トポロジー:** 明示的下流 H/L 通信は万能ではなく、その価値は残基表現に強く依存する。  
2. **アノテーション:** IMGT / CDR-FR は万能の改善ではなく文脈依存の帰納バイアスである。特に XREG 下の AbLang2 における REGION 利得と、BASE が有利な ESM 構成の対比が明確である。  
3. **ペア推論:** マッチしたペア対鎖別の PLM コンテキストは誤差面を変えるが、AbLang2 と CurrAb のパターンは異なり、下流トポロジーは PLM 内部ペアリングと互換な代替ではない。  
4. **Scratch:** 事前学習残基がないとトポロジー／アノテーション選択の影響は大きい。事前学習入力は設計選択の必要性を消すのではなく、有用なバイアスを組み替える。

### 非主張（Non-claims）

本レポートは、PLM が「抗体構造を理解している」、REG が物理的 H/L 接触に対応する、ペア PLM が「界面を学習した」、IMGT 情報が内部に符号化されている、といった主張を（別途実証されない限り）行わない。また結論を TmApp の外へ拡張しない。

---

## 6. 参考文献

1. Shehata, L., Maurer, D. P., Wec, A. Z., Lilov, A., Champney, E., Sun, T., Archambault, K., Burnina, I., Lynaugh, H., Zhi, X., Xu, Y., & Walker, L. M. (2019). Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability. *Cell Reports*, *28*(13), 3300–3308.e4. https://doi.org/10.1016/j.celrep.2019.08.056

2. TmApp の文言および補足完全例数に用いたリポジトリ・アッセイ定義: `gate_b1/reports/assay_definitions.md`（Shehata et al. 2019 補足列 **TmApp (°C)**）。

---

## 7. 再現性ポインタ

| 項目 | 場所 |
| --- | --- |
| 人間可読 master（200 セル） | `technical_report/tmapp_factorial/data/tmapp_factorial_human_master.csv` |
| Figures 1–7 | `technical_report/tmapp_factorial/figures/` |
| Tables 1–5 | `technical_report/tmapp_factorial/tables/` |
| 用語 | `technical_report/tmapp_factorial/terminology.yaml` |
| 凍結機械可読結果 | `results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv` |
| Internal freeze | `results/TMAPP_REP_TOPO_ANNOT_INTERNAL_FREEZE.yaml` |
| プラットフォーム | `DL_FOLDLOCAL_COSINE_V3`、seed 101 |
| 旧トポロジー対応 | A→SEP、B1→JOINT、B2→REG-SEP、C→XREG、D→FUSE |

---

**TmApp REPORT FROZEN.** 本ドキュメントから新たな TmApp 実験・再学習・再設計は起動しない。

*TmApp 技術レポート終わり。本ドキュメント作成にあたり新規実験は行っていない。*
