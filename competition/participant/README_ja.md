# Antibody Developability Competition

English version: [README.md](README.md)

**Competition title:** Antibody Developability — TmApp & HIC  
**Package version:** 1.0  

**Tracks:** TmApp · HIC retention time  
**各トラックの主評価指標:** Mean Absolute Error（MAE）— 小さいほど良い  
**入力:** 抗体可変領域のペア配列（`heavy`, `light`）

この README は、抗体開発の専門知識が少ないコンピュータサイエンス / 機械学習 / データサイエンス参加者を想定して書かれています。

---

## 1. Developability（開発適性）とは何か

標的に強く結合し、高い薬効が期待できる抗体であっても、それだけで優れた医薬品候補になるとは限りません。

治療用抗体候補は、次のような点でも実務上使える必要があります。

- 発現（expression）  
- 精製（purification）  
- 製剤化（formulation）  
- 保存  
- 輸送  
- 再現性のある製造  

また、次のような問題をできるだけ避けることが望ましいです。

- 構造安定性の低さ  
- 過度な自己会合（self-association）  
- 凝集傾向  
- 表面疎水性の問題  
- 溶解性の問題  
- 非常に高い粘度  
- 非特異的相互作用  

**Developability（開発適性）** とは、生物学的に有望な分子が、医薬品として開発・製造していくうえで問題になりにくい物性を持っているかどうか、という広い概念です。

**単一の万能な developability score があるわけではありません。** Developability は複数の実験指標で評価されます。本コンペでは、そのうち **2つ** を扱います。

---

## 2. 「効くか？」と「開発していけるか？」

非生物系の参加者向けの直感的な対比です。

| 問いの種類 | 日常的な意味 | 例 |
|---|---|---|
| **生物学的な問い** | 「この抗体は狙った働きをするか？」 | 結合（binding）、特異性（specificity）、生物活性 |
| **Developability の問い** | 「この分子を現実的な医薬品として開発していけるか？」 | 安定性、溶解性、疎水性、自己会合、凝集関連リスク、粘度、製造適性（manufacturability） |

これらは完全に独立ではありませんが、区別して考えるとわかりやすいです。**本コンペが扱うのは、後者のうちごく一部の物性軸です。**

---

## 3. なぜ早期のリスク予測が重要なのか

早期の探索段階では、候補抗体が多数あることが普通です。一方、候補が次のような後期工程に進むほど、実験コストと運用負荷は大きく上がります。

- 詳細な物性評価  
- プロセス開発  
- 精製プロセス開発  
- 製剤開発  
- スケールアップ  
- 安定性試験  
- 外部製造（CMO/CDMO など）  

望ましくない物性が、すでに大きなプロセス開発・製造リソースを投下した **後** に見つかると、追加実験、製剤やり直し、精製条件変更、候補再設計、遅延、コスト増などが起きえます。

したがって、早期 developability 評価の重要な目的は次です。

> 高価なプロセス開発や製造に進む前に、将来問題になりうる物性リスクを早期に把握すること。

**注意:** TmApp や HIC だけでは、「この抗体は製造できる / できない」を直接判断することはできません。これらは、定義された測定条件のもとで得られる、**developability に関連する早期リスク指標**です。製造成功そのものを直接予測する指標ではありません。

---

## 4. 抗体配列の短い入門

抗体は **重鎖（heavy chain）** と **軽鎖（light chain）** からなります。本データでは:

- `heavy` = 重鎖可変領域（VH）のアミノ酸配列  
- `light` = 軽鎖可変領域（VL）のアミノ酸配列  

可変領域は抗原認識に中心的です。比較的保存された **フレームワーク領域（framework region）** と、より可変な **相補性決定領域（CDR; complementarity-determining region）** を含みます。特に CDR は、抗原結合部位の形状や抗原との結合様式に強く関与します。

配列から物性を予測しうる理由の直感は次のとおりです。

> 配列 → 局所化学 → 構造 / 表面物性 → 測定可能な物理化学的挙動

免疫学の専門家である必要はありません。配列記述子や protein language model（PLM）だけから始める強いアプローチも多いです。

---

## 5. 配列由来の抗体アノテーション

抗体配列は、単なる20種類のアミノ酸の文字列として扱うこともできますが、抗体特有の生物学的な観点から整理することもできます。便宜のため、VH/VL 配列そのものから推定した **配列由来の抗体アノテーション（sequence-derived antibody annotations）** を、任意利用の追加ファイルとして提供します。

| ファイル | 役割 | N |
|---|---|---:|
| `data/dev_annotations.csv` | Dev ID 向けアノテーション | 162 |
| `data/test_annotations.csv` | Test ID 向けアノテーション | 162 |

`dev.csv` / `test_features.csv` とは `id` で結合します。`heavy` / `light` / `TmApp` / `HIC` は重複しません。

### 含まれる内容（提供できるもの）

- 重鎖・軽鎖の推定 **V/J family**  
- **light-chain type**（`kappa` / `lambda`）  
- **CDR length**（HCDR1–3、LCDR1–3）  
- 重鎖・軽鎖 V 領域の **germline identity**（0–1）

定義・ツール・番号付け規約の詳細は `DATA_DICTIONARY.md` を参照してください。

### germline（生殖細胞系列）の直感的な説明

抗体可変領域は、生まれつきゲノムに存在する複数の **germline（生殖細胞系列）** gene segment を組み合わせ（**V(D)J recombination**）、その後さらに配列変化を蓄積することで多様化します。観測された抗体配列を reference germline sequence と比較することで、どの **V/J family** に近いか、germline からどの程度離れているか（**germline identity**）を推定できます。

germline identity が低い（つまり inferred germline から遠い）ことは、**somatic hypermutation（体細胞超変異, SHM）** — 抗体の成熟過程で蓄積しうる配列多様化 — などによる配列の乖離を反映し得ます。

**重要:** germline から遠いほど良い／近いほど良い、という意味ではありません。SHM の多さが親和性や developability の良し悪しを直接意味するわけでもありません。記述子として扱い、予測との関係は各自で検討してください。

### CDR length

**CDR**（相補性決定領域）は抗原認識に関わる可変性の高いループです。長さや配列組成は局所的な形・構造・露出する表面の化学的性質に影響しうるため、CDR length は有用な配列由来の記述子になり得ます。特定の CDR が TmApp / HIC を決める、といった主張はしません。

### light-chain type（κ / λ）

ヒト抗体の軽鎖には、よく見られるタイプとして **kappa（κ）** と **lambda（λ）** があります。アノテーションは配列から推定したタイプを記録します。κ / λ に優劣があるという意味ではありません。

### 任意リソースであり、特権的なラベル情報ではない

> これらのアノテーションファイルは **任意の便宜的リソース** です。提供された抗体配列から導出されており、追加の実験的な TmApp / HIC 情報は含みません。

無視して `heavy` / `light` だけを使っても構いません。公式スコアラーはアノテーションファイルを使いません。


## 6. Target 1 — TmApp（apparent melting temperature, °C）

### 直感

タンパク質は通常、特定の3次元構造に折りたたまれています。温度を上げると、その構造はやがて熱変性・unfolding を起こします。**Differential scanning fluorimetry（DSF）** は、その過程に伴う温度依存的な蛍光変化を追跡します。**TmApp** は、測定条件下で観測された特徴的な **見かけの（apparent）** 熱転移温度をまとめた値です。

- **単位:** 摂氏（°C）  
- **TmApp が高い** ほど、その測定条件下では一般に **熱・構造安定性がより高い** ことを示します。

### 原論文で何を測っているか

Shehata らの研究では:

- **TmApp** は **apparent melting temperature（見かけの融解温度）** を意味します。  
- 測定対象は抗体の **Fab** 断片（抗原結合腕）です。  
- 熱安定性は **DSF** で測定されています。  
- 精製 Fab 試料を加熱しながら蛍光をモニターします。  
- 見かけの融解 / 熱転移温度は、熱蛍光曲線（またはその微分）から、当該研究の方法に従って割り当てられます。  
- TmApp は、構造 / 熱安定性や unfolding への耐性に関連する指標として用いられました。

**メカニズム表現の注意:** 一般的な dye-based DSF では、本来埋もれていた疎水性領域が露出すると応答する色素を使うことがあります。これは **一般的な dye-based DSF の直感** として扱ってください。本論文に固有のメカニズム主張として必須ではありません。本コンペでは次で十分です。

> DSFでは、試料を徐々に加熱しながら、タンパク質のunfoldingに伴う温度依存的な蛍光変化を追跡します。

### なぜ “apparent” なのか

**TmApp は次のものではありません。**

- 普遍的な熱力学定数  
- 絶対的な破壊温度  
- pass/fail 閾値  

測定される転移は、コンストラクト、バッファ / 溶液条件、測定プロトコル、昇温条件などの assay context に依存します。タンパク質の unfolding が理想的な可逆平衡として振る舞うとは限りません。

したがって:

> **TmApp = 定義された測定条件下での見かけの融解 / 熱転移温度**

例えば **TmApp = 70 °C** は、「その測定条件でおおよそその温度付近に特徴的な熱転移が見られた」ことを意味します。**69.9 °C では完全に安定で、70.0 °C で急に壊れる** という意味ではありません。

### Developability における意味

構造安定性が高い分子は、熱などによる構造変化に対してより頑健であることが期待されます。熱安定性が低い場合には、折りたたみ状態が比較的脆く、developability 上のリスクが高まることがあります。

**必須の注意:**

- TmApp 自体は **凝集測定ではありません**。  
- TmApp 単独の pass/fail 基準ではありません。  
- TmApp が高いからといって、他の developability がすべて良いとは限りません。  

TmApp は developability に関連する **1つの軸** として扱ってください。

---

## 7. Target 2 — HIC retention time（分）

### 直感

**HIC** = **Hydrophobic Interaction Chromatography（疎水性相互作用クロマトグラフィー）**

この実験は、指定されたクロマトグラフィプロトコルのもとで、分子が **疎水性固定相** とどれだけ強く相互作用するかを測ります。本データセットの HIC 値は、原研究で報告された **IgG** の HIC retention time に対応します。

コンペの予測対象は **retention time（保持時間）** です。

- **単位:** 分（min）  
- **保持時間が長い** ほど、その測定条件下では一般に **より強い疎水的相互作用** / より強い有効な表面疎水性を反映します。

### だから何が問題なのか？

概念的な連鎖:

1. 表面に強い疎水性領域が露出（有効な疎水性）  
2. → より強い疎水的相互作用（プロトコル下でより長い HIC retention）  
3. → 抗体分子同士の不要な相互作用 / self-association（自己会合）の傾向が増す可能性  
4. → 凝集関連・溶解性・製剤化などの developability リスクと関連しうる  

HIC retention time が長い抗体は、その測定条件ではより強い疎水的相互作用を示します。抗体表面に強い疎水性領域が露出していると、抗体同士の不要な相互作用や自己会合が起こりやすくなり、場合によっては凝集、溶解性、製剤化などの問題と関連する可能性があります。

**必須の注意:**

> ただし、HICは凝集そのものを測定する実験ではありません。  
> HIC値が高いからといって、その抗体が必ず凝集するという意味ではありません。

「高 HIC = 凝集する抗体」と同一視しないでください。

### Assay 依存性

HIC retention time は **プロトコル依存** です。抗体の普遍的な固有定数ではありません。絶対的な数値スケールは、カラム化学、移動相、塩条件、グラジエント、流速などに依存します。

本コンペが予測するのは:

> **本データセットが表す測定文脈での HIC retention time**

であり、普遍的な「疎水性スコア」ではありません。

### 解釈用バンド（competition class ではない）

スケール理解のための **解釈用 / 診断用の参考 band** です。

| Band | HIC retention time |
|---|---|
| LOW | < 10.5 min |
| MEDIUM | 10.5–11.5 min |
| HIGH | > 11.5 min |

これらは competition class でも、普遍的な臨床閾値でも、公式の pass/fail ルールでもありません。

コンペのターゲットは引き続き **連続値** です。採点は **数値の HIC** を使います。自分の探索分析以外で、分類タスクとして扱わないでください。

---

## 8. ターゲット比較表

| Target | 実験で主に見ているもの | 単位 | 高い値のおおまかな意味 | Developability 上の直感 |
|---|---|---|---|---|
| **TmApp** | Fab の DSF による熱転移 | °C | 測定条件下でより高い熱・構造安定性 | 折りたたみ構造がより頑健 |
| **HIC** | IgG の疎水性相互作用クロマトグラフィー保持時間 | min | より強い有効疎水性相互作用 | 自己会合・凝集関連リスクと関連しうる |

TmApp も HIC も、それ単独で抗体の developability や manufacturability を決めるものではありません。  
**TmApp は凝集測定ではありません。**  
**HIC も凝集測定ではありません。**

これらは、早期 developability 評価に関連する **2つの異なる物性軸** です。

---

## 9. なぜ配列から予測できる可能性があるのか

アミノ酸配列は、側鎖化学、電荷、疎水性、芳香族性、水素結合、ループ組成、パッキングなどを決めます。それらが折りたたみ安定性、露出表面化学、疎水性パッチ、自己相互作用傾向に影響します。

したがって VH/VL 配列には、TmApp と HIC の両方に関連する情報が含まれえます。

使える方法の例:

- sequence descriptors（配列記述子）  
- protein language model（PLM）  
- antibody-specific language model  
- 構造予測に基づく特徴量  
- アンサンブル  

検証可能な範囲で選んでください。主催者のベンチマーク成績は公開していません。

---

## 10. 原研究と本コンペの違い

Shehata らの研究は、もともと sequence-to-property の機械学習ベンチマークとして設計されたものではありません。ヒト抗体パネルの生物学的・生物物理学的性質（親和性成熟と物性の関係など）を調べた研究です。

本コンペは、実験測定値を次の教師あり予測タスクとして再利用しています。

- VH + VL 配列 → **TmApp**  
- VH + VL 配列 → **HIC retention time**

原著者が PLM・配列予測モデル・本コンペ形式を評価したかのように書かないでください。

---

## 11. データセット

配布ファイルの正確な件数:

| File | 役割 | N |
|---|---|---:|
| `data/dev.csv` | ラベル付き開発 / 学習セット | 162 |
| `data/test_features.csv` | ラベルなしテスト配列 | 162 |
| `data/dev_annotations.csv` | 任意の配列由来アノテーション（Dev） | 162 |
| `data/test_annotations.csv` | 任意の配列由来アノテーション（Test） | 162 |
| `data/sample_submission.csv` | 提出テンプレート例 | 162 |

本パッケージの抗体は、主催者側では TmApp と HIC の両方を持ちます（`dev.csv` にも両方のラベルがあります）。

### 列

| Column | 出現箇所 | 意味 |
|---|---|---|
| `id` | すべて | 抗体 ID |
| `heavy` | dev, test_features | VH アミノ酸配列 |
| `light` | dev, test_features | VL アミノ酸配列 |
| `TmApp` | dev, submission | Apparent melting temperature（°C） |
| `HIC` | dev, submission | HIC retention time（min） |

正式定義は `DATA_DICTIONARY.md` を参照してください。

---

## 12. 独立した 2 つのリーダーボード

本コンペには **2つのトラック** があります。

1. **TmApp**  
2. **HIC**

それぞれ **独立に** 採点されます。

**主評価指標:** Mean Absolute Error（MAE）

\[
\mathrm{MAE} = \frac{1}{n}\sum_i \lvert y_i - \hat{y}_i\rvert
\]

小さいほど良いです。

次のものは **ありません**。

- 合算スコア  
- 重み付き平均  
- 全体のグランドメトリック  

優勝はトラックごとに分かれえます（TmApp Champion / HIC Champion）。

### 同点ルール（科学的定義）

各トラックの最終順位は **Private MAE**（小さいほど良い）で決まります。

あるトラックで 2 チームの **Private MAE が完全に同じ** 場合、それらは **同点（共有順位）** です。

Exact Private MAE の同点を、Pearson・Spearman・RMSE・Public スコア・提出時刻・提出 ID で **打破しません**。

ホスト基盤の表示都合で同点が別扱いに見える場合があっても、科学的ルールは「Private MAE 完全一致は同点」です。

---

## 13. Public / Private リーダーボード

- Test 全体: **N = 162**  
- Public: **N = 81**  
- Private（最終順位用）: **N = 81**  

Public と Private は、TmApp と HIC で **同じ分割** を使います。

Public は Test の固定部分集合、最終順位は held-out の Private を使います。

どの行が Public / Private に属するかは非公開です。

---

## 14. 推奨する検証の考え方

Public leaderboard は有限サンプルなのでノイズを含みます。小さな Public 変動を追いすぎないでください。

実務的な勧め:

1. `dev.csv` 上で信頼できる **local CV** を作る  
2. CV と Public の **両方を不完全な証拠** として見る  
3. 小さな Public 変動に過剰反応しない  

**HIC** については、比較的 sparse な high-HIC 尾部があるため、単純なランダム分割だと fold 間ばらつきが大きくなりえます。出発点としては、**sequence-group-aware** な分割に加えて **target-distribution-aware stratification** が現実的です。

「CV が常に勝つ」わけではありません。

---

## 15. 提出形式

提出 CSV の列:

```text
id,TmApp,HIC
```

要件:

- Test ID 集合が `test_features.csv` / `sample_submission.csv` と完全一致  
- ID は一意  
- `TmApp` と `HIC` の両方があり、いずれも **有限値**（NaN / Inf 不可）  
- 採点は **`id` で join**（行順には依存しない）  

**片方のトラックだけ試したい場合:** それでも両列が必要です。積極的にモデル化しない側は、学習セット中央値などの単純な予測で構いません。

`sample_submission.csv` は両列とも Train 中央値の placeholder です。

---

## 16. 重要な科学的注意

- TmApp は assay 依存であり、普遍的な安定性定数ではありません。  
- HIC retention time はクロマトグラフィプロトコル依存です。  
- HIC は直接の凝集測定ではありません。  
- TmApp も直接の凝集測定ではありません。  
- どちらも単独では developability を定義しません。  
- 配列ベース予測はスクリーニング / 優先順位付けの道具であり、実験評価の代替ではありません。

---

## 17. データ由来と attribution

本コンペデータセットは、次の報告に由来します。

> Shehata, L., Maurer, D. P., Wec, A. Z., Lilov, A., Champney, E., Sun, T., Archambault, K., Burnina, I., Lynaugh, H., Zhi, X., Xu, Y., & Walker, L. M. (2019). Affinity maturation enhances antibody specificity but compromises conformational stability. *Cell Reports*, *28*(13), 3300–3308.e4.  
> DOI: [10.1016/j.celrep.2019.08.056](https://doi.org/10.1016/j.celrep.2019.08.056)

Version of Record は Creative Commons Attribution 4.0 International（**CC BY 4.0**）で配布されています。  
https://creativecommons.org/licenses/by/4.0/

元データを機械学習コンペ用に再構成・分割しています。具体的には次を行っています。

- 著者提供の抗体 ID と VH/VL 配列を使用  
- 配列を大文字アミノ酸文字列として保持  
- TmApp と HIC の両方がある抗体に絞り込み  
- ラベル付き開発セット（`dev.csv`）とラベルなしテスト（`test_features.csv`）へ再編成  
- Test 内に固定の Public/Private 分割を適用  

**原著者は本コンペの運営・責任主体ではありません。** データ利用は、原著者による endorsement（推奨・承認）を意味しません。

---

## 配布ファイル

```text
README.md
README_ja.md
data/dev.csv
data/test_features.csv
data/dev_annotations.csv
data/test_annotations.csv
data/sample_submission.csv
data/DATA_DICTIONARY.md
```

健闘を祈ります。覚えておいてください: **トラックは2つ、各MAE、合算スコアなし。**
