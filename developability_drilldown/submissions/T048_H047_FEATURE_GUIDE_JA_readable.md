# EXP-T048 / EXP-H047 特徴量ガイド（人間向け整理版）

この文書は、`EXP-T048` と `EXP-H047` が**実際に何を入力として使っているのか**を、実装監査の内容を保ったまま、人間が読みやすい形に整理したものです。

細かな provenance や個々の関数名を最初から並べるのではなく、

1. まずモデル全体の考え方
2. 次に各特徴量が「何を測っているのか」
3. 最後に実装上の詳細

の順に説明します。

---

# 1. まず全体像

## EXP-T048 — TmApp 予測

一言でいうと、

> **抗体配列そのものの表現**に、**局所CDR表現**、**構造揺らぎ**、**配列と構造の整合性**を足し、RBF-SVR で TmApp を予測するモデル。

です。

```text
AbLang2 の抗体全体表現
+ 基本的な配列統計
+ BioEmu 由来の構造揺らぎ
+ ProteinMPNN 由来の配列–構造適合度
+ AbLingua の CDR3 局所表現
+ AbLingua の全CDR局所表現
+ AbLang2 の「露出CDR」局所表現
        ↓
     RBF-SVR
        ↓
      TmApp
```

raw feature は 6,649 次元ですが、高次元の局所 PLM block 3本は CV 内でそれぞれ PCA32 に圧縮されてから SVR に入ります。

---

## EXP-H047 — HIC 予測

一言でいうと、

> **ESM-2 による配列表現**に、**抗体表面の疎水性・芳香族露出・電荷状態**と**露出したCDR3の局所表現**を足し、RBF-SVR で HIC retention time を予測するモデル。

です。

```text
ESM-2 Heavy 全体表現
+ 拡張配列記述子
+ 露出芳香族トポロジー
+ 連続疎水性 field
+ pH に対する電荷変化の形
+ ESM-2 の「露出CDR3」局所表現
        ↓
     RBF-SVR
        ↓
 HIC retention time
```

raw feature は 2,728 次元で、最後の RASA-CDR3 PLM block だけ PCA32 に圧縮されます。

---

# 2. まず覚えておくとよい違い

| 観点 | T048 | H047 |
|---|---|---|
| Target | TmApp | HIC retention time |
| 回帰器 | RBF-SVR | RBF-SVR |
| 主なPLM | AbLang2 + AbLingua | ESM-2 |
| Heavy/Light | H/Lをかなり明示的に使う | global ESM-2 は Heavy |
| 主な構造情報 | 構造揺らぎ、配列–構造適合 | 表面疎水性、芳香族露出、電荷状態 |
| 局所領域 | CDR3 / 全CDR | CDR3 |
| raw次元 | 6649 | 2728 |

---

# 3. T048 — 各特徴量は何をしているのか

## 3.1 AbLang2 paired H+L embedding — 「抗体全体を480次元で表す」

Heavy と Light の配列を **1組の抗体ペア**として `ablang2-paired` に入力し、480次元の sequence-level embedding を得ています。

```text
Heavy sequence
Light sequence
      ↓
 AbLang2 paired
      ↓
 480-d vector
```

これは `Heavy embedding ∥ Light embedding` の単純連結ではありません。AbLang2 の paired mode が返す `seqcoding` をそのまま使っています。

狙いは、単純なAA組成では表現できない antibody-specific な配列文脈や Heavy/Light の組合せを compact に表すことです。

実装メモ:
- model: `ablang2-paired`
- package: `ablang2` 0.2.1
- call: `ablang(seqs, mode="seqcoding")`
- output: 480 dimensions
- PCA: なし
- 明示的 L2 normalization: なし
- 学習前に StandardScaler

---

## 3.2 SEQ_BASIC — 「ごく基本的な配列統計 78個」

Heavy、Light、H+L全体について、

- 長さ
- 20種類のアミノ酸組成
- 疎水性残基の割合
- 芳香族残基の割合
- 荷電残基の割合
- Gly の割合
- Pro の割合

を数えています。

**SEQ_BASIC には CDR 長は入っていません。** 過去の一部ドキュメントには “CDR length summaries” と書かれていましたが、実装と canonical parquet を見る限り誤りです。

定義:

```text
HYDRO   = A, I, L, M, F, V, W, Y
AROM    = F, W, Y
CHARGED = K, R, D, E
```

Histidine は aromatic に含めていません。

Heavy 26 + Light 26 + H/L combined 26 = **78 features** です。

---

## 3.3 BioEmu NEW_PAIRWISE — 「構造がどのくらい揺れるか」

BioEmu で VH 単独、VL 単独について複数の structure conformer を作り、各 conformer 間の **Cα RMSD** を全ペアについて計算します。

```text
conformer 1
conformer 2
...
conformer 8
       ↓
all pairwise Cα RMSD
       ↓
median / 90th percentile
```

使う物理フレーム数は QC 後に **Nphys = 8** に固定されています。

VH と VL それぞれについて pairwise RMSD の median と q90 を計算し、それぞれについて

- VH
- VL
- mean(VH, VL)
- max(VH, VL)
- abs(VH − VL)

を作るので、**2 metrics × 5 summaries = 10 features** です。

要するに、これは一つの予測構造そのものではなく、**その可変領域がどの程度さまざまな構造を取りうるか**を簡易的に表す特徴です。

注意点として、Fab 全体の dynamics ではなく、VH と VL を**別々に** BioEmu へ入れています。また `NEW_PAIRWISE` の “NEW” は新しい数式ではなく、QC 後 Nphys=8 で再計算した新しい pipeline 世代を指します。

---

## 3.4 ProteinMPNN — 「この構造に、この配列は自然か？」

ESMFold で得た Fv 構造を ProteinMPNN に与え、その backbone 構造の上で、実際の native sequence がどれくらい尤もらしいかを score しています。

実体は **masked average negative log-likelihood (NLL)** です。

```text
ESMFold Fv backbone
+ actual antibody sequence
        ↓
   ProteinMPNN
        ↓
 average native-sequence NLL
```

feature は **1個だけ**です。

値が小さいほど「この backbone にこの sequence が乗っていることを ProteinMPNN が自然だとみなす」方向です。FoldX の ΔG などではなく、あくまで sequence–structure compatibility score です。

---

## 3.5 AbLingua CDR3 — 「CDR3だけをPLMで読む」

AbLingua の最終層 residue embedding を取り、Heavy CDR3 と Light CDR3 についてそれぞれ平均し、最後に連結します。

```text
AbLingua residue embeddings
        ↓
IMGT CDR3 mask
        ↓
mean(H-CDR3) ∥ mean(L-CDR3)
        ↓
2560 dimensions
        ↓
PCA32 in each CV fold
```

1280 + 1280 = 2560 raw dimensions です。ここでは RASA は使いません。

---

## 3.6 AbLang2 RASA × CDR — 「露出しているCDRを強く読む」

AbLang2 の residue embedding に対して、

1. IMGT CDR1/2/3 に限定
2. ESMFold構造から各残基の RASA を計算
3. RASA が高い残基ほど大きな重みを与える
4. weighted mean pooling

を行います。

式は、

\[
z=\frac{\sum_i RASA_i\,e_i}{\sum_i RASA_i}
\]

です。実際には RASA は 0–1 に clip されます。

RASA の計算は、
- structure: ESMFold Fv
- SASA: Bio.PDB `ShrakeRupley`
- probe radius: 1.4 Å
- n_points: 100
- normalization: Tien 2013 MaxASA

で、`RASA = SASA / MaxASA(residue type)` です。

Heavy CDR と Light CDR を別々に pool し、480 + 480 = 960 raw dimensions。その後 CV 内で PCA32 です。

---

## 3.7 AbLingua all-CDR — 「CDR1/2/3 全部をPLMで読む」

CDR3 block と同じ AbLingua residue embedding を使いますが、対象が `CDR1 ∪ CDR2 ∪ CDR3` です。

Heavy / Light を別々に平均し、1280 + 1280 = 2560 raw dimensions として、CV 内で PCA32 にします。RASA は使いません。

---

# 4. T048 を一枚で理解する

```text
                    ┌─────────────────────────────┐
                    │ AbLang2 paired: 抗体全体     │ 480
                    ├─────────────────────────────┤
Heavy + Light ─────►│ SEQ_BASIC: 基本配列統計      │ 78
                    ├─────────────────────────────┤
                    │ BioEmu: VH/VL構造揺らぎ      │ 10
                    ├─────────────────────────────┤
                    │ ProteinMPNN: 構造との整合性  │ 1
                    └─────────────────────────────┘
                                +
                    ┌─────────────────────────────┐
                    │ AbLingua CDR3               │ 2560 → PCA32
                    ├─────────────────────────────┤
                    │ AbLang2 RASA-weighted CDR   │ 960  → PCA32
                    ├─────────────────────────────┤
                    │ AbLingua all-CDR            │ 2560 → PCA32
                    └─────────────────────────────┘
                                ↓
                          StandardScaler
                                ↓
                           RBF-SVR
                     C=100, epsilon=0.05
                                ↓
                             TmApp
```

---

# 5. H047 — 各特徴量は何をしているのか

## 5.1 ESM-2 Heavy embedding — 「Heavy配列全体を1280次元で表す」

`facebook/esm2_t33_650M_UR50D` に Heavy sequence を入れ、special token を除いた residue hidden state を平均します。

```text
Heavy sequence
      ↓
 ESM-2 650M
      ↓
mean residue embedding
      ↓
1280 dimensions
```

PCA はかけません。この global ESM-2 block は **Heavy-only** です。

---

## 5.2 SEQ_ALL — 「SEQ_BASICを少し本格的にした115個」

SEQ_ALL は `SEQ_BASIC 78 + additional 37 = 115` です。

追加されるのは主に、

- positive / negative residue fraction
- polar fraction
- Cys fraction
- positive − negative
- pI
- GRAVY
- aromaticity
- pH 7 net charge
- Heavy–Light の pI / GRAVY / charge 差
- sequence entropy
- unique AA count

です。一部は `Bio.SeqUtils.ProtParam.ProteinAnalysis` を使っています。

含まれないものは、germline identity、CDR length、ANARCI region composition です。

---

## 5.3 AROMATIC_TOPO — 「表面に出た芳香族残基が、どんな塊を作っているか」

対象となる芳香族残基は **F, W, Y** のみです。His は含みません。

ESMFold Fv から SASA / RASA を計算し、

```text
RASA >= 0.20 : exposed
RASA >= 0.50 : strongly exposed
```

と定義します。

さらに、露出芳香族残基同士について `Cα distance <= 8 Å` なら接続しているとみなし、graph の connected component を aromatic patch とします。

したがってこれは、単に「露出F/W/Yがいくつあるか」だけでなく、「それらが空間的に集まっているか」を見ています。

特徴の例は、
- exposed aromatic count
- exposed aromatic SASA
- CDR 内にある割合
- aromatic patch 数
- 最大 patch size
- patch statistics
- 近傍 10 Å 内の aromatic SASA

などで、合計 **19 features** です。

---

## 5.4 HYDRO_FIELD — 「分子表面に疎水性の地図を作る」

まず FreeSASA で分子表面点を作ります。

その表面上の各点 \(s\) について、近くの原子から疎水性を足し合わせます。

\[
H(s)=\sum_{i:\|s-x_i\|\le7\AA}\pi_i e^{-\alpha\|s-x_i\|}
\]

ここで、
- \(\pi_i\): その残基の Fauchère–Pliska hydrophobicity
- \(x_i\): heavy atom の位置
- \(\alpha=1.0\ \AA^{-1}\)
- radius = 7 Å

です。

つまり、**近くに疎水性の高い残基が多い表面点ほど H(s) が高くなる**という連続的な疎水性 map を作っています。

その後、表面全体の H(s) の分布から、mean、quantile、high-H surface fraction、patch statistics などをまとめます。high-H の閾値は構造ごとの **80 percentile**、patch link distance は **2 Å** です。

重要なのは、これは canonical SAP ではないことです。`HYDRO_FIELD ≠ SAP` です。

---

## 5.5 TITRATION_SHAPE — 「pHを変えたとき電荷がどう動くか」

まず ESMFold Fv 構造に対して PROPKA3 で各 ionizable residue の pKa を予測します。

次に Henderson–Hasselbalch 式で、pH 4.0 から 10.0 まで 0.25 刻みで分子電荷 \(Q(pH)\) を計算します。

対象残基は、
- acidic: ASP, GLU, CYS, TYR
- basic: HIS, LYS, ARG

です。

そこから、
- Q(pH)
- dQ/dpH
- charge transition width
- major switch region 数
- CDR 上の charge

など、**pHに対して電荷状態がどう変化するかという曲線の形**を特徴にしています。合計 **18 features** です。

この family には pI の zero-crossing 値そのものは入っていません。sequence-based pI は SEQ_ALL にあります。

---

## 5.6 ESM-2 RASA × CDR3 — 「露出したHeavy CDR3をESM-2で読む」

T048 の RASA-weighted CDR pooling とほぼ同じ考え方ですが、

- PLM = ESM-2
- chain = Heavy only
- region = CDR3 only

です。

\[
z=\frac{\sum_i RASA_i e_i}{\sum_i RASA_i}
\]

を Heavy CDR3 について計算し、raw 1280 dimensions → CV 内 PCA32 とします。

---

# 6. H047 を一枚で理解する

```text
Heavy sequence
     │
     ├── ESM-2 global embedding ───────────── 1280
     │
H + L│
     └── SEQ_ALL ──────────────────────────── 115

ESMFold Fv structure
     │
     ├── exposed aromatic topology ────────── 19
     ├── continuous hydrophobic surface ───── 16
     └── PROPKA → Q(pH) titration shape ───── 18

Heavy CDR3
     │
     └── ESM-2 residue emb × RASA
           weighted pooling ───────────────── 1280 → PCA32

                         ↓
                   StandardScaler
                         ↓
                      RBF-SVR
                C=1.0, epsilon=0.05
                         ↓
                 HIC retention time
```

---

# 7. 分かりにくい略称を日本語にすると

| 名前 | 実際にしていること |
|---|---|
| `SEQ_BASIC` | H/L/Fv の長さ、AA組成、疎水/芳香/荷電/Gly/Pro割合 |
| `SEQ_ALL` | SEQ_BASIC + pI/GRAVY/charge/entropyなど |
| `BIOEMU_NEW_PAIRWISE` | VH/VLの複数conformer間 Cα-RMSD の median/q90 |
| `M1_PROTEINMPNN` | ESMFold backbone上でnative配列の平均NLL |
| `FB_AL_CDR3` | AbLingua residue embeddingを H/L CDR3 で平均 |
| `FB_AL2_RASA_CDR` | AbLang2 residue embeddingを露出度RASAで重み付けして全CDR平均 |
| `AROMATIC_TOPO` | 表面に露出したF/W/Yの量と空間patch構造 |
| `HYDRO_FIELD` | 分子表面上に連続的な疎水性 field を作って分布を要約 |
| `TITRATION_SHAPE` | PROPKA pKaからQ(pH)曲線を作り、その形を要約 |
| `FB_ESM2_RASA_CDR3` | Heavy CDR3 のESM-2 residue embeddingをRASA重み付き平均 |

---

# 8. raw feature 数と実際にSVRへ入る情報量

## T048

| block | raw |
|---|---:|
| AbLang2 paired | 480 |
| SEQ_BASIC | 78 |
| BioEmu NEW_PAIRWISE | 10 |
| ProteinMPNN | 1 |
| AbLingua CDR3 | 2560 → PCA32 |
| AbLang2 RASA-CDR | 960 → PCA32 |
| AbLingua all-CDR | 2560 → PCA32 |
| **raw total** | **6649** |

PCA 後の effective dimensionality は概ね `480 + 78 + 10 + 1 + 32 + 32 + 32 = 665` です。

## H047

| block | raw |
|---|---:|
| ESM-2 Heavy | 1280 |
| SEQ_ALL | 115 |
| AROMATIC_TOPO | 19 |
| HYDRO_FIELD | 16 |
| TITRATION_SHAPE | 18 |
| ESM-2 RASA-CDR3 | 1280 → PCA32 |
| **raw total** | **2728** |

PCA 後の effective dimensionality は概ね `1280 + 115 + 19 + 16 + 18 + 32 = 1480` です。

---

# 9. この2モデルの科学的な違い

## T048

T048 は、かなり大ざっぱに言えば、

> **「この抗体配列はどんな抗体なのか」**
> ＋ **「CDRの局所配列文脈はどうか」**
> ＋ **「構造がどれくらい揺れそうか」**
> ＋ **「その配列は予測backboneと整合しているか」**

をまとめています。

TmApp は Fab の apparent melting temperature なので、このような sequence / local region / conformational compatibility の混合は科学的には自然です。ただし、どの特徴も thermodynamic ΔG を直接計算しているわけではありません。

## H047

H047 は、

> **「Heavy配列はどんな文脈を持つか」**
> ＋ **「表面にどれくらい疎水性が集中しているか」**
> ＋ **「芳香族残基が露出して塊を作っているか」**
> ＋ **「pHで表面電荷がどう変わるか」**
> ＋ **「Heavy CDR3のうち露出している部分はどんな配列表現か」**

をまとめています。

これは HIC retention time に関係しうる exposed hydrophobicity、aromatic surface exposure、charge state、local CDR chemistry を明示的に表現しようとしたモデルです。HIC retention time は aggregation assay ではありません。

---

# 10. 実装詳細を確認したいときの早見表

| family | 主なtool / model | key calculation |
|---|---|---|
| AbLang2 paired | `ablang2-paired` | `seqcoding`, 480-d |
| SEQ_BASIC | Python `Counter` | AA fraction / length |
| BioEmu | `bioemu-v1.2` + mdtraj | pairwise Cα RMSD |
| ProteinMPNN | soluble `v_48_030.pt` | masked average NLL |
| AbLingua CDR | `IDEA-AI4S/AbLingua` | final residue hidden mean |
| RASA | Bio.PDB ShrakeRupley | SASA / Tien2013 MaxASA |
| ESM-2 | `esm2_t33_650M_UR50D` | residue hidden mean |
| AROMATIC_TOPO | ESMFold + ShrakeRupley | exposed F/W/Y + 8Å graph |
| HYDRO_FIELD | FreeSASA | Fauchère–Pliska hydrophobic field |
| TITRATION_SHAPE | PROPKA3 | pKa → Henderson–Hasselbalch Q(pH) |

---

# 11. まだ完全には解決していない点

1. AbLang2 `seqcoding` 内部で package がどの pooling をしているか
2. 過去文書の一部で SEQ_BASIC に CDR length が含まれるように書かれていた齟齬
3. bundle parquet への packaging / rename step の完全な追跡
4. H047 の global ESM-2 と residue ESM-2 が別 stack（HF / fair-esm）で生成されている点
5. HYDRO_FIELD family に混在した `phi_finite_frac` QC列
6. BioEmu 全抗体の元 XTC / individual seed log を今回再計算してはいない点
7. TITRATION_SHAPE 自体には pI zero-crossing は含まれない点

---

# 12. 最後に一行で

```text
T048:
抗体全体 + CDR局所 + 構造揺らぎ + 配列–構造整合性
→ RBF-SVR
→ TmApp

H047:
ESM-2配列 + 表面疎水性 + 芳香族露出 + pH依存電荷 + 露出CDR3
→ RBF-SVR
→ HIC
```
