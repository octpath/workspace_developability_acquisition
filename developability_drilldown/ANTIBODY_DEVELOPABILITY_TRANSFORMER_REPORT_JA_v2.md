# 抗体配列のDevelopability予測へのTransformerの応用
## — 残基表現、Heavy/Light相互作用、構造・物性情報を統合するモデル設計 —

**文書位置づけ:** 研究設計・実装・結果の統合レポート  
**対象:** TmApp / HIC  
**実装監査:** `TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md` に基づき、主要実装は repository と照合済み  
**注記:** 本文では研究上の主題を Transformer framework とその設計思想に置き、AROMATIC_TOPO / HYDRO_FIELD などの詳細 feature 定義は Appendix にまとめる。

---

# 1. Executive Summary

本検討では、抗体の VH/VL 配列から developability に関連する 2 つの連続値、
**TmApp（apparent melting temperature）** と **HIC retention time** を予測するために、
残基レベル Transformer を中心としたモデル設計を検討した。

本検討で得られた主な成果は、単一の protein language model（PLM）や単一の最良スコアではない。
より本質的には、

> **抗体を「Heavy鎖とLight鎖からなる残基列」として保持し、残基表現、抗体固有annotation、H/L間通信、summary token、構造・物性情報を独立した設計要素として組み替えられる Transformer framework を構築したこと**

にある。

この framework では、各残基の content representation を

- amino-acid identity から学習する **Scratch embedding**
- 抗体特異的 PLM の **AbLang2**
- protein PLM の **ESM-2**

などから選び、その後の Transformer 構造は共通化できる。

さらに、抗体特有の

- Heavy / Light chain identity
- sequence position
- IMGT position
- CDR / framework region

を明示的に与え、
H/L の communication pattern を attention mask や cross-attention として設計できる。

実験結果として、TmApp では **AbLang2 + H/L integration** が最も強く、
特に

- **ARCH-3:** joint unrestricted dual REG
- **ARCH-7:** REG-only cross-attention

が最有力クラスターとなった。

一方で、Scratch representation でも適切な H/L architecture を使うことで
PLM モデルに比較的近い内部性能まで到達した。
これは、性能が PLM の事前学習知識だけで決まるのではなく、
**抗体に適した inductive bias と residue-level aggregation 自体にも大きな価値がある**
ことを示唆する。

HIC では、Transformer architecture の差よりも、
明示的な分子表面の物理・化学情報を late fusion することが有効だった。
特に

$$
\mathrm{SURFACE}
=
\mathrm{AROMATIC\_TOPO}_{19}
+
\mathrm{HYDRO\_FIELD}_{16}
$$

が一貫して有効であり、
Scratch Transformer + SURFACE でも ESM-2 系モデルと同程度の有望性能が得られた。

したがって、本検討からは次の2つの方向性が得られた。

> **TmApp:** residue representation と H/L integration の設計が重要。  
> **HIC:** residue model に加え、明示的な surface physicochemistry が重要。

また、この Transformer framework は将来的に

- 残基間距離の attention bias
- 接触グラフ
- Graph Transformer / GNN 型 message passing
- residue-level RASA / electrostatics
- multi-conformer / structural ensemble
- multispecific antibody の多鎖処理

へ自然に拡張できる。

---

# 2. なぜ抗体Developability予測にTransformerを使うのか

## 2.1 fixed-length feature の限界

抗体配列の予測では、従来よく使われる方法として、

1. 配列全体を数十〜数百個の descriptor に変換する
2. PLM embedding を mean pooling して 1 ベクトルにする
3. Ridge / SVR / XGBoost などへ入力する

といった fixed-length model がある。

これらは small-N では非常に強力であり、本プロジェクトでも H047 など優秀な classical model が得られた。

しかし、この方式では

- どの残基がどの残基と関係するか
- Heavy と Light の情報をどこで統合するか
- CDR3 のような特定領域を prediction task に応じて動的に重視するか

といった **残基間関係そのもの** は明示的には扱いにくい。

Transformer を使う最大の理由は、残基列をそのまま保持して、
$$
\text{residue}
\rightarrow
\text{residue interaction}
\rightarrow
\text{antibody-level representation}
$$
という計算を学習できる点にある。

---

# 3. 本研究で構築した antibody-aware residue Transformer framework

## 3.1 基本思想

本 framework では、「PLMを使うかどうか」と「抗体をどう処理するか」を分離する。

概念図:

```text
                 residue content
              ┌────────┴────────┐
              │                 │
       Scratch AA embedding   pretrained PLM
                            AbLang2 / ESM-2
              │                 │
              └────────┬────────┘
                       ↓
              + sequence position
              + H/L chain identity
              + IMGT position
              + CDR/FR region
                       ↓
        ┌────────────────────────────┐
        │ antibody-aware Transformer │
        │                            │
        │ H/L communication          │
        │ REG token aggregation      │
        │ chain-specific masks       │
        │ cross-attention            │
        └────────────────────────────┘
                       ↓
           antibody-level vector
                       ↓
        ┌──────────────┴──────────────┐
        │                             │
 direct regression            physical-feature fusion
                              / structure information
```

重要なのは、

> **PLMは「各残基のcontentを作る一つの方法」であり、Transformer frameworkそのものではない**

という点である。

したがって、
$$
\text{Scratch}
\leftrightarrow
\text{AbLang2}
\leftrightarrow
\text{ESM-2}
$$
を比較しても、下流の H/L processing を共通化できる。

---

## 3.2 residue token に何を持たせるか

各 residue $i$ の Transformer 入力は概念的に
$$
\mathbf{x}_i
=
\mathbf{x}^{content}_i
+
\mathbf{x}^{pos}_i
+
\mathbf{x}^{chain}_i
+
\mathbf{x}^{IMGT}_i
+
\mathbf{x}^{region}_i
$$
で構成する。

### content

Scratch の場合:
$$
\mathbf{x}^{content}_i
=
\mathrm{Embedding}(\mathrm{AA}_i)
$$
PLM の場合:
$$
\mathbf{x}^{content}_i
=
W_{proj}\mathbf{h}^{PLM}_i
$$
AbLang2 では raw 480-dimensional residue representation を
trainable `Linear(480→128)` で d_model=128 へ射影する。

### antibody-specific annotation

それに加えて、

- sequence position
- H / L chain
- IMGT position
- CDR / FR region

を embedding として加える。

この設計により、
PLM が持つ一般的な sequence context と、
抗体特有の「どこにある残基か」という情報を分離して扱える。

---

# 4. REG token: 残基列から抗体全体をどう要約するか

## 4.1 REG token の意味

REG token は BERT の `[CLS]` に近い、学習可能な summary token である。

単純 mean pooling なら、
$$
\mathbf{z}
=
\frac{1}{N}
\sum_{i=1}^{N}\mathbf{x}_i
$$
となり、全残基を同じ重みで圧縮する。

一方 REG token は attention を通じて、
$$
\mathbf{r}^{(l+1)}
=
\mathrm{Attention}
\left(
Q=\mathbf{r}^{(l)},
K=X^{(l)},
V=X^{(l)}
\right)
$$
として残基情報を選択的に集約する。

つまり REG token は

> **property-specific learnable pooling**

として働く。

---

## 4.2 single REG と dual REG

抗体では Heavy と Light が明確に異なる chain であるため、
summary token の設計にも自由度がある。

### single REG

```text
[REG] H1 H2 ... Hn L1 L2 ... Lm
```

抗体全体を1個の summary token に圧縮する。

### dual REG

```text
[REG_H] H1 ... Hn [REG_L] L1 ... Lm
```

Heavy / Light に対応する 2 個の summary token を持つ。

最終的には concat または mean で統合できる。

本研究では merge 方法の比較も行ったが、
本レポートの主題は merge 自体ではなく、
**REG token が H/L interaction と antibody-level aggregation の設計点になっている**
ことである。

---

# 5. 有望だったTransformer architecture

本研究では多くの H/L interaction pattern を試したが、
ここでは現在の理解に重要なものだけを示す。

---

# 5.1 ARCH-3 — Joint unrestricted dual REG

## 概念

Heavy と Light を同じ Transformer sequence に入れ、
すべての有効 token 間で self-attention を許す。

```text
[REG_H] H1 H2 ... Hn [REG_L] L1 L2 ... Lm
        \________________________________/
                    ↓
          unrestricted self-attention
                    ↓
              REG_H, REG_L
```

padding 以外の H/L restriction はない。

1 head では、
$$
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V
$$
$$
A=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
+
M_{padding}
\right)
$$
$$
Y=AV
$$
であり、H residue と L residue の間にも直接 attention が張られる。

したがって、
$$
H_i \leftrightarrow L_j
$$
だけでなく、
$$
REG_H \leftrightarrow L_j,\qquad
REG_L \leftrightarrow H_i
$$
も可能である。

## 設計思想

これは

> **Heavy と Light を最初から一つの抗体として contextualize する**

構造である。

一方で summary token を2個持つため、
完全に1 vectorへ早期圧縮するわけではない。

## TmApp

AbLang2 + ARCH-3 の EXP-T113 は、

- OOF TEST Primary ≈ 2.995
- OOF TEST Shadow ≈ 3.283
- TEST mean ≈ 3.139
- external Overall ≈ 3.052

で、現在の TmApp 最有力クラスターである。

---

# 5.2 ARCH-7 — REG-only cross-attention

## 概念

ARCH-7 はより強い inductive bias を持つ。

Heavy / Light の residue encoder は別々に適用する。

```text
Heavy residues                     Light residues
      │                                  │
 shared encoder                     shared encoder
      │                                  │
 REG_H + H residues                 REG_L + L residues
```

その後、**REG token だけが相手鎖の residue を読む**。
$$
\Delta r_H
=
\mathrm{MHA}
(Q=r_H,\ K=L,\ V=L)
$$
$$
\Delta r_L
=
\mathrm{MHA}
(Q=r_L,\ K=H,\ V=H)
$$
$$
r'_H=r_H+\Delta r_H
$$
$$
r'_L=r_L+\Delta r_L
$$
となる。

実装では、

- H/L は同じ shared encoder を別々に通す
- cross-attention は full encoder 後に一度だけ
- H→L / L→H は同じ MHA module を共有
- query は REG token 1個
- key/value は相手鎖の real residues のみ
- REG token は K/V から除外
- padding mask のみ
- cross 後に additional LayerNorm / FFN / gate はない

という単純な構造である。

## 1 query × 全相手残基

Heavy summary $r_H$ が Light 残基 $L_1,\dots,L_m$ を読む場合、
$$
s_j
=
\frac{
(r_HW_Q)(L_jW_K)^\top
}{
\sqrt{d_h}
}
$$
$$
a_j
=
\frac{\exp(s_j)}
{\sum_{k=1}^{m}\exp(s_k)}
$$
$$
\Delta r_H
=
\sum_j a_j(L_jW_V)
$$
となる。

つまり REG_H は Light 鎖を一度 mean pool した vector を受け取るのではなく、
**Light 全残基へ query を出し、task に応じて必要な residue を選択的に読む**。

これは
$$
\mathrm{MLP}([REG_H;REG_L])
$$
とは本質的に異なる。
MLP はすでに圧縮された summary しか見られないが、
ARCH-7 は final aggregation の直前に residue-level information へ再アクセスできる。

## TmApp

AbLang2 + ARCH-7 の EXP-T121 は、

- OOF TEST Primary ≈ 3.023
- OOF TEST Shadow ≈ 3.252
- TEST mean ≈ 3.138

で、T113 とほぼ同等の内部最良クラスターに入った。

---

# 5.3 ARCH-4 — Joint residues + chain-specific dual REG

ARCH-4 では biological residues は H/L 間で communication できるが、
REG_H / REG_L には chain-specific な役割を課す。

概念的には、

```text
residue level:
    H ↔ L communication allowed

summary level:
    REG_H -> Heavy-oriented summary
    REG_L -> Light-oriented summary
```

となる。

実装の attention permission は以下。

| query \ key | REG_H | H residues | REG_L | L residues |
|---|:---:|:---:|:---:|:---:|
| REG_H | 1 | 1 | 0 | 0 |
| H residue | 1 | 1 | 0 | 1 |
| REG_L | 0 | 0 | 1 | 1 |
| L residue | 0 | 1 | 1 | 1 |

つまり H/L residue 同士は communication できるが、
REG token は対応する chain を中心に summary を作る。

TmApp Scratch ではこの種の chain-specific joint design が強く、
HIC では ESM-2 + ARCH-4 + SURFACE が有望モデルとなった。

---

# 5.4 Joint single REG

最も単純な H/L joint architecture は、

```text
[REG] H1 H2 ... Hn L1 L2 ... Lm
              ↓
       joint self-attention
              ↓
             REG
```

である。

これは summary bottleneck を1個に絞り、

> **抗体全体を一つの latent representation として学習する**

構造である。

HIC Scratch ではこの単純な joint single REG が強く、
SURFACE を組み合わせた H090 が有力モデルになった。

---

# 6. Scratch と PLM を分離して比較する意味

## 6.1 Scratch とは何か

本研究でいう Scratch は、
PLM residue embedding を使わず、

- amino-acid identity
- sequence position
- chain identity
- IMGT position
- CDR / FR region

のみから Transformer が representation を学習する構成である。

つまり、PLM由来の進化的・配列統計的知識を初期表現に持たない。

---

## 6.2 TmApp での結果

代表例:

| Model | Representation | Architecture | OOF TEST mean |
|---|---|---|---:|
| T113 | AbLang2 | ARCH-3 | 3.139 |
| T121 | AbLang2 | ARCH-7 | 3.138 |
| T096 | Scratch | ARCH-4 | 3.258 |

Scratch の最良帯は AbLang2 よりやや劣るが、
差は内部 MAE でおよそ 0.12 程度である。

これは

> **PLMなしでは全く学習できない**

という状況ではない。

適切な H/L interaction と antibody annotation を用いることで、
Scratch でもかなり競争力のある性能まで到達した。

一方で external performance では Scratch の T096 は AbLang2 より弱かったため、
「ScratchとPLMが同等」と結論するのは適切ではない。

より慎重には、

> **PLMが重要である一方、性能のかなりの部分は antibody-aware Transformer architecture 側でも獲得できる**

と解釈するのが妥当である。

---

## 6.3 HIC での結果

SURFACE late fusion を組み合わせた代表例:

| Model | Residue representation | Backbone | Added features | OOF TEST mean | Overall |
|---|---|---|---|---:|---:|
| H086 | ESM-2 | ARCH-4 | SURFACE | 0.486 | 0.406 |
| H090 | Scratch | joint single REG | SURFACE | 0.472 | 0.409 |

HIC では Scratch + SURFACE が ESM-2 + SURFACE とほぼ同じ性能帯に入った。

これは HIC では

> **PLMの高度化よりも、explicit surface chemistry の有無が大きな差を作る場合がある**

ことを示唆する。

---

# 7. TmApp から分かったこと

TmApp では、最終的に

- AbLang2 residue representation
- small Transformer
- H/L joint or controlled cross-chain aggregation

が有力だった。

一方、

- d_model 128 → 256
- layers 2 → 3
- heads 4 → 8
- FFN 256 → 512

という単純 capacity expansion は改善しなかった。

したがって、現在のデータ規模では、

> **より大きいモデルを作ることより、どういう residue representation と H/L inductive bias を与えるかが重要**

と考えられる。

---

# 8. HIC から分かったこと

HIC では broad architecture search だけでは大きな改善が得られず、
内部 OOF TEST mean は概ね 0.50 前後だった。

一方、H047 に由来する explicit surface / physicochemical features を
late fusion すると明確な改善が得られた。

特に有効だったのが
$$
\boxed{
\mathrm{SURFACE}
=
\mathrm{AROMATIC\_TOPO}_{19}
+
\mathrm{HYDRO\_FIELD}_{16}
}
$$
である。

H086 / H090 などでは external Overall も historical H047 と同等以上の有望帯に入った。

さらに auxiliary feature block を permutation すると、
held-out MAE が Primary で概ね +0.13〜+0.20 悪化した。

これは、

> **SURFACE branch が単に学習軌道を変えただけでなく、学習済みモデルが実際にその情報へ依存している**

ことを支持する。

---

# 9. SURFACE late fusion

## 9.1 基本構造

```text
H/L sequence
    ↓
ESM-2 or Scratch residue Transformer
    ↓
z_DL
                              predicted Fv structure
                                      ↓
                           AROMATIC_TOPO + HYDRO_FIELD
                                      ↓
                                   x_surface
                                      ↓
                                   aux MLP
                                      ↓
                                    z_aux

z = concat(z_DL, z_aux)
        ↓
regression head
```

auxiliary branch は
$$
x_{aux}
\rightarrow
Linear(p,64)
\rightarrow
GELU
\rightarrow
Dropout(0.2)
\rightarrow
Linear(64,32)
\rightarrow
GELU
$$
である。

SURFACE の場合 $p=35$。

---

## 9.2 AROMATIC_TOPO の簡潔な説明

AROMATIC_TOPO は、ESMFold Fv structure 上で

- F / W / Y 芳香族残基を抽出
- SASA / RASA を計算
- solvent-exposed aromatic residue を特定
- exposed aromatic residues の 3D cluster / patch を計算

して、

> **「芳香族残基が分子表面にどれだけ露出し、どのように空間集積しているか」**

を19個の descriptor にする。

詳細な全19列は Appendix A に示す。

---

## 9.3 HYDRO_FIELD の簡潔な説明

HYDRO_FIELD は分子表面に surface point を配置し、
各点 $s$ について周囲の heavy atoms から受ける hydrophobic field
$$
H(s)
=
\sum_{\|s-x_i\|\le7\mathrm{\AA}}
\pi_i
\exp(-\|s-x_i\|)
$$
を計算する。

$\pi_i$ は residue-level Fauchère–Pliska hydrophobicity を
その residue の heavy atoms に割り当てた値である。

その後、

- surface mean
- upper quantiles
- high-H surface area
- high-H patch number / largest patch
- CDR-specific field

などを16列に要約する。

つまり HYDRO_FIELD は

> **分子表面全体の連続的な疎水性 landscape**

を表す。

詳細は Appendix B に示す。

---

# 10. Transformer framework の今後の拡張性

ここが本 framework の大きな利点である。

---

# 10.1 residue representation の交換

Transformer architecture を固定したまま、
$$
\text{Scratch}
\rightarrow
\text{AbLang2}
\rightarrow
\text{ESM-2}
\rightarrow
\text{structure-aware PLM}
$$
と content representation を交換できる。

これにより、

> **representation の能力と architecture の能力を分離して比較できる**

---

# 10.2 残基間距離を attention に入れる

通常の attention score は
$$
s_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d_h}}
$$
である。

構造情報を入れる場合、
$$
s_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d_h}}
+
b(d_{ij})
$$
とできる。

ここで $d_{ij}$ は Cα distance など。

本研究でも H/L cross-attention に
RBF distance bias を入れる ARCH-6G を実装した。

今回の結果では、
学習後に geometry bias を zero 化しても prediction がほぼ変わらず、
この具体的な距離表現は実質的には利用されなかった。

したがって、

> **「距離情報が有効だった」わけではない**

が、

> **attention score 自体へ residue-pair information を注入できる framework 上の拡張点がある**

ことは確認できた。

---

# 10.3 Transformer を Graph Neural Network / Graph Transformer 的に使う

Transformer attention は、見方を変えると

> **全 token 間に edge がある complete graph 上の message passing**

と解釈できる。

通常:
$$
\alpha_{ij}
=
\mathrm{softmax}_j
\left(
\frac{q_i^\top k_j}{\sqrt d}
\right)
$$
$$
x'_i
=
x_i
+
\sum_j
\alpha_{ij}W_Vx_j
$$
Graph Transformer 的にするなら、
通信先を neighbor set $N(i)$ に制限し、
$$
\alpha_{ij}
=
\mathrm{softmax}_{j\in N(i)}
\left[
\frac{q_i^\top k_j}{\sqrt d}
+
f(e_{ij})
\right]
$$
$$
x'_i
=
x_i
+
\sum_{j\in N(i)}
\alpha_{ij}W_Vx_j
$$
とすればよい。

ここで edge feature $e_{ij}$ に

- Cα distance
- Cβ distance
- sequence separation
- same-chain / cross-chain flag
- predicted contact
- H/L interface flag
- orientation
- electrostatic relation

などを入れられる。

---

## 10.4 抗体を graph として表現する例

node:

```text
amino-acid residue
```

node feature:

```text
Scratch or PLM embedding
+ chain
+ IMGT
+ CDR/FR
+ RASA
+ charge
+ local hydrophobicity
```

edge:

```text
sequence-neighbor edge
same-chain spatial-contact edge
H-L interface edge
```

edge feature:

```text
distance
sequence separation
edge type
orientation
contact probability
```

REG token は global node として残せる。

```text
                     REG_H / REG_L
                      /   |   \
                     /    |    \
           Heavy residue graph
                 \      /
                  \    /
              H-L interface
                  /    \
                 /      \
           Light residue graph
```

これにより、

- local residue-residue message passing
- H/L interface communication
- global antibody-level pooling

を一つのモデルで扱える。

---

# 10.5 現在のARCH群はgraph topology設計としても解釈できる

今回の architecture は次のように再解釈できる。

### ARCH-3

```text
complete H/L graph
```

すべての valid token が相互に communication。

### ARCH-4

```text
typed / masked graph
```

residue-residue communication は H/L 間で許すが、
REG token への edge を chain-specific に制限。

### ARCH-7

```text
separate residue graphs
+
global REG nodes with cross-chain edges
```

residue-level graph は分離し、
summary node だけが opposite chain を読む。

### ARCH-6

```text
two chain graphs
+
explicit residue-level H-L cross edges
```

この意味で、今回の H/L attention 設計はすでに

> **Graph Transformer へ拡張可能な graph connectivity design**

の初歩とみなすことができる。

ただし、現時点のモデルを厳密に「GNN」と呼ぶ必要はない。
より正確には、

> **attention connectivity を graph message passing として再解釈できる**

という位置づけである。

---

# 10.6 residue-level physical features

現在 SURFACE は antibody-level late fusion で入れている。

将来的には
$$
x_i
=
x^{content}_i
+
x^{annotation}_i
+
x^{physical}_i
$$
として、

- RASA
- SASA
- local hydrophobic field
- electrostatic potential
- local packing
- flexibility
- predicted pKa

などを residue token 自体へ追加できる。

この方式なら、モデルは

> **どの residue の物理情報が prediction に重要か**

を attention を通じて学習できる可能性がある。

---

# 10.7 multi-conformer / ensemble information

現在は主に単一 predicted structure の descriptor を使っている。

将来的には、
$$
d_{ij}^{(1)}, d_{ij}^{(2)}, \dots, d_{ij}^{(K)}
$$
のような複数 conformer を用いて、

- mean distance
- distance variance
- contact probability
- exposure variability

などを edge feature にできる。

これは構造予測の単一点誤差に対する robustness を上げる可能性がある。

---

# 10.8 multispecific / multichain antibody への拡張

現 framework は H/L chain identity を明示的に扱うため、
概念的には chain type を一般化して
$$
chain \in \{H_1,L_1,H_2,L_2,\dots\}
$$
とすれば、多鎖 antibody へ拡張できる。

attention mask / cross-attention topology を変えることで、

- chain-specific communication
- selected interface only
- global REG node

を設計できる。

---

# 11. 本研究の意義

本検討から得られた重要な点は、
単に「AbLang2 が強かった」「SURFACE が強かった」ということだけではない。

より一般的には、

> **抗体Developability予測を、残基表現・抗体annotation・H/L通信・global pooling・構造物性の5つの設計軸に分解できた**

ことである。

特に Scratch model が一定の性能を示したことから、

> **抗体に適した architecture 自体が予測性能へ寄与し得る**

ことが示唆された。

これは今後、

- 新PLM
- structure-aware representation
- GNN / Graph Transformer
- explicit physics
- multi-conformer modeling

を比較するときに、
同じ共通 backbone 上で controlled experiment を行えることを意味する。

---

# 12. 現時点での推奨構成

## TmApp

第一候補群:

1. **AbLang2 + ARCH-3 joint unrestricted dual REG**
2. **AbLang2 + ARCH-7 REG-only cross-attention**

両者は内部ではほぼ同格とみなす。

Scratch は PLM より external generalization で弱いものの、
architecture study の control として重要。

---

## HIC

第一候補群:

1. **ESM-2 residue Transformer + SURFACE late fusion**
2. **Scratch residue Transformer + SURFACE late fusion**

SURFACE:
$$
\mathrm{AROMATIC\_TOPO}_{19}
+
\mathrm{HYDRO\_FIELD}_{16}
$$
F4 all-aux も有力だが、
permutation diagnostic では改善の中心は SURFACE。

---

# 13. 結論

本検討では、抗体配列の developability prediction に対して
residue-level Transformer を単なる deep regressor としてではなく、

> **抗体の構造的・生物学的単位を組み込む柔軟な representation / message-passing framework**

として用いた。

その結果、

- PLM と Scratch を同じ architecture で比較できた
- H/L interaction の設計差を検討できた
- REG token による antibody-level pooling を設計できた
- TmApp では AbLang2 + ARCH-3 / ARCH-7 が強かった
- Scratch でも一定の競争力が得られた
- HIC では SURFACE late fusion が明確に有効だった
- Cα distance bias は今回の形では有効利用されなかった
- しかし attention score / connectivity を介して構造情報や graph topology を導入できることが明確になった

今後は、この framework を Graph Transformer 型へ拡張し、
residue node と spatial/contact edge を用いた message passing を行うことで、

> **sequence representation と3D structure / physics を同じ residue graph 上で統合する**

方向が自然な発展候補である。

---

# Appendix A. AROMATIC_TOPO — 19 features の完全定義

入力は **ESMFold predicted Fv structure**。

芳香族残基:
$$
\{F,W,Y\}
$$
Residue SASA は Bio.PDB `ShrakeRupley`:

- probe radius = 1.4 Å
- n_points = 100

RASA:
$$
R_i=
\frac{SASA_i}{MaxASA_{Tien2013}(AA_i)}
$$
exposed:
$$
R_i\ge0.20
$$
strongly exposed:
$$
R_i\ge0.50
$$
aromatic patch:

- node = exposed F/W/Y
- edge = Cα distance ≤ 8 Å
- patch = connected component

local aromatic SASA:

- center = exposed aromatic residue
- neighbor = exposed aromatic residues within 10 Å
- self included
- maximum local SASA sum used

| # | column | exact definition | unit |
|---:|---|---|---|
| 1 | `aro_exposed_TYR_count` | exposed Y residue count | count |
| 2 | `aro_exposed_TRP_count` | exposed W residue count | count |
| 3 | `aro_exposed_PHE_count` | exposed F residue count | count |
| 4 | `aro_exposed_aromatic_total_count` | exposed F/W/Y total count | count |
| 5 | `aro_aromatic_exposed_SASA_total` | sum SASA over exposed aromatic residues | Å² |
| 6 | `aro_aromatic_exposed_SASA_fraction` | exposed aromatic SASA / total Fv SASA | fraction |
| 7 | `aro_strongly_exposed_aromatic_count` | aromatic residues with RASA≥0.50 | count |
| 8 | `aro_strongly_exposed_aromatic_SASA` | sum SASA over strongly exposed aromatics | Å² |
| 9 | `aro_CDR_exposed_aromatic_count` | exposed aromatics in IMGT CDR | count |
| 10 | `aro_CDR_aromatic_SASA` | exposed aromatic SASA in CDR | Å² |
| 11 | `aro_CDR_aromatic_fraction` | CDR exposed aromatic SASA / all exposed aromatic SASA | fraction |
| 12 | `aro_aromatic_patch_count` | number of connected components among exposed aromatics | count |
| 13 | `aro_largest_aromatic_patch_n_res` | maximum residue count among aromatic patches | count |
| 14 | `aro_largest_aromatic_patch_exposed_SASA` | exposed SASA sum in largest patch | Å² |
| 15 | `aro_max_local_aromatic_SASA` | maximum 10 Å local exposed-aromatic SASA sum | Å² |
| 16 | `aro_sequence_aromatic_count` | F/W/Y total count regardless of exposure | count |
| 17 | `aro_sequence_TYR_count` | Y count | count |
| 18 | `aro_sequence_TRP_count` | W count | count |
| 19 | `aro_sequence_PHE_count` | F count | count |

Empty set → 0。  
Singleton aromatic patch は size-1 connected component として count する。

---

# Appendix B. HYDRO_FIELD — 16 features の完全定義

## B.1 molecular surface

FreeSASA Lee–Richards:

- probe radius = 1.4 Å

Fibonacci surface sampling:

- density = 0.35 points / Å²
- maximum = 2000 points

## B.2 hydrophobicity assignment

各 residue $r$ の Fauchère–Pliska hydrophobicity を $\pi_r$ とする。

その residue の全 heavy atom $i$ に
$$
\pi_i=\pi_{r(i)}
$$
を割り当てる。

## B.3 hydrophobic field

surface point $s$ について
$$
H(s)
=
\sum_{i:\|s-x_i\|\le7\mathrm{\AA}}
\pi_i
\exp
\left(
-\alpha\|s-x_i\|
\right)
$$
$$
\alpha=1.0\ \mathrm{\AA}^{-1}
$$
各 atom contribution に SASA を掛けない。

## B.4 high-H patch

同じ structure 内の
$$
q_{0.8}=Q_{0.8}(H)
$$
を threshold とし、
$$
H(s)\ge q_{0.8}
$$
を high-H point とする。

high-H point 間距離 ≤ 2.0 Å で graph edge を張り、
connected component を patch とする。

largest patch は **vertex count 最大の component**。

## B.5 16 output columns

| # | column | exact definition |
|---:|---|---|
| 1 | `mean_H_surface` | area-weighted mean of H(s) |
| 2 | `q75_H_surface` | 75th percentile of H(s) |
| 3 | `q90_H_surface` | 90th percentile of H(s) |
| 4 | `q95_H_surface` | 95th percentile of H(s) |
| 5 | `max_H_surface` | maximum H(s) |
| 6 | `positive_H_area_fraction` | area fraction with H(s)>0 |
| 7 | `top10_H_mean` | mean H over top ceil(10% N) points |
| 8 | `top10_H_area_fraction` | area fraction occupied by those top 10% points |
| 9 | `high_H_patch_count` | number of q80 high-H connected components |
| 10 | `largest_high_H_patch_area_fraction` | area of max-vertex-count high-H component / total area |
| 11 | `largest_high_H_patch_n_vertices` | vertex count of largest high-H component |
| 12 | `CDR_mean_H` | mean H over CDR surface points |
| 13 | `CDR_q90_H` | q90 H over CDR surface points |
| 14 | `CDR_high_H_area_fraction` | high-H area within CDR / total CDR area |
| 15 | `n_surface_points` | number of sampled surface points |
| 16 | `phi_finite_frac` | fraction of finite APBS potential samples; QC column, not hydrophobic field itself |

---

# Appendix C. HIC auxiliary features beyond SURFACE

## C.1 SEQ_ALL

115-dimensional sequence descriptors。

含む代表量:

- H/L/combined amino-acid composition
- positive / negative / polar / Cys fractions
- pI
- GRAVY
- aromaticity
- pH 7 net-charge proxy
- H-L differences
- sequence entropy
- unique amino-acid count

---

## C.2 TITRATION_SHAPE

ESMFold Fv に PROPKA3 を適用し residue pKa を得る。

pH 4.0〜10.0、0.25刻みで
Henderson–Hasselbalch により residue charge を計算。

酸性 residue:
$$
q_i(pH)
=
-\frac{1}{1+10^{pK_{a,i}-pH}}
$$
塩基性 residue:
$$
q_i(pH)
=
+\frac{1}{1+10^{pH-pK_{a,i}}}
$$
$$
Q(pH)=\sum_i q_i(pH)
$$
を作り、

- Q(pH)
- $|dQ/dpH|$
- transition width
- switch region
- CDR charge

などを要約する。

---

## C.3 LOCAL_RASA_CDR3

Heavy CDR3 の ESM-2 residue embedding $e_i$ を
RASA $R_i$ で重み付け。
$$
w_i=\mathrm{clip}(R_i,0,1)
$$
$$
e_{CDR3}
=
\frac{\sum_iw_ie_i}
{\sum_iw_i}
$$
1280-dimensional vector を fold TRAIN のみで PCA32 へ圧縮。

今回の late-fusion permutation diagnostic では
追加寄与はほぼゼロだった。

---

# Appendix D. 実装監査済みの主要 repository component

- Transformer:
  `developability_drilldown/models/antibody_transformer/model.py`
- Late fusion:
  `developability_drilldown/models/antibody_transformer/late_fusion.py`
- H047 auxiliary feature preprocessing:
  `h047_aux_features.py`
- AbLang2 residue extraction:
  `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py`
- AROMATIC_TOPO:
  `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py`
- HYDRO_FIELD:
  `organizer_extension/feature_prospecting/common/hydro_surface.py`

主要 architecture / feature semantics は
`TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md`
にて repository implementation と照合済み。
