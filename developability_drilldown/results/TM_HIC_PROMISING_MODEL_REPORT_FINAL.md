# 抗体 Developability 予測における有望モデル構成
## TmApp と HIC の現時点での整理 — residue-level Transformer と surface physicochemistry

**文書状態:** FINAL — repository implementation audit 完了（HEAD `df1da853`）。  
**目的:** これまでの `developability_drilldown` 実験を、単なる leaderboard 表ではなく、
「どの情報を、どのモデル構造で統合すると有効だったか」という観点から整理する。  
**監査:** 旧 Draft の `[CURSOR_VERIFY]` は authoritative source と照合済み。詳細は `TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md`。

---

# 1. Executive summary

現時点では、TmApp と HIC は「同じ抗体配列から予測する2つの物性」でありながら、
有効な情報設計がかなり異なる。

**TmApp** では、単純に Transformer を大きくするよりも、
抗体に適した residue-level representation と Heavy/Light (H/L) 情報の統合方法が重要だった。
とくに現在の有力クラスターは以下である。

- **EXP-T113:** AbLang2 residue representation + **ARCH-3**
  （H/L joint self-attention + unrestricted dual REG）+ mean merge
- **EXP-T121:** AbLang2 residue representation + **ARCH-7**
  （H/L は基本別処理し、最終 summary token だけが相手鎖残基を cross-attend）+ mean merge

両者の内部 OOF TEST mean MAE はほぼ同等で、T113 ≈ 3.139、T121 ≈ 3.138。
一方、T124–T129 で depth/width を増しても改善しなかったため、
「モデル容量不足」より **表現と inductive bias の選択**が重要と考えるのが妥当である。

**HIC** では事情が異なる。Residue Transformer の architecture を広く探索しても
内部 MAE は概ね 0.50 前後であったのに対し、H047 由来の明示的な表面物性を late fusion すると
0.47 前後まで改善した。特に有効だった中心情報は

> **SURFACE = AROMATIC_TOPO + HYDRO_FIELD**

である。

これは、「どのアミノ酸があるか」だけでなく、
**芳香族残基が分子表面にどれだけ露出し、どのような空間的 patch を作るか**、
および **分子表面上の連続的な疎水性ポテンシャルがどう分布するか**
を数値化した特徴量である。

HIC の late-fusion 実験では、SURFACE block を permute すると held-out MAE が大きく悪化したため、
単に別 run の学習軌道が変わっただけでなく、**学習済みモデルが SURFACE 情報を実際に利用している**
ことが支持された。これは、Cα geometry bias を zero 化しても予測がほぼ変わらなかった
geometry 系実験とは対照的である。

したがって、現時点での要約は次のとおりである。

> **TmApp:** 強い residue representation（特に AbLang2）と、H/L を適切に統合する小型 Transformer。  
> **HIC:** sequence/residue representation に加え、明示的な surface hydrophobicity / aromatic topology を融合するモデル。

---

# 2. 共通用語

## 2.1 residue-level representation

各アミノ酸残基 \(i\) に対してベクトル

\[
\mathbf{x}_i \in \mathbb{R}^{d_\mathrm{input}}
\]

を持たせる表現である。

PLM を使う場合、\(\mathbf{x}_i\) は pretrained protein language model の hidden representation。
Scratch の場合は amino-acid identity を learned embedding に変換し、そこへ position / chain /
IMGT position / region annotation を組み合わせる。

本プロジェクトでは、PLM の「抗体全体を1ベクトルに mean pool した fixed-length feature」だけを
回帰に渡す方式とは別に、**残基ごとのベクトル列を保持したまま小型 Transformer に入力する**
方式を広く検討した。

## 2.2 H/L interaction

Heavy chain (H) と Light chain (L) の残基表現を、モデル内部で直接相互作用させることを指す。

代表的には以下の2系統がある。

1. **joint self-attention:** H と L を1つの token sequence に置き、
   同じ self-attention 内で H↔L を許す。
2. **cross-attention:** H と L をいったん別々に表現した後、
   H query が L key/value を読む、またはその逆を行う。

## 2.3 REG token

通常の Transformer の `[CLS]` に近い「回帰用 summary token」である。
実データのアミノ酸残基ではなく、学習可能なベクトルを sequence に追加し、
attention を通じて残基情報を集約させる。

Heavy/Light を別々に要約する構成では

\[
\mathrm{REG}_H,\qquad \mathrm{REG}_L
\]

の2個を用いる。

Dual REG の最終統合には主に

\[
\mathrm{CONCAT} =
[\mathbf{r}_H;\mathbf{r}_L]
\]

と

\[
\mathrm{MEAN} =
\frac{\mathbf{r}_H+\mathbf{r}_L}{2}
\]

を試した。本稿では merge 自体の詳細比較は主題としないが、
現在の TmApp 有力モデル T113/T121 はいずれも MEAN である。

## 2.4 self-attention と cross-attention

1 head の scaled dot-product attention を簡略化して書くと、

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)
\]

\[
Y=AV
\]

である。\(M\) は padding や許可/禁止する token pair を表す mask。

**self-attention** では Q/K/V が同じ token 集合から作られる。

**cross-attention** では、たとえば Heavy が Light を読む場合、

\[
Q_H = H W_Q,\qquad
K_L = L W_K,\qquad
V_L = L W_V
\]

\[
C_{H\leftarrow L}
=
\mathrm{softmax}
\left(
\frac{Q_H K_L^\top}{\sqrt{d_h}} + M_{HL}
\right)
V_L
\]

となる。

multi-head attention ではこの計算を複数 head で行い、head 出力を concat して output projection する。

---

# 3. TmApp — 現時点での有望構造

## 3.1 結論

TmApp では、**AbLang2 residue representation + 小型 Transformer** が最も有望である。
現在は T113 と T121 を「ほぼ同格の有力クラスター」とみなすのが妥当である。

| Experiment | Representation | Core architecture | OOF TEST P | OOF TEST S | TEST mean | Public | Private | Overall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| EXP-T113 | AbLang2 | ARCH-3 joint unrestricted dual REG | 2.995 | 3.283 | 3.139 | 3.105 | 2.999 | 3.052 |
| EXP-T121 | AbLang2 | ARCH-7 REG-only cross-attention | 3.023 | 3.252 | 3.138 | 3.247 | 3.194 | 3.221 |

※ `cv_*` は現在の registry では V3 protocol の OOF TEST を記録している。

T124–T129 で d_model / heads / FFN / layer 数を増しても baseline を超えなかった。
したがって、これ以上の単純な capacity expansion は優先度が低い。

---

## 3.2 AbLang2 residue representation

現在の T113/T121 residue path で使う AbLang2 は次のとおり（historical 480-d `seqcoding` とは**非同義**）。

| 項目 | 実装 |
|---|---|
| package | `ablang2==0.2.1` |
| checkpoint | `ablang2-paired`（`random_init=False`） |
| 抽出 API | `AbLang.AbRep(tokens).last_hidden_states` から AA index のみ slice |
| H/L 呼び出し | **別々に encode**: Heavy=`[heavy,""]`、Light=`["",light]`（相手を空にし pre-contextualize しない） |
| special tokens | `<`,`>`,`|` を index 選択で除外（`EXACT_AA_1TO1_SPECIAL_STRIPPED`） |
| raw dim | **480**（残基ごと） |
| dtype | cache float16 → load float32 |
| cache | `top_models_feature_bundle/residue_level/ablang2/{heavy,light}_embeddings.npy` |
| 学習時 | 埋め込みは凍結キャッシュ；`nn.Linear(480→128)` で投影（trainable） |

したがって downstream Transformer で H/L interaction を入れる実験には意味がある。
PLM 出力時点ですでに H が L を見ているわけではない。

**出典:** `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py`；
`residue_level/ablang2/metadata.json`；`models/antibody_transformer/model.py`（`plm_proj`）。

---

# 4. TmApp 有力構造 A — ARCH-3: joint unrestricted dual REG

## 4.1 概念

ARCH-3 では Heavy と Light を1本の token sequence として扱い、
\(\mathrm{REG}_H\) と \(\mathrm{REG}_L\) の2個の learned summary token を含める。

概念的には

\[
[
\mathrm{REG}_H,
H_1,\ldots,H_n,
\mathrm{REG}_L,
L_1,\ldots,L_m
]
\]

を1回の Transformer encoder に入力する。

**unrestricted** の意味は、padding mask 以外には H/L communication を制限する
chain-specific attention mask を置かず、有効 token 同士が互いを attention 可能であること。

したがって理論上、

- \(H_i\rightarrow H_j\)
- \(H_i\rightarrow L_j\)
- \(L_i\rightarrow H_j\)
- \(L_i\rightarrow L_j\)
- \(\mathrm{REG}_H\rightarrow H/L\)
- \(\mathrm{REG}_L\rightarrow H/L\)
- \(\mathrm{REG}_H\leftrightarrow\mathrm{REG}_L\)

が可能である。

## 4.2 attention 計算

各 layer の self-attention では、全 token matrix を \(X\) とすると

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
A=\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M_\mathrm{padding}
\right)
\]

\[
X_\mathrm{attn}=AV
\]

となる。ここでは H/L を分離する block mask は使わない、というのが ARCH-3 の本質である。

最終 layer 後に

\[
\mathbf{r}_H=X[\mathrm{REG}_H],\qquad
\mathbf{r}_L=X[\mathrm{REG}_L]
\]

を取り出し、T113 では

\[
\mathbf{z}=\frac{\mathbf{r}_H+\mathbf{r}_L}{2}
\]

を回帰 head に渡す。


### 実装確定事項（ARCH-3 / EXP-T113）

| 項目 | 値 |
|---|---|
| class / path | `AnnotatedTransformer.encode_joint_hl_dual_reg`（`chain_specific_reg=False`）；`models/antibody_transformer/model.py` |
| token 順 | `[REG_H, H_1..H_n, REG_L, L_1..L_m]` |
| REG init | `nn.Parameter` → `normal_(std=0.02)`；chain embedding のみ（IMGT/region/pos なし） |
| 残基 embedding 加算順 | `content(=plm_proj) + pos + chain` → `+imgt` → `+region` |
| attention | `attn_mask=None`；**`src_key_padding_mask` のみ**（「padding 以外 unrestricted」は正確） |
| Pre-LN | `norm_first=True` |
| d_model / layers / heads / FFN / dropout | 128 / 2 / 4 / 256 / 0.2 |
| regression head | `Linear(128→128)→GELU→Dropout(0.2)→Linear(128→1)` |
| trainable params（実ラン） | 382849（YAML preregistered 373633；語彙長差 +9216） |


## 4.3 なぜ有望か

ARCH-3 の強みは、H/L の相互作用を最も直接的に許しつつ、
summary capacity は Heavy/Light で2個保持する点にある。

Scratch でも joint 系が separate baseline を大きく回復させたため、
「抗体全体の property prediction では H/L を別々の独立物として扱うだけでは不十分な場合がある」
という経験的証拠とも整合する。

ただし、「joint attention が物理的な VH–VL contact を直接学習した」とは言えない。
入力は一次配列由来 representation であり、attention weight が物理的 interaction energyを意味するわけではない。

---

# 5. TmApp 有力構造 B — ARCH-7: REG-only cross-attention

## 5.1 概念

ARCH-7 は、ARCH-3 より強い inductive bias を持つ。

基本の H/L residue encoding は分離して行う。
つまり Heavy residues と Light residues を最初から全面的に混ぜない。

その後、**抗体全体を要約する REG token だけが相手鎖の残基列を読む**。

概念的には

\[
H \xrightarrow{\text{separate encoder}} 
(\mathbf{r}_H,\mathbf{H})
\]

\[
L \xrightarrow{\text{same/shared encoder}}
(\mathbf{r}_L,\mathbf{L})
\]

の後、

\[
\Delta\mathbf{r}_H
=
\mathrm{MHA}
(
Q=\mathbf{r}_H,\,
K=\mathbf{L},\,
V=\mathbf{L}
)
\]

\[
\Delta\mathbf{r}_L
=
\mathrm{MHA}
(
Q=\mathbf{r}_L,\,
K=\mathbf{H},\,
V=\mathbf{H}
)
\]

を計算し、

\[
\mathbf{r}'_H
=
\mathbf{r}_H+\Delta\mathbf{r}_H
\]

\[
\mathbf{r}'_L
=
\mathbf{r}_L+\Delta\mathbf{r}_L
\]

としてから T121 では

\[
\mathbf{z}
=
\frac{\mathbf{r}'_H+\mathbf{r}'_L}{2}
\]

を回帰 head に渡す、というのが科学的な設計意図である。

### 実装確定式（ARCH-7 / EXP-T121）— コード通り

1. 共有 `self.encoder` で H/L を別々に full encode（REG は各鎖先頭）。
2. その**後に一度だけ** REG-only cross-attention（`nn.MultiheadAttention`, `batch_first=True`,
   heads=4, dropout=0.2；**H→L と L→H で同一 module**）。
3. K/V は相手鎖 residue のみ（REG は `[:,1:]` で除外）。mask は `key_padding_mask` のみ。
4. cross 後に LayerNorm / FFN / gate **なし**。

Exact residual:

```
delta_h = MHA(reg_h, l_res, l_res; key_padding_mask=~ml)
delta_l = MHA(reg_l, h_res, h_res; key_padding_mask=~mh)
reg_h' = reg_h + delta_h
reg_l' = reg_l + delta_l
z = (reg_h' + reg_l') / 2
```

depth 増加（T127/T129）は encoder self-attention 層のみ。cross stage は増やさない。

**出典:** `encode_reg_only_cross_attention`（`model.py` L779–829）。実ラン params ≈ 448897。

## 5.2 1 query × 相手鎖全残基という計算

\(\mathbf{r}_H\) が1 token、Light が \(m\) residues の場合、
1 head の cross-attention score は

\[
s_j
=
\frac{
(\mathbf{r}_H W_Q)
(\mathbf{L}_jW_K)^\top
}
{\sqrt{d_h}}
\]

\[
a_j
=
\frac{\exp(s_j)}
{\sum_{k=1}^{m}\exp(s_k)}
\]

\[
\Delta\mathbf{r}_H
=
\sum_{j=1}^{m}
a_j
(\mathbf{L}_jW_V)
\]

となる。

つまり REG_H は「Light の全 residue を同じように平均する」のではなく、
query-dependent な重み \(a_j\) を学習し、必要な Light residue を選択的に summary へ取り込める。

REG_L 側も同様に Heavy residues を読む。

このため ARCH-7 は単なる

\[
\mathrm{MLP}([\mathrm{REG}_H;\mathrm{REG}_L])
\]

とは異なる。MLP はすでに圧縮済みの2つの summary しか見ないのに対し、
ARCH-7 の REG_H は Light の**残基レベル情報に再アクセス**できる。

## 5.3 なぜ有望か

T121 は内部 TEST mean で T113 とほぼ同等の最上位。
AbLingua でも ARCH-7 MEAN (T087) は強かった。

したがって、

> 各鎖内部の表現は独立に形成しつつ、最終的な antibody-level summary を作る段階で相手鎖を見る

という inductive bias は、少量データに対して合理的な候補である。

（上記「実装確定式」で監査完了。）

---

# 6. 参考: ARCH-6 ungated residue cross-attention

ARCH-6 は現時点で TmApp の最終本命ではないため簡潔に記すが、
H/L interaction の意味を理解するうえで重要である。

ARCH-6 は、H/L を別々に self-attend した途中で**全 residue 同士の cross-attention**を挟む。

Heavy update:

\[
C_H=
\mathrm{MHA}
(Q=H,\ K=L,\ V=L)
\]

\[
H' = H + C_H
\]

Light update:

\[
C_L=
\mathrm{MHA}
(Q=L,\ K=H,\ V=H)
\]

\[
L' = L + C_L
\]

**ungated** とは、この residual branch に

\[
g_H,\ g_L
\]

のような trainable scalar gate を掛けないことを意味する。

旧 gated ARCH-5 では

\[
H'=H+g_H C_H,\qquad
L'=L+g_L C_L
\]

で、\(g_H,g_L\) をゼロ初期化した。
実際、T079 では学習後も gate が非常に小さく、
gate=0 inference でも予測がほぼ変わらなかった。

ARCH-6 はこの最適化ボトルネックを除去し、cross-attention branch を最初から有効にする。
ただし TmApp の最終最良は ARCH-6 ではなく T113/T121 クラスターだった。

---

# 7. TmApp に関する現時点の結論

1. **AbLang2 residue representation が現在最も有望。**
2. **ARCH-3 joint unrestricted dual REG** と **ARCH-7 REG-only cross-attention** が最有力。
3. d_model=128、2-layer 程度から depth/width を増しても改善せず、単純 capacity expansion は終了候補。
4. Scratch でも joint 系が強く、H/L 統合 architecture の重要性は PLM 専用の現象ではない。
5. Cα distance RBF geometry bias は zero-ablation で予測依存がほぼ消えず、現方式は優先度低。
6. 「attention が物理的相互作用そのものを学習している」とは解釈しない。

---

# 8. HIC — 現時点での有望構成

## 8.1 結論

HIC では、Transformer topology の細かな差よりも、
**明示的な molecular-surface information を residue model に追加すること**が重要だった。

代表的な有望モデル:

| Experiment | DL backbone | Added features | OOF TEST mean | Public | Private | Overall |
|---|---|---|---:|---:|---:|---:|
| EXP-H086 | ESM-2, ARCH-4, H+L | SURFACE | 0.486 | 0.402 | 0.411 | 0.406 |
| EXP-H090 | Scratch, ARCH-2, H+L | SURFACE | 0.472 | 0.406 | 0.413 | 0.409 |
| EXP-H089 | ESM-2, ARCH-4 | F4 all auxiliary | 0.470 | 0.402 | 0.439 | 0.421 |
| EXP-H085 | ESM-2 Heavy-only | F4 all auxiliary | 0.481 | 0.405 | 0.428 | 0.417 |

比較対象の historical H047 は Overall ≈0.427。
したがって、late-fusion 系の複数モデルが少なくともこの external split 上では H047 と同等以上の
有望帯へ入った。

ただし多数のモデルを試した後の external best を未知データ性能の不偏推定とはみなさない。

---

# 9. SURFACE とは何か

本稿でいう **SURFACE** は抽象語ではなく、次の固定 feature bundle である。

\[
\boxed{
\mathrm{SURFACE}
=
\mathrm{AROMATIC\_TOPO}_{19}
+
\mathrm{HYDRO\_FIELD}_{16}
}
\]

合計 35 raw features。

2つは似ているようで、数値化の発想が異なる。

- **AROMATIC_TOPO:** F/W/Y という芳香族残基を discrete objects として扱い、
  露出量と patch topology を数える。
- **HYDRO_FIELD:** 分子表面の各点に連続値の hydrophobic field \(H(s)\) を定義し、
  その分布・high-field patch を要約する。

つまり前者は「**露出した芳香族残基の空間配置**」、
後者は「**表面全体の連続的な疎水性 landscape**」を表す。

---

# 10. AROMATIC_TOPO — 具体的アルゴリズム

## 10.1 入力構造

**ESMFold で予測した Fv structure** を使用する。

## 10.2 芳香族残基の定義

対象は厳密に

\[
\{F,W,Y\}
\]

のみ。

Histidine (H) はこの family の aromatic residue には含めない。

## 10.3 residue SASA

Bio.PDB の `ShrakeRupley` を使用。

- probe radius = **1.4 Å**
- `n_points = 100`
- residue-level SASA を計算

残基 \(i\) の SASA を \(S_i\) とする。

## 10.4 RASA

Tien2013 の residue-specific MaxASA \(S^{\max}_{a_i}\) を用い、

\[
R_i=\frac{S_i}{S^{\max}_{a_i}}
\]

を計算する。

## 10.5 exposed / strongly exposed

芳香族残基 \(i\) について

\[
R_i\ge 0.20
\]

なら **exposed aromatic residue**。

\[
R_i\ge 0.50
\]

なら **strongly exposed aromatic residue**。

## 10.6 aromatic patch graph

exposed aromatic residues を node とする graph \(G\) を作る。

2つの node \(i,j\) の Cα 座標を
\(\mathbf{c}_i,\mathbf{c}_j\) として、

\[
\|\mathbf{c}_i-\mathbf{c}_j\|_2 \le 8~\text{Å}
\]

なら edge を張る。

この graph の**連結成分 (connected component)** を aromatic patch と定義する。

したがって「patch」は sequence 上で連続している必要はなく、
3D Cα 距離で 8 Å 以下の exposed F/W/Y が鎖状に連結されれば同一 component に入る。

## 10.7 local 10 Å SASA statistic

各 exposed aromatic residue を中心に 10 Å の空間近傍を定義し、
その近傍に属する対象残基の SASA を合計する。
その局所 SASA sum の最大値を feature 化する。

近傍集合は **exposed aromatic のみ**（全 aromatic でも全残基でもない）。
各 exposed aromatic（Cαあり）i について自己を含む 10 Å 内の exposed aromatic SASA 合計 L_i を取り、
feature は max_i L_i（空なら 0）。

**出典:** `aromatic_features` L82–89；`LOCAL_R=10.0`。

## 10.8 19-dimensional output

EXP-H047 / late-fusion 列名は `aro_` 接頭辞。15 canonical + 4 QC。すべて **Fv 合算**。
aromatic={F,W,Y}。exposed: RASA≥0.20。strongly: RASA≥0.50。

| # | column name | definition | unit | scope | source |
|--:|---|---|---|---|---|
| 1 | `aro_exposed_TYR_count` | # exposed Y | count | Fv | `aromatic_features` L92 |
| 2 | `aro_exposed_TRP_count` | # exposed W | count | Fv | L93 |
| 3 | `aro_exposed_PHE_count` | # exposed F | count | Fv | L94 |
| 4 | `aro_exposed_aromatic_total_count` | # exposed F∪W∪Y | count | Fv | L95 |
| 5 | `aro_aromatic_exposed_SASA_total` | sum S over exposed arom | Å² | Fv | L63,96 |
| 6 | `aro_aromatic_exposed_SASA_fraction` | (#5) / sum_Fv S | fraction | den=**all Fv residues** | L62–63,97 |
| 7 | `aro_strongly_exposed_aromatic_count` | # aromatic RASA≥0.50 | count | Fv | L98 |
| 8 | `aro_strongly_exposed_aromatic_SASA` | sum S over strongly exposed arom | Å² | Fv | L99 |
| 9 | `aro_CDR_exposed_aromatic_count` | # exposed arom ∩ IMGT CDR | count | H+L CDR | L100 |
| 10 | `aro_CDR_aromatic_SASA` | sum S over CDR∩exposed arom | Å² | H+L CDR | L101 |
| 11 | `aro_CDR_aromatic_fraction` | (#10)/(#5) | fraction | den=**all exposed arom SASA** | L102 |
| 12 | `aro_aromatic_patch_count` | # CC; edge if Cα≤8 Å among exposed arom with CA | count | Fv | L71–73,103 |
| 13 | `aro_largest_aromatic_patch_n_res` | max \|c\|; empty→0; singleton=size-1 patch | count | Fv | L74–76,104 |
| 14 | `aro_largest_aromatic_patch_exposed_SASA` | sum S in largest CC; empty→0 | Å² | Fv | L77,105 |
| 15 | `aro_max_local_aromatic_SASA` | max local 10 Å sum (§10.7); empty→0 | Å² | exposed arom only | L82–89,106 |
| 16 | `aro_sequence_aromatic_count` | # F/W/Y (ignore exposure) | count | Fv | L107 |
| 17 | `aro_sequence_TYR_count` | # Y | count | Fv | L108 |
| 18 | `aro_sequence_TRP_count` | # W | count | Fv | L109 |
| 19 | `aro_sequence_PHE_count` | # F | count | Fv | L110 |

SASA: Bio.PDB `ShrakeRupley(probe_radius=1.4, n_points=100)`。
RASA: S / Tien2013 `MAX_ASA`。parquet 19 列はコードと exact set 一致。

**出典:** `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py`；
`common/structure_utils.py`；`AROMATIC-TOPO/FEATURE_SPEC.json`。

---

# 11. HYDRO_FIELD — 具体的アルゴリズム

## 11.1 これは SAP ではない

HYDRO_FIELD は canonical Spatial Aggregation Propensity (SAP) ではない。
Black–Mould hydrophobicity × RASA の単純な residue SAP と同一視してはいけない。

ここでは**分子表面上の連続的 hydrophobic potential**を独自に構成している。

## 11.2 molecular surface

FreeSASA の **Lee–Richards** surface を使用。

- probe radius = **1.4 Å**

原子 SASA に比例して表面点を sampling する。

surface points は Fibonacci 型 sampling を用い、

- density = **0.35 points / Å²**
- max = **2000 points**

とする。

表面点を \(\mathbf{s}\) とする。

## 11.3 原子に割り当てる hydrophobicity

各 residue \(r\) に **Fauchère–Pliska hydrophobicity** \(\pi_r\) を割り当てる。

その residue の**全 heavy atom** \(i\) に同じ residue-level value

\[
\pi_i=\pi_{r(i)}
\]

を与える。

したがって atom type ごとの hydrophobic constant ではなく、
residue type による値を heavy atoms に複製している。

## 11.4 surface hydrophobic field

表面点 \(\mathbf{s}\) における field は厳密に

\[
\boxed{
H(\mathbf{s})
=
\sum_{i:\|\mathbf{s}-\mathbf{x}_i\|\le 7\text{Å}}
\pi_i
\exp\left(
-\alpha\|\mathbf{s}-\mathbf{x}_i\|
\right)
}
\]

\[
\alpha = 1.0~\text{Å}^{-1}
\]

で計算する。

ここで

- \(\mathbf{x}_i\): heavy atom \(i\) の3D座標
- cutoff: **7 Å**
- \(\pi_i\): その atom が属する residue の Fauchère–Pliska hydrophobicity

である。

重要なのは、**field の各 atom contribution 自体には SASA を掛けない**こと。

SASA は surface point sampling の密度、すなわち「どの表面領域をどれだけ代表させるか」に使われる。

## 11.5 面積重み付き平均

surface point \(s_k\) に対応する area weight を \(a_k\) とすれば、

\[
\mathrm{mean\_H\_surface}
=
\frac{\sum_k a_k H(s_k)}
{\sum_k a_k}
\]

として surface 上の平均 hydrophobic field を求める。

## 11.6 high-hydrophobic field

同一構造内の \(H(s)\) 分布について **0.80 quantile** を threshold とする。

\[
q_{0.8}
=
Q_{0.8}(\{H(s_k)\})
\]

\[
\text{high-H point}
\iff
H(s_k)\ge q_{0.8}
\]

と定義する。

絶対閾値ではなく、各 antibody の surface field 分布に対する相対 threshold である。

## 11.7 high-H patch

high-H surface points の3D位置を用い、
点間距離が **2.0 Å** 以下なら接続する graph を構成し、
connected components を high-H patches とする。

実装は `hydro_surface.connected_components`（Union–Find、全ペア Euclidean ≤ LINK=2.0 Å）。
**最大パッチは頂点数最大の CC**（面積最大ではない）。その CC の面積和で fraction を取る。

## 11.8 16-dimensional output

14 canonical + 2 QC。EXP-H047 列名（接頭辞なし）:

| # | column name | formula / algorithm | unit | weighting | threshold | source |
|--:|---|---|---|---|---|---|
| 1 | `mean_H_surface` | area-weighted mean of H(s) | field | area | — | `hydro_summaries` L234 |
| 2 | `q75_H_surface` | quantile(H, 0.75) | field | none | 0.75 | L236 |
| 3 | `q90_H_surface` | quantile(H, 0.90) | field | none | 0.90 | L237 |
| 4 | `q95_H_surface` | quantile(H, 0.95) | field | none | 0.95 | L238 |
| 5 | `max_H_surface` | max(H) | field | none | — | L239 |
| 6 | `positive_H_area_fraction` | area(H>0)/total area | fraction | area | H>0 | L240 |
| 7 | `top10_H_mean` | mean of top ceil(0.1 N) points by H | field | none | top 10% count | L224–225,241 |
| 8 | `top10_H_area_fraction` | area of those / total | fraction | area | top 10% | L242 |
| 9 | `high_H_patch_count` | # CC of high-H points | count | none | q≥0.80; link≤2.0 Å | L243 |
| 10 | `largest_high_H_patch_area_fraction` | area(max-\|V\| CC)/total | fraction | area | same | L244 |
| 11 | `largest_high_H_patch_n_vertices` | \|V\| of that CC | count | none | same | L245 |
| 12 | `CDR_mean_H` | mean(H[cdr]) else 0 | field | none | CDR flag | L246 |
| 13 | `CDR_q90_H` | quantile(H[cdr],0.90) else 0 | field | none | 0.90 | L247 |
| 14 | `CDR_high_H_area_fraction` | area(cdr∧high)/area(cdr) else 0 | fraction | area | high∩CDR | L248–250 |
| 15 | `n_surface_points` | # surface points (≤2000) | count | — | MAX_POINTS | L250 |
| 16 | `phi_finite_frac` | mean(isfinite(φ)) on APBS DX samples（**疎水場以外の QC**；copatch 同居） | fraction | none | — | `extract_hydro_copatch_batch2.py` |

Field: FreeSASA Lee–Richards probe 1.4；Fibonacci 0.35/Å²；Fauchère–Pliska π を重原子へ複製；
寄与に SASA 非乗算；cutoff 7 Å；α=1.0。

**出典:** `organizer_extension/feature_prospecting/common/hydro_surface.py`；
`HYDRO-FIELD/FEATURE_SPEC.json`。

---

# 12. SURFACE が HIC に効いたという意味

SURFACE は単なる「疎水性アミノ酸の割合」ではない。

AROMATIC_TOPO では

1. residue が F/W/Y か
2. その residue が solvent exposed か
3. exposed aromatic が3D的に集まって patch を作るか

を区別する。

HYDRO_FIELD ではさらに、各 surface location が周囲7 Å以内の heavy atoms から
どれだけ疎水性寄与を受けるかを連続値として表現する。

したがって、同じ amino-acid composition の抗体でも、

- 芳香族残基が buried か exposed か
- exposed F/W/Y が孤立するか cluster するか
- 疎水的 residue の配置が局所的に重なって高い surface field を作るか

によって feature が変わる。

HIC は IgG の chromatographic retention time であり、
この SURFACE は IgG 全体を直接計算したものではなく **predicted Fv** に由来する点に注意が必要である。
それでも今回の late-fusion / permutation 診断では、SURFACE 情報が再現性のある追加情報を
持つことが支持された。

---

# 13. HIC の有望 late-fusion 構成

## 13.1 基本形

有望な HIC モデルは、

\[
\text{residue-level DL backbone}
+
\text{explicit surface feature branch}
\]

の late fusion である。

概念的には

\[
\text{H/L sequence}
\rightarrow
\text{ESM-2 or Scratch residue Transformer}
\rightarrow
\mathbf{z}_{DL}
\]

と

\[
\text{predicted Fv structure}
\rightarrow
\mathrm{AROMATIC\_TOPO}
+
\mathrm{HYDRO\_FIELD}
\rightarrow
\mathbf{x}_{surface}
\rightarrow
\mathbf{z}_{aux}
\]

を別々に作り、

\[
\mathbf{z}
=
[\mathbf{z}_{DL};\mathbf{z}_{aux}]
\]

として regression head へ渡す。

## 13.2 auxiliary branch

production（`LateFusionAuxMLP`）は preregistration と一致。**モデル内 LayerNorm はない**
（標準化は TRAIN-only `StandardScaler` を外部前処理で適用）:

```
x_aux -> Linear(p,64) -> GELU -> Dropout(0.2) -> Linear(64,32) -> GELU = z_aux ∈ R^32
z = [z_DL ; z_aux]
  -> Linear(dim(z), d_model) -> GELU -> Dropout -> Linear(d_model, 1)
```

F1: p=35、z_DL∈R^128（MEAN）、concat dim=160。gate / FiLM / token injection なし。

前処理（scheme/fold ごと TRAIN のみ fit）:

- F1: median impute → StandardScaler；concat 順 = **AROMATIC_TOPO19 → HYDRO_FIELD16**
- F4: 各 block 独立に impute →（FB のみ）**PCA32** → StandardScaler → concat
  順 = ARO → HYDRO → SEQ → TITR → PCA32（計 200）。**PCA 前の StandardScaler は行わない。**
- グローバル ESM2_H は fusion bundle に含めない。

**出典:** `models/antibody_transformer/late_fusion.py`；`h047_aux_features.py`。

---

# 14. HIC 有望 backbone 1 — ESM-2 + ARCH-4 + SURFACE

EXP-H086 は

- frozen ESM-2 residue representation
- H/L joint processing
- chain-specific dual REG
- SURFACE late fusion

を組み合わせたモデル。

ARCH-4 では biological H/L residues は相互に joint attention できる一方、
REG_H と REG_L に明示的な役割分担を課す。

科学的な意図は

> residue-level では H/L context を交換させるが、
> summary token は Heavy / Light の役割を保つ

こと。

### ARCH-4 query→key permission matrix（1=allowed, 0=blocked）

実装: `build_chain_specific_dual_reg_attn_mask`。PyTorch では True=blocked。
layout `[REG_H, H…, REG_L, L…]`。

| query \ key | REG_H | H res | REG_L | L res |
|---|:-:|:-:|:-:|:-:|
| REG_H | 1 | 1 | 0 | 0 |
| H res | 1 | 1 | 0 | 1 |
| REG_L | 0 | 0 | 1 | 1 |
| L res | 0 | 1 | 1 | 1 |

REG は自鎖＋自己のみ；残基同士の H↔L は許可；他鎖 REG への残基 attention は禁止。


H086 の external Overall ≈0.406 は historical H047 ≈0.427 より良好だったが、
大量探索後の post-competition external comparison であるため、
これを未知データへの不偏な優位性推定とは扱わない。

---

# 15. HIC 有望 backbone 2 — Scratch joint single REG + SURFACE

EXP-H090 は PLM residue representation を使わず、

- learned amino-acid embedding
- position / chain / IMGT / region annotations
- H/L joint Transformer
- single REG summary
- SURFACE late fusion

を用いる。

内部 OOF TEST mean ≈0.472、external Overall ≈0.409。

重要なのは、H090 の主 backbone が Scratch であるにもかかわらず、
ESM-2 backbone + SURFACE と同等の有望帯に入ったことである。

これは HIC では

> PLM の高度化だけではなく、
> **明示的な surface chemistry 情報を追加すること自体が大きい**

ことを示唆する。

ただし Scratch + F3/F4 のように ESM2_RASA_CDR3 block を使う場合は
モデル全体を「完全 PLM-free」と呼べない。
H090 は F1=SURFACE のため、その surface block 自体は ESM2 embedding を含まない。

---

# 16. H047 と F4 の補助 feature

SURFACE が現在もっとも一貫して有効だったが、H047/F4 には他の情報もある。

## 16.1 SEQ_ALL

115-dimensional sequence descriptors。

SEQ_BASIC に加え、

- H/L/combined の正・負・極性・Cys counts/fractions
- ProtParam pI
- GRAVY
- aromaticity
- pH 7 net-charge proxy
- H/L 間の pI / GRAVY / charge difference
- sequence entropy
- number of unique residue types

などを含む。

`Bio.SeqUtils.ProtParam.ProteinAnalysis` を使う。
GRAVY は Kyte–Doolittle 系。

## 16.2 TITRATION_SHAPE

ESMFold Fv structure に PROPKA3 (`propka==3.5.1`) を適用して residue pKa を得る。

pH = 4.0, 4.25, ..., 10.0 で、独立 site Henderson–Hasselbalch により

酸性残基 ASP/GLU/CYS/TYR:

\[
q_i(pH)
=
-\frac{1}{1+10^{pK_{a,i}-pH}}
\]

塩基性残基 HIS/LYS/ARG:

\[
q_i(pH)
=
+\frac{1}{1+10^{pH-pK_{a,i}}}
\]

を計算し、

\[
Q(pH)=\sum_i q_i(pH)
\]

を作る。

さらに `np.gradient` で \(|dQ/dpH|\) を計算し、
Q(pH) の値、傾き、transition width、major-switch regions、CDR charge などを要約する。

主 canonical Q では termini を除外。
Cys で pKa≥99 の disulfide sentinel も除外。
この family 自体には zero-crossing pI 列はなく、sequence pI は SEQ_ALL 側にある。

## 16.3 LOCAL_RASA_CDR3

Heavy-chain CDR3 の ESM-2 residue embedding \(\mathbf{e}_i\) を
RASA \(R_i\) で重み付けし、

\[
w_i=\mathrm{clip}(R_i,0,1)
\]

\[
\mathbf{e}_{CDR3}
=
\frac{\sum_i w_i\mathbf{e}_i}
{\sum_i w_i}
\]

として 1280-d vector を作り、fold TRAIN のみで PCA32 へ圧縮する。

ただし最近の late-fusion block permutation では、この LOCAL_RASA block の追加寄与はほぼゼロだった。

---

# 17. HIC の permutation diagnostic

Late-fusion model が fixed surface features を「本当に使っているか」を確認するため、
held-out sample 上で auxiliary feature vector を sample 間で permutation し、
DL input は固定したまま MAE を再計算した。

\[
\Delta MAE
=
MAE_{\mathrm{permuted}}
-
MAE_{\mathrm{normal}}
\]

SURFACE/F4 では Primary でおおむね +0.13〜+0.20 程度の悪化が観測され、
feature branch が prediction に実質寄与していることが示された。

F4 の sub-block permutation では概ね

\[
\mathrm{SURFACE}
\gg
\mathrm{SEQ+TITRATION}
\gg
\mathrm{LOCAL\_RASA}\approx 0
\]

だった。

これは「別 architecture を入れた結果たまたま学習軌道が変わった」だけでは説明しにくく、
SURFACE が学習済み predictor の入力情報として使われていることを支持する。

---

# 18. TmApp と HIC の対照

| 観点 | TmApp | HIC |
|---|---|---|
| 現在の主要 representation | AbLang2 residue | ESM-2 residue または Scratch |
| 主な改善点 | H/L integration / summary architecture | explicit surface physicochemistry |
| 有望構造 | ARCH-3, ARCH-7 | DL backbone + SURFACE late fusion |
| 単純 capacity 増加 | 改善せず | 主研究軸ではない |
| Scratch | PLMにかなり接近 | SURFACE付与でPLM系と競争 |
| Cα geometry bias | 推論依存ほぼなし | 推論依存ほぼなし |
| Surface feature | 主役ではない | 明確に有効 |
| 解釈 | sequence-context / antibody-level aggregation | exposed aromatic / hydrophobic surface information |

したがって、単純化すると

> **TmApp は「どの residue representation をどう抗体全体へ統合するか」の問題が強い。**  
> **HIC は「配列表現だけでは不足する surface physicochemistry をどう与えるか」の問題が強い。**

ただしこれは現在の dataset / split / feature implementation に対する経験的結論であり、
一般の全抗体 Tm/HIC assay にそのまま普遍化しない。

---

# 19. 現時点で優先度を下げるもの

- TmApp の単純な depth/width expansion
- 現方式の Cα-distance RBF geometry bias
- HIC の broad H/L cross-attention topology sweep
- HIC LOCAL_RASA_CDR3 単独追加
- HIC における「PLMだけをさらに強くすれば解ける」という単一路線

---

# 20. 実装監査ステータス

本 FINAL では実装依存記述を repository と照合済み。
監査ログ: `TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md`。

| ID | 項目 | 結果 |
|---|---|---|
| A1 | AROMATIC_TOPO 19列 | exact |
| A2 | 10 Å local SASA 近傍 | exposed aromatic only |
| H1 | HYDRO_FIELD 16列 | exact（14 canonical + 2 QC） |
| H2 | high-H CC | Union–Find ≤2.0 Å；最大=max-|V| |
| T1 | ARCH-3 | padding-only unrestricted；exact |
| T2 | ARCH-7 | encode後・共有 MHA・ungated residual・Norm/FFNなし |
| T3 | ARCH-4 4×4 | exact |
| F1–F2 | late-fusion / preprocess | exact（aux に LayerNorm なし） |
| P1 | AbLang2 residue | exact；≠ seqcoding |

---

# 21. 最終版で採用する推奨表現

## TmApp

**第一候補群**
- AbLang2 + ARCH-3 joint unrestricted dual REG
- AbLang2 + ARCH-7 REG-only cross-attention

両者を「勝者/敗者」と強く分けず、有力クラスターとして扱う。

## HIC

**第一候補群**
- ESM-2 residue Transformer + SURFACE late fusion
- Scratch residue Transformer + SURFACE late fusion

SURFACE は必ず

\[
\mathrm{AROMATIC\_TOPO}_{19}
+
\mathrm{HYDRO\_FIELD}_{16}
\]

と展開して記述し、「surface features など」のように曖昧化しない。

F4 all-aux は secondary candidate。
その改善の中心は permutation diagnostic 上 SURFACE であり、
LOCAL_RASA の追加価値は小さい。

---

# 22. Evidence / provenance notes

- Git HEAD: `df1da85339a67599018ed0a1a902afc8fe56b2c2`
- Registry: `developability_drilldown/results/experiments.csv`
- AROMATIC_TOPO: `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py`,
  `common/structure_utils.py`, `AROMATIC-TOPO/FEATURE_SPEC.json`
- HYDRO_FIELD: `common/hydro_surface.py`, `HYDRO-FIELD/FEATURE_SPEC.json`,
  `scripts/extract_hydro_copatch_batch2.py`
- Transformers: `developability_drilldown/models/antibody_transformer/model.py`
- Late fusion: `late_fusion.py`, `h047_aux_features.py`
- AbLang2 residue: `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py`
- Configs: `EXP-T113.yaml`, `EXP-T121.yaml`, `EXP-H061.yaml`, `EXP-H086.yaml`, …
- Score convention: V3 の `cv_*` = OOF TEST（`dl_foldlocal_cosine_v3_oof_test`）

companion: `TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md`
