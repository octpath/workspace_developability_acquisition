# EXP-T048 / EXP-H047 特徴量技術解説

一次情報（canonical parquet・config・FEATURE_SETS/BLOCKS・生成コード）に基づく監査。新規実験・特徴生成は行っていない。

---

## 1. Executive summary

| 項目 | EXP-T048 | EXP-H047 |
|------|----------|----------|
| target | TmApp | HIC |
| estimator | SVR (`kernel="rbf"`) | SVR (`kernel="rbf"`) |
| feature_set | `FS_TM_BIOEMU_MPNN+FB_AL_CDR3+FB_AL2_RASA_CDR+FB_AL_CDR_ALL` | `FS_HIC_HYDRO_TITRATION+FB_ESM2_RASA_CDR3` |
| registry id | `FS_EXP-T048` | `FS_EXP-H047` |
| raw n_features | **6649** | **2728** |
| canonical parquet | `experiments/features/EXP-T048.parquet` | `experiments/features/EXP-H047.parquet` |
| fitted (fold0 / full-dev refit) | C=100, gamma=`scale`, epsilon=0.05 | C=1.0, gamma=`scale`, epsilon=0.05 |

共通前処理（`classical_features/cv_eval.py`）:

1. block ごとの train-median impute（非有限 → 列中央値、中央値が非有限なら 0）
2. `pca=true` かつ幅 > 32 の block のみ fold-local PCA32（`random_state=0`）
3. 連結後 `StandardScaler`
4. SVR(RBF)

T048 は抗体ペア PLM（AbLang2 / AbLingua）＋配列基本量＋BioEmu 構造揺らぎ＋ProteinMPNN 適合度を TmApp に投入する。H047 は Heavy ESM-2＋拡張配列記述＋ESMFold 由来の表面疎水／芳香族トポロジー／滴定曲線＋CDR3 の RASA 重み付き ESM-2 を HIC に投入する。

---

## 2. EXP-T048 全体構成

```
EXP-T048
  experiment_id: SVR_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_SVR
  Target: TmApp
  Estimator: SVR(kernel=rbf, C=100, gamma=scale, epsilon=0.05)  # primary fold0 → full-dev refit

  Feature composition (registry blocks):
    [A] FS_TM_BIOEMU_MPNN                         569
        ├─ AbLang2_HL_paired                      480  (PCA false)
        ├─ SEQ_BASIC                               78  (PCA false)
        ├─ BIOEMU_NEW_PAIRWISE                     10  (PCA false)
        └─ M1_PROTEINMPNN                           1  (PCA false)
    [B] FB_AL_CDR3                               2560  (PCA true → 32)
    [C] FB_AL2_RASA_CDR                           960  (PCA true → 32)
    [D] FB_AL_CDR_ALL                            2560  (PCA true → 32)
    ------------------------------------------------
    TOTAL raw                                    6649

  Processing:
    raw features
      ↓ per-block median impute
      ↓ PCA32 on [B][C][D] only
      ↓ StandardScaler
      ↓ SVR(RBF)
      ↓ TmApp
```

出典: `experiments/configs/EXP-T048.yaml`, `results/FEATURE_SETS.csv` (`FS_EXP-T048`, `FS_TM_BIOEMU_MPNN`), `results/experiments.csv`, parquet 実測。

---

## 3. T048 — 各 feature block の詳細

### 3.1 AbLang2 paired H+L embedding（480）

| 項目 | 実装上の事実 |
|------|----------------|
| モデル | `ablang2-paired` |
| package | `ablang2`（residue metadata では **0.2.1**） |
| 入力 | `[[heavy, light]]` を1ペアとして渡す |
| paired の意味 | パッケージのペア抗体モード。H mean ∥ L mean の 960 連結結合ではない |
| 呼び出し | `ablang(seqs, mode="seqcoding")` |
| 表現 | **sequence-level seqcoding**（residue embedding ではない） |
| layer / pooling | パッケージ内部の `seqcoding`（本リポジトリに内部式は未収録） |
| 出力次元 | **480**（列 `ablang2_0000`…`0479`） |
| 生成 | `gate_b1/scripts/06_plm_embeddings.py::embed_ablang2` → Stage2 / `round1_embeddings.npz::ablang2__HL_paired` → bundle `ablang2.parquet` |
| 正規化 | 特徴生成時の明示的 L2 正規化なし（訓練時は StandardScaler） |
| 訓練時 PCA | **なし** |

補足: residue 資産 `residue_level/ablang2/`（`AbRep.last_hidden_states`, 480/residue）は **FB_AL2_RASA_CDR 用**で、本 480-d 固定ブロックとは別経路。

### 3.2 SEQ_BASIC（78）

生成: `virtual_participant/stage1_features/scripts/run_stage1.py::build_all_feature_tables`  
ライブラリ: 標準 `Counter` のみ（ProtParam **不使用**）。接頭辞 `seqB_`。

定数:

- `HYDRO = {A,I,L,M,F,V,W,Y}`
- `AROM = {F,W,Y}`（**H を含まない**）
- `CHARGED = {K,R,D,E}`

| カテゴリ | 列 | 計算法 | 領域 |
|----------|-----|--------|------|
| length H | `seqB_h_len` | `len(H)` | Heavy |
| AAC H | `seqB_h_aa_{A…Y}` (20) | `count/len` | Heavy |
| physchem H | `hydro/arom/charged/gly/pro_frac` | 集合カウント / len | Heavy |
| length L | `seqB_l_len` | `len(L)` | Light |
| AAC L | `seqB_l_aa_*` (20) | 同上 | Light |
| physchem L | 同上 5 | 同上 | Light |
| Fv length | `total_len`, `hl_len_ratio`, `hl_len_diff` | len(H+L); len(H)/max(len(L),1); len(H)−len(L) | HL |
| AAC Fv | `seqB_c_aa_*` (20) | concat 上の fraction | HL |
| physchem Fv | `c_hydro/arom/charged_frac` | concat | HL |

**合計 26+26+26 = 78。CDR 長は含まれない**（一部 recipe 文言の「CDR-length summaries」は実装と不一致 → unresolved/ドキュメント齟齬）。

### 3.3 BioEmu NEW_PAIRWISE（10）

| 項目 | 実装上の事実 |
|------|----------------|
| モデル | SPEC: **`bioemu-v1.2`** |
| 入力 | **孤立 VH / 孤立 VL**（Fab ではない）を別々にサンプリング |
| 物理フレーム | QC 後凍結 **Nphys=8**（`BIOEMU_ISOLATED_NPHYS_DECISION.md`） |
| 原子 | **Cα のみ**（`traj.atom_slice(... name CA)`） |
| 整列 | `ca.superpose(ca, 0)` |
| pairwise | **conformer-vs-conformer** の全ペア Cα-RMSD（`md.rmsd`） |
| 集計 | ペア RMSD の **median** と **q90** |
| H/L 結合 | 各メトリクスについて `VH_*`, `VL_*`, `mean_*`, `max_*`, `absdiff_*` |
| 最終列 | 上記 × `{ca_rmsd_median, ca_rmsd_q90}` = **10** |
| 単位 | nm（mdtraj） |
| 欠損 | 生成時は物理フレーム不足で skip 可能; 訓練時は median impute |
| 実装 | `.../bioemu_isolated_reassessment/scripts/analyze_convergence.py::{pairwise_ca_rmsd, descriptors}` |
| NEW vs OLD | **NEW** = 物理 QC 後 Nphys=8 で再計算した reassessment 特徴家族。**OLD** = 生 N≈16 扱いの旧 `BIOEMU_V12_FEATURES`。アルゴリズム名ではなくパイプライン世代ラベル |

FS には `*ca_rmsd*` のみ。FLEX/CONTACT/SHAPE 家系は本実験に未使用。

### 3.4 ProteinMPNN（1）

| 項目 | 実装上の事実 |
|------|----------------|
| 実装 | `tools/ProteinMPNN` + `m1_proteinmpnn.py` |
| weights | `soluble_model_weights`；コードは `sorted(glob("*.pt"))[-1]` → 実ディレクトリでは **`v_48_030.pt`** |
| 構造 | ESMFold Fv PDB（`STRUCTURE_INPUT_CROSSWALK_v2.csv` の `esmfold_canonical_path`） |
| native-score | `_scores` = **masked average NLL**（`torch.nn.NLLLoss`） |
| 式 | `sum(loss * mask) / sum(mask)` → `float(native_score.mean())` |
| 列 | **`MPNN_native_score` のみ** |
| temperature | スコアリング経路では未使用（`augment_eps=0.0`） |

### 3.5 FB_AL_CDR3（2560 → PCA32）

| 項目 | 実装上の事実 |
|------|----------------|
| モデル | `IDEA-AI4S/AbLingua`（HF revision `4d1272df…`） |
| layer | `outputs.hidden_states[-1]` |
| tokenizer | AbLingua `BioTokenizer`（TripleAA） |
| CDR 境界 | IMGT（`cdr_sequence_index_imgt.csv` / annotations）`region == "CDR3"` |
| pooling | `pool_mean` = マスク内 **単純平均** |
| H/L | `concat(H_CDR3, L_CDR3)` = 1280+1280 |
| PCA | parquet は raw 2560；CV で PCA32 |

### 3.6 FB_AL2_RASA_CDR（960 → PCA32）

| 項目 | 実装上の事実 |
|------|----------------|
| PLM | residue AbLang2（hidden 480） |
| RASA | ESMFold Fv + Bio.PDB `ShrakeRupley(probe=1.4, n_points=100)` + **Tien2013 MaxASA**；`RASA = SASA/MaxASA` |
| CDR | `region="CDR"` → IMGT CDR1∪CDR2∪CDR3 |
| 重み式（コード通り） | `clipped = clip(nan_to_num(RASA,0), 0, 1)`；`w = clipped ** rasa_power`（`rasa_power=1.0`）；`pooled = Σ (emb·w) / Σ w`（`pool_weighted`） |
| 欠損 | 非有限 RASA は weight 0 かつ選択から除外；`Σw≤0` なら **ゼロベクトル** |
| H/L | 鎖別 pool → concat → **960** |
| registry | `cdr_g=1.0;cdr3_g=None;rasa_p=1.0` |

これは「閾値選択」でも「単純 mean(emb×RASA)」でもなく、**正規化付き RASA 重み付き平均**である。

### 3.7 FB_AL_CDR_ALL（2560 → PCA32）

CDR3 と同モデル・同層・同トークン処理。マスクは `region ∈ {CDR1,CDR2,CDR3}`。`pool_mean` → H∥L concat。RASA なし。PCA32。

---

## 4. EXP-H047 全体構成

```
EXP-H047
  experiment_id: SVR_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_SVR
  Target: HIC
  Estimator: SVR(kernel=rbf, C=1.0, gamma=scale, epsilon=0.05)  # fold0 primary

  Feature composition:
    [A] FS_HIC_HYDRO_TITRATION                    1448
        ├─ ESM2_H                                 1280  (PCA false)
        ├─ SEQ_ALL                                 115  (PCA false)
        ├─ AROMATIC_TOPO                            19  (PCA false)
        ├─ HYDRO_FIELD                              16  (PCA false)
        └─ TITRATION_SHAPE                          18  (PCA false)
    [B] FB_ESM2_RASA_CDR3                         1280  (PCA true → 32)
    ------------------------------------------------
    TOTAL raw                                    2728

  Processing:
    raw features
      ↓ per-block median impute
      ↓ PCA32 on [B] only
      ↓ StandardScaler
      ↓ SVR(RBF)
      ↓ HIC retention time
```

出典: `EXP-H047.yaml`, `FEATURE_SETS.csv` (`FS_HIC_HYDRO_TITRATION` = `ESM2_H|SEQ_ALL|AROMATIC_TOPO|HYDRO_FIELD|TITRATION_SHAPE`), parquet 実測。

---

## 5. H047 — 各 feature block の詳細

### 5.1 ESM-2 Heavy embedding（1280）

| 項目 | 実装上の事実 |
|------|----------------|
| モデル | `facebook/esm2_t33_650M_UR50D`（650M, 33 layers, hidden **1280**） |
| pooled 経路 | HF `transformers.AutoModel`（`gate_b1/.../06_plm_embeddings.py::embed_esm`） |
| Heavy only | Stage2 が HL concat の **先頭 1280** を H として切り出し → `esm2_heavy.parquet` |
| pooling | special token を除く AA token の mean（`hidden[1:-1].mean(0)`） |
| PCA | **なし** |

residue pack（FB 用）は **fair-esm** `repr_layers=[33]`。mean-pool との数値差は metadata 上 ~1e-6。

### 5.2 SEQ_ALL（115）= SEQ_BASIC（78）+ extras（37）

```
SEQ_BASIC (78)
  + pos/neg/polar/cys/pos_minus_neg (H/L/C)
  + ProtParam pI, gravy, aromaticity, net_charge_ph7 (H/L/C)
  + hl_pI_diff, hl_gravy_diff, hl_charge_diff
  + h/l entropy, h/l n_unique
  = SEQ_ALL (115)
```

- ProtParam: `Bio.SeqUtils.ProtParam.ProteinAnalysis`
- GRAVY: ProtParam（Kyte–Doolittle 系）
- **含まれない**: germline identity、CDR length、ANARCI region composition（それらは Stage1 の別家族）

### 5.3 AROMATIC_TOPO（19）

| 項目 | 実装上の事実 |
|------|----------------|
| 構造 | ESMFold Fv |
| 芳香族 | **F, W, Y のみ（H なし）** |
| SASA | Bio.PDB ShrakeRupley probe=1.4, n_points=100, residue-level |
| RASA | SASA / Tien2013 MaxASA |
| exposed | RASA ≥ **0.20**；strongly ≥ **0.50** |
| topology | 露出芳香族の **Cα** 間距離 ≤ **8 Å** で連結成分（patch） |
| local | 露出芳香族の近傍 **10 Å** 内 SASA 和の最大 |

主要列（prefix `aro_`）: exposed counts / SASA / CDR 分率 / patch 統計 / sequence QC counts（SPEC では qc だが bundle に含まれる）。

### 5.4 HYDRO_FIELD（16）— SAP ではない

実装: `common/hydro_surface.py` + FEATURE_SPEC。

表面: FreeSASA **Lee–Richards**, probe 1.4, Fibonacci SAS 点（密度 0.35/Å², max 2000）。

場の式（コード・SPEC 一致）:

\[
H(s)=\sum_{i:\ \|s-x_i\|\le 7}\ \pi_i\,\mathrm{e}^{-\alpha\|s-x_i\|},\quad \alpha=1.0\ \mathrm{Å}^{-1}
\]

- \(\pi_i\): **Fauchère–Pliska** 残基疎水性をその残基の全 heavy atom に割当
- 場自体は SASA 重みなし；表面点サンプリングは原子 SASA に比例
- `mean_H_surface` は面積重み付き平均
- high-H: 構造内 quantile **0.80**；patch link **2.0 Å**

canonical 14 + QC `n_surface_points` + copatch 由来 `phi_finite_frac` = **16**。

**STATIC-SAP（Black–Mould × RASA）とは別家族。同一視しない。**

### 5.5 TITRATION_SHAPE（18）

| 項目 | 実装上の事実 |
|------|----------------|
| pKa | **PROPKA3**（`propka==3.5.1`）構造依存 |
| 電荷 | Henderson–Hasselbalch 独立サイト |
| pH grid | 4.0 → 10.0 step **0.25** |
| acids | ASP, GLU, CYS, TYR → \(-1/(1+10^{pK_a-pH})\) |
| bases | HIS, LYS, ARG → \(+1/(1+10^{pH-pK_a})\) |
| termini | 主 Q から **N+/C- 除外**（QC 列に termini 付き Q） |
| CYS | pKa≥99（S–S sentinel）除外 |
| 形状 | Q(pH)、\|dQ/dpH\|（`np.gradient`）、遷移幅、major switch 領域数、CDR 上 Q |
| pI | **本家族にゼロ交差 pI 列はない**（配列 pI は SEQ_ALL） |

14 canonical + 4 QC = 18。

### 5.6 FB_ESM2_RASA_CDR3（1280 → PCA32）

T048 の RASA pool と同関数（`pooling.py`）。差:

- PLM = ESM-2 residue（1280）
- **Heavy only**（`chains=["H"]`; L pack なし）
- region = **CDR3 のみ**
- `rasa_power=1.0` → \(w_i=\mathrm{clip}(RASA_i,0,1)\)
- \(pooled=\sum w_i e_i / \sum w_i\)

---

## 6. Feature dimension accounting

### T048

| block | n | PCA in CV |
|-------|--:|-----------|
| AbLang2_HL_paired | 480 | false |
| SEQ_BASIC | 78 | false |
| BIOEMU_NEW_PAIRWISE | 10 | false |
| M1_PROTEINMPNN | 1 | false |
| FB_AL_CDR3 | 2560 | true → 32 |
| FB_AL2_RASA_CDR | 960 | true → 32 |
| FB_AL_CDR_ALL | 2560 | true → 32 |
| **TOTAL raw** | **6649** | |
| EXP-T048.parquet 実測 | **6649**（meta: id, split） | 一致 |

### H047

| block | n | PCA in CV |
|-------|--:|-----------|
| ESM2_H | 1280 | false |
| SEQ_ALL | 115 | false |
| AROMATIC_TOPO | 19 | false |
| HYDRO_FIELD | 16 | false |
| TITRATION_SHAPE | 18 | false |
| FB_ESM2_RASA_CDR3 | 1280 | true → 32 |
| **TOTAL raw** | **2728** | |
| EXP-H047.parquet 実測 | **2728** | 一致 |

FS_HIC_HYDRO_TITRATION = 1280+115+19+16+18 = **1448**（FEATURE_SETS と一致）。

---

## 7. 使用 tool/model 一覧

| Tool/model | Used for | Exact version/checkpoint | Input | Output used |
|------------|----------|--------------------------|-------|-------------|
| AbLang2 | T048 paired seq embedding + residue RASA-CDR pool | package 0.2.1; `ablang2-paired` | H+L sequences | seqcoding 480; residue 480-d for FB |
| AbLingua | T048 CDR3 / CDR_ALL pool | `IDEA-AI4S/AbLingua` rev `4d1272df…` | H+L sequences | last hidden → residue → mean pool 2560 |
| BioEmu | T048 NEW_PAIRWISE | `bioemu-v1.2`; Nphys=8 | isolated VH/VL | pairwise Cα RMSD summaries (10) |
| ProteinMPNN | T048 native score | soluble `v_48_030.pt` | ESMFold Fv PDB + seq | masked NLL `MPNN_native_score` |
| ESM-2 | H047 H embedding + RASA-CDR3 | `facebook/esm2_t33_650M_UR50D` | Heavy (and residue H) | 1280 mean-pool; 1280 RASA-CDR3 |
| ESMFold | structure source for MPNN/RASA/ARO/HYDRO/TITR | canonical Fv PDB paths in crosswalk | sequences | PDB coordinates |
| FreeSASA | HYDRO_FIELD surface | 2.2.1 Lee–Richards | ESMFold Fv | SAS points + areas |
| Bio.PDB ShrakeRupley | RASA / AROMATIC SASA | probe 1.4, n_points 100 | ESMFold Fv | residue SASA → RASA |
| PROPKA3 | TITRATION_SHAPE pKa | propka==3.5.1 | ESMFold Fv | side-chain pKa → HH Q(pH) |
| mdtraj | BioEmu RMSD | (env dependency) | BioEmu traj | Cα RMSD |
| Biopython ProtParam | SEQ_ALL physchem | Bio.SeqUtils.ProtParam | sequences | pI/GRAVY/aromaticity/charge |

---

## 8. 「名前」→「実際の計算」の対応表

| shorthand | 実際に計算しているもの |
|-----------|------------------------|
| SEQ_BASIC | H/L/Fv の長さ・20AA組成・疎水/芳香/荷電/G/P 分率（集合カウント）。CDR長なし |
| SEQ_ALL | SEQ_BASIC + ProtParam(pI/GRAVY等) + pos/neg/polar/Cys + H−L差分 + entropy |
| NEW_PAIRWISE | 孤立 VH/VL BioEmu 物理8フレームの **conformer間 Cα-RMSD** の median/q90 とその HL集約 |
| ProteinMPNN / M1 | ESMFold Fv 上の native 配列 masked **平均 NLL** |
| AROMATIC_TOPO | ESMFold 上 F/W/Y の露出(RASA≥0.2)カウント・SASA・Cα 8Å patch 統計 |
| HYDRO_FIELD | FreeSASA 表面上の Fauchère MLP \(H(s)=\sum\pi_i e^{-\alpha d}\) の分位・patch 要約（≠SAP） |
| TITRATION_SHAPE | PROPKA3 pKa → HH で Q(pH) 曲線を作り、値・傾き・遷移幅を要約 |
| RASA_CDR / RASA_CDR3 | `Σ (RASA^p · emb) / Σ RASA^p`（p=1）を指定領域・鎖で pool |
| AbLang2 paired | `ablang2-paired` の **seqcoding** 480-d（residue mean ではない） |
| AbLingua CDR* | 最終層 residue emb の IMGT CDR マスク平均（H∥L） |

---

## 9. T048 と H047 の科学的な違い

**実装上の事実の差**

- T048: 抗体特化 PLM（AbLang2/AbLingua）＋配列基本量＋単鎖構造 ensemble 揺らぎ＋配列–構造適合 NLL。標的は TmApp。
- H047: 汎用 ESM-2 Heavy＋拡張配列物性＋**表面**疎水場・露出芳香族トポロジー・滴定曲線＋CDR3 局所 ESM-2。標的は HIC。

**科学的解釈（意図；実装事実とは分離）**

- T048 の特徴群は、配列全体の抗体表現・局所 CDR 表現・鎖の構造揺らぎ・バックボーン適合度といった、**熱安定性（見かけの Tm）に関わる配列–構造情報**を SVR に渡す意図と読める。ただし特徴は ΔG そのものではない。
- H047 の特徴群は、**疎水性相互作用クロマトでの保持に関係しうる表面疎水・芳香族露出・電荷状態・CDR3 局所表現**を渡す意図と読める。HIC retention を aggregation と同一視しない。

---

## 10. unresolved provenance

1. **AbLang2 `seqcoding` のパッケージ内部プーリング式**（attention/mean 等）— 呼び出しは確認済みだが ablang2 内部はベンダー依存で本リポジトリ非収録。
2. **SEQ_BASIC に関する一部ドキュメント**が「CDR length summaries」と書くが、実装・parquet に CDR 長はない。
3. **bundle parquet を prefix 付きで書き出した packaging スクリプト**のチェックイン有無 — 値は生成ソースと一致するが copy/rename ステップ自体は未特定。
4. **ESM2_H の dual stack**: pooled は HF gate_b1→Stage2、residue FB は fair-esm。数値は整合するが同一呼び出しではない。
5. **HYDRO_FIELD の `phi_finite_frac`**: 疎水場公式の一部ではなく、copatch batch が同居させた静電 QC 列。
6. **BioEmu 全抗体の xtc 実体・個別 seed ログの再検証**は本監査で未再計算（SPEC/凍結 Nphys=8 は文書確認）。
7. **TITRATION_SHAPE に曲線ゼロ交差 pI は無い**（意図的。配列 pI は SEQ_ALL）。

---

## 付録: dictionary パス

- `results/EXP-T048_FEATURE_DICTIONARY.csv`（family/pattern 単位）
- `results/EXP-H047_FEATURE_DICTIONARY.csv`（family/pattern 単位）
