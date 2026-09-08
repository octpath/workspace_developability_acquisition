# Organizer 上位モデル用特徴量バンドル

English: [`README.md`](README.md)

## これは何か

Organizer 側の canonical Simple TVT CV で上位となった **feature-level recipe** を
再現・改変しやすくするために、ターゲット値を用いずに生成された特徴量をまとめたものです
（Ridge / Lasso endgame → 先進モデル段階向け）。

## ファイル

ブロック → ファイル対応は `feature_manifest.csv` を参照。

| ファイル | 役割 |
|------|------|
| `folds.csv` | fold_primary / fold_shadow |
| `recipes.csv` | ターゲットごとの Top-3（CV 順位は **Public/Private より前**に凍結） |
| `data/base_sequences.parquet` | id, heavy, light |
| `data/*.parquet` | 特徴量ブロック |

## 推奨フォールド

- `scheme == primary` → fold_primary
- `scheme == shadow` → fold_shadow

回転: TEST=k, VAL=(k+1)%5, TRAIN=残り3折り。

## Top recipes（先進モデル用凍結; CV のみ）

ここで示す Top recipe は、**canonical Simple TVT CV のみ**で凍結した
**feature-level** recipe です。過去の予測値ブレンドや stack を含む
Organizer 全履歴の順位ではありません。

**これらの recipe の選定に Public / Private スコアは用いていません。**

歴史的な HIC ≈0.42 MAE は予測値レベルの SVR ブレンドに属し、別途
`organizer_extension/linear_model_closure/` に記録されています。

権威ある凍結定義:
`organizer_extension/linear_model_closure/ADVANCED_MODEL_FEATURE_RECIPE_FREEZE.json`
（本バンドルの `recipes.csv` にも反映）。

### TmApp

1. `TM_PARENT_ABLINGUA_CDR3__RIDGE` — RIDGE — CV Primary/Shadow **2.732072376654289 / 2.784957206877823**
   特徴量: AbLang2 の重鎖・軽鎖ペア埋め込み（タンパク質言語モデル／PLM）
   ＋ stage-1 SEQ_BASIC 配列記述子（長さ・組成・CDR 長要約）
   ＋ BioEmu 孤立 VH/VL NEW_PAIRWISE アンサンブル Cα RMSD 記述子
   ＋ ProteinMPNN 配列–構造適合ネイティブスコア記述子
   ＋ AbLingua-600M masked-mean 大域（GLOBAL）重鎖+軽鎖連結埋め込み
   ＋ AbLingua-600M CDR3 誘導残基プーリング埋め込み

2. `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` — RIDGE — CV Primary/Shadow **2.7466001170280285 / 2.822725206703513**
   特徴量: (1) と同じだが **AbLingua CDR3 誘導プーリングなし**
   （AbLang2 ＋ SEQ_BASIC ＋ BioEmu NEW_PAIRWISE ＋ ProteinMPNN ＋ AbLingua GLOBAL）

3. `TM_BASE_BIOEMU_MPNN__RIDGE` — RIDGE — CV Primary/Shadow **2.702681240871287 / 2.845936772811553**
   特徴量: AbLang2 の重鎖・軽鎖ペア埋め込み
   ＋ stage-1 SEQ_BASIC 配列記述子
   ＋ BioEmu 孤立 VH/VL NEW_PAIRWISE アンサンブル Cα RMSD 記述子
   ＋ ProteinMPNN 配列–構造適合ネイティブスコア記述子
   （AbLingua ブロックなし）

### HIC

1. `HIC_HYDRO_TITRATION__LASSO` — LASSO — CV Primary/Shadow **0.4872225781431885 / 0.4833719610538652**
   特徴量: ESM-2 重鎖のタンパク質言語モデル（PLM）埋め込み
   ＋ SEQ_ALL 配列記述子
   ＋ AROMATIC-TOPO 露出芳香族残基の構造トポロジー特徴量
   ＋ HYDRO_FIELD 疎水性場に基づく表面特徴量
   ＋ TITRATION_SHAPE 静電的滴定形状特徴量

2. `HIC_ARO_CONTINUOUS_SURFACE__LASSO` — LASSO — CV Primary/Shadow **0.4822073134231601 / 0.4898232886862832**
   特徴量: ESM-2 重鎖のタンパク質言語モデル（PLM）埋め込み
   ＋ SEQ_ALL 配列記述子
   ＋ AROMATIC-TOPO 露出芳香族残基の構造トポロジー特徴量
   ＋ CONTINUOUS_SURFACE 連続分子表面特徴量

3. `HIC_ESM2_SEQ_AROMATIC__LASSO` — LASSO — CV Primary/Shadow **0.4927303900206625 / 0.4808724742416768**
   特徴量: ESM-2 重鎖のタンパク質言語モデル（PLM）埋め込み
   ＋ SEQ_ALL 配列記述子
   ＋ AROMATIC-TOPO 露出芳香族残基の構造トポロジー特徴量
   （連続分子表面 / 疎水性場 / 滴定形状ブロックなし）

## 簡易利用例

```python
import pandas as pd
train = pd.read_csv("official_dev.csv")  # competition train table
feat = pd.read_parquet("data/ablang2.parquet")
train = train.merge(feat, on="id", how="left")
```

## Ridge / Lasso の例

- `examples/train_ridge.py`
- `examples/train_lasso.py`

## 本リリースに含まれるブロック

- AROMATIC_TOPO
- AbLang2_HL_paired
- AbLingua_CDR3
- AbLingua_HL_mean
- BIOEMU_NEW_PAIRWISE
- CONTINUOUS_SURFACE
- ESM2_H
- HYDRO_FIELD
- M1_PROTEINMPNN
- SEQ_ALL
- SEQ_BASIC
- TITRATION_SHAPE

## 重要

- scaler / PCA / 欠損補完は **学習 fold の内側**で fit すること
- モデル選定に Public / Private メタデータを使わないこと
- 特徴量ファイルはターゲット値を用いずに生成されている（target-blind）
- 一部のモデル由来ブロックには再配布条件があり得る — `feature_manifest.csv` の `license_status` を確認
- Organizer の Ridge は AbLingua ブロックに PCA32 を使う場合があるが、本バンドルは **生の埋め込み**を配布する

## Organizer CV の再現

このバンドル単独で学習 / CV できます（PLM / BioEmu / ProteinMPNN の再計算なし）:

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv
```

任意の提出ペア:

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv \
    --tm-recipe TM_PARENT_ABLINGUA_CDR3__RIDGE \
    --hic-recipe HIC_HYDRO_TITRATION__LASSO \
    --submission outputs/submission.csv
```

`dev.csv` スキーマ: `id,heavy,light,TmApp,HIC`  
`test.csv` スキーマ: `id,heavy,light`

**`solution.csv` は学習・CV に不要です。** 事後の Public / Private 評価専用です
（競技終了までは organizer 内部用）:

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv \
    --solution solution.csv
```

詳細は `RELEASE_FILE_POLICY.md` と `BUNDLE_REPRODUCTION_AUDIT.md` を参照。

## スコープの明確化

以前の会話での「Organizer Top-3」は、endgame Ridge/Lasso の **狭い**
feature-recipe 在庫内の話であり、履歴を含む全 organizer 線形モデルの総合順位では
ありません。歴史的な HIC ≈0.42 は主に **SVR 基底の予測値ブレンド**であり、
`organizer_extension/linear_model_closure/` に別途記録されています。
