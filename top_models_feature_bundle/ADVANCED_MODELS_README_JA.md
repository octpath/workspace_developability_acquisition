# 先進モデル — 参加者向けガイド

English: [`ADVANCED_MODELS_README.md`](ADVANCED_MODELS_README.md)

このスイートは、本バンドル内の凍結 Top-3 特徴レシピ／残基アセット上で
**XGBoost** と小さな **アノテーション付き Transformer** を学習します。

Public/Private ラベル（`solution.csv`）はモデル選択・ハイパラ選択に
**一切使いません**。指定された場合のみ、CV 選択の**後**に採点します
（**事後解析専用 / POSTMORTEM ONLY**）。

## 環境

```bash
cd top_models_feature_bundle

uv venv
uv pip install -r advanced_models/requirements.txt

python advanced_models/validate_environment.py
```

確認項目: Python、torch、`torch.cuda.is_available()`、GPU 名、
XGBoost 版、残基アセット、DEV/Test 各 162 件。

## 線形モデルの再現（変更なし）

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv
```

主催者向け事後採点:

```bash
python reproduce_top_recipes.py \
    --dev dev.csv \
    --test test.csv \
    --solution solution.csv
```

## XGBoost

単一レシピ例:

```bash
python advanced_models/run.py \
    --target TmApp \
    --model xgboost \
    --recipe TM_PARENT_ABLINGUA_CDR3__RIDGE \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target HIC \
    --model xgboost \
    --recipe HIC_HYDRO_TITRATION__LASSO \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

Top-3 全 6 本:

```bash
python advanced_models/run_benchmark.py \
    --stage xgboost \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

## Scratch Transformer

```bash
python advanced_models/run.py \
    --target TmApp \
    --model scratch_transformer \
    --annotation-mode full \
    --merge concat \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target TmApp \
    --model scratch_transformer \
    --annotation-mode full \
    --merge mean \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target HIC \
    --model scratch_transformer \
    --annotation-mode full \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

## 凍結 PLM Transformer

```bash
python advanced_models/run.py \
    --target TmApp \
    --model frozen_transformer \
    --annotation-mode full \
    --merge concat \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

```bash
python advanced_models/run.py \
    --target HIC \
    --model frozen_transformer \
    --annotation-mode full \
    --device cuda \
    --dev dev.csv \
    --test test.csv
```

## フルベンチマーク

```bash
python advanced_models/run_benchmark.py \
    --stage all \
    --device cuda \
    --dev dev.csv \
    --test test.csv \
    --write-submissions
```

主催者事後解析（選択には関与しません）:

```bash
python advanced_models/run_benchmark.py \
    --stage all \
    --device cuda \
    --dev dev.csv \
    --test test.csv \
    --solution solution.csv \
    --write-submissions
```

## スモーク（正規スコアではない）

```bash
python advanced_models/run_benchmark.py \
    --stage transformer_sequence \
    --device cuda \
    --quick \
    --dev dev.csv \
    --test test.csv
```

## 出力先

| パス | 内容 |
|------|------|
| `advanced_outputs/ADVANCED_MODEL_RESULTS.csv` | CV／任意の PP スコア |
| `advanced_outputs/ADVANCED_CV_SELECTED_CONFIGS.json` | CV 勝者 |
| `advanced_outputs/predictions/` | Test 予測 |
| `advanced_outputs/submissions/submission_best_cv.csv` | `id,TmApp,HIC` |
| `advanced_outputs/logs/` | fold キャッシュと設定ハッシュ |

提出スキーマは厳密に `id,TmApp,HIC`（162 行）。

## 残基アセットとライセンス

`residue_level/RESIDUE_ASSET_AUDIT.md` および `RELEASE_FILE_POLICY.md` を参照。
AbLingua / ESM-2 残基テンソルは **REVIEW_MODEL_OUTPUT** です。
ローカル再生成: `python scripts/build_residue_assets.py --device cuda`。
