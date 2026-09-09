# Top-3 Ensemble Quickcheck

参加者配布の **Top-3 特徴レシピのみ** を候補とした、予測レベル・アンサンブル／厳格スタッキングのクイックチェックです。

`top_models_feature_bundle/` は **読み取り専用** です。成果物はすべて本ディレクトリに出力します。

## 検証済み再実行コマンド

```bash
python organizer_extension/top3_ensemble_quickcheck/run_all.py --bundle top_models_feature_bundle
```

オーガナイザー・ポストモルテム（solution あり）:

```bash
python organizer_extension/top3_ensemble_quickcheck/run_all.py --bundle top_models_feature_bundle --solution top_models_feature_bundle/solution.csv
```

## 出力

| ファイル | 内容 |
|---------|------|
| `BASE_REPRODUCTION_AUDIT.csv` | 6基本モデル再現監査 |
| `TOP3_ERROR_DIVERSITY.csv` | 残差多様性診断 |
| `EQUAL_MEAN_ALL_SUBSETS.csv` | 7部分集合の等重み平均 |
| `MEDIAN3_RESULT.csv` | 3モデル中央値 |
| `STRICT_STACKING_RESULTS.csv` | Convex / Ridge 厳格スタッキング |
| `BOOTSTRAP_DIAGNOSTIC.csv` | ペア・ブートストラップ |
| `TOP3_ENSEMBLE_SUMMARY.csv` | 要約 |
| `FINAL_ENSEMBLE_SELECTION.json` | CV選択・判定 |
| `TOP3_ENSEMBLE_REPORT_JA.md` | 日本語レポート |
| `final_predictions/` | 凍結後 Test 予測 |

## 注意

- `solution.csv` は POSTMORTEM スコア専用。部分集合・重み・alpha・手法選択には不使用。
- 歴史的 SVR/meta blend は候補に含めない（記述比較のみ）。
