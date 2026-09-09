# Participant candidate: Top-3 ensemble

このディレクトリは **候補** です。まだ `top_models_feature_bundle/` にはコピーしません。

## 前提

先にバンドル内で:

```bash
python reproduce_top_recipes.py --dev dev.csv --test test.csv --outdir outputs
```

## 使い方

```bash
python ensemble_top3.py --bundle top_models_feature_bundle \
  --target HIC --method equal_mean --subset H1+H3 --out hic_ens.csv
```

対応 method: `equal_mean`, `median3`, `convex_stack`, `ridge_stack`
