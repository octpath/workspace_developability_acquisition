# FeNNix Fab Micro-Dynamics — 最終レポート

UTC: 2026-09-08

## 冒頭 Q&A

1. **5 ps は Fab 1本あたり何分？** 計測ベース **約 34 分/本**（0.344 s/step × 6000 steps ≈ 0.57 h）。本番フル軌跡は未実行（ゲート失敗）。
2. **323本フル生成は現実的か？** **いいえ。** 5 ps で約 **185 h**、短縮 2 ps でも約 **93 h** ≫ 12 h ゲート → **`MICRODYNAMICS_TOO_SLOW`**。
3. **完走抗体数？** フルコホート **0/323**（タイミング打ち切り）。ベンチ ADI-45391 で 0.5 ps（500 steps）は完走・有限。
4. **TmApp は改善したか？** **未評価**（特徴表なし）。
5. **BioEmu + ProteinMPNN レシピへの上乗せ？** **未評価**。
6. **動的 aromatic/hydrophobic 露出は HIC を改善したか？** **未評価**。
7. **Primary / Shadow 方向は一致したか？** **N/A**。
8. **BASE_FIXED_ALPHA は生存したか？** **N/A**。
9. **競技判定？** スプリント打ち切りのため特徴ブロックは **COMP_DROP（未生成）**。CV で落とすのではなく **速度ゲート**で閉じた。
10. **feature_extension に追加すべきか？** **No**（フル表が無い）。静的 Fab FeNNix（v1.1）はそのまま有効。

## 結論（短）

FeNNol Langevin（`fennix-bio1S`、RTX 3090、~6400 atoms Fab）は動くが、指定の 1 fs / 1+5 ps プロトコルは競技スプリントの時間予算に入らない。規則どおり追加ハイパー探索はせずクローズ。

## 技術メモ

- JAX CUDA は env の `nvidia/*/lib` を `LD_LIBRARY_PATH` に入れないと CPU フォールバックする。
- 既存の静的/曲率 FeNNix 成果物は未改変。
- 意図した特徴定義はラベル前に `MICRODYNAMICS_FEATURE_SPEC.md` / `MICRODYNAMICS_FEATURE_FREEZE.json` に凍結（適用は速度ゲートで停止）。

## 成果物

| File | Note |
|------|------|
| `MICRODYNAMICS_PILOT_REPORT.md` | タイミング・安定性 |
| `MICRODYNAMICS_FEATURE_SPEC.md` | 意図特徴 |
| `MICRODYNAMICS_FEATURE_FREEZE.json` | `NOT_APPLIED` |
| `FENNIX_MICRODYNAMICS_FEATURE_DICTIONARY.csv` | 辞書のみ |
| `MICRODYNAMICS_SIMPLE_TVT_RESULTS.csv` | `NOT_RUN` |
| `results/PILOT_TIMING.json` | 数値根拠 |
| `FENNIX_MICRODYNAMICS_FEATURES.parquet` | **未作成** |
