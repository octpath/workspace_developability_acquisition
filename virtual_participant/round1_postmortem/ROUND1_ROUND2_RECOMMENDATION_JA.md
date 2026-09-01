# Round 2 Recommendation

**結論:** `ROUND2_GO`

## 観察（Round 1 単回 holdout）

- **TmApp:** Primary CV（2.71）より Test（3.22）が大幅に悪化。stack の meta-level CV gain が Test では generalize しなかった。
- **HIC:** Primary CV（0.425）と Test（0.422）は整合。simple blend は Test でも conservative / PLM-only secondary を上回った。
- **HIC tail:** Test N=13（≥ 10.5372）すべて underprediction。tail bias −2.13、slope ≈ 0.20 で shrinkage 再現。

## 仮説（最大3）

1. **HIC high-tail shrinkage** — MAE 最適化モデルは rare high-HIC を系統的に underpredict。Round2 では Dev-only で quantile / robust transformation を predeclare 検証（Test label は training に使わない）。
2. **TmApp stack instability** — small-N meta learner の CV gain が Test で反転。Round2 候補: family pruning または conservative fusion への回帰を Shadow プロトコルで比較。
3. **Repeated nested CV** — Round1 は単一 frozen Test。Round2 では organizer split 以外の frozen split ロバストネス監査を追加（Round2 score は adaptive evaluation として明示）。

Round2 score は Round1 と同意味の unbiased unseen-test evidence ではない。
