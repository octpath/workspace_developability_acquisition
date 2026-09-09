# Top-3 Ensemble Quickcheck Report（日本語）
参加者配布 Top-3 特徴レシピのみを候補とした、予測レベル・アンサンブル／スタッキングの厳格クイックチェック。
**BASE_REPRODUCTION = PASS**
## 1. Top-3 基本モデルと正準スコア
- `TM_PARENT_ABLINGUA_CDR3__RIDGE` (T1): Primary=2.732072, Shadow=2.784957
- `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` (T2): Primary=2.746600, Shadow=2.822725
- `TM_BASE_BIOEMU_MPNN__RIDGE` (T3): Primary=2.702681, Shadow=2.845937
- `HIC_HYDRO_TITRATION__LASSO` (H1): Primary=0.487223, Shadow=0.483372
- `HIC_ARO_CONTINUOUS_SURFACE__LASSO` (H2): Primary=0.482207, Shadow=0.489823
- `HIC_ESM2_SEQ_AROMATIC__LASSO` (H3): Primary=0.492730, Shadow=0.480872

## 2. ペア残差相関（Primary）
### TmApp
- T1/T2: residual Pearson=0.9972, Spearman=0.9956, mean|Δpred|=0.2064
- T1/T3: residual Pearson=0.9854, Spearman=0.9744, mean|Δpred|=0.4764
- T2/T3: residual Pearson=0.9921, Spearman=0.9861, mean|Δpred|=0.3479
### HIC
- H1/H2: residual Pearson=0.9723, Spearman=0.9437, mean|Δpred|=0.0994
- H1/H3: residual Pearson=0.9861, Spearman=0.9642, mean|Δpred|=0.0724
- H2/H3: residual Pearson=0.9840, Spearman=0.9703, mean|Δpred|=0.0633

ネストしたレシピ構造のため、高い残差相関は事前に期待される。

## 3. Equal-mean 全部分集合
### TmApp
- T1 (n=1): P=2.732072 S=2.784957 mean=2.758515 worst=2.784957
- T1+T3 (n=2): P=2.703905 S=2.798592 mean=2.751248 worst=2.798592
- T1+T2 (n=2): P=2.737709 S=2.800898 mean=2.769303 worst=2.800898
- T1+T2+T3 (n=3): P=2.717611 S=2.804693 mean=2.761152 worst=2.804693
- T2 (n=1): P=2.746600 S=2.822725 mean=2.784663 worst=2.822725
- T2+T3 (n=2): P=2.716040 S=2.829968 mean=2.773004 worst=2.829968
- T3 (n=1): P=2.702681 S=2.845937 mean=2.774309 worst=2.845937
### HIC
- H1+H2 (n=2): P=0.481298 S=0.478198 mean=0.479748 worst=0.481298
- H1+H2+H3 (n=3): P=0.484055 S=0.476747 mean=0.480401 worst=0.484055
- H2+H3 (n=2): P=0.484874 S=0.484960 mean=0.484917 worst=0.484960
- H1+H3 (n=2): P=0.487061 S=0.474426 mean=0.480744 worst=0.487061
- H1 (n=1): P=0.487223 S=0.483372 mean=0.485297 worst=0.487223
- H2 (n=1): P=0.482207 S=0.489823 mean=0.486015 worst=0.489823
- H3 (n=1): P=0.492730 S=0.480872 mean=0.486801 worst=0.492730

## 4. 2モデル平均は効いたか
- TmApp: best-1 worst=2.784957 (T1) vs best-2 worst=2.798592 (T1+T3); Δworst=-0.013635
- HIC: best-1 worst=0.487223 (H1) vs best-2 worst=0.481298 (H1+H2); Δworst=0.005925

## 5. 3モデル平均は効いたか
- TmApp: best-1 worst=2.784957 vs ALL3 worst=2.804693; Δworst=-0.019736
- HIC: best-1 worst=0.487223 vs ALL3 worst=0.484055; Δworst=0.003168

## 6. Median-of-three
- TmApp: P=2.727960 S=2.813849 mean=2.770904 worst=2.813849
- HIC: P=0.489608 S=0.480641 mean=0.485124 worst=0.489608

## 7. Convex MAE stacking
- TmApp: P=2.738262 S=2.810097 worst=2.810097 (Δworst vs best single=-0.025140)
- HIC: P=0.482989 S=0.482399 worst=0.482989 (Δworst vs best single=0.004234)

## 8. Ridge stacking
### RIDGE_CV_ALPHA
- TmApp: P=2.802570 S=2.875991 worst=2.875991 (Δworst=-0.091034)
- HIC: P=0.498456 S=0.494445 worst=0.498456 (Δworst=-0.011233)
### RIDGE_FIXED_ALPHA_1
- TmApp: P=2.798044 S=2.887289 worst=2.887289 (Δworst=-0.102332)
- HIC: P=0.488570 S=0.490811 worst=0.490811 (Δworst=-0.003589)

## 9. Primary / Shadow 一貫性
- TmApp: ΔPrimary=0.000000, ΔShadow=0.000000, both_improve=False
- HIC: ΔPrimary=0.005925, ΔShadow=0.005174, both_improve=True

## 10. CV選択ベスト
- TmApp: `SINGLE:T1` (SINGLE) P=2.732072 S=2.784957 worst=2.784957
- HIC: `EQUAL_MEAN:H1+H2` (EQUAL_MEAN) P=0.481298 S=0.478198 worst=0.481298

## 11. ベスト単体との差分
- TmApp: ΔP=0.000000 ΔS=0.000000 Δworst=0.000000
- HIC: ΔP=0.005925 ΔS=0.005174 Δworst=0.005925

## 12. Bootstrap 安定性診断
- TmApp/primary: meanΔ=0.000000 medianΔ=0.000000 95%CI=[0.000000,0.000000] P(Δ>0)=0.000
- TmApp/shadow: meanΔ=0.000000 medianΔ=0.000000 95%CI=[0.000000,0.000000] P(Δ>0)=0.000
- HIC/primary: meanΔ=0.005954 medianΔ=0.005848 95%CI=[-0.006649,0.018345] P(Δ>0)=0.827
- HIC/shadow: meanΔ=0.005080 medianΔ=0.005062 95%CI=[-0.008928,0.019568] P(Δ>0)=0.761

## 13. 最終構成 / 重み
```json
{
  "TmApp": {
    "type": "single",
    "model": "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "weight": 1.0
  },
  "HIC": {
    "type": "equal_mean",
    "models": [
      "HIC_HYDRO_TITRATION__LASSO",
      "HIC_ARO_CONTINUOUS_SURFACE__LASSO"
    ],
    "weights": [
      0.5,
      0.5
    ]
  }
}
```

## 14. Public / Private（POSTMORTEM ONLY）
- TmApp: Public=3.118525 Private=3.289918 Overall=3.204221 **[POSTMORTEM ONLY]**
- HIC: Public=0.473711 Private=0.468509 Overall=0.471110 **[POSTMORTEM ONLY]**

## 15. 歴史的オーガナイザー・アンサンブルとの比較（記述のみ）
- HIC historical `HIC__SIMPLE_blend_seq_surf_adv`: CV~0.4252/0.4204, Pub/Priv~0.4239/0.4222。
- 本実験の CV 選択 H1+H2 equal-mean: CV worst~0.4813、POSTMORTEM Pub/Priv~0.4737/0.4685。
- 結論: 配布 Top-3 のみでは歴史的 SVR/予測ブレンド水準（~0.42）は回収できない。小さな CV 改善（~0.006）に留まる。
- TmApp: 歴史 meta/blend はより広い候補プールに依存。Top-3 平均は cv_worst を悪化させ、増分なし。
- 歴史スコアは新ランキングに混在させない（プロトコル非互換）。

## 16. 最終判定
- TmApp: **NO_ENSEMBLE_INCREMENT**
- HIC: **ENSEMBLE_USEFUL**

## 17. 参加者バンドルへの後入れ？
- YES（candidate を organizer_extension 側に用意。バンドルへのコピーは後続）
