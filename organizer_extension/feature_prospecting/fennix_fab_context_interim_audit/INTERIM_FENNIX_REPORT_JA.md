# Interim FeNNix + Gap Closure Combination — 日本語報告

**注意:** FeNNix 本計算は未完了。本報告は暫定診断のみ。最終特徴定義・実行中ワーカーは変更していない。

## 冒頭 10 問

1. **利用可能な FeNNix Dev 完了 Abs 数は？**  
   **100**（BCM SUCCESS + C↔M 座標一致 + matched/r1）。全完了（Dev+非Dev）usable=199 / BCM完了=199。

2. **完了サブセットは Dev 全体を近似的に代表するか？**  
   **`COMPLETION_BIAS_UNCERTAIN`**（詳細: `INTERIM_FENNIX_COMPLETION_BIAS.md`）。

3. **DELTA_GEOM に暫定 TmApp 情報は？**  
   Primary ΔMAE(inc−cand)=-0.4452 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.5703 (`PROVISIONAL_NO_SIGNAL_YET`)。同方向改善=False。

4. **DELTA_ENV に暫定 TmApp 情報は？**  
   Primary ΔMAE(inc−cand)=-0.3297 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.4902 (`PROVISIONAL_NO_SIGNAL_YET`)。同方向改善=False。

5. **CONSTANT:**  
   Primary ΔMAE(inc−cand)=-0.3092 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.4608 (`PROVISIONAL_NO_SIGNAL_YET`)。同方向改善=False。
   **INTERFACE:**  
   Primary ΔMAE(inc−cand)=-0.0387 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.2077 (`PROVISIONAL_NO_SIGNAL_YET`)。同方向改善=False。
   **FULL_FAB_NORMALIZED:**  
   Primary ΔMAE(inc−cand)=-0.0641 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.0459 (`PROVISIONAL_NO_SIGNAL_YET`)。同方向改善=False。

6. **同一部分コホートで AbLang2 を改善する family は？**  
   明確な同方向改善は見えない（暫定）。

7. **同一部分コホートで TmApp incumbent を改善する family は？**  
    Primary ΔMAE(inc−cand)=-0.4135 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.5730 (`PROVISIONAL_NO_SIGNAL_YET`)。同方向改善=False。 個別は結果 CSV 参照。

8. **Primary/Shadow 方向一致は？**  
   DELTA_GEOM:一致, DELTA_ENV:一致, CONSTANT:一致, INTERFACE:一致, FULL_FAB_NORMALIZED:一致, PREP_RELAX_SENSITIVITY:一致, COMBINED_PREDECLARED:一致

9. **見かけの信号は prep/force/size アーティファクトか？**  
   |ρ|>0.7 フラグ数=0。詳細は `INTERIM_FENNIX_TARGETBLIND_QC.csv`。DELTA_ENV はサイト数・F_rms・サイズとの相関を必ず確認。

10. **最終 full-Fab FeNNix が科学的に面白くなりそうか（暫定）？**  
   暫定的には弱い信号の可能性はあるが、最終コホートで消失しうる。現時点で科学的に「確定的に面白い」とは言えない。

## Gap Closure combination closure（要約）
family           framework              
CORE_DEFECT      residual_plus_oof         -0.190696
                 ridge_concat_oof_scalar   -0.279696
FAB_INTERFACE    residual_plus_oof         -0.103625
                 ridge_concat_oof_scalar   -0.267617
GAP_ALL          residual_plus_oof         -0.263857
                 ridge_concat_oof_scalar   -0.379970
HIC_SURFACE_ALL  residual_plus_oof         -0.080561
                 ridge_concat_oof_scalar   -0.087743

両 framework を報告（事後選択なし）。詳細: `GAP_CLOSURE_COMBINATION_CLOSURE.csv`。

## 折りたたみ統計
- Primary folds (usable Dev): {0.0: 23, 1.0: 22, 2.0: 16, 3.0: 22, 4.0: 17}
- Shadow folds (usable Dev): {0.0: 18, 1.0: 23, 2.0: 18, 3.0: 21, 4.0: 20}
