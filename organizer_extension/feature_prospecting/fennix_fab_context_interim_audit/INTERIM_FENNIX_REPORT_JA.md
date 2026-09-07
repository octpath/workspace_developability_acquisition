# Interim FeNNix + Gap Closure Combination — 日本語報告

**注意:** FeNNix 本計算は未完了。本報告は暫定診断のみ。最終特徴定義・実行中ワーカーは変更していない。

## 冒頭 10 問

1. **利用可能な FeNNix Dev 完了 Abs 数は？**  
   **100**（BCM SUCCESS + C↔M 座標一致 + matched/r1）。全完了（Dev+非Dev）usable=199 / BCM完了=199。

2. **完了サブセットは Dev 全体を近似的に代表するか？**  
   **`COMPLETION_BIAS_UNCERTAIN`**（詳細: `INTERIM_FENNIX_COMPLETION_BIAS.md`）。

3. **DELTA_GEOM に暫定 TmApp 情報は？**  
   Primary ΔMAE(inc−cand)=-0.4452 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.5703 (`PROVISIONAL_NO_SIGNAL_YET`)。方向ラベルは悪化側の一致であり改善ではない。

4. **DELTA_ENV に暫定 TmApp 情報は？**  
   Primary ΔMAE(inc−cand)=-0.3297 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.4902 (`PROVISIONAL_NO_SIGNAL_YET`)。方向ラベルは悪化側の一致であり改善ではない。

5. **CONSTANT:**  
   Primary ΔMAE(inc−cand)=-0.3092 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.4608 (`PROVISIONAL_NO_SIGNAL_YET`)。方向ラベルは悪化側の一致であり改善ではない。
   **INTERFACE:**  
   Primary ΔMAE(inc−cand)=-0.0387 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.2077 (`PROVISIONAL_NO_SIGNAL_YET`)。方向ラベルは悪化側の一致であり改善ではない。
   **FULL_FAB_NORMALIZED:**  
   Primary ΔMAE(inc−cand)=-0.0641 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.0459 (`PROVISIONAL_NO_SIGNAL_YET`)。方向ラベルは悪化側の一致であり改善ではない。

6. **同一部分コホートで AbLang2 を改善する family は？**  
   明確な同方向改善は見えない（暫定）。

7. **同一部分コホートで TmApp incumbent を改善する family は？**  
    Primary ΔMAE(inc−cand)=-0.4135 (`PROVISIONAL_NO_SIGNAL_YET`), Shadow=-0.5730 (`PROVISIONAL_NO_SIGNAL_YET`)。方向ラベルは悪化側の一致であり改善ではない。 個別は結果 CSV 参照。

8. **Primary/Shadow 方向一致は？**  
   （ΔMAE>0=改善。`CONSISTENT_WORSENING` は陽性シグナルではない。）  
   STRICT_NESTED（concat）: DELTA_GEOM / DELTA_ENV / CONSTANT / INTERFACE / FULL_FAB_NORMALIZED / COMBINED_PREDECLARED → いずれも **`CONSISTENT_WORSENING`**。  
   STRICT_NESTED（INTERFACE residual）のみ **`MIXED_DIRECTION`**（OLD の residual `CONSISTENT_IMPROVEMENT` は汚染由来で破棄）。詳細: `STRICT_NESTED_INCREMENT_REPORT_JA.md`。

9. **見かけの信号は prep/force/size アーティファクトか？**  
   |ρ|>0.7 フラグ数=0。詳細は `INTERIM_FENNIX_TARGETBLIND_QC.csv`。DELTA_ENV はサイト数・F_rms・サイズとの相関を必ず確認。

10. **最終 full-Fab FeNNix が科学的に面白くなりそうか（暫定）？**  
   暫定的には弱い信号の可能性はあるが、最終コホートで消失しうる。現時点で科学的に「確定的に面白い」とは言えない。

## Gap Closure combination closure（要約）

旧 global-OOF 増分は **cross-fold contamination** あり。最終判定は STRICT（いずれも `NO_INCREMENT` / `CONSISTENT_WORSENING`）。

| family | framework | STRICT Primary Δ | STRICT Shadow Δ |
|--------|-----------|------------------|-----------------|
| CORE_DEFECT | residual | −0.158 | −0.172 |
| CORE_DEFECT | concat | −0.283 | −0.243 |
| FAB_INTERFACE | residual | −0.069 | −0.073 |
| FAB_INTERFACE | concat | −0.206 | −0.275 |
| GAP_ALL | residual | −0.196 | −0.255 |
| GAP_ALL | concat | −0.374 | −0.342 |
| HIC_SURFACE_ALL | residual | −0.066 | −0.090 |
| HIC_SURFACE_ALL | concat | −0.072 | −0.099 |

両 framework を報告（事後選択なし）。詳細: `STRICT_NESTED_INCREMENT_RESULTS.csv` / 汚染監査: `STRICT_NESTED_CONTAMINATION_AUDIT.md`。

## 折りたたみ統計
- Primary folds (usable Dev): {0.0: 23, 1.0: 22, 2.0: 16, 3.0: 22, 4.0: 17}
- Shadow folds (usable Dev): {0.0: 18, 1.0: 23, 2.0: 18, 3.0: 21, 4.0: 20}
