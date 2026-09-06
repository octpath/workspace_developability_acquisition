# Structure Marathon 報告書（日本語）

対象読者: 情報科学 / ML（抗体専門知識は仮定しない）

評価プロトコル（凍結）: Primary/Shadow CV、fold-local StandardScaler → Ridge（高次元は PCA32）、incumbent は既存 OOF。

Incumbent:
- TmApp: `TmApp__META_performance__ridge_100.0`
- HIC: `HIC__SIMPLE_blend_seq_surf_adv`

---

## 冒頭の 13 問への回答

1. **TmApp を成熟 PLM incumbent より実質改善したか？**  
   **いいえ（明確な改善なし）。** 最良は `S1_PATCH_NEIGHBORS`（Primary INC ΔMAE≈+0.027）だが 95% CI が 0 を跨ぎ、Shadow との一貫した decisive 改善ではない。

2. **HIC で露出芳香族+PLM 解を超える新規構造表現はあったか？**  
   **いいえ。** 試験した全ファミリーが incumbent 対比で非正（悪化寄り）。

3. **構造誘導 PLM プーリングは助けたか？**  
   **限定的。** TmApp でパッチ近傍マスクに微弱な正の点推定があるが、統計的に決定的でない。HIC の「強く露出芳香族」仮説は incumbent を超えず **REDUNDANT_SIGNAL**。

4. **ProteinMPNN に有用な構造条件付き信号は？**  
   **実質なし（NO_SIGNAL）。** native LL 集約は incumbent を改善せず。

5. **ESM-IF1 は？**  
   **実質なし。** TmApp で微小な正の点推定（Δ≈+0.003）のみ。CUDA では公式 util が CPU テンソルとのデバイス不一致で失敗したため **CPU で正式実行**。

6. **SaProt / ProSST は？**  
   SaProt 35M（AA+3Di）は 324/324 完了だが incumbent 改善なし。ProSST は依存関係で **DEFERRED_TECHNICAL**。

7. **生成器間不一致は？**  
   **NO_SIGNAL**（TmApp/HIC とも incumbent 非改善）。

8. **SPURS / ThermoMPNN は？**  
   ともに **DEFERRED_TECHNICAL**（セットアップ予算内で公式推論スタック未到達）。

9. **生成器ロバストだった知見は？**  
   予測的な正の知見が無いため、ABB2/Boltz2 複製の優先度は低い。不一致特徴自体も弱い。

10. **PLM と冗長だったものは？**  
    HIC 向け露出芳香族プーリング、SaProt 埋め込み、多くの表面グラフ特徴（standalone で多少見えても PLM/incumbent を超えない）。

11. **full Fab が Fv を超えて何を加えたか（FeNNix）？**  
    **未判定**（パイロット技術実行中。TmApp スコア前）。

12. **さらに追うべき構造アプローチは？**  
    FeNNix A/B/C/M の DELTA_ENV（同一座標での定数ドメイン環境効果）。それ以外の「別スカラー再計算」は停止。

13. **止めるべきは？**  
    露出芳香族 SASA の再発明、SaProt/MPNN/IF1 の広範スイープ、生成器不一致の拡張、GearNet。

---

## OBSERVATION / INTERPRETATION / HYPOTHESIS

### OBSERVATION
- P0（S1–S4）と P1（MPNN, IF1, SaProt）を target-blind 定義で完走しスコア化した。
- Incumbent を両 CV で明確に上回るファミリーは無い。
- HIC では構造ファミリーの多くが aromatic+PLM に対し冗長または悪化。

### INTERPRETATION
- 配列/PLM が既に捉えている情報と、予測構造から取れる幾何・逆折りたたみ尤度が大きく重複している。
- 「構造を別表現に写す」だけでは、現データ規模（N≈162 train）では incumbent を押し上げにくい。

### HYPOTHESIS
- TmApp に残る余地があるとすれば、**Fab 測定と整合する定数ドメイン環境（DELTA_ENV）**のような、PLM が明示的に持たない物理分解である。
- HIC は既に露出芳香族幾何を取り込んでおり、同系の構造再符号化は頭打ち。

---

## ファミリー別 verdict（要約）

| family | TmApp | HIC |
|--------|-------|-----|
| S1 pooling | WEAK_SIGNAL〜NO_SIGNAL | REDUNDANT_SIGNAL / NO_SIGNAL |
| S2 disagreement | NO_SIGNAL | NO_SIGNAL |
| S3 surface | NO_SIGNAL | NO_SIGNAL |
| S4 contact | NO_SIGNAL | NO_SIGNAL |
| M1 MPNN | NO_SIGNAL | NO_SIGNAL |
| M2 IF1 | NO_SIGNAL | NO_SIGNAL |
| M3 SaProt | NO_SIGNAL | NO_SIGNAL |
| M4 ProSST | TECHNICAL_BLOCK | TECHNICAL_BLOCK |
| T1/T2 | TECHNICAL_BLOCK | n/a |
| Fusion top3 | WEAK（単体と同程度） | NO_SIGNAL |

詳細数値: `results/STRUCTURE_FAMILY_SCORECARD.csv`

---

## 最終状態

`ORGANIZER_STRUCTURE_MARATHON_PARTIAL_COMPLETE`
