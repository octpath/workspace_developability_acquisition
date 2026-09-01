# Stage 1 — 配列統計量と抗体特有情報による予測

## 1. このStageで何をしたか

Stage 0でfreezeした共通5-fold CVを固定したまま、古典的な配列記述子と抗体特有annotationを評価した。目的は、タンパク質言語モデルや立体構造特徴に進む前に、「通常の配列統計量と抗体特有の知識だけでどこまで予測できるか」を確定することである。

Stage 1最良モデルでは、median baselineに対してTmAppで約9.8%、HICで約8.6%のMAE低減が得られ、Shadow CVでも改善方向は維持された。TmAppでは単純配列組成、CDR長、germline情報など複数の特徴表現がほぼ同程度の性能に到達した。HICではRBF-SVRによる非線形モデリングが主な改善経路だった。抗体annotationを加えたモデルがPrimary最良だったが、配列特徴のみとの差は小さく、annotation追加の独立した効果はまだ確定していない。次Stageでは、PLMがこのclassical-feature baselineを明確に上回るかを検証する。

## 2. 今回使った特徴量

| 特徴量群 | 内容 | なぜ有用と考えたか |
|---|---|---|
| SEQ_BASIC | VH/VL長、アミノ酸組成、疎水性・芳香族・電荷などの単純組成 | 配列の大まかな物理化学的傾向を捉える基礎量 |
| SEQ_PHYS | pI、GRAVY、近似電荷、H/L差など | 安定性や疎水性相互作用と関係しうる量 |
| SEQ_HEAVY / SEQ_LIGHT / SEQ_COMBINED | 鎖別・結合配列のみ | 鎖ごとの予測情報の有無を切り分けるため |
| ANN_GERMLINE | V/J、κ/λ、germline identityとその派生 | 系列背景・成熟度の代理指標になりうる |
| ANN_CDR_LENGTH | 配布CDR長 | ループ長が局所構造や露出に影響しうる |
| ANN_CDR_COMPOSITION | ANARCI IMGTに基づく各CDR/FW組成 | 疎水性・芳香族・電荷の局在 |
| ANN_HCDR3 | HCDR3組成のみ | 可変性が高く、独立した情報源になりうる |
| SEQ_ALL / ANTIBODY_ALL / SEQ_PLUS_ANTIBODY | 上記の統合 | 相補的情報の重ね合わせ効果を見る |

ANARCI領域抽出はDev配列のみから再計算し、`cache/anarci_imgt_regions_dev.csv` に保存した（162件すべて成功）。カテゴリ変数は各学習fold内で出現回数の少ない水準をまとめてからone-hot化した。スケーラ等の前処理もfold外リークが起きないよう学習側だけでfitした。

## 3. 評価方法

- **Primary CV**（`../stage0_cv/cv_primary.csv`）: 特徴量比較・モデル選択・Optunaの目的関数。
- **Shadow CV**（`../stage0_cv/cv_shadow.csv`）: Primary上位最大5モデルのみ再評価し、Primary過学習を点検。
- 指標はMAE。中央値予測との差（ΔMAE）を記録。
- Foldの再探索は行っていない。Testラベルやorganizer資料は未使用。
- 上位モデル間のMAE差がごく小さい場合は、順位そのものを強く解釈しない。
- Stage 1では、全特徴量群に対して同一の非線形モデル・同一Optuna予算を割り当てる完全な総当たり比較は行っていない。まず正則化線形モデル等で特徴量群をscreeningし、有望候補をSVR / GBDT / Optunaへ進めた。したがって、最終的なモデル順位は「Stage 1で採用した探索方針の下で得られた順位」と解釈する。

## 4. TmApp結果

中央値ベースライン Primary MAE = **3.4374**  
Stage 1最良 Primary MAE = **3.1011**（相対MAE低減 ≈ **9.8%**）

| 特徴量 | モデル | Primary MAE | Shadow MAE | Δmedian |
|---|---|---:|---:|---:|
| SEQ_BASIC | SVROpt | 3.1011 | 3.1716 | -0.3364 |
| ANN_CDR_LENGTH | SVROpt | 3.1041 | 3.2183 | -0.3333 |
| SEQ_ALL | LGBMOpt | 3.1063 | 3.1885 | -0.3311 |
| SEQ_BASIC | ElasticNetOpt | 3.1182 | 3.1360 | -0.3192 |
| ANN_CDR_LENGTH | RidgeOpt | 3.1324 | 3.1451 | -0.3050 |
| ANN_CDR_LENGTH | Ridge | 3.1361 | — | -0.3013 |
| ANN_GERMLINE | Ridge | 3.1815 | — | -0.2559 |
| SEQ_COMBINED | Ridge | 3.2525 | — | -0.1850 |

**Stage1最良（Primary順位）:** `TmApp__SEQ_BASIC__SVROpt`  
Primary MAE=**3.1011**, Shadow MAE=**3.1716**, 状態=`SHADOW_CONFIRMED`

ただし上位3モデル（3.1011 / 3.1041 / 3.1063）の差は約0.005 °Cと極めて小さい。Primary順位ではSEQ_BASIC+SVRが最良だったが、**この順位そのものを強く解釈すべきではない**。線形モデルだけでも `ANN_CDR_LENGTH`（MAE 3.1361）と `ANN_GERMLINE`（3.1815）はmedian baselineを明確に上回った。高次元の `SEQ_PLUS_ANTIBODY` / `ANTIBODY_ALL` は正則化が弱いと中央値より悪化した。

## 5. HIC結果

中央値ベースライン Primary MAE = **0.5180**  
Stage 1最良 Primary MAE = **0.4732**（相対MAE低減 ≈ **8.6%**）

| 特徴量 | モデル | Primary MAE | Shadow MAE | Δmedian |
|---|---|---:|---:|---:|
| SEQ_PLUS_ANTIBODY | SVROpt | 0.4732 | 0.4788 | -0.0448 |
| SEQ_ALL | SVROpt | 0.4766 | 0.4795 | -0.0413 |
| SEQ_COMBINED | SVROpt | 0.4775 | 0.4724 | -0.0405 |
| SEQ_COMBINED | RidgeOpt | 0.5044 | 0.5235 | -0.0135 |
| SEQ_ALL | XGBOpt | 0.5072 | 0.5651 | -0.0108 |
| SEQ_COMBINED | Lasso | 0.5126 | — | -0.0053 |

**Stage1最良（Primary順位）:** `HIC__SEQ_PLUS_ANTIBODY__SVROpt`  
Primary MAE=**0.4732**, Shadow MAE=**0.4788**, 状態=`SHADOW_CONFIRMED`

線形モデルでは中央値を明確に下回る候補が少なく、最良でも `SEQ_COMBINED` + Lasso でわずかに改善（0.5126）にとどまった。RBF-SVRでは複数の配列特徴群がMAE≈0.47台に入り、改善自体はShadowでも維持された。一方、上位SVR間の差は小さく、Shadowでは `SEQ_COMBINED` が最良（0.4724）だった。XGBoostの一部はPrimaryのみ改善でShadowでは悪化（`PRIMARY_ONLY_GAIN`）した。

### HICにおける抗体annotationの追加効果（incremental value）

| 比較 | Primary MAE | Shadow MAE | 解釈 |
|---|---:|---:|---|
| SEQ_COMBINED + SVROpt | 0.4775 | 0.4724 | 結合配列の物性特徴 |
| SEQ_ALL + SVROpt | 0.4766 | 0.4795 | 配列特徴のみ |
| SEQ_PLUS_ANTIBODY + SVROpt | 0.4732 | 0.4788 | 配列 + annotation |
| SEQ_PLUS_ANTIBODY − SEQ_ALL | **-0.0034** | **-0.0007** | 追加効果は小さく、現時点では未確定 |

## 6. 何が効いたか

**TmApp**では、SEQ_BASIC、CDR長、SEQ_ALLなど異なる特徴表現がほぼ同程度のMAE≈3.10に到達した。germline系情報も単独の線形モデルでmedianを明確に上回る。したがって、一つの特徴量群だけが支配的というより、組成・抗体特有の幾何/系列背景など**複数の低次元情報源**に予測情報が含まれていると考える方が妥当である。

**HIC**では、単純な線形モデルの改善は小さく、RBF-SVRの利用が大きな改善要因だった。SEQ_COMBINED / SEQ_ALL / SEQ_PLUS_ANTIBODY のSVRはほぼ同一レンジにあり、「annotation追加が決定的」とは言えない。Stage 1から最も強く言えるのは、**単純な線形関係では捉えきれないが、低次元の配列・物性特徴を非線形に組み合わせると再現性のある改善が得られる**ことである。

## 7. HeavyとLightのどちらが効いたか

同一モデル種・同一既定ハイパーパラメータでの比較（registry再集計）:

### TmApp（Primary MAE）

| 特徴量群 | Ridge | ElasticNet | Lasso |
|---|---:|---:|---:|
| SEQ_COMBINED | 3.2525 | 3.2574 | 3.2599 |
| SEQ_HEAVY | 3.3183 | 3.3294 | 3.3310 |
| SEQ_LIGHT | 3.5541 | 3.5582 | 3.5276 |

### HIC（Primary MAE）

| 特徴量群 | Ridge | ElasticNet | Lasso |
|---|---:|---:|---:|
| SEQ_COMBINED | 0.5192 | 0.5148 | 0.5126 |
| SEQ_HEAVY | 0.5455 | 0.5340 | 0.5277 |
| SEQ_LIGHT | 0.6470 | 0.6244 | 0.5949 |

Stage 1で評価した線形モデル群では、Light-only特徴はHeavy-onlyまたはcombined特徴より一貫して低い性能を示した。ただし、最終最良モデル（SVROpt等）まで含めた全比較では探索経路が完全に揃っているわけではないため、**これをHeavy鎖の情報量が本質的に大きいことの厳密なablation証明とはみなさない**。Stage 2では同一PLM・同一pooling・同一regressorの条件で Heavy-only / Light-only / H+L を比較する。

## 8. ANARCI / germline / CDR情報は役立ったか

### TmApp

抗体特有annotation単独でも予測情報が認められた。

- `ANN_CDR_LENGTH` + Ridge ≈ **3.1361**
- `ANN_GERMLINE` + Ridge ≈ **3.1815**

いずれもmedian 3.4374より明確に良い。最終上位でもCDR長ベースのモデルがSEQ_BASICとほぼ同程度だった。

### HIC

annotation単独の線形モデルでは改善が小さかった。`SEQ_PLUS_ANTIBODY` + SVRがPrimary最良だったが、`SEQ_ALL` + SVRとの差はPrimaryで0.0034 min、Shadowで0.0007 minにすぎない。またShadowでは `SEQ_COMBINED` + SVRが最良だった。したがって、**HICではANARCI/annotationのincremental valueは現時点では明確ではない**（限定的または未確定）。

ANARCI由来の `ANN_CDR_COMPOSITION` 単独は次元が高く、線形では過学習しやすい。`ANN_HCDR3` 単独も弱い改善〜中立程度だった。

まとめ:

- **TmApp**では配布アノテーションが有力な特徴量群である。
- **HIC**では統合候補として利用価値はあるが、Stage 1だけでは追加効果が明確とは言えない。

## 9. 科学的にどう解釈できるか

断定はできないが、次の読みは結果と矛盾しない。

- **TmApp（見かけの融解温度）**: CDR長やgermline背景は、Fab構造・sequence background・affinity maturation historyなどと関係する代理指標になりうる。今回のStage 1ではHeavy由来特徴がLight-only特徴より良好な傾向を示した。ただし、この比較だけからHeavy鎖がTmAppを因果的に支配すると結論することはできない。
- **HIC（疎水性相互作用クロマトグラフィー）**: 配列上の疎水性・芳香族性・電荷特徴が、立体構造上の表面化学を間接的に反映している可能性がある。Stage 1では3D exposureを測っていない。この仮説は後続のstructure stageで SASA / RASA / surface patch 等を用いて直接検証する。

これらは相関の解釈であり、因果関係や製造可否を意味しない。相対MAE低減（TmApp約9.8%、HIC約8.6%）は本コンペのmedian baselineに対する改善であり、assay精度そのものの効果量とは解釈しない。

### 係数・単変量の参考（最良特徴量群）

探索的結果として:

- TmApp（`SEQ_BASIC`）平均|係数|上位例: `l_aa_D`, `h_aa_M`, `h_aa_E`, `l_aa_A` など組成項。単変量では `h_aa_M`（負相関）、`c_aa_Y` / `c_arom_frac`（正相関）などが相対的に大きい。
- HIC（`SEQ_PLUS_ANTIBODY`）平均|係数|上位例: `c_pI`, `heavy_j_gene_JH4`, `h_cdr1_hydro_frac`, `l_cdr1_arom_frac`。単変量では `h_cdr3_length`（Spearman≈0.35）が目立つ。

これらは探索的な相関・係数であり、相関した特徴量間の代替やsmall-Nによる不安定性があるため、**個々の特徴の因果効果とは解釈しない**。

## 10. PrimaryとShadowは一致したか

### TmApp
上位5モデルはいずれもShadowでもmedian baselineを上回り、改善方向は維持された（いずれも `SHADOW_CONFIRMED`）。最良SVRはPrimary 3.1011 → Shadow 3.1716 とやや戻るが、改善自体は残る。一方、Primaryで僅差だった候補間（3.1011 / 3.1041 / 3.1063）の順位やMAE差はShadowでは変動している。したがって、「古典特徴から再現性のある改善が得られた」とは言えるが、特定の特徴量群を一意に最良とみなすべきではない。

### HIC
- SVR系3モデル: `SHADOW_CONFIRMED`（Primary≈0.47–0.48、Shadowも同程度）
- ただしShadow最良は `SEQ_COMBINED` + SVROpt（0.4724）であり、Primary最良のannotation付きモデルとの差は実質的にない。
- `SEQ_COMBINED__RidgeOpt`, `SEQ_ALL__XGBOpt`: `PRIMARY_ONLY_GAIN`（Shadowでは中央値より悪化）

HICでは木モデルのPrimary改善を過信しない方がよい。Stage2以降もSVR級の安定候補を基準にする。

## 11. Stage 1で分かったこと

1. 古典的な配列特徴だけでも、TmApp/HIC双方で中央値予測を再現性よく上回れた。
2. Stage 1最良では、median baseline比でTmApp約9.8%、HIC約8.6%のMAE低減が得られた。
3. TmAppでは、単純配列組成、CDR長、germline情報など複数の特徴表現が予測情報を持つ。
4. TmApp上位モデル間の差は非常に小さく、一つの特徴量群だけが支配的とは言えない。
5. HICでは、線形モデルよりRBF-SVRの改善が明確で、複数の配列物性の非線形な組合せが重要である可能性が高い。
6. HICでSEQ_PLUS_ANTIBODYがPrimary最良だったが、SEQ_ALLとの差は小さく、annotation追加の独立した価値はまだ確定していない。
7. Stage 1で評価したモデル群ではLight-only特徴が弱い傾向を示したが、完全なcontrolled ablationではない。
8. Primaryで改善したGBDTの一部はShadowで再現せず、小標本ではPrimary CVへの適応を警戒する必要がある。
9. 古典特徴だけではなお残差が大きく、PLMが追加情報を提供できる余地がある。

## 12. 次に試すべきこと

Stage 2（PLM）で検証可能な仮説:

### Hypothesis A
PLM embedding単独は、Stage 1 classical featuresを上回るか？

### Hypothesis B
抗体特化PLMは、generic protein PLMを上回るか？

### Hypothesis C
Heavy-only / Light-only / H+Lの差は、同一PLM・同一pooling・同一regressor条件でも再現するか？  
（Stage 1の観察をcontrolled ablationとして再検証する）

### Hypothesis D
PLM + Stage1 featuresは、PLM単独より改善するか？  
（classical featuresとPLMが相補的情報を持つかを検証する）

### Hypothesis E
HICでは、PLMが単純なGRAVY / compositionでは表せないsequence contextを捉えるか？

運用上の基準:

- Stage1最終候補（TmApp: SEQ_BASIC+SVR、HIC: SEQ_PLUS_ANTIBODY+SVR、およびHICのSEQ_ALL/SEQ_COMBINED+SVR）をclassical baselineとして固定する。
- PLM追加の採否はPrimaryとShadowの両方のΔMAEで判定する。
- 立体構造特徴（SASA / RASA / surface patch等）は、HICの表面化学仮説を直接検証する段階まで温存する。

---

**最終状態:** `STAGE1_FINAL_REPORT_FROZEN`
