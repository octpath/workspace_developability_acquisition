# Stage 2 — Protein Language Modelによる配列表現

## 1. このStageで何をしたか

Stage 0の共通5-fold CVを固定したまま、ESM-1b・ESM-2・AbLang2の凍結埋め込みを公平比較した。目的は、Stage 1の古典的配列特徴では捉えきれない配列文脈をPLMが追加できるかを検証することである。Stage 1基準は TmApp MAE=3.1011 / HIC MAE=0.4732（いずれもPrimary）である。

TmAppでは抗体特化のAbLang2（paired）がPLM-only最良（Primary 2.923 / Shadow 3.036）となり、SEQ_BASICとのfusionでさらに 2.776 / 2.832 まで改善した。HICではgenericのESM-2 Heavy-onlyがPLM-only最良（0.455 / 0.453）で、SEQ_ALL fusionはわずかに追加改善（0.448 / 0.451）した。controlled ablationでは両ターゲットでLight-onlyが最も弱く、H+L（またはHeavy）が相対的に良い傾向が再現した。構造特徴やfine-tuningには進んでいない。

## 2. Stage 1からの出発点

| Target | Stage1 best | Primary MAE | Shadow MAE |
|---|---|---:|---:|
| TmApp | SEQ_BASIC + SVROpt | 3.1011 | 3.1716 |
| HIC | SEQ_PLUS_ANTIBODY + SVROpt | 0.4732 | 0.4788 |

HIC参考:
- SEQ_ALL + SVROpt: Primary 0.4766 / Shadow 0.4795
- SEQ_COMBINED + SVROpt: Primary 0.4775 / Shadow 0.4724

PLMの価値は median との差だけでなく、**このStage1 bestとの差**で評価する。

## 3. 使用したPLM

| PLM | 種類 | embedding | pooling | 備考 |
|---|---|---|---|---|
| ESM-1b | generic protein LM | 残基埋め込み 1280d/鎖 | whole-chain mean → H/L/HL | gate_b1 raw cache再利用 |
| ESM-2 (650M) | generic protein LM | 同上 | 同上 | 同上 |
| AbLang2 | antibody-specific LM | paired seqcoding 480d | H/L別seqcoding + concat | paired 480dも補助比較 |

**generic PLM**は一般タンパク質配列で学習された表現、**antibody-specific PLM**は抗体（対）配列に特化した表現である。どちらがdevelopability予測に有利かは事前に決めず、実測で比較した。

キャッシュ監査: `PLM_CACHE_AUDIT_JA.md`

## 4. 評価方法

- Primary CV: モデル選択・Optuna・表現比較
- Shadow CV: Primary上位のみ監査（Optuna目的には未使用）
- PCA / StandardScaler は各training fold内でfit
- Fixed pipeline（公平比較）と個別Optunaを分離して記録
- Optuna目安: 主要候補あたり最大約40 trials（fusionは30）
- Stage 2ではfold-local PCAを用いて複数次元（8/16/32/48/64）を探索したが、PCAを全く使用しないraw embedding条件については systematic な controlled comparison を必須条件としていなかった（既存registry上も `pca_dim=None` の実験は無い）。したがって、「PLMにはPCAが必要である」または「PCAが性能改善の主因である」とは Stage 2 の結果だけでは結論しない。この点は Stage 2b で独立に監査する。

## 5. Fixed-pipelineによるPLM公平比較

固定条件例: `PCA=32` + `Ridge`（および同条件SVR）。

### TmApp（PCA32 + Ridge）
| PLM | chain | Primary MAE | Δ vs Stage1 |
|---|---|---:|---:|
| ablang2 | HL | 3.0164 | -0.0847 |
| ablang2 | HL_paired | 3.0351 | -0.0660 |
| esm2 | HL | 3.2498 | +0.1487 |
| ablang2 | H | 3.3092 | +0.2081 |
| esm1b | HL | 3.3505 | +0.2494 |
| ablang2 | L | 3.3857 | +0.2846 |
| esm2 | H | 3.5436 | +0.4425 |
| esm1b | H | 3.5920 | +0.4909 |
| esm2 | L | 3.7154 | +0.6143 |
| esm1b | L | 3.7641 | +0.6630 |

### HIC（PCA32 + Ridge）
| PLM | chain | Primary MAE | Δ vs Stage1 |
|---|---|---:|---:|
| esm2 | H | 0.5076 | +0.0344 |
| esm2 | HL | 0.5390 | +0.0658 |
| ablang2 | H | 0.5489 | +0.0757 |
| esm1b | H | 0.5541 | +0.0809 |
| ablang2 | HL_paired | 0.5722 | +0.0990 |
| ablang2 | HL | 0.5997 | +0.1265 |
| esm1b | HL | 0.6187 | +0.1455 |
| ablang2 | L | 0.6408 | +0.1676 |
| esm2 | L | 0.6442 | +0.1710 |
| esm1b | L | 0.6600 | +0.1868 |

### TmApp（PCA32 + SVR）
| PLM | chain | Primary MAE | Δ vs Stage1 |
|---|---|---:|---:|
| ablang2 | L | 3.2516 | +0.1505 |
| ablang2 | HL_paired | 3.2580 | +0.1569 |
| ablang2 | H | 3.2988 | +0.1977 |
| esm2 | L | 3.3279 | +0.2268 |
| ablang2 | HL | 3.3685 | +0.2674 |
| esm1b | H | 3.4100 | +0.3089 |
| esm1b | L | 3.4176 | +0.3165 |
| esm2 | H | 3.4305 | +0.3294 |
| esm2 | HL | 3.4440 | +0.3429 |
| esm1b | HL | 3.4479 | +0.3468 |

### HIC（PCA32 + SVR）
| PLM | chain | Primary MAE | Δ vs Stage1 |
|---|---|---:|---:|
| esm2 | H | 0.5703 | +0.0971 |
| ablang2 | HL | 0.5800 | +0.1068 |
| ablang2 | HL_paired | 0.5894 | +0.1162 |
| esm1b | H | 0.5901 | +0.1169 |
| esm2 | HL | 0.5953 | +0.1221 |
| esm1b | HL | 0.5996 | +0.1264 |
| ablang2 | H | 0.6000 | +0.1268 |
| esm2 | L | 0.6035 | +0.1303 |
| ablang2 | L | 0.6089 | +0.1357 |
| esm1b | L | 0.6296 | +0.1564 |

## 6. TmApp結果

Stage1 Primary **3.1011** → 最良PLM-only Primary **2.9231**（Δ=-0.1780）  
Shadow: Stage1 **3.1716** → PLM **3.0361**（status=`PLM_SHADOW_CONFIRMED`）

| Model family | Primary MAE | Shadow MAE | Δ vs Stage1 Primary | 解釈 |
|---|---:|---:|---:|---|
| Stage1 classical best | 3.1011 | 3.1716 | 0.0000 | 基準 |
| best ablang2 | 2.9231 | 3.0361 | -0.1780 | PLM-only |
| best esm1b | 3.1046 | 3.1543 | +0.0035 | PLM-only |
| best esm2 | 3.1152 | — | +0.0141 | PLM-only |
| best PLM+classical | 2.7756 | 2.8316 | -0.3255 | fusion |

Fusion最良: `TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt`  
Primary=2.7756, Shadow=2.8316

## 7. HIC結果

Stage1 Primary **0.4732** → 最良PLM-only Primary **0.4552**（Δ=-0.0180）  
Shadow: Stage1 **0.4788** → PLM **0.4529**（status=`PLM_SHADOW_CONFIRMED`）

| Model family | Primary MAE | Shadow MAE | Δ vs Stage1 Primary | 解釈 |
|---|---:|---:|---:|---|
| Stage1 classical best | 0.4732 | 0.4788 | 0.0000 | 基準 |
| best esm2 | 0.4552 | 0.4529 | -0.0180 | PLM-only |
| best esm1b | 0.4637 | 0.4533 | -0.0095 | PLM-only |
| best ablang2 | 0.4808 | 0.4787 | +0.0076 | PLM-only |
| best PLM+classical | 0.4485 | 0.4510 | -0.0247 | fusion |

Fusion最良: `HIC__FUSION__esm2__H__SEQ_ALL__SVROpt`  
Primary=0.4485, Shadow=0.4510

### Stage 2 bestモデルの前処理・hyperparameter（既存registry）

すべて StandardScaler（fold内fit）+ fold-local PCA を使用。固定PCA32+SVR（C=3, gamma=0.01, epsilon=0.1）との差は、PCA次元とSVRハイパーパラメータの両方の変化が候補である（因果は断定しない）。

| Target | 種別 | PLM | chain | PCA | Regressor | C | gamma | epsilon | Primary | Shadow |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| TmApp | PLM-only | AbLang2 | HL_paired | 64 | SVROpt | 35.23 | 4.45e-4 | 0.0173 | 2.9231 | 3.0361 |
| TmApp | fusion (+SEQ_BASIC) | AbLang2 | HL_paired | 48 | SVROpt | 18.19 | 2.91e-4 | 0.3568 | 2.7756 | 2.8316 |
| HIC | PLM-only | ESM-2 | H | 48 | SVROpt | 0.758 | 7.54e-4 | 0.0069 | 0.4552 | 0.4529 |
| HIC | fusion (+SEQ_ALL) | ESM-2 | H | 8 | SVROpt | 2.713 | 2.10e-4 | 0.0635 | 0.4485 | 0.4510 |

参考（同一PLM/chainの固定条件）: TmApp AbLang2 HL_paired PCA32+SVR ≈ 3.258; HIC ESM-2 H PCA32+SVR ≈ 0.570。

## 8. Heavy / Light / H+L controlled ablation

同一条件: whole-chain mean、PCA=32、Ridge。

### TmApp
| PLM | Heavy-only | Light-only | H+L concat |
|---|---:|---:|---:|
| esm1b | 3.5920 | 3.7641 | 3.3505 |
| esm2 | 3.5436 | 3.7154 | 3.2498 |
| ablang2 | 3.3092 | 3.3857 | 3.0164 |

### HIC
| PLM | Heavy-only | Light-only | H+L concat |
|---|---:|---:|---:|
| esm1b | 0.5541 | 0.6600 | 0.6187 |
| esm2 | 0.5076 | 0.6442 | 0.5390 |
| ablang2 | 0.5489 | 0.6408 | 0.5997 |

Stage 1で見られた「Light-onlyが弱い」傾向は、同一PLM・同一pipeline（PCA32 + Ridge）でも再現した。TmAppでは H+L < Heavy < Light、HICでは Heavy ≲ H+L ≪ Light の順で、Light単独は一貫して最も悪い。HICではHeavy-onlyがH+Lより良い場合があり、単純に「常にconcatが最良」ではない。

## 9. Generic PLM vs antibody-specific PLM

| Target | best ESM-1b | best ESM-2 | best AbLang2 |
|---|---:|---:|---:|
| TmApp | 3.1046 | 3.1152 | 2.9231 |
| HIC | 0.4637 | 0.4552 | 0.4808 |

TmAppでは、今回のStage 2探索範囲ではAbLang2系が最良だった。HICではgeneric ESM系（特にESM-2 Heavy-only）がAbLang2を上回った。抗体特化モデルが常に優位とは言えない。差がごく小さい比較は「同程度」と読む。

## 10. PLMはStage 1特徴を超えたか（Hypothesis A）

**TmApp: はい（再現性あり）。**  
PLM-only最良（AbLang2 paired + SVR）は Primary 2.923（ΔStage1=-0.178）、Shadow 3.036（Δ=-0.135）。Stage1を両CVで上回った。

**HIC: はい、ただし改善幅は小さめ。**  
PLM-only最良（ESM-2 Heavy + SVR）は Primary 0.455（Δ=-0.018）、Shadow 0.453（Δ=-0.026）。Stage1を両CVで下回るが、差は約0.02 min程度であり過大解釈しない。

## 11. PLMとclassical featuresは相補的だったか（Hypothesis D）

**TmApp: 相補性が明確。**  
AbLang2 + SEQ_BASIC fusionは Primary 2.776（PLM-only比 Δ=-0.148）、Shadow 2.832（Stage1比 Δ=-0.340）。classical追加がPLM-onlyをさらに改善し、Shadowでも維持された。

**HIC: 追加効果は小さい。**  
ESM-2 Heavy + SEQ_ALL は Primary 0.448（PLM-only比 Δ=-0.007）、Shadow 0.451。改善方向は一致するが、差が小さく「明確な相補性」とまでは言わない。SEQ_PLUS_ANTIBODY追加も同程度かそれ以下だった。

## 12. HICでsequence contextは有効だったか（Hypothesis E）

限定的に支持。ESM-2 Heavy-onlyがStage1 classical bestをShadowでも下回ったことは、単純組成だけでは足りない情報がある可能性を示す。ただし改善幅は小さく、残差とStage1の相関も高い（最良PLMでもPearson≈0.94）ため、PLMが全く別種のエラーを支配しているわけではない。表面露出など構造記述子での追加検証が自然な次段階である。

## 13. PrimaryとShadowは一致したか

### TmApp Shadow監査
- `TmApp__ablang2__HL_paired__SVROpt`: P=2.9231 → S=3.0361 (ΔS1 P=-0.1780 / S=-0.1355) **PLM_SHADOW_CONFIRMED**
- `TmApp__ablang2__HL__RidgeOpt`: P=2.9376 → S=3.0306 (ΔS1 P=-0.1635 / S=-0.1410) **PLM_SHADOW_CONFIRMED**
- `TmApp__ablang2__HL_paired__RidgeOpt`: P=2.9661 → S=3.0617 (ΔS1 P=-0.1350 / S=-0.1099) **PLM_SHADOW_CONFIRMED**
- `TmApp__ablang2__HL__SVROpt`: P=3.0299 → S=3.0312 (ΔS1 P=-0.0712 / S=-0.1404) **PLM_SHADOW_CONFIRMED**
- `TmApp__esm1b__HL__SVROpt`: P=3.1046 → S=3.1543 (ΔS1 P=+0.0035 / S=-0.0173) **PLM_NO_GAIN**
- `TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt`: P=2.7756 → S=2.8316 (ΔS1 P=-0.3255 / S=-0.3400) **PLM_SHADOW_CONFIRMED**
- `TmApp__FUSION__ablang2__HL_paired__ANN_CDR_LENGTH__SVROpt`: P=2.8750 → S=2.8740 (ΔS1 P=-0.2261 / S=-0.2976) **PLM_SHADOW_CONFIRMED**
- `TmApp__FUSION__ablang2__HL_paired__ANN_GERMLINE__SVROpt`: P=2.8983 → S=2.9620 (ΔS1 P=-0.2028 / S=-0.2096) **PLM_SHADOW_CONFIRMED**
- `TmApp__FUSION__esm1b__HL__SEQ_BASIC__SVROpt`: P=3.1146 → S=3.1559 (ΔS1 P=+0.0135 / S=-0.0157) **PLM_NO_GAIN**
- `TmApp__FUSION__esm1b__HL__ANN_CDR_LENGTH__SVROpt`: P=3.1141 → S=3.1528 (ΔS1 P=+0.0130 / S=-0.0188) **PLM_NO_GAIN**
- `TmApp__FUSION__esm1b__HL__ANN_GERMLINE__SVROpt`: P=3.1045 → S=3.1783 (ΔS1 P=+0.0034 / S=+0.0067) **PLM_NO_GAIN**
- `TmApp__FUSION__esm2__HL__SEQ_BASIC__SVROpt`: P=3.1265 → S=3.1555 (ΔS1 P=+0.0254 / S=-0.0161) **PLM_NO_GAIN**
- `TmApp__FUSION__esm2__HL__ANN_CDR_LENGTH__SVROpt`: P=3.1280 → S=3.1413 (ΔS1 P=+0.0269 / S=-0.0303) **PLM_NO_GAIN**
- `TmApp__FUSION__esm2__HL__ANN_GERMLINE__SVROpt`: P=3.1258 → S=3.1354 (ΔS1 P=+0.0247 / S=-0.0362) **PLM_NO_GAIN**

### HIC Shadow監査
- `HIC__esm2__H__SVROpt`: P=0.4552 → S=0.4529 (ΔS1 P=-0.0180 / S=-0.0259) **PLM_SHADOW_CONFIRMED**
- `HIC__esm1b__H__SVROpt`: P=0.4637 → S=0.4533 (ΔS1 P=-0.0095 / S=-0.0255) **PLM_SHADOW_CONFIRMED**
- `HIC__ablang2__H__SVROpt`: P=0.4808 → S=0.4787 (ΔS1 P=+0.0076 / S=-0.0001) **PLM_NO_GAIN**
- `HIC__esm2__HL__SVROpt`: P=0.4849 → S=0.4840 (ΔS1 P=+0.0117 / S=+0.0052) **PLM_NO_GAIN**
- `HIC__ablang2__HL_paired__SVROpt`: P=0.4863 → S=0.4697 (ΔS1 P=+0.0131 / S=-0.0091) **PLM_NO_GAIN**
- `HIC__FUSION__esm2__H__SEQ_ALL__SVROpt`: P=0.4485 → S=0.4510 (ΔS1 P=-0.0247 / S=-0.0278) **PLM_SHADOW_CONFIRMED**
- `HIC__FUSION__esm2__H__SEQ_PLUS_ANTIBODY__SVROpt`: P=0.4527 → S=0.4599 (ΔS1 P=-0.0205 / S=-0.0189) **PLM_SHADOW_CONFIRMED**
- `HIC__FUSION__esm2__H__SEQ_BASIC__SVROpt`: P=0.4553 → S=0.4531 (ΔS1 P=-0.0179 / S=-0.0257) **PLM_SHADOW_CONFIRMED**
- `HIC__FUSION__esm1b__H__SEQ_ALL__SVROpt`: P=0.4623 → S=0.4506 (ΔS1 P=-0.0109 / S=-0.0282) **PLM_SHADOW_CONFIRMED**
- `HIC__FUSION__esm1b__H__SEQ_PLUS_ANTIBODY__SVROpt`: P=0.4618 → S=0.4533 (ΔS1 P=-0.0114 / S=-0.0255) **PLM_SHADOW_CONFIRMED**
- `HIC__FUSION__esm1b__H__SEQ_BASIC__SVROpt`: P=0.4628 → S=0.4515 (ΔS1 P=-0.0104 / S=-0.0273) **PLM_SHADOW_CONFIRMED**
- `HIC__FUSION__ablang2__H__SEQ_ALL__SVROpt`: P=0.4798 → S=0.4791 (ΔS1 P=+0.0066 / S=+0.0003) **PLM_NO_GAIN**
- `HIC__FUSION__ablang2__H__SEQ_PLUS_ANTIBODY__SVROpt`: P=0.4758 → S=0.4849 (ΔS1 P=+0.0026 / S=+0.0061) **PLM_NO_GAIN**
- `HIC__FUSION__ablang2__H__SEQ_BASIC__SVROpt`: P=0.4765 → S=0.4745 (ΔS1 P=+0.0033 / S=-0.0043) **PLM_NO_GAIN**


## 14. 残差の相補性

OOF残差相関（低いほど相補的な誤りの可能性）:

### TmApp
- `TmApp__ablang2__HL__SVROpt` vs `STAGE1_TmApp`: Pearson=0.773
- `TmApp__ablang2__HL_paired__RidgeOpt` vs `STAGE1_TmApp`: Pearson=0.776
- `TmApp__ablang2__HL__RidgeOpt` vs `STAGE1_TmApp`: Pearson=0.796
- `TmApp__ablang2__HL_paired__SVROpt` vs `STAGE1_TmApp`: Pearson=0.826
- `TmApp__FUSION__ablang2__HL_paired__ANN_GERMLINE__SVROpt` vs `STAGE1_TmApp`: Pearson=0.840
- `TmApp__esm2__HL__RidgeOpt` vs `STAGE1_TmApp`: Pearson=0.841
- `TmApp__FUSION__ablang2__HL_paired__ANN_CDR_LENGTH__SVROpt` vs `STAGE1_TmApp`: Pearson=0.850
- `TmApp__FUSION__esm1b__HL__ANN_GERMLINE__SVROpt` vs `STAGE1_TmApp`: Pearson=0.902

### HIC
- `HIC__esm1b__H__RidgeOpt` vs `STAGE1_HIC`: Pearson=0.814
- `HIC__esm2__H__RidgeOpt` vs `STAGE1_HIC`: Pearson=0.840
- `HIC__esm2__HL__RidgeOpt` vs `STAGE1_HIC`: Pearson=0.842
- `HIC__esm1b__HL__RidgeOpt` vs `STAGE1_HIC`: Pearson=0.863
- `HIC__ablang2__HL_paired__RidgeOpt` vs `STAGE1_HIC`: Pearson=0.886
- `HIC__ablang2__H__RidgeOpt` vs `STAGE1_HIC`: Pearson=0.908
- `HIC__ablang2__HL__RidgeOpt` vs `STAGE1_HIC`: Pearson=0.912
- `HIC__esm2__H__SVROpt` vs `STAGE1_HIC`: Pearson=0.940

本Stageでは本格stackingは行わない。相補性が高そうでも、structure Stage以降の候補に留める。

## 15. Stage 2で分かったこと

1. ESM-1b / ESM-2 / AbLang2を、同一CV・同一前処理方針で比較できた。
2. Heavy / Light / H+Lのcontrolled ablationで、Light-onlyが両ターゲットで最も弱い傾向が再現した。
3. TmAppではAbLang2がStage1を再現性よく上回り、classical（SEQ_BASIC）とのfusionがさらに効いた。
4. HICではESM-2 Heavy-onlyがStage1を小さく上回った（約0.02 min）。fusionの追加はごく小さい。
5. 抗体特化PLMが常に優位ではない（HICではgeneric ESMの方が良い）。
6. 固定PCA32・固定ハイパーパラメータのpipelineではStage1に届かないケースが多かった一方、個別最適化後には大きく改善する候補があった。したがって、PLM familyだけでなく、PCA次元や回帰モデルの正則化・hyperparameter設定も性能に大きく影響した可能性がある。ただし、PCA自体の必要性はStage 2bで切り分ける。
7. Primary改善の一部はShadowで消える（特にgeneric ESMのTmAppやAbLang2のHIC）。
8. Stage1との残差相関は高めであり、完全に独立な信号ではない。
9. 構造特徴なしでも、PLMの伸びしろ（TmApp大・HIC小）を切り分けられた。

## 16. 次に試すべきこと（structure Stage）

仮説:

1. HICの残差が表面露出疎水性と対応するなら、SASA/RASAやhydrophobic patchがPLM残差を説明できる。
2. TmApp残差がパッキングやループ露出と関係するなら、構造記述子がAbLang2+classicalを補完しうる。
3. 構造fusionの採否は、現行best（TmApp: AbLang2+SEQ_BASIC、HIC: ESM-2 Heavy±SEQ_ALL）をShadowでも上回る場合に限る。
4. fine-tuning / learned poolingは、frozen PLMの頭打ちが明確な場合に限る。

---

**最終状態:** `STAGE2_PLM_CORE_FROZEN`  
（PCA/no-PCAの追加監査は Stage 2b）
