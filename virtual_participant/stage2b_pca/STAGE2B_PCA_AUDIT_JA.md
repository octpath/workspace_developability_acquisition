# Stage 2b — PLM embeddingにPCAは必要か

## 1. この追加監査で何を調べたか

Stage 2ではPLM埋め込みをPCAで圧縮してからRidgeやRBF-SVRに渡していた。しかし「PCAを使わないraw embedding」との公平比較が必須ではなかった。本監査（Stage 2b）は、同じPLM・同じ鎖表現・同じ回帰モデルのもとで、**PCAあり/なし**および**PCA次元**がMAEにどう効くかを切り分ける。新しいPLMや構造特徴は導入していない。

## 2. なぜこの監査が必要だったか

Stage 2では、固定PCA32 + 固定SVRと、個別最適化SVRの間に大きな性能差があった。

| 例 | 固定PCA32+SVR | 最適化後 | 差 |
|---|---:|---:|---:|
| TmApp AbLang2 HL_paired | ≈3.258 | ≈2.923 | ≈0.335 |
| HIC ESM-2 Heavy | ≈0.570 | ≈0.455 | ≈0.115 |

この差の候補要因は (1) PCA次元 (2) C/gamma/epsilon などSVRハイパーパラメータ である。Stage 2だけでは「PCAが必要だった」とは言えないため、raw条件を明示的に入れた。

## 3. Stage 2で既に試されていたPCA条件

詳細: `PCA_EXISTING_RESULT_AUDIT_JA.md`

- 既存 `pca_dim`: 8 / 16 / 32 / 48 / 64 のみ
- **raw（None）は Stage 2 registry に存在しない**

## 4. Raw vs PCA32 controlled comparison

固定ハイパーパラメータ（Ridge α=10、SVR C=3 / gamma=0.01 / epsilon=0.1）。差は「PCAを入れたこと」だけ。

### TmApp — Ridge
| PLM | chain | raw MAE | PCA32 MAE | Δ(raw−PCA32) |
|---|---|---:|---:|---:|
| ablang2 | HL_paired | 3.0548 | 3.0351 | +0.0197 |
| ablang2 | HL | 3.0694 | 3.0164 | +0.0530 |
| ablang2 | H | 3.7993 | 3.3092 | +0.4900 |
| esm1b | HL | 3.8382 | 3.3505 | +0.4877 |
| esm2 | HL | 3.9125 | 3.2498 | +0.6627 |
| esm1b | H | 4.1001 | 3.5920 | +0.5081 |
| ablang2 | L | 4.1545 | 3.3857 | +0.7688 |
| esm2 | H | 4.3474 | 3.5436 | +0.8039 |
| esm1b | L | 4.3672 | 3.7641 | +0.6032 |
| esm2 | L | 5.6154 | 3.7154 | +1.9000 |

### TmApp — SVR
| PLM | chain | raw MAE | PCA32 MAE | Δ(raw−PCA32) |
|---|---|---:|---:|---:|
| ablang2 | L | 3.3257 | 3.2516 | +0.0741 |
| ablang2 | H | 3.3600 | 3.2988 | +0.0612 |
| ablang2 | HL_paired | 3.3632 | 3.2580 | +0.1052 |
| esm2 | L | 3.3690 | 3.3279 | +0.0411 |
| esm1b | L | 3.3871 | 3.4176 | -0.0306 |
| ablang2 | HL | 3.4451 | 3.3685 | +0.0765 |
| esm1b | H | 3.4479 | 3.4100 | +0.0379 |
| esm2 | H | 3.4482 | 3.4305 | +0.0177 |
| esm2 | HL | 3.4516 | 3.4440 | +0.0076 |
| esm1b | HL | 3.4517 | 3.4479 | +0.0038 |

### HIC — Ridge
| PLM | chain | raw MAE | PCA32 MAE | Δ(raw−PCA32) |
|---|---|---:|---:|---:|
| esm2 | H | 0.5894 | 0.5076 | +0.0819 |
| ablang2 | HL_paired | 0.6355 | 0.5722 | +0.0633 |
| esm2 | HL | 0.6464 | 0.5390 | +0.1074 |
| esm1b | H | 0.6513 | 0.5541 | +0.0972 |
| ablang2 | H | 0.6683 | 0.5489 | +0.1194 |
| esm1b | HL | 0.7140 | 0.6187 | +0.0954 |
| ablang2 | HL | 0.7238 | 0.5997 | +0.1241 |
| ablang2 | L | 0.8068 | 0.6408 | +0.1660 |
| esm1b | L | 1.0108 | 0.6600 | +0.3509 |
| esm2 | L | 1.0774 | 0.6442 | +0.4332 |

### HIC — SVR
| PLM | chain | raw MAE | PCA32 MAE | Δ(raw−PCA32) |
|---|---|---:|---:|---:|
| ablang2 | H | 0.5905 | 0.6000 | -0.0096 |
| ablang2 | HL_paired | 0.5914 | 0.5894 | +0.0020 |
| esm2 | H | 0.5944 | 0.5703 | +0.0240 |
| ablang2 | HL | 0.5970 | 0.5800 | +0.0171 |
| esm1b | H | 0.5976 | 0.5901 | +0.0075 |
| ablang2 | L | 0.5980 | 0.6089 | -0.0108 |
| esm2 | HL | 0.5988 | 0.5953 | +0.0035 |
| esm1b | HL | 0.5989 | 0.5996 | -0.0007 |
| esm2 | L | 0.6032 | 0.6035 | -0.0004 |
| esm1b | L | 0.6124 | 0.6296 | -0.0172 |

解釈メモ: Δ(raw−PCA32) が負なら raw の方が良い（PCAが損）、正なら PCA32 の方が良い。

## 5. PCA dimension比較（joint Optuna）

主要候補について Optuna（`pca_dim` ∈ {None,8,16,32,48,64} と Ridge/SVR hyperparameters の **joint search**、約40 trials）。

以下の selected pca_dim は「PCA単独の最適次元」ではなく、**今回のjoint探索で選ばれた最良configurationに含まれた pca_dim** である。

- `ablang2/HL_paired/RidgeOpt`: selected pca_dim=None, MAE=2.8634
- `ablang2/HL/RidgeOpt`: selected pca_dim=None, MAE=2.9203
- `ablang2/HL_paired/SVROpt`: selected pca_dim=64, MAE=2.9678
- `ablang2/HL/SVROpt`: selected pca_dim=64, MAE=3.1241
- `esm1b/L/SVROpt`: selected pca_dim=None, MAE=3.3264
- `esm1b/L/RidgeOpt`: selected pca_dim=8, MAE=3.3848

- `esm2/H/SVROpt`: selected pca_dim=None, MAE=0.4530
- `esm1b/H/SVROpt`: selected pca_dim=64, MAE=0.4703
- `esm2/H/RidgeOpt`: selected pca_dim=32, MAE=0.5045
- `esm1b/L/SVROpt`: selected pca_dim=32, MAE=0.5177
- `ablang2/L/SVROpt`: selected pca_dim=32, MAE=0.5186
- `esm1b/H/RidgeOpt`: selected pca_dim=64, MAE=0.5410
- `ablang2/L/RidgeOpt`: selected pca_dim=8, MAE=0.5881
- `esm1b/L/RidgeOpt`: selected pca_dim=8, MAE=0.6031

## 6. RidgeではPCAは必要だったか

Ridge固定条件では、rawがPCA32より良い割合≈0%、PCA32が良い割合≈100%、平均Δ(raw−PCA32)=+0.3968。

固定α=10のRidgeでは、高次元raw（特にESM 1280/2560d）に対してPCA32がほぼ一貫して有利だった（Case B）。一方、AbLang2 paired（480d）では、今回のjoint Optuna探索で選ばれた最良configurationが **pca_dim=None + 強い正則化（大きいα）** であり、ShadowでもStage2を上回った。したがってRidgeでも「常にPCA必須」ではなく、**次元と正則化の組合せ**の問題である。

## 7. RBF-SVRではPCAは必要だったか

SVR固定条件では、rawが良い割合≈30%、PCA32が良い割合≈70%、平均Δ(raw−PCA32)=+0.0205。

固定SVRではPCA32がやや有利なことが多いが、差はRidgeほど大きくない。Stage 2の大きなスコア差（例: 3.258→2.923）は、固定PCA32だけでなく **SVRのC/gamma/epsilon最適化**が大きく寄与した可能性が高い。HICのESM-2 Heavyでは、今回のjoint探索で選ばれた最良configurationに `pca_dim=None` が含まれたが、Stage2 bestとの差は小さく同等扱いとした。

## 8. TmApp結論

- Stage2 PLM-only: **2.9231** (AbLang2 HL_paired, PCA64, SVROpt)
- Stage2b PLM-only best: **2.8634 / Shadow 2.9803** (AbLang2 HL_paired, **今回のjoint探索で pca_dim=None を含む最良configuration**, RidgeOpt α≈54) — `PCA_NO_PCA_GAIN_CONFIRMED`
- Primary / Shadow双方で約0.056–0.060 °C改善し、HICより明瞭な再現改善だったため **PLM-only incumbentをStage2bモデルへ更新した**。
- Stage2b fusion (raw + SEQ_BASIC): Primary 2.763 とわずかに良いが Shadow 2.903 で Stage2 fusion（2.832）より悪化 → fusionは更新しない
- Decision: **PLM-only は USE_STAGE2B（raw Ridge）**、**fusion は KEEP_STAGE2**

## 9. HIC結論

- Stage2 PLM-only: **0.4552** (ESM-2 Heavy, PCA48, SVROpt)
- Stage2bのESM-2 Heavy raw + SVRは、Stage 2 PLM-onlyに対して Primary / Shadow の双方で数値上わずかに良かった（Primary ≈0.4530、Shadow ≈0.4480）。ただし改善幅は Primary約0.002 min、Shadow約0.005 min と小さい。この差だけから明確な性能優位と判断することは避け、モデル管理上の連続性も考慮して **Stage 2 incumbentを維持した**。なお0.01 minを統計的有意差や formal な採用閾値として用いたわけではない。
- Decision: **KEEP_STAGE2**

## 10. Primary / Shadow整合性

| Target | PLM | Chain | Regressor | PCA dim | Primary MAE | Shadow MAE |
|---|---|---|---|---|---:|---:|
| TmApp | ablang2 | HL_paired | RidgeOpt | None | 2.8634 | 2.9803 |
| TmApp | ablang2 | HL | RidgeOpt | None | 2.9203 | 2.9537 |
| TmApp | ablang2 | HL_paired | SVROpt | 64 | 2.9678 | 2.9844 |
| HIC | esm2 | H | SVROpt | None | 0.4530 | 0.4480 |
| HIC | esm1b | H | SVROpt | 64 | 0.4703 | 0.4609 |
| HIC | esm2 | H | RidgeOpt | 32 | 0.5045 | 0.5041 |

## 11. Stage 2 best modelを更新すべきか

| Target | Stage2 best (PLM-only) | Stage2b best | ΔPrimary | ΔShadow | Decision |
|---|---:|---:|---:|---:|---|
| TmApp | 2.9231 | 2.8634 | -0.0597 | -0.0558 | USE_STAGE2B |
| HIC | 0.4552 | 0.4530 | -0.0022 | -0.0049 | KEEP_STAGE2 |

方針: Primary/Shadow双方での改善の大きさ・再現性、およびモデル管理上の連続性を総合して Stage2b を新 incumbent とするかを判断した。`stage2_best_models.json` は上書きせず、`stage2b_best_models.json` に記録。

なお **0.01 °C / 0.01 min を統計的有意差や formal な採用閾値として用いたわけではない**。

## 12. Stage 3 structureへ持ち越すモデル

| Target | track | source | Primary | Shadow |
|---|---|---|---:|---:|
| TmApp | PLM-only | Stage2b AbLang2 HL_paired Ridge **raw** | 2.8634 | 2.9803 |
| TmApp | overall (+classical) | Stage2 AbLang2+SEQ_BASIC fusion (PCA48 SVR) | 2.7756 | 2.8316 |
| HIC | overall | Stage2 ESM-2 Heavy + SEQ_ALL fusion | 0.4485 | 0.4510 |

補足: TmAppのoverallはShadowが良いStage2 fusionを維持。PLM-only比較の基準はStage2b raw Ridgeへ更新する。

---

**最終状態:** `STAGE2B_FINAL_REPORT_FROZEN`
