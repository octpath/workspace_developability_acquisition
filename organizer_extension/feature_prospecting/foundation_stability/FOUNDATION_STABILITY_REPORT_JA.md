# Foundation-Model Stability Prospecting — TmApp 報告

**状態:** `ORGANIZER_TMAPP_FOUNDATION_STABILITY_PROSPECTING_COMPLETE`  
**証拠境界:** ORGANIZER-EXPLORATORY（Public/Private は post-reveal 複製のみ）  
**範囲不一致（全モデル）:** 実験 TmApp は **Fab**；構造入力は **Fv**（CH1/CL は捏造しない）  
**HIC:** 本フェーズでは最適化しない

---

## 冒頭回答（§14）

1. **実際に成功したモデル**  
   - **FeNNix-Bio1S:** Gate F1 成功 + Dev/Test 全 324×2 generators 特徴抽出成功 → **唯一のフル評価対象**  
   - **LiTEN-FF:** Gate F1 成功（3 Abs × ESMFold+ABB2、CPU）；GPU は Fv で OOM；フル Dev 抽出は未完 → **PILOT_ONLY（技術成功・未スコア）**  
   - **BioEmu:** 鎖ごと VH パイロット成功（n=4）；Fv 多量体は不使用；Dev×32 は未完 → **PILOT_ONLY**  
   - **CGSchNet:** 二本鎖 Fv の公式 CG 写像なし → **BLOCKED**

2. **最強 STANDALONE TmApp 信号:** **FeNNix on ABodyBuilder2**（CV MAE 3.19 vs median 3.44；ESMFold 単独は中央値以下）

3. **最強 PLM 増分:** **FeNNix + AbLang2 on ABB2**（CV ΔMAE ≈ −0.17）だが Public/Private では悪化 → **GENERATOR_SPECIFIC / 非再現的**

4. **PLM 残差を説明:** ABB2 上で弱い再構成改善（Δ≈−0.15）；ESMFold では実質なし → **弱い／未解決**

5. **古典 OpenMM strain を上回るか:** 予測価値としては **明確な上回はない**。FeNNix 摂動応答は clash / 原子数と強く相関（特に ABB2）し、古典ひずみと同型の **構造生成アーティファクト** リスクが大きい

6. **BioEmu アンサンブル:** Dev スコアリング未実施のため **TmApp 増分の証拠なし**（鎖ごとパイロットのみ；界面ダイナミクス欠如）

7. **ESMFold vs ABB2 頑健性:** **FRAGILE**（力・一部 ΔE の generator 間 Spearman 中央値低）

8. **clash / 予測構造アーティファクト:** **強く疑われる**（`severe_clash_count` と P1/P3 ΔE の Spearman 最大 ~0.94 on ABB2）

9. **最終 TmApp モデルへの持ち込み:** **推奨しない**（どの foundation family も最終候補に入れない）

---

## 解釈クラス（モデル別）

| Model | STANDALONE | INCREMENTAL | FOUNDATION PHYSICS |
|-------|------------|-------------|--------------------|
| FeNNix-Bio1S (ESMFold) | NO_EVIDENCE〜WEAK（median 未達） | NO_INCREMENT（悪化） | MOSTLY_STRUCTURE_ARTIFACT / MOSTLY_REDUNDANT_WITH_PLM |
| FeNNix-Bio1S (ABB2) | WEAK_SIGNAL | WEAK_OR_MIXED_INCREMENT / GENERATOR_SPECIFIC | MOSTLY_STRUCTURE_ARTIFACT |
| LiTEN-FF | 未評価（PILOT_ONLY） | 未評価 | UNRESOLVED（技術のみ） |
| BioEmu | 未評価（PILOT_ONLY） | 未評価 | UNRESOLVED（鎖ごとのみ） |
| CGSchNet | BLOCKED | BLOCKED | UNRESOLVED |

---

## Gate F0 / F1 要約

- 監査: `FOUNDATION_MODEL_FEASIBILITY_AUDIT.md`  
- パイロット Abs（TmApp ブラインド）: ADI-47078 / ADI-46705 / ADI-46684  
- 構造: ESMFold 主、ABB2 副；`pdb2pqr --ff=AMBER --keep-chain`  
- 特徴仕様凍結: `FOUNDATION_FEATURE_SPEC.json`（摂動振幅はスコア前固定）

---

## FeNNix 評価（Primary CV）

基準: `TRAIN_MEDIAN_BASELINE` CV MAE ≈ **3.438**  
PLM: `ablang2__HL_paired` → fold-local StandardScaler → PCA32 → Ridge（Fusion Closure P0 と同一レシピ）

| Generator | Mode | CV MAE | Δ vs ref | 95% CI (Δ) | p_improve |
|-----------|------|--------|----------|------------|-----------|
| — | MEDIAN | 3.438 | 0 | — | — |
| ESMFold | STANDALONE | 3.574 | +0.136 vs median | [−0.20, +0.51] | 0.23 |
| ABB2 | STANDALONE | 3.188 | −0.250 vs median | [−0.49, −0.01] | 0.98 |
| any | PLM_ONLY | 3.029 | −0.410 vs median | [−0.68, −0.14] | ≈1.00 |
| ESMFold | PLM+FeNNix | 3.185 | **+0.156 vs PLM** | [−0.15, +0.55] | 0.19 |
| ABB2 | PLM+FeNNix | 2.859 | **−0.169 vs PLM** | [−0.35, +0.01] | 0.97 |
| ESMFold | RESIDUAL_RECON | 3.011 | −0.018 vs PLM | [−0.23, +0.22] | 0.58 |
| ABB2 | RESIDUAL_RECON | 2.879 | −0.150 vs PLM | [−0.32, +0.01] | 0.96 |

Shadow も ABB2 増分は同方向だが、**Public/Private では PLM+FeNNix が PLM 単独より悪化**（過適合／generator 特異の典型）。

頑健性: `FOUNDATION_ROBUSTNESS.csv` → **FRAGILE**

---

## 古典物理との比較

`results/FOUNDATION_VS_CLASSICAL_PHYSICS.csv`

- ABB2: FeNNix `P1_dE_mean` ↔ `severe_clash_count_before` Spearman ≈ **0.94**  
- ESMFold: 同相関は弱いが依然 clash/サイズと連動  
- OpenMM `initial_force_RMS` との関係は generator 依存（ABB2 で強い順位相関あり）  
→ 「新しい熱力学安定性情報」より **予測構造の局所ひずみ／clash の再符号化** と解釈するのが妥当

---

## LiTEN / BioEmu / CG

### LiTEN-FF
- インストール成功；H2O および Fv で E/F 有限値  
- RTX 3090 で Fv グラフ **CUDA OOM**；CPU でパイロット **3×2 = 6/6 SUCCESS**（FeNNix と同スキーマ）  
- パイロット上 `P1_dE_mean` は FeNNix と高い相関（ESMFold n=3 で ≈0.99）だが **n 極小・Dev スコアなし**  
- **Dev 全件スコアリングなし** → TmApp A/B/C の主張不可

### BioEmu
- 科学的に妥当な一次プロトコルは **鎖ごと VH/VL**（リンカー scFv は一次にしない）  
- パイロット: ADI-47078 VH, n=4 → 記述子取得成功  
- 界面ダイナミクス欠如を明示；フルコホート未実施

### CGSchNet / mlcg
- 二本鎖 Ab Fv への clean な transferable CG 適用が保証されない  
- **BLOCKED**（疑わしい写像を発明しない）

Secondary（UMA / OrbMol / MACE）: first-wave が「全ブロック」ではないため **未展開**

---

## 最終推奨

1. **最終 TmApp モデルに foundation ML potential / BioEmu / CG 特徴を持ち込まない**  
2. FeNNix の ABB2 単独・増分は **構造生成器特異かつ clash 連動** のため「新安定性情報」と主張しない  
3. 古典 OPENMM-STRAIN 系を「MLFF で置き換えれば解決」する仮説は、本パイロットでは **支持されない**  
4. 今後やるなら（本フェーズ外）: Fab 構造・明示溶媒・clash 条件付け残差など、アーティファクト分離を先に設計すること

---

## 成果物

| File | Role |
|------|------|
| `FOUNDATION_MODEL_FEASIBILITY_AUDIT.md` | Gate F0 |
| `FOUNDATION_FEATURE_SPEC.json` | 凍結仕様 |
| `FOUNDATION_FEATURE_MANIFEST.csv` | 特徴一覧 |
| `FOUNDATION_STANDALONE_RESULTS.csv` | 単独 + PLM |
| `FOUNDATION_INCREMENTAL_RESULTS.csv` | PLM+physics |
| `FOUNDATION_RESIDUAL_RESULTS.csv` | 残差再構成 |
| `FOUNDATION_ROBUSTNESS.csv` | ESMFold vs ABB2 |
| `FOUNDATION_STABILITY_REPORT_JA.md` | 本報告 |
| `results/FOUNDATION_ALL_EVAL.csv` | 全評価行 |
| `results/FOUNDATION_VS_CLASSICAL_PHYSICS.csv` | vs OpenMM/clash |
| `cache/fennix_features/` | FeNNix 特徴 |
| `pilots/` | Gate F1 |

**Do not modify** prior `feature_prospecting` family results outside this new branch.
