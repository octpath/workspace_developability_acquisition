# FeNNix Full-Fab Context — 日本語報告（途中〜パイロット後）

## 冒頭 10 問への回答（現時点）

1. **化学的に一貫した Fab 準備は可能か？**  
   **はい（パイロット範囲）.** パイロット 12/12 の調製済み Fab で FeNNix B/C/M が完走。既存 CUDA 調製 113/324 も再利用候補。全コホートは CPU 監査後。

2. **kappa/lambda の H–L ジスルフィドは再現可能に調製できたか？**  
   **パイロットでは yes.** QC 上 HL Sγ–Sγ ≈ 2.05 Å、ドメイン内 SS も ~2.04–2.06 Å。全 324 の最終監査は未完了。

3. **Fab 予測文脈は Fv FeNNix 特徴を変えたか？**  
   **技術的には B（Fab 幾何由来 Fv）が完走。** TmApp スコア（DELTA_GEOM）は **未実施**（特徴凍結・全コホート後）。

4. **同一 Fv 座標で定数ドメインはエネルギー応答を変えたか？**  
   **DELTA_ENV の前提（C↔M 座標一致）はパイロット 12/12 で成立**（max_abs=0、許容 1e-4 Å）。数値的な TmApp 信号は未スコア。

5. **CH1/CL に TmApp 信号は？** — **未スコア**

6. **VH–CH1 / VL–CL / CH1–CL 界面に信号は？** — **未スコア**

7. **AbLang2 を超える増分は？** — **未スコア**

8. **現行 practical incumbent を超える増分は？** — **未スコア**

9. **見かけの信号はアーティファクトか？** — **未スコア**（パイロット技術ゲートは通過）

10. **full Fab は FeNNix-v2 の否定を救済したか？** — **未判定**（スコア前）

---

## パイロット技術結果

**Verdict: `PILOT_PASS_WITH_LIMITATIONS`**

| 項目 | 結果 |
|------|------|
| B/C/M 完走 | 12/12 |
| C↔M 可変領域座標一致 | 12/12（完全一致） |
| 提示特徴の有限性 | OK（条件外 NaN は除外して判定） |
| DELTA_ENV 前提 | VALID |
| 制限 | 一部 FIRE 未収束、B で稀に severe clash 残、C の LCDR3 曲率プール空 |

TmApp/HIC はパイロット分類に **未使用**。

---

## 分子スコープの不一致

- TmApp 実験: **Fab**
- 旧 FeNNix-v2: **孤立 Fv**
- 本実験: 再構成 experimental-like Fab（完全実験配列ではない）

## なぜ Fab−Fv 全エネルギー差を使わないか

原子数・組成が異なり、絶対エネルギーは比較不能。代わりに A/B/C/M 分解と同一サイトでの曲率差を使う。

## A/B/C/M

- A: 孤立 Fv  
- B: Fab 予測から切り出し独立緩和 Fv  
- C: 調製 full Fab  
- M: C から CH1/CL 削除・**再緩和なし**（座標固定）

DELTA_GEOM = B−A、DELTA_ENV = C_var−M_var（一次）。

## 調製・監査の現状

- OpenMM **CUDA 連続 prep は再開しない**
- ADI-47317: orphan quarantine 後、**CPU 診断成功**（HL Sγ–Sγ ≈ 2.05 Å）
- CPU/CUDA 等価監査（電源断で一度中断 → 再開完了）:
  - 調製 QC: `PREP_METRICS_COMPATIBLE`
  - 曲率（4 Abs × condition C）: Spearman **0.984**、median NAD **0.019**
  - 判定: **`CPU_CUDA_EQUIVALENT_REUSE_113_CUDA_REMAINING_ON_CPU`**
- プロトコル凍結: `PROTOCOL_FREEZE_CPU_CUDA.json`
- Fab prep: **323/324 complete**（16-core OpenMM CPU、`OPENMM_CPU_THREADS=16` / taskset 0–15）
- **ADI-47265**: ソロ再試行でも再現する `QC_FAIL HL≈8.15 Å`（intra 4/4 OK）→ **`TECHNICAL_SKIP`**（SIGSEGV ではなく構造 QC）
- FeNNix 全コホート B/C/M: **16-core 上限で実行中**（`scripts/12_run_fennix_16cap.py`、resume 安全）

## 次ステップ（権威ある順序）

1. ~~パイロット 12~~ ✅  
2. ~~ADI-47317 CPU 診断~~ ✅  
3. ~~CPU/CUDA 等価監査~~ ✅  
4. ~~プロトコル凍結 → 残余 CPU バッチ prep~~ ✅（323/324; ADI-47265 skip）  
5. 全 FeNNix B/C/M（実行中）→ 特徴凍結 → TmApp スコア → path-scoped commit

## 状態ラベル（暫定）

`ORGANIZER_TMAPP_FENNIX_FAB_CONTEXT_FENNIX_FULL_RUNNING_16CAP`
