# Simple TVT Competition Rescreen — 最終報告

**日付:** 2026-09-08  
**範囲:** 凍結済み構造/物理ブロックの直接特徴融合（Simple TVT）。`STRICT_NESTED_INCREMENT` の代替ではない。  
**FeNNix 本番:** 未変更（進捗とは独立）。

**プロトコル修正:** 初回再スクリーニングで高次元 BASE 全体に PCA をかけた誤りを発見。最終結果は **STRUCTURE ブロックのみ dim>200 で PCA32**、BASE はそのまま Ridge。CONTINUOUS_SURFACE / INTERIM_CONSTANT は初回 Simple TVT と整合。

---

## 冒頭 7 問

1. **BioEmu は直接 Simple TVT で効いたか？**  
   **はい（TmApp）。** `NEW_PAIRWISE` が最安定（P/S Δ≈+0.034/+0.033、fixed 生存）。CONTACT / FLEX / COMBINED も両方正。V12 旧セットは効かない。HIC 探索は混合〜無効。詳細: `BIOEMU_SIMPLE_TVT_REPORT_JA.md`。

2. **以前 late-fusion 陰性だったが直接融合で陽性になったものは？**  
   - BioEmu NEW_*（TmApp）  
   - HIC `CONTINUOUS_SURFACE` / `HIC_SURFACE_ALL`（弱い）  
   - Interim `CONSTANT`（暫定 N=100、強い）  
   - Marathon `M1_PROTEINMPNN`、`S2_GENERATOR_DISAGREEMENT`、`M3_SAPROT`（弱い〜中）  
   Gap Closure の CORE_DEFECT / GAP_ALL / FAB_INTERFACE は **引き続き非改善**。

3. **fixed-alpha 制御を生き残るものは？**  
   上記の両方正例のほぼすべて（INTERIM 系・BioEmu NEW・M1・S2・M3・HIC 表面/HYDRO/TITRATION 等）。`INTERIM_FULL_FAB_NORMALIZED` は FREE では正に見えても fixed で崩れ、除外。

4. **Primary∧Shadow で陽性のファミリーは？**  
   TmApp: INTERIM_PREP_RELAX_SENSITIVITY, INTERIM_CONSTANT, BIOEMU_NEW_PAIRWISE/FLEX/CONTACT/COMBINED, M1_PROTEINMPNN, S2, M3_SAPROT。  
   HIC: HYDRO_FIELD, TITRATION_SHAPE, HIC_SURFACE_ALL, CONTINUOUS_SURFACE, STATIC_SAP, AROMATIC_TOPO（いずれも Shadow Δ は小さい）。

5. **最良の Dev 組み合わせは？**  
   - TmApp（暫定 N=100）: **`BIOEMU_NEW_CONTACT + INTERIM_CONSTANT`**（worst Δ≈+0.119、fixed 生存）  
   - TmApp（フル Dev N=162）: **`BIOEMU_NEW_PAIRWISE + M1_PROTEINMPNN`**（worst Δ≈+0.051）  
   - HIC: **`HYDRO_FIELD + TITRATION_SHAPE`**（worst Δ≈+0.0087）または **`CONTINUOUS_SURFACE + TITRATION_SHAPE`**（worst≈+0.0067）

6. **競技最適と厳密科学結論は違うか？**  
   **違う。** Strict nested late-fusion は多くのブロックで `NO_INCREMENT`。直接融合は複数ブロックで正。科学側は「incumbent late-fusion 増分なし」を維持しつつ、競技側は直接融合レシピを `COMP_TRY/PRIORITY` として保持してよい。機序の確定とは区別する。

7. **最終競技評価に持ち込む 1–3 レシピは？**  
   1. **TmApp-A（フル Dev）:** BASE(AbLang2+SEQ_BASIC) + `BIOEMU_NEW_PAIRWISE` + `M1_PROTEINMPNN`  
   2. **TmApp-B（Fab FeNNix 完了後に再確認）:** BASE + `BIOEMU_NEW_CONTACT` + `INTERIM_CONSTANT`→フル Fab FeNNix CONSTANT 相当  
   3. **HIC-A:** BASE(ESM2+SEQ+ARO) + `CONTINUOUS_SURFACE` + `TITRATION_SHAPE`（または HYDRO_FIELD+TITRATION）

   Public/Private での選択・チューニングは行っていない。

---

## 成果物

| ファイル | 内容 |
|----------|------|
| `SIMPLE_TVT_FEATURE_INVENTORY.csv` | 52 行の凍結ファミリー一覧と Simple TVT 既実施フラグ |
| `SIMPLE_TVT_ALL_BLOCKS.csv` | 単ブロック FREE/FIXED 全結果 |
| `SIMPLE_TVT_COMPETITION_SHORTLIST.csv` | ターゲット別上位短リスト |
| `SIMPLE_TVT_LIMITED_COMBINATIONS.csv` | 事前指定の少数組み合わせ |
| `BIOEMU_SIMPLE_TVT_REPORT_JA.md` | BioEmu 専用 8 問 |
| `SIMPLE_TVT_RESCREEN_REPORT_JA.md` | 本報告 |

## FeNNix フルコホート

本番完了後、同一 Simple TVT プロトコルをフル usable Dev で自動再実行する（interim 結果でプロトコルは変更しない）。
