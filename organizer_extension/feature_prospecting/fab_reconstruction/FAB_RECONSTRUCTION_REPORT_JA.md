# Shehata Fab 再構成レポート

**Git HEAD（開始時）:** `f1a40e32b06f40dc64a6375f30b7aef78056e262`（`main`）  
**作業日:** 2026-09-05  
**出力:** `organizer_extension/feature_prospecting/fab_reconstruction/`

---

## 冒頭回答（14項目）

1. **正確な実験 Fab 配列を同定できたか?** → **No**
2. **再構成クラス** → **`RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB`**
3. **使用 CH1** → UniProt P01857 由来 `ASTKGPS…VDKKVEPKSC`（CH1 + `EPKSC`；ID `IGHG1_CH1_EPKSC_UNIPROT_P01857`）
4. **Cκ** → UniProt P01834（IGKC 全長）
5. **Cλ** → UniProt P0DOY2（IGLC2 全長；λ 全例に共通・事前宣言）
6. **κ / λ / unresolved** → **238 / 86 / 0**（N=324）
7. **VH/VL 境界補正** → **不要**（324/324 `junction_status=OK`；定数領域の重複なし）
8. **papain / hinge 末端** → papain 2 h / 30°C・Protein A・CaptureSelect IgG-CH1 は **HIGH**。切断後の正確な HC C 末端残基は **未確定**（`EPKSC` は化学的に完全な再構成仮定）
9. **12-Fab ESMFold パイロット** → **`PASS_FULL_COHORT`**（12/12 成功）
10. **全件生成** → 実行中（パイロット合格後に自動開始；完了数はマニフェスト参照）
11. **CH1/CL 界面** → パイロットで CA 接触（8 Å）中央付近 **32–46**；系統的な乖離なし
12. **H–L ジスルフィド幾何** → パイロット最小 SG–SG 距離 **約 2.3–3.3 Å**（概ね妥当；ESMFold は結合を強制しない）
13. **Fv vs Fab の VH/VL 変化（パイロット）** → 結合 Fv 骨格 RMSD 平均 **≈ 0.59 Å**（鎖内は小さい）。配向差メトリクスは補助的
14. **FeNNix-v2 前の残存不確実性** → 正確な allele / クローニング接合 / papain 末端；酵母ベクター固有の定数改変の有無；再構成 Fab ≠ 実験分子の同一性

---

## 事実（実験ソース）

- 組換え抗体は **full-length IgG1** を engineered *S. cerevisiae* で発現（HIGH）
- VH/VL は single-cell PCR → homologous recombination で expression vector へ
- Fab は IgG の **papain（2 h, 30°C）** → Protein A → CaptureSelect **IgG-CH1** 精製（HIGH）
- プロジェクト凍結配列: `gate_b1/data/shehata_b1_full.csv` + `numbering_germline.csv`（κ/λ 注釈）
- 関連 Adimab ワークフロー（例: PMC7685319）が同一 papain / CaptureSelect 手順を裏づけ

## 再構成仮定

- CH1 / Cκ / Cλ の **正確な allele は不明** → UniProt 正規配列を使用
- λ は抗体ごとに IGLC を割り当てず **IGLC2 を共通**
- 重鎖末端は papain 産物の正確な長さが不明なため **`EPKSC`（Cys220 まで）** を明示ラベル付きで採用
- クラスは **RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB**（exact とは呼ばない）

## 構造予測の観察（パイロット 12）

| 指標 | 概要 |
|------|------|
| global pLDDT | ≈ 83–88 |
| CH1 / CL pLDDT | ≈ 81–87 |
| CH1–CL 接触 | 32–46 |
| H–L SS（Å） | ≈ 2.3–3.3 |
| severe clash | 0 |
| 判定 | `PASS_FULL_COHORT` |

ESMFold 設定は gate_b2 と整合（`HEAVY:LIGHT`、chunk=48、recycles 既定 4）。GPU1（1080 Ti）は現行 PyTorch 非対応のため **GPU0（3090）** で BioEmu と共存（chunk 縮小）。

---

## 成果物チェック

- `forensic/CONSTRUCT_FORENSIC_AUDIT.md`
- `sequences/*`（監査・政策・324 Fab・FASTA）
- `qc/FAB_SEQUENCE_*` / `ESMFOLD_FAB_PILOT_*`
- `structures/pilot/*.pdb`（12）・`ESMFOLD_FAB_CONFIG.json`
- `comparisons/FV_VS_FAB_*`（パイロット）
- `scripts/run_fennix_v2_on_fab.py`（TmApp 未実行）

**状態（本レポート時点）:** 配列再構成 + パイロット合格。全 324 ESMFold はバックグラウンド実行中 → 完了後に `ORGANIZER_SHEHATA_FAB_RECONSTRUCTION_COMPLETE`。
