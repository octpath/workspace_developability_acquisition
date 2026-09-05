# Shehata Fab 再構成レポート

**開始時 git HEAD:** `f1a40e32b06f40dc64a6375f30b7aef78056e262`（`main`）  
**最終状態:** `ORGANIZER_SHEHATA_FAB_RECONSTRUCTION_COMPLETE`  
**完了コミット（構造/QC）:** `2d05abae`  
**最終コミット:** `259f4af0`（本ヘッダ修正含む後続コミットで更新）  
**作業日:** 2026-09-05  
**出力:** `organizer_extension/feature_prospecting/fab_reconstruction/`

---

## 冒頭回答（14項目）

1. **正確な実験 Fab 配列を同定できたか?** → **No**
2. **再構成クラス** → **`RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB`**
3. **使用 CH1** → UniProt P01857 由来 CH1+`EPKSC`（`IGHG1_CH1_EPKSC_UNIPROT_P01857`）
4. **Cκ** → UniProt P01834（IGKC）
5. **Cλ** → UniProt P0DOY2（IGLC2・λ共通）
6. **κ / λ / unresolved** → **238 / 86 / 0**
7. **VH/VL 境界補正** → **不要**（324/324 OK）
8. **papain / hinge 末端** → papain 2 h/30°C・Protein A・CaptureSelect IgG-CH1 は HIGH。正確な HC C 末端は未確定（`EPKSC` は再構成仮定）
9. **12-Fab パイロット** → **`PASS_FULL_COHORT`**
10. **全件 ESMFold** → **324 / 324 成功**（失敗 0）
11. **CH1/CL 界面** → 接触（8 Å）中央 ≈39；系統的な乖離なし
12. **H–L Cys 距離** → 全体中央値 **2.80 Å**（κ 2.74 / λ 3.34）。近接 ≠ 共有結合。FeNNix 緩和後 QC 対象。
13. **Fv vs Fab（324）** → 結合 Fv CA RMSD 中央値 **0.58 Å**（q90 1.09）。配向 COM 中央値 **0.50 Å**。**大筋で Fv 幾何は変わらない**（外れ値 ADI-47265 は QC 扱い）。
14. **FeNNix 前の残存不確実性** → allele/接合/papain 末端；H–L ジスルフィドをエネルギー特徴が支配しないよう緩和後検証が必要。

---

## 事実（実験ソース）

- 組換え **IgG1**（yeast）・VH/VL single-cell PCR → homologous recombination（HIGH）
- Fab = papain（2 h, 30°C）→ Protein A → CaptureSelect IgG-CH1（HIGH）
- 凍結配列: `gate_b1/data/shehata_b1_full.csv` + `numbering_germline.csv`

## 再構成仮定

- CH1/Cκ/Cλ allele 未確定 → UniProt 正規配列
- λ は IGLC2 を全例共通
- 重鎖末端 `EPKSC` は化学的完全性のための仮定（実験正確末端ではない）
- クラス: **RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB**

## 構造予測（全 324）

| 指標 | 中央値 / 要約 |
|------|----------------|
| global pLDDT | median 86.7（P10–P90 84.7–88.3） |
| severe clash | **0**（全例） |
| CH1–CL 接触 | median 39 |
| H–L Sγ–Sγ | median 2.80 Å；<2.6Å 12.3%；≥3.0Å 25.6% |
| Fv vs Fab combined RMSD | median 0.58 Å；q90 1.09；max 24.72 |
| 芳香族露出 SASA Δ | median -30.5 Å² |

ESMFold 設定: gate_b2 相当（`H:L`、chunk=48、recycles 既定）。BioEmu と GPU0 共存。GPU1 は非対応。

## Fv 幾何への影響（結論）

**No (bulk).** Adding CH1/CL does not materially rewrite predicted Fv backbone geometry for the cohort: combined Fv CA RMSD median **0.58 Å**, IQR 0.29, q90 **1.09 Å**. VH/VL orientation COM shift after VH-align median **0.50 Å**. Interface contact Δ median **0.0**. Exposed aromatic SASA Δ (Fab Fv-portion − Fv) median **-30.5 Å²** (constant domains can bury some aromatics). One extreme outlier (ADI-47265, combined RMSD 24.7 Å) should be treated as QC failure, not biology.

## FeNNix-on-Fab への注意

H–L Cys 近接は **入力 QC / 緩和後チェック**。共有結合形成を ESMFold 結果から断定しない。ジスルフィド歪みがエネルギー特徴を支配しないこと。

TmApp スコアリングは本ブランチでは **未実施**（target-blind 維持）。
