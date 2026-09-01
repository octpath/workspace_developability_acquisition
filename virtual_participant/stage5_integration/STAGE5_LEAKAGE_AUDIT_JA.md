# Stage 5 — Leakage Audit

Stage5 では meta learner / calibration / residual model すべて **nested cross-fitting** を使用した。

## 1. Base prediction leakage
- 各 outer fold の test 行は base learner の training に含めない。
- inner OOF 生成時も outer test fold を inner training から除外。

## 2. Meta learner leakage
- meta learner は outer-train 上の inner OOF predictions + labels のみで fit。
- outer-test 行の base predictions は fit に使わず、適用のみ。
- 既存 Stage1–4 の full Primary OOF を meta-training matrix としては使用していない。

## 3. Calibration leakage
- affine calibration も outer nested procedure。inner OOF incumbent prediction から calibration 係数を推定。

## 4. Residual target leakage
- residual target = y − incumbent_inner_oof_pred。各 row の residual target 作成にその row 自身の in-sample prediction は使わない。

## 5. Validation tests
- `tests/test_stage5_leakage.py` で fold 分離・162行一意・tail N=17 等を自動確認。

**Status:** PASS（自動テスト実行後に確定）
