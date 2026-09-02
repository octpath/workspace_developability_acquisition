# VHL-ANGLE_v1

Paper: [Dunbar et al., Protein Eng Des Sel 2013 — ABangle](https://doi.org/10.1093/protein/gzt020)

Software/repository: [jaredsampson/ABangle](https://github.com/jaredsampson/ABangle)（vendored commit `5283cd39`）

**State:** `ORGANIZER_FEATURE_PROSPECTING_GATE2B_VHL_ANGLE_V1_COMPLETE`  
**Numbering:** Chothia via ANARCI 2026.2.13.2

## Bottom line

| Target | Mechanistic prior | Empirical verdict | One-line conclusion |
|--------|-------------------|-------------------|---------------------|
| TmApp | `5_LIKELY_RELEVANT` | `PROMISING_BUT_REDUNDANT` | ESMFold ABangle 6パラメータは median baseline を再現的に下回るが、Round1 residual への増分はない |
| HIC | `3_RELATED_BUT_INDIRECT` | `NO_EVIDENCE_IN_CURRENT_DATA` | 同 feature の cross-endpoint audit で signal なし |

Primary generator: **ESMFold**。ABB2 の ABangle 値は dc/HL 分布が系統的にずれ **FRAGILE**（解釈注意）。

---

## 1. VH–VL orientationとは

抗体 Fv は VH と VL の2ドメインから成る。ABangle は両者の相対姿勢を **5角度（HL, HC1, LC1, HC2, LC2）+ 距離 dc** で絶対座標的に表す（単一 packing angle ではない）。

## 2. Why TmApp

ドメイン接合幾何は Fab の安定性・packing と mechanistically 結びつきやすい（prior `5_LIKELY_RELEVANT`）。ANM（大域力学）とは独立な antibody-specific 軸。

## 3. Why HIC is indirect

HIC 主因は表面疎水。orientation は表面露出を間接的に変え得るのみ（`3_RELATED_BUT_INDIRECT`）。

## 4. Method / references

- Preference path **A**: published ABangle implementation
- Parameters: HL, HC1, LC1, HC2, LC2, dc
- Consensus coreset CA → Bio.PDB Superimposer（本コードパスは TMalign 非使用）
- Crosswalk v2 のみ（独自 path 禁止）

## 5. Three-generator robustness

| Pair | median Spearman |
|------|----------------:|
| ESMFold vs ABB2 | 0.018 |
| ESMFold vs Boltz2 | 0.558 |
| ABB2 vs Boltz2 | 0.035 |

- **Class: `FRAGILE`**（discard しない）
- Extraction: ESMFold **324/324**, Boltz2 **324/324**, ABB2 **323/324**（`ADI-47177` FAIL: coreset atom list size mismatch; DEV）
- ESMFold/Boltz2 の dc 中央値 ≈ 16 Å；ABB2 ≈ 6.8 Å → ABB2 側の numbering/coreset 登録が不安定な可能性が高い

## 6. Standalone TmApp（ESMFold）

| Split | MAE | baseline | Spearman |
|-------|----:|---------:|---------:|
| CV | 3.335 | 3.438 | （正相関条件を満たす） |
| Public | 3.593 | | |
| Private | 3.741 | | |

Standalone: **`REPRODUCIBLE`**

## 7. Incremental TmApp

Round1 ref MAE CV=2.713。delta MAE: CV **+0.016**, Public **−0.053**, Private **+0.061** → **`NO_INCREMENT`**

Residual Ridge audit: **PASS**（`common/ridge_eval.py` 再利用）

## 8. Tm range diagnostic

quintile / low·high bias を保存。v1 内で特徴量は調整しない。

## 9. HIC cross-endpoint

`SECONDARY_CROSS_ENDPOINT_AUDIT` — standalone/incremental とも **NO_SIGNAL / NO_INCREMENT**（delta MAE は正＝悪化）。

## 10. Generator-specific findings

- TmApp generator signal: `NO_SIGNAL`（residual 軸）
- ESMFold / ABB2: standalone REPRODUCIBLE だが incremental なし → `PROMISING_BUT_REDUNDANT`
- Boltz2: `MIXED`（WEAK + CV_ONLY_INCREMENT）

## 11. Confounds / limitations

- ABB2 ABangle は ESMFold/Boltz と整合せず（FRAGILE の主因）
- author residue numbering では framework RMSD QC が NaN（Chothia 再番号前の coreset 不一致）
- 角度 Spearman は raw degrees；同時に circular abs-diff も記録

## 12. Scientific interpretation

事前指定パラメータでは Dev で `dc` / `HC2` 等が TmApp と弱い関連（|ρ|≈0.23）を示す場合があるが、**予測の増分価値は Round1 に対し再現しない**。standalone の baseline 超えは「orientation が全く無情報ではない」ことを示唆しつつ、incumbent 補完としては不十分。
