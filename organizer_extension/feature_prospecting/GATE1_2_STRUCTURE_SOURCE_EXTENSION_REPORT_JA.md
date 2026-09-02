# Gate 1.2 — Structure Source Extension Report

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`（予測完了後に最終数値を同期）

---

## 1. Gate1.2目的

feature family 実験前に:

1. CANONICAL_RESIDUAL_RIDGE の outer-CV leakage を修正
2. mechanistic relevance を 5段階 numeric display に統一
3. Boltz-2 を第三 structure generator として追加・凍結
4. structure crosswalk を 3-generator（v2）へ拡張

target-based feature experiment / TmApp·HIC training は行わない。

---

## 2. Residual Ridge CV correction

Gate1.1 の「outer_train residual = y − global frozen OOF」は、outer-heldout fold のラベルが global OOF に混入し得るため **PRIMARY 禁止**。

Gate1.2:

- outer_test baseline: frozen Round1 OOF（可）
- outer_train residual: **outer_train のみ**で Round1 reference recipe を cross-fit
- outer_test を reference cross-fit に使わない
- Test: Dev162 frozen OOF residual で fit（Test labels 不使用）
- global OOF shortcut は `NON_NESTED_DIAGNOSTIC_ONLY` のみ

詳細: `SIGNAL_CLASSIFICATION_SPEC.md` §5。

---

## 3. Mechanistic relevance 5-level scale

| score | display |
|------:|---------|
| 5 | `5_LIKELY_RELEVANT` |
| 4 | `4_PLAUSIBLY_RELEVANT` |
| 3 | `3_RELATED_BUT_INDIRECT` |
| 2 | `2_UNLIKELY_PRIMARY` |
| 1 | `1_NO_CLEAR_MECHANISTIC_LINK` |

empirical verdict には numeric を付けない。

---

## 4. Current relevance map

| Family | TmApp prior | HIC prior |
|--------|-------------|-----------|
| ANM-SPECTRUM | 5_LIKELY_RELEVANT | 2_UNLIKELY_PRIMARY |
| VHL-ANGLE | 5_LIKELY_RELEVANT | 3_RELATED_BUT_INDIRECT |
| PKA-SHIFT | 5_LIKELY_RELEVANT | 4_PLAUSIBLY_RELEVANT |
| 3DI-FROZEN | 4_PLAUSIBLY_RELEVANT | 4_PLAUSIBLY_RELEVANT |
| AROMATIC-TOPO | 3_RELATED_BUT_INDIRECT | 5_LIKELY_RELEVANT |
| STATIC-SAP | 3_RELATED_BUT_INDIRECT | 5_LIKELY_RELEVANT |
| VOID-EXPLICIT | 5_LIKELY_RELEVANT | 2_UNLIKELY_PRIMARY |
| POLAR-SAT | 5_LIKELY_RELEVANT | 3_RELATED_BUT_INDIRECT |
| OPENMM-STRAIN | 4_PLAUSIBLY_RELEVANT | 1_NO_CLEAR_MECHANISTIC_LINK |
| ENCOM-CHEM | 5_LIKELY_RELEVANT | 2_UNLIKELY_PRIMARY |
| INTERFACE-ENERGY | 5_LIKELY_RELEVANT | 3_RELATED_BUT_INDIRECT |
| HYDRO-FIELD | 3_RELATED_BUT_INDIRECT | 5_LIKELY_RELEVANT |
| ELEC-HYDRO-COPATCH | 3_RELATED_BUT_INDIRECT | **4_PLAUSIBLY_RELEVANT**（Gate1.2 修正） |
| GEARNET-FROZEN | 4_PLAUSIBLY_RELEVANT | 4_PLAUSIBLY_RELEVANT |
| SURFACE-DL | 3_RELATED_BUT_INDIRECT | 4_PLAUSIBLY_RELEVANT |
| SHORT-ENSEMBLE | 4_PLAUSIBLY_RELEVANT | 3_RELATED_BUT_INDIRECT |
| MLFF | 4_PLAUSIBLY_RELEVANT | 1_NO_CLEAR_MECHANISTIC_LINK |

---

## 5. Boltz-2 software/version

| Item | Value |
|------|-------|
| Package | `boltz==2.2.1`（`pip install boltz[cuda] -U`） |
| Python | 3.12（`.venv_boltz`） |
| Torch | 2.13.0+cu130 |
| GPU | NVIDIA GeForce RTX 3090 |
| Checkpoint | `~/.boltz_cache` → `/workspace_developability_acquisition/.boltz_cache/boltz2_conf.ckpt` |
| License | MIT（code + weights; official README） |

CLI discrepancy: help 文が checkpoint「Boltz-1 default」と書くが、`--model` default は `boltz2`。installed behavior を正とする。

---

## 6. Boltz-2 frozen inference protocol

`BOLTZ2_FV_STANDARD_v1`（`BOLTZ2_STRUCTURE_SPEC.json`）:

- recycling_steps=3, sampling_steps=200, diffusion_samples=1
- step_scale=1.5（Boltz-2 official default）
- output_format=mmcif（native）+ deterministic PDB conversion
- seed=42, num_workers=0（/dev/shm=64MB 制約回避）

---

## 7. MSA / template / potentials policy

| Policy | Freeze |
|--------|--------|
| MSA | enabled（ColabFold `https://api.colabfold.com`, pairing=`greedy`） |
| MSA generation | `precompute_msa.py` → CSV embed in YAML（resume-safe; still MSA-enabled） |
| Templates | **none** |
| use_potentials | **false** |
| NOT RUN | `BOLTZ2_NOMSA_v1`, `BOLTZ2_POTENTIALS_v1`, `BOLTZ2_MULTISAMPLE_v1` |

---

## 8. Boltz-2 prediction completion

実行中。最新カウントは `structures_mmcif/` と `BOLTZ2_STRUCTURE_AUDIT.json` を参照。

目標: **324/324 PASS**。未達の場合は success/fail ID を報告し feature experiment に進まない。

---

## 9. Sequence / chain audit

成功構造について H/L を competition VH/VL と照合。ADI-37123 は EXACT_MATCH（H/L）。全件は audit スクリプトで集計。

---

## 10. Confidence/QC summary

target 相関は計算しない。confidence_score / ptm / iptm / complex_plddt 等を JSON 保存。

---

## 11. Crosswalk v2

`STRUCTURE_INPUT_CROSSWALK_v2.csv` — ESMFold/ABB2 列を維持し Boltz-2 列を追加。

---

## 12. Three-generator robustness contract

Primary: **minimum pairwise median Spearman** across 3 pairs。  
ROBUST / MODERATE / FRAGILE。ensemble feature はまだ作らない。

---

## 13. DeepResearch provenance

`DEEP_RESEARCH_FULL_REPORT_NOT_AVAILABLE_IN_REPO`  
（`references/DEEP_RESEARCH_PROVENANCE.md`）

---

## 14. Next step

Human review → first-wave family（ANM-SPECTRUM 等）開始は **自動開始しない**。
