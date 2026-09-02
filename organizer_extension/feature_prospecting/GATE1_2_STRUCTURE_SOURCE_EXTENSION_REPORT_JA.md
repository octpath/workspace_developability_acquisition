# Gate 1.2 — Structure Source Extension Report

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`  
**Boltz完了:** 2026-09-02T15:31:37Z — **324/324 PASS**

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

| Item | Count |
|------|------:|
| MSA ready | 324 |
| mmCIF | 324 |
| PDB | 324 |
| confidence JSON | 324 |
| **PASS** | **324 / 324** |
| FAIL | 0 |

詳細: `structure_sources/boltz2_fv_standard_v1/BOLTZ2_RUN_REPORT_JA.md`

---

## 9. Sequence / chain audit

Competition VH/VL との照合（PDB 復元）:

| Check | Result |
|-------|--------|
| H EXACT_MATCH | 324 / 324 |
| L EXACT_MATCH | 324 / 324 |
| mapping_status PASS | 324 / 324 |

silent fix なし。FAIL / PASS_EXPLAINED = 0。

---

## 10. Confidence/QC summary

target 相関は計算していない（target-blind QC のみ）。

| Metric | min | median | max |
|--------|----:|-------:|----:|
| confidence_score | 0.902 | 0.954 | 0.978 |
| ptm | 0.930 | 0.960 | 0.978 |
| iptm | 0.915 | 0.946 | 0.972 |
| complex_plddt | 0.893 | 0.955 | 0.983 |

QC: 全件 chains=`H,L`、NaN coordinates=0、n_res median=231。

---

## 11. Crosswalk v2

`STRUCTURE_INPUT_CROSSWALK_v2.csv` + `STRUCTURE_INPUT_CROSSWALK_V2_AUDIT.json`

| Generator | PASS |
|-----------|-----:|
| ESMFold | 324 |
| ABodyBuilder2 | 324 |
| Boltz-2 | 324 |

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
