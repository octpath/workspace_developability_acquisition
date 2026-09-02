# Advanced Batch 2 Report（Gate 2H–2K）

**Batch state:** `ORGANIZER_FEATURE_PROSPECTING_ADVANCED_BATCH2_COMPLETE`  
**Anti-posthoc:** 4× `FEATURE_SPEC` を target 前に SHA-256 freeze（`all_family_specs_frozen_before_any_target_score = true`）。  
Manifest: [`ADVANCED_BATCH2_TARGET_BLIND_FREEZE_MANIFEST.json`](ADVANCED_BATCH2_TARGET_BLIND_FREEZE_MANIFEST.json)

Blocked families: **なし**（全4 family 実行完了）

---

## 現時点で何が効いていそうか

### TmApp — 「結局、何が効いていそうか？」

埋没極性・局所 packing 寄りの **POLAR-SAT** と、弱いが再現する **VHL-ANGLE / PKA（ESMFold）** が依然として最も情報がある。  
本 batch の **INTERFACE-ENERGY（固定座標 FF 代理）** と **3DI-FROZEN** は、現状データでは **明確な TmApp 上積みを示さなかった**。  
→ global dynamics（ANM）より local polar/interface 仮説は残るが、**静的 OpenMM 界面エネルギーや frozen 3Di 埋め込みだけでは足りない**。

### HIC — 「結局、何が効いていそうか？」

**露出芳香族トポロジー（AROMATIC-TOPO）が依然唯一の明瞭な HIC standalone。**  
一般連続疎水場（HYDRO-FIELD）は univariate では弱い相関があるが Ridge では **CV で baseline を安定して超えず**、芳香族特異的 signal を一般疎水に拡張できなかった。  
静電×疎水 co-patch（ELEC-HYDRO-COPATCH）も **疎水単独より悪く、静電は現状 HIC に足さない**。  
→ HIC は「広い疎水場」より **芳香族露出幾何** に寄っている可能性が高い。

---

## TmApp table（全完了 family）

| Family | Prior | Robustness | Standalone (ESMFold) | Incremental | Verdict |
|--------|-------|------------|----------------------|-------------|---------|
| ANM-SPECTRUM | 5 | MODERATE | TEST_ONLY_POSTHOC | CV_ONLY | NO_EVIDENCE_IN_CURRENT_DATA |
| VHL-ANGLE | 5 | PROVISIONAL_TECHNICAL_MAPPING_CONCERN | REPRODUCIBLE | NO_INCREMENT | PROMISING_BUT_REDUNDANT |
| PKA-SHIFT corrected | 5 | FRAGILE | REPRODUCIBLE | WEAK_OR_MIXED | MIXED |
| AROMATIC-TOPO | 3 | MODERATE | WEAK | NO_INCREMENT | MIXED |
| STATIC-SAP | 3 | MODERATE | WEAK | NO_INCREMENT | MIXED |
| VOID-EXPLICIT | 5 | FRAGILE | WEAK | NO_INCREMENT | MIXED |
| POLAR-SAT | 5 | FRAGILE | REPRODUCIBLE | NO_INCREMENT | PROMISING_BUT_REDUNDANT |
| HYDRO-FIELD | 3 | MODERATE | WEAK | NO_INCREMENT | MIXED |
| ELEC-HYDRO-COPATCH | 3 | FRAGILE | TEST_ONLY_POSTHOC | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| INTERFACE-ENERGY | 5 | FRAGILE | NO_SIGNAL | NO_INCREMENT | UNLIKELY_RELEVANT_UNDER_CURRENT_SETUP |
| 3DI-FROZEN | 4 | MODERATE | TEST_ONLY_POSTHOC | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |

## HIC table（全完了 family）

| Family | Prior | Robustness | Standalone (ESMFold) | Incremental | Verdict |
|--------|-------|------------|----------------------|-------------|---------|
| ANM-SPECTRUM | 2 | MODERATE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| VHL-ANGLE | 3 | PROVISIONAL… | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| PKA-SHIFT corrected | 4 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| AROMATIC-TOPO | 5 | MODERATE | REPRODUCIBLE | NO_INCREMENT | **PROMISING_BUT_REDUNDANT** |
| STATIC-SAP | 5 | MODERATE | TEST_ONLY_POSTHOC | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| VOID-EXPLICIT | 2 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| POLAR-SAT | 3 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| HYDRO-FIELD | 5 | MODERATE | TEST_ONLY_POSTHOC | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| ELEC-HYDRO-COPATCH | 4 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| INTERFACE-ENERGY | 3 | FRAGILE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |
| 3DI-FROZEN | 4 | MODERATE | NO_SIGNAL | NO_INCREMENT | NO_EVIDENCE_IN_CURRENT_DATA |

---

## Pre-specified comparisons

### HIC: AROMATIC vs STATIC-SAP vs HYDRO-FIELD vs COPATCH

| Family | HIC verdict | Note |
|--------|-------------|------|
| AROMATIC-TOPO | PROMISING_BUT_REDUNDANT | 唯一の再現 standalone |
| STATIC-SAP | NO_EVIDENCE | 静的 SAP 近似では不足 |
| HYDRO-FIELD | NO_EVIDENCE | 連続疎水場は aromatic を置換できない（univariate ρ≈0.36 あるが Ridge CV 失敗） |
| ELEC-HYDRO-COPATCH | NO_EVIDENCE | 静電 co-localization は疎水単独より悪化 |

**問いへの答え:** (1) HIC signal は芳香族特異的に見える。(2) 一般連続疎水場は現規約では十分な Ridge signal を持たない。(3) いずれも incumbent への incremental なし。

### TmApp: VHL vs POLAR vs INTERFACE vs 3DI（+ANM/PKA/VOID）

| Family | TmApp verdict |
|--------|---------------|
| POLAR-SAT | PROMISING_BUT_REDUNDANT（最有力の物理系） |
| VHL-ANGLE | PROMISING_BUT_REDUNDANT（弱い） |
| PKA corrected | MIXED（ESMFold only） |
| VOID | MIXED / FRAGILE |
| ANM | NO_EVIDENCE |
| INTERFACE-ENERGY | UNLIKELY（固定座標 proxy；GEOMETRY_ARTIFACT_CONCERN） |
| 3DI-FROZEN | NO_EVIDENCE（埋め込みは generator 間 MODERATE だが target 情報なし） |

---

## Family details

### 1. HYDRO-FIELD_v1

- Method status: `TRANSPARENT_LITERATURE_INSPIRED_IMPLEMENTATION`
- Field: Fauchère exponential MLP \(H(s)=\sum_i \pi_i e^{-\alpha r}\)（α=1 Å⁻¹, cutoff 7 Å）on FreeSASA Lee–Richards SAS points
- π scale: Fauchère–Pliska 1983 · Concept: Audry et al. 1986 MLP（関連 DOI: [10.1016/0223-5234(89)90109-8](https://doi.org/10.1016/0223-5234(89)90109-8)）
- Software: [FreeSASA](https://github.com/mittinatten/freesasa) 2.2.1
- Robustness: **MODERATE**
- HIC ESMFold: MAE CV/Pub/Pri **0.527 / 0.540 / 0.504** · TEST_ONLY_POSTHOC · Δ **+0.032 / +0.034 / +0.023** · **NO_EVIDENCE**
- Verdict HIC: **NO_EVIDENCE_IN_CURRENT_DATA** · TmApp secondary: MIXED

### 2. ELEC-HYDRO-COPATCH_v1

- Same surface + H(s); PDB2PQR AMBER pH6.5 + APBS Stage4（I=0.15 M）
- Paper/soft: [APBS](https://doi.org/10.1002/jcc.10344) · [repo](https://github.com/Electrostatics/apbs) · [pdb2pqr](https://github.com/Electrostatics/pdb2pqr)
- Limitation: `ARTIFICIAL_FV_C_TERMINUS_LIMITATION`（`--neutralc` は PARSE のみ；AMBER と併用不可）
- Robustness: **FRAGILE**
- HIC: NO_SIGNAL / NO_INCREMENT → **NO_EVIDENCE**
- **静電は疎水に足さない**（standalone MAE は HYDRO より悪い）

### 3. INTERFACE-ENERGY_v1

- Terminology: **fixed-coordinate VH–VL force-field interaction proxy**（ΔG_binding ではない）
- OpenMM 8.6.0 + `amber14-all.xml` · NoCutoff · no min/MD · iface heavy-atom ≤4.5 Å
- Paper: [OpenMM 7](https://doi.org/10.1371/journal.pcbi.1005659) · [repo](https://github.com/openmm/openmm)
- Robustness: **FRAGILE**
- TmApp: NO_SIGNAL / NO_INCREMENT → **UNLIKELY_RELEVANT_UNDER_CURRENT_SETUP**
- **GEOMETRY_ARTIFACT_CONCERN**: Spearman(E_vdw, clash) ESMFold 0.61 / Boltz2 0.54

### 4. 3DI-FROZEN_v1

- method_id: `FOLDSEEK_3DI__PROSTT5_FROZEN`
- Foldseek `941cd33` structure→3Di（AA→3Di 予測は不使用）→ ProstT5 encoder `<fold2AA>` · HL_CONCAT · nested PCA32+Ridge
- Papers: [Foldseek](https://doi.org/10.1038/s41587-023-01773-0) · [ProstT5](https://doi.org/10.1093/nargab/lqae152) · [ProstT5 repo](https://github.com/mheinzinger/ProstT5) · [HF Rostlab/ProstT5](https://huggingface.co/Rostlab/ProstT5)
- Embedding robustness: **MODERATE**（token id≈0.77–0.82, cosine≈0.99, dist Spearman≈0.71–0.75 < ROBUST の 0.80）
- TmApp/HIC: **NO_EVIDENCE_IN_CURRENT_DATA**

---

## Generator dependence

- INTERFACE / COPATCH: **FRAGILE**（構造生成器依存大）
- HYDRO / 3DI: **MODERATE**
- INTERFACE は clash と E_vdw の相関が ESMFold/Boltz で強く、予測幾何アーティファクト懸念

---

## Recommend next（開始しない）

情報利得優先:

1. **TITRATION-SHAPE** — PKA MIXED の延長で、形状×滴定の別仮説
2. **OPENMM-STRAIN** — 本 batch の固定座標が失敗したため、緩和ひずみが必要か検証（INTERFACE と明確に分離）
3. **SURFACE-DL** — HIC で芳香族を超える学習表面表現の探索（license 確認必須）
4. ENCOM-CHEM / GEARNET — ANM/3Di が空振り後の優先度は相対的に低い

---

**Final batch state:** `ORGANIZER_FEATURE_PROSPECTING_ADVANCED_BATCH2_COMPLETE`
