# PKA-SHIFT_v1

Paper: [Olsson et al., JCTC 2011 — PROPKA3](https://doi.org/10.1021/ct100578z)  
Related: [Sondergaard et al., JCTC 2011](https://doi.org/10.1021/ct200133y)  
Repository: [jensengroup/propka](https://github.com/jensengroup/propka)  
Documentation: [github.com/jensengroup/propka](https://github.com/jensengroup/propka)

Software: **PROPKA 3.5.1** (LGPL v2.1; Round1 `.venv_stage4`)

## Bottom line

| Target | Mechanistic prior | Empirical verdict | One-line conclusion |
|--------|-------------------|-------------------|---------------------|
| TmApp | `5_LIKELY_RELEVANT` | **MIXED** | ESMFold 25-ΔpKa Ridge is standalone-reproducible vs median baseline, but residual increment is weak/mixed; ABB2/Boltz do not reproduce |
| HIC | `4_PLAUSIBLY_RELEVANT` | **NO_EVIDENCE_IN_CURRENT_DATA`** | Same frozen features show no standalone/incremental HIC signal |

Pre-Gate2C audit: [GATE2AB_CONSISTENCY_AUDIT.md](../GATE2AB_CONSISTENCY_AUDIT.md) — common Ridge infra **PASS**; VHL ABB2 → `VHL-ANGLE_v1_TECHNICAL_REVIEW`.

---

## 1. pKa shiftとは何か

同じ Asp や Glu でも、周囲の電荷・埋もれ方・水素結合環境によって pKa が通常値（model compound）からずれる。そのずれ（ΔpKa）を、局所的な特殊な電気的環境・フラストレーションの指標として使う。

---

## 2. Why TmApp

強い pKa シフトは埋没電荷や異常な静電環境を示唆し、Fv/Fab の熱安定性（TmApp）と直接結びつきやすい。

## 3. Why HIC is plausible but less direct

電荷／静電環境は表面相互作用や疎水露出を変え得るが、HIC の主因は疎水面パッチであり、ΔpKa との関係は一段間接的（`4_PLAUSIBLY_RELEVANT`）。

---

## 4. Method / references

- Canonical method: **PROPKA only**（pKAI / PypKa は未実行 → 将来 `PKA-METHOD-SENSITIVITY_v1`）
- ΔpKa = predicted − model_pKa（`propka.cfg` 凍結値: ASP 3.8, GLU 4.5, HIS 6.5, CYS 9.0, TYR 10.0, LYS 10.5, ARG 12.5）
- 前処理: 生成構造そのまま（最小化なし）
- Titration-curve（Q(pH), pI）は **含めない**（将来 `TITRATION-SHAPE_v1`）

## 5. Residue / termini policy（重要）

予測構造は **孤立 VH/VL** であり、実験 Fab では VH→CH1 / VL→CL へ続く。したがって人工的な Fv **N/C 末端**（PROPKA の `N+` / `C-`）の pKa は canonical 25特徴から **除外**（QC のみ）。

側鎖セット: ASP, GLU, HIS, LYS, ARG, TYR, CYS。

マッピング鍵: `(id, chain H/L, sequence_index, AA)` — PDB resseq は使わない。CDR は競争配列上の **IMGT/ANARCI**（`cdr_sequence_index_imgt.csv`）。Interface: CA–CA ≤ 5 Å。Buried: RASA < 0.20（Stage4）。

Extraction: **324/324 × 3**。AA mismatch 0。末端除外は抗体あたり通常 2–4。

## 6. Three-generator robustness

| Pair | median feature Spearman |
|------|-------------------------|
| ESMFold–ABB2 | 0.449 |
| ESMFold–Boltz2 | 0.631 |
| ABB2–Boltz2 | 0.508 |

**Class: FRAGILE**（捨てない。pKa は構造敏感な科学的結果）。

Residue-level ΔpKa（同一 sequence_index）: per-ab median Spearman ≈ **0.80 / 0.87 / 0.84** — 残基対応は比較的良いが、抗体要約特徴はより fragile。

定数特徴の注意: ほぼ全抗体でジスルフィド Cys が predicted_pKa=99.99 → `max_abs_delta_pKa` 等 = **90.99**（分散ゼロ）。v1 では変更しない（confound / limitation）。

## 7. Standalone TmApp（Ridge, nested Primary CV）

| Generator | CV MAE (base) | Public | Private | Class |
|-----------|---------------|--------|---------|-------|
| ESMFold | **3.268** (3.438) | 3.747 (3.784) | 3.770 (3.772) | **REPRODUCIBLE** |
| ABB2 | 3.551 (3.438) | 3.933 | 3.679 | TEST_ONLY_POSTHOC |
| Boltz2 | 3.403 (3.438) | 3.889 | 4.136 | CV_ONLY |

## 8. Incremental TmApp（CANONICAL_RESIDUAL_RIDGE）

| Generator | ΔCV | ΔPublic | ΔPrivate | Class |
|-----------|-----|---------|----------|-------|
| ESMFold | +0.074 | **−0.026** | **−0.029** | WEAK_OR_MIXED_INCREMENT |
| ABB2 | +0.113 | +0.103 | −0.080 | NO_INCREMENT |
| Boltz2 | +0.004 | +0.153 | +0.019 | NO_INCREMENT |

Residual Ridge audit: **PASS**。

## 9. Which physical summaries associate with Tm

ESMFold Dev univariate |Spearman| は全体に弱い（例: `mean_abs_delta_pKa` ≈ 0.11, `cdr_max_abs_delta` ≈ 0.12, `buried_mean_abs_delta` ≈ 0.10）。単変量で「効く角度」を選ばない。関連の存在と Ridge 予測力は区別する。

## 10. Tm range diagnostic

候補 OOF: low-Tm bias ≈ +4.08, high-Tm bias ≈ −3.39; pred SD / true SD ≈ 0.29。range compression は残存。強い ΔpKa 要約だけでは incumbent が見逃す低安定極端を十分には補えない。

## 11. HIC cross-endpoint

全 generator: standalone **NO_SIGNAL**, incremental **NO_INCREMENT** → **NO_EVIDENCE_IN_CURRENT_DATA**。  
（候補の high-tail ROC 等は参考診断のみ；MAE は悪化。）

## 12. Generator-specific findings

- TmApp residual axis label: **ESMFOLD_ONLY_SIGNAL**（WEAK_OR_MIXED on ESMFold only）
- HIC: **NO_SIGNAL**
- Primary empirical reading uses ESMFold (sequence-mapped, termini-excluded canonical path)

## 13. Confounds

- ジスルフィド Cys → 定数 max|ΔpKa|
- PROPKA 経験式依存（方法感度は未テスト）
- Fv-only vs Fab 静電環境
- 要約特徴の cross-generator FRAGILE（残基レベルはより一致）
- 組成／サイズ QC は canonical に入れていないが、完全分離ではない

## 14. Scientific interpretation

Mechanistic prior（TmApp 5）に対し、ESMFold 上では ΔpKa 要約に **standalone の再現可能な弱い予測力**がある一方、Round1 residual への安定した上積みはなく、他 generator では再現しない。よって empirical は **MIXED**。HIC への交差シグナルは現データでは見えない。

## 15. Limitations

- PROPKA-only; no titration-shape features
- No post-hoc removal of Cys-99.99 / max features in v1
- Public/Private 既知（ORGANIZER-EXPLORATORY）

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_GATE2C_PKA_SHIFT_V1_COMPLETE`
