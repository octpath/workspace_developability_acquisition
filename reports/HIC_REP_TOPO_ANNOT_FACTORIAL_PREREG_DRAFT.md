# HIC Representation × Topology × Annotation Factorial — PREREGISTRATION DRAFT

**STATUS:** `READY_FOR_FORMAL_FREEZE` — **not** `PREREGISTERED_FROZEN`  
**Training / EXP-H issuance / registry write:** **NOT authorized** until human formal-freeze approval.

| Field | Value |
|-------|--------|
| Evaluation freeze SHA | **`379e0751a93c2af8f6fbfeedaad4d72f3556996b`** |
| Evaluation contract | `reports/HIC_EVALUATION_PROTOCOL_FREEZE.md` |
| Design | **FULL FACTORIAL 10 × 5 × 4 = 200** |
| Cell manifest | `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_CELL_MANIFEST.csv` |
| **Manifest SHA256** | **`ffdf88d6d59cc6af9864fa041c4a057d398f523ce724fb55d392c50282e5f7cf`** |
| REUSE audit | `reports/HIC_FACTORIAL_REUSE_AUDIT.csv` |
| Embedding inventory | `reports/HIC_FACTORIAL_EMBEDDING_INVENTORY.csv` |
| Surface / physics aux | **Excluded** |

---

## Gate summary (this finalization)

| Check | Result |
|-------|--------|
| Manifest rows / unique cells | **200 / 200 PASS** |
| REUSE confirmed | **0** |
| REUSE → RETRAIN_REQUIRED | **6/6** (share_hl_encoder & pair_interaction_mode UNKNOWN) |
| Expected new training | **200** |
| Embedding blockers | **None** (all 9 PLM assets N=324; Scratch OK) |
| Topology implementation blockers | **None** (TmApp flag semantics available; HIC must copy them) |
| Annotation blockers | **None** (annotations.parquet 324 Abs; model modes match) |
| Bootstrap structural audit | **PASS** |
| Pub/Priv dependency in manifest | **False** |
| Surface flags in manifest | **False** |

---

## 1. Status / scope

Human-approved Full factorial. This document is the **pre-freeze gate** output.  
Still forbidden: training, GPU jobs, Public/Private scoring, midway MAE cell selection, factor changes, surface addition, formal FROZEN prereg commit, EXP-H issuance.

---

## 2. Evaluation freeze SHA

All cells inherit `reports/HIC_EVALUATION_PROTOCOL_FREEZE.md` at **`379e0751…`**.  
Conflict → freeze wins.

---

## 3. Scientific questions

Unchanged from prior draft: HIC headroom in Rep/Topo/Annot without surface; context contrasts; post-hoc TmApp comparison only.

---

## 4. Complete 200-cell design

| | |
|--|--:|
| Total cells | **200** |
| CONFIRMED_REUSE | **0** |
| NEW training | **200** |

Platform lock (at formal prereg): `DL_FOLDLOCAL_COSINE_V3`, seed **101**, `d_model=128`, merge **mean**, pooling REG, fold-local LR, `share_hl_encoder=True` **explicit**, Primary/Shadow V3 schemes.

Manifest asserts: 10 reps × 5 topos × 4 annots; no duplicates.

---

## 5–7. Factor / provenance / hierarchy

As Design A / TmApp terminology. Context metadata over allocation names.  
Families (8) + AbLang2/CurrAb contexts analyzed hierarchically without collapsing the 10-level grid.

---

## 8. Reuse policy — FINAL GATE OUTCOME

| Code | Cell | audit_verdict | formal disposition |
|------|------|---------------|--------------------|
| H056 | ESM2×SEP×FULL | UNRESOLVED | **RETRAIN_REQUIRED** |
| H059 | ESM2×JOINT×FULL | UNRESOLVED | **RETRAIN_REQUIRED** |
| H061 | ESM2×REG-SEP×FULL | UNRESOLVED | **RETRAIN_REQUIRED** |
| H070 | Scratch×SEP×FULL | UNRESOLVED | **RETRAIN_REQUIRED** |
| H073 | Scratch×JOINT×FULL | UNRESOLVED | **RETRAIN_REQUIRED** |
| H075 | Scratch×REG-SEP×FULL | UNRESOLVED | **RETRAIN_REQUIRED** |

**Reason:** `share_hl_encoder` and `pair_interaction_mode` not recorded in config/run_state → UNKNOWN. Rule: unknown ≠ PASS.  
Other fields (target HIC, arch flags, merge mean, seed 101, V3, TEST_mean=mean(P,S), OOF artifacts) were consistent but insufficient for CONFIRMED_REUSE.

→ **No REUSE in formal plan; train all 200.**

---

## 9. Exact primary contrasts

* Topology − SEP (fixed Rep×Annot)  
* Annotation − BASE (fixed Rep×Topo)  
* PLM − Scratch (fixed Topo×Annot; separate from absolute ranking)  
* PAIRED_NATIVE − SEPARATE_CHAIN (AbLang2 & CurrAb; fixed Topo×Annot)  

Secondary: `Δ_P`/`Δ_S` sign agreement; improvement ⇒ both `<0`. Does not replace Primary `TEST_mean`.

---

## 10. Interactions

Level-1: Rep×Topo, Rep×Annot, Topo×Annot.  
Exploratory: AbLang2/CurrAb Family×Context×Topo/Annot.  
Conclusion hierarchy: factor patterns > predefined contrasts > single best cell (descriptive).

---

## 11. Primary / Secondary / Diagnostic / External

| Role | Rule |
|------|------|
| Primary | `TEST_mean = mean(TEST_P, TEST_S)` |
| Secondary | directional Δ replication |
| HIGH-tail | `HIC>11.5`; record only; no selection |
| External | GEN_0001 after internal freeze only; embargo on selection/reinterpretation |

Surface excluded. TmApp results do **not** alter HIC selection.

---

## 12. Bootstrap / uncertainty — AUDITED

| Item | Definition |
|------|------------|
| Method | Antibody-level paired residual bootstrap (`paired_boot` / `did_boot` family) |
| Resampling unit | **antibody `id`** (Dev OOF vector) |
| N_BOOT | **2000** |
| Seed | **101** |
| CI | percentile **2.5% / 97.5%** |
| Pairing | same id index for candidate vs baseline predictions and `y=HIC` |
| Primary vs Shadow | **bootstrap separately** on `oof_test_primary` / `oof_test_shadow` |
| TEST_mean reporting | rank/select on mean of scheme MAEs; for uncertainty on mean-level contrasts, report both scheme bootstraps + require sign agreement; do **not** treat folds as i.i.d. resample units |
| Forbidden | resampling folds as independent samples; mixing Public/Private ids into internal bootstrap |

**Structural PASS:** HIC V3 OOF (`EXP-H056`) has 162 Dev ids, Primary/Shadow aligned, subset of competition Dev, HIC labels present — TmApp bootstrap math applies. Analysis code must load **HIC** paths/labels (not copy TmApp blindly).

---

## 13. Cross-target TmApp vs HIC

After HIC internal freeze only. Align names/contrasts. Never use TmApp to select HIC cells.

---

## 14. Technical gates (MAE-blind)

1. REUSE checksum — **done** → 0 reuse  
2. Representation asset inventory — **PASS**  
3. Annotation wiring smoke — config/asset **PASS**; optional forward smoke at execution  
4. Topology wiring smoke — semantics **PASS** vs TmApp flags; smoke at execution  
5. Primary/Shadow prediction smoke — at execution  
6. Artifact completeness / nonzero variance — at execution  
7. Analysis pipeline dry-run — at execution  

---

## 15–16. Failure / missing-cell policy (FROZEN for this draft)

### Technical failure (OOM, code, corrupt asset)

Repair and **re-run the same cell**.

### Scientifically bad MAE

**Keep the cell.** Not a failure.

### Asset impossible

Mark **BLOCKED**; record reason. **No substitution** of another representation.

### Missing cells

Aim for complete 200 before analysis. If unavoidable BLOCKED remain: document broken contrasts; **do not freeze strong factor-level conclusions** until human review.

---

## 17. No-midway-selection

All 200 predefined. No dropping factors from intermediate scores. No surface injection mid-batch.

---

## 18. Public / Private embargo

GEN_0001 only after internal freeze. No selection / HP / reinterpretation use.

---

## 19. Required artifacts

Manifest + SHA256; reuse audit; embedding inventory; per-cell configs/OOF; master table; contrasts+bootstrap; HIGH-tail diagnostics; internal freeze yaml; optional external table post-embargo.

---

## 20. Experiment ID policy draft (**no issuance**)

| Item | Value |
|------|--------|
| Last issued HIC code | **EXP-H139** |
| Next free | **EXP-H140** |
| Contiguous range for 200 new | **EXP-H140 … EXP-H339** |
| If any future REUSE revived | fewer new IDs; not applicable now |

**Do not reserve/write registry in this gate.**

---

## 21. Binding commitments (to sign at formal freeze)

* Full 200; no mid-batch factor drop  
* Primary TEST_mean; Secondary Δ signs; HIGH-tail & Pub/Priv non-selection  
* Context metadata authority  
* TmApp ≠ HIC selector  
* Surface excluded  
* Missing/failed policy as §15–16  

---

## 22. Remaining blockers before formal freeze

1. **Human approval** to set status `PREREGISTERED_FROZEN` and commit  
2. Authorize **EXP-H140–H339** issuance / registry write  
3. Pin formal prereg git SHA + re-hash manifest at freeze commit  
4. Confirm runner will set `share_hl_encoder=True` and TmApp-identical topology flags explicitly  
5. Optional: enrich AbLang2/ESM2 metadata context fields (non-blocking; inventory OK)

**No embedding blockers. No topology/annotation definition blockers. Bootstrap PASS.**

---

**STOP — await human formal-freeze approval.**
