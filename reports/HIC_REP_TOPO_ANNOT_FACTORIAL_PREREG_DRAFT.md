# HIC Representation × Topology × Annotation Factorial — PREREGISTRATION DRAFT

**STATUS:** `DRAFT_ONLY` — **not** `PREREGISTERED_FROZEN`  
**This document does not authorize training, ID issuance, or registry mutation.**

| Field | Value |
|-------|--------|
| Evaluation freeze SHA | **`379e0751a93c2af8f6fbfeedaad4d72f3556996b`** |
| Evaluation contract | `reports/HIC_EVALUATION_PROTOCOL_FREEZE.md` |
| Design proposal | `reports/HIC_FACTORIAL_DESIGN_PROPOSAL.md` |
| Final design | **Design A — Full 10 × 5 × 4 = 200** |
| Target | **HIC only** |
| Surface / physics aux | **Excluded**（separate future physics program） |

TmApp reference definitions: `developability_drilldown/technical_report/tmapp_factorial/terminology.yaml`；`results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_*`。

---

## 1. Status / scope

* Draft for human review after adopting **Full factorial**.  
* No EXP-H codes reserved or issued in this draft.  
* No embeddings extracted, no GPU jobs, no Public/Private scoring.  
* Formal freeze requires a later commit that sets status to `PREREGISTERED_FROZEN` with plan SHA256 + binding commitments.

---

## 2. Evaluation freeze SHA

All cells inherit:

* Primary / Secondary / HIGH-tail / External rules from  
  **`reports/HIC_EVALUATION_PROTOCOL_FREEZE.md`**  
  at commit **`379e0751a93c2af8f6fbfeedaad4d72f3556996b`**.

Any conflict between this draft and that freeze → **freeze wins**.

---

## 3. Scientific questions

1. Without surface/physics features, where is reproducible HIC headroom across **Representation / Topology / Annotation**?  
2. Are topology gains representation-dependent on HIC as on TmApp?  
3. Does annotation (BASE→IMGT/REGION/FULL) help, harm, or interact on HIC given historical FULL-only V3 work?  
4. For AbLang2 and CurrAb, how does **PAIRED_NATIVE vs SEPARATE_CHAIN** change HIC error surfaces relative to downstream topologies?  
5. After HIC internal freeze, how do factor patterns compare to the TmApp 200-cell factorial (**post hoc only**; never for HIC cell selection)?

Non-questions for this batch: surface patch / SASA / HSP late fusion; leaderboard optimization; HIGH-tail as selection objective.

---

## 4. Complete 200-cell design

| Factor | n | Levels |
|--------|--:|--------|
| Representation | 10 | see §5–6 |
| Topology | 5 | SEP, JOINT, REG-SEP, XREG, FUSE |
| Annotation | 4 | BASE, IMGT, REGION, FULL |
| **Cells** | **200** | full Cartesian product |

| Execution | Count |
|-----------|------:|
| Total | 200 |
| REUSE (after checksum) | **0–6** |
| New training | **194–200** |

Platform (intended; locked at formal prereg):

* `DL_FOLDLOCAL_COSINE_V3`  
* seed **101**  
* `d_model=128`, `n_layers=2`, merge **mean**, pooling **REG**  
* fold-local LR on VAL only；`no_full_dev_refit=true`  
* Primary / Shadow schemes as in V3  

---

## 5. Exact factor definitions

### 5.1 Topology（TmApp terminology 正本）

| ID | Meaning (abbrev.) |
|----|-------------------|
| **SEP** | Separate H/L；no downstream H/L interaction before final merge |
| **JOINT** | Unrestricted joint Transformer over H/L residues + REG_H/REG_L |
| **REG-SEP** | Joint residues；chain-specific REGs |
| **XREG** | Separate encode then REG reads opposite-chain residues |
| **FUSE** | Separate encode then REG↔REG via frozen D3 (`pair_interaction_mode=d3_symmetric_mlp`) |

**REG** = learned chain-level token gathering residue information in the downstream Transformer（do not call “summary”）.

Implementation flags follow TmApp factorial `topology_flags` / ARCH-1/3/4/7/D3 mapping with **mean** merge only.

### 5.2 Annotation

Sequence-position and chain-ID embeddings **on in all cells**. Plus:

| ID | IMGT | CDR/FR region |
|----|------|---------------|
| BASE | off | off |
| IMGT | on | off |
| REGION | off | on |
| FULL | on | on |

---

## 6. Representation provenance

Authority: **metadata `representation_context`**, not historical allocation nicknames.

| Display name | Typical internal / subdir | Context |
|--------------|---------------------------|---------|
| Scratch | — | LEARNED_AA |
| AbLingua | ablingua600m | SEPARATE_CHAIN |
| AbLang1 | ablang1 | SEPARATE_CHAIN |
| AbLang2（鎖別推論） | `ablang2`（alloc historically `ablang2_paired`） | **SEPARATE_CHAIN** |
| AbLang2（H/Lペア推論） | `ablang2_unpaired` | **PAIRED_NATIVE** |
| ESM-1b | esm1b | SEPARATE_CHAIN |
| ESM-2 | esm2 | SEPARATE_CHAIN |
| ESM-C 600M | esmc600m | SEPARATE_CHAIN |
| CurrAb（鎖別推論） | currab_unpaired | **SEPARATE_CHAIN** |
| CurrAb（H/Lペア推論） | currab | **PAIRED_NATIVE** |

Matched AbLang2 contexts share the same AbLang2 checkpoint family；CurrAb contexts share revision **`92e28534663e163f1b398f773b3fe041085737d9`** (as in TmApp factorial). Formal prereg must pin content hashes / manifest entries for N=324 coverage.

**No new PLMs** beyond this list.

---

## 7. Hierarchical family / context definition

### Grid layer（primary factorial）

Treat **10 representation levels** as the factorial factor (TmApp parity).

### Hierarchy layer（additional analysis）

**Families (8):** Scratch, AbLingua, AbLang1, AbLang2, ESM-1b, ESM-2, ESM-C, CurrAb  

**Contexts (where applicable):** SEPARATE_CHAIN, PAIRED_NATIVE  

| Family | Levels in 10-grid |
|--------|-------------------|
| AbLang2 | 鎖別 + H/Lペア |
| CurrAb | 鎖別 + H/Lペア |
| Others | single context each |

**Do not conflate:** reporting “best of 10 representations” vs “AbLang2 family context effect.” Both are allowed；separate sections.

---

## 8. Reuse policy

### Provisional candidates

`EXP-H056`, `EXP-H059`, `EXP-H061`, `EXP-H070`, `EXP-H073`, `EXP-H075`

### Required exact match checklist

target；representation；embedding source + hash/metadata；topology implementation flags；annotation；**`share_hl_encoder`**；**`pair_interaction_mode`**；`d_model`；merge mode；seed；LR procedure；V3 protocol；Primary/Shadow definitions；evaluation implementation.

### Current draft finding

`share_hl_encoder` / explicit `pair_interaction_mode` are **not recorded** in candidate yamls/run_state → under the rule “不明なら REUSE しない,” **confirmed REUSE count = 0** until the REUSE checksum gate resolves each field from training-time evidence.

If any check fails → cell becomes **new training**.  
Planning envelope: **reuse 0–6；new 194–200**.

Surface/late-fusion H082–H139: **never** reusable for this factorial.

---

## 9. Exact primary contrasts（predefined）

All contrasts use MAE；**negative Δ = improvement**.

### 9.1 Topology（within fixed Representation × Annotation）

Baseline = **SEP**

* JOINT − SEP  
* REG-SEP − SEP  
* XREG − SEP  
* FUSE − SEP  

Compute on `TEST_mean` for ranking tables；also on `TEST_P` and `TEST_S` for Secondary.

### 9.2 Annotation（within fixed Representation × Topology）

Baseline = **BASE**

* IMGT − BASE  
* REGION − BASE  
* FULL − BASE  

### 9.3 PLM vs Scratch（within fixed Topology × Annotation）

For each non-Scratch representation level:

`PLM − Scratch`

Report **separately** from absolute 10-way ranking of representations.

### 9.4 Inference-context（AbLang2 and CurrAb；within fixed Topology × Annotation）

`PAIRED_NATIVE − SEPARATE_CHAIN`

Pre-registered as a **cross-target priority contrast** (compare later to TmApp Figure-6-style analyses).  
Does **not** replace the 10-level grid.

### 9.5 Secondary directional replication

For any predefined contrast with baseline B and candidate C:

`Δ = MAE(C) − MAE(B)` on Primary and Shadow OOF (`Δ_P`, `Δ_S`).

**Directionally replicated** iff signs agree；for an improvement claim, require **`Δ_P < 0` and `Δ_S < 0`** (unless formal prereg amends with written exception).  
Secondary never replaces Primary `TEST_mean` ranking.

---

## 10. Interaction analysis

### Primary interactions（Level 1 patterns）

* Representation × Topology  
* Representation × Annotation  
* Topology × Annotation  

### Hierarchical / exploratory（AbLang2, CurrAb）

* Family × Context × Topology  
* Family × Context × Annotation  
（or equivalent paired-context interaction tables）

Label exploratory unless Primary/Shadow agree and bootstrap CIs support.

### Analysis hierarchy（conclusion strength）

| Level | Content | Role |
|------:|---------|------|
| **1** | Factor-pattern conclusions across many cells | **Primary scientific weight** |
| **2** | Predefined paired contrasts (§9) + uncertainty | Confirmatory support |
| **3** | Individual best cell | **Descriptive only** — not main conclusion |

---

## 11. Primary / Secondary / Diagnostic / External rules

| Role | Definition | Selection use |
|------|------------|---------------|
| **Primary** | `TEST_mean = mean(TEST_P, TEST_S)` | **Yes** — cell/factor ranking |
| **Secondary** | `Δ_P`/`Δ_S` sign replication vs baseline | No — does not override Primary |
| **HIGH-tail diagnostic** | `HIC > 11.5 min`；record n, MAE, mean signed error `(pred−true)`, observed range, predicted range | **No** |
| **External diagnostic** | GEN_0001 Public/Private MAE（after internal factorial freeze） | **No** for rep/topo/annot/HP/cell selection or rewriting factorial interpretation |

`cv_worst_mae` is legacy classical read-only；**not** used in this factorial.

---

## 12. Bootstrap / uncertainty plan

### Source (TmApp)

`developability_drilldown/scripts/run_exp_t151_t156_plm_topology.py` → `paired_boot` / `did_boot`；used by `analyze_t161_t337_factorial.py` with:

* `SEED = 101`  
* `N_BOOT = 2000`  

### Definition to reuse for HIC（do not blindly copy scripts without path checks）

**Resampling unit:** antibody `id` in the Dev OOF prediction vector for a given scheme（Primary or Shadow），aligned on common ids.

**Paired structure:** for contrast of predictors `a` vs `b` with labels `y`:

* per-id absolute errors `e_a = |a−y|`, `e_b = |b−y|`  
* `d_i = e_a,i − e_b,i`  
* point estimate = `mean(d)`  
* bootstrap: resample ids with replacement `N_BOOT` times；each replicate = `mean(d_boot)`  
* **CI:** percentile **2.5% / 97.5%** of bootstrap means  

（`paired_boot` docstring: Δ=MAE(a)−MAE(b)；negative ⇒ a better.）

**DiD:** same antibody-level pairing via `did_boot` for non-additivity / interaction contrasts；exploratory unless Primary & Shadow agree.

### Why applicable to HIC V3

HIC V3 cells already emit `oof_test_primary.csv` / `oof_test_shadow.csv` under `experiments/predictions/EXP-Hxxx/`（verified for reuse candidates）.  
`TEST_P`/`TEST_S` are MAE of those OOF vectors — **same structure** as TmApp factorial internal metrics.  
Resampling antibodies (not folds as independent units) matches TmApp’s paired residual bootstrap.

### Reporting rules

* Prefer contrasts with **Primary/Shadow agreement**.  
* If CI includes 0, do not claim strong superiority.  
* Stars/markers = descriptive（TmApp policy）.

Formal prereg must name the HIC analysis entrypoint (new or adapted script) and confirm it loads **HIC** OOF paths and **HIC** labels (`HIC` column), not TmApp.

---

## 13. Cross-target TmApp vs HIC comparison plan

**Timing:** only after HIC **internal** factorial freeze.

**Align:** factor names, level names, SEP/BASE/Scratch/context contrasts, topology/annotation gain tables.

**Compare (descriptive):**

* best representation family patterns  
* topology gain vs SEP  
* annotation gain vs BASE  
* Rep×Topo / Rep×Annot  
* AbLang2 context effect；CurrAb context effect  
* Scratch vs PLM behavior  

**Forbidden:** using TmApp results for HIC representation/topology/annotation/HP/cell selection at any time before or during HIC execution.

---

## 14. Technical gates（MAE-blind）

| # | Gate | Pass criterion (technical) |
|---|------|----------------------------|
| 1 | REUSE checksum | All checklist fields resolved；else demote to new |
| 2 | Representation asset inventory | All 10 reps load；coverage N=324；context metadata OK |
| 3 | Annotation wiring smoke | BASE/IMGT/REGION/FULL toggle without crash |
| 4 | Topology wiring smoke | SEP and at least one of XREG/FUSE train/eval with variance>0 |
| 5 | Primary/Shadow prediction smoke | OOF artifacts written for both schemes |
| 6 | Artifact completeness | configs, OOF, run_state；nonzero pred variance |
| 7 | Analysis pipeline dry-run | contrast + bootstrap code runs on smoke outputs |

**MAE quality is not a gate.** Gates must not drop scientifically inconvenient cells.

---

## 15. Failure / blocked-cell policy

* Train failures: **one** automatic retry（TmApp-style）；then mark FAILED with reason.  
* Blocked (missing asset / irreversible config error): BLOCKED with reason；do not silently skip without record.  
* No replacement topology/annotation/PLM mid-batch.  
* Matrix remains 200 logical cells；FAILED/BLOCKED counted explicitly in reports.

---

## 16. Missing-cell policy

* Every planned cell appears in the plan table with status ∈ {PLANNED, REUSE, COMPLETE, FAILED, BLOCKED}.  
* Missing artifacts after claimed COMPLETE → treat as FAILED until restored.  
* No imputation of MAE for missing cells in primary tables.

---

## 17. No-midway-selection commitment

* All 200 cells defined before inspecting new factorial HIC results.  
* **No** dropping PLM / topology / annotation from intermediate `TEST_mean`.  
* **No** expanding surface features into this batch.  
* Gates may only enforce technical validity.

---

## 18. Public / Private embargo

* GEN_0001 scores computed only after **internal** freeze of all planned cells’ internal metrics.  
* Embargo on using Public/Private for selection, HP, or reinterpretation of internal factor conclusions.  
* Post-embargo analyses labeled **external diagnostic** / post-hoc.

---

## 19. Required artifacts（at formal prereg / execution）

1. Frozen plan CSV/YAML（200 rows）+ SHA256  
2. REUSE checksum report  
3. Embedding inventory（path, context, hash, N）  
4. Per-cell config + OOF Primary/Shadow + run_state  
5. Master results table（`TEST_P/S/mean/worst`）  
6. Contrast tables（topo/annot/Scratch/context）+ bootstrap CIs  
7. HIGH-tail diagnostic table（non-selection）  
8. Internal freeze yaml（pre-external）  
9. Optional post-freeze external score table  
10. Analysis report + cross-target comparison note  

---

## 20. Items still requiring human approval before execution

1. Approve **Design A** prereg draft → authorize **formal** `PREREGISTERED_FROZEN` commit  
2. Resolve REUSE 0–6 via checksum（especially `share_hl_encoder` evidence）  
3. Confirm embedding coverage/hashes for all 10 reps（extract only if approved）  
4. Authorize EXP-H **code issuance** / registry writes  
5. Authorize technical gate smoke **training** (minimal) vs full 194–200 train  
6. Confirm analysis script ownership (adapt TmApp analyzer for HIC labels/paths)  
7. Confirm no surface leakage in runners  
8. Schedule external embargo end conditions  

---

## Binding commitments（to be signed at formal prereg）

* Design A 200 cells；no mid-batch factor dropping  
* Primary=`TEST_mean`；Secondary=Δ sign replication；HIGH-tail & Pub/Priv non-selection  
* Context metadata over allocation nicknames  
* TmApp results never select HIC cells  
* Surface/physics excluded from this factorial  

---

**END OF DRAFT — stop here until human approval.**
