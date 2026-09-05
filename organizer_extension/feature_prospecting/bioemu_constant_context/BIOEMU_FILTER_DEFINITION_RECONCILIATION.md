# BioEmu Filter Definition Reconciliation

**Evidence:** ORGANIZER-EXPLORATORY  
**Scope:** Reconcile `foundation_stability_v2` isolated VH/VL “high completion” language vs Gate B0.5 physical pass rates  
**Constraint:** No large resampling; audit of existing caches + code paths only  
**Package:** `bioemu==1.4.1`, model `bioemu-v1.2` (same env for v2 and constant-context)

---

## Executive answer

The apparent discrepancy is **not** a scientific contradiction between isolated domains and Gate B0.5.

It is a **definition mismatch**:

| Phrase in v2 | What was actually counted |
|--------------|---------------------------|
| “648/648 completed”, `sample_log.valid == 16`, status SUCCESS/CACHED | **Unfiltered NPZ frames** via `count_samples_in_output_dir` |
| Gate B0.5 “physical pass rate” | **Filtered XTC frames / NPZ frames** after official `filter_unphysical_traj` |

Under an **apples-to-apples** definition (official-mask pass = `n(samples.xtc) / n(batch_*.npz)` after `filter_samples=True` conversion):

| Construct | Median pass rate |
|-----------|------------------|
| Isolated VH (v2, n=324 chains) | **0.312** |
| Isolated VL (v2, n=324 chains) | **0.312** |
| Isolated VL (B0.5 domain test, n=4) | **0.375** |
| Isolated CL (B0.5, n=4) | **0.578** |
| VL+CL (B0.5, n=4 / 12) | **≈0.10** |

Isolated VL ≈ 0.31–0.38 in both eras. VL+CL ≈ 0.10 remains **specifically worse**.

---

## 1. Statement inventory

### A. foundation_stability_v2 — “high usable completion”

**Source statements**

- Report: “324/324 Abs、VH+VL **648/648** 鎖” completed  
- `cache/bioemu_samples/sample_log.csv`: almost all rows `valid=16`, status `SUCCESS`/`CACHED`  
- Sampler success criterion in `scripts/bioemu_sample.py`

| Field | Value |
|-------|--------|
| Numerator (logged as `valid`) | `count_samples_in_output_dir(out_dir)` = sum of lengths encoded in `batch_*_*.npz` filenames |
| Denominator / target | `requested` = frozen N (16) |
| Object counted | **Generated NPZ frames (pre-physical-filter)** |
| Not counted | XTC frames; official mask passes |
| Model | `bioemu-v1.2` |
| `filter_samples` | **`True`** at `bioemu.sample.main` call |
| When filtering runs | During **NPZ→PDB/XTC conversion** inside `save_pdb_and_xtc(..., filter_samples=True)` → `filter_unphysical_traj` |
| Success check after filter? | **No** — loop exits when NPZ count ≥ requested |

**Code fact:** `count_samples_in_output_dir` docstring: counts samples in **npz files**, not filtered trajectories.

### B. Gate B0.5 — isolated VL ≈ 0.38, VL+CL ≈ 0.10

| Field | Value |
|-------|--------|
| Numerator | Frames in `samples.xtc` after official conversion filter (= frames that passed the mask) |
| Denominator | NPZ unfiltered count (64 in domain test) |
| Object | **Per-frame official physical-mask pass rate** |
| Filter function | `bioemu.convert_chemgraph.filter_unphysical_traj` / `_filter_unphysical_traj_masks` |
| Defaults | `max_ca_seq_distance=4.5` Å, `max_cn_seq_distance=2.0` Å, `clash_distance=1.0` Å |
| Model / package | Same `bioemu-v1.2` / 1.4.1 |
| Retrospective re-filter? | Domain-test rates from conversion filter; failure taxonomy also rebuilt RAW then reapplied same masks (consistent) |

### C. Retrospective check on v2 XTC

Re-applying `_filter_unphysical_traj_masks` to existing `samples.xtc` yields **pass rate 1.0 vs XTC** (spot-check 20/20 VL).  
⇒ XTC already contains **only** post-filter frames. No second hidden filter in feature code.

---

## 2. Explicit answers

### Q1. Were previous isolated VH/VL TmApp features based on physically filtered frames?

**Yes.**

`bioemu_features.py` → `load_traj()` reads `samples.xtc` + `topology.pdb`.  
That XTC is the **output of** `filter_samples=True` conversion.  
Descriptors (`n_valid` = `traj.n_frames`) therefore use **filtered** frames only.

### Q2. If yes, what was their true per-frame official-mask pass rate?

From existing v2 caches (`results/RECONCILE_V2_ISO_PASS_RATES.csv`):

| Chain | Median `phys_xtc / npz` | Mean | Median `phys_xtc` when npz≈16 |
|-------|-------------------------|------|-------------------------------|
| VH | **0.312** | 0.343 | **4.5** |
| VL | **0.312** | 0.352 | **5.0** |

So the true official-mask pass rate for isolated variable domains was **~31% median**, **not** ~100%.

Logged `valid=16` meant “16 NPZ written,” while the ensemble actually used for features was typically **~5 filtered frames** (median).

Only ~3% of chains have `phys_xtc == npz` (100% pass).  
~20% of chains have `phys_xtc ≥ 8`.

### Q3. Does any discrepancy affect validity of the completed isolated VH/VL BioEmu result?

**Split:**

| Claim | Impact |
|-------|--------|
| **Pipeline completion** (“648/648 dirs with NPZ≥16”) | Still true under the NPZ definition; wording overstated “physical validity.” |
| **Features = filtered-frame descriptors** | Still true. |
| **TmApp null / no PLM increment** | **Directionally intact** as an exploratory null on those descriptors; not rescued by redefining pass rate. |
| **Sample-count freeze N=16** | **Weakened.** Convergence compared prefixes of XTC that often had **≪16** frames; “N=16 sufficient” was not a clean 16-physical-frame study. Treat frozen N as an operational choice, not a strong physical-ensemble convergence proof. |
| **“High physical validity” narrative** | **Incorrect** if read as per-frame pass ≈1. Must be corrected to: high **NPZ quota fulfillment**, moderate **~30% mask pass**, small filtered ensembles. |

**Do not** reopen v2 scoring on this audit alone; do **document** the mislabeling.

### Q4. Is “VL+CL specifically much worse than isolated domains” still robust apples-to-apples?

**Yes.**

Same definition `phys_xtc / npz`, same package/model/filter:

| | Isolated VL | VL+CL |
|--|-------------|-------|
| v2 cohort median | 0.31 | — |
| B0.5 4-Ab median | 0.38 | **0.10** |
| B0.5 12-Ab VLCL (filter summary) | — | **≈0.095** |

Paired example (ADI-47230): VL pass **0.375** (v2 6/16 and B0.5 24/64) vs VLCL **0.094**.

Gate B0.5 conclusion that low pass is **multi-domain / interdomain-clash dominated** does **not** depend on the v2 wording error.

---

## 3. Side-by-side definition table

| Quantity | v2 `sample_log.valid` | v2 feature `n_valid` | B0.5 pass rate |
|----------|----------------------|----------------------|----------------|
| Numerator | NPZ count | XTC frames loaded | XTC frames |
| Denominator | requested N | — (absolute count) | NPZ count |
| Filtered? | No | Yes (implicit) | Yes / Yes |
| Typical magnitude (VL) | 16 | ~5 | ~0.31–0.38 |

---

## 4. Why B0.5 isolated VL (0.38) looked “worse” than v2 narrative

1. v2 narrative counted **job success = NPZ quota**, casually readable as “physical OK.”  
2. B0.5 reported **mask pass fraction**.  
3. Same biology + same filter ⇒ isolated VL ~0.3–0.4 in both; only the **label** differed.  
4. CL alone (~0.58) and VL+CL (~0.10) were new measurements; they never appeared in v2.

---

## 5. Residual caveats (no new sampling)

- 5 v2 chains have unreadable XTC (`phys_xtc=-1`); negligible vs 648.  
- Some v2 dirs have `npz > 16` (oversample retries); pass rates still `phys/npz`.  
- MSA path differs (v2 often Boltz a3m; B0.5 domain test singleseq) — pass rates still land in the same ballpark for VL, so MSA is not the driver of the v2-vs-B0.5 “discrepancy.”  
- Full-cohort constant-context sampling remains **blocked** (`BIOEMU_EXTENDED_CHAIN_UNRESOLVED`).

---

## 6. Files

| File | Role |
|------|------|
| `BIOEMU_FILTER_DEFINITION_RECONCILIATION.md` | This note |
| `results/RECONCILE_V2_ISO_PASS_RATES.csv` | Per-chain v2 `npz`, `phys_xtc`, `pass_rate` (648 rows) |
| `results/B05_DOMAIN_COMPARE_PASS.csv` | B0.5 VL/CL/VLCL pass rates |
| `BIOEMU_LOW_PASS_DIAGNOSTIC.md` | Gate B0.5 (unchanged conclusion) |
