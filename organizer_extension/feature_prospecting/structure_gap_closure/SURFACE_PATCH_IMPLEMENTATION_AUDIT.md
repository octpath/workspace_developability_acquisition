# SURFACE_PATCH_IMPLEMENTATION_AUDIT

**Date:** 2026-09-07  
**Classification: `RESIDUE_GRAPH_ONLY_OR_INCOMPLETE`**

Therefore Task B (true continuous molecular-surface HIC features) **runs**.

---

## Scope audited

| Artifact | Path |
|----------|------|
| Marathon S3 | `structure_marathon/scripts/s3_s4_graphs.py`, `surface_patch_graph/S3_FEATURES.csv` |
| AROMATIC-TOPO / STATIC-SAP | `feature_prospecting/scripts/extract_physical_batch1.py`, family SPECs |
| HYDRO-FIELD | `common/hydro_surface.py` |
| ADV surface patch | Round1 / stage4 ADV extractors |

---

## Answers

### 1. Graph nodes?

**Residues (CA coordinates)** for S3 `surface_patch_graph`.  
Not atoms; not molecular-surface mesh vertices.

### 2. Solvent exposure?

S3 uses distance proxy `rasa_proxy = 1 - (#CA within 10Å)/25`, threshold 0.25 — **not** FreeSASA/Shrake–Rupley RASA.  
Related families (AROMATIC-TOPO, ADV) use residue Shrake–Rupley / FreeSASA SASA.

### 3. Triangulated SAS/SES molecular surface?

**No** for S3.  
HYDRO-FIELD samples exterior points on probe-expanded spheres from FreeSASA atom areas — **point cloud**, not a triangulated mesh.

### 4. True surface AREA on connected patches?

**No** for S3 (`*_largest_sasa` = sum of `rasa_proxy`).  
AROMATIC-TOPO/ADV sum residue SASA (Å²) over residue patches — still residue graphs, not mesh-face area.

### 5. Patch adjacency?

**Residue–residue CA distance** (S3: ≤6 Å).  
Not adjacency on a molecular-surface mesh.

### 6. Descriptor checklist (S3 vs needed)

| Quantity | S3 | Notes |
|----------|----|-------|
| largest hydrophobic patch area | pseudo only | proxy sum |
| perimeter | **missing** | |
| compactness | graph clustering only | not geometric |
| connected-component area | largest_n + proxy | |
| aromatic fraction in hydro patch | count only | not fraction |
| CDR / FW fraction of patch | **missing** | |
| fragmentation | weak (n/density) | |
| patch shape | **missing** | |

### 7. Hydrophobicity scales?

- **S3:** binary set `{A,I,L,M,F,V,W,Y}` — no continuous scale  
- Stage3 SURFACE_CHEM: Kyte–Doolittle  
- STATIC-SAP: Black & Mould 1991  
- HYDRO-FIELD: Fauchère–Pliska π  

### 8. Same descriptor across generators?

- **S3:** ESMFold Fv only  
- AROMATIC-TOPO / STATIC-SAP / HYDRO-FIELD: ESMFold + ABB2 + Boltz2  

---

## Overlap with Task B

HYDRO-FIELD already has continuous **hydrophobic field on surface samples**, but:

- optimized for field quantiles / high-H area **fractions**, not the frozen HIC patch geometry list (perimeter, compactness, aromatic/CDR/FW fractions of largest hydro patch, multi-scale hydrophobicity patches, Fv↔Fab ΔPATCH_CONTEXT, generator replication of that family).

**Decision:** Task B proceeds with a **frozen PRIMARY_SURFACE_DEF** (FreeSASA LR + exterior sampling + scale-threshold hydrophobic patches) documented in `STRUCTURE_GAP_CLOSURE_SPEC.md`. Explicitly distinct from residue-CA S3 and from reporting HYDRO-FIELD as “already done continuous patch geometry.”
