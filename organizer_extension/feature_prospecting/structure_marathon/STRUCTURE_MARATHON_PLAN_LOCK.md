# Structure Marathon — PLAN LOCK

**State:** `STRUCTURE_MARATHON_PLAN_LOCKED`  
**Locked before target scoring:** true  
**Campaign start (UTC):** see `cache/campaign_start_utc.txt`  
**Authority:** Autonomous Structure-Information Campaign message (this session)

## Targets

- **TmApp** (Fab experimental melting temperature, °C)
- **HIC** (IgG HIC retention time, min)

## Evidence rules

All structural feature definitions are **target-blind**. Public/Private = post-reveal replication only.  
Do not use TmApp/HIC to choose masks, thresholds, checkpoints, or pilot validity.

## Molecular scope

- TmApp measured on **Fab**; HIC on **IgG**
- Reconstructed Fab = `RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB` (not exact experimental)
- Full-Fab HIC surface features = `HIC_FAB_SURFACE_EXPLORATORY` only

## Screening model (frozen)

- Low-dim ≤30: fold-local StandardScaler → Ridge, nested α ∈ {0.1, 1, 10, 100}
- High-dim embeddings: fold-local StandardScaler → PCA32 → Ridge (PCA fold-local; no forced PCA if dim<32)
- No Optuna per family; SVR only as secondary confirmation if warranted
- Bootstrap B=10000 for families improving incumbent on Primary+Shadow

## Family priority (predeclared)

| Pri | ID | Family |
|-----|-----|--------|
| P0 | S1 | Structure-guided PLM pooling (ESM2 residue) |
| P0 | S2 | Cross-generator disagreement |
| P0 | S3 | Surface-patch graph (HIC-oriented) |
| P0 | S4 | Contact/packing graph (TmApp-oriented) |
| P1 | M1 | ProteinMPNN (soluble primary) |
| P1 | M2 | ESM-IF1 |
| P1 | M3 | SaProt (35M smoke → 650M if practical) |
| P1 | M4 | ProSST (≤45 min setup) |
| P2 | T1 | SPURS mutational robustness (TmApp) |
| P2 | T2 | ThermoMPNN if SPURS blocked / early finish |
| P2 | G1 | GearNet optional (≤30–45 min) |
| P3 | FeNNix Fab | Workstream A (`fennix_fab_context/`) |

## S1 masks (frozen labels)

COMMON: CDR_ALL, HCDR3, LCDR3, BURIED_CORE, EXPOSED, STRONGLY_EXPOSED, VH_VL_INTERFACE, HIGH_CONTACT_DENSITY_CORE  
HIC: EXPOSED_AROMATIC, STRONGLY_EXPOSED_AROMATIC, EXPOSED_HYDROPHOBIC, CDR_EXPOSED, LARGEST_HYDROPHOBIC_SURFACE_PATCH (+ neighbors)  
TmApp/Fab: VH_CH1_INTERFACE, VL_CL_INTERFACE, CH1_CL_INTERFACE, VARIABLE_CONSTANT_INTERFACE_UNION  

Pooling: mean; optional RASA-/contact-weighted if cheap. Each mask = own family (no mega-concat).

Geometric thresholds (target-blind freeze):
- Exposed: RASA ≥ 0.25; Strongly exposed: RASA ≥ 0.40 (relative SASA if available; else absolute SASA proxy documented per source)
- Contact edge: Cβ–Cβ (or CA for Gly) ≤ 8.0 Å
- VH/VL interface: heavy-atom contact ≤ 4.5 Å between chains
- Surface patch edge: exposed residues with CA–CA ≤ 6.0 Å

## Workstream A — FeNNix order (authoritative)

1. FeNNix pilot 12 B/C/M (no TmApp)  
2. ADI-47317 CPU isolated diagnostic  
3. CPU/CUDA equivalence audit (**before** mass remaining prep)  
4. Preparation protocol freeze  
5. Full-cohort Fab preparation (CPU small-batch)  
6. Full FeNNix B/C/M  
7. Target-blind feature freeze  
8. TmApp scoring  

### CPU/CUDA equivalence acceptance (frozen)

Matched FeNNix curvature on frozen site subset:
- Spearman(CPU, CUDA) ≥ 0.95  
- median normalized absolute difference ≤ 10%  
Plus no systematic platform shift in backbone/domain RMSD, S–S geometry, clashes, residual force, deformation.

### Pilot classification

`PILOT_PASS` | `PILOT_PASS_WITH_LIMITATIONS` | `PILOT_FAIL` — technical/physical only. No predictive scoring.

### Matched C/M requirement

Before perturbation: C variable-region coordinates == M within recorded numerical tolerance (default: max |Δ| < 1e-4 Å per atom, or RMSD < 1e-6 Å). DELTA_ENV invalid otherwise.

## Host safety

- No persistent OpenMM CUDA full-Fab prep  
- One substantial GPU inference process at a time  
- CPU Fab prep may co-run with GPU inference only if RAM/I/O/load safe; else serialize  
- Optional model setup budget ~30–45 min → DEFERRED_TECHNICAL  

## Forbidden

IgG-from-scratch, broad MD, BioEmu rerun, Rosetta campaigns, fine-tuning large models on N=162, Optuna sweeps, target-tuned masks, Public/Private optimization.

## Verdict classes

CLEAR_INCREMENT | WEAK_SIGNAL | REDUNDANT_SIGNAL | NO_SIGNAL | GENERATOR_FRAGILE | ARTIFACT_SUSPECT | TECHNICAL_BLOCK
