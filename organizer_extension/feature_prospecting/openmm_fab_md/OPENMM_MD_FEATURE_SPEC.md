# OPENMM Fab MD — Feature SPEC (target-blind freeze)

**Interpretation:** short **implicit-solvent classical-MD dynamic descriptors**.  
**Not** rigorous solution MD, unfolding, or direct Tm simulation.

## Cohort

- Usable Fab/FeNNix Dev IDs only (`results/USABLE_DEV_IDS.json`, N=161)
- Missing Dev: ADI-47265
- Start structures: `fennix_fab_context/cache/r1/{id}_C_r1.pdb`

## Frozen MD protocol

| Parameter | Value |
|-----------|-------|
| Engine | OpenMM **8.2**, platform **CUDA** (RTX 3090, DeviceIndex=0) |
| Force field | `amber14-all.xml` |
| Solvent | `implicit/obc2.xml` (OBC2 GB) |
| Nonbonded | CutoffNonPeriodic, cutoff 2.0 nm |
| Constraints | HBonds |
| Integrator | LangevinMiddle |
| T | 300 K |
| γ | 1 / ps |
| dt | **2 fs** |
| Minimization | `minimizeEnergy(maxIterations=200)` |
| Equilibration | **5 ps** (2500 steps), velocities from Maxwell–Boltzmann @ 300 K |
| Production | **0.5 ns** (250_000 steps); one-time fallback **0.25 ns** if timing gate fails |
| Trajectory dump | every **10 ps** (5000 steps) during production |
| Seed | `uint32(sha256(id.encode()).digest()[:4])` |

Do **not** tune FF / solvent / T / dt / lengths from target labels.

## Domains

- Chain A: VH = resseq 1..VH_len_used; CH1 = remainder  
- Chain B: VL = resseq 1..VL_len_used; CL = remainder  
- Lengths from `SHEHATA_RECONSTRUCTED_FAB.csv`

## Native contacts

- Defined on **initial** heavy atoms between domain pairs  
- Cutoff: **4.5 Å** (frozen)  
- Occupancy = mean fraction of native pairs within cutoff over production frames

## Feature families

### DYN_GLOBAL
- `backbone_rmsd_mean`, `backbone_rmsd_sd`, `backbone_rmsd_q90`
- `rg_mean`, `rg_sd`

### DYN_RMSF
- `{VH,VL,CH1,CL}_rmsf_mean`, `{VH,VL,CH1,CL}_rmsf_q90` (backbone)

### DYN_INTERFACE
- `{VH_VL,VH_CH1,VL_CL,CH1_CL}_native_contact_occupancy`
- `{...}_native_contact_occupancy_sd`, `{...}_native_contact_occupancy_min`

### DYN_EXPOSURE (HIC-oriented)
- aromatic / hydrophobic SASA mean, sd, q90 over frames
- Tyr / Phe / Trp SASA mean
- `delta_*` vs start frame where natural
- largest hydrophobic patch: **SKIP** (continuous-surface mesh not used)

## QC fields (sidecar, not in participant feature matrix)

`simulation_success`, `finite_energy`, `finite_coordinates`, `constraint_failure`,  
`max_backbone_rmsd`, `temperature_mean`, `feature_finite`, `runtime_seconds`
