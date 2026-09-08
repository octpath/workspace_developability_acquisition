# OPENMM_ENDGAME_BENCHMARK

- OpenMM `8.2` / platform **CUDA** (RTX 3090 DeviceIndex=0)
- Force field (implicit): `amber14-all.xml + implicit/obc2.xml`
- Structure: `ADI-45391` `/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context/cache/r1/ADI-45391_C_r1.pdb`
- Atoms (implicit): **6434**
- Constraints: HBonds; dt=2 fs; LangevinMiddle; T=300 K; γ=1/ps
- Stability: finite=True; no NaN/Inf observed

## Implicit solvent (OBC2) performance

| metric | value |
|---|---:|
| production steps | 25000 |
| simulated ns | 0.0500 |
| wall s | 8.21 |
| **ns/day** | **526.0** |
| GPU util % | 98.0 |
| GPU mem MiB | 312.0 |

## Projected DEV N=161 wall time (1 GPU, serial) — implicit

| traj / Ab | projected hours |
|---:|---:|
| 0.25 ns | 1.84 |
| 0.5 ns | 3.67 |
| 1.0 ns | 7.35 |
| 2.0 ns | 14.69 |

## Explicit solvent

- status=**OK**; atoms=106208; ns/day=**190.7**; finite=True
- setup: explicit tip3p + tip3pfb FF params + PME + 0.15M, pad 1nm

| traj / Ab | projected hours |
|---:|---:|
| 0.25 ns | 5.07 |
| 0.5 ns | 10.13 |
| 1.0 ns | 20.27 |
| 2.0 ns | 40.53 |

## Decision gate

- measured ns/day (implicit) = **526.0** → band **≥500**
- recommended traj length = **0.5 ns / DEV antibody**
- endgame choice = **OPENMM_GO**

## FeNNix status

- completed 37/161 preserved (not deleted/overwritten)
- CUDA OOM on ADI-46682; isolated subprocess resume prepared (`04_resume_isolated.py`)
- **Not launched** during this OpenMM benchmark
