# THERMAL_MICROPROBE — Feature SPEC (target-blind freeze)

**Interpretation:** short finite-temperature **thermal-response / relaxation** descriptor.  
**Not** MD sampling, equilibrium dynamics, or unfolding.

## Cohort (Dev gate)

- Usable FeNNix Dev IDs only (`USABLE_DEV_IDS.json`, N=161)
- Missing Dev: ADI-47265
- **No Test** until Dev Simple TVT gate passes

## Protocol (frozen)

| Parameter | Value |
|-----------|-------|
| Engine | FeNNol `fennol.md` Langevin (LGV) NVT |
| Model | `foundation_stability_v2/cache/fennix-bio1S.fnx` |
| Start structure | `fennix_fab_context/cache/r1/{id}_C_r1.pdb` |
| T | 300 K |
| dt | 1 fs |
| Steps | **250** (no separate equilibration) |
| Thermostat | LGV, γ = 1 / ps |
| Seed | deterministic `sha256(id) → uint32` (frozen formula) |

Do **not** tune T / dt / steps / thermostat / seed from targets.

## Domains

- Chain A: heavy = VH (res 1..VH_len) + CH1 (rest)
- Chain B: light = VL (res 1..VL_len) + CL (rest)
- Lengths from `SHEHATA_RECONSTRUCTED_FAB.csv` (`VH_len_used`, `VL_len_used`)

## Native contacts

- Heavy-atom pairs between domain pairs within **4.5 Å** on the start structure
- Retention at endpoint = fraction still ≤ 4.5 Å

## Features (small block)

### Thermal / geometry

- `endpoint_backbone_rmsd`
- `max_backbone_rmsd` (over sparse dumped frames)
- `VH_endpoint_displacement`, `VL_endpoint_displacement`, `CH1_endpoint_displacement`, `CL_endpoint_displacement` (CA COM distance start→end)
- `VH_VL_contact_retention_end`, `VH_CH1_contact_retention_end`, `VL_CL_contact_retention_end`, `CH1_CL_contact_retention_end`
- `potential_energy_start`, `potential_energy_end`, `delta_potential_energy`
- `force_rms_start`, `force_rms_end`, `delta_force_rms`
- `rg_start`, `rg_end`, `delta_rg`

### Exposure (start & endpoint only)

- `aromatic_sasa_start`, `aromatic_sasa_end`, `delta_aromatic_sasa`
- `hydrophobic_sasa_start`, `hydrophobic_sasa_end`, `delta_hydrophobic_sasa`
- `delta_tyr_sasa`, `delta_phe_sasa`, `delta_trp_sasa`

No continuous-surface / patch mesh along the probe.
