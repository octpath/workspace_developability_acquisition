# Boltz-2 Run Report — `BOLTZ2_FV_STANDARD_v1`

**完了:** 2026-09-02T15:31:37Z（`progress_watch.log` に `ALL_STRUCTURES_DONE`）

## Summary

| Metric | Value |
|--------|------:|
| Success | **324 / 324** |
| Failure | 0 |
| H sequence EXACT_MATCH | 324 / 324 |
| L sequence EXACT_MATCH | 324 / 324 |
| mapping_status PASS | 324 / 324 |

## Protocol

- Package: `boltz==2.2.1`
- Checkpoint: `boltz2_conf.ckpt`（SHA-256 in `BOLTZ2_STRUCTURE_SPEC.json`）
- MSA: ColabFold API + greedy pairing（precompute → YAML embed）
- Templates: none
- use_potentials: false
- diffusion_samples: 1
- recycling_steps: 3 / sampling_steps: 200 / step_scale: 1.5
- output: native mmCIF + BioPython PDB（H/L 保持）

## Confidence（target-blind）

| Metric | min | median | max |
|--------|----:|-------:|----:|
| confidence_score | 0.902 | 0.954 | 0.978 |
| ptm | 0.930 | 0.960 | 0.978 |
| iptm | 0.915 | 0.946 | 0.972 |
| complex_plddt | 0.893 | 0.955 | 0.983 |

## QC

- chains: 全件 `H,L`
- NaN coordinates: 0
- residue count: 221–247（median 231）
- atom count: 1670–1914（median 1770）

## Artifacts

- Spec: `BOLTZ2_STRUCTURE_SPEC.json`
- Manifest（per-ID hashes）: `BOLTZ2_STRUCTURE_MANIFEST.csv`
- Audit: `BOLTZ2_STRUCTURE_AUDIT.json`
- Structures: `structures_mmcif/`, `structures_pdb/`, `confidence/`
- MSA CSV（local, large）: `msa/`（git 非追跡推奨）
- Native dumps: `predictions_native/`（git 非追跡推奨）

## Policy notes

- successful outputs は silent overwrite しない
- MSA transient failure: retry（spec `msa_max_retry=12`）
- predict transient: max 2 retry
- target labels / TmApp·HIC 相関は未計算
