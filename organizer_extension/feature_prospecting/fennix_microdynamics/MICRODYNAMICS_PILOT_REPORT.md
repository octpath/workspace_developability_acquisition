# MICRODYNAMICS_PILOT_REPORT

UTC bench: 2026-09-08 (RTX 3090, FeNNol 2026.6.29, fennix-bio1S)

## Goal

Fast check whether short FeNNix finite-T dynamics can supply competition features beyond static/curvature Fab FeNNix.

## Environment

| Item | Value |
|------|-------|
| Python | `foundation_stability/envs/fennol/bin/python` |
| JAX CUDA | Requires `LD_LIBRARY_PATH` to env `nvidia/*/lib` (cusparse etc.) |
| Device | `cuda:0` RTX 3090 (GTX 1080 Ti present but unused) |
| Engine | `fennol.md.dynamic.config_and_run_dynamic` Langevin LGV |
| Structure | `ADI-45391` `*_C_r1.pdb` (6434 atoms) |

CPU-only force eval ≈ 0.71 s/step → ruled out immediately (~380 h for 323×6k).

## Pilot ID selection (target-blind)

12 IDs stratified by kappa/lambda (6/6) and Fab sequence length tertiles; QC metadata recorded without TmApp/HIC.

See `results/PILOT_IDS.csv`, `results/PILOT_ID_METADATA.csv`.

**Full 5 ps trajectories were not run on all 12** after the timing gate failed on the instrumented bench antibody.

## Timing / resource bench (ADI-45391)

| Metric | Value |
|--------|------:|
| Bench steps | 500 |
| Wall | 172.0 s |
| sec / step | **0.344** |
| Peak GPU mem (force path) | ~18.4 GiB |
| NaN/Inf | none observed |
| Catastrophic explosion | no |
| T behavior | rising ~181→230 K over 0.5 ps (toward 300 K) |
| Epot | finite throughout |

Projection (same sec/step):

| Protocol | Steps | h / Ab | h × 323 |
|----------|------:|-------:|--------:|
| 1 ps eq + **5 ps** prod | 6000 | 0.57 | **185** |
| 1 ps eq + **2 ps** prod | 3000 | 0.29 | **93** |

Sprint gate: full cohort ≲ **12 h**.

## Pilot acceptance

| Criterion | Result |
|-----------|--------|
| ≥11/12 trajectories complete | **N/A** (cohort path aborted after timing gate) |
| No explosions / systematic NaN | Bench OK |
| Projected 323 runtime ≤12 h | **FAIL** (185 h @ 5 ps; **93 h @ 2 ps**) |

## Verdict

**`MICRODYNAMICS_TOO_SLOW`**

Per sprint rules: do not optimize thermostats/timesteps/lengths further; stop.

## Artifacts

- `results/PILOT_TIMING.json`
- `/tmp/fennix_md_bench/run500.log` (local bench log; not authoritative cache)
- This report
