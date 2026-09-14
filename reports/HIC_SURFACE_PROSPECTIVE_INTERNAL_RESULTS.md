# HIC SURFACE Prospective Replication — Internal Results Freeze

**STATUS: `HIC_SURFACE_PROSPECTIVE_INTERNAL_FROZEN`**

| Field | Value |
|-------|--------|
| Prereg SHA | `993b8a9e268f65ed8d203949c34cbe3ef87446e6` |
| Code/config SHA at freeze | `f53039ffe606a2ca685fe64efaaeafb9c3993c6b` |
| External scoring | **NONE (embargo)** |
| solution.csv loaded | **NO** |
| Public/Private consulted | **NO** |

## Parent verification

- Config match status: **PASS** (parents JOINT/FULL verified; SHAM/REAL YAML diffs limited to aux identity fields)

- ok: **True**
- EXP-H187 (ablang1): ok=True
- EXP-H167 (ablingua): ok=True

## Experiment codes

- `EXP-H340` — Prospective AbLang1 JOINT FULL + SHAM35
- `EXP-H341` — Prospective AbLang1 JOINT FULL + REAL F1_SURFACE35
- `EXP-H342` — Prospective AbLingua JOINT FULL + SHAM35
- `EXP-H343` — Prospective AbLingua JOINT FULL + REAL F1_SURFACE35

## PRIMARY contrast (REAL − SHAM)

| Rep | ΔCV_P | ΔCV_S | ΔCV_mean |
|-----|-------|-------|----------|
| ablang1 | -0.048033 | -0.021103 | -0.034568 |
| ablingua | -0.015863 | -0.063067 | -0.039465 |

- Scheme-level improve count: **4/4**
- Aggregated Primary antibody treatment effect mean=-0.031948 CI=[-0.081355, +0.018820]

**Preregistered internal verdict:** `INTERNAL_SURFACE_REPLICATION_DIRECTIONAL`

## HIGH-tail diagnostic (HIC>11.5; not a success criterion)

- ablang1/primary: n_high=6, ΔMAE_high=-0.1443, Δsigned=+0.1443, ΔMAE_nonHIGH=-0.0443
- ablang1/shadow: n_high=6, ΔMAE_high=+0.0109, Δsigned=-0.0109, ΔMAE_nonHIGH=-0.0223
- ablingua/primary: n_high=6, ΔMAE_high=-0.1446, Δsigned=+0.1446, ΔMAE_nonHIGH=-0.0109
- ablingua/shadow: n_high=6, ΔMAE_high=+0.0548, Δsigned=-0.0548, ΔMAE_nonHIGH=-0.0676

## Secondary (descriptive): REAL vs sequence-only parent

See `HIC_SURFACE_PROSPECTIVE_SECONDARY_VS_PARENT.csv` (not primary causal contrast).

## Embargo confirmation

- Test predictions generated: **YES** (unscored)
- Public/Private/Test Overall in this report: **ABSENT**

