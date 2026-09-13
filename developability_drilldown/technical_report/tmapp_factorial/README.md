## Finished report

- `TMAPP_TECHNICAL_REPORT.md` — finalized TmApp-only technical report

# TmApp factorial — technical report analysis package

Presentation-layer analysis only. No new training. Historical experiment IDs and result files are not rewritten.

## Source of truth

- Machine results: `results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv`
- Human master: `data/tmapp_factorial_human_master.csv` (200 rows)
- Terminology: `terminology.yaml`

## REG definition

A REG is **a learned chain-level token that gathers information from the residues processed by the downstream Transformer**.

## Topology display IDs

| legacy | ID | display |
| --- | --- | --- |
| A | SEP | Separate H/L |
| B1 | JOINT | Full Joint |
| B2 | REG-SEP | Joint Residues, Separate REGs |
| C | XREG | Cross-REG Read |
| D | FUSE | REG Fusion |

## AbLang2 / CurrAb naming

Uses `representation_context`, not historical allocation labels:

- AbLang2 (separate-chain) ← `ablang2_paired` / SEPARATE_CHAIN
- AbLang2 (paired H/L) ← `ablang2_unpaired` / PAIRED_NATIVE
- CurrAb (paired H/L) ← `currab_paired` / PAIRED_NATIVE
- CurrAb (separate-chain) ← `currab_unpaired` / SEPARATE_CHAIN

## Figures

- `Figure_01_factorial_topology_schematic.png` (+ PDF)
- `Figure_02_best_per_representation_mean.png` (+ PDF)
- `Figure_02_best_per_representation_worst.png` (+ PDF)
- `Figure_03A_absolute_scratch.png` (+ PDF)
- `Figure_03B_absolute_general_protein.png` (+ PDF)
- `Figure_03C_absolute_antibody_ablang1_ablingua.png` (+ PDF)
- `Figure_03D_absolute_ablang2.png` (+ PDF)
- `Figure_03E_absolute_currab.png` (+ PDF)
- `Figure_03_individual_AbLang1.png` (+ PDF)
- `Figure_03_individual_AbLang2_paired_H_L.png` (+ PDF)
- `Figure_03_individual_AbLang2_separate-chain.png` (+ PDF)
- `Figure_03_individual_AbLingua.png` (+ PDF)
- `Figure_03_individual_CurrAb_paired_H_L.png` (+ PDF)
- `Figure_03_individual_CurrAb_separate-chain.png` (+ PDF)
- `Figure_03_individual_ESM-1b.png` (+ PDF)
- `Figure_03_individual_ESM-2.png` (+ PDF)
- `Figure_03_individual_ESM-C_600M.png` (+ PDF)
- `Figure_03_individual_Scratch.png` (+ PDF)
- `Figure_04_topology_effects_vs_SEP.png` (+ PDF)
- `Figure_05_annotation_effects_vs_BASE.png` (+ PDF)
- `Figure_06_paired_minus_separate.png` (+ PDF)
- `Figure_07_primary_shadow_robustness.png` (+ PDF)

## Tables

- `Table_01_representation_definitions.csv`
- `Table_02_topology_definitions.csv`
- `Table_03_annotation_definitions.csv`
- `Table_04_best_cell_per_representation.csv`
- `Table_05_top15_by_mean_ps.csv`
- `Table_05b_top15_by_worst_ps.csv`

## Notes

- `notes/Figure_XX_interpretation.md`
- `notes/first_pass_scientific_questions.md`
- `TECHNICAL_REPORT_OUTLINE.md`

## Rebuild

```bash
PYTHONPATH=scripts:models ../.venv_b1/bin/python technical_report/tmapp_factorial/scripts/build_report_package.py
```
