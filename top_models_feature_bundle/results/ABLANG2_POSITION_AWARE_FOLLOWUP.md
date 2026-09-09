# AbLang2 × antibody-position-aware Transformer follow-up

TmApp-only pre-registered follow-up. CV selection used Primary/Shadow only; Public/Private are POSTMORTEM.

## Verdict

**PARTIALLY_SUPPORTED / INCONCLUSIVE** for “AbLang2 residue + explicit IMGT/FR-CDR re-aggregation explains TmApp strength.”

- Best sequence: `AL2F3_FULL_MEAN` (worst ≈ 3.043) — better than AbLang2 minimal/full-concat, still far from linear T1.
- Annotation AL2F1→AL2F2: Shadow improves, Primary does not (mixed).
- Mean merge beats concat; region-gate does **not** improve and weights stay ~uniform (CDR3 not elevated).
- Best fusion (BIOEMU+MPNN) worst ≈ 2.835 — does not beat old TMF2 fusion (~2.773).
- Cross-family equal-mean with AL2F3 included: worst ≈ 2.735 vs Phase-1 2.738 (**tiny** CV gain).

See `results/ablang2_followup/` for tables and `ABLANG2_POSITION_AWARE_FOLLOWUP_JA.md` for Q1–Q10.
