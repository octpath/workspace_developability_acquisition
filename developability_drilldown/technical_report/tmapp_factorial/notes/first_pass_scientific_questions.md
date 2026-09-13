# First-pass scientific questions (figure-oriented)

REG definition used throughout: a learned chain-level token that gathers information from the residues processed by the downstream Transformer.

## Q1. How strongly does optimal H/L topology depend on residue representation?

Supported by Figures 2–4: best topologies differ across representations (Scratch REG-SEP; AbLang2 XREG; ESM-C JOINT; ESM-2 FUSE; ESM-1b XREG). Topology Δ vs SEP maps are not interchangeable.

## Q2. Does explicit antibody annotation help uniformly?

No. Figure 5 shows signed Δ vs BASE that change with both representation and topology. REGION/FULL are not universal improvements.

## Q3. Does pretrained residue information reduce the need for downstream H/L interaction vs Scratch?

Partially consistent with Figure 4: Scratch often shows large negative Δ for interaction topologies; some PLMs (e.g., ESM-C under FULL) prefer SEP. This is comparative predictive evidence, not a mechanistic reduction proof.

## Q4. Does pretrained information reduce need for explicit IMGT/CDR-FR?

Not uniformly (Figure 5). Some PLM cells improve under BASE relative to FULL; others benefit from REGION. Avoid claiming that annotations are “already inside” the PLM.

## Q5. Within the same checkpoint, what changes with paired H/L inference context?

Figure 6: PAIR−SEPARATE surfaces are non-flat for AbLang2 and CurrAb. Absolute best cells can be similar (AbLang2 XREG+REGION in both contexts) while average Δ still favors one context.

## Q6. Are AbLang2 and CurrAb paired-vs-separate patterns similar?

Figure 6 suggests **not identical**: CurrAb’s average paired advantage is clearer; AbLang2 is more annotation/topology-dependent. Shared scale enables direct visual comparison.

## Q7. Simple monotonic ordering by newer/general/antibody PLMs?

Figure 2/3: **no simple monotonic story**. Antibody-oriented AbLang2 leads this matrix, but ESM-C can beat several antibody PLMs depending on cell choice; age/domain labels are descriptive only.

## Special cases (examples, not proof points)

- Scratch: REG-SEP + FULL best (EXP-T096, mean=3.258)
- AbLang2: XREG + REGION leads both contexts (EXP-T205, EXP-T225)
- ESM-C: JOINT + BASE (EXP-T285, mean=3.123) far better than a C+FULL-only smoke reading
- ESM-1b: XREG + BASE (EXP-T253)
- ESM-2: FUSE + FULL (EXP-T156)
- CurrAb: average paired-context advantage with near-tie of best separate/paired cells
