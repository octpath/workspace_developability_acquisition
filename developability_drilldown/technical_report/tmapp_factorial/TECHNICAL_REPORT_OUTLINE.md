# TECHNICAL_REPORT_OUTLINE.md

Proposed structure for a later technical report. **Do not treat this as polished manuscript prose.**

1. Motivation
2. Dataset and prediction task
3. Experimental design (Representation × Topology × Annotation factorial)
4. Residue representations (display names; `representation_context` as ground truth)
5. H/L topology definitions (SEP / JOINT / REG-SEP / XREG / FUSE; REG definition)
6. Annotation definitions (BASE / IMGT / REGION / FULL; position+chain always on)
7. Evaluation protocol (DL_FOLDLOCAL_COSINE_V3; Primary/Shadow; seed 101)
8. Overall performance landscape (Figures 2–3; Tables 4–5)
9. Representation-dependent topology effects (Figure 4)
10. Representation-dependent annotation effects (Figure 5)
11. Paired vs separate PLM context (Figure 6; matched checkpoints)
12. Scratch as control
13. Robustness and uncertainty (Figure 7; bootstrap tables)
14. External diagnostic (post-internal-freeze only)
15. Limitations
16. Conclusions (predictive inductive-bias evidence; non-claims)

Figure/table inventory lives in `README.md`.
