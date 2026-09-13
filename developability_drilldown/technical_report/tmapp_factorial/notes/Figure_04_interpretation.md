# Figure 4 — Topology effects vs SEP

## What the figure directly shows

ΔMAE = MAE(topology) − MAE(SEP) for JOINT, REG-SEP, XREG, FUSE within each representation×annotation. Negative values indicate improvement over SEP.

## Strongest patterns

Scratch shows large negative deltas for REG-SEP/JOINT/XREG under several annotations (especially FULL). ESM-2 shows broad negative deltas including FUSE under FULL. ESM-C often shows positive deltas (SEP preferred) under FULL. Antibody PLMs are mixed: AbLang2 contexts favor XREG/JOINT in several annotations, not uniformly.

## Robustness

Prefer patterns that repeat across Primary and Shadow in the bootstrap/topology-gain tables. Single-panel outliers should stay exploratory.

## What we should NOT claim

That negative ΔMAE proves the topology “implements” biological H/L pairing, or that REG tokens correspond to physical interface contacts.

## Possible technical interpretation

Interpretation: pretrained residue information can change how much explicit downstream H/L communication helps relative to Scratch, without implying equivalence of mechanisms.
