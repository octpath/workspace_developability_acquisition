# VHL-ANGLE_v1_TECHNICAL_REVIEW

Registered from Gate2C pre-audit (`GATE2AB_CONSISTENCY_AUDIT.md`).

## Status

ABB2 ABangle path: **TECHNICAL_MAPPING_CONCERN** (geometry not trustworthy for cross-generator comparison).  
ESMFold / Boltz2: **TECHNICALLY_VALID**.

## Issue

ABB2 PDBs use Chothia-like residue numbering with gaps. Vendored ABangle `AtomIterator` inserts `X` placeholders → ANARCI renumbering produces duplicate/truncated Chothia numbers → implausible `dc≈6.8 Å` (vs ≈16 Å on ESM/Boltz).

## Action

Do **not** fix VHL-ANGLE_v1 in place. Future remediation (gap-aware sequence extraction / skip re-numbering when already Chothia) belongs in a reviewed follow-up, not silent v1 edits.
