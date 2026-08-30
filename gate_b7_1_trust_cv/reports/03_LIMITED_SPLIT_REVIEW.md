# Phase 3 — Limited split review

## Restriction
No new Public/Private mask search. Conceptual comparison only against previously shortlisted candidates.

## Candidates considered
- **CAND_12528** (production candidate; B6.1 bakeoff SELECT)
- **CAND_04974** (B6 selected / stronger pre-model balance)
- **CAND_12207** (compromise; failed HIC Public→Private floor in B6.1)
- **CURRENT_BASELINE_SPLIT** (control)

## Trust-CV remapping status
Full Trust-CV re-audit on alternate splits requires remapping every model’s Public/Private predictions onto each candidate’s ID lists. That remapping was **not** performed here (and is out of scope for this limited review). Therefore no alternate can claim measured TrustCV_rate dominance on the B7.1 validation bank.

## Prior bakeoff evidence (unchanged)
CAND_12528 already won on participant-selection harm (lowest Public-winner Private regret on both targets) with acceptable transfer. CAND_04974 is closer on balance/local stability but worse on selection regret. CAND_12207 fails the HIC transfer floor.

## Educational property after B7.1 stress
On CAND_12528, material CV↔Public disagreements are uncommon for HIC; when they occur, Private more often follows Public (TrustCV_rate≈0.13). TmApp is near coin-flip (≈0.48). This **fails** a strong “Trust CV on disagreement” claim, but does **not** by itself create STRONG_REASON_TO_SWITCH without a remapped alternative that improves both targets without harming B6 safety.

## Decision
**KEEP CAND_12528.** Evidence reaches at most MARGINAL_PREFERENCE speculation, not STRONG_REASON_TO_SWITCH.

Recommended FINAL wording: `FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED`.
