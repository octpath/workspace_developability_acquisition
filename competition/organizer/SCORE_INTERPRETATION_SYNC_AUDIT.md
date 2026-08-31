# Score interpretation documentation — sync audit

Date: 2026-08-31  
Sources of truth: `SCORE_INTERPRETATION_STATS.json`  
Documents:

| Level | Path |
|---|---|
| 1 Technical (EN) | `SCORE_INTERPRETATION_TECHNICAL.md` |
| 1 Technical (JA) | `SCORE_INTERPRETATION_TECHNICAL_JA.md` |
| 2 Explainer (JA) | `SCORE_INTERPRETATION_EXPLAINER_JA.md` |
| 3 Participant EN | `../participant/SCORE_GUIDE.md` |
| 3 Participant JA | `../participant/SCORE_GUIDE_ja.md` |

Overall: **PASS**

---

## Key-statement checks

| Statement | Stats JSON | L1 | L2 | L3 EN | L3 JA | Result |
|---|---|---|---|---|---|---|
| Units TmApp °C / HIC min | yes | yes | yes | yes | yes | **PASS** |
| Primary metric MAE | yes | yes | yes | yes | yes | **PASS** |
| Local-CV TmApp baseline ~3.54 / ~3.5 | 3.543 | yes | yes | ~3.5 | ~3.5 | **PASS** |
| Local-CV HIC baseline ~0.518 / ~0.52 | 0.518 | yes | yes | ~0.52 | ~0.52 | **PASS** |
| Strong TmApp ~2.8–3.0 | yes | yes | yes | yes | yes | **PASS** |
| Strong HIC ~0.45–0.48 | yes | yes | yes | yes | yes | **PASS** |
| Exceptional TmApp <~2.5 | yes | yes | yes | yes | yes | **PASS** |
| Exceptional HIC <~0.40 | yes | yes | yes | yes | yes | **PASS** |
| TmApp skill ~21.6% | 21.6 | yes | yes | (omitted; OK for L3) | (omitted) | **PASS** |
| HIC skill ~10.0% | 10.0 | yes | yes | (omitted; OK for L3) | (omitted) | **PASS** |
| Landmarks are local-CV, not Public targets | yes | yes | yes | yes | yes | **PASS** |
| Public N=81 noisy | — | yes | yes | yes | yes | **PASS** |
| Not industrial / developability thresholds | — | yes | yes | yes | yes | **PASS** |
| Not assay replacement | — | yes | yes | yes | yes | **PASS** |
| MAE ≠ pairwise resolution | — | yes | yes | yes | yes | **PASS** |
| Shehata-specific noise floor not claimed | yes | yes | yes | yes | yes | **PASS** |
| Related DSF sub-degree–~1 °C wording | yes | yes | yes | qualitative | qualitative | **PASS** |
| Closely matched HIC ~0.12 min (Jain) | yes | yes | yes | ~0.1 scale mention | ~0.1 scale mention | **PASS** |
| N=324 distribution stats used | yes | yes | yes | (not required) | (not required) | **PASS** |
| No organizer model architecture secrets in L3 | — | details internal | internal | approximate only | approximate only | **PASS** |
| Release CSVs unmodified | — | docs only | docs only | docs only | docs only | **PASS** |

---

## Quality tests

### Level 1 (senior Q1–Q8)

All eight reviewer questions are answerable from `SCORE_INTERPRETATION_TECHNICAL.md` Appendix — **PASS**.

### Level 2 (three rulers)

Baseline → biological spread → assay repeatability framework present — **PASS**.

### Level 3 (30-second answers)

Beat ~3.5 / ~0.52; strong ~3.0 / ~0.47; exceptional <2.5 / <0.40; not assay replacement — **PASS**.

---

## Unresolved scientific uncertainties (documented, not blockers)

1. No direct Shehata replicate table for DSF or HIC technical SD in available packaging evidence.  
2. Related-assay repeatability numbers remain Grade B/C (DSF) or A/B close-protocol (HIC), not proven Shehata floors.  
3. Exact mapping of every external DSF protocol to Shehata Fab DSF is imperfect.

These uncertainties are explicitly caveated; they do **not** block documentation readiness.

---

## Final status

**SCORE_INTERPRETATION_DOCUMENTATION_READY**
