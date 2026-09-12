# TmApp C/D Architecture Exploration Report (T142–T150)

**PLM:** frozen AbLang2 · **Target:** TmApp · **Protocol:** DL_FOLDLOCAL_COSINE_V3
**Controls:** A=EXP-T110 mean=3.2416; B=EXP-T113 mean=3.1391; C0=EXP-T121 mean=3.1377
**Git at analyze:** `ea2449715f3b3726524cc6a103ad01bd1245929a`

## Central table

| Code | Family | Variant | Primary | Shadow | Mean | Worst | Δ control | Bootstrap | Added params | Functional ablation |
|------|--------|---------|---------|--------|------|-------|-----------|-----------|--------------|---------------------|
| EXP-T121 | C | C0 | 3.0232 | 3.2521 | 3.1377 | 3.2521 | 0 | — | 0 | — |
| EXP-T142 | C | C1 | 3.1448 | 3.2270 | 3.1859 | 3.2270 | +0.0483 | P:+0.122[-0.071,+0.303] S:-0.025[-0.197,+0.149] | 6689 | P:+0.068 S:-0.035 |
| EXP-T143 | C | C2 | 3.0525 | 3.4182 | 3.2354 | 3.4182 | +0.0977 | P:+0.029[-0.200,+0.270] S:+0.166[-0.022,+0.357] | 6800 | P:+0.003 S:+0.418 |
| EXP-T144 | C | C3 | 3.1264 | 3.2398 | 3.1831 | 3.2398 | +0.0454 | P:+0.103[-0.112,+0.312] S:-0.012[-0.158,+0.140] | 8608 | P:+0.106 S:-0.008 |
| EXP-T145 | C | C4 | 3.2151 | 3.5526 | 3.3838 | 3.5526 | +0.2462 | P:+0.192[-0.044,+0.430] S:+0.301[+0.076,+0.542] | 4641 | P:+0.016 S:+0.883 |
| EXP-T146 | C | C5 | 3.0931 | 3.1494 | 3.1212 | 3.1494 | -0.0164 | P:+0.070[-0.155,+0.290] S:-0.103[-0.303,+0.106] | 4880 | P:-0.000 S:+0.000 |
| EXP-T147 | D | D1 | 3.2589 | 3.4039 | 3.3314 | 3.4039 | +0.0898 | P:+0.056[-0.201,+0.313] S:+0.124[-0.083,+0.319] | 2320 | P:-0.031 S:-0.067 |
| EXP-T148 | D | D2 | 3.1925 | 3.4360 | 3.3143 | 3.4360 | +0.0727 | P:-0.010[-0.294,+0.242] S:+0.156[-0.082,+0.400] | 4352 | P:-0.008 S:-0.025 |
| EXP-T149 | D | D3 | 3.2233 | 3.3715 | 3.2974 | 3.3715 | +0.0558 | P:+0.020[-0.241,+0.269] S:+0.091[-0.125,+0.304] | 5264 | P:+0.007 S:-0.017 |
| EXP-T150 | D | D4 | 3.0550 | 3.4200 | 3.2375 | 3.4200 | -0.0041 | P:-0.148[-0.404,+0.093] S:+0.140[-0.101,+0.368] | 66304 | P:-0.041 S:-0.032 |

Δ control = mean(P,S)_model − mean(P,S)_control (C→C0, D→A).
Bootstrap: ΔMAE with 95% CI (negative ⇒ candidate better).
Functional ablation: MAE_off − MAE_on (positive ⇒ mechanism helps when enabled).

## Selected

- **C\* = C0 / EXP-T121** — No new C showed convincing incremental value over C0/T121.
- **D\* = D3 / EXP-T149** — D4 capacity flagged; selected simpler D by mean/worst/params (D4 mean gain vs A not large enough to justify +66k params).

Freeze artifacts: `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.yaml`, `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.md`

## Answers

1. **Did data-dependent gating improve C0?** No — no gated C beat C0 on mean with bootstrap + functional criteria. Best new C mean=3.1212 vs C0 3.1377 (C5 lower mean but primary worse + ablation≈0).
2. **Scalar or feature-wise gating?** C1 mean=3.1859 (ablation ΔP=+0.068); C2 mean=3.2354 (ablation ΔP=+0.003). Neither selected over C0; scalar showed clearer primary-side gate use.
3. **Did nonlinear REG processing help?** C3 mean=3.1831, ablation ΔP=+0.106. Not better than C0 overall.
4. **Did a second cross-chain read help?** C4 mean=3.3838, ablation ΔP=+0.016. No — worse and unstable.
5. **Did multiple summary queries help?** C5 mean=3.1212, ablation ΔP=-0.000. Slight mean edge but mechanism unused (≈C0) and primary not improved; not selected.
6. **What is C\*?** **C0 / EXP-T121**
7. **Does global H/L compatibility carry signal without residue cross-attention?** Weak / mixed on AbLang2. Best D mean among D=3.2375 vs A 3.2416. Low-rank D1–D3 do not beat A; D4 capacity-heavy.
8. **Which D implementation is best?** **D3** (EXP-T149) by robust score / parsimony.
9. **What is D\*?** **D3 / EXP-T149**
10. **Are learned H/L effects asymmetric despite symmetric parameterization?** Descriptive only — see diagnostics YAML / asymmetry block in freeze. Shared params; any H≠L activations are input-driven.
11. **Are A/B/C\*/D\* sufficiently distinct conceptually?** Yes: A no cross; B joint residue Transformer; C* REG-level opposite-chain read; D* post-summary pair interaction only.
12. **Ready for PLM × architecture comparison?** **Yes after this freeze.** Do **not** start EXP-T151 / PLM matrix in this batch.

## Parameter notes

D4 token-attention capacity is larger than D1–D3; flagged in prereg incremental counts. Prefer simpler D when tied.

## Artifacts

- Bootstrap: `results/T142_T150_PAIRED_BOOTSTRAP.csv`
- Ablation: `results/T142_T150_FUNCTIONAL_ABLATION.csv`
- Diagnostics: `results/T142_T150_CD_DIAGNOSTICS.yaml`
- Freeze: `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.{yaml,md}`

**STOP.** Next code EXP-T151 reserved; PLM matrix not started.
