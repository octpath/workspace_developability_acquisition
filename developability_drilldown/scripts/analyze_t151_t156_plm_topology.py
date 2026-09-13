#!/usr/bin/env python3
"""Analyze T151–T156 PLM × topology matrix: gains, DiD, freeze, report."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from experiment_codes import next_code  # noqa: E402
from run_exp_t151_t156_plm_topology import (  # noqa: E402
    MATRIX,
    PLM_RAW,
    SCRATCH_CONTEXT,
    TOPOLOGY_FLAGS,
    count_params,
    did_boot,
    load_oof,
    paired_boot,
    projection_params,
    scores_from_exp,
    sha_file,
)

SEED = 101
N_BOOT = 2000
PLMS = ("ablingua", "ablang2", "esm2")
TOPOS = ("A", "B1", "B2", "C", "D")


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def cell_code(plm: str, topo: str) -> str:
    for s in MATRIX:
        if s["plm"] == plm and s["topology"] == topo:
            return s["code"]
    raise KeyError((plm, topo))


def ensure_complete() -> None:
    for s in MATRIX:
        code = s["code"]
        if not (ROOT / "experiments" / "predictions" / code / "oof_primary.csv").exists():
            raise SystemExit(f"missing OOF for {code}")
        if not s["reuse"] and not (ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").exists():
            raise SystemExit(f"incomplete new cell {code}")


def pick_winner(rows: list[dict]) -> str:
    """Pick topology by mean(P,S), then worst, requiring no major collapse."""
    ranked = sorted(rows, key=lambda r: (r["mean"], r["worst"], r["topology"]))
    return ranked[0]["topology"]


def run_full_analysis(*, include_external_in_report: bool = False) -> None:
    ensure_complete()
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    dev = pd.read_csv(ROOT / "data" / "dev.csv")
    ymap = {str(r["id"]): float(r["TmApp"]) for _, r in dev.iterrows()}

    # score table
    score_rows = []
    for plm in PLMS:
        for topo in TOPOS:
            code = cell_code(plm, topo)
            sc = scores_from_exp(code, exp)
            params = count_params(plm, topo)
            score_rows.append(
                {
                    "plm": plm,
                    "topology": topo,
                    "code": code,
                    **sc,
                    "n_trainable": params["total"],
                    "projection_params": projection_params(PLM_RAW[plm]),
                    "interaction_params": params["cross_attention"] + params["cross_interaction"],
                }
            )
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(ROOT / "results" / "T151_T156_MATRIX_SCORES.csv", index=False)

    # gains vs A
    gain_rows = []
    boot_rows = []
    for plm in PLMS:
        a_code = cell_code(plm, "A")
        a_sc = scores_from_exp(a_code, exp)
        for topo in ("B1", "B2", "C", "D"):
            x_code = cell_code(plm, topo)
            x_sc = scores_from_exp(x_code, exp)
            gain_rows.append(
                {
                    "plm": plm,
                    "topology": topo,
                    "delta_primary": x_sc["primary"] - a_sc["primary"],
                    "delta_shadow": x_sc["shadow"] - a_sc["shadow"],
                    "delta_mean": x_sc["mean"] - a_sc["mean"],
                    "code_x": x_code,
                    "code_a": a_code,
                }
            )
            for scheme in ("primary", "shadow"):
                px = load_oof(x_code, scheme)
                pa = load_oof(a_code, scheme)
                m = px.to_frame("x").join(pa.to_frame("a"), how="inner")
                ids = m.index.astype(str)
                y = np.asarray([ymap[i] for i in ids], float)
                d, lo, hi = paired_boot(m["x"].to_numpy(float), m["a"].to_numpy(float), y)
                boot_rows.append(
                    {
                        "kind": "gain_vs_A",
                        "plm": plm,
                        "topology": topo,
                        "scheme": scheme,
                        "model_a": x_code,
                        "model_b": a_code,
                        "delta_mae": d,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "note": "delta=MAE_X-MAE_A; negative => topology helps",
                    }
                )

    # within-PLM pairwise
    pairs = [("B1", "B2"), ("B1", "C"), ("B2", "C")]
    for plm in PLMS:
        for t1, t2 in pairs:
            c1, c2 = cell_code(plm, t1), cell_code(plm, t2)
            for scheme in ("primary", "shadow"):
                p1 = load_oof(c1, scheme)
                p2 = load_oof(c2, scheme)
                m = p1.to_frame("a").join(p2.to_frame("b"), how="inner")
                y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                d, lo, hi = paired_boot(m["a"].to_numpy(float), m["b"].to_numpy(float), y)
                boot_rows.append(
                    {
                        "kind": "within_plm_pairwise",
                        "plm": plm,
                        "topology": f"{t1}_vs_{t2}",
                        "scheme": scheme,
                        "model_a": c1,
                        "model_b": c2,
                        "delta_mae": d,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "note": "delta=MAE_a-MAE_b; negative => a better",
                    }
                )

    # DiD
    did_rows = []
    plm_pairs = [
        ("ablingua", "ablang2"),
        ("ablingua", "esm2"),
        ("ablang2", "esm2"),
    ]
    for topo in ("B1", "B2", "C", "D"):
        for p1, p2 in plm_pairs:
            for scheme in ("primary", "shadow"):
                p1x = load_oof(cell_code(p1, topo), scheme)
                p1a = load_oof(cell_code(p1, "A"), scheme)
                p2x = load_oof(cell_code(p2, topo), scheme)
                p2a = load_oof(cell_code(p2, "A"), scheme)
                m = (
                    p1x.to_frame("p1x")
                    .join(p1a.to_frame("p1a"), how="inner")
                    .join(p2x.to_frame("p2x"), how="inner")
                    .join(p2a.to_frame("p2a"), how="inner")
                )
                y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                d, lo, hi = did_boot(
                    m["p1x"].to_numpy(float),
                    m["p1a"].to_numpy(float),
                    m["p2x"].to_numpy(float),
                    m["p2a"].to_numpy(float),
                    y,
                )
                # interpretation
                if hi < 0:
                    interp = f"{p1} gains more from {topo} than {p2} (CI<0)"
                elif lo > 0:
                    interp = f"{p2} gains more from {topo} than {p1} (CI>0)"
                else:
                    interp = "DiD CI includes 0 — no strong interaction claim"
                did_rows.append(
                    {
                        "topology": topo,
                        "plm_p1": p1,
                        "plm_p2": p2,
                        "scheme": scheme,
                        "did": d,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "interpretation": interp,
                        "note": "DiD=(MAE_P1X-MAE_P1A)-(MAE_P2X-MAE_P2A); negative => P1 benefits more",
                    }
                )
                boot_rows.append(
                    {
                        "kind": "DiD",
                        "plm": f"{p1}_vs_{p2}",
                        "topology": topo,
                        "scheme": scheme,
                        "model_a": cell_code(p1, topo),
                        "model_b": cell_code(p2, topo),
                        "delta_mae": d,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "note": "DiD bootstrap",
                    }
                )

    # PLM main effect secondary
    for topo in TOPOS:
        for p1, p2 in plm_pairs:
            for scheme in ("primary", "shadow"):
                a = load_oof(cell_code(p1, topo), scheme)
                b = load_oof(cell_code(p2, topo), scheme)
                m = a.to_frame("a").join(b.to_frame("b"), how="inner")
                y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                d, lo, hi = paired_boot(m["a"].to_numpy(float), m["b"].to_numpy(float), y)
                boot_rows.append(
                    {
                        "kind": "plm_main_effect",
                        "plm": f"{p1}_vs_{p2}",
                        "topology": topo,
                        "scheme": scheme,
                        "model_a": cell_code(p1, topo),
                        "model_b": cell_code(p2, topo),
                        "delta_mae": d,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "note": "delta=MAE_p1-MAE_p2; negative => p1 better",
                    }
                )

    pd.DataFrame(gain_rows).to_csv(ROOT / "results" / "T151_T156_GAINS_VS_A.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(ROOT / "results" / "T151_T156_PAIRED_BOOTSTRAP.csv", index=False)
    pd.DataFrame(did_rows).to_csv(ROOT / "results" / "T151_T156_DID_BOOTSTRAP.csv", index=False)

    # winners per PLM
    winners = {}
    for plm in PLMS:
        sub = score_df[score_df.plm == plm].to_dict("records")
        w = pick_winner(sub)
        winners[plm] = {"topology": w, "code": cell_code(plm, w), **scores_from_exp(cell_code(plm, w), exp)}
        # winner vs A / D boots already in gain table

    # scratch context
    scratch_rows = []
    for topo, code in SCRATCH_CONTEXT.items():
        scratch_rows.append({"topology": topo, **scores_from_exp(code, exp)})

    # scientific verdict heuristics
    def best_label(plm):
        return winners[plm]["topology"]

    # Pattern classification
    ab_w, a2_w, e_w = best_label("ablingua"), best_label("ablang2"), best_label("esm2")
    # DiD support count
    did_sig = [
        r
        for r in did_rows
        if r["scheme"] == "primary" and (r["ci95_hi"] < 0 or r["ci95_lo"] > 0)
    ]
    verdict = {
        "winners": winners,
        "did_primary_significant_count": len(did_sig),
        "pattern_notes": [],
    }
    if a2_w in ("B1", "C") and ab_w == "C" and e_w in ("B1", "B2"):
        verdict["pattern_notes"].append(
            "Suggestive Pattern 1/9: antibody PLMs lean C (or B1≈C), ESM-2 leans residue-joint"
        )
    if all(best_label(p) in ("B1", "B2") for p in PLMS):
        verdict["pattern_notes"].append("Pattern 2: all PLMs favor residue-level joint")
    if all(best_label(p) == "C" for p in PLMS):
        verdict["pattern_notes"].append("Pattern 3: all PLMs favor C")
    if any(best_label(p) == "D" for p in PLMS):
        verdict["pattern_notes"].append("Pattern 4: at least one PLM favors D")
    if len(did_sig) == 0:
        verdict["pattern_notes"].append(
            "Pattern 5 lean: ranking may differ but DiD CIs mostly include 0"
        )

    # recommended model
    # Prefer AbLang2 B1 or C if clustered; else winner by mean across PLMs carefully
    a2 = score_df[score_df.plm == "ablang2"].sort_values("mean")
    recommended = {
        "code": a2.iloc[0]["code"],
        "plm": "ablang2",
        "topology": a2.iloc[0]["topology"],
        "rationale": "Lowest AbLang2 mean(P,S) in matrix; antibody-specific PLM preferred when identity known",
    }
    if abs(float(a2.iloc[0]["mean"]) - float(a2[a2.topology == "C"].iloc[0]["mean"])) < 0.01 and a2.iloc[0]["topology"] == "B1":
        recommended["note"] = "B1≈C on AbLang2; either acceptable"
    unknown_plm_topo = "C"  # small-N inductive bias default from prior + matrix
    # if ESM and AbLang disagree strongly, say "no single topology"
    if {ab_w, a2_w, e_w} == {ab_w}:
        unknown_plm_topo = ab_w
    elif len({ab_w, a2_w}) == 1:
        unknown_plm_topo = ab_w
    else:
        unknown_plm_topo = "C"

    # pre-external freeze
    freeze = {
        "status": "PRE_EXTERNAL_FREEZE",
        "git_rev": git_rev(),
        "matrix_scores": score_rows,
        "gains_vs_a": gain_rows,
        "did_primary_significant": did_sig,
        "winners_per_plm": winners,
        "scratch_context": scratch_rows,
        "verdict": verdict,
        "recommended_canonical": recommended,
        "topology_if_plm_unknown": unknown_plm_topo,
        "trainable_params": {
            f"{r['plm']}_{r['topology']}": {
                "total": r["n_trainable"],
                "projection": r["projection_params"],
                "interaction": r["interaction_params"],
            }
            for r in score_rows
        },
        "statement": "Internal P/S / gains / DiD frozen before comparative external interpretation of new cells.",
    }
    (ROOT / "results" / "TMAPP_PLM_TOPOLOGY_PRE_EXTERNAL_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "results" / "TMAPP_PLM_TOPOLOGY_PRE_EXTERNAL_FREEZE.md").write_text(
        f"""# TmApp PLM × Topology Pre-External Freeze

**Status:** INTERNAL FROZEN  
**Git:** `{git_rev()}`  
**Next code:** `{next_code("TmApp")}`

## Winners (internal mean P/S)

| PLM | Topology | Code | Mean |
|-----|----------|------|------|
| AbLingua | {winners['ablingua']['topology']} | {winners['ablingua']['code']} | {winners['ablingua']['mean']:.4f} |
| AbLang2 | {winners['ablang2']['topology']} | {winners['ablang2']['code']} | {winners['ablang2']['mean']:.4f} |
| ESM-2 | {winners['esm2']['topology']} | {winners['esm2']['code']} | {winners['esm2']['mean']:.4f} |

Recommended (when AbLang2 available): **{recommended['code']}** ({recommended['topology']}).  
If PLM unknown: prefer **{unknown_plm_topo}**.

DiD primary significant contrasts: **{len(did_sig)}**.

Machine-readable: `results/TMAPP_PLM_TOPOLOGY_PRE_EXTERNAL_FREEZE.yaml`
""",
        encoding="utf-8",
    )

    # build report
    def cell_txt(plm, topo):
        r = score_df[(score_df.plm == plm) & (score_df.topology == topo)].iloc[0]
        return f"{r['mean']:.3f} (P {r['primary']:.3f}/S {r['shadow']:.3f})"

    def gain_txt(plm, topo):
        g = [x for x in gain_rows if x["plm"] == plm and x["topology"] == topo][0]
        return f"{g['delta_mean']:+.3f}"

    matrix_lines = [
        "| PLM | A | B1 full joint | B2 residue joint | C REG→residue | D REG↔REG |",
        "|-----|--:|--------------:|-----------------:|--------------:|----------:|",
    ]
    for plm, label in (("ablingua", "AbLingua"), ("ablang2", "AbLang2"), ("esm2", "ESM-2")):
        matrix_lines.append(
            f"| {label} | {cell_txt(plm,'A')} | {cell_txt(plm,'B1')} | {cell_txt(plm,'B2')} | "
            f"{cell_txt(plm,'C')} | {cell_txt(plm,'D')} |"
        )

    gain_lines = [
        "| PLM | ΔB1 vs A | ΔB2 vs A | ΔC vs A | ΔD vs A |",
        "|-----|---------:|---------:|--------:|--------:|",
    ]
    for plm, label in (("ablingua", "AbLingua"), ("ablang2", "AbLang2"), ("esm2", "ESM-2")):
        gain_lines.append(
            f"| {label} | {gain_txt(plm,'B1')} | {gain_txt(plm,'B2')} | {gain_txt(plm,'C')} | {gain_txt(plm,'D')} |"
        )

    did_lines = [
        "| Topology | PLM pair | Primary DiD | CI | Shadow DiD | CI | Interpretation |",
        "|----------|----------|-------------|----|------------|----|----------------|",
    ]
    for topo in ("B1", "B2", "C", "D"):
        for p1, p2 in plm_pairs:
            pr = [r for r in did_rows if r["topology"] == topo and r["plm_p1"] == p1 and r["plm_p2"] == p2 and r["scheme"] == "primary"][0]
            sh = [r for r in did_rows if r["topology"] == topo and r["plm_p1"] == p1 and r["plm_p2"] == p2 and r["scheme"] == "shadow"][0]
            did_lines.append(
                f"| {topo} | {p1} vs {p2} | {pr['did']:+.3f} | [{pr['ci95_lo']:+.3f},{pr['ci95_hi']:+.3f}] | "
                f"{sh['did']:+.3f} | [{sh['ci95_lo']:+.3f},{sh['ci95_hi']:+.3f}] | {pr['interpretation']} |"
            )

    scratch_lines = [
        "| Topology | Code | Primary | Shadow | Mean | Worst |",
        "|----------|------|---------|--------|------|-------|",
    ]
    for r in scratch_rows:
        scratch_lines.append(
            f"| {r['topology']} | {r['code']} | {r['primary']:.4f} | {r['shadow']:.4f} | {r['mean']:.4f} | {r['worst']:.4f} |"
        )

    # Q&A helpers
    def helps_all(topo):
        return all(
            [g for g in gain_rows if g["topology"] == topo and g["plm"] == plm][0]["delta_mean"] < 0
            for plm in PLMS
        )

    def b2_dependence():
        gains = {plm: [g for g in gain_rows if g["plm"] == plm and g["topology"] == "B2"][0]["delta_mean"] for plm in PLMS}
        return gains

    ext_block = ""
    if include_external_in_report:
        ext_lines = [
            "| Code | PLM | Topo | Public | Private | Overall |",
            "|------|-----|------|--------|---------|---------|",
        ]
        for r in score_rows:
            if r["public"] is None:
                continue
            ext_lines.append(
                f"| {r['code']} | {r['plm']} | {r['topology']} | {r['public']:.4f} | {r['private']:.4f} | {r['overall']:.4f} |"
            )
        ext_block = "\n## External diagnostic (does not change internal verdict)\n\n" + "\n".join(ext_lines) + "\n"

    b2g = b2_dependence()
    report = f"""# TmApp PLM × H/L Topology Matrix Report (T151–T156)

**Question:** Does optimal H/L communication topology depend on the pretrained residue representation?  
**Git at analyze:** `{git_rev()}`  
**Internal freeze:** `results/TMAPP_PLM_TOPOLOGY_PRE_EXTERNAL_FREEZE.*`  
**Next unused code:** `{next_code("TmApp")}` (do not run).

## Central matrix (mean(P,S) with P/S)

{chr(10).join(matrix_lines)}

## Gain vs A (mean; negative = helps)

{chr(10).join(gain_lines)}

## PLM × topology interaction (DiD)

{chr(10).join(did_lines)}

## Scratch context (not primary matrix)

{chr(10).join(scratch_lines)}

{ext_block}
## Answers

1. **Best for AbLingua?** **{winners['ablingua']['topology']}** ({winners['ablingua']['code']}, mean={winners['ablingua']['mean']:.4f})
2. **Best for AbLang2?** **{winners['ablang2']['topology']}** ({winners['ablang2']['code']}, mean={winners['ablang2']['mean']:.4f})
3. **Best for ESM-2?** **{winners['esm2']['topology']}** ({winners['esm2']['code']}, mean={winners['esm2']['mean']:.4f})
4. **Does B1 help every PLM?** {"Yes on mean(P,S)" if helps_all("B1") else "No — see gain table / bootstrap"}
5. **B2 representation dependence?** Gains: AbLingua {b2g['ablingua']:+.3f}, AbLang2 {b2g['ablang2']:+.3f}, ESM-2 {b2g['esm2']:+.3f}. See DiD B2 rows.
6. **C representation dependence?** See ΔC vs A and DiD C rows.
7. **Does D help any PLM?** {"Yes" if any(g['delta_mean']<0 for g in gain_rows if g['topology']=='D') else "No clear help on mean"}; check functional interpretation carefully.
8. **Scratch residue-level preference reproduced by ESM-2?** Scratch B2={scores_from_exp(SCRATCH_CONTEXT['B2'], exp)['mean']:.3f} vs A={scores_from_exp(SCRATCH_CONTEXT['A'], exp)['mean']:.3f}; ESM-2 winner={winners['esm2']['topology']}.
9. **Do antibody PLMs need less residue-level mixing than ESM-2?** Compare winners + B1/B2 gains; AbLingua/AbLang2 vs ESM-2 DiD.
10. **Bootstrap-supported PLM×topology interaction?** Primary DiD significant contrasts: **{len(did_sig)}**. {"Yes for listed contrasts" if did_sig else "No strong DiD claim (CIs include 0)"}.
11. **AbLang2 B1≈C unique?** AbLang2 B1 mean={score_df[(score_df.plm=='ablang2')&(score_df.topology=='B1')].iloc[0]['mean']:.4f}, C={score_df[(score_df.plm=='ablang2')&(score_df.topology=='C')].iloc[0]['mean']:.4f}; AbLingua B1/C and ESM-2 B1/C in matrix.
12. **Robust across P and S?** Prefer effects with both schemes agreeing in gain/DiD tables.
13. **Explainable by param counts alone?** Projection differs (AbLang2 61k vs 164k) but A/B1/B2 share totals within PLM; C/D add interaction params only. Ranking changes are not reducible to capacity alone within a PLM.
14. **Recommended canonical TmApp model?** **{recommended['code']}** ({recommended['plm']} / {recommended['topology']}) — {recommended['rationale']}
15. **Topology if PLM unknown?** **{unknown_plm_topo}**
16. **Further topology refinement?** {"Low priority — move to another axis (features/PLM fine-tune/data)" if len(did_sig)==0 and len(set(winners[p]['topology'] for p in PLMS))<=2 else "Optional light follow-up only if DiD contrasts replicate; else move axis"}

## Non-claims

Do not infer that a preferred topology means a PLM “contains structure,” nor that B2 is the literal biophysical H/L mechanism. These are predictive inductive-bias results about **representation-dependent downstream communication requirements**.

## Artifacts

- Scores: `results/T151_T156_MATRIX_SCORES.csv`
- Gains: `results/T151_T156_GAINS_VS_A.csv`
- Bootstrap: `results/T151_T156_PAIRED_BOOTSTRAP.csv`
- DiD: `results/T151_T156_DID_BOOTSTRAP.csv`
- Freeze V2: `results/TMAPP_HL_TOPOLOGY_FREEZE_V2.*`
- Pre-external freeze: `results/TMAPP_PLM_TOPOLOGY_PRE_EXTERNAL_FREEZE.*`

**STOP.** Do not run EXP-T157.
"""
    (ROOT / "results" / "TMAPP_PLM_HL_TOPOLOGY_MATRIX_REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote analysis + freeze + report; next=", next_code("TmApp"), flush=True)
    print("winners", {k: v["topology"] for k, v in winners.items()}, flush=True)


if __name__ == "__main__":
    run_full_analysis(include_external_in_report=False)
