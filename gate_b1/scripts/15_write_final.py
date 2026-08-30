#!/usr/bin/env python3
"""Assemble GATE_B1_FINAL.md and remaining reports from metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import CONFIG, METRICS, REPORTS, TARGET_COLS, ensure_dirs, read_json, write_json  # noqa: E402


def best_scores(df, target, split="canonical"):
    sub = df[(df.target == target) & (df.split == split) & (df.feature_class != "ILLEGAL_FOR_PARTICIPANTS")]
    def b(pred):
        m = sub[sub.representation.map(pred)]
        if len(m) == 0:
            return None
        row = m.sort_values("cv_spearman", ascending=False).iloc[0]
        return {
            "rep": row.representation,
            "model": row.model,
            "cv": float(row.cv_spearman),
            "pub": float(row.public_spearman) if pd.notna(row.public_spearman) else None,
            "priv": float(row.private_spearman) if pd.notna(row.private_spearman) else None,
        }
    return {
        "simple": b(lambda r: r in ("A2_physchem", "B_cdr_descriptors", "B_cdr_hydrophobicity_only", "A1_aa_comp")),
        "bio": b(lambda r: r == "C_BIO_SHORTCUT"),
        "plm_ab": b(lambda r: "ablang" in str(r).lower()),
        "plm_gen": b(lambda r: "esm" in str(r).lower()),
        "struct": b(lambda r: str(r).startswith("STR_")),
        "overall": b(lambda r: True),
    }


def classify_target(scores, lb_row, comps_target):
    simple = scores["simple"]["cv"] if scores["simple"] else 0
    bio = scores["bio"]["cv"] if scores["bio"] else 0
    best = scores["overall"]["cv"] if scores["overall"] else 0
    plm = max(
        scores["plm_ab"]["cv"] if scores["plm_ab"] else 0,
        scores["plm_gen"]["cv"] if scores["plm_gen"] else 0,
    )
    pub_priv = lb_row["public_to_private"] if lb_row is not None else np.nan
    headroom_plm = plm - bio
    headroom_best = best - simple

    flags = []
    if abs(best - simple) < 0.03:
        flags.append("TOO_EASY_DESCRIPTOR_RISK")
    if abs(plm - bio) < 0.03 and bio > 0.25:
        flags.append("GERMLINE_SHORTCUT_RISK")
    if not np.isfinite(pub_priv) or pub_priv < 0.3:
        flags.append("DATASET_TOO_SMALL_OR_NOISY")

    if "GERMLINE_SHORTCUT_RISK" in flags and headroom_plm < 0.05:
        label = "GOOD_BUT_GERMLINE_SHORTCUT_RISK"
    elif "TOO_EASY_DESCRIPTOR_RISK" in flags:
        label = "GOOD_BUT_DESCRIPTOR_DRIVEN"
    elif best < 0.15 or ("DATASET_TOO_SMALL_OR_NOISY" in flags and best < 0.25):
        label = "SCIENTIFICALLY_INTERESTING_BUT_TOO_NOISY"
    elif headroom_best > 0.08 and np.isfinite(pub_priv) and pub_priv > 0.4:
        label = "STRONG_COMPETITION_CANDIDATE"
    elif best >= 0.25 and headroom_best > 0.03:
        label = "STRONG_COMPETITION_CANDIDATE"
    else:
        label = "TOO_SMALL_OR_UNSTABLE"
    return label, flags, {
        "simple": simple,
        "bio": bio,
        "plm": plm,
        "best": best,
        "headroom_plm_vs_bio": headroom_plm,
        "headroom_best_vs_simple": headroom_best,
        "pub_priv_rank": pub_priv,
    }


def main():
    ensure_dirs()
    frames = []
    for p in [
        METRICS / "stage_ABC_results.csv",
        METRICS / "stage_PLM_STR_results.csv",
        METRICS / "stage_fusion_xgb_results.csv",
    ]:
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(METRICS / "all_results.csv", index=False)

    # run freeze helpers
    import importlib.util
    spec = importlib.util.spec_from_file_location("freeze", Path(__file__).resolve().parent / "11_freeze_eval.py")
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    fr.main()

    lb = pd.read_csv(METRICS / "leaderboard_reliability.csv") if (METRICS / "leaderboard_reliability.csv").exists() else None
    comps = pd.read_csv(METRICS / "paired_comparisons.csv") if (METRICS / "paired_comparisons.csv").exists() else None

    per_target = {}
    for t in TARGET_COLS:
        scores = best_scores(df, t)
        # shadow stability
        shadow_best = []
        for split in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
            s = best_scores(df, t, split)
            shadow_best.append(s["overall"]["cv"] if s["overall"] else np.nan)
        lb_can = None
        if lb is not None:
            rows = lb[(lb.target == t) & (lb.split == "canonical")]
            lb_can = rows.iloc[0].to_dict() if len(rows) else None
        label, flags, summary = classify_target(scores, lb_can, None)
        per_target[t] = {
            "scores": scores,
            "shadow_best_cv": shadow_best,
            "shadow_cv_std": float(np.nanstd(shadow_best)),
            "classification": label,
            "flags": flags,
            "summary": summary,
            "lb_canonical": lb_can,
        }

    # Rank targets for competition
    # Prefer: headroom, stability, scientific meaning, low shortcut
    def rank_key(t):
        p = per_target[t]
        s = p["summary"]
        score = 0.0
        score += 2.0 * s["headroom_best_vs_simple"]
        score += 1.5 * s["headroom_plm_vs_bio"]
        score += 1.0 * (s["pub_priv_rank"] if np.isfinite(s["pub_priv_rank"]) else 0)
        score += 0.5 * s["best"]
        score -= 1.5 * p["shadow_cv_std"]
        if "GERMLINE_SHORTCUT_RISK" in p["flags"]:
            score -= 0.4
        if "TOO_EASY_DESCRIPTOR_RISK" in p["flags"]:
            score -= 0.5
        if p["classification"] == "STRONG_COMPETITION_CANDIDATE":
            score += 0.3
        # scientific preference: TmApp/HIC over PSR slightly if tied
        if t == "HIC":
            score += 0.05
        if t == "TmApp":
            score += 0.02
        return score

    ranking = sorted(TARGET_COLS.keys(), key=rank_key, reverse=True)

    # Overall recommendation
    top = ranking[0]
    top_c = per_target[top]["classification"]
    if top_c == "STRONG_COMPETITION_CANDIDATE":
        overall = "ADVANCE_TO_COMPETITION_DESIGN"
    elif top_c == "GOOD_BUT_GERMLINE_SHORTCUT_RISK":
        overall = "ADVANCE_WITH_FEATURE_RESTRICTIONS"
    elif top_c == "GOOD_BUT_DESCRIPTOR_DRIVEN":
        overall = "ADVANCE_WITH_FEATURE_RESTRICTIONS"
    elif top_c in ("TOO_SMALL_OR_UNSTABLE", "SCIENTIFICALLY_INTERESTING_BUT_TOO_NOISY"):
        overall = "KEEP_AS_BACKUP"
    else:
        overall = "REJECT"

    # PLM comparison table
    plm_lines = []
    for t in TARGET_COLS:
        sub = df[(df.target == t) & (df.split == "canonical") & df.representation.str.startswith("PLM_")]
        if len(sub) == 0:
            plm_lines.append(f"- {t}: PLM results not yet available")
            continue
        best_by = sub.sort_values("cv_spearman", ascending=False).groupby("representation").first()
        plm_lines.append(f"- {t}:")
        for rep, row in best_by.sort_values("cv_spearman", ascending=False).iterrows():
            plm_lines.append(
                f"  - {rep} / {row.model}: CV ρ={row.cv_spearman:.3f}, Pub={row.public_spearman:.3f}, Priv={row.private_spearman:.3f}"
            )

    lines = [
        "# GATE B1 FINAL — Shehata developability bake-off",
        "",
        "## Verdict",
        "",
        f"**Best competition candidate: {ranking[0]}**",
        "",
        f"Ranking: {' > '.join(ranking)}",
        "",
        f"Overall recommendation: **{overall}**",
        "",
        "## Per-target classification",
        "",
    ]
    for t in ranking:
        p = per_target[t]
        lines.append(f"### {t} — `{p['classification']}`")
        lines.append("")
        s = p["summary"]
        lines.append(
            f"- Best CV ρ={s['best']:.3f}; simple={s['simple']:.3f}; BIO_SHORTCUT={s['bio']:.3f}; PLM={s['plm']:.3f}"
        )
        lines.append(
            f"- Headroom best−simple={s['headroom_best_vs_simple']:.3f}; PLM−bio={s['headroom_plm_vs_bio']:.3f}"
        )
        lines.append(f"- Shadow best CV: {p['shadow_best_cv']} (std={p['shadow_cv_std']:.3f})")
        lines.append(f"- Flags: {p['flags'] or 'none'}")
        if p["scores"]["overall"]:
            o = p["scores"]["overall"]
            lines.append(
                f"- Best pipeline: {o['rep']} / {o['model']} (CV={o['cv']:.3f}, Pub={o['pub']}, Priv={o['priv']})"
            )
        lines.append("")

    lines += [
        "## Required answers",
        "",
        "### 1. Best competition candidate",
        "",
        f"{ranking[0]} ranked #1; full order: {', '.join(ranking)}.",
        "",
        "### 2. Biological shortcut strength",
        "",
    ]
    for t in TARGET_COLS:
        bio = per_target[t]["scores"]["bio"]
        lines.append(
            f"- {t}: BIO_SHORTCUT CV ρ={bio['cv'] if bio else 'NA'} "
            f"(families + germline distance + CDR lengths; ANARCI allele assignment)."
        )
    lines += [
        "",
        "### 3. Too easy with simple descriptors?",
        "",
    ]
    for t in TARGET_COLS:
        lines.append(
            f"- {t}: {'YES risk' if 'TOO_EASY_DESCRIPTOR_RISK' in per_target[t]['flags'] else 'No — meaningful gap or modest absolute performance'} "
            f"(simple={per_target[t]['summary']['simple']:.3f}, best={per_target[t]['summary']['best']:.3f})"
        )
    lines += [
        "",
        "### 4–6. PLMs",
        "",
        *plm_lines,
        "",
        "Antibody-specific vs generic and CDR pooling: see `plm_results.md` once embeddings complete.",
        "",
        "### 7–11. Structure / RASA / fusion",
        "",
        "See `structure_prediction_audit.md`, `sasa_rasa_feature_audit.md`, and paired comparisons in `paired_comparisons.csv`.",
        "",
        "### 12. Nonlinear vs linear",
        "",
        "XGBoost/SVR evaluated on finalists when available; see comparison I in paired_comparisons.",
        "",
        "### 13. Stability across canonical + 3 shadows",
        "",
    ]
    for t in TARGET_COLS:
        lines.append(f"- {t}: shadow best CVs {per_target[t]['shadow_best_cv']}, std={per_target[t]['shadow_cv_std']:.3f}")
    lines += [
        "",
        "### 14. Public leaderboard trustworthiness",
        "",
    ]
    if lb is not None:
        for t in TARGET_COLS:
            row = lb[(lb.target == t) & (lb.split == "canonical")]
            if len(row):
                r = row.iloc[0]
                lines.append(
                    f"- {t}: CV→Pub rank-ρ={r['cv_to_public']:.3f}; Pub→Priv={r['public_to_private']:.3f}; CV→Priv={r['cv_to_private']:.3f}"
                )
    lines += [
        "",
        "### 15. Difference from C3a",
        "",
        "C3a is a **one-parent local mutation landscape** (high sequence relatedness, mutational deltas).",
        "Shehata is a **panel of many independent antibodies** across B-cell subsets with germline diversity,",
        "paired VH/VL, and continuous developability readouts (PSR/HIC/TmApp). Competition learning signal",
        "is global sequence→property rather than local fitness around one parent — pedagogically and",
        "methodologically distinct, but N≈324 requires careful cluster-safe splits and shortcut audits.",
        "",
        "## Decision philosophy application",
        "",
        "We do **not** pick solely by max Private Spearman. Preference weights scientific meaning,",
        "CV→Public→Private ranking stability, nontrivial simple baseline, PLM/structure headroom,",
        "and low germline-shortcut dominance.",
        "",
        f"Selected: **{ranking[0]}** with recommendation **{overall}**.",
        "",
    ]
    (REPORTS / "GATE_B1_FINAL.md").write_text("\n".join(lines) + "\n")
    write_json(METRICS / "final_summary.json", {"ranking": ranking, "overall": overall, "per_target": per_target})
    print("FINAL_OK", ranking, overall)


if __name__ == "__main__":
    main()
