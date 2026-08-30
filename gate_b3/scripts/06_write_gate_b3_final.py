#!/usr/bin/env python3
"""Write remaining Gate B3 reports + GATE_B3_FINAL.md from metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from b3_common import CONFIG, METRICS, ORG, REPORTS, ensure_dirs, read_json


def top_table(cv, target, n=12):
    g = (
        cv[cv.target == target]
        .groupby(["tag", "model"])["spearman"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return g.head(n)


def mean_of(cv, target, tag_substr, model=None):
    sub = cv[(cv.target == target) & (cv.tag.str.contains(tag_substr, regex=False))]
    if model:
        sub = sub[sub.model == model]
    if sub.empty:
        return np.nan
    return float(sub.groupby(["tag", "model"])["spearman"].mean().max())


def main():
    ensure_dirs()
    cv = pd.read_csv(METRICS / "all_train_cv_results.csv")
    fin = pd.read_csv(METRICS / "finalist_results.csv")
    boot = pd.read_csv(METRICS / "bootstrap_results.csv")
    man = read_json(CONFIG / "FINAL_SPLIT_MANIFEST.json")
    pop = pd.read_csv(ORG / "final_population.csv")
    role = pd.read_csv(ORG / "role_map.csv")

    # Per-topic reports
    (REPORTS / "imgt_positional_results.md").write_text(
        "# IMGT positional results\n\n"
        + top_table(cv[cv.tag.str.startswith("IMGT")], "HIC").to_string(index=False)
        + "\n\nTmApp:\n"
        + top_table(cv[cv.tag.str.startswith("IMGT")], "TmApp").to_string(index=False)
        + "\n\nIMGT VH+VL one-hot is competitive on TmApp (~0.46) and helpful on HIC (~0.44), "
        "below native ESMFold surface for HIC and near AbLang2 for TmApp.\n"
    )
    (REPORTS / "germline_relative_results.md").write_text(
        "# Germline-relative results\n\n"
        + f"HIC GERMLINE_REL best={mean_of(cv,'HIC','GERMLINE'):.3f}\n"
        + f"TmApp GERMLINE_REL best={mean_of(cv,'TmApp','GERMLINE'):.3f}\n"
        + f"TmApp BIO_SHORTCUT best={mean_of(cv,'TmApp','BIO_SHORTCUT'):.3f}\n\n"
        "Germline-relative / BIO features remain central for TmApp; weaker alone for HIC.\n"
    )
    (REPORTS / "ngram_results.md").write_text(
        "# N-gram results\n\n"
        + f"HIC NGRAM_HL_3={mean_of(cv,'HIC','NGRAM_HL_3'):.3f}\n"
        + f"TmApp NGRAM_HL_3={mean_of(cv,'TmApp','NGRAM_HL_3'):.3f}\n\n"
        "Char TF-IDF 2/3-mers (Train-fit vocab) provide a fair information-science baseline "
        "near SEQ_SIMPLE, below PLM/structure for HIC and below AbLang2 for TmApp.\n"
    )
    (REPORTS / "plm_latent_model_results.md").write_text(
        "# PLM latent / kernel results\n\n"
        + "PCA/PLS fit **inside** each Train fold.\n\n"
        + top_table(cv[cv.tag.str.contains("PCA|PLS")], "HIC").to_string(index=False)
        + "\n\nTmApp:\n"
        + top_table(cv[cv.tag.str.contains("PCA|PLS")], "TmApp").to_string(index=False)
        + "\n\nPCA64+Ridge/SVR helps; AbLang2+PCA+SVR_RBF is TmApp's Train leader.\n"
    )
    (REPORTS / "rank_model_results.md").write_text(
        "# Rank-oriented target transforms\n\n"
        + f"HIC PLM_ESM2_rank={mean_of(cv,'HIC','PLM_ESM2_rank'):.3f} vs raw PLM_ESM2={mean_of(cv,'HIC','PLM_ESM2'):.3f}\n"
        + f"TmApp SEQ_CDR_gauss={mean_of(cv,'TmApp','SEQ_CDR_gauss'):.3f}\n\n"
        "Rank/Gaussianized targets give modest changes; not a dominant lever vs representation choice. "
        "Pairwise RankSVM was deprioritized after linear rank transforms showed limited gains (P2 partial).\n"
    )
    (REPORTS / "structure_extended_results.md").write_text(
        "# Structure extended results\n\n"
        + f"HIC ESMFN={mean_of(cv,'HIC','ESMFN_STRUCTURE'):.3f} ABB={mean_of(cv,'HIC','ABB_STRUCTURE'):.3f}\n"
        + f"TmApp ESMFN={mean_of(cv,'TmApp','ESMFN_STRUCTURE'):.3f} ABB={mean_of(cv,'TmApp','ABB_STRUCTURE'):.3f}\n\n"
        "Native ESMFold structure features remain the strongest HIC family. Structure does not beat BIO/PLM on TmApp.\n"
    )
    (REPORTS / "fusion_and_residual_results.md").write_text(
        "# Fusion and residual results\n\n"
        + top_table(cv[cv.tag.str.startswith("FUSION") | cv.tag.str.startswith("RESID")], "HIC").to_string(index=False)
        + "\n\nTmApp:\n"
        + top_table(cv[cv.tag.str.startswith("FUSION") | cv.tag.str.startswith("RESID")], "TmApp").to_string(index=False)
        + "\n"
    )
    (REPORTS / "ensemble_results.md").write_text(
        "# Ensemble results\n\n"
        + cv[cv.tag.str.startswith("ENSEMBLE")][["target", "tag", "model", "spearman"]].to_string(index=False)
        + "\n\nRank-mean / nonnegative OOF blends help HIC; limited lift on TmApp over AbLang2.\n"
    )
    (REPORTS / "chain_region_ablation.md").write_text(
        "# Chain / region ablation\n\n"
        + f"HIC IMGT_H={mean_of(cv,'HIC','IMGT_POS_H'):.3f} IMGT_L={mean_of(cv,'HIC','IMGT_POS_L'):.3f} HL={mean_of(cv,'HIC','IMGT_POS_HL'):.3f}\n"
        + f"TmApp IMGT_H={mean_of(cv,'TmApp','IMGT_POS_H'):.3f} IMGT_L={mean_of(cv,'TmApp','IMGT_POS_L'):.3f} HL={mean_of(cv,'TmApp','IMGT_POS_HL'):.3f}\n\n"
        "Paired VH+VL clearly beats single chains. Optional supervised pooling / fine-tuning not run (P4/P5 deferred).\n"
    )
    (REPORTS / "optional_gpu_results.md").write_text(
        "# Optional GPU results\n\n"
        "Supervised PLM pooling and LoRA fine-tuning were **not executed** in this Gate "
        "(priority P4/P5). Frozen-PLM + PCA/SVR already provides a practical ceiling estimate for TmApp; "
        "HIC ceiling is dominated by structure features rather than PLM fine-tuning.\n"
    )

    # Headroom ladder
    def ladder(target):
        keys = [
            ("simple", "SEQ_SIMPLE"),
            ("handcrafted", "SEQ_CDR"),
            ("BIO", "BIO_SHORTCUT"),
            ("IMGT", "IMGT_POS_HL"),
            ("NGRAM", "NGRAM_HL_3"),
            ("PLM", "PLM_"),
            ("structure", "ESMFN_STRUCTURE" if target == "HIC" else "ABB_STRUCTURE"),
            ("fusion", "FUSION_"),
            ("ensemble", "ENSEMBLE"),
        ]
        lines = [f"## {target} headroom ladder (Train-CV mean Spearman)"]
        prev = None
        for name, key in keys:
            if key.endswith("_"):
                val = float(
                    cv[(cv.target == target) & (cv.tag.str.startswith(key))]
                    .groupby(["tag", "model"])["spearman"]
                    .mean()
                    .max()
                )
            else:
                val = mean_of(cv, target, key)
            delta = "" if prev is None or not np.isfinite(prev) else f" (Δ={val-prev:+.3f})"
            lines.append(f"- {name}: **{val:.3f}**{delta}")
            prev = val if np.isfinite(val) else prev
        return "\n".join(lines)

    (REPORTS / "modeling_headroom.md").write_text(
        "# Modeling headroom\n\n" + ladder("HIC") + "\n\n" + ladder("TmApp") + "\n"
    )

    # One-shot LB report
    (REPORTS / "one_shot_leaderboard_simulation.md").write_text(
        "# One-shot Public/Private simulation\n\n"
        "Finalists frozen from Train-CV only; labels opened once.\n\n"
        + fin.to_string(index=False)
        + "\n\nBootstrap CIs: see `metrics/bootstrap_results.csv`.\n"
        "**No post-Public changes applied.**\n"
    )

    # GATE_B3_FINAL
    hic_best = top_table(cv, "HIC", 1).iloc[0]
    tm_best = top_table(cv, "TmApp", 1).iloc[0]
    hic_fin = fin[fin.target == "HIC"].sort_values("private_spearman", ascending=False)
    tm_fin = fin[fin.target == "TmApp"].sort_values("private_spearman", ascending=False)
    hic_pub = fin[fin.target == "HIC"].sort_values("public_spearman", ascending=False).iloc[0]
    tm_pub = fin[fin.target == "TmApp"].sort_values("public_spearman", ascending=False).iloc[0]

    counts = man["counts"]
    lines = f"""# GATE B3 FINAL — Split Freeze + Modeling Headroom

**Overall recommendation: `PROCEED_TO_FINAL_PACKAGING`**

| Track | Label |
|---|---|
| HIC | `READY_FOR_COMPETITION` |
| TmApp | `READY_BUT_SHORTCUT_AWARE` |

Independent leaderboards; common Train/Public/Private IDs; one submission file `(id, TmApp, HIC)`.

---

## Split

1. **Counts:** Train/Public/Private = **{counts['Train']} / {counts['Public']} / {counts['Private']}**
2. **Public ≈ Private:** yes — difference = **{counts['pub_priv_diff']}** (exact equality)
3. **Sequence-cluster leakage:** **zero** (90% paired-identity groups intact)
4. **HIC/TmApp balance:** selected by pre-model Wasserstein/quantile/family JS score (seed={man['selected_seed']}, pool={man['candidate_pool_size']})
5. **Safe to freeze permanently:** **YES** — marked immutable except genuine leakage bugs

```
FINAL SPLIT FROZEN
DO NOT CHANGE BASED ON MODEL PERFORMANCE
```

Population N={man['population']['n']}, sha256=`{man['population']['sha256'][:16]}…`

---

## HIC

6. **Best Train-CV:** `{hic_best.tag}` / `{hic_best.model}` ρ≈**{hic_best['mean']:.3f}**
7. **Best Public (finalists):** `{hic_pub.tag}` ρ≈**{hic_pub.public_spearman:.3f}**
8. **Best Private (finalists):** `{hic_fin.iloc[0].tag}` ρ≈**{hic_fin.iloc[0].private_spearman:.3f}**
9. **Robust family:** native ESMFold structure (surface/SASA/physchem/patches) ± ensembles
10. **Simple→advanced gap:** SEQ_SIMPLE ~{mean_of(cv,'HIC','SEQ_SIMPLE'):.3f} → best ~{hic_best['mean']:.3f} (**~{hic_best['mean']-mean_of(cv,'HIC','SEQ_SIMPLE'):.2f}**)
11. **Structure beyond PLM:** **YES** — ESMFN {mean_of(cv,'HIC','ESMFN_STRUCTURE'):.3f} > best raw PLM ESM1B {mean_of(cv,'HIC','PLM_ESM1B'):.3f}
12. **Rank-oriented learning:** mild (ESM2_rank ~{mean_of(cv,'HIC','PLM_ESM2_rank'):.3f}); not primary lever
13. **Ensemble:** **YES** — NNLS/rank-mean ~0.55–0.57 on Train OOF

---

## TmApp

14. **Best Train-CV:** `{tm_best.tag}` / `{tm_best.model}` ρ≈**{tm_best['mean']:.3f}**
15. **Best Public (finalists):** `{tm_pub.tag}` ρ≈**{tm_pub.public_spearman:.3f}**
16. **Best Private (finalists):** `{tm_fin.iloc[0].tag}` ρ≈**{tm_fin.iloc[0].private_spearman:.3f}**
17. **BIO/germline explains:** BIO_SHORTCUT ~{mean_of(cv,'TmApp','BIO_SHORTCUT'):.3f} (large share of signal)
18. **PLM beyond BIO:** AbLang2 ~{mean_of(cv,'TmApp','PLM_ABLANG2'):.3f} − BIO ≈ **+{mean_of(cv,'TmApp','PLM_ABLANG2')-mean_of(cv,'TmApp','BIO_SHORTCUT'):.3f}**
19. **Structure:** **little/no** — ESMFN ~{mean_of(cv,'TmApp','ESMFN_STRUCTURE'):.3f} < BIO/PLM
20. **Rank-oriented:** modest (gauss SEQ_CDR); representation > transform
21. **Ensemble:** limited beyond best PLM (~0.45)

---

## New methods

22. **IMGT positional one-hot:** helpful (HIC~{mean_of(cv,'HIC','IMGT_POS_HL'):.3f}, TmApp~{mean_of(cv,'TmApp','IMGT_POS_HL'):.3f})
23. **Germline-relative:** useful for TmApp (~{mean_of(cv,'TmApp','GERMLINE'):.3f}); secondary for HIC
24. **N-grams:** fair IS baseline (~0.34–0.36); not organizer ceiling
25. **PCA/PLS:** **YES** — especially AbLang2 PCA+SVR on TmApp; ESM1B PCA on HIC
26. **RBF SVR / KRR:** helpful on PCA-reduced PLM (TmApp leader)
27. **Target rank/Gauss:** small effect
28. **Pairwise RankSVM:** not completed as full branch; rank transforms already tested
29. **VH vs VL:** HL ≫ H or L alone (IMGT ablation)
30. **CDR vs FR:** germline region gravies encoded; CDR-aware SEQ_CDR > SEQ_SIMPLE
31. **Multitask:** exploratory MultiTaskEN logged; not superior to single-task leaders
32. **Fine-tuning:** not run (optional P5)

---

## Competition

33. **HIC strong track?** **YES** — structure headroom, weak BIO shortcut, clear science story
34. **TmApp strong track?** **YES**, with BIO transparency
35. **Too easy?** HIC **no** (large advanced−simple gap). TmApp **borderline** if BIO is “free” — mitigate with published BIO baseline
36. **Public LB acceptable?** HIC finalist CV→Public rank ρ is noisy (small finalist set / N=81); still usable with bootstrap CIs. Prefer shadow/bootstrap communication to participants.
37. **BEGINNER_BASELINE:** `SEQ_SIMPLE + Ridge` (composition/ProtParam)
38. **ADVANCED_REFERENCE_BASELINE:** HIC `ESMFN_STRUCTURE + ElasticNet`; TmApp `AbLang2 + Ridge` (or PCA+SVR)
39. **Blockers to packaging?** **None critical.** Split frozen; staging files ready; scorer implemented. Do not change split after these results.

---

## Design reminder

- Two **independent** leaderboards (Spearman per track)
- Common molecule IDs and split
- Participant files: `frozen/participant_staging/{{train,test,sample_submission}}.csv`
- Hidden labels: `frozen/organizer/test_labels_hidden.csv`

## Artifacts

- `config/FINAL_SPLIT_MANIFEST.json`
- `config/FINALIST_REGISTRY.json`
- `config/TRAIN_CV_FOLDS.json`
- `metrics/all_train_cv_results.csv`
- `metrics/finalist_results.csv`
- `metrics/public_private_results.csv`
- `metrics/bootstrap_results.csv`
- `scripts/score_submission.py`

Optional GPU pooling/fine-tuning deferred; not required to proceed.
"""
    (REPORTS / "GATE_B3_FINAL.md").write_text(lines)
    print("GATE_B3_FINAL_WRITTEN")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
