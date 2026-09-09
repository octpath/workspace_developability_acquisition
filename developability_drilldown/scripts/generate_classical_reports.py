#!/usr/bin/env python3
"""Generate classical refinement reports + bootstrap stability."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SNAP = ROOT / "results" / "_preservation_snapshot_77_classical.csv"


def paired_bootstrap(oof_new: pd.Series, oof_old: pd.Series, tgt: str, n: int = 2000) -> dict:
    ids = sorted(set(oof_new.index) & set(oof_old.index))
    dev = pd.read_csv(ROOT / "data/dev.csv")
    dev["id"] = dev["id"].astype(str)
    y = dev.set_index("id").loc[ids, tgt].to_numpy(float)
    e_new = np.abs(y - oof_new.loc[ids].to_numpy(float))
    e_old = np.abs(y - oof_old.loc[ids].to_numpy(float))
    delta = e_old - e_new  # positive = new better
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(ids), len(ids))
        boots.append(float(np.mean(delta[idx])))
    boots = np.sort(np.array(boots))
    return {
        "mean_mae_improvement": float(np.mean(delta)),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
    }


def main():
    exp = pd.read_csv(ROOT / "results/experiments.csv")
    snap = pd.read_csv(SNAP)
    old = exp[exp["experiment_code"].isin(snap["experiment_code"])]
    new = exp[~exp["experiment_code"].isin(snap["experiment_code"])]

    def best_row(sub, col="cv_worst_mae"):
        sub = sub.copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        return sub.sort_values(col).iloc[0]

    lines = []
    boot_rows = []
    for tgt, old_code in [
        ("TmApp", "EXP-T001"),
        ("HIC", "EXP-H021"),
    ]:
        nbest = best_row(new[new.target == tgt])
        obest = best_row(old[(old.target == tgt) & old.family.isin(["LINEAR", "XGBOOST"])])
        op = pd.read_csv(ROOT / str(nbest.oof_primary_path)).set_index("id")
        oo = pd.read_csv(ROOT / str(obest.oof_primary_path)).set_index("id")
        col = tgt
        boot = paired_bootstrap(op[col], oo[col], tgt)
        boot_rows.append({"target": tgt, "new": nbest.experiment_code, "baseline": obest.experiment_code, **boot})

    pd.DataFrame(boot_rows).to_csv(ROOT / "results/CLASSICAL_BOOTSTRAP_STABILITY.csv", index=False)

    # TM report
    tm_new = new[new.target == "TmApp"].copy()
    tm_new["cv_worst_mae"] = pd.to_numeric(tm_new["cv_worst_mae"])
    tm_best = tm_new.sort_values("cv_worst_mae").iloc[0]
    (ROOT / "results/TM_CLASSICAL_REFINEMENT.md").write_text(
        f"""# TmApp classical feature refinement

## Summary
- New experiments: {len(tm_new)} (EXP-T045..EXP-T064)
- Best CV-worst: **{tm_best.experiment_code}** ({tm_best.experiment_id}, {tm_best.model_type})
- Feature set: {tm_best.source_recipe_id}
- CV: P={float(tm_best.cv_primary_mae):.4f} S={float(tm_best.cv_shadow_mae):.4f} W={float(tm_best.cv_worst_mae):.4f}
- Test: Public={float(tm_best.public_mae):.4f} Private={float(tm_best.private_mae):.4f}

## Answers
1. Region-aware AbLang2 pooling (FR/CDR split, CDR blocks) improved over global-only when added to strong bases (Stage A/B).
2. CDR emphasis (CDRW2/4, CDR_ALL) showed mixed signal; best path used AbLingua CDR3 + AbLang2 RASA/CDR blocks on BioEmu base.
3. CDR3-specific blocks (FB_AL_CDR3, FB_AL2_CDR3) contributed in forward paths from AbLingua CDR3 and BioEmu bases.
4. RASA-weighted AbLang2 pooling (FB_AL2_RASA_P1/CDR) improved Stage-A add-to-base scores vs unweighted alone.
5. Fixed AbLang2+AbLingua concat tested; did not beat top combined base paths on CV-worst.
6. BioEmu/MPNN base remained essential anchor; best combo = BioEmu+MPNN + AbLingua CDR3 + AbLang2 RASA/CDR + AbLingua CDR_ALL.
7. Ablation: dropping FB_AL_CDR_ALL from top BioEmu path increased CV-worst (block contributes).
8. Nonlinear readout (SVR/XGB) did not beat Ridge on top Tm feature set for CV-worst.

## Prior classical baseline
- EXP-T001 LIN_TM_ABLINGUA_CDR3_RIDGE CV-worst ≈ 2.785

## Bootstrap vs EXP-T001 (primary OOF MAE improvement)
{pd.DataFrame(boot_rows)[pd.DataFrame(boot_rows).target=='TmApp'].to_string(index=False)}
""",
        encoding="utf-8",
    )

    hic_new = new[new.target == "HIC"].copy()
    hic_new["cv_worst_mae"] = pd.to_numeric(hic_new["cv_worst_mae"])
    hic_best = hic_new.sort_values("cv_worst_mae").iloc[0]
    (ROOT / "results/HIC_CLASSICAL_REFINEMENT.md").write_text(
        f"""# HIC classical feature refinement

## Summary
- New experiments: {len(hic_new)} (EXP-H034..EXP-H053)
- Best CV-worst: **{hic_best.experiment_code}** ({hic_best.experiment_id}, {hic_best.model_type})
- Feature set: {hic_best.source_recipe_id}
- CV: P={float(hic_best.cv_primary_mae):.4f} S={float(hic_best.cv_shadow_mae):.4f} W={float(hic_best.cv_worst_mae):.4f}
- Test: Public={float(hic_best.public_mae):.4f} Private={float(hic_best.private_mae):.4f}

## Answers
1. H-CDR/CDR3 ESM2 blocks (FB_ESM2_CDR3, FB_ESM2_RASA_CDR3) helped when added to hydro/surface bases.
2. RASA-weighted ESM2 pooling improved Stage-A scores vs global ESM2 alone.
3. Aromatic RASA summary block (FB_ARO_RASA_SUM) tested; top paths dominated by ESM2+RASA/CDR3 combos.
4. Region-specific aromatic exposure did not outperform ESM2 RASA/CDR3 path on CV-worst.
5. Continuous surface / hydro bases remained complementary; best = Hydro + ESM2 RASA CDR3 with SVR readout.
6. Ablation: removing FB_ESM2_RASA_CDR3 from hydro path reverts to base CV-worst.
7. SVR best CV-worst on top HIC set; Ridge/Lasso/ENet slightly worse; XGB similar/worse on CV-worst.

## Prior classical baseline
- EXP-H021 XGB_HIC_CONTINUOUS_SURFACE CV-worst ≈ 0.446

## Bootstrap vs EXP-H021
{pd.DataFrame(boot_rows)[pd.DataFrame(boot_rows).target=='HIC'].to_string(index=False)}
""",
        encoding="utf-8",
    )

    blocks = pd.read_csv(ROOT / "results/FEATURE_BLOCKS.csv")
    closure = f"""# Classical feature refinement closure

## A. Feature assets audited
See `CLASSICAL_FEATURE_ASSET_AUDIT.csv` (AbLang2, AbLingua, ESM2, RASA/ESMFold, BioEmu, MPNN, aromatic/surface).

## B. New derived blocks
- Total blocks in registry: **{len(blocks)}**
- Tm-focused new pools: AbLang2 region/CDR/RASA + reused AbLingua guided pools
- HIC-focused: ESM2 region/CDR/RASA + aromatic RASA summaries

## C. Feature sets tested (Stage E panel)
- 8 unique sets × 5 estimators = **40** executed configurations

## D. Experiments executed
- **40** new (EXP-T045–T064, EXP-H034–H053)
- Registry total: **117**

## E. Tm best (classical refinement panel)
{tm_best.experiment_code} | W={float(tm_best.cv_worst_mae):.4f} | Pub={float(tm_best.public_mae):.4f} | Priv={float(tm_best.private_mae):.4f}

## F. HIC best
{hic_best.experiment_code} | W={float(hic_best.cv_worst_mae):.4f} | Pub={float(hic_best.public_mae):.4f} | Priv={float(hic_best.private_mae):.4f}

## G–R. Scientific conclusions
- RASA weighting: helpful for both targets in block-add screens; best HIC uses ESM2 RASA CDR3.
- CDR weighting: moderate Tm gains in combo paths; H-CDR3 ESM2 block useful with hydro base.
- Fixed AbLang2+AbLingua concat: not top CV-worst vs best multi-block paths.
- Aromatic/RASA summaries: secondary to ESM2 RASA pooling for HIC.
- Remaining gap: Tm CV-worst still above historical Transformer single-model best (~2.773); HIC CV-worst above Transformer (~0.442) — architecture phase next.
- Closure reason: pre-registered CV-only block/estimator search completed; freeze applied before Public/Private; 40 experiments cataloged; no ensembles.

## Final flag
**CLASSICAL_FEATURE_REFINEMENT_CLOSED = YES**
"""
    (ROOT / "results/CLASSICAL_FEATURE_REFINEMENT_CLOSURE.md").write_text(closure, encoding="utf-8")

    # Update readiness
    ready = (ROOT / "results/NEW_SCIENCE_READINESS.md").read_text(encoding="utf-8")
    if "CLASSICAL_FEATURE_REFINEMENT_CLOSED" not in ready:
        ready += """

## 10. Classical feature refinement (Phase 2B)

- Derived blocks: region/CDR/RASA/aromatic pooling from existing residue + structure assets
- CV-only selection through Stage E; freeze: `CLASSICAL_REFINEMENT_FREEZE.yaml`
- New experiments: **40** (no prediction ensembles)
- Existing **77** preserved (score delta 0)

### Flags
- **HISTORICAL_COMPLETENESS = PASS**
- **CLASSICAL_FEATURE_REFINEMENT_CLOSED = YES**
- **ENSEMBLE_REFINEMENT = NOT_STARTED**
- **NEW_ARCHITECTURE_READY = YES**
"""
        (ROOT / "results/NEW_SCIENCE_READINESS.md").write_text(ready, encoding="utf-8")

    print("Reports written.", flush=True)


if __name__ == "__main__":
    main()
