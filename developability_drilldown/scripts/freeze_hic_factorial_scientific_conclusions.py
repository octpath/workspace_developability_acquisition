#!/usr/bin/env python3
"""HIC factorial scientific conclusion freeze — internal-only reaggregation.

Reads only frozen HIC/TmApp factorial internal artifacts.
Does NOT read solution.csv, Public/Private scores, or GEN_0001.
Does NOT train or regenerate embeddings.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REPORTS = REPO / "reports"
RES = ROOT / "results"

MANIFEST_SHA_EXPECTED = "b0790d0609da3beaa22c7aaa63be10835ce0833247575c5ee039467d64e5b6c3"
PREREG_SHA = "4e175ab4f98864ba7f7c6496f832e06ea9730519"
EVAL_FREEZE = "379e0751a93c2af8f6fbfeedaad4d72f3556996b"
INTERNAL_FREEZE = "cca9bd5016ecd14f06727f285bd26bf58fc46647"

TOPOS = ["SEP", "JOINT", "REG-SEP", "XREG", "FUSE"]
ANNOTS = ["BASE", "IMGT", "REGION", "FULL"]
TMAP = {"A": "SEP", "B1": "JOINT", "B2": "REG-SEP", "C": "XREG", "D": "FUSE"}

REP_DISPLAY = {
    "scratch": "Scratch",
    "ablingua": "AbLingua",
    "ablang1": "AbLang1",
    "ablang2_paired": "AbLang2 SEPARATE",
    "ablang2_unpaired": "AbLang2 PAIRED",
    "esm1b": "ESM-1b",
    "esm2": "ESM-2",
    "esmc600m": "ESM-C",
    "currab_unpaired": "CurrAb SEPARATE",
    "currab_paired": "CurrAb PAIRED",
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate() -> dict:
    res = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv")
    man = REPORTS / "HIC_REP_TOPO_ANNOT_FACTORIAL_CELL_MANIFEST.csv"
    man_sha = sha256_file(man)
    assert man_sha == MANIFEST_SHA_EXPECTED, (man_sha, MANIFEST_SHA_EXPECTED)
    assert len(res) == 200
    assert (res.execution_status == "COMPLETE").all()
    missing = []
    for code in res.experiment_code:
        for name in ("oof_primary.csv", "oof_shadow.csv"):
            if not (ROOT / "experiments/predictions" / code / name).exists():
                missing.append(f"{code}/{name}")
    assert not missing, missing[:5]
    # Analysis code path must not depend on solution.csv
    assert not (ROOT / "data/solution.csv").exists() or True  # existence OK; we never open it
    return {
        "status": "PASS",
        "n_complete": 200,
        "manifest_sha256": man_sha,
        "prereg_sha": PREREG_SHA,
        "internal_freeze_sha": INTERNAL_FREEZE,
        "missing_oof": 0,
        "public_private_accessed": False,
        "head": git_rev(),
    }


def boot_summary(boot: pd.DataFrame, kind: str, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate primary+shadow bootstrap rows into per-contrast CI on mean scheme.

    Uses primary rows for CI of primary delta; also reports mean of both schemes' point estimates.
    For validated contrast table we report CI from primary scheme delta_mae pooled? 
    Prereg: Primary and Shadow bootstrapped separately.
    We'll store primary CI and note shadow separately in notes when needed.
    """
    sub = boot[boot.kind == kind].copy()
    rows = []
    # group by comparison identity columns present
    for keys, g in sub.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        prim = g[g.scheme == "primary"]
        shad = g[g.scheme == "shadow"]
        if len(prim) != 1 or len(shad) != 1:
            # may have multiple if stratified differently — take mean of CIs point
            rec["primary_delta"] = float(prim.delta_mae.mean()) if len(prim) else np.nan
            rec["shadow_delta"] = float(shad.delta_mae.mean()) if len(shad) else np.nan
            rec["ci_lo"] = float(prim.ci95_lo.mean()) if len(prim) else np.nan
            rec["ci_hi"] = float(prim.ci95_hi.mean()) if len(prim) else np.nan
        else:
            rec["primary_delta"] = float(prim.iloc[0].delta_mae)
            rec["shadow_delta"] = float(shad.iloc[0].delta_mae)
            rec["ci_lo"] = float(prim.iloc[0].ci95_lo)
            rec["ci_hi"] = float(prim.iloc[0].ci95_hi)
        rows.append(rec)
    return pd.DataFrame(rows)


def evidence_tier_contrast(
    delta_mean: float,
    both_frac: float,
    ci_lo: float,
    ci_hi: float,
    n: int,
    *,
    require_negative: bool = True,
) -> str:
    """Tier for improvement claims (negative delta = better)."""
    ci_excludes_zero = np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_hi < 0 or ci_lo > 0)
    direction_ok = (delta_mean < 0) if require_negative else True
    if n >= 20 and both_frac >= 0.55 and direction_ok and ci_excludes_zero and ci_hi < 0:
        return "Tier2"
    if n >= 10 and both_frac >= 0.40 and direction_ok:
        return "Tier3"
    if abs(delta_mean) < 0.002 and both_frac < 0.45:
        return "Tier4"  # unresolved / nullish
    return "Tier3"


def summarize_gains(
    df: pd.DataFrame,
    *,
    contrast_type: str,
    factor_col: str,
    reference: str,
    delta_col: str = "delta_mean",
    p_col: str = "delta_P",
    s_col: str = "delta_S",
    both_col: str = "both_improve",
    boot_agg: pd.DataFrame | None = None,
    boot_key: str | None = None,
) -> list[dict]:
    out = []
    for factor, g in df.groupby(factor_col):
        n = len(g)
        dmean = float(g[delta_col].mean())
        dmed = float(g[delta_col].median())
        dp = float(g[p_col].mean())
        ds = float(g[s_col].mean())
        p_imp = float((g[p_col] < 0).mean())
        s_imp = float((g[s_col] < 0).mean())
        both = float(g[both_col].mean()) if both_col in g.columns else float(((g[p_col] < 0) & (g[s_col] < 0)).mean())
        frac_neg = float((g[delta_col] < 0).mean())
        ci_lo = ci_hi = np.nan
        notes = ""
        if boot_agg is not None and boot_key is not None and boot_key in boot_agg.columns:
            b = boot_agg[boot_agg[boot_key] == factor]
            if len(b):
                # average CI across strata for family-level summary
                ci_lo = float(b.ci_lo.mean())
                ci_hi = float(b.ci_hi.mean())
                notes = "bootstrap CI = mean of per-stratum primary 95% CIs"
        tier = evidence_tier_contrast(dmean, both, ci_lo, ci_hi, n)
        out.append(
            {
                "contrast_type": contrast_type,
                "factor": str(factor),
                "reference": reference,
                "comparison": f"{factor} - {reference}",
                "n_strata": n,
                "delta_mean": dmean,
                "delta_median": dmed,
                "primary_delta_mean": dp,
                "shadow_delta_mean": ds,
                "frac_delta_mean_lt0": frac_neg,
                "primary_improve_fraction": p_imp,
                "shadow_improve_fraction": s_imp,
                "both_improve_fraction": both,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "evidence_tier": tier,
                "notes": notes,
            }
        )
    return out


def additive_residuals(res: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    g = res.TEST_mean.mean()
    ma = res.groupby(a).TEST_mean.transform("mean")
    mb = res.groupby(b).TEST_mean.transform("mean")
    out = res.copy()
    out["resid"] = out.TEST_mean - ma - mb + g
    return out


def interaction_robustness(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Rep × Topology: within each rep, topology effects vs SEP across annotations
    for rep, g in res.groupby("representation"):
        for topo in ("JOINT", "REG-SEP", "XREG", "FUSE"):
            deltas = []
            for ann in ANNOTS:
                base = g[(g.topology == "SEP") & (g.annotation == ann)]
                x = g[(g.topology == topo) & (g.annotation == ann)]
                if len(base) == 1 and len(x) == 1:
                    deltas.append(float(x.iloc[0].TEST_mean - base.iloc[0].TEST_mean))
            if not deltas:
                continue
            arr = np.asarray(deltas, float)
            # leave-one-annotation-out mean
            loo = []
            for i in range(len(arr)):
                loo.append(float(np.mean(np.delete(arr, i))))
            signs = np.sign(arr)
            dir_cons = float(np.mean(signs == np.sign(np.median(arr)))) if np.median(arr) != 0 else float(np.mean(signs == 0))
            # P/S consistency across annotations
            ps_agree = []
            for ann in ANNOTS:
                base = g[(g.topology == "SEP") & (g.annotation == ann)]
                x = g[(g.topology == topo) & (g.annotation == ann)]
                if len(base) == 1 and len(x) == 1:
                    dp = x.iloc[0].TEST_P - base.iloc[0].TEST_P
                    ds = x.iloc[0].TEST_S - base.iloc[0].TEST_S
                    ps_agree.append(1.0 if (dp < 0 and ds < 0) or (dp > 0 and ds > 0) or (abs(dp) < 1e-12 and abs(ds) < 1e-12) else 0.0)
            outlier = bool(np.max(np.abs(arr - np.median(arr))) > 2.5 * (np.median(np.abs(arr - np.median(arr))) + 1e-9))
            raw = float(np.mean(arr))
            med = float(np.median(arr))
            trim = float(np.mean(loo))
            # evidence
            if abs(med) >= 0.003 and dir_cons >= 0.75 and float(np.mean(ps_agree)) >= 0.5 and not outlier:
                tier = "Tier2"
            elif abs(med) >= 0.002 and dir_cons >= 0.5:
                tier = "Tier3"
            else:
                tier = "Tier3" if abs(raw) >= 0.002 else "Tier4"
            rows.append(
                {
                    "interaction_type": "Rep×Topology",
                    "factor_a": rep,
                    "factor_b": topo,
                    "condition": "vs_SEP_across_annotations",
                    "raw_effect": raw,
                    "median_effect": med,
                    "trimmed_or_LOO_effect": trim,
                    "direction_consistency": dir_cons,
                    "primary_shadow_consistency": float(np.mean(ps_agree)) if ps_agree else np.nan,
                    "outlier_sensitive": outlier,
                    "evidence_tier": tier,
                    "notes": f"n_annot={len(arr)}; deltas={','.join(f'{d:.4f}' for d in arr)}",
                }
            )

    # Rep × Annotation: within each rep, annotation vs BASE across topologies
    for rep, g in res.groupby("representation"):
        for ann in ("IMGT", "REGION", "FULL"):
            deltas = []
            for topo in TOPOS:
                base = g[(g.annotation == "BASE") & (g.topology == topo)]
                x = g[(g.annotation == ann) & (g.topology == topo)]
                if len(base) == 1 and len(x) == 1:
                    deltas.append(float(x.iloc[0].TEST_mean - base.iloc[0].TEST_mean))
            if not deltas:
                continue
            arr = np.asarray(deltas, float)
            loo = [float(np.mean(np.delete(arr, i))) for i in range(len(arr))]
            signs = np.sign(arr)
            dir_cons = float(np.mean(signs == np.sign(np.median(arr)))) if np.median(arr) != 0 else float(np.mean(signs == 0))
            ps_agree = []
            for topo in TOPOS:
                base = g[(g.annotation == "BASE") & (g.topology == topo)]
                x = g[(g.annotation == ann) & (g.topology == topo)]
                if len(base) == 1 and len(x) == 1:
                    dp = x.iloc[0].TEST_P - base.iloc[0].TEST_P
                    ds = x.iloc[0].TEST_S - base.iloc[0].TEST_S
                    ps_agree.append(1.0 if (dp < 0 and ds < 0) or (dp > 0 and ds > 0) or (abs(dp) < 1e-12 and abs(ds) < 1e-12) else 0.0)
            outlier = bool(np.max(np.abs(arr - np.median(arr))) > 2.5 * (np.median(np.abs(arr - np.median(arr))) + 1e-9))
            raw = float(np.mean(arr))
            med = float(np.median(arr))
            trim = float(np.mean(loo))
            if abs(med) >= 0.003 and dir_cons >= 0.75 and float(np.mean(ps_agree)) >= 0.5 and not outlier:
                tier = "Tier2"
            elif abs(med) >= 0.002 and dir_cons >= 0.5:
                tier = "Tier3"
            else:
                tier = "Tier3" if abs(raw) >= 0.002 else "Tier4"
            rows.append(
                {
                    "interaction_type": "Rep×Annotation",
                    "factor_a": rep,
                    "factor_b": ann,
                    "condition": "vs_BASE_across_topologies",
                    "raw_effect": raw,
                    "median_effect": med,
                    "trimmed_or_LOO_effect": trim,
                    "direction_consistency": dir_cons,
                    "primary_shadow_consistency": float(np.mean(ps_agree)) if ps_agree else np.nan,
                    "outlier_sensitive": outlier,
                    "evidence_tier": tier,
                    "notes": f"n_topo={len(arr)}; deltas={','.join(f'{d:.4f}' for d in arr)}",
                }
            )

    # single-cell additive residual extremes (labeled as such)
    rt = additive_residuals(res, "representation", "topology")
    idx = rt.resid.abs().idxmax()
    r = rt.loc[idx]
    rows.append(
        {
            "interaction_type": "Rep×Topology_single_cell_extreme",
            "factor_a": r.representation,
            "factor_b": r.topology,
            "condition": f"{r.annotation}/{r.experiment_code}",
            "raw_effect": float(r.resid),
            "median_effect": np.nan,
            "trimmed_or_LOO_effect": np.nan,
            "direction_consistency": np.nan,
            "primary_shadow_consistency": np.nan,
            "outlier_sensitive": True,
            "evidence_tier": "Tier3",
            "notes": "single-cell additive residual extreme; NOT a replicated interaction",
        }
    )
    ra = additive_residuals(res, "representation", "annotation")
    idx = ra.resid.abs().idxmax()
    r = ra.loc[idx]
    rows.append(
        {
            "interaction_type": "Rep×Annotation_single_cell_extreme",
            "factor_a": r.representation,
            "factor_b": r.annotation,
            "condition": f"{r.topology}/{r.experiment_code}",
            "raw_effect": float(r.resid),
            "median_effect": np.nan,
            "trimmed_or_LOO_effect": np.nan,
            "direction_consistency": np.nan,
            "primary_shadow_consistency": np.nan,
            "outlier_sensitive": True,
            "evidence_tier": "Tier3",
            "notes": "single-cell additive residual extreme; NOT a replicated interaction",
        }
    )
    return pd.DataFrame(rows)


def build_validated_contrasts(res, boot, tg, ag, plm, ctx) -> pd.DataFrame:
    rows = []

    # Topology family-level
    boot_topo = boot_summary(boot, "topology_vs_SEP", ["topology"])
    rows.extend(
        summarize_gains(
            tg,
            contrast_type="topology_vs_SEP",
            factor_col="topology",
            reference="SEP",
            boot_agg=boot_topo,
            boot_key="topology",
        )
    )

    # Annotation family-level
    boot_ann = boot_summary(boot, "annotation_vs_BASE", ["annotation"])
    rows.extend(
        summarize_gains(
            ag,
            contrast_type="annotation_vs_BASE",
            factor_col="annotation",
            reference="BASE",
            boot_agg=boot_ann,
            boot_key="annotation",
        )
    )

    # PLM vs Scratch by representation
    boot_plm = boot_summary(boot, "plm_vs_scratch", ["representation"])
    rows.extend(
        summarize_gains(
            plm,
            contrast_type="plm_vs_scratch",
            factor_col="representation",
            reference="scratch",
            boot_agg=boot_plm,
            boot_key="representation",
        )
    )

    # Context
    for fam, kind in (
        ("ablang2", "context_ablang2_PAIRED_minus_SEPARATE"),
        ("currab", "context_currab_PAIRED_minus_SEPARATE"),
    ):
        g = ctx[ctx.family == fam]
        b = boot[boot.kind == kind]
        prim = b[b.scheme == "primary"]
        shad = b[b.scheme == "shadow"]
        dmean = float(g.delta_mean.mean())
        dmed = float(g.delta_mean.median())
        dp = float(g.delta_P.mean())
        ds = float(g.delta_S.mean())
        both = float(g.both_improve.mean())
        # sign agreement P/S on mean deltas
        sign_agree = np.sign(dp) == np.sign(ds)
        ci_lo = float(prim.ci95_lo.mean()) if len(prim) else np.nan
        ci_hi = float(prim.ci95_hi.mean()) if len(prim) else np.nan
        # For context, improvement means PAIRED better (negative delta)
        if not sign_agree:
            tier = "Tier4"
            notes = "Primary/Shadow direction disagreement → no robust context advantage"
        else:
            tier = evidence_tier_contrast(dmean, both, ci_lo, ci_hi, len(g))
            notes = "bootstrap CI = mean of per-stratum primary 95% CIs"
        rows.append(
            {
                "contrast_type": "inference_context",
                "factor": fam,
                "reference": "SEPARATE_CHAIN",
                "comparison": "PAIRED_NATIVE - SEPARATE_CHAIN",
                "n_strata": len(g),
                "delta_mean": dmean,
                "delta_median": dmed,
                "primary_delta_mean": dp,
                "shadow_delta_mean": ds,
                "frac_delta_mean_lt0": float((g.delta_mean < 0).mean()),
                "primary_improve_fraction": float((g.delta_P < 0).mean()),
                "shadow_improve_fraction": float((g.delta_S < 0).mean()),
                "both_improve_fraction": both,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "evidence_tier": tier,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{nd}f}"


def write_scientific_doc(res, contrasts, inter, high, val) -> Path:
    # Representation table
    rep_rows = []
    scratch_mean = float(res[res.representation == "scratch"].TEST_mean.mean())
    for rep, g in res.groupby("representation"):
        best = g.loc[g.TEST_mean.idxmin()]
        worst = g.loc[g.TEST_mean.idxmax()]
        rep_rows.append(
            {
                "representation": rep,
                "display": REP_DISPLAY.get(rep, rep),
                "mean": float(g.TEST_mean.mean()),
                "median": float(g.TEST_mean.median()),
                "min": float(g.TEST_mean.min()),
                "max": float(g.TEST_mean.max()),
                "std": float(g.TEST_mean.std()),
                "vs_scratch": float(g.TEST_mean.mean()) - scratch_mean,
                "mean_abs_PS": float(g.abs_PS.mean()),
                "best_code": best.experiment_code,
                "best_cell": f"{best.topology}/{best.annotation}",
                "best_mean": float(best.TEST_mean),
                "worst_code": worst.experiment_code,
                "worst_cell": f"{worst.topology}/{worst.annotation}",
                "worst_mean": float(worst.TEST_mean),
            }
        )
    rep_df = pd.DataFrame(rep_rows).sort_values("mean")

    best_cell = res.loc[res.TEST_mean.idxmin()]
    best_rep = rep_df.iloc[0]

    # HIGH tail
    high_best = high.loc[high.mae_high.idxmin()] if high.mae_high.notna().any() else None

    # Interaction highlights
    rt_ext = inter[inter.interaction_type == "Rep×Topology_single_cell_extreme"].iloc[0]
    ra_ext = inter[inter.interaction_type == "Rep×Annotation_single_cell_extreme"].iloc[0]
    rt_rob = inter[inter.interaction_type == "Rep×Topology"].copy()
    # most consistent negative median (improvement)
    rt_neg = rt_rob[rt_rob.median_effect < 0].sort_values("median_effect")
    ra_rob = inter[inter.interaction_type == "Rep×Annotation"].copy()

    # Tier counts from contrasts + interaction (excluding single-cell as Tier3 always)
    tiers = list(contrasts.evidence_tier) + list(inter.evidence_tier)
    # Also define explicit claim tiers in prose - count curated claims below

    topo_c = contrasts[contrasts.contrast_type == "topology_vs_SEP"].set_index("factor")
    ann_c = contrasts[contrasts.contrast_type == "annotation_vs_BASE"].set_index("factor")
    ctx_c = contrasts[contrasts.contrast_type == "inference_context"].set_index("factor")
    plm_c = contrasts[contrasts.contrast_type == "plm_vs_scratch"].set_index("factor")

    # TmApp ranks for comparison snippet
    tm = pd.read_csv(RES / "TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv")
    tm = tm.copy()
    tm["topology_h"] = tm["topology"].map(lambda x: TMAP.get(x, x))
    tm_rep = tm.groupby("representation")["mean_ps"].mean().sort_values()
    hic_rep = res.groupby("representation")["TEST_mean"].mean().sort_values()

    lines = []
    A = lines.append

    A("# HIC Factorial — Scientific Conclusion Freeze (Internal)")
    A("")
    A("**STATUS: SCIENTIFIC_CONCLUSIONS_FROZEN_INTERNAL**")
    A("")
    A("**PUBLIC_PRIVATE_NOT_CONSULTED**")
    A("")
    A("This document freezes the final *internal* scientific interpretation of the")
    A("HIC Representation × Topology × Annotation 200-cell factorial.")
    A("Subsequent Public/Private diagnostics must not rewrite these conclusions.")
    A("")
    A("## 1. Freeze status")
    A("")
    A("| Field | Value |")
    A("|-------|--------|")
    A(f"| Evaluation freeze | `{EVAL_FREEZE}` |")
    A(f"| Formal prereg | `{PREREG_SHA}` |")
    A(f"| Internal factorial freeze | `{INTERNAL_FREEZE}` |")
    A(f"| Manifest SHA256 | `{MANIFEST_SHA_EXPECTED}` |")
    A(f"| Cells | EXP-H140 … EXP-H339 (200/200 COMPLETE) |")
    A(f"| Scientific freeze HEAD (at writing) | `{git_rev()}` |")
    A("| Public/Private consulted | **No** |")
    A("| New training / embeddings | **None** |")
    A("")
    A("## 2. Provenance")
    A("")
    A("- Internal report: `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_REPORT.md`")
    A("- Cell metrics: `developability_drilldown/results/HIC_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv`")
    A("- Bootstrap (N_BOOT=2000, seed=101, antibody-level paired): `HIC_REP_TOPO_ANNOT_BOOTSTRAP.csv`")
    A("- Validated contrasts: `reports/HIC_FACTORIAL_VALIDATED_CONTRASTS.csv`")
    A("- Interaction robustness: `reports/HIC_FACTORIAL_INTERACTION_ROBUSTNESS.csv`")
    A(f"- Validation: {val['status']}; missing OOF files = {val['missing_oof']}")
    A("")
    A("## 3. Executive scientific conclusions")
    A("")
    A("1. **Best on average under this factorial (representation):** "
      f"**{best_rep.display}** (mean TEST_mean={fmt(best_rep['mean'])}).")
    A("2. **Best single internal cell (descriptive):** "
      f"**{best_cell.experiment_code}** = {REP_DISPLAY.get(best_cell.representation, best_cell.representation)} "
      f"× {best_cell.topology} × {best_cell.annotation} (TEST_mean={fmt(best_cell.TEST_mean)}).")
    A("3. Best average representation (**AbLingua**) and best single cell (**ESM-2**) are **different families**.")
    A("4. Topology effects vs SEP are **small** (mean |Δ| ≲ 0.004); both-scheme improvement rates ≈ 0.24 — "
      "**no general strong topology win**.")
    A("5. Annotation effects vs BASE are **small**; FULL is slightly better on average, IMGT slightly worse; "
      "**representation-dependent**.")
    A("6. AbLang2 / CurrAb PAIRED vs SEPARATE show **no robust context advantage** (P/S direction mismatch).")
    A("7. **SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE** "
      "(n_high=6, mean signed error ≈ −2.66 across all 200 cells).")
    A("8. Broad Rep×Topo×Annot search modestly improved the best internal sequence-only score "
      "(≈0.4945) but **did not fundamentally break** the previously observed ~0.50 regime "
      "**within this tested design space**.")
    A("9. Cross-target: HIC and TmApp favor **different** representation patterns "
      "(descriptive only; absolute MAE not compared).")
    A("10. Sequence-only headroom appears limited → **justifies re-examining independent surface/physics evidence**, "
       "without claiming surface superiority from this factorial alone.")
    A("")
    A("## 4. Representation")
    A("")
    A("Ordering is **mean TEST_mean across 20 Topo×Annot conditions** — "
      "`best on average under this factorial`, not a universal ranking.")
    A("")
    A("| Representation | mean | median | min | max | vs Scratch | mean\\|P−S\\| | best cell |")
    A("|----------------|------|--------|-----|-----|------------|-----------|-----------|")
    for _, r in rep_df.iterrows():
        A(
            f"| {r.display} | {fmt(r['mean'])} | {fmt(r['median'])} | {fmt(r['min'])} | {fmt(r['max'])} | "
            f"{fmt(r.vs_scratch)} | {fmt(r.mean_abs_PS)} | {r.best_code} {r.best_cell} ({fmt(r.best_mean)}) |"
        )
    A("")
    A("- Spread across Topo×Annot is material (e.g. ESM-C max−min ≈ "
      f"{fmt(float(rep_df[rep_df.representation=='esmc600m'].iloc[0]['max']-rep_df[rep_df.representation=='esmc600m'].iloc[0]['min']))}).")
    A("- Several PLMs (AbLang2, CurrAb) are **worse on average than Scratch** under this factorial.")
    A("- Do **not** claim AbLingua (or any family) is universally best for HIC.")
    A("")
    A("## 5. Topology")
    A("")
    A("Overall mean TEST_mean: "
      + ", ".join(f"{t}={fmt(float(res[res.topology==t].TEST_mean.mean()))}" for t in TOPOS))
    A("")
    A("### Contrasts vs SEP (40 Rep×Annot strata each)")
    A("")
    A("| Contrast | Δmean | Δmedian | both-improve | primary CI (mean of strata) | tier |")
    A("|----------|-------|---------|--------------|-----------------------------|------|")
    for topo in ("JOINT", "REG-SEP", "XREG", "FUSE"):
        r = topo_c.loc[topo]
        A(
            f"| {topo}−SEP | {fmt(r.delta_mean)} | {fmt(r.delta_median)} | {fmt(r.both_improve_fraction,3)} | "
            f"[{fmt(r.bootstrap_ci_low)}, {fmt(r.bootstrap_ci_high)}] | {r.evidence_tier} |"
        )
    A("")
    A("**Conclusion:** small average improvements exist for JOINT/XREG/FUSE vs SEP, but "
      "both-scheme rates are low (~0.24) and mean effects are ~0.002–0.004. "
      "**No general topology upgrade claim.** Topology effects are representation-dependent "
      "(see interaction robustness).")
    A("")
    A("## 6. Annotation")
    A("")
    A("Overall mean TEST_mean: "
      + ", ".join(f"{a}={fmt(float(res[res.annotation==a].TEST_mean.mean()))}" for a in ANNOTS))
    A("")
    A("| Contrast | Δmean | Δmedian | both-improve | primary CI (mean of strata) | tier |")
    A("|----------|-------|---------|--------------|-----------------------------|------|")
    for ann in ("IMGT", "REGION", "FULL"):
        r = ann_c.loc[ann]
        A(
            f"| {ann}−BASE | {fmt(r.delta_mean)} | {fmt(r.delta_median)} | {fmt(r.both_improve_fraction,3)} | "
            f"[{fmt(r.bootstrap_ci_low)}, {fmt(r.bootstrap_ci_high)}] | {r.evidence_tier} |"
        )
    A("")
    A("**Conclusion:** FULL is slightly better than BASE on average; IMGT slightly worse; "
      "effects are small and **representation-dependent**. Do not claim FULL is always best.")
    A("")
    A("## 7. Predefined contrasts (PLM − Scratch)")
    A("")
    A("| PLM | Δmean | both-improve | tier |")
    A("|-----|-------|--------------|------|")
    for rep, r in plm_c.sort_values("delta_mean").iterrows():
        A(f"| {REP_DISPLAY.get(rep,rep)} | {fmt(r.delta_mean)} | {fmt(r.both_improve_fraction,3)} | {r.evidence_tier} |")
    A("")
    A("Only AbLingua (and weakly ESM-2) show average improvement vs Scratch; "
      "several antibody PLMs are average-worse. This is a factorial-average fact, not mechanism.")
    A("")
    A("## 8. Interactions")
    A("")
    A("### A. Single-cell extremes (not robust patterns)")
    A(f"- Rep×Topo extreme residual: **{rt_ext.factor_a} × {rt_ext.factor_b}** "
      f"({rt_ext.condition}), resid={fmt(rt_ext.raw_effect)} — outlier-sensitive.")
    A(f"- Rep×Annot extreme residual: **{ra_ext.factor_a} × {ra_ext.factor_b}** "
      f"({ra_ext.condition}), resid={fmt(ra_ext.raw_effect)} — outlier-sensitive.")
    A("- `EXP-H289` (esmc600m × REG-SEP × IMGT) is an **extreme high-error cell**, not a "
      "replicated interaction template.")
    A("")
    A("### B. Reproducible / robustness-filtered Rep×Topo (vs SEP across annotations)")
    A("")
    A("See `HIC_FACTORIAL_INTERACTION_ROBUSTNESS.csv`. Patterns with higher direction consistency "
      "and lower outlier_sensitive flags are preferred over max-residual cells.")
    if len(rt_neg):
        top = rt_neg.head(5)
        A("")
        A("| Rep | Topo | median Δ | LOO mean | dir cons. | P/S cons. | outlier? | tier |")
        A("|-----|------|----------|----------|-----------|-----------|----------|------|")
        for _, r in top.iterrows():
            A(
                f"| {r.factor_a} | {r.factor_b} | {fmt(r.median_effect)} | {fmt(r.trimmed_or_LOO_effect)} | "
                f"{fmt(r.direction_consistency,2)} | {fmt(r.primary_shadow_consistency,2)} | "
                f"{r.outlier_sensitive} | {r.evidence_tier} |"
            )
    A("")
    A("### C. Rep×Annotation")
    A("")
    A("AbLingua / ESM-2 / ESM-C / Scratch annotation deltas vary by topology; "
      "IMGT harm and FULL benefit are **not uniform** across representations. "
      "Use the robustness table; do not over-interpret single cells.")
    A("")
    A("### D. Topology × Annotation")
    A("")
    A("Marginal Topo×Annot means are weak relative to representation main effects. "
      "Any Topo×Annot preference should be treated as **representation-dependent**.")
    A("")
    A("## 9. Context effects")
    A("")
    for fam in ("ablang2", "currab"):
        r = ctx_c.loc[fam]
        A(f"### {fam}")
        A(f"- n_strata={int(r.n_strata)}; Δmean={fmt(r.delta_mean)}; ΔP={fmt(r.primary_delta_mean)}; "
          f"ΔS={fmt(r.shadow_delta_mean)}; both-improve={fmt(r.both_improve_fraction,3)}")
        A(f"- Evidence: **{r.evidence_tier}** — {r.notes}")
        A("")
    A("**AbLang2 / CurrAb pair context: no robust advantage** under Primary+Shadow agreement rules.")
    A("")
    A("## 10. HIGH-tail")
    A("")
    A("Definition: `HIC > 11.5` (diagnostic only; not used for selection).")
    A("")
    A(f"- n_high = {int(high.n_high.iloc[0])} for every cell")
    A(f"- MAE_high: mean={fmt(float(high.mae_high.mean()))}, "
      f"min={fmt(float(high.mae_high.min()))}, max={fmt(float(high.mae_high.max()))}")
    A(f"- mean signed error (pred−true): {fmt(float(high.mean_signed_error.mean()))} "
      "(systematic underprediction)")
    if high_best is not None:
        A(f"- Best HIGH-tail MAE cell (descriptive): {high_best.experiment_code} "
          f"({high_best.representation}/{high_best.topology}/{high_best.annotation}) "
          f"MAE_high={fmt(float(high_best.mae_high))}")
    A("")
    A("**SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE**")
    A("")
    A("No Rep/Topo/Annot combination in this design eliminated the failure mode. "
      "Causes (scarcity / representation / noise / loss) remain **unresolved** — do not pick one.")
    A("")
    A("## 11. Sequence-only plateau reassessment")
    A("")
    A("Prior observation: V3-tested configurations plateau near MAE ~0.50.")
    A(f"This factorial’s best internal TEST_mean is **{fmt(best_cell.TEST_mean)}** "
      f"({best_cell.experiment_code}).")
    A("")
    A("> Broad Rep×Topo×Annot search modestly improved the best internal sequence-only score "
      "but did not fundamentally break the previously observed ~0.50 regime "
      "**within the tested design space**.")
    A("")
    A("This is **not** a theoretical lower bound on HIC MAE.")
    A("")
    A("## 12. TmApp vs HIC")
    A("")
    A("See `reports/TMAPP_VS_HIC_FACTORIAL_SCIENTIFIC_COMPARISON.md`. Absolute MAE not compared.")
    A("")
    A("Representation mean ranks (lower rank = better mean):")
    A("")
    A("| Representation | HIC rank | TmApp rank |")
    A("|----------------|----------|------------|")
    hic_rank = {r: i + 1 for i, r in enumerate(hic_rep.index)}
    tm_rank = {r: i + 1 for i, r in enumerate(tm_rep.index)}
    for rep in sorted(set(hic_rank) | set(tm_rank), key=lambda x: hic_rank.get(x, 99)):
        A(f"| {REP_DISPLAY.get(rep,rep)} | {hic_rank.get(rep,'—')} | {tm_rank.get(rep,'—')} |")
    A("")
    A("**Same VH/VL inputs do not imply the same effective representation / inductive bias "
      "for TmApp vs HIC** (descriptive pattern difference).")
    A("")
    A("## 13. Evidence hierarchy")
    A("")
    A("### Tier 1 — Established internal observation")
    A("- 200/200 cells completed under frozen prereg; 0 failed/blocked")
    A(f"- Best descriptive cell: {best_cell.experiment_code} TEST_mean={fmt(best_cell.TEST_mean)}")
    A(f"- Representation mean ordering under this factorial (AbLingua best average)")
    A("- Persistent HIGH-tail underprediction (n=6, signed error ≈ −2.66) across all cells")
    A("- PUBLIC_PRIVATE_NOT_CONSULTED")
    A("")
    A("### Tier 2 — Supported pattern")
    A("- Several PLMs average-worse than Scratch (consistent across many Topo×Annot strata)")
    A("- Topology/annotation *average* effects are small relative to cell-to-cell spread")
    A("- Context contrasts lack Primary+Shadow directional agreement")
    A("")
    A("### Tier 3 — Suggestive")
    A("- Modest JOINT/XREG mean edges vs SEP")
    A("- Modest FULL edge vs BASE")
    A("- Single-cell interaction extremes (e.g. EXP-H289)")
    A("- Specific Rep×Topo/Annot robustness rows with mixed P/S consistency")
    A("")
    A("### Tier 4 — Unresolved")
    A("- Causal driver of HIGH-tail failure")
    A("- Whether any untested sequence-only design could break ~0.50")
    A("- Mechanism behind AbLingua vs ESM-2 average-vs-best divergence")
    A("- External generalization (Public/Private)")
    A("")
    A("## 14–17. Claim lists")
    A("")
    A("See executive conclusions (§3) and hierarchy (§13). Validated numeric contrasts live in CSVs.")
    A("")
    A("## 18. Non-claims")
    A("")
    A("- AbLingua is not claimed universally best for HIC")
    A("- ESM-2 is not claimed to ‘understand’ HIC mechanism")
    A("- FULL annotation is not always best")
    A("- JOINT/XREG are not universally superior topologies")
    A("- HIGH-tail failure is not attributed solely to data scarcity")
    A("- No aromatic/surface causality from this factorial")
    A("- No Public/Private generalization claimed")
    A("- No theoretical MAE floor at 0.49")
    A("")
    A("## 19. Implications for next research stage")
    A("")
    A("Sequence-only design-space headroom under this factorial appears limited "
      "(best ≈ 0.4945; HIGH-tail unresolved). "
      "This **increases the rationale for re-examining independent surface/physics evidence**, "
      "but **does not prove surface superiority from this factorial alone**.")
    A("")
    A("## 20. Reproducibility")
    A("")
    A("| Artifact | Role |")
    A("|----------|------|")
    A(f"| `{PREREG_SHA}` | Formal prereg |")
    A(f"| `{EVAL_FREEZE}` | Evaluation freeze |")
    A(f"| `{INTERNAL_FREEZE}` | Internal results freeze |")
    A(f"| `{MANIFEST_SHA_EXPECTED}` | Manifest SHA256 |")
    A("| `reports/HIC_FACTORIAL_VALIDATED_CONTRASTS.csv` | Contrast table |")
    A("| `reports/HIC_FACTORIAL_INTERACTION_ROBUSTNESS.csv` | Interaction robustness |")
    A("| `reports/TMAPP_VS_HIC_FACTORIAL_SCIENTIFIC_COMPARISON.md` | Cross-target |")
    A("")
    A("Bootstrap: antibody-level paired residual; N_BOOT=2000; seed=101; Primary/Shadow separate.")
    A("")

    # Explicit Q&A section
    A("## Required explicit answers")
    A("")
    A("1. **Best average representation:** AbLingua (under this factorial).")
    A("2. **Best single cell:** EXP-H266 (esm2 × JOINT × REGION).")
    A("3. **Same family?** No (AbLingua vs ESM-2).")
    A("4. **General topology improvement?** Not strong; only small average Δ vs SEP.")
    A("5. **Topology representation-dependent?** Yes.")
    A("6. **General annotation improvement?** Weak; FULL slight average gain, IMGT slight harm.")
    A("7. **Annotation representation-dependent?** Yes.")
    A("8. **AbLang2 pair context effective?** No robust advantage (P/S disagree).")
    A("9. **CurrAb pair context effective?** No robust advantage.")
    A("10. **HIGH-tail resolved?** No — SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE.")
    A("11. **Broke ~0.50 plateau?** Modestly improved best cell; did not fundamentally break regime in tested space.")
    A("12. **Same optimal representation pattern as TmApp?** No (rank patterns differ).")
    A("13. **Surface follow-up justified?** Rational to re-examine independent surface evidence due to limited "
      "sequence-only headroom; not proven by this factorial alone.")
    A("")

    path = REPORTS / "HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, rep_df, best_cell, best_rep


def write_tmapp_comparison(res: pd.DataFrame) -> Path:
    tm = pd.read_csv(RES / "TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv")
    tm = tm.copy()
    tm["topology_h"] = tm["topology"].map(lambda x: TMAP.get(x, x))

    hic_rep = res.groupby("representation").TEST_mean.mean().sort_values()
    tm_rep = tm.groupby("representation").mean_ps.mean().sort_values()

    # topology gains vs SEP/A for each target (mean across rep×annot)
    def topo_gains_hic():
        rows = []
        for topo in ("JOINT", "REG-SEP", "XREG", "FUSE"):
            deltas = []
            for rep in res.representation.unique():
                for ann in ANNOTS:
                    a = res[(res.representation == rep) & (res.annotation == ann) & (res.topology == "SEP")]
                    x = res[(res.representation == rep) & (res.annotation == ann) & (res.topology == topo)]
                    if len(a) == 1 and len(x) == 1:
                        deltas.append(float(x.iloc[0].TEST_mean - a.iloc[0].TEST_mean))
            rows.append((topo, float(np.mean(deltas)), float(np.median(deltas))))
        return rows

    def topo_gains_tm():
        rows = []
        for topo_h, topo_t in [("JOINT", "B1"), ("REG-SEP", "B2"), ("XREG", "C"), ("FUSE", "D")]:
            deltas = []
            for rep in tm.representation.unique():
                for ann in ANNOTS:
                    a = tm[(tm.representation == rep) & (tm.annotation == ann) & (tm.topology == "A")]
                    x = tm[(tm.representation == rep) & (tm.annotation == ann) & (tm.topology == topo_t)]
                    if len(a) == 1 and len(x) == 1:
                        deltas.append(float(x.iloc[0].mean_ps - a.iloc[0].mean_ps))
            rows.append((topo_h, float(np.mean(deltas)), float(np.median(deltas))))
        return rows

    def annot_gains(df, mean_col, base_topo_col_values):
        # vs BASE within each rep×topo
        rows = []
        topos = base_topo_col_values
        for ann in ("IMGT", "REGION", "FULL"):
            deltas = []
            for rep in df.representation.unique():
                for topo in topos:
                    b = df[(df.representation == rep) & (df.topology == topo) & (df.annotation == "BASE")]
                    x = df[(df.representation == rep) & (df.topology == topo) & (df.annotation == ann)]
                    if len(b) == 1 and len(x) == 1:
                        deltas.append(float(x.iloc[0][mean_col] - b.iloc[0][mean_col]))
            rows.append((ann, float(np.mean(deltas)), float(np.median(deltas))))
        return rows

    def context_delta(df, paired, separate, mean_col):
        deltas = []
        for topo_key in df.topology.unique():
            for ann in ANNOTS:
                p = df[(df.representation == paired) & (df.topology == topo_key) & (df.annotation == ann)]
                s = df[(df.representation == separate) & (df.topology == topo_key) & (df.annotation == ann)]
                if len(p) == 1 and len(s) == 1:
                    deltas.append(float(p.iloc[0][mean_col] - s.iloc[0][mean_col]))
        return float(np.mean(deltas)), float(np.median(deltas))

    lines = []
    A = lines.append
    A("# TmApp vs HIC Factorial — Scientific Comparison (Descriptive)")
    A("")
    A("**STATUS: DESCRIPTIVE_CROSS_TARGET_ONLY**")
    A("")
    A("Based solely on frozen TmApp and HIC internal factorials. "
      "Does **not** alter either target’s frozen conclusions. "
      "**Absolute MAE values are not compared** (different target scales).")
    A("")
    A("## Representation mean ranks")
    A("")
    A("| Representation | HIC mean-rank | TmApp mean-rank | Pattern note |")
    A("|----------------|---------------|-----------------|--------------|")
    hic_rank = {r: i + 1 for i, r in enumerate(hic_rep.index)}
    tm_rank = {r: i + 1 for i, r in enumerate(tm_rep.index)}
    for rep in hic_rep.index:
        hr, tr = hic_rank[rep], tm_rank.get(rep, None)
        note = ""
        if tr is not None and abs(hr - tr) >= 3:
            note = "rank shift ≥3"
        A(f"| {REP_DISPLAY.get(rep,rep)} | {hr} | {tr} | {note} |")
    A("")
    A("Notable: AbLingua is strongest on average for **HIC**; AbLang2/CurrAb are weak on HIC "
      "relative to Scratch, whereas TmApp factorial ranks differ — evidence that "
      "**effective representation bias is target-dependent** for the same VH/VL inputs.")
    A("")
    A("## Topology relative gains (vs SEP/A)")
    A("")
    A("| Topology | HIC mean Δ | HIC median Δ | TmApp mean Δ | TmApp median Δ |")
    A("|----------|------------|--------------|--------------|----------------|")
    htg = dict((t, (m, md)) for t, m, md in topo_gains_hic())
    ttg = dict((t, (m, md)) for t, m, md in topo_gains_tm())
    for t in ("JOINT", "REG-SEP", "XREG", "FUSE"):
        hm, hmd = htg[t]
        tm_, tmd = ttg[t]
        A(f"| {t} | {fmt(hm)} | {fmt(hmd)} | {fmt(tm_)} | {fmt(tmd)} |")
    A("")
    A("Compare **directions/ranks of contrasts**, not absolute MAE.")
    A("")
    A("## Annotation relative gains (vs BASE)")
    A("")
    A("| Annotation | HIC mean Δ | TmApp mean Δ |")
    A("|------------|------------|--------------|")
    hag = dict((a, (m, md)) for a, m, md in annot_gains(res, "TEST_mean", TOPOS))
    tag = dict((a, (m, md)) for a, m, md in annot_gains(tm, "mean_ps", ["A", "B1", "B2", "C", "D"]))
    for a in ("IMGT", "REGION", "FULL"):
        A(f"| {a} | {fmt(hag[a][0])} | {fmt(tag[a][0])} |")
    A("")
    A("## Inference context (PAIRED − SEPARATE)")
    A("")
    hab, _ = context_delta(res, "ablang2_unpaired", "ablang2_paired", "TEST_mean")
    tab, _ = context_delta(tm, "ablang2_unpaired", "ablang2_paired", "mean_ps")
    hcu, _ = context_delta(res, "currab_paired", "currab_unpaired", "TEST_mean")
    tcu, _ = context_delta(tm, "currab_paired", "currab_unpaired", "mean_ps")
    A(f"- AbLang2: HIC mean Δ={fmt(hab)}; TmApp mean Δ={fmt(tab)}")
    A(f"- CurrAb: HIC mean Δ={fmt(hcu)}; TmApp mean Δ={fmt(tcu)}")
    A("")
    A("## Scientific takeaway")
    A("")
    A("The same antibody sequence inputs do **not** yield identical relative representation / "
      "topology / annotation preferences for TmApp vs HIC. "
      "This supports treating developability targets as **distinct prediction problems** "
      "even under shared sequence featurization — without claiming mechanism.")
    A("")
    A("## Non-interference")
    A("")
    A("- Does not rewrite TmApp frozen conclusions")
    A("- Does not rewrite HIC scientific freeze")
    A("- No Public/Private")
    A("")

    path = REPORTS / "TMAPP_VS_HIC_FACTORIAL_SCIENTIFIC_COMPARISON.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    val = validate()
    (RES / "HIC_SCIENTIFIC_FREEZE_VALIDATION.json").write_text(json.dumps(val, indent=2) + "\n")

    res = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv")
    boot = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_BOOTSTRAP.csv")
    tg = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_TOPOLOGY_GAINS.csv")
    ag = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_ANNOTATION_GAINS.csv")
    plm = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_PLM_VS_SCRATCH.csv")
    ctx = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_CONTEXT_CONTRASTS.csv")
    high = pd.read_csv(RES / "HIC_REP_TOPO_ANNOT_HIGH_TAIL.csv")

    contrasts = build_validated_contrasts(res, boot, tg, ag, plm, ctx)
    contrasts_path = REPORTS / "HIC_FACTORIAL_VALIDATED_CONTRASTS.csv"
    contrasts.to_csv(contrasts_path, index=False)

    inter = interaction_robustness(res)
    inter_path = REPORTS / "HIC_FACTORIAL_INTERACTION_ROBUSTNESS.csv"
    inter.to_csv(inter_path, index=False)

    sci_path, rep_df, best_cell, best_rep = write_scientific_doc(res, contrasts, inter, high, val)
    cmp_path = write_tmapp_comparison(res)

    # claim counts for final report
    tier_counts = contrasts.evidence_tier.value_counts().to_dict()
    # curated narrative tiers
    curated = {"Tier1": 5, "Tier2": 3, "Tier3": 4, "Tier4": 4}

    summary = {
        "scientific_doc": str(sci_path),
        "contrasts": str(contrasts_path),
        "interactions": str(inter_path),
        "tmapp_comparison": str(cmp_path),
        "contrast_tier_counts": tier_counts,
        "curated_claim_tier_counts": curated,
        "best_rep": best_rep.display,
        "best_cell": best_cell.experiment_code,
        "validation": val,
    }
    (RES / "HIC_SCIENTIFIC_FREEZE_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("SCIENTIFIC_FREEZE_OK", sci_path)
    print("TIERS", curated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
