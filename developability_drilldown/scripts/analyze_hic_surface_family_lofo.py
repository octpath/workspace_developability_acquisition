#!/usr/bin/env python3
"""Analyze F1_SURFACE physical-family LOFO (FULL_MINUS_F vs FULL)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "developability_drilldown"
REPORTS = REPO / "reports"
sys.path.insert(0, str(ROOT / "scripts"))
from _lib import load_solution, mae  # noqa: E402

PREREG_SHA = "e45d9ea8c4720ac92388f7219e7c0d6a0b92cf3f"
FREEZE_V3_SHA = "1bfdf9107f55d14c58117aaa7f64e255dc387ed9"
ARO_HYDRO_SHA = "ebf2fda9615c12f037914bc5b605a6ec6ba2eabc"
N_BOOT = 10_000
BOOT_SEED = 101
HIGH_THR = 11.5
TOL = 1e-6
ANCHORS = ["EXP-H030", "EXP-H107", "EXP-H137"]
FULL_BASELINES = {"ablang1": "EXP-H341", "ablingua": "EXP-H343"}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_series() -> list[dict]:
    return json.loads((ROOT / "results/H348_FAMILY_LOFO_SERIES.json").read_text())


def load_tax() -> dict:
    return json.loads((ROOT / "results/HIC_SURFACE_FAMILY_TAXONOMY_FROZEN.json").read_text())


def load_oof(code: str, name: str) -> pd.Series:
    df = pd.read_csv(ROOT / "experiments/predictions" / code / name)
    return df.set_index("id")["y_pred"].astype(float)


def load_test_pred(code: str) -> pd.Series:
    df = pd.read_csv(ROOT / "experiments/predictions" / code / "test.csv")
    idc = "id" if "id" in df.columns else df.columns[0]
    pc = "y_pred" if "y_pred" in df.columns else df.columns[1]
    return df.set_index(idc)[pc].astype(float)


def paired_boot(delta: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(delta)
    boots = np.empty(N_BOOT, float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[i] = float(delta[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n": n,
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "frac_worsen": float((delta > 0).mean()),
        "frac_improve": float((delta < 0).mean()),
    }


def score_ext(pred: pd.Series, sol2: pd.DataFrame) -> dict:
    ids = sol2.index.tolist()
    pub = sol2.index[sol2["is_public"].astype(bool)]
    priv = sol2.index[sol2["is_private"].astype(bool)]
    y = sol2["HIC"].astype(float)
    te = pred.reindex(ids)
    if te.isna().any():
        raise ValueError("missing prediction ids")
    pub_m = float(mae(y.loc[pub].to_numpy(), te.loc[pub].to_numpy(float)))
    priv_m = float(mae(y.loc[priv].to_numpy(), te.loc[priv].to_numpy(float)))
    ov = float(mae(y.loc[ids].to_numpy(), te.loc[ids].to_numpy(float)))
    return {"public_mae": pub_m, "private_mae": priv_m, "test_overall_mae": ov, "public_private_gap": abs(pub_m - priv_m)}


def evidence_class(row_a1: dict, row_a2: dict) -> str:
    """Predeclared directional class from external contrasts only."""
    tests = [row_a1["delta_test"] > 0, row_a2["delta_test"] > 0]
    pubs = [row_a1["delta_public"] > 0, row_a2["delta_public"] > 0]
    privs = [row_a1["delta_private"] > 0, row_a2["delta_private"] > 0]
    if all(tests) and all(pubs) and all(privs):
        return "ROBUST_CONDITIONAL_CONTRIBUTOR"
    if all(tests):
        return "EXTERNAL_DIRECTIONAL_CONTRIBUTOR"
    # one rep Test worsens, other not
    if tests[0] != tests[1]:
        return "CONTEXT_DEPENDENT_CONTRIBUTOR"
    # both Test not worsen
    return "LITTLE_UNIQUE_CONDITIONAL_VALUE"


def global_interpretation(class_map: dict[str, str], contrasts: pd.DataFrame) -> str:
    aro_fams = [f for f in class_map if f.startswith("ARO_")]
    hydro_fams = [f for f in class_map if f.startswith("HYDRO_")]
    contrib = {"ROBUST_CONDITIONAL_CONTRIBUTOR", "EXTERNAL_DIRECTIONAL_CONTRIBUTOR"}
    aro_c = [f for f in aro_fams if class_map[f] in contrib]
    hydro_c = [f for f in hydro_fams if class_map[f] in contrib]
    ctx = [f for f, c in class_map.items() if c == "CONTEXT_DEPENDENT_CONTRIBUTOR"]
    little = [f for f, c in class_map.items() if c == "LITTLE_UNIQUE_CONDITIONAL_VALUE"]

    # mean Test delta across reps per family
    mean_test = (
        contrasts[contrasts.metric == "test"].groupby("family_id")["delta"].mean().to_dict()
    )
    positive_test = [f for f, d in mean_test.items() if d > 0]

    if len(ctx) >= 3 and len(aro_c) + len(hydro_c) <= 1:
        return "CONTEXT_DEPENDENT_SURFACE_SIGNAL"
    if len(aro_c) >= 1 and len(hydro_c) >= 1:
        return "AROMATIC_AND_HYDRO_FAMILIES_COMPLEMENTARY"
    if len(aro_c) >= 2 and len(hydro_c) == 0:
        return "AROMATIC_FAMILY_ENRICHED"
    if len(hydro_c) >= 2 and len(aro_c) == 0:
        return "HYDRO_FAMILY_ENRICHED"
    if len(positive_test) >= 4:
        return "DISTRIBUTED_SURFACE_SIGNAL"
    if len(little) == len(class_map):
        return "FAMILY_LOCALIZATION_INCONCLUSIVE"
    if len(aro_c) + len(hydro_c) >= 3:
        return "DISTRIBUTED_SURFACE_SIGNAL"
    if len(aro_c) == 1 and len(hydro_c) == 0:
        return "AROMATIC_FAMILY_ENRICHED"
    if len(hydro_c) == 1 and len(aro_c) == 0:
        return "HYDRO_FAMILY_ENRICHED"
    if ctx:
        return "CONTEXT_DEPENDENT_SURFACE_SIGNAL"
    return "FAMILY_LOCALIZATION_INCONCLUSIVE"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    series = load_series()
    tax = load_tax()
    sol = load_solution().set_index("id")
    exp = pd.read_csv(ROOT / "results/experiments.csv").set_index("experiment_code")
    y_dev = pd.read_csv(ROOT / "data/dev.csv").set_index("id")["HIC"].astype(float)
    y_te = sol["HIC"].astype(float)
    te_ids = sol.index.tolist()
    high_te = [i for i in te_ids if y_te.loc[i] > HIGH_THR]
    non_te = [i for i in te_ids if y_te.loc[i] <= HIGH_THR]
    rng = np.random.default_rng(BOOT_SEED)

    # pipeline validation
    max_d = 0.0
    for code in ANCHORS:
        sc = score_ext(load_test_pred(code), sol)
        reg = exp.loc[code]
        for k, col in [("public_mae", "public_mae"), ("private_mae", "private_mae"), ("test_overall_mae", "test_overall_mae")]:
            max_d = max(max_d, abs(sc[k] - float(reg[col])))
    if max_d > TOL:
        raise SystemExit(f"PIPELINE_FAIL {max_d}")

    # config audit
    audit_rows = []
    for s in series:
        code = s["experiment_code"]
        cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
        doc = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
        audit_rows.append(
            {
                "experiment_code": code,
                "representation": s["representation"],
                "family_id": s["family_id"],
                "aux_dim": cfg.get("aux_dim"),
                "mask_stage": cfg.get("mask_stage"),
                "zero_indices": json.dumps(s["zero_indices"]),
                "seed": cfg.get("seed"),
                "topology_id": cfg.get("topology_id"),
                "annotation_mode": cfg.get("annotation_mode"),
                "n_trainable": doc.get("n_trainable"),
                "full_baseline": s["full_baseline"],
            }
        )
    # baselines
    for rep, code in FULL_BASELINES.items():
        cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
        doc = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
        audit_rows.append(
            {
                "experiment_code": code,
                "representation": rep,
                "family_id": "FULL35",
                "aux_dim": cfg.get("aux_dim"),
                "mask_stage": "none",
                "zero_indices": "[]",
                "seed": cfg.get("seed"),
                "topology_id": cfg.get("topology_id"),
                "annotation_mode": cfg.get("annotation_mode"),
                "n_trainable": doc.get("n_trainable"),
                "full_baseline": code,
            }
        )
    pd.DataFrame(audit_rows).to_csv(REPORTS / "HIC_SURFACE_FAMILY_LOFO_CONFIG_AUDIT.csv", index=False)

    # scores
    score_rows = []
    for rep, code in FULL_BASELINES.items():
        oof = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
        sc = score_ext(load_test_pred(code), sol)
        score_rows.append(
            {
                "experiment_code": code,
                "representation": rep,
                "family_id": "FULL35",
                "arm": "FULL35",
                "cv_primary": float(oof["oof_test"]["primary"]),
                "cv_shadow": float(oof["oof_test"]["shadow"]),
                "cv_mean": float(oof["oof_test"]["mean"]),
                **sc,
            }
        )
    for s in series:
        code = s["experiment_code"]
        oof = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
        sc = score_ext(load_test_pred(code), sol)
        score_rows.append(
            {
                "experiment_code": code,
                "representation": s["representation"],
                "family_id": s["family_id"],
                "arm": s["arm"],
                "cv_primary": float(oof["oof_test"]["primary"]),
                "cv_shadow": float(oof["oof_test"]["shadow"]),
                "cv_mean": float(oof["oof_test"]["mean"]),
                **sc,
            }
        )
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(REPORTS / "HIC_SURFACE_FAMILY_LOFO_SCORES.csv", index=False)

    # fill experiments.csv for new arms
    exp2 = pd.read_csv(ROOT / "results/experiments.csv")
    for _, r in scores_df[scores_df.family_id != "FULL35"].iterrows():
        m = exp2.experiment_code == r.experiment_code
        if m.any():
            exp2.loc[m, "public_mae"] = r.public_mae
            exp2.loc[m, "private_mae"] = r.private_mae
            exp2.loc[m, "test_overall_mae"] = r.test_overall_mae
            if "public_private_gap" in exp2.columns:
                exp2.loc[m, "public_private_gap"] = r.public_private_gap
            exp2.loc[m, "status"] = "COMPLETE"
    exp2.to_csv(ROOT / "results/experiments.csv", index=False)

    # contrasts FULL_MINUS_F - FULL
    metric_map = {
        "cv_primary": "cv_primary",
        "cv_shadow": "cv_shadow",
        "cv_mean": "cv_mean",
        "public": "public_mae",
        "private": "private_mae",
        "test": "test_overall_mae",
    }
    contrast_rows = []
    boot_rows = []
    by_rep_fam = {(s["representation"], s["family_id"]): s for s in series}

    for fam in tax["family_order"]:
        for rep in ("ablang1", "ablingua"):
            full_code = FULL_BASELINES[rep]
            minus = by_rep_fam[(rep, fam)]
            minus_code = minus["experiment_code"]
            sf = scores_df[(scores_df.experiment_code == full_code)].iloc[0]
            sm = scores_df[(scores_df.experiment_code == minus_code)].iloc[0]
            for metric, col in metric_map.items():
                va = float(sm[col])
                vb = float(sf[col])
                contrast_rows.append(
                    {
                        "representation": rep,
                        "family_id": fam,
                        "minus_code": minus_code,
                        "full_code": full_code,
                        "metric": metric,
                        "value_minus": va,
                        "value_full": vb,
                        "delta": va - vb,
                    }
                )

            # OOF primary bootstrap
            pm = load_oof(minus_code, "oof_test_primary.csv")
            pf = load_oof(full_code, "oof_test_primary.csv")
            ids = sorted(set(pm.index) & set(pf.index) & set(y_dev.index))
            delta = np.abs(pm.loc[ids].to_numpy() - y_dev.loc[ids].to_numpy()) - np.abs(
                pf.loc[ids].to_numpy() - y_dev.loc[ids].to_numpy()
            )
            boot_rows.append({"representation": rep, "family_id": fam, "split": "oof_primary", **paired_boot(delta, rng)})

            tm = load_test_pred(minus_code).reindex(te_ids)
            tf = load_test_pred(full_code).reindex(te_ids)
            delta_t = np.abs(tm.to_numpy(float) - y_te.loc[te_ids].to_numpy()) - np.abs(
                tf.to_numpy(float) - y_te.loc[te_ids].to_numpy()
            )
            boot_rows.append({"representation": rep, "family_id": fam, "split": "test", **paired_boot(delta_t, rng)})

    contrasts_df = pd.DataFrame(contrast_rows)
    contrasts_df.to_csv(REPORTS / "HIC_SURFACE_FAMILY_LOFO_CONTRASTS.csv", index=False)
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(REPORTS / "HIC_SURFACE_FAMILY_LOFO_BOOTSTRAP.csv", index=False)

    # evidence classes
    class_map = {}
    class_rows = []
    for fam in tax["family_order"]:
        def pack(rep):
            d = {}
            for m in ("public", "private", "test", "cv_primary", "cv_shadow"):
                d[f"delta_{m}"] = float(
                    contrasts_df[
                        (contrasts_df.family_id == fam)
                        & (contrasts_df.representation == rep)
                        & (contrasts_df.metric == m)
                    ].iloc[0]["delta"]
                )
            return d

        a1, a2 = pack("ablang1"), pack("ablingua")
        lab = evidence_class(a1, a2)
        class_map[fam] = lab
        class_rows.append({"family_id": fam, "evidence_class": lab, **{f"ablang1_{k}": v for k, v in a1.items()}, **{f"ablingua_{k}": v for k, v in a2.items()}})
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(REPORTS / "HIC_SURFACE_FAMILY_LOFO_EVIDENCE_CLASSES.csv", index=False)

    # HIGH-tail
    high_rows = []
    for rep, full_code in FULL_BASELINES.items():
        tf = load_test_pred(full_code)
        for subset, id_list in (("HIGH", high_te), ("nonHIGH", non_te)):
            yy = y_te.loc[id_list].to_numpy()
            pp = tf.loc[id_list].to_numpy(float)
            high_rows.append(
                {
                    "representation": rep,
                    "family_id": "FULL35",
                    "subset": subset,
                    "n": len(id_list),
                    "mae": float(np.abs(pp - yy).mean()),
                    "signed_error": float((pp - yy).mean()),
                    "delta_mae_vs_full": 0.0,
                    "delta_signed_vs_full": 0.0,
                }
            )
        for fam in tax["family_order"]:
            code = by_rep_fam[(rep, fam)]["experiment_code"]
            tm = load_test_pred(code)
            for subset, id_list in (("HIGH", high_te), ("nonHIGH", non_te)):
                yy = y_te.loc[id_list].to_numpy()
                pm = tm.loc[id_list].to_numpy(float)
                pf = tf.loc[id_list].to_numpy(float)
                mae_m = float(np.abs(pm - yy).mean())
                mae_f = float(np.abs(pf - yy).mean())
                sig_m = float((pm - yy).mean())
                sig_f = float((pf - yy).mean())
                high_rows.append(
                    {
                        "representation": rep,
                        "family_id": fam,
                        "subset": subset,
                        "n": len(id_list),
                        "mae": mae_m,
                        "signed_error": sig_m,
                        "delta_mae_vs_full": mae_m - mae_f,
                        "delta_signed_vs_full": sig_m - sig_f,
                    }
                )
    high_df = pd.DataFrame(high_rows)
    high_df.to_csv(REPORTS / "HIC_SURFACE_FAMILY_LOFO_HIGHTAIL.csv", index=False)

    # heterogeneity
    het_rows = []
    for fam in tax["family_order"]:
        worsen_maps = {}
        for rep, full_code in FULL_BASELINES.items():
            code = by_rep_fam[(rep, fam)]["experiment_code"]
            tm = load_test_pred(code).reindex(te_ids).to_numpy(float)
            tf = load_test_pred(full_code).reindex(te_ids).to_numpy(float)
            yv = y_te.loc[te_ids].to_numpy()
            d = np.abs(tm - yv) - np.abs(tf - yv)
            worsen_maps[rep] = d > 0
            for thr_lab, mask in (
                ("ALL", np.ones(len(te_ids), bool)),
                ("HIGH", yv > HIGH_THR),
                ("nonHIGH", yv <= HIGH_THR),
            ):
                dd = d[mask]
                het_rows.append(
                    {
                        "representation": rep,
                        "family_id": fam,
                        "subset": thr_lab,
                        "n": int(mask.sum()),
                        "frac_worsen_on_removal": float((dd > 0).mean()),
                        "frac_improve_on_removal": float((dd < 0).mean()),
                        "mean_delta_ae": float(dd.mean()),
                    }
                )
        agree = float((worsen_maps["ablang1"] == worsen_maps["ablingua"]).mean())
        het_rows.append(
            {
                "representation": "both",
                "family_id": fam,
                "subset": "ALL",
                "n": len(te_ids),
                "frac_worsen_agree": agree,
                "frac_both_worsen": float((worsen_maps["ablang1"] & worsen_maps["ablingua"]).mean()),
            }
        )
    pd.DataFrame(het_rows).to_csv(REPORTS / "HIC_SURFACE_FAMILY_LOFO_HETEROGENEITY.csv", index=False)

    ginterp = global_interpretation(class_map, contrasts_df)

    # aromatic hypothesis update
    aro_contrib = [f for f, c in class_map.items() if f.startswith("ARO_") and c in ("ROBUST_CONDITIONAL_CONTRIBUTOR", "EXTERNAL_DIRECTIONAL_CONTRIBUTOR")]
    hydro_contrib = [f for f, c in class_map.items() if f.startswith("HYDRO_") and c in ("ROBUST_CONDITIONAL_CONTRIBUTOR", "EXTERNAL_DIRECTIONAL_CONTRIBUTOR")]
    if aro_contrib and hydro_contrib:
        aro_status = (
            "PARTIAL_AT_FAMILY_LEVEL — aromatic families carry some conditional value, but hydro families also do. "
            "Do not collapse to exposed-aromatic-only causality; do not claim specific aromatic feature X is causal."
        )
        next_rec = "Optional later: within robust families only, coarse sub-probes — still no 35-way search."
    elif aro_contrib and not hydro_contrib:
        aro_status = (
            "SUPPORTED_AT_ARO_FAMILY_LEVEL — aromatic physical families dominate conditional contribution among LOFO arms. "
            "Still not causal; specific exposed-aromatic feature causality not licensed."
        )
        next_rec = "Consider ARO-family-internal coarse probes for robust aromatic families only."
    elif hydro_contrib and not aro_contrib:
        aro_status = "NOT_SUPPORTED_AS_PRIMARY_STORY — hydro families dominate; avoid aromatics-only narrative."
        next_rec = "Consider HYDRO-family-internal coarse probes for robust hydro families only."
    else:
        aro_status = "INCONCLUSIVE_AT_FAMILY_LEVEL — no clear robust family contributors under LOFO."
        next_rec = "Stop or revisit only with new independent evidence; no feature-level search."

    # markdown
    lines = []
    A = lines.append
    A("# HIC SURFACE Physical-Family LOFO Results")
    A("")
    A("**STATUS: `HIC_SURFACE_FAMILY_LOFO_COMPLETE`**")
    A("")
    A("| Field | Value |")
    A("|-------|--------|")
    A(f"| Taxonomy / prereg SHA | `{PREREG_SHA}` |")
    A(f"| Freeze v3 SHA | `{FREEZE_V3_SHA}` |")
    A(f"| ARO/HYDRO decomp SHA | `{ARO_HYDRO_SHA}` |")
    A(f"| Analysis code SHA | `{git_rev()}` |")
    A(f"| n_families | {len(tax['family_order'])} |")
    A(f"| Global interpretation | `{ginterp}` |")
    A("")
    A("## 1–3. Taxonomy")
    A("")
    A("See `HIC_SURFACE_FAMILY_TAXONOMY.csv` (35 columns, exactly-once family membership).")
    A("")
    for fam in tax["family_order"]:
        idxs = tax["family_indices_0based"][fam]
        cols = [tax["canonical_columns"][i] for i in idxs]
        A(f"- **`{fam}`** indices={idxs}: {', '.join(cols)}")
    A("")
    A("## 4. Mask implementation audit")
    A("")
    A("Mask stage: **post_standardscaler_model_input** (removed coords set to exact 0 after fold-local scale).")
    A("Smoke: `tests/test_hic_surface_family_lofo_mask.py`. Config audit CSV written.")
    A("")
    A("## 5–6. Experiment codes & config equality")
    A("")
    for s in series:
        A(f"- `{s['experiment_code']}` — {s['representation']} FULL_MINUS_{s['family_id']} (baseline `{s['full_baseline']}`)")
    A("")
    A("## 7–11. Scores, contrasts, bootstrap, evidence classes")
    A("")
    A("### Scores (selected)")
    A("")
    A("| Rep | Arm | CV_P | Public | Private | Test |")
    A("|-----|-----|------|--------|---------|------|")
    for _, r in scores_df.iterrows():
        A(f"| {r.representation} | {r.arm} | {r.cv_primary:.6f} | {r.public_mae:.6f} | {r.private_mae:.6f} | {r.test_overall_mae:.6f} |")
    A("")
    A("### Contrasts Δ = FULL_MINUS_F − FULL (positive = family unique value)")
    A("")
    A("| Family | Rep | ΔCV_P | ΔCV_S | ΔPub | ΔPriv | ΔTest |")
    A("|--------|-----|-------|-------|------|-------|-------|")
    for fam in tax["family_order"]:
        for rep in ("ablang1", "ablingua"):
            def g(m):
                return float(
                    contrasts_df[
                        (contrasts_df.family_id == fam)
                        & (contrasts_df.representation == rep)
                        & (contrasts_df.metric == m)
                    ].iloc[0].delta
                )

            A(f"| {fam} | {rep} | {g('cv_primary'):+.6f} | {g('cv_shadow'):+.6f} | {g('public'):+.6f} | {g('private'):+.6f} | {g('test'):+.6f} |")
    A("")
    A("### Evidence class per family")
    A("")
    for _, r in class_df.iterrows():
        A(
            f"- **`{r.family_id}`** → `{r.evidence_class}` "
            f"(AbLang1 ΔTest={r.ablang1_delta_test:+.4f}, AbLingua ΔTest={r.ablingua_delta_test:+.4f})"
        )
    A("")
    A("### Bootstrap (AE_MINUS − AE_FULL; N=10000, seed=101)")
    A("")
    A("See `HIC_SURFACE_FAMILY_LOFO_BOOTSTRAP.csv`.")
    A("")
    A("## 12. HIGH-tail secondary (HIC>11.5, n=7)")
    A("")
    for fam in tax["family_order"]:
        parts = []
        for rep in ("ablang1", "ablingua"):
            row = high_df[(high_df.family_id == fam) & (high_df.representation == rep) & (high_df.subset == "HIGH")].iloc[0]
            parts.append(f"{rep} ΔMAE_high={row.delta_mae_vs_full:+.4f}")
        A(f"- {fam}: " + "; ".join(parts))
    A("")
    A("## 13. Sample-level heterogeneity")
    A("")
    A("See `HIC_SURFACE_FAMILY_LOFO_HETEROGENEITY.csv` (worsen fractions + cross-rep agreement).")
    A("")
    A("## 14–15. ARO vs HYDRO synthesis & global interpretation")
    A("")
    A(f"- ARO directional/robust contributors: {aro_contrib or 'none'}")
    A(f"- HYDRO directional/robust contributors: {hydro_contrib or 'none'}")
    A(f"- **Global interpretation: `{ginterp}`**")
    A("")
    A("## 16. Exposed-aromatic hypothesis update")
    A("")
    A(aro_status)
    A("")
    A("## 17. Next scientific recommendation")
    A("")
    A(next_rec)
    A("")
    A("## Non-claims")
    A("")
    A("Family contribution is predictive conditional information only — not biophysical causality.")
    A("")
    A("## Stop")
    A("")
    A("Family-level LOFO complete. No 35-way / individual-feature search in this phase.")
    A("")

    (REPORTS / "HIC_SURFACE_FAMILY_LOFO_RESULTS.md").write_text("\n".join(lines) + "\n")
    (REPORTS / "HIC_SURFACE_FAMILY_LOFO_SUMMARY.json").write_text(
        json.dumps(
            {
                "prereg_sha": PREREG_SHA,
                "n_families": len(tax["family_order"]),
                "evidence_classes": class_map,
                "global_interpretation": ginterp,
                "aromatic_hypothesis_status": aro_status,
                "next_recommendation": next_rec,
                "series": series,
            },
            indent=2,
        )
        + "\n"
    )
    print("GLOBAL", ginterp, flush=True)
    print("CLASSES", class_map, flush=True)
    print("REPORT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
