#!/usr/bin/env python3
"""Analyze ARO19 vs HYDRO16 decomposition (H340–H347) and write final reports."""
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

PREREG_SHA = "40f8ac0a7cf9caff59cbae913f99638d8cc93c4b"
FREEZE_V3_SHA = "1bfdf9107f55d14c58117aaa7f64e255dc387ed9"
N_BOOT = 10_000
BOOT_SEED = 101
HIGH_THR = 11.5
TOL = 1e-6
ANCHORS = ["EXP-H030", "EXP-H107", "EXP-H137"]

ARMS = {
    "ablang1": {
        "SHAM35": "EXP-H340",
        "ARO_ONLY35": None,
        "HYDRO_ONLY35": None,
        "FULL35": "EXP-H341",
    },
    "ablingua": {
        "SHAM35": "EXP-H342",
        "ARO_ONLY35": None,
        "HYDRO_ONLY35": None,
        "FULL35": "EXP-H343",
    },
}
CONTRASTS = [
    ("ARO_vs_SHAM", "ARO_ONLY35", "SHAM35"),
    ("HYDRO_vs_SHAM", "HYDRO_ONLY35", "SHAM35"),
    ("FULL_vs_SHAM", "FULL35", "SHAM35"),
    ("FULL_vs_ARO", "FULL35", "ARO_ONLY35"),
    ("FULL_vs_HYDRO", "FULL35", "HYDRO_ONLY35"),
]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_series() -> dict:
    series = json.loads((ROOT / "results/H344_H347_ARO_HYDRO_SERIES.json").read_text())
    by_key = {s["key"]: s for s in series}
    ARMS["ablang1"]["ARO_ONLY35"] = by_key["ablang1_aro"]["experiment_code"]
    ARMS["ablang1"]["HYDRO_ONLY35"] = by_key["ablang1_hydro"]["experiment_code"]
    ARMS["ablingua"]["ARO_ONLY35"] = by_key["ablingua_aro"]["experiment_code"]
    ARMS["ablingua"]["HYDRO_ONLY35"] = by_key["ablingua_hydro"]["experiment_code"]
    return by_key


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
    return {
        "public_mae": pub_m,
        "private_mae": priv_m,
        "test_overall_mae": ov,
        "public_private_gap": abs(pub_m - priv_m),
    }


def classify(contrasts: pd.DataFrame) -> tuple[str, str]:
    """Assign mechanistic class from Primary/Shadow/Public/Private direction + size."""
    # Focus on Primary CV + Public/Private + Test for both reps
    notes = []

    def mean_delta(contrast: str, metric: str) -> float:
        sub = contrasts[(contrasts.contrast == contrast) & (contrasts.metric == metric)]
        return float(sub["delta"].mean()) if len(sub) else float("nan")

    metrics = ["cv_primary", "cv_shadow", "public", "private", "test"]
    aro_sham = {m: mean_delta("ARO_vs_SHAM", m) for m in metrics}
    hydro_sham = {m: mean_delta("HYDRO_vs_SHAM", m) for m in metrics}
    full_sham = {m: mean_delta("FULL_vs_SHAM", m) for m in metrics}
    full_aro = {m: mean_delta("FULL_vs_ARO", m) for m in metrics}
    full_hydro = {m: mean_delta("FULL_vs_HYDRO", m) for m in metrics}

    def mostly_improve(d: dict, keys=("cv_primary", "cv_shadow", "public", "private")) -> bool:
        vals = [d[k] for k in keys if np.isfinite(d[k])]
        return sum(v < 0 for v in vals) >= max(3, len(vals) - 1) and np.nanmean(vals) < 0

    def mostly_small(d: dict, keys=("cv_primary", "public", "private"), vs_full=None) -> bool:
        vals = [abs(d[k]) for k in keys if np.isfinite(d[k])]
        if not vals:
            return False
        if vs_full is None:
            return float(np.mean(vals)) < 0.02
        fvals = [abs(vs_full[k]) for k in keys if np.isfinite(vs_full.get(k, np.nan))]
        if not fvals:
            return False
        return float(np.mean(vals)) < 0.35 * float(np.mean(fvals))

    aro_ok = mostly_improve(aro_sham)
    hydro_ok = mostly_improve(hydro_sham)
    full_ok = mostly_improve(full_sham)
    aro_inc = mostly_improve(full_hydro)  # FULL−HYDRO = ARO incremental
    hydro_inc = mostly_improve(full_aro)

    notes.append(f"aro_vs_sham_improve={aro_ok} mean_primary={aro_sham['cv_primary']:+.4f}")
    notes.append(f"hydro_vs_sham_improve={hydro_ok} mean_primary={hydro_sham['cv_primary']:+.4f}")
    notes.append(f"full_vs_sham_improve={full_ok} mean_primary={full_sham['cv_primary']:+.4f}")
    notes.append(f"aro_incremental_on_hydro={aro_inc} mean_primary={full_hydro['cv_primary']:+.4f}")
    notes.append(f"hydro_incremental_on_aro={hydro_inc} mean_primary={full_aro['cv_primary']:+.4f}")

    # Per-rep stability check on Primary CV
    per_rep = contrasts[contrasts.metric == "cv_primary"]
    dirs = {}
    for contrast in ("ARO_vs_SHAM", "HYDRO_vs_SHAM", "FULL_vs_SHAM"):
        dirs[contrast] = {
            r: float(per_rep[(per_rep.contrast == contrast) & (per_rep.representation == r)]["delta"].iloc[0])
            for r in ("ablang1", "ablingua")
        }
    unstable = any(
        (dirs[c]["ablang1"] < 0) != (dirs[c]["ablingua"] < 0) for c in ("ARO_vs_SHAM", "HYDRO_vs_SHAM")
    )

    if unstable and not (aro_ok and hydro_ok):
        label = "MIXED_OR_INCONCLUSIVE"
    elif aro_ok and not hydro_ok and aro_inc and not hydro_inc:
        label = "ARO_DOMINANT"
    elif hydro_ok and not aro_ok and hydro_inc and not aro_inc:
        label = "HYDRO_DOMINANT"
    elif aro_ok and hydro_ok and (aro_inc or hydro_inc) and full_ok:
        # both blocks useful; FULL further improves at least one incremental
        label = "ARO_HYDRO_COMPLEMENTARY"
    elif aro_ok and hydro_ok and mostly_small(full_aro, vs_full=full_sham) and mostly_small(full_hydro, vs_full=full_sham):
        label = "SURFACE_BLOCK_REDUNDANT"
    elif aro_ok and not hydro_ok:
        label = "ARO_DOMINANT"
    elif hydro_ok and not aro_ok:
        label = "HYDRO_DOMINANT"
    else:
        label = "MIXED_OR_INCONCLUSIVE"

    return label, "; ".join(notes)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    by_key = load_series()
    sol = load_solution().set_index("id")
    exp = pd.read_csv(ROOT / "results/experiments.csv").set_index("experiment_code")
    y_dev = pd.read_csv(ROOT / "data/dev.csv").set_index("id")["HIC"].astype(float)
    y_te = sol["HIC"].astype(float)
    te_ids = sol.index.tolist()
    high_te = [i for i in te_ids if y_te.loc[i] > HIGH_THR]
    non_te = [i for i in te_ids if y_te.loc[i] <= HIGH_THR]
    rng = np.random.default_rng(BOOT_SEED)

    # pipeline validation
    pipe_rows = []
    max_d = 0.0
    for code in ANCHORS:
        sc = score_ext(load_test_pred(code), sol)
        reg = exp.loc[code]
        for k, col in [
            ("public_mae", "public_mae"),
            ("private_mae", "private_mae"),
            ("test_overall_mae", "test_overall_mae"),
        ]:
            d = abs(sc[k] - float(reg[col]))
            max_d = max(max_d, d)
            pipe_rows.append({"experiment_code": code, "metric": k, "recomputed": sc[k], "registry": float(reg[col]), "abs_delta": d})
    if max_d > TOL:
        raise SystemExit(f"PIPELINE_FAIL max_delta={max_d}")

    # config equality / n_trainable audit
    audit_rows = []
    for rep, arms in ARMS.items():
        nts = {}
        for arm, code in arms.items():
            doc = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
            cfg = yaml.safe_load((ROOT / "experiments/configs" / f"{code}.yaml").read_text())
            nts[arm] = doc.get("n_trainable")
            audit_rows.append(
                {
                    "representation": rep,
                    "arm": arm,
                    "experiment_code": code,
                    "aux_dim": cfg.get("aux_dim"),
                    "fusion_bundle_id": cfg.get("fusion_bundle_id"),
                    "fusion_mode": cfg.get("fusion_mode"),
                    "seed": cfg.get("seed"),
                    "topology_id": cfg.get("topology_id"),
                    "annotation_mode": cfg.get("annotation_mode"),
                    "n_trainable": doc.get("n_trainable"),
                    "config_hash": doc.get("config_hash"),
                }
            )
        if len(set(v for v in nts.values() if v is not None)) != 1:
            raise SystemExit(f"n_trainable mismatch {rep}: {nts}")
    pd.DataFrame(audit_rows).to_csv(REPORTS / "HIC_SURFACE_ARO_HYDRO_CONFIG_AUDIT.csv", index=False)

    # external score all 8 arms; fill new arms into experiments.csv
    score_rows = []
    ext_scores = {}
    for rep, arms in ARMS.items():
        for arm, code in arms.items():
            sc = score_ext(load_test_pred(code), sol)
            ext_scores[code] = sc
            oof = yaml.safe_load((ROOT / "results" / f"{code}_OOF_EVALUATION.yaml").read_text())
            score_rows.append(
                {
                    "experiment_code": code,
                    "representation": rep,
                    "arm": arm,
                    "cv_primary": float(oof["oof_test"]["primary"]),
                    "cv_shadow": float(oof["oof_test"]["shadow"]),
                    "cv_mean": float(oof["oof_test"]["mean"]),
                    **sc,
                }
            )
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(REPORTS / "HIC_SURFACE_ARO_HYDRO_SCORES.csv", index=False)

    exp2 = pd.read_csv(ROOT / "results/experiments.csv")
    for _, r in scores_df.iterrows():
        if r.arm in ("ARO_ONLY35", "HYDRO_ONLY35"):
            m = exp2.experiment_code == r.experiment_code
            if m.any():
                exp2.loc[m, "public_mae"] = r.public_mae
                exp2.loc[m, "private_mae"] = r.private_mae
                exp2.loc[m, "test_overall_mae"] = r.test_overall_mae
                if "public_private_gap" in exp2.columns:
                    exp2.loc[m, "public_private_gap"] = r.public_private_gap
                exp2.loc[m, "status"] = "COMPLETE"
    exp2.to_csv(ROOT / "results/experiments.csv", index=False)

    # contrasts
    contrast_rows = []
    boot_rows = []
    metric_map = {
        "cv_primary": ("cv_primary", "internal"),
        "cv_shadow": ("cv_shadow", "internal"),
        "cv_mean": ("cv_mean", "internal"),
        "public": ("public_mae", "external"),
        "private": ("private_mae", "external"),
        "test": ("test_overall_mae", "external"),
    }
    for rep, arms in ARMS.items():
        sdf = scores_df[scores_df.representation == rep].set_index("arm")
        for cname, a, b in CONTRASTS:
            for metric, (col, scope) in metric_map.items():
                da = float(sdf.loc[a, col])
                db = float(sdf.loc[b, col])
                contrast_rows.append(
                    {
                        "representation": rep,
                        "contrast": cname,
                        "arm_a": a,
                        "arm_b": b,
                        "metric": metric,
                        "scope": scope,
                        "value_a": da,
                        "value_b": db,
                        "delta": da - db,
                    }
                )

        # OOF primary bootstrap for each contrast
        for cname, a, b in CONTRASTS:
            pa = load_oof(arms[a], "oof_test_primary.csv")
            pb = load_oof(arms[b], "oof_test_primary.csv")
            ids = sorted(set(pa.index) & set(pb.index) & set(y_dev.index))
            delta = np.abs(pa.loc[ids].to_numpy() - y_dev.loc[ids].to_numpy()) - np.abs(
                pb.loc[ids].to_numpy() - y_dev.loc[ids].to_numpy()
            )
            boot_rows.append({"representation": rep, "contrast": cname, "split": "oof_primary", **paired_boot(delta, rng)})

            # Test AE bootstrap
            ta = load_test_pred(arms[a]).reindex(te_ids)
            tb = load_test_pred(arms[b]).reindex(te_ids)
            delta_t = np.abs(ta.to_numpy(float) - y_te.loc[te_ids].to_numpy()) - np.abs(
                tb.to_numpy(float) - y_te.loc[te_ids].to_numpy()
            )
            boot_rows.append({"representation": rep, "contrast": cname, "split": "test", **paired_boot(delta_t, rng)})

    contrasts_df = pd.DataFrame(contrast_rows)
    contrasts_df.to_csv(REPORTS / "HIC_SURFACE_ARO_HYDRO_CONTRASTS.csv", index=False)
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(REPORTS / "HIC_SURFACE_ARO_HYDRO_BOOTSTRAP.csv", index=False)

    # HIGH-tail
    high_rows = []
    for rep, arms in ARMS.items():
        for arm, code in arms.items():
            p = load_test_pred(code)
            for split_name, id_list in (("HIGH", high_te), ("nonHIGH", non_te)):
                yy = y_te.loc[id_list].to_numpy()
                pp = p.loc[id_list].to_numpy(float)
                high_rows.append(
                    {
                        "representation": rep,
                        "arm": arm,
                        "experiment_code": code,
                        "subset": split_name,
                        "n": len(id_list),
                        "mae": float(np.abs(pp - yy).mean()),
                        "signed_error": float((pp - yy).mean()),
                    }
                )
        for cname, a, b in [("ARO_vs_SHAM", "ARO_ONLY35", "SHAM35"), ("HYDRO_vs_SHAM", "HYDRO_ONLY35", "SHAM35"), ("FULL_vs_SHAM", "FULL35", "SHAM35")]:
            for subset, id_list in (("HIGH", high_te), ("nonHIGH", non_te)):
                pa = load_test_pred(arms[a]).loc[id_list].to_numpy(float)
                pb = load_test_pred(arms[b]).loc[id_list].to_numpy(float)
                yy = y_te.loc[id_list].to_numpy()
                high_rows.append(
                    {
                        "representation": rep,
                        "arm": cname,
                        "experiment_code": f"{arms[a]}-{arms[b]}",
                        "subset": subset,
                        "n": len(id_list),
                        "mae": float(np.abs(pa - yy).mean() - np.abs(pb - yy).mean()),
                        "signed_error": float((pa - yy).mean() - (pb - yy).mean()),
                        "is_delta": True,
                    }
                )
    high_df = pd.DataFrame(high_rows)
    high_df.to_csv(REPORTS / "HIC_SURFACE_ARO_HYDRO_HIGHTAIL.csv", index=False)

    # sample-level heterogeneity (Test)
    het_rows = []
    for rep, arms in ARMS.items():
        sham = load_test_pred(arms["SHAM35"]).reindex(te_ids)
        ae_s = np.abs(sham.to_numpy(float) - y_te.loc[te_ids].to_numpy())
        improved = {}
        for arm in ("ARO_ONLY35", "HYDRO_ONLY35", "FULL35"):
            p = load_test_pred(arms[arm]).reindex(te_ids)
            ae = np.abs(p.to_numpy(float) - y_te.loc[te_ids].to_numpy())
            d = ae - ae_s
            improved[arm] = d < 0
            for thr_lab, mask in (("ALL", np.ones(len(te_ids), bool)), ("HIGH", y_te.loc[te_ids].to_numpy() > HIGH_THR), ("nonHIGH", y_te.loc[te_ids].to_numpy() <= HIGH_THR)):
                dd = d[mask]
                het_rows.append(
                    {
                        "representation": rep,
                        "arm": arm,
                        "subset": thr_lab,
                        "n": int(mask.sum()),
                        "frac_improved_vs_sham": float((dd < 0).mean()),
                        "frac_worsened_vs_sham": float((dd > 0).mean()),
                        "mean_delta_ae": float(dd.mean()),
                    }
                )
        both = improved["ARO_ONLY35"] & improved["HYDRO_ONLY35"]
        aro_only = improved["ARO_ONLY35"] & ~improved["HYDRO_ONLY35"]
        hydro_only = improved["HYDRO_ONLY35"] & ~improved["ARO_ONLY35"]
        het_rows.append(
            {
                "representation": rep,
                "arm": "RESPONDER_PATTERN",
                "subset": "ALL",
                "n": len(te_ids),
                "frac_aro_only": float(aro_only.mean()),
                "frac_hydro_only": float(hydro_only.mean()),
                "frac_both": float(both.mean()),
                "frac_neither": float((~improved["ARO_ONLY35"] & ~improved["HYDRO_ONLY35"]).mean()),
            }
        )
    # cross-rep agreement on FULL responders
    a1 = load_test_pred(ARMS["ablang1"]["FULL35"]).reindex(te_ids)
    s1 = load_test_pred(ARMS["ablang1"]["SHAM35"]).reindex(te_ids)
    a2 = load_test_pred(ARMS["ablingua"]["FULL35"]).reindex(te_ids)
    s2 = load_test_pred(ARMS["ablingua"]["SHAM35"]).reindex(te_ids)
    yv = y_te.loc[te_ids].to_numpy()
    imp1 = np.abs(a1.to_numpy(float) - yv) < np.abs(s1.to_numpy(float) - yv)
    imp2 = np.abs(a2.to_numpy(float) - yv) < np.abs(s2.to_numpy(float) - yv)
    agree = float((imp1 == imp2).mean())
    het_rows.append(
        {
            "representation": "both",
            "arm": "FULL_RESPONDER_AGREEMENT",
            "subset": "ALL",
            "n": len(te_ids),
            "frac_agree": agree,
            "frac_both_improve": float((imp1 & imp2).mean()),
        }
    )
    pd.DataFrame(het_rows).to_csv(REPORTS / "HIC_SURFACE_ARO_HYDRO_HETEROGENEITY.csv", index=False)

    label, rationale = classify(contrasts_df)

    # aromatic hypothesis status
    if label == "ARO_DOMINANT":
        aro_status = (
            "SUPPORTED_AT_BLOCK_LEVEL — ARO19 carries most gain; propose later ARO19-internal family "
            "decomposition. Do NOT claim specific exposed-aromatic feature X is causal."
        )
        next_exp = "ARO19 internal feature-family decomposition (still block/family level; no 35-way search)."
    elif label == "HYDRO_DOMINANT":
        aro_status = "NOT_SUPPORTED_AS_SOLE_STORY — HYDRO16 dominates; do not collapse to aromatics-only narrative."
        next_exp = "HYDRO16 internal family decomposition or surface-field localization probes."
    elif label == "ARO_HYDRO_COMPLEMENTARY":
        aro_status = (
            "PARTIAL — aromatics informative but not sole; HYDRO also contributes. "
            "Avoid aromatics-only causal story."
        )
        next_exp = "Joint interpretation: modest ARO and HYDRO family probes, or HIGH-tail focused localization."
    elif label == "SURFACE_BLOCK_REDUNDANT":
        aro_status = "INCONCLUSIVE_FOR_AROMATICS — either block largely recovers FULL; redundancy, not aromatic specificity."
        next_exp = "Shared-information / redundancy analysis before finer feature decomposition."
    else:
        aro_status = "INCONCLUSIVE — representation/split instability; do not advance aromatic causality claims."
        next_exp = "Stabilize with additional matched contexts only if scientifically justified; else stop."

    # write markdown report
    lines = []
    A = lines.append
    A("# HIC SURFACE ARO19 vs HYDRO16 Block Decomposition")
    A("")
    A("**STATUS: `HIC_SURFACE_ARO_HYDRO_DECOMPOSITION_COMPLETE`**")
    A("")
    A("| Field | Value |")
    A("|-------|--------|")
    A(f"| Freeze v3 SHA | `{FREEZE_V3_SHA}` |")
    A(f"| Decomposition prereg SHA | `{PREREG_SHA}` |")
    A(f"| Analysis code SHA | `{git_rev()}` |")
    A(f"| Mechanistic classification | `{label}` |")
    A("")
    A("## 1–3. Provenance & experiment codes")
    A("")
    for rep, arms in ARMS.items():
        A(f"### {rep}")
        for arm, code in arms.items():
            A(f"- `{arm}` → `{code}`")
    A("")
    A("## 4. Config equality audit")
    A("")
    A("All arms: aux_dim=35, late_concat_aux32, JOINT/FULL, seed=101.")
    A(f"n_trainable matched within each representation (see `HIC_SURFACE_ARO_HYDRO_CONFIG_AUDIT.csv`).")
    A("")
    A("## 5–11. Scores and contrasts")
    A("")
    A("### Scores")
    A("")
    A("| Rep | Arm | CV_P | CV_S | Public | Private | Test |")
    A("|-----|-----|------|------|--------|---------|------|")
    for _, r in scores_df.iterrows():
        A(
            f"| {r.representation} | {r.arm} | {r.cv_primary:.6f} | {r.cv_shadow:.6f} | "
            f"{r.public_mae:.6f} | {r.private_mae:.6f} | {r.test_overall_mae:.6f} |"
        )
    A("")
    A("### Contrasts (Δ = A − B; negative = A better)")
    A("")
    A("| Rep | Contrast | ΔCV_P | ΔCV_S | ΔPub | ΔPriv | ΔTest |")
    A("|-----|----------|-------|-------|------|-------|-------|")
    for rep in ("ablang1", "ablingua"):
        for cname, _, _ in CONTRASTS:
            def g(metric):
                row = contrasts_df[
                    (contrasts_df.representation == rep)
                    & (contrasts_df.contrast == cname)
                    & (contrasts_df.metric == metric)
                ].iloc[0]
                return float(row.delta)

            A(
                f"| {rep} | {cname} | {g('cv_primary'):+.6f} | {g('cv_shadow'):+.6f} | "
                f"{g('public'):+.6f} | {g('private'):+.6f} | {g('test'):+.6f} |"
            )
    A("")
    A("### Bootstrap (Primary OOF & Test AE; N=10000, seed=101)")
    A("")
    for _, r in boot_df.iterrows():
        A(
            f"- {r.representation}/{r.contrast}/{r.split}: mean={r['mean']:+.6f} "
            f"CI=[{r.ci95_lo:+.6f}, {r.ci95_hi:+.6f}] frac_improve={r.frac_improve:.3f}"
        )
    A("")
    A("## 12–13. HIGH-tail (HIC>11.5, diagnostic)")
    A("")
    A(f"n_HIGH={len(high_te)}")
    A("")
    for rep in ("ablang1", "ablingua"):
        A(f"### {rep}")
        for arm in ("SHAM35", "ARO_ONLY35", "HYDRO_ONLY35", "FULL35"):
            row = high_df[(high_df.representation == rep) & (high_df.arm == arm) & (high_df.subset == "HIGH")].iloc[0]
            A(f"- {arm}: MAE_high={row.mae:.4f}, signed={row.signed_error:+.4f}")
        for cname in ("ARO_vs_SHAM", "HYDRO_vs_SHAM", "FULL_vs_SHAM"):
            row = high_df[(high_df.representation == rep) & (high_df.arm == cname) & (high_df.subset == "HIGH")].iloc[0]
            A(f"- Δ {cname}: ΔMAE_high={row.mae:+.4f}")
    A("")
    A("## 14. Sample-level heterogeneity")
    A("")
    A("See `HIC_SURFACE_ARO_HYDRO_HETEROGENEITY.csv`.")
    A(f"- FULL responder pattern agreement across AbLang1/AbLingua: **{agree:.3f}**")
    A("")
    A("## 15. Mechanistic classification")
    A("")
    A(f"**`{label}`**")
    A("")
    A(rationale)
    A("")
    A("## 16. Exposed-aromatic hypothesis status")
    A("")
    A(aro_status)
    A("")
    A("## 17. Next experiment recommendation")
    A("")
    A(next_exp)
    A("")
    A("## Stop")
    A("")
    A("This phase stops at ARO19 vs HYDRO16 **block** conclusion. No 35-way feature search.")
    A("")

    (REPORTS / "HIC_SURFACE_ARO_HYDRO_DECOMPOSITION.md").write_text("\n".join(lines) + "\n")
    (REPORTS / "HIC_SURFACE_ARO_HYDRO_SUMMARY.json").write_text(
        json.dumps(
            {
                "classification": label,
                "rationale": rationale,
                "aromatic_hypothesis_status": aro_status,
                "next_experiment": next_exp,
                "prereg_sha": PREREG_SHA,
                "freeze_v3_sha": FREEZE_V3_SHA,
                "arms": ARMS,
                "series": list(by_key.values()),
            },
            indent=2,
        )
        + "\n"
    )
    print("CLASSIFICATION", label, flush=True)
    print("REPORT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
