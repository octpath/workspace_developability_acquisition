#!/usr/bin/env python3
"""Internal analysis for HIC Representation × Topology × Annotation factorial (H140–H339).

No Public/Private / GEN_0001 inspection.
"""
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

from run_exp_t151_t156_plm_topology import load_oof, paired_boot  # noqa: E402

SEED = 101
N_BOOT = 2000
HIGH_THR = 11.5
TOPOS = ("SEP", "JOINT", "REG-SEP", "XREG", "FUSE")
ANNOTS = ("BASE", "IMGT", "REGION", "FULL")
PLAN_CSV = ROOT / "results/HIC_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv"
OUT_DIR = ROOT / "results/h140_h339_analysis"
REPORTS = REPO / "reports"


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


def load_plan() -> pd.DataFrame:
    return pd.read_csv(PLAN_CSV)


def scores_for_row(row, exp: pd.DataFrame) -> dict:
    code = str(row["experiment_code"])
    st = str(row.get("execution_status") or "")
    out = {
        "experiment_code": code,
        "representation": row["representation"],
        "representation_id": row.get("representation_id", row["representation"]),
        "plm_family": row.get("plm_family"),
        "representation_context": row.get("representation_context"),
        "paired_match_group": row.get("paired_match_group"),
        "topology": row["topology"],
        "annotation": row["annotation"],
        "execution_status": st,
        "TEST_P": np.nan,
        "TEST_S": np.nan,
        "TEST_mean": np.nan,
        "TEST_worst": np.nan,
        "abs_PS": np.nan,
    }
    yml = ROOT / f"results/{code}_OOF_EVALUATION.yaml"
    if yml.exists():
        doc = yaml.safe_load(yml.read_text())
        if doc.get("technical_smoke") or doc.get("quick"):
            return out
        ot = doc.get("oof_test") or {}
        p, s = float(ot["primary"]), float(ot["shadow"])
        out.update(
            TEST_P=p,
            TEST_S=s,
            TEST_mean=float(ot.get("mean", (p + s) / 2)),
            TEST_worst=float(ot.get("worst", max(p, s))),
            abs_PS=abs(p - s),
            execution_status="COMPLETE",
        )
        return out
    m = exp[exp.experiment_code == code]
    if len(m) == 1 and str(m.iloc[0].get("artifact_status")) == "FULL":
        r = m.iloc[0]
        p, s = float(r.cv_primary_mae), float(r.cv_shadow_mae)
        out.update(
            TEST_P=p,
            TEST_S=s,
            TEST_mean=float(r.cv_mean_mae),
            TEST_worst=float(r.cv_worst_mae),
            abs_PS=abs(p - s),
            execution_status="COMPLETE",
        )
    return out


def high_tail_diag(code: str, ymap: dict, thr: float = HIGH_THR) -> dict:
    pred_path = ROOT / "experiments/predictions" / code / "oof_primary.csv"
    out = {
        "experiment_code": code,
        "n_high": 0,
        "mae_high": np.nan,
        "mean_signed_error": np.nan,
        "obs_min": np.nan,
        "obs_max": np.nan,
        "pred_min": np.nan,
        "pred_max": np.nan,
    }
    if not pred_path.exists():
        return out
    pred = pd.read_csv(pred_path)
    ids = pred["id"].astype(str).tolist()
    y = np.asarray([ymap[i] for i in ids], float)
    p = pred["HIC"].to_numpy(float)
    mask = y > thr
    out["n_high"] = int(mask.sum())
    if out["n_high"] == 0:
        return out
    yh, ph = y[mask], p[mask]
    out["mae_high"] = float(np.mean(np.abs(ph - yh)))
    out["mean_signed_error"] = float(np.mean(ph - yh))
    out["obs_min"] = float(yh.min())
    out["obs_max"] = float(yh.max())
    out["pred_min"] = float(ph.min())
    out["pred_max"] = float(ph.max())
    return out


def completion_audit(res: pd.DataFrame, ymap: dict) -> dict:
    errs = []
    n_complete = int((res.execution_status == "COMPLETE").sum())
    st = pd.read_csv(REPORTS / "HIC_H140_H339_EXECUTION_STATUS.csv") if (REPORTS / "HIC_H140_H339_EXECUTION_STATUS.csv").exists() else None
    n_fail = int((st.status == "FAILED_TECHNICAL").sum()) if st is not None else 0
    n_block = int((st.status == "BLOCKED").sum()) if st is not None else 0
    if n_complete != 200:
        errs.append(f"COMPLETE={n_complete} expected 200")
    if n_fail:
        errs.append(f"FAILED_TECHNICAL={n_fail}")
    if n_block:
        errs.append(f"BLOCKED={n_block}")
    for _, r in res.iterrows():
        if r.execution_status != "COMPLETE":
            continue
        code = r.experiment_code
        for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
            p = ROOT / "experiments/predictions" / code / name
            if not p.exists():
                errs.append(f"{code} missing {name}")
                continue
            df = pd.read_csv(p)
            if list(df.columns) != ["id", "HIC"]:
                errs.append(f"{code} {name} columns")
            if name.startswith("oof_") and set(df.id.astype(str)) != set(ymap):
                # oof is Dev only
                if not set(df.id.astype(str)).issubset(set(ymap)):
                    errs.append(f"{code} {name} id mismatch")
            vals = df["HIC"].to_numpy(float)
            if not np.all(np.isfinite(vals)):
                errs.append(f"{code} {name} nonfinite")
            if name.startswith("oof_") and float(np.nanvar(vals)) <= 0:
                errs.append(f"{code} {name} zero var")
        cfg = yaml.safe_load((ROOT / f"experiments/configs/{code}.yaml").read_text())
        if cfg.get("target") != "HIC" or cfg.get("share_hl_encoder") is not True:
            errs.append(f"{code} config mismatch")
        oof = yaml.safe_load((ROOT / f"results/{code}_OOF_EVALUATION.yaml").read_text())
        if oof.get("public_private_inspected"):
            errs.append(f"{code} pub/priv inspected")
        if oof.get("quick") or oof.get("technical_smoke"):
            errs.append(f"{code} smoke/quick artifact counted as complete")
    return {
        "status": "PASS" if not errs else "FAIL",
        "n_complete": n_complete,
        "n_failed": n_fail,
        "n_blocked": n_block,
        "errors": errs[:100],
        "n_errors": len(errs),
        "code_sha": git_rev(),
    }


def contrast_boot(code_x: str, code_a: str, scheme: str, ymap: dict):
    px = load_oof(code_x, scheme)
    pa = load_oof(code_a, scheme)
    # load_oof returns series indexed by id; for HIC column may be target-named
    if isinstance(px, pd.DataFrame):
        px = px.set_index("id")["HIC"] if "HIC" in px.columns else px.iloc[:, -1]
    if isinstance(pa, pd.DataFrame):
        pa = pa.set_index("id")["HIC"] if "HIC" in pa.columns else pa.iloc[:, -1]
    m = px.to_frame("x").join(pa.to_frame("a"), how="inner")
    y = np.asarray([ymap[str(i)] for i in m.index], float)
    return paired_boot(m["x"].to_numpy(float), m["a"].to_numpy(float), y)


def run_analysis() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = load_plan()
    assert len(plan) == 200
    exp = pd.read_csv(ROOT / "results/experiments.csv")
    dev = pd.read_csv(ROOT / "data/dev.csv")
    ymap = {str(r["id"]): float(r["HIC"]) for _, r in dev.iterrows()}

    res = pd.DataFrame([scores_for_row(r, exp) for _, r in plan.iterrows()])
    res_path = ROOT / "results/HIC_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv"
    res.to_csv(res_path, index=False)

    audit = completion_audit(res, ymap)
    (OUT_DIR / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    audit_md = [
        "# HIC H140–H339 completion audit",
        "",
        f"**STATUS: {audit['status']}**",
        "",
        f"- COMPLETE: {audit['n_complete']}",
        f"- FAILED_TECHNICAL: {audit['n_failed']}",
        f"- BLOCKED: {audit['n_blocked']}",
        f"- code_sha: `{audit['code_sha']}`",
        f"- Public/Private consulted: **No**",
        "",
    ]
    if audit["errors"]:
        audit_md.append("## Errors")
        audit_md.extend(f"- {e}" for e in audit["errors"][:50])
    (REPORTS / "HIC_H140_H339_COMPLETION_AUDIT.md").write_text("\n".join(audit_md) + "\n")
    if audit["status"] != "PASS":
        print("COMPLETION_AUDIT FAIL", audit["n_errors"], flush=True)
        return 1

    done = res[res.execution_status == "COMPLETE"].copy()

    # HIGH-tail
    high_rows = [high_tail_diag(c, ymap) for c in done.experiment_code]
    high = pd.DataFrame(high_rows)
    high = high.merge(done[["experiment_code", "representation", "topology", "annotation", "TEST_mean"]], on="experiment_code")
    high.to_csv(ROOT / "results/HIC_REP_TOPO_ANNOT_HIGH_TAIL.csv", index=False)

    boot_rows = []
    # Topology vs SEP
    topo_rows = []
    for rep in done.representation.unique():
        for ann in ANNOTS:
            base = done[(done.representation == rep) & (done.annotation == ann) & (done.topology == "SEP")]
            if len(base) != 1:
                continue
            a = base.iloc[0]
            for topo in ("JOINT", "REG-SEP", "XREG", "FUSE"):
                x = done[(done.representation == rep) & (done.annotation == ann) & (done.topology == topo)]
                if len(x) != 1:
                    continue
                xr = x.iloc[0]
                row = {
                    "representation": rep,
                    "annotation": ann,
                    "topology": topo,
                    "delta_P": xr.TEST_P - a.TEST_P,
                    "delta_S": xr.TEST_S - a.TEST_S,
                    "delta_mean": xr.TEST_mean - a.TEST_mean,
                    "both_improve": (xr.TEST_P - a.TEST_P) < 0 and (xr.TEST_S - a.TEST_S) < 0,
                    "code_x": xr.experiment_code,
                    "code_sep": a.experiment_code,
                }
                topo_rows.append(row)
                for scheme in ("primary", "shadow"):
                    d, lo, hi = contrast_boot(xr.experiment_code, a.experiment_code, scheme, ymap)
                    boot_rows.append(
                        {
                            "kind": "topology_vs_SEP",
                            "representation": rep,
                            "annotation": ann,
                            "topology": topo,
                            "scheme": scheme,
                            "delta_mae": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "code_x": xr.experiment_code,
                            "code_ref": a.experiment_code,
                        }
                    )
    topo_gains = pd.DataFrame(topo_rows)
    topo_gains.to_csv(ROOT / "results/HIC_REP_TOPO_ANNOT_TOPOLOGY_GAINS.csv", index=False)

    # Annotation vs BASE
    annot_rows = []
    for rep in done.representation.unique():
        for topo in TOPOS:
            base = done[(done.representation == rep) & (done.topology == topo) & (done.annotation == "BASE")]
            if len(base) != 1:
                continue
            b = base.iloc[0]
            for ann in ("IMGT", "REGION", "FULL"):
                x = done[(done.representation == rep) & (done.topology == topo) & (done.annotation == ann)]
                if len(x) != 1:
                    continue
                xr = x.iloc[0]
                annot_rows.append(
                    {
                        "representation": rep,
                        "topology": topo,
                        "annotation": ann,
                        "delta_P": xr.TEST_P - b.TEST_P,
                        "delta_S": xr.TEST_S - b.TEST_S,
                        "delta_mean": xr.TEST_mean - b.TEST_mean,
                        "both_improve": (xr.TEST_P - b.TEST_P) < 0 and (xr.TEST_S - b.TEST_S) < 0,
                        "code_x": xr.experiment_code,
                        "code_base": b.experiment_code,
                    }
                )
                for scheme in ("primary", "shadow"):
                    d, lo, hi = contrast_boot(xr.experiment_code, b.experiment_code, scheme, ymap)
                    boot_rows.append(
                        {
                            "kind": "annotation_vs_BASE",
                            "representation": rep,
                            "topology": topo,
                            "annotation": ann,
                            "scheme": scheme,
                            "delta_mae": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "code_x": xr.experiment_code,
                            "code_ref": b.experiment_code,
                        }
                    )
    annot_gains = pd.DataFrame(annot_rows)
    annot_gains.to_csv(ROOT / "results/HIC_REP_TOPO_ANNOT_ANNOTATION_GAINS.csv", index=False)

    # PLM vs Scratch
    plm_rows = []
    for topo in TOPOS:
        for ann in ANNOTS:
            sc = done[(done.representation == "scratch") & (done.topology == topo) & (done.annotation == ann)]
            if len(sc) != 1:
                continue
            srow = sc.iloc[0]
            for rep in done.representation.unique():
                if rep == "scratch":
                    continue
                x = done[(done.representation == rep) & (done.topology == topo) & (done.annotation == ann)]
                if len(x) != 1:
                    continue
                xr = x.iloc[0]
                plm_rows.append(
                    {
                        "representation": rep,
                        "topology": topo,
                        "annotation": ann,
                        "delta_P": xr.TEST_P - srow.TEST_P,
                        "delta_S": xr.TEST_S - srow.TEST_S,
                        "delta_mean": xr.TEST_mean - srow.TEST_mean,
                        "both_improve": (xr.TEST_P - srow.TEST_P) < 0 and (xr.TEST_S - srow.TEST_S) < 0,
                        "code_x": xr.experiment_code,
                        "code_scratch": srow.experiment_code,
                    }
                )
                for scheme in ("primary", "shadow"):
                    d, lo, hi = contrast_boot(xr.experiment_code, srow.experiment_code, scheme, ymap)
                    boot_rows.append(
                        {
                            "kind": "plm_vs_scratch",
                            "representation": rep,
                            "topology": topo,
                            "annotation": ann,
                            "scheme": scheme,
                            "delta_mae": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "code_x": xr.experiment_code,
                            "code_ref": srow.experiment_code,
                        }
                    )
    plm_gains = pd.DataFrame(plm_rows)
    plm_gains.to_csv(ROOT / "results/HIC_REP_TOPO_ANNOT_PLM_VS_SCRATCH.csv", index=False)

    # Context contrasts AbLang2 / CurrAb
    ctx_rows = []
    pairs = [
        ("ablang2", "ablang2_unpaired", "ablang2_paired"),  # PAIRED_NATIVE - SEPARATE
        ("currab", "currab_paired", "currab_unpaired"),
    ]
    for fam, paired, separate in pairs:
        for topo in TOPOS:
            for ann in ANNOTS:
                p = done[(done.representation == paired) & (done.topology == topo) & (done.annotation == ann)]
                s = done[(done.representation == separate) & (done.topology == topo) & (done.annotation == ann)]
                if len(p) != 1 or len(s) != 1:
                    continue
                pr, sr = p.iloc[0], s.iloc[0]
                ctx_rows.append(
                    {
                        "family": fam,
                        "topology": topo,
                        "annotation": ann,
                        "delta_P": pr.TEST_P - sr.TEST_P,
                        "delta_S": pr.TEST_S - sr.TEST_S,
                        "delta_mean": pr.TEST_mean - sr.TEST_mean,
                        "both_improve": (pr.TEST_P - sr.TEST_P) < 0 and (pr.TEST_S - sr.TEST_S) < 0,
                        "code_paired": pr.experiment_code,
                        "code_separate": sr.experiment_code,
                    }
                )
                for scheme in ("primary", "shadow"):
                    d, lo, hi = contrast_boot(pr.experiment_code, sr.experiment_code, scheme, ymap)
                    boot_rows.append(
                        {
                            "kind": f"context_{fam}_PAIRED_minus_SEPARATE",
                            "representation": fam,
                            "topology": topo,
                            "annotation": ann,
                            "scheme": scheme,
                            "delta_mae": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "code_x": pr.experiment_code,
                            "code_ref": sr.experiment_code,
                        }
                    )
    ctx = pd.DataFrame(ctx_rows)
    ctx.to_csv(ROOT / "results/HIC_REP_TOPO_ANNOT_CONTEXT_CONTRASTS.csv", index=False)

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(ROOT / "results/HIC_REP_TOPO_ANNOT_BOOTSTRAP.csv", index=False)

    # Interaction summaries
    rep_topo = done.pivot_table(index="representation", columns="topology", values="TEST_mean", aggfunc="mean")
    rep_annot = done.pivot_table(index="representation", columns="annotation", values="TEST_mean", aggfunc="mean")
    topo_annot = done.pivot_table(index="topology", columns="annotation", values="TEST_mean", aggfunc="mean")
    rep_topo.to_csv(OUT_DIR / "rep_topo_mean.csv")
    rep_annot.to_csv(OUT_DIR / "rep_annot_mean.csv")
    topo_annot.to_csv(OUT_DIR / "topo_annot_mean.csv")

    best = done.loc[done.TEST_mean.idxmin()]
    # Robustness: fraction with |P-S| small and scheme agreement on topo gains
    agree = float(topo_gains["both_improve"].mean()) if len(topo_gains) else np.nan

    # TmApp comparison if available
    tm_path = ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv"
    tm_cmp = []
    if tm_path.exists():
        tm = pd.read_csv(tm_path)
        # map topo labels
        tmap = {"A": "SEP", "B1": "JOINT", "B2": "REG-SEP", "C": "XREG", "D": "FUSE"}
        if "topology" in tm.columns and tm.topology.isin(list(tmap)).any():
            tm = tm.copy()
            tm["topology_h"] = tm["topology"].map(lambda x: tmap.get(x, x))
        else:
            tm["topology_h"] = tm["topology"]
        for metric_name, hic_col, tm_col in (("mean", "TEST_mean", "mean_ps"),):
            h_rep = done.groupby("representation")[hic_col].mean()
            if tm_col in tm.columns:
                t_rep = tm.groupby("representation")[tm_col].mean()
                for rep in sorted(set(h_rep.index) & set(t_rep.index)):
                    tm_cmp.append(
                        {
                            "facet": "representation_mean",
                            "level": rep,
                            "HIC": float(h_rep.loc[rep]),
                            "TmApp": float(t_rep.loc[rep]),
                        }
                    )
        pd.DataFrame(tm_cmp).to_csv(ROOT / "results/HIC_VS_TMAPP_FACTORIAL_COMPARISON.csv", index=False)

    # Report
    man = REPORTS / "HIC_REP_TOPO_ANNOT_FACTORIAL_CELL_MANIFEST.csv"
    lines = [
        "# HIC Representation × Topology × Annotation Factorial — INTERNAL REPORT",
        "",
        "**STATUS: INTERNAL_ANALYSIS (Public/Private embargo)**",
        "",
        f"- code_sha: `{git_rev()}`",
        f"- manifest_sha256: `{sha256_file(man)}`",
        f"- cells COMPLETE: 200 / 200",
        f"- best TEST_mean (descriptive): `{best.experiment_code}` "
        f"{best.representation}/{best.topology}/{best.annotation} = {best.TEST_mean:.6f}",
        "",
        "## 1. Execution completeness",
        "All 200 cells COMPLETE; 0 FAILED; 0 BLOCKED. See `HIC_H140_H339_COMPLETION_AUDIT.md`.",
        "",
        "## 2. Representation main patterns",
        done.groupby("representation")["TEST_mean"].agg(["mean", "median", "min", "max"]).sort_values("mean").to_markdown(),
        "",
        "## 3. Topology patterns (mean TEST_mean)",
        done.groupby("topology")["TEST_mean"].mean().reindex(list(TOPOS)).to_markdown(),
        "",
        "## 4. Annotation patterns (mean TEST_mean)",
        done.groupby("annotation")["TEST_mean"].mean().reindex(list(ANNOTS)).to_markdown(),
        "",
        "## 5–7. Interactions",
        f"- Rep×Topo table: `{OUT_DIR / 'rep_topo_mean.csv'}`",
        f"- Rep×Annot table: `{OUT_DIR / 'rep_annot_mean.csv'}`",
        f"- Topo×Annot table: `{OUT_DIR / 'topo_annot_mean.csv'}`",
        "",
        "## 8–9. Context (PAIRED_NATIVE − SEPARATE_CHAIN)",
        ctx.groupby("family")[["delta_mean", "delta_P", "delta_S"]].mean().to_markdown() if len(ctx) else "_n/a_",
        "",
        "## 10. Scratch controls",
        f"Scratch mean TEST_mean = {done[done.representation=='scratch'].TEST_mean.mean():.6f}",
        "",
        "## 11. Primary/Shadow robustness",
        f"- mean |P−S| = {done.abs_PS.mean():.6f}",
        f"- topology-vs-SEP both-scheme improve rate = {agree:.3f}",
        "",
        "## 12. Bootstrap",
        f"N_BOOT={N_BOOT}, seed={SEED}; tables in `HIC_REP_TOPO_ANNOT_BOOTSTRAP.csv`.",
        "",
        "## 13. HIGH-tail diagnostic (HIC > 11.5)",
        f"mean n_high across cells = {high.n_high.mean():.2f}; "
        f"mean MAE_high = {high.mae_high.mean():.4f}; "
        f"mean signed error = {high.mean_signed_error.mean():.4f}",
        "_Diagnostic only — not used for factor selection._",
        "",
        "## 14. Best cells (descriptive only)",
        done.nsmallest(5, "TEST_mean")[
            ["experiment_code", "representation", "topology", "annotation", "TEST_P", "TEST_S", "TEST_mean"]
        ].to_markdown(index=False),
        "",
        "## 15–17. Claims / non-claims",
        "- Established: internal 200-cell matrix completed under frozen prereg.",
        "- Weak/exploratory: cross-target TmApp comparison is descriptive only.",
        "- Non-claims: no Public/Private conclusions; HIGH-tail not used for selection; smoke MAE irrelevant.",
        "",
        "## 18. Reproducibility",
        f"- prereg: `4e175ab4f98864ba7f7c6496f832e06ea9730519`",
        f"- evaluation freeze: `379e0751a93c2af8f6fbfeedaad4d72f3556996b`",
        f"- analysis HEAD: `{git_rev()}`",
        "",
        "## Cross-target TmApp comparison",
        "See `developability_drilldown/results/HIC_VS_TMAPP_FACTORIAL_COMPARISON.csv` (descriptive; does not alter HIC selection).",
        "",
    ]
    report_path = REPORTS / "HIC_REP_TOPO_ANNOT_FACTORIAL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    freeze = {
        "status": "INTERNAL_FROZEN",
        "public_private_consulted": False,
        "n_complete": 200,
        "manifest_sha256": sha256_file(man),
        "results_sha256": sha256_file(res_path),
        "report_sha256": sha256_file(report_path),
        "prereg_sha": "4e175ab4f98864ba7f7c6496f832e06ea9730519",
        "evaluation_freeze_sha": "379e0751a93c2af8f6fbfeedaad4d72f3556996b",
        "analysis_head": git_rev(),
        "best_cell_descriptive": {
            "experiment_code": best.experiment_code,
            "representation": best.representation,
            "topology": best.topology,
            "annotation": best.annotation,
            "TEST_mean": float(best.TEST_mean),
        },
        "major_patterns": {
            "best_rep_by_mean": done.groupby("representation").TEST_mean.mean().idxmin(),
            "best_topo_by_mean": done.groupby("topology").TEST_mean.mean().idxmin(),
            "best_annot_by_mean": done.groupby("annotation").TEST_mean.mean().idxmin(),
        },
    }
    (REPORTS / "HIC_H140_H339_INTERNAL_FREEZE.yaml").write_text(yaml.safe_dump(freeze, sort_keys=False))
    print("ANALYSIS_OK", best.experiment_code, best.TEST_mean, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_analysis())
