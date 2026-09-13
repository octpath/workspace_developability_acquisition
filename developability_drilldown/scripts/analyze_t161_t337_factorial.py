#!/usr/bin/env python3
"""Full analysis for TmApp Representation × Topology × Annotation factorial (200 cells)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from run_exp_t151_t156_plm_topology import did_boot, load_oof, paired_boot  # noqa: E402

SEED = 101
N_BOOT = 2000
TOPOS = ("A", "B1", "B2", "C", "D")
ANNOTS = ("BASE", "IMGT", "REGION", "FULL")
PLAN_CSV = ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv"
FIG_DIR = ROOT / "results/t161_t337_figures"

PRE_REG_SHA = "7727ecc0f82798c8788d4b2f9ba3295e50502802"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def effective_code(row) -> str:
    src = row.get("reuse_source")
    if isinstance(src, str) and src.strip():
        return src.strip()
    return str(row["experiment_code"])


def load_plan() -> pd.DataFrame:
    return pd.read_csv(PLAN_CSV)


def scores_for_row(row, exp: pd.DataFrame) -> dict:
    code = effective_code(row)
    st = str(row["execution_status"])
    out = {
        "representation": row["representation"],
        "plm_family": row.get("plm_family"),
        "representation_context": row.get("representation_context"),
        "paired_match_group": row.get("paired_match_group"),
        "topology": row["topology"],
        "annotation": row["annotation"],
        "experiment_code": row["experiment_code"],
        "effective_code": code,
        "execution_status": st,
        "reuse_source": row.get("reuse_source") or "",
        "raw_dim": row.get("raw_dim"),
        "primary_mae": np.nan,
        "shadow_mae": np.nan,
        "mean_ps": np.nan,
        "worst_ps": np.nan,
        "abs_ps": np.nan,
    }
    if st not in ("COMPLETE", "REUSE"):
        return out
    # Prefer plan scores, else experiments.csv / OOF yaml
    if pd.notna(row.get("primary_mae")):
        p, s = float(row["primary_mae"]), float(row["shadow_mae"])
        out.update(
            primary_mae=p,
            shadow_mae=s,
            mean_ps=float(row["mean_ps"]) if pd.notna(row.get("mean_ps")) else (p + s) / 2,
            worst_ps=float(row["worst_ps"]) if pd.notna(row.get("worst_ps")) else max(p, s),
            abs_ps=abs(p - s),
        )
        return out
    m = exp[exp.experiment_code == code]
    if len(m) == 1:
        r = m.iloc[0]
        p, s = float(r.cv_primary_mae), float(r.cv_shadow_mae)
        out.update(
            primary_mae=p,
            shadow_mae=s,
            mean_ps=float(r.cv_mean_mae),
            worst_ps=float(r.cv_worst_mae),
            abs_ps=abs(p - s),
        )
        return out
    yml = ROOT / f"results/{code}_OOF_EVALUATION.yaml"
    if yml.exists():
        doc = yaml.safe_load(yml.read_text())
        ot = doc.get("oof_test") or doc["scores"]["oof_test"]
        p, s = float(ot["primary"]), float(ot["shadow"])
        out.update(
            primary_mae=p,
            shadow_mae=s,
            mean_ps=float(ot["mean"]),
            worst_ps=float(ot["worst"]),
            abs_ps=abs(p - s),
        )
    return out


def heatmap(mat: pd.DataFrame, title: str, path: Path, cmap="RdYlGn_r", center=None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = mat.to_numpy(dtype=float)
    if center is None:
        im = ax.imshow(data, aspect="auto", cmap=cmap)
    else:
        vmax = np.nanmax(np.abs(data - center))
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=center - vmax, vmax=center + vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(list(mat.columns))
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(list(mat.index))
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pivot_mean(res: pd.DataFrame, representation: str) -> pd.DataFrame:
    sub = res[res.representation == representation]
    return sub.pivot_table(index="annotation", columns="topology", values="mean_ps").reindex(
        index=list(ANNOTS), columns=list(TOPOS)
    )


def run_analysis() -> None:
    plan = load_plan()
    assert len(plan) == 200, len(plan)
    n_done = plan.execution_status.isin(["COMPLETE", "REUSE"]).sum()
    n_fail = (plan.execution_status == "FAILED").sum()
    n_block = plan.execution_status.astype(str).str.startswith("BLOCKED").sum()
    n_plan = (plan.execution_status == "PLANNED").sum()
    print(f"analysis: done={n_done} failed={n_fail} blocked={n_block} planned={n_plan}", flush=True)
    if n_plan > 0:
        print("matrix incomplete — writing partial RESULTS only", flush=True)

    exp = pd.read_csv(ROOT / "results/experiments.csv")
    dev = pd.read_csv(ROOT / "data/dev.csv")
    ymap = {str(r["id"]): float(r["TmApp"]) for _, r in dev.iterrows()}

    rows = [scores_for_row(r, exp) for _, r in plan.iterrows()]
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv", index=False)

    if n_plan > 0 or n_done < 200 - n_fail - n_block:
        # Allow analysis only when no PLANNED remain (FAILED/BLOCKED OK)
        if n_plan > 0:
            return

    reps = list(dict.fromkeys(res["representation"].tolist()))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Heatmaps per representation
    for rep in reps:
        mat = pivot_mean(res, rep)
        heatmap(mat, f"{rep}: mean(P,S)", FIG_DIR / f"heatmap_{rep}.png")

    # Topology gains vs A
    topo_gain_rows = []
    boot_rows = []
    for rep in reps:
        for ann in ANNOTS:
            base = res[(res.representation == rep) & (res.annotation == ann) & (res.topology == "A")]
            if len(base) != 1 or not np.isfinite(base.iloc[0]["mean_ps"]):
                continue
            a = base.iloc[0]
            for topo in ("B1", "B2", "C", "D"):
                x = res[(res.representation == rep) & (res.annotation == ann) & (res.topology == topo)]
                if len(x) != 1 or not np.isfinite(x.iloc[0]["mean_ps"]):
                    continue
                xr = x.iloc[0]
                topo_gain_rows.append(
                    {
                        "representation": rep,
                        "annotation": ann,
                        "topology": topo,
                        "delta_primary": xr.primary_mae - a.primary_mae,
                        "delta_shadow": xr.shadow_mae - a.shadow_mae,
                        "delta_mean": xr.mean_ps - a.mean_ps,
                        "code_x": xr.effective_code,
                        "code_a": a.effective_code,
                    }
                )
                for scheme, col in (("primary", "primary_mae"), ("shadow", "shadow_mae")):
                    try:
                        px = load_oof(xr.effective_code, scheme)
                        pa = load_oof(a.effective_code, scheme)
                    except Exception:
                        continue
                    m = px.to_frame("x").join(pa.to_frame("a"), how="inner")
                    y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                    d, lo, hi = paired_boot(m["x"].to_numpy(float), m["a"].to_numpy(float), y)
                    boot_rows.append(
                        {
                            "kind": "topology_gain_vs_A",
                            "representation": rep,
                            "annotation": ann,
                            "topology": topo,
                            "scheme": scheme,
                            "delta_mae": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "code_x": xr.effective_code,
                            "code_a": a.effective_code,
                        }
                    )

    topo_gains = pd.DataFrame(topo_gain_rows)
    topo_gains.to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_TOPOLOGY_GAINS.csv", index=False)

    # Annotation gains vs BASE
    annot_gain_rows = []
    for rep in reps:
        for topo in TOPOS:
            base = res[(res.representation == rep) & (res.topology == topo) & (res.annotation == "BASE")]
            if len(base) != 1 or not np.isfinite(base.iloc[0]["mean_ps"]):
                continue
            b = base.iloc[0]
            vals = {"BASE": b}
            for ann in ("IMGT", "REGION", "FULL"):
                x = res[(res.representation == rep) & (res.topology == topo) & (res.annotation == ann)]
                if len(x) != 1 or not np.isfinite(x.iloc[0]["mean_ps"]):
                    continue
                xr = x.iloc[0]
                vals[ann] = xr
                annot_gain_rows.append(
                    {
                        "representation": rep,
                        "topology": topo,
                        "annotation": ann,
                        "delta_primary": xr.primary_mae - b.primary_mae,
                        "delta_shadow": xr.shadow_mae - b.shadow_mae,
                        "delta_mean": xr.mean_ps - b.mean_ps,
                        "code_x": xr.effective_code,
                        "code_base": b.effective_code,
                    }
                )
                for scheme in ("primary", "shadow"):
                    try:
                        px = load_oof(xr.effective_code, scheme)
                        pb = load_oof(b.effective_code, scheme)
                    except Exception:
                        continue
                    m = px.to_frame("x").join(pb.to_frame("a"), how="inner")
                    y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                    d, lo, hi = paired_boot(m["x"].to_numpy(float), m["a"].to_numpy(float), y)
                    boot_rows.append(
                        {
                            "kind": "annotation_gain_vs_BASE",
                            "representation": rep,
                            "topology": topo,
                            "annotation": ann,
                            "scheme": scheme,
                            "delta_mae": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "code_x": xr.effective_code,
                            "code_base": b.effective_code,
                        }
                    )
            # non-additivity: FULL - IMGT - REGION + BASE
            if all(k in vals for k in ("FULL", "IMGT", "REGION", "BASE")):
                for scheme, attr in (("primary", "primary_mae"), ("shadow", "shadow_mae"), ("mean", "mean_ps")):
                    contrast = (
                        float(vals["FULL"][attr])
                        - float(vals["IMGT"][attr])
                        - float(vals["REGION"][attr])
                        + float(vals["BASE"][attr])
                    )
                    annot_gain_rows.append(
                        {
                            "representation": rep,
                            "topology": topo,
                            "annotation": "INTERACTION_FULL_IMGT_REGION",
                            "delta_primary": contrast if scheme == "primary" else np.nan,
                            "delta_shadow": contrast if scheme == "shadow" else np.nan,
                            "delta_mean": contrast if scheme == "mean" else np.nan,
                            "code_x": vals["FULL"].effective_code,
                            "code_base": b.effective_code,
                            "scheme_note": scheme,
                        }
                    )
                for scheme in ("primary", "shadow"):
                    try:
                        pf = load_oof(vals["FULL"].effective_code, scheme)
                        pi = load_oof(vals["IMGT"].effective_code, scheme)
                        pr = load_oof(vals["REGION"].effective_code, scheme)
                        pb = load_oof(vals["BASE"].effective_code, scheme)
                    except Exception:
                        continue
                    m = (
                        pf.to_frame("f")
                        .join(pi.to_frame("i"), how="inner")
                        .join(pr.to_frame("r"), how="inner")
                        .join(pb.to_frame("b"), how="inner")
                    )
                    y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                    # contrast on abs errors: (F-B) - (I-B) - (R-B) = F - I - R + B
                    dvec = (
                        np.abs(m["f"].to_numpy(float) - y)
                        - np.abs(m["i"].to_numpy(float) - y)
                        - np.abs(m["r"].to_numpy(float) - y)
                        + np.abs(m["b"].to_numpy(float) - y)
                    )
                    rng = np.random.default_rng(SEED)
                    boots = np.asarray(
                        [float(dvec[rng.integers(0, len(dvec), len(dvec))].mean()) for _ in range(N_BOOT)]
                    )
                    boot_rows.append(
                        {
                            "kind": "annotation_nonadditivity",
                            "representation": rep,
                            "topology": topo,
                            "annotation": "FULL_IMGT_REGION",
                            "scheme": scheme,
                            "delta_mae": float(dvec.mean()),
                            "ci95_lo": float(np.quantile(boots, 0.025)),
                            "ci95_hi": float(np.quantile(boots, 0.975)),
                        }
                    )

    annot_gains = pd.DataFrame(annot_gain_rows)
    annot_gains.to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_ANNOTATION_GAINS.csv", index=False)

    # DiD topology across representation pairs (within annotation)
    did_topo_rows = []
    for ann in ANNOTS:
        for topo in ("B1", "B2", "C", "D"):
            for i, r1 in enumerate(reps):
                for r2 in reps[i + 1 :]:
                    for scheme in ("primary", "shadow"):
                        try:
                            p1x = load_oof(
                                res[
                                    (res.representation == r1)
                                    & (res.annotation == ann)
                                    & (res.topology == topo)
                                ].iloc[0].effective_code,
                                scheme,
                            )
                            p1a = load_oof(
                                res[
                                    (res.representation == r1)
                                    & (res.annotation == ann)
                                    & (res.topology == "A")
                                ].iloc[0].effective_code,
                                scheme,
                            )
                            p2x = load_oof(
                                res[
                                    (res.representation == r2)
                                    & (res.annotation == ann)
                                    & (res.topology == topo)
                                ].iloc[0].effective_code,
                                scheme,
                            )
                            p2a = load_oof(
                                res[
                                    (res.representation == r2)
                                    & (res.annotation == ann)
                                    & (res.topology == "A")
                                ].iloc[0].effective_code,
                                scheme,
                            )
                        except Exception:
                            continue
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
                        did_topo_rows.append(
                            {
                                "annotation": ann,
                                "topology": topo,
                                "rep1": r1,
                                "rep2": r2,
                                "scheme": scheme,
                                "did": d,
                                "ci95_lo": lo,
                                "ci95_hi": hi,
                                "note": "DiD=gain(r1)-gain(r2); negative => r1 benefits more from topology",
                            }
                        )
    pd.DataFrame(did_topo_rows).to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_DID_TOPOLOGY.csv", index=False)

    # DiD annotation across representation pairs (within topology)
    did_ann_rows = []
    for topo in TOPOS:
        for ann in ("IMGT", "REGION", "FULL"):
            for i, r1 in enumerate(reps):
                for r2 in reps[i + 1 :]:
                    for scheme in ("primary", "shadow"):
                        try:
                            p1x = load_oof(
                                res[
                                    (res.representation == r1)
                                    & (res.topology == topo)
                                    & (res.annotation == ann)
                                ].iloc[0].effective_code,
                                scheme,
                            )
                            p1b = load_oof(
                                res[
                                    (res.representation == r1)
                                    & (res.topology == topo)
                                    & (res.annotation == "BASE")
                                ].iloc[0].effective_code,
                                scheme,
                            )
                            p2x = load_oof(
                                res[
                                    (res.representation == r2)
                                    & (res.topology == topo)
                                    & (res.annotation == ann)
                                ].iloc[0].effective_code,
                                scheme,
                            )
                            p2b = load_oof(
                                res[
                                    (res.representation == r2)
                                    & (res.topology == topo)
                                    & (res.annotation == "BASE")
                                ].iloc[0].effective_code,
                                scheme,
                            )
                        except Exception:
                            continue
                        m = (
                            p1x.to_frame("p1x")
                            .join(p1b.to_frame("p1a"), how="inner")
                            .join(p2x.to_frame("p2x"), how="inner")
                            .join(p2b.to_frame("p2a"), how="inner")
                        )
                        y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                        d, lo, hi = did_boot(
                            m["p1x"].to_numpy(float),
                            m["p1a"].to_numpy(float),
                            m["p2x"].to_numpy(float),
                            m["p2a"].to_numpy(float),
                            y,
                        )
                        did_ann_rows.append(
                            {
                                "topology": topo,
                                "annotation": ann,
                                "rep1": r1,
                                "rep2": r2,
                                "scheme": scheme,
                                "did": d,
                                "ci95_lo": lo,
                                "ci95_hi": hi,
                                "note": "DiD=annot_gain(r1)-annot_gain(r2)",
                            }
                        )
    pd.DataFrame(did_ann_rows).to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_DID_ANNOTATION.csv", index=False)

    # Matched paired vs unpaired
    paired_rows = []
    threeway = []
    match_groups = {
        "ablang2": ("ablang2_paired", "ablang2_unpaired"),
        "currab": ("currab_paired", "currab_unpaired"),
    }
    for family, (rp, ru) in match_groups.items():
        for ann in ANNOTS:
            for topo in TOPOS:
                sp = res[(res.representation == rp) & (res.annotation == ann) & (res.topology == topo)]
                su = res[(res.representation == ru) & (res.annotation == ann) & (res.topology == topo)]
                if len(sp) != 1 or len(su) != 1:
                    continue
                sp, su = sp.iloc[0], su.iloc[0]
                if not (np.isfinite(sp.mean_ps) and np.isfinite(su.mean_ps)):
                    continue
                delta_pair = {
                    "family": family,
                    "annotation": ann,
                    "topology": topo,
                    "delta_pair_primary": sp.primary_mae - su.primary_mae,
                    "delta_pair_shadow": sp.shadow_mae - su.shadow_mae,
                    "delta_pair_mean": sp.mean_ps - su.mean_ps,
                    "code_paired": sp.effective_code,
                    "code_unpaired": su.effective_code,
                }
                for scheme in ("primary", "shadow"):
                    try:
                        pp = load_oof(sp.effective_code, scheme)
                        pu = load_oof(su.effective_code, scheme)
                    except Exception:
                        continue
                    m = pp.to_frame("p").join(pu.to_frame("u"), how="inner")
                    y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                    d, lo, hi = paired_boot(m["p"].to_numpy(float), m["u"].to_numpy(float), y)
                    boot_rows.append(
                        {
                            "kind": "paired_minus_unpaired",
                            "family": family,
                            "annotation": ann,
                            "topology": topo,
                            "scheme": scheme,
                            "delta_mae": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                        }
                    )
                    delta_pair[f"boot_{scheme}"] = d
                    delta_pair[f"boot_{scheme}_lo"] = lo
                    delta_pair[f"boot_{scheme}_hi"] = hi
                paired_rows.append(delta_pair)

            # pairing × topology interaction vs A
            for topo in ("B1", "B2", "C", "D"):
                for scheme in ("primary", "shadow"):
                    try:
                        pp_t = load_oof(
                            res[
                                (res.representation == rp) & (res.annotation == ann) & (res.topology == topo)
                            ].iloc[0].effective_code,
                            scheme,
                        )
                        pp_a = load_oof(
                            res[
                                (res.representation == rp) & (res.annotation == ann) & (res.topology == "A")
                            ].iloc[0].effective_code,
                            scheme,
                        )
                        pu_t = load_oof(
                            res[
                                (res.representation == ru) & (res.annotation == ann) & (res.topology == topo)
                            ].iloc[0].effective_code,
                            scheme,
                        )
                        pu_a = load_oof(
                            res[
                                (res.representation == ru) & (res.annotation == ann) & (res.topology == "A")
                            ].iloc[0].effective_code,
                            scheme,
                        )
                    except Exception:
                        continue
                    m = (
                        pp_t.to_frame("p1x")
                        .join(pp_a.to_frame("p1a"), how="inner")
                        .join(pu_t.to_frame("p2x"), how="inner")
                        .join(pu_a.to_frame("p2a"), how="inner")
                    )
                    y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                    d, lo, hi = did_boot(
                        m["p1x"].to_numpy(float),
                        m["p1a"].to_numpy(float),
                        m["p2x"].to_numpy(float),
                        m["p2a"].to_numpy(float),
                        y,
                    )
                    threeway.append(
                        {
                            "kind": "pairing_x_topology",
                            "family": family,
                            "annotation": ann,
                            "topology": topo,
                            "scheme": scheme,
                            "contrast": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "note": "topology_gain_paired - topology_gain_unpaired",
                        }
                    )

        # pairing × annotation interaction vs BASE (once per family)
        for ann_x in ("IMGT", "REGION", "FULL"):
            for topo in TOPOS:
                for scheme in ("primary", "shadow"):
                    try:
                        pp_x = load_oof(
                            res[
                                (res.representation == rp)
                                & (res.topology == topo)
                                & (res.annotation == ann_x)
                            ].iloc[0].effective_code,
                            scheme,
                        )
                        pp_b = load_oof(
                            res[
                                (res.representation == rp)
                                & (res.topology == topo)
                                & (res.annotation == "BASE")
                            ].iloc[0].effective_code,
                            scheme,
                        )
                        pu_x = load_oof(
                            res[
                                (res.representation == ru)
                                & (res.topology == topo)
                                & (res.annotation == ann_x)
                            ].iloc[0].effective_code,
                            scheme,
                        )
                        pu_b = load_oof(
                            res[
                                (res.representation == ru)
                                & (res.topology == topo)
                                & (res.annotation == "BASE")
                            ].iloc[0].effective_code,
                            scheme,
                        )
                    except Exception:
                        continue
                    m = (
                        pp_x.to_frame("p1x")
                        .join(pp_b.to_frame("p1a"), how="inner")
                        .join(pu_x.to_frame("p2x"), how="inner")
                        .join(pu_b.to_frame("p2a"), how="inner")
                    )
                    y = np.asarray([ymap[i] for i in m.index.astype(str)], float)
                    d, lo, hi = did_boot(
                        m["p1x"].to_numpy(float),
                        m["p1a"].to_numpy(float),
                        m["p2x"].to_numpy(float),
                        m["p2a"].to_numpy(float),
                        y,
                    )
                    threeway.append(
                        {
                            "kind": "pairing_x_annotation",
                            "family": family,
                            "annotation": ann_x,
                            "topology": topo,
                            "scheme": scheme,
                            "contrast": d,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                            "note": "annotation_gain_paired - annotation_gain_unpaired",
                        }
                    )

        # paired-unpaired Δpair heatmaps
        sub = pd.DataFrame([r for r in paired_rows if r["family"] == family])
        if len(sub):
            mat = sub.pivot_table(index="annotation", columns="topology", values="delta_pair_mean").reindex(
                index=list(ANNOTS), columns=list(TOPOS)
            )
            heatmap(
                mat,
                f"{family}: Δpair = MAE_paired − MAE_unpaired (mean P/S)",
                FIG_DIR / f"delta_pair_{family}.png",
                cmap="coolwarm",
                center=0.0,
            )

    pd.DataFrame(paired_rows).to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_PAIRED_UNPAIRED.csv", index=False)
    pd.DataFrame(threeway).to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_THREEWAY.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_BOOTSTRAP.csv", index=False)

    # Topology / annotation gain heatmaps (mean)
    for rep in reps:
        g = topo_gains[topo_gains.representation == rep]
        if len(g):
            mat = g.pivot_table(index="annotation", columns="topology", values="delta_mean").reindex(
                index=list(ANNOTS), columns=["B1", "B2", "C", "D"]
            )
            heatmap(mat, f"{rep}: topology gain vs A", FIG_DIR / f"topo_gain_{rep}.png", cmap="coolwarm", center=0.0)
        g2 = annot_gains[
            (annot_gains.representation == rep) & (annot_gains.annotation.isin(["IMGT", "REGION", "FULL"]))
        ]
        if len(g2):
            mat = g2.pivot_table(index="annotation", columns="topology", values="delta_mean").reindex(
                index=["IMGT", "REGION", "FULL"], columns=list(TOPOS)
            )
            heatmap(
                mat, f"{rep}: annotation gain vs BASE", FIG_DIR / f"annot_gain_{rep}.png", cmap="coolwarm", center=0.0
            )

    # P vs S agreement
    ok = res[np.isfinite(res.primary_mae) & np.isfinite(res.shadow_mae)]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(ok.primary_mae, ok.shadow_mae, s=18, alpha=0.7)
    lims = [min(ok.primary_mae.min(), ok.shadow_mae.min()) - 0.05, max(ok.primary_mae.max(), ok.shadow_mae.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Primary OOF MAE")
    ax.set_ylabel("Shadow OOF MAE")
    ax.set_title("Primary vs Shadow agreement")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "primary_vs_shadow.png", dpi=140)
    plt.close(fig)

    # Best per representation summary plot
    best_rows = []
    for rep in reps:
        sub = res[(res.representation == rep) & np.isfinite(res.mean_ps)]
        if not len(sub):
            continue
        best = sub.sort_values(["mean_ps", "worst_ps"]).iloc[0]
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)
    if len(best_df):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(range(len(best_df)), best_df.mean_ps.to_numpy(float))
        ax.set_xticks(range(len(best_df)))
        ax.set_xticklabels(
            [f"{r.representation}\n{r.topology}/{r.annotation}" for r in best_df.itertuples()],
            rotation=0,
            fontsize=7,
        )
        ax.set_ylabel("best mean(P,S)")
        ax.set_title("Best cell per representation")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "best_per_representation.png", dpi=140)
        plt.close(fig)

    # Rankings / freeze / report
    valid = res[np.isfinite(res.mean_ps)].copy()
    top_mean = valid.sort_values(["mean_ps", "worst_ps"]).head(15)
    top_worst = valid.sort_values(["worst_ps", "mean_ps"]).head(15)

    def best_cell_text(sub: pd.DataFrame) -> str:
        if not len(sub):
            return "n/a"
        r = sub.sort_values(["mean_ps", "worst_ps"]).iloc[0]
        return f"{r.experiment_code} {r.representation}/{r.topology}/{r.annotation} mean={r.mean_ps:.4f} (P={r.primary_mae:.4f}, S={r.shadow_mae:.4f})"

    conclusions = {
        "best_overall": best_cell_text(valid),
        "best_scratch": best_cell_text(valid[valid.representation == "scratch"]),
        "best_per_representation": {
            rep: best_cell_text(valid[valid.representation == rep]) for rep in reps
        },
        "n_registered": 200,
        "n_reuse": int((res.execution_status == "REUSE").sum()),
        "n_complete_new": int((res.execution_status == "COMPLETE").sum()),
        "n_failed": int((res.execution_status == "FAILED").sum()),
        "n_blocked": int(res.execution_status.astype(str).str.startswith("BLOCKED").sum()),
        "pre_reg_commit": PRE_REG_SHA,
        "internal_freeze_commit": "PENDING",
        "head_sha": git_rev(),
    }

    freeze = {
        "statement": "Internal P/S / gains / DiD / paired-unpaired frozen before comparative external interpretation.",
        "conclusions": conclusions,
        "top_by_mean_ps": top_mean[
            ["experiment_code", "representation", "topology", "annotation", "primary_mae", "shadow_mae", "mean_ps", "worst_ps"]
        ].to_dict(orient="records"),
        "top_by_worst_ps": top_worst[
            ["experiment_code", "representation", "topology", "annotation", "primary_mae", "shadow_mae", "mean_ps", "worst_ps"]
        ].to_dict(orient="records"),
        "non_claims": [
            "Predictive inductive-bias evidence only; not mechanistic/structural claims about PLM internals.",
            "DiD and three-way contrasts are exploratory unless Primary and Shadow agree.",
            "Tiny MAE differences are not treated as decisive.",
            "AbLang2 allocation labels: historical ablang2_paired asset is SEPARATE_CHAIN; ablang2_unpaired is matched PAIRED_NATIVE joint inference — interpret via representation_context.",
        ],
    }
    (ROOT / "results/TMAPP_REP_TOPO_ANNOT_INTERNAL_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )

    # Human report (compact but complete section list)
    lines = [
        "# TmApp Representation × H/L Topology × Annotation Factorial Report",
        "",
        f"- PRE_REG_COMMIT_SHA: `{PRE_REG_SHA}`",
        f"- HEAD at report generation: `{git_rev()}`",
        f"- Registered cells: **200** (REUSE={conclusions['n_reuse']}, COMPLETE_new={conclusions['n_complete_new']}, FAILED={conclusions['n_failed']}, BLOCKED={conclusions['n_blocked']})",
        "",
        "## 1. Experiment design",
        "",
        "Frozen platform `DL_FOLDLOCAL_COSINE_V3`, seed 101.",
        "Axes: 10 representations × 5 topologies (A/B1/B2/C/D) × 4 annotations (BASE/IMGT/REGION/FULL).",
        "",
        "## 2. Representation provenance",
        "",
        "See plan YAML / prereg. Matched AbLang2 and CurrAb pairs hold checkpoint identity fixed;",
        "only inference-time H/L context differs (see `representation_context`).",
        "",
        "## 3. Complete 200-cell matrix status",
        "",
        "```",
        res.execution_status.value_counts().to_string(),
        "```",
        "",
        "## 4. Overall results",
        "",
        f"- Best overall: {conclusions['best_overall']}",
        f"- Best Scratch: {conclusions['best_scratch']}",
        "",
        "Per-representation best cells:",
    ]
    for rep, txt in conclusions["best_per_representation"].items():
        lines.append(f"- **{rep}**: {txt}")
    lines += [
        "",
        "Full long table: `results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv`",
        "Per-rep heatmaps: `results/t161_t337_figures/heatmap_*.png`",
        "",
        "## 5. H/L topology effects",
        "",
        "Gains vs A: `results/TMAPP_REP_TOPO_ANNOT_TOPOLOGY_GAINS.csv` (negative ⇒ interaction helps).",
        "",
        "## 6. Annotation effects",
        "",
        "Gains vs BASE + non-additivity: `results/TMAPP_REP_TOPO_ANNOT_ANNOTATION_GAINS.csv`.",
        "",
        "## 7. Representation × topology interactions",
        "",
        "DiD table: `results/TMAPP_REP_TOPO_ANNOT_DID_TOPOLOGY.csv`.",
        "",
        "## 8. Representation × annotation interactions",
        "",
        "DiD table: `results/TMAPP_REP_TOPO_ANNOT_DID_ANNOTATION.csv`.",
        "",
        "## 9. AbLang2 paired-vs-unpaired",
        "",
        "See `results/TMAPP_REP_TOPO_ANNOT_PAIRED_UNPAIRED.csv` (family=ablang2) and figure `delta_pair_ablang2.png`.",
        "Interpret with representation_context (historical SEPARATE vs matched PAIRED_NATIVE).",
        "",
        "## 10. CurrAb paired-vs-unpaired",
        "",
        "See paired/unpaired CSV (family=currab) and `delta_pair_currab.png`.",
        "",
        "## 11. Scratch control",
        "",
        "Scratch remains a scientific control for pretrained residue information vs downstream inductive bias.",
        f"Best Scratch cell: {conclusions['best_scratch']}",
        "",
        "## 12. PLM generation/domain observations",
        "",
        "Descriptive only (general-protein vs antibody-oriented; old vs newer related models).",
        "Wording: consistent with representation-dependent downstream requirements — not causal claims about corpora.",
        "",
        "## 13. Primary/Shadow robustness",
        "",
        "See `primary_vs_shadow.png` and bootstrap CIs; prefer effects agreeing on both schemes.",
        "",
        "## 14. Internal ranking",
        "",
        "Top by mean(P,S):",
        "",
        "```",
        top_mean[
            ["experiment_code", "representation", "topology", "annotation", "mean_ps", "worst_ps", "primary_mae", "shadow_mae"]
        ].to_string(index=False),
        "```",
        "",
        "## 15. Scientific interpretation",
        "",
        "Answers to the six preregistered questions are encoded in topology/annotation/DiD/paired tables above.",
        "Broad multi-comparison findings are exploratory unless replicated across Primary and Shadow.",
        "",
        "## 16. Non-claims",
        "",
        "- Predictive inductive-bias evidence ≠ mechanistic/structural claims.",
        "- No claim that downstream topology is literally equivalent to PLM-internal pairing.",
        "- External Public/Private diagnostics (if any) do not rewrite this internal freeze.",
        "",
        "## 17. Failures / blocked cells",
        "",
        res[res.execution_status.isin(["FAILED"]) | res.execution_status.astype(str).str.startswith("BLOCKED")][
            ["experiment_code", "representation", "topology", "annotation", "execution_status"]
        ].to_string(index=False)
        if ((res.execution_status == "FAILED").any() or res.execution_status.astype(str).str.startswith("BLOCKED").any())
        else "None.",
        "",
        "## 18. Reproducibility / commits",
        "",
        f"- PRE_REG_COMMIT_SHA: `{PRE_REG_SHA}`",
        f"- INTERNAL_FREEZE_COMMIT_SHA: recorded after this freeze is committed",
        f"- Platform: DL_FOLDLOCAL_COSINE_V3 / seed 101",
        "",
    ]
    (ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "results/TMAPP_REP_TOPO_ANNOT_INTERNAL_FREEZE.md").write_text(
        "# Internal freeze\n\n"
        + yaml.safe_dump(freeze, sort_keys=False)
        + "\nSee also FACTORIAL_REPORT.md and machine-readable CSVs.\n",
        encoding="utf-8",
    )
    print("analysis complete", flush=True)


def run_external() -> None:
    freeze = ROOT / "results/TMAPP_REP_TOPO_ANNOT_INTERNAL_FREEZE.yaml"
    if not freeze.exists():
        raise SystemExit("internal freeze missing — refuse external diagnostic")
    res = pd.read_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv")
    exp = pd.read_csv(ROOT / "results/experiments.csv")
    rows = []
    for _, r in res.iterrows():
        code = r.get("effective_code") or r["experiment_code"]
        m = exp[exp.experiment_code == code]
        if len(m) != 1:
            continue
        e = m.iloc[0]
        rows.append(
            {
                "experiment_code": r["experiment_code"],
                "effective_code": code,
                "representation": r["representation"],
                "topology": r["topology"],
                "annotation": r["annotation"],
                "mean_ps": r["mean_ps"],
                "public_mae": e.get("public_mae"),
                "private_mae": e.get("private_mae"),
                "test_overall_mae": e.get("test_overall_mae"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_EXTERNAL_DIAGNOSTIC.csv", index=False)
    md = [
        "# External diagnostic (post-internal-freeze)",
        "",
        "Diagnostic only. Does not trigger retraining or redesign.",
        "",
        f"Rows with external scores: {len(out)}",
        "",
        "See CSV for Public/Private/overall.",
    ]
    (ROOT / "results/TMAPP_REP_TOPO_ANNOT_EXTERNAL_DIAGNOSTIC.md").write_text("\n".join(md), encoding="utf-8")
    print("external diagnostic written", flush=True)


if __name__ == "__main__":
    run_analysis()
