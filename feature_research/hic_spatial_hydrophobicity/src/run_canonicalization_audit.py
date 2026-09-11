#!/usr/bin/env python3
"""HSP robustness / canonicalization audit (feature-research only).

Reuses Stage-1 VAL + antibody/residue atlas. No TEST selection, no EXP-H114,
no Transformer training.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1] if (ROOT.parents[1] / "developability_drilldown").exists() else ROOT.parent
# ROOT = feature_research/hic_spatial_hydrophobicity
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "developability_drilldown" / "scripts"))

from geometry_cache import load_antibody_arrays  # noqa: E402
from scales import property_raw, scale_hash  # noqa: E402
from spatial_engine import (  # noqa: E402
    RADII,
    aggregate,
    exposure_vector,
    family_id,
    feature_name,
    hydro_vector,
    pairwise_centroid,
    pairwise_closest_sc,
    spatial_scores,
    spec_hash,
)

RES = ROOT / "results"
FEAT = ROOT / "features"
RES.mkdir(parents=True, exist_ok=True)

BM_SCALE, BM_TRANSFORM, BM_EXPOSURE = "BM", "RAW", "TOTAL_RASA_TIEN"
EIS_SCALE, EIS_TRANSFORM, EIS_EXPOSURE = "EIS", "RAW", "SIDECHAIN_SASA_ABS"
NEIGHS = ("CLOSEST_SC", "CENTROID")
SELECTED = {
    "BM": {"neigh": "CLOSEST_SC", "R": 5.0},
    "EIS": {"neigh": "CENTROID", "R": 8.0},
}
PLATEAU_RADII = [5.0, 6.0, 7.5, 8.0]


def fid(scale: str, exposure: str, neigh: str, R: float) -> str:
    return family_id(scale, "RAW", exposure, neigh, R)


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _py(o):
    """Convert numpy scalars / arrays for YAML/JSON."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def geometry_analysis(geo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ids = sorted(geo["id"].unique())
    count_rows = []
    jaccard_rows = []
    # paired distance distributions for selected definitions
    cent_given_closest5 = []
    closest_given_cent8 = []
    j_c5_vs_cen8 = []

    for ai, aid in enumerate(ids):
        ab = load_antibody_arrays(geo, aid)
        Dc = pairwise_centroid(ab["centroids"])
        Ds = pairwise_closest_sc(ab["sc_heavy"])
        n = Dc.shape[0]
        # neighbor counts
        for neigh, D in (("CENTROID", Dc), ("CLOSEST_SC", Ds)):
            for R in RADII:
                # include self
                counts = (np.isfinite(D) & (D <= R)).sum(axis=1).astype(float)
                count_rows.append(
                    {
                        "id": aid,
                        "neighborhood": neigh,
                        "radius": R,
                        "mean_neighbor_count": float(np.nanmean(counts)),
                        "median_neighbor_count": float(np.nanmedian(counts)),
                        "n_residues": n,
                    }
                )
        # Jaccard matrices between all CLOSEST_SC Rx and CENTROID Ry (per residue then mean)
        for Rc in RADII:
            for Ry in RADII:
                jacs = []
                for i in range(n):
                    a = set(np.where(np.isfinite(Ds[i]) & (Ds[i] <= Rc))[0].tolist())
                    b = set(np.where(np.isfinite(Dc[i]) & (Dc[i] <= Ry))[0].tolist())
                    if not a and not b:
                        j = 1.0
                    else:
                        j = len(a & b) / max(1, len(a | b))
                    jacs.append(j)
                arr = np.asarray(jacs)
                jaccard_rows.append(
                    {
                        "id": aid,
                        "closest_R": Rc,
                        "centroid_R": Ry,
                        "mean_jaccard": float(arr.mean()),
                        "median_jaccard": float(np.median(arr)),
                        "q10_jaccard": float(np.quantile(arr, 0.10)),
                        "q90_jaccard": float(np.quantile(arr, 0.90)),
                        "frac_identical": float((arr >= 1.0 - 1e-12).mean()),
                        "frac_jaccard_ge_0p8": float((arr >= 0.8).mean()),
                        "frac_jaccard_ge_0p5": float((arr >= 0.5).mean()),
                    }
                )
                if Rc == 5.0 and Ry == 8.0:
                    j_c5_vs_cen8.append(arr)

        # distance cross distributions (upper triangle pairs, exclude self)
        iu = np.triu_indices(n, k=1)
        pair_sc = Ds[iu]
        pair_c = Dc[iu]
        m5 = np.isfinite(pair_sc) & (pair_sc <= 5.0) & np.isfinite(pair_c)
        if m5.any():
            cent_given_closest5.append(pair_c[m5])
        m8 = np.isfinite(pair_c) & (pair_c <= 8.0) & np.isfinite(pair_sc)
        if m8.any():
            closest_given_cent8.append(pair_sc[m8])
        if (ai + 1) % 50 == 0:
            print(f"geometry {ai+1}/{len(ids)}", flush=True)

    counts_df = pd.DataFrame(count_rows)
    jac_df = pd.DataFrame(jaccard_rows)

    def dist_summary(arrays: list[np.ndarray], name: str) -> dict:
        if not arrays:
            return {"name": name}
        v = np.concatenate(arrays)
        return {
            "name": name,
            "n_pairs": int(len(v)),
            "median": float(np.median(v)),
            "q10": float(np.quantile(v, 0.10)),
            "q25": float(np.quantile(v, 0.25)),
            "q75": float(np.quantile(v, 0.75)),
            "q90": float(np.quantile(v, 0.90)),
            "q95": float(np.quantile(v, 0.95)),
            "mean": float(np.mean(v)),
        }

    # aggregate Jaccard CLOSEST_SC R5 vs CENTROID R8 across Abs (pool residue jaccards)
    jac_pool = np.concatenate(j_c5_vs_cen8) if j_c5_vs_cen8 else np.array([])
    summary = {
        "CLOSEST_SC_R5_vs_CENTROID_R8": {
            "mean_jaccard": float(jac_pool.mean()) if len(jac_pool) else None,
            "median_jaccard": float(np.median(jac_pool)) if len(jac_pool) else None,
            "q10": float(np.quantile(jac_pool, 0.10)) if len(jac_pool) else None,
            "q90": float(np.quantile(jac_pool, 0.90)) if len(jac_pool) else None,
            "frac_identical": float((jac_pool >= 1.0 - 1e-12).mean()) if len(jac_pool) else None,
            "frac_jaccard_ge_0p8": float((jac_pool >= 0.8).mean()) if len(jac_pool) else None,
            "frac_jaccard_ge_0p5": float((jac_pool >= 0.5).mean()) if len(jac_pool) else None,
            "n_residue_centers": int(len(jac_pool)),
        },
        "centroid_dist_given_closest_sc_le_5": dist_summary(cent_given_closest5, "centroid|closest<=5"),
        "closest_sc_dist_given_centroid_le_8": dist_summary(closest_given_cent8, "closest|centroid<=8"),
        "mean_neighbor_counts": {
            "CLOSEST_SC_R5": float(
                counts_df[(counts_df.neighborhood == "CLOSEST_SC") & (counts_df.radius == 5.0)][
                    "mean_neighbor_count"
                ].mean()
            ),
            "CENTROID_R8": float(
                counts_df[(counts_df.neighborhood == "CENTROID") & (counts_df.radius == 8.0)][
                    "mean_neighbor_count"
                ].mean()
            ),
        },
    }
    return counts_df, jac_df, summary


def feature_similarity(ab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, exposure, tag in (
        (BM_SCALE, BM_EXPOSURE, "BM"),
        (EIS_SCALE, EIS_EXPOSURE, "EIS"),
    ):
        fams = [fid(scale, exposure, n, R) for n in NEIGHS for R in RADII]
        for agg in ("MAX", "MEAN", "SUM"):
            cols = {f: feature_name(f, "ALL_FV", agg) for f in fams}
            for f1 in fams:
                for f2 in fams:
                    c1, c2 = cols[f1], cols[f2]
                    x, y = ab[c1].to_numpy(float), ab[c2].to_numpy(float)
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() < 3:
                        pear = spear = float("nan")
                    else:
                        pear = float(np.corrcoef(x[m], y[m])[0, 1])
                        spear = float(pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman"))
                    rows.append(
                        {
                            "family_group": tag,
                            "aggregation": agg,
                            "family_a": f1,
                            "family_b": f2,
                            "pearson": pear,
                            "spearman": spear,
                        }
                    )
    # cross BM selected vs EIS selected
    for agg in ("MAX", "MEAN", "SUM"):
        fbm = fid(BM_SCALE, BM_EXPOSURE, "CLOSEST_SC", 5.0)
        feis = fid(EIS_SCALE, EIS_EXPOSURE, "CENTROID", 8.0)
        c1 = feature_name(fbm, "ALL_FV", agg)
        c2 = feature_name(feis, "ALL_FV", agg)
        x, y = ab[c1].to_numpy(float), ab[c2].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        rows.append(
            {
                "family_group": "BM_vs_EIS_SELECTED",
                "aggregation": agg,
                "family_a": fbm,
                "family_b": feis,
                "pearson": float(np.corrcoef(x[m], y[m])[0, 1]),
                "spearman": float(pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def extract_val_landscapes(s1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def one(scale, exposure, group):
        rows = []
        for neigh in NEIGHS:
            for R in RADII:
                fam = fid(scale, exposure, neigh, R)
                sub = s1[(s1.family_id == fam) & (s1.bundle == "B3") & (s1.kind == "GENERIC")]
                for ctx in ("A_ALONE", "B_SURFACE"):
                    for est in ("Ridge", "SVR"):
                        r = sub[(sub.context == ctx) & (sub.estimator == est)]
                        if len(r) != 1:
                            # Stage1 has primary/shadow folded into VAL_P/S already as one row per est
                            pass
                        if r.empty:
                            continue
                        row = r.iloc[0]
                        rows.append(
                            {
                                "family_group": group,
                                "family_id": fam,
                                "neighborhood": neigh,
                                "radius": R,
                                "context": ctx,
                                "estimator": est,
                                "VAL_P": float(row["VAL_P"]),
                                "VAL_S": float(row["VAL_S"]),
                                "VAL_mean": float(row["VAL_mean"]),
                                "VAL_worst": float(row["VAL_worst"]),
                                "n_features": int(row.get("n_features", 3)),
                                "source": "STAGE1_VAL_SCREEN.csv reused",
                            }
                        )
        return pd.DataFrame(rows)

    return one(BM_SCALE, BM_EXPOSURE, "BM"), one(EIS_SCALE, EIS_EXPOSURE, "EIS")


def landscape_tables(df: pd.DataFrame, group: str) -> dict[str, pd.DataFrame]:
    out = {}
    for ctx, est in (
        ("A_ALONE", "Ridge"),
        ("A_ALONE", "SVR"),
        ("B_SURFACE", "Ridge"),
        ("B_SURFACE", "SVR"),
    ):
        sub = df[(df.context == ctx) & (df.estimator == est)]
        piv = sub.pivot_table(index="neighborhood", columns="radius", values="VAL_mean")
        piv = piv.reindex(index=list(NEIGHS), columns=RADII)
        out[f"{group}_{ctx}_{est}"] = piv
    return out


def plot_landscapes(df: pd.DataFrame, group: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panels = [
        ("A_ALONE", "Ridge", axes[0, 0]),
        ("A_ALONE", "SVR", axes[0, 1]),
        ("B_SURFACE", "Ridge", axes[1, 0]),
        ("B_SURFACE", "SVR", axes[1, 1]),
    ]
    for ctx, est, ax in panels:
        for neigh, style in (("CLOSEST_SC", "-o"), ("CENTROID", "--s")):
            sub = df[(df.context == ctx) & (df.estimator == est) & (df.neighborhood == neigh)].sort_values(
                "radius"
            )
            ax.plot(sub["radius"], sub["VAL_mean"], style, label=neigh)
        ax.set_title(f"{group} {ctx} {est}")
        ax.set_xlabel("radius (Å)")
        ax.set_ylabel("VAL_mean MAE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def robustness_for_group(df: pd.DataFrame, group: str, sel_neigh: str, sel_R: float) -> dict:
    # Use SURFACE Ridge as primary confirmatory screen context (matches mainline scientific test)
    # Also report alone Ridge
    out = {"group": group, "selected": {"neighborhood": sel_neigh, "radius": sel_R}}
    for ctx in ("A_ALONE", "B_SURFACE"):
        block = {}
        for est in ("Ridge", "SVR"):
            sub = df[(df.context == ctx) & (df.estimator == est)].copy()
            plateau = sub[sub.radius.isin(PLATEAU_RADII)]
            # fold-scheme variability proxy: |VAL_P - VAL_S| / 2 as 1-SE style width at selected
            sel = sub[(sub.neighborhood == sel_neigh) & (sub.radius == sel_R)]
            if sel.empty:
                continue
            sel_row = sel.iloc[0]
            se = abs(sel_row.VAL_P - sel_row.VAL_S) / 2.0
            best_val = float(sub["VAL_mean"].min())
            plateau_vals = plateau["VAL_mean"].to_numpy(float)
            nearby_same_neigh = sub[(sub.neighborhood == sel_neigh) & (sub.radius.isin(PLATEAU_RADII))]
            med_nearby = float(nearby_same_neigh["VAL_mean"].median())
            sel_val = float(sel_row.VAL_mean)
            # plateau membership: within best + SE (one-SE of selected scheme variability)
            in_plateau = {}
            for _, r in nearby_same_neigh.iterrows():
                in_plateau[f"{r.neighborhood}_R{r.radius}"] = bool(r.VAL_mean <= best_val + se + 1e-12) or bool(
                    abs(r.VAL_mean - sel_val) <= se + 1e-12
                )
            # also: how many of 4 plateau radii for selected neigh within SE of selected
            n_compat = sum(
                1
                for _, r in nearby_same_neigh.iterrows()
                if abs(r.VAL_mean - sel_val) <= max(se, 0.005)  # se or 0.005 MAE descriptive floor for tiny SE
            )
            # neighborhood swap at selected R
            alt = sub[(sub.radius == sel_R) & (sub.neighborhood != sel_neigh)]
            alt_delta = float(alt.iloc[0].VAL_mean - sel_val) if len(alt) else float("nan")
            block[est] = {
                "selected_VAL_mean": sel_val,
                "selected_VAL_P": float(sel_row.VAL_P),
                "selected_VAL_S": float(sel_row.VAL_S),
                "scheme_half_range_SE": se,
                "best_VAL_mean_in_grid": best_val,
                "delta_selected_minus_best": sel_val - best_val,
                "plateau_radii_range": float(plateau_vals.max() - plateau_vals.min()) if len(plateau_vals) else None,
                "plateau_radii_sd": float(plateau_vals.std(ddof=0)) if len(plateau_vals) else None,
                "selected_minus_median_nearby_same_neigh": sel_val - med_nearby,
                "n_plateau_radii_compatible_with_selected": n_compat,
                "n_plateau_radii": int(len(nearby_same_neigh)),
                "alt_neighborhood_delta_at_selected_R": alt_delta,
                "rank_selected_in_grid": int((sub["VAL_mean"] < sel_val - 1e-12).sum() + 1),
                "n_grid": int(len(sub)),
            }
        out[ctx] = block
    return out


def classify(rob: dict) -> str:
    """Classify family using SURFACE Ridge primarily + alone consistency."""
    surf = rob.get("B_SURFACE", {}).get("Ridge", {})
    alone = rob.get("A_ALONE", {}).get("Ridge", {})
    if not surf:
        return "INCONCLUSIVE"
    n_compat = surf.get("n_plateau_radii_compatible_with_selected", 0)
    n_plat = surf.get("n_plateau_radii", 4)
    range_p = surf.get("plateau_radii_range", 999)
    alt = abs(surf.get("alt_neighborhood_delta_at_selected_R", 999) or 999)
    # Isolated: selected much better than nearby OR few compatible radii
    isolated = (n_compat <= 1) and (surf.get("selected_minus_median_nearby_same_neigh", 0) < -0.01)
    # or range across plateau tiny and many compatible -> robust
    robust_radius = (n_compat >= 3) or (range_p is not None and range_p <= 0.02 and n_compat >= 2)
    # geometry equivalent if neighborhood swap small
    geom_eq = alt <= 0.015
    if isolated:
        return "HYPERPARAMETER_SENSITIVE"
    if robust_radius and geom_eq:
        return "ROBUST_BUT_GEOMETRY_EQUIVALENT"
    if robust_radius:
        return "ROBUST_CANONICALIZABLE"
    if alone and alone.get("n_plateau_radii_compatible_with_selected", 0) >= 3:
        return "ROBUST_CANONICALIZABLE"
    return "INCONCLUSIVE"


def build_canonical_residue(geo: pd.DataFrame, ab: pd.DataFrame, families: dict) -> pd.DataFrame:
    """Build residue table for canonical families; reconstruct B3 QC."""
    fam_ids = list(families.values())
    dset = ds.dataset(str(FEAT / "residue_spatial_hydrophobicity.parquet"))
    tbl = dset.to_table(
        filter=ds.field("family_id").isin(fam_ids),
        columns=["id", "chain", "residue_index", "aa", "region", "family_id", "P_i"],
    )
    long = tbl.to_pandas()
    # merge geometry flags
    gmeta = geo[
        ["id", "chain", "residue_index", "is_cdr", "is_h_cdr3", "is_l_cdr3", "aa", "region"]
    ].drop_duplicates()
    # pivot
    wide = long.pivot_table(
        index=["id", "chain", "residue_index"], columns="family_id", values="P_i", aggfunc="first"
    ).reset_index()
    rename = {}
    for key, fam in families.items():
        rename[fam] = f"HSP_{key}_i"
    wide = wide.rename(columns=rename)
    wide = wide.merge(gmeta, on=["id", "chain", "residue_index"], how="left", suffixes=("", "_g"))
    if "aa_g" in wide.columns:
        wide["aa"] = wide["aa"].fillna(wide["aa_g"])
        wide = wide.drop(columns=[c for c in wide.columns if c.endswith("_g")])
    # note: IMGT not in HSP geometry cache — store null with provenance
    wide["imgt_position"] = pd.NA
    wide["imgt_note"] = "UNAVAILABLE_IN_HSP_GEOMETRY_CACHE_v1; use chain+residue_index+region"
    return wide


def residue_qc(wide: pd.DataFrame, ab: pd.DataFrame, families: dict) -> dict:
    qc = {"n_antibodies": int(wide.id.nunique()), "n_residues": int(len(wide)), "coverage_324": wide.id.nunique() == 324}
    recon_ok = {}
    for key, fam in families.items():
        col = f"HSP_{key}_i"
        s = wide[col]
        qc[f"{key}_score"] = {
            "finite_frac": float(np.isfinite(s).mean()),
            "min": float(np.nanmin(s)),
            "q05": float(np.nanquantile(s, 0.05)),
            "median": float(np.nanmedian(s)),
            "q95": float(np.nanquantile(s, 0.95)),
            "max": float(np.nanmax(s)),
        }
        # H vs L
        for ch in ("H", "L"):
            v = wide.loc[wide.chain == ch, col]
            qc[f"{key}_mean_{ch}"] = float(np.nanmean(v))
        # CDR vs FR
        qc[f"{key}_mean_CDR"] = float(np.nanmean(wide.loc[wide.is_cdr.fillna(False), col]))
        qc[f"{key}_mean_FR"] = float(np.nanmean(wide.loc[~wide.is_cdr.fillna(False), col]))
        qc[f"{key}_mean_HCDR3"] = float(np.nanmean(wide.loc[wide.is_h_cdr3.fillna(False), col]))
        # reconstruct B3
        rows = []
        for aid, sub in wide.groupby("id"):
            v = sub[col].to_numpy(float)
            ag = aggregate(v, np.isfinite(v))
            rows.append({"id": aid, "MAX": ag["MAX"], "MEAN": ag["MEAN"], "SUM": ag["SUM"]})
        rec = pd.DataFrame(rows)
        ok = True
        diffs = {}
        for agg in ("MAX", "MEAN", "SUM"):
            c = feature_name(fam, "ALL_FV", agg)
            m = rec.merge(ab[["id", c]], on="id")
            d = (m[agg] - m[c]).abs()
            diffs[agg] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}
            if d.max() > 1e-6:
                ok = False
        recon_ok[key] = {"exact": ok, "diffs": diffs}
    if "HSP_BM_i" in wide.columns and "HSP_EIS_i" in wide.columns:
        x = wide["HSP_BM_i"].to_numpy(float)
        y = wide["HSP_EIS_i"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        qc["BM_i_vs_EIS_i"] = {
            "pearson": float(np.corrcoef(x[m], y[m])[0, 1]),
            "spearman": float(pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")),
            "n": int(m.sum()),
        }
    qc["b3_reconstruction"] = recon_ok
    return qc


def main() -> int:
    print("=== HSP canonicalization audit ===", flush=True)
    from experiment_codes import next_code

    assert next_code("HIC") == "EXP-H114", next_code("HIC")

    geo = pd.read_parquet(FEAT / "cache" / "residue_geometry_sasa.parquet")
    ab = pd.read_parquet(FEAT / "antibody_spatial_hydrophobicity.parquet")
    s1 = pd.read_csv(RES / "STAGE1_VAL_SCREEN.csv")

    print("geometry analysis...", flush=True)
    counts_df, jac_df, geom_summary = geometry_analysis(geo)
    counts_df.to_csv(RES / "HSP_NEIGHBOR_COUNT_SUMMARY.csv", index=False)
    # long overlap table: keep antibody-level Jaccard for closest vs centroid pairs of interest + full
    jac_df.to_csv(RES / "HSP_GEOMETRY_OVERLAP.csv", index=False)
    (RES / "HSP_GEOMETRY_OVERLAP_SUMMARY.json").write_text(json.dumps(geom_summary, indent=2))

    print("feature similarity...", flush=True)
    sim = feature_similarity(ab)
    sim.to_csv(RES / "HSP_FEATURE_SIMILARITY.csv", index=False)

    print("VAL landscapes (reuse Stage-1)...", flush=True)
    bm_val, eis_val = extract_val_landscapes(s1)
    bm_val.to_csv(RES / "HSP_BM_RADIUS_NEIGHBORHOOD_VAL.csv", index=False)
    eis_val.to_csv(RES / "HSP_EIS_RADIUS_NEIGHBORHOOD_VAL.csv", index=False)
    for name, df in (("BM", bm_val), ("EIS", eis_val)):
        tabs = landscape_tables(df, name)
        for k, piv in tabs.items():
            piv.to_csv(RES / f"{k}_VAL_MEAN_PIVOT.csv")
        plot_landscapes(df, name, RES / f"HSP_{name}_VAL_LANDSCAPE.png")

    rob_bm = robustness_for_group(bm_val, "BM", "CLOSEST_SC", 5.0)
    rob_eis = robustness_for_group(eis_val, "EIS", "CENTROID", 8.0)
    class_bm = classify(rob_bm)
    class_eis = classify(rob_eis)
    rob_bm["classification"] = class_bm
    rob_eis["classification"] = class_eis

    # Feature-level near-equivalence can upgrade to GEOMETRY_EQUIVALENT
    def mean_spearman_selected_vs_alts(group_df, sel_fam, group):
        sub = sim[(sim.family_group == group) & (sim.agg == "MEAN") & (sim.family_a == sel_fam)]
        return float(sub["spearman"].median())

    bm_sel = fid(BM_SCALE, BM_EXPOSURE, "CLOSEST_SC", 5.0)
    eis_sel = fid(EIS_SCALE, EIS_EXPOSURE, "CENTROID", 8.0)
    # within-family median spearman of MEAN vs other geometries
    bm_med_sim = float(
        sim[
            (sim.family_group == "BM")
            & (sim["aggregation"] == "MEAN")
            & (sim.family_a == bm_sel)
            & (sim.family_b != bm_sel)
        ]["spearman"].median()
    )
    eis_med_sim = float(
        sim[
            (sim.family_group == "EIS")
            & (sim["aggregation"] == "MEAN")
            & (sim.family_a == eis_sel)
            & (sim.family_b != eis_sel)
        ]["spearman"].median()
    )
    if class_bm == "ROBUST_CANONICALIZABLE" and bm_med_sim >= 0.95:
        class_bm = "ROBUST_BUT_GEOMETRY_EQUIVALENT"
        rob_bm["classification"] = class_bm
    if class_eis == "ROBUST_CANONICALIZABLE" and eis_med_sim >= 0.95:
        class_eis = "ROBUST_BUT_GEOMETRY_EQUIVALENT"
        rob_eis["classification"] = class_eis

    # Canonical decision: keep mainline definitions if class allows
    canons = {}
    if class_bm in ("ROBUST_CANONICALIZABLE", "ROBUST_BUT_GEOMETRY_EQUIVALENT"):
        canons["BM"] = {
            "canonical_id": "HSP_BM_CANONICAL",
            "family_id": bm_sel,
            "scale": "BM",
            "transform": "RAW",
            "exposure": "TOTAL_RASA_TIEN",
            "neighborhood": "CLOSEST_SC",
            "radius_A": 5.0,
            "include_self": True,
            "aggregation_bundle": "B3",
            "scope": "ALL_FV",
            "aggregations": ["MAX", "MEAN", "SUM"],
            "structure_scope": "Fv_ESMFold",
            "reason": "Keep mainline P1; central in plateau; CLOSEST_SC is patch-sensitive; R5 literature-compatible with SAP-like scale",
            "classification": class_bm,
            "property_table_hash": scale_hash("BM", "RAW"),
            "spec_hash": spec_hash(
                scale="BM",
                transform="RAW",
                exposure="TOTAL_RASA_TIEN",
                neighborhood="CLOSEST_SC",
                radius=5.0,
                scale_table_hash=scale_hash("BM", "RAW"),
                sasa="ShrakeRupley_probe1.4_n100",
                structure="ESMFold_Fv_crosswalk_v2",
                include_self=True,
            ),
            "per_residue_formula": "P_i = sum_j I[min_heavy_SC(i,j)<=5] * total_rASA_Tien_j * BM_RAW_j (self included)",
        }
    if class_eis in ("ROBUST_CANONICALIZABLE", "ROBUST_BUT_GEOMETRY_EQUIVALENT"):
        canons["EIS"] = {
            "canonical_id": "HSP_EIS_CANONICAL",
            "family_id": eis_sel,
            "scale": "EIS",
            "transform": "RAW",
            "exposure": "SIDECHAIN_SASA_ABS",
            "neighborhood": "CENTROID",
            "radius_A": 8.0,
            "include_self": True,
            "aggregation_bundle": "B3",
            "scope": "ALL_FV",
            "aggregations": ["MAX", "MEAN", "SUM"],
            "structure_scope": "Fv_ESMFold",
            "reason": "Keep mainline P3; plateau-compatible; CENTROID R8 simple; exposure matches side-chain absolute SASA",
            "classification": class_eis,
            "property_table_hash": scale_hash("EIS", "RAW"),
            "spec_hash": spec_hash(
                scale="EIS",
                transform="RAW",
                exposure="SIDECHAIN_SASA_ABS",
                neighborhood="CENTROID",
                radius=8.0,
                scale_table_hash=scale_hash("EIS", "RAW"),
                sasa="ShrakeRupley_probe1.4_n100",
                structure="ESMFold_Fv_crosswalk_v2",
                include_self=True,
            ),
            "per_residue_formula": "P_i = sum_j I[centroid_dist(i,j)<=8] * sidechain_SASA_j * EIS_RAW_j (self included)",
        }

    # Residue artifact
    fam_map = {k: v["family_id"] for k, v in canons.items()}
    residue_justified = len(canons) >= 1
    qc = {}
    if fam_map:
        print("canonical residue artifact...", flush=True)
        wide = build_canonical_residue(geo, ab, fam_map)
        out_res = FEAT / "hsp_canonical_residue_scores.parquet"
        wide.to_parquet(out_res, index=False)
        qc = residue_qc(wide, ab, fam_map)
        qc["artifact"] = str(out_res)
        qc["imgt"] = "UNAVAILABLE — not in HSP geometry cache; chain+residue_index+region provided"
        (RES / "HSP_CANONICAL_RESIDUE_QC.json").write_text(json.dumps(qc, indent=2))
        # markdown QC
        lines = [
            "# HSP Canonical Residue QC",
            "",
            f"- antibodies: {qc['n_antibodies']} (324 required: {qc['coverage_324']})",
            f"- residues: {qc['n_residues']}",
            f"- IMGT: {qc['imgt']}",
            "",
            "## B3 reconstruction",
            "",
            yaml.safe_dump(qc.get("b3_reconstruction", {}), sort_keys=False),
            "",
            "## Distributions / regions",
            "",
            "```json",
            json.dumps({k: v for k, v in qc.items() if k not in ("b3_reconstruction", "artifact", "imgt")}, indent=2),
            "```",
            "",
        ]
        (RES / "HSP_CANONICAL_RESIDUE_QC.md").write_text("\n".join(lines))

    # Spec yaml
    spec_doc = {
        "status": "CANONICALIZED" if canons else "NO_CANONICAL",
        "mainline_next_hic": "EXP-H114",
        "consumed_EXP_H114": False,
        "no_TEST_used_for_selection": True,
        "no_external_used": True,
        "source_stage1": str(RES / "STAGE1_VAL_SCREEN.csv"),
        "geometry_summary": geom_summary,
        "classifications": {"BM": class_bm, "EIS": class_eis},
        "robustness": {"BM": rob_bm, "EIS": rob_eis},
        "feature_similarity_median_spearman_MEAN": {"BM": bm_med_sim, "EIS": eis_med_sim},
        "canonical_descriptors": canons,
        "residue_level_justified": residue_justified and all(
            qc.get("b3_reconstruction", {}).get(k, {}).get("exact", False) for k in canons
        ),
        "n_residue_channels": len(canons),
    }
    (RES / "HSP_CANONICAL_DESCRIPTOR_SPEC.yaml").write_text(
        yaml.safe_dump(_py(spec_doc), sort_keys=False)
    )

    # Robustness summary MD
    def piv_md(path: Path) -> str:
        if not path.exists():
            return "(missing)"
        return pd.read_csv(path, index_col=0).to_string()

    rob_md = [
        "# HSP Robustness Summary",
        "",
        "Stage-1 VAL only. No TEST / Public / Private used for decisions.",
        "",
        f"## Geometry: CLOSEST_SC R5 vs CENTROID R8",
        "",
        "```json",
        json.dumps(geom_summary.get("CLOSEST_SC_R5_vs_CENTROID_R8"), indent=2),
        "```",
        "",
        f"Mean neighbor counts: {geom_summary.get('mean_neighbor_counts')}",
        "",
        "## BM classification",
        f"**{class_bm}**",
        "",
        "```yaml",
        yaml.safe_dump(_py(rob_bm), sort_keys=False),
        "```",
        "",
        "### BM SURFACE Ridge VAL_mean pivot",
        "",
        "```",
        piv_md(RES / "BM_B_SURFACE_Ridge_VAL_MEAN_PIVOT.csv"),
        "```",
        "",
        "## EIS classification",
        f"**{class_eis}**",
        "",
        "```yaml",
        yaml.safe_dump(_py(rob_eis), sort_keys=False),
        "```",
        "",
        "### EIS SURFACE Ridge VAL_mean pivot",
        "",
        "```",
        piv_md(RES / "EIS_B_SURFACE_Ridge_VAL_MEAN_PIVOT.csv"),
        "```",
        "",
        "## BM vs EIS selected antibody-level similarity (MEAN)",
        "",
        sim[(sim.family_group == "BM_vs_EIS_SELECTED")].to_string(index=False),
        "",
    ]
    (RES / "HSP_ROBUSTNESS_SUMMARY.md").write_text("\n".join(rob_md))

    # Canonicalization report
    canon_md = [
        "# HSP Canonicalization Report",
        "",
        f"- BM: **{class_bm}** → canonical: `{canons.get('BM', {}).get('canonical_id', 'NONE')}`",
        f"- EIS: **{class_eis}** → canonical: `{canons.get('EIS', {}).get('canonical_id', 'NONE')}`",
        "",
        "Selection followed interpretability / literature / plateau-centrality; **not** lowest VAL.",
        "Existing mainline P1/P3 geometries retained when robust.",
        "",
        "## Exact canonical definitions",
        "",
        "```yaml",
        yaml.safe_dump(_py(canons), sort_keys=False) if canons else "NONE — no family passed robustness gate",
        "```",
        "",
        f"## Residue-level modeling justified? **{spec_doc['residue_level_justified']}**",
        f"Channels: {spec_doc['n_residue_channels']}",
        "",
        "If neither family is ROBUST_*: do **not** proceed to residue-level injection",
        "merely because H103/H107 looked good under the exact selected geometries.",
        "",
        "See `HSP_CANONICAL_RESIDUE_QC.md` and `HSP_RESIDUE_LEVEL_EXPERIMENT_DESIGN.md`.",
        "",
    ]
    (RES / "HSP_CANONICALIZATION_REPORT.md").write_text("\n".join(canon_md))

    # Always write residue QC artifact (even if empty / not justified)
    if not fam_map:
        qc_empty = {
            "n_antibodies": 0,
            "n_residues": 0,
            "coverage_324": False,
            "status": "NO_CANONICAL_FAMILY",
            "b3_reconstruction": {},
            "note": "No ROBUST_* family; residue canonical parquet not written.",
        }
        (RES / "HSP_CANONICAL_RESIDUE_QC.json").write_text(json.dumps(qc_empty, indent=2))
        (RES / "HSP_CANONICAL_RESIDUE_QC.md").write_text(
            "\n".join(
                [
                    "# HSP Canonical Residue QC",
                    "",
                    "**Status: NO_CANONICAL_FAMILY**",
                    "",
                    "Both BM and EIS were classified as non-robust for canonicalization.",
                    "No `hsp_canonical_residue_scores.parquet` was written.",
                    "Residue-level modeling is **not** justified under the preregistered gate.",
                    "",
                ]
            )
        )

    # Future residue experiment design (DO NOT RUN)
    design = [
        "# HSP Residue-Level Experiment Design (DO NOT RUN)",
        "",
        "Status: design note only. No Transformer training. EXP-H114 unused.",
        "",
        "## Channels",
        "",
    ]
    if "BM" in canons:
        design.append("- `HSP_BM_i` from `HSP_BM_CANONICAL` (CLOSEST_SC R5, BM RAW, TOTAL_RASA_TIEN)")
    if "EIS" in canons:
        design.append("- `HSP_EIS_i` from `HSP_EIS_CANONICAL` (CENTROID R8, EIS RAW, SIDECHAIN_SASA_ABS)")
    if not canons:
        design.append("- NONE — residue-level experiment NOT justified.")
    k = len(canons)
    design += [
        "",
        f"k = {k} residue HSP channels.",
        "",
        "## Comparison arms (future)",
        "",
        "| Arm | Aux / injection | Purpose |",
        "|-----|-----------------|--------|",
        "| A | SURFACE only | baseline |",
        "| B | SURFACE + antibody-level canonical HSP (B3) | Ab-level control |",
        "| C | SURFACE + residue-level canonical HSP | location-preserving |",
        "| D | SURFACE + Ab-level HSP + residue-level HSP | complementarity |",
        "",
        "## First injection mechanism (fixed, minimal)",
        "",
        "```",
        "hsp_i ∈ R^k   # TRAIN-fold StandardScaler on HSP channels",
        "hsp_proj: Linear(k, d_model)  # init weights/bias = 0",
        "token_i ← token_i + hsp_proj(hsp_i)",
        "```",
        "",
        "Init-equivalence QC required:",
        "",
        "```",
        "pred(residue_HSP at init) == pred(no residue_HSP)",
        "```",
        "",
        "## Functional-use diagnostics (future)",
        "",
        "1. force hsp_proj = 0 at inference",
        "2. permute HSP among residues within antibody",
        "3. permute HSP profiles across antibodies",
        "",
        "## Questions answered by this design",
        "",
        "1. Spatial hydrophobicity useful? — largely answered by Ab-level H102–H113",
        "2. Does WHERE the patch occurs help?",
        "3. Does residue-level replace global aggregation?",
        "4. Are global and local HSP complementary?",
        "",
        "Do not compare multiple injection mechanisms in the first residue batch.",
        "",
    ]
    (RES / "HSP_RESIDUE_LEVEL_EXPERIMENT_DESIGN.md").write_text("\n".join(design))

    print("Wrote audit artifacts to", RES, flush=True)
    print("BM", class_bm, "EIS", class_eis, "residue_justified", spec_doc["residue_level_justified"], flush=True)
    print("next HIC", next_code("HIC"), "unused OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
