#!/usr/bin/env python3
"""Build HSP spatial hydrophobicity atlas (antibody + residue) — target-blind."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from geometry_cache import build_or_load_cache, load_antibody_arrays  # noqa: E402
from literature_anchors import all_literature_anchors  # noqa: E402
from scales import UNAVAILABLE, all_scale_ids, scale_hash  # noqa: E402
from spatial_engine import (  # noqa: E402
    AGGS,
    EXPOSURES,
    NEIGHBORHOODS,
    RADII,
    SCOPES,
    TRANSFORMS,
    aggregate,
    exposure_vector,
    family_id,
    feature_name,
    hydro_vector,
    pairwise_centroid,
    pairwise_closest_sc,
    scope_mask,
    spatial_scores,
    spec_hash,
)

ROOT = HERE.parent
FEAT = ROOT / "features"
RES = ROOT / "results"
FEAT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def iter_generic_families():
    for scale in all_scale_ids():
        for transform in TRANSFORMS:
            for exposure in EXPOSURES:
                for neigh in NEIGHBORHOODS:
                    for R in RADII:
                        yield scale, transform, exposure, neigh, R


def main() -> int:
    cache_path = build_or_load_cache()
    geo = pd.read_parquet(cache_path)
    ids = sorted(geo["id"].unique())
    families = list(iter_generic_families())
    family_ids = [family_id(*f) for f in families]
    print(f"antibodies={len(ids)} generic_families={len(families)}", flush=True)

    catalog = []
    for scale, transform, exposure, neigh, R in families:
        fid = family_id(scale, transform, exposure, neigh, R)
        sh = spec_hash(
            scale=scale,
            transform=transform,
            exposure=exposure,
            neighborhood=neigh,
            radius=R,
            scale_table_hash=scale_hash(scale, transform),
            sasa="ShrakeRupley_probe1.4_n100",
            structure="ESMFold_Fv_crosswalk_v2",
            include_self=True,
        )
        for scope in SCOPES:
            for agg in AGGS:
                catalog.append(
                    {
                        "feature_name": feature_name(fid, scope, agg),
                        "family_id": fid,
                        "literature_anchor": False,
                        "scale": scale,
                        "raw_minmax": transform,
                        "exposure": exposure,
                        "neighborhood": neigh,
                        "radius": R,
                        "scope": scope,
                        "aggregation": agg,
                        "source_reference": "generic_spatial_factory_v1",
                        "implementation_version": "HSP_v1",
                        "spec_hash": sh,
                    }
                )

    ab_rows = []
    # residue wide: accumulate per-antibody blocks then concat
    res_blocks = []

    for bi, aid in enumerate(ids):
        ab = load_antibody_arrays(geo, aid)
        n = len(ab["aa"])
        D_cent = pairwise_centroid(ab["centroids"])
        D_sc = pairwise_closest_sc(ab["sc_heavy"])
        row = {"id": aid}
        hydro_cache = {
            (s, t): hydro_vector(ab["aa"], s, t) for s in all_scale_ids() for t in TRANSFORMS
        }
        exp_cache = {e: exposure_vector(ab, e) for e in EXPOSURES}

        res_block = {
            "id": [aid] * n,
            "chain": ab["chain"],
            "residue_index": ab["residue_index"].tolist(),
            "aa": ab["aa"],
            "region": ab["region"],
        }

        for scale, transform, exposure, neigh, R in families:
            fid = family_id(scale, transform, exposure, neigh, R)
            D = D_cent if neigh == "CENTROID" else D_sc
            scores = spatial_scores(D, exp_cache[exposure], hydro_cache[(scale, transform)], R)
            bad = ~np.isfinite(ab["centroids"]).all(axis=1)
            scores = scores.astype(np.float32)
            scores[bad] = np.nan
            res_block[fid] = scores
            for scope in SCOPES:
                ag = aggregate(scores.astype(float), scope_mask(ab, scope))
                for agg, val in ag.items():
                    row[feature_name(fid, scope, agg)] = val

        lit = all_literature_anchors(ab)
        row.update(lit)
        ab_rows.append(row)
        res_blocks.append(pd.DataFrame(res_block))

        if (bi + 1) % 20 == 0:
            print(f"atlas {bi+1}/{len(ids)} n_feats={len(row)-1}", flush=True)

    ab_df = pd.DataFrame(ab_rows)
    ab_path = FEAT / "antibody_spatial_hydrophobicity.parquet"
    ab_df.to_parquet(ab_path, index=False)
    print("Wrote", ab_path, ab_df.shape, flush=True)

    res_df = pd.concat(res_blocks, ignore_index=True)
    # long-form residue table for portability
    id_vars = ["id", "chain", "residue_index", "aa", "region"]
    long = res_df.melt(id_vars=id_vars, value_vars=family_ids, var_name="family_id", value_name="P_i")
    res_path = FEAT / "residue_spatial_hydrophobicity.parquet"
    long.to_parquet(res_path, index=False)
    print("Wrote", res_path, long.shape, flush=True)
    # also keep wide for fast reload
    res_df.to_parquet(FEAT / "residue_spatial_hydrophobicity_wide.parquet", index=False)

    lit_cols = [c for c in ab_df.columns if c.startswith("LIT")]
    for c in lit_cols:
        catalog.append(
            {
                "feature_name": c,
                "family_id": c.split("__")[0],
                "literature_anchor": True,
                "scale": "",
                "raw_minmax": "",
                "exposure": "",
                "neighborhood": "",
                "radius": "",
                "scope": "",
                "aggregation": "",
                "source_reference": "literature_anchor",
                "implementation_version": "HSP_v1",
                "spec_hash": spec_hash(feature=c, version="HSP_v1"),
            }
        )
    cat = pd.DataFrame(catalog)
    cat.to_csv(RES / "FEATURE_CATALOG.csv", index=False)

    reg = []
    for i, fid in enumerate(sorted(cat["family_id"].unique()), start=1):
        reg.append({"hsp_id": f"HSP-F{i:04d}", "family_id": fid, "status": "COMPUTED"})
    pd.DataFrame(reg).to_csv(RES / "feature_screen.csv", index=False)

    meta = {
        "n_antibodies": int(len(ab_df)),
        "n_antibody_features": int(len(ab_df.columns) - 1),
        "n_generic_families": len(families),
        "n_residue_rows_long": int(len(long)),
        "unavailable_scales": UNAVAILABLE,
        "mainline_next_hic": "EXP-H102",
        "consumed_EXP_H102": False,
    }
    (RES / "ATLAS_META.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
