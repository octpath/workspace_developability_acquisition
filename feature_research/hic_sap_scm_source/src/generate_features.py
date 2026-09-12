#!/usr/bin/env python3
"""Generate residue-level SAP/SCM scores and antibody-level SOURCE24+extensions."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
FEAT = TRACK / "features"
RES = TRACK / "results"
ROOT = TRACK.parents[1]
HSP_CACHE = (
    ROOT
    / "feature_research/hic_spatial_hydrophobicity/features/cache/residue_geometry_sasa.parquet"
)
HSP_META = (
    ROOT
    / "feature_research/hic_spatial_hydrophobicity/features/cache/residue_geometry_meta.json"
)
HSP_AB = (
    ROOT
    / "feature_research/hic_spatial_hydrophobicity/features/antibody_spatial_hydrophobicity.parquet"
)
FP = ROOT / "organizer_extension/feature_prospecting"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FP))

from aggregation_engine import (  # noqa: E402
    SAP_EXTRA20_NAMES,
    SAP_GLOBAL6_NAMES,
    SCM_EXTRA20_NAMES,
    SCM_GLOBAL6_NAMES,
    SOURCE_RADII_A,
    SOURCE_REGIONS,
    SOURCE_SAP24_NAMES,
    SOURCE_SCM24_NAMES,
    SOURCE_STATS,
    EXTENSION_REGIONS,
    EXTENSION_STATS,
    aggregate_block,
    local_scores,
    pairwise_centroid,
    region_masks,
)
from properties import (  # noqa: E402
    AA20,
    CHARGE,
    CHARGE_HASH,
    KD_NORM,
    KD_NORM_HASH,
    KD_RAW,
    KD_RAW_HASH,
    TIEN_MAXASA,
    TIEN_MAXASA_HASH,
)
from common.structure_utils import load_cdr_map, load_sequences  # noqa: E402


def _seq_hash(heavy: str, light: str) -> str:
    return hashlib.sha256(f"{heavy}|{light}".encode()).hexdigest()


def generate() -> None:
    FEAT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    geo = pd.read_parquet(HSP_CACHE)
    meta = json.loads(HSP_META.read_text())
    assert geo["id"].nunique() == 324, geo["id"].nunique()

    seqs = load_sequences()
    cdr_map = load_cdr_map()
    seq_hashes = {
        aid: _seq_hash(str(seqs.loc[aid, "heavy"]), str(seqs.loc[aid, "light"]))
        for aid in seqs.index.astype(str)
    }

    residue_rows: list[dict] = []
    ab_rows: list[dict] = []

    for aid, sub in geo.groupby("id", sort=True):
        sub = sub.sort_values(["chain", "residue_index"]).reset_index(drop=True)
        centroids = sub[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(float)
        aa = sub["aa"].tolist()
        chain = sub["chain"].tolist()
        is_cdr = sub["is_cdr"].to_numpy(bool)
        masks = region_masks(chain, is_cdr)
        for reg in SOURCE_REGIONS:
            if int(masks[reg].sum()) == 0:
                raise SystemExit(
                    f"STOP FEATURE GENERATION: empty region {reg} for antibody {aid}"
                )

        kd_raw = np.asarray([KD_RAW.get(a, np.nan) for a in aa], float)
        kd_norm = np.asarray([KD_NORM.get(a, np.nan) for a in aa], float)
        charge = np.asarray([CHARGE.get(a, 0.0) if a in AA20 else np.nan for a in aa], float)
        tien = sub["Tien_MaxASA"].to_numpy(float)
        sasa = sub["total_SASA"].to_numpy(float)
        rasa_raw = np.where(tien > 0, sasa / tien, np.nan)
        rasa_clip = np.clip(rasa_raw, 0.0, 1.0)
        # Prefer cached clipped RASA for consistency with HSP
        rasa_clip = sub["total_rASA_Tien"].to_numpy(float)

        D = pairwise_centroid(centroids)
        bad = ~np.isfinite(centroids).all(axis=1)
        sap = {}
        scm = {}
        for R in SOURCE_RADII_A:
            s = local_scores(D, kd_norm, rasa_clip, R)
            c = local_scores(D, charge, rasa_clip, R)
            s[bad] = np.nan
            c[bad] = np.nan
            sap[R] = s
            scm[R] = c
            if np.isfinite(s).any() and float(np.nanmin(s)) < -1e-9:
                raise RuntimeError(f"negative SAP local for {aid} R={R}")

        # IMGT join
        cdr_rows = {(r["chain"], int(r["sequence_index"])): r for r in cdr_map.get(aid, [])}

        for i in range(len(sub)):
            key = (chain[i], int(sub.loc[i, "residue_index"]))
            cr = cdr_rows.get(key, {})
            imgt = cr.get("imgt_number", None)
            residue_rows.append(
                {
                    "id": aid,
                    "chain": chain[i],
                    "residue_index": int(sub.loc[i, "residue_index"]),
                    "IMGT_position": imgt if imgt is not None else np.nan,
                    "region": sub.loc[i, "region"],
                    "is_cdr": bool(is_cdr[i]),
                    "aa": aa[i],
                    "RASA_raw": float(rasa_raw[i]) if np.isfinite(rasa_raw[i]) else np.nan,
                    "RASA_clip": float(rasa_clip[i]) if np.isfinite(rasa_clip[i]) else np.nan,
                    "KD_raw": float(kd_raw[i]) if np.isfinite(kd_raw[i]) else np.nan,
                    "KD_norm": float(kd_norm[i]) if np.isfinite(kd_norm[i]) else np.nan,
                    "charge": float(charge[i]) if np.isfinite(charge[i]) else np.nan,
                    "SAP_R5": float(sap[5.0][i]),
                    "SAP_R10": float(sap[10.0][i]),
                    "SCM_R5": float(scm[5.0][i]),
                    "SCM_R10": float(scm[10.0][i]),
                    "sequence_hash": seq_hashes[str(aid)],
                    "structure_hash": str(sub.loc[i, "structure_sha256"]),
                    "structure_path": str(sub.loc[i, "structure_path"]),
                    "centroid_prov": str(sub.loc[i, "centroid_prov"]),
                }
            )

        feats: dict = {"id": aid, "sequence_hash": seq_hashes[str(aid)]}
        feats.update(
            aggregate_block(sap, masks, "SAP", SOURCE_REGIONS, SOURCE_RADII_A, SOURCE_STATS)
        )
        feats.update(
            aggregate_block(scm, masks, "SCM", SOURCE_REGIONS, SOURCE_RADII_A, SOURCE_STATS)
        )
        feats.update(
            aggregate_block(sap, masks, "SAP", ("ALL_FV",), SOURCE_RADII_A, SOURCE_STATS)
        )
        feats.update(
            aggregate_block(scm, masks, "SCM", ("ALL_FV",), SOURCE_RADII_A, SOURCE_STATS)
        )
        feats.update(
            aggregate_block(sap, masks, "SAP", EXTENSION_REGIONS, SOURCE_RADII_A, EXTENSION_STATS)
        )
        feats.update(
            aggregate_block(scm, masks, "SCM", EXTENSION_REGIONS, SOURCE_RADII_A, EXTENSION_STATS)
        )
        ab_rows.append(feats)

    res_df = pd.DataFrame(residue_rows)
    ab_df = pd.DataFrame(ab_rows)
    assert ab_df["id"].nunique() == 324
    assert len(SOURCE_SAP24_NAMES) == 24
    assert len(SOURCE_SCM24_NAMES) == 24

    # H094-style GLOBAL3 from existing HSP atlas (do not recompute differently)
    hsp = pd.read_parquet(HSP_AB)[
        [
            "id",
            "LIT1_STATIC_SAP_KD_v1__ALL_FV__MAX",
            "LIT1_STATIC_SAP_KD_v1__ALL_FV__MEAN",
            "LIT1_STATIC_SAP_KD_v1__ALL_FV__SUM",
        ]
    ].rename(
        columns={
            "LIT1_STATIC_SAP_KD_v1__ALL_FV__MAX": "H094_GLOBAL3_MAX",
            "LIT1_STATIC_SAP_KD_v1__ALL_FV__MEAN": "H094_GLOBAL3_MEAN",
            "LIT1_STATIC_SAP_KD_v1__ALL_FV__SUM": "H094_GLOBAL3_SUM",
        }
    )
    ab_df = ab_df.merge(hsp, on="id", how="left")
    assert ab_df[["H094_GLOBAL3_MAX", "H094_GLOBAL3_MEAN", "H094_GLOBAL3_SUM"]].notna().all().all()

    res_path = FEAT / "residue_source_sap_scm.parquet"
    res_df.to_parquet(res_path, index=False)

    def write_block(path: Path, cols: list[str]) -> None:
        out = ab_df[["id"] + cols].copy()
        out.to_parquet(path, index=False)

    write_block(FEAT / "antibody_source_sap24.parquet", SOURCE_SAP24_NAMES)
    write_block(FEAT / "antibody_source_scm24.parquet", SOURCE_SCM24_NAMES)
    write_block(
        FEAT / "antibody_source_sap_scm48.parquet",
        SOURCE_SAP24_NAMES + SOURCE_SCM24_NAMES,
    )
    write_block(
        FEAT / "antibody_source_sap30.parquet",
        SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES,
    )
    write_block(
        FEAT / "antibody_source_scm30.parquet",
        SOURCE_SCM24_NAMES + SCM_GLOBAL6_NAMES,
    )
    write_block(
        FEAT / "antibody_source_sap_scm60.parquet",
        SOURCE_SAP24_NAMES
        + SAP_GLOBAL6_NAMES
        + SOURCE_SCM24_NAMES
        + SCM_GLOBAL6_NAMES,
    )
    write_block(
        FEAT / "antibody_source_sap50.parquet",
        SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES + SAP_EXTRA20_NAMES,
    )
    write_block(
        FEAT / "antibody_source_scm50.parquet",
        SOURCE_SCM24_NAMES + SCM_GLOBAL6_NAMES + SCM_EXTRA20_NAMES,
    )
    write_block(
        FEAT / "antibody_source_sap_scm100.parquet",
        SOURCE_SAP24_NAMES
        + SAP_GLOBAL6_NAMES
        + SAP_EXTRA20_NAMES
        + SOURCE_SCM24_NAMES
        + SCM_GLOBAL6_NAMES
        + SCM_EXTRA20_NAMES,
    )
    # full wide for reconstruction / H094
    ab_df.to_parquet(FEAT / "antibody_source_sap_scm_wide.parquet", index=False)

    provenance = {
        "geometry_cache": str(HSP_CACHE),
        "geometry_meta": meta,
        "structure_scope": "Fv",
        "KD_RAW_HASH": KD_RAW_HASH,
        "KD_NORM_HASH": KD_NORM_HASH,
        "CHARGE_HASH": CHARGE_HASH,
        "TIEN_MAXASA_HASH": TIEN_MAXASA_HASH,
        "n_antibodies": 324,
        "n_residues": len(res_df),
        "SOURCE_SAP24_dim": 24,
        "SOURCE_SCM24_dim": 24,
        "radii_A": list(SOURCE_RADII_A),
        "self_neighbor": "included",
        "neighborhood": "sidechain_centroid",
        "SCM_exposure": "SOURCE_DERIVED_INFERENCE same RASAclip as SAP",
    }
    (RES / "GENERATION_PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print("Wrote features", FEAT, "residues", len(res_df), flush=True)


def reconstruct_from_residue(res_df: pd.DataFrame, aid: str) -> dict[str, float]:
    """Independent reconstruction for tests."""
    sub = res_df[res_df["id"] == aid].sort_values(["chain", "residue_index"])
    chain = sub["chain"].tolist()
    is_cdr = sub["is_cdr"].to_numpy(bool)
    masks = region_masks(chain, is_cdr)
    sap = {5.0: sub["SAP_R5"].to_numpy(float), 10.0: sub["SAP_R10"].to_numpy(float)}
    scm = {5.0: sub["SCM_R5"].to_numpy(float), 10.0: sub["SCM_R10"].to_numpy(float)}
    out: dict[str, float] = {}
    out.update(aggregate_block(sap, masks, "SAP", SOURCE_REGIONS, SOURCE_RADII_A, SOURCE_STATS))
    out.update(aggregate_block(scm, masks, "SCM", SOURCE_REGIONS, SOURCE_RADII_A, SOURCE_STATS))
    out.update(aggregate_block(sap, masks, "SAP", ("ALL_FV",), SOURCE_RADII_A, SOURCE_STATS))
    out.update(aggregate_block(scm, masks, "SCM", ("ALL_FV",), SOURCE_RADII_A, SOURCE_STATS))
    out.update(
        aggregate_block(sap, masks, "SAP", EXTENSION_REGIONS, SOURCE_RADII_A, EXTENSION_STATS)
    )
    out.update(
        aggregate_block(scm, masks, "SCM", EXTENSION_REGIONS, SOURCE_RADII_A, EXTENSION_STATS)
    )
    return out


if __name__ == "__main__":
    generate()
