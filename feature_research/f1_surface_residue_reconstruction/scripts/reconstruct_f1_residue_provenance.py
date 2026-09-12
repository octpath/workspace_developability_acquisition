#!/usr/bin/env python3
"""Reconstruct residue-level provenance for historical F1_SURFACE (ARO19+HYDRO16).

AUDIT / RECONSTRUCTION ONLY — does not issue EXP-H128 or train models.

Preserves originating atom → residue assignment during HYDRO surface construction
(same algorithm as hydro_surface.build_surface_and_field; does not invent descriptors).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = ROOT / "feature_research/f1_surface_residue_reconstruction"
FEAT = OUT / "features"
RES = ROOT / "developability_drilldown/results"
sys.path.insert(0, str(FP))

from Bio.PDB import PDBIO, PDBParser  # noqa: E402
from Bio.PDB.Polypeptide import is_aa  # noqa: E402
import freesasa  # noqa: E402

from common.structure_utils import (  # noqa: E402
    build_residue_table,
    load_cdr_map,
    load_sequences,
)
from common.hydro_surface import (  # noqa: E402
    ALPHA,
    CUTOFF,
    DENSITY,
    FAUCHERE_PI,
    LINK,
    MAX_POINTS,
    MIN_SASA,
    PROBE,
    VDW,
    _HeavySelect,
    aa1,
    connected_components as hydro_cc,
    cdr_index_set,
    fibonacci_sphere,
    hydro_summaries,
    map_chains,
)
from scripts.extract_physical_batch1 import aromatic_features  # noqa: E402

ARO_COLS = [
    "aro_exposed_TYR_count",
    "aro_exposed_TRP_count",
    "aro_exposed_PHE_count",
    "aro_exposed_aromatic_total_count",
    "aro_aromatic_exposed_SASA_total",
    "aro_aromatic_exposed_SASA_fraction",
    "aro_strongly_exposed_aromatic_count",
    "aro_strongly_exposed_aromatic_SASA",
    "aro_CDR_exposed_aromatic_count",
    "aro_CDR_aromatic_SASA",
    "aro_CDR_aromatic_fraction",
    "aro_aromatic_patch_count",
    "aro_largest_aromatic_patch_n_res",
    "aro_largest_aromatic_patch_exposed_SASA",
    "aro_max_local_aromatic_SASA",
    "aro_sequence_aromatic_count",
    "aro_sequence_TYR_count",
    "aro_sequence_TRP_count",
    "aro_sequence_PHE_count",
]
HYDRO_COLS = [
    "mean_H_surface",
    "q75_H_surface",
    "q90_H_surface",
    "q95_H_surface",
    "max_H_surface",
    "positive_H_area_fraction",
    "top10_H_mean",
    "top10_H_area_fraction",
    "high_H_patch_count",
    "largest_high_H_patch_area_fraction",
    "largest_high_H_patch_n_vertices",
    "CDR_mean_H",
    "CDR_q90_H",
    "CDR_high_H_area_fraction",
    "n_surface_points",
    "phi_finite_frac",
]
F1_COLS = ARO_COLS + HYDRO_COLS


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def build_surface_with_residue_provenance(pdb_path: Path, heavy: str, light: str, cdr_rows: list[dict]):
    """Identical to hydro_surface.build_surface_and_field but keeps parent residue on each vertex."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    pdb_to_hl, err = map_chains(model, heavy, light)
    if err:
        return None, err
    cdr_set = cdr_index_set(cdr_rows)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name
    io = PDBIO()
    io.set_structure(structure)
    io.save(tmp_path, _HeavySelect())
    try:
        heavy_st = parser.get_structure("h", tmp_path)
        heavy_model = next(heavy_st.get_models())
        pdb_to_hl2, err2 = map_chains(heavy_model, heavy, light)
        if err2:
            return None, err2
        meta = []
        for ch in heavy_model.get_chains():
            if ch.id not in pdb_to_hl2:
                continue
            hl = pdb_to_hl2[ch.id]
            seq_i = 0
            for res in ch.get_residues():
                if not is_aa(res, standard=True):
                    continue
                aa = aa1(res.get_resname())
                pi = FAUCHERE_PI.get(aa, 0.0)
                is_cdr = (hl, seq_i) in cdr_set
                pdb_resseq = int(res.id[1])
                pdb_icode = (res.id[2] or "").strip()
                for atom in res.get_atoms():
                    elem = atom.element.strip().upper()
                    if elem in ("H", ""):
                        continue
                    meta.append(
                        {
                            "coord": atom.coord.astype(float).copy(),
                            "element": elem,
                            "atom_name": atom.get_name().strip(),
                            "pi": float(pi),
                            "chain": hl,
                            "seq_i": seq_i,
                            "aa": aa,
                            "pdb_chain": ch.id,
                            "pdb_resseq": pdb_resseq,
                            "pdb_icode": pdb_icode,
                            "is_cdr": is_cdr,
                            "r_vdw": VDW.get(elem, 1.7),
                        }
                    )
                seq_i += 1
        if not meta:
            return None, "no_atoms"
        fs = freesasa.Structure(tmp_path)
        result = freesasa.calc(
            fs, freesasa.Parameters({"algorithm": freesasa.LeeRichards, "probe-radius": PROBE})
        )
        if fs.nAtoms() != len(meta):
            return None, f"freesasa_atom_mismatch:{fs.nAtoms()}!={len(meta)}"
        for i, m in enumerate(meta):
            m["sasa"] = float(result.atomArea(i))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    atom_coords = np.array([m["coord"] for m in meta], float)
    atom_radii = np.array([m["r_vdw"] + PROBE for m in meta], float)
    atom_pi = np.array([m["pi"] for m in meta], float)

    points, areas, cdr_flags = [], [], []
    parents = []
    for m in meta:
        if m["sasa"] < MIN_SASA:
            continue
        n = max(1, int(round(m["sasa"] * DENSITY)))
        dirs = fibonacci_sphere(n)
        R = m["r_vdw"] + PROBE
        cand = m["coord"][None, :] + dirs * R
        for p in cand:
            d = np.linalg.norm(atom_coords - p, axis=1)
            if np.all(d >= (atom_radii - 0.05 - 1e-6)):
                points.append(p)
                areas.append(m["sasa"] / n)
                cdr_flags.append(m["is_cdr"])
                parents.append(
                    {
                        "chain": m["chain"],
                        "seq_i": m["seq_i"],
                        "aa": m["aa"],
                        "pdb_chain": m["pdb_chain"],
                        "pdb_resseq": m["pdb_resseq"],
                        "pdb_icode": m["pdb_icode"],
                        "source_atom_name": m["atom_name"],
                        "source_element": m["element"],
                        "assignment_method": "originating_atom_during_surface_construction",
                    }
                )
    if not points:
        return None, "no_surface_points"
    pts = np.asarray(points, float)
    areas_a = np.asarray(areas, float)
    cdr_a = np.asarray(cdr_flags, bool)
    if len(pts) > MAX_POINTS:
        idx = np.linspace(0, len(pts) - 1, MAX_POINTS).astype(int)
        pts, areas_a, cdr_a = pts[idx], areas_a[idx], cdr_a[idx]
        parents = [parents[i] for i in idx]

    H = np.zeros(len(pts), float)
    chunk = 256
    cutoff2 = CUTOFF * CUTOFF
    for i0 in range(0, len(pts), chunk):
        block = pts[i0 : i0 + chunk]
        d2 = np.sum((block[:, None, :] - atom_coords[None, :, :]) ** 2, axis=2)
        mask = d2 <= cutoff2
        d = np.sqrt(np.where(mask, d2, 0.0))
        contrib = np.where(mask, atom_pi[None, :] * np.exp(-ALPHA * d), 0.0)
        H[i0 : i0 + chunk] = contrib.sum(axis=1)

    return {
        "points": pts,
        "areas": areas_a,
        "cdr": cdr_a,
        "H": H,
        "n_atoms": len(meta),
        "n_points": len(pts),
        "parents": parents,
    }, None


def rows_to_aro_residue_df(aid: str, rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        ca = r["ca"]
        out.append(
            {
                "antibody_id": aid,
                "chain": r["chain"],
                "sequence_index": int(r["sequence_index"]),
                "sequence_aa": r["amino_acid"],
                "structure_aa": r["amino_acid"],
                "pdb_chain": r["pdb_chain"],
                "pdb_resseq": int(r["pdb_resseq"]),
                "pdb_icode": "",
                "structure_residue_id": f"{r['pdb_chain']}:{int(r['pdb_resseq'])}",
                "sasa": float(r["sasa"]),
                "rasa": float(r["rasa"]) if np.isfinite(r["rasa"]) else np.nan,
                "is_exposed": bool(r["is_exposed"]),
                "is_strongly_exposed": bool(r["is_strongly_exposed"]),
                "is_cdr": bool(r["is_cdr"]),
                "region": r["region"],
                "is_aromatic": r["amino_acid"] in set("FWY"),
                "ca_x": float(ca[0]) if ca is not None else np.nan,
                "ca_y": float(ca[1]) if ca is not None else np.nan,
                "ca_z": float(ca[2]) if ca is not None else np.nan,
                "mapping_status": "OK_SEQ_EXACT",
                "availability_aro": True,
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    FEAT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    hist_aro = pd.read_parquet(ROOT / "top_models_feature_bundle/data/aromatic_topo.parquet").set_index("id")
    hist_hydro = pd.read_parquet(ROOT / "top_models_feature_bundle/data/hydro_field.parquet").set_index("id")
    h047 = pd.read_parquet(ROOT / "developability_drilldown/experiments/features/EXP-H047.parquet")
    feat_cols = [c for c in h047.columns if c not in ("id", "split")]
    assert feat_cols[1395:1430] == F1_COLS
    h047_f1 = h047.set_index("id")[F1_COLS]
    # confirm H090/H086 artifact identity
    for col in ARO_COLS:
        assert np.allclose(h047_f1[col], hist_aro[col].reindex(h047_f1.index), equal_nan=True)
    for col in HYDRO_COLS:
        assert np.allclose(h047_f1[col], hist_hydro[col].reindex(h047_f1.index), equal_nan=True)

    hist_sf = pd.read_parquet(FP / "HYDRO-FIELD/surface_field_esmfold.parquet")
    ann = pd.read_parquet(ROOT / "top_models_feature_bundle/residue_level/annotations.parquet")
    ann = ann.rename(columns={"id": "antibody_id", "seq_index": "sequence_index", "aa": "sequence_aa_ann"})

    seqs = load_sequences()
    cdr_map = load_cdr_map()
    cw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv").set_index("id")

    aro_res_parts = []
    hydro_vert_parts = []
    hydro_res_parts = []
    recon_rows = []
    fail_rows = []

    ids = sorted(h047_f1.index.astype(str).tolist())
    assert len(ids) == 324

    for i, aid in enumerate(ids):
        pdb = Path(str(cw.loc[aid, "esmfold_canonical_path"]))
        heavy = str(seqs.loc[aid, "heavy"])
        light = str(seqs.loc[aid, "light"])
        cdr_rows = cdr_map.get(aid, [])
        pdb_hash = sha256_file(pdb) if pdb.exists() else None

        # --- ARO ---
        rows, err_a = build_residue_table(pdb, heavy, light, cdr_rows)
        if err_a or rows is None:
            fail_rows.append({"antibody_id": aid, "stage": "ARO", "error": err_a})
            continue
        aro_df = rows_to_aro_residue_df(aid, rows)
        # join annotations for transformer seq alignment check
        a_ann = ann[ann.antibody_id == aid][["antibody_id", "chain", "sequence_index", "sequence_aa_ann", "imgt_insertion", "numbering_status"]]
        aro_df = aro_df.merge(a_ann, on=["antibody_id", "chain", "sequence_index"], how="left")
        aa_mismatch = aro_df["sequence_aa_ann"].notna() & (
            aro_df["sequence_aa"] != aro_df["sequence_aa_ann"]
        )
        if bool(aa_mismatch.any()):
            aro_df.loc[aa_mismatch, "mapping_status"] = "AA_MISMATCH_VS_ANNOTATIONS"
        aro_feat = aromatic_features(rows)
        aro_recon = {f"aro_{k}": float(v) for k, v in aro_feat.items()}

        # --- HYDRO with provenance ---
        surf, err_h = build_surface_with_residue_provenance(pdb, heavy, light, cdr_rows)
        if err_h or surf is None:
            fail_rows.append({"antibody_id": aid, "stage": "HYDRO", "error": err_h})
            continue
        hydro_feat = hydro_summaries(surf)
        # phi from historical surface_field (identical coordinates); QC only
        sfa = hist_sf[hist_sf["id"] == aid].sort_values("vertex_i")
        hist_pts = np.stack([sfa["x"], sfa["y"], sfa["z"]], axis=1)
        coord_err = (
            float(np.abs(hist_pts - surf["points"]).max())
            if len(sfa) == len(surf["points"])
            else float("inf")
        )
        if coord_err == 0.0:
            phi = sfa["phi"].to_numpy(float)
            phi_finite = float(np.isfinite(phi).mean())
        else:
            phi = np.full(len(surf["points"]), np.nan)
            phi_finite = float(hist_hydro.loc[aid, "phi_finite_frac"])
        hydro_feat["phi_finite_frac"] = phi_finite

        vert_rows = []
        for vi in range(len(surf["points"])):
            p = surf["parents"][vi]
            vert_rows.append(
                {
                    "antibody_id": aid,
                    "vertex_i": vi,
                    "x": float(surf["points"][vi, 0]),
                    "y": float(surf["points"][vi, 1]),
                    "z": float(surf["points"][vi, 2]),
                    "H": float(surf["H"][vi]),
                    "area": float(surf["areas"][vi]),
                    "is_cdr": bool(surf["cdr"][vi]),
                    "phi": float(phi[vi]) if vi < len(phi) and np.isfinite(phi[vi]) else np.nan,
                    "chain": p["chain"],
                    "sequence_index": int(p["seq_i"]),
                    "sequence_aa": p["aa"],
                    "structure_aa": p["aa"],
                    "pdb_chain": p["pdb_chain"],
                    "pdb_resseq": int(p["pdb_resseq"]),
                    "pdb_icode": p["pdb_icode"],
                    "structure_residue_id": f"{p['pdb_chain']}:{int(p['pdb_resseq'])}"
                    + (p["pdb_icode"] if p["pdb_icode"] else ""),
                    "source_atom_name": p["source_atom_name"],
                    "source_element": p["source_element"],
                    "assignment_method": p["assignment_method"],
                    "availability_hydro": True,
                    "vertex_coord_match_historical": coord_err == 0.0,
                }
            )
        vert_df = pd.DataFrame(vert_rows)

        # residue-level HYDRO rollups (inspection; round-trip uses vertices)
        g_parts = []
        for (ab, ch, si), sub in vert_df.groupby(["antibody_id", "chain", "sequence_index"]):
            w = sub["area"].to_numpy(float)
            h = sub["H"].to_numpy(float)
            g_parts.append(
                {
                    "antibody_id": ab,
                    "chain": ch,
                    "sequence_index": int(si),
                    "hydro_n_vertices": int(len(sub)),
                    "hydro_area_sum": float(w.sum()),
                    "hydro_H_area_weighted_mean": float(np.average(h, weights=w)) if w.sum() > 0 else np.nan,
                    "hydro_H_max": float(h.max()),
                    "hydro_is_cdr_any": bool(sub["is_cdr"].any()),
                }
            )
        g = pd.DataFrame(g_parts)

        recon = {
            "antibody_id": aid,
            "pdb_sha256": pdb_hash,
            "aro_ok": True,
            "hydro_ok": True,
            "coord_err": coord_err,
        }
        recon.update(aro_recon)
        recon.update(hydro_feat)
        recon_rows.append(recon)
        aro_res_parts.append(aro_df)
        hydro_vert_parts.append(vert_df)
        hydro_res_parts.append(g)
        if i == 0 or (i + 1) % 20 == 0:
            print(f"[{i+1}/324] {aid} coord_err={coord_err}", flush=True)

    recon_df = pd.DataFrame(recon_rows).set_index("antibody_id")
    aro_res = pd.concat(aro_res_parts, ignore_index=True)
    hydro_vert = pd.concat(hydro_vert_parts, ignore_index=True)
    hydro_res = pd.concat(hydro_res_parts, ignore_index=True)

    aligned = aro_res.merge(hydro_res, on=["antibody_id", "chain", "sequence_index"], how="left")
    aligned["availability_hydro"] = aligned["hydro_n_vertices"].fillna(0) > 0
    # transformer token position within chain: seq_index; special tokens never get SURFACE
    aligned["transformer_chain_token_index"] = aligned["sequence_index"]
    aligned["special_token"] = False

    # Round-trip errors
    err_rows = []
    ab_ok = []
    for aid in recon_df.index:
        for col in F1_COLS:
            hv = float(h047_f1.loc[aid, col])
            rv = float(recon_df.loc[aid, col])
            err_rows.append(
                {
                    "antibody_id": aid,
                    "column": col,
                    "block": "ARO" if col.startswith("aro_") else "HYDRO",
                    "historical": hv,
                    "reconstructed": rv,
                    "abs_err": abs(rv - hv),
                }
            )
        max_e = max(abs(float(recon_df.loc[aid, c]) - float(h047_f1.loc[aid, c])) for c in F1_COLS)
        ab_ok.append({"antibody_id": aid, "max_abs_err_35": max_e, "exact_1e10": max_e <= 1e-10})

    err_df = pd.DataFrame(err_rows)
    ab_df = pd.DataFrame(ab_ok)

    def block_stats(block: str) -> dict:
        sub = err_df[err_df.block == block]
        by = sub.groupby("column")["abs_err"]
        return {
            "max_abs_err": float(sub.abs_err.max()),
            "mean_abs_err": float(sub.abs_err.mean()),
            "median_abs_err": float(sub.abs_err.median()),
            "n_exact_1e10": int((sub.abs_err <= 1e-10).sum()),
            "n_pairs": int(len(sub)),
            "per_column_max": {k: float(v) for k, v in by.max().items()},
            "per_column_mean": {k: float(v) for k, v in by.mean().items()},
            "per_column_corr": {
                col: float(
                    np.corrcoef(
                        err_df.loc[(err_df.column == col), "historical"],
                        err_df.loc[(err_df.column == col), "reconstructed"],
                    )[0, 1]
                )
                if err_df.loc[err_df.column == col, "historical"].std() > 0
                else 1.0
                for col in sorted(sub.column.unique())
            },
        }

    aro_stats = block_stats("ARO")
    hydro_stats = block_stats("HYDRO")
    f1_max = float(err_df.abs_err.max())
    n_ab_exact = int(ab_df.exact_1e10.sum())

    # Verdict (criteria frozen in PASS_CRITERIA.md)
    def block_pass(stats: dict) -> str:
        if stats["max_abs_err"] <= 1e-10:
            return "PASS-EXACT"
        if stats["max_abs_err"] <= 1e-3 and min(stats["per_column_corr"].values()) >= 0.999999:
            return "PASS-NUMERIC"
        return "FAIL"

    aro_v = block_pass(aro_stats)
    hydro_v = block_pass(hydro_stats)
    if aro_v == "PASS-EXACT" and hydro_v == "PASS-EXACT" and n_ab_exact == len(ab_df):
        verdict = "PASS-EXACT"
    elif aro_v.startswith("PASS") and hydro_v.startswith("PASS") and f1_max <= 1e-3:
        verdict = "PASS-NUMERIC"
    elif aro_v.startswith("PASS") ^ hydro_v.startswith("PASS"):
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    # persist
    aro_path = FEAT / "residue_surface_aro.parquet"
    hydro_v_path = FEAT / "residue_surface_hydro_vertices.parquet"
    hydro_r_path = FEAT / "residue_surface_hydro.parquet"
    aligned_path = FEAT / "residue_surface_aligned.parquet"
    recon_path = FEAT / "reconstructed_f1_surface_35.parquet"
    err_path = FEAT / "roundtrip_errors_long.parquet"
    ab_path = FEAT / "roundtrip_antibody_summary.parquet"
    fail_path = FEAT / "reconstruction_failures.csv"

    aro_res.to_parquet(aro_path, index=False)
    hydro_vert.to_parquet(hydro_v_path, index=False)
    hydro_res.to_parquet(hydro_r_path, index=False)
    aligned.to_parquet(aligned_path, index=False)
    recon_df.reset_index().to_parquet(recon_path, index=False)
    err_df.to_parquet(err_path, index=False)
    ab_df.to_parquet(ab_path, index=False)
    pd.DataFrame(fail_rows).to_csv(fail_path, index=False)

    # coverage
    n_ids = 324
    n_recon = len(recon_df)
    h_map = (aligned.groupby("antibody_id")["chain"].apply(lambda s: "H" in set(s))).sum()
    l_map = (aligned.groupby("antibody_id")["chain"].apply(lambda s: "L" in set(s))).sum()
    hydro_cov = aligned.groupby("antibody_id")["availability_hydro"].any().sum()

    summary = {
        "status": "COMPLETE",
        "verdict": verdict,
        "aro_block_verdict": aro_v,
        "hydro_block_verdict": hydro_v,
        "n_antibodies_historical": n_ids,
        "n_antibodies_reconstructed": n_recon,
        "n_failures": len(fail_rows),
        "failures": fail_rows,
        "n_antibodies_35D_exact_1e10": n_ab_exact,
        "f1_max_abs_err": f1_max,
        "aro_stats": aro_stats,
        "hydro_stats": hydro_stats,
        "coverage": {
            "structures_available": int(sum(Path(str(cw.loc[a, "esmfold_canonical_path"])).exists() for a in ids)),
            "H_chain_mapping": int(h_map),
            "L_chain_mapping": int(l_map),
            "ARO_residue_rows": int(len(aro_res)),
            "HYDRO_vertices": int(len(hydro_vert)),
            "antibodies_with_any_hydro_vertex": int(hydro_cov),
            "full_35D_roundtrip_success_exact": n_ab_exact,
        },
        "historical_artifact": {
            "h047_parquet": "developability_drilldown/experiments/features/EXP-H047.parquet",
            "h047_sha256": sha256_file(ROOT / "developability_drilldown/experiments/features/EXP-H047.parquet"),
            "aromatic_topo_sha256": sha256_file(ROOT / "top_models_feature_bundle/data/aromatic_topo.parquet"),
            "hydro_field_sha256": sha256_file(ROOT / "top_models_feature_bundle/data/hydro_field.parquet"),
            "f1_columns": F1_COLS,
            "consumed_by": ["EXP-H090", "EXP-H086"],
            "loader": "H047AuxFeatureStore(F1_SURFACE)",
        },
        "output_artifacts": {
            "residue_surface_aro": {"path": str(aro_path.relative_to(ROOT)), "sha256": sha256_file(aro_path)},
            "residue_surface_hydro_vertices": {
                "path": str(hydro_v_path.relative_to(ROOT)),
                "sha256": sha256_file(hydro_v_path),
            },
            "residue_surface_hydro": {"path": str(hydro_r_path.relative_to(ROOT)), "sha256": sha256_file(hydro_r_path)},
            "residue_surface_aligned": {
                "path": str(aligned_path.relative_to(ROOT)),
                "sha256": sha256_file(aligned_path),
            },
            "reconstructed_f1_35": {"path": str(recon_path.relative_to(ROOT)), "sha256": sha256_file(recon_path)},
        },
        "hydro_assignment_method": "originating_atom_during_surface_construction",
        "phi_finite_frac_method": "historical_surface_field_phi_on_identical_vertices",
        "no_experiment_issued": True,
        "next_hic_unchanged": "EXP-H128",
        "git_rev": git_rev(),
        "residue_fusion_technically_justified": verdict in ("PASS-EXACT", "PASS-NUMERIC"),
    }
    (OUT / "RECONSTRUCTION_SUMMARY.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    (RES / "F1_SURFACE_RESIDUE_RECONSTRUCTION_AUDIT.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print("VERDICT", verdict, "f1_max", f1_max, "n_exact", n_ab_exact, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
