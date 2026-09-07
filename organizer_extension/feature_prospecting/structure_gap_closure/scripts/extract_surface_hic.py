#!/usr/bin/env python3
"""Task B: continuous SAS-sampled hydrophobic patch features (FreeSASA)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    ARO,
    CACHE,
    CROSSWALK,
    DENSITY,
    LINK,
    MAX_POINTS,
    MIN_SASA,
    PREP_FAB,
    PROBE,
    RAW_FAB,
    RESULTS,
    SCALES,
    VDW,
    aa1,
    connected_components,
    domain_of,
    fibonacci_sphere,
    load_cdr,
    load_crosswalk,
    load_fab_meta,
    load_hl_seqs,
    map_chains,
    write_heavy_tmp,
)

import freesasa  # noqa: E402


def build_surface(pdb_path: Path, heavy: str, light: str, cdr_rows, vh_len: int, vl_len: int, variable_only: bool):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    pdb_to_hl, err = map_chains(model, heavy, light)
    if err:
        return None, err
    cdr_set = set()
    fw_set = set()
    for r in cdr_rows:
        key = (str(r["chain"]), int(r["sequence_index"]))
        if str(r.get("is_cdr", "")).lower() in ("1", "true", "t", "yes") or r.get("is_cdr") is True:
            cdr_set.add(key)
        if str(r.get("is_framework", "")).lower() in ("1", "true", "t", "yes") or r.get("is_framework") is True:
            fw_set.add(key)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_heavy_tmp(structure, tmp_path)
        heavy_st = parser.get_structure("h", str(tmp_path))
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
                dom = domain_of(hl, seq_i, vh_len, vl_len)
                if variable_only and dom not in ("VH", "VL"):
                    seq_i += 1
                    continue
                key = (hl, seq_i)
                for atom in res.get_atoms():
                    elem = atom.element.strip().upper()
                    if elem in ("H", ""):
                        continue
                    meta.append(
                        {
                            "coord": atom.coord.astype(float).copy(),
                            "element": elem,
                            "aa": aa,
                            "chain": hl,
                            "seq_i": seq_i,
                            "domain": dom,
                            "is_cdr": key in cdr_set,
                            "is_fw": key in fw_set or (dom in ("VH", "VL") and key not in cdr_set),
                            "r_vdw": VDW.get(elem, 1.7),
                        }
                    )
                seq_i += 1
        if not meta:
            return None, "no_atoms"
        fs = freesasa.Structure(str(tmp_path))
        # When variable_only we still computed full structure SASA then filter meta —
        # recompute on filtered atoms by rewriting PDB of selected atoms only.
        if variable_only:
            # rebuild temp with only variable residues
            from Bio.PDB import PDBIO
            from Bio.PDB.StructureBuilder import StructureBuilder

            # simpler: compute full then keep atoms whose seq in VH/VL via rematch
            result = freesasa.calc(
                fs, freesasa.Parameters({"algorithm": freesasa.LeeRichards, "probe-radius": PROBE})
            )
            # full-atom meta for matching FreeSASA order
            meta_full = []
            for ch in heavy_model.get_chains():
                if ch.id not in pdb_to_hl2:
                    continue
                hl = pdb_to_hl2[ch.id]
                seq_i = 0
                for res in ch.get_residues():
                    if not is_aa(res, standard=True):
                        continue
                    aa = aa1(res.get_resname())
                    dom = domain_of(hl, seq_i, vh_len, vl_len)
                    key = (hl, seq_i)
                    for atom in res.get_atoms():
                        elem = atom.element.strip().upper()
                        if elem in ("H", ""):
                            continue
                        meta_full.append(
                            {
                                "coord": atom.coord.astype(float).copy(),
                                "element": elem,
                                "aa": aa,
                                "chain": hl,
                                "seq_i": seq_i,
                                "domain": dom,
                                "is_cdr": key in cdr_set,
                                "is_fw": key in fw_set or (dom in ("VH", "VL") and key not in cdr_set),
                                "r_vdw": VDW.get(elem, 1.7),
                                "keep": dom in ("VH", "VL"),
                            }
                        )
                    seq_i += 1
            if fs.nAtoms() != len(meta_full):
                return None, f"freesasa_mismatch:{fs.nAtoms()}!={len(meta_full)}"
            for i, m in enumerate(meta_full):
                m["sasa"] = float(result.atomArea(i))
            meta = [m for m in meta_full if m["keep"]]
        else:
            result = freesasa.calc(
                fs, freesasa.Parameters({"algorithm": freesasa.LeeRichards, "probe-radius": PROBE})
            )
            if fs.nAtoms() != len(meta):
                return None, f"freesasa_mismatch:{fs.nAtoms()}!={len(meta)}"
            for i, m in enumerate(meta):
                m["sasa"] = float(result.atomArea(i))
    finally:
        tmp_path.unlink(missing_ok=True)

    atom_coords = np.array([m["coord"] for m in meta], float)
    atom_radii = np.array([m["r_vdw"] + PROBE for m in meta], float)
    points, areas, attrs = [], [], []
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
                attrs.append(m)
    if not points:
        return None, "no_surface_points"
    pts = np.asarray(points, float)
    areas_a = np.asarray(areas, float)
    if len(pts) > MAX_POINTS:
        idx = np.linspace(0, len(pts) - 1, MAX_POINTS).astype(int)
        pts, areas_a = pts[idx], areas_a[idx]
        attrs = [attrs[i] for i in idx]
    return {"points": pts, "areas": areas_a, "attrs": attrs}, None


def patch_features(surf: dict, scale_id: str) -> dict:
    scale, thr = SCALES[scale_id]
    pts = surf["points"]
    areas = surf["areas"]
    attrs = surf["attrs"]
    vals = np.array([scale.get(a["aa"], 0.0) for a in attrs], float)
    hydro = vals > thr
    aro = np.array([a["aa"] in ARO for a in attrs], bool)
    tyr = np.array([a["aa"] == "Y" for a in attrs], bool)
    phe = np.array([a["aa"] == "F" for a in attrs], bool)
    trp = np.array([a["aa"] == "W" for a in attrs], bool)
    cdr = np.array([a["is_cdr"] for a in attrs], bool)
    fw = np.array([a["is_fw"] for a in attrs], bool)

    hydro_area = float(areas[hydro].sum())
    comps = connected_components(hydro, pts, LINK)
    comps = sorted(comps, key=lambda c: areas[c].sum(), reverse=True)
    n_patches = len(comps)
    a1 = float(areas[comps[0]].sum()) if comps else 0.0
    a2 = float(areas[comps[1]].sum()) if len(comps) > 1 else 0.0
    largest = comps[0] if comps else []

    # perimeter proxy: edges from hydro points to non-hydro neighbors within LINK
    perim = 0.0
    if largest:
        L = set(largest)
        for i in largest:
            for j in range(len(pts)):
                if j in L:
                    continue
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d <= LINK:
                    perim += d
        perim *= 0.5
    compact = (perim ** 2) / (4.0 * np.pi * a1) if a1 > 1e-9 else np.nan
    aro_in = float(areas[[i for i in largest if aro[i]]].sum()) if largest else 0.0
    out = {
        f"{scale_id}__hydro_area_total": hydro_area,
        f"{scale_id}__patch_area_max": a1,
        f"{scale_id}__patch_area_2nd": a2,
        f"{scale_id}__patch_area_max_over_total": (a1 / hydro_area) if hydro_area > 1e-9 else 0.0,
        f"{scale_id}__n_patches": float(n_patches),
        f"{scale_id}__fragmentation": float(n_patches / max(hydro_area / 100.0, 1e-6)),
        f"{scale_id}__perimeter_max": perim,
        f"{scale_id}__compactness_max": float(compact) if np.isfinite(compact) else np.nan,
        f"{scale_id}__aro_area_in_max": aro_in,
        f"{scale_id}__aro_frac_max": (aro_in / a1) if a1 > 1e-9 else 0.0,
        f"{scale_id}__tyr_area_in_max": float(areas[[i for i in largest if tyr[i]]].sum()) if largest else 0.0,
        f"{scale_id}__phe_area_in_max": float(areas[[i for i in largest if phe[i]]].sum()) if largest else 0.0,
        f"{scale_id}__trp_area_in_max": float(areas[[i for i in largest if trp[i]]].sum()) if largest else 0.0,
        f"{scale_id}__cdr_frac_max": float(areas[[i for i in largest if cdr[i]]].sum() / a1) if a1 > 1e-9 else 0.0,
        f"{scale_id}__fw_frac_max": float(areas[[i for i in largest if fw[i]]].sum() / a1) if a1 > 1e-9 else 0.0,
    }
    return out


def _one_job(job):
    ab_id, pdb, heavy, light, cdr_rows, vh_len, vl_len, variable_only, tag = job
    if not Path(pdb).exists():
        return ab_id, tag, None, "missing_pdb"
    surf, err = build_surface(Path(pdb), heavy, light, cdr_rows, vh_len, vl_len, variable_only)
    if err:
        return ab_id, tag, None, err
    feats = {}
    for sid in SCALES:
        feats.update(patch_features(surf, sid))
    feats["id"] = ab_id
    feats["structure_tag"] = tag
    return ab_id, tag, feats, None


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    fab = load_fab_meta()
    seqs = load_hl_seqs()
    cdr = load_cdr()
    cw = load_crosswalk()
    jobs = []

    # Fv generators (PRIMARY for HIC)
    for gen, col in [
        ("esmfold", "esmfold_canonical_path"),
        ("abodybuilder2", "abodybuilder2_path"),
        ("boltz2", "boltz2_pdb_path"),
    ]:
        for ab_id in fab.index.astype(str):
            if ab_id not in seqs.index or ab_id not in cw.index:
                continue
            pdb = cw.loc[ab_id, col]
            if not isinstance(pdb, str) or not Path(pdb).exists():
                continue
            row = fab.loc[ab_id]
            jobs.append(
                (
                    ab_id,
                    pdb,
                    str(seqs.loc[ab_id, "heavy"]),
                    str(seqs.loc[ab_id, "light"]),
                    cdr.get(ab_id, []),
                    int(row.VH_len_used),
                    int(row.VL_len_used),
                    False,
                    f"fv_{gen}",
                )
            )

    # Fab raw full + variable-in-Fab
    for ab_id in fab.index.astype(str):
        if ab_id not in seqs.index:
            continue
        raw = RAW_FAB / f"{ab_id}.pdb"
        prep = PREP_FAB / f"{ab_id}_prepared.pdb"
        row = fab.loc[ab_id]
        heavy_fab = str(row.heavy_fab_seq)
        light_fab = str(row.light_fab_seq)
        base = (
            ab_id,
            heavy_fab,
            light_fab,
            cdr.get(ab_id, []),
            int(row.VH_len_used),
            int(row.VL_len_used),
        )
        if raw.exists():
            jobs.append((ab_id, str(raw), heavy_fab, light_fab, cdr.get(ab_id, []), int(row.VH_len_used), int(row.VL_len_used), False, "fab_raw_full"))
            jobs.append((ab_id, str(raw), heavy_fab, light_fab, cdr.get(ab_id, []), int(row.VH_len_used), int(row.VL_len_used), True, "fab_raw_var"))
        if prep.exists():
            jobs.append((ab_id, str(prep), heavy_fab, light_fab, cdr.get(ab_id, []), int(row.VH_len_used), int(row.VL_len_used), False, "fab_prep_full"))
            jobs.append((ab_id, str(prep), heavy_fab, light_fab, cdr.get(ab_id, []), int(row.VH_len_used), int(row.VL_len_used), True, "fab_prep_var"))

    print(f"surface jobs={len(jobs)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    errs = []
    workers = int(os.environ.get("GAP_WORKERS", "6"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one_job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            ab_id, tag, feats, err = fut.result()
            if err:
                errs.append({"id": ab_id, "tag": tag, "error": err})
            else:
                rows.append(feats)
            if k % 50 == 0:
                print(f"  surface {k}/{len(jobs)} ok={len(rows)} err={len(errs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "HIC_CONTINUOUS_SURFACE_RAW.csv", index=False)
    pd.DataFrame(errs).to_csv(CACHE / "HIC_CONTINUOUS_SURFACE_ERRORS.csv", index=False)

    # Wide primary tables
    pieces = []
    for tag, name in [
        ("fv_esmfold", "fv_esmfold"),
        ("fv_abodybuilder2", "fv_abb2"),
        ("fv_boltz2", "fv_boltz2"),
        ("fab_raw_full", "fab_raw"),
        ("fab_raw_var", "fab_raw_var"),
        ("fab_prep_full", "fab_prep"),
        ("fab_prep_var", "fab_prep_var"),
    ]:
        sub = df[df.structure_tag == tag].drop(columns=["structure_tag"])
        if sub.empty:
            continue
        ren = {c: f"{name}__{c}" for c in sub.columns if c != "id"}
        pieces.append(sub.rename(columns=ren).set_index("id"))
    wide = pieces[0]
    for p in pieces[1:]:
        wide = wide.join(p, how="outer")
    # context deltas: fab_raw_var - fv_esmfold
    feat_cols = [c for c in df.columns if c not in ("id", "structure_tag")]
    fv = df[df.structure_tag == "fv_esmfold"].set_index("id")
    fabv = df[df.structure_tag == "fab_raw_var"].set_index("id")
    ids = sorted(set(fv.index) & set(fabv.index))
    delta_rows = []
    for i in ids:
        row = {"id": i}
        for c in feat_cols:
            row[f"delta_ctx__{c}"] = float(fabv.loc[i, c]) - float(fv.loc[i, c])
        delta_rows.append(row)
    delta = pd.DataFrame(delta_rows).set_index("id")
    wide = wide.join(delta, how="left")
    wide.reset_index().to_csv(RESULTS / "HIC_CONTINUOUS_SURFACE_FEATURES.csv", index=False)
    print("wrote HIC_CONTINUOUS_SURFACE_FEATURES", wide.shape, "errors", len(errs), flush=True)


if __name__ == "__main__":
    main()
