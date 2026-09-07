#!/usr/bin/env python3
"""Tasks C/D/E: Fab packing/cavity, buried-unsat/H-bond, domain interfaces."""
from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    CACHE,
    MAX_ASA,
    PREP_FAB,
    PROBE,
    RASA_BURIED,
    RAW_FAB,
    RESULTS,
    VDW,
    aa1,
    domain_of,
    load_cdr,
    load_fab_meta,
    map_chains,
    write_heavy_tmp,
)

VOXEL = 1.0
MIN_CAV = 15.0
PACK_R = 4.5
HBOND_R = 3.5
SALT_R = 4.0
IFACE_CA = 8.0


DONOR_ATOMS = {"N", "NE", "NH1", "NH2", "ND1", "ND2", "NE1", "NE2", "NZ", "OG", "OG1", "OH", "SG"}
ACCEPTOR_ATOMS = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "ND1", "NE2", "SG"}
NEG = {"ASP", "GLU"}
POS = {"ARG", "LYS", "HIS"}


def _residues(model, pdb_to_hl, vh_len, vl_len, cdr_set):
    rows = []
    for ch in model.get_chains():
        if ch.id not in pdb_to_hl:
            continue
        hl = pdb_to_hl[ch.id]
        seq_i = 0
        for res in ch.get_residues():
            if not is_aa(res, standard=True):
                continue
            aa = aa1(res.get_resname())
            dom = domain_of(hl, seq_i, vh_len, vl_len)
            atoms = []
            for atom in res.get_atoms():
                elem = atom.element.strip().upper()
                atoms.append(
                    {
                        "name": atom.get_name().strip(),
                        "elem": elem,
                        "coord": atom.coord.astype(float).copy(),
                        "is_h": elem == "H",
                    }
                )
            ca = res["CA"].coord.astype(float).copy() if "CA" in res else None
            rows.append(
                {
                    "hl": hl,
                    "seq_i": seq_i,
                    "aa": aa,
                    "res3": res.get_resname().strip().upper(),
                    "domain": dom,
                    "is_cdr": (hl, seq_i) in cdr_set and dom in ("VH", "VL"),
                    "is_fw": dom in ("VH", "VL") and (hl, seq_i) not in cdr_set,
                    "atoms": atoms,
                    "ca": ca,
                    "res": res,
                }
            )
            seq_i += 1
    return rows


def compute_rasa(rows):
    # Bio.PDB SASA already on residues if computed
    for r in rows:
        sasa = float(getattr(r["res"], "sasa", np.nan))
        maxasa = MAX_ASA.get(r["aa"])
        rasa = sasa / maxasa if maxasa and np.isfinite(sasa) else np.nan
        r["sasa"] = sasa
        r["rasa"] = rasa
        r["buried"] = bool(np.isfinite(rasa) and rasa < RASA_BURIED)


def cavity_features(rows):
    heavies = []
    radii = []
    for r in rows:
        for a in r["atoms"]:
            if a["is_h"]:
                continue
            heavies.append(a["coord"])
            radii.append(VDW.get(a["elem"], 1.7))
    if not heavies:
        return {}
    coords = np.asarray(heavies, float)
    radii = np.asarray(radii, float)
    mn = coords.min(axis=0) - 3.0
    mx = coords.max(axis=0) + 3.0
    dims = np.ceil((mx - mn) / VOXEL).astype(int) + 1
    # guard memory
    if int(np.prod(dims)) > 2_500_000:
        # coarsen
        scale = (np.prod(dims) / 2_000_000) ** (1 / 3)
        vox = VOXEL * max(scale, 1.0)
        dims = np.ceil((mx - mn) / vox).astype(int) + 1
    else:
        vox = VOXEL
    nx, ny, nz = map(int, dims)
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    for c, rad in zip(coords, radii):
        ijk = ((c - mn) / vox).astype(int)
        rbin = int(np.ceil(rad / vox))
        for di in range(-rbin, rbin + 1):
            for dj in range(-rbin, rbin + 1):
                for dk in range(-rbin, rbin + 1):
                    ii, jj, kk = ijk[0] + di, ijk[1] + dj, ijk[2] + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        if di * di + dj * dj + dk * dk <= rbin * rbin + 1:
                            occ[ii, jj, kk] = 1
    # exterior flood fill on empty
    from collections import deque

    empty = occ == 0
    exterior = np.zeros_like(occ, dtype=bool)
    q = deque()
    for i in range(nx):
        for j in range(ny):
            for k in (0, nz - 1):
                if empty[i, j, k] and not exterior[i, j, k]:
                    exterior[i, j, k] = True
                    q.append((i, j, k))
            for k in range(nz):
                for ii, jj in ((i, 0), (i, ny - 1), (0, j), (nx - 1, j)):
                    if 0 <= ii < nx and 0 <= jj < ny and empty[ii, jj, k] and not exterior[ii, jj, k]:
                        exterior[ii, jj, k] = True
                        q.append((ii, jj, k))
    # fix: properly seed all boundary empty voxels
    q = deque()
    exterior[:] = False
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1):
                    if empty[i, j, k]:
                        exterior[i, j, k] = True
                        q.append((i, j, k))
    while q:
        i, j, k = q.popleft()
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ii, jj, kk = i + di, j + dj, k + dk
            if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                if empty[ii, jj, kk] and not exterior[ii, jj, kk]:
                    exterior[ii, jj, kk] = True
                    q.append((ii, jj, kk))
    cavity = empty & ~exterior
    # connected components
    visited = np.zeros_like(cavity, dtype=bool)
    vols = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if cavity[i, j, k] and not visited[i, j, k]:
                    q = deque([(i, j, k)])
                    visited[i, j, k] = True
                    n = 0
                    while q:
                        x, y, z = q.popleft()
                        n += 1
                        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                            ii, jj, kk = x + di, y + dj, z + dk
                            if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                                if cavity[ii, jj, kk] and not visited[ii, jj, kk]:
                                    visited[ii, jj, kk] = True
                                    q.append((ii, jj, kk))
                    vol = n * (vox ** 3)
                    if vol >= MIN_CAV:
                        vols.append(vol)
    vols = sorted(vols, reverse=True)
    bbox_vol = float(np.prod(dims) * (vox ** 3))
    total = float(sum(vols))
    feats = {
        "cav_total": total,
        "cav_largest": float(vols[0]) if vols else 0.0,
        "cav_count": float(len(vols)),
        "cav_vol_frac": total / bbox_vol if bbox_vol else 0.0,
    }
    # per-domain: assign cavity voxels to nearest CA domain (approximate)
    # skip expensive per-domain voxel assignment; use packing-domain cavities via residue centers in empty regions — simplified:
    for dom in ("VH", "VL", "CH1", "CL"):
        dcoords = np.array([r["ca"] for r in rows if r["domain"] == dom and r["ca"] is not None], float)
        if len(dcoords) == 0:
            feats[f"cav_{dom}_total"] = 0.0
            feats[f"cav_{dom}_largest"] = 0.0
            continue
        # volume proxy: count cavity voxels within 6A of any domain CA
        # subsample cavity indices
        cav_idx = np.argwhere(cavity)
        if len(cav_idx) == 0:
            feats[f"cav_{dom}_total"] = 0.0
            feats[f"cav_{dom}_largest"] = 0.0
            continue
        if len(cav_idx) > 20000:
            cav_idx = cav_idx[np.linspace(0, len(cav_idx) - 1, 20000).astype(int)]
        pts = mn + (cav_idx + 0.5) * vox
        # nearest domain distance
        # chunk
        near = np.zeros(len(pts), bool)
        for i0 in range(0, len(pts), 500):
            block = pts[i0 : i0 + 500]
            d2 = np.sum((block[:, None, :] - dcoords[None, :, :]) ** 2, axis=2)
            near[i0 : i0 + 500] = d2.min(axis=1) <= 36.0  # 6A
        # inflate by sampling ratio
        ratio = float(np.prod(cavity.shape)) / max(len(np.argwhere(cavity)), 1) if False else 1.0
        n_all = int(cavity.sum())
        frac = float(near.mean()) if len(near) else 0.0
        dom_vol = total * frac
        feats[f"cav_{dom}_total"] = float(dom_vol)
        feats[f"cav_{dom}_largest"] = float(dom_vol)  # proxy
    return feats


def packing_features(rows):
    heavy = []
    owners = []
    for ri, r in enumerate(rows):
        for a in r["atoms"]:
            if a["is_h"]:
                continue
            heavy.append(a["coord"])
            owners.append(ri)
    if not heavy:
        return {}
    coords = np.asarray(heavy, float)
    owners = np.asarray(owners, int)
    dens = []
    dens_meta = []
    # build neighbor lists via coarse grid optional; for N~3000 atoms OK with chunked
    for ri, r in enumerate(rows):
        if not r.get("buried"):
            continue
        my = coords[owners == ri]
        if len(my) == 0:
            continue
        other = coords[owners != ri]
        if len(other) == 0:
            continue
        near = np.zeros(len(other), bool)
        for i0 in range(0, len(my), 16):
            block = my[i0 : i0 + 16]
            d2 = np.min(np.sum((other[:, None, :] - block[None, :, :]) ** 2, axis=2), axis=1)
            near |= d2 <= PACK_R * PACK_R
        dens.append(float(near.sum()))
        dens_meta.append(r)

    def summ(vals, prefix):
        if not vals:
            return {f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan, f"{prefix}_q10": np.nan}
        a = np.asarray(vals, float)
        return {
            f"{prefix}_mean": float(np.mean(a)),
            f"{prefix}_median": float(np.median(a)),
            f"{prefix}_q10": float(np.quantile(a, 0.10)),
        }

    out = {}
    out.update(summ(dens, "pack_fab"))
    for label, pred in [
        ("VH", lambda r: r["domain"] == "VH"),
        ("VL", lambda r: r["domain"] == "VL"),
        ("CH1", lambda r: r["domain"] == "CH1"),
        ("CL", lambda r: r["domain"] == "CL"),
        ("CDR", lambda r: r["is_cdr"]),
        ("FW", lambda r: r["is_fw"]),
    ]:
        vals = [d for d, r in zip(dens, dens_meta) if pred(r)]
        out.update(summ(vals, f"pack_{label}"))
    return out


def _polar_sites(rows):
    donors, acceptors = [], []
    for ri, r in enumerate(rows):
        for a in r["atoms"]:
            if a["is_h"]:
                continue
            nm = a["name"]
            if nm in DONOR_ATOMS or (nm == "N" and r["aa"] != "P"):
                donors.append((ri, a["coord"], nm))
            if nm in ACCEPTOR_ATOMS or nm == "O":
                acceptors.append((ri, a["coord"], nm))
    return donors, acceptors


def hbond_pairs(donors, acceptors, max_r):
    pairs = []
    if not donors or not acceptors:
        return pairs
    dcoord = np.array([d[1] for d in donors], float)
    acoord = np.array([a[1] for a in acceptors], float)
    for i, (ri, _, _) in enumerate(donors):
        d2 = np.sum((acoord - dcoord[i]) ** 2, axis=1)
        for j in np.where(d2 <= max_r * max_r)[0]:
            rj = acceptors[j][0]
            if ri == rj:
                continue
            pairs.append((ri, rj, float(np.sqrt(d2[j]))))
    return pairs


def buried_unsat_and_network(rows):
    donors, acceptors = _polar_sites(rows)
    pairs = hbond_pairs(donors, acceptors, HBOND_R)
    sat_d = set()
    sat_a = set()
    for di_list_i, (ri, _, _) in enumerate(donors):
        pass
    # mark satisfied donors/acceptors by residue-atom index
    donor_sat = np.zeros(len(donors), bool)
    acc_sat = np.zeros(len(acceptors), bool)
    dcoord = np.array([d[1] for d in donors], float) if donors else np.zeros((0, 3))
    acoord = np.array([a[1] for a in acceptors], float) if acceptors else np.zeros((0, 3))
    for i in range(len(donors)):
        if len(acoord) == 0:
            break
        d2 = np.sum((acoord - dcoord[i]) ** 2, axis=1)
        js = np.where(d2 <= HBOND_R * HBOND_R)[0]
        for j in js:
            if donors[i][0] == acceptors[j][0]:
                continue
            donor_sat[i] = True
            acc_sat[j] = True
    # buried unsatisfied
    unsat_d = unsat_a = 0
    per_dom = {d: {"d": 0, "a": 0} for d in ("VH", "VL", "CH1", "CL")}
    for i, (ri, _, _) in enumerate(donors):
        r = rows[ri]
        if r.get("buried") and not donor_sat[i]:
            unsat_d += 1
            per_dom[r["domain"]]["d"] += 1
    for j, (rj, _, _) in enumerate(acceptors):
        r = rows[rj]
        if r.get("buried") and not acc_sat[j]:
            unsat_a += 1
            per_dom[r["domain"]]["a"] += 1
    nres = max(len(rows), 1)
    feats = {
        "buried_unsat_donor_count": float(unsat_d),
        "buried_unsat_acceptor_count": float(unsat_a),
        "buried_unsat_total": float(unsat_d + unsat_a),
        "buried_unsat_per_100_residues": float(100.0 * (unsat_d + unsat_a) / nres),
        "hbond_count": float(len(pairs)),
    }
    for d in ("VH", "VL", "CH1", "CL"):
        feats[f"buried_unsat_donor_{d}"] = float(per_dom[d]["d"])
        feats[f"buried_unsat_acceptor_{d}"] = float(per_dom[d]["a"])
        feats[f"buried_unsat_total_{d}"] = float(per_dom[d]["d"] + per_dom[d]["a"])

    # salt bridges
    neg_atoms, pos_atoms = [], []
    for ri, r in enumerate(rows):
        if r["res3"] in NEG:
            for a in r["atoms"]:
                if a["name"] in ("OD1", "OD2", "OE1", "OE2"):
                    neg_atoms.append((ri, a["coord"], r["domain"]))
        if r["res3"] in POS:
            for a in r["atoms"]:
                if a["name"] in ("NZ", "NH1", "NH2", "NE", "ND1", "NE2"):
                    pos_atoms.append((ri, a["coord"], r["domain"]))
    sb = 0
    if neg_atoms and pos_atoms:
        nc = np.array([a[1] for a in neg_atoms], float)
        pc = np.array([a[1] for a in pos_atoms], float)
        for i in range(len(nc)):
            d2 = np.sum((pc - nc[i]) ** 2, axis=1)
            sb += int(np.any(d2 <= SALT_R * SALT_R))
    feats["salt_bridge_count"] = float(sb)

    # H-bond network connected components on residues
    adj = defaultdict(set)
    for ri, rj, _ in pairs:
        adj[ri].add(rj)
        adj[rj].add(ri)
    seen = set()
    comps = []
    for i in range(len(rows)):
        if i in seen or i not in adj:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    feats["hbond_largest_cc"] = float(max((len(c) for c in comps), default=0))
    deg = [len(adj[i]) for i in range(len(rows)) if i in adj]
    feats["hbond_mean_degree"] = float(np.mean(deg)) if deg else 0.0
    return feats, pairs, neg_atoms, pos_atoms


def interface_features(rows, pairs, neg_atoms, pos_atoms):
    # BSA via residue SASA already is monomer; approximate BSA by interface contacts
    domains = ("VH", "VL", "CH1", "CL")
    pairs_if = [("VH", "VL"), ("VH", "CH1"), ("VL", "CL"), ("CH1", "CL")]
    by_dom = defaultdict(list)
    for i, r in enumerate(rows):
        if r["ca"] is not None:
            by_dom[r["domain"]].append(i)
    feats = {}
    for a, b in pairs_if:
        key = f"{a}_{b}"
        ia, ib = by_dom[a], by_dom[b]
        if not ia or not ib:
            for suf in (
                "bsa_proxy",
                "n_res",
                "contact_density",
                "hydrophobic_buried_area",
                "polar_buried_area",
                "hbond",
                "salt_bridge",
            ):
                feats[f"iface_{key}_{suf}"] = 0.0
            continue
        ca_a = np.array([rows[i]["ca"] for i in ia], float)
        ca_b = np.array([rows[i]["ca"] for i in ib], float)
        iface_a, iface_b = set(), set()
        for i, p in zip(ia, ca_a):
            d2 = np.sum((ca_b - p) ** 2, axis=1)
            if np.any(d2 <= IFACE_CA * IFACE_CA):
                iface_a.add(i)
        for j, p in zip(ib, ca_b):
            d2 = np.sum((ca_a - p) ** 2, axis=1)
            if np.any(d2 <= IFACE_CA * IFACE_CA):
                iface_b.add(j)
        iface = iface_a | iface_b
        # BSA proxy: sum SASA of interface residues * 0.5 (crude)
        bsa = 0.5 * sum(rows[i].get("sasa", 0.0) or 0.0 for i in iface)
        hydro = sum(rows[i].get("sasa", 0.0) or 0.0 for i in iface if rows[i]["aa"] in set("AILMFVWY"))
        polar = sum(rows[i].get("sasa", 0.0) or 0.0 for i in iface if rows[i]["aa"] not in set("AILMFVWY"))
        # contacts
        n_contact = 0
        for i in iface_a:
            for j in iface_b:
                if np.linalg.norm(rows[i]["ca"] - rows[j]["ca"]) <= IFACE_CA:
                    n_contact += 1
        hb = sum(1 for ri, rj, _ in pairs if (ri in iface_a and rj in iface_b) or (rj in iface_a and ri in iface_b))
        sb = 0
        if neg_atoms and pos_atoms:
            for ri, c1, d1 in neg_atoms:
                if rows[ri]["domain"] not in (a, b):
                    continue
                for rj, c2, d2 in pos_atoms:
                    if rows[rj]["domain"] not in (a, b):
                        continue
                    if {d1, d2} == {a, b} and np.linalg.norm(c1 - c2) <= SALT_R:
                        sb += 1
                        break
        feats[f"iface_{key}_bsa_proxy"] = float(bsa)
        feats[f"iface_{key}_n_res"] = float(len(iface))
        feats[f"iface_{key}_contact_density"] = float(n_contact / max(len(iface), 1))
        feats[f"iface_{key}_hydrophobic_buried_area"] = float(hydro)
        feats[f"iface_{key}_polar_buried_area"] = float(polar)
        feats[f"iface_{key}_hbond"] = float(hb)
        feats[f"iface_{key}_salt_bridge"] = float(sb)
    feats["shape_complementarity"] = np.nan
    feats["shape_complementarity_status"] = "SHAPE_COMPLEMENTARITY_TECHNICAL_BLOCK"
    # elbow proxy: angle between VH–CH1 COM vector and VL–CL COM vector
    com = {}
    for d in domains:
        pts = [rows[i]["ca"] for i in by_dom[d] if rows[i]["ca"] is not None]
        com[d] = np.mean(pts, axis=0) if pts else None
    if all(com[d] is not None for d in ("VH", "CH1", "VL", "CL")):
        v1 = com["CH1"] - com["VH"]
        v2 = com["CL"] - com["VL"]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
            feats["elbow_angle_deg"] = float(np.degrees(np.arccos(c)))
        else:
            feats["elbow_angle_deg"] = np.nan
    else:
        feats["elbow_angle_deg"] = np.nan
    return feats


def process_one(ab_id: str, pdb: Path, heavy: str, light: str, vh_len: int, vl_len: int, cdr_rows, source: str):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", str(pdb))
    model = next(st.get_models())
    pdb_to_hl, err = map_chains(model, heavy, light)
    if err:
        return None, err
    cdr_set = set()
    for r in cdr_rows:
        if str(r.get("is_cdr", "")).lower() in ("1", "true", "t", "yes") or r.get("is_cdr") is True:
            cdr_set.add((str(r["chain"]), int(r["sequence_index"])))
    # SASA
    try:
        sr = ShrakeRupley(probe_radius=PROBE, n_points=60)
        sr.compute(model, level="R")
    except Exception as e:
        return None, f"sasa_fail:{e}"
    rows = _residues(model, pdb_to_hl, vh_len, vl_len, cdr_set)
    compute_rasa(rows)
    feats = {"id": ab_id, "structure_source": source}
    try:
        feats.update(cavity_features(rows))
    except Exception as e:
        feats["cav_error"] = str(e)[:120]
    feats.update(packing_features(rows))
    unsat, pairs, neg_a, pos_a = buried_unsat_and_network(rows)
    feats.update(unsat)
    feats.update(interface_features(rows, pairs, neg_a, pos_a))
    # correlations helpers
    feats["n_res"] = float(len(rows))
    feats["n_atoms_heavy"] = float(sum(1 for r in rows for a in r["atoms"] if not a["is_h"]))
    return feats, None


def _job(args):
    return process_one(*args)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    fab = load_fab_meta()
    cdr = load_cdr()
    jobs = []
    for ab_id in fab.index.astype(str):
        row = fab.loc[ab_id]
        heavy, light = str(row.heavy_fab_seq), str(row.light_fab_seq)
        vh, vl = int(row.VH_len_used), int(row.VL_len_used)
        raw = RAW_FAB / f"{ab_id}.pdb"
        prep = PREP_FAB / f"{ab_id}_prepared.pdb"
        if raw.exists():
            jobs.append((ab_id, raw, heavy, light, vh, vl, cdr.get(ab_id, []), "raw_esmfold_fab"))
        if prep.exists():
            jobs.append((ab_id, prep, heavy, light, vh, vl, cdr.get(ab_id, []), "prepared_fab"))
    print(f"fab packing/iface jobs={len(jobs)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, errs = [], []
    workers = int(os.environ.get("GAP_WORKERS", "4"))  # cavity is RAM-heavy
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            feats, err = fut.result()
            if err:
                errs.append({"error": err, "job": str(k)})
            else:
                rows.append(feats)
            if k % 20 == 0:
                print(f"  fab {k}/{len(jobs)} ok={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "FAB_GAP_FEATURES_RAW.csv", index=False)
    pd.DataFrame(errs).to_csv(CACHE / "FAB_GAP_FEATURES_ERRORS.csv", index=False)

    raw = df[df.structure_source == "raw_esmfold_fab"].drop(columns=["structure_source"])
    prep = df[df.structure_source == "prepared_fab"].drop(columns=["structure_source"])

    cav_cols = [c for c in raw.columns if c.startswith("cav_") or c.startswith("pack_")]
    unsat_cols = [c for c in raw.columns if c.startswith("buried_") or c.startswith("hbond") or c.startswith("salt_")]
    iface_cols = [c for c in raw.columns if c.startswith("iface_") or c in ("elbow_angle_deg", "shape_complementarity")]

    # PRIMARY TmApp: raw Fab for geometric; prepared preferred for unsat — write both tagged
    raw[["id"] + cav_cols].to_csv(RESULTS / "TMAPP_PACKING_CAVITY_FEATURES.csv", index=False)
    # unsat primary = prepared when available
    u = prep[["id"] + unsat_cols].copy() if len(prep) else raw[["id"] + unsat_cols].copy()
    u.to_csv(RESULTS / "TMAPP_BURIED_UNSAT_FEATURES.csv", index=False)
    raw[["id"] + iface_cols].to_csv(RESULTS / "TMAPP_INTERFACE_FEATURES.csv", index=False)

    # also store prepared cavity/iface for sensitivity
    if len(prep):
        prep[["id"] + cav_cols].to_csv(CACHE / "TMAPP_PACKING_CAVITY_PREPARED.csv", index=False)
        prep[["id"] + iface_cols].to_csv(CACHE / "TMAPP_INTERFACE_PREPARED.csv", index=False)
        raw[["id"] + unsat_cols].to_csv(CACHE / "TMAPP_BURIED_UNSAT_RAW.csv", index=False)

    print("wrote packing/unsat/iface", len(raw), len(prep), "errs", len(errs), flush=True)


if __name__ == "__main__":
    main()
