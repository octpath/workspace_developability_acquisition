#!/usr/bin/env python3
"""
Stage 4 feature extraction (label-independent).
Uses .venv_stage4 for PROPKA/PDB2PQR/geometry; APBS binary; ESM-IF via subprocess to .venv_esmif.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "virtual_participant/stage4_advanced_structure"
CACHE = OUT / "cache"
FEAT = CACHE / "features"
PQR_DIR = CACHE / "pqr"
PKA_DIR = CACHE / "pka"
APBS_DIR = CACHE / "apbs"
IF_DIR = CACHE / "invfold"
ESMFOLD = ROOT / "esmfold_native"
DEV = ROOT / "competition/data/distribution/dev.csv"
S3_FEAT = ROOT / "virtual_participant/stage3_structure/cache/features_ESMFold.csv"
STAGE4_PY = ROOT / ".venv_stage4/bin/python"
ESMIF_PY = ROOT / ".venv_esmif/bin/python"
APBS_BIN = Path("/usr/bin/apbs")

PH = 6.5
IONIC = 0.15
PROBE = 1.4
RASA_EXPOSED = 0.20
PATCH_R = 8.0
LOCAL_R = 10.0
SB_DIST = 4.0
HB_DIST = 3.5
CLASH_DIST = 2.2
CONTACT_DIST = 4.5

HYDROPHOBIC = set("AILMFVW")
AROMATIC = set("FWY")
POS = set("KR")
NEG = set("DE")
POLAR = set("STNQYH")
IONIZABLE = set("DEKRHYC")

MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


def aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def ensure_dirs():
    for d in [OUT, CACHE, FEAT, PQR_DIR, PKA_DIR, APBS_DIR, IF_DIR, OUT / "oof", OUT / "plots", OUT / "artifacts", OUT / "scripts"]:
        d.mkdir(parents=True, exist_ok=True)


def residue_table(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", str(pdb_path))
    sr = ShrakeRupley(probe_radius=PROBE, n_points=100)
    sr.compute(st, level="R")
    rows = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                a = aa1(res.get_resname())
                sasa = float(res.sasa)
                maxasa = MAX_ASA.get(a)
                rasa = sasa / maxasa if maxasa else np.nan
                ca = res["CA"].coord.copy() if "CA" in res else None
                atoms = []
                for atom in res:
                    if atom.element == "H":
                        continue
                    atoms.append((atom.get_name(), atom.coord.copy(), atom.element))
                rows.append(
                    {
                        "chain": chain.id,
                        "resseq": res.id[1],
                        "aa": a,
                        "sasa": sasa,
                        "rasa": rasa,
                        "ca": ca,
                        "atoms": atoms,
                        "exposed": bool(np.isfinite(rasa) and rasa >= RASA_EXPOSED),
                    }
                )
        break
    return rows


def map_hl(rows, vh_len, vl_len):
    chains = sorted({r["chain"] for r in rows})
    if "H" in chains and "L" in chains:
        return {c: c for c in chains}
    if len(chains) < 2:
        return {chains[0]: "H"} if chains else {}
    lens = {c: sum(1 for r in rows if r["chain"] == c) for c in chains}
    best = None
    for a in chains:
        for b in chains:
            if a == b:
                continue
            score = abs(lens[a] - vh_len) + abs(lens[b] - vl_len)
            if best is None or score < best[0]:
                best = (score, a, b)
    return {best[1]: "H", best[2]: "L"}


def run_propka_pqr(pdb_path: Path, aid: str) -> dict:
    out = {"antibody_id": aid, "propka_ok": False, "pqr_ok": False, "error": ""}
    pka_path = PKA_DIR / f"{aid}.pka"
    pqr_path = PQR_DIR / f"{aid}.pqr"
    work = PKA_DIR / aid
    work.mkdir(parents=True, exist_ok=True)
    try:
        if not pka_path.exists():
            r = subprocess.run(
                [str(STAGE4_PY.parent / "propka3"), str(pdb_path)],
                cwd=str(work),
                capture_output=True,
                text=True,
                timeout=120,
            )
            produced = list(work.glob("*.pka"))
            if not produced:
                # propka writes next to pdb sometimes
                produced = list(Path(pdb_path).parent.glob(pdb_path.stem + ".pka"))
                produced += list(work.glob("*"))
            # copy pka
            cand = work / (pdb_path.stem + ".pka")
            if not cand.exists():
                cands = list(work.glob("*.pka"))
                if cands:
                    cand = cands[0]
            if cand.exists():
                pka_path.write_text(cand.read_text())
            elif r.returncode != 0:
                out["error"] = (r.stderr or r.stdout)[:500]
                return out
        out["propka_ok"] = pka_path.exists()
        if not pqr_path.exists():
            r = subprocess.run(
                [
                    str(STAGE4_PY.parent / "pdb2pqr"),
                    "--ff",
                    "AMBER",
                    "--with-ph",
                    str(PH),
                    "--titration-state-method",
                    "propka",
                    str(pdb_path),
                    str(pqr_path),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if r.returncode != 0 and not pqr_path.exists():
                out["error"] = (r.stderr or r.stdout)[-500:]
                return out
        out["pqr_ok"] = pqr_path.exists()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def parse_propka_features(pka_path: Path, rows, chain_map) -> dict:
    feat = {}
    if not pka_path.exists():
        return feat
    text = pka_path.read_text(errors="ignore")
    # parse lines like:    ASP  35 A    3.50 ...
    pkas = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        if parts[0] in {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG", "N+", "C-"} or (
            len(parts[0]) <= 3 and parts[0].isalpha() and parts[0].upper() == parts[0]
        ):
            try:
                resname = parts[0]
                resseq = int(parts[1])
                chain = parts[2]
                pka = float(parts[3])
                pkas.append((chain, resseq, resname, pka))
            except Exception:
                continue
    if not pkas:
        # fallback: charge proxy from sequence groups at pH
        pass
    # charge at pH from Henderson–Hasselbalch for ionizable
    # Use residue aa from rows
    key_to_row = {(r["chain"], r["resseq"]): r for r in rows}
    charges = []
    for ch, rs, rn, pka in pkas:
        mapped = chain_map.get(ch, ch)
        aa = None
        if (ch, rs) in key_to_row:
            aa = key_to_row[(ch, rs)]["aa"]
        # acid residues: negative when deprotonated
        acids = {"ASP", "GLU", "CYS", "TYR", "C-"}
        bases = {"HIS", "LYS", "ARG", "N+"}
        frac_deprot = 1.0 / (1.0 + 10 ** (pka - PH))
        if rn in acids or (aa in NEG | {"C", "Y"}):
            q = -frac_deprot
        elif rn in bases or (aa in POS | {"H"}):
            q = 1.0 - frac_deprot
        else:
            continue
        exposed = False
        if (ch, rs) in key_to_row:
            exposed = key_to_row[(ch, rs)]["exposed"]
        charges.append({"chain": mapped, "q": q, "exposed": exposed, "pka": pka})
    if charges:
        qs = np.array([c["q"] for c in charges])
        feat["propka_n_ionizable"] = len(charges)
        feat["propka_net_charge"] = float(qs.sum())
        feat["propka_abs_charge"] = float(np.abs(qs).sum())
        feat["propka_mean_pka"] = float(np.mean([c["pka"] for c in charges]))
        eq = [c["q"] for c in charges if c["exposed"]]
        feat["propka_exposed_net_charge"] = float(sum(eq)) if eq else 0.0
        feat["propka_exposed_pos"] = float(sum(q for q in eq if q > 0))
        feat["propka_exposed_neg"] = float(sum(q for q in eq if q < 0))
        for lab, pred in [("VH", "H"), ("VL", "L")]:
            sub = [c["q"] for c in charges if c["chain"] == pred]
            feat[f"propka_{lab}_net_charge"] = float(sum(sub)) if sub else 0.0
    return feat


def parse_pqr_charges(pqr_path: Path) -> dict:
    feat = {}
    if not pqr_path.exists():
        return feat
    qs, coords = [], []
    for line in pqr_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # PDB/PQR: charge and radius at end
        parts = line.split()
        try:
            # standard pdb2pqr: x y z charge radius
            x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
            q = float(parts[-2])
            qs.append(q)
            coords.append([x, y, z])
        except Exception:
            continue
    if not qs:
        return feat
    qs = np.asarray(qs, float)
    feat["pqr_n_atoms"] = len(qs)
    feat["pqr_net_charge"] = float(qs.sum())
    feat["pqr_abs_charge"] = float(np.abs(qs).sum())
    feat["pqr_charge_std"] = float(qs.std())
    feat["pqr_pos_charge_frac"] = float((qs > 0).mean())
    feat["pqr_neg_charge_frac"] = float((qs < 0).mean())
    return feat


def write_apbs_in(pqr_path: Path, out_prefix: Path) -> Path:
    # rough molecule size for mg-auto
    coords = []
    for line in pqr_path.read_text().splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            parts = line.split()
            try:
                coords.append([float(parts[-5]), float(parts[-4]), float(parts[-3])])
            except Exception:
                pass
    coords = np.asarray(coords, float)
    cmin = coords.min(0) - 20
    cmax = coords.max(0) + 20
    center = 0.5 * (cmin + cmax)
    length = cmax - cmin
    inp = out_prefix.with_suffix(".in")
    # APBS 3.x input
    text = f"""read
  mol pqr {pqr_path}
end
elec name pot
  mg-auto
  dime 97 97 97
  cglen {length[0]:.3f} {length[1]:.3f} {length[2]:.3f}
  fglen {max(length[0]-10,20):.3f} {max(length[1]-10,20):.3f} {max(length[2]-10,20):.3f}
  cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
  fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
  mol 1
  lpbe
  bcfl sdh
  pdie 2.0
  sdie 78.54
  srfm smol
  chgm spl2
  sdens 10.0
  srad 1.4
  swin 0.3
  temp 298.15
  ion charge 1 conc {IONIC} radius 2.0
  ion charge -1 conc {IONIC} radius 2.0
  calcenergy no
  calcforce no
  write pot dx {out_prefix}_pot
end
quit
"""
    inp.write_text(text)
    return inp


def load_dx(path: Path):
    """Minimal OpenDX scalar grid loader."""
    lines = path.read_text().splitlines()
    dims = None
    origin = None
    deltas = []
    data = []
    mode = None
    for line in lines:
        if line.startswith("object 1"):
            # gridpositions counts nx ny nz
            parts = line.split()
            dims = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
        elif line.startswith("origin"):
            origin = np.array(list(map(float, line.split()[1:4])))
        elif line.startswith("delta"):
            deltas.append(list(map(float, line.split()[1:4])))
        elif line.startswith("object 3") or line.startswith("object 2 class array"):
            mode = "data"
        elif mode == "data":
            if line.startswith("attribute") or line.startswith("object") or line.startswith("end"):
                continue
            data.extend(map(float, line.split()))
    if dims is None or origin is None or len(deltas) < 3:
        raise ValueError("bad dx")
    grid = np.asarray(data, float).reshape(dims, order="C")
    # APBS often writes z-fast; reshape may need order F — try size match
    if grid.size != math.prod(dims):
        grid = np.asarray(data, float).reshape(dims, order="F")
    dmat = np.array(deltas)
    return origin, dmat, grid


def sample_potential(origin, dmat, grid, points):
    # trilinear on grid indices along orthogonal deltas if diagonal
    nx, ny, nz = grid.shape
    # assume orthogonal deltas
    dx = dmat[0, 0] if abs(dmat[0, 0]) > 1e-9 else np.linalg.norm(dmat[0])
    dy = dmat[1, 1] if abs(dmat[1, 1]) > 1e-9 else np.linalg.norm(dmat[1])
    dz = dmat[2, 2] if abs(dmat[2, 2]) > 1e-9 else np.linalg.norm(dmat[2])
    vals = []
    for p in points:
        i = (p[0] - origin[0]) / dx
        j = (p[1] - origin[1]) / dy
        k = (p[2] - origin[2]) / dz
        i0, j0, k0 = int(np.floor(i)), int(np.floor(j)), int(np.floor(k))
        if not (0 <= i0 < nx - 1 and 0 <= j0 < ny - 1 and 0 <= k0 < nz - 1):
            continue
        # nearest
        vals.append(grid[i0, j0, k0])
    return np.asarray(vals, float)


def run_apbs_features(aid: str, rows, chain_map) -> dict:
    feat = {"apbs_ok": 0}
    pqr = PQR_DIR / f"{aid}.pqr"
    if not pqr.exists() or not APBS_BIN.exists():
        return feat
    prefix = APBS_DIR / aid
    dx = Path(str(prefix) + "_pot.dx")
    try:
        if not dx.exists():
            inp = write_apbs_in(pqr, prefix)
            r = subprocess.run(
                [str(APBS_BIN), str(inp)],
                cwd=str(APBS_DIR),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if not dx.exists():
                # sometimes named differently
                alts = list(APBS_DIR.glob(f"{aid}*.dx"))
                if alts:
                    dx = alts[0]
                else:
                    feat["apbs_error"] = (r.stderr or r.stdout)[-300:]
                    return feat
        origin, dmat, grid = load_dx(dx)
        # sample at exposed CA + 1.4A along arbitrary +x as crude surface offset
        pts = []
        labels = []
        for r in rows:
            if not r["exposed"] or r["ca"] is None:
                continue
            p = r["ca"] + np.array([1.4, 0.0, 0.0])
            pts.append(p)
            labels.append(chain_map.get(r["chain"], r["chain"]))
        if not pts:
            return feat
        vals = sample_potential(origin, dmat, grid, pts)
        if len(vals) == 0:
            feat["apbs_error"] = "no_samples"
            return feat
        feat["apbs_ok"] = 1
        feat["apbs_mean_potential"] = float(vals.mean())
        feat["apbs_median_potential"] = float(np.median(vals))
        feat["apbs_std_potential"] = float(vals.std())
        feat["apbs_q10_potential"] = float(np.quantile(vals, 0.1))
        feat["apbs_q90_potential"] = float(np.quantile(vals, 0.9))
        feat["apbs_abs_mean_potential"] = float(np.mean(np.abs(vals)))
        feat["apbs_pos_frac"] = float((vals > 0).mean())
        feat["apbs_neg_frac"] = float((vals < 0).mean())
        feat["apbs_strong_pos_frac"] = float((vals > 25).mean())
        feat["apbs_strong_neg_frac"] = float((vals < -25).mean())
        # electrostatic patches on exposed residues via CA graph + sign
        exposed = [r for r in rows if r["exposed"] and r["ca"] is not None]
        if len(exposed) >= 2 and len(vals) == len(exposed):
            for name, pred in [("pos", lambda v: v > 10), ("neg", lambda v: v < -10)]:
                idxs = [i for i, v in enumerate(vals) if pred(v)]
                if not idxs:
                    feat[f"apbs_{name}_patch_count"] = 0
                    feat[f"apbs_{name}_max_patch"] = 0
                    continue
                parent = {i: i for i in idxs}

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for a in idxs:
                    for b in idxs:
                        if a >= b:
                            continue
                        if np.linalg.norm(exposed[a]["ca"] - exposed[b]["ca"]) <= PATCH_R:
                            ra, rb = find(a), find(b)
                            if ra != rb:
                                parent[rb] = ra
                comps = defaultdict(list)
                for i in idxs:
                    comps[find(i)].append(i)
                sizes = [len(v) for v in comps.values()]
                feat[f"apbs_{name}_patch_count"] = len(sizes)
                feat[f"apbs_{name}_max_patch"] = int(max(sizes))
    except Exception as e:
        feat["apbs_error"] = f"{type(e).__name__}:{e}"
    return feat


def advanced_patch_features(rows, chain_map) -> dict:
    feat = {}
    exposed = [r for r in rows if r["exposed"] and r["ca"] is not None]
    if len(exposed) < 2:
        return feat
    coords = np.array([r["ca"] for r in exposed])
    # local exposed hydrophobic / aromatic SASA within radius
    loc_h, loc_a = [], []
    for i, r in enumerate(exposed):
        d = np.linalg.norm(coords - coords[i], axis=1)
        nbr = [exposed[j] for j in np.where(d <= LOCAL_R)[0]]
        loc_h.append(sum(x["sasa"] for x in nbr if x["aa"] in HYDROPHOBIC))
        loc_a.append(sum(x["sasa"] for x in nbr if x["aa"] in AROMATIC))
    loc_h = np.asarray(loc_h)
    loc_a = np.asarray(loc_a)
    feat["adv_max_local_hydrophobic_sasa"] = float(loc_h.max())
    feat["adv_mean_local_hydrophobic_sasa"] = float(loc_h.mean())
    feat["adv_topk_local_hydrophobic_sasa"] = float(np.sort(loc_h)[-min(5, len(loc_h)) :].mean())
    feat["adv_max_local_aromatic_sasa"] = float(loc_a.max())
    feat["adv_mean_local_aromatic_sasa"] = float(loc_a.mean())
    # spatial hydrophobicity descriptor (generic; not claiming SAP formula)
    # weight = exposed hydrophobic SASA * local density
    spat = loc_h * np.array([r["sasa"] if r["aa"] in HYDROPHOBIC else 0.0 for r in exposed])
    feat["adv_spatial_hydrophobicity_sum"] = float(spat.sum())
    feat["adv_spatial_hydrophobicity_max"] = float(spat.max()) if len(spat) else 0.0

    def components(pred):
        idxs = [i for i, r in enumerate(exposed) if pred(r)]
        if not idxs:
            return 0, 0, 0.0, 0.0, 0.0
        parent = {i: i for i in idxs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a_i, a in enumerate(idxs):
            for b in idxs[a_i + 1 :]:
                if np.linalg.norm(exposed[a]["ca"] - exposed[b]["ca"]) <= PATCH_R:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        comps = defaultdict(list)
        for i in idxs:
            comps[find(i)].append(i)
        sizes = [len(v) for v in comps.values()]
        areas = [sum(exposed[i]["sasa"] for i in v) for v in comps.values()]
        # compactness: area / (size)^(2/3)
        compact = [areas[k] / (sizes[k] ** (2 / 3)) for k in range(len(sizes))]
        extents = []
        for v in comps.values():
            pts = np.array([exposed[i]["ca"] for i in v])
            extents.append(float(np.linalg.norm(pts.max(0) - pts.min(0))))
        return len(sizes), int(max(sizes)), float(max(areas)), float(np.mean(compact) if compact else 0), float(max(extents) if extents else 0)

    n, mx, area, compact, extent = components(lambda r: r["aa"] in HYDROPHOBIC)
    feat.update(
        {
            "adv_hydrophobic_patch_n": n,
            "adv_hydrophobic_patch_max_size": mx,
            "adv_hydrophobic_patch_max_sasa": area,
            "adv_hydrophobic_patch_mean_compact": compact,
            "adv_hydrophobic_patch_max_extent": extent,
        }
    )
    n, mx, area, compact, extent = components(lambda r: r["aa"] in AROMATIC)
    feat.update(
        {
            "adv_aromatic_patch_n": n,
            "adv_aromatic_patch_max_size": mx,
            "adv_aromatic_patch_max_sasa": area,
            "adv_aromatic_patch_mean_compact": compact,
            "adv_aromatic_patch_max_extent": extent,
        }
    )
    # mixed aromatic-hydrophobic
    n, mx, area, compact, extent = components(lambda r: r["aa"] in (HYDROPHOBIC | AROMATIC))
    feat["adv_mixed_arom_hydro_patch_n"] = n
    feat["adv_mixed_arom_hydro_patch_max_sasa"] = area
    return feat


def interaction_features(rows, chain_map) -> dict:
    feat = {}
    # remap chain labels
    for r in rows:
        r["hl"] = chain_map.get(r["chain"], r["chain"])
    # salt bridges: ASP/GLU OD/OE vs LYS/ARG NZ/NH within 4A
    donors = []
    acceptors = []
    for r in rows:
        for name, coord, el in r["atoms"]:
            if r["aa"] in NEG and name.startswith(("OD", "OE")):
                acceptors.append((r, coord))
            if r["aa"] in POS and name.startswith(("NZ", "NH", "NE")):
                donors.append((r, coord))
    sb = 0
    sb_iface = 0
    for d_r, d_c in donors:
        for a_r, a_c in acceptors:
            if np.linalg.norm(d_c - a_c) <= SB_DIST:
                sb += 1
                if d_r["hl"] != a_r["hl"]:
                    sb_iface += 1
    feat["salt_bridge_count"] = sb
    feat["salt_bridge_interface"] = sb_iface

    # H-bond geometry proxy: N/O heavy atoms within 3.5A, different residues
    heavy = []
    for r in rows:
        for name, coord, el in r["atoms"]:
            if el in {"N", "O"} or name[0] in {"N", "O"}:
                heavy.append((r, name, coord, el if el else name[0]))
    hb = 0
    hb_buried = 0
    hb_iface = 0
    for i in range(len(heavy)):
        ri, ni, ci, ei = heavy[i]
        for j in range(i + 1, len(heavy)):
            rj, nj, cj, ej = heavy[j]
            if ri is rj:
                continue
            if {ei, ej} == {"N", "O"} or (ni[0] in "NO" and nj[0] in "NO" and ni[0] != nj[0]):
                dist = np.linalg.norm(ci - cj)
                if dist <= HB_DIST:
                    hb += 1
                    if (not ri["exposed"]) and (not rj["exposed"]):
                        hb_buried += 1
                    if ri["hl"] != rj["hl"]:
                        hb_iface += 1
    feat["hbond_proxy_count"] = hb
    feat["hbond_proxy_buried"] = hb_buried
    feat["hbond_proxy_interface"] = hb_iface

    # clashes / contacts on heavy atoms
    all_heavy = []
    for r in rows:
        for name, coord, el in r["atoms"]:
            all_heavy.append((r, coord))
    clash = 0
    contacts = 0
    core_h = 0
    # subsample pairs via CA contacts first for speed
    ca_rows = [r for r in rows if r["ca"] is not None]
    for i, ri in enumerate(ca_rows):
        for rj in ca_rows[i + 1 :]:
            dca = np.linalg.norm(ri["ca"] - rj["ca"])
            if dca > 8:
                continue
            for _, ci in [(a, c) for a, c, _e in ri["atoms"]]:
                for _, cj in [(a, c) for a, c, _e in rj["atoms"]]:
                    d = np.linalg.norm(ci - cj)
                    if d < CLASH_DIST:
                        clash += 1
                    if d < CONTACT_DIST:
                        contacts += 1
                        if (not ri["exposed"]) and (not rj["exposed"]) and ri["aa"] in HYDROPHOBIC and rj["aa"] in HYDROPHOBIC:
                            core_h += 1
    nres = max(len(rows), 1)
    feat["clash_proxy_count"] = clash
    feat["heavy_contact_count"] = contacts
    feat["contact_density"] = contacts / nres
    feat["core_hydrophobic_contacts"] = core_h
    buried = [r for r in rows if np.isfinite(r["rasa"]) and r["rasa"] < RASA_EXPOSED]
    feat["buried_frac"] = len(buried) / nres
    # packing degree via CA neighbors < 8A
    degrees = []
    for i, ri in enumerate(ca_rows):
        deg = sum(1 for j, rj in enumerate(ca_rows) if i != j and np.linalg.norm(ri["ca"] - rj["ca"]) < 8.0)
        degrees.append(deg)
    if degrees:
        deg = np.asarray(degrees, float)
        feat["packing_degree_mean"] = float(deg.mean())
        feat["packing_degree_q10"] = float(np.quantile(deg, 0.1))
        feat["packing_degree_q90"] = float(np.quantile(deg, 0.9))
        feat["low_packed_frac"] = float((deg <= np.quantile(deg, 0.2)).mean())
    # interface metrics
    h = [r for r in rows if r["hl"] == "H" and r["ca"] is not None]
    l = [r for r in rows if r["hl"] == "L" and r["ca"] is not None]
    iface_n = 0
    hydro_bsa_proxy = 0.0
    arom_bsa_proxy = 0.0
    for ri in h:
        for rj in l:
            if np.linalg.norm(ri["ca"] - rj["ca"]) < 8.0:
                iface_n += 1
                if ri["aa"] in HYDROPHOBIC:
                    hydro_bsa_proxy += ri["sasa"]
                if rj["aa"] in HYDROPHOBIC:
                    hydro_bsa_proxy += rj["sasa"]
                if ri["aa"] in AROMATIC:
                    arom_bsa_proxy += ri["sasa"]
                if rj["aa"] in AROMATIC:
                    arom_bsa_proxy += rj["sasa"]
    feat["interface_contact_pairs"] = iface_n
    feat["interface_contact_density"] = iface_n / max(len(h) + len(l), 1)
    feat["hydrophobic_interface_sasa_proxy"] = hydro_bsa_proxy
    feat["aromatic_interface_sasa_proxy"] = arom_bsa_proxy
    if h and l:
        feat["vh_vl_center_dist"] = float(
            np.linalg.norm(np.mean([r["ca"] for r in h], 0) - np.mean([r["ca"] for r in l], 0))
        )
    return feat


def cavity_and_unsat_features(rows) -> dict:
    feat = {}
    ca_rows = [r for r in rows if r["ca"] is not None]
    if len(ca_rows) < 10:
        return feat
    coords = np.array([r["ca"] for r in ca_rows])
    # geometry cavity proxy: buried residues with low neighbor degree
    degrees = []
    for i in range(len(coords)):
        deg = int((np.linalg.norm(coords - coords[i], axis=1) < 8).sum() - 1)
        degrees.append(deg)
    degrees = np.asarray(degrees)
    buried = np.array([1 if (np.isfinite(r["rasa"]) and r["rasa"] < RASA_EXPOSED) else 0 for r in ca_rows])
    low = (degrees <= np.quantile(degrees, 0.15)) & (buried == 1)
    feat["cavity_proxy_n_sites"] = int(low.sum())
    feat["cavity_proxy_frac"] = float(low.mean())
    # approximate volume proxy: count * 4/3 pi (2.5)^3
    feat["cavity_proxy_volume"] = float(low.sum() * (4 / 3) * math.pi * (2.5**3))
    # buried unsatisfied polar proxy: buried polar/charged with few N/O neighbors
    unsat = 0
    for i, r in enumerate(ca_rows):
        if not (np.isfinite(r["rasa"]) and r["rasa"] < RASA_EXPOSED):
            continue
        if r["aa"] not in (POLAR | POS | NEG):
            continue
        # count nearby donor/acceptor heavy atoms from other residues
        n_near = 0
        for j, rj in enumerate(ca_rows):
            if i == j:
                continue
            if np.linalg.norm(coords[i] - coords[j]) > 6:
                continue
            for name, coord, el in rj["atoms"]:
                if name[0] in {"N", "O"} and np.linalg.norm(coord - coords[i]) < 3.5:
                    n_near += 1
        if n_near == 0:
            unsat += 1
    feat["buried_unsatisfied_polar_proxy"] = unsat
    feat["buried_unsatisfied_polar_proxy_frac"] = unsat / max(len(ca_rows), 1)
    return feat


def extract_one(aid: str, heavy: str, light: str) -> dict:
    pdb = ESMFOLD / f"{aid}.pdb"
    feat = {"antibody_id": aid}
    status = {"antibody_id": aid}
    if not pdb.exists():
        status["error"] = "missing_pdb"
        return feat, status
    rows = residue_table(pdb)
    chain_map = map_hl(rows, len(heavy), len(light))
    st = run_propka_pqr(pdb, aid)
    status.update(st)
    feat.update(parse_propka_features(PKA_DIR / f"{aid}.pka", rows, chain_map))
    feat.update(parse_pqr_charges(PQR_DIR / f"{aid}.pqr"))
    feat.update(run_apbs_features(aid, rows, chain_map))
    feat.update(advanced_patch_features(rows, chain_map))
    feat.update(interaction_features(rows, chain_map))
    feat.update(cavity_and_unsat_features(rows))
    return feat, status


def run_invfold_all(ids):
    script = CACHE / "score_invfold_batch.py"
    script.write_text(
        """
import json, sys
from pathlib import Path
import numpy as np
import esm
from esm.inverse_folding.util import load_coords, score_sequence

pdb_dir = Path(sys.argv[1])
out_json = Path(sys.argv[2])
ids = json.loads(Path(sys.argv[3]).read_text())
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()
rows = []
for i, aid in enumerate(ids):
    pdb = pdb_dir / f"{aid}.pdb"
    rec = {"antibody_id": aid, "invfold_ok": 0}
    try:
        lls = []
        lens = []
        for ch in ["A", "B"]:
            coords, seq = load_coords(str(pdb), ch)
            ll = float(score_sequence(model, alphabet, coords, seq)[0])
            lls.append(ll)
            lens.append(len(seq))
            rec[f"invfold_{ch}_ll"] = ll
            rec[f"invfold_{ch}_nll"] = -ll
            rec[f"invfold_{ch}_len"] = len(seq)
        rec["invfold_fv_mean_ll"] = float(np.mean(lls))
        rec["invfold_fv_mean_nll"] = float(-np.mean(lls))
        rec["invfold_ll_var"] = float(np.var(lls))
        rec["invfold_worst_chain_ll"] = float(min(lls))
        rec["invfold_ok"] = 1
    except Exception as e:
        rec["error"] = f"{type(e).__name__}:{e}"
    rows.append(rec)
    if (i + 1) % 10 == 0:
        print(f"invfold {i+1}/{len(ids)}", flush=True)
        Path(out_json).write_text(json.dumps(rows))
Path(out_json).write_text(json.dumps(rows))
print("done", len(rows))
"""
    )
    id_json = CACHE / "dev_ids.json"
    id_json.write_text(json.dumps(ids))
    out_json = IF_DIR / "invfold_scores.json"
    if out_json.exists() and len(json.loads(out_json.read_text())) >= len(ids):
        return pd.DataFrame(json.loads(out_json.read_text()))
    subprocess.run(
        [str(ESMIF_PY), str(script), str(ESMFOLD), str(out_json), str(id_json)],
        check=False,
    )
    if out_json.exists():
        return pd.DataFrame(json.loads(out_json.read_text()))
    return pd.DataFrame({"antibody_id": ids})


def classify_family(name: str) -> str:
    n = name.lower()
    if name.startswith("propka") or "pka" in n:
        return "ADV_PROPKA"
    if name.startswith("pqr"):
        return "ADV_PQR_CHARGE"
    if name.startswith("apbs"):
        return "ADV_ELECTROSTATICS"
    if name.startswith("adv_") or "patch" in n or "spatial_hydrophobic" in n:
        return "ADV_SURFACE_PATCH"
    if any(x in n for x in ["salt_bridge", "hbond", "clash", "contact", "packing", "interface", "vh_vl"]):
        return "ADV_INTERACTIONS"
    if "cavity" in n:
        return "ADV_CAVITY"
    if "unsatisfied" in n:
        return "ADV_UNSAT_POLAR"
    if "invfold" in n:
        return "ADV_INVFOLD"
    return "ADV_OTHER"


def main():
    ensure_dirs()
    # write APBS param manifest
    (CACHE / "apbs_params.json").write_text(
        json.dumps(
            {
                "pH_for_protonation": PH,
                "ionic_strength_M": IONIC,
                "pdie": 2.0,
                "sdie": 78.54,
                "srad": 1.4,
                "temp": 298.15,
                "label": "generic_fixed_condition_not_assay_matched",
            },
            indent=2,
        )
    )
    dev = pd.read_csv(DEV)
    rows_f, statuses = [], []
    feat_path = FEAT / "advanced_features_partial.csv"
    done = set()
    if feat_path.exists():
        prev = pd.read_csv(feat_path)
        done = set(prev.antibody_id)
        rows_f = prev.to_dict("records")
        print(f"resume {len(done)}", flush=True)

    for i, r in dev.iterrows():
        aid = r["id"]
        if aid in done:
            continue
        print(f"[{i+1}/162] {aid}", flush=True)
        try:
            feat, st = extract_one(aid, r["heavy"], r["light"])
        except Exception as e:
            feat = {"antibody_id": aid}
            st = {"antibody_id": aid, "error": traceback.format_exc()[-500:]}
            print("  FAIL", e, flush=True)
        rows_f.append(feat)
        statuses.append(st)
        if (i + 1) % 5 == 0:
            pd.DataFrame(rows_f).to_csv(feat_path, index=False)
            pd.DataFrame(statuses).to_csv(FEAT / "extraction_status.csv", index=False)

    pd.DataFrame(rows_f).to_csv(FEAT / "advanced_features.csv", index=False)
    pd.DataFrame(statuses).to_csv(FEAT / "extraction_status.csv", index=False)

    print("=== inverse folding ===", flush=True)
    inv = run_invfold_all(list(dev["id"]))
    inv.to_csv(FEAT / "invfold_features.csv", index=False)

    # merge
    base = pd.DataFrame(rows_f).set_index("antibody_id")
    inv = inv.set_index("antibody_id")
    merged = base.join(inv, how="left", rsuffix="_if")
    merged = merged.reset_index()
    # align to dev order
    merged = merged.set_index("antibody_id").loc[dev["id"]].reset_index()
    merged.to_csv(FEAT / "stage4_all_features.csv", index=False)

    # manifest
    man = []
    for c in merged.columns:
        if c == "antibody_id":
            continue
        fam = classify_family(c)
        man.append(
            {
                "feature_name": c,
                "family": fam,
                "structure_source": "ESMFold_native",
                "definition": c,
                "unit": "mixed",
                "missing_count": int(merged[c].isna().sum()) if merged[c].dtype != object else int(merged[c].isna().sum()),
                "label_independent": True,
                "notes": "Stage4 advanced",
            }
        )
    pd.DataFrame(man).to_csv(OUT / "stage4_feature_manifest.csv", index=False)
    print("features", merged.shape, flush=True)


if __name__ == "__main__":
    main()
