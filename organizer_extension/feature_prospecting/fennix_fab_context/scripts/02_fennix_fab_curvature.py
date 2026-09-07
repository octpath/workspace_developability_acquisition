#!/usr/bin/env python3
"""FeNNix-v2 curvature on Fab conditions B / C / M (+ matched variable sites)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, chemical_symbols
from ase.optimize import BFGS, FIRE
from Bio.PDB import PDBIO, PDBParser, Select

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
V2 = FP / "foundation_stability_v2"
CTX = FP / "fennix_fab_context"
FAB = FP / "fab_reconstruction"
SEQ = FAB / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
CDR_IDX = FP / "cdr_sequence_index_imgt.csv"
MODEL = V2 / "cache/fennix-bio1S.fnx"
SPEC = json.loads((V2 / "FOUNDATION_STABILITY_V2_SPEC.json").read_text())
FX = SPEC["fennix_v2"]
FMAX = float(FX["reference_states"]["R1_SIDECHAIN_RELAXED"]["fmax_eV_A"])
MAX_STEPS = int(FX["reference_states"]["R1_SIDECHAIN_RELAXED"]["max_steps"])
PHI_PSI_DEG = float(FX["perturbations"]["backbone_phi_psi_deg"])
CHI1_DEG = float(FX["perturbations"]["sidechain_chi1_deg"])
SEED = int(FX["site_selection"]["seed"])
BB_FIX = set(FX["reference_states"]["R1_SIDECHAIN_RELAXED"]["fix_backbone_heavy"])
PILOT = json.loads((FAB / "structures/pilot_ids.json").read_text())
CONTACT_A = 4.5

OUT = CTX
CACHE = CTX / "cache"


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def force_stats(F):
    nrm = np.linalg.norm(F, axis=1)
    return float(np.sqrt(np.mean(nrm**2))), float(np.quantile(nrm, 0.9)), float(np.max(nrm))


def clash_counts(coords, meta):
    heavy = [(i, m) for i, m in enumerate(meta) if m["is_heavy"]]
    severe = allc = 0
    min_nb = np.inf
    for a in range(len(heavy)):
        i, mi = heavy[a]
        for b in range(a + 1, len(heavy)):
            j, mj = heavy[b]
            if mi["chain"] == mj["chain"] and abs(mi["resseq"] - mj["resseq"]) <= 1:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            min_nb = min(min_nb, d)
            if d < 1.5:
                severe += 1
            if d < 2.0:
                allc += 1
    return severe, allc, (min_nb if min_nb < np.inf else np.nan)


def load_structure(pdb: Path):
    s = PDBParser(QUIET=True).get_structure("x", str(pdb))
    species, coords, meta = [], [], []
    for a in s.get_atoms():
        el = (a.element or "").strip()
        if not el:
            el = a.get_name()[0]
        el = el.capitalize() if len(el) == 1 else el.title()
        if el not in atomic_numbers:
            el = a.get_name()[0].capitalize()
        species.append(atomic_numbers[el])
        coords.append(a.coord.copy())
        res = a.get_parent()
        meta.append(
            {
                "chain": res.get_parent().id,
                "resseq": int(res.id[1]),
                "resname": res.get_resname(),
                "name": a.get_name().strip(),
                "element": el,
                "is_backbone_heavy": a.get_name().strip() in BB_FIX,
                "is_heavy": el != "H",
            }
        )
    return np.asarray(species, np.int32), np.asarray(coords, float), meta


def to_ase(species, coords):
    return Atoms(symbols=[chemical_symbols[int(z)] for z in species], positions=coords)


def energy_forces_model(model, species, coords):
    E, F, _ = model.energy_and_forces(species=species, coordinates=coords, unit="ev")
    return float(np.asarray(E).reshape(-1)[0]), np.asarray(F, float)


def relax_r1(model, species, coords, meta):
    atoms = to_ase(species, coords)
    try:
        from fennol.ase import FENNIXCalculator

        atoms.calc = FENNIXCalculator(model, unit="eV")
    except Exception:
        from ase.calculators.calculator import Calculator, all_changes

        class Wrap(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
                Calculator.calculate(self, atoms, properties, system_changes)
                e, f = energy_forces_model(model, species, atoms.get_positions())
                self.results = {"energy": e, "forces": f}

        atoms.calc = Wrap()

    fix_idx = [i for i, m in enumerate(meta) if m["is_backbone_heavy"]]
    atoms.set_constraint(FixAtoms(indices=fix_idx))
    opt_name = "FIRE"
    try:
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=FMAX, steps=MAX_STEPS)
    except Exception as e:
        opt_name = f"BFGS_after_FIRE_fail:{type(e).__name__}"
        atoms.set_positions(coords)
        atoms.set_constraint(FixAtoms(indices=fix_idx))
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=FMAX, steps=MAX_STEPS)
    final = atoms.get_positions()
    e, f = energy_forces_model(model, species, final)
    free = np.ones(len(meta), bool)
    free[fix_idx] = False
    fmax = float(np.max(np.linalg.norm(f[free], axis=1))) if free.any() else float(np.max(np.linalg.norm(f, axis=1)))
    return final, e, f, bool(fmax <= FMAX + 1e-6), int(getattr(opt, "nsteps", MAX_STEPS)), fmax, opt_name


def agg_K(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {k: np.nan for k in ["mean", "median", "q10", "q90", "iqr", "frac_neg", "n"]}
    q10, q90 = np.quantile(v, [0.1, 0.9])
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "q10": float(q10),
        "q90": float(q90),
        "iqr": float(np.quantile(v, 0.75) - np.quantile(v, 0.25)),
        "frac_neg": float(np.mean(v < 0)),
        "n": int(len(v)),
    }


def classify_residue(row_region: str) -> str:
    r = str(row_region).upper()
    if "H3" in r or r in {"CDR3", "HCDR3"}:
        return "HCDR3"
    if "L3" in r or r == "LCDR3":
        return "LCDR3"
    if "CDR" in r:
        return "CDR"
    return "FW"


def region_table(ab_id: str):
    if not CDR_IDX.exists():
        return pd.DataFrame()
    idx = pd.read_csv(CDR_IDX)
    return idx[idx.id == ab_id].copy()


def build_res_region(ab_id, meta, vh_len, vl_len):
    """Map (chain,resseq) -> region label using CDR index + domain lengths."""
    residues = {}
    for i, m in enumerate(meta):
        residues.setdefault((m["chain"], m["resseq"]), {})[m["name"]] = i
    regions = region_table(ab_id)
    res_region = {}
    # Variable region from CDR file
    if len(regions):
        for ch_src, ch_dst, maxlen in [("H", "A", vh_len), ("L", "B", vl_len)]:
            sub = regions[regions.chain.astype(str) == ch_src].sort_values("sequence_index")
            keys = sorted([k for k in residues if k[0] == ch_dst and 1 <= k[1] <= maxlen], key=lambda x: x[1])
            for i, (_, r) in enumerate(sub.iterrows()):
                if i >= len(keys):
                    break
                lab = classify_residue(r.get("region", ""))
                # refine LCDR3
                reg = str(r.get("region", "")).upper()
                if "L3" in reg or reg == "LCDR3":
                    lab = "LCDR3"
                elif lab == "CDR" and ch_src == "L":
                    lab = "LCDR"
                elif lab == "CDR" and ch_src == "H" and "H3" not in reg:
                    lab = "HCDR"
                res_region[keys[i]] = lab
    for k in residues:
        ch, rs = k
        if k in res_region:
            continue
        if ch == "A" and rs <= vh_len:
            res_region[k] = "VH_FW"
        elif ch == "B" and rs <= vl_len:
            res_region[k] = "VL_FW"
        elif ch == "A" and rs > vh_len:
            res_region[k] = "CH1"
        elif ch == "B" and rs > vl_len:
            res_region[k] = "CL"
        else:
            res_region[k] = "FW"
    # normalize FW labels for variable
    for k, v in list(res_region.items()):
        if v == "FW":
            ch, rs = k
            if ch == "A" and rs <= vh_len:
                res_region[k] = "VH_FW"
            elif ch == "B" and rs <= vl_len:
                res_region[k] = "VL_FW"
    return residues, res_region


def pick_variable_sites(ab_id, meta, vh_len, vl_len):
    """Same counts/seed as FeNNix-v2, restricted to VH+VL."""
    residues, res_region = build_res_region(ab_id, meta, vh_len, vl_len)

    def valid_bb(key):
        at = residues[key]
        return all(n in at for n in ("N", "CA", "C"))

    def valid_chi1(key):
        at = residues[key]
        if "CA" not in at or "CB" not in at:
            return False
        return any(n.startswith(("CG", "OG", "SG", "NG")) for n in at)

    def take(pool, n, tag):
        if not pool:
            return []
        h = stable_hash(f"{ab_id}|{tag}|{SEED}")
        rng = np.random.default_rng(h % (2**32))
        order = np.argsort(rng.random(len(pool)))
        return [pool[i] for i in order[: min(n, len(pool))]]

    var_keys = [k for k in residues if (k[0] == "A" and k[1] <= vh_len) or (k[0] == "B" and k[1] <= vl_len)]
    fw = [k for k in var_keys if res_region.get(k, "").endswith("FW") and valid_bb(k)]
    # CDR pools: HCDR / LCDR / HCDR3 / LCDR3
    hcdr = [k for k in var_keys if res_region.get(k) in ("HCDR", "CDR") and k[0] == "A" and valid_bb(k)]
    lcdr = [k for k in var_keys if res_region.get(k) in ("LCDR", "CDR") and k[0] == "B" and valid_bb(k)]
    h3 = [k for k in var_keys if res_region.get(k) == "HCDR3" and valid_bb(k)]
    # match v2 counts: FW 8, CDR 8 (combine H+L CDR non-H3), HCDR3 extra 4, chi1 12
    cdr = [k for k in hcdr + lcdr if k not in h3]
    sel_fw = take(fw, FX["site_selection"]["framework_backbone"], "FW")
    sel_cdr = take(cdr, FX["site_selection"]["cdr_backbone"], "CDR")
    sel_h3 = take(h3, FX["site_selection"]["hcdr3_backbone_extra"], "HCDR3")
    chi_pool = [k for k in var_keys if valid_chi1(k)]
    sel_chi = take(chi_pool, FX["site_selection"]["chi1"], "CHI1")

    sites = []
    for k in sel_fw:
        sites += [
            {"kind": "phi", "region": res_region.get(k, "VH_FW"), "chain": k[0], "resseq": k[1], "scope": "variable"},
            {"kind": "psi", "region": res_region.get(k, "VH_FW"), "chain": k[0], "resseq": k[1], "scope": "variable"},
        ]
    for k in sel_cdr:
        sites += [
            {"kind": "phi", "region": res_region.get(k, "CDR"), "chain": k[0], "resseq": k[1], "scope": "variable"},
            {"kind": "psi", "region": res_region.get(k, "CDR"), "chain": k[0], "resseq": k[1], "scope": "variable"},
        ]
    for k in sel_h3:
        sites += [
            {"kind": "phi", "region": "HCDR3", "chain": k[0], "resseq": k[1], "scope": "variable"},
            {"kind": "psi", "region": "HCDR3", "chain": k[0], "resseq": k[1], "scope": "variable"},
        ]
    for k in sel_chi:
        sites.append(
            {"kind": "chi1", "region": res_region.get(k, "FW"), "chain": k[0], "resseq": k[1], "scope": "variable"}
        )
    return sites, residues, res_region


def pick_extra_region_sites(ab_id, meta, vh_len, vl_len, residues, res_region, n_per=6):
    """Low-dim constant / interface site samples (target-blind)."""

    def valid_bb(key):
        return all(n in residues[key] for n in ("N", "CA", "C"))

    def take(pool, n, tag):
        if not pool:
            return []
        h = stable_hash(f"{ab_id}|{tag}|{SEED}")
        rng = np.random.default_rng(h % (2**32))
        order = np.argsort(rng.random(len(pool)))
        return [pool[i] for i in order[: min(n, len(pool))]]

    # Contact interfaces from CA distances
    ca = {}
    for i, m in enumerate(meta):
        if m["name"] == "CA":
            ca[(m["chain"], m["resseq"])] = meta and np.zeros(3)
    # rebuild from coords later — pass coords separately
    return []


def interface_residues(coords, meta, vh_len, vl_len, pair: str):
    """Frozen geometric contact: any heavy atom < 4.5 A between domains."""
    groups = {
        "VH": [(i, m) for i, m in enumerate(meta) if m["is_heavy"] and m["chain"] == "A" and m["resseq"] <= vh_len],
        "CH1": [(i, m) for i, m in enumerate(meta) if m["is_heavy"] and m["chain"] == "A" and m["resseq"] > vh_len],
        "VL": [(i, m) for i, m in enumerate(meta) if m["is_heavy"] and m["chain"] == "B" and m["resseq"] <= vl_len],
        "CL": [(i, m) for i, m in enumerate(meta) if m["is_heavy"] and m["chain"] == "B" and m["resseq"] > vl_len],
    }
    mapping = {"VH_CH1": ("VH", "CH1"), "VL_CL": ("VL", "CL"), "CH1_CL": ("CH1", "CL")}
    g1, g2 = mapping[pair]
    hit = set()
    for i, mi in groups[g1]:
        for j, mj in groups[g2]:
            if float(np.linalg.norm(coords[i] - coords[j])) < CONTACT_A:
                hit.add((mi["chain"], mi["resseq"]))
                hit.add((mj["chain"], mj["resseq"]))
    return sorted(hit)


def pick_region_sites(ab_id, residues, res_region, keys, n, tag):
    def valid_bb(key):
        return key in residues and all(n in residues[key] for n in ("N", "CA", "C"))

    pool = [k for k in keys if valid_bb(k)]
    if not pool:
        return []
    h = stable_hash(f"{ab_id}|{tag}|{SEED}")
    rng = np.random.default_rng(h % (2**32))
    order = np.argsort(rng.random(len(pool)))
    sel = [pool[i] for i in order[: min(n, len(pool))]]
    sites = []
    for k in sel:
        sites.append({"kind": "phi", "region": res_region.get(k, tag), "chain": k[0], "resseq": k[1], "scope": tag})
        sites.append({"kind": "psi", "region": res_region.get(k, tag), "chain": k[0], "resseq": k[1], "scope": tag})
    return sites


def rodrigues(coords, idxs, origin, axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    out = coords.copy()
    c, s = np.cos(ang), np.sin(ang)
    for i in idxs:
        v = out[i] - origin
        out[i] = origin + v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
    return out


def apply_torsion(coords, residues, site, delta_rad):
    key = (site["chain"], site["resseq"])
    if key not in residues:
        return None
    at = residues[key]
    kind = site["kind"]
    if kind == "phi":
        if not all(n in at for n in ("N", "CA")):
            return None
        origin = coords[at["N"]]
        axis = coords[at["CA"]] - coords[at["N"]]
        move = [i for n, i in at.items() if n not in ("N",) and not (n.startswith("H") and n in ("H", "H1", "H2", "H3"))]
        return rodrigues(coords, move, origin, axis, delta_rad)
    if kind == "psi":
        if not all(n in at for n in ("CA", "C")):
            return None
        origin = coords[at["CA"]]
        axis = coords[at["C"]] - coords[at["CA"]]
        move = [at[n] for n in ("O", "OXT") if n in at]
        nxt = (site["chain"], site["resseq"] + 1)
        if nxt in residues:
            for n in ("N", "H", "H1"):
                if n in residues[nxt]:
                    move.append(residues[nxt][n])
        return rodrigues(coords, move, origin, axis, delta_rad)
    if kind == "chi1":
        if not all(n in at for n in ("CA", "CB")):
            return None
        origin = coords[at["CA"]]
        axis = coords[at["CB"]] - coords[at["CA"]]
        move = []
        for n, i in at.items():
            if n in ("N", "CA", "C", "O", "OXT", "H", "HA", "CB", "HB1", "HB2", "HB3", "HB"):
                continue
            move.append(i)
        return rodrigues(coords, move, origin, axis, delta_rad)
    return None


def evaluate_sites(model, species, coords, residues, sites):
    E0, _ = energy_forces_model(model, species, coords)
    out = []
    for site in sites:
        deg = CHI1_DEG if site["kind"] == "chi1" else PHI_PSI_DEG
        delta = np.deg2rad(deg)
        cp = apply_torsion(coords, residues, site, +delta)
        cm = apply_torsion(coords, residues, site, -delta)
        if cp is None or cm is None:
            continue
        Ep, _ = energy_forces_model(model, species, cp)
        Em, _ = energy_forces_model(model, species, cm)
        Ke = (Ep + Em - 2.0 * E0) / (delta * delta)
        rec = dict(site)
        rec["K_E"] = float(Ke)
        out.append(rec)
    return E0, out


def extract_fv_pdb(src: Path, dst: Path, vh_len: int, vl_len: int):
    class FV(Select):
        def accept_residue(self, residue):
            ch = residue.get_parent().id
            rs = residue.id[1]
            if ch == "A" and rs <= vh_len:
                return 1
            if ch == "B" and rs <= vl_len:
                return 1
            return 0

    st = PDBParser(QUIET=True).get_structure("x", str(src))
    io = PDBIO()
    io.set_structure(st)
    dst.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(dst), FV())


def add_hydrogens_simple(src: Path, dst: Path):
    """PDBFixer H-add for Fv extracts (no disulfide requirement)."""
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def feature_from_site_rows(ab_id, condition, site_rows):
    buckets = {
        "all_bb": [],
        "FW": [],
        "CDR": [],
        "HCDR3": [],
        "chi1": [],
        "VH_FW": [],
        "HCDR": [],
        "VL_FW": [],
        "LCDR": [],
        "LCDR3": [],
        "CH1": [],
        "CL": [],
        "VH_CH1": [],
        "VL_CL": [],
        "CH1_CL": [],
        "SS_neigh": [],
        "all": [],
    }
    for r in site_rows:
        ke = r["K_E"]
        buckets["all"].append(ke)
        if r["kind"] == "chi1":
            buckets["chi1"].append(ke)
        else:
            buckets["all_bb"].append(ke)
        reg = r.get("region", "")
        scope = r.get("scope", "")
        if reg in buckets:
            buckets[reg].append(ke)
        if reg in ("VH_FW", "VL_FW") or reg.endswith("FW"):
            buckets["FW"].append(ke)
        if reg in ("HCDR", "LCDR", "CDR"):
            buckets["CDR"].append(ke)
        if scope in buckets:
            buckets[scope].append(ke)
    feat = {"id": ab_id, "condition": condition, "extraction_status": "SUCCESS", "n_sites": len(site_rows)}
    for name, vals in buckets.items():
        ag = agg_K(vals)
        for k, v in ag.items():
            feat[f"K_{name}_{k}"] = v
    return feat


def process_condition(model, ab_id, condition, pdb: Path, vh_len, vl_len, do_relax: bool, extra_sites=True):
    t0 = time.time()
    species, coords0, meta = load_structure(pdb)
    if do_relax:
        coords1, E1, F1, converged, steps, fmax, opt_name = relax_r1(model, species, coords0, meta)
    else:
        coords1 = coords0.copy()
        E1, F1 = energy_forces_model(model, species, coords1)
        converged, steps, fmax, opt_name = True, 0, float(np.max(np.linalg.norm(F1, axis=1))), "NO_RELAX_MATCHED"

    sev0, _, _ = clash_counts(coords0, meta)
    sev1, _, min1 = clash_counts(coords1, meta)
    frm1, fq1, fmx1 = force_stats(F1)
    bb_idx = np.array([m["is_backbone_heavy"] for m in meta])
    bb_rmsd = float(np.sqrt(np.mean((coords1[bb_idx] - coords0[bb_idx]) ** 2))) if do_relax and bb_idx.any() else 0.0

    sites_var, residues, res_region = pick_variable_sites(ab_id, meta, vh_len, vl_len)
    sites = list(sites_var)

    if extra_sites and condition == "C":
        # constant domain sites
        ch1_keys = [k for k in residues if k[0] == "A" and k[1] > vh_len]
        cl_keys = [k for k in residues if k[0] == "B" and k[1] > vl_len]
        sites += pick_region_sites(ab_id, residues, res_region, ch1_keys, 6, "CH1")
        sites += pick_region_sites(ab_id, residues, res_region, cl_keys, 6, "CL")
        for pair, n, tag in [("VH_CH1", 4, "VH_CH1"), ("VL_CL", 4, "VL_CL"), ("CH1_CL", 4, "CH1_CL")]:
            keys = interface_residues(coords1, meta, vh_len, vl_len, pair)
            sites += pick_region_sites(ab_id, residues, res_region, keys, n, tag)
        # disulfide neighborhood (secondary): residues within 5A of HL SG
        # approximate: terminal Cys residues ±1
        hl_keys = [("A", vh_len + 103), ("B", vl_len + (107 if True else 105))]
        # better from topo later; use terminal Cys on each chain
        term = []
        for ch, mx in [("A", int(max(k[1] for k in residues if k[0] == "A"))), ("B", int(max(k[1] for k in residues if k[0] == "B")))]:
            for rs in range(max(1, mx - 2), mx + 1):
                if (ch, rs) in residues:
                    term.append((ch, rs))
        sites += pick_region_sites(ab_id, residues, res_region, term, 4, "SS_neigh")

    E0, site_rows = evaluate_sites(model, species, coords1, residues, sites)
    feat = feature_from_site_rows(ab_id, condition, site_rows)
    qc = {
        "id": ab_id,
        "condition": condition,
        "n_atoms": len(species),
        "R1_E": E1,
        "E0_eval": E0,
        "R1_F_rms": frm1,
        "R1_F_q90": fq1,
        "R1_F_max": fmx1,
        "R0_severe_clash": sev0,
        "R1_severe_clash": sev1,
        "R1_min_heavy_dist": min1,
        "backbone_RMSD": bb_rmsd,
        "converged": converged,
        "steps": steps,
        "final_fmax": fmax,
        "optimizer": opt_name,
        "runtime_s": time.time() - t0,
        "n_site_evals": len(site_rows),
    }
    # cache R1
    r1_path = CACHE / "r1" / f"{ab_id}_{condition}.npz"
    r1_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(r1_path, species=species, coords=coords1, E=E1, meta_chain=[m["chain"] for m in meta], meta_resseq=[m["resseq"] for m in meta], meta_name=[m["name"] for m in meta])
    return qc, feat, site_rows, (species, coords1, meta)


def write_r1_and_matched(prepared_pdb: Path, coords, meta, vh_len, vl_len, r1_pdb: Path, matched_pdb: Path):
    """Write full R1 PDB and matched Fv (identical Fv coords, CH1/CL removed)."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("c", str(prepared_pdb))
    # map (chain,resseq,name) -> coord index
    idx = {(m["chain"], m["resseq"], m["name"]): i for i, m in enumerate(meta)}
    for atom in st.get_atoms():
        res = atom.get_parent()
        key = (res.get_parent().id, int(res.id[1]), atom.get_name().strip())
        if key in idx:
            atom.coord = coords[idx[key]]
    r1_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(st)
    io.save(str(r1_pdb))

    class FV(Select):
        def accept_residue(self, residue):
            ch = residue.get_parent().id
            rs = residue.id[1]
            if ch == "A" and rs <= vh_len:
                return 1
            if ch == "B" and rs <= vl_len:
                return 1
            return 0

    matched_pdb.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(matched_pdb), FV())


def write_matched_fv_pdb(species, coords, meta, vh_len, vl_len, dst: Path):
    """Legacy fallback writer."""
    lines = []
    serial = 1
    for i, m in enumerate(meta):
        if m["chain"] == "A" and m["resseq"] > vh_len:
            continue
        if m["chain"] == "B" and m["resseq"] > vl_len:
            continue
        x, y, z = coords[i]
        el = m["element"]
        name = m["name"]
        # PDB atom name alignment
        if len(name) < 4:
            an = f" {name:<3s}"
        else:
            an = name[:4]
        lines.append(
            f"ATOM  {serial:5d} {an} {m.get('resname','UNK'):3s} {m['chain']}{m['resseq']:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}"
        )
        serial += 1
    lines.append("END")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import fennol

    model = fennol.load(str(MODEL))
    fab = pd.read_csv(SEQ).set_index("id")
    ids = PILOT if "--pilot" in sys.argv else fab.index.astype(str).tolist()
    if "--ids" in sys.argv:
        raw = sys.argv[sys.argv.index("--ids") + 1 :]
        ids = []
        for x in raw:
            if x.startswith("--"):
                break
            ids.append(x)

    conditions = {"B", "C", "M"}
    if "--conditions" in sys.argv:
        conditions = {x.strip().upper() for x in sys.argv[sys.argv.index("--conditions") + 1].split(",") if x.strip()}

    shard_tag = "all"
    if "--shard" in sys.argv:
        spec = sys.argv[sys.argv.index("--shard") + 1]
        si, sn = map(int, spec.split("/"))
        ids = [i for j, i in enumerate(ids) if j % sn == si]
        shard_tag = f"shard{si}of{sn}"
        print(f"SHARD {si}/{sn} n={len(ids)}", flush=True)

    prep_qc = pd.read_csv(CTX / "FAB_PREP_QC.csv") if (CTX / "FAB_PREP_QC.csv").exists() else pd.DataFrame()

    if "--pilot" in sys.argv:
        feat_path = CACHE / "features" / "pilot_features.csv"
        qc_path = CACHE / "features" / "pilot_qc.csv"
        sites_path = CACHE / "features" / "pilot_sites.jsonl"
    elif "--audit-tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--audit-tag") + 1]
        adir = CACHE / "cpu_cuda_audit" / tag
        adir.mkdir(parents=True, exist_ok=True)
        feat_path = adir / "features.csv"
        qc_path = adir / "qc.csv"
        sites_path = adir / "sites.jsonl"
    else:
        feat_path = CACHE / "features" / f"features_partial_{shard_tag}.csv"
        qc_path = CACHE / "features" / f"qc_partial_{shard_tag}.csv"
        sites_path = CACHE / "features" / f"sites_{shard_tag}.jsonl"
    feat_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    feat_rows, qc_rows = [], []
    if feat_path.exists() and "--fresh" not in sys.argv:
        prev = pd.read_csv(feat_path)
        feat_rows = prev.to_dict("records")
        done = {(r["id"], r["condition"]) for r in feat_rows if r.get("extraction_status") == "SUCCESS"}
    # Also treat global features_partial successes as done when not pilot/audit
    global_partial = CACHE / "features" / "features_partial.csv"
    if (not ("--pilot" in sys.argv)) and (not ("--audit-tag" in sys.argv)) and global_partial.exists() and "--fresh" not in sys.argv:
        g = pd.read_csv(global_partial)
        for r in g.to_dict("records"):
            if r.get("extraction_status") == "SUCCESS":
                done.add((r["id"], r["condition"]))
    if qc_path.exists() and "--fresh" not in sys.argv:
        qc_rows = pd.read_csv(qc_path).to_dict("records")

    print(f"resume done_pairs={len(done)} remaining_ids≈{len(ids)} conditions={sorted(conditions)}", flush=True)

    for ab_id in ids:
        row = fab.loc[ab_id]
        vh_len, vl_len = int(row.VH_len_used), int(row.VL_len_used)
        prepared = CACHE / "prepared_fab" / f"{ab_id}_prepared.pdb"
        renum = CACHE / "renumbered_fab" / f"{ab_id}.pdb"
        if not prepared.exists():
            print(f"SKIP {ab_id} no prepared", flush=True)
            continue
        if len(prep_qc) and ab_id in set(prep_qc.id.astype(str)):
            ok_val = prep_qc.set_index("id").loc[ab_id]
            if isinstance(ok_val, pd.DataFrame):
                ok_val = ok_val.iloc[-1]
            if not bool(ok_val.ok):
                print(f"SKIP {ab_id} prep not ok", flush=True)
                continue

        # B: Fab-geom Fv extract + H-add + R1
        if "B" in conditions and (ab_id, "B") not in done:
            print(f"COND B {ab_id}", flush=True)
            fv_raw = CACHE / "fabgeom_fv" / f"{ab_id}_fv.pdb"
            fv_h = CACHE / "fabgeom_fv" / f"{ab_id}_fv_h.pdb"
            if not renum.exists():
                extract_fv_pdb(prepared, fv_raw, vh_len, vl_len)
            else:
                extract_fv_pdb(renum, fv_raw, vh_len, vl_len)
            if not fv_h.exists():
                try:
                    add_hydrogens_simple(fv_raw, fv_h)
                except Exception:
                    raise RuntimeError(f"Missing H-added Fv PDB: {fv_h}")
            qc, feat, site_rows, _ = process_condition(model, ab_id, "B", fv_h, vh_len, vl_len, do_relax=True, extra_sites=False)
            feat_rows.append(feat)
            qc_rows.append(qc)
            with sites_path.open("a") as fh:
                for s in site_rows:
                    fh.write(json.dumps({"id": ab_id, "condition": "B", **s}) + "\n")
            pd.DataFrame(feat_rows).to_csv(feat_path, index=False)
            pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

        # C: full Fab prepared + R1 + region sites
        if "C" in conditions and (ab_id, "C") not in done:
            print(f"COND C {ab_id}", flush=True)
            qc, feat, site_rows, pack = process_condition(model, ab_id, "C", prepared, vh_len, vl_len, do_relax=True, extra_sites=True)
            feat_rows.append(feat)
            qc_rows.append(qc)
            with sites_path.open("a") as fh:
                for s in site_rows:
                    fh.write(json.dumps({"id": ab_id, "condition": "C", **s}) + "\n")
            species, coords1, meta = pack
            r1_pdb = CACHE / "r1" / f"{ab_id}_C_r1.pdb"
            matched = CACHE / "matched_fv" / f"{ab_id}_matched.pdb"
            write_r1_and_matched(prepared, coords1, meta, vh_len, vl_len, r1_pdb, matched)
            pd.DataFrame(feat_rows).to_csv(feat_path, index=False)
            pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

        # M: matched Fv from C, no re-relax
        if "M" in conditions and (ab_id, "M") not in done:
            print(f"COND M {ab_id}", flush=True)
            matched = CACHE / "matched_fv" / f"{ab_id}_matched.pdb"
            if not matched.exists():
                print("  missing matched pdb", flush=True)
            else:
                qc, feat, site_rows, _ = process_condition(
                    model, ab_id, "M", matched, vh_len, vl_len, do_relax=False, extra_sites=False
                )
                feat_rows.append(feat)
                qc_rows.append(qc)
                with sites_path.open("a") as fh:
                    for s in site_rows:
                        fh.write(json.dumps({"id": ab_id, "condition": "M", **s}) + "\n")
                pd.DataFrame(feat_rows).to_csv(feat_path, index=False)
                pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass

    print("DONE", pd.DataFrame(feat_rows).groupby(["condition", "extraction_status"]).size().to_dict() if feat_rows else {}, flush=True)


if __name__ == "__main__":
    main()
