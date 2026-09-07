#!/usr/bin/env python3
"""Gate F0: topology-based Fab disulfide audit + chemically explicit OpenMM prep."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FAB = FP / "fab_reconstruction"
CTX = FP / "fennix_fab_context"
SEQ = FAB / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
PDB_DIR = FAB / "structures/esmfold_fab"
OUT_TOPO = CTX / "FAB_DISULFIDE_TOPOLOGY.csv"
OUT_PREP_QC = CTX / "FAB_PREP_QC.csv"
OUT_PREP_DIR = CTX / "cache/prepared_fab"
OUT_QC_MD = CTX / "FAB_DISULFIDE_QC.md"
PILOT = json.loads((FAB / "structures/pilot_ids.json").read_text())

# Constant-domain Cys offsets (1-based within constant segment)
CH1_INTRA = (27, 83)
CH1_HL = 103  # EPKSC terminal
CK_INTRA = (27, 87)
CK_HL = 107
CL_INTRA = (28, 87)
CL_HL = 105


def cys_positions(seq: str) -> list[int]:
    return [i + 1 for i, a in enumerate(seq) if a == "C"]


def pick_variable_intra(seq: str, domain: str) -> tuple[int, int] | None:
    """Topology-first VH/VL intradomain: early FR Cys + late FR Cys (not CDR extras)."""
    cs = cys_positions(seq)
    if len(cs) < 2:
        return None
    early = [c for c in cs if c <= 40]
    late = [c for c in cs if 70 <= c <= len(seq)]
    if not early or not late:
        # fallback: first and second-to-last if terminal Cys rare in V
        if len(cs) >= 2:
            return cs[0], cs[1] if len(cs) == 2 else cs[-1] if cs[-1] < len(seq) - 2 else cs[-2]
        return None
    c1 = early[0]
    # prefer late Cys closest to canonical ~92–96 window
    target = 92 if domain == "VH" else 88
    c2 = min(late, key=lambda c: abs(c - target))
    if c1 == c2:
        return None
    return c1, c2


def expected_pairs(row: pd.Series) -> list[dict]:
    vh = row.VH_used
    vl = row.VL_used
    vh_len = int(row.VH_len_used)
    vl_len = int(row.VL_len_used)
    locus = str(row.light_locus)
    pairs = []

    vi = pick_variable_intra(vh, "VH")
    if vi:
        pairs.append({"bond": "VH_intra", "chain": "A", "res1": vi[0], "res2": vi[1], "role": "intradomain"})
    li = pick_variable_intra(vl, "VL")
    if li:
        pairs.append({"bond": "VL_intra", "chain": "B", "res1": li[0], "res2": li[1], "role": "intradomain"})

    # CH1 / CL relative to Fab residue numbering (chain A/B continuous)
    c1a, c1b = CH1_INTRA
    pairs.append(
        {
            "bond": "CH1_intra",
            "chain": "A",
            "res1": vh_len + c1a,
            "res2": vh_len + c1b,
            "role": "intradomain",
        }
    )
    if locus == "kappa":
        cl_a, cl_b = CK_INTRA
        hl_l = vl_len + CK_HL
    else:
        cl_a, cl_b = CL_INTRA
        hl_l = vl_len + CL_HL
    pairs.append(
        {
            "bond": "CL_intra",
            "chain": "B",
            "res1": vl_len + cl_a,
            "res2": vl_len + cl_b,
            "role": "intradomain",
        }
    )
    pairs.append(
        {
            "bond": "HL_inter",
            "chain": "A:B",
            "res1": vh_len + CH1_HL,
            "res2": hl_l,
            "role": "interchain",
            "heavy_chain": "A",
            "light_chain": "B",
            "heavy_res": vh_len + CH1_HL,
            "light_res": hl_l,
        }
    )
    return pairs


def find_res(structure, chain: str, resseq: int):
    """Find residue by chain + resseq ignoring hetero flag (CYX may be H_CYX)."""
    try:
        ch = structure[0][chain]
    except KeyError:
        return None
    for res in ch:
        if res.id[1] == resseq:
            return res
    return None


def sg_coord(structure, chain: str, resseq: int):
    res = find_res(structure, chain, resseq)
    if res is not None and "SG" in res:
        return np.asarray(res["SG"].coord, float)
    return None


def dist_sg(structure, chain1, res1, chain2, res2):
    a = sg_coord(structure, chain1, res1)
    b = sg_coord(structure, chain2, res2)
    if a is None or b is None:
        return np.nan
    return float(np.linalg.norm(a - b))


def ca_coords(struct, chain, r0, r1):
    xs = []
    for resseq in range(r0, r1 + 1):
        res = find_res(struct, chain, resseq)
        if res is not None and "CA" in res:
            xs.append(res["CA"].coord)
    return np.asarray(xs, float) if xs else np.zeros((0, 3))


def load_renumbered(ab_id: str):
    raw = PDB_DIR / f"{ab_id}.pdb"
    ren = CTX / "cache/renumbered_fab" / f"{ab_id}.pdb"
    if raw.exists():
        renumber_fab_pdb(raw, ren)
        return PDBParser(QUIET=True).get_structure(ab_id, str(ren))
    return None


def audit_topology(ids: list[str] | None = None) -> pd.DataFrame:
    fab = pd.read_csv(SEQ)
    if ids:
        fab = fab[fab.id.isin(ids)].copy()
    rows = []
    for _, r in fab.iterrows():
        struct = load_renumbered(r.id)
        for p in expected_pairs(r):
            if p["bond"] == "HL_inter":
                ch1, r1, ch2, r2 = "A", p["heavy_res"], "B", p["light_res"]
            else:
                ch1 = ch2 = p["chain"]
                r1, r2 = p["res1"], p["res2"]
            d = dist_sg(struct, ch1, r1, ch2, r2) if struct is not None else np.nan
            rows.append(
                {
                    "id": r.id,
                    "light_locus": r.light_locus,
                    "bond": p["bond"],
                    "role": p["role"],
                    "chain1": ch1,
                    "res1": r1,
                    "chain2": ch2,
                    "res2": r2,
                    "raw_SG_SG_A": d,
                    "topology_source": "sequence_domain_Ig_canonical_renumbered_1based",
                }
            )
    return pd.DataFrame(rows)


def residue_lookup(topology):
    """Map (chainId, resSeq:int) -> Residue. OpenMM stores res.id as str."""
    out = {}
    for chain in topology.chains():
        for res in chain.residues():
            out[(str(chain.id), int(res.id))] = res
    return out


def renumber_fab_pdb(src: Path, dst: Path) -> dict:
    """Rewrite Fab PDB with chain A=1..H, chain B=1..L (ESMFold often offsets B)."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("fab", str(src))
    mapping = {"A": {}, "B": {}}
    for chain in st[0]:
        cid = chain.id
        residues = [r for r in chain if r.id[0] == " "]
        for i, res in enumerate(residues, start=1):
            old = res.id[1]
            mapping[cid][old] = i
            res.id = (" ", i, " ")
    io = PDBIO()
    io.set_structure(st)
    dst.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(dst))
    return mapping


def _sanitize_prepared_pdb(src: Path, dst: Path, pairs: list[dict]):
    """Rewrite OpenMM PDB: CYX->CYS, blank hetero, strip HG on bonded cysteines."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", str(src))
    bonded = set()
    for p in pairs:
        if p["bond"] == "HL_inter":
            bonded.add(("A", p["heavy_res"]))
            bonded.add(("B", p["light_res"]))
        else:
            bonded.add((p["chain"], p["res1"]))
            bonded.add((p["chain"], p["res2"]))
    for chain in st[0]:
        for res in list(chain):
            if res.resname in ("CYX", "CYS"):
                res.resname = "CYS"
                # Clear hetero flag so FeNNix/BioPython see standard residues
                res.id = (" ", res.id[1], res.id[2])
                if (chain.id, res.id[1]) in bonded:
                    for atom in list(res):
                        if atom.get_name() in ("HG", "HG1"):
                            res.detach_child(atom.get_id())
    io = PDBIO()
    io.set_structure(st)
    io.save(str(dst))


def prepare_one(ab_id: str, row: pd.Series, pairs: list[dict], out_pdb: Path) -> dict:
    """Chemically explicit disulfide prep via PDBFixer + OpenMM restrained min."""
    from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
    from pdbfixer import PDBFixer

    t0 = time.time()
    raw = PDB_DIR / f"{ab_id}.pdb"
    src = CTX / "cache/renumbered_fab" / f"{ab_id}.pdb"
    renumber_fab_pdb(raw, src)
    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    # Hydrogens after we mark disulfides: use Modeller
    modeller = Modeller(fixer.topology, fixer.positions)
    resmap = residue_lookup(modeller.topology)

    # Build disulfide residue pairs for createDisulfideBond
    ss_res = []
    for p in pairs:
        if p["bond"] == "HL_inter":
            key1 = ("A", p["heavy_res"])
            key2 = ("B", p["light_res"])
        else:
            key1 = (p["chain"], p["res1"])
            key2 = (p["chain"], p["res2"])
        r1 = resmap.get(key1)
        r2 = resmap.get(key2)
        if r1 is None or r2 is None:
            return {"id": ab_id, "ok": False, "error": f"MISSING_RES {key1}/{key2}", "runtime_s": time.time() - t0}
        ss_res.append((r1, r2))

    # Do NOT call modeller.createDisulfideBond: terminal Cys become CCYX and
    # amber99sbildn templates fail. Enforce S–S chemically via CustomBondForce
    # after stripping thiol hydrogens.
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    try:
        modeller.addHydrogens(ff, pH=7.0)
    except Exception:
        fixer2 = PDBFixer(topology=modeller.topology, positions=modeller.positions)
        fixer2.addMissingHydrogens(7.0)
        modeller = Modeller(fixer2.topology, fixer2.positions)

    # Keep thiol H for ForceField templates (CYS-without-HG matches CYX/CCYX and
    # fails when no FF disulfide bond exists). SS geometry enforced by CustomBondForce;
    # thiol H stripped only in the sanitized FeNNix output PDB.
    resmap = residue_lookup(modeller.topology)

    # Vacuum minimization with backbone restraints (no solvent for speed / FeNNix input)
    ff_prot = ForceField("amber99sbildn.xml")
    try:
        system = ff_prot.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
            removeCMMotion=False,
        )
    except Exception as e:
        return {"id": ab_id, "ok": False, "error": f"CREATE_SYSTEM:{type(e).__name__}:{e}", "runtime_s": time.time() - t0}

    # Harmonic restraint on backbone heavy atoms (strong; preserve fold)
    force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 500.0 * unit.kilocalories_per_mole / unit.angstrom**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    positions = modeller.positions
    bb = {"N", "CA", "C", "O"}
    for atom in modeller.topology.atoms():
        if atom.name in bb:
            pos = positions[atom.index]
            force.addParticle(atom.index, [pos.x, pos.y, pos.z])
    system.addForce(force)

    # Explicit SG–SG harmonic tethers so chemically intended disulfides form even if FF bond missing
    from openmm import CustomBondForce

    sg_map = {}
    for atom in modeller.topology.atoms():
        if atom.name == "SG":
            sg_map[(str(atom.residue.chain.id), int(atom.residue.id))] = atom.index
    ss_bond = CustomBondForce("0.5*kss*(r-r0)^2")
    ss_bond.addGlobalParameter("kss", 800.0 * unit.kilocalories_per_mole / unit.angstrom**2)
    ss_bond.addGlobalParameter("r0", 2.05 * unit.angstrom)
    n_ss_added = 0
    sg_pairs = []
    for p in pairs:
        if p["bond"] == "HL_inter":
            k1, k2 = ("A", p["heavy_res"]), ("B", p["light_res"])
        else:
            k1, k2 = (p["chain"], p["res1"]), (p["chain"], p["res2"])
        if k1 in sg_map and k2 in sg_map:
            ss_bond.addBond(sg_map[k1], sg_map[k2], [])
            sg_pairs.append((sg_map[k1], sg_map[k2]))
            n_ss_added += 1
    if n_ss_added:
        system.addForce(ss_bond)

    # Zero nonbonded between bonded SG atoms so tether can reach ~2.05 A
    from openmm import NonbondedForce

    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            for i, j in sg_pairs:
                force.addException(i, j, 0.0, 1.0, 0.0, replace=True)
            break

    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    import os as _os
    prefer = _os.environ.get("FENNIX_PREP_PLATFORM", "CUDA").upper()
    platform_name = "CPU"
    props = {}
    cpu_threads = int(_os.environ.get("OPENMM_CPU_THREADS", "16"))
    if prefer != "CPU":
        try:
            platform = Platform.getPlatformByName("CUDA")
            props = {"CudaPrecision": "mixed", "DeviceIndex": "0"}
            sim = Simulation(modeller.topology, system, integrator, platform, props)
            platform_name = "CUDA"
        except Exception:
            platform = Platform.getPlatformByName("CPU")
            props = {"Threads": str(cpu_threads)}
            sim = Simulation(modeller.topology, system, integrator, platform, props)
            platform_name = "CPU"
    else:
        platform = Platform.getPlatformByName("CPU")
        props = {"Threads": str(cpu_threads)}
        sim = Simulation(modeller.topology, system, integrator, platform, props)
        platform_name = "CPU"

    # Verify OpenMM CPU thread property when on CPU
    openmm_threads_reported = None
    if platform_name == "CPU":
        try:
            openmm_threads_reported = sim.context.getPlatform().getPropertyValue(sim.context, "Threads")
        except Exception:
            try:
                openmm_threads_reported = platform.getPropertyDefaultValue("Threads")
            except Exception:
                openmm_threads_reported = props.get("Threads")
        print(
            f"  OpenMM platform=CPU requested_Threads={cpu_threads} reported_Threads={openmm_threads_reported}",
            flush=True,
        )
        if str(openmm_threads_reported) != str(cpu_threads):
            print(
                f"  WARNING: OpenMM Threads mismatch (want {cpu_threads}, got {openmm_threads_reported})",
                flush=True,
            )

    sim.context.setPositions(positions)
    state0 = sim.context.getState(getEnergy=True, getPositions=True)
    e0 = state0.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    pos0 = state0.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    sim.minimizeEnergy(maxIterations=800)
    state1 = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state1.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    pos1 = state1.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdb = out_pdb.with_suffix(".openmm.pdb")
    with open(tmp_pdb, "w") as fh:
        PDBFile.writeFile(sim.topology, state1.getPositions(), fh, keepIds=True)

    # Sanitize for FeNNix: CYX->CYS, clear hetero flag, keep no thiol H on bonded Cys
    _sanitize_prepared_pdb(tmp_pdb, out_pdb, pairs)
    tmp_pdb.unlink(missing_ok=True)

    parser = PDBParser(QUIET=True)
    st = parser.get_structure(ab_id, str(out_pdb))
    d_after = {}
    d_before = {}
    src_st = parser.get_structure("raw", str(src))
    for p in pairs:
        if p["bond"] == "HL_inter":
            args = ("A", p["heavy_res"], "B", p["light_res"])
        else:
            args = (p["chain"], p["res1"], p["chain"], p["res2"])
        d_after[p["bond"]] = dist_sg(st, *args)
        d_before[p["bond"]] = dist_sg(src_st, *args)

    bb_idx = [atom.index for atom in modeller.topology.atoms() if atom.name in bb]
    if bb_idx:
        dpos = pos1[bb_idx] - pos0[bb_idx]
        bb_rmsd = float(np.sqrt(np.mean(np.sum(dpos**2, axis=1))))
    else:
        bb_rmsd = np.nan

    vh_len = int(row.VH_len_used)
    vl_len = int(row.VL_len_used)
    heavy_len = int(row.heavy_length)
    light_len = int(row.light_length)

    def domain_rmsd(ch, a0, a1):
        c0 = ca_coords(src_st, ch, a0, a1)
        c1 = ca_coords(st, ch, a0, a1)
        n = min(len(c0), len(c1))
        if n < 3:
            return np.nan
        return float(np.sqrt(np.mean(np.sum((c0[:n] - c1[:n]) ** 2, axis=1))))

    coords = []
    meta = []
    for a in st.get_atoms():
        el = (a.element or "X").strip().upper()
        if el == "H":
            continue
        coords.append(a.coord)
        res = a.get_parent()
        meta.append((res.get_parent().id, res.id[1]))
    coords = np.asarray(coords, float)
    severe = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if meta[i][0] == meta[j][0] and abs(meta[i][1] - meta[j][1]) <= 1:
                continue
            if float(np.linalg.norm(coords[i] - coords[j])) < 1.5:
                severe += 1

    hl = d_after.get("HL_inter", np.nan)
    intra = [d_after.get(k, np.nan) for k in ("VH_intra", "VL_intra", "CH1_intra", "CL_intra")]
    intra_ok = sum(1 for d in intra if np.isfinite(d) and d < 2.85)
    # Pass if HL formed into disulfide-compatible range, backbone barely moved,
    # majority intradomain formed, no large domain drift. Report distributions;
    # do not hard-code biology from a single cutoff.
    ok = bool(
        np.isfinite(hl)
        and hl < 2.85
        and bb_rmsd < 0.5
        and intra_ok >= 3
        and domain_rmsd("A", 1, vh_len) < 1.0
        and domain_rmsd("B", 1, vl_len) < 1.0
    )

    return {
        "id": ab_id,
        "ok": ok,
        "error": "" if ok else f"QC_FAIL HL={hl:.3f} intra_ok={intra_ok}/4 bb={bb_rmsd:.3f}",
        "light_locus": row.light_locus,
        "n_ss_tethers": n_ss_added,
        "energy_before_kcal": e0,
        "energy_after_kcal": e1,
        "backbone_RMSD": bb_rmsd,
        "VH_CA_RMSD": domain_rmsd("A", 1, vh_len),
        "VL_CA_RMSD": domain_rmsd("B", 1, vl_len),
        "CH1_CA_RMSD": domain_rmsd("A", vh_len + 1, heavy_len),
        "CL_CA_RMSD": domain_rmsd("B", vl_len + 1, light_len),
        "HL_SG_SG_before": d_before.get("HL_inter", np.nan),
        "HL_SG_SG_after": hl,
        "VH_intra_SG_SG_after": d_after.get("VH_intra", np.nan),
        "VL_intra_SG_SG_after": d_after.get("VL_intra", np.nan),
        "CH1_intra_SG_SG_after": d_after.get("CH1_intra", np.nan),
        "CL_intra_SG_SG_after": d_after.get("CL_intra", np.nan),
        "severe_clash_after": severe,
        "n_atoms_out": sum(1 for _ in st.get_atoms()),
        "runtime_s": time.time() - t0,
        "out_pdb": str(out_pdb),
        "prep_platform": platform_name,
        "prep_batch": __import__("os").environ.get("FENNIX_PREP_BATCH", ""),
        "prep_attempt": __import__("os").environ.get("FENNIX_PREP_ATTEMPT", "1"),
        "wall_time": time.time() - t0,
        "openmm_cpu_threads_requested": cpu_threads if platform_name == "CPU" else None,
        "openmm_cpu_threads_reported": openmm_threads_reported if platform_name == "CPU" else None,
    }


def write_qc_md(topo: pd.DataFrame, prep: pd.DataFrame):
    lines = [
        "# Fab Disulfide Preparation QC",
        "",
        "## Topology rule",
        "- Intradomain VH/VL: earliest Cys ≤40 + late Cys nearest canonical (~92 VH / ~88 VL) among Cys ≥70.",
        "- CH1 intradomain: constant offsets 27–83; HL: CH1 terminal Cys (EPKSC +103).",
        "- CL intradomain: κ 27–87 / λ 28–87; HL light terminal Cys.",
        "- κ and λ audited separately; topology from sequence/domain, **not** nearest-neighbor alone.",
        "",
        "## Raw ESMFold SG–SG (all bonds in topology table)",
    ]
    if len(topo):
        g = topo.groupby(["light_locus", "bond"])["raw_SG_SG_A"].describe()
        lines.append("```")
        lines.append(g.to_string())
        lines.append("```")
    lines += ["", "## After OpenMM restrained prep"]
    if len(prep):
        lines.append(f"- n={len(prep)} ok={int(prep.ok.sum())} fail={int((~prep.ok).sum())}")
        for col in ["HL_SG_SG_after", "backbone_RMSD", "severe_clash_after"]:
            if col in prep.columns:
                lines.append(
                    f"- {col}: median={prep[col].median():.3f} q90={prep[col].quantile(0.9):.3f}"
                )
        if "light_locus" in prep.columns:
            for loc in ["kappa", "lambda"]:
                sub = prep[prep.light_locus == loc]
                if len(sub) and "HL_SG_SG_after" in sub.columns:
                    lines.append(
                        f"- {loc} HL after: median={sub.HL_SG_SG_after.median():.3f} "
                        f"ok={int(sub.ok.sum())}/{len(sub)}"
                    )
    lines += [
        "",
        "## Interpretation note",
        "Proximity / SSBOND records alone are insufficient; this prep defines bonds and relaxes local geometry.",
        "Do not hard-code biology from a single cutoff — distributions are primary.",
        "",
    ]
    OUT_QC_MD.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--topology-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = args.ids
    if args.pilot:
        ids = PILOT
    fab = pd.read_csv(SEQ)
    if ids:
        fab = fab[fab.id.isin(ids)].copy()
    if args.limit:
        fab = fab.head(args.limit)

    # Always refresh topology for requested set (merge into full later)
    topo_all = audit_topology(None)
    if OUT_TOPO.exists() and ids:
        prev = pd.read_csv(OUT_TOPO)
        prev = prev[~prev.id.isin(fab.id)]
        topo = pd.concat([prev, topo_all[topo_all.id.isin(fab.id)]], ignore_index=True)
    else:
        topo = topo_all
    topo.to_csv(OUT_TOPO, index=False)
    print("wrote", OUT_TOPO, len(topo))

    if args.topology_only:
        write_qc_md(topo, pd.DataFrame())
        return

    prep_rows = []
    if OUT_PREP_QC.exists():
        old = pd.read_csv(OUT_PREP_QC)
        done = set(old.loc[old.get("ok", False) == True, "id"].astype(str)) if "ok" in old.columns else set()
    else:
        old = pd.DataFrame()
        done = set()

    force = bool(args.ids) or args.pilot
    for _, r in fab.iterrows():
        if (not force) and (r.id in done):
            continue
        pairs = expected_pairs(r)
        out_pdb = OUT_PREP_DIR / f"{r.id}_prepared.pdb"
        print(f"[prep] {r.id} …", flush=True)
        try:
            qc = prepare_one(r.id, r, pairs, out_pdb)
        except Exception as e:
            qc = {"id": r.id, "ok": False, "error": f"{type(e).__name__}:{e}", "light_locus": r.light_locus}
        prep_rows.append(qc)
        print(f"  ok={qc.get('ok')} HL={qc.get('HL_SG_SG_after')} err={str(qc.get('error',''))[:120]}", flush=True)
        # Atomic completion marker (required for success accounting)
        if qc.get("ok") and out_pdb.exists():
            marker = OUT_PREP_DIR / f"{r.id}_prepared.complete"
            marker.write_text(
                json.dumps(
                    {
                        "id": r.id,
                        "ok": True,
                        "out_pdb": str(out_pdb),
                        "prep_platform": qc.get("prep_platform"),
                        "prep_batch": qc.get("prep_batch"),
                        "wall_time": qc.get("wall_time"),
                    },
                    indent=2,
                )
                + "\n"
            )
        # incremental save
        new_tmp = pd.DataFrame(prep_rows)
        if len(old) and "id" in old.columns:
            merged = pd.concat([old[~old["id"].isin(new_tmp["id"])], new_tmp], ignore_index=True)
        else:
            merged = new_tmp
        merged.to_csv(OUT_PREP_QC, index=False)

    new = pd.DataFrame(prep_rows)
    if len(new) == 0:
        prep = old
    elif len(old) and "id" in old.columns:
        old = old[~old["id"].isin(new["id"])]
        prep = pd.concat([old, new], ignore_index=True)
    else:
        prep = new
    prep.to_csv(OUT_PREP_QC, index=False)
    write_qc_md(topo, prep)
    print("wrote", OUT_PREP_QC, OUT_QC_MD)


if __name__ == "__main__":
    main()
