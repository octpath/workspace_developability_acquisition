#!/usr/bin/env python3
"""OpenMM Fab MD endgame — DEV campaign (0.5 ns implicit OBC2)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "openmm_fab_md"
CACHE = OUT / "cache" / "dev"
RES = OUT / "results"
R1 = FP / "fennix_fab_context/cache/r1"
SEQ = FP / "fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
ENV = ROOT / ".mamba/envs/openmm_cuda"
FREEZE = json.loads((OUT / "OPENMM_MD_FEATURE_FREEZE.json").read_text())

os.environ["LD_LIBRARY_PATH"] = f"{ENV}/lib:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENMM_PLUGIN_DIR"] = str(ENV / "lib/plugins")

DT_PS = 0.002
EQ_STEPS = 2500  # 5 ps
DUMP_STEPS = 5000  # 10 ps
CONTACT_A = 4.5
BB = {"N", "CA", "C", "O"}
ARO = {"PHE", "TYR", "TRP"}
HYDRO = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
GATE_H = 6.0


def seed_for(aid: str) -> int:
    return int.from_bytes(hashlib.sha256(aid.encode()).digest()[:4], "little")


def setup_openmm():
    from openmm import Platform

    Platform.loadPluginsFromDirectory(str(ENV / "lib/plugins"))
    names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    assert "CUDA" in names, names


def build_meta_from_openmm(topology, vh_len: int, vl_len: int):
    meta = []
    for atom in topology.atoms():
        res = atom.residue
        chain = res.chain.id
        # OpenMM residue id may be string
        try:
            resseq = int(res.id)
        except Exception:
            resseq = int("".join(ch for ch in str(res.id) if ch.isdigit()) or 0)
        name = atom.name
        element = (atom.element.symbol if atom.element is not None else name[0]).upper()
        if chain == "A":
            dom = "VH" if resseq <= vh_len else "CH1"
        elif chain == "B":
            dom = "VL" if resseq <= vl_len else "CL"
        else:
            dom = "OTHER"
        meta.append(
            {
                "chain": chain,
                "resseq": resseq,
                "resname": res.name,
                "name": name,
                "is_bb": name in BB,
                "is_ca": name == "CA",
                "is_heavy": element != "H",
                "domain": dom,
            }
        )
    return meta


def coords_to_bio_structure(topology, coords_A):
    """Build a Bio.PDB structure matching OpenMM atom order for SASA."""
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    structure = Structure("omm")
    model = Model(0)
    structure.add(model)
    chains = {}
    res_map = {}
    for i, atom in enumerate(topology.atoms()):
        res = atom.residue
        cid = res.chain.id
        if cid not in chains:
            chains[cid] = Chain(cid)
            model.add(chains[cid])
        try:
            resseq = int(res.id)
        except Exception:
            resseq = int("".join(ch for ch in str(res.id) if ch.isdigit()) or 0)
        key = (cid, resseq, res.name)
        if key not in res_map:
            r = Residue((" ", resseq, " "), res.name, "")
            chains[cid].add(r)
            res_map[key] = r
        else:
            r = res_map[key]
        element = atom.element.symbol if atom.element is not None else atom.name[0]
        aname = atom.name
        full = Atom(aname, coords_A[i], 0.0, 1.0, " ", aname, i, element=element)
        try:
            r.add(full)
        except Exception:
            full = Atom(aname + str(i % 10), coords_A[i], 0.0, 1.0, " ", aname, i, element=element)
            r.add(full)
    return structure


def sasa_sums_structure(structure):
    from Bio.PDB import SASA

    SASA.ShrakeRupley().compute(structure, level="R")
    aro = hydro = tyr = phe = trp = 0.0
    for r in structure.get_residues():
        v = float(getattr(r, "sasa", 0.0) or 0.0)
        rn = r.resname
        if rn in ARO:
            aro += v
        if rn in HYDRO:
            hydro += v
        if rn == "TYR":
            tyr += v
        elif rn == "PHE":
            phe += v
        elif rn == "TRP":
            trp += v
    return aro, hydro, tyr, phe, trp


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    a2 = a @ R
    return float(np.sqrt(((a2 - b) ** 2).sum(axis=1).mean()))


def rg(coords):
    c = coords - coords.mean(0, keepdims=True)
    return float(np.sqrt((c * c).sum(axis=1).mean()))


def native_pairs(coords, meta, dom_a, dom_b):
    ia = [i for i, m in enumerate(meta) if m["domain"] == dom_a and m["is_heavy"]]
    ib = [i for i, m in enumerate(meta) if m["domain"] == dom_b and m["is_heavy"]]
    if not ia or not ib:
        return []
    A = coords[ia]
    B = coords[ib]
    pairs = []
    for ii, p in zip(ia, A):
        d = np.linalg.norm(B - p, axis=1)
        for jloc in np.where(d <= CONTACT_A)[0]:
            pairs.append((ii, ib[int(jloc)]))
    return pairs


def occupancy(coords, pairs):
    if not pairs:
        return 1.0
    ok = sum(1 for i, j in pairs if np.linalg.norm(coords[i] - coords[j]) <= CONTACT_A)
    return ok / len(pairs)


def sasa_from_coords(template_struct, coords, meta):
    from Bio.PDB import SASA

    s = template_struct.copy()
    atoms = list(s.get_atoms())
    assert len(atoms) == len(coords)
    for a, c in zip(atoms, coords):
        a.set_coord(c)
    SASA.ShrakeRupley().compute(s, level="R")
    aro = hydro = tyr = phe = trp = 0.0
    for r in s.get_residues():
        v = float(getattr(r, "sasa", 0.0) or 0.0)
        rn = r.resname
        if rn in ARO:
            aro += v
        if rn in HYDRO:
            hydro += v
        if rn == "TYR":
            tyr += v
        elif rn == "PHE":
            phe += v
        elif rn == "TRP":
            trp += v
    return aro, hydro, tyr, phe, trp


def run_md(aid: str, pdb_path: Path, n_prod_steps: int, seed: int, vh_len: int, vl_len: int):
    import openmm
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, PDBFile, Simulation

    setup_openmm()
    pdb = PDBFile(str(pdb_path))
    ff = ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = Modeller(pdb.topology, pdb.positions)
    try:
        modeller.addHydrogens(ff)
    except Exception:
        pass
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=openmm.app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
    )
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(int(seed) % (2**31 - 1))
    platform = Platform.getPlatformByName("CUDA")
    props = {"DeviceIndex": "0", "Precision": "mixed"}
    sim = Simulation(modeller.topology, system, integrator, platform, props)
    sim.context.setPositions(modeller.positions)
    meta = build_meta_from_openmm(modeller.topology, vh_len, vl_len)

    constraint_failure = False
    try:
        sim.minimizeEnergy(maxIterations=200)
    except Exception:
        constraint_failure = True
        raise

    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, int(seed) % (2**31 - 1))
    sim.step(EQ_STEPS)

    frames = []
    energies = []
    st0 = sim.context.getState(getPositions=True, getEnergy=True)
    pos0 = st0.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    frames.append(np.array(pos0, float))
    energies.append(float(st0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)))

    done = 0
    while done < n_prod_steps:
        chunk = min(DUMP_STEPS, n_prod_steps - done)
        sim.step(chunk)
        done += chunk
        st = sim.context.getState(getPositions=True, getEnergy=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        frames.append(np.array(pos, float))
        e = float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        energies.append(e)

    temp_mean = 300.0
    try:
        from openmm import unit as u

        stf = sim.context.getState(getEnergy=True)
        ke = stf.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
        nat = modeller.topology.getNumAtoms()
        kB = 0.008314462618
        temp_mean = float(ke / (0.5 * max(3 * nat - 3, 1) * kB))
    except Exception:
        temp_mean = 300.0

    finite_energy = bool(np.isfinite(energies).all())
    finite_coords = bool(all(np.isfinite(f).all() for f in frames))
    return {
        "frames": frames,
        "energies": energies,
        "finite_energy": finite_energy,
        "finite_coordinates": finite_coords,
        "constraint_failure": constraint_failure,
        "temperature_mean": temp_mean,
        "natoms": modeller.topology.getNumAtoms(),
        "topology": modeller.topology,
        "meta": meta,
    }


def extract_features(aid, frames, meta, topology, qc_extra):
    ref = frames[0]
    bb_idx = [i for i, m in enumerate(meta) if m["is_bb"]]
    prod = frames[1:] if len(frames) > 1 else frames
    rmsds_prod = [kabsch_rmsd(f[bb_idx], ref[bb_idx]) for f in prod]
    rgs = [rg(f) for f in prod]

    feat = {}
    ref_bb = ref[bb_idx]
    ref_com = ref_bb.mean(0)
    aligned = []
    for f in prod:
        f2 = f.copy()
        f2 = f2 - f[bb_idx].mean(0) + ref_com
        aligned.append(f2)
    aligned = np.stack(aligned, axis=0)
    mean_xyz = aligned.mean(axis=0)
    rmsf_atom = np.sqrt(((aligned - mean_xyz) ** 2).sum(axis=2).mean(axis=0))

    for dom in ("VH", "VL", "CH1", "CL"):
        idx = [i for i, m in enumerate(meta) if m["domain"] == dom and m["is_bb"]]
        if not idx:
            feat[f"{dom}_rmsf_mean"] = 0.0
            feat[f"{dom}_rmsf_q90"] = 0.0
        else:
            vals = rmsf_atom[idx]
            feat[f"{dom}_rmsf_mean"] = float(np.mean(vals))
            feat[f"{dom}_rmsf_q90"] = float(np.quantile(vals, 0.9))

    pairs = {
        "VH_VL": native_pairs(ref, meta, "VH", "VL"),
        "VH_CH1": native_pairs(ref, meta, "VH", "CH1"),
        "VL_CL": native_pairs(ref, meta, "VL", "CL"),
        "CH1_CL": native_pairs(ref, meta, "CH1", "CL"),
    }
    for key, pr in pairs.items():
        occ = [occupancy(f, pr) for f in prod]
        feat[f"{key}_native_contact_occupancy"] = float(np.mean(occ))
        feat[f"{key}_native_contact_occupancy_sd"] = float(np.std(occ))
        feat[f"{key}_native_contact_occupancy_min"] = float(np.min(occ))

    # SASA: start + every other production frame to keep cost bounded (~25 frames)
    sasa_coords = [frames[0]] + prod[::2]
    aro_l, hydro_l, tyr_l, phe_l, trp_l = [], [], [], [], []
    for f in sasa_coords:
        st = coords_to_bio_structure(topology, f)
        a, h, t, p, w = sasa_sums_structure(st)
        aro_l.append(a)
        hydro_l.append(h)
        tyr_l.append(t)
        phe_l.append(p)
        trp_l.append(w)
    aro_l = np.asarray(aro_l)
    hydro_l = np.asarray(hydro_l)
    aro_p, hydro_p = aro_l[1:], hydro_l[1:]
    tyr_p = np.asarray(tyr_l)[1:]
    phe_p = np.asarray(phe_l)[1:]
    trp_p = np.asarray(trp_l)[1:]

    feat.update(
        {
            "backbone_rmsd_mean": float(np.mean(rmsds_prod)),
            "backbone_rmsd_sd": float(np.std(rmsds_prod)),
            "backbone_rmsd_q90": float(np.quantile(rmsds_prod, 0.9)),
            "rg_mean": float(np.mean(rgs)),
            "rg_sd": float(np.std(rgs)),
            "aromatic_sasa_mean": float(np.mean(aro_p)),
            "aromatic_sasa_sd": float(np.std(aro_p)),
            "aromatic_sasa_q90": float(np.quantile(aro_p, 0.9)),
            "Tyr_sasa_mean": float(np.mean(tyr_p)),
            "Phe_sasa_mean": float(np.mean(phe_p)),
            "Trp_sasa_mean": float(np.mean(trp_p)),
            "hydrophobic_sasa_mean": float(np.mean(hydro_p)),
            "hydrophobic_sasa_sd": float(np.std(hydro_p)),
            "hydrophobic_sasa_q90": float(np.quantile(hydro_p, 0.9)),
            "delta_aromatic_sasa_mean": float(np.mean(aro_p) - aro_l[0]),
            "delta_hydrophobic_sasa_mean": float(np.mean(hydro_p) - hydro_l[0]),
        }
    )
    feat["id"] = aid
    finite = bool(np.isfinite([feat[k] for k in FREEZE["feature_names"]]).all())
    qc = {
        **qc_extra,
        "max_backbone_rmsd": float(np.max(rmsds_prod)),
        "feature_finite": finite,
        "n_frames": len(prod),
        "n_sasa_frames": len(sasa_coords),
    }
    return feat, qc


def process_one(aid: str, seq_row, n_prod_steps: int) -> dict:
    t0 = time.time()
    work = CACHE / aid
    work.mkdir(parents=True, exist_ok=True)
    done = work / ".complete"
    feat_path = work / "features.json"
    qc_path = work / "qc.json"
    if done.exists() and feat_path.exists():
        return json.loads(feat_path.read_text())

    pdb = R1 / f"{aid}_C_r1.pdb"
    if not pdb.exists():
        raise FileNotFoundError(pdb)
    seed = seed_for(aid)
    vh_len = int(seq_row.VH_len_used)
    vl_len = int(seq_row.VL_len_used)

    md = run_md(aid, pdb, n_prod_steps, seed, vh_len, vl_len)
    feat, qc = extract_features(
        aid,
        md["frames"],
        md["meta"],
        md["topology"],
        {
            "id": aid,
            "simulation_success": True,
            "finite_energy": md["finite_energy"],
            "finite_coordinates": md["finite_coordinates"],
            "constraint_failure": md["constraint_failure"],
            "temperature_mean": md["temperature_mean"],
            "runtime_seconds": time.time() - t0,
            "n_prod_steps": n_prod_steps,
            "production_ns": n_prod_steps * DT_PS / 1000.0,
            "seed": seed,
            "natoms": md["natoms"],
        },
    )
    if not qc["feature_finite"] or not md["finite_energy"] or not md["finite_coordinates"]:
        qc["simulation_success"] = False
        raise RuntimeError(f"nonfinite {aid}")

    tmp = work / "features.json.tmp"
    tmp.write_text(json.dumps(feat, indent=2))
    tmp.rename(feat_path)
    qc_path.write_text(json.dumps(qc, indent=2))
    done.write_text("ok\n")
    return feat


def assemble(ids):
    rows = []
    qcs = []
    for aid in ids:
        fp = CACHE / aid / "features.json"
        qp = CACHE / aid / "qc.json"
        if fp.exists() and (CACHE / aid / ".complete").exists():
            rows.append(json.loads(fp.read_text()))
            if qp.exists():
                qcs.append(json.loads(qp.read_text()))
    cols = ["id"] + FREEZE["feature_names"]
    df = pd.DataFrame(rows)[cols].sort_values("id").reset_index(drop=True)
    return df, pd.DataFrame(qcs)


def write_dictionary():
    rows = [
        ("backbone_rmsd_mean", "DYN_GLOBAL", "Mean backbone RMSD vs post-eq start over production frames", "Angstrom"),
        ("backbone_rmsd_sd", "DYN_GLOBAL", "SD backbone RMSD", "Angstrom"),
        ("backbone_rmsd_q90", "DYN_GLOBAL", "q90 backbone RMSD", "Angstrom"),
        ("rg_mean", "DYN_GLOBAL", "Mean radius of gyration", "Angstrom"),
        ("rg_sd", "DYN_GLOBAL", "SD radius of gyration", "Angstrom"),
    ]
    for dom in ("VH", "VL", "CH1", "CL"):
        rows.append((f"{dom}_rmsf_mean", "DYN_RMSF", f"Mean backbone RMSF of {dom}", "Angstrom"))
        rows.append((f"{dom}_rmsf_q90", "DYN_RMSF", f"q90 backbone RMSF of {dom}", "Angstrom"))
    for iface in ("VH_VL", "VH_CH1", "VL_CL", "CH1_CL"):
        rows.append((f"{iface}_native_contact_occupancy", "DYN_INTERFACE", f"Mean native-contact occupancy {iface} (4.5A)", "fraction"))
        rows.append((f"{iface}_native_contact_occupancy_sd", "DYN_INTERFACE", f"SD occupancy {iface}", "fraction"))
        rows.append((f"{iface}_native_contact_occupancy_min", "DYN_INTERFACE", f"Min occupancy {iface}", "fraction"))
    rows += [
        ("aromatic_sasa_mean", "DYN_EXPOSURE", "Mean aromatic SASA over frames", "Angstrom^2"),
        ("aromatic_sasa_sd", "DYN_EXPOSURE", "SD aromatic SASA", "Angstrom^2"),
        ("aromatic_sasa_q90", "DYN_EXPOSURE", "q90 aromatic SASA", "Angstrom^2"),
        ("Tyr_sasa_mean", "DYN_EXPOSURE", "Mean Tyr SASA", "Angstrom^2"),
        ("Phe_sasa_mean", "DYN_EXPOSURE", "Mean Phe SASA", "Angstrom^2"),
        ("Trp_sasa_mean", "DYN_EXPOSURE", "Mean Trp SASA", "Angstrom^2"),
        ("hydrophobic_sasa_mean", "DYN_EXPOSURE", "Mean hydrophobic SASA", "Angstrom^2"),
        ("hydrophobic_sasa_sd", "DYN_EXPOSURE", "SD hydrophobic SASA", "Angstrom^2"),
        ("hydrophobic_sasa_q90", "DYN_EXPOSURE", "q90 hydrophobic SASA", "Angstrom^2"),
        ("delta_aromatic_sasa_mean", "DYN_EXPOSURE", "Mean aromatic SASA minus start", "Angstrom^2"),
        ("delta_hydrophobic_sasa_mean", "DYN_EXPOSURE", "Mean hydrophobic SASA minus start", "Angstrom^2"),
    ]
    pd.DataFrame(rows, columns=["feature", "family", "definition", "units"]).to_csv(
        RES / "OPENMM_FAB_MD_FEATURE_DICTIONARY.csv", index=False
    )
    # also top-level required name
    pd.DataFrame(rows, columns=["feature", "family", "definition", "units"]).to_csv(
        OUT / "OPENMM_FAB_MD_FEATURE_DICTIONARY.csv", index=False
    )


def sanity_and_gate(ids, seq, n_prod_steps):
    # 5 target-blind representatives from pilot-like stratification
    pick = []
    for locus in ("kappa", "lambda"):
        sub = seq.loc[ids]
        sub = sub[sub.light_locus == locus]
        if len(sub) == 0:
            continue
        sub = sub.assign(fab_len=sub.heavy_length + sub.light_length).sort_values("fab_len")
        for frac in (0.1, 0.5, 0.9):
            pick.append(sub.iloc[int(frac * (len(sub) - 1))].name)
    # unique keep 5
    seen = []
    for a in pick:
        if a not in seen and a in ids:
            seen.append(a)
    pilot = seen[:5]
    (RES / "SANITY_IDS.json").write_text(json.dumps(pilot, indent=2))
    walls = []
    for aid in pilot:
        # force recompute for timing even if complete? Use fresh workdir tag
        # Clear only if we want pure timing — if already complete, still ok to re-run timing by temp
        # For gate, re-run into cache (idempotent skip would cheat timing). Force by removing complete for pilot only if not part of final? 
        # Better: run in cache/sanity/
        print(f"SANITY {aid}", flush=True)
        # temporarily process into main cache — if already exists skip would under-estimate; remove complete for these 5 only for timing
        for p in (CACHE / aid / ".complete", CACHE / aid / "features.json"):
            if p.exists():
                p.unlink()
        t0 = time.time()
        process_one(aid, seq.loc[aid], n_prod_steps)
        w = time.time() - t0
        walls.append(w)
        print(f"  wall_s={w:.1f}", flush=True)
    mean_w = float(np.mean(walls))
    proj = mean_w * 161 / 3600
    meta = {"pilot_ids": pilot, "walls_s": walls, "mean_wall_s": mean_w, "proj_dev161_h": proj, "n_prod_steps": n_prod_steps}
    (RES / "SANITY_TIMING.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    write_dictionary()
    ids = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]
    seq = pd.read_csv(SEQ).set_index("id")
    assert len(ids) == 161

    n_prod = 250_000  # 0.5 ns
    if "--full-only" not in sys.argv:
        meta = sanity_and_gate(ids, seq, n_prod)
        if meta["proj_dev161_h"] > GATE_H:
            n_prod = 125_000  # 0.25 ns once
            print("REDUCE_TO_0.25NS", flush=True)
            # clear sanity completes to retime? retime 3 of them
            for aid in meta["pilot_ids"][:3]:
                for p in (CACHE / aid / ".complete", CACHE / aid / "features.json"):
                    if p.exists():
                        p.unlink()
            meta2 = sanity_and_gate(ids, seq, n_prod)
            if meta2["proj_dev161_h"] > GATE_H:
                (RES / "VERDICT.json").write_text(json.dumps({"verdict": "OPENMM_ENDGAME_TOO_SLOW", **meta2}))
                print("OPENMM_ENDGAME_TOO_SLOW", flush=True)
                return 2
            meta = meta2
        (RES / "PROTOCOL_LENGTH.json").write_text(
            json.dumps({"production_ns": n_prod * DT_PS / 1000.0, "n_prod_steps": n_prod, "sanity": meta}, indent=2)
        )
    else:
        n_prod = json.loads((RES / "PROTOCOL_LENGTH.json").read_text())["n_prod_steps"]

    # full DEV
    t0 = time.time()
    fails = []
    for i, aid in enumerate(ids, 1):
        try:
            feat = process_one(aid, seq.loc[aid], n_prod)
            print(f"[{i}/161] {aid} OK wall~{feat.get('id') and (CACHE/aid/'qc.json')}", flush=True)
            if (CACHE / aid / "qc.json").exists():
                qc = json.loads((CACHE / aid / "qc.json").read_text())
                print(f"[{i}/161] {aid} runtime_s={qc.get('runtime_seconds'):.1f} elapsed_h={(time.time()-t0)/3600:.2f}", flush=True)
        except Exception as e:
            fails.append({"id": aid, "error": repr(e), "tb": traceback.format_exc()[-1500:]})
            (CACHE / aid).mkdir(parents=True, exist_ok=True)
            (CACHE / aid / "error.json").write_text(json.dumps(fails[-1], indent=2))
            print(f"[{i}/161] {aid} FAIL {e}", flush=True)
            continue

    df, qcdf = assemble(ids)
    df.to_parquet(RES / "OPENMM_FAB_MD_FEATURES_DEV.parquet", index=False)
    df.to_csv(RES / "OPENMM_FAB_MD_FEATURES_DEV.csv", index=False)
    qcdf.to_csv(RES / "OPENMM_FAB_MD_QC_DEV.csv", index=False)
    (OUT / "OPENMM_FAB_MD_FEATURES_DEV.parquet").write_bytes((RES / "OPENMM_FAB_MD_FEATURES_DEV.parquet").read_bytes())
    summary = {
        "n_success": int(len(df)),
        "n_fail": len(fails),
        "fails": fails,
        "total_wall_h": (time.time() - t0) / 3600,
        "production_ns": n_prod * DT_PS / 1000.0,
        "mean_runtime_s": float(qcdf["runtime_seconds"].mean()) if len(qcdf) else None,
    }
    (RES / "DEV_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if len(df) >= 150 else 1


if __name__ == "__main__":
    raise SystemExit(main())
