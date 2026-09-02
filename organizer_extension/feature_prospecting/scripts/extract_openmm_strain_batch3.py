#!/usr/bin/env python3
"""Target-blind OPENMM-STRAIN_v1: local minimization relaxation response."""
from __future__ import annotations

import json
import subprocess
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from openmm import Context, LangevinMiddleIntegrator, LocalEnergyMinimizer, NonbondedForce, Platform, unit
from openmm.app import ForceField, NoCutoff, PDBFile

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FAMILY = FP / "OPENMM-STRAIN"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
PDB2PQR = ROOT / ".venv_stage4/bin/pdb2pqr30"
CACHE = FP / "_batch3_cache" / "openmm_strain"
CACHE.mkdir(parents=True, exist_ok=True)
IFACE_CACHE = FP / "_batch2_cache" / "interface"
GENS = ["esmfold", "abodybuilder2", "boltz2"]
GEN_PATH = {
    "esmfold": "esmfold_canonical_path",
    "abodybuilder2": "abodybuilder2_path",
    "boltz2": "boltz2_pdb_path",
}
CLASH = 2.2
IFACE = 4.5
TOL = 10.0  # kJ/mol/nm
MAXITER = 500
N_WORKERS = 4


def aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def load_sequences():
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    return pd.concat([dev, te], ignore_index=True).drop_duplicates("id").set_index("id")


def map_hl_topology(pdb_path: Path, heavy: str, light: str):
    model = next(PDBParser(QUIET=True).get_structure("x", str(pdb_path)).get_models())
    chain_seqs = {}
    for ch in model.get_chains():
        chain_seqs[ch.id] = "".join(aa1(r.get_resname()) for r in ch.get_residues() if is_aa(r, standard=True))
    m = {}
    for cid, seq in chain_seqs.items():
        if seq == heavy:
            m[cid] = "H"
        elif seq == light:
            m[cid] = "L"
    if set(m.values()) != {"H", "L"}:
        return None, f"chain_map_fail:{m}"
    return m, None


def count_clashes(positions, heavy_mask):
    idx = np.where(heavy_mask)[0]
    n = 0
    for i_a, a in enumerate(idx):
        d = np.linalg.norm(positions[idx[i_a + 1 :]] - positions[a], axis=1)
        n += int((d < CLASH).sum())
    return n


def run_one(args):
    aid, gen, pdb_path, heavy, light = args
    row = {"id": aid, "generator": gen, "extraction_status": "FAIL", "error": ""}
    try:
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            row["error"] = "missing_pdb"
            return row
        work = CACHE / f"{aid}_{gen}"
        work.mkdir(parents=True, exist_ok=True)
        prot = work / "protonated.pdb"
        # reuse INTERFACE protonation if identical protocol
        alt = IFACE_CACHE / f"{aid}_{gen}" / "protonated.pdb"
        if not prot.exists():
            if alt.exists():
                prot.write_bytes(alt.read_bytes())
            else:
                proc = subprocess.run(
                    [
                        str(PDB2PQR),
                        "--ff=AMBER",
                        "--with-ph=7.0",
                        "--drop-water",
                        "--pdb-output",
                        str(prot),
                        str(pdb_path),
                        str(work / "tmp.pqr"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if not prot.exists():
                    row["error"] = f"pdb2pqr:{(proc.stderr or proc.stdout)[-300:]}"
                    return row

        hl_map, err = map_hl_topology(prot, heavy, light)
        if err:
            row["error"] = err
            return row

        pdb = PDBFile(str(prot))
        ff = ForceField("amber14-all.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None, removeCMMotion=False)
        integ = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        ctx = Context(system, integ, Platform.getPlatformByName("CPU"))
        ctx.setPositions(pdb.positions)

        state0 = ctx.getState(getEnergy=True, getForces=True, getPositions=True)
        e0 = state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f0 = state0.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        fn0 = np.linalg.norm(f0, axis=1)
        pos0 = state0.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        LocalEnergyMinimizer.minimize(
            ctx,
            tolerance=TOL * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=MAXITER,
        )
        state1 = ctx.getState(getEnergy=True, getForces=True, getPositions=True)
        e1 = state1.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f1 = state1.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        fn1 = np.linalg.norm(f1, axis=1)
        pos1 = state1.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        # atom metadata
        atom_hl = []
        atom_heavy = []
        atom_ca = []
        atom_reskey = []
        for atom in pdb.topology.atoms():
            cid = atom.residue.chain.id
            atom_hl.append(hl_map.get(cid))
            elem = atom.element.symbol.upper() if atom.element is not None else ""
            atom_heavy.append(elem not in ("H", ""))
            atom_ca.append(atom.name == "CA")
            atom_reskey.append((hl_map.get(cid), atom.residue.id, atom.residue.name))

        atom_hl = np.array(atom_hl, dtype=object)
        atom_heavy = np.asarray(atom_heavy, bool)
        atom_ca = np.asarray(atom_ca, bool)

        # interface residues by heavy proximity before min
        H_h = np.where((atom_hl == "H") & atom_heavy)[0]
        L_h = np.where((atom_hl == "L") & atom_heavy)[0]
        iface_res = set()
        min_d = 1e9
        for i in H_h:
            d = np.linalg.norm(pos0[L_h] - pos0[i], axis=1)
            min_d = min(min_d, float(d.min()) if len(d) else min_d)
            hit = L_h[d <= IFACE]
            if len(hit):
                iface_res.add(atom_reskey[i])
                for j in hit:
                    iface_res.add(atom_reskey[j])
        iface_ca = np.array(
            [atom_ca[i] and atom_reskey[i] in iface_res for i in range(len(atom_ca))], dtype=bool
        )

        def rmsd(mask):
            if not mask.any():
                return float("nan")
            return float(np.sqrt(np.mean(np.sum((pos1[mask] - pos0[mask]) ** 2, axis=1))))

        clash_before = count_clashes(pos0, atom_heavy)
        clash_after = count_clashes(pos1, atom_heavy)
        relief = (clash_before - clash_after) / clash_before if clash_before > 0 else 0.0
        n_res = len({(a.residue.chain.id, a.residue.id) for a in pdb.topology.atoms()})
        dE = float(e1 - e0)

        # save minimized PDB hash content
        # write XYZ for hash of positions
        np.savez_compressed(work / "positions_pre_post.npz", pre=pos0, post=pos1)

        row.update(
            {
                "extraction_status": "SUCCESS",
                "error": "",
                "delta_E_total": dE,
                "delta_E_per_residue": dE / n_res if n_res else dE,
                "CA_RMSD_pre_post": rmsd(atom_ca),
                "all_heavy_RMSD_pre_post": rmsd(atom_heavy),
                "VH_CA_RMSD": rmsd(atom_ca & (atom_hl == "H")),
                "VL_CA_RMSD": rmsd(atom_ca & (atom_hl == "L")),
                "interface_CA_RMSD": rmsd(iface_ca),
                "initial_force_RMS": float(np.sqrt(np.mean(fn0**2))),
                "initial_force_q95": float(np.quantile(fn0, 0.95)),
                "initial_force_max": float(fn0.max()),
                "final_force_RMS": float(np.sqrt(np.mean(fn1**2))),
                "final_force_q95": float(np.quantile(fn1, 0.95)),
                "clash_relief_fraction": float(relief),
                "severe_clash_count_before": int(clash_before),
                "severe_clash_count_after": int(clash_after),
                "E_before": float(e0),
                "E_after": float(e1),
                "n_residues": int(n_res),
                "min_HL_atom_distance_before": float(min_d),
            }
        )
        return row
    except Exception as e:
        row["error"] = f"{type(e).__name__}:{e}"
        row["traceback"] = traceback.format_exc()[-400:]
        return row


def main():
    xw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    seqs = load_sequences()
    jobs = []
    for _, r in xw.iterrows():
        aid = r["id"]
        for gen in GENS:
            jobs.append((aid, gen, r[GEN_PATH[gen]], seqs.loc[aid, "heavy"], seqs.loc[aid, "light"]))
    print("strain jobs", len(jobs), flush=True)
    rows = {g: [] for g in GENS}
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            rows[row["generator"]].append(row)
            done += 1
            if done % 40 == 0:
                print(f"strain done {done}/{len(jobs)}", flush=True)
    all_df = []
    art = {}
    from scipy.stats import spearmanr

    for gen in GENS:
        df = pd.DataFrame(rows[gen]).sort_values("id")
        df.to_parquet(FAMILY / f"features_{gen}.parquet", index=False)
        print(gen, (df.extraction_status == "SUCCESS").sum(), "/", len(df), flush=True)
        ok = df.query("extraction_status == 'SUCCESS'")
        if len(ok) >= 10:
            r1 = float(spearmanr(ok["delta_E_total"], ok["severe_clash_count_before"]).statistic)
            r2 = float(spearmanr(ok["CA_RMSD_pre_post"], ok["severe_clash_count_before"]).statistic)
            dominated = (abs(r1) >= 0.7) or (abs(r2) >= 0.7)
            art[gen] = {
                "spearman_deltaE_vs_clash": r1,
                "spearman_CA_RMSD_vs_clash": r2,
                "GENERATOR_GEOMETRY_ARTIFACT_DOMINATED": dominated,
            }
            df.loc[df.extraction_status == "SUCCESS", "GENERATOR_GEOMETRY_ARTIFACT_DOMINATED_flag"] = int(dominated)
            df.to_parquet(FAMILY / f"features_{gen}.parquet", index=False)
        all_df.append(df)
    qc = pd.concat(all_df, ignore_index=True)
    qc.to_csv(FAMILY / "extraction_qc.csv", index=False)
    qc.to_csv(FAMILY / "FEATURE_MANIFEST.csv", index=False)
    (FAMILY / "geometry_artifact_audit.json").write_text(json.dumps(art, indent=2) + "\n")
    print("OPENMM-STRAIN done", art)


if __name__ == "__main__":
    main()
