#!/usr/bin/env python3
"""Dev-only FeNNix 250-step thermal microprobe (NOT equilibrium MD)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from Bio.PDB import PDBIO, PDBParser, SASA, StructureBuilder
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "fennix_thermal_microprobe"
CACHE = OUT / "cache" / "dev"
RES = OUT / "results"
MODEL = FP / "foundation_stability_v2/cache/fennix-bio1S.fnx"
R1 = FP / "fennix_fab_context/cache/r1"
SEQ = FP / "fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
SITE = FP / "foundation_stability/envs/fennol/lib/python3.11/site-packages"
NSTEPS = 250
DT_PS = 0.001
CONTACT_A = 4.5
ARO = {"PHE", "TYR", "TRP"}
HYDRO = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
BB = {"N", "CA", "C", "O"}
SEC_PER_STEP_REF = 0.344
GATE_H = 5.0


def setup_cuda_env() -> None:
    parts = [
        SITE / "nvidia/cusparse/lib",
        SITE / "nvidia/cublas/lib",
        SITE / "nvidia/cuda_runtime/lib",
        SITE / "nvidia/cudnn/lib",
        SITE / "nvidia/cufft/lib",
        SITE / "nvidia/cusolver/lib",
        SITE / "nvidia/nvjitlink/lib",
        Path("/usr/local/cuda/lib64"),
    ]
    lp = ":".join(str(p) for p in parts if p.exists())
    os.environ["LD_LIBRARY_PATH"] = lp + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ["FENNOL_DEVICE"] = "cuda:0"


def seed_for(aid: str) -> int:
    return int.from_bytes(hashlib.sha256(aid.encode()).digest()[:4], "little")


def load_pdb_atoms(pdb: Path):
    s = PDBParser(QUIET=True).get_structure("x", str(pdb))
    species, coords, meta = [], [], []
    for a in s.get_atoms():
        el = (a.element or a.get_name()[0]).strip()
        el = el.capitalize() if len(el) == 1 else el.title()
        if el not in atomic_numbers:
            el = a.get_name()[0].capitalize()
        res = a.get_parent()
        chain = res.get_parent().id
        name = a.get_name().strip()
        species.append(atomic_numbers[el])
        coords.append(np.array(a.coord, float))
        meta.append(
            {
                "chain": chain,
                "resseq": int(res.id[1]),
                "resname": res.get_resname(),
                "name": name,
                "element": el,
                "is_bb": name in BB,
                "is_heavy": el != "H",
                "is_ca": name == "CA",
            }
        )
    return np.asarray(species, np.int32), np.asarray(coords, float), meta, s


def domain_masks(meta, vh_len, vl_len):
    masks = {k: np.zeros(len(meta), bool) for k in ("VH", "VL", "CH1", "CL")}
    for i, m in enumerate(meta):
        if m["chain"] == "A":
            masks["VH" if m["resseq"] <= vh_len else "CH1"][i] = True
        elif m["chain"] == "B":
            masks["VL" if m["resseq"] <= vl_len else "CL"][i] = True
    return masks


def write_xyz(path: Path, species, coords, comment: str):
    from ase.data import chemical_symbols

    lines = [str(len(species)), comment]
    for z, c in zip(species, coords):
        el = chemical_symbols[int(z)]
        lines.append(f"{el} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}")
    path.write_text("\n".join(lines) + "\n")


def read_xyz_frames(path: Path):
    text = path.read_text().strip().splitlines()
    frames = []
    i = 0
    while i < len(text):
        n = int(text[i])
        i += 2
        coords = []
        for _ in range(n):
            parts = text[i].split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        frames.append(np.asarray(coords, float))
    return frames


def rg(coords):
    c = coords - coords.mean(axis=0, keepdims=True)
    return float(np.sqrt((c * c).sum(axis=1).mean()))


def bb_rmsd(coords, ref, meta):
    idx = [i for i, m in enumerate(meta) if m["is_bb"]]
    a = coords[idx]
    b = ref[idx]
    a = a - a.mean(0)
    b = b - b.mean(0)
    # Kabsch
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    a2 = a @ R
    return float(np.sqrt(((a2 - b) ** 2).sum(axis=1).mean()))


def com_ca(coords, mask, meta):
    idx = [i for i, m in enumerate(meta) if mask[i] and m["is_ca"]]
    if not idx:
        return np.zeros(3)
    return coords[idx].mean(axis=0)


def native_pairs(coords, mask_a, mask_b, meta):
    ia = [i for i, m in enumerate(meta) if mask_a[i] and m["is_heavy"]]
    ib = [i for i, m in enumerate(meta) if mask_b[i] and m["is_heavy"]]
    if not ia or not ib:
        return []
    A = coords[ia]
    B = coords[ib]
    # chunked distances
    pairs = []
    for i, p in zip(ia, A):
        d = np.linalg.norm(B - p, axis=1)
        hits = np.where(d <= CONTACT_A)[0]
        for j in hits:
            pairs.append((i, ib[j]))
    return pairs


def retention(coords, pairs):
    if not pairs:
        return 1.0
    ok = 0
    for i, j in pairs:
        if np.linalg.norm(coords[i] - coords[j]) <= CONTACT_A:
            ok += 1
    return ok / len(pairs)


def force_rms(F):
    nrm = np.linalg.norm(F, axis=1)
    return float(np.sqrt(np.mean(nrm**2)))


def sasa_sums(structure):
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


def coords_to_structure(template_structure, coords, meta):
    """Clone residue topology; replace coordinates."""
    # mutate atom coords in a deep-ish copy via PDB roundtrip is slow; update in place on a parsed copy
    s = template_structure.copy()
    atoms = list(s.get_atoms())
    assert len(atoms) == len(coords)
    for a, c in zip(atoms, coords):
        a.set_coord(c)
    return s


def run_fennol_md(work: Path, species, coords0, seed: int) -> list[np.ndarray]:
    from fennol.md.dynamic import config_and_run_dynamic

    xyz = work / "start.xyz"
    write_xyz(xyz, species, coords0, "thermal_microprobe_start")
    for p in work.glob("*.restart*"):
        p.unlink()
    for p in work.glob("microprobe*"):
        if p.suffix in {".xyz", ".arc", ".fnl"} or "restart" in p.name:
            if p.name != "start.xyz":
                p.unlink(missing_ok=True)
    fnl = work / "run.fnl"
    # Single endpoint dump; nprint=nsteps to cut Python I/O overhead.
    fnl.write_text(
        "\n".join(
            [
                "device cuda:0",
                f"model_file {MODEL}",
                f"xyz_input/file {xyz}",
                "xyz_input/has_comment_line yes",
                "system_name microprobe",
                f"dt {DT_PS}",
                f"nsteps {NSTEPS}",
                "temperature 300.0",
                "thermostat LGV",
                "gamma 1.0",
                f"tdump {DT_PS * NSTEPS}",
                "traj_format xyz",
                f"nprint {NSTEPS}",
                f"nsummary {NSTEPS + 1}",
                f"random_seed {seed}",
                "energy_unit eV",
                "per_atom_energy no",
                "",
            ]
        )
    )
    cwd = Path.cwd()
    os.chdir(work)
    try:
        # Silence fennol step spam (keeps errors on exception).
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            config_and_run_dynamic(fnl)
        (work / "fennol_md.log").write_text(buf.getvalue()[-5000:])
    finally:
        os.chdir(cwd)
    traj = work / "microprobe.xyz"
    if not traj.exists():
        cands = [c for c in work.glob("*.xyz") if c.name != "start.xyz"]
        if not cands:
            raise FileNotFoundError("trajectory xyz missing")
        traj = cands[0]
    frames = read_xyz_frames(traj)
    return frames

def process_one(aid: str, seq_row, model) -> dict:
    t_wall0 = time.time()
    pdb = R1 / f"{aid}_C_r1.pdb"
    work = CACHE / aid
    work.mkdir(parents=True, exist_ok=True)
    done = work / ".complete"
    feat_path = work / "features.json"
    if done.exists() and feat_path.exists():
        return json.loads(feat_path.read_text())

    species, coords0, meta, struct0 = load_pdb_atoms(pdb)
    vh_len = int(seq_row.VH_len_used)
    vl_len = int(seq_row.VL_len_used)
    masks = domain_masks(meta, vh_len, vl_len)

    # start energy/forces
    E0, F0, _ = model.energy_and_forces(species=species, coordinates=coords0, unit="ev")
    E0 = float(np.asarray(E0).reshape(-1)[0])
    F0 = np.asarray(F0, float)
    frms0 = force_rms(F0)

    seed = seed_for(aid)
    frames = run_fennol_md(work, species, coords0, seed)
    if not frames:
        raise RuntimeError("empty trajectory")
    coords1 = frames[-1]

    E1, F1, _ = model.energy_and_forces(species=species, coordinates=coords1, unit="ev")
    E1 = float(np.asarray(E1).reshape(-1)[0])
    F1 = np.asarray(F1, float)
    frms1 = force_rms(F1)

    # Endpoint-only dump: max_backbone_rmsd == endpoint (still recorded for schema).
    endpoint_rmsd = bb_rmsd(coords1, coords0, meta)
    max_rmsd = float(endpoint_rmsd)
    disp = {}
    for dom in ("VH", "VL", "CH1", "CL"):
        c0 = com_ca(coords0, masks[dom], meta)
        c1 = com_ca(coords1, masks[dom], meta)
        disp[dom] = float(np.linalg.norm(c1 - c0))

    pairs = {
        "VH_VL": native_pairs(coords0, masks["VH"], masks["VL"], meta),
        "VH_CH1": native_pairs(coords0, masks["VH"], masks["CH1"], meta),
        "VL_CL": native_pairs(coords0, masks["VL"], masks["CL"], meta),
        "CH1_CL": native_pairs(coords0, masks["CH1"], masks["CL"], meta),
    }
    ret = {k: retention(coords1, v) for k, v in pairs.items()}

    rg0, rg1 = rg(coords0), rg(coords1)

    aro0, hydro0, tyr0, phe0, trp0 = sasa_sums(struct0)
    struct1 = coords_to_structure(struct0, coords1, meta)
    aro1, hydro1, tyr1, phe1, trp1 = sasa_sums(struct1)

    feat = {
        "id": aid,
        "endpoint_backbone_rmsd": endpoint_rmsd,
        "max_backbone_rmsd": max_rmsd,
        "VH_endpoint_displacement": disp["VH"],
        "VL_endpoint_displacement": disp["VL"],
        "CH1_endpoint_displacement": disp["CH1"],
        "CL_endpoint_displacement": disp["CL"],
        "VH_VL_contact_retention_end": ret["VH_VL"],
        "VH_CH1_contact_retention_end": ret["VH_CH1"],
        "VL_CL_contact_retention_end": ret["VL_CL"],
        "CH1_CL_contact_retention_end": ret["CH1_CL"],
        "potential_energy_start": E0,
        "potential_energy_end": E1,
        "delta_potential_energy": E1 - E0,
        "force_rms_start": frms0,
        "force_rms_end": frms1,
        "delta_force_rms": frms1 - frms0,
        "rg_start": rg0,
        "rg_end": rg1,
        "delta_rg": rg1 - rg0,
        "aromatic_sasa_start": aro0,
        "aromatic_sasa_end": aro1,
        "delta_aromatic_sasa": aro1 - aro0,
        "hydrophobic_sasa_start": hydro0,
        "hydrophobic_sasa_end": hydro1,
        "delta_hydrophobic_sasa": hydro1 - hydro0,
        "delta_tyr_sasa": tyr1 - tyr0,
        "delta_phe_sasa": phe1 - phe0,
        "delta_trp_sasa": trp1 - trp0,
        "wall_s": time.time() - t_wall0,
        "nsteps": NSTEPS,
        "seed": seed,
        "natoms": int(len(species)),
        "status": "SUCCESS",
    }
    if not np.isfinite([feat[k] for k in feat if isinstance(feat[k], float)]).all():
        feat["status"] = "NONFINITE"
        raise RuntimeError(f"nonfinite features for {aid}")

    tmp = work / "features.json.tmp"
    tmp.write_text(json.dumps(feat, indent=2))
    tmp.rename(feat_path)
    done.write_text("ok\n")
    return feat


def assemble(rows: list[dict]) -> pd.DataFrame:
    freeze = json.loads((OUT / "THERMAL_MICROPROBE_FEATURE_FREEZE.json").read_text())
    cols = ["id"] + freeze["feature_names"]
    df = pd.DataFrame(rows)
    df = df[cols].sort_values("id").reset_index(drop=True)
    return df


def main(assemble_only: bool = False):
    setup_cuda_env()
    CACHE.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    ids = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]
    if assemble_only:
        ok = []
        for aid in ids:
            fp = CACHE / aid / "features.json"
            if fp.exists() and (CACHE / aid / ".complete").exists():
                ok.append(json.loads(fp.read_text()))
        if len(ok) != len(ids):
            print(f"assemble_only incomplete {len(ok)}/{len(ids)}", flush=True)
            return 1
        df = assemble(ok)
        df.to_parquet(RES / "FENNIX_THERMAL_MICROPROBE_DEV.parquet", index=False)
        df.to_csv(RES / "FENNIX_THERMAL_MICROPROBE_DEV.csv", index=False)
        print(f"assembled {df.shape}", flush=True)
        return 0

    # Soft pre-gate from historical MD-only rate; hard stop only if first measured Ab implies >> budget.
    proj_h = SEC_PER_STEP_REF * NSTEPS * len(ids) / 3600
    print(f"usable_dev={len(ids)} projected_h_ref={proj_h:.2f} soft_gate={GATE_H}", flush=True)
    if proj_h > GATE_H * 1.5:
        (RES / "VERDICT.json").write_text(
            json.dumps({"verdict": "THERMAL_MICROPROBE_TOO_SLOW", "projected_h": proj_h})
        )
        print("THERMAL_MICROPROBE_TOO_SLOW", flush=True)
        return 2

    import fennol

    t_load = time.time()
    model = fennol.load(str(MODEL))
    print(f"model_loaded_s={time.time()-t_load:.1f}", flush=True)

    seq = pd.read_csv(SEQ).set_index("id")
    rows = []
    t0 = time.time()
    for i, aid in enumerate(ids, 1):
        try:
            feat = process_one(aid, seq.loc[aid], model)
            rows.append(feat)
            print(
                f"[{i}/{len(ids)}] {aid} status={feat.get('status')} wall_s={feat.get('wall_s',0):.1f} "
                f"elapsed_h={(time.time()-t0)/3600:.2f}",
                flush=True,
            )
        except Exception as e:
            err = {"id": aid, "status": "FAIL", "error": repr(e)}
            (CACHE / aid).mkdir(parents=True, exist_ok=True)
            (CACHE / aid / "error.json").write_text(json.dumps(err, indent=2))
            print(f"[{i}/{len(ids)}] {aid} FAIL {e}", flush=True)
            raise

    ok = []
    for aid in ids:
        fp = CACHE / aid / "features.json"
        if fp.exists():
            ok.append(json.loads(fp.read_text()))
    df = assemble(ok)
    df.to_parquet(RES / "FENNIX_THERMAL_MICROPROBE_DEV.parquet", index=False)
    df.to_csv(RES / "FENNIX_THERMAL_MICROPROBE_DEV.csv", index=False)
    walls = [r["wall_s"] for r in ok if "wall_s" in r]
    meta = {
        "n_success": len(ok),
        "n_expected": len(ids),
        "mean_wall_s": float(np.mean(walls)) if walls else None,
        "median_wall_s": float(np.median(walls)) if walls else None,
        "total_wall_h": (time.time() - t0) / 3600,
        "feature_dim": int(df.shape[1] - 1),
    }
    (RES / "DEV_RUN_META.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    assemble_only = "--assemble-only" in sys.argv
    sys.exit(main(assemble_only=assemble_only))
