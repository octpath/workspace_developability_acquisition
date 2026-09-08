#!/usr/bin/env python3
"""Reproduce FeNNix Fab microdynamics timing bench (does not run full cohort)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
SITE = FP / "foundation_stability/envs/fennol/lib/python3.11/site-packages"
MODEL = FP / "foundation_stability_v2/cache/fennix-bio1S.fnx"
PDB = FP / "fennix_fab_context/cache/r1/ADI-45391_C_r1.pdb"
OUT = FP / "fennix_microdynamics/results"


def _cuda_lib_path() -> str:
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
    return ":".join(str(p) for p in parts if p.exists())


def pdb_to_xyz(pdb: Path, xyz: Path) -> int:
    from Bio.PDB import PDBParser
    from ase.data import atomic_numbers

    s = PDBParser(QUIET=True).get_structure("x", str(pdb))
    atoms = list(s.get_atoms())
    lines = [str(len(atoms)), f"from {pdb.name}"]
    for a in atoms:
        el = (a.element or a.get_name()[0]).strip()
        el = el.capitalize() if len(el) == 1 else el.title()
        if el not in atomic_numbers:
            el = a.get_name()[0].capitalize()
        x, y, z = map(float, a.coord)
        lines.append(f"{el} {x:.8f} {y:.8f} {z:.8f}")
    xyz.write_text("\n".join(lines) + "\n")
    return len(atoms)


def main(nsteps: int = 500) -> None:
    os.environ["LD_LIBRARY_PATH"] = _cuda_lib_path() + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ["FENNOL_DEVICE"] = "cuda:0"

    work = OUT / "bench_workdir"
    work.mkdir(parents=True, exist_ok=True)
    xyz = work / "fab.xyz"
    nat = pdb_to_xyz(PDB, xyz)
    fnl = work / "bench.fnl"
    fnl.write_text(
        "\n".join(
            [
                "device cuda:0",
                f"model_file {MODEL}",
                f"xyz_input/file {xyz}",
                "xyz_input/has_comment_line yes",
                "system_name microdyn_bench",
                "dt 0.001",
                f"nsteps {nsteps}",
                "temperature 300.0",
                "thermostat LGV",
                "gamma 1.0",
                "tdump 1.0",
                "traj_format xyz",
                "random_seed 42",
                "energy_unit eV",
                "per_atom_energy no",
                "",
            ]
        )
    )

    from fennol.md.dynamic import config_and_run_dynamic

    t0 = time.time()
    config_and_run_dynamic(fnl)
    wall = time.time() - t0
    sps = wall / nsteps
    payload = {
        "natoms": nat,
        "nsteps": nsteps,
        "wall_s": wall,
        "sec_per_step": sps,
        "hours_323_5ps": sps * 6000 * 323 / 3600,
        "hours_323_2ps": sps * 3000 * 323 / 3600,
        "verdict": "MICRODYNAMICS_TOO_SLOW"
        if sps * 3000 * 323 / 3600 > 12
        else "TIMING_OK",
    }
    (OUT / "PILOT_TIMING_RERUN.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
