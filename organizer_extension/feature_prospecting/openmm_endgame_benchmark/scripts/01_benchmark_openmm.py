#!/usr/bin/env python3
"""OpenMM endgame Fab MD benchmark (implicit GB; optional explicit). Does not start full DEV."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "openmm_endgame_benchmark"
PDB = FP / "fennix_fab_context/cache/r1/ADI-45391_C_r1.pdb"
ENV = ROOT / ".mamba/envs/openmm_cuda"

os.environ["LD_LIBRARY_PATH"] = f"{ENV}/lib:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENMM_PLUGIN_DIR"] = str(ENV / "lib/plugins")


def gpu_stats():
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            text=True,
        ).strip()
        util, mem_u, mem_t = [x.strip() for x in out.split(",")]
        return {"gpu_util_pct": float(util), "gpu_mem_used_mib": float(mem_u), "gpu_mem_total_mib": float(mem_t)}
    except Exception as e:
        return {"error": repr(e)}


def ensure_platforms():
    from openmm import Platform

    Platform.loadPluginsFromDirectory(str(ENV / "lib/plugins"))
    names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    return names


def run_implicit(n_prod_steps: int = 25000):
    import openmm
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, PDBFile, Simulation

    OUT.mkdir(parents=True, exist_ok=True)
    platforms = ensure_platforms()
    assert "CUDA" in platforms, platforms

    pdb = PDBFile(str(PDB))
    ff = ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = Modeller(pdb.topology, pdb.positions)
    # C_r1 already has H; still allow FF to add missing if any
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
    platform = Platform.getPlatformByName("CUDA")
    props = {"DeviceIndex": "0", "Precision": "mixed"}
    sim = Simulation(modeller.topology, system, integrator, platform, props)
    sim.context.setPositions(modeller.positions)

    natoms = modeller.topology.getNumAtoms()
    meta = {
        "openmm_version": openmm.__version__,
        "platforms": platforms,
        "platform": "CUDA",
        "device": "0",
        "forcefield": "amber14-all.xml + implicit/obc2.xml",
        "solvent": "OBC2 implicit GB",
        "constraints": "HBonds",
        "dt_fs": 2,
        "temperature_K": 300,
        "friction_per_ps": 1.0,
        "natoms": natoms,
        "pdb": str(PDB),
    }

    # minimize
    t0 = time.time()
    sim.minimizeEnergy(maxIterations=200)
    meta["minimize_s"] = time.time() - t0
    state = sim.context.getState(getEnergy=True)
    e0 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    meta["Epot_after_min_kJmol"] = float(e0)

    # short equilibration 5 ps
    n_eq = 2500
    t0 = time.time()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.step(n_eq)
    meta["equilibration_ps"] = n_eq * 0.002
    meta["equilibration_s"] = time.time() - t0

    # production timing
    t0 = time.time()
    sim.step(n_prod_steps)
    wall = time.time() - t0
    sim_ns = n_prod_steps * 0.002 / 1000.0  # ns
    ns_per_day = sim_ns / (wall / 86400.0)
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    finite = bool(np.isfinite(pos).all() and np.isfinite(e1))

    meta.update(
        {
            "n_prod_steps": n_prod_steps,
            "prod_sim_ns": sim_ns,
            "prod_wall_s": wall,
            "ns_per_day": ns_per_day,
            "Epot_end_kJmol": float(e1),
            "finite": finite,
            "nan_or_inf": not finite,
            "gpu_during": gpu_stats(),
        }
    )
    # projections for DEV=161
    for L in (0.25, 0.5, 1.0, 2.0):
        hours = (L / ns_per_day) * 24 * 161
        meta[f"proj_dev161_{L}ns_hours"] = hours

    (OUT / "IMPLICIT_BENCH.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def run_explicit(n_prod_steps: int = 10000, time_budget_s: float = 1800):
    """Optional; SKIP if setup exceeds budget."""
    t_start = time.time()
    try:
        import openmm
        from openmm import LangevinMiddleIntegrator, Platform, unit
        from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile, Simulation

        ensure_platforms()
        pdb = PDBFile(str(PDB))
        ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = Modeller(pdb.topology, pdb.positions)
        try:
            modeller.addHydrogens(ff)
        except Exception:
            pass
        if time.time() - t_start > time_budget_s:
            return {"status": "SKIP_EXPLICIT_BENCHMARK", "reason": "time before solvate"}
        modeller.addSolvent(ff, model="tip3pfb", padding=1.0 * unit.nanometer, ionicStrength=0.15 * unit.molar)
        if time.time() - t_start > time_budget_s:
            return {"status": "SKIP_EXPLICIT_BENCHMARK", "reason": "time after solvate"}
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds,
            rigidWater=True,
        )
        integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        platform = Platform.getPlatformByName("CUDA")
        props = {"DeviceIndex": "0", "Precision": "mixed"}
        sim = Simulation(modeller.topology, system, integrator, platform, props)
        sim.context.setPositions(modeller.positions)
        natoms = modeller.topology.getNumAtoms()
        sim.minimizeEnergy(maxIterations=200)
        sim.context.setVelocitiesToTemperature(300)
        sim.step(2500)  # 5 ps eq
        t0 = time.time()
        sim.step(n_prod_steps)
        wall = time.time() - t0
        sim_ns = n_prod_steps * 0.002 / 1000.0
        ns_per_day = sim_ns / (wall / 86400.0)
        state = sim.context.getState(getEnergy=True, getPositions=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        finite = bool(np.isfinite(pos).all() and np.isfinite(e))
        meta = {
            "status": "OK",
            "solvent": "explicit tip3pfb + PME + 0.15M ions, padding 1.0 nm",
            "forcefield": "amber14-all.xml + amber14/tip3pfb.xml",
            "natoms": natoms,
            "n_prod_steps": n_prod_steps,
            "prod_sim_ns": sim_ns,
            "prod_wall_s": wall,
            "ns_per_day": ns_per_day,
            "finite": finite,
            "gpu_during": gpu_stats(),
            "setup_wall_s": time.time() - t_start,
        }
        for L in (0.25, 0.5, 1.0, 2.0):
            meta[f"proj_dev161_{L}ns_hours"] = (L / ns_per_day) * 24 * 161
        (OUT / "EXPLICIT_BENCH.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return meta
    except Exception as e:
        meta = {"status": "SKIP_EXPLICIT_BENCHMARK", "reason": repr(e), "elapsed_s": time.time() - t_start}
        (OUT / "EXPLICIT_BENCH.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return meta


def decide(imp: dict) -> dict:
    nspd = float(imp["ns_per_day"])
    if nspd >= 1000:
        rec_len, choice = 1.0, "OPENMM_GO"
    elif nspd >= 500:
        rec_len, choice = 0.5, "OPENMM_GO"
    elif nspd >= 250:
        rec_len, choice = 0.25, "OPENMM_GO"
    else:
        rec_len, choice = None, "OPENMM_NOT_FAST_ENOUGH_FOR_ENDGAME_FULL_DEV"
    # vs FeNNix thermal microprobe ~110s for 0.00025 ns (250*1fs) → tiny physical time
    return {
        "ns_per_day": nspd,
        "recommended_traj_ns": rec_len,
        "recommendation": choice if choice.startswith("OPENMM") else choice,
        "endgame_choice": "OPENMM_GO" if choice == "OPENMM_GO" else "RESUME_FENNIX",
        "note": "Primary candidate is implicit GB; FeNNix thermal microprobe preserved (37/161) not resumed.",
    }


def write_md(imp: dict, exp: dict, dec: dict):
    lines = [
        "# OPENMM_ENDGAME_BENCHMARK",
        "",
        f"- OpenMM `{imp['openmm_version']}` / platform **{imp['platform']}** (RTX 3090)",
        f"- Force field: `{imp['forcefield']}`",
        f"- Structure: `{imp['pdb']}` (ADI-45391 C_r1)",
        f"- Atoms (implicit): **{imp['natoms']}**",
        f"- Constraints: HBonds; dt=2 fs; Langevin; T=300 K",
        f"- Stability: finite={imp['finite']} nan_or_inf={imp['nan_or_inf']}",
        "",
        "## Implicit solvent (OBC2) performance",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| production steps | {imp['n_prod_steps']} |",
        f"| simulated ns | {imp['prod_sim_ns']:.4f} |",
        f"| wall s | {imp['prod_wall_s']:.2f} |",
        f"| **ns/day** | **{imp['ns_per_day']:.1f}** |",
        f"| GPU util % | {imp.get('gpu_during',{}).get('gpu_util_pct')} |",
        f"| GPU mem MiB | {imp.get('gpu_during',{}).get('gpu_mem_used_mib')} |",
        "",
        "## Projected DEV N=161 wall time (1 GPU, serial)",
        "",
        "| traj / Ab | projected hours |",
        "|---:|---:|",
    ]
    for L in (0.25, 0.5, 1.0, 2.0):
        lines.append(f"| {L} ns | {imp[f'proj_dev161_{L}ns_hours']:.2f} |")
    lines += ["", "## Explicit solvent", ""]
    if exp.get("status") == "OK":
        lines += [
            f"- atoms={exp['natoms']}; ns/day=**{exp['ns_per_day']:.1f}**; finite={exp['finite']}",
            "",
            "| traj / Ab | projected hours |",
            "|---:|---:|",
        ]
        for L in (0.25, 0.5, 1.0, 2.0):
            lines.append(f"| {L} ns | {exp[f'proj_dev161_{L}ns_hours']:.2f} |")
    else:
        lines.append(f"- **{exp.get('status', 'SKIP')}**: {exp.get('reason', '')}")
    lines += [
        "",
        "## Decision gate",
        "",
        f"- measured ns/day (implicit) = **{dec['ns_per_day']:.1f}**",
        f"- recommended traj length = **{dec['recommended_traj_ns']}** ns / DEV Ab",
        f"- endgame choice = **{dec['endgame_choice']}**",
        "",
        "FeNNix thermal microprobe: 37/161 preserved; resume path prepared separately; **not launched** during this bench.",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "OPENMM_ENDGAME_BENCHMARK.md").write_text(text)
    # also top-level alias under thermal microprobe sibling
    (FP / "OPENMM_ENDGAME_BENCHMARK.md").write_text(text)
    (OUT / "DECISION.json").write_text(json.dumps(dec, indent=2))


def main():
    print("GPU before:", gpu_stats(), flush=True)
    imp = run_implicit(25000)
    exp = run_explicit(10000)
    dec = decide(imp)
    write_md(imp, exp, dec)
    print("DECISION", json.dumps(dec, indent=2), flush=True)


if __name__ == "__main__":
    main()
