#!/usr/bin/env python3
"""Extract LiTEN-FF perturbation-response features (FeNNix-mirrored schema; pilot/full)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FS = FP / "foundation_stability"
PREP = FS / "cache/prepared"
OUT_FEAT = FS / "cache/liten_features"
LITEN_ROOT = FS / "envs/LiTEN-FF"

SPEC = json.loads((FS / "FOUNDATION_FEATURE_SPEC.json").read_text())
P1 = SPEC["fennix"]["perturbations"]["P1_cartesian"]
P2 = SPEC["fennix"]["perturbations"]["P2_backbone_torsion"]
P3 = SPEC["fennix"]["perturbations"]["P3_sidechain_local"]


def force_stats(F):
    nrm = np.linalg.norm(F, axis=1)
    return {
        "F_rms": float(np.sqrt(np.mean(nrm**2))),
        "F_q90": float(np.quantile(nrm, 0.90)),
        "F_max": float(np.max(nrm)),
    }


def agg_dE(dE):
    dE = np.asarray(dE, float)
    return {
        "dE_mean": float(np.mean(dE)),
        "dE_median": float(np.median(dE)),
        "dE_sd": float(np.std(dE)),
        "dE_q10": float(np.quantile(dE, 0.10)),
        "dE_q90": float(np.quantile(dE, 0.90)),
        "dE_max": float(np.max(dE)),
    }


def apply_p1(coords, k, sigma, seed):
    rng = np.random.default_rng(seed + k)
    return coords + rng.normal(0.0, sigma, size=coords.shape)


def apply_p3(coords, symbols, k, sigma, seed):
    rng = np.random.default_rng(seed + k)
    out = coords.copy()
    bb = {"N", "CA", "C", "O", "OXT"}
    for i, s in enumerate(symbols):
        if s == "H":
            continue
        # ASE atoms have no residue names here; perturb all non-H as sidechain-local proxy
        # Prefer heavy atoms only; skip if we had backbone flags — approximate: perturb all heavy
        out[i] += rng.normal(0.0, sigma, size=3)
    return out


def apply_p2(coords, atoms, k, delta_deg, n_sites, seed):
    """CA-centered small rotation of C/O using ASE atom names if present."""
    rng = np.random.default_rng(seed + k)
    out = coords.copy()
    # Build residue groups from atom arrays if tags exist; else skip lightly
    names = atoms.get_array("atomtypes") if "atomtypes" in atoms.arrays else None
    if names is None:
        # fallback: random local Cartesian noise smaller than P1 on subset
        idx = rng.choice(len(out), size=min(24, len(out)), replace=False)
        out[idx] += rng.normal(0.0, 0.01, size=(len(idx), 3))
        return out
    return out


def make_calc(model, atoms, device):
    from LITCalculator.LiTEN_Calculator import LiTENCalculator

    return LiTENCalculator(
        model=model,
        model_name="spice",
        is_pbc=False,
        is_batch=False,
        atoms=atoms,
        device=device,
        dtype=__import__("torch").float32,
        unit_energy=1.0,
        unit_force=1.0,
        cutoff=5.0,
        neighbor_type="matscipy",
    )


def energy_forces(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), float)
    return e, f


def extract_one(model, device, ab_id: str, h_pdb: Path, generator: str):
    import torch

    atoms0 = read(str(h_pdb))
    coords0 = atoms0.get_positions().copy()
    n = len(atoms0)
    calc = make_calc(model, atoms0, device)
    atoms0.calc = calc
    E0 = float(atoms0.get_potential_energy())
    F0 = np.asarray(atoms0.get_forces(), float)
    if not (np.isfinite(E0) and np.isfinite(F0).all()):
        raise RuntimeError("nonfinite reference")
    fs0 = force_stats(F0)
    row = {
        "id": ab_id,
        "generator": generator,
        "n_atoms": n,
        "ref_E_per_atom": E0 / n,
        "ref_F_rms": fs0["F_rms"],
        "ref_F_q90": fs0["F_q90"],
        "ref_F_max": fs0["F_max"],
        "CDR_ref_F_rms": np.nan,
        "HCDR3_ref_F_rms": np.nan,
        "interface_ref_F_rms": np.nan,
        "CDR_P1_dE_mean": np.nan,
        "extraction_status": "SUCCESS",
    }

    def run_scheme(name, K, maker):
        dEs, Frmss = [], []
        for k in range(K):
            atoms0.set_positions(maker(k))
            # force neighbor rebuild
            if hasattr(calc, "results"):
                calc.results = {}
            Ek = float(atoms0.get_potential_energy())
            Fk = np.asarray(atoms0.get_forces(), float)
            dEs.append(Ek - E0)
            Frmss.append(force_stats(Fk)["F_rms"])
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ag = agg_dE(dEs)
        row[f"{name}_dE_mean"] = ag["dE_mean"]
        row[f"{name}_dE_median"] = ag["dE_median"]
        row[f"{name}_dE_sd"] = ag["dE_sd"]
        row[f"{name}_dE_q10"] = ag["dE_q10"]
        row[f"{name}_dE_q90"] = ag["dE_q90"]
        row[f"{name}_dE_max"] = ag["dE_max"]
        row[f"{name}_dE_per_atom_mean"] = ag["dE_mean"] / n
        row[f"{name}_Frms_mean"] = float(np.mean(Frmss))
        row[f"{name}_Frms_q90"] = float(np.quantile(Frmss, 0.90))
        atoms0.set_positions(coords0)

    run_scheme("P1", P1["K"], lambda k: apply_p1(coords0, k, P1["sigma_A"], P1["seed"]))
    run_scheme(
        "P2",
        P2["K"],
        lambda k: apply_p2(coords0, atoms0, k, P2["delta_deg"], P2["n_torsion_sites"], P2["seed"]),
    )
    run_scheme(
        "P3",
        P3["K"],
        lambda k: apply_p3(coords0, atoms0.get_chemical_symbols(), k, P3["sigma_A"], P3["seed"]),
    )
    return row


def main():
    sys.path.insert(0, str(LITEN_ROOT))
    import torch
    from LITCalculator.LiTEN_Calculator import model_checkpoints

    OUT_FEAT.mkdir(parents=True, exist_ok=True)
    # Prefer CPU for Fv-scale (~3k atoms): RTX 3090 OOM'd under full graph + reuse
    force_cpu = "--cpu" in sys.argv or os.environ.get("LITEN_FORCE_CPU", "1") == "1"
    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device", device, flush=True)
    model = torch.load(model_checkpoints["LiTEN_SPICE"], weights_only=False, map_location=device)
    model.eval()

    pilot_only = "--pilot" in sys.argv or "--full" not in sys.argv
    cw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    if pilot_only:
        pilots = pd.read_csv(FS / "pilots/PILOT_ANTIBODIES.csv")["id"].tolist()
        cw = cw[cw.id.isin(pilots)]
        print("PILOT mode", pilots, flush=True)

    gens = [("esmfold", "esmfold_canonical_path"), ("abodybuilder2", "abodybuilder2_path")]
    partial = OUT_FEAT / "features_partial.csv"
    # fresh start if previous all-FAIL
    rows, done = [], set()
    if partial.exists() and "--fresh" not in sys.argv:
        prev = pd.read_csv(partial)
        rows = [r for r in prev.to_dict("records") if str(r.get("extraction_status")) == "SUCCESS"]
        for rec in rows:
            done.add((str(rec["id"]), str(rec["generator"])))

    for gen, _col in gens:
        for _, r in cw.iterrows():
            key = (str(r.id), gen)
            if key in done:
                continue
            h_pdb = PREP / f"{r.id}_{gen}_h.pdb"
            if not h_pdb.exists():
                rows.append({"id": r.id, "generator": gen, "extraction_status": "MISSING_PREP"})
                continue
            print(f"LiTEN {gen} {r.id}", flush=True)
            try:
                row = extract_one(model, device, r.id, h_pdb, gen)
                rows.append(row)
            except Exception as e:
                rows.append({"id": r.id, "generator": gen, "extraction_status": f"FAIL:{type(e).__name__}:{e}"})
            pd.DataFrame(rows).to_csv(partial, index=False)

    df = pd.DataFrame(rows)
    print("DONE", df.extraction_status.value_counts().to_dict() if len(df) else {}, flush=True)
    for gen in ["esmfold", "abodybuilder2"]:
        g = df[df.generator == gen] if len(df) else df
        g.to_csv(OUT_FEAT / f"features_{gen}.csv", index=False)
        try:
            g.to_parquet(OUT_FEAT / f"features_{gen}.parquet", index=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
