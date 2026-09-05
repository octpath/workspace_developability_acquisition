#!/usr/bin/env python3
"""FeNNix-v2: R1 sidechain-relax + symmetric torsional curvature features."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
from Bio.PDB import PDBParser

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
V2 = FP / "foundation_stability_v2"
PREP = V2 / "cache/prepared_v1"
MODEL = V2 / "cache/fennix-bio1S.fnx"
CDR_IDX = FP / "cdr_sequence_index_imgt.csv"
OUT_QC = V2 / "FENNIX_V2_RELAXATION_QC.csv"
OUT_FEAT = V2 / "FENNIX_V2_CURVATURE_FEATURES.csv"
OUT_SITES = V2 / "cache/fennix_curvature/selected_sites.jsonl"
R1_DIR = V2 / "cache/fennix_r1"
PARTIAL = V2 / "cache/fennix_curvature/features_partial.csv"

SPEC = json.loads((V2 / "FOUNDATION_STABILITY_V2_SPEC.json").read_text())
FX = SPEC["fennix_v2"]
FMAX = float(FX["reference_states"]["R1_SIDECHAIN_RELAXED"]["fmax_eV_A"])
MAX_STEPS = int(FX["reference_states"]["R1_SIDECHAIN_RELAXED"]["max_steps"])
PHI_PSI_DEG = float(FX["perturbations"]["backbone_phi_psi_deg"])
CHI1_DEG = float(FX["perturbations"]["sidechain_chi1_deg"])
SEED = int(FX["site_selection"]["seed"])
BB_FIX = set(FX["reference_states"]["R1_SIDECHAIN_RELAXED"]["fix_backbone_heavy"])

from ase.data import atomic_numbers  # noqa: E402


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def force_stats(F):
    nrm = np.linalg.norm(F, axis=1)
    return float(np.sqrt(np.mean(nrm**2))), float(np.quantile(nrm, 0.9)), float(np.max(nrm))


def clash_counts(coords, meta):
    """Inter-residue heavy-atom clashes only (exclude bonded intra-residue pairs)."""
    heavy = [(i, m) for i, m in enumerate(meta) if m["is_heavy"]]
    severe = allc = 0
    min_nb = np.inf
    for a in range(len(heavy)):
        i, mi = heavy[a]
        for b in range(a + 1, len(heavy)):
            j, mj = heavy[b]
            if mi["chain"] == mj["chain"] and abs(mi["resseq"] - mj["resseq"]) <= 1:
                # skip same / adjacent residue (covers most covalent neighbors)
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
    atoms_b = list(s.get_atoms())
    species = []
    coords = []
    meta = []
    for a in atoms_b:
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
    symbols = []
    from ase.data import chemical_symbols

    for z in species:
        symbols.append(chemical_symbols[int(z)])
    return Atoms(symbols=symbols, positions=coords)


def region_table(ab_id: str):
    if not CDR_IDX.exists():
        return pd.DataFrame()
    idx = pd.read_csv(CDR_IDX)
    return idx[idx.id == ab_id].copy()


def classify_residue(row_region: str) -> str:
    r = str(row_region).upper()
    if "H3" in r or r in {"CDR3", "HCDR3"}:
        return "HCDR3"
    if "CDR" in r:
        return "CDR"
    return "FW"


def pick_sites(ab_id: str, meta, regions: pd.DataFrame):
    """Deterministic torsion site selection."""
    # residue keys
    residues = {}
    for i, m in enumerate(meta):
        key = (m["chain"], m["resseq"])
        residues.setdefault(key, {})[m["name"]] = i

    # map region from CDR index: chain H/L vs structure chain ids
    reg_map = {}
    if len(regions):
        for _, r in regions.iterrows():
            ch = str(r.chain)
            # map H->first heavy-like chain, L->light; structures often H/L or A/B
            reg_map[(ch, int(r.imgt_number) if "imgt_number" in regions.columns else int(r.sequence_index))] = classify_residue(
                r.get("region", "")
            )

    # Fallback region by CDR index match on sequence_index within chain order
    # Better: match by amino acid sequence position using CDR file sequence_index + chain letter
    chain_ids = sorted({m["chain"] for m in meta})
    # Prefer H/L naming
    heavy = "H" if "H" in chain_ids else (chain_ids[0] if chain_ids else "A")
    light = "L" if "L" in chain_ids else (chain_ids[1] if len(chain_ids) > 1 else chain_ids[0])

    # Build per-residue region using CDR_IDX (chain, sequence_index) aligned to sorted resseq within chain
    res_region = {}
    if len(regions):
        for ch_src, ch_dst in [("H", heavy), ("L", light)]:
            sub = regions[regions.chain.astype(str) == ch_src].sort_values("sequence_index")
            keys = sorted([k for k in residues if k[0] == ch_dst], key=lambda x: x[1])
            for i, (_, r) in enumerate(sub.iterrows()):
                if i >= len(keys):
                    break
                res_region[keys[i]] = classify_residue(r.get("region", ""))
    for k in residues:
        res_region.setdefault(k, "FW")

    def valid_bb(key):
        at = residues[key]
        return all(n in at for n in ("N", "CA", "C"))

    def valid_chi1(key):
        at = residues[key]
        # need CA, CB, and CG or OG or SG etc.
        if "CA" not in at or "CB" not in at:
            return False
        for n in at:
            if n.startswith(("CG", "OG", "SG", "NG")):
                return True
        return False

    fw = [k for k, r in res_region.items() if r == "FW" and valid_bb(k)]
    cdr = [k for k, r in res_region.items() if r == "CDR" and valid_bb(k)]
    h3 = [k for k, r in res_region.items() if r == "HCDR3" and valid_bb(k)]
    # CDR includes HCDR3 in some labels — keep h3 separate extra
    cdr = [k for k in cdr if k not in h3]

    def take(pool, n, tag):
        if not pool:
            return []
        h = stable_hash(f"{ab_id}|{tag}|{SEED}")
        rng = np.random.default_rng(h % (2**32))
        order = np.argsort(rng.random(len(pool)))
        sel = [pool[i] for i in order[: min(n, len(pool))]]
        return sel

    sel_fw = take(fw, FX["site_selection"]["framework_backbone"], "FW")
    sel_cdr = take(cdr, FX["site_selection"]["cdr_backbone"], "CDR")
    sel_h3 = take(h3, FX["site_selection"]["hcdr3_backbone_extra"], "HCDR3")
    chi_pool = [k for k in residues if valid_chi1(k)]
    sel_chi = take(chi_pool, FX["site_selection"]["chi1"], "CHI1")

    sites = []
    for k in sel_fw:
        sites.append({"kind": "phi", "region": "FW", "chain": k[0], "resseq": k[1]})
        sites.append({"kind": "psi", "region": "FW", "chain": k[0], "resseq": k[1]})
    for k in sel_cdr:
        sites.append({"kind": "phi", "region": "CDR", "chain": k[0], "resseq": k[1]})
        sites.append({"kind": "psi", "region": "CDR", "chain": k[0], "resseq": k[1]})
    for k in sel_h3:
        sites.append({"kind": "phi", "region": "HCDR3", "chain": k[0], "resseq": k[1]})
        sites.append({"kind": "psi", "region": "HCDR3", "chain": k[0], "resseq": k[1]})
    for k in sel_chi:
        sites.append({"kind": "chi1", "region": res_region.get(k, "FW"), "chain": k[0], "resseq": k[1]})
    return sites, residues, res_region


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
    at = residues[key]
    kind = site["kind"]
    if kind == "phi":
        if not all(n in at for n in ("N", "CA")):
            return None
        origin = coords[at["N"]]
        axis = coords[at["CA"]] - coords[at["N"]]
        move = []
        for n, i in at.items():
            if n in ("N",):
                continue
            if n.startswith("H") and n in ("H", "H1", "H2", "H3"):
                # leave amide H on N
                continue
            move.append(i)
        return rodrigues(coords, move, origin, axis, delta_rad)
    if kind == "psi":
        if not all(n in at for n in ("CA", "C")):
            return None
        origin = coords[at["CA"]]
        axis = coords[at["C"]] - coords[at["CA"]]
        move = [at[n] for n in ("O", "OXT") if n in at]
        # next residue N/H if present
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
            if n in ("N", "CA", "C", "O", "OXT", "H", "HA", "CB"):
                continue
            # hydrogens on CB stay? rotate with CB side
            if n in ("HB1", "HB2", "HB3", "HB"):
                continue
            move.append(i)
        return rodrigues(coords, move, origin, axis, delta_rad)
    return None


def energy_forces_model(model, species, coords):
    E, F, _ = model.energy_and_forces(species=species, coordinates=coords, unit="ev")
    E = float(np.asarray(E).reshape(-1)[0])
    F = np.asarray(F, float)
    return E, F


def relax_r1(model, species, coords, meta):
    atoms = to_ase(species, coords)
    # Attach calculator via fennol ASE if available; else custom
    try:
        from fennol.ase import FENNIXCalculator

        atoms.calc = FENNIXCalculator(model, unit="eV")
    except Exception:
        # fallback: wrap energy_and_forces
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
    # convergence: max force on free atoms
    free = np.ones(len(meta), bool)
    free[fix_idx] = False
    fmax = float(np.max(np.linalg.norm(f[free], axis=1))) if free.any() else float(np.max(np.linalg.norm(f, axis=1)))
    converged = fmax <= FMAX + 1e-6
    steps = int(getattr(opt, "nsteps", MAX_STEPS))
    return final, e, f, converged, steps, fmax, opt_name


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


def process_one(model, ab_id, generator, pdb: Path):
    species, coords0, meta = load_structure(pdb)
    is_heavy = np.array([m["is_heavy"] for m in meta])
    n = len(species)

    # R0 diagnostics
    E0r, F0r = energy_forces_model(model, species, coords0)
    frm0, fq0, fmx0 = force_stats(F0r)
    sev0, all0, min0 = clash_counts(coords0, meta)

    # R1
    coords1, E1, F1, converged, steps, fmax, opt_name = relax_r1(model, species, coords0, meta)
    frm1, fq1, fmx1 = force_stats(F1)
    sev1, all1, min1 = clash_counts(coords1, meta)

    # RMSDs
    bb_idx = np.array([m["is_backbone_heavy"] for m in meta])
    sc_heavy = np.array([m["is_heavy"] and not m["is_backbone_heavy"] for m in meta])
    bb_rmsd = float(np.sqrt(np.mean((coords1[bb_idx] - coords0[bb_idx]) ** 2))) if bb_idx.any() else np.nan
    sc_rmsd = float(np.sqrt(np.mean((coords1[sc_heavy] - coords0[sc_heavy]) ** 2))) if sc_heavy.any() else np.nan

    qc = {
        "id": ab_id,
        "generator": generator,
        "n_atoms": n,
        "R0_E": E0r,
        "R0_F_rms": frm0,
        "R0_F_q90": fq0,
        "R0_F_max": fmx0,
        "R0_severe_clash": sev0,
        "R0_all_clash": all0,
        "R0_min_heavy_dist": min0,
        "R1_E": E1,
        "R1_F_rms": frm1,
        "R1_F_q90": fq1,
        "R1_F_max": fmx1,
        "R1_severe_clash": sev1,
        "R1_all_clash": all1,
        "R1_min_heavy_dist": min1,
        "delta_severe_clash": sev1 - sev0,
        "delta_F_rms": frm1 - frm0,
        "sidechain_heavy_RMSD": sc_rmsd,
        "backbone_RMSD": bb_rmsd,
        "converged": bool(converged),
        "steps": steps,
        "final_fmax": fmax,
        "optimizer": opt_name,
        "failure_mode": "" if converged else "NOT_CONVERGED_WITHIN_MAX_STEPS",
    }

    # Save R1 pdb-like xyz cache
    R1_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(R1_DIR / f"{ab_id}_{generator}.npz", species=species, coords=coords1, E=E1)

    regions = region_table(ab_id)
    sites, residues, res_region = pick_sites(ab_id, meta, regions)
    with OUT_SITES.open("a") as fh:
        fh.write(json.dumps({"id": ab_id, "generator": generator, "sites": sites}) + "\n")

    # E0 on R1
    E_ref = E1
    buckets = {"all_bb": [], "FW": [], "CDR": [], "HCDR3": [], "chi1": []}

    for site in sites:
        deg = CHI1_DEG if site["kind"] == "chi1" else PHI_PSI_DEG
        delta = np.deg2rad(deg)
        cp = apply_torsion(coords1, residues, site, +delta)
        cm = apply_torsion(coords1, residues, site, -delta)
        if cp is None or cm is None:
            continue
        Ep, _ = energy_forces_model(model, species, cp)
        Em, _ = energy_forces_model(model, species, cm)
        Ke = (Ep + Em - 2.0 * E_ref) / (delta * delta)
        if site["kind"] == "chi1":
            buckets["chi1"].append(Ke)
        else:
            buckets["all_bb"].append(Ke)
            buckets[site["region"]].append(Ke)
            if site["region"] == "HCDR3":
                pass
            # HCDR3 also in CDR? keep separate only

    feat = {"id": ab_id, "generator": generator, "extraction_status": "SUCCESS", "n_sites": len(sites)}
    for name, vals in buckets.items():
        ag = agg_K(vals)
        for k, v in ag.items():
            feat[f"K_{name}_{k}"] = v
    # keep <=25 primary: drop some raw n's later in eval
    return qc, feat


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import fennol

    model = fennol.load(str(MODEL))
    cw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    pilot = "--pilot" in sys.argv
    if pilot:
        # 3 target-blind from v1 pilots
        ids = pd.read_csv(FP / "foundation_stability/pilots/PILOT_ANTIBODIES.csv").id.tolist()
        cw = cw[cw.id.isin(ids)]
        print("PILOT", ids, flush=True)

    # optional shard: --shard 0/2
    shard_tag = "all"
    if "--shard" in sys.argv:
        spec = sys.argv[sys.argv.index("--shard") + 1]
        si, sn = map(int, spec.split("/"))
        ids_all = cw.id.astype(str).tolist()
        keep = {i for j, i in enumerate(ids_all) if j % sn == si}
        cw = cw[cw.id.astype(str).isin(keep)]
        shard_tag = f"shard{si}of{sn}"
        print(f"SHARD {si}/{sn} n={len(cw)}", flush=True)

    partial_path = V2 / "cache/fennix_curvature" / f"features_partial_{shard_tag}.csv"
    qc_path_partial = V2 / "cache/fennix_curvature" / f"qc_partial_{shard_tag}.csv"

    done = set()
    qc_rows, feat_rows = [], []
    # resume from shard file + global merged success
    for p in [partial_path, PARTIAL]:
        if p.exists() and "--fresh" not in sys.argv:
            prev = pd.read_csv(p)
            for r in prev.to_dict("records"):
                if r.get("extraction_status") == "SUCCESS":
                    done.add((str(r["id"]), str(r["generator"])))
    if partial_path.exists() and "--fresh" not in sys.argv:
        feat_rows = pd.read_csv(partial_path).to_dict("records")
    if qc_path_partial.exists() and "--fresh" not in sys.argv:
        qc_rows = pd.read_csv(qc_path_partial).to_dict("records")
    print("resume", len(done), flush=True)

    gens = ["esmfold", "abodybuilder2"]
    for gen in gens:
        for _, r in cw.iterrows():
            key = (str(r.id), gen)
            if key in done:
                continue
            pdb = PREP / f"{r.id}_{gen}_h.pdb"
            print(f"FENNIX_V2 {gen} {r.id}", flush=True)
            try:
                qc, feat = process_one(model, r.id, gen, pdb)
                qc_rows.append(qc)
                feat_rows.append(feat)
            except Exception as e:
                feat_rows.append(
                    {"id": r.id, "generator": gen, "extraction_status": f"FAIL:{type(e).__name__}:{e}"}
                )
                qc_rows.append({"id": r.id, "generator": gen, "failure_mode": f"{type(e).__name__}:{e}"})
            pd.DataFrame(feat_rows).to_csv(partial_path, index=False)
            pd.DataFrame(qc_rows).to_csv(qc_path_partial, index=False)
            try:
                import jax

                jax.clear_caches()
            except Exception:
                pass

    pd.DataFrame(qc_rows).to_csv(V2 / "cache/fennix_curvature" / f"qc_{shard_tag}.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(V2 / "cache/fennix_curvature" / f"features_{shard_tag}.csv", index=False)
    print("DONE feats", pd.DataFrame(feat_rows).extraction_status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
