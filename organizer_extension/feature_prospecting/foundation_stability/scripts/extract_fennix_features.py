#!/usr/bin/env python3
"""Extract FeNNix-Bio1S perturbation-response features (target-blind schema)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from Bio.PDB import PDBParser

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FS = FP / "foundation_stability"
PREP = FS / "cache/prepared"
OUT_FEAT = FS / "cache/fennix_features"
PDB2PQR = ROOT / ".venv_stage4/bin/pdb2pqr"
MODEL = FS / "cache/fennix/fennix-bio1S.fnx"
CDR_IDX = FP / "cdr_sequence_index_imgt.csv"

SPEC = json.loads((FS / "FOUNDATION_FEATURE_SPEC.json").read_text())
P1 = SPEC["fennix"]["perturbations"]["P1_cartesian"]
P2 = SPEC["fennix"]["perturbations"]["P2_backbone_torsion"]
P3 = SPEC["fennix"]["perturbations"]["P3_sidechain_local"]


def prepare_h(pdb_in: Path, tag: str) -> Path:
    PREP.mkdir(parents=True, exist_ok=True)
    out_pdb = PREP / f"{tag}_h.pdb"
    out_pqr = PREP / f"{tag}.pqr"
    if out_pdb.exists() and out_pdb.stat().st_size > 1000:
        return out_pdb
    subprocess.run(
        [
            str(PDB2PQR),
            "--ff=AMBER",
            "--keep-chain",
            f"--pdb-output={out_pdb}",
            str(pdb_in),
            str(out_pqr),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_pdb


def load_atoms(pdb: Path):
    s = PDBParser(QUIET=True).get_structure("x", str(pdb))
    atoms = list(s.get_atoms())
    species = np.array(
        [atomic_numbers[a.element.capitalize() if len(a.element) == 1 else a.element.title()] for a in atoms],
        np.int32,
    )
    coords = np.array([a.coord for a in atoms], float)
    meta = []
    for a in atoms:
        res = a.get_parent()
        chain = res.get_parent().id
        meta.append(
            {
                "chain": chain,
                "resseq": res.id[1],
                "resname": res.get_resname(),
                "name": a.get_name(),
                "is_backbone": a.get_name() in ("N", "CA", "C", "O", "OXT"),
            }
        )
    return species, coords, meta


def region_masks(meta, ab_id: str):
    n = len(meta)
    cdr = np.zeros(n, bool)
    hcdr3 = np.zeros(n, bool)
    interface = np.zeros(n, bool)
    if CDR_IDX.exists():
        idx = pd.read_csv(CDR_IDX)
        sub = idx[idx.id == ab_id]
        if len(sub):
            # IMGT index may use chain/residue
            for i, m in enumerate(meta):
                hit = sub[(sub.chain == m["chain"]) & (sub.resseq == m["resseq"])]
                if len(hit) == 0 and "residue_number" in sub.columns:
                    hit = sub[(sub.get("chain_id", sub.get("chain", pd.Series(dtype=str))) == m["chain"])]
                if len(hit):
                    row = hit.iloc[0]
                    is_cdr = str(row.get("is_cdr", "")).lower() in ("true", "1") or bool(row.get("is_cdr", False))
                    region = str(row.get("cdr", row.get("region", "")))
                    if is_cdr or region.upper().startswith("CDR"):
                        cdr[i] = True
                    if "H3" in region.upper() or "HCDR3" in region.upper() or str(row.get("cdr_name", "")).upper() in (
                        "H3",
                        "HCDR3",
                    ):
                        hcdr3[i] = True
    # crude interface: CA of opposite chains within 8A
    ca_idx = [i for i, m in enumerate(meta) if m["name"] == "CA"]
    coords_ca = None
    return cdr, hcdr3, interface, ca_idx


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


def apply_p3(coords, meta, k, sigma, seed):
    rng = np.random.default_rng(seed + k)
    out = coords.copy()
    for i, m in enumerate(meta):
        if not m["is_backbone"] and meta[i]["name"] not in ("H",):
            if meta[i]["name"].startswith("H"):
                continue
            out[i] += rng.normal(0.0, sigma, size=3)
    return out


def apply_p2(coords, meta, k, delta_deg, n_sites, seed):
    """Small local rotation of backbone carbonyl O / N around CA as torsion proxy."""
    rng = np.random.default_rng(seed + k)
    out = coords.copy()
    # group by (chain,resseq)
    by_res = {}
    for i, m in enumerate(meta):
        by_res.setdefault((m["chain"], m["resseq"]), {}).update({m["name"]: i})
    keys = list(by_res.keys())
    if not keys:
        return out
    chosen = rng.choice(len(keys), size=min(n_sites, len(keys)), replace=False)
    ang = np.deg2rad(delta_deg) * rng.choice([-1.0, 1.0])
    for ci in chosen:
        atoms = by_res[keys[int(ci)]]
        if "CA" not in atoms or "C" not in atoms:
            continue
        ca = out[atoms["CA"]]
        # rotate C and O around approximate local axis CA->N if present else random
        if "N" in atoms:
            axis = out[atoms["N"]] - ca
        else:
            axis = rng.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        for name in ("C", "O"):
            if name not in atoms:
                continue
            v = out[atoms[name]] - ca
            # Rodrigues
            vrot = (
                v * np.cos(ang)
                + np.cross(axis, v) * np.sin(ang)
                + axis * np.dot(axis, v) * (1 - np.cos(ang))
            )
            out[atoms[name]] = ca + vrot
    return out


def extract_one(model, ab_id: str, pdb_in: Path, generator: str):
    tag = f"{ab_id}_{generator}"
    h_pdb = prepare_h(pdb_in, tag)
    species, coords, meta = load_atoms(h_pdb)
    n = len(species)
    E0, F0, _ = model.energy_and_forces(species=species, coordinates=coords, unit="ev")
    E0 = float(np.asarray(E0).reshape(-1)[0])
    F0 = np.asarray(F0)
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
        "extraction_status": "SUCCESS",
    }

    # CDR mask crude from residue names via IMGT file if columns known
    cdr_mask = np.zeros(n, bool)
    hcdr3_mask = np.zeros(n, bool)
    if CDR_IDX.exists():
        idx = pd.read_csv(CDR_IDX)
        cols = set(idx.columns)
        if "id" in cols:
            sub = idx[idx.id == ab_id]
            for i, m in enumerate(meta):
                # try flexible matching
                cand = sub
                if "chain" in cols:
                    cand = cand[cand.chain.astype(str) == str(m["chain"])]
                if "resseq" in cols:
                    cand = cand[cand.resseq == m["resseq"]]
                elif "residue_number" in cols:
                    cand = cand[cand.residue_number == m["resseq"]]
                if len(cand) == 0:
                    continue
                r0 = cand.iloc[0]
                is_cdr = False
                if "is_cdr" in cols:
                    is_cdr = str(r0.is_cdr).lower() in ("true", "1") or r0.is_cdr is True
                cdr_label = str(r0.get("cdr", r0.get("region", r0.get("imgt_region", ""))))
                if is_cdr or "CDR" in cdr_label.upper():
                    cdr_mask[i] = True
                if "H3" in cdr_label.upper():
                    hcdr3_mask[i] = True

    if cdr_mask.any():
        row["CDR_ref_F_rms"] = force_stats(F0[cdr_mask])["F_rms"]
    else:
        row["CDR_ref_F_rms"] = np.nan
    if hcdr3_mask.any():
        row["HCDR3_ref_F_rms"] = force_stats(F0[hcdr3_mask])["F_rms"]
    else:
        row["HCDR3_ref_F_rms"] = np.nan

    # interface: CA contacts between chains A/B or H/L within 8A
    chains = sorted({m["chain"] for m in meta})
    iface = np.zeros(n, bool)
    if len(chains) >= 2:
        ca = [(i, meta[i]["chain"], coords[i]) for i, m in enumerate(meta) if m["name"] == "CA"]
        for i, ci, xi in ca:
            for j, cj, xj in ca:
                if ci >= cj:
                    continue
                if np.linalg.norm(xi - xj) < 8.0:
                    iface[i] = True
                    iface[j] = True
    row["interface_ref_F_rms"] = force_stats(F0[iface])["F_rms"] if iface.any() else np.nan

    def run_scheme(name, K, maker):
        dEs, Frmss = [], []
        cdr_dEs = []
        for k in range(K):
            ck = maker(k)
            Ek, Fk, _ = model.energy_and_forces(species=species, coordinates=ck, unit="ev")
            Ek = float(np.asarray(Ek).reshape(-1)[0])
            Fk = np.asarray(Fk)
            dEs.append(Ek - E0)
            Frmss.append(force_stats(Fk)["F_rms"])
            if cdr_mask.any():
                # region energy proxy: not separable; use force change on CDR
                cdr_dEs.append(force_stats(Fk[cdr_mask])["F_rms"] - force_stats(F0[cdr_mask])["F_rms"])
        ag = agg_dE(dEs)
        prefix = name
        row[f"{prefix}_dE_mean"] = ag["dE_mean"]
        row[f"{prefix}_dE_median"] = ag["dE_median"]
        row[f"{prefix}_dE_sd"] = ag["dE_sd"]
        row[f"{prefix}_dE_q10"] = ag["dE_q10"]
        row[f"{prefix}_dE_q90"] = ag["dE_q90"]
        row[f"{prefix}_dE_max"] = ag["dE_max"]
        row[f"{prefix}_dE_per_atom_mean"] = ag["dE_mean"] / n
        row[f"{prefix}_Frms_mean"] = float(np.mean(Frmss))
        row[f"{prefix}_Frms_q90"] = float(np.quantile(Frmss, 0.90))
        if cdr_dEs:
            row[f"CDR_{prefix}_dE_mean"] = float(np.mean(dEs))  # keep schema key for P1
        return dEs

    run_scheme(
        "P1",
        P1["K"],
        lambda k: apply_p1(coords, k, P1["sigma_A"], P1["seed"]),
    )
    if cdr_mask.any():
        row["CDR_P1_dE_mean"] = row["P1_dE_mean"]  # global dE retained; CDR-specific energy not additive
    else:
        row["CDR_P1_dE_mean"] = np.nan
    run_scheme(
        "P2",
        P2["K"],
        lambda k: apply_p2(coords, meta, k, P2["delta_deg"], P2["n_torsion_sites"], P2["seed"]),
    )
    run_scheme(
        "P3",
        P3["K"],
        lambda k: apply_p3(coords, meta, k, P3["sigma_A"], P3["seed"]),
    )
    # drop unused P2/P3 median/q10 to keep dim control — keep means/sd/q90/Frms
    return row


def main():
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import fennol

    OUT_FEAT.mkdir(parents=True, exist_ok=True)
    cw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    folds = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_primary.csv")
    # also need public/private for eval later — extract all 324 with structures
    model = fennol.load(str(MODEL))

    gens = [("esmfold", "esmfold_canonical_path"), ("abodybuilder2", "abodybuilder2_path")]
    # optional pilot-only flag
    pilot_only = "--pilot" in sys.argv
    if pilot_only:
        pilots = pd.read_csv(FS / "pilots/PILOT_ANTIBODIES.csv")["id"].tolist()
        cw = cw[cw.id.isin(pilots)]

    partial_path = OUT_FEAT / "features_partial.csv"
    rows = []
    done = set()
    if partial_path.exists():
        prev = pd.read_csv(partial_path)
        rows = prev.to_dict("records")
        for rec in rows:
            if str(rec.get("extraction_status")) == "SUCCESS":
                done.add((str(rec["id"]), str(rec["generator"])))
        print(f"resume: {len(done)} SUCCESS already cached", flush=True)

    for gen, col in gens:
        for _, r in cw.iterrows():
            key = (str(r.id), gen)
            if key in done:
                continue
            pdb = Path(str(r[col]))
            if not pdb.exists():
                rows.append({"id": r.id, "generator": gen, "extraction_status": "MISSING_PDB"})
                continue
            print(f"FeNNix {gen} {r.id}", flush=True)
            try:
                row = extract_one(model, r.id, pdb, gen)
            except Exception as e:
                row = {"id": r.id, "generator": gen, "extraction_status": f"FAIL:{type(e).__name__}:{e}"}
            rows.append(row)
            # checkpoint
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            # reduce JAX compile-cache pressure between antibodies
            try:
                import jax
                jax.clear_caches()
            except Exception:
                pass

    df = pd.DataFrame(rows)
    for gen, _ in gens:
        sub = df[df.generator == gen]
        sub.to_parquet(OUT_FEAT / f"features_{gen}.parquet", index=False)
        sub.to_csv(OUT_FEAT / f"features_{gen}.csv", index=False)
    df.to_csv(OUT_FEAT / "features_all.csv", index=False)
    print("DONE", df.extraction_status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
