#!/usr/bin/env python3
"""Target-blind VHL-ANGLE_v1 extraction via vendored ABangle (Dunbar et al.)."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FAMILY = FP / "VHL-ANGLE"
VENDOR = FAMILY / "vendor/ABangle"
sys.path.insert(0, str(VENDOR))

from abangle.calculate import find_angles, coresets  # noqa: E402

CROSSWALK = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
GEN_PATH = {
    "esmfold": "esmfold_canonical_path",
    "abodybuilder2": "abodybuilder2_path",
    "boltz2": "boltz2_pdb_path",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ca_counts(pdb_path: Path) -> dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    n_h = n_l = 0
    chains = []
    for ch in model.get_chains():
        chains.append(ch.id)
        n_ca = sum(1 for r in ch.get_residues() if is_aa(r, standard=True) and "CA" in r)
        # provisional: H/L or A/B mapping unknown until ANARCI; store total later
        if ch.id in ("H", "A"):
            n_h = n_ca
        elif ch.id in ("L", "B"):
            n_l = n_ca
    return {"n_CA_H_or_A": n_h, "n_CA_L_or_B": n_l, "input_chains": ",".join(chains)}


def framework_ca_coords(pdb_path: Path, chain: str) -> np.ndarray | None:
    """CA coords for ABangle coreset residue numbers on chain H or L (author numbering may differ).
    For RMSD QC we use all CA on mapped H/L-like chains after a light rename heuristic.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    # map A->H, B->L if needed
    chain_map = {}
    ids = [c.id for c in model.get_chains()]
    if "H" in ids and "L" in ids:
        chain_map = {"H": "H", "L": "L"}
    elif "A" in ids and "B" in ids:
        chain_map = {"H": "A", "L": "B"}
    else:
        return None
    cid = chain_map[chain]
    wanted = set(coresets[chain])
    coords = []
    for res in model[cid].get_residues():
        if not is_aa(res, standard=True) or "CA" not in res:
            continue
        if res.id[1] in wanted:
            coords.append(res.get_coord())
    if len(coords) < 5:
        return None
    return np.asarray(coords, dtype=float)


def pairwise_framework_rmsd(path_a: Path, path_b: Path) -> float | None:
    """RMSD of overlapping coreset CA on L then report; uses author/Chothia-ish resseq match."""
    try:
        ca_a = []
        ca_b = []
        for chain in ("L", "H"):
            a = framework_ca_coords(path_a, chain)
            b = framework_ca_coords(path_b, chain)
            if a is None or b is None:
                return None
            n = min(len(a), len(b))
            ca_a.append(a[:n])
            ca_b.append(b[:n])
        A = np.vstack(ca_a)
        B = np.vstack(ca_b)
        if len(A) < 10:
            return None
        # Kabsch via Bio Superimposer using dummy Atom-like arrays is heavy; use SVD
        Ac = A - A.mean(0)
        Bc = B - B.mean(0)
        U, _, Vt = np.linalg.svd(Ac.T @ Bc)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        B_al = Bc @ R
        return float(np.sqrt(np.mean(np.sum((Ac - B_al) ** 2, axis=1))))
    except Exception:
        return None


def process_one(aid: str, gen: str, path: Path) -> dict:
    out = {
        "id": aid,
        "generator": gen,
        "pdb_path": str(path),
        "pdb_sha256": sha256_file(path) if path.exists() else None,
        "extraction_status": "FAIL",
    }
    if not path.exists():
        out["error"] = "missing_pdb"
        return out
    out.update(ca_counts(path))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ang = find_angles(path)
        for k in CANON:
            out[k] = float(ang[k])
        out["extraction_status"] = "SUCCESS"
        out["missing_coreset_flag"] = False
    except Exception as e:
        out["error"] = f"{type(e).__name__}:{e}"
        out["extraction_status"] = "FAIL"
        out["missing_coreset_flag"] = True
    return out


def main() -> None:
    FAMILY.mkdir(parents=True, exist_ok=True)
    cw = pd.read_csv(CROSSWALK)
    assert len(cw) == 324
    all_qc = []
    man = []
    paths_by_gen = {g: {} for g in GEN_PATH}
    for gen, col in GEN_PATH.items():
        rows = []
        for i, r in cw.iterrows():
            aid = r["id"]
            path = Path(str(r[col]))
            paths_by_gen[gen][aid] = path
            rec = process_one(aid, gen, path)
            rows.append(rec)
            all_qc.append(rec)
            man.append(
                {
                    "id": aid,
                    "generator": gen,
                    "extraction_status": rec["extraction_status"],
                    "pdb_path": rec.get("pdb_path"),
                    "pdb_sha256": rec.get("pdb_sha256"),
                    "error": rec.get("error"),
                }
            )
            if (i + 1) % 50 == 0:
                print(gen, i + 1, flush=True)
        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda x: "" if x is None else str(x))
        outp = FAMILY / f"features_{gen}.parquet"
        df.to_parquet(outp, index=False)
        ok = int((df["extraction_status"] == "SUCCESS").sum())
        print(f"{gen}: {ok}/{len(df)}", flush=True)

    # pairwise framework RMSD QC (not canonical features)
    qc_extra = []
    ids = list(cw["id"])
    for aid in ids:
        row = {"id": aid}
        for a, b, key in [
            ("esmfold", "abodybuilder2", "framework_CA_rmsd_esmfold_vs_abb2"),
            ("esmfold", "boltz2", "framework_CA_rmsd_esmfold_vs_boltz2"),
            ("abodybuilder2", "boltz2", "framework_CA_rmsd_abb2_vs_boltz2"),
        ]:
            row[key] = pairwise_framework_rmsd(paths_by_gen[a][aid], paths_by_gen[b][aid])
        qc_extra.append(row)
    pd.DataFrame(qc_extra).to_csv(FAMILY / "framework_rmsd_qc.csv", index=False)

    pd.DataFrame(all_qc).to_csv(FAMILY / "extraction_qc.csv", index=False)
    pd.DataFrame(man).to_csv(FAMILY / "FEATURE_MANIFEST.csv", index=False)

    freeze = {"state": "VHL_ANGLE_V1_FEATURE_SPEC_FROZEN", "target_scoring_started_after_feature_freeze": False, "files": {}}
    for p in [
        FAMILY / "FEATURE_SPEC.json",
        FAMILY / "FEATURE_MANIFEST.csv",
        FAMILY / "extraction_qc.csv",
        FAMILY / "framework_rmsd_qc.csv",
        FAMILY / "features_esmfold.parquet",
        FAMILY / "features_abodybuilder2.parquet",
        FAMILY / "features_boltz2.parquet",
    ]:
        freeze["files"][str(p.relative_to(ROOT))] = {"sha256": sha256_file(p), "nbytes": p.stat().st_size}
    (FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print("wrote target-blind hashes")


if __name__ == "__main__":
    main()
