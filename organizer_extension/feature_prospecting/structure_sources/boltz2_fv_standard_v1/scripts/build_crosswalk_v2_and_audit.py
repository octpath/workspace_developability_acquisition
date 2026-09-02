#!/usr/bin/env python3
"""Target-blind Boltz-2 sequence/QC audit + STRUCTURE_INPUT_CROSSWALK_v2 builder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser, MMCIFParser

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
BASE = FP / "structure_sources/boltz2_fv_standard_v1"
V1 = FP / "STRUCTURE_INPUT_CROSSWALK.csv"
OUT_V2 = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
OUT_AUDIT = FP / "STRUCTURE_INPUT_CROSSWALK_V2_AUDIT.json"
MANIFEST = BASE / "BOLTZ2_STRUCTURE_MANIFEST.csv"
BOLTZ_AUDIT = BASE / "BOLTZ2_STRUCTURE_AUDIT.json"

AA = {k.upper(): v for k, v in protein_letters_3to1.items()}
AA["MSE"] = "M"


def sha256_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seq_from_pdb(path: Path) -> dict[str, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(path))
    model = next(structure.get_models())
    out = {}
    for chain in model.get_chains():
        seq = []
        seen = set()
        for res in chain.get_residues():
            het, resseq, icode = res.id
            if "CA" not in res:
                continue
            key = (resseq, icode)
            if key in seen:
                continue
            aa = AA.get(res.get_resname().upper())
            if aa is None:
                continue
            seen.add(key)
            seq.append(aa)
        out[chain.id] = "".join(seq)
    return out


def match(expected: str, observed: str | None) -> str:
    if observed is None:
        return "MISSING"
    if observed == expected:
        return "EXACT_MATCH"
    if expected in observed or observed in expected:
        return "PASS_EXPLAINED_SUBSTRING"
    return "MISMATCH"


def qc_pdb(path: Path) -> dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(path))
    model = next(structure.get_models())
    n_res = 0
    n_atoms = 0
    nan_coords = 0
    chains = []
    for chain in model.get_chains():
        chains.append(chain.id)
        for res in chain.get_residues():
            if "CA" in res:
                n_res += 1
            for atom in res.get_atoms():
                n_atoms += 1
                coord = atom.get_coord()
                if any(pd.isna(c) for c in coord):
                    nan_coords += 1
    return {
        "parse_ok": True,
        "chains": ",".join(chains),
        "has_H": "H" in chains,
        "has_L": "L" in chains,
        "n_res": n_res,
        "n_atoms": n_atoms,
        "nan_coords": nan_coords,
    }


def main() -> None:
    v1 = pd.read_csv(V1)
    # sequences
    seqs = {}
    for csvp in [
        ROOT / "competition/data/distribution/dev.csv",
        ROOT / "competition/data/distribution/test_features.csv",
    ]:
        df = pd.read_csv(csvp)
        for _, r in df.iterrows():
            seqs[r["id"]] = (r["heavy"], r["light"])

    rows = []
    man_rows = []
    n_boltz_pass = 0
    for _, r in v1.iterrows():
        aid = r["id"]
        vh, vl = seqs[aid]
        cif = BASE / "structures_mmcif" / f"{aid}.cif"
        pdb = BASE / "structures_pdb" / f"{aid}.pdb"
        confp = BASE / "confidence" / f"{aid}.json"
        exists = cif.exists() and pdb.exists()
        conf = {}
        if confp.exists():
            conf = json.loads(confp.read_text())
        h_m = l_m = "MISSING"
        map_status = "FAIL_MISSING"
        notes = []
        qc = {"parse_ok": False}
        if exists:
            try:
                obs = seq_from_pdb(pdb)
                h_m = match(vh, obs.get("H"))
                l_m = match(vl, obs.get("L"))
                qc = qc_pdb(pdb)
                if h_m == "EXACT_MATCH" and l_m == "EXACT_MATCH" and qc.get("has_H") and qc.get("has_L"):
                    map_status = "PASS"
                    n_boltz_pass += 1
                elif "MISMATCH" in (h_m, l_m) or "MISSING" in (h_m, l_m):
                    map_status = "FAIL"
                else:
                    map_status = "PASS_EXPLAINED"
            except Exception as e:
                map_status = "FAIL"
                notes.append(f"parse_error:{type(e).__name__}")
        else:
            notes.append("structure_not_generated")

        row = dict(r)
        row.update(
            {
                "boltz2_protocol_id": "BOLTZ2_FV_STANDARD_v1",
                "boltz2_native_cif_path": str(cif) if cif.exists() else "",
                "boltz2_pdb_path": str(pdb) if pdb.exists() else "",
                "boltz2_exists": bool(exists),
                "boltz2_heavy_sequence_match": h_m,
                "boltz2_light_sequence_match": l_m,
                "boltz2_mapping_status": map_status,
                "boltz2_confidence_score": conf.get("confidence_score"),
                "boltz2_ptm": conf.get("ptm"),
                "boltz2_iptm": conf.get("iptm"),
                "boltz2_complex_plddt": conf.get("complex_plddt"),
                "notes": (str(r.get("notes") or "") + ";" + ";".join(notes)).strip(";"),
            }
        )
        rows.append(row)
        man_rows.append(
            {
                "id": aid,
                "dataset_split": r["dataset_split"],
                "boltz2_exists": exists,
                "mapping_status": map_status,
                "heavy_match": h_m,
                "light_match": l_m,
                "cif_sha256": sha256_file(cif),
                "pdb_sha256": sha256_file(pdb),
                "confidence_sha256": sha256_file(confp),
                "confidence_score": conf.get("confidence_score"),
                "ptm": conf.get("ptm"),
                "iptm": conf.get("iptm"),
                "protein_iptm": conf.get("protein_iptm"),
                "complex_plddt": conf.get("complex_plddt"),
                "complex_iplddt": conf.get("complex_iplddt"),
                "n_res": qc.get("n_res"),
                "n_atoms": qc.get("n_atoms"),
                "nan_coords": qc.get("nan_coords"),
                "chains": qc.get("chains"),
            }
        )

    pd.DataFrame(rows).to_csv(OUT_V2, index=False)
    pd.DataFrame(man_rows).to_csv(MANIFEST, index=False)
    audit = {
        "crosswalk_version": "v2",
        "n_rows": len(rows),
        "esmfold_pass": int((v1["canonical_mapping_status"] == "PASS").sum()),
        "abb2_pass": int((v1["canonical_mapping_status"] == "PASS").sum()),
        "boltz2_exists": int(sum(1 for x in rows if x["boltz2_exists"])),
        "boltz2_pass": n_boltz_pass,
        "boltz2_mapping_status_counts": pd.Series([x["boltz2_mapping_status"] for x in rows]).value_counts().to_dict(),
        "target_labels_used": False,
        "output": str(OUT_V2),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    BOLTZ_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
