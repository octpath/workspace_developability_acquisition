#!/usr/bin/env python3
"""Target-blind Physical Batch1 extraction for all four families (one SASA pass)."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
import sys

sys.path.insert(0, str(FP))
from common.structure_utils import (  # noqa: E402
    AROMATIC,
    BLACK_MOULD_SAP,
    GEN_PATH,
    LOCAL_R,
    MAX_ASA,
    PATCH_R,
    RASA_EXPOSED,
    RASA_STRONG,
    build_residue_table,
    connected_components,
    load_cdr_map,
    load_sequences,
)

PDB2PQR = ROOT / ".venv_stage4/bin/pdb2pqr30"
N_WORKERS = 6

SPECS = {
    "AROMATIC-TOPO": json.loads((FP / "AROMATIC-TOPO/FEATURE_SPEC.json").read_text()),
    "STATIC-SAP": json.loads((FP / "STATIC-SAP/FEATURE_SPEC.json").read_text()),
    "VOID-EXPLICIT": json.loads((FP / "VOID-EXPLICIT/FEATURE_SPEC.json").read_text()),
    "POLAR-SAT": json.loads((FP / "POLAR-SAT/FEATURE_SPEC.json").read_text()),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def aromatic_features(rows: list[dict]) -> dict:
    arom = [r for r in rows if r["amino_acid"] in AROMATIC]
    exp = [r for r in arom if r["is_exposed"]]
    strong = [r for r in arom if r["is_strongly_exposed"]]
    total_sasa = float(sum(r["sasa"] for r in rows if np.isfinite(r["sasa"])))
    arom_sasa = float(sum(r["sasa"] for r in exp if np.isfinite(r["sasa"])))
    cdr_exp = [r for r in exp if r["is_cdr"]]
    cdr_sasa = float(sum(r["sasa"] for r in cdr_exp if np.isfinite(r["sasa"])))

    # patches among exposed aromatics
    idxs = list(range(len(exp)))
    coords = np.array([r["ca"] for r in exp if r["ca"] is not None])
    valid = [r for r in exp if r["ca"] is not None]
    if len(valid) >= 1:
        comps = connected_components(list(range(len(valid))), np.array([r["ca"] for r in valid]), PATCH_R)
        patch_count = len(comps)
        sizes = [len(c) for c in comps]
        largest = comps[int(np.argmax(sizes))] if comps else []
        largest_n = len(largest)
        largest_sasa = float(sum(valid[i]["sasa"] for i in largest)) if largest else 0.0
    else:
        patch_count = largest_n = 0
        largest_sasa = 0.0

    # max local aromatic SASA within LOCAL_R among exposed aromatics
    max_local = 0.0
    if valid:
        C = np.array([r["ca"] for r in valid])
        S = np.array([r["sasa"] for r in valid])
        for i in range(len(valid)):
            d = np.linalg.norm(C - C[i], axis=1)
            max_local = max(max_local, float(S[d <= LOCAL_R].sum()))

    return {
        "exposed_TYR_count": sum(1 for r in exp if r["amino_acid"] == "Y"),
        "exposed_TRP_count": sum(1 for r in exp if r["amino_acid"] == "W"),
        "exposed_PHE_count": sum(1 for r in exp if r["amino_acid"] == "F"),
        "exposed_aromatic_total_count": len(exp),
        "aromatic_exposed_SASA_total": arom_sasa,
        "aromatic_exposed_SASA_fraction": arom_sasa / total_sasa if total_sasa > 0 else 0.0,
        "strongly_exposed_aromatic_count": len(strong),
        "strongly_exposed_aromatic_SASA": float(sum(r["sasa"] for r in strong if np.isfinite(r["sasa"]))),
        "CDR_exposed_aromatic_count": len(cdr_exp),
        "CDR_aromatic_SASA": cdr_sasa,
        "CDR_aromatic_fraction": cdr_sasa / arom_sasa if arom_sasa > 0 else 0.0,
        "aromatic_patch_count": int(patch_count),
        "largest_aromatic_patch_n_res": int(largest_n),
        "largest_aromatic_patch_exposed_SASA": largest_sasa,
        "max_local_aromatic_SASA": max_local,
        "sequence_aromatic_count": len(arom),
        "sequence_TYR_count": sum(1 for r in arom if r["amino_acid"] == "Y"),
        "sequence_TRP_count": sum(1 for r in arom if r["amino_acid"] == "W"),
        "sequence_PHE_count": sum(1 for r in arom if r["amino_acid"] == "F"),
    }


def static_sap_features(rows: list[dict]) -> dict:
    usable = [r for r in rows if r["ca"] is not None and np.isfinite(r["rasa"])]
    coords = np.array([r["ca"] for r in usable])
    rasa = np.array([r["rasa"] for r in usable])
    hphi = np.array([BLACK_MOULD_SAP.get(r["amino_acid"], 0.0) for r in usable])
    out = {}
    for R, tag in [(5.0, "R5"), (10.0, "R10")]:
        sap = np.zeros(len(usable))
        for i in range(len(usable)):
            d = np.linalg.norm(coords - coords[i], axis=1)
            m = d <= R
            sap[i] = float((rasa[m] * hphi[m]).sum())
        pos = sap[sap > 0]
        out[f"{tag}_max_positive_static_SAP"] = float(pos.max()) if len(pos) else 0.0
        out[f"{tag}_mean_positive_static_SAP"] = float(pos.mean()) if len(pos) else 0.0
        out[f"{tag}_sum_positive_static_SAP"] = float(pos.sum()) if len(pos) else 0.0
        out[f"{tag}_positive_SAP_residue_fraction"] = float((sap > 0).mean()) if len(sap) else 0.0
        order = np.sort(sap)[::-1]
        out[f"{tag}_top3_mean_static_SAP"] = float(order[:3].mean()) if len(order) else 0.0
        out[f"{tag}_top5_mean_static_SAP"] = float(order[:5].mean()) if len(order) else 0.0
        # positive patch
        pos_idx = [i for i, v in enumerate(sap) if v > 0]
        if pos_idx:
            C = coords[pos_idx]
            comps = connected_components(list(range(len(pos_idx))), C, PATCH_R)
            out[f"{tag}_largest_positive_patch_size"] = int(max(len(c) for c in comps)) if comps else 0
        else:
            out[f"{tag}_largest_positive_patch_size"] = 0
        cdr_mask = np.array([usable[i]["is_cdr"] for i in range(len(usable))])
        cdr_pos = sap[cdr_mask & (sap > 0)]
        out[f"{tag}_CDR_sum_positive_static_SAP"] = float(cdr_pos.sum()) if len(cdr_pos) else 0.0
        out[f"{tag}_CDR_max_positive_static_SAP"] = float(cdr_pos.max()) if len(cdr_pos) else 0.0
        # store residue SAP for overlap QC via side channel? skip for speed
    return out


def void_features(pdb_path: Path, rows: list[dict]) -> tuple[dict, list[dict]]:
    from pyKVFinder import run_workflow

    res = run_workflow(
        str(pdb_path),
        include_depth=True,
        include_hydropathy=False,
        volume_cutoff=5.0,
        step=0.6,
        probe_in=1.4,
        probe_out=4.0,
        removal_distance=2.4,
        nthreads=1,
    )
    if res is None:
        feat = {
            "internal_cavity_count": 0,
            "total_internal_cavity_volume": 0.0,
            "max_internal_cavity_volume": 0.0,
            "mean_internal_cavity_volume": 0.0,
            "max_internal_cavity_depth": 0.0,
            "mean_internal_cavity_depth": 0.0,
            "internal_cavity_area_total": 0.0,
            "volume_per_residue": 0.0,
            "interface_cavity_volume": 0.0,
            "interface_cavity_count": 0,
            "external_cavity_count": 0,
            "external_cavity_volume_total": 0.0,
            "all_cavity_count": 0,
        }
        return feat, []
    # map pdb chain+resseq -> residue row
    by_pdb = {(r["pdb_chain"], r["pdb_resseq"]): r for r in rows}
    cav_rows = []
    internal = []
    external = []
    for name in res.volume.keys():
        vol = float(res.volume[name])
        area = float(res.area[name])
        avg_d = float(res.avg_depth.get(name, np.nan))
        max_d = float(res.max_depth.get(name, np.nan))
        is_int = bool(np.isfinite(max_d) and np.isfinite(avg_d) and max_d >= 2.0 and avg_d >= 0.5)
        lining = res.residues.get(name, [])
        chains = set()
        mapped = 0
        for item in lining:
            # ['38','A','ARG']
            try:
                rs, ch, _aa = item[0], item[1], item[2]
                key = (ch, int(rs))
                if key in by_pdb:
                    mapped += 1
                    chains.add(by_pdb[key]["chain"])
            except Exception:
                continue
        is_iface = is_int and ("H" in chains and "L" in chains)
        rec = {
            "cavity_id": name,
            "volume": vol,
            "area": area,
            "avg_depth": avg_d,
            "max_depth": max_d,
            "is_internal": is_int,
            "is_interface": is_iface,
            "n_lining_mapped": mapped,
        }
        cav_rows.append(rec)
        (internal if is_int else external).append(rec)

    nres = max(len(rows), 1)
    feat = {
        "internal_cavity_count": len(internal),
        "total_internal_cavity_volume": float(sum(c["volume"] for c in internal)),
        "max_internal_cavity_volume": float(max((c["volume"] for c in internal), default=0.0)),
        "mean_internal_cavity_volume": float(np.mean([c["volume"] for c in internal])) if internal else 0.0,
        "max_internal_cavity_depth": float(max((c["max_depth"] for c in internal), default=0.0)),
        "mean_internal_cavity_depth": float(np.mean([c["avg_depth"] for c in internal])) if internal else 0.0,
        "internal_cavity_area_total": float(sum(c["area"] for c in internal)),
        "volume_per_residue": float(sum(c["volume"] for c in internal) / nres),
        "interface_cavity_volume": float(sum(c["volume"] for c in internal if c["is_interface"])),
        "interface_cavity_count": int(sum(1 for c in internal if c["is_interface"])),
        "external_cavity_count": len(external),
        "external_cavity_volume_total": float(sum(c["volume"] for c in external)),
        "all_cavity_count": int(res.ncav),
    }
    return feat, cav_rows


# Polar atom role helpers on PQR atom names
DONOR_SIDE = {
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
    "TRP": {"NE1"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}
ACC_SIDE = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "HIS": {"ND1", "NE2"},
}


def polar_sat_features(pdb_path: Path, rows: list[dict]) -> tuple[dict, list[dict]]:
    """pdb2pqr AMBER hydrogens + D-A distance/angle on buried polars."""
    with tempfile.TemporaryDirectory(prefix="pqr_") as td:
        td = Path(td)
        pqr = td / "struct.pqr"
        proc = subprocess.run(
            [str(PDB2PQR), "--ff=AMBER", "--keep-chain", str(pdb_path), str(pqr)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0 or not pqr.exists():
            return {"extraction_status": "FAIL", "error": f"pdb2pqr_fail:{proc.stderr[-200:]}"}, []

        # parse PQR atoms grouped by (chain, resseq)
        atoms_by_res: dict[tuple, list] = {}
        for line in pqr.read_text().splitlines():
            if not line.startswith("ATOM"):
                continue
            # PDB-like columns; PQR may use whitespace
            name = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21].strip() or " "
            try:
                resseq = int(line[22:26])
            except Exception:
                parts = line.split()
                if len(parts) < 9:
                    continue
                name, resn, chain, resseq = parts[2], parts[3], parts[4], int(parts[5])
                x, y, z = map(float, parts[6:9])
            else:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            key = (chain, resseq)
            atoms_by_res.setdefault(key, []).append({"name": name, "resn": resn, "xyz": np.array([x, y, z])})

    # Build polar sites from rows + pqr
    sites = []  # dict with xyz_heavy, xyz_h or None, role donor/acceptor, row ref
    for r in rows:
        key = (r["pdb_chain"], r["pdb_resseq"])
        pats = atoms_by_res.get(key, [])
        if not pats:
            continue
        byname = {a["name"]: a for a in pats}
        resn = r["resname3"]
        # backbone N donor if H present
        if "N" in byname and "H" in byname:
            sites.append({"row": r, "role": "donor", "heavy": byname["N"]["xyz"], "h": byname["H"]["xyz"], "atom": "N"})
        if "O" in byname:
            sites.append({"row": r, "role": "acceptor", "heavy": byname["O"]["xyz"], "h": None, "atom": "O"})
        for an in DONOR_SIDE.get(resn, ()):
            if an not in byname:
                continue
            # find bonded H: names starting with H near this atom
            hs = [a for a in pats if a["name"].startswith("H") and np.linalg.norm(a["xyz"] - byname[an]["xyz"]) < 1.3]
            if not hs:
                continue
            sites.append({"row": r, "role": "donor", "heavy": byname[an]["xyz"], "h": hs[0]["xyz"], "atom": an})
        for an in ACC_SIDE.get(resn, ()):
            if an in byname:
                sites.append({"row": r, "role": "acceptor", "heavy": byname[an]["xyz"], "h": None, "atom": an})

    satisfied: set[int] = set()
    for i, si in enumerate(sites):
        if si["role"] != "donor" or si["h"] is None:
            continue
        for j, sj in enumerate(sites):
            if i == j or sj["role"] != "acceptor":
                continue
            if np.linalg.norm(si["heavy"] - sj["heavy"]) > 3.5:
                continue
            v1 = si["heavy"] - si["h"]
            v2 = sj["heavy"] - si["h"]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            ang = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
            if ang >= 120.0:
                satisfied.add(i)
                satisfied.add(j)

    # residue-level buried polar / unsat
    buried_polar_res = set()
    unsat_res = set()
    donor_unsat = acceptor_unsat = 0
    atom_rows = []
    for i, s in enumerate(sites):
        r = s["row"]
        if not r["is_buried"]:
            continue
        buried_polar_res.add((r["chain"], r["sequence_index"]))
        ok = i in satisfied
        if not ok:
            unsat_res.add((r["chain"], r["sequence_index"]))
            if s["role"] == "donor":
                donor_unsat += 1
            else:
                acceptor_unsat += 1
        atom_rows.append(
            {
                "chain": r["chain"],
                "sequence_index": r["sequence_index"],
                "residue_type": r["resname3"],
                "atom": s["atom"],
                "role": s["role"],
                "is_buried": True,
                "is_satisfied": ok,
                "is_cdr": r["is_cdr"],
                "is_framework": r["is_framework"],
                "is_vh_vl_interface": r["is_vh_vl_interface"],
            }
        )

    nres = max(len(rows), 1)
    n_buried_polar = len(buried_polar_res)
    n_unsat = len(unsat_res)

    def count_unsat(pred):
        return sum(1 for ch, si in unsat_res if pred(next(r for r in rows if r["chain"] == ch and r["sequence_index"] == si)))

    feat = {
        "buried_polar_count": n_buried_polar,
        "buried_unsatisfied_polar_count": n_unsat,
        "fraction_buried_polar_unsatisfied": float(n_unsat / n_buried_polar) if n_buried_polar else 0.0,
        "buried_unsatisfied_per_100_res": float(100.0 * n_unsat / nres),
        "VH_buried_unsat_count": sum(1 for ch, _ in unsat_res if ch == "H"),
        "VL_buried_unsat_count": sum(1 for ch, _ in unsat_res if ch == "L"),
        "CDR_buried_unsat_count": count_unsat(lambda r: r["is_cdr"]),
        "framework_buried_unsat_count": count_unsat(lambda r: r["is_framework"]),
        "interface_buried_unsat_count": count_unsat(lambda r: r["is_vh_vl_interface"]),
        "donor_unsat_count": int(donor_unsat),
        "acceptor_unsat_count": int(acceptor_unsat),
        "extraction_status": "SUCCESS",
        "error": "",
    }
    return feat, atom_rows


def process_one(args):
    aid, gen, pdb_path, heavy, light, cdr_rows = args
    pdb_path = Path(pdb_path)
    base = {"id": aid, "generator": gen, "pdb_path": str(pdb_path)}
    out = {fam: dict(base) for fam in SPECS}
    cav_rows = []
    polar_rows = []
    if not pdb_path.exists():
        for fam in out:
            out[fam].update({"extraction_status": "FAIL", "error": "missing_pdb"})
        return out, cav_rows, polar_rows

    rows, err = build_residue_table(pdb_path, heavy, light, cdr_rows)
    if err:
        for fam in out:
            out[fam].update({"extraction_status": "FAIL", "error": err})
        return out, cav_rows, polar_rows

    # AROMATIC
    try:
        feat = aromatic_features(rows)
        out["AROMATIC-TOPO"].update(feat)
        out["AROMATIC-TOPO"]["extraction_status"] = "SUCCESS"
        out["AROMATIC-TOPO"]["error"] = ""
    except Exception as e:
        out["AROMATIC-TOPO"].update({"extraction_status": "FAIL", "error": f"{type(e).__name__}:{e}"})

    # STATIC-SAP
    try:
        feat = static_sap_features(rows)
        out["STATIC-SAP"].update(feat)
        out["STATIC-SAP"]["extraction_status"] = "SUCCESS"
        out["STATIC-SAP"]["error"] = ""
    except Exception as e:
        out["STATIC-SAP"].update({"extraction_status": "FAIL", "error": f"{type(e).__name__}:{e}"})

    # VOID
    try:
        feat, cav_rows = void_features(pdb_path, rows)
        out["VOID-EXPLICIT"].update(feat)
        out["VOID-EXPLICIT"]["extraction_status"] = "SUCCESS"
        out["VOID-EXPLICIT"]["error"] = ""
        for c in cav_rows:
            c["id"] = aid
            c["generator"] = gen
    except Exception as e:
        out["VOID-EXPLICIT"].update({"extraction_status": "FAIL", "error": f"{type(e).__name__}:{e}"})

    # POLAR
    try:
        feat, polar_rows = polar_sat_features(pdb_path, rows)
        out["POLAR-SAT"].update(feat)
        if feat.get("extraction_status") != "FAIL":
            out["POLAR-SAT"]["extraction_status"] = "SUCCESS"
        for p in polar_rows:
            p["id"] = aid
            p["generator"] = gen
    except Exception as e:
        out["POLAR-SAT"].update({"extraction_status": "FAIL", "error": f"{type(e).__name__}:{e}"})

    return out, cav_rows, polar_rows


def main():
    # freeze hash of specs first
    freeze_specs = {
        "all_family_specs_frozen_before_any_target_score": True,
        "states": {k: v["state"] for k, v in SPECS.items()},
        "spec_sha256": {},
    }
    for fam in SPECS:
        p = FP / fam / "FEATURE_SPEC.json"
        freeze_specs["spec_sha256"][fam] = sha256_file(p)
    (FP / "PHYSICAL_BATCH1_SPEC_FREEZE.json").write_text(json.dumps(freeze_specs, indent=2) + "\n")
    print("SPECS FROZEN", freeze_specs["states"])

    cw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    assert len(cw) == 324
    seqs = load_sequences()
    cdr_by_id = load_cdr_map()

    fam_rows = {fam: [] for fam in SPECS}
    all_cav = []
    all_polar = []

    for gen, col in GEN_PATH.items():
        jobs = []
        for _, r in cw.iterrows():
            aid = r["id"]
            jobs.append((aid, gen, str(r[col]), seqs.loc[aid, "heavy"], seqs.loc[aid, "light"], cdr_by_id[aid]))
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(process_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                out, cav, pol = fut.result()
                for fam in SPECS:
                    fam_rows[fam].append(out[fam])
                all_cav.extend(cav)
                all_polar.extend(pol)
                if i % 40 == 0:
                    print(gen, i, flush=True)
        print(gen, "done", flush=True)

    for fam, rows in fam_rows.items():
        famdir = FP / fam
        df = pd.DataFrame(rows)
        # split by generator
        for gen in GEN_PATH:
            sub = df[df.generator == gen].copy()
            sub.to_parquet(famdir / f"features_{gen}.parquet", index=False)
            ok = int((sub.extraction_status == "SUCCESS").sum())
            print(f"{fam} {gen}: {ok}/{len(sub)}")
        df.to_csv(famdir / "extraction_qc.csv", index=False)
        man = df[["id", "generator", "extraction_status", "pdb_path", "error"]].copy() if "error" in df.columns else df[["id", "generator", "extraction_status", "pdb_path"]]
        man.to_csv(famdir / "FEATURE_MANIFEST.csv", index=False)

    if all_cav:
        pd.DataFrame(all_cav).to_parquet(FP / "VOID-EXPLICIT/cavity_level.parquet", index=False)
    if all_polar:
        pd.DataFrame(all_polar).to_parquet(FP / "POLAR-SAT/residue_level.parquet", index=False)
    print("extraction complete")


if __name__ == "__main__":
    main()
