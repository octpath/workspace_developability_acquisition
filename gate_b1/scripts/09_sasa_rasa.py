#!/usr/bin/env python3
"""SASA / RASA / surface physicochemistry / interface features from predicted structures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    AROMATIC,
    CACHE,
    CONFIG,
    DATA,
    HYDROPHOBIC,
    KD,
    MAX_ASA_TIEN2013,
    NEGATIVE,
    POLAR,
    POSITIVE,
    REPORTS,
    ensure_dirs,
    read_json,
    write_json,
)

# Frozen SASA params
PROBE_RADIUS = 1.4
N_POINTS = 100
BIOPYTHON_NOTE = "Bio.PDB.SASA.ShrakeRupley"


def residue_sasa(structure) -> list[dict]:
    sr = ShrakeRupley(probe_radius=PROBE_RADIUS, n_points=N_POINTS)
    sr.compute(structure, level="R")
    rows = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                aa = res.get_resname()
                # convert 3-letter
                from Bio.Data.IUPACData import protein_letters_3to1

                try:
                    aa1 = protein_letters_3to1[aa.capitalize()]
                except Exception:
                    aa1 = "X"
                sasa = float(res.sasa)
                maxasa = MAX_ASA_TIEN2013.get(aa1)
                rasa = (sasa / maxasa) if maxasa else np.nan
                rows.append(
                    {
                        "chain": chain.id,
                        "resseq": res.id[1],
                        "icode": res.id[2],
                        "aa": aa1,
                        "sasa": sasa,
                        "maxasa": maxasa,
                        "rasa": rasa,
                    }
                )
    return rows


def aggregate(res_rows: list[dict], cdr_map: dict | None, prefix: str) -> dict:
    """cdr_map: res key -> region label optional."""
    if not res_rows:
        return {f"{prefix}_empty": 1}
    sasa = np.array([r["sasa"] for r in res_rows], dtype=float)
    rasa = np.array([r["rasa"] for r in res_rows], dtype=float)
    aa = [r["aa"] for r in res_rows]
    feat = {
        f"{prefix}_total_sasa": float(sasa.sum()),
        f"{prefix}_mean_sasa": float(sasa.mean()),
        f"{prefix}_mean_rasa": float(np.nanmean(rasa)),
        f"{prefix}_median_rasa": float(np.nanmedian(rasa)),
        f"{prefix}_max_rasa": float(np.nanmax(rasa)),
        f"{prefix}_frac_rasa_gt_0_25": float(np.nanmean(rasa > 0.25)),
        f"{prefix}_frac_rasa_gt_0_50": float(np.nanmean(rasa > 0.50)),
        f"{prefix}_frac_rasa_gt_1": float(np.nanmean(rasa > 1.0)),
        f"{prefix}_n_res": len(res_rows),
    }
    # chemically grouped SASA
    for name, aset in [
        ("hydrophobic", HYDROPHOBIC),
        ("aromatic", AROMATIC),
        ("positive", POSITIVE),
        ("negative", NEGATIVE),
        ("polar", POLAR),
    ]:
        feat[f"{prefix}_sasa_{name}"] = float(
            sum(r["sasa"] for r in res_rows if r["aa"] in aset)
        )
    # RASA-weighted physicochemistry
    w_h = sum((r["rasa"] if np.isfinite(r["rasa"]) else 0) * KD.get(r["aa"], 0) for r in res_rows)
    w_c = sum(
        (r["rasa"] if np.isfinite(r["rasa"]) else 0)
        * {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}.get(r["aa"], 0)
        for r in res_rows
    )
    feat[f"{prefix}_rasa_weighted_hydrophobicity_sum"] = float(w_h)
    feat[f"{prefix}_rasa_weighted_hydrophobicity_mean"] = float(w_h / max(len(res_rows), 1))
    feat[f"{prefix}_rasa_weighted_charge_sum"] = float(w_c)
    feat[f"{prefix}_rasa_weighted_aromaticity_mean"] = float(
        np.mean([(r["rasa"] if np.isfinite(r["rasa"]) else 0) * (1 if r["aa"] in AROMATIC else 0) for r in res_rows])
    )
    feat[f"{prefix}_exposed_pos_charge"] = float(
        sum((r["rasa"] if np.isfinite(r["rasa"]) else 0) for r in res_rows if r["aa"] in POSITIVE)
    )
    feat[f"{prefix}_exposed_neg_charge"] = float(
        sum((r["rasa"] if np.isfinite(r["rasa"]) else 0) for r in res_rows if r["aa"] in NEGATIVE)
    )
    return feat


def chain_sasa_isolated(pdb_path: Path, chain_id: str) -> float:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    # remove other chains
    model = structure[0]
    for chain in list(model):
        if chain.id != chain_id:
            model.detach_child(chain.id)
    sr = ShrakeRupley(probe_radius=PROBE_RADIUS, n_points=N_POINTS)
    sr.compute(structure, level="R")
    total = 0.0
    for res in model[chain_id]:
        if res.id[0] == " ":
            total += float(res.sasa)
    return total


def process_one(pdb_path: Path, antibody_id: str, num_row: pd.Series | None, prefix: str) -> dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(antibody_id, str(pdb_path))
    res_rows = residue_sasa(structure)
    feat = {"antibody_id": antibody_id}
    feat.update(aggregate(res_rows, None, f"{prefix}_Fv"))
    # by chain — ImmuneBuilder uses H and L
    for ch in sorted({r["chain"] for r in res_rows}):
        label = "VH" if ch.upper() in ("H", "A") else ("VL" if ch.upper() in ("L", "B", "K") else ch)
        sub = [r for r in res_rows if r["chain"] == ch]
        feat.update(aggregate(sub, None, f"{prefix}_{label}"))
    # CDR mapping via IMGT resseq if numbering preserved
    # Author CDR lengths — approximate by IMGT number ranges on structure
    cdr_defs = {
        "H_CDR1": ("H", 27, 38),
        "H_CDR2": ("H", 56, 65),
        "H_CDR3": ("H", 105, 117),
        "L_CDR1": ("L", 27, 38),
        "L_CDR2": ("L", 56, 65),
        "L_CDR3": ("L", 105, 117),
    }
    all_cdr = []
    for name, (ch, lo, hi) in cdr_defs.items():
        sub = [
            r
            for r in res_rows
            if r["chain"].upper() == ch and lo <= r["resseq"] <= hi
        ]
        all_cdr.extend(sub)
        feat.update(aggregate(sub, None, f"{prefix}_{name}"))
    feat.update(aggregate(all_cdr, None, f"{prefix}_all_CDR"))
    fr = [
        r
        for r in res_rows
        if not any(
            r["chain"].upper() == ch and lo <= r["resseq"] <= hi
            for ch, lo, hi in [(c[0], c[1], c[2]) for c in cdr_defs.values()]
        )
    ]
    # simpler FR: not in any CDR range
    fr_rows = []
    for r in res_rows:
        in_cdr = False
        for name, (ch, lo, hi) in cdr_defs.items():
            if r["chain"].upper() == ch and lo <= r["resseq"] <= hi:
                in_cdr = True
                break
        if not in_cdr:
            fr_rows.append(r)
    feat.update(aggregate(fr_rows, None, f"{prefix}_FR"))

    # Interface BSA
    try:
        chains = sorted({r["chain"] for r in res_rows})
        if len(chains) >= 2:
            h_ch = "H" if "H" in chains else chains[0]
            l_ch = "L" if "L" in chains else chains[1]
            sasa_h = chain_sasa_isolated(pdb_path, h_ch)
            sasa_l = chain_sasa_isolated(pdb_path, l_ch)
            sasa_c = feat[f"{prefix}_Fv_total_sasa"]
            bsa = sasa_h + sasa_l - sasa_c
            feat[f"{prefix}_SASA_VH_isolated"] = sasa_h
            feat[f"{prefix}_SASA_VL_isolated"] = sasa_l
            feat[f"{prefix}_BSA"] = bsa
            feat[f"{prefix}_interface_area_BSA_over_2"] = bsa / 2.0
    except Exception as e:
        feat[f"{prefix}_interface_error"] = str(e)

    feat[f"{prefix}_raw_rasa_gt1_frac_Fv"] = feat.get(f"{prefix}_Fv_frac_rasa_gt_1")
    return feat


def main(predictor: str = "abodybuilder2"):
    ensure_dirs()
    if predictor == "abodybuilder2":
        manifest = pd.read_csv(CACHE / "structures" / "abb2_manifest.csv")
        prefix = "ABB"
        pdb_key = "pdb_path"
    else:
        manifest = pd.read_csv(CACHE / "structures" / "esmfold_manifest.csv")
        prefix = "ESMF"
        pdb_key = "pdb_path"

    num = pd.read_csv(DATA / "numbering_germline.csv")
    rows = []
    rasa_gt1 = []
    for _, m in manifest.iterrows():
        if not m.get("success", True):
            continue
        pdb = Path(m[pdb_key])
        if not pdb.exists():
            continue
        feat = process_one(pdb, m["antibody_id"], None, prefix)
        rows.append(feat)
        rasa_gt1.append(feat.get(f"{prefix}_Fv_frac_rasa_gt_1", np.nan))
        if len(rows) % 50 == 0:
            print("SASA progress", len(rows), flush=True)

    out = pd.DataFrame(rows)
    out_path = CACHE / "structure_features" / f"{predictor}_sasa_rasa.csv"
    out.to_csv(out_path, index=False)

    audit = {
        "predictor": predictor,
        "biopython_sasa": BIOPYTHON_NOTE,
        "probe_radius": PROBE_RADIUS,
        "n_points": N_POINTS,
        "maxasa_scale": "Tien2013/Wilke",
        "maxasa_values": MAX_ASA_TIEN2013,
        "n_structures": len(out),
        "mean_frac_rasa_gt_1": float(np.nanmean(rasa_gt1)) if rasa_gt1 else None,
        "max_rasa_observed": float(out[f"{prefix}_Fv_max_rasa"].max()) if len(out) else None,
        "note": "RASA values >1 are preserved (not clipped).",
    }
    write_json(CACHE / "structure_features" / f"{predictor}_sasa_audit.json", audit)
    md_path = REPORTS / "sasa_rasa_feature_audit.md"
    prev = md_path.read_text() if md_path.exists() else "# SASA / RASA feature audit\n\n"
    prev += f"\n## {predictor}\n\n"
    for k, v in audit.items():
        if k == "maxasa_values":
            continue
        prev += f"- {k}: {v}\n"
    md_path.write_text(prev)

    # registry
    reg = read_json(CONFIG / "representation_registry.json")
    reg[f"STR_{prefix}_SASA"] = {
        "file": str(out_path.name),
        "cols_prefix": [f"{prefix}_"],
        "include_substrings": ["_sasa", "_total_sasa", "_mean_sasa"],
        "exclude_substrings": ["_rasa", "weighted", "BSA", "interface"],
        "class": "PARTICIPANT_LEGAL",
        "feature_file": str(out_path),
    }
    reg[f"STR_{prefix}_SASA_RASA"] = {
        "feature_file": str(out_path),
        "cols_prefix": [f"{prefix}_"],
        "include_substrings": ["sasa", "rasa"],
        "exclude_substrings": ["weighted", "BSA"],
        "class": "PARTICIPANT_LEGAL",
    }
    reg[f"STR_{prefix}_SURFACE_PHYS"] = {
        "feature_file": str(out_path),
        "cols_prefix": [f"{prefix}_"],
        "include_substrings": ["rasa_weighted", "exposed_", "sasa_hydrophobic", "sasa_aromatic"],
        "class": "PARTICIPANT_LEGAL",
    }
    reg[f"STR_{prefix}_INTERFACE"] = {
        "feature_file": str(out_path),
        "cols_prefix": [f"{prefix}_"],
        "include_substrings": ["BSA", "interface", "isolated"],
        "class": "PARTICIPANT_LEGAL",
    }
    write_json(CONFIG / "representation_registry.json", reg)
    print("SASA_OK", predictor, len(out), audit.get("mean_frac_rasa_gt_1"))


if __name__ == "__main__":
    pred = sys.argv[1] if len(sys.argv) > 1 else "abodybuilder2"
    main(pred)
