#!/usr/bin/env python3
"""Stage A/B/C feature extraction: simple, CDR, biological shortcut, organizer oracles."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    AA20,
    AROMATIC,
    CACHE,
    DATA,
    HYDROPHOBIC,
    KD,
    NEGATIVE,
    POLAR,
    POSITIVE,
    ensure_dirs,
    write_json,
)


def safe_protparam(seq: str) -> dict:
    seq = "".join(c for c in seq if c in AA20)
    if len(seq) < 1:
        return {
            "length": 0,
            "mw": np.nan,
            "pI": np.nan,
            "gravy": np.nan,
            "aromaticity": np.nan,
            "instability": np.nan,
            "charge_ph7": np.nan,
            "frac_polar": np.nan,
            "frac_hydrophobic": np.nan,
            "frac_acidic": np.nan,
            "frac_basic": np.nan,
            "frac_aromatic": np.nan,
        }
    pa = ProteinAnalysis(seq)
    # net charge approx at pH7
    charge = sum({"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}.get(a, 0) for a in seq)
    n = len(seq)
    return {
        "length": n,
        "mw": float(pa.molecular_weight()),
        "pI": float(pa.isoelectric_point()),
        "gravy": float(pa.gravy()),
        "aromaticity": float(pa.aromaticity()),
        "instability": float(pa.instability_index()),
        "charge_ph7": float(charge),
        "frac_polar": sum(a in POLAR for a in seq) / n,
        "frac_hydrophobic": sum(a in HYDROPHOBIC for a in seq) / n,
        "frac_acidic": sum(a in NEGATIVE for a in seq) / n,
        "frac_basic": sum(a in POSITIVE for a in seq) / n,
        "frac_aromatic": sum(a in AROMATIC for a in seq) / n,
    }


def aa_comp(seq: str, prefix: str) -> dict:
    seq = "".join(c for c in seq if c in AA20)
    n = max(len(seq), 1)
    return {f"{prefix}_aa_{a}": seq.count(a) / n for a in AA20}


def region_feats(seq: str, prefix: str) -> dict:
    seq = "".join(c for c in str(seq) if c in AA20)
    n = max(len(seq), 1)
    charge = sum({"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}.get(a, 0) for a in seq)
    gravy = sum(KD.get(a, 0) for a in seq) / n
    out = {
        f"{prefix}_len": len(seq),
        f"{prefix}_charge": charge,
        f"{prefix}_gravy": gravy,
        f"{prefix}_aromaticity": sum(a in AROMATIC for a in seq) / n,
        f"{prefix}_frac_polar": sum(a in POLAR for a in seq) / n,
        f"{prefix}_frac_hydrophobic": sum(a in HYDROPHOBIC for a in seq) / n,
    }
    out.update(aa_comp(seq, prefix))
    return out


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    num = pd.read_csv(DATA / "numbering_germline.csv")
    m = df.merge(num, on="antibody_id")

    rows_a = []
    rows_b = []
    rows_c = []
    rows_org = []

    for _, r in m.iterrows():
        h, l = r["heavy"], r["light"]
        hl = h + l
        # A0/A1/A2
        feat = {"antibody_id": r["antibody_id"]}
        feat["A0_vh_len"] = len(h)
        feat["A0_vl_len"] = len(l)
        feat["A0_total_len"] = len(hl)
        feat.update(aa_comp(h, "A1_H"))
        feat.update(aa_comp(l, "A1_L"))
        feat.update(aa_comp(hl, "A1_HL"))
        for prefix, seq in [("A2_H", h), ("A2_L", l), ("A2_HL", hl)]:
            pp = safe_protparam(seq)
            for k, v in pp.items():
                feat[f"{prefix}_{k}"] = v
        rows_a.append(feat)

        # Stage B CDR-aware
        b = {"antibody_id": r["antibody_id"]}
        b.update(region_feats(h, "B_H_full"))
        b.update(region_feats(l, "B_L_full"))
        for chain in ["H", "L"]:
            for reg in ["CDR1", "CDR2", "CDR3"]:
                b.update(region_feats(r[f"{chain}_{reg}"], f"B_{chain}_{reg}"))
        cdr_all = "".join(
            [
                r["H_CDR1"],
                r["H_CDR2"],
                r["H_CDR3"],
                r["L_CDR1"],
                r["L_CDR2"],
                r["L_CDR3"],
            ]
        )
        fr_all = "".join(
            [
                r["H_FR1"],
                r["H_FR2"],
                r["H_FR3"],
                r["H_FR4"],
                r["L_FR1"],
                r["L_FR2"],
                r["L_FR3"],
                r["L_FR4"],
            ]
        )
        b.update(region_feats(cdr_all, "B_all_CDR"))
        b.update(region_feats(fr_all, "B_all_FR"))
        rows_b.append(b)

        # Stage C BIO_SHORTCUT participant-legal
        c = {"antibody_id": r["antibody_id"]}
        c["C_vh_family"] = r.get("PL_vh_family_filled")
        c["C_vl_family"] = r.get("PL_vl_family_filled")
        c["C_family_pair"] = f"{c['C_vh_family']}|{c['C_vl_family']}"
        c["C_kappa_lambda"] = r.get("ORG_kappa_lambda") or r.get("PL_kappa_lambda")
        for col in [
            "PL_vh_germline_distance",
            "PL_vl_germline_distance",
            "PL_combined_germline_distance",
            "PL_vh_fr_germline_distance",
            "PL_vh_cdr12_germline_distance",
            "PL_H_CDR1_len",
            "PL_H_CDR2_len",
            "PL_H_CDR3_len",
            "PL_L_CDR1_len",
            "PL_L_CDR2_len",
            "PL_L_CDR3_len",
        ]:
            c[col.replace("PL_", "C_")] = r.get(col)
        rows_c.append(c)

        # Organizer-only oracles ILLEGAL_FOR_PARTICIPANTS
        o = {
            "antibody_id": r["antibody_id"],
            "ILLEGAL_b_cell_subset": r["b_cell_subset"],
            "ILLEGAL_donor_proxy": None,  # no donor column in mmc2
            "ILLEGAL_author_vh_germline": r.get("ORG_author_vh_germline") or r.get("vh_germline"),
            "ILLEGAL_author_vl_germline": r.get("ORG_author_vl_germline") or r.get("vl_germline"),
        }
        rows_org.append(o)

    fa = pd.DataFrame(rows_a)
    fb = pd.DataFrame(rows_b)
    fc = pd.DataFrame(rows_c)
    fo = pd.DataFrame(rows_org)
    fa.to_csv(CACHE / "features" / "stage_A_simple.csv", index=False)
    fb.to_csv(CACHE / "features" / "stage_B_cdr.csv", index=False)
    fc.to_csv(CACHE / "features" / "stage_C_shortcut.csv", index=False)
    fo.to_csv(CACHE / "features" / "stage_ORG_illegal.csv", index=False)

    # Representation registry
    registry = {
        "A0_length": {
            "cols": ["A0_vh_len", "A0_vl_len", "A0_total_len"],
            "file": "stage_A_simple.csv",
            "class": "PARTICIPANT_LEGAL",
        },
        "A1_aa_comp": {
            "cols_prefix": ["A1_"],
            "file": "stage_A_simple.csv",
            "class": "PARTICIPANT_LEGAL",
        },
        "A2_physchem": {
            "cols_prefix": ["A2_"],
            "file": "stage_A_simple.csv",
            "class": "PARTICIPANT_LEGAL",
        },
        "B_cdr_descriptors": {
            "cols_prefix": ["B_"],
            "file": "stage_B_cdr.csv",
            "class": "PARTICIPANT_LEGAL",
        },
        "B_cdr_hydrophobicity_only": {
            "cols": [
                "B_H_CDR1_gravy",
                "B_H_CDR2_gravy",
                "B_H_CDR3_gravy",
                "B_L_CDR1_gravy",
                "B_L_CDR2_gravy",
                "B_L_CDR3_gravy",
                "B_all_CDR_gravy",
                "B_H_full_gravy",
                "B_L_full_gravy",
            ],
            "file": "stage_B_cdr.csv",
            "class": "PARTICIPANT_LEGAL",
            "diagnostic": "HIC_SIMPLE_HYDROPHOBICITY_RISK",
        },
        "C_BIO_SHORTCUT": {
            "cols_prefix": ["C_"],
            "file": "stage_C_shortcut.csv",
            "class": "PARTICIPANT_LEGAL",
            "categorical": ["C_vh_family", "C_vl_family", "C_family_pair", "C_kappa_lambda"],
        },
        "ORG_subset_only": {
            "cols": ["ILLEGAL_b_cell_subset"],
            "file": "stage_ORG_illegal.csv",
            "class": "ILLEGAL_FOR_PARTICIPANTS",
            "categorical": ["ILLEGAL_b_cell_subset"],
        },
        "ORG_germline_author": {
            "cols": ["ILLEGAL_author_vh_germline", "ILLEGAL_author_vl_germline"],
            "file": "stage_ORG_illegal.csv",
            "class": "ILLEGAL_FOR_PARTICIPANTS",
            "categorical": [
                "ILLEGAL_author_vh_germline",
                "ILLEGAL_author_vl_germline",
            ],
        },
        "ORG_subset_plus_germline": {
            "cols": [
                "ILLEGAL_b_cell_subset",
                "ILLEGAL_author_vh_germline",
                "ILLEGAL_author_vl_germline",
            ],
            "file": "stage_ORG_illegal.csv",
            "class": "ILLEGAL_FOR_PARTICIPANTS",
            "categorical": [
                "ILLEGAL_b_cell_subset",
                "ILLEGAL_author_vh_germline",
                "ILLEGAL_author_vl_germline",
            ],
        },
    }
    write_json(Path(__file__).resolve().parents[1] / "config" / "representation_registry.json", registry)
    print("FEATURES_OK", len(fa), len(fb.columns), len(fc.columns))


if __name__ == "__main__":
    main()
