#!/usr/bin/env python3
"""ANARCI germline gene assignment + germline distance vs assigned V allele AA."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from anarci import anarci, all_germlines, run_germline_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import DATA, REPORTS, ensure_dirs, write_json  # noqa: E402


def family_from_gene(g: str | None) -> str | None:
    if g is None or (isinstance(g, float) and pd.isna(g)):
        return None
    s = str(g).upper()
    m = re.match(r"(IGHV|IGKV|IGLV|VH|VK|VL)(\d+)", s)
    if not m:
        return None
    pref = {"IGHV": "VH", "IGKV": "VK", "IGLV": "VL", "VH": "VH", "VK": "VK", "VL": "VL"}[m.group(1)]
    return f"{pref}{m.group(2)}"


def assign(seq: str, allow: set):
    n, d, h = anarci([("q", seq)], scheme="imgt", output=False, allow=allow, allowed_species=["human"])
    if not n or not n[0]:
        return {}
    numbered = n[0][0][0]
    chain = d[0][0]["chain_type"]
    state_vector = []
    idx = 0
    for (pos, ins), aa in numbered:
        if aa == "-":
            continue
        key = (pos, "m") if ins == " " else (pos, ins)
        state_vector.append((key, idx))
        idx += 1
    genes = run_germline_assignment(state_vector, seq, chain, allowed_species=["human"])
    v = genes.get("v_gene", [None, None])
    j = genes.get("j_gene", [None, None])
    v_gene = v[0][1] if v and v[0] else None
    v_ident = v[1] if v and v[0] else None
    j_gene = j[0][1] if j and j[0] else None
    j_ident = j[1] if j and j[0] else None
    species = v[0][0] if v and v[0] else None
    # germline AA for distance excluding CDR3 (IMGT 105-117)
    gl_seq = None
    if species and v_gene and chain in all_germlines["V"] and species in all_germlines["V"][chain]:
        gl_seq = all_germlines["V"][chain][species].get(v_gene)
    # Build query IMGT state sequence same as ANARCI
    state_dict = {(i, "m"): None for i in range(1, 129)}
    state_dict.update(dict(state_vector))
    # mutation fraction on FR+CDR1/2 positions (1-104), not CDR3
    mut_sites = mismatches = covered = 0
    if gl_seq:
        for i in range(1, 105):  # through FR3 end before CDR3
            qi = state_dict.get((i, "m"))
            ga = gl_seq[i - 1] if i - 1 < len(gl_seq) else "-"
            if ga == "-" and qi is None:
                continue
            if qi is None or ga == "-":
                continue
            covered += 1
            if seq[qi] != ga:
                mismatches += 1
        mut_frac = mismatches / covered if covered else None
    else:
        mut_frac = None
    return {
        "v_gene": v_gene,
        "v_identity": v_ident,
        "j_gene": j_gene,
        "j_identity": j_ident,
        "chain_type": chain,
        "species": species,
        "germline_distance": None if v_ident is None else 1.0 - float(v_ident),
        "mutation_fraction_V_aligned_excl_CDR3": mut_frac,
        "bitscore": d[0][0].get("bitscore"),
    }


def main():
    ensure_dirs()
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    num = pd.read_csv(DATA / "numbering_germline.csv")
    rows = []
    for _, r in df.iterrows():
        h = assign(r["heavy"], {"H"})
        l = assign(r["light"], {"L", "K"})
        rows.append(
            {
                "antibody_id": r["antibody_id"],
                "PL_anarci_vh_v_gene": h.get("v_gene"),
                "PL_anarci_vh_j_gene": h.get("j_gene"),
                "PL_anarci_vh_v_identity": h.get("v_identity"),
                "PL_anarci_vh_germline_distance": h.get("germline_distance"),
                "PL_anarci_vh_mutfrac_excl_CDR3": h.get("mutation_fraction_V_aligned_excl_CDR3"),
                "PL_anarci_vl_v_gene": l.get("v_gene"),
                "PL_anarci_vl_j_gene": l.get("j_gene"),
                "PL_anarci_vl_v_identity": l.get("v_identity"),
                "PL_anarci_vl_germline_distance": l.get("germline_distance"),
                "PL_anarci_vl_mutfrac_excl_CDR3": l.get("mutation_fraction_V_aligned_excl_CDR3"),
                "PL_anarci_vh_family": family_from_gene(h.get("v_gene")),
                "PL_anarci_vl_family": family_from_gene(l.get("v_gene")),
                "PL_anarci_kappa_lambda": "kappa" if l.get("chain_type") == "K" else ("lambda" if l.get("chain_type") == "L" else None),
                "PL_anarci_H_ok": bool(h.get("v_gene")),
                "PL_anarci_L_ok": bool(l.get("v_gene")),
            }
        )
        if len(rows) % 50 == 0:
            print("progress", len(rows), flush=True)
    ann = pd.DataFrame(rows)
    # drop old anarci cols
    drop_cols = [c for c in num.columns if c.startswith("PL_anarci_") or c in [
        "PL_vh_v_gene", "PL_vl_v_gene", "PL_vh_family", "PL_vl_family", "PL_kappa_lambda",
        "PL_vh_family_filled", "PL_vl_family_filled", "PL_family_source",
        "PL_vh_germline_distance", "PL_vl_germline_distance", "PL_combined_germline_distance",
        "PL_vh_fr_germline_distance", "PL_vh_cdr12_germline_distance",
    ]]
    out = num.drop(columns=drop_cols, errors="ignore").merge(ann, on="antibody_id")
    out["PL_vh_v_gene"] = out["PL_anarci_vh_v_gene"]
    out["PL_vl_v_gene"] = out["PL_anarci_vl_v_gene"]
    out["PL_vh_family"] = out["PL_anarci_vh_family"]
    out["PL_vl_family"] = out["PL_anarci_vl_family"]
    out["PL_kappa_lambda"] = out["PL_anarci_kappa_lambda"]
    out["PL_vh_germline_distance"] = out["PL_anarci_vh_germline_distance"]
    out["PL_vl_germline_distance"] = out["PL_anarci_vl_germline_distance"]
    out["PL_combined_germline_distance"] = 0.5 * (
        out["PL_vh_germline_distance"].astype(float) + out["PL_vl_germline_distance"].astype(float)
    )
    out["PL_vh_mutfrac_excl_CDR3"] = out["PL_anarci_vh_mutfrac_excl_CDR3"]
    out["PL_vl_mutfrac_excl_CDR3"] = out["PL_anarci_vl_mutfrac_excl_CDR3"]
    out["PL_vh_family_filled"] = out["PL_vh_family"].fillna(out["ORG_author_vh_family"])
    out["PL_vl_family_filled"] = out["PL_vl_family"].fillna(out["ORG_author_vl_family"])
    out["PL_family_source"] = np.where(
        out["PL_vh_family"].notna() & out["PL_vl_family"].notna(), "anarci", "author_fallback"
    )
    # author agreement at family level
    agree_h = float((out["PL_vh_family"] == out["ORG_author_vh_family"]).mean())
    agree_l = float((out["PL_vl_family"] == out["ORG_author_vl_family"]).mean())
    out.to_csv(DATA / "numbering_germline.csv", index=False)
    audit = {
        "H_ok": float(out["PL_anarci_H_ok"].mean()),
        "L_ok": float(out["PL_anarci_L_ok"].mean()),
        "family_source": out["PL_family_source"].value_counts().to_dict(),
        "family_agree_author_H": agree_h,
        "family_agree_author_L": agree_l,
        "mean_vh_germline_distance": float(out["PL_vh_germline_distance"].mean()),
        "mean_vl_germline_distance": float(out["PL_vl_germline_distance"].mean()),
    }
    write_json(DATA / "anarci_germline_audit.json", audit)
    with open(REPORTS / "numbering_and_germline_audit.md", "a") as f:
        f.write("\n## ANARCI allele-level germline assignment\n\n")
        for k, v in audit.items():
            f.write(f"- {k}: {v}\n")
        f.write(
            "\nGermline distance = 1 - ANARCI V-allele identity. "
            "Mutation fraction uses IMGT positions 1–104 only (CDR3 excluded).\n"
        )
    print("GERMLINE_OK", audit)


if __name__ == "__main__":
    main()
