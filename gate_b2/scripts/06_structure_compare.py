#!/usr/bin/env python3
"""Lightweight ABB vs native ESMFold vs B1 HF-linker structure comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import B1_CACHE, CACHE, DATA, REPORTS, ensure_dirs, write_json  # noqa: E402


def ca_atoms(structure, chain_ids=None):
    atoms = []
    for model in structure:
        for chain in model:
            if chain_ids and chain.id not in chain_ids:
                continue
            for res in chain:
                if res.id[0] != " " or not is_aa(res, standard=True):
                    continue
                if "CA" in res:
                    atoms.append(res["CA"])
    return atoms


def chain_com(structure, chain_id):
    coords = []
    for res in structure[0][chain_id]:
        if res.id[0] != " " or "CA" not in res:
            continue
        coords.append(res["CA"].coord)
    if not coords:
        return None
    return np.mean(coords, axis=0)


def map_chains(st, vh_len, vl_len):
    chains = list(st[0])
    if len(chains) < 2:
        return None, None
    lens = {c.id: sum(1 for r in c if r.id[0] == " ") for c in chains}
    ids = list(lens)
    if "H" in lens and "L" in lens:
        return "H", "L"
    if vh_len and vl_len:
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                if lens[a] == vh_len and lens[b] == vl_len:
                    return a, b
    ordered = sorted(ids, key=lambda c: -lens[c])
    return ordered[0], ordered[1]


def rmsd_align(ref_atoms, mob_atoms):
    n = min(len(ref_atoms), len(mob_atoms))
    if n < 10:
        return np.nan
    ref = ref_atoms[:n]
    mob = mob_atoms[:n]
    sup = Superimposer()
    sup.set_atoms(ref, mob)
    return float(sup.rms)


def main():
    ensure_dirs()
    abb_man = pd.read_csv(CACHE / "structures" / "abb2_manifest.csv")
    esm_path = CACHE / "structures" / "esmfold_native_manifest.csv"
    b1_path = B1_CACHE / "structures" / "esmfold_manifest.csv"
    if not esm_path.exists():
        print("WAIT_ESMFOLD")
        return
    esm_man = pd.read_csv(esm_path)
    b1_man = pd.read_csv(b1_path) if b1_path.exists() else None

    union = pd.read_csv(DATA / "hic_tmapp_union.csv")
    lens = {r.antibody_id: (len(r.heavy), len(r.light)) for r in union.itertuples()}

    parser = PDBParser(QUIET=True)
    rows = []
    abb_ok = abb_man if "success" not in abb_man.columns else abb_man[abb_man["success"] == True]
    esm_ok = esm_man[esm_man["success"] == True] if "success" in esm_man.columns else esm_man
    common = set(abb_ok["antibody_id"]) & set(esm_ok["antibody_id"])
    # sample up to 80 for RMSD (expensive-ish) but compute COM for all
    for aid in sorted(common):
        arow = abb_man[abb_man.antibody_id == aid].iloc[0]
        erow = esm_man[esm_man.antibody_id == aid].iloc[0]
        ap = Path(arow["pdb_path"])
        if not ap.exists():
            ap = CACHE / "structures" / "abodybuilder2" / f"{arow['pair_hash']}.pdb"
        ep = Path(erow["pdb_path"])
        if not ap.exists() or not ep.exists():
            continue
        vh_l, vl_l = lens.get(aid, (None, None))
        try:
            sa = parser.get_structure("a", str(ap))
            se = parser.get_structure("e", str(ep))
            ah, al = map_chains(sa, vh_l, vl_l)
            eh, el = map_chains(se, vh_l, vl_l)
            com_a = np.linalg.norm(chain_com(sa, ah) - chain_com(sa, al))
            com_e = np.linalg.norm(chain_com(se, eh) - chain_com(se, el))
            rec = {
                "antibody_id": aid,
                "abb_com_dist": float(com_a),
                "esmn_com_dist": float(com_e),
                "com_delta": float(com_e - com_a),
                "esmn_mean_plddt": erow.get("mean_plddt"),
            }
            # RMSD on subset
            if len(rows) < 100:
                ra = ca_atoms(sa)
                re = ca_atoms(se)
                # align by sequential CA count
                rec["fv_ca_rmsd_abb_vs_esmn"] = rmsd_align(ra, re)
                rec["vh_ca_rmsd"] = rmsd_align(ca_atoms(sa, {ah}), ca_atoms(se, {eh}))
                rec["vl_ca_rmsd"] = rmsd_align(ca_atoms(sa, {al}), ca_atoms(se, {el}))
            # B1 HF linker if present
            if b1_man is not None and aid in set(b1_man.antibody_id):
                brow = b1_man[b1_man.antibody_id == aid].iloc[0]
                bp = Path(brow.get("pdb_path", ""))
                if not bp.exists() and "pair_hash" in brow:
                    bp = B1_CACHE / "structures" / "esmfold" / f"{brow['pair_hash']}.pdb"
                if bp.exists():
                    sb = parser.get_structure("b", str(bp))
                    bh, bl = map_chains(sb, vh_l, vl_l)
                    com_b = np.linalg.norm(chain_com(sb, bh) - chain_com(sb, bl))
                    rec["b1hf_com_dist"] = float(com_b)
                    if len(rows) < 100:
                        rec["fv_ca_rmsd_b1hf_vs_esmn"] = rmsd_align(ca_atoms(sb), ca_atoms(se))
            rows.append(rec)
        except Exception as e:
            rows.append({"antibody_id": aid, "error": str(e)})
        if len(rows) % 50 == 0:
            print("CMP", len(rows), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "structure_features" / "structure_comparison.csv", index=False)
    ok = out.dropna(subset=["abb_com_dist", "esmn_com_dist"]) if "abb_com_dist" in out.columns else out
    lines = [
        "# Structure comparison — ABB vs native ESMFold (and B1 HF/linker)",
        "",
        f"- Compared antibodies: {len(ok)}",
        f"- ABB COM dist mean±sd: {ok['abb_com_dist'].mean():.2f} ± {ok['abb_com_dist'].std():.2f} Å",
        f"- Native ESMFold COM dist mean±sd: {ok['esmn_com_dist'].mean():.2f} ± {ok['esmn_com_dist'].std():.2f} Å",
        f"- ΔCOM (ESMN−ABB) mean: {ok['com_delta'].mean():.2f} Å",
    ]
    if "fv_ca_rmsd_abb_vs_esmn" in ok.columns:
        r = ok["fv_ca_rmsd_abb_vs_esmn"].dropna()
        lines.append(f"- Fv CA RMSD ABB vs ESMN (n={len(r)}): mean {r.mean():.2f} Å")
    if "fv_ca_rmsd_b1hf_vs_esmn" in ok.columns:
        r = ok["fv_ca_rmsd_b1hf_vs_esmn"].dropna()
        lines.append(f"- Fv CA RMSD B1-HF/linker vs native ESMN (n={len(r)}): mean {r.mean():.2f} Å")
    lines += [
        "",
        "## Interpretation",
        "",
        "Purpose is not structure benchmarking; ask whether geometry differences are large enough",
        "to change downstream developability features. See surface/feature ablations for predictive impact.",
        "",
        "ESMFold supports multimer inputs through colon-separated chains and chain-aware inference",
        "machinery; it is not a true AlphaFold-Multimer model.",
        "",
    ]
    # merge feature deltas if available
    abb_f = CACHE / "structure_features" / "abb_sasa_rasa_patch.csv"
    esm_f = CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv"
    if abb_f.exists() and esm_f.exists():
        a = pd.read_csv(abb_f)
        e = pd.read_csv(esm_f)
        m = a.merge(e, on="antibody_id", suffixes=("_abb", "_esmn"))
        for col in ["Fv_total_sasa", "BSA", "largest_hydrophobic_patch_sasa", "Fv_mean_rasa"]:
            ca, ce = f"ABB_{col}", f"ESMFN_{col}"
            # actual column names
        for pair in [
            ("ABB_Fv_total_sasa", "ESMFN_Fv_total_sasa"),
            ("ABB_BSA", "ESMFN_BSA"),
            ("ABB_largest_hydrophobic_patch_sasa", "ESMFN_largest_hydrophobic_patch_sasa"),
            ("ABB_Fv_mean_rasa", "ESMFN_Fv_mean_rasa"),
        ]:
            if pair[0] in m.columns and pair[1] in m.columns:
                d = (m[pair[1]] - m[pair[0]]).abs()
                lines.append(f"- |Δ| {pair[0].replace('ABB_','')} mean={d.mean():.2f} spearman={m[pair[0]].corr(m[pair[1]], method='spearman'):.3f}")
    (REPORTS / "structure_comparison.md").write_text("\n".join(lines) + "\n")
    write_json(CACHE / "structure_features" / "structure_comparison_summary.json", {"n": len(ok)})
    print("STRUCTURE_COMPARE_OK", len(ok))


if __name__ == "__main__":
    main()
