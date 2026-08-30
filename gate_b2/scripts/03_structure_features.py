#!/usr/bin/env python3
"""SASA/RASA/surface physchem/patches/interface for ABB (reuse B1) + native ESMFold."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.SASA import ShrakeRupley
from Bio.Data.IUPACData import protein_letters_3to1

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import (  # noqa: E402
    AROMATIC,
    B1_CACHE,
    CACHE,
    DATA,
    HYDROPHOBIC,
    KD,
    MAX_ASA_TIEN2013,
    NEGATIVE,
    N_POINTS,
    PATCH_CONTACT_A,
    POLAR,
    POSITIVE,
    PROBE_RADIUS,
    RASA_SURFACE_THRESHOLD,
    REPORTS,
    ensure_dirs,
    pair_hash,
    write_json,
)
import Bio

CDR_DEFS = {
    "H_CDR1": ("H", 27, 38),
    "H_CDR2": ("H", 56, 65),
    "H_CDR3": ("H", 105, 117),
    "L_CDR1": ("L", 27, 38),
    "L_CDR2": ("L", 56, 65),
    "L_CDR3": ("L", 105, 117),
}


def aa1(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def residue_rows(structure):
    sr = ShrakeRupley(probe_radius=PROBE_RADIUS, n_points=N_POINTS)
    sr.compute(structure, level="R")
    rows = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                a = aa1(res.get_resname())
                sasa = float(res.sasa)
                maxasa = MAX_ASA_TIEN2013.get(a)
                rasa = sasa / maxasa if maxasa else np.nan
                # CA coord
                ca = res["CA"].coord if "CA" in res else None
                rows.append(
                    {
                        "chain": chain.id,
                        "resseq": res.id[1],
                        "aa": a,
                        "sasa": sasa,
                        "rasa": rasa,
                        "ca": ca.copy() if ca is not None else None,
                    }
                )
    return rows


REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]


def normalize_chain_labels(rows, vh_len=None, vl_len=None):
    """Map A/B or H/L to H/L by length if needed. Keep pdb_chain for file I/O."""
    for r in rows:
        r.setdefault("pdb_chain", r["chain"])
    chains = sorted({r["chain"] for r in rows})
    mapping = {}
    if "H" in chains and "L" in chains:
        return rows
    if len(chains) >= 2:
        lens = {c: sum(1 for r in rows if r["chain"] == c) for c in chains}
        c0, c1 = chains[0], chains[1]
        if vh_len and vl_len:
            if lens[c0] == vh_len and lens[c1] == vl_len:
                mapping = {c0: "H", c1: "L"}
            elif lens[c1] == vh_len and lens[c0] == vl_len:
                mapping = {c1: "H", c0: "L"}
        if not mapping:
            ordered = sorted(chains, key=lambda c: -lens[c])
            mapping = {ordered[0]: "H", ordered[1]: "L"}
        for r in rows:
            r["chain"] = mapping.get(r["chain"], r["chain"])
    return rows


def looks_imgt_numbered(rows, chain: str) -> bool:
    ch = [r for r in rows if r["chain"] == chain]
    if not ch:
        return False
    return max(r["resseq"] for r in ch) >= 105


def tag_regions(rows, numbering_row=None):
    """Attach region labels: IMGT resseq for ABB; author FR/CDR lengths for sequential ESMFold."""
    for r in rows:
        r["region"] = "UNK"
        r["is_cdr"] = False
    if numbering_row is None:
        # IMGT fallback
        for r in rows:
            for name, (ch, lo, hi) in CDR_DEFS.items():
                if r["chain"] == ch and lo <= r["resseq"] <= hi:
                    r["region"] = name.split("_", 1)[1]  # CDR1..
                    r["is_cdr"] = True
        return rows

    for chain, prefix in [("H", "H"), ("L", "L")]:
        ch_rows = sorted([r for r in rows if r["chain"] == chain], key=lambda r: (r["resseq"], r["aa"]))
        if not ch_rows:
            continue
        if looks_imgt_numbered(rows, chain):
            for r in ch_rows:
                for name, (ch, lo, hi) in CDR_DEFS.items():
                    if ch == chain and lo <= r["resseq"] <= hi:
                        r["region"] = name.split("_", 1)[1]
                        r["is_cdr"] = True
            continue
        # Sequential: slice by frozen author region lengths
        regions = {k: str(numbering_row.get(f"{prefix}_{k}") or "") for k in REGION_ORDER}
        expected = sum(len(regions[k]) for k in REGION_ORDER)
        if expected == 0:
            continue
        # tolerate small length mismatch by truncating/padding UNK
        idx = 0
        for name in REGION_ORDER:
            for _ in regions[name]:
                if idx >= len(ch_rows):
                    break
                ch_rows[idx]["region"] = name
                ch_rows[idx]["is_cdr"] = name.startswith("CDR")
                idx += 1
    return rows

def aggregate(rows, prefix):
    if not rows:
        return {f"{prefix}_empty": 1}
    sasa = np.array([r["sasa"] for r in rows])
    rasa = np.array([r["rasa"] for r in rows], float)
    feat = {
        f"{prefix}_total_sasa": float(sasa.sum()),
        f"{prefix}_mean_sasa": float(sasa.mean()),
        f"{prefix}_mean_rasa": float(np.nanmean(rasa)),
        f"{prefix}_median_rasa": float(np.nanmedian(rasa)),
        f"{prefix}_max_rasa": float(np.nanmax(rasa)),
        f"{prefix}_frac_rasa_gt_0_20": float(np.nanmean(rasa >= 0.20)),
        f"{prefix}_frac_rasa_gt_0_25": float(np.nanmean(rasa > 0.25)),
        f"{prefix}_frac_rasa_gt_0_50": float(np.nanmean(rasa > 0.50)),
        f"{prefix}_frac_rasa_gt_1": float(np.nanmean(rasa > 1.0)),
        f"{prefix}_n_res": len(rows),
        f"{prefix}_n_exposed": int(np.nansum(rasa >= RASA_SURFACE_THRESHOLD)),
    }
    for name, aset in [
        ("hydrophobic", HYDROPHOBIC),
        ("aromatic", AROMATIC),
        ("positive", POSITIVE),
        ("negative", NEGATIVE),
        ("polar", POLAR),
    ]:
        feat[f"{prefix}_sasa_{name}"] = float(sum(r["sasa"] for r in rows if r["aa"] in aset))
    # RASA-weighted
    wh = sum((r["rasa"] if np.isfinite(r["rasa"]) else 0) * KD.get(r["aa"], 0) for r in rows)
    wc = sum(
        (r["rasa"] if np.isfinite(r["rasa"]) else 0)
        * {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}.get(r["aa"], 0)
        for r in rows
    )
    feat[f"{prefix}_rasa_w_hydrophobicity_sum"] = float(wh)
    feat[f"{prefix}_rasa_w_hydrophobicity_mean"] = float(wh / max(len(rows), 1))
    feat[f"{prefix}_rasa_w_charge_sum"] = float(wc)
    feat[f"{prefix}_exposed_pos"] = float(
        sum((r["rasa"] if np.isfinite(r["rasa"]) else 0) for r in rows if r["aa"] in POSITIVE)
    )
    feat[f"{prefix}_exposed_neg"] = float(
        sum((r["rasa"] if np.isfinite(r["rasa"]) else 0) for r in rows if r["aa"] in NEGATIVE)
    )
    return feat


def surface_patches(rows, prefix):
    """Connected components among exposed residues by CA distance."""
    exposed = [r for r in rows if np.isfinite(r["rasa"]) and r["rasa"] >= RASA_SURFACE_THRESHOLD and r["ca"] is not None]
    if len(exposed) < 2:
        return {
            f"{prefix}_n_hydrophobic_patches": 0,
            f"{prefix}_largest_hydrophobic_patch_sasa": 0.0,
            f"{prefix}_total_hydrophobic_patch_sasa": 0.0,
            f"{prefix}_largest_positive_patch_sasa": 0.0,
            f"{prefix}_largest_negative_patch_sasa": 0.0,
            f"{prefix}_cdr_hydrophobic_sasa_exposed": 0.0,
            f"{prefix}_h3_hydrophobic_sasa_exposed": 0.0,
        }

    def patch_stats(pred):
        idxs = [i for i, r in enumerate(exposed) if pred(r)]
        if not idxs:
            return 0, 0.0, 0.0
        parent = {i: i for i in idxs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                if np.linalg.norm(exposed[a]["ca"] - exposed[b]["ca"]) <= PATCH_CONTACT_A:
                    union(a, b)
        comps = {}
        for i in idxs:
            comps.setdefault(find(i), []).append(i)
        areas = [sum(exposed[i]["sasa"] for i in members) for members in comps.values()]
        return len(areas), float(max(areas) if areas else 0.0), float(sum(areas))

    n_h, largest_h, total_h = patch_stats(lambda r: r["aa"] in HYDROPHOBIC)
    _, largest_p, _ = patch_stats(lambda r: r["aa"] in POSITIVE)
    _, largest_n, _ = patch_stats(lambda r: r["aa"] in NEGATIVE)
    cdr_rows = [r for r in exposed if r.get("is_cdr") and r["aa"] in HYDROPHOBIC]
    h3 = [r for r in exposed if r["chain"] == "H" and r.get("region") == "CDR3" and r["aa"] in HYDROPHOBIC]
    return {
        f"{prefix}_n_hydrophobic_patches": n_h,
        f"{prefix}_largest_hydrophobic_patch_sasa": largest_h,
        f"{prefix}_total_hydrophobic_patch_sasa": total_h,
        f"{prefix}_largest_positive_patch_sasa": largest_p,
        f"{prefix}_largest_negative_patch_sasa": largest_n,
        f"{prefix}_cdr_hydrophobic_sasa_exposed": float(sum(r["sasa"] for r in cdr_rows)),
        f"{prefix}_h3_hydrophobic_sasa_exposed": float(sum(r["sasa"] for r in h3)),
    }


def interface_features(pdb_path, rows, prefix):
    parser = PDBParser(QUIET=True)
    feat = {}
    try:
        sasa_c = sum(r["sasa"] for r in rows)

        def iso_sasa(pdb_chain_id):
            st = parser.get_structure("x", str(pdb_path))
            model = st[0]
            for ch in list(model):
                if ch.id != pdb_chain_id:
                    model.detach_child(ch.id)
            sr = ShrakeRupley(probe_radius=PROBE_RADIUS, n_points=N_POINTS)
            sr.compute(st, level="R")
            return sum(float(res.sasa) for res in model[pdb_chain_id] if res.id[0] == " ")

        h_rows = [r for r in rows if r["chain"] == "H"]
        l_rows = [r for r in rows if r["chain"] == "L"]
        if not h_rows or not l_rows:
            chains = sorted({r["chain"] for r in rows})
            h_rows = [r for r in rows if r["chain"] == chains[0]]
            l_rows = [r for r in rows if r["chain"] == chains[1]]
        h_pdb = h_rows[0].get("pdb_chain", h_rows[0]["chain"])
        l_pdb = l_rows[0].get("pdb_chain", l_rows[0]["chain"])
        sh, sl = iso_sasa(h_pdb), iso_sasa(l_pdb)
        bsa = sh + sl - sasa_c
        feat[f"{prefix}_SASA_VH_iso"] = sh
        feat[f"{prefix}_SASA_VL_iso"] = sl
        feat[f"{prefix}_BSA"] = bsa
        feat[f"{prefix}_interface_area_BSA_over_2"] = bsa / 2.0
        iface = []
        for r in rows:
            if r["ca"] is None:
                continue
            other = [o for o in rows if o["chain"] != r["chain"] and o["ca"] is not None]
            if not other:
                continue
            dmin = min(np.linalg.norm(r["ca"] - o["ca"]) for o in other)
            if dmin <= 5.0:
                iface.append(r)
        feat[f"{prefix}_n_interface_res"] = len(iface)
        if iface:
            feat[f"{prefix}_iface_frac_hydrophobic"] = sum(r["aa"] in HYDROPHOBIC for r in iface) / len(iface)
            feat[f"{prefix}_iface_frac_aromatic"] = sum(r["aa"] in AROMATIC for r in iface) / len(iface)
            feat[f"{prefix}_iface_frac_positive"] = sum(r["aa"] in POSITIVE for r in iface) / len(iface)
            feat[f"{prefix}_iface_frac_negative"] = sum(r["aa"] in NEGATIVE for r in iface) / len(iface)
    except Exception as e:
        feat[f"{prefix}_interface_error"] = str(e)
    return feat


def process_pdb(pdb_path, antibody_id, prefix, vh_len=None, vl_len=None, numbering_row=None, conf_meta=None):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure(antibody_id, str(pdb_path))
    rows = residue_rows(st)
    rows = normalize_chain_labels(rows, vh_len, vl_len)
    rows = tag_regions(rows, numbering_row)
    feat = {"antibody_id": antibody_id}
    feat.update(aggregate(rows, f"{prefix}_Fv"))
    for ch, lab in [("H", "VH"), ("L", "VL")]:
        sub = [r for r in rows if r["chain"] == ch]
        feat.update(aggregate(sub, f"{prefix}_{lab}"))
    all_cdr = [r for r in rows if r.get("is_cdr")]
    for name, (ch, _, _) in CDR_DEFS.items():
        cdr_name = name.split("_", 1)[1]
        sub = [r for r in rows if r["chain"] == ch and r.get("region") == cdr_name]
        feat.update(aggregate(sub, f"{prefix}_{name}"))
    feat.update(aggregate(all_cdr, f"{prefix}_all_CDR"))
    fr = [r for r in rows if not r.get("is_cdr")]
    feat.update(aggregate(fr, f"{prefix}_FR"))
    feat.update(surface_patches(rows, f"{prefix}"))
    feat.update(interface_features(pdb_path, rows, f"{prefix}"))
    if conf_meta:
        feat[f"{prefix}_mean_confidence"] = conf_meta.get("mean_plddt")
        feat[f"{prefix}_mean_plddt"] = conf_meta.get("mean_plddt")
    return feat

def symlink_or_copy_abb():
    """Reuse B1 ABB structures into gate_b2 cache."""
    src = B1_CACHE / "structures" / "abodybuilder2"
    dst = CACHE / "structures" / "abodybuilder2"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.glob("*.pdb"):
        target = dst / p.name
        if not target.exists():
            try:
                target.symlink_to(p)
            except Exception:
                target.write_bytes(p.read_bytes())
        n += 1
    # copy manifest
    man = B1_CACHE / "structures" / "abb2_manifest.csv"
    if man.exists():
        pd.read_csv(man).to_csv(CACHE / "structures" / "abb2_manifest.csv", index=False)
    return n


def main():
    ensure_dirs()
    n_abb = symlink_or_copy_abb()
    print("ABB linked", n_abb, flush=True)

    union = pd.read_csv(DATA / "hic_tmapp_union.csv") if (DATA / "hic_tmapp_union.csv").exists() else pd.read_csv(
        Path("/workspace_developability_acquisition/gate_b1/data/shehata_b1_full.csv")
    )
    id_to_lens = {r.antibody_id: (len(r.heavy), len(r.light)) for r in union.itertuples()}
    num = pd.read_csv(Path("/workspace_developability_acquisition/gate_b1/data/numbering_germline.csv"))
    num_map = {r.antibody_id: r for r in num.itertuples(index=False)}

    # Process ABB (skip if already written)
    abb_out = CACHE / "structure_features" / "abb_sasa_rasa_patch.csv"
    if abb_out.exists() and len(pd.read_csv(abb_out)) >= 300:
        print("ABB_FEATS_CACHED", len(pd.read_csv(abb_out)), flush=True)
    else:
        abb_rows = []
        abb_man = pd.read_csv(CACHE / "structures" / "abb2_manifest.csv")
        for _, m in abb_man.iterrows():
            if not m.get("success", True):
                continue
            pdb = Path(m["pdb_path"])
            if not pdb.exists():
                pdb = CACHE / "structures" / "abodybuilder2" / f"{m['pair_hash']}.pdb"
            if not pdb.exists():
                continue
            vh_l, vl_l = id_to_lens.get(m["antibody_id"], (None, None))
            nrow = num_map.get(m["antibody_id"])
            feat = process_pdb(pdb, m["antibody_id"], "ABB", vh_l, vl_l, numbering_row=nrow)
            abb_rows.append(feat)
            if len(abb_rows) % 50 == 0:
                print("ABB SASA", len(abb_rows), flush=True)
                pd.DataFrame(abb_rows).to_csv(abb_out, index=False)
        pd.DataFrame(abb_rows).to_csv(abb_out, index=False)
        print("ABB_FEATS", len(abb_rows), flush=True)

    # Process native ESMFold when ready (poll briefly)
    esm_man_path = CACHE / "structures" / "esmfold_native_manifest.csv"
    if not esm_man_path.exists():
        print("ESMFN_MANIFEST_PENDING", flush=True)
    else:
        esm_out = CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv"
        esm_man = pd.read_csv(esm_man_path)
        done_ids = set()
        if esm_out.exists():
            prev = pd.read_csv(esm_out)
            done_ids = set(prev["antibody_id"])
            esm_rows = prev.to_dict("records")
        else:
            esm_rows = []
        for _, m in esm_man.iterrows():
            if not m.get("success", False):
                continue
            if m["antibody_id"] in done_ids:
                continue
            pdb = Path(m["pdb_path"])
            if not pdb.exists():
                continue
            vh_l, vl_l = id_to_lens.get(m["antibody_id"], (None, None))
            nrow = num_map.get(m["antibody_id"])
            meta_path = pdb.with_suffix(".json")
            conf = json.loads(meta_path.read_text()) if meta_path.exists() else {"mean_plddt": m.get("mean_plddt")}
            feat = process_pdb(pdb, m["antibody_id"], "ESMFN", vh_l, vl_l, numbering_row=nrow, conf_meta=conf)
            esm_rows.append(feat)
            done_ids.add(m["antibody_id"])
            if len(esm_rows) % 50 == 0:
                print("ESMFN SASA", len(esm_rows), flush=True)
                pd.DataFrame(esm_rows).to_csv(esm_out, index=False)
        pd.DataFrame(esm_rows).to_csv(esm_out, index=False)
        print("ESMFN_FEATS", len(esm_rows), flush=True)

    write_json(
        CACHE / "structure_features" / "sasa_params.json",
        {
            "probe_radius": PROBE_RADIUS,
            "n_points": N_POINTS,
            "biopython": Bio.__version__,
            "maxasa": "Tien2013/Wilke",
            "rasa_surface_threshold": RASA_SURFACE_THRESHOLD,
            "patch_contact_A": PATCH_CONTACT_A,
            "cdr_mapping": "IMGT resseq for ABB; author FR/CDR lengths for sequential ESMFold",
        },
    )
    print("SASA_FEATURES_DONE")


if __name__ == "__main__":
    main()
