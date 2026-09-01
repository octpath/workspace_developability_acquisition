#!/usr/bin/env python3
"""Build participant-safe Test features for Round 1 (label-independent)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "virtual_participant/round1_finalization/cache"
DEV = ROOT / "competition/data/distribution/dev.csv"
TEST = ROOT / "competition/data/distribution/test_features.csv"
TEST_ANN = ROOT / "competition/data/distribution/test_annotations.csv"
PLM_ROOT = ROOT / "gate_b1/cache/plm"
G2_FEAT = ROOT / "gate_b2/cache/structure_features/esmfold_native_sasa_rasa_patch.csv"
ESMFOLD = ROOT / "esmfold_native"
VENV_B1 = ROOT / ".venv_b1/bin/python"
VENV_ESMIF = ROOT / ".venv_esmif/bin/python"

sys.path.insert(0, str(ROOT / "virtual_participant/stage3_structure/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage4_advanced_structure/scripts"))


def ensure_out():
    OUT.mkdir(parents=True, exist_ok=True)


def build_anarci_regions_test():
    """Run ANARCI region extraction for Test 162 via venv_b1."""
    out_path = OUT / "anarci_imgt_regions_test.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        if len(df) == 162:
            return df
    script = OUT / "_anarci_test.py"
    script.write_text(
        '''
import sys
sys.path.insert(0, "/workspace_developability_acquisition/gate_b1/scripts/anarci_shim")
import pandas as pd
from anarci import anarci

REGIONS = {
    "fr1": (1, 26), "cdr1": (27, 38), "fr2": (39, 55), "cdr2": (56, 65),
    "fr3": (66, 104), "cdr3": (105, 117), "fr4": (118, 200),
}

def extract_regions(seq, allow, prefix):
    n, d, h = anarci([("q", seq)], scheme="imgt", output=False, allow=allow, allowed_species=["human"])
    row = {f"{prefix}_anarci_ok": bool(n and n[0])}
    for name in REGIONS:
        row[f"{prefix}_{name}"] = ""
    if not (n and n[0]):
        return row
    numbered = n[0][0][0]
    buckets = {k: [] for k in REGIONS}
    for (pos, ins), aa in numbered:
        if aa == "-":
            continue
        for rname, (lo, hi) in REGIONS.items():
            if lo <= pos <= hi:
                buckets[rname].append(aa)
    for rname in REGIONS:
        row[f"{prefix}_{rname}"] = "".join(buckets[rname])
    return row

test = pd.read_csv("/workspace_developability_acquisition/competition/data/distribution/test_features.csv")
rows = []
for _, r in test.iterrows():
    row = {"id": r["id"]}
    row.update(extract_regions(r["heavy"], {"H"}, "h"))
    row.update(extract_regions(r["light"], {"L", "K"}, "l"))
    rows.append(row)
pd.DataFrame(rows).to_csv("/workspace_developability_acquisition/virtual_participant/round1_finalization/cache/anarci_imgt_regions_test.csv", index=False)
print("done", len(rows))
'''
    )
    subprocess.run([str(VENV_B1), str(script)], check=True)
    return pd.read_csv(out_path)


def build_packing_test():
    from run_stage3 import compute_packing_table, pdb_seq_and_coords, packing_from_cas, center_dist, CONTACT_A
    import numpy as np

    out_path = OUT / "packing_esmfold_test.csv"
    if out_path.exists() and len(pd.read_csv(out_path)) == 162:
        return pd.read_csv(out_path)

    test = pd.read_csv(TEST)
    rows = []
    for _, r in test.iterrows():
        aid = r["id"]
        pdb = ESMFOLD / f"{aid}.pdb"
        feat = {"antibody_id": aid}
        if not pdb.exists():
            rows.append(feat)
            continue
        chains = pdb_seq_and_coords(pdb)
        # ESMFold: A=VH, B=VL by length convention used in stage3
        hid, lid = "A", "B"
        if hid not in chains or lid not in chains:
            rows.append(feat)
            continue
        h = packing_from_cas(chains[hid]["ca"])
        l = packing_from_cas(chains[lid]["ca"])
        fv_cas = chains[hid]["ca"] + chains[lid]["ca"]
        fv = packing_from_cas(fv_cas)
        for k, v in fv.items():
            feat[f"Fv_{k}"] = v
        for k, v in h.items():
            feat[f"VH_{k}"] = v
        for k, v in l.items():
            feat[f"VL_{k}"] = v
        feat["VH_VL_center_dist"] = center_dist(chains[hid]["ca"], chains[lid]["ca"])
        ha = np.array([c for c in chains[hid]["ca"] if c is not None], float)
        la = np.array([c for c in chains[lid]["ca"] if c is not None], float)
        if len(ha) and len(la):
            d = np.linalg.norm(ha[:, None, :] - la[None, :, :], axis=-1)
            feat["interface_contact_count"] = int((d < CONTACT_A).sum())
            feat["interface_contact_density"] = float(feat["interface_contact_count"] / (len(ha) + len(la)))
        rows.append(feat)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def build_stage3_combined():
    """Merge dev + test ESMFold structure features (gate_b2 + packing)."""
    from run_stage3 import load_precomputed

    out_path = OUT / "features_ESMFold_combined.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        if len(df) == 324:
            return out_path

    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    all_df = pd.concat([dev, test], ignore_index=True)

    esm = load_precomputed(all_df, "ESMFN_", G2_FEAT)
    pack_dev = pd.read_csv(ROOT / "virtual_participant/stage3_structure/cache/packing_esmfold.csv")
    pack_test = build_packing_test()
    packing = pd.concat([pack_dev, pack_test], ignore_index=True)
    esm = esm.merge(packing, on="antibody_id", how="left")

    def add_ratios(df):
        out = df.copy()
        if "Fv_total_sasa" in out.columns and "Fv_sasa_hydrophobic" in out.columns:
            tot = out["Fv_total_sasa"].replace(0, np.nan)
            out["Fv_hydrophobic_sasa_ratio"] = out["Fv_sasa_hydrophobic"] / tot
            out["Fv_aromatic_sasa_ratio"] = out["Fv_sasa_aromatic"] / tot
            out["Fv_charge_balance"] = out.get("Fv_sasa_positive", 0) - out.get("Fv_sasa_negative", 0)
            if "Fv_exposed_pos" in out.columns:
                out["Fv_exposed_charge_proxy"] = out["Fv_exposed_pos"] - out["Fv_exposed_neg"]
        if "Fv_frac_rasa_gt_0_20" in out.columns:
            out["Fv_buried_frac"] = 1.0 - out["Fv_frac_rasa_gt_0_20"]
        return out

    esm = add_ratios(esm)
    assert len(esm) == 324
    esm.to_csv(out_path, index=False)
    return out_path


def build_stage4_test_geometry():
    """Geometry + invfold features for Test (no APBS — not used by Round1 models)."""
    from extract_stage4_features import (
        advanced_patch_features,
        cavity_and_unsat_features,
        interaction_features,
        map_hl,
        residue_table,
        run_invfold_all,
        IF_DIR,
        classify_family,
    )

    out_partial = OUT / "stage4_test_geometry.csv"
    dev_feat = pd.read_csv(ROOT / "virtual_participant/stage4_advanced_structure/cache/features/stage4_all_features.csv")
    if out_partial.exists() and len(pd.read_csv(out_partial)) == 162:
        test_geom = pd.read_csv(out_partial)
    else:
        test = pd.read_csv(TEST)
        rows = []
        for i, r in test.iterrows():
            aid = r["id"]
            pdb = ESMFOLD / f"{aid}.pdb"
            feat = {"antibody_id": aid}
            if pdb.exists():
                rows_pdb = residue_table(pdb)
                chain_map = map_hl(rows_pdb, len(r["heavy"]), len(r["light"]))
                feat.update(advanced_patch_features(rows_pdb, chain_map))
                feat.update(interaction_features(rows_pdb, chain_map))
                feat.update(cavity_and_unsat_features(rows_pdb))
            rows.append(feat)
            if (i + 1) % 20 == 0:
                print(f"stage4 geom {i+1}/162", flush=True)
        test_geom = pd.DataFrame(rows)
        test_geom.to_csv(out_partial, index=False)

    test_ids = list(pd.read_csv(TEST)["id"])
    inv_path = OUT / "invfold_test.json"
    if not inv_path.exists():
        inv = run_invfold_all(test_ids)
        inv.to_csv(OUT / "invfold_test.csv", index=False)
    else:
        inv = pd.read_csv(OUT / "invfold_test.csv") if (OUT / "invfold_test.csv").exists() else pd.DataFrame(json.loads(inv_path.read_text()))

    test_m = test_geom.set_index("antibody_id")
    inv = inv.set_index("antibody_id")
    test_full = test_m.join(inv, how="left", rsuffix="_if").reset_index()

    # Align columns to dev: use dev column order, fill missing with NaN
    for c in dev_feat.columns:
        if c not in test_full.columns:
            test_full[c] = np.nan
    test_full = test_full[dev_feat.columns]

    combined = pd.concat([dev_feat, test_full], ignore_index=True)
    out_combined = OUT / "stage4_all_features_combined.csv"
    combined.to_csv(out_combined, index=False)
    return out_combined


def build_embeddings_combined():
    """Dev + Test PLM embeddings aligned to combined id order."""
    out_path = OUT / "round1_embeddings.npz"
    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    ids = list(dev["id"]) + list(test["id"])
    if out_path.exists():
        z = np.load(out_path, allow_pickle=True)
        if len(z["ids"]) == 324:
            return out_path

    emb_dev = np.load(ROOT / "virtual_participant/stage2_plm/cache/stage2_embeddings.npz", allow_pickle=True)
    plm_keys = [k for k in emb_dev.files if k != "ids"]

    def load_plm_block(plm, mid, repr_suffix):
        man = pd.read_csv(PLM_ROOT / f"manifest_{mid}.csv").set_index("antibody_id")
        vecs = []
        for aid in ids:
            path = Path(man.loc[aid, "path"])
            if "HL_concat_mean" not in path.name and repr_suffix == "HL":
                path = PLM_ROOT / mid / f"HL_concat_mean_{man.loc[aid, 'pair_hash']}.npy"
            arr = np.load(path).astype(np.float32)
            if repr_suffix == "H":
                arr = arr[:1280]
            elif repr_suffix == "L":
                arr = arr[1280:]
            vecs.append(arr)
        return np.vstack(vecs)

    out = {"ids": np.array(ids)}
    for plm, mid in [("esm1b", "esm1b_t33_650M_UR50S"), ("esm2", "esm2_t33_650M_UR50D")]:
        out[f"{plm}__H"] = load_plm_block(plm, mid, "H")
        out[f"{plm}__L"] = load_plm_block(plm, mid, "L")
        out[f"{plm}__HL"] = load_plm_block(plm, mid, "HL")

    # AbLang2
    man = pd.read_csv(PLM_ROOT / "manifest_ablang2_default.csv").set_index("antibody_id")
    cache = ROOT / "virtual_participant/stage2_plm/cache"
    paired, h_list, l_list, hl_cat = [], [], [], []
    need = []
    for i, aid in enumerate(ids):
        path = Path(man.loc[aid, "path"])
        paired.append(np.load(path).astype(np.float32))
        hp = cache / f"ablang2_H_{aid}.npy"
        lp = cache / f"ablang2_L_{aid}.npy"
        if hp.exists() and lp.exists():
            h_list.append(np.load(hp))
            l_list.append(np.load(lp))
        else:
            need.append(aid)
            h_list.append(None)
            l_list.append(None)
    if need:
        script = OUT / "_ablang2_chains.py"
        script.write_text(
            '''
import numpy as np
from pathlib import Path
import pandas as pd
import ablang2

ROOT = Path("/workspace_developability_acquisition")
cache = ROOT / "virtual_participant/stage2_plm/cache"
dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv").set_index("id")
test = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv").set_index("id")
need = ''' + repr(need) + '''
ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device="cpu")
for i, aid in enumerate(need):
    row = test.loc[aid] if aid in test.index else dev.loc[aid]
    hv = np.asarray(ablang([[row["heavy"], ""]], mode="seqcoding")).reshape(-1).astype(np.float32)
    lv = np.asarray(ablang([["", row["light"]]], mode="seqcoding")).reshape(-1).astype(np.float32)
    np.save(cache / f"ablang2_H_{aid}.npy", hv)
    np.save(cache / f"ablang2_L_{aid}.npy", lv)
    if (i + 1) % 20 == 0:
        print(f"ablang2 chains {i+1}/{len(need)}", flush=True)
print("done", len(need))
'''
        )
        subprocess.run([str(VENV_B1), str(script)], check=True)
        for i, aid in enumerate(ids):
            if h_list[i] is None:
                h_list[i] = np.load(cache / f"ablang2_H_{aid}.npy")
                l_list[i] = np.load(cache / f"ablang2_L_{aid}.npy")
    for i in range(len(ids)):
        hl_cat.append(np.concatenate([h_list[i], l_list[i]]))
    out["ablang2__H"] = np.vstack(h_list)
    out["ablang2__L"] = np.vstack(l_list)
    out["ablang2__HL"] = np.vstack(hl_cat)
    out["ablang2__HL_paired"] = np.vstack(paired)

    np.savez(out_path, **out)
    return out_path


def main():
    ensure_out()
    print("=== ANARCI test regions ===", flush=True)
    build_anarci_regions_test()
    print("=== Stage3 combined features ===", flush=True)
    build_stage3_combined()
    print("=== Stage4 test geometry ===", flush=True)
    build_stage4_test_geometry()
    print("=== Combined embeddings ===", flush=True)
    build_embeddings_combined()
    print("ROUND1_TEST_FEATURES_READY", flush=True)


if __name__ == "__main__":
    main()
