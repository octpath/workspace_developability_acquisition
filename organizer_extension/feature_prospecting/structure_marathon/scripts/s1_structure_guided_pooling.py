#!/usr/bin/env python3
"""S1: Structure-guided ESM2 residue pooling (target-blind masks)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
SM = FP / "structure_marathon"
OUT = SM / "structure_guided_pooling"
OUT.mkdir(parents=True, exist_ok=True)
CW = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
CDR = FP / "cdr_sequence_index_imgt.csv"
SEQ = FP / "fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
CACHE_EMB = SM / "cache/esm2_residue"
CACHE_EMB.mkdir(parents=True, exist_ok=True)

EXPOSED = 0.25
STRONG = 0.40
CONTACT = 8.0
IFACE = 4.5
ARO = set("FYW")
HYDRO = set("AILMFVWY")


def load_pdb_atoms(pdb: Path):
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    residues = []
    for ch in st[0]:
        for res in ch:
            if res.id[0] != " ":
                continue
            atoms = {a.get_name().strip(): a.coord.copy() for a in res}
            if "CA" not in atoms:
                continue
            residues.append(
                {
                    "chain": ch.id,
                    "resseq": int(res.id[1]),
                    "resname": res.get_resname(),
                    "ca": atoms["CA"],
                    "cb": atoms.get("CB", atoms["CA"]),
                    "heavy": [a.coord.copy() for a in res if a.element != "H"],
                }
            )
    return residues


def approx_rasa(residues):
    """Cheap proxy: neighbor density → exposure score in [0,1]."""
    n = len(residues)
    if n == 0:
        return np.zeros(0)
    coords = np.asarray([r["ca"] for r in residues], float)
    rasa = np.zeros(n)
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        nn = int(((d > 0.1) & (d < 10.0)).sum())
        rasa[i] = float(np.clip(1.0 - nn / 25.0, 0.0, 1.0))
    return rasa


def aa1(resname: str) -> str:
    from Bio.Data.IUPACData import protein_letters_3to1

    return protein_letters_3to1.get(resname.capitalize(), "X")


def build_masks(residues, rasa, cdr_set, vh_len_hint=None):
    n = len(residues)
    chains = sorted({r["chain"] for r in residues})
    heavy = chains[0]
    light = chains[1] if len(chains) > 1 else chains[0]
    # map CDR by sequential index within chain using CDR file keys (H/L, seq index)
    masks = {k: np.zeros(n, bool) for k in [
        "CDR_ALL", "HCDR3", "LCDR3", "BURIED_CORE", "EXPOSED", "STRONGLY_EXPOSED",
        "VH_VL_INTERFACE", "HIGH_CONTACT_DENSITY_CORE", "EXPOSED_AROMATIC",
        "STRONGLY_EXPOSED_AROMATIC", "EXPOSED_HYDROPHOBIC", "CDR_EXPOSED",
    ]}
    # contact density
    coords = np.asarray([r["cb"] for r in residues], float)
    deg = np.zeros(n)
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        deg[i] = ((d > 0.1) & (d <= CONTACT)).sum()

    # interface: any heavy-atom between chains < IFACE
    iface = np.zeros(n, bool)
    for i, ri in enumerate(residues):
        for j, rj in enumerate(residues):
            if ri["chain"] == rj["chain"]:
                continue
            for a in ri["heavy"]:
                for b in rj["heavy"]:
                    if np.linalg.norm(a - b) < IFACE:
                        iface[i] = True
                        break
                if iface[i]:
                    break

    for i, r in enumerate(residues):
        aa = aa1(r["resname"])
        key = (r["chain"], r["resseq"])
        # CDR via sequential: use cdr_set with remapped chain H/L
        is_cdr = False
        is_h3 = False
        is_l3 = False
        # approximate: use provided cdr map by absolute seq index along chain order
        # filled below after building chain-order index

    # rebuild with sequence index within chain
    per_chain = {}
    for i, r in enumerate(residues):
        per_chain.setdefault(r["chain"], []).append(i)
    for ch, idxs in per_chain.items():
        src = "H" if ch == heavy else "L"
        for local_i, gi in enumerate(idxs, start=1):
            r = residues[gi]
            aa = aa1(r["resname"])
            lab = cdr_set.get((src, local_i), "FW")
            if "CDR" in lab or "H3" in lab or "L3" in lab:
                masks["CDR_ALL"][gi] = True
            if "H3" in lab or lab == "HCDR3":
                masks["HCDR3"][gi] = True
            if "L3" in lab or lab == "LCDR3":
                masks["LCDR3"][gi] = True
            if rasa[gi] >= EXPOSED:
                masks["EXPOSED"][gi] = True
            if rasa[gi] >= STRONG:
                masks["STRONGLY_EXPOSED"][gi] = True
            if rasa[gi] < EXPOSED:
                masks["BURIED_CORE"][gi] = True
            if iface[gi]:
                masks["VH_VL_INTERFACE"][gi] = True
            if deg[gi] >= np.quantile(deg, 0.75):
                masks["HIGH_CONTACT_DENSITY_CORE"][gi] = True
            if rasa[gi] >= EXPOSED and aa in ARO:
                masks["EXPOSED_AROMATIC"][gi] = True
            if rasa[gi] >= STRONG and aa in ARO:
                masks["STRONGLY_EXPOSED_AROMATIC"][gi] = True
            if rasa[gi] >= EXPOSED and aa in HYDRO:
                masks["EXPOSED_HYDROPHOBIC"][gi] = True
            if masks["CDR_ALL"][gi] and rasa[gi] >= EXPOSED:
                masks["CDR_EXPOSED"][gi] = True

    # largest hydrophobic surface patch among EXPOSED_HYDROPHOBIC
    hydro_idx = np.where(masks["EXPOSED_HYDROPHOBIC"])[0]
    patch = np.zeros(n, bool)
    neigh = np.zeros(n, bool)
    if len(hydro_idx):
        # BFS on CA<=6
        adj = {i: [] for i in hydro_idx}
        for a in range(len(hydro_idx)):
            for b in range(a + 1, len(hydro_idx)):
                i, j = hydro_idx[a], hydro_idx[b]
                if np.linalg.norm(residues[i]["ca"] - residues[j]["ca"]) <= 6.0:
                    adj[i].append(j)
                    adj[j].append(i)
        seen = set()
        best = []
        for s in hydro_idx:
            if s in seen:
                continue
            stack = [s]
            comp = []
            seen.add(s)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            if len(comp) > len(best):
                best = comp
        patch[best] = True
        for i in best:
            for j in range(n):
                if np.linalg.norm(residues[i]["ca"] - residues[j]["ca"]) <= 8.0:
                    neigh[j] = True
    masks["LARGEST_HYDROPHOBIC_SURFACE_PATCH"] = patch
    masks["PATCH_NEIGHBORS"] = neigh
    return masks


def cdr_lookup(ab_id: str):
    if not CDR.exists():
        return {}
    df = pd.read_csv(CDR)
    df = df[df.id == ab_id]
    out = {}
    for _, r in df.iterrows():
        out[(str(r.chain), int(r.sequence_index))] = str(r.get("region", "FW"))
    return out


def extract_residue_emb(model, alphabet, batch_converter, seq: str, device):
    data = [("x", seq)]
    _, _, toks = batch_converter(data)
    toks = toks.to(device)
    with torch.no_grad():
        out = model(toks, repr_layers=[33], return_contacts=False)
    # tokens: BOS + seq + EOS
    rep = out["representations"][33][0, 1 : 1 + len(seq)].cpu().numpy().astype(np.float32)
    return rep


def pool(emb, mask):
    if mask.sum() == 0 or emb is None:
        return np.full(emb.shape[1] if emb is not None else 1280, np.nan, np.float32)
    return emb[mask].mean(axis=0)


def main():
    import esm

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    cw = pd.read_csv(CW)
    seqs = pd.read_csv(SEQ).set_index("id") if SEQ.exists() else None
    # Use gate_b1 sequences if needed
    b1 = ROOT / "gate_b1/data/shehata_b1_full.csv"
    if b1.exists():
        b1df = pd.read_csv(b1)
        if "antibody_id" in b1df.columns:
            b1df = b1df.set_index("antibody_id")
        elif "id" in b1df.columns:
            b1df = b1df.set_index("id")
        else:
            b1df = None
    else:
        b1df = None

    feat_rows = []
    t0 = time.time()
    for _, row in cw.iterrows():
        ab = str(row.id)
        pdb = Path(str(row.esmfold_canonical_path))
        if not pdb.exists():
            continue
        # sequences
        if seqs is not None and ab in seqs.index:
            vh, vl = str(seqs.loc[ab].VH_used), str(seqs.loc[ab].VL_used)
        elif b1df is not None and ab in b1df.index:
            vh = str(b1df.loc[ab]["heavy"]); vl = str(b1df.loc[ab]["light"])
        else:
            continue

        cache = CACHE_EMB / f"{ab}.npz"
        if cache.exists():
            z = np.load(cache)
            emb_h, emb_l = z["H"], z["L"]
        else:
            emb_h = extract_residue_emb(model, alphabet, batch_converter, vh, device)
            emb_l = extract_residue_emb(model, alphabet, batch_converter, vl, device)
            np.savez_compressed(cache, H=emb_h, L=emb_l)

        residues = load_pdb_atoms(pdb)
        # map structure residues to sequence embeddings by chain order
        chains = sorted({r["chain"] for r in residues})
        heavy, light = chains[0], (chains[1] if len(chains) > 1 else chains[0])
        # truncate/pad to seq lengths
        h_idx = [i for i, r in enumerate(residues) if r["chain"] == heavy]
        l_idx = [i for i, r in enumerate(residues) if r["chain"] == light]
        n_h, n_l = len(h_idx), len(l_idx)
        emb = np.zeros((len(residues), emb_h.shape[1]), np.float32)
        for k, i in enumerate(h_idx):
            if k < len(emb_h):
                emb[i] = emb_h[k]
        for k, i in enumerate(l_idx):
            if k < len(emb_l):
                emb[i] = emb_l[k]

        rasa = approx_rasa(residues)
        masks = build_masks(residues, rasa, cdr_lookup(ab))
        feat = {"id": ab}
        for name, m in masks.items():
            # restrict mask to residues with non-zero emb rows
            valid = m & (np.linalg.norm(emb, axis=1) > 0)
            v = pool(emb, valid)
            # store PCA-ready: keep mean of first 32 dims? No — store full then PCA in eval.
            # For file size, store 32 random projection? Spec wants full then PCA32 fold-local.
            # Save mean pooled 1280 — large but OK for 324.
            for j, x in enumerate(v):
                feat[f"S1_{name}_{j}"] = float(x)
            feat[f"S1_{name}_n"] = int(valid.sum())
        # contrasts (aligned dim)
        for a, b, cname in [
            ("EXPOSED", "BURIED_CORE", "exposed_minus_buried"),
            ("CDR_ALL", "BURIED_CORE", "cdr_minus_buried"),
        ]:
            if a in masks and b in masks:
                va = pool(emb, masks[a] & (np.linalg.norm(emb, axis=1) > 0))
                vb = pool(emb, masks[b] & (np.linalg.norm(emb, axis=1) > 0))
                d = va - vb
                for j, x in enumerate(d):
                    feat[f"S1_{cname}_{j}"] = float(x)
        feat_rows.append(feat)
        if len(feat_rows) % 20 == 0:
            print(f"S1 {len(feat_rows)}/{len(cw)} elapsed={time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(feat_rows).to_csv(OUT / "S1_FEATURES_partial.csv", index=False)

    df = pd.DataFrame(feat_rows)
    df.to_csv(OUT / "S1_FEATURES.csv", index=False)
    # also write compact family slices for high-priority HIC masks (first 32 dims for quick Ridge without full 1280*many)
    # Full eval uses PCA32 on each mask family separately.
    meta = {
        "n_abs": len(df),
        "embedding": "esm2_t33_650M_UR50D",
        "structure": "esmfold_fv",
        "exposed_thr": EXPOSED,
        "strong_thr": STRONG,
        "runtime_s": time.time() - t0,
    }
    (OUT / "S1_META.json").write_text(json.dumps(meta, indent=2))
    print("DONE S1", len(df), meta["runtime_s"], flush=True)


if __name__ == "__main__":
    main()
