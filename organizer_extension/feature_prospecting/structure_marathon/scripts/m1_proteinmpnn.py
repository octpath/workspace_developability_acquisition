#!/usr/bin/env python3
"""M1 ProteinMPNN native scoring on Fv ESMFold PDBs (soluble weights if present)."""
from __future__ import annotations
import json, os, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
MPNN = ROOT / "tools/ProteinMPNN"
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "structure_marathon/proteinmpnn"
OUT.mkdir(parents=True, exist_ok=True)
CW = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"

def pick_weights():
    for cand in [
        MPNN / "soluble_model_weights",
        MPNN / "vanilla_model_weights",
        MPNN / "ca_model_weights",
    ]:
        if cand.exists() and any(cand.glob("*.pt")):
            return cand
    return None

def main():
    wdir = pick_weights()
    if wdir is None:
        # download soluble weights via protein_mpnn helper if possible
        print("NO_WEIGHTS", flush=True)
        (OUT / "M1_STATUS.json").write_text(json.dumps({"status": "TECHNICAL_BLOCK", "reason": "no_weights"}))
        return
    print("weights", wdir, flush=True)
    cw = pd.read_csv(CW)
    # Use official score script if available
    score_py = MPNN / "protein_mpnn_run.py"
    # Simpler path: parse PDBs with their utils in-process
    import sys
    sys.path.insert(0, str(MPNN))
    import torch
    from protein_mpnn_utils import parse_PDB, StructureDatasetPDB, _scores, _S_to_seq, tied_featurize, loss_nll
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    ckpt = sorted(wdir.glob("*.pt"))[-1]
    print("ckpt", ckpt, device, flush=True)
    checkpoint = torch.load(ckpt, map_location=device)
    # Use ProteinMPNN class from utils
    from protein_mpnn_utils import ProteinMPNN
    hidden = 128
    model = ProteinMPNN(num_letters=21, node_features=hidden, edge_features=hidden, hidden_dim=hidden,
                        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    rows=[]
    t0=time.time()
    for _, r in cw.iterrows():
        ab=str(r.id)
        pdb=Path(str(r.esmfold_canonical_path))
        if not pdb.exists():
            continue
        try:
            pdb_dict_list = parse_PDB(str(pdb))
            dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=10000)
            # score native
            for batch in [[dataset[0]]]:
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch, device, None, None, None, None, None, False)
                noise = torch.zeros_like(X)
                with torch.no_grad():
                    randn = torch.randn(chain_M.shape, device=device)
                    log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn)
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S, log_probs, mask_for_loss)
                    native_score = scores.cpu().data.numpy()
                rows.append({"id": ab, "status":"SUCCESS", "MPNN_native_score": float(native_score.mean()), "MPNN_n_res": int(mask_for_loss.sum().item())})
        except Exception as e:
            rows.append({"id": ab, "status": f"FAIL:{type(e).__name__}:{e}"})
        if len(rows)%40==0:
            pd.DataFrame(rows).to_csv(OUT/"M1_FEATURES_partial.csv", index=False)
            print(f"MPNN {len(rows)} {time.time()-t0:.0f}s", flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"M1_FEATURES.csv", index=False)
    (OUT/"M1_META.json").write_text(json.dumps({"n":len(df),"success":int(df.status.eq('SUCCESS').sum()),"weights":str(wdir),"runtime_s":time.time()-t0},indent=2))
    print("DONE", df.status.value_counts().to_dict(), flush=True)

if __name__=='__main__':
    main()
