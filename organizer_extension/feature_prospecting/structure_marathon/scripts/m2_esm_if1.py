#!/usr/bin/env python3
"""M2 ESM-IF1 native sequence scoring on Fv ESMFold."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "structure_marathon/esm_if1"
OUT.mkdir(parents=True, exist_ok=True)
CW = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
SEQ = FP / "fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
B1 = ROOT / "gate_b1/data/shehata_b1_full.csv"

def main():
    import esm
    import esm.inverse_folding
    # Official util keeps token indices on CPU; model.to(cuda) causes RuntimeError.
    # CPU is scientifically equivalent for native LL scoring and avoids dual-GPU CC issues.
    device = torch.device("cpu")
    print("device", device, flush=True)
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    cw = pd.read_csv(CW)
    seqs = pd.read_csv(SEQ).set_index("id")
    b1 = pd.read_csv(B1).set_index("antibody_id")
    rows = []
    done = set()
    partial = OUT / "M2_ESMIF1_FEATURES_partial.csv"
    if partial.exists():
        prev = pd.read_csv(partial)
        ok_prev = prev[prev.status.astype(str).str.startswith("SUCCESS")]
        rows = ok_prev.to_dict("records")
        done = {str(x) for x in ok_prev["id"]}
        print(f"resume from partial: {len(done)}", flush=True)
    t0 = time.time()
    for _, r in cw.iterrows():
        ab = str(r.id)
        if ab in done:
            continue
        pdb = Path(str(r.esmfold_canonical_path))
        if not pdb.exists():
            continue
        try:
            try:
                structure = esm.inverse_folding.util.load_structure(str(pdb), ["A", "B"])
            except Exception:
                structure = esm.inverse_folding.util.load_structure(str(pdb), ["H", "L"])
            coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
            chs = list(coords.keys())
            scores = {}
            for ch in chs:
                out = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                    model, alphabet, coords, ch, native_seqs[ch]
                )
                scores[ch] = float(out[0] if isinstance(out, (tuple, list)) else out)
            vals = list(scores.values())
            rows.append({
                "id": ab,
                "status": "SUCCESS",
                "IF1_ll_chain0": vals[0],
                "IF1_ll_chain1": vals[1] if len(vals) > 1 else np.nan,
                "IF1_ll_mean": float(np.mean(vals)),
                "IF1_ll_sum": float(np.sum(vals)),
            })
        except Exception as e:
            try:
                structure = esm.inverse_folding.util.load_structure(str(pdb))
                coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
                ll, _ = esm.inverse_folding.util.score_sequence(model, alphabet, coords, seq)
                rows.append({"id": ab, "status": "SUCCESS_SINGLE", "IF1_ll_chain0": float(ll), "IF1_ll_chain1": np.nan, "IF1_ll_mean": float(ll), "IF1_ll_sum": float(ll)})
            except Exception as e2:
                rows.append({"id": ab, "status": f"FAIL:{type(e2).__name__}"})
        if len(rows) % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT / "M2_ESMIF1_FEATURES_partial.csv", index=False)
            print(f"IF1 {len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "M2_ESMIF1_FEATURES.csv", index=False)
    ok = int(df.status.astype(str).str.startswith("SUCCESS").sum())
    (OUT / "M2_META.json").write_text(json.dumps({"n": len(df), "success": ok, "runtime_s": time.time()-t0}, indent=2))
    print("DONE IF1", df.status.value_counts().to_dict(), flush=True)

if __name__ == "__main__":
    main()
