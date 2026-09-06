#!/usr/bin/env python3
"""M3 SaProt 35M embeddings with AA+3Di tokens (Foldseek). Target-blind."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
SM = FP / "structure_marathon"
OUT = SM / "saprot"
OUT.mkdir(parents=True, exist_ok=True)
CW = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
PILOT = FP / "fab_reconstruction/structures/pilot_ids.json"
FOLDSEEK = ROOT / "tools/foldseek/bin/foldseek"
SAPROT_DIR = SM / "tools/SaProt"
sys.path.insert(0, str(SAPROT_DIR))

MODEL_ID = "westlake-repl/SaProt_35M_AF2"


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    # hidden: [1, L, D]; skip special tokens via mask
    mask = attention_mask.unsqueeze(-1).float()
    s = (hidden * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1.0)
    return (s / d).squeeze(0).detach().cpu().numpy()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from utils.foldseek_util import get_struc_seq
    from transformers import EsmTokenizer, EsmModel

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Prefer GPU0 (compatible CC); avoid unsupported 1080 Ti as default device
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    print("device", device, "model", MODEL_ID, flush=True)

    tokenizer = EsmTokenizer.from_pretrained(MODEL_ID)
    model = EsmModel.from_pretrained(MODEL_ID)
    model = model.eval().to(device)

    cw = pd.read_csv(CW)
    if args.pilot:
        ids = set(json.loads(PILOT.read_text()))
        cw = cw[cw.id.isin(ids)].copy()
        print("pilot n=", len(cw), flush=True)

    rows = []
    emb_rows = []
    t0 = time.time()
    for i, r in cw.iterrows():
        ab = str(r.id)
        pdb = Path(str(r.esmfold_canonical_path))
        if not pdb.exists():
            rows.append({"id": ab, "status": "FAIL:missing_pdb"})
            continue
        try:
            parsed = get_struc_seq(str(FOLDSEEK), str(pdb), ["A", "B"], plddt_mask=False, process_id=hash(ab) % 10_000)
            # chain-wise then concat mean of chain means (no artificial linker)
            chain_means = []
            lens = {}
            for ch in ["A", "B"]:
                if ch not in parsed:
                    continue
                _aa, _fs, combined = parsed[ch]
                inputs = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                vec = mean_pool(out.last_hidden_state, inputs["attention_mask"])
                chain_means.append(vec)
                lens[ch] = len(_aa)
            if not chain_means:
                raise RuntimeError("no_chains")
            # Fixed order A then B pools + global mean
            va = chain_means[0]
            vb = chain_means[1] if len(chain_means) > 1 else np.full_like(va, np.nan)
            vg = np.nanmean(np.stack(chain_means), axis=0)
            row = {"id": ab, "status": "SUCCESS", "len_A": lens.get("A", np.nan), "len_B": lens.get("B", np.nan)}
            for j, x in enumerate(vg):
                row[f"SaProt35_global_{j}"] = float(x)
            for j, x in enumerate(va):
                row[f"SaProt35_VH_{j}"] = float(x)
            if not np.isnan(vb).all():
                for j, x in enumerate(vb):
                    row[f"SaProt35_VL_{j}"] = float(x)
            rows.append(row)
            emb_rows.append(row)
        except Exception as e:
            rows.append({"id": ab, "status": f"FAIL:{type(e).__name__}:{e}"})
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(OUT / ("M3_FEATURES_pilot_partial.csv" if args.pilot else "M3_FEATURES_partial.csv"), index=False)
            print(f"SaProt {len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out_name = "M3_FEATURES_pilot.csv" if args.pilot else "M3_FEATURES.csv"
    df.to_csv(OUT / out_name, index=False)
    ok = int(df.status.astype(str).str.startswith("SUCCESS").sum())
    meta = {"n": len(df), "success": ok, "runtime_s": time.time() - t0, "model": MODEL_ID, "pilot": args.pilot}
    (OUT / ("M3_META_pilot.json" if args.pilot else "M3_META.json")).write_text(json.dumps(meta, indent=2))
    print("DONE SaProt", df.status.value_counts().to_dict(), meta, flush=True)


if __name__ == "__main__":
    main()
