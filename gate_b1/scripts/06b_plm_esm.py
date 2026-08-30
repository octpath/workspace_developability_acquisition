#!/usr/bin/env python3
"""ESM-1b / ESM-2 embeddings only (AbLang2 already cached; original AbLang may hang on download)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import CACHE, CONFIG, DATA, ensure_dirs, read_json, set_gpu0, write_json  # noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location("plm", Path(__file__).resolve().parent / "06_plm_embeddings.py")
plm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plm)


def main():
    set_gpu0()
    ensure_dirs()
    device = "cuda:0"
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    summary = {}

    # AbLang2 already done?
    man2 = CACHE / "plm" / "manifest_ablang2_default.csv"
    if not man2.exists():
        n = len(list((CACHE / "plm" / "ablang2_default").glob("*.npy")))
        if n >= 400:
            # rebuild manifest
            rows = []
            from b1_common import pair_hash
            import numpy as np
            for _, r in df.iterrows():
                ph = pair_hash(r["heavy"], r["light"])
                p = CACHE / "plm" / "ablang2_default" / f"HL_concat_mean_{ph}.npy"
                rows.append({"antibody_id": r["antibody_id"], "pair_hash": ph, "path": str(p), "dim": int(np.load(p).shape[0])})
            pd.DataFrame(rows).to_csv(man2, index=False)
            summary["ablang2_default"] = {"n": len(rows), "dim0": rows[0]["dim"]}
    else:
        summary["ablang2_default"] = {"n": len(pd.read_csv(man2)), "from_cache": True}

    print("=== ESM-1b ===", flush=True)
    try:
        rows = plm.embed_esm(df, "facebook/esm1b_t33_650M_UR50S", "esm1b_t33_650M_UR50S", device)
        pd.DataFrame(rows).to_csv(CACHE / "plm" / "manifest_esm1b_t33_650M_UR50S.csv", index=False)
        summary["esm1b"] = {"n": len(rows), "dim0": rows[0]["dim"]}
    except Exception as e:
        summary["esm1b"] = {"error": str(e)}
        print("ESM1B_ERROR", e)

    print("=== ESM-2 ===", flush=True)
    try:
        rows = plm.embed_esm(df, "facebook/esm2_t33_650M_UR50D", "esm2_t33_650M_UR50D", device)
        pd.DataFrame(rows).to_csv(CACHE / "plm" / "manifest_esm2_t33_650M_UR50D.csv", index=False)
        summary["esm2"] = {"n": len(rows), "dim0": rows[0]["dim"]}
    except Exception as e:
        summary["esm2"] = {"error": str(e)}
        print("ESM2_ERROR", e)

    print("=== ESM-2 CDR6 ===", flush=True)
    num = pd.read_csv(DATA / "numbering_germline.csv")
    try:
        rows = plm.cdr_pooling_esm(df, num, "facebook/esm2_t33_650M_UR50D", "esm2_t33_650M_UR50D", device)
        pd.DataFrame(rows).to_csv(CACHE / "plm" / "manifest_esm2_t33_650M_UR50D_CDR6.csv", index=False)
        summary["esm2_cdr6"] = {"n": len(rows), "dim0": rows[0]["dim"]}
    except Exception as e:
        summary["esm2_cdr6"] = {"error": str(e)}
        print("CDR_ERROR", e)

    # Try original AbLang with short per-call; skip on failure
    print("=== AbLang original (best effort) ===", flush=True)
    try:
        rows = plm.embed_ablang_original(df.head(3), device)  # smoke 3
        if rows:
            rows = plm.embed_ablang_original(df, device)
            pd.DataFrame(rows).to_csv(CACHE / "plm" / "manifest_ablang_original.csv", index=False)
            summary["ablang_original"] = {"n": len(rows)}
        else:
            summary["ablang_original"] = {"error": "smoke failed"}
    except Exception as e:
        summary["ablang_original"] = {"error": str(e)}
        print("ABLANG_SKIP", e)

    write_json(CACHE / "plm" / "embed_summary.json", summary)
    reg = read_json(CONFIG / "representation_registry.json")
    for mid in ["ablang2_default", "ablang_original", "esm1b_t33_650M_UR50S", "esm2_t33_650M_UR50D"]:
        man = CACHE / "plm" / f"manifest_{mid}.csv"
        if man.exists():
            reg[f"PLM_{mid}"] = {
                "type": "dense_embedding",
                "model_id": mid,
                "pooling": "HL_concat_mean",
                "manifest": man.name,
                "class": "PARTICIPANT_LEGAL",
            }
    if (CACHE / "plm" / "manifest_esm2_t33_650M_UR50D_CDR6.csv").exists():
        reg["PLM_esm2_CDR6"] = {
            "type": "dense_embedding",
            "model_id": "esm2_t33_650M_UR50D",
            "pooling": "CDR6_concat_mean",
            "manifest": "manifest_esm2_t33_650M_UR50D_CDR6.csv",
            "class": "PARTICIPANT_LEGAL",
        }
    write_json(CONFIG / "representation_registry.json", reg)
    print("PLM_EMBED_OK", summary)


if __name__ == "__main__":
    main()
