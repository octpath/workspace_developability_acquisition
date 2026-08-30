#!/usr/bin/env python3
"""Generate and cache PLM embeddings: AbLang2, AbLang, ESM-1b, ESM-2."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    CACHE,
    CONFIG,
    DATA,
    ensure_dirs,
    pair_hash,
    read_json,
    seq_hash,
    set_gpu0,
    sha256_text,
    write_json,
)


def cache_path(model_id: str, kind: str, key: str) -> Path:
    d = CACHE / "plm" / model_id.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_{key}.npy"


def save_emb(path: Path, arr: np.ndarray, meta: dict) -> None:
    np.save(path, arr.astype(np.float32))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_emb(path: Path):
    if path.exists():
        return np.load(path), json.loads(path.with_suffix(".json").read_text())
    return None, None


def mean_pool(token_emb: torch.Tensor, attention_mask: torch.Tensor | None = None) -> np.ndarray:
    """token_emb: [L, D] or [1,L,D]."""
    if token_emb.dim() == 3:
        token_emb = token_emb[0]
    if attention_mask is not None:
        m = attention_mask[0].bool() if attention_mask.dim() == 2 else attention_mask.bool()
        return token_emb[m].mean(0).detach().cpu().numpy()
    return token_emb.mean(0).detach().cpu().numpy()


def embed_esm(df: pd.DataFrame, model_name: str, model_id: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    revision = getattr(model.config, "_name_or_path", model_name)
    rows = []
    with torch.no_grad():
        for _, r in df.iterrows():
            ph = pair_hash(r["heavy"], r["light"])
            out_path = cache_path(model_id, "HL_concat_mean", ph)
            arr, meta = load_emb(out_path)
            if arr is None:
                embs = []
                metas = {}
                for chain, seq in [("H", r["heavy"]), ("L", r["light"])]:
                    inputs = tok(seq, return_tensors="pt", add_special_tokens=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out = model(**inputs)
                    # skip special tokens for mean
                    hidden = out.last_hidden_state[0]
                    # remove CLS/EOS if present
                    if hidden.size(0) >= 3:
                        pooled = hidden[1:-1].mean(0).cpu().numpy()
                    else:
                        pooled = hidden.mean(0).cpu().numpy()
                    embs.append(pooled)
                    metas[chain] = {"len": len(seq), "dim": int(pooled.shape[0])}
                arr = np.concatenate(embs, axis=0)
                meta = {
                    "model": model_name,
                    "model_id": model_id,
                    "pooling": "H_mean+L_mean_concat",
                    "pair_hash": ph,
                    "chains": metas,
                    "revision": revision,
                }
                save_emb(out_path, arr, meta)
            rows.append({"antibody_id": r["antibody_id"], "pair_hash": ph, "path": str(out_path), "dim": int(arr.shape[0])})
    del model
    torch.cuda.empty_cache()
    return rows


def embed_ablang2(df: pd.DataFrame, device: str):
    import ablang2

    # AbLang2 paired antibody model
    model_id = "ablang2_default"
    # API: ablang2.pretrained()
    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device=device)
    rows = []
    for _, r in df.iterrows():
        ph = pair_hash(r["heavy"], r["light"])
        out_path = cache_path(model_id, "HL_concat_mean", ph)
        arr, meta = load_emb(out_path)
        if arr is None:
            # AbLang2 expects list of [heavy, light]
            seqs = [[r["heavy"], r["light"]]]
            try:
                emb = ablang(seqs, mode="seqcoding")
                # emb shape handling
                emb = np.asarray(emb)
                if emb.ndim == 2 and emb.shape[0] == 1:
                    arr = emb[0]
                elif emb.ndim == 3:
                    arr = emb[0].mean(0)
                else:
                    arr = emb.reshape(-1)
            except Exception as e:
                # fallback: encode chains separately if API differs
                try:
                    h = np.asarray(ablang([[r["heavy"], ""]], mode="seqcoding")).reshape(-1)
                    l = np.asarray(ablang([["", r["light"]]], mode="seqcoding")).reshape(-1)
                    arr = np.concatenate([h, l])
                except Exception as e2:
                    print("ABLANG2_FAIL", r["antibody_id"], e, e2)
                    continue
            meta = {
                "model": "ablang2-paired",
                "model_id": model_id,
                "pooling": "seqcoding_or_HL_concat",
                "pair_hash": ph,
                "dim": int(arr.shape[0]),
            }
            save_emb(out_path, arr.astype(np.float32), meta)
        rows.append({"antibody_id": r["antibody_id"], "pair_hash": ph, "path": str(out_path), "dim": int(np.load(out_path).shape[0])})
    return rows


def embed_ablang_original(df: pd.DataFrame, device: str):
    import ablang

    model_id = "ablang_original"
    # separate heavy/light models
    heavy_model = ablang.pretrained("heavy", device=device)
    light_model = ablang.pretrained("light", device=device)
    rows = []
    for _, r in df.iterrows():
        ph = pair_hash(r["heavy"], r["light"])
        out_path = cache_path(model_id, "HL_concat_mean", ph)
        arr, meta = load_emb(out_path)
        if arr is None:
            try:
                h = np.asarray(heavy_model([r["heavy"]], mode="seqcoding"))
                l = np.asarray(light_model([r["light"]], mode="seqcoding"))
                h = h[0] if h.ndim > 1 else h
                l = l[0] if l.ndim > 1 else l
                arr = np.concatenate([h.reshape(-1), l.reshape(-1)])
            except Exception as e:
                print("ABLANG_FAIL", r["antibody_id"], e)
                continue
            meta = {
                "model": "ablang_original_heavy+light",
                "model_id": model_id,
                "pooling": "H_seqcoding+L_seqcoding_concat",
                "pair_hash": ph,
                "dim": int(arr.shape[0]),
            }
            save_emb(out_path, arr.astype(np.float32), meta)
        rows.append({"antibody_id": r["antibody_id"], "pair_hash": ph, "path": str(out_path), "dim": int(np.load(out_path).shape[0])})
    return rows


def cdr_pooling_esm(df: pd.DataFrame, num: pd.DataFrame, model_name: str, model_id: str, device: str):
    """CDR-aware pooling for a leading PLM (ESM-2 default)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    m = df.merge(num, on="antibody_id")
    rows = []
    with torch.no_grad():
        for _, r in m.iterrows():
            ph = pair_hash(r["heavy"], r["light"])
            out_path = cache_path(model_id, "CDR6_concat_mean", ph)
            arr, meta = load_emb(out_path)
            if arr is None:
                cdr_embs = []
                for chain, seq_col in [("H", "heavy"), ("L", "light")]:
                    seq = r[seq_col]
                    inputs = tok(seq, return_tensors="pt", add_special_tokens=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    hidden = model(**inputs).last_hidden_state[0]  # includes CLS
                    # map residues 0..L-1 to hidden[1:L+1]
                    for reg in ["CDR1", "CDR2", "CDR3"]:
                        cdr_seq = str(r[f"{chain}_{reg}"])
                        start = seq.find(cdr_seq)
                        if start < 0 or not cdr_seq:
                            # zeros
                            cdr_embs.append(np.zeros(hidden.size(-1), dtype=np.float32))
                            continue
                        # token positions
                        hslice = hidden[1 + start : 1 + start + len(cdr_seq)]
                        cdr_embs.append(hslice.mean(0).cpu().numpy())
                arr = np.concatenate(cdr_embs, axis=0)
                meta = {"model": model_name, "pooling": "H/L CDR1/2/3 means concat", "pair_hash": ph}
                save_emb(out_path, arr, meta)
            rows.append({"antibody_id": r["antibody_id"], "path": str(out_path), "dim": int(np.load(out_path).shape[0])})
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    set_gpu0()
    ensure_dirs()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("DEVICE", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    # Prefer triple+full unique — embed all 400 once
    manifests = {}

    print("=== AbLang2 ===", flush=True)
    try:
        manifests["ablang2_default"] = embed_ablang2(df, device)
    except Exception as e:
        manifests["ablang2_default"] = {"error": str(e)}
        print("ABLANG2_ERROR", e)

    print("=== AbLang original ===", flush=True)
    try:
        manifests["ablang_original"] = embed_ablang_original(df, device)
    except Exception as e:
        manifests["ablang_original"] = {"error": str(e)}
        print("ABLANG_ERROR", e)

    print("=== ESM-1b ===", flush=True)
    try:
        manifests["esm1b_t33_650M_UR50S"] = embed_esm(
            df, "facebook/esm1b_t33_650M_UR50S", "esm1b_t33_650M_UR50S", device
        )
    except Exception as e:
        manifests["esm1b_t33_650M_UR50S"] = {"error": str(e)}
        print("ESM1B_ERROR", e)

    print("=== ESM-2 650M ===", flush=True)
    try:
        manifests["esm2_t33_650M_UR50D"] = embed_esm(
            df, "facebook/esm2_t33_650M_UR50D", "esm2_t33_650M_UR50D", device
        )
    except Exception as e:
        manifests["esm2_t33_650M_UR50D"] = {"error": str(e)}
        print("ESM2_ERROR", e)

    print("=== ESM-2 CDR pooling ===", flush=True)
    num = pd.read_csv(DATA / "numbering_germline.csv")
    try:
        manifests["esm2_t33_650M_UR50D_CDR6"] = cdr_pooling_esm(
            df, num, "facebook/esm2_t33_650M_UR50D", "esm2_t33_650M_UR50D", device
        )
    except Exception as e:
        manifests["esm2_t33_650M_UR50D_CDR6"] = {"error": str(e)}
        print("ESM2_CDR_ERROR", e)

    # summarize
    summary = {}
    for k, v in manifests.items():
        if isinstance(v, dict) and "error" in v:
            summary[k] = v
        else:
            summary[k] = {"n": len(v), "dim0": v[0]["dim"] if v else None}
            pd.DataFrame(v).to_csv(CACHE / "plm" / f"manifest_{k}.csv", index=False)
    write_json(CACHE / "plm" / "embed_summary.json", summary)

    # update representation registry
    reg_path = CONFIG / "representation_registry.json"
    reg = read_json(reg_path) if reg_path.exists() else {}
    for mid in ["ablang2_default", "ablang_original", "esm1b_t33_650M_UR50S", "esm2_t33_650M_UR50D"]:
        reg[f"PLM_{mid}"] = {
            "type": "dense_embedding",
            "model_id": mid,
            "pooling": "HL_concat_mean",
            "manifest": f"manifest_{mid}.csv",
            "class": "PARTICIPANT_LEGAL",
        }
    reg["PLM_esm2_CDR6"] = {
        "type": "dense_embedding",
        "model_id": "esm2_t33_650M_UR50D",
        "pooling": "CDR6_concat_mean",
        "manifest": "manifest_esm2_t33_650M_UR50D_CDR6.csv",
        "class": "PARTICIPANT_LEGAL",
    }
    write_json(reg_path, reg)
    print("PLM_EMBED_OK", summary)


if __name__ == "__main__":
    main()
