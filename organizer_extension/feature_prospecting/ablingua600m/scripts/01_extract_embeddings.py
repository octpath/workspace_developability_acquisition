#!/usr/bin/env python3
"""Extract AbLingua-600M H/L MASKED_MEAN embeddings for all 324 competition Abs.

Uses official BioTokenizer + Simple_Collator from baysicx/AbLingua and
HF weights IDEA-AI4S/AbLingua. CLS is NOT inserted by the official collator
→ CLS_UNSUPPORTED (no invented CLS).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForMaskedLM

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/ablingua600m"
VENDOR = OUT / "vendor"
EMB = OUT / "embeddings"
CACHE = ROOT / ".cache/huggingface"
MODEL_ID = "IDEA-AI4S/AbLingua"
BATCH = 8  # FP32 on 3090; 256-seq × 1280 × 30 layers is comfortable
MAX_LEN = 256

sys.path.insert(0, str(VENDOR))
from AbLingua.collate import Simple_Collator  # noqa: E402
from AbLingua.tokenizer import BioTokenizer  # noqa: E402


def make_collator(tok: BioTokenizer) -> Simple_Collator:
    import argparse

    parser = argparse.ArgumentParser()
    parser = Simple_Collator.add_args(parser)
    args = parser.parse_args([])
    args.max_len = MAX_LEN
    args.truncation = True
    args.truncation_mode = "cut"
    args.padding = True
    return Simple_Collator(tok, args)


def masked_mean(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Mean over real tokens (attention_mask==1). Official collator has no CLS/SEP."""
    mask = attn.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tok = BioTokenizer(vocab_path=str(VENDOR / "AbLingua/tokens.txt"))
    collator = make_collator(tok)
    cls_id = int(tok.encode("[CLS]"))
    # Official tokenize() never inserts CLS — verify on a probe
    probe_ids = tok.tokenize("QVQLVQSGAEVKKPGASVKVSCKAS")
    cls_in_seq = cls_id in probe_ids
    cls_supported = False  # official embedding path does not use CLS
    print(
        f"CLS token id={cls_id} appears_in_official_tokenize={cls_in_seq} "
        f"→ CLS_SUPPORTED={cls_supported}",
        flush=True,
    )

    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=str(CACHE))
    print(
        f"config: layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"max_pos={config.max_position_embeddings} vocab={config.vocab_size}",
        flush=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE),
        output_hidden_states=True,
        return_dict=True,
    )
    model.eval()
    model.to(device)
    dtype_str = "float32"

    # Recover revision if present
    rev = None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(MODEL_ID)
        rev = info.sha
    except Exception as e:
        rev = f"unrecoverable:{e}"

    import transformers

    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda
    tf_ver = transformers.__version__

    # Sequences: DEV + Test, no labels
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")
    test = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")
    all_df = pd.concat(
        [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
        ignore_index=True,
    )
    all_df["id"] = all_df["id"].astype(str)
    assert len(all_df) == 324 and all_df["id"].nunique() == 324

    # Token-length audit BEFORE extraction
    lengths = []
    over = []
    for r in all_df.itertuples(index=False):
        for chain_name, seq in (("H", r.heavy), ("L", r.light)):
            n = len(tok.tokenize(seq))
            lengths.append({"id": r.id, "chain": chain_name, "token_len": n, "aa_len": len(seq)})
            if n > MAX_LEN:
                over.append({"id": r.id, "chain": chain_name, "token_len": n, "aa_len": len(seq)})
    len_df = pd.DataFrame(lengths)
    len_df.to_csv(OUT / "TOKEN_LENGTH_AUDIT.csv", index=False)
    if over:
        print("FATAL: sequences exceed max_position_embeddings; STOP", flush=True)
        print(json.dumps(over, indent=2))
        (OUT / "TRUNCATION_STOP.json").write_text(json.dumps(over, indent=2))
        sys.exit(2)

    print(
        f"token_len min/median/max = "
        f"{len_df.token_len.min()}/{len_df.token_len.median():.0f}/{len_df.token_len.max()}",
        flush=True,
    )

    ids = all_df["id"].tolist()
    H_mean = np.zeros((len(ids), config.hidden_size), dtype=np.float32)
    L_mean = np.zeros((len(ids), config.hidden_size), dtype=np.float32)

    t0 = time.time()
    for chain_key, store in (("heavy", H_mean), ("light", L_mean)):
        seqs = all_df[chain_key].tolist()
        for start in range(0, len(seqs), BATCH):
            batch_seqs = seqs[start : start + BATCH]
            batch = collator(batch_seqs)
            # Official collator may produce multiple windows if trunc; we asserted len<=256
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            if input_ids.shape[0] != len(batch_seqs):
                raise RuntimeError(
                    f"unexpected batch rows {input_ids.shape[0]} vs {len(batch_seqs)} "
                    "(possible truncation windows)"
                )
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attn)
                hidden = out.hidden_states[-1]  # [B, T, H]
                pooled = masked_mean(hidden, attn)
            store[start : start + len(batch_seqs)] = pooled.float().cpu().numpy()
            if (start // BATCH) % 5 == 0:
                print(
                    f"  {chain_key} {start}/{len(seqs)} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    wall = time.time() - t0
    HL = np.concatenate([H_mean, L_mean], axis=1)

    def save_parquet(arr, name, prefix):
        cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        df.insert(0, "id", ids)
        path = EMB / name
        df.to_parquet(path, index=False)
        return path

    p_h = save_parquet(H_mean, "ablingua600m_H_mean.parquet", "H_mean")
    p_l = save_parquet(L_mean, "ablingua600m_L_mean.parquet", "L_mean")
    p_hl = save_parquet(HL, "ablingua600m_HL_mean_concat.parquet", "HL_mean")

    meta = {
        "model_repo": MODEL_ID,
        "hf_revision": rev,
        "tokenizer": "baysicx/AbLingua BioTokenizer 3-gram (vendor/AbLingua/tokens.txt)",
        "tokenizer_note": "HF IDEA-AI4S/AbLingua has no tokenizer files; official GitHub tokenizer used",
        "transformers_version": tf_ver,
        "torch_version": torch_ver,
        "cuda_version": cuda_ver,
        "dtype": dtype_str,
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "max_position_embeddings": int(config.max_position_embeddings),
        "vocab_size_config": int(config.vocab_size),
        "vocab_size_tokenizer": int(tok.get_size()),
        "token_length_min": int(len_df.token_len.min()),
        "token_length_median": float(len_df.token_len.median()),
        "token_length_max": int(len_df.token_len.max()),
        "batch_size": BATCH,
        "extraction_wall_time_sec": wall,
        "n_ids": len(ids),
        "device": str(device),
        "representation_source": "outputs.hidden_states[-1]",
        "pooling": ["MASKED_MEAN"],
        "CLS_SUPPORTED": cls_supported,
        "CLS_REASON": "Official Simple_Collator/BioTokenizer.tokenize does not insert [CLS]; marking CLS_UNSUPPORTED",
        "files": {
            "H_mean": str(p_h),
            "L_mean": str(p_l),
            "HL_mean_concat": str(p_hl),
        },
        "gpu": "RTX 3090 (cuda:0)",
    }
    (OUT / "ABLINGUA_EXTRACTION_METADATA.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"DONE wall={wall:.1f}s wrote {p_h.name}, {p_l.name}, {p_hl.name}", flush=True)


if __name__ == "__main__":
    main()
