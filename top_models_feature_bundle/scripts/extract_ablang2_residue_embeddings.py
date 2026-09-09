#!/usr/bin/env python3
"""Extract AbLang2 residue embeddings (official rescoding path) for the bundle.

Uses the same ablang2-paired checkpoint/API as Stage-2 seqcoding.
Special tokens (<, >, |) are stripped with an exact character-index map;
amino-acid token positions are 1:1 with sequence residues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parent
OUT = BUNDLE / "residue_level" / "ablang2"
MODEL_TO_USE = "ablang2-paired"
PACKAGE_VERSION = "0.2.1"
SPECIAL = set("<>|")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def format_pair(heavy: str, light: str) -> str:
    """Match ablang2.pretrained.add_extra_tokens."""
    return f"<{heavy}>|<{light}>".replace("<>", "")


def aa_token_indices(fmt: str, heavy: str, light: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact AA indices in the formatted AbLang2 string for H and L."""
    if heavy and light:
        expect = f"<{heavy}>|<{light}>"
        if fmt != expect:
            raise RuntimeError(f"format mismatch HL: {fmt!r} vs {expect!r}")
        h_idx = np.arange(1, 1 + len(heavy), dtype=np.int64)
        l_start = 1 + len(heavy) + 3  # skip >, |, <
        l_idx = np.arange(l_start, l_start + len(light), dtype=np.int64)
        return h_idx, l_idx
    if heavy and not light:
        expect = f"<{heavy}>|"
        if fmt != expect:
            raise RuntimeError(f"format mismatch H: {fmt!r}")
        return np.arange(1, 1 + len(heavy), dtype=np.int64), np.zeros(0, dtype=np.int64)
    if light and not heavy:
        expect = f"|<{light}>"
        if fmt != expect:
            raise RuntimeError(f"format mismatch L: {fmt!r}")
        return np.zeros(0, dtype=np.int64), np.arange(2, 2 + len(light), dtype=np.int64)
    raise RuntimeError("empty heavy and light")


def pack_padded(
    ids: list[str], arrays: list[np.ndarray], max_len: int, hidden: int
) -> tuple[np.ndarray, np.ndarray]:
    N = len(ids)
    emb = np.zeros((N, max_len, hidden), dtype=np.float16)
    mask = np.zeros((N, max_len), dtype=np.bool_)
    for i, a in enumerate(arrays):
        L = a.shape[0]
        if L > max_len:
            raise RuntimeError(f"seq longer than max_len: {ids[i]} {L}>{max_len}")
        if a.shape[1] != hidden:
            raise RuntimeError(f"hidden mismatch {ids[i]}")
        if not np.isfinite(a.astype(np.float32)).all():
            raise RuntimeError(f"non-finite {ids[i]}")
        emb[i, :L] = a.astype(np.float16)
        mask[i, :L] = True
    return emb, mask


def split_npy(path: Path) -> list[Path]:
    """Deterministic half/half byte split into .part0 + .part1 (repo convention).

    Stale part files are removed before writing. Reassembly must equal original.
    """
    data = path.read_bytes()
    for stale in sorted(path.parent.glob(path.name + ".part*")):
        stale.unlink()
    mid = len(data) // 2
    parts = []
    for i, chunk in enumerate((data[:mid], data[mid:])):
        p = Path(str(path) + f".part{i}")
        p.write_bytes(chunk)
        parts.append(p)
    rebuilt = parts[0].read_bytes() + parts[1].read_bytes()
    if rebuilt != data:
        raise RuntimeError(f"part reassembly failed for {path}")
    if hashlib.sha256(rebuilt).hexdigest() != hashlib.sha256(data).hexdigest():
        raise RuntimeError(f"SHA mismatch after split for {path}")
    return parts


def extract_chain_batch(
    ablang,
    ids: list[str],
    seqs: pd.DataFrame,
    chain: str,
    device: str,
) -> tuple[list[np.ndarray], list[dict]]:
    rows = []
    arrays = []
    for ab in ids:
        heavy = str(seqs.loc[ab, "heavy"])
        light = str(seqs.loc[ab, "light"])
        if chain == "H":
            pair = [heavy, ""]
            aa_len = len(heavy)
        else:
            pair = ["", light]
            aa_len = len(light)
        fmt = format_pair(pair[0], pair[1])
        h_idx, l_idx = aa_token_indices(fmt, pair[0], pair[1])
        aa_idx = h_idx if chain == "H" else l_idx
        if len(aa_idx) != aa_len:
            raise RuntimeError(f"{ab} {chain}: AA index len {len(aa_idx)} != {aa_len}")
        # verify characters
        for j, ti in enumerate(aa_idx):
            ch = fmt[int(ti)]
            expect = (heavy if chain == "H" else light)[j]
            if ch != expect:
                raise RuntimeError(f"{ab} {chain} map fail at {j}: {ch}!={expect}")
        tokens = ablang.tokenizer([fmt], pad=True, w_extra_tkns=False, device=device)
        with torch.no_grad():
            hs = ablang.AbRep(tokens).last_hidden_states[0].detach().float().cpu().numpy()
        if hs.shape[0] < len(fmt):
            raise RuntimeError(f"{ab} {chain}: truncated hidden {hs.shape[0]} < {len(fmt)}")
        # special-token treatment: drop < > | via AA index selection only
        res = hs[aa_idx]
        if res.shape[0] != aa_len:
            raise RuntimeError(f"{ab} {chain}: residue len mismatch")
        if not np.isfinite(res).all():
            raise RuntimeError(f"{ab} {chain}: non-finite")
        arrays.append(res.astype(np.float32))
        rows.append(
            {
                "id": ab,
                "chain": chain,
                "sequence_length": aa_len,
                "token_length": int(len(fmt)),
                "residue_length": int(res.shape[0]),
                "hidden_dimension": int(res.shape[1]),
                "dtype": "float32",
                "finite": True,
                "mapping_status": "EXACT_AA_1TO1_SPECIAL_STRIPPED",
                "formatted_prefix": fmt[:8],
                "formatted_suffix": fmt[-8:],
                "n_special_in_format": int(sum(c in SPECIAL for c in fmt)),
            }
        )
    return arrays, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-note", default="", help="unused; extraction is per-id for exact map audit")
    args = ap.parse_args()

    import ablang2

    OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(BUNDLE / "dev.csv")
    test = pd.read_csv(BUNDLE / "test.csv")
    seqs = pd.concat(
        [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
        ignore_index=True,
    )
    seqs["id"] = seqs["id"].astype(str)
    ids = seqs["id"].tolist()
    if len(ids) != 324 or seqs["id"].duplicated().any():
        raise RuntimeError("expected 324 unique ids")
    seqs = seqs.set_index("id")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Loading AbLang2 {MODEL_TO_USE} on {device} ...", flush=True)
    t0 = time.time()
    ablang = ablang2.pretrained(
        model_to_use=MODEL_TO_USE, random_init=False, ncpu=1, device=device
    )
    ablang.freeze()

    all_meta_rows = []
    h_arrays: list[np.ndarray] = []
    l_arrays: list[np.ndarray] = []

    for chain, sink in (("H", h_arrays), ("L", l_arrays)):
        for i, ab in enumerate(ids):
            arrs, rows = extract_chain_batch(ablang, [ab], seqs, chain, device)
            sink.extend(arrs)
            all_meta_rows.extend(rows)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  AbLang2 {chain} {i+1}/{len(ids)}", flush=True)

    hidden = int(h_arrays[0].shape[1])
    max_h = max(a.shape[0] for a in h_arrays)
    max_l = max(a.shape[0] for a in l_arrays)
    emb_h, mask_h = pack_padded(ids, h_arrays, max_h, hidden)
    emb_l, mask_l = pack_padded(ids, l_arrays, max_l, hidden)

    np.save(OUT / "ids.npy", np.asarray(ids, dtype=object))
    np.save(OUT / "heavy_embeddings.npy", emb_h)
    np.save(OUT / "light_embeddings.npy", emb_l)
    np.save(OUT / "heavy_mask.npy", mask_h)
    np.save(OUT / "light_mask.npy", mask_l)

    # QC table
    qc = pd.DataFrame(all_meta_rows)
    qc_path = OUT / "ablang2_residue_qc.csv"
    qc.to_csv(qc_path, index=False)

    # Compare AA-mean vs Stage-2 seqcoding (includes specials) → NOT comparable
    # Also compare official rescoding mean (full fmt) vs seqcoding on a sample
    pool = pd.read_parquet(BUNDLE / "data/ablang2.parquet")
    pool = pool.set_index(pool["id"].astype(str))
    # bundle ablang2.parquet columns?
    feat_cols = [c for c in pool.columns if c != "id"]
    # Stage-2 uses H and L separately then often concat; check parquet schema
    sample_diffs_aa_mean = []
    sample_diffs_fmt_mean = []
    for ab in ids[:5]:
        # rebuild fmt mean for H
        heavy = str(seqs.loc[ab, "heavy"])
        fmt = format_pair(heavy, "")
        tokens = ablang.tokenizer([fmt], pad=True, w_extra_tkns=False, device=device)
        with torch.no_grad():
            hs = ablang.AbRep(tokens).last_hidden_states[0].float().cpu().numpy()
        fmt_mean = hs[: len(fmt)].mean(0)
        aa_mean = h_arrays[ids.index(ab)].astype(np.float64).mean(0)
        seqc = np.asarray(
            ablang([[heavy, ""]], mode="seqcoding"), dtype=np.float64
        ).reshape(-1)
        sample_diffs_fmt_mean.append(float(np.max(np.abs(fmt_mean - seqc))))
        sample_diffs_aa_mean.append(float(np.max(np.abs(aa_mean - seqc))))

    semantic = {
        "aa_residue_mean_vs_seqcoding": "NOT_SEMANTICALLY_COMPARABLE",
        "reason": (
            "Official seqcoding averages all formatted tokens including <, >, |; "
            "this asset stores AA-only residue vectors after exact special-token strip."
        ),
        "fmt_token_mean_vs_seqcoding_max_abs_sample5": float(max(sample_diffs_fmt_mean)),
        "aa_mean_vs_seqcoding_max_abs_sample5": float(max(sample_diffs_aa_mean)),
        "pool_parquet_feature_cols": len(feat_cols),
    }

    summary = {
        "n_antibodies": 324,
        "n_heavy": int((qc.chain == "H").sum()),
        "n_light": int((qc.chain == "L").sum()),
        "duplicate_antibody_ids": int(qc.drop_duplicates(["id"]).duplicated(subset=["id"]).any()),
        "duplicate_id_chain_rows": int(qc.duplicated(subset=["id", "chain"]).any()),
        "n_unique_ids": int(qc["id"].nunique()),
        "all_residue_eq_seq_len": bool(
            (qc["residue_length"] == qc["sequence_length"]).all()
        ),
        "finite_100pct": bool(qc["finite"].all()),
        "hidden_dim_constant": bool((qc["hidden_dimension"] == hidden).all()),
        "hidden_dim": hidden,
        "max_len_H": max_h,
        "max_len_L": max_l,
        "mapping_statuses": sorted(qc["mapping_status"].unique().tolist()),
        "unknown_mapping_count": 0,
        "silent_truncation": False,
        "semantic_vs_pooled": semantic,
        "extraction_wall_time_sec": float(time.time() - t0),
        "device": device,
    }
    (OUT / "ablang2_residue_qc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # Split large npy + SHA manifest
    manifest = {"files": {}, "parts": {}, "policy": "deterministic byte split; numeric part suffixes"}
    for name in (
        "ids.npy",
        "heavy_mask.npy",
        "light_mask.npy",
        "heavy_embeddings.npy",
        "light_embeddings.npy",
    ):
        p = OUT / name
        digest = sha256_file(p)
        manifest["files"][name] = {
            "sha256": digest,
            "bytes": p.stat().st_size,
        }
        if name.endswith("_embeddings.npy"):
            parts = split_npy(p)
            manifest["parts"][name] = [
                {"path": x.name, "sha256": sha256_file(x), "bytes": x.stat().st_size}
                for x in parts
            ]

    (OUT / "SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    meta = {
        "model_to_use": MODEL_TO_USE,
        "package": "ablang2",
        "package_version": PACKAGE_VERSION,
        "representation_source": "AbLang.AbRep(tokens).last_hidden_states",
        "official_api_modes_used": ["rescoding_path_equivalent", "AbRep.last_hidden_states"],
        "special_token_rule": (
            "Format with ablang2 add_extra_tokens (<heavy>|<light>); "
            "select only amino-acid character positions; drop <, >, |."
        ),
        "mapping": "EXACT_AA_1TO1_SPECIAL_STRIPPED",
        "n_ids": 324,
        "max_len_H": max_h,
        "max_len_L": max_l,
        "hidden_dim": hidden,
        "dtype_on_disk": "float16",
        "dtype_in_memory": "float32",
        "license_status": "REVIEW_MODEL_OUTPUT",
        "code_license_note": "ablang2 PyPI package 0.2.1",
        "model_weight_license_note": "ablang2-paired weights — review redistribution",
        "derived_output_redistribution": "REVIEW_MODEL_OUTPUT",
        "extraction_wall_time_sec": float(time.time() - t0),
        "device": device,
        "local_regeneration": (
            "python scripts/extract_ablang2_residue_embeddings.py --device cuda:0"
        ),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    audit = f"""# AbLang2 residue embedding audit

## Model identity

- Package: `ablang2=={PACKAGE_VERSION}`
- Checkpoint: `{MODEL_TO_USE}` (`random_init=False`)
- Same family as Stage-2 / bundle AbLang2 seqcoding

## Residue API

Official `rescoding` uses `AbRep(tokens).last_hidden_states` then
`res_to_list(state, formatted_seq)` which keeps **all** formatted tokens
including `<`, `>`, `|`.

This asset keeps **amino-acid positions only**, via exact character-index
mapping of the formatted string (1:1 AA↔token). Special tokens are removed
by index selection, not heuristic pooling.

## QC summary

```json
{json.dumps(summary, indent=2)}
```

## Semantic comparison to pooled AbLang2

{semantic['aa_residue_mean_vs_seqcoding']}: {semantic['reason']}

Fmt-token mean vs seqcoding (sample5 max abs): {semantic['fmt_token_mean_vs_seqcoding_max_abs_sample5']:.3e}

## License

`REVIEW_MODEL_OUTPUT` — do not claim redistribution OK without organizer review.
"""
    (OUT / "ABLANG2_RESIDUE_AUDIT.md").write_text(audit)
    print(json.dumps(summary, indent=2))
    print("DONE", OUT)


if __name__ == "__main__":
    main()
