#!/usr/bin/env python3
"""Materialize ESM-2 Light residue embeddings alongside the heavy-only pack.

Source: organizer_extension/.../cache/esm2_residue/{id}.npz with keys H and L
(facebook/esm2_t33_650M_UR50D, separate H/L encoding, dim 1280).

Writes into top_models_feature_bundle/residue_level/esm2/:
  light_embeddings.npy, light_mask.npy (+ .part0/.part1 split)
and updates metadata.json (chain: HL).

Heavy files are left untouched; ids order matches existing ids.npy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "top_models_feature_bundle"
OUT = BUNDLE / "residue_level" / "esm2"
ESM_CACHE = (
    ROOT
    / "organizer_extension/feature_prospecting/structure_marathon/cache/esm2_residue"
)


def pack_padded(ids, arrays, max_len: int, hidden: int):
    N = len(ids)
    emb = np.zeros((N, max_len, hidden), dtype=np.float16)
    mask = np.zeros((N, max_len), dtype=np.bool_)
    for i, a in enumerate(arrays):
        L = a.shape[0]
        if L > max_len:
            raise RuntimeError(f"seq longer than max_len: {ids[i]} {L}>{max_len}")
        if a.shape[1] != hidden:
            raise RuntimeError(f"hidden mismatch {ids[i]} {a.shape}")
        if not np.isfinite(a.astype(np.float32)).all():
            raise RuntimeError(f"non-finite emb {ids[i]}")
        emb[i, :L] = a.astype(np.float16)
        mask[i, :L] = True
    return emb, mask


def split_npy(path: Path) -> list:
    """Deterministic half/half byte split into .part0 + .part1 (repo convention)."""
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


def materialize_esm2_light_from_cache(
    *,
    ids=None,
    seqs=None,
    out_dir: Path = OUT,
    cache_dir: Path = ESM_CACHE,
    split_parts: bool = True,
) -> dict:
    """Build light_embeddings.npy / light_mask.npy from per-id npz cache.

    Can be called on first need from loaders; returns metadata dict.
    """
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"ESM2 residue cache missing: {cache_dir}")

    if ids is None:
        ids_path = out_dir / "ids.npy"
        if not ids_path.exists():
            raise FileNotFoundError(f"expected existing heavy pack ids: {ids_path}")
        ids = [str(x) for x in np.load(ids_path, allow_pickle=True).tolist()]

    if seqs is None:
        dev = pd.read_csv(BUNDLE / "dev.csv")
        test = pd.read_csv(BUNDLE / "test.csv")
        seqs = pd.concat(
            [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
            ignore_index=True,
        )
        seqs["id"] = seqs["id"].astype(str)
    seqs = seqs.set_index("id")

    arrays = []
    max_len = 0
    for ab in ids:
        z = np.load(cache_dir / f"{ab}.npz")
        if "L" not in z.files:
            raise RuntimeError(f"{ab}.npz missing key L")
        l = z["L"].astype(np.float32)
        expect = len(str(seqs.loc[ab, "light"]))
        if l.shape[0] != expect:
            raise RuntimeError(f"ESM L len mismatch {ab}: {l.shape[0]} vs {expect}")
        arrays.append(l)
        max_len = max(max_len, l.shape[0])

    hidden = int(arrays[0].shape[1])
    emb, mask = pack_padded(ids, arrays, max_len, hidden)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "light_embeddings.npy"
    np.save(emb_path, emb)
    np.save(out_dir / "light_mask.npy", mask)
    if split_parts:
        split_npy(emb_path)

    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "model": meta.get("model", "facebook/esm2_t33_650M_UR50D"),
            "chain": "HL",
            "n_ids": len(ids),
            "max_len_L": max_len,
            "hidden_dim": hidden,
            "source_cache": str(cache_dir),
            "light_materialized": True,
            "light_builder": "developability_drilldown/scripts/build_esm2_light_residue_bundle.py",
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--cache-dir", type=Path, default=ESM_CACHE)
    ap.add_argument("--no-split", action="store_true", help="skip .part0/.part1 split")
    args = ap.parse_args()
    meta = materialize_esm2_light_from_cache(
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        split_parts=not args.no_split,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
