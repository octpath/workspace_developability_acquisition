#!/usr/bin/env python3
"""Precompute BioEmu ColabFold embeds from local a3m (parallel CPU-friendly sequential with resume)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
V2 = ROOT / "organizer_extension/feature_prospecting/foundation_stability_v2"
A3M = V2 / "cache/bioemu_a3m"
CACHE = V2 / "cache/bioemu_embeds"


def main():
    from bioemu.get_embeds import get_colabfold_embeds, shahexencode

    seq = pd.read_csv(ROOT / "gate_b3/frozen/organizer/final_population.csv")
    if "--shard" in sys.argv:
        si, sn = map(int, sys.argv[sys.argv.index("--shard") + 1].split("/"))
        seq = seq.iloc[si::sn]
        print(f"shard {si}/{sn} n={len(seq)}", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    n_done = n_new = 0
    for _, r in seq.iterrows():
        for chain, s in [("VH", str(r.heavy)), ("VL", str(r.light))]:
            h = shahexencode(s)
            if (CACHE / f"{h}_single.npy").exists() and (CACHE / f"{h}_pair.npy").exists():
                n_done += 1
                continue
            a3m = A3M / f"{r.id}_{chain}.a3m"
            print(f"EMBED {r.id} {chain}", flush=True)
            get_colabfold_embeds(
                seq=s,
                cache_embeds_dir=str(CACHE),
                msa_file=str(a3m) if a3m.exists() else None,
            )
            n_new += 1
    print("DONE", {"cached": n_done, "new": n_new}, flush=True)


if __name__ == "__main__":
    main()
