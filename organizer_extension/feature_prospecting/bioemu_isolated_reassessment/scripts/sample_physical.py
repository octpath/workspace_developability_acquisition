#!/usr/bin/env python3
"""Oversample isolated VH/VL until N PHYSICAL (post official filter) frames exist.

Reuses v2 NPZ by copying into a new cache (does not modify v2 in place).
Physical count = mdtraj frames in samples.xtc after filter_samples=True conversion.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
V2 = FP / "foundation_stability_v2"
R = FP / "bioemu_isolated_reassessment"
SPEC = json.loads((R / "BIOEMU_ISOLATED_REASSESS_SPEC.json").read_text())
OUT = R / "cache/bioemu_samples"
V2_SAMPLES = V2 / "cache/bioemu_samples"
A3M_V2 = R / "cache/bioemu_a3m_v2"
EMB = R / "cache/bioemu_embeds"
EMB.mkdir(parents=True, exist_ok=True)
# prefer shared embeds
if (R / "cache/bioemu_embeds_v2").exists():
    EMB = R / "cache/bioemu_embeds_v2"

MODEL = SPEC["model_name"]
CHUNK = 64  # overridden in main for small Nphys
MAX_NPZ_FACTOR = 40  # safety cap vs target physical
LOG = R / "cache/bioemu_samples/sample_physical_log.csv"


def load_sequences():
    frames = []
    for p in [
        ROOT / "gate_b3/frozen/organizer/final_population.csv",
        ROOT / "competition/data/distribution/dev.csv",
        ROOT / "competition/data/secret/test.csv",
    ]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        idc, hc, lc = cols.get("id"), cols.get("heavy") or cols.get("vh"), cols.get("light") or cols.get("vl")
        if idc and hc and lc:
            frames.append(df[[idc, hc, lc]].rename(columns={idc: "id", hc: "heavy", lc: "light"}))
    return pd.concat(frames).drop_duplicates("id").set_index("id")


def count_npz(d: Path) -> int:
    from bioemu.sample import count_samples_in_output_dir

    try:
        return int(count_samples_in_output_dir(d))
    except Exception:
        return 0


def count_physical(d: Path) -> int:
    top, xtc = d / "topology.pdb", d / "samples.xtc"
    if not (top.exists() and xtc.exists()):
        return 0
    try:
        return int(md.load(str(xtc), top=str(top)).n_frames)
    except Exception:
        return 0


def seed_from_v2(ab: str, chain: str, dest: Path):
    """Copy existing v2 NPZ (+fasta) into dest without touching v2."""
    src = V2_SAMPLES / f"{ab}_{chain}"
    if not src.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    if count_npz(dest) > 0:
        return
    for p in src.glob("batch_*.npz"):
        shutil.copy2(p, dest / p.name)
    if (src / "sequence.fasta").exists():
        shutil.copy2(src / "sequence.fasta", dest / "sequence.fasta")
    # do not copy old xtc/topology — force reconvert from all npz on next sample call


def find_a3m(ab: str, chain: str) -> Path | None:
    p = A3M_V2 / f"{ab}_{chain}.a3m"
    return p if p.exists() else None


def ensure_physical(ab: str, chain: str, sequence: str, n_phys: int, seed: int, chunk: int | None = None) -> dict:
    from bioemu.sample import main as bioemu_sample

    chunk = int(chunk or CHUNK)
    dest = OUT / f"{ab}_{chain}"
    seed_from_v2(ab, chain, dest)
    dest.mkdir(parents=True, exist_ok=True)

    a3m = find_a3m(ab, chain)
    if a3m is None:
        a3m = R / "cache/bioemu_a3m" / f"{ab}_{chain}.a3m"
        a3m.parent.mkdir(parents=True, exist_ok=True)
        if not a3m.exists():
            a3m.write_text(f">{ab}_{chain}\n{sequence}\n")

    t0 = time.time()
    attempts = 0
    max_npz = n_phys * MAX_NPZ_FACTOR
    phys = count_physical(dest)
    npz = count_npz(dest)

    # Already have enough physical frames (e.g. pilot reuse)
    if phys >= n_phys and npz > 0:
        return {
            "id": ab,
            "chain": chain,
            "requested_physical": n_phys,
            "physical": phys,
            "npz": npz,
            "pass_rate": float(phys / max(npz, 1)),
            "status": "SUCCESS",
            "attempts": 0,
            "runtime_s": time.time() - t0,
            "msa": "v2_a3m" if find_a3m(ab, chain) else "singleseq",
        }

    # If we have npz but no/stale xtc, force a reconvert by asking bioemu for current npz count
    if npz > 0 and phys < n_phys:
        # trigger conversion of existing npz first
        try:
            bioemu_sample(
                sequence=str(a3m),
                num_samples=npz,  # no new samples; reconvert
                output_dir=str(dest),
                model_name=MODEL,
                cache_embeds_dir=str(EMB),
                filter_samples=True,
                base_seed=seed,
                batch_size_100=40,
            )
        except Exception as e:
            # if num_samples==npz and all exist, main may just reconvert — ok
            print("reconvert note", ab, chain, e, flush=True)
        phys = count_physical(dest)
        npz = count_npz(dest)

    while phys < n_phys and npz < max_npz and attempts < 40:
        rate = max(phys / max(npz, 1), 0.05)
        need = int(np.ceil((n_phys - phys) / rate)) + 4
        target_npz = min(max_npz, max(npz + chunk, npz + need))
        target_npz = max(target_npz, npz + chunk)
        print(f"  {ab}_{chain}: phys={phys}/{n_phys} npz={npz} -> request_npz={target_npz}", flush=True)
        try:
            bioemu_sample(
                sequence=str(a3m),
                num_samples=target_npz,
                output_dir=str(dest),
                model_name=MODEL,
                cache_embeds_dir=str(EMB),
                filter_samples=True,
                base_seed=seed + attempts * 10007,
                batch_size_100=40,
            )
        except Exception as e:
            print("ERROR", ab, chain, e, flush=True)
        npz = count_npz(dest)
        phys = count_physical(dest)
        attempts += 1

    status = "SUCCESS" if phys >= n_phys else "INSUFFICIENT_PHYSICAL"
    return {
        "id": ab,
        "chain": chain,
        "requested_physical": n_phys,
        "physical": phys,
        "npz": npz,
        "pass_rate": float(phys / max(npz, 1)),
        "status": status,
        "attempts": attempts,
        "runtime_s": time.time() - t0,
        "msa": "v2_a3m" if find_a3m(ab, chain) else "singleseq",
    }


def append_log(row: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(LOG, mode="a", header=not LOG.exists() or LOG.stat().st_size == 0, index=False)


def main():
    mode = "pilot" if "--pilot" in sys.argv else "full"
    n_phys = int(sys.argv[sys.argv.index("--n-phys") + 1]) if "--n-phys" in sys.argv else 64
    chains = ["VH", "VL"]
    if "--vh-only" in sys.argv:
        chains = ["VH"]
    if "--vl-only" in sys.argv:
        chains = ["VL"]

    seqs = load_sequences()
    if mode == "pilot":
        ids = pd.read_csv(R / "pilots/BIOEMU_ISOLATED_REASSESS_PILOT.csv").id.tolist()
    else:
        ids = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv").id.tolist()
        nfile = R / "BIOEMU_ISOLATED_FROZEN_NPHYS.txt"
        if nfile.exists() and "--n-phys" not in sys.argv:
            n_phys = int(nfile.read_text().strip())

    if "--shard" in sys.argv:
        si, sn = map(int, sys.argv[sys.argv.index("--shard") + 1].split("/"))
        ids = [x for i, x in enumerate(ids) if i % sn == si]

    # Smaller chunks when target physical count is modest (full-cohort Nphys=8)
    chunk = 16 if n_phys <= 16 else 64
    print(f"mode={mode} n_phys={n_phys} chunk={chunk} chains={chains} n_ids={len(ids)}", flush=True)
    rows = []
    for i, ab in enumerate(ids):
        for ch in chains:
            seq = str(seqs.loc[ab, "heavy" if ch == "VH" else "light"])
            seed = abs(hash(f"{ab}_{ch}_reassess")) % (2**31 - 1)
            print(f"[{i+1}/{len(ids)}] {ab} {ch} len={len(seq)}", flush=True)
            rec = ensure_physical(ab, ch, seq, n_phys, seed, chunk=chunk)
            append_log(rec)
            rows.append(rec)
            print(rec, flush=True)
    sdf = pd.DataFrame(rows)
    print("DONE", sdf.groupby(["chain", "status"]).size().to_dict() if len(sdf) else {}, flush=True)
    if len(sdf):
        print("pass_rate median", sdf.groupby("chain").pass_rate.median().to_dict(), flush=True)


if __name__ == "__main__":
    main()
