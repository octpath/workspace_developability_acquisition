#!/usr/bin/env python3
"""BioEmu-v1.2 sampling for VL+CL (primary) and VH+CH1 (UNPAIRED_HEAVY_CONTEXT).

CRITICAL: count_samples_in_output_dir counts UNFILTERED npz frames.
Physicality filtering is applied when writing samples.xtc. This script
oversamples until filtered XTC frames >= requested N (filter stays ON).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "bioemu_constant_context"
SPEC = json.loads((CTX / "BIOEMU_CONTEXT_SPEC.json").read_text())
MAN = pd.read_csv(CTX / "BIOEMU_CONTEXT_SEQUENCE_MANIFEST.csv").set_index("id")
OUT_SAMPLES = CTX / "cache/bioemu_samples"
CACHE_EMB = CTX / "cache/bioemu_embeds"
A3M_DIR = CTX / "cache/bioemu_a3m"
MODEL_NAME = SPEC["model_name"]
LOG = CTX / "cache/bioemu_samples/sample_log.csv"

ARMS = {
    "LIGHT": ("VLCL", "light_fab_seq"),
    "HEAVY": ("VHCH1", "heavy_fab_seq"),
}

# Oversample budget: unfiltered npz cap = target_physical * this factor
MAX_OVERSAMPLE_FACTOR = 25
CHUNK = 128  # npz increment per bioemu call


def ensure_singleseq_a3m(ab_id: str, arm: str, sequence: str) -> Path:
    A3M_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ARMS[arm][0]
    p = A3M_DIR / f"{ab_id}_{suffix}.a3m"
    if not p.exists():
        p.write_text(f">{ab_id}_{suffix}\n{sequence}\n")
    return p


def count_npz(out_dir: Path) -> int:
    from bioemu.sample import count_samples_in_output_dir

    try:
        return int(count_samples_in_output_dir(out_dir))
    except Exception:
        return 0


def count_physical(out_dir: Path) -> int:
    top = out_dir / "topology.pdb"
    xtc = out_dir / "samples.xtc"
    if not (top.exists() and xtc.exists()):
        return 0
    try:
        return int(md.load(str(xtc), top=str(top)).n_frames)
    except Exception:
        return 0


def sample_until_physical(sequence: str, out_dir: Path, n_physical: int, seed: int, ab_id: str, arm: str):
    from bioemu.sample import main as bioemu_sample

    out_dir.mkdir(parents=True, exist_ok=True)
    a3m = ensure_singleseq_a3m(ab_id, arm, sequence)
    t0 = time.time()
    max_npz = n_physical * MAX_OVERSAMPLE_FACTOR
    attempts = 0
    last_err = ""

    phys = count_physical(out_dir)
    npz = count_npz(out_dir)
    if phys >= n_physical:
        return {
            "requested_physical": n_physical,
            "valid_physical": phys,
            "npz_unfiltered": npz,
            "rejection_frac": 1.0 - phys / max(npz, 1),
            "status": "CACHED",
            "msa": "singleseq_a3m",
            "runtime_s": 0.0,
            "attempts": 0,
            "error": "",
        }

    while phys < n_physical and npz < max_npz and attempts < 30:
        # request more unfiltered samples
        target_npz = min(max_npz, npz + CHUNK)
        if target_npz <= npz:
            target_npz = npz + CHUNK
        try:
            bioemu_sample(
                sequence=str(a3m),
                num_samples=target_npz,
                output_dir=str(out_dir),
                model_name=MODEL_NAME,
                cache_embeds_dir=str(CACHE_EMB),
                filter_samples=True,
                base_seed=seed + attempts * 10007,
                batch_size_100=20,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}:{e}"
            print("ERROR", ab_id, arm, last_err, flush=True)
        npz = count_npz(out_dir)
        phys = count_physical(out_dir)
        attempts += 1
        print(
            f"  progress {ab_id} {arm}: phys={phys}/{n_physical} npz={npz}/{max_npz} rej={1-phys/max(npz,1):.2f}",
            flush=True,
        )
        if phys >= n_physical:
            break

    status = "SUCCESS" if phys >= n_physical else "INSUFFICIENT_PHYSICAL"
    if last_err and status != "SUCCESS":
        status = "FAIL"
    return {
        "requested_physical": n_physical,
        "valid_physical": phys,
        "npz_unfiltered": npz,
        "rejection_frac": float(1.0 - phys / max(npz, 1)),
        "status": status,
        "msa": "singleseq_a3m",
        "runtime_s": time.time() - t0,
        "attempts": attempts,
        "error": last_err,
    }


def append_log(row: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    hdr = not LOG.exists() or LOG.stat().st_size == 0
    # use a new log for oversample phase
    df.to_csv(LOG, mode="a", header=hdr, index=False)


def main():
    mode = "convergence" if "--convergence" in sys.argv else "full"
    arms = ["LIGHT", "HEAVY"]
    if "--light-only" in sys.argv:
        arms = ["LIGHT"]
    if "--heavy-only" in sys.argv:
        arms = ["HEAVY"]

    n_phys = 64 if mode == "convergence" else None
    if "--n" in sys.argv:
        n_phys = int(sys.argv[sys.argv.index("--n") + 1])

    if mode == "convergence":
        ids = pd.read_csv(CTX / "pilots/BIOEMU_CONTEXT_CONVERGENCE_ANTIBODIES.csv").id.tolist()
        n_phys = 64
    else:
        ids = MAN.index.tolist()
        if n_phys is None:
            light_n = CTX / "BIOEMU_CONTEXT_FROZEN_N_LIGHT.txt"
            n_phys = int(light_n.read_text().strip()) if light_n.exists() else 16

    if "--shard" in sys.argv:
        spec = sys.argv[sys.argv.index("--shard") + 1]
        si, sn = map(int, spec.split("/"))
        ids = [x for i, x in enumerate(ids) if i % sn == si]

    # rotate log for this run
    run_log = CTX / "cache/bioemu_samples/sample_log_physical.csv"
    global LOG
    LOG = run_log

    print(f"mode={mode} n_physical={n_phys} arms={arms} n_ids={len(ids)} max_oversample_x={MAX_OVERSAMPLE_FACTOR}", flush=True)
    summary = []
    for i, ab in enumerate(ids):
        row = MAN.loc[ab]
        for arm in arms:
            suffix, scol = ARMS[arm]
            n_use = n_phys
            if mode == "full":
                nf = CTX / f"BIOEMU_CONTEXT_FROZEN_N_{'LIGHT' if arm == 'LIGHT' else 'HEAVY'}.txt"
                if nf.exists():
                    n_use = int(nf.read_text().strip())
            out_dir = OUT_SAMPLES / f"{ab}_{suffix}"
            seq = str(row[scol])
            seed = abs(hash(f"{ab}_{suffix}_phys")) % (2**31 - 1)
            print(f"[{i+1}/{len(ids)}] {ab} {arm} len={len(seq)} target_phys={n_use}", flush=True)
            res = sample_until_physical(seq, out_dir, n_use, seed, ab, arm)
            rec = {
                "id": ab,
                "arm": arm,
                "suffix": suffix,
                "seq_len": len(seq),
                "light_locus": row.light_locus,
                **res,
            }
            append_log(rec)
            summary.append(rec)
            print(rec, flush=True)

    sdf = pd.DataFrame(summary)
    print("DONE", sdf.groupby(["arm", "status"]).size().to_dict() if len(sdf) else {}, flush=True)
    if len(sdf):
        print("rejection_frac median", sdf.groupby("arm").rejection_frac.median().to_dict(), flush=True)


if __name__ == "__main__":
    main()
