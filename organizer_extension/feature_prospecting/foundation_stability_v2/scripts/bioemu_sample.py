#!/usr/bin/env python3
"""BioEmu-v1.2 chainwise sampling (VH/VL) with resume + sample-count convergence."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
V2 = FP / "foundation_stability_v2"
SPEC = json.loads((V2 / "FOUNDATION_STABILITY_V2_SPEC.json").read_text())
OUT_SAMPLES = V2 / "cache/bioemu_samples"
CACHE_EMB = V2 / "cache/bioemu_embeds"
MODEL_NAME = SPEC["bioemu_v2"]["model_name"]
MAX_REGEN = float(SPEC["bioemu_v2"]["max_regen_factor"])


def load_sequences():
    frames = []
    paths = [
        ROOT / "gate_b3/frozen/organizer/final_population.csv",
        ROOT / "competition/data/distribution/dev.csv",
        ROOT / "competition/data/secret/test.csv",
        ROOT / "competition/data/distribution/test.csv",
    ]
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        idc = cols.get("id")
        hc = cols.get("heavy") or cols.get("h_seq") or cols.get("vh")
        lc = cols.get("light") or cols.get("l_seq") or cols.get("vl")
        if idc and hc and lc:
            frames.append(df[[idc, hc, lc]].rename(columns={idc: "id", hc: "heavy", lc: "light"}))
    if not frames:
        raise FileNotFoundError("No sequence table with heavy/light found")
    seq = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    return seq.set_index("id")


def find_local_a3m(ab_id: str, chain: str) -> Path | None:
    p = V2 / "cache/bioemu_a3m" / f"{ab_id}_{chain}.a3m"
    return p if p.exists() else None


def sample_chain(sequence: str, out_dir: Path, num_samples: int, seed: int, ab_id: str = "", chain: str = ""):
    from bioemu.sample import main as bioemu_sample

    out_dir.mkdir(parents=True, exist_ok=True)
    from bioemu.sample import count_samples_in_output_dir

    have = 0
    try:
        have = int(count_samples_in_output_dir(out_dir))
    except Exception:
        pass
    if have >= num_samples:
        return {"requested": num_samples, "valid": have, "status": "CACHED", "msa": "cached"}

    a3m = find_local_a3m(ab_id, chain) if ab_id else None
    seq_arg = str(a3m) if a3m is not None else sequence
    msa_mode = "local_boltz_a3m" if a3m is not None else "colabfold_remote"

    attempts = 0
    while have < num_samples and attempts < 3:
        need = max(num_samples - have, num_samples)
        need = min(int(np.ceil(num_samples * MAX_REGEN)), max(need, num_samples))
        bioemu_sample(
            sequence=seq_arg,
            num_samples=need,
            output_dir=str(out_dir),
            model_name=MODEL_NAME,
            cache_embeds_dir=str(CACHE_EMB),
            filter_samples=True,
            base_seed=seed + attempts * 10007,
            batch_size_100=40,
        )
        have = int(count_samples_in_output_dir(out_dir))
        attempts += 1
        if have >= num_samples:
            break
    status = "SUCCESS" if have >= num_samples else "INSUFFICIENT_VALID"
    return {
        "requested": num_samples,
        "valid": have,
        "status": status,
        "attempts": attempts,
        "msa": msa_mode,
    }


def main():
    mode = "convergence" if "--convergence" in sys.argv else "full"
    n_req = 64 if mode == "convergence" else None
    if "--n" in sys.argv:
        n_req = int(sys.argv[sys.argv.index("--n") + 1])

    seqs = load_sequences()
    if mode == "convergence":
        ids = pd.read_csv(V2 / "pilots/BIOEMU_CONVERGENCE_ANTIBODIES.csv").id.tolist()
        n_req = 64
    else:
        # full cohort from crosswalk
        ids = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv").id.tolist()
        if n_req is None:
            if (V2 / "BIOEMU_FROZEN_N.txt").exists():
                n_req = int((V2 / "BIOEMU_FROZEN_N.txt").read_text().strip())
            else:
                n_req = 32  # temporary until decision; full mode should not run before freeze
                print("WARNING: N not frozen; defaulting", n_req, flush=True)

    if "--shard" in sys.argv:
        spec = sys.argv[sys.argv.index("--shard") + 1]
        si, sn = map(int, spec.split("/"))
        ids = [x for j, x in enumerate(ids) if j % sn == si]
        print(f"BIOEMU_SHARD {si}/{sn} n={len(ids)}", flush=True)

    OUT_SAMPLES.mkdir(parents=True, exist_ok=True)
    CACHE_EMB.mkdir(parents=True, exist_ok=True)
    log_rows = []
    log_path = V2 / "cache/bioemu_samples/sample_log.csv"

    for ab_id in ids:
        if ab_id not in seqs.index:
            log_rows.append({"id": ab_id, "chain": "VH", "status": "MISSING_SEQ"})
            continue
        heavy = str(seqs.loc[ab_id, "heavy"])
        light = str(seqs.loc[ab_id, "light"])
        for chain, sequence in [("VH", heavy), ("VL", light)]:
            out_dir = OUT_SAMPLES / f"{ab_id}_{chain}"
            print(f"BIOEMU {MODEL_NAME} {ab_id} {chain} N={n_req}", flush=True)
            try:
                info = sample_chain(
                    sequence, out_dir, n_req, seed=stable_seed(ab_id, chain), ab_id=ab_id, chain=chain
                )
                log_rows.append({"id": ab_id, "chain": chain, **info, "model": MODEL_NAME})
            except Exception as e:
                log_rows.append(
                    {
                        "id": ab_id,
                        "chain": chain,
                        "status": f"FAIL:{type(e).__name__}:{e}",
                        "requested": n_req,
                        "valid": 0,
                        "model": MODEL_NAME,
                    }
                )
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print("DONE", pd.DataFrame(log_rows).status.value_counts().to_dict(), flush=True)


def stable_seed(ab_id, chain):
    import hashlib

    return int(hashlib.sha256(f"{ab_id}|{chain}|bioemu-v1.2".encode()).hexdigest()[:8], 16)


if __name__ == "__main__":
    main()
