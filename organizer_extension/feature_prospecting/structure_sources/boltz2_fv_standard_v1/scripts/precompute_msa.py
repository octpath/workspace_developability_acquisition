#!/usr/bin/env python3
"""Precompute paired+unpaired MSAs via ColabFold API (same logic as boltz.main.compute_msa).

Resume-capable. Serial requests with backoff (ColabFold API is rate-limited / flaky).
Saves CSV per chain under msa/<id>/{H,L}.csv and updates YAML msa paths.
Does not use TmApp/HIC labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from boltz.data import const
from boltz.data.msa.mmseqs2 import run_mmseqs2

ROOT = Path("/workspace_developability_acquisition")
BASE = ROOT / "organizer_extension/feature_prospecting/structure_sources/boltz2_fv_standard_v1"
INP = BASE / "inputs"
MSA = BASE / "msa"
STATUS = BASE / "msa_status.jsonl"
MSA_URL = "https://api.colabfold.com"
MSA_PAIR = "greedy"
MAX_RETRY = 12
SLEEP_BASE = 15
INTER_JOB_SLEEP = 20


def append_status(rec: dict) -> None:
    with STATUS.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def msa_ready(aid: str) -> bool:
    h = MSA / aid / "H.csv"
    l = MSA / aid / "L.csv"
    return h.exists() and l.exists() and h.stat().st_size > 50 and l.stat().st_size > 50


def write_yaml(aid: str, heavy: str, light: str) -> None:
    yaml = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: H\n"
        f"      sequence: {heavy}\n"
        f"      msa: {MSA / aid / 'H.csv'}\n"
        "  - protein:\n"
        "      id: L\n"
        f"      sequence: {light}\n"
        f"      msa: {MSA / aid / 'L.csv'}\n"
    )
    (INP / f"{aid}.yaml").write_text(yaml)


def wipe_tmp(msa_dir: Path) -> None:
    if not msa_dir.exists():
        return
    for p in msa_dir.glob("*_tmp*"):
        shutil.rmtree(p, ignore_errors=True)


def compute_one(aid: str, heavy: str, light: str) -> None:
    data = {"H": heavy, "L": light}
    msa_dir = MSA / aid
    wipe_tmp(msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)
    pairing_mode = "paired+unpaired"
    try:
        paired_msas = run_mmseqs2(
            list(data.values()),
            str(msa_dir / f"{aid}_paired_tmp"),
            use_env=True,
            use_pairing=True,
            host_url=MSA_URL,
            pairing_strategy=MSA_PAIR,
        )
    except Exception as e:
        pairing_mode = f"unpaired_only_after_pair_fail:{type(e).__name__}"
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        str(msa_dir / f"{aid}_unpaired_tmp"),
        use_env=True,
        use_pairing=False,
        host_url=MSA_URL,
        pairing_strategy=MSA_PAIR,
    )
    for idx, name in enumerate(data):
        if paired_msas[idx]:
            paired = paired_msas[idx].strip().splitlines()
            paired = paired[1::2]
            paired = paired[: const.max_paired_seqs]
            keys = [i for i, s in enumerate(paired) if s != "-" * len(s)]
            paired = [s for s in paired if s != "-" * len(s)]
        else:
            paired, keys = [], []
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]
        if not paired and not unpaired:
            raise RuntimeError("empty MSA after paired/unpaired fetch")
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]
        out = msa_dir / f"{name}.csv"
        out.write_text("\n".join(csv_str) + "\n")
        if out.stat().st_size < 50:
            raise RuntimeError(f"MSA csv too small for {name}")
    write_yaml(aid, heavy, light)
    meta = {
        "id": aid,
        "status": "SUCCESS",
        "pairing_mode": pairing_mode,
        "H_sha256": sha256_file(msa_dir / "H.csv"),
        "L_sha256": sha256_file(msa_dir / "L.csv"),
        "H_nlines": sum(1 for _ in (msa_dir / "H.csv").open()) - 1,
        "L_nlines": sum(1 for _ in (msa_dir / "L.csv").open()) - 1,
        "msa_server_url": MSA_URL,
        "msa_pairing_strategy": MSA_PAIR,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (msa_dir / "msa_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    append_status(meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", type=str, default="")
    args = ap.parse_args()

    MSA.mkdir(parents=True, exist_ok=True)
    seqs = {}
    for csvp in [
        ROOT / "competition/data/distribution/dev.csv",
        ROOT / "competition/data/distribution/test_features.csv",
    ]:
        df = pd.read_csv(csvp)
        for _, r in df.iterrows():
            seqs[r["id"]] = (r["heavy"], r["light"])

    ids = sorted(seqs)
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.limit:
        ids = ids[: args.limit]

    for aid in ids:
        if msa_ready(aid):
            heavy, light = seqs[aid]
            write_yaml(aid, heavy, light)
            print(f"SKIP {aid} (msa exists)", flush=True)
            continue
        heavy, light = seqs[aid]
        ok = False
        for attempt in range(1, MAX_RETRY + 1):
            try:
                print(f"MSA {aid} attempt={attempt}", flush=True)
                t0 = time.time()
                compute_one(aid, heavy, light)
                print(f"  OK {aid} in {time.time()-t0:.1f}s", flush=True)
                ok = True
                break
            except Exception as e:
                wipe_tmp(MSA / aid)
                append_status(
                    {
                        "id": aid,
                        "status": "FAIL",
                        "attempt": attempt,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-1500:],
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"  FAIL {aid}: {e}", flush=True)
                time.sleep(min(180, SLEEP_BASE * attempt))
        if not ok:
            print(f"GIVEUP {aid}", flush=True)
        else:
            time.sleep(INTER_JOB_SLEEP)

    n = sum(1 for i in ids if msa_ready(i))
    print(f"MSA ready {n}/{len(ids)}")


if __name__ == "__main__":
    main()
