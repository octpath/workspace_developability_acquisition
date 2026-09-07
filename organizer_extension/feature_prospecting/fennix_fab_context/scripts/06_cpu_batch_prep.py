#!/usr/bin/env python3
"""OpenMM CPU Fab prep with bounded CPU budget (default 16 threads + affinity)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
SCRIPT = CTX / "scripts/01_fab_disulfide_prep.py"


def cpu_budget_env(n_threads: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OPENMM_DEFAULT_PLATFORM"] = "CPU"
    env["FENNIX_PREP_PLATFORM"] = "CPU"
    env["OPENMM_CPU_THREADS"] = str(n_threads)
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[k] = str(n_threads)
    # Avoid OpenMP nested oversubscription
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "cores"
    return env


def affinity_list(n: int) -> str:
    # Fixed first N logical CPUs
    return ",".join(str(i) for i in range(n))


def run_ids(ids: list[str], batch_tag: str, attempt: int = 1, n_threads: int = 16, use_taskset: bool = True):
    py = ROOT / ".venv_b1/bin/python"
    env = cpu_budget_env(n_threads)
    env["FENNIX_PREP_BATCH"] = batch_tag
    env["FENNIX_PREP_ATTEMPT"] = str(attempt)
    inner = [str(py), str(SCRIPT), "--ids", *ids]
    if use_taskset:
        cmd = ["taskset", "-c", affinity_list(n_threads), *inner]
    else:
        cmd = inner
    t0 = time.time()
    print(
        f"BATCH {batch_tag} n={len(ids)} threads={n_threads} taskset={use_taskset} start",
        flush=True,
    )
    try:
        free = subprocess.check_output(["bash", "-lc", "free -h | head -2; uptime; nproc"], text=True)
        print(free, flush=True)
    except Exception:
        pass
    r = subprocess.run(cmd, env=env)
    print(f"BATCH {batch_tag} exit={r.returncode} wall={time.time()-t0:.1f}s", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--batch-tag", default="cpu_manual")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("HOST_CPU_BUDGET", "16")))
    ap.add_argument("--no-taskset", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run_ids(args.ids, args.batch_tag, args.attempt, args.threads, not args.no_taskset))


if __name__ == "__main__":
    main()
