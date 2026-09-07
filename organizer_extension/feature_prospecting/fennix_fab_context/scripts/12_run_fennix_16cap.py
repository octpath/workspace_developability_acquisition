#!/usr/bin/env python3
"""FeNNix full-cohort runner with <=16 effective CPU cores (affinity + thread caps).

Does not change scientific feature definitions — only host resource limits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
FENNOL = ROOT / "organizer_extension/feature_prospecting/foundation_stability/envs/fennol/bin/python"
SCRIPT = CTX / "scripts/02_fennix_fab_curvature.py"
BUDGET_PATH = CTX / "cache/HOST_CPU_BUDGET.json"
LOG = CTX / "cache/logs/fennix_full_16cap.log"


def _python_worker_alive(script_name: str) -> bool:
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        toks = raw.split()
        if not toks:
            continue
        a0 = toks[0]
        if not (a0.endswith("python") or "/bin/python" in a0):
            continue
        if script_name in raw:
            return True
    return False


def main():
    st = json.loads(BUDGET_PATH.read_text()) if BUDGET_PATH.exists() else {"budget_cores": 16}
    if st.get("status") == "HOST_STABILITY_BLOCK":
        print("HOST_STABILITY_BLOCK — refuse FeNNix", flush=True)
        raise SystemExit(2)
    # Do not run if prep/FeNNix still active; clear stale stamps after crash/reboot
    if st.get("active_run_stamp"):
        if _python_worker_alive("02_fennix_fab_curvature.py") or _python_worker_alive(
            "01_fab_disulfide_prep.py"
        ) or _python_worker_alive("06_cpu_batch_prep.py") or _python_worker_alive(
            "11_run_cpu_prep_stable.py"
        ):
            print("Refuse: another controlled CPU run is stamped active", flush=True)
            raise SystemExit(3)
        print(
            f"Clearing stale active_run_stamp={st.get('active_run_stamp')} (no worker alive)",
            flush=True,
        )
        st["active_run_stamp"] = None
        BUDGET_PATH.write_text(json.dumps(st, indent=2) + "\n")
    n = int(st.get("budget_cores", 16))
    n = min(n, 16)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["OMP_NUM_THREADS"] = str(n)
    env["MKL_NUM_THREADS"] = str(n)
    env["OPENBLAS_NUM_THREADS"] = str(n)
    env["NUMEXPR_NUM_THREADS"] = str(n)
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count={n}"
    # PyTorch / torch intra-op if used by dependency chain
    env["TORCH_NUM_THREADS"] = str(n)
    env["PYTHONUNBUFFERED"] = "1"
    affinity = ",".join(str(i) for i in range(n))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", affinity, str(FENNOL), str(SCRIPT), *sys.argv[1:]]
    print(f"FeNNix start cores={n} affinity={affinity} {datetime.now(timezone.utc).isoformat()}", flush=True)
    st["active_run_stamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    st["fennix_cpu_budget"] = n
    BUDGET_PATH.write_text(json.dumps(st, indent=2) + "\n")
    with LOG.open("a", buffering=1) as fh:
        fh.write(f"\n=== START {datetime.now(timezone.utc).isoformat()} cores={n} ===\n")
        r = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    st = json.loads(BUDGET_PATH.read_text())
    st["active_run_stamp"] = None
    BUDGET_PATH.write_text(json.dumps(st, indent=2) + "\n")
    print(f"FeNNix exit={r.returncode}", flush=True)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
