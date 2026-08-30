#!/usr/bin/env python3
"""Wait for ESMFold + structure features, then run modeling → diagnostics → GATE_B2_FINAL."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
B2 = ROOT / "gate_b2"
PY = str(ROOT / ".venv_b1" / "bin" / "python")
LOGS = B2 / "logs"


def run(script: str, log: str, timeout=None):
    logp = LOGS / log
    print(f"START {script}", flush=True)
    with open(logp, "a") as f:
        f.write(f"\n=== RUN {script} {time.ctime()} ===\n")
    with open(logp, "a") as f:
        p = subprocess.run(
            [PY, "-u", str(B2 / "scripts" / script)],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            env={
                **dict(**{k: v for k, v in __import__("os").environ.items()}),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        )
    print(f"DONE {script} rc={p.returncode}", flush=True)
    return p.returncode


def wait_file(path: Path, label: str, poll=30):
    while not path.exists():
        print(f"WAIT {label} …", flush=True)
        time.sleep(poll)
    print(f"HAVE {label}", flush=True)


def wait_esmfold_done(poll=60):
    man = B2 / "cache" / "structures" / "esmfold_native_manifest.csv"
    # also detect process exit
    while True:
        if man.exists():
            import pandas as pd

            df = pd.read_csv(man)
            n_ok = int((df.get("success") == True).sum()) if "success" in df.columns else len(df)
            print(f"ESMFOLD_MANIFEST n={len(df)} ok={n_ok}", flush=True)
            if n_ok >= 300:
                return
        # progress from pdb count
        n = len(list((B2 / "cache" / "structures" / "esmfold_native").glob("*.pdb")))
        print(f"WAIT ESMFold pdbs={n}/370", flush=True)
        time.sleep(poll)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    # Wait for current ABB structure job or skip if cached
    abb = B2 / "cache" / "structure_features" / "abb_sasa_rasa_patch.csv"
    while not abb.exists() or len(open(abb).readlines()) < 300:
        print(f"WAIT ABB feats exists={abb.exists()}", flush=True)
        time.sleep(60)

    wait_esmfold_done()
    # Structure features for ESMFold (and refresh params)
    rc = run("03_structure_features.py", "03_struct_feats_esmn.log")
    if rc != 0:
        sys.exit(rc)
    run("06_structure_compare.py", "06_structure_compare.log")
    rc = run("04_modeling.py", "04_modeling.log")
    if rc != 0:
        print("MODELING_FAILED", rc)
    # residual + ensemble extension if present
    if (B2 / "scripts" / "04b_residuals_ensembles.py").exists():
        run("04b_residuals_ensembles.py", "04b_residuals.log")
    run("05_diagnostics_final.py", "05_diagnostics.log")
    run("07_write_gate_b2_final.py", "07_final.log")
    print("ORCHESTRATOR_DONE")


if __name__ == "__main__":
    main()
