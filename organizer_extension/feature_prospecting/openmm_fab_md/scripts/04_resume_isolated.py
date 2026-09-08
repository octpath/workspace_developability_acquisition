#!/usr/bin/env python3
"""Resume-safe OpenMM DEV supervisor: one antibody per subprocess."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd

OUT = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/openmm_fab_md")
RES = OUT / "results"
CACHE = OUT / "cache" / "dev"
PY = Path("/workspace_developability_acquisition/.mamba/envs/openmm_cuda/bin/python")
WORKER = OUT / "scripts/_worker_one_ab.py"
LOG = RES / "dev_run_isolated.log"


def assemble():
    freeze = json.loads((OUT / "OPENMM_MD_FEATURE_FREEZE.json").read_text())
    ids = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]
    rows, qcs = [], []
    fails = []
    for aid in ids:
        if (CACHE / aid / ".complete").exists() and (CACHE / aid / "features.json").exists():
            rows.append(json.loads((CACHE / aid / "features.json").read_text()))
            if (CACHE / aid / "qc.json").exists():
                qcs.append(json.loads((CACHE / aid / "qc.json").read_text()))
        else:
            fails.append(aid)
    cols = ["id"] + freeze["feature_names"]
    df = pd.DataFrame(rows)[cols].sort_values("id").reset_index(drop=True)
    df.to_parquet(RES / "OPENMM_FAB_MD_FEATURES_DEV.parquet", index=False)
    df.to_csv(RES / "OPENMM_FAB_MD_FEATURES_DEV.csv", index=False)
    (OUT / "OPENMM_FAB_MD_FEATURES_DEV.parquet").write_bytes((RES / "OPENMM_FAB_MD_FEATURES_DEV.parquet").read_bytes())
    qcdf = pd.DataFrame(qcs)
    qcdf.to_csv(RES / "OPENMM_FAB_MD_QC_DEV.csv", index=False)
    proto = json.loads((RES / "PROTOCOL_LENGTH.json").read_text())
    summary = {
        "n_success": len(df),
        "n_fail": len(fails),
        "fail_ids": fails,
        "mean_runtime_s": float(qcdf["runtime_seconds"].mean()) if len(qcdf) else None,
        "production_ns": proto["production_ns"],
    }
    (RES / "DEV_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ids = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]
    n_prod = json.loads((RES / "PROTOCOL_LENGTH.json").read_text())["n_prod_steps"]
    pending = [a for a in ids if not ((CACHE / a / ".complete").exists() and (CACHE / a / "features.json").exists())]
    print(f"preserved={len(ids)-len(pending)} pending={len(pending)} n_prod={n_prod}", flush=True)
    with LOG.open("a") as log:
        log.write(f"\n# isolated resume pending={len(pending)}\n")
        for i, aid in enumerate(pending, 1):
            t0 = time.time()
            print(f"[{i}/{len(pending)}] {aid}", flush=True)
            p = subprocess.run([str(PY), "-u", str(WORKER), aid, str(n_prod)], cwd=str(OUT), capture_output=True, text=True)
            wall = time.time() - t0
            log.write(p.stdout + "\n" + p.stderr[-2000:] + f"\n# {aid} rc={p.returncode} wall={wall:.1f}\n")
            log.flush()
            ok = (CACHE / aid / ".complete").exists()
            print(f"  rc={p.returncode} ok={ok} wall={wall:.1f}", flush=True)
            if not ok:
                (CACHE / aid).mkdir(parents=True, exist_ok=True)
                (CACHE / aid / "error.json").write_text(
                    json.dumps({"id": aid, "rc": p.returncode, "stderr_tail": p.stderr[-2000:]}, indent=2)
                )
                # continue
    assemble()


if __name__ == "__main__":
    main()
