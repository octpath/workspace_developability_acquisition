#!/usr/bin/env python3
"""Orchestrator: as MSAs become ready, run Boltz-2 predictions. Resume-safe."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
BASE = ROOT / "organizer_extension/feature_prospecting/structure_sources/boltz2_fv_standard_v1"
INP = BASE / "inputs"
MSA = BASE / "msa"
MMCIF = BASE / "structures_mmcif"
PY = ROOT / ".venv_boltz/bin/python"
RUN = BASE / "scripts/run_boltz2_batch.py"
PRE = BASE / "scripts/precompute_msa.py"


def ready_msa(aid: str) -> bool:
    return (MSA / aid / "H.csv").exists() and (MSA / aid / "L.csv").exists()


def done_struct(aid: str) -> bool:
    return (MMCIF / f"{aid}.cif").exists() and (MMCIF / f"{aid}.cif").stat().st_size > 0


def main() -> None:
    ids = sorted(p.stem for p in INP.glob("*.yaml"))
    # Ensure MSA worker is progressing: kick once
    subprocess.Popen(
        [str(PY), str(PRE)],
        stdout=open(BASE / "logs/msa_all2.log", "a"),
        stderr=subprocess.STDOUT,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "BOLTZ_CACHE": str(ROOT / ".boltz_cache")},
    )
    while True:
        pending_msa = [i for i in ids if not ready_msa(i)]
        pending_pred = [i for i in ids if ready_msa(i) and not done_struct(i)]
        n_done = sum(1 for i in ids if done_struct(i))
        print(
            f"[{time.strftime('%H:%M:%S')}] structs={n_done}/324 msa_pending={len(pending_msa)} pred_queue={len(pending_pred)}",
            flush=True,
        )
        if n_done == 324:
            print("ALL DONE")
            break
        if pending_pred:
            # predict one at a time for shm safety
            aid = pending_pred[0]
            print(f"predict {aid}", flush=True)
            subprocess.run(
                [str(PY), str(RUN), "--ids", aid, "--retry-failed"],
                check=False,
                env={
                    **dict(**{k: v for k, v in __import__("os").environ.items()}),
                    "CUDA_VISIBLE_DEVICES": "0",
                    "BOLTZ_CACHE": str(ROOT / ".boltz_cache"),
                    "TMPDIR": str(ROOT / ".tmp_torch"),
                },
            )
        else:
            time.sleep(30)
        if not pending_msa and not pending_pred and n_done < 324:
            print("STALLED: MSA incomplete and no pred queue")
            break


if __name__ == "__main__":
    main()
