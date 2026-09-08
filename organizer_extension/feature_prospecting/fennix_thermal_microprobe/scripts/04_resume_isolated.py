#!/usr/bin/env python3
"""
Safer FeNNix thermal-microprobe resume: one antibody per fresh subprocess.

PREPARED ONLY — do not launch while OpenMM benchmark owns the GPU.
Preserves existing .complete markers (37 SUCCESS). Retries failed IDs.
No scientific protocol changes vs 01_run_dev_microprobe.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "fennix_thermal_microprobe"
RES = OUT / "results"
CACHE = OUT / "cache" / "dev"
WORKER = OUT / "scripts" / "_worker_one_ab.py"
PY = FP / "foundation_stability/envs/fennol/bin/python"
LOG = RES / "dev_run_isolated.log"


def main():
    ids = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]
    pending = []
    done = 0
    for aid in ids:
        if (CACHE / aid / ".complete").exists() and (CACHE / aid / "features.json").exists():
            done += 1
        else:
            pending.append(aid)
    print(f"preserved_complete={done} pending={len(pending)}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as log:
        log.write(f"\n# resume_isolated start pending={len(pending)} preserved={done}\n")
        for i, aid in enumerate(pending, 1):
            t0 = time.time()
            # Fresh process: releases JAX/CUDA after each Ab
            cmd = [str(PY), "-u", str(WORKER), aid]
            print(f"[{i}/{len(pending)}] launch {aid}", flush=True)
            p = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
            wall = time.time() - t0
            log.write(p.stdout)
            log.write(p.stderr)
            log.write(f"# {aid} returncode={p.returncode} wall_s={wall:.1f}\n")
            log.flush()
            ok = (CACHE / aid / ".complete").exists()
            print(f"  -> rc={p.returncode} ok={ok} wall_s={wall:.1f}", flush=True)
            if p.returncode != 0 and not ok:
                # leave retryable; continue
                (CACHE / aid).mkdir(parents=True, exist_ok=True)
                (CACHE / aid / "error_isolated.json").write_text(
                    json.dumps({"id": aid, "rc": p.returncode, "stderr_tail": p.stderr[-2000:]}, indent=2)
                )
    # assemble if all done
    ok_ids = [aid for aid in ids if (CACHE / aid / ".complete").exists()]
    print(f"complete_after={len(ok_ids)}/{len(ids)}", flush=True)
    if len(ok_ids) == len(ids):
        subprocess.check_call([str(PY), "-u", str(OUT / "scripts" / "01_run_dev_microprobe.py"), "--assemble-only"], cwd=str(OUT))


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        ids = json.loads((RES / "USABLE_DEV_IDS.json").read_text())["ids"]
        done = sum(1 for aid in ids if (CACHE / aid / ".complete").exists())
        print(json.dumps({"preserved_complete": done, "pending": len(ids) - done, "worker": str(WORKER), "ready": WORKER.exists()}))
        sys.exit(0)
    main()
