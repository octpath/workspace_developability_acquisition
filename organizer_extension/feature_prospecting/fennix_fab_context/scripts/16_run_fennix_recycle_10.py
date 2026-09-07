#!/usr/bin/env python3
"""FeNNix full-cohort orchestrator: recycle native process every ≤10 antibodies.

Scientific protocol unchanged — only process lifetime. Uses existing
12_run_fennix_16cap.py / 02_fennix_fab_curvature.py with --ids batches.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
CACHE = CTX / "cache"
FEAT = CACHE / "features" / "features_partial_all.csv"
QC = CACHE / "features" / "qc_partial_all.csv"
SEQ = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv"
PREP_QC = CTX / "FAB_PREP_QC.csv"
LAUNCHER = CTX / "scripts/12_run_fennix_16cap.py"
PY = ROOT / ".venv_b1/bin/python"
BUDGET = CACHE / "HOST_CPU_BUDGET.json"
TELEMETRY = CACHE / "logs/FENNIX_WORKER_TELEMETRY.jsonl"
STATE = CACHE / "FENNIX_RECYCLE_STATE.json"
BATCH = 10
CONDITIONS = ("B", "C", "M")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _done_pairs() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if FEAT.exists():
        df = pd.read_csv(FEAT)
        for r in df.to_dict("records"):
            if r.get("extraction_status") == "SUCCESS":
                done.add((str(r["id"]), str(r["condition"])))
    return done


def _eligible_ids() -> list[str]:
    fab = pd.read_csv(SEQ)
    prep = pd.read_csv(PREP_QC) if PREP_QC.exists() else pd.DataFrame()
    last_ok = set()
    if len(prep):
        last = prep.groupby("id").tail(1)
        last_ok = set(last[last.ok].id.astype(str))
    done = _done_pairs()
    out = []
    for ab_id in fab.id.astype(str).tolist():
        prepared = CACHE / "prepared_fab" / f"{ab_id}_prepared.complete"
        if not prepared.exists():
            continue
        if last_ok and ab_id not in last_ok:
            continue
        need = [c for c in CONDITIONS if (ab_id, c) not in done]
        if need:
            out.append(ab_id)
    return out


def _python_alive(script: str) -> bool:
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
        if not (toks[0].endswith("python") or "/bin/python" in toks[0]):
            continue
        if script in raw:
            return True
    return False


def _max_rss_kb(pid: int) -> int:
    """Peak RSS of process tree via /proc smaps-ish VmHWM if present else VmRSS."""
    peak = 0
    try:
        kids = {pid}
        # include children
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                with open(f"/proc/{p}/status") as fh:
                    st = fh.read()
                if f"PPid:\t{pid}" in st or p == str(pid):
                    kids.add(int(p))
            except Exception:
                continue
        for p in list(kids):
            try:
                with open(f"/proc/{p}/status") as fh:
                    for line in fh:
                        if line.startswith("VmHWM:") or line.startswith("VmRSS:"):
                            peak = max(peak, int(line.split()[1]))
            except Exception:
                continue
    except Exception:
        pass
    return peak


def run_batch(worker_id: str, ids: list[str]) -> dict:
    start = _utc()
    t0 = time.time()
    before = _done_pairs()
    cmd = [str(PY), str(LAUNCHER), "--ids", *ids]
    # Clear stamp so 12_run can start (orchestrator owns sequencing)
    st = json.loads(BUDGET.read_text()) if BUDGET.exists() else {}
    st["active_run_stamp"] = None
    st["recycle_mode"] = True
    st["recycle_batch"] = BATCH
    BUDGET.write_text(json.dumps(st, indent=2) + "\n")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak = 0
    log_lines = []
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line:
            log_lines.append(line.rstrip())
            print(line, end="", flush=True)
        peak = max(peak, _max_rss_kb(proc.pid))
        # also probe children
        if proc.poll() is not None:
            # drain
            rest = proc.stdout.read()
            if rest:
                print(rest, end="", flush=True)
                log_lines.extend(rest.splitlines())
            break
        time.sleep(2.0)
        peak = max(peak, _max_rss_kb(proc.pid))

    rc = proc.returncode
    after = _done_pairs()
    new_pairs = after - before
    abs_done = sorted({i for i, c in new_pairs})
    rec = {
        "worker_id": worker_id,
        "worker_start_time": start,
        "worker_end_time": _utc(),
        "wall_s": round(time.time() - t0, 1),
        "ids_requested": ids,
        "antibodies_processed": len(abs_done),
        "pairs_processed": len(new_pairs),
        "new_pairs": sorted([f"{i}:{c}" for i, c in new_pairs]),
        "max_rss_kb": peak,
        "exit_status": rc,
        "batch_size_cap": BATCH,
    }
    TELEMETRY.parent.mkdir(parents=True, exist_ok=True)
    with TELEMETRY.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"TELEMETRY {json.dumps(rec)}", flush=True)
    return rec


def main():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    # Wait until no long-lived 02 worker if --wait-clear
    if "--wait-clear" in __import__("sys").argv:
        print("Waiting for existing 02_fennix worker to exit…", flush=True)
        while _python_alive("02_fennix_fab_curvature.py") or _python_alive("12_run_fennix_16cap.py"):
            time.sleep(15)
        print("Clear.", flush=True)

    remaining = _eligible_ids()
    print(f"recycle orchestrator remaining_abs={len(remaining)} batch={BATCH}", flush=True)
    STATE.write_text(
        json.dumps(
            {
                "mode": "recycle_10",
                "started": _utc(),
                "remaining_at_start": len(remaining),
                "batch": BATCH,
            },
            indent=2,
        )
        + "\n"
    )
    w = 0
    while True:
        remaining = _eligible_ids()
        if not remaining:
            print("ALL_COMPLETE", flush=True)
            break
        chunk = remaining[:BATCH]
        w += 1
        worker_id = f"recycle_{_utc().replace(':','').replace('-','')}_w{w:04d}"
        print(f"\n=== WORKER {worker_id} n={len(chunk)} ids={chunk} ===", flush=True)
        rec = run_batch(worker_id, chunk)
        # Always start fresh next loop regardless of partial success
        if rec["exit_status"] not in (0, None) and rec["pairs_processed"] == 0:
            print("WARN zero progress; sleep then continue", flush=True)
            time.sleep(30)
        # Drop stamp between workers
        st = json.loads(BUDGET.read_text()) if BUDGET.exists() else {}
        st["active_run_stamp"] = None
        BUDGET.write_text(json.dumps(st, indent=2) + "\n")
        time.sleep(2)

    STATE.write_text(
        json.dumps({"mode": "recycle_10", "finished": _utc(), "workers": w}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
