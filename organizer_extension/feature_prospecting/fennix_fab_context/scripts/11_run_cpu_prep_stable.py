#!/usr/bin/env python3
"""Resume remaining Fab CPU prep under controlled HOST CPU budget.

- Default budget: 16 cores (OPENMM_CPU_THREADS + taskset + BLAS/OMP caps)
- First resumed batch: size 5 + telemetry
- Later batches: size 10 (never >10)
- One heavy CPU job at a time; process exits between batches
- Failover: if reboot while stamped at 16, next resume uses 8 once;
  if reboot again at 8 -> HOST_STABILITY_BLOCK
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
FAB = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
PREP = CTX / "cache/prepared_fab"
LOG_DIR = CTX / "cache/logs"
BATCH_PY = CTX / "scripts/06_cpu_batch_prep.py"
BUDGET_PATH = CTX / "cache/HOST_CPU_BUDGET.json"
TELEMETRY = LOG_DIR / "prep_batch_telemetry.jsonl"
PY = ROOT / ".venv_b1/bin/python"


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def boot_id() -> str:
    try:
        return subprocess.check_output(["bash", "-lc", "who -b | awk '{print $3\" \"$4\" \"$5}'"], text=True).strip()
    except Exception:
        return "unknown"


def load_budget() -> dict:
    default = {
        "budget_cores": 16,
        "reboots_at_controlled_budget": 0,
        "last_boot": None,
        "active_run_stamp": None,
        "status": "OK",
    }
    if BUDGET_PATH.exists():
        d = json.loads(BUDGET_PATH.read_text())
        default.update(d)
    return default


def save_budget(d: dict):
    BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_PATH.write_text(json.dumps(d, indent=2) + "\n")


def resolve_budget(st: dict) -> dict:
    """Apply failover if host rebooted during a stamped controlled run."""
    cur = boot_id()
    if st.get("status") == "HOST_STABILITY_BLOCK":
        return st
    if st.get("active_run_stamp") and st.get("last_boot") and st["last_boot"] != cur:
        # Reboot occurred while a controlled budget run was active
        st["reboots_at_controlled_budget"] = int(st.get("reboots_at_controlled_budget", 0)) + 1
        prev = int(st.get("budget_cores", 16))
        if st["reboots_at_controlled_budget"] == 1 and prev >= 16:
            st["budget_cores"] = 8
            st["status"] = "FAILOVER_8"
            st["note"] = f"Reboot during controlled {prev}-core run; reducing to 8 for one retry"
        elif st["reboots_at_controlled_budget"] >= 2 or prev <= 8:
            st["status"] = "HOST_STABILITY_BLOCK"
            st["note"] = "Reboot again at controlled 8-core budget; stop sustained CPU work"
        st["active_run_stamp"] = None
    # First application of controlled 16 after uncontrolled crash: keep 16
    if st.get("budget_cores") is None:
        st["budget_cores"] = 16
    st["last_boot"] = cur
    save_budget(st)
    return st


def quarantine_incomplete(ab: str):
    qdir = CTX / "cache/quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    pdb = PREP / f"{ab}_prepared.pdb"
    marker = PREP / f"{ab}_prepared.complete"
    tmp = PREP / f"{ab}_prepared.openmm.pdb"
    for p in [pdb, marker, tmp]:
        if p.exists():
            dest = qdir / f"{ab}_{p.name}_incomplete_{utc_now().replace(':','')}"
            shutil.move(str(p), str(dest))
            print(f"quarantine {p} -> {dest}", flush=True)


def completed_ids() -> set[str]:
    done = set()
    for p in PREP.glob("*_prepared.complete"):
        done.add(p.name.replace("_prepared.complete", ""))
    if (CTX / "FAB_PREP_QC.csv").exists():
        import pandas as pd

        qc = pd.read_csv(CTX / "FAB_PREP_QC.csv")
        for _, r in qc.iterrows():
            ab = str(r.id)
            if bool(r.get("ok", False)) and (PREP / f"{ab}_prepared.pdb").exists() and (PREP / f"{ab}_prepared.pdb").stat().st_size > 0:
                if (PREP / f"{ab}_prepared.complete").exists():
                    done.add(ab)
    # empty pdb without complete is NOT done
    return done


def remaining_ids() -> list[str]:
    import pandas as pd

    seq = pd.read_csv(FAB / "sequences/SHEHATA_RECONSTRUCTED_FAB.csv")
    done = completed_ids()
    return [str(i) for i in seq.id.astype(str).tolist() if i not in done]


def snapshot_host() -> dict:
    out = {"utc": utc_now()}
    try:
        out["uptime"] = subprocess.check_output(["uptime"], text=True).strip()
    except Exception:
        pass
    try:
        out["free"] = subprocess.check_output(["bash", "-lc", "free -m | head -3"], text=True).strip()
    except Exception:
        pass
    try:
        out["nproc"] = int(subprocess.check_output(["nproc"], text=True).strip())
    except Exception:
        pass
    # temperature / throttling best-effort
    for cmd, key in [
        ("sensors 2>/dev/null | head -40", "sensors"),
        ("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null", "cpu0_freq_khz"),
        ("cat /sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count 2>/dev/null", "throttle_count"),
        ("grep -E 'MemAvailable|SwapFree' /proc/meminfo", "meminfo"),
    ]:
        try:
            out[key] = subprocess.check_output(["bash", "-lc", cmd], text=True).strip()
        except Exception:
            pass
    return out


def append_telemetry(rec: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TELEMETRY.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_batch(ids: list[str], tag: str, threads: int) -> int:
    cmd = [
        str(PY),
        str(BATCH_PY),
        "--ids",
        *ids,
        "--batch-tag",
        tag,
        "--attempt",
        "1",
        "--threads",
        str(threads),
    ]
    # Outer process also affinity-limited
    cmd = ["taskset", "-c", ",".join(str(i) for i in range(threads)), *cmd]
    t0 = time.time()
    before = snapshot_host()
    print(f"=== RUN {tag} n={len(ids)} threads={threads} ===", flush=True)
    # Monitor peak via /usr/bin/time if available, else plain
    r = subprocess.run(cmd)
    after = snapshot_host()
    # wall times from QC for this batch
    walls = []
    threads_rep = []
    if (CTX / "FAB_PREP_QC.csv").exists():
        import pandas as pd

        qc = pd.read_csv(CTX / "FAB_PREP_QC.csv")
        sub = qc[qc["id"].astype(str).isin(ids)]
        if "prep_batch" in sub.columns:
            sub = sub[sub["prep_batch"].astype(str) == tag]
        if "wall_time" in sub.columns:
            walls = [float(x) for x in sub["wall_time"].dropna().tolist()]
        if "openmm_cpu_threads_reported" in sub.columns:
            threads_rep = [str(x) for x in sub["openmm_cpu_threads_reported"].dropna().astype(str).tolist()]
    rec = {
        "tag": tag,
        "ids": ids,
        "threads_requested": threads,
        "exit": r.returncode,
        "wall_batch_s": time.time() - t0,
        "wall_per_ab_s": walls,
        "openmm_threads_reported": threads_rep,
        "host_before": before,
        "host_after": after,
    }
    append_telemetry(rec)
    print(json.dumps({k: rec[k] for k in ["tag", "exit", "wall_batch_s", "wall_per_ab_s", "openmm_threads_reported"]}, indent=2), flush=True)
    return r.returncode


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    st = resolve_budget(load_budget())
    if st.get("status") == "HOST_STABILITY_BLOCK":
        print("HOST_STABILITY_BLOCK — refusing sustained CPU prep", flush=True)
        (CTX / "HOST_STABILITY_BLOCK.md").write_text(
            f"# HOST_STABILITY_BLOCK\n\nFrozen at {utc_now()}\n\n{json.dumps(st, indent=2)}\n"
        )
        raise SystemExit(2)

    threads = int(st.get("budget_cores", 16))
    print(f"budget_cores={threads} status={st.get('status')} boot={st.get('last_boot')}", flush=True)

    # Inspect / quarantine interrupted ADI-45450
    p454 = PREP / "ADI-45450_prepared.pdb"
    c454 = PREP / "ADI-45450_prepared.complete"
    if p454.exists() and (p454.stat().st_size == 0 or not c454.exists()):
        quarantine_incomplete("ADI-45450")

    remain = remaining_ids()
    print(f"remaining={len(remain)} completed={324 - len(remain)}", flush=True)
    if not remain:
        print("ALL_PREP_ALREADY_DONE", flush=True)
        return

    # Stamp active controlled run
    st["active_run_stamp"] = utc_now()
    st["budget_cores"] = threads
    save_budget(st)

    # First resumed batch: size 5
    first = remain[:5]
    rest = remain[5:]
    rc = run_batch(first, f"resume16_b5_{utc_now().replace(':','')[:15]}", threads)
    if rc != 0:
        print("FIRST_BATCH_FAILED", rc, flush=True)
        st["active_run_stamp"] = None
        save_budget(st)
        raise SystemExit(rc)

    # Verify Threads=16 (or budget) appeared in telemetry
    last = json.loads(TELEMETRY.read_text().strip().splitlines()[-1])
    reps = set(str(x).split(".")[0] for x in (last.get("openmm_threads_reported") or []))
    print(f"first_batch OpenMM Threads reported={reps}", flush=True)
    if reps and str(threads) not in reps:
        print("WARNING: OpenMM did not report expected thread count", flush=True)
    else:
        print(f"VERIFIED OpenMM Threads={threads}", flush=True)

    # Continue with batch size 10 (or 5 after a prior hard fault in this session)
    batch_size = 5 if st.get("last_batch_fault") else 10
    bi = 0
    while True:
        rest = remaining_ids()
        # exclude permanently quarantined
        skip = set(st.get("skip_ids") or [])
        rest = [i for i in rest if i not in skip]
        if not rest:
            break
        chunk = rest[:batch_size]
        tag = f"resume{threads}_b{batch_size}_{bi:03d}"
        bi += 1
        rc = run_batch(chunk, tag, threads)
        if rc != 0:
            print(f"BATCH_FAULT {tag} rc={rc} — isolating incomplete and continuing", flush=True)
            st["last_batch_fault"] = True
            batch_size = 5
            # quarantine incomplete members of this chunk
            for ab in chunk:
                pdb = PREP / f"{ab}_prepared.pdb"
                marker = PREP / f"{ab}_prepared.complete"
                if marker.exists() and pdb.exists() and pdb.stat().st_size > 0:
                    continue
                # solo retry once
                print(f"SOLO_RETRY {ab}", flush=True)
                src = run_batch([ab], f"solo_{ab}_{utc_now().replace(':','')[:15]}", threads)
                marker2 = PREP / f"{ab}_prepared.complete"
                if src != 0 or not marker2.exists():
                    quarantine_incomplete(ab)
                    st.setdefault("skip_ids", []).append(ab)
                    print(f"QUARANTINE_SKIP {ab}", flush=True)
            save_budget(st)
            continue

    st["active_run_stamp"] = None
    st["status"] = "PREP_COMPLETE" if not remaining_ids() else st.get("status")
    save_budget(st)
    print(f"ALL_PREP_DONE remaining={len(remaining_ids())} skipped={st.get('skip_ids', [])} {utc_now()}", flush=True)


if __name__ == "__main__":
    main()
