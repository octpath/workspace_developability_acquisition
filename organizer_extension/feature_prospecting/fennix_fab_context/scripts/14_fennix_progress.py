#!/usr/bin/env python3
"""Write FeNNix full-run progress snapshot (read-only vs running job)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CTX = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_fab_context")
FEAT = CTX / "cache/features/features_partial_all.csv"
LOG = CTX / "cache/logs/fennix_full_16cap.log"
OUT = CTX / "cache/FENNIX_FULL_PROGRESS.json"
NEED_IDS = 323
NEED_PAIRS = NEED_IDS * 3


def _worker_alive() -> bool:
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
        if "02_fennix_fab_curvature.py" in raw:
            return True
    return False


def main():
    alive = _worker_alive()
    n_success = 0
    by = {}
    ids = 0
    if FEAT.exists():
        df = pd.read_csv(FEAT)
        ok = df[df.extraction_status == "SUCCESS"]
        n_success = len(ok)
        ids = int(ok.id.nunique())
        by = ok.groupby("condition").size().to_dict()
    last = ""
    if LOG.exists():
        lines = LOG.read_text().strip().splitlines()
        last = lines[-1] if lines else ""
    remain = max(0, NEED_PAIRS - n_success)
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "worker_alive": alive,
        "success_pairs": n_success,
        "need_pairs": NEED_PAIRS,
        "remain_pairs": remain,
        "unique_ids_success": ids,
        "by_condition": by,
        "last_log": last,
        "frac": round(n_success / NEED_PAIRS, 4) if NEED_PAIRS else None,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
