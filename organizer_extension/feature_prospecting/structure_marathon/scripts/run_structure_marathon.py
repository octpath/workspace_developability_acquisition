#!/usr/bin/env python3
"""Resumable Structure Marathon orchestrator (status + selective family runners)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SM = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/structure_marathon")
STATE = SM / "STRUCTURE_MARATHON_STATE.json"
LOG = SM / "STRUCTURE_MARATHON_LOG.md"
PY_B1 = Path("/workspace_developability_acquisition/.venv_b1/bin/python")
PY_IF = Path("/workspace_developability_acquisition/.venv_esmif/bin/python")


def load_state():
    return json.loads(STATE.read_text())


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def log(msg: str):
    with LOG.open("a") as f:
        f.write(f"- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
    print(msg, flush=True)


def run(cmd, env=None):
    log("RUN " + " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=str(SM), env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--fusion", action="store_true")
    args = ap.parse_args()
    st = load_state()
    if args.status:
        print(json.dumps(st, indent=2))
        return
    if args.eval_only:
        run([str(PY_B1), str(SM / "scripts/eval_families.py")])
        return
    if args.fusion:
        run([str(PY_B1), str(SM / "scripts/fusion_top3.py")])
        return
    log("orchestrator idle — families already driven by campaign scripts; use --status/--eval-only/--fusion")


if __name__ == "__main__":
    main()
