#!/usr/bin/env python3
"""Analysis stub for T161–T337 factorial — expanded after bulk completion."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def run_analysis() -> None:
    import pandas as pd

    plan = pd.read_csv(ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv")
    n_done = (plan.execution_status.isin(["COMPLETE", "REUSE"])).sum()
    n_fail = (plan.execution_status == "FAILED").sum()
    n_plan = (plan.execution_status == "PLANNED").sum()
    print(f"analysis: done={n_done} failed={n_fail} planned={n_plan}", flush=True)
    if n_plan > 0 or n_done < 200:
        print("matrix incomplete — skipping full analysis", flush=True)
        return
    # Full analysis implemented in finalize after bulk; write results shell
    out = ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv"
    plan.to_csv(out, index=False)
    print("Wrote", out, flush=True)


def run_external() -> None:
    print("external diagnostic requires internal freeze first", flush=True)


if __name__ == "__main__":
    run_analysis()
