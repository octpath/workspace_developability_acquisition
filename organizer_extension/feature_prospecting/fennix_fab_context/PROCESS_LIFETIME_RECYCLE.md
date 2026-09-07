# FeNNix process lifetime (operational only)

**Policy (from 2026-09-07):** recycle native worker every **≤10 antibodies** (≤30 B/C/M pairs).

- Science unchanged: same model, structures, perturbations, features, QC, completion markers.
- CPU cap remains **16** (`12_run_fennix_16cap.py`).
- Orchestrator: `scripts/16_run_fennix_recycle_10.py`
- Telemetry: `cache/logs/FENNIX_WORKER_TELEMETRY.jsonl`
- Resume helper: `scripts/15_resume_fennix_if_needed.sh` → recycle orchestrator

**Post-cohort (pending full finish):** fresh-worker numerical reproducibility check on crash-associated Abs ADI-47060 / 47313 / 45469 / 45472.
