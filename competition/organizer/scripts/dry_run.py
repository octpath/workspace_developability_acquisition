#!/usr/bin/env python3
"""End-to-end dry-run: emulate a participant with distribution files only, then score.

Participant temp workspace MUST NOT contain solution.csv or SPLIT_MANIFEST.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMP = ROOT / "competition"
DIST = COMP / "data" / "distribution"
PART = COMP / "participant"
SCRIPTS = COMP / "organizer" / "scripts"
SOLUTION = COMP / "data" / "secret" / "solution.csv"
PYTHON = Path(sys.executable)


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def main() -> int:
    assert DIST.exists(), DIST
    assert PART.exists(), PART
    assert SOLUTION.exists(), SOLUTION

    with tempfile.TemporaryDirectory(prefix="comp_participant_") as tmp:
        tmp_p = Path(tmp)
        # STEP 1–2: isolated participant workspace with ONLY distribution + participant docs
        shutil.copytree(DIST, tmp_p / "data")
        shutil.copytree(PART, tmp_p / "participant")
        # copy baseline script as if provided in a starter kit (organizer-side helper)
        starter = tmp_p / "starter"
        starter.mkdir()
        shutil.copy2(SCRIPTS / "baseline.py", starter / "baseline.py")

        # Leakage hard-check: no solution / split manifest in temp tree
        leaked = []
        for p in tmp_p.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if name in {"solution.csv", "split_manifest.json"}:
                leaked.append(str(p))
            if "secret" in p.parts:
                leaked.append(str(p))
        if leaked:
            print("FAIL: secret material in participant workspace:", leaked)
            return 1

        # STEP 3–4: run baseline (median + ridge)
        out_dir = tmp_p / "submissions"
        out_dir.mkdir()
        for method in ("median", "ridge"):
            out = out_dir / f"submission_{method}.csv"
            cp = run(
                [
                    str(PYTHON),
                    str(starter / "baseline.py"),
                    "--dev",
                    str(tmp_p / "data" / "dev.csv"),
                    "--test",
                    str(tmp_p / "data" / "test_features.csv"),
                    "--out",
                    str(out),
                    "--method",
                    method,
                ]
            )
            print(cp.stdout.strip())

        # STEP 5–7: score on organizer side
        results = {}
        for method in ("median", "ridge"):
            sub = out_dir / f"submission_{method}.csv"
            # copy submission out of temp for scoring clarity
            cp = run(
                [
                    str(PYTHON),
                    str(SCRIPTS / "score_submission.py"),
                    str(sub),
                    "--solution",
                    str(SOLUTION),
                    "--json",
                ]
            )
            results[method] = json.loads(cp.stdout)
            assert results[method]["status"] == "OK", results[method]
            s = results[method]["scores"]
            for k in (
                "TmApp_public_mae",
                "TmApp_private_mae",
                "HIC_public_mae",
                "HIC_private_mae",
            ):
                assert np_isfinite(s[k]), (method, k, s[k])

        summary = {
            "status": "PASS",
            "participant_workspace_had_secrets": False,
            "scores": {
                m: results[m]["scores"] for m in results
            },
        }
        print(json.dumps(summary, indent=2))
        return 0


def np_isfinite(x) -> bool:
    import math

    try:
        return math.isfinite(float(x))
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
