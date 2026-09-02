#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
CW = ROOT / "organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv"
OUT_DIR = ROOT / "organizer_extension/feature_prospecting/ANM-SPECTRUM/boltz2_rows"
WORKER = Path("/tmp/anm_one.py")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKER.write_text(
    '''
import sys, json
from pathlib import Path
import importlib.util
import pandas as pd
import prody as pr
pr.confProDy(verbosity="none")
aid, pdb, out = sys.argv[1], sys.argv[2], sys.argv[3]
cw = pd.read_csv("/workspace_developability_acquisition/organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv")
row = cw[cw.id == aid].iloc[0]
spec = importlib.util.spec_from_file_location(
    "ext",
    "/workspace_developability_acquisition/organizer_extension/feature_prospecting/ANM-SPECTRUM/scripts/extract_anm_spectrum_v1.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rec = mod.process_one(aid, "boltz2", Path(pdb), row)
safe = {}
for k, v in rec.items():
    if v is None:
        safe[k] = None
    elif hasattr(v, "item"):
        try:
            safe[k] = v.item()
        except Exception:
            safe[k] = str(v)
    else:
        safe[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
Path(out).write_text(json.dumps(safe))
print("OK", aid)
'''
)

cw = pd.read_csv(CW)
ok = fail = 0
for i, r in cw.iterrows():
    aid = r["id"]
    out = OUT_DIR / f"{aid}.json"
    if out.exists():
        ok += 1
        continue
    rc = subprocess.run(
        [sys.executable, str(WORKER), aid, str(r["boltz2_pdb_path"]), str(out)],
        capture_output=True,
        text=True,
    )
    if rc.returncode == 0 and out.exists():
        ok += 1
    else:
        fail += 1
        print("FAIL", aid, rc.returncode, (rc.stderr or rc.stdout)[-300:], flush=True)
    if (i + 1) % 25 == 0:
        print("progress", i + 1, "ok", ok, "fail", fail, flush=True)
print("DONE ok", ok, "fail", fail, flush=True)
