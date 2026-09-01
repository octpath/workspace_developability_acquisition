#!/usr/bin/env python3
"""Generate Round 1 Pre-Test Lock artifacts (PART B)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FIN = ROOT / "virtual_participant/round1_finalization"
BASE = json.loads((ROOT / "virtual_participant/stage5_integration/stage5_base_models.json").read_text())
SHADOW = pd.read_csv(ROOT / "virtual_participant/stage5_integration/stage5_shadow_results.csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def cfg(eid: str) -> dict:
    for t in ["TmApp", "HIC"]:
        for c in BASE[t]:
            if c["experiment_id"] == eid:
                return {"target": t, **c}
    raise KeyError(eid)


def shadow_scores(eid: str) -> dict:
    row = SHADOW[SHADOW.experiment_id == eid]
    if len(row):
        r = row.iloc[0]
        return {"Primary_MAE": None, "Shadow_MAE": float(r.Shadow_MAE), "status": r.status}
    # from base models ref
    c = cfg(eid)
    return {"Primary_MAE": c.get("ref_primary"), "Shadow_MAE": c.get("ref_shadow"), "status": "registry"}


PRIMARY_TM = "TmApp__META_performance__ridge_100.0"
PRIMARY_HIC = "HIC__SIMPLE_blend_seq_surf_adv"
TM_BASES = [
    "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt",
    "TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt",
    "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt",
    "TmApp__ADV_TMAPP_ALL__SVROpt",
]
HIC_BASES = [
    "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt",
    "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt",
    "HIC__ADV_SURFACE_PATCH__SVROpt",
]
SECONDARY = [
    {"role": "TmApp_conservative", "target": "TmApp", "experiment_id": "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt",
     "recipe": "single", "output_key": "TmApp_SECONDARY_A", "diagnostic_submission": "ROUND1_DIAGNOSTIC_TM_CONSERVATIVE.csv"},
    {"role": "TmApp_plm_seq", "target": "TmApp", "experiment_id": "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt",
     "recipe": "single", "output_key": "TmApp_SECONDARY_B", "diagnostic_submission": None},
    {"role": "HIC_conservative", "target": "HIC", "experiment_id": "HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt",
     "recipe": "single", "output_key": "HIC_SECONDARY_A", "diagnostic_submission": "ROUND1_DIAGNOSTIC_HIC_CONSERVATIVE.csv"},
    {"role": "HIC_plm_only", "target": "HIC", "experiment_id": "HIC__esm2__H__SVROpt",
     "recipe": "single", "output_key": "HIC_SECONDARY_B", "diagnostic_submission": None},
]


def full_spec_entry(c: dict, role: str, recipe: str, extra: dict | None = None) -> dict:
    entry = {
        "experiment_id": c["experiment_id"],
        "target": c["target"],
        "role": role,
        "recipe": recipe,
        "stage": c.get("stage"),
        "modality": c.get("modality"),
        "model_kind": c.get("model_kind"),
        "hyperparameters": c.get("params"),
        "pca_dim": c.get("pca_dim"),
        "plm_key": c.get("plm_key"),
        "classical_fam": c.get("classical_fam"),
        "struct_fam": c.get("struct_fam"),
        "s3_subset": c.get("s3_subset"),
        "adv_fam": c.get("adv_fam"),
        "random_seed": 0,
        "training_data": "full Dev N=162",
        "cv_artifact": "virtual_participant/stage0_cv/cv_primary.csv",
        **shadow_scores(c["experiment_id"]),
    }
    if extra:
        entry.update(extra)
    return entry


def main():
    FIN.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    commit = git_commit()

    specs = {
        "round": 1,
        "state": "ROUND1_PRETEST_LOCKED",
        "timestamp": ts,
        "git_commit": commit,
        "primary": {
            "name": "ROUND1_PRIMARY",
            "TmApp": {
                "experiment_id": PRIMARY_TM,
                "recipe": "nested_stack_full_dev_refit",
                "base_model_ids": TM_BASES,
                "meta_method": "Ridge",
                "meta_alpha": 100.0,
                "meta_set": "performance",
                "Primary_MAE": 2.7134783500422097,
                "Shadow_MAE": 2.759100007677969,
                "status": "STAGE5_SHADOW_CONFIRMED",
                "note": "Final Test meta fit uses full Dev OOF base matrix; not outer-fold weight average.",
            },
            "HIC": {
                "experiment_id": PRIMARY_HIC,
                "recipe": "equal_mean_blend_full_dev_refit",
                "base_model_ids": HIC_BASES,
                "blend_weights": [1 / 3, 1 / 3, 1 / 3],
                "Primary_MAE": 0.42517769877091044,
                "Shadow_MAE": 0.43207646713858877,
                "status": "STAGE5_SHADOW_CONFIRMED",
            },
        },
        "secondary": SECONDARY,
        "models": [],
    }

    # Full model entries
    tm_stack_bases = [full_spec_entry(cfg(b), "primary_base", "base") for b in TM_BASES]
    hic_blend_bases = [full_spec_entry(cfg(b), "primary_base", "base") for b in HIC_BASES]
    specs["models"].append({
        **specs["primary"]["TmApp"],
        "target": "TmApp",
        "base_models": tm_stack_bases,
    })
    specs["models"].append({
        **specs["primary"]["HIC"],
        "target": "HIC",
        "base_models": hic_blend_bases,
    })
    for sec in SECONDARY:
        specs["models"].append(full_spec_entry(cfg(sec["experiment_id"]), sec["role"], sec["recipe"]))

    (FIN / "round1_final_model_specs.json").write_text(json.dumps(specs, indent=2))

    # Shortlist CSV
    rows = [
        {"target": "TmApp", "role": "PRIMARY", "experiment_id": PRIMARY_TM, "Primary_MAE": 2.7135, "Shadow_MAE": 2.7591, "status": "STAGE5_SHADOW_CONFIRMED"},
        {"target": "HIC", "role": "PRIMARY", "experiment_id": PRIMARY_HIC, "Primary_MAE": 0.4252, "Shadow_MAE": 0.4321, "status": "STAGE5_SHADOW_CONFIRMED"},
    ]
    for sec in SECONDARY:
        sc = shadow_scores(sec["experiment_id"])
        c = cfg(sec["experiment_id"])
        rows.append({
            "target": sec["target"],
            "role": sec["role"],
            "experiment_id": sec["experiment_id"],
            "Primary_MAE": round(c["ref_primary"], 4),
            "Shadow_MAE": round(c["ref_shadow"], 4) if c.get("ref_shadow") else sc.get("Shadow_MAE"),
            "status": sc.get("status", "pre_registered"),
        })
    pd.DataFrame(rows).to_csv(FIN / "round1_model_shortlist.csv", index=False)

    # Hashes
    hash_files = [
        FIN / "round1_final_model_specs.json",
        FIN / "round1_model_shortlist.csv",
        ROOT / "virtual_participant/stage5_integration/STAGE5_REPORT_JA.md",
    ]
    hash_lines = []
    file_hashes = {}
    for p in hash_files:
        if p.exists():
            digest = sha256(p)
            file_hashes[str(p.relative_to(ROOT))] = digest
            hash_lines.append(f"{digest}  {p.relative_to(ROOT)}")

    (FIN / "round1_artifact_hashes.txt").write_text("\n".join(hash_lines) + "\n")

    lock = {
        "timestamp": ts,
        "git_commit": commit,
        "state": "ROUND1_PRETEST_LOCKED",
        "primary_model_ids": {"TmApp": PRIMARY_TM, "HIC": PRIMARY_HIC},
        "secondary_model_ids": {s["role"]: s["experiment_id"] for s in SECONDARY},
        "files": file_hashes,
    }
    (FIN / "ROUND1_PRETEST_LOCK.json").write_text(json.dumps(lock, indent=2))

    md = f"""# Round 1 Pre-Test Lock

**状態:** `ROUND1_PRETEST_LOCKED`  
**タイムスタンプ:** {ts}  
**git commit:** `{commit or 'unknown'}`

## 目的

Test prediction 作成前に、Round 1 として評価するモデルを完全 freeze する。Public / Private 結果を見てから final model を選ぶことを禁止する。

## PRIMARY — ROUND1_PRIMARY

| Target | experiment_id | Primary CV | Shadow CV | status |
|---|---|---:|---:|---|
| TmApp | `{PRIMARY_TM}` | 2.7135 | 2.7591 | STAGE5_SHADOW_CONFIRMED |
| HIC | `{PRIMARY_HIC}` | 0.4252 | 0.4321 | STAGE5_SHADOW_CONFIRMED |

### TmApp stack recipe (frozen)

- Base models: {', '.join(f'`{b}`' for b in TM_BASES)}
- Meta: Ridge α=100
- Final Test: full Dev 162 cross-fitted OOF → meta fit → full Dev base refit → Test predict

### HIC blend recipe (frozen)

- Base models: {', '.join(f'`{b}`' for b in HIC_BASES)}
- Weights: 1/3, 1/3, 1/3 (equal mean, no refit)

## SECONDARY — pre-registered diagnostics

| Target | role | experiment_id |
|---|---|---|
| TmApp | conservative | `TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt` |
| TmApp | plm_seq | `TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt` |
| HIC | conservative | `HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt` |
| HIC | plm_only | `HIC__esm2__H__SVROpt` |

**SECONDARY は reveal 後 postmortem 用。primary 選択には使用しない。**

## SHA-256 (pre-lock)

```
{chr(10).join(hash_lines)}
```

**Public / Private / organizer secret は本 lock 時点で未参照。**
"""
    (FIN / "ROUND1_PRETEST_LOCK_JA.md").write_text(md)
    print("ROUND1_PRETEST_LOCK_DONE")


if __name__ == "__main__":
    main()
