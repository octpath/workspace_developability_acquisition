#!/usr/bin/env python3
"""ESMFold paired VH/VL structures with explicit chain-break handling.

Uses a fixed 25-Gly linker between VH and VL (HuggingFace EsmForProteinFolding
does not reliably preserve colon-separated true multichain geometry). Linker
residues are recorded and excluded from SASA/RASA aggregates. This limitation
is documented — we do NOT silently treat concat-without-break as multichain.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import CACHE, DATA, REPORTS, ensure_dirs, pair_hash, set_gpu0, write_json  # noqa: E402

LINKER = "G" * 25
MODEL_ID = "facebook/esmfold_v1"


def write_pdb(coords, aatype, plddt, path: Path, hlen: int, linker_len: int):
    """Minimal CA-only or full-atom PDB from ESMFold output tensors.

    Prefer model.output_to_pdb if available.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    set_gpu0()
    ensure_dirs()
    out_dir = CACHE / "structures" / "esmfold"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    device = "cuda:0"

    from transformers import AutoTokenizer, EsmForProteinFolding

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmForProteinFolding.from_pretrained(MODEL_ID, low_cpu_mem_usage=True).to(device)
    model.eval()

    # Validate multichain representation on 2 antibodies
    validation = []
    for _, r in df.head(2).iterrows():
        seq = r["heavy"] + LINKER + r["light"]
        inputs = tok([seq], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        plddt = out.plddt[0].detach().cpu().numpy()
        hlen = len(r["heavy"])
        llen = len(r["light"])
        assert plddt.shape[0] == hlen + len(LINKER) + llen
        # inter-chain CA distance sanity using positions if available
        val = {
            "antibody_id": r["antibody_id"],
            "len_total": int(plddt.shape[0]),
            "hlen": hlen,
            "llen": llen,
            "linker_len": len(LINKER),
            "mean_plddt": float(plddt.mean()),
            "heavy_plddt": float(plddt[:hlen].mean()),
            "linker_plddt": float(plddt[hlen : hlen + len(LINKER)].mean()),
            "light_plddt": float(plddt[hlen + len(LINKER) :].mean()),
            "method": "25xGly_linker_between_VH_VL",
            "true_multichain_colon_supported": False,
            "note": (
                "HF EsmForProteinFolding folds a single polymer with Gly linker; "
                "chain boundaries preserved by index bookkeeping, not by native "
                "multichain decoder. Linker residues excluded from feature extraction."
            ),
        }
        validation.append(val)
    write_json(CACHE / "structures" / "esmfold_multichain_validation.json", validation)

    rows = []
    n_ok = n_fail = 0
    t_all = time.perf_counter()
    for i, r in df.iterrows():
        ph = pair_hash(r["heavy"], r["light"])
        pdb_path = out_dir / f"{ph}.pdb"
        meta_path = out_dir / f"{ph}.json"
        if pdb_path.exists() and meta_path.exists():
            rows.append(json.loads(meta_path.read_text()))
            n_ok += 1
            continue
        t0 = time.perf_counter()
        try:
            h, l = r["heavy"], r["light"]
            seq = h + LINKER + l
            inputs = tok([seq], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            plddt = out.plddt[0].detach().cpu().numpy().astype(float)
            # Use official PDB writer
            pdb_strs = model.output_to_pdb(out)
            pdb_text = pdb_strs[0] if isinstance(pdb_strs, (list, tuple)) else pdb_strs
            # Relabel chains: residues 1..hlen -> H, skip linker, then L
            # output_to_pdb is single chain A — rewrite chain IDs by residue index
            new_lines = []
            res_counter = 0
            prev_res = None
            atom_lines = []
            for line in pdb_text.splitlines():
                if line.startswith(("ATOM", "HETATM")):
                    atom_lines.append(line)
            # Parse and rewrite
            rewritten = []
            current_reskey = None
            res_idx = -1  # 0-based residue along polymer
            for line in atom_lines:
                reskey = line[17:26]  # resname + chain + resseq
                if reskey != current_reskey:
                    current_reskey = reskey
                    res_idx += 1
                if res_idx < len(h):
                    chain = "H"
                    new_resseq = res_idx + 1
                    keep = True
                elif res_idx < len(h) + len(LINKER):
                    keep = False  # drop linker from PDB used for SASA
                    chain = "X"
                    new_resseq = res_idx - len(h) + 1
                else:
                    chain = "L"
                    new_resseq = res_idx - len(h) - len(LINKER) + 1
                    keep = True
                if not keep:
                    continue
                # PDB columns: chain is 21, resseq 22-26
                newline = line[:21] + chain + f"{new_resseq:4d}" + line[26:]
                rewritten.append(newline)
            pdb_body = "\n".join(rewritten) + "\nEND\n"
            pdb_path.write_text(pdb_body)
            meta = {
                "antibody_id": r["antibody_id"],
                "pair_hash": ph,
                "predictor": "ESMFold",
                "model": MODEL_ID,
                "pdb_path": str(pdb_path),
                "structure_hash": hashlib.sha256(pdb_body.encode()).hexdigest()[:32],
                "runtime_s": time.perf_counter() - t0,
                "success": True,
                "hlen": len(h),
                "llen": len(l),
                "linker": LINKER,
                "linker_len": len(LINKER),
                "global_mean_plddt": float(plddt.mean()),
                "heavy_mean_plddt": float(plddt[: len(h)].mean()),
                "light_mean_plddt": float(plddt[len(h) + len(LINKER) :].mean()),
                "linker_mean_plddt": float(plddt[len(h) : len(h) + len(LINKER)].mean()),
                "chain_break_method": "25xGly_linker_excluded_from_PDB",
                "error": None,
            }
            n_ok += 1
        except Exception as e:
            meta = {
                "antibody_id": r["antibody_id"],
                "pair_hash": ph,
                "predictor": "ESMFold",
                "success": False,
                "error": str(e),
                "tb": traceback.format_exc()[-1500:],
                "runtime_s": time.perf_counter() - t0,
            }
            n_fail += 1
            print("FAIL", r["antibody_id"], e, flush=True)
        meta_path.write_text(json.dumps(meta, indent=2))
        rows.append(meta)
        if len(rows) % 10 == 0:
            print(f"PROGRESS {len(rows)}/{len(df)} ok={n_ok} fail={n_fail}", flush=True)

    pd.DataFrame(rows).to_csv(CACHE / "structures" / "esmfold_manifest.csv", index=False)
    audit = {
        "n_total": len(df),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "success_rate": n_ok / max(len(df), 1),
        "total_runtime_s": time.perf_counter() - t_all,
        "multichain_note": validation[0]["note"] if validation else "",
        "mean_plddt": float(
            np.mean([r["global_mean_plddt"] for r in rows if r.get("success") and "global_mean_plddt" in r])
        )
        if any(r.get("success") for r in rows)
        else None,
    }
    write_json(CACHE / "structures" / "esmfold_audit.json", audit)
    md = REPORTS / "structure_prediction_audit.md"
    prev = md.read_text() if md.exists() else "# Structure prediction audit\n"
    prev += "\n## ESMFold\n\n" + "\n".join(f"- {k}: {v}" for k, v in audit.items()) + "\n"
    md.write_text(prev)
    print("ESMFOLD_OK", audit)


if __name__ == "__main__":
    main()
