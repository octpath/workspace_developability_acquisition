#!/usr/bin/env python3
"""Extract residue-level embeddings for AbLang1 / ESM-1b / ESM-C 600M / CurrAb.

Creates bundles under top_models_feature_bundle/residue_level/{ablang1,esm1b,esmc600m,currab}/
matching the canonical asset contract (ids, heavy/light embeddings+masks, metadata, .part splits).

Usage:
  CUDA_VISIBLE_DEVICES=0 python extract_new_plm_residue_embeddings.py --plm all --device cuda:0
  CUDA_VISIBLE_DEVICES=0 python extract_new_plm_residue_embeddings.py --plm ablang1 esm1b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parent
DRILL = ROOT / "developability_drilldown"
RES = BUNDLE / "residue_level"

# Prefer competition sequences; fall back to existing pack ID order.
DEV = DRILL / "data" / "dev.csv"
TEST = DRILL / "data" / "test.csv"
REF_IDS = RES / "ablang2" / "ids.npy"

MAX_H = 140
MAX_L = 120


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_npy(path: Path) -> list[Path]:
    data = path.read_bytes()
    for stale in sorted(path.parent.glob(path.name + ".part*")):
        stale.unlink()
    mid = len(data) // 2
    parts = []
    for i, chunk in enumerate((data[:mid], data[mid:])):
        p = Path(str(path) + f".part{i}")
        p.write_bytes(chunk)
        parts.append(p)
    rebuilt = parts[0].read_bytes() + parts[1].read_bytes()
    if rebuilt != data:
        raise RuntimeError(f"part reassembly failed for {path}")
    return parts


def load_sequences() -> pd.DataFrame:
    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    seqs = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
    seqs["id"] = seqs["id"].astype(str)
    # canonical order from existing residue packs
    if REF_IDS.exists():
        ref = [str(x) for x in np.load(REF_IDS, allow_pickle=True).tolist()]
        if set(ref) != set(seqs["id"]):
            raise RuntimeError("sequence IDs do not match reference residue pack IDs")
        seqs = seqs.set_index("id").loc[ref].reset_index()
    else:
        seqs = seqs.sort_values("id").reset_index(drop=True)
    if len(seqs) != 324:
        raise RuntimeError(f"expected 324 antibodies, got {len(seqs)}")
    if int(seqs["heavy"].str.len().max()) > MAX_H or int(seqs["light"].str.len().max()) > MAX_L:
        raise RuntimeError("sequence longer than canonical max_len")
    return seqs


def pack_padded(arrays: list[np.ndarray], max_len: int, hidden: int) -> tuple[np.ndarray, np.ndarray]:
    N = len(arrays)
    emb = np.zeros((N, max_len, hidden), dtype=np.float16)
    mask = np.zeros((N, max_len), dtype=np.bool_)
    for i, a in enumerate(arrays):
        L = int(a.shape[0])
        if L > max_len:
            raise RuntimeError(f"len {L} > max_len {max_len}")
        if a.shape[1] != hidden:
            raise RuntimeError(f"hidden mismatch {a.shape[1]} vs {hidden}")
        if not np.isfinite(a.astype(np.float32)).all():
            raise RuntimeError("non-finite embedding")
        emb[i, :L] = a.astype(np.float16)
        mask[i, :L] = True
    return emb, mask


def write_bundle(
    out_dir: Path,
    ids: list[str],
    heavy: list[np.ndarray],
    light: list[np.ndarray],
    hidden: int,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # length QC vs sequences embedded in meta caller
    eh, mh = pack_padded(heavy, MAX_H, hidden)
    el, ml = pack_padded(light, MAX_L, hidden)
    np.save(out_dir / "ids.npy", np.asarray(ids, dtype=object))
    np.save(out_dir / "heavy_embeddings.npy", eh)
    np.save(out_dir / "light_embeddings.npy", el)
    np.save(out_dir / "heavy_mask.npy", mh)
    np.save(out_dir / "light_mask.npy", ml)
    for name in ("heavy_embeddings.npy", "light_embeddings.npy"):
        split_npy(out_dir / name)
    meta = dict(meta)
    meta.update(
        {
            "n_ids": len(ids),
            "max_len_H": MAX_H,
            "max_len_L": MAX_L,
            "hidden_dim": int(hidden),
            "dtype_on_disk": "float16",
            "dtype_in_memory": "float32",
            "ids_sha256": sha256_file(out_dir / "ids.npy"),
            "heavy_emb_sha256": sha256_file(out_dir / "heavy_embeddings.npy"),
            "light_emb_sha256": sha256_file(out_dir / "light_embeddings.npy"),
            "heavy_mask_sha256": sha256_file(out_dir / "heavy_mask.npy"),
            "light_mask_sha256": sha256_file(out_dir / "light_mask.npy"),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # diagnostics
    diag = embedding_diagnostics(heavy + light)
    (out_dir / "EMBEDDING_DIAGNOSTICS.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir} hidden={hidden} diag={diag}", flush=True)


def embedding_diagnostics(arrays: list[np.ndarray]) -> dict:
    norms = []
    n_zero = 0
    n_nonfinite = 0
    n_vec = 0
    for a in arrays:
        a32 = a.astype(np.float32)
        n_nonfinite += int((~np.isfinite(a32)).sum())
        for v in a32:
            n_vec += 1
            nrm = float(np.linalg.norm(v))
            norms.append(nrm)
            if nrm < 1e-12:
                n_zero += 1
    norms = np.asarray(norms, dtype=np.float64)
    return {
        "n_vectors": int(n_vec),
        "mean_l2": float(norms.mean()),
        "std_l2": float(norms.std()),
        "p05_l2": float(np.quantile(norms, 0.05)),
        "p95_l2": float(np.quantile(norms, 0.95)),
        "fraction_zero_vectors": float(n_zero / max(n_vec, 1)),
        "n_nonfinite_elements": int(n_nonfinite),
    }


# -------------------- extractors --------------------


def extract_ablang1(seqs: pd.DataFrame, device: str) -> tuple[list, list, int, dict]:
    import ablang
    from pathlib import Path as _P

    t0 = time.time()
    pkg = _P(ablang.__file__).parent
    heavy_dir = pkg / "model-weights-heavy"
    light_dir = pkg / "model-weights-light"
    for d, name in ((heavy_dir, "heavy"), (light_dir, "light")):
        if not (d / "amodel.pt").is_file():
            raise RuntimeError(
                f"AbLang1 {name} weights missing at {d}/amodel.pt — "
                f"download https://opig.stats.ox.ac.uk/data/downloads/ablang-{name}.tar.gz"
            )
    # Prefer local official cache (avoid re-download hang on empty tmp.tar.gz)
    heavy_model = ablang.pretrained("heavy", model_folder=str(heavy_dir), device=device if device.startswith("cuda") else "cpu")
    light_model = ablang.pretrained("light", model_folder=str(light_dir), device=device if device.startswith("cuda") else "cpu")
    heavy_model.AbLang.eval()
    light_model.AbLang.eval()
    H_arr, L_arr = [], []
    with torch.no_grad():
        for _, row in seqs.iterrows():
            h, l = str(row.heavy), str(row.light)
            hr = np.asarray(heavy_model.rescoding([h])[0], dtype=np.float32)
            lr = np.asarray(light_model.rescoding([l])[0], dtype=np.float32)
            if hr.shape[0] != len(h) or lr.shape[0] != len(l):
                raise RuntimeError(
                    f"AbLang1 mapping fail {row.id}: H {hr.shape[0]}!={len(h)} L {lr.shape[0]}!={len(l)}"
                )
            if hr.shape[1] != 768 or lr.shape[1] != 768:
                raise RuntimeError(f"AbLang1 unexpected dim {hr.shape} {lr.shape}")
            H_arr.append(hr)
            L_arr.append(lr)
    hidden = 768
    meta = {
        "model_name": "ablang-heavy / ablang-light",
        "package": "ablang",
        "package_version": getattr(ablang, "__version__", "unknown"),
        "representation_source": "ablang.pretrained(...).rescoding (res-codings)",
        "raw_hidden_dimension": hidden,
        "native_input_convention": "separate_chain_models",
        "representation_context": "SEPARATE_CHAIN",
        "paired_vs_separate": "separate",
        "special_token_mapping_rule": "rescoding returns AA-only vectors; no BOS/EOS retained",
        "mapping": "EXACT_AA_1TO1",
        "license_status": "REVIEW_MODEL_OUTPUT",
        "local_weight_dirs": {"heavy": str(heavy_dir), "light": str(light_dir)},
        "extraction_wall_time_sec": time.time() - t0,
        "device": device,
        "extraction_command": "extract_new_plm_residue_embeddings.py --plm ablang1",
    }
    return H_arr, L_arr, hidden, meta


def extract_esm1b(seqs: pd.DataFrame, device: str) -> tuple[list, list, int, dict]:
    """ESM-1b via HuggingFace (fair-esm is shadowed by biohub `esm` 3.x in this venv)."""
    from transformers import EsmModel, EsmTokenizer
    import transformers

    t0 = time.time()
    repo = "facebook/esm1b_t33_650M_UR50S"
    tok = EsmTokenizer.from_pretrained(repo)
    # EsmModel last_hidden_state == final transformer layer (layer 33 / index 33)
    model = EsmModel.from_pretrained(repo)
    model = model.eval().to(device)
    H_arr, L_arr = [], []
    with torch.no_grad():
        for _, row in seqs.iterrows():
            for chain_seq, out_list in ((str(row.heavy), H_arr), (str(row.light), L_arr)):
                enc = tok(chain_seq, return_tensors="pt", add_special_tokens=True)
                tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
                if tokens[0] not in ("<cls>", "<bos>") or tokens[-1] not in ("<eos>", "<sep>"):
                    # ESM-1b HF uses <cls> ... <eos>
                    if not (tokens[0] == "<cls>" and tokens[-1] == "<eos>"):
                        raise RuntimeError(f"ESM-1b unexpected specials {row.id}: {tokens[0]!r}..{tokens[-1]!r}")
                if len(tokens) != len(chain_seq) + 2:
                    raise RuntimeError(
                        f"ESM-1b token length {row.id}: {len(tokens)} vs {len(chain_seq)+2}"
                    )
                if "".join(tokens[1:-1]) != chain_seq:
                    raise RuntimeError(f"ESM-1b AA token mismatch {row.id}")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                rep = out.last_hidden_state[0, 1:-1].detach().float().cpu().numpy().astype(np.float32)
                if rep.shape[0] != len(chain_seq):
                    raise RuntimeError(f"ESM-1b BOS/EOS strip fail {row.id} {rep.shape} vs {len(chain_seq)}")
                out_list.append(rep)
    hidden = int(H_arr[0].shape[1])
    if hidden != 1280:
        raise RuntimeError(f"ESM-1b unexpected hidden {hidden}")
    meta = {
        "model_name": repo,
        "exact_model_alias": "esm1b_t33_650M_UR50S",
        "package": "transformers (EsmModel; fair-esm shadowed by biohub esm)",
        "package_version": transformers.__version__,
        "representation_source": "EsmModel.last_hidden_state (final layer 33)",
        "final_layer": 33,
        "raw_hidden_dimension": hidden,
        "native_input_convention": "single_protein",
        "representation_context": "SEPARATE_CHAIN",
        "paired_vs_separate": "separate",
        "special_token_mapping_rule": "strip <cls> and <eos>; keep exactly len(seq) AA vectors",
        "mapping": "EXACT_AA_1TO1_BOS_EOS_STRIPPED",
        "license_status": "REVIEW_MODEL_OUTPUT",
        "extraction_wall_time_sec": time.time() - t0,
        "device": device,
        "extraction_command": "extract_new_plm_residue_embeddings.py --plm esm1b",
    }
    return H_arr, L_arr, hidden, meta


def extract_esmc600m(seqs: pd.DataFrame, device: str) -> tuple[list, list, int, dict]:
    warnings.filterwarnings("ignore")
    from esm.pretrained import ESMC_600M_202412, ESMC_600M_HF_REPO
    from esm.sdk.api import ESMProtein, LogitsConfig
    import esm as biohub_esm

    t0 = time.time()
    model = ESMC_600M_202412(device=device, use_flash_attn=False)
    cfg = LogitsConfig(sequence=False, return_embeddings=True)
    H_arr, L_arr = [], []
    hidden = None
    with torch.no_grad():
        for _, row in seqs.iterrows():
            for chain_seq, out_list in ((str(row.heavy), H_arr), (str(row.light), L_arr)):
                t = model.encode(ESMProtein(sequence=chain_seq))
                # sequence tokens = [BOS] + AA + [EOS]
                if int(t.sequence.numel()) != len(chain_seq) + 2:
                    raise RuntimeError(
                        f"ESMC token length mismatch {row.id}: {t.sequence.numel()} vs {len(chain_seq)+2}"
                    )
                out = model.logits(t, cfg)
                emb = out.embeddings[0].detach().float().cpu().numpy()
                # strip BOS/EOS
                aa = emb[1:-1]
                if aa.shape[0] != len(chain_seq):
                    raise RuntimeError(f"ESMC strip fail {row.id}: {aa.shape[0]} vs {len(chain_seq)}")
                if hidden is None:
                    hidden = int(aa.shape[1])
                elif aa.shape[1] != hidden:
                    raise RuntimeError("ESMC hidden dim inconsistency")
                out_list.append(aa.astype(np.float32))
    assert hidden is not None
    meta = {
        "model_name": ESMC_600M_HF_REPO,
        "model_revision": None,
        "package": "esm (Biohub)",
        "package_version": getattr(biohub_esm, "__version__", "unknown"),
        "representation_source": "ESMC.logits(..., return_embeddings=True) final embeddings",
        "raw_hidden_dimension": int(hidden),
        "inferred_from_checkpoint": True,
        "native_input_convention": "single_protein ESMProtein.encode",
        "representation_context": "SEPARATE_CHAIN",
        "paired_vs_separate": "separate",
        "special_token_mapping_rule": "encode adds BOS/EOS; strip indices [0] and [-1]",
        "mapping": "EXACT_AA_1TO1_BOS_EOS_STRIPPED",
        "tokenizer_version": "esm ESMC internal",
        "license_status": "REVIEW_MODEL_OUTPUT",
        "license_note": "biohub/ESMC-600M MIT/other — see THIRD_PARTY_NOTICE",
        "extraction_wall_time_sec": time.time() - t0,
        "device": device,
        "extraction_command": "extract_new_plm_residue_embeddings.py --plm esmc600m",
        "note_te_fallback": "Transformer Engine / flash-attn may be unavailable; numerical fallback noted by esm package",
    }
    return H_arr, L_arr, int(hidden), meta


def extract_currab(seqs: pd.DataFrame, device: str) -> tuple[list, list, int, dict]:
    from transformers import EsmForMaskedLM, EsmTokenizer
    from huggingface_hub import model_info

    t0 = time.time()
    repo = "brineylab/CurrAb"
    info = model_info(repo)
    rev = getattr(info, "sha", None)
    tok = EsmTokenizer.from_pretrained(repo)
    model = EsmForMaskedLM.from_pretrained(repo)
    model = model.eval().to(device)
    H_arr, L_arr = [], []
    hidden = int(model.config.hidden_size)
    with torch.no_grad():
        for _, row in seqs.iterrows():
            h, l = str(row.heavy), str(row.light)
            # Native paired form from README: Heavy + <cls> + Light ; no extra BOS/EOS
            s = f"{h}<cls>{l}"
            enc = tok(s, return_tensors="pt", add_special_tokens=False)
            tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
            if len(tokens) != len(h) + 1 + len(l):
                raise RuntimeError(f"CurrAb token length {row.id}: {len(tokens)} vs {len(h)+1+len(l)}")
            if tokens[len(h)] != "<cls>":
                raise RuntimeError(f"CurrAb separator missing/misplaced {row.id}: {tokens[len(h)]}")
            # verify AA identity
            if "".join(tokens[: len(h)]) != h or "".join(tokens[len(h) + 1 :]) != l:
                raise RuntimeError(f"CurrAb AA token mismatch {row.id}")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-1][0].detach().float().cpu().numpy()
            he = hs[: len(h)].astype(np.float32)
            le = hs[len(h) + 1 :].astype(np.float32)
            if he.shape[0] != len(h) or le.shape[0] != len(l):
                raise RuntimeError(f"CurrAb split fail {row.id}")
            H_arr.append(he)
            L_arr.append(le)
    meta = {
        "model_name": repo,
        "model_revision": rev,
        "package": "transformers",
        "package_version": __import__("transformers").__version__,
        "representation_source": "EsmForMaskedLM last_hidden_state",
        "raw_hidden_dimension": hidden,
        "native_input_convention": "paired Heavy<cls>Light with add_special_tokens=False",
        "representation_context": "PAIRED_NATIVE",
        "paired_vs_separate": "paired",
        "special_token_mapping_rule": "exclude single <cls> separator; no BOS/EOS when add_special_tokens=False",
        "mapping": "EXACT_AA_1TO1_CLS_SEPARATOR_STRIPPED",
        "tokenizer_version": "EsmTokenizer brineylab/CurrAb",
        "license_status": "REVIEW_MODEL_OUTPUT",
        "license_note": "MIT (CurrAb README)",
        "extraction_wall_time_sec": time.time() - t0,
        "device": device,
        "extraction_command": "extract_new_plm_residue_embeddings.py --plm currab",
    }
    return H_arr, L_arr, hidden, meta


EXTRACTORS = {
    "ablang1": ("ablang1", extract_ablang1),
    "esm1b": ("esm1b", extract_esm1b),
    "esmc600m": ("esmc600m", extract_esmc600m),
    "currab": ("currab", extract_currab),
}


def reextract_subset_check(plm: str, seqs: pd.DataFrame, device: str, out_dir: Path, n: int = 3) -> dict:
    """Re-extract first n antibodies and compare to saved bundle.

    Bundles are stored as float16; agreement is checked after casting the fresh
    float32 extraction through float16 (disk dtype), with a small extra eps.
    """
    sub = seqs.head(n).copy()
    _, fn = EXTRACTORS[plm]
    H, L, hidden, _ = fn(sub, device)
    eh = np.load(out_dir / "heavy_embeddings.npy")
    el = np.load(out_dir / "light_embeddings.npy")
    mh = np.load(out_dir / "heavy_mask.npy")
    ml = np.load(out_dir / "light_mask.npy")
    max_abs = 0.0
    max_abs_raw = 0.0
    for i in range(n):
        lh = int(mh[i].sum())
        ll = int(ml[i].sum())
        h_disklike = H[i].astype(np.float16).astype(np.float32)
        l_disklike = L[i].astype(np.float16).astype(np.float32)
        dh = float(np.max(np.abs(eh[i, :lh].astype(np.float32) - h_disklike)))
        dl = float(np.max(np.abs(el[i, :ll].astype(np.float32) - l_disklike)))
        max_abs = max(max_abs, dh, dl)
        max_abs_raw = max(
            max_abs_raw,
            float(np.max(np.abs(eh[i, :lh].astype(np.float32) - H[i]))),
            float(np.max(np.abs(el[i, :ll].astype(np.float32) - L[i]))),
        )
    # Exact match after fp16 round-trip; allow tiny numeric noise for SDPA/bf16 models
    tol = 1e-5 if plm != "esmc600m" else 5e-2
    ok = max_abs <= tol
    return {
        "plm": plm,
        "n": n,
        "max_abs_diff_vs_fp16_roundtrip": max_abs,
        "max_abs_diff_vs_raw_fp32": max_abs_raw,
        "tolerance": tol,
        "disk_dtype": "float16",
        "pass": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plm", nargs="+", default=["all"], choices=["all", *EXTRACTORS.keys()])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-reextract-check", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    plms = list(EXTRACTORS.keys()) if "all" in args.plm else list(args.plm)
    seqs = load_sequences()
    ids = seqs["id"].tolist()
    status = {}
    for plm in plms:
        subdir, fn = EXTRACTORS[plm]
        out = RES / subdir
        print(f"==== EXTRACT {plm} -> {out} ====", flush=True)
        try:
            H, L, hidden, meta = fn(seqs, args.device)
            # length QC vs sequences
            for i, (_, row) in enumerate(seqs.iterrows()):
                if H[i].shape[0] != len(str(row.heavy)) or L[i].shape[0] != len(str(row.light)):
                    raise RuntimeError(f"length QC fail {row.id}")
            write_bundle(out, ids, H, L, hidden, meta)
            if not args.skip_reextract_check:
                chk = reextract_subset_check(plm, seqs, args.device, out)
                (out / "REEXTRACT_CHECK.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
                print("reextract", chk, flush=True)
                if not chk["pass"]:
                    status[plm] = "FAILED_TRAINING"
                    print(f"BLOCKED reextract tolerance for {plm}", flush=True)
                    continue
            status[plm] = "PASS"
        except Exception as e:
            status[plm] = f"BLOCKED: {type(e).__name__}: {e}"
            print(status[plm], flush=True)
            import traceback

            traceback.print_exc()
    (RES / "NEW_PLM_EXTRACTION_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STATUS", status, flush=True)
    return 0 if all(v == "PASS" for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
