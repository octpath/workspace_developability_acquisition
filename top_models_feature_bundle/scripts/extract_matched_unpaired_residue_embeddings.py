#!/usr/bin/env python3
"""Extract matched AbLang2 joint-paired and CurrAb unpaired residue bundles.

AbLang2 audit (critical):
  The historical residue_level/ablang2/ bundle was extracted with *single-chain*
  formats (<H>| and |<L>) using the ablang2-paired *checkpoint*. Factorial R3
  (ablang2_paired) keeps that existing asset for reuse of T110–T149.

  Factorial R4 (ablang2_unpaired allocation name) stores the complementary
  *joint* paired inference (<H>|<L> one forward) with the SAME checkpoint so the
  matched inference-context contrast is scientifically available. Metadata records
  representation_context=PAIRED_NATIVE and the audit note.

CurrAb unpaired:
  Official forms: Heavy → "H<cls>"; Light → "<cls>L"; same revision as T160.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parent
RES = BUNDLE / "residue_level"
DRILL = ROOT / "developability_drilldown"
MAX_H, MAX_L = 140, 120
CURRAB_REV = "92e28534663e163f1b398f773b3fe041085737d9"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_npy(path: Path) -> None:
    data = path.read_bytes()
    for stale in sorted(path.parent.glob(path.name + ".part*")):
        stale.unlink()
    mid = len(data) // 2
    for i, chunk in enumerate((data[:mid], data[mid:])):
        Path(str(path) + f".part{i}").write_bytes(chunk)
    if Path(str(path) + ".part0").read_bytes() + Path(str(path) + ".part1").read_bytes() != data:
        raise RuntimeError(f"part reassembly failed {path}")


def load_seqs() -> pd.DataFrame:
    ref = [str(x) for x in np.load(RES / "ablang2" / "ids.npy", allow_pickle=True)]
    dev = pd.read_csv(DRILL / "data/dev.csv")
    test = pd.read_csv(DRILL / "data/test.csv")
    seqs = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
    seqs["id"] = seqs["id"].astype(str)
    if set(ref) != set(seqs["id"]):
        raise RuntimeError("ID set mismatch vs ablang2 pack")
    return seqs.set_index("id").loc[ref].reset_index()


def pack_padded(arrays, max_len, hidden):
    N = len(arrays)
    emb = np.zeros((N, max_len, hidden), dtype=np.float16)
    mask = np.zeros((N, max_len), dtype=np.bool_)
    for i, a in enumerate(arrays):
        L = a.shape[0]
        if L > max_len:
            raise RuntimeError("truncation")
        if not np.isfinite(a.astype(np.float32)).all():
            raise RuntimeError("nonfinite")
        emb[i, :L] = a.astype(np.float16)
        mask[i, :L] = True
    return emb, mask


def diagnostics(arrays):
    norms = []
    n_zero = n_nf = n = 0
    for a in arrays:
        a32 = a.astype(np.float32)
        n_nf += int((~np.isfinite(a32)).sum())
        for v in a32:
            n += 1
            nrm = float(np.linalg.norm(v))
            norms.append(nrm)
            if nrm < 1e-12:
                n_zero += 1
    norms = np.asarray(norms, float)
    return {
        "n_vectors": n,
        "mean_l2": float(norms.mean()),
        "std_l2": float(norms.std()),
        "fraction_zero_vectors": float(n_zero / max(n, 1)),
        "n_nonfinite_elements": int(n_nf),
    }


def write_bundle(out: Path, ids, H, L, hidden, meta):
    out.mkdir(parents=True, exist_ok=True)
    eh, mh = pack_padded(H, MAX_H, hidden)
    el, ml = pack_padded(L, MAX_L, hidden)
    np.save(out / "ids.npy", np.asarray(ids, dtype=object))
    np.save(out / "heavy_embeddings.npy", eh)
    np.save(out / "light_embeddings.npy", el)
    np.save(out / "heavy_mask.npy", mh)
    np.save(out / "light_mask.npy", ml)
    for name in ("heavy_embeddings.npy", "light_embeddings.npy"):
        split_npy(out / name)
    meta = dict(meta)
    meta.update(
        {
            "n_ids": len(ids),
            "max_len_H": MAX_H,
            "max_len_L": MAX_L,
            "hidden_dim": int(hidden),
            "dtype_on_disk": "float16",
            "dtype_in_memory": "float32",
            "ids_sha256": sha256_file(out / "ids.npy"),
            "heavy_emb_sha256": sha256_file(out / "heavy_embeddings.npy"),
            "light_emb_sha256": sha256_file(out / "light_embeddings.npy"),
        }
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "EMBEDDING_DIAGNOSTICS.json").write_text(
        json.dumps(diagnostics(H + L), indent=2), encoding="utf-8"
    )


def extract_ablang2_joint(seqs, device):
    """True joint <H>|<L> forward with ablang2-paired checkpoint."""
    import ablang2

    t0 = time.time()
    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device=device)
    ablang.freeze()
    H, L = [], []
    with torch.no_grad():
        for _, row in seqs.iterrows():
            h, l = str(row.heavy), str(row.light)
            fmt = f"<{h}>|<{l}>"
            h_idx = np.arange(1, 1 + len(h), dtype=np.int64)
            l_idx = np.arange(1 + len(h) + 3, 1 + len(h) + 3 + len(l), dtype=np.int64)
            if "".join(fmt[i] for i in h_idx) != h or "".join(fmt[i] for i in l_idx) != l:
                raise RuntimeError(f"AbLang2 joint AA map fail {row.id}")
            tokens = ablang.tokenizer([fmt], pad=True, w_extra_tkns=False, device=device)
            hs = ablang.AbRep(tokens).last_hidden_states[0].detach().float().cpu().numpy()
            he, le = hs[h_idx].astype(np.float32), hs[l_idx].astype(np.float32)
            if he.shape[0] != len(h) or le.shape[0] != len(l):
                raise RuntimeError(f"len fail {row.id}")
            H.append(he)
            L.append(le)
    hidden = int(H[0].shape[1])
    meta = {
        "factorial_representation_id": "ablang2_unpaired",
        "model_to_use": "ablang2-paired",
        "package": "ablang2",
        "package_version": "0.2.1",
        "representation_source": "AbRep.last_hidden_states joint <H>|<L>",
        "representation_context": "PAIRED_NATIVE",
        "paired_vs_separate": "paired",
        "matched_to": "ablang2_paired",
        "raw_hidden_dimension": hidden,
        "mapping": "EXACT_AA_1TO1_SPECIAL_STRIPPED",
        "special_token_mapping_rule": "joint <H>|<L>; keep AA indices only; drop <,>,|",
        "audit_note": (
            "Historical residue_level/ablang2/ used SEPARATE_CHAIN formats with this same "
            "ablang2-paired checkpoint. Factorial allocation name ablang2_unpaired holds the "
            "complementary JOINT paired inference so matched context contrast is available; "
            "see TMAPP_REP_TOPO_ANNOT_FACTORIAL_PREREG.md."
        ),
        "license_status": "REVIEW_MODEL_OUTPUT",
        "extraction_wall_time_sec": time.time() - t0,
        "device": device,
    }
    return H, L, hidden, meta


def extract_currab_unpaired(seqs, device):
    from transformers import EsmForMaskedLM, EsmTokenizer
    import transformers

    t0 = time.time()
    repo = "brineylab/CurrAb"
    tok = EsmTokenizer.from_pretrained(repo, revision=CURRAB_REV)
    model = EsmForMaskedLM.from_pretrained(repo, revision=CURRAB_REV)
    model = model.eval().to(device)
    H, L = [], []
    hidden = int(model.config.hidden_size)
    with torch.no_grad():
        for _, row in seqs.iterrows():
            h, l = str(row.heavy), str(row.light)
            # Official unpaired: Heavy → H<cls> ; Light → <cls>L
            for chain, s, expect_len, sink in (
                ("H", f"{h}<cls>", len(h), H),
                ("L", f"<cls>{l}", len(l), L),
            ):
                enc = tok(s, return_tensors="pt", add_special_tokens=False)
                tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
                if chain == "H":
                    if tokens[-1] != "<cls>" or len(tokens) != len(h) + 1:
                        raise RuntimeError(f"CurrAb unpaired H format {row.id}: {tokens[:3]}..{tokens[-2:]}")
                    if "".join(tokens[:-1]) != h:
                        raise RuntimeError(f"CurrAb unpaired H AA {row.id}")
                    aa_slice = slice(0, len(h))
                else:
                    if tokens[0] != "<cls>" or len(tokens) != len(l) + 1:
                        raise RuntimeError(f"CurrAb unpaired L format {row.id}")
                    if "".join(tokens[1:]) != l:
                        raise RuntimeError(f"CurrAb unpaired L AA {row.id}")
                    aa_slice = slice(1, 1 + len(l))
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states[-1][0].detach().float().cpu().numpy()
                aa = hs[aa_slice].astype(np.float32)
                if aa.shape[0] != expect_len:
                    raise RuntimeError(f"CurrAb unpaired len {row.id} {chain}")
                sink.append(aa)
    meta = {
        "factorial_representation_id": "currab_unpaired",
        "model_name": repo,
        "model_revision": CURRAB_REV,
        "package": "transformers",
        "package_version": transformers.__version__,
        "representation_source": "EsmForMaskedLM last_hidden_state",
        "representation_context": "SEPARATE_CHAIN",
        "paired_vs_separate": "separate",
        "matched_to": "currab_paired",
        "raw_hidden_dimension": hidden,
        "native_input_convention": "unpaired Heavy<cls> and <cls>Light; add_special_tokens=False",
        "special_token_mapping_rule": "exclude trailing/leading <cls>; no BOS/EOS",
        "mapping": "EXACT_AA_1TO1_CLS_STRIPPED",
        "license_status": "REVIEW_MODEL_OUTPUT",
        "extraction_wall_time_sec": time.time() - t0,
        "device": device,
    }
    return H, L, hidden, meta


def reextract_check(plm, seqs, device, out, fn, n=3):
    sub = seqs.head(n)
    H, L, _, _ = fn(sub, device)
    eh = np.load(out / "heavy_embeddings.npy")
    el = np.load(out / "light_embeddings.npy")
    mh = np.load(out / "heavy_mask.npy")
    ml = np.load(out / "light_mask.npy")
    max_abs = 0.0
    for i in range(n):
        lh, ll = int(mh[i].sum()), int(ml[i].sum())
        dh = float(np.max(np.abs(eh[i, :lh].astype(np.float32) - H[i].astype(np.float16).astype(np.float32))))
        dl = float(np.max(np.abs(el[i, :ll].astype(np.float32) - L[i].astype(np.float16).astype(np.float32))))
        max_abs = max(max_abs, dh, dl)
    chk = {"plm": plm, "n": n, "max_abs_diff_vs_fp16_roundtrip": max_abs, "tolerance": 1e-5, "pass": max_abs <= 1e-5}
    (out / "REEXTRACT_CHECK.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plm", nargs="+", default=["ablang2_unpaired", "currab_unpaired"])
    args = ap.parse_args()
    seqs = load_seqs()
    ids = seqs["id"].tolist()
    status = {}
    extractors = {
        "ablang2_unpaired": (RES / "ablang2_unpaired", extract_ablang2_joint),
        "currab_unpaired": (RES / "currab_unpaired", extract_currab_unpaired),
    }
    for plm in args.plm:
        out, fn = extractors[plm]
        print(f"==== {plm} -> {out} ====", flush=True)
        H, L, hidden, meta = fn(seqs, args.device)
        for i, row in enumerate(seqs.itertuples(index=False)):
            if H[i].shape[0] != len(str(row.heavy)) or L[i].shape[0] != len(str(row.light)):
                raise RuntimeError(f"QC fail {row.id}")
        write_bundle(out, ids, H, L, hidden, meta)
        chk = reextract_check(plm, seqs, args.device, out, fn)
        print("reextract", chk, flush=True)
        status[plm] = "PASS" if chk["pass"] else "FAIL"
    (RES / "MATCHED_UNPAIRED_EXTRACTION_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STATUS", status, flush=True)
    return 0 if all(v == "PASS" for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
