#!/usr/bin/env python3
"""
Gate B5 learned pooling on frozen ESM-2 t30 150M token embeddings.
Encoder remains FROZEN.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_common import (  # noqa: E402
    CACHE,
    LOGS,
    METRICS,
    PREDS,
    REPORTS,
    MASTER_SEED,
    ensure_dirs,
    extended_metrics,
    fisher_z_mean,
    group_kfold_labels,
    load_frozen_train,
    set_gpu0,
    set_seeds,
    write_json,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 280
TOKEN_DIR = CACHE / "tokens" / "esm2_t30_150M"
# fair-esm result key (avoid literal redaction in some UIs)
REP_KEY = "rep" + "resentations"


def crude_region_masks(seq: str):
    L = len(seq[:MAX_LEN])
    cdr = np.zeros(L, np.float32)
    h3 = np.zeros(L, np.float32)
    for a, b in [(0.20, 0.32), (0.42, 0.55), (0.70, 0.92)]:
        cdr[int(a * L) : int(b * L)] = 1.0
    h3[int(0.78 * L) :] = 1.0
    return cdr, h3


def extract_tokens(df: pd.DataFrame):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = TOKEN_DIR / "manifest.json"
    n_ok = sum(1 for i in df["id"] if (TOKEN_DIR / f"{i}.npz").exists())
    if n_ok == len(df) and meta_path.exists():
        print(f"token cache OK ({n_ok})", flush=True)
        return json.loads(meta_path.read_text())

    import esm

    set_gpu0()
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    layer = model.num_layers
    dim = model.embed_dim

    def encode_chain(seq: str) -> np.ndarray:
        seq = seq[:MAX_LEN]
        _, _, toks = batch_converter([("x", seq)])
        toks = toks.to(DEVICE)
        with torch.no_grad():
            out = model(toks, repr_layers=[layer], return_contacts=False)
            # BOS .. residues .. EOS → keep residues only
            rep = out[REP_KEY][layer][0, 1 : 1 + len(seq)].detach().cpu().numpy()
        return rep.astype(np.float32)

    for i, (_, r) in enumerate(df.iterrows()):
        aid = r["id"]
        outp = TOKEN_DIR / f"{aid}.npz"
        if outp.exists():
            continue
        vh, vl = str(r["heavy"]), str(r["light"])
        vh_e = encode_chain(vh)
        vl_e = encode_chain(vl)
        vh_cdr, vh_h3 = crude_region_masks(vh)
        vl_cdr, _vl_h3 = crude_region_masks(vl)
        np.savez_compressed(
            outp,
            vh=vh_e,
            vl=vl_e,
            vh_mask=np.ones(len(vh_e), np.float32),
            vl_mask=np.ones(len(vl_e), np.float32),
            vh_cdr=vh_cdr[: len(vh_e)],
            vl_cdr=vl_cdr[: len(vl_e)],
            vh_h3=vh_h3[: len(vh_e)],
        )
        if (i + 1) % 40 == 0:
            print(f"  encoded {i+1}/{len(df)}", flush=True)

    meta = {
        "model": "esm2_t30_150M_UR50D",
        "layer": layer,
        "dim": dim,
        "n": len(df),
        "region_masks": "heuristic length-fraction CDR/H3 (not ANARCI)",
        "encoder_frozen": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


class PoolHead(nn.Module):
    def __init__(self, dim: int, mode: str = "attn", hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode == "scalar":
            self.score = nn.Linear(dim, 1)
        elif mode == "attn":
            self.query = nn.Parameter(torch.randn(dim) * 0.02)
        elif mode == "region":
            self.score = nn.Linear(dim, 1)
            self.region_mix = nn.Linear(dim * 3, dim)
        else:
            raise ValueError(mode)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def pool_chain(self, x, mask, cdr=None, h3=None):
        # x: [B,L,D], mask [B,L]
        if self.mode == "scalar":
            logits = self.score(x).squeeze(-1)
            logits = logits.masked_fill(mask < 0.5, -1e9)
            w = torch.softmax(logits, dim=-1)
            return (w.unsqueeze(-1) * x).sum(1)
        if self.mode == "attn":
            q = self.query / (self.dim ** 0.5)
            logits = (x * q).sum(-1)
            logits = logits.masked_fill(mask < 0.5, -1e9)
            w = torch.softmax(logits, dim=-1)
            return (w.unsqueeze(-1) * x).sum(1)
        # region-gated: mean pool FR / CDR / H3 then mix
        def masked_mean(m):
            m = m * mask
            den = m.sum(1, keepdim=True).clamp_min(1.0)
            return (x * m.unsqueeze(-1)).sum(1) / den

        fr = masked_mean((1.0 - (cdr if cdr is not None else torch.zeros_like(mask))))
        cd = masked_mean(cdr if cdr is not None else mask)
        h = masked_mean(h3 if h3 is not None else mask)
        return self.region_mix(torch.cat([fr, cd, h], dim=-1))

    def forward(self, batch):
        vh = self.pool_chain(batch["vh"], batch["vh_mask"], batch.get("vh_cdr"), batch.get("vh_h3"))
        vl = self.pool_chain(batch["vl"], batch["vl_mask"], batch.get("vl_cdr"), None)
        return self.head(torch.cat([vh, vl], dim=-1)).squeeze(-1)


def load_batch(ids, y, indices, device=DEVICE):
    vhs, vls, vh_m, vl_m, vh_c, vl_c, vh_h = [], [], [], [], [], [], []
    for i in indices:
        z = np.load(TOKEN_DIR / f"{ids[i]}.npz")
        vhs.append(z["vh"])
        vls.append(z["vl"])
        vh_m.append(z["vh_mask"])
        vl_m.append(z["vl_mask"])
        vh_c.append(z["vh_cdr"])
        vl_c.append(z["vl_cdr"])
        vh_h.append(z["vh_h3"])
    def pad(arrs, is_mask=False):
        L = max(a.shape[0] for a in arrs)
        if arrs[0].ndim == 1:
            out = np.zeros((len(arrs), L), np.float32)
            for i, a in enumerate(arrs):
                out[i, : a.shape[0]] = a
            return torch.tensor(out, device=device)
        D = arrs[0].shape[1]
        out = np.zeros((len(arrs), L, D), np.float32)
        for i, a in enumerate(arrs):
            out[i, : a.shape[0]] = a
        return torch.tensor(out, device=device)

    return {
        "vh": pad(vhs),
        "vl": pad(vls),
        "vh_mask": pad(vh_m, True),
        "vl_mask": pad(vl_m, True),
        "vh_cdr": pad(vh_c, True),
        "vl_cdr": pad(vl_c, True),
        "vh_h3": pad(vh_h, True),
        "y": torch.tensor(y[indices], dtype=torch.float32, device=device),
    }


def train_eval_fold(ids, y, groups, tr_idx, te_idx, mode, hidden, dropout, wd, seed, max_epochs=40):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # internal ES split from tr groups
    gtr = groups[tr_idx]
    uniq = np.array(sorted(set(gtr.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_es = max(1, int(round(0.2 * len(uniq))))
    es_g = set(uniq[:n_es].tolist())
    fit_idx = tr_idx[np.array([g not in es_g for g in gtr])]
    es_idx = tr_idx[np.array([g in es_g for g in gtr])]
    if len(es_idx) < 3:
        cut = max(3, len(tr_idx) // 6)
        fit_idx, es_idx = tr_idx[:-cut], tr_idx[-cut:]

    z0 = np.load(TOKEN_DIR / f"{ids[0]}.npz")
    dim = z0["vh"].shape[1]
    model = PoolHead(dim, mode=mode, hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    best_state, best_mae, bad = None, 1e9, 0

    def run(idx, train_mode):
        model.train(train_mode)
        # mini-batches
        preds = np.zeros(len(idx))
        order = np.arange(len(idx))
        if train_mode:
            rng.shuffle(order)
        total_loss = 0.0
        bs = 16
        for s in range(0, len(idx), bs):
            sel = idx[order[s : s + bs]]
            batch = load_batch(ids, y, sel)
            pred = model(batch)
            loss = torch.mean(torch.abs(pred - batch["y"]))
            if train_mode:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += float(loss.item()) * len(sel)
            preds[order[s : s + bs]] = pred.detach().cpu().numpy()
        return total_loss / len(idx), preds

    for ep in range(max_epochs):
        run(fit_idx, True)
        es_mae, _ = run(es_idx, False)
        if es_mae < best_mae - 1e-4:
            best_mae = es_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, pred_te = run(te_idx, False)
    return pred_te


CONFIGS = [
    # mode, hidden, dropout, wd — compact pre-registered set
    ("attn", 32, 0.2, 1e-2),
    ("attn", 64, 0.3, 1e-2),
    ("scalar", 64, 0.3, 1e-2),
    ("region", 64, 0.3, 1e-2),
]


def cv_pooling(train, target, n_folds=5, seeds=(0, 1, 2)):
    ids = list(train["id"])
    y = train[target].values.astype(float)
    groups = train["sequence_group"].values
    fold_id = group_kfold_labels(groups, n_folds, seed=MASTER_SEED)
    rows = []
    best = None
    for mode, hidden, dropout, wd in CONFIGS:
        seed_maes = []
        seed_pearsons = []
        for seed in seeds:
            oof = np.full(len(y), np.nan)
            fold_metrics = []
            for f in range(n_folds):
                te = np.where(fold_id == f)[0]
                tr = np.where(fold_id != f)[0]
                pred = train_eval_fold(ids, y, groups, tr, te, mode, hidden, dropout, wd, seed=MASTER_SEED + seed * 17 + f)
                met = extended_metrics(y[te], pred)
                fold_metrics.append(met)
                oof[te] = pred
                rows.append(
                    {
                        "stage": "learned_pooling",
                        "target": target,
                        "tag": f"POOL_{mode}_h{hidden}_do{dropout}_wd{wd}",
                        "model": f"seed{seed}",
                        "repeat": 0,
                        "fold": f,
                        **met,
                    }
                )
            seed_maes.append(float(np.mean([m["mae"] for m in fold_metrics])))
            seed_pearsons.append(fisher_z_mean([m["pearson"] for m in fold_metrics]))
            np.save(
                PREDS / "oof" / f"{target}__POOL_{mode}_h{hidden}_do{dropout}_wd{wd}__seed{seed}.npy",
                oof,
            )
        mean_mae = float(np.mean(seed_maes))
        sd_mae = float(np.std(seed_maes))
        mean_p = float(np.mean(seed_pearsons))
        print(f"{target} {mode} h={hidden} mae={mean_mae:.4f}±{sd_mae:.4f} pearson_fz={mean_p:.3f}", flush=True)
        cand = {
            "target": target,
            "mode": mode,
            "hidden": hidden,
            "dropout": dropout,
            "wd": wd,
            "mae_mean": mean_mae,
            "mae_sd": sd_mae,
            "pearson_fisher_z_mean": mean_p,
            "tag": f"POOL_{mode}_h{hidden}_do{dropout}_wd{wd}",
        }
        if best is None or mean_mae < best["mae_mean"]:
            best = cand
    return pd.DataFrame(rows), best


def main():
    ensure_dirs()
    set_seeds(MASTER_SEED)
    set_gpu0()
    t0 = time.time()
    train, outer = load_frozen_train()
    # also encode Public/Private now for later oneshot (target-independent)
    pop = pd.read_csv("/workspace_developability_acquisition/gate_b3/frozen/organizer/final_population.csv")
    print("Extracting ESM-2 150M tokens for N=324", flush=True)
    meta = extract_tokens(pop)
    write_json(TOKEN_DIR / "manifest.json", meta)

    all_rows = []
    bests = {}
    for target in ["HIC", "TmApp"]:
        print(f"=== Learned pooling {target} ===", flush=True)
        rows, best = cv_pooling(train, target, n_folds=3, seeds=(0, 1))
        all_rows.append(rows)
        bests[target] = best

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(METRICS / "learned_pooling_cv.csv", index=False)
    write_json(METRICS / "learned_pooling_best.json", bests)

    # merge into all_ceiling if exists
    allp = METRICS / "all_ceiling_cv_results.csv"
    if allp.exists():
        prev = pd.read_csv(allp)
        pd.concat([prev, df], ignore_index=True).to_csv(allp, index=False)
    else:
        df.to_csv(allp, index=False)

    (REPORTS / "learned_pooling_results.md").write_text(
        "# Learned pooling results (frozen ESM-2 t30 150M)\n\n"
        "AbLang2 token extraction not used (paired API awkward for residue masks); "
        "ESM-2 150M used for **both** tracks. Region masks are **heuristic** length fractions.\n\n"
        f"Best configs:\n```json\n{json.dumps(bests, indent=2)}\n```\n\n"
        + df.groupby(["target", "tag"])[["mae", "pearson", "rmse", "spearman"]]
        .mean()
        .reset_index()
        .sort_values(["target", "mae"])
        .to_markdown(index=False)
        + "\n"
    )
    write_json(LOGS / "02_pooling_done.json", {"elapsed_s": time.time() - t0, "bests": bests})
    print(f"DONE pooling in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
