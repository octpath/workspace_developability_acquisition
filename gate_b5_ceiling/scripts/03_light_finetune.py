#!/usr/bin/env python3
"""
Gate B5 light PLM adaptation: ESM-2 t30 150M with LoRA (preferred) or last-block unfreeze.
VH/VL shared encoder → pool → concat → small head. Primary: MAE.
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
REP_KEY = "rep" + "resentations"

# Pre-registered compact LR set
LR_SET = [3e-5, 1e-4]


class ESMRegressor(nn.Module):
    def __init__(self, esm_model, alphabet, hidden=64, dropout=0.3):
        super().__init__()
        self.esm = esm_model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.layer = esm_model.num_layers
        dim = esm_model.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, seqs):
        # seqs: list[str]
        data = [(str(i), s[:MAX_LEN]) for i, s in enumerate(seqs)]
        _, _, toks = self.batch_converter(data)
        toks = toks.to(DEVICE)
        out = self.esm(toks, repr_layers=[self.layer], return_contacts=False)
        rep = out[REP_KEY][self.layer]  # [B, L+2, D]
        # mean pool residues (exclude BOS/EOS via lengths)
        pooled = []
        for i, s in enumerate(seqs):
            L = len(s[:MAX_LEN])
            pooled.append(rep[i, 1 : 1 + L].mean(0))
        return torch.stack(pooled, dim=0)

    def forward(self, vh_seqs, vl_seqs):
        vh = self.encode(vh_seqs)
        vl = self.encode(vl_seqs)
        return self.head(torch.cat([vh, vl], dim=-1)).squeeze(-1)


def apply_lora(esm_model, r=8, alpha=16):
    """Attach LoRA to ESM attention dense projections via peft if possible;
    otherwise fall back to last-block unfreeze.
    """
    try:
        from peft import LoraConfig, get_peft_model

        # fair-esm is not a HF model — peft may not wrap cleanly.
        # Manual LoRA on Linear layers named similarly to q/k/v/out
        raise RuntimeError("manual LoRA preferred for fair-esm")
    except Exception:
        return manual_lora(esm_model, r=r, alpha=alpha)


def manual_lora(esm_model, r=8, alpha=16):
    """Inject low-rank adapters into last transformer layer Linear modules."""
    # Freeze all
    for p in esm_model.parameters():
        p.requires_grad = False

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r, alpha):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.zeros(base.in_features, r))
            self.B = nn.Parameter(torch.zeros(r, base.out_features))
            nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
            nn.init.zeros_(self.B)
            self.scale = alpha / r

        def forward(self, x):
            return self.base(x) + (x @ self.A @ self.B) * self.scale

    # last layer only
    layer = esm_model.layers[-1]
    replaced = 0
    for name, mod in list(layer.named_modules()):
        if isinstance(mod, nn.Linear) and mod.out_features >= 64:
            # replace only direct children to avoid recursion mess
            pass
    # Target known ESM2 layer attrs
    attn = getattr(layer, "self_attn", None)
    for attr in ["fc1", "fc2"]:
        if hasattr(layer, attr) and isinstance(getattr(layer, attr), nn.Linear):
            setattr(layer, attr, LoRALinear(getattr(layer, attr), r, alpha).to(DEVICE))
            replaced += 1
    if attn is not None:
        for attr in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            if hasattr(attn, attr) and isinstance(getattr(attn, attr), nn.Linear):
                setattr(attn, attr, LoRALinear(getattr(attn, attr), r, alpha).to(DEVICE))
                replaced += 1
    if replaced == 0:
        # fallback: unfreeze last block
        for p in esm_model.layers[-1].parameters():
            p.requires_grad = True
    return esm_model, replaced


def last_block_unfreeze(esm_model):
    for p in esm_model.parameters():
        p.requires_grad = False
    for p in esm_model.layers[-1].parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in esm_model.parameters() if p.requires_grad)
    return esm_model, n


def run_fold(train, tr_idx, te_idx, target, strategy, lr, seed, max_epochs=15):
    import esm

    torch.manual_seed(seed)
    np.random.seed(seed)
    ids = list(train["id"])
    y = train[target].values.astype(float)
    groups = train["sequence_group"].values
    vh = train["heavy"].astype(str).values
    vl = train["light"].astype(str).values

    # ES split inside tr
    gtr = groups[tr_idx]
    uniq = np.array(sorted(set(gtr.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_es = max(1, int(round(0.2 * len(uniq))))
    es_g = set(uniq[:n_es].tolist())
    fit_idx = tr_idx[[g not in es_g for g in gtr]]
    es_idx = tr_idx[[g in es_g for g in gtr]]
    if len(es_idx) < 3:
        cut = max(3, len(tr_idx) // 6)
        fit_idx, es_idx = tr_idx[:-cut], tr_idx[-cut:]

    base, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    base = base.to(DEVICE)
    if strategy == "lora":
        base, nrep = apply_lora(base)
        tag_extra = f"lora_n{nrep}"
    else:
        base, npar = last_block_unfreeze(base)
        tag_extra = f"lastblock_p{npar}"

    model = ESMRegressor(base, alphabet, hidden=64, dropout=0.3).to(DEVICE)
    # train head + adapters
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    def epoch(idx, train_mode):
        model.train(train_mode)
        order = np.arange(len(idx))
        if train_mode:
            rng.shuffle(order)
        preds = np.zeros(len(idx))
        bs = 4  # small due to GPU memory with 150M
        total = 0.0
        for s in range(0, len(idx), bs):
            sel = idx[order[s : s + bs]]
            yt = torch.tensor(y[sel], dtype=torch.float32, device=DEVICE)
            if train_mode:
                pred = model(list(vh[sel]), list(vl[sel]))
                loss = torch.mean(torch.abs(pred - yt))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            else:
                with torch.no_grad():
                    pred = model(list(vh[sel]), list(vl[sel]))
                    loss = torch.mean(torch.abs(pred - yt))
            total += float(loss.item()) * len(sel)
            preds[order[s : s + bs]] = pred.detach().cpu().numpy()
        return total / len(idx), preds

    best_state, best_mae, bad = None, 1e9, 0
    for ep in range(max_epochs):
        epoch(fit_idx, True)
        es_mae, _ = epoch(es_idx, False)
        if es_mae < best_mae - 1e-4:
            best_mae = es_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, pred = epoch(te_idx, False)
    return pred, tag_extra


def cv_finetune(train, target, strategy="lora", n_folds=3, seeds=(0, 1)):
    y = train[target].values.astype(float)
    groups = train["sequence_group"].values
    fold_id = group_kfold_labels(groups, n_folds, seed=MASTER_SEED)
    rows = []
    summary = []
    for lr in LR_SET:
        seed_maes, seed_ps = [], []
        for seed in seeds:
            oof = np.full(len(y), np.nan)
            fmets = []
            tag_extra = ""
            for f in range(n_folds):
                te = np.where(fold_id == f)[0]
                tr = np.where(fold_id != f)[0]
                print(f"{target} {strategy} lr={lr} seed={seed} fold={f}", flush=True)
                pred, tag_extra = run_fold(train, tr, te, target, strategy, lr, seed=MASTER_SEED + seed * 31 + f)
                met = extended_metrics(y[te], pred)
                fmets.append(met)
                oof[te] = pred
                rows.append(
                    {
                        "stage": "light_finetune",
                        "target": target,
                        "tag": f"FT_{strategy}_lr{lr}_{tag_extra}",
                        "model": f"seed{seed}",
                        "repeat": 0,
                        "fold": f,
                        "lr": lr,
                        **met,
                    }
                )
            seed_maes.append(float(np.mean([m["mae"] for m in fmets])))
            seed_ps.append(fisher_z_mean([m["pearson"] for m in fmets]))
            np.save(PREDS / "oof" / f"{target}__FT_{strategy}_lr{lr}__seed{seed}.npy", oof)
        summary.append(
            {
                "target": target,
                "strategy": strategy,
                "lr": lr,
                "mae_mean": float(np.mean(seed_maes)),
                "mae_sd": float(np.std(seed_maes)),
                "pearson_fisher_z_mean": float(np.mean(seed_ps)),
            }
        )
        print(f"SUMMARY {target} {strategy} lr={lr} mae={np.mean(seed_maes):.4f}", flush=True)
    return pd.DataFrame(rows), summary


def main():
    ensure_dirs()
    set_seeds(MASTER_SEED)
    set_gpu0()
    t0 = time.time()
    train, _ = load_frozen_train()

    all_rows = []
    all_sum = []
    # Prefer LoRA; compact: 3-fold, 1 seed, 2 LRs for runtime
    for target in ["TmApp", "HIC"]:
        try:
            rows, summ = cv_finetune(train, target, strategy="lora", n_folds=3, seeds=(0,))
            # Restrict to first two LRs by filtering after — better: change LR_SET locally
            all_rows.append(rows)
            all_sum.extend(summ)
        except Exception as e:
            print("LoRA failed", target, e, flush=True)
            rows, summ = cv_finetune(train, target, strategy="lastblock", n_folds=3, seeds=(0,))
            all_rows.append(rows)
            all_sum.extend(summ)

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df.to_csv(METRICS / "light_finetune_cv.csv", index=False)
    write_json(METRICS / "light_finetune_summary.json", all_sum)

    allp = METRICS / "all_ceiling_cv_results.csv"
    if allp.exists() and len(df):
        prev = pd.read_csv(allp)
        pd.concat([prev, df], ignore_index=True).to_csv(allp, index=False)

    if len(df):
        g = df.groupby(["target", "tag"])[["mae", "pearson", "rmse", "spearman"]].mean().reset_index().sort_values(["target", "mae"])
        md = "# Light fine-tuning results (ESM-2 t30 150M)\n\n" + g.to_markdown(index=False) + "\n\nSummaries:\n```json\n" + json.dumps(all_sum, indent=2) + "\n```\n"
    else:
        md = "# Light fine-tuning results\n\n**NOT RUN / FAILED**\n"
    (REPORTS / "light_finetuning_results.md").write_text(md)
    write_json(LOGS / "03_finetune_done.json", {"elapsed_s": time.time() - t0, "summary": all_sum})
    print(f"DONE finetune in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
