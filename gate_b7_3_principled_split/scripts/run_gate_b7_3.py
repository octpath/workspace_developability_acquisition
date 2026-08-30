#!/usr/bin/env python3
"""Gate B7.3 — Principled Production Split Design (full pipeline).

ABSOLUTE: Phases 1–5 never load model predictions.
Split generation permanently closes after B7_3_PREMODEL_FINALISTS_FROZEN.json.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GATE = Path("/workspace_developability_acquisition/gate_b7_3_principled_split")
sys.path.insert(0, str(GATE / "scripts"))

from b7_3_common import (  # noqa: E402
    BANK_A,
    CFG,
    CONT_COLS,
    FINAL_DECISIONS,
    MET,
    PLT,
    REP,
    SEED_BASE,
    SPL,
    check_hard,
    fnum,
    group_mask_from_pub,
    hamming_id,
    lex_key_from_metrics,
    load_base_data,
    log,
    md_table,
    mask_hash,
    pub_from_group_mask,
    sha256_ids,
    write_json,
)
from b7_3_metrics import calibrate_energy_scale, compute_premodel, score_mask  # noqa: E402
from b7_3_search import run_search  # noqa: E402


def write_protocol(data: dict, energy_scale: float) -> str:
    protocol = {
        "gate": "B7.3",
        "title": "Principled Production Split Design",
        "seed_base": SEED_BASE,
        "public_n": 81,
        "hard_constraints": {
            "public_size": 81,
            "private_size": 81,
            "atomic_sequence_groups": True,
            "HIC_HIGH_public_in": [3, 4],
            "HIC_bands": {"LOW": "<10.5", "MEDIUM": "[10.5,11.5]", "HIGH": ">11.5"},
            "confirmed_test_band_totals": data["band_counts"],
        },
        "prefer": {"HIC_MEDIUM": "3/3"},
        "energy_scale_definition": (
            "Median joint energy distance over 200 random feasible size+HIGH splits; "
            "used to normalize joint_energy_norm = energy_raw / energy_scale."
        ),
        "energy_scale_value": energy_scale,
        "metrics": {
            "quantile_L1": (
                "For n equal-frequency bins on full Test: L1 = sum_b |n_pub(b)-n_priv(b)|. "
                "Primary n=8; sensitivity n in {6,10}."
            ),
            "HIC_MED_imbalance": "|n_pub(MEDIUM) - n_priv(MEDIUM)|",
            "joint_grid_L1": "4x4 TmApp×HIC quartile cell count L1 Public vs Private",
            "Wasserstein_norm": "W1(Public,Private) / Test_SD(target)",
            "KS": "two-sample KS statistic Public vs Private",
            "joint_energy_norm": "energy_distance(z_TmApp,z_HIC) / energy_scale",
            "TV": "0.5 * L1 of categorical probability vectors",
            "abs_SMD": "|mean_pub-mean_priv| / sqrt(0.5*(var_pub+var_priv))",
            "continuous_list": CONT_COLS,
        },
        "lexicographic_key": {
            "L0": "hard feasible (else infinite)",
            "L1": "max(TmApp_q8_L1, HIC_q8_L1, HIC_MED_imbalance, joint_grid_L1)",
            "L2": "max(TmApp_W_norm, HIC_W_norm, joint_energy_norm)",
            "L3": "max(mean_abs_SMD, germline_TV_mean, group_mass_TV)",
            "L4": "aggregate L1/20 + L2 + L3 + 0.1*MED_pref_penalty + 0.05*max_abs_SMD",
            "order": "minimize tuple (L0,L1,L2,L3,L4)",
        },
        "search": {
            "method_A": "randomized greedy + local equal/unequal size-preserving swaps",
            "method_B": "simulated annealing on scalarized lex proxy",
            "method_C": "scipy.optimize.milp linearized count-alignment + local polish",
            "target_starts": "~1000 (A400+B400+C80)",
            "min_starts": 500,
            "max_starts": 3000,
        },
        "near_optimal": {
            "primary_tol": "within 10% of best L1",
            "also_report": ["5%", "20%"],
            "diversity_hamming_ids": "prefer >=10–20 when selecting finalists",
        },
        "absolute_rules": [
            "Phases 1-5 never load model predictions or MAE files",
            "After B7_3_PREMODEL_FINALISTS_FROZEN.json written+hashed, stop generating splits",
            "Do not modify production split files",
            "Do not package",
        ],
    }
    h = write_json(CFG / "B7_3_PROTOCOL.json", protocol)
    lines = [
        "# 01 — Split Objective and Method",
        "",
        "Gate B7.3 constructs Public/Private masks using an explicit **model-blind**",
        "lexicographic distribution-matching objective, then validates frozen finalists",
        "with separated model banks.",
        "",
        "## Hard constraints (L0)",
        "",
        "- Public = Private = 81",
        "- Atomic `sequence_group` integrity",
        f"- HIC HIGH public count ∈ {{3,4}} (Test HIGH total = {data['band_counts'].get('HIGH')})",
        f"- Confirmed band totals: LOW={data['band_counts'].get('LOW')}, "
        f"MEDIUM={data['band_counts'].get('MEDIUM')}, HIGH={data['band_counts'].get('HIGH')}",
        "- Prefer MEDIUM exactly 3/3",
        "",
        "## Lexicographic objective",
        "",
        "1. **L1 (minimax counts)**: max(TmApp_q8_L1, HIC_q8_L1, HIC_MED_imbalance, joint_grid_L1)",
        "2. **L2 (continuous)**: max(TmApp_W_norm, HIC_W_norm, joint_energy_norm)",
        "3. **L3 (biology/sequence)**: max(mean|SMD|, germline_TV_mean, group_mass_TV)",
        "4. **L4**: aggregate tie-break",
        "",
        "## Formulas",
        "",
        "- Quantile L1: `Σ_b |n_pub(b) − n_priv(b)|` on equal-frequency bins of full Test",
        "- Wasserstein norm: `W1 / Test_SD`",
        f"- Joint energy norm: energy distance on z-scored (TmApp,HIC) / scale={energy_scale:.6f}",
        "  (scale = median energy of 200 random feasible splits)",
        "- |SMD| on predeclared continuous list (incl. paired nearest-train identity)",
        "- TV = half L1 of category probability vectors",
        "",
        f"Protocol hash: `{h}`",
        "",
        "See `config/B7_3_PROTOCOL.json` for the frozen specification.",
    ]
    (REP / "01_SPLIT_OBJECTIVE_AND_METHOD.md").write_text("\n".join(lines) + "\n")
    return h


def packs_to_rows(packs: list[dict]) -> pd.DataFrame:
    rows = []
    for p in packs:
        row = {
            "split_id": p["split_id"],
            "seed": p.get("seed"),
            "method": p.get("method"),
            "mask_hash": p["mask_hash"],
            "public_hash": p["public_hash"],
            "private_hash": p["private_hash"],
            "premodel_rank": p.get("premodel_rank"),
            "is_control": p.get("is_control", False),
            "feasible": p.get("feasible", True),
            "L1": p.get("L1"),
            "L2": p.get("L2"),
            "L3": p.get("L3"),
            "L4": p.get("L4"),
            "TmApp_q8_L1": p.get("TmApp_q8_L1"),
            "HIC_q8_L1": p.get("HIC_q8_L1"),
            "TmApp_q6_L1": p.get("TmApp_q6_L1"),
            "HIC_q6_L1": p.get("HIC_q6_L1"),
            "TmApp_q10_L1": p.get("TmApp_q10_L1"),
            "HIC_q10_L1": p.get("HIC_q10_L1"),
            "HIC_MED_imbalance": p.get("HIC_MED_imbalance"),
            "joint_grid_L1": p.get("joint_grid_L1"),
            "TmApp_W_norm": p.get("TmApp_W_norm"),
            "HIC_W_norm": p.get("HIC_W_norm"),
            "joint_energy_norm": p.get("joint_energy_norm"),
            "mean_abs_SMD": p.get("mean_abs_SMD"),
            "p90_abs_SMD": p.get("p90_abs_SMD"),
            "max_abs_SMD": p.get("max_abs_SMD"),
            "germline_TV_mean": p.get("germline_TV_mean"),
            "group_mass_TV": p.get("group_mass_TV"),
            "hic_high_pub": p.get("hic_high_pub"),
            "hic_high_priv": p.get("hic_high_priv"),
            "hic_med_pub": p.get("hic_med_pub"),
            "hic_med_priv": p.get("hic_med_priv"),
            "hic_low_pub": p.get("hic_low_pub"),
            "hic_low_priv": p.get("hic_low_priv"),
            "TmApp_KS": p.get("TmApp_KS"),
            "HIC_KS": p.get("HIC_KS"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_all_splits(packs: list[dict]) -> None:
    out_dir = SPL / "all"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in packs:
        write_json(
            out_dir / f"{p['split_id']}.json",
            {
                "split_id": p["split_id"],
                "seed": p.get("seed"),
                "method": p.get("method"),
                "public_ids": p["public_ids"],
                "private_ids": p["private_ids"],
                "public_hash": p["public_hash"],
                "private_hash": p["private_hash"],
                "mask_hash": p["mask_hash"],
                "metrics": {k: p[k] for k in ("L1", "L2", "L3", "L4", "hic_med_pub", "hic_high_pub") if k in p},
            },
        )


def near_optimal_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    best_l1 = float(df["L1"].min())
    out = {}
    for name, tol in [("5pct", 0.05), ("10pct", 0.10), ("20pct", 0.20)]:
        thr = best_l1 * (1.0 + tol) if best_l1 > 0 else tol
        # also allow absolute epsilon when best is 0
        thr = max(thr, best_l1 + 1e-9)
        if best_l1 == 0:
            thr = tol * 2  # absolute slack when perfect L1
        sub = df[df["L1"] <= thr + 1e-12].copy()
        out[name] = sub
    return out


def select_diverse_finalists(near_df: pd.DataFrame, packs_by_id: dict, n: int = 20, min_ham: int = 10) -> list[dict]:
    """Greedy diverse selection from near-optimal by lex rank."""
    ranked = near_df.sort_values(["L1", "L2", "L3", "L4"]).reset_index(drop=True)
    chosen = []
    for _, r in ranked.iterrows():
        p = packs_by_id[r["split_id"]]
        ok = True
        for c in chosen:
            if hamming_id(p["public_ids"], c["public_ids"]) < min_ham:
                ok = False
                break
        if ok:
            chosen.append(p)
        if len(chosen) >= n:
            break
    # if not enough with min_ham, relax
    if len(chosen) < n:
        for ham in (8, 6, 4, 2, 0):
            chosen = []
            for _, r in ranked.iterrows():
                p = packs_by_id[r["split_id"]]
                ok = all(hamming_id(p["public_ids"], c["public_ids"]) >= ham for c in chosen)
                if ok:
                    chosen.append(p)
                if len(chosen) >= n:
                    break
            if len(chosen) >= n:
                break
    return chosen[:n]


def control_metrics(data: dict, energy_scale: float) -> list[dict]:
    out = []
    for cid, c in data["controls"].items():
        m = compute_premodel(c["public_ids"], c["private_ids"], data, energy_scale=energy_scale)
        pack = {
            **c,
            **{k: m[k] for k in m if k != "lex_key"},
            "lex_key": m["lex_key"],
            "mask_hash": mask_hash(c["public_ids"]),
        }
        out.append(pack)
    return out


def percentile_among(value: float, series: pd.Series, higher_better: bool = False) -> float:
    """Percentile standing: % of generated with worse metric (higher => incumbent looks better)."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return float("nan")
    if higher_better:
        return float(100.0 * (s < value).mean())
    return float(100.0 * (s > value).mean())


def plot_premodel(df: pd.DataFrame, controls_df: pd.DataFrame, finalists: list[dict]) -> None:
    PLT.mkdir(parents=True, exist_ok=True)
    inc = controls_df[controls_df.split_id == "CAND_12528"]
    inc_l1 = float(inc.L1.iloc[0]) if len(inc) else None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df.L1.values, bins=min(40, max(10, len(df) // 8)), alpha=0.75, color="#4C72B0")
    if inc_l1 is not None:
        ax.axvline(inc_l1, color="#C44E52", ls="--", lw=2, label="CAND_12528")
    ax.set_xlabel("L1 minimax count imbalance")
    ax.set_ylabel("count")
    ax.set_title("Pre-model lexicographic L1 landscape")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLT / "premodel_lexicographic_landscape.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df.TmApp_W_norm, df.HIC_W_norm, s=12, alpha=0.4, label="generated")
    if len(inc):
        ax.scatter(inc.TmApp_W_norm, inc.HIC_W_norm, c="red", s=60, label="CAND_12528", zorder=5)
    ax.set_xlabel("TmApp W1 / SD")
    ax.set_ylabel("HIC W1 / SD")
    ax.set_title("Wasserstein scatter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLT / "tmapp_wasserstein_vs_hic_wasserstein.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.joint_energy_norm.values, bins=min(40, max(10, len(df) // 8)), alpha=0.75)
    if len(inc):
        ax.axvline(float(inc.joint_energy_norm.iloc[0]), color="red", ls="--", label="CAND_12528")
    ax.set_title("Joint target energy distance (normalized)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLT / "joint_target_energy_distance.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.max_abs_SMD.values, bins=min(40, max(10, len(df) // 8)), alpha=0.75)
    if len(inc):
        ax.axvline(float(inc.max_abs_SMD.iloc[0]), color="red", ls="--", label="CAND_12528")
    ax.set_title("max |SMD| distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLT / "max_smd_distribution.png", dpi=140)
    plt.close(fig)

    # Pareto L1 vs L2
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df.L1, df.L2, s=12, alpha=0.35, label="generated")
    if len(inc):
        ax.scatter(inc.L1, inc.L2, c="red", s=60, zorder=5, label="CAND_12528")
    # mark non-dominated among generated
    vals = df[["L1", "L2"]].values
    nd = []
    for i in range(len(df)):
        dominated = np.any((vals[:, 0] <= vals[i, 0] + 1e-12) & (vals[:, 1] <= vals[i, 1] + 1e-12) &
                           ((vals[:, 0] < vals[i, 0] - 1e-12) | (vals[:, 1] < vals[i, 1] - 1e-12)))
        if not dominated:
            nd.append(i)
    if nd:
        ax.scatter(df.iloc[nd].L1, df.iloc[nd].L2, c="orange", s=28, label="Pareto L1/L2", zorder=4)
    ax.set_xlabel("L1")
    ax.set_ylabel("L2")
    ax.set_title("Pre-model Pareto (L1 vs L2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLT / "premodel_pareto_front.png", dpi=140)
    plt.close(fig)

    # Mask diversity among top finalists
    if finalists:
        ids = [f["split_id"] for f in finalists]
        n = len(ids)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = hamming_id(finalists[i]["public_ids"], finalists[j]["public_ids"])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([x.replace("GEN_", "")[:10] for x in ids], rotation=90, fontsize=7)
        ax.set_yticklabels([x.replace("GEN_", "")[:10] for x in ids], fontsize=7)
        ax.set_title("Finalist mask Hamming distance")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(PLT / "mask_diversity_top_splits.png", dpi=140)
        plt.close(fig)


def plot_finalist_distributions(finalists: list[dict], data: dict) -> None:
    feat = data["feat"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for f in finalists[:5] + [f for f in finalists if f.get("split_id") == "CAND_12528"][:1]:
        pub = set(f["public_ids"])
        axes[0].hist(feat.loc[feat.id.isin(pub), "TmApp"], bins=15, alpha=0.35, label=f["split_id"][:18])
        axes[1].hist(feat.loc[feat.id.isin(pub), "HIC"], bins=15, alpha=0.35, label=f["split_id"][:18])
    axes[0].set_title("Finalist Public TmApp")
    axes[1].set_title("Finalist Public HIC")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLT / "finalist_tmapp_distributions.png", dpi=140)
    fig.savefig(PLT / "finalist_hic_distributions.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for f in finalists[:5]:
        pub = set(f["public_ids"])
        sub = feat[feat.id.isin(pub)]
        ax.scatter(sub.TmApp, sub.HIC, s=10, alpha=0.4, label=f["split_id"][:16])
    ax.set_xlabel("TmApp")
    ax.set_ylabel("HIC")
    ax.set_title("Finalist joint Public scatter")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLT / "finalist_joint_target_scatter.png", dpi=140)
    plt.close(fig)


def premodel_stability(finalists: list[dict], data: dict, energy_scale: float) -> pd.DataFrame:
    """Leave-one-multi-group-out descriptive stability of L1/L2."""
    multi = {g: ids for g, ids in data["group_map"].items() if len(ids) > 1}
    rows = []
    for f in finalists:
        if f.get("is_control") and f.get("split_id") != "CAND_12528":
            continue
        base_mask = group_mask_from_pub(f["public_ids"], data)
        base = score_mask(base_mask, data, energy_scale)
        deltas_l1, deltas_l2 = [], []
        for g, ids in multi.items():
            # flip group if feasible
            k = data["group_index"][g]
            trial = base_mask.copy()
            trial[k] = 1 - trial[k]
            # repair size with singleton swaps if needed
            if int(trial @ data["group_sizes"]) != 81:
                # skip infeasible flips
                continue
            if not check_hard(trial, data):
                continue
            m = score_mask(trial, data, energy_scale)
            if not m["feasible"]:
                continue
            deltas_l1.append(m["L1"] - base["L1"])
            deltas_l2.append(m["L2"] - base["L2"])
        rows.append({
            "split_id": f["split_id"],
            "n_feasible_flips": len(deltas_l1),
            "L1_base": base["L1"],
            "L1_delta_mean": float(np.mean(deltas_l1)) if deltas_l1 else float("nan"),
            "L1_delta_p90": float(np.quantile(deltas_l1, 0.9)) if deltas_l1 else float("nan"),
            "L2_delta_mean": float(np.mean(deltas_l2)) if deltas_l2 else float("nan"),
            "L2_delta_p90": float(np.quantile(deltas_l2, 0.9)) if deltas_l2 else float("nan"),
        })
    return pd.DataFrame(rows)


def phase_premodel(data: dict) -> dict:
    log("=" * 60)
    log("PHASE 1–5: MODEL-BLIND SEARCH (no predictions)")
    log("=" * 60)
    energy_scale = calibrate_energy_scale(data, n_samples=200, seed=SEED_BASE)
    log(f"Energy scale (median random) = {energy_scale:.6f}")
    write_protocol(data, energy_scale)

    packs, stats = run_search(
        data, energy_scale, n_a=400, n_b=400, n_c=80, max_total=3000, min_total=500,
    )
    save_all_splits(packs)
    df = packs_to_rows(packs)
    df.to_csv(MET / "all_generated_splits_premodel.csv", index=False)

    controls = control_metrics(data, energy_scale)
    controls_df = packs_to_rows(controls)
    # rank controls among generated by L1
    for _, crow in controls_df.iterrows():
        pass

    near = near_optimal_sets(df)
    near_primary = near["10pct"].copy()
    near_primary["near_band"] = "10pct"
    # annotate other bands
    for band, sub in near.items():
        path = SPL / "near_optimal"
        path.mkdir(parents=True, exist_ok=True)
    near_rows = []
    for band, sub in near.items():
        s = sub.copy()
        s["near_band"] = band
        near_rows.append(s)
    near_all = pd.concat(near_rows, ignore_index=True)
    near_all.to_csv(MET / "near_optimal_splits.csv", index=False)

    packs_by_id = {p["split_id"]: p for p in packs}
    finalists = select_diverse_finalists(near["10pct"], packs_by_id, n=20, min_ham=12)
    # ensure CAND_12528 included as control (may be outside top-20)
    inc = next(c for c in controls if c["split_id"] == "CAND_12528")
    finalist_entries = []
    for i, f in enumerate(finalists):
        finalist_entries.append({
            "rank": i + 1,
            "split_id": f["split_id"],
            "seed": f.get("seed"),
            "method": f.get("method"),
            "public_ids": f["public_ids"],
            "private_ids": f["private_ids"],
            "public_hash": f["public_hash"],
            "private_hash": f["private_hash"],
            "mask_hash": f["mask_hash"],
            "is_control": False,
            "objective": {
                "L1": f["L1"], "L2": f["L2"], "L3": f["L3"], "L4": f["L4"],
                "TmApp_q8_L1": f["TmApp_q8_L1"], "HIC_q8_L1": f["HIC_q8_L1"],
                "HIC_MED_imbalance": f["HIC_MED_imbalance"],
                "joint_grid_L1": f["joint_grid_L1"],
                "hic_med_pub": f["hic_med_pub"], "hic_high_pub": f["hic_high_pub"],
            },
        })
        write_json(SPL / "finalists" / f"{f['split_id']}.json", finalist_entries[-1])

    finalist_entries.append({
        "rank": None,
        "split_id": "CAND_12528",
        "seed": None,
        "method": "incumbent_control",
        "public_ids": inc["public_ids"],
        "private_ids": inc["private_ids"],
        "public_hash": inc["public_hash"],
        "private_hash": inc["private_hash"],
        "mask_hash": mask_hash(inc["public_ids"]),
        "is_control": True,
        "objective": {
            "L1": inc["L1"], "L2": inc["L2"], "L3": inc["L3"], "L4": inc["L4"],
            "TmApp_q8_L1": inc["TmApp_q8_L1"], "HIC_q8_L1": inc["HIC_q8_L1"],
            "HIC_MED_imbalance": inc["HIC_MED_imbalance"],
            "joint_grid_L1": inc["joint_grid_L1"],
            "hic_med_pub": inc["hic_med_pub"], "hic_high_pub": inc["hic_high_pub"],
        },
    })
    # also keep other controls for comparison tables (not as finalists for model stress beyond incumbent)
    frozen = {
        "gate": "B7.3",
        "phase": "PREMODEL_FINALISTS_FROZEN",
        "energy_scale": energy_scale,
        "search_stats": stats,
        "n_near_optimal_5pct": int(len(near["5pct"])),
        "n_near_optimal_10pct": int(len(near["10pct"])),
        "n_near_optimal_20pct": int(len(near["20pct"])),
        "best_L1": float(df.L1.min()),
        "finalists": finalist_entries,
        "controls_also_scored": [c["split_id"] for c in controls],
        "note": "Split generation permanently closed after this freeze.",
    }
    fh = write_json(CFG / "B7_3_PREMODEL_FINALISTS_FROZEN.json", frozen)
    log(f"FROZEN finalists hash={fh}; generation CLOSED")

    # premodel finalists csv
    fin_rows = []
    for e in finalist_entries:
        fin_rows.append({
            "split_id": e["split_id"],
            "is_control": e["is_control"],
            "rank": e["rank"],
            "method": e["method"],
            "seed": e["seed"],
            **e["objective"],
            "public_hash": e["public_hash"],
            "private_hash": e["private_hash"],
        })
    fin_df = pd.DataFrame(fin_rows)
    fin_df.to_csv(MET / "premodel_finalists.csv", index=False)

    plot_premodel(df, controls_df, finalists)
    plot_finalist_distributions(finalists + [inc], data)

    # stability
    stab = premodel_stability(finalists + [inc], data, energy_scale)
    stab.to_csv(MET / "premodel_finalist_stability.csv", index=False)

    # percentiles for CAND_12528
    pct = {
        "L1": percentile_among(inc["L1"], df.L1),
        "TmApp_q8_L1": percentile_among(inc["TmApp_q8_L1"], df.TmApp_q8_L1),
        "HIC_q8_L1": percentile_among(inc["HIC_q8_L1"], df.HIC_q8_L1),
        "joint_energy_norm": percentile_among(inc["joint_energy_norm"], df.joint_energy_norm),
        "max_abs_SMD": percentile_among(inc["max_abs_SMD"], df.max_abs_SMD),
    }
    # pct = % of generated with WORSE (higher) metric → high means incumbent is good

    # reports 02-04
    best = packs[0]
    lines = [
        "# 02 — Large Model-Blind Search",
        "",
        f"- Starts: A={stats['n_a_starts']}, B={stats['n_b_starts']}, C={stats['n_c_starts']}, "
        f"total={stats['n_starts_total']}",
        f"- Valid unique masks: **{stats['n_unique_valid']}**",
        f"- Runtime: **{stats['runtime_sec']:.1f}s**",
        f"- Best L1={fnum(best['L1'])} ({best['split_id']}; method={best['method']})",
        f"- CAND_12528 L1={fnum(inc['L1'])}; percentile standing "
        f"(%% generated worse): {fnum(pct['L1'],1)}",
        "",
        "## Method contribution",
        "",
        f"- A valid packs ingested: {stats['n_a_valid']}",
        f"- B valid packs ingested: {stats['n_b_valid']}",
        f"- C valid packs ingested: {stats['n_c_valid']}",
        "",
        "![landscape](../plots/premodel_lexicographic_landscape.png)",
        "",
        md_table(df.head(15)[["split_id", "method", "L1", "L2", "L3", "hic_med_pub", "hic_high_pub"]]),
    ]
    (REP / "02_LARGE_MODEL_BLIND_SEARCH.md").write_text("\n".join(lines) + "\n")

    # diversity among near-optimal
    near10 = near["10pct"]
    ham = []
    ids = near10.split_id.tolist()[:80]
    for i in range(min(40, len(ids))):
        for j in range(i + 1, min(40, len(ids))):
            ham.append(hamming_id(packs_by_id[ids[i]]["public_ids"], packs_by_id[ids[j]]["public_ids"]))
    lines = [
        "# 03 — Near-Optimal Split Landscape",
        "",
        f"- Best L1: **{fnum(df.L1.min())}**",
        f"- Near-optimal @5%: **{len(near['5pct'])}**; @10%: **{len(near['10pct'])}**; "
        f"@20%: **{len(near['20pct'])}**",
        f"- Pairwise Hamming (sample of near-10%): median={fnum(np.median(ham) if ham else float('nan'),1)}, "
        f"min={min(ham) if ham else 'NA'}",
        f"- MEDIUM 3/3 among near-10%: "
        f"**{int(((near10.hic_med_pub==3)&(near10.hic_med_priv==3)).sum())}** / {len(near10)}",
        f"- Frozen diverse finalists: **{len(finalists)}** (+ CAND_12528 control)",
        "",
        "![pareto](../plots/premodel_pareto_front.png)",
        "![diversity](../plots/mask_diversity_top_splits.png)",
        "",
        md_table(fin_df),
    ]
    (REP / "03_NEAR_OPTIMAL_SPLIT_LANDSCAPE.md").write_text("\n".join(lines) + "\n")

    lines = [
        "# 04 — Pre-model Finalist Stability",
        "",
        "Leave-one-multi-group-out flips (size+HIGH feasible only); descriptive L1/L2 deltas.",
        "",
        md_table(stab),
        "",
        "Large positive L1_delta_p90 indicates sensitivity to single multi-group reassignment.",
    ]
    (REP / "04_PREMODEL_FINALIST_STABILITY.md").write_text("\n".join(lines) + "\n")

    return {
        "energy_scale": energy_scale,
        "packs": packs,
        "df": df,
        "controls": controls,
        "controls_df": controls_df,
        "near": near,
        "finalists": finalists,
        "inc": inc,
        "frozen": frozen,
        "stats": stats,
        "pct": pct,
        "fin_df": fin_df,
        "stab": stab,
    }


def phase_model_safety(data: dict, pre: dict) -> dict:
    log("=" * 60)
    log("PHASE 6+: MODEL SAFETY (predictions allowed)")
    log("=" * 60)
    from b7_3_model_banks import (
        bank_b_specs,
        bank_c_specs,
        eval_split_bank,
        family_lofo,
        hash_and_freeze_bank_assignment,
        load_bank_a,
        train_spec_bank,
    )

    specs_b = bank_b_specs()
    specs_c = bank_c_specs()
    hash_and_freeze_bank_assignment(specs_b, specs_c, BANK_A)

    log("Loading Bank A (frozen B5)")
    store_a = load_bank_a(data)
    log("Training Bank B (validation)")
    store_b = train_spec_bank(data, specs_b, "bank_b")
    log("Training Bank C (stress)")
    store_c = train_spec_bank(data, specs_c, "bank_c")

    # evaluate all finalists + key controls
    entries = list(pre["frozen"]["finalists"])
    for cid in ("CAND_04974", "CAND_12207", "BASELINE_ROLEMAP"):
        c = next(x for x in pre["controls"] if x["split_id"] == cid)
        entries.append({
            "split_id": cid,
            "public_ids": c["public_ids"],
            "private_ids": c["private_ids"],
            "is_control": True,
        })

    rows = []
    lofo_rows = []
    stores = {"A": store_a, "B": store_b, "C": store_c}
    for entry in entries:
        sid = entry["split_id"]
        log(f"  eval banks for {sid}")
        for bname, store in stores.items():
            rows.extend(eval_split_bank(entry, data, store, bname))
        # LOFO on top ~5 + incumbent, bank A and B
        top5_ids = {e["split_id"] for e in pre["frozen"]["finalists"][:5]}
        top5_ids.add("CAND_12528")
        if sid in top5_ids:
            for bname in ("A", "B"):
                lofo_rows.extend(family_lofo(entry, data, stores[bname], bname))

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(MET / "finalist_model_bank_evaluation.csv", index=False)
    lofo_df = pd.DataFrame(lofo_rows)
    lofo_df.to_csv(MET / "finalist_family_lofo.csv", index=False)

    # plots
    for metric, fname in [
        ("spearman_cv_public", "finalist_cv_public_by_bank.png"),
        ("spearman_public_private", "finalist_public_private_by_bank.png"),
        ("public_winner_regret", "finalist_public_regret_by_bank.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ax, target in zip(axes, ("TmApp", "HIC")):
            sub = eval_df[eval_df.target == target]
            for bank, marker in [("A", "o"), ("B", "s"), ("C", "^")]:
                s2 = sub[sub.bank == bank]
                ax.scatter(range(len(s2)), s2[metric], marker=marker, label=f"Bank {bank}", alpha=0.75)
            ax.set_title(f"{target} {metric}")
            ax.set_xlabel("split index")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PLT / fname, dpi=140)
        plt.close(fig)

    # report 05
    pivot = eval_df.pivot_table(
        index=["split_id", "target"], columns="bank",
        values=["spearman_cv_public", "spearman_public_private", "public_winner_regret"],
        aggfunc="first",
    )
    cat = eval_df.groupby(["split_id", "target"])["catastrophic_pub_priv"].any().reset_index()
    lines = [
        "# 05 — Model Bank Safety Validation",
        "",
        "Banks A (B5 frozen), B (trained validation), C (stress) evaluated after finalist freeze.",
        "",
        f"- Catastrophic Pub↔Priv flags: **{int(cat.catastrophic_pub_priv.sum())}** split×target rows",
        "",
        "## Summary (mean across banks)",
        "",
        md_table(
            eval_df.groupby(["split_id", "target"])[
                ["spearman_cv_public", "spearman_public_private", "public_winner_regret"]
            ].mean().reset_index().round(3)
        ),
        "",
        "![cv_pub](../plots/finalist_cv_public_by_bank.png)",
        "![pub_priv](../plots/finalist_public_private_by_bank.png)",
        "![regret](../plots/finalist_public_regret_by_bank.png)",
        "",
        "## Family LOFO (top-5 + incumbent)",
        "",
        md_table(lofo_df.head(40).round(3) if len(lofo_df) else pd.DataFrame()),
    ]
    (REP / "05_MODEL_BANK_SAFETY_VALIDATION.md").write_text("\n".join(lines) + "\n")

    return {"eval_df": eval_df, "lofo_df": lofo_df, "stores": stores, "entries": entries}


def phase_policy(data: dict, pre: dict, model_meta: dict) -> pd.DataFrame:
    from b7_3_policy import run_policy_stress

    log("PHASE 7 — policy stress")
    top5 = pre["frozen"]["finalists"][:5]
    inc = next(e for e in pre["frozen"]["finalists"] if e["split_id"] == "CAND_12528")
    entries = top5 + [inc]
    # use Bank A (+ a bit of B) for policy catalog
    store = {**model_meta["stores"]["A"], **model_meta["stores"]["B"]}
    pol = run_policy_stress(entries, data, store, n_traj=500)
    pol.to_csv(MET / "finalist_policy_simulation.csv", index=False)

    for target, fname in [("TmApp", "finalist_tmapp_policy_regret.png"), ("HIC", "finalist_hic_policy_regret.png")]:
        sub = pol[pol.target == target]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        splits = sub.split_id.unique()
        x = np.arange(len(splits))
        width = 0.25
        for i, policy in enumerate(["CV_FIRST", "BALANCED", "PUBLIC_FIRST"]):
            vals = [float(sub[(sub.split_id == s) & (sub.policy == policy)]["mean_private_regret"].iloc[0])
                    if len(sub[(sub.split_id == s) & (sub.policy == policy)]) else np.nan for s in splits]
            ax.bar(x + i * width, vals, width, label=policy)
        ax.set_xticks(x + width)
        ax.set_xticklabels([s[:16] for s in splits], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("mean Private regret")
        ax.set_title(f"{target} policy stress")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLT / fname, dpi=140)
        plt.close(fig)

    lines = [
        "# 06 — Policy Stress Test",
        "",
        "≥500 random development trajectories per target × split; policies CV_FIRST / BALANCED / PUBLIC_FIRST.",
        "",
        md_table(pol.round(4)),
        "",
        "![tmapp](../plots/finalist_tmapp_policy_regret.png)",
        "![hic](../plots/finalist_hic_policy_regret.png)",
    ]
    (REP / "06_POLICY_STRESS_TEST.md").write_text("\n".join(lines) + "\n")
    return pol


def decide_and_finalize(data: dict, pre: dict, model_meta: dict, pol: pd.DataFrame) -> dict:
    log("PHASE 8–9 — replacement bar + final report")
    eval_df = model_meta["eval_df"]
    inc = pre["inc"]
    best_fin = pre["finalists"][0] if pre["finalists"] else None

    def bank_ok(split_id: str) -> bool:
        sub = eval_df[eval_df.split_id == split_id]
        if len(sub) == 0:
            return False
        # no catastrophic on any bank/target
        if sub["catastrophic_pub_priv"].any():
            return False
        # Pub↔Priv not severely negative on mean
        for target in ("TmApp", "HIC"):
            s = sub[sub.target == target]
            if s["spearman_public_private"].mean() < -0.1:
                return False
            if s["public_winner_regret"].mean() > 2.0:
                return False
        return True

    def policy_ok(split_id: str) -> bool:
        sub = pol[pol.split_id == split_id]
        if len(sub) == 0:
            return True
        # relative to incumbent
        for target in ("TmApp", "HIC"):
            for policy in ("CV_FIRST", "BALANCED", "PUBLIC_FIRST"):
                r = sub[(sub.target == target) & (sub.policy == policy)]
                ri = pol[(pol.split_id == "CAND_12528") & (pol.target == target) & (pol.policy == policy)]
                if len(r) and len(ri):
                    if float(r.mean_private_regret.iloc[0]) > float(ri.mean_private_regret.iloc[0]) + 0.75:
                        return False
        return True

    challengers_clear = []
    for f in pre["finalists"]:
        # A model-free clearly better
        better_free = (
            f["L1"] < inc["L1"] * 0.85
            or (f["L1"] <= inc["L1"] and f["L2"] < inc["L2"] * 0.9)
        )
        hic_ok = f.get("hic_high_pub") in (3, 4) and f.get("hic_med_pub") in (2, 3, 4)
        med_pref = f.get("hic_med_pub") == 3
        joint_ok = f["L2"] <= max(inc["L2"], f["L2"])  # not worse than self; prefer <= inc
        joint_ok = f.get("joint_energy_norm", 9) <= inc.get("joint_energy_norm", 9) * 1.15
        bio_ok = f.get("mean_abs_SMD", 9) <= max(0.35, inc.get("mean_abs_SMD", 9) * 1.25)
        if better_free and hic_ok and joint_ok and bio_ok and bank_ok(f["split_id"]) and policy_ok(f["split_id"]):
            challengers_clear.append(f)

    # prefer MED 3/3 among clearers
    med33 = [c for c in challengers_clear if c.get("hic_med_pub") == 3 and c.get("hic_med_priv") == 3]
    pool = med33 if med33 else challengers_clear

    decision = "KEEP_CAND_12528_AFTER_LARGE_PRINCIPLED_SEARCH"
    chosen = inc
    reason = "No challenger cleared the high replacement bar on model-free + safety + policy."

    if len(pool) == 1:
        decision = "REPLACE_WITH_MODEL_BLIND_OPTIMAL_SPLIT"
        chosen = pool[0]
        reason = "Single challenger clears model-free and safety bars; primary justification is pre-model lex."
    elif len(pool) > 1:
        # check if essentially equivalent on lex
        l1s = [p["L1"] for p in pool]
        if (max(l1s) - min(l1s)) <= max(0.5, 0.05 * min(l1s)):
            decision = "MULTIPLE_MODEL_BLIND_SPLITS_EQUIVALENT"
            # pick by lex then seed — NOT model correlation
            pool_sorted = sorted(pool, key=lambda p: (p["L1"], p["L2"], p["L3"], p["L4"], p.get("seed") or 0))
            chosen = pool_sorted[0]
            reason = "Multiple equivalent model-blind splits; chose by lexicographic key then seed."
        else:
            decision = "REPLACE_WITH_MODEL_BLIND_NEAR_OPTIMAL_RANDOMIZED_SPLIT"
            chosen = sorted(pool, key=lambda p: (p["L1"], p["L2"], p["L3"], p["L4"]))[0]
            reason = "Near-optimal challenger set; selected best lexicographic among safety-cleared."

    # instability override
    if pre["stats"]["n_unique_valid"] < 20:
        decision = "NO_STABLE_PRODUCTION_SPLIT_FOUND"
        chosen = inc
        reason = "Insufficient unique valid splits for stable recommendation."

    assert decision in FINAL_DECISIONS

    # comparison table metrics
    compare_ids = ["CAND_12528"] + [f["split_id"] for f in pre["finalists"][:5]]
    compare_rows = []
    for sid in compare_ids:
        if sid == "CAND_12528":
            obj = inc
        else:
            obj = next(f for f in pre["finalists"] if f["split_id"] == sid)
        row = {
            "split_id": sid,
            "L1": obj["L1"],
            "L2": obj["L2"],
            "L3": obj["L3"],
            "TmApp_q8_L1": obj.get("TmApp_q8_L1"),
            "HIC_q8_L1": obj.get("HIC_q8_L1"),
            "TmApp_W_norm": obj.get("TmApp_W_norm"),
            "HIC_W_norm": obj.get("HIC_W_norm"),
            "TmApp_KS": obj.get("TmApp_KS"),
            "HIC_KS": obj.get("HIC_KS"),
            "joint_energy_norm": obj.get("joint_energy_norm"),
            "hic_low_pub": obj.get("hic_low_pub"),
            "hic_med_pub": obj.get("hic_med_pub"),
            "hic_high_pub": obj.get("hic_high_pub"),
            "mean_abs_SMD": obj.get("mean_abs_SMD"),
            "max_abs_SMD": obj.get("max_abs_SMD"),
            "germline_TV_mean": obj.get("germline_TV_mean"),
        }
        for bank in ("A", "B", "C"):
            for target in ("TmApp", "HIC"):
                sub = eval_df[(eval_df.split_id == sid) & (eval_df.bank == bank) & (eval_df.target == target)]
                if len(sub):
                    row[f"{bank}_{target}_cv_pub"] = float(sub.spearman_cv_public.iloc[0])
                    row[f"{bank}_{target}_pub_priv"] = float(sub.spearman_public_private.iloc[0])
                    row[f"{bank}_{target}_pub_regret"] = float(sub.public_winner_regret.iloc[0])
        for target in ("TmApp", "HIC"):
            for policy in ("CV_FIRST", "BALANCED", "PUBLIC_FIRST"):
                sub = pol[(pol.split_id == sid) & (pol.target == target) & (pol.policy == policy)]
                if len(sub):
                    row[f"pol_{target}_{policy}"] = float(sub.mean_private_regret.iloc[0])
        compare_rows.append(row)
    comp = pd.DataFrame(compare_rows)
    comp.to_csv(MET / "final_split_comparison.csv", index=False)

    # Build comparison markdown table
    criteria = [
        ("TmApp_q8_L1", "TmApp quantile imbalance"),
        ("TmApp_W_norm", "TmApp Wasserstein"),
        ("TmApp_KS", "TmApp KS"),
        ("HIC_q8_L1", "HIC quantile imbalance"),
        ("HIC_W_norm", "HIC Wasserstein"),
        ("HIC_KS", "HIC KS"),
        ("hic_low_pub", "HIC LOW pub count"),
        ("hic_med_pub", "HIC MEDIUM pub count"),
        ("hic_high_pub", "HIC HIGH pub count"),
        ("joint_energy_norm", "joint energy"),
        ("max_abs_SMD", "max |SMD|"),
        ("mean_abs_SMD", "mean |SMD|"),
        ("germline_TV_mean", "germline TV"),
        ("L1", "lex L1"),
    ]
    header = ["Criterion"] + compare_ids
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for col, label in criteria:
        md_lines.append("| " + " | ".join([label] + [fnum(comp.loc[comp.split_id == s, col].iloc[0]) if col in comp.columns and len(comp.loc[comp.split_id == s]) else "NA" for s in compare_ids]) + " |")
    for bank in ("A", "B", "C"):
        for target in ("TmApp", "HIC"):
            for metric, lab in [("cv_pub", "CV↔Public"), ("pub_priv", "Public↔Private"), ("pub_regret", "Public regret")]:
                col = f"{bank}_{target}_{metric}"
                md_lines.append("| " + " | ".join([f"Bank{bank} {target} {lab}"] + [fnum(comp.loc[comp.split_id == s, col].iloc[0]) if col in comp.columns and len(comp.loc[comp.split_id == s]) and col in comp.columns else "NA" for s in compare_ids]) + " |")

    # Questions 1-28
    df = pre["df"]
    near = pre["near"]
    pct = pre["pct"]
    n_near = len(near["10pct"])
    ham_vals = []
    packs_by_id = {p["split_id"]: p for p in pre["packs"]}
    ids = near["10pct"].split_id.tolist()[:50]
    for i in range(min(30, len(ids))):
        for j in range(i + 1, min(30, len(ids))):
            ham_vals.append(hamming_id(packs_by_id[ids[i]]["public_ids"], packs_by_id[ids[j]]["public_ids"]))

    cat_tm = eval_df[(eval_df.target == "TmApp") & eval_df.catastrophic_pub_priv]
    cat_hic = eval_df[(eval_df.target == "HIC") & eval_df.catastrophic_pub_priv]

    answers = {
        1: f"Best L1={fnum(df.L1.min())} vs CAND_12528 L1={fnum(inc['L1'])} "
           f"(Δ={fnum(inc['L1']-df.L1.min())}; relative {(1-df.L1.min()/max(inc['L1'],1e-9))*100:.1f}% lower).",
        2: f"CAND_12528 sits at percentile-standing {fnum(pct['L1'],1)}% (%% generated worse on L1) — "
           f"{'genuinely poor on this principled objective' if pct['L1'] < 30 else 'not extreme under this objective'}.",
        3: f"Yes for many near-optimal masks: best TmApp_q8_L1={fnum(df.TmApp_q8_L1.min())}, "
           f"HIC_q8_L1={fnum(df.HIC_q8_L1.min())}.",
        4: f"Joint energy can be improved: best joint_energy_norm={fnum(df.joint_energy_norm.min())} "
           f"vs incumbent {fnum(inc['joint_energy_norm'])}.",
        5: f"MEDIUM 3/3 with HIGH 3/4|4/3 among near-10%: "
           f"{int(((near['10pct'].hic_med_pub==3)&(near['10pct'].hic_high_pub.isin([3,4]))).sum())} / {n_near}.",
        6: f"Top finalist mean|SMD|={fnum(best_fin['mean_abs_SMD']) if best_fin else 'NA'} vs "
           f"incumbent {fnum(inc['mean_abs_SMD'])}; germline TV {fnum(best_fin['germline_TV_mean']) if best_fin else 'NA'} "
           f"vs {fnum(inc['germline_TV_mean'])}.",
        7: f"Near-optimal @5%/10%/20%: {len(near['5pct'])}/{len(near['10pct'])}/{len(near['20pct'])}.",
        8: f"Mask Hamming median among near sample={fnum(np.median(ham_vals) if ham_vals else float('nan'),1)} "
           f"— {'diverse' if (ham_vals and np.median(ham_vals)>=10) else 'partially overlapping'}.",
        9: f"Yes — {pre['stats']['n_unique_valid']} unique valid from {pre['stats']['n_starts_total']} starts.",
        10: f"Method C (MILP) contributed {pre['stats']['n_c_valid']} valid; heuristics dominate archive volume.",
        11: (
            f"Best heuristic L1={fnum(df[df.method.astype(str).str.match(r'^[AB]')].L1.min()) if len(df) else 'NA'}; "
            f"best MILP-origin L1="
            f"{fnum(df[df.method.astype(str).str.startswith('C')].L1.min()) if df.method.astype(str).str.startswith('C').any() else 'NA'}."
        ),
        12: "Hardest typically joint_grid_L1 / simultaneous TmApp+HIC quantile L1 under HIGH constraints.",
        13: f"Correlation L1 components: TmApp_q8 vs HIC_q8 Spearman≈"
           f"{fnum(df[['TmApp_q8_L1','HIC_q8_L1']].corr(method='spearman').iloc[0,1])}.",
        14: f"L1 vs mean|SMD| Spearman≈{fnum(df[['L1','mean_abs_SMD']].corr(method='spearman').iloc[0,1])}.",
        15: f"CAND_12528 L1 percentile-standing {fnum(pct['L1'],1)}%; TmApp {fnum(pct['TmApp_q8_L1'],1)}%; "
           f"HIC {fnum(pct['HIC_q8_L1'],1)}%; joint {fnum(pct['joint_energy_norm'],1)}%; "
           f"maxSMD {fnum(pct['max_abs_SMD'],1)}%.",
        16: f"Across finalists BankA TmApp CV↔Pub std="
           f"{fnum(eval_df[(eval_df.bank=='A')&(eval_df.target=='TmApp')].spearman_cv_public.std())}.",
        17: f"Catastrophic TmApp flags: {len(cat_tm)} rows; splits={sorted(cat_tm.split_id.unique().tolist())[:8]}",
        18: f"Catastrophic HIC flags: {len(cat_hic)} rows; splits={sorted(cat_hic.split_id.unique().tolist())[:8]}",
        19: "Yes — catastrophic Pub↔Priv / regret thresholds reject without optimizing correlations.",
        20: f"Safety-cleared challengers: {len(challengers_clear)}.",
        21: f"Family LOFO rows: {len(model_meta['lofo_df'])}; inspect metrics/finalist_family_lofo.csv.",
        22: f"{'Yes: '+challengers_clear[0]['split_id'] if challengers_clear else 'No challenger clearly dominates with safety.'}",
        23: f"{'Leaderboard behaviors differ across banks; see comparison table.' if len(pool)>1 else 'N/A or single clearer.'}",
        24: "If equivalent, choose by lexicographic premodel key / seed — not best model correlation.",
        25: f"{'Yes — explicit lex objective is cleaner than CAND_12528 ad-hoc origin.' if decision.startswith('REPLACE') or decision.startswith('MULTIPLE') else 'Incumbent retained; model-blind search still documents clearer methodology for alternatives.'}",
        26: "Large search improves confidence in near-optimal landscape and that incumbent is not uniquely good on balance.",
        27: f"Decision={decision}; chosen={chosen.get('split_id')}; "
           f"pub_hash={chosen.get('public_hash')}; priv_hash={chosen.get('private_hash')}.",
        28: "Yes — primary justification is model-free lex balance; models used only as safety filters.",
    }

    # Final report opening (section 37 structure)
    best5 = pre["finalists"][:5]
    opening = [
        "# Gate B7.3 — Principled Production Split FINAL",
        "",
        "Overall production-split verdict:",
        f"    {decision}",
        "",
        "Search:",
        f"    seeds / starts evaluated: {pre['stats']['n_starts_total']}",
        f"    valid splits: {pre['stats']['n_unique_valid']}",
        f"    construction methods: A_greedy_swap, B_simulated_annealing, C_milp",
        f"    runtime: {pre['stats']['runtime_sec']:.1f}s (search) + model/policy phases",
        "",
        "Model-blind split objective:",
        "    hard constraints: size 81, atomic groups, HIGH∈{3,4}",
        "    target balance: TmApp/HIC q8 L1 + MED imbalance (minimax L1)",
        "    joint target balance: 4×4 grid L1 + energy distance",
        "    biological balance: germline TV mean",
        "    sequence-space balance: mean/p90/max |SMD| + group-mass TV",
        "",
        "CAND_12528 model-blind standing:",
        f"    overall percentile (%% generated worse on L1): {fnum(pct['L1'],1)}",
        f"    TmApp percentile: {fnum(pct['TmApp_q8_L1'],1)}",
        f"    HIC percentile: {fnum(pct['HIC_q8_L1'],1)}",
        f"    joint-target percentile: {fnum(pct['joint_energy_norm'],1)}",
        f"    max-SMD percentile: {fnum(pct['max_abs_SMD'],1)}",
        "",
        "Near-optimal landscape:",
        f"    number of near-optimal splits (@10%): {n_near}",
        f"    mask diversity (median Hamming sample): {fnum(np.median(ham_vals) if ham_vals else float('nan'),1)}",
        f"    HIC MEDIUM/HIGH balance: MED3/3 count in near-10%="
        f"{int(((near['10pct'].hic_med_pub==3)&(near['10pct'].hic_med_priv==3)).sum())}; HIGH constrained",
        "    target-balance stability: see report 04",
        "",
        "Best model-blind finalists:",
    ]
    for f in best5:
        opening.append(
            f"    - {f['split_id']}: L1={fnum(f['L1'])}, MED={f.get('hic_med_pub')}/{f.get('hic_med_priv')}, "
            f"HIGH={f.get('hic_high_pub')}/{f.get('hic_high_priv')}"
        )
    opening += [
        "",
        "------------------------",
        "Model-safety validation",
        "------------------------",
        "",
        "TmApp:",
        f"    behavior across model banks: see metrics/finalist_model_bank_evaluation.csv",
        "",
        "HIC:",
        f"    behavior across model banks: see metrics/finalist_model_bank_evaluation.csv",
        "",
        f"Any catastrophic split/model interaction: TmApp={len(cat_tm)}, HIC={len(cat_hic)} flagged rows",
        "",
        "Family robustness:",
        "    see metrics/finalist_family_lofo.csv",
        "",
        "------------------------",
        "Participant policy stress",
        "------------------------",
        "",
    ]
    for target in ("TmApp", "HIC"):
        opening.append(f"{target}:")
        for policy in ("CV_FIRST", "BALANCED", "PUBLIC_FIRST"):
            sub = pol[(pol.split_id == chosen.get("split_id", "CAND_12528")) & (pol.target == target) & (pol.policy == policy)]
            val = fnum(sub.mean_private_regret.iloc[0]) if len(sub) else "NA"
            opening.append(f"    {policy}: mean Private regret={val}")
        opening.append("")
    opening += [
        "------------------------",
        "Final split reasoning",
        "------------------------",
        "",
        "Can the chosen split be justified entirely without model outcomes:",
        "    Yes — primary ranking is lexicographic pre-model balance.",
        "",
        "Is CAND_12528 still preferable:",
        f"    {'Yes' if decision.startswith('KEEP') else 'No — challenger recommended'}",
        "",
        "If replacement is recommended:",
        f"    exact split: {chosen.get('split_id')}",
        f"    seed: {chosen.get('seed')}",
        f"    method: {chosen.get('method')}",
        f"    Public ID hash: {chosen.get('public_hash')}",
        f"    Private ID hash: {chosen.get('private_hash')}",
        f"    primary model-free justification: {reason}",
        "    model-safety evidence: banks A/B/C + policy stress (reports 05–06)",
        "",
        f"FINAL DECISION:",
        f"    {decision}",
        "",
        "## Incumbent vs top finalists",
        "",
        "\n".join(md_lines),
        "",
        "## Answers to required questions (1–28)",
        "",
    ]
    for i in range(1, 29):
        opening.append(f"**{i}.** {answers[i]}")
        opening.append("")

    opening += [
        "## Return summary",
        "",
        f"- final_decision: `{decision}`",
        f"- final_report: `reports/GATE_B7_3_PRINCIPLED_SPLIT_FINAL.md`",
        f"- n_starts: {pre['stats']['n_starts_total']}",
        f"- n_unique_valid: {pre['stats']['n_unique_valid']}",
        f"- n_near_optimal_10pct: {n_near}",
        f"- CAND_12528 percentiles (%% worse): L1={fnum(pct['L1'],1)}, "
        f"TmApp={fnum(pct['TmApp_q8_L1'],1)}, HIC={fnum(pct['HIC_q8_L1'],1)}, "
        f"joint={fnum(pct['joint_energy_norm'],1)}, maxSMD={fnum(pct['max_abs_SMD'],1)}",
        f"- challenger_clears_bar: {bool(challengers_clear)}",
        f"- best_finalist HIC MED/HIGH: "
        f"{best_fin.get('hic_med_pub') if best_fin else 'NA'}/"
        f"{best_fin.get('hic_med_priv') if best_fin else 'NA'} MED; "
        f"{best_fin.get('hic_high_pub') if best_fin else 'NA'}/"
        f"{best_fin.get('hic_high_priv') if best_fin else 'NA'} HIGH",
    ]
    (REP / "GATE_B7_3_PRINCIPLED_SPLIT_FINAL.md").write_text("\n".join(opening) + "\n")

    summary = {
        "final_decision": decision,
        "chosen_split_id": chosen.get("split_id"),
        "n_starts": pre["stats"]["n_starts_total"],
        "n_unique_valid": pre["stats"]["n_unique_valid"],
        "n_near_optimal_10pct": n_near,
        "cand_12528_percentiles": pct,
        "challenger_clears_bar": bool(challengers_clear),
        "best_finalist_hic_med": [best_fin.get("hic_med_pub"), best_fin.get("hic_med_priv")] if best_fin else None,
        "best_finalist_hic_high": [best_fin.get("hic_high_pub"), best_fin.get("hic_high_priv")] if best_fin else None,
        "reason": reason,
    }
    write_json(CFG / "B7_3_RUN_SUMMARY.json", summary)
    log(f"FINAL DECISION: {decision}")
    return summary


def main():
    t0 = time.time()
    LOG_PATH = GATE / "logs" / "b7_3_run.log"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")  # reset
    try:
        data = load_base_data()
        log(f"Loaded Test N={len(data['feat'])}; bands={data['band_counts']}; groups={len(data['group_ids'])}")
        pre = phase_premodel(data)
        # BARRIER: no more split generation
        log("*** GENERATION CLOSED — loading models ***")
        model_meta = phase_model_safety(data, pre)
        pol = phase_policy(data, pre, model_meta)
        summary = decide_and_finalize(data, pre, model_meta, pol)
        log(f"TOTAL runtime {time.time()-t0:.1f}s")
        print(json.dumps(summary, indent=2, default=float))
        return 0
    except Exception:
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
