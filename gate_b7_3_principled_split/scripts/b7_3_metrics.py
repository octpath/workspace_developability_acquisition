"""Gate B7.3 pre-model metrics and lexicographic objective."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from b7_3_common import (
    CAT_COLS,
    CONT_COLS,
    N_Q_PRIMARY,
    PUBLIC_N,
    check_hard,
    lex_key_from_metrics,
    pub_from_group_mask,
)


def _bin_l1(bins: np.ndarray, pub_mask_rows: np.ndarray, n_bins: int) -> float:
    """L1 count imbalance Public vs Private across fixed bins."""
    total = 0.0
    for b in range(n_bins):
        in_b = bins == b
        n_pub = int((in_b & pub_mask_rows).sum())
        n_priv = int((in_b & ~pub_mask_rows).sum())
        total += abs(n_pub - n_priv)
    return float(total)


def _tv(a_labels, b_labels) -> float:
    keys = sorted(set(a_labels) | set(b_labels))
    if not keys:
        return 0.0
    na, nb = len(a_labels), len(b_labels)
    pa = np.array([np.sum(a_labels == k) / max(na, 1) for k in keys], float)
    pb = np.array([np.sum(b_labels == k) / max(nb, 1) for k in keys], float)
    return 0.5 * float(np.abs(pa - pb).sum())


def _abs_smd(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = float(np.sqrt(0.5 * (np.var(a) + np.var(b))) + 1e-12)
    return abs(float(np.mean(a) - np.mean(b)) / pooled)


def _energy_distance(xa: np.ndarray, xb: np.ndarray) -> float:
    """Energy distance between two 2D samples."""
    xa = np.asarray(xa, float)
    xb = np.asarray(xb, float)
    na, nb = len(xa), len(xb)
    if na < 2 or nb < 2:
        return 0.0

    def mean_pw(u, v):
        # mean pairwise Euclidean
        d = np.sqrt(((u[:, None, :] - v[None, :, :]) ** 2).sum(-1))
        return float(d.mean())

    return 2.0 * mean_pw(xa, xb) - mean_pw(xa, xa) - mean_pw(xb, xb)


def continuous_summaries(a, b, full_sd, full_iqr) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    def iqr(x):
        return float(np.subtract(*np.percentile(x, [75, 25])))

    return {
        "mean_diff_norm": abs(a.mean() - b.mean()) / (full_sd + 1e-12),
        "sd_diff_norm": abs(a.std() - b.std()) / (full_sd + 1e-12),
        "median_diff_norm": abs(np.median(a) - np.median(b)) / (full_sd + 1e-12),
        "iqr_diff_norm": abs(iqr(a) - iqr(b)) / (full_iqr + 1e-12),
        "W_norm": float(wasserstein_distance(a, b)) / (full_sd + 1e-12),
        "KS": float(ks_2samp(a, b, method="asymp").statistic),
    }


def compute_premodel(
    pub_ids,
    priv_ids,
    data: dict,
    *,
    energy_scale: float | None = None,
    require_hard: bool = True,
) -> dict:
    feat = data["feat"]
    pub_ids = sorted(map(str, pub_ids))
    priv_ids = sorted(map(str, priv_ids))
    pub_set = set(pub_ids)
    rows_pub = feat.id.isin(pub_set).values

    out: dict = {
        "feasible": False,
        "n_public": len(pub_ids),
        "n_private": len(priv_ids),
    }

    # atomic groups
    for g, ids in data["group_map"].items():
        n_in = sum(1 for i in ids if i in pub_set)
        if 0 < n_in < len(ids):
            out.update(_inf_metrics())
            return out

    if len(pub_ids) != PUBLIC_N or len(priv_ids) != PUBLIC_N:
        out.update(_inf_metrics())
        return out
    if set(pub_ids) | set(priv_ids) != set(feat.id):
        out.update(_inf_metrics())
        return out

    sub_pub = feat[rows_pub]
    sub_priv = feat[~rows_pub]
    high_pub = int((sub_pub.hic_band == "HIGH").sum())
    high_priv = int((sub_priv.hic_band == "HIGH").sum())
    med_pub = int((sub_pub.hic_band == "MEDIUM").sum())
    med_priv = int((sub_priv.hic_band == "MEDIUM").sum())
    low_pub = int((sub_pub.hic_band == "LOW").sum())
    low_priv = int((sub_priv.hic_band == "LOW").sum())
    out.update(
        {
            "hic_high_pub": high_pub,
            "hic_high_priv": high_priv,
            "hic_med_pub": med_pub,
            "hic_med_priv": med_priv,
            "hic_low_pub": low_pub,
            "hic_low_priv": low_priv,
        }
    )
    if high_pub not in (3, 4):
        out.update(_inf_metrics())
        return out

    out["feasible"] = True

    # Quantile L1 primary + sensitivity
    for nq in (6, 8, 10):
        out[f"TmApp_q{nq}_L1"] = _bin_l1(data["qbins"][nq]["TmApp"], rows_pub, nq)
        out[f"HIC_q{nq}_L1"] = _bin_l1(data["qbins"][nq]["HIC"], rows_pub, nq)

    # MED imbalance: |pub - priv| (= 2*|pub-3| when total=6)
    out["HIC_MED_imbalance"] = float(abs(med_pub - med_priv))
    out["HIC_MED_pref_penalty"] = float(abs(med_pub - 3) + abs(med_priv - 3))

    # Joint 4x4 grid L1
    joint = data["tm_quart"] * 4 + data["hic_quart"]
    out["joint_grid_L1"] = _bin_l1(joint, rows_pub, 16)

    # Continuous target distances
    tm_s = continuous_summaries(
        sub_pub.TmApp.values, sub_priv.TmApp.values, data["tmapp_sd"], data["tmapp_iqr"]
    )
    hic_s = continuous_summaries(
        sub_pub.HIC.values, sub_priv.HIC.values, data["hic_sd"], data["hic_iqr"]
    )
    for k, v in tm_s.items():
        out[f"TmApp_{k}"] = v
    for k, v in hic_s.items():
        out[f"HIC_{k}"] = v

    # Joint energy
    z = data["z2"]
    e_raw = _energy_distance(z[rows_pub], z[~rows_pub])
    if energy_scale is None or energy_scale <= 0:
        # Test diameter in z-space
        # use max pairwise among a subsample for speed if needed
        diam = float(np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)).max())
        energy_scale = max(diam, 1e-12)
    out["joint_energy_raw"] = float(e_raw)
    out["joint_energy_norm"] = float(e_raw) / float(energy_scale)
    out["energy_scale_used"] = float(energy_scale)

    # Biology TV
    tvs = []
    for col in CAT_COLS:
        tv = _tv(sub_pub[col].astype(str).values, sub_priv[col].astype(str).values)
        out[f"TV_{col}"] = tv
        tvs.append(tv)
    out["germline_TV_mean"] = float(np.mean(tvs))

    # Group mass TV: distribution of group sizes on each side
    gsize_pub = sub_pub.groupby("sequence_group").size().values
    gsize_priv = sub_priv.groupby("sequence_group").size().values
    # TV over size categories
    def size_hist(sizes):
        keys = sorted(set(sizes.tolist()) | set(gsize_priv.tolist()) | set(gsize_pub.tolist()))
        h = np.array([(sizes == k).sum() for k in keys], float)
        return h / max(h.sum(), 1)

    hp, hr = size_hist(gsize_pub), size_hist(gsize_priv)
    out["group_mass_TV"] = 0.5 * float(np.abs(hp - hr).sum())

    # Continuous |SMD|
    smds = []
    for col in CONT_COLS:
        s = _abs_smd(sub_pub[col].values, sub_priv[col].values)
        out[f"SMD_{col}"] = s
        smds.append(s)
    smds = np.array(smds, float)
    out["mean_abs_SMD"] = float(smds.mean())
    out["p90_abs_SMD"] = float(np.quantile(smds, 0.90))
    out["max_abs_SMD"] = float(smds.max())

    # Lex levels
    out["L1"] = float(
        max(
            out["TmApp_q8_L1"],
            out["HIC_q8_L1"],
            out["HIC_MED_imbalance"],
            out["joint_grid_L1"],
        )
    )
    out["L2"] = float(
        max(out["TmApp_W_norm"], out["HIC_W_norm"], out["joint_energy_norm"])
    )
    out["L3"] = float(
        max(out["mean_abs_SMD"], out["germline_TV_mean"], out["group_mass_TV"])
    )
    # Aggregate L4: sum of normalized components
    out["L4"] = float(
        out["L1"] / 20.0
        + out["L2"]
        + out["L3"]
        + 0.1 * out["HIC_MED_pref_penalty"]
        + 0.05 * out["max_abs_SMD"]
    )
    out["lex_key"] = lex_key_from_metrics(out)
    return out


def _inf_metrics() -> dict:
    return {
        "feasible": False,
        "L1": 1e9,
        "L2": 1e9,
        "L3": 1e9,
        "L4": 1e9,
        "TmApp_q8_L1": 1e9,
        "HIC_q8_L1": 1e9,
        "HIC_MED_imbalance": 1e9,
        "joint_grid_L1": 1e9,
        "TmApp_W_norm": 1e9,
        "HIC_W_norm": 1e9,
        "joint_energy_norm": 1e9,
        "mean_abs_SMD": 1e9,
        "germline_TV_mean": 1e9,
        "group_mass_TV": 1e9,
        "lex_key": (1, 1e9, 1e9, 1e9, 1e9),
    }


def score_mask(mask: np.ndarray, data: dict, energy_scale: float | None = None) -> dict:
    if not check_hard(mask, data):
        m = _inf_metrics()
        m["feasible"] = False
        return m
    pub, priv = pub_from_group_mask(mask, data)
    return compute_premodel(pub, priv, data, energy_scale=energy_scale)


def calibrate_energy_scale(data: dict, n_samples: int = 200, seed: int = 0) -> float:
    """Median joint energy of random feasible splits (size+HIGH only)."""
    rng = np.random.default_rng(seed)
    vals = []
    attempts = 0
    while len(vals) < n_samples and attempts < n_samples * 200:
        attempts += 1
        mask = _random_feasible_mask(data, rng)
        if mask is None:
            continue
        pub, priv = pub_from_group_mask(mask, data)
        rows = data["feat"].id.isin(set(pub)).values
        e = _energy_distance(data["z2"][rows], data["z2"][~rows])
        vals.append(e)
    if not vals:
        z = data["z2"]
        return float(np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)).max())
    return float(np.median(vals))


def _random_feasible_mask(data: dict, rng: np.random.Generator) -> np.ndarray | None:
    """Random group assignment with exact size 81 and HIGH in {3,4}."""
    sizes = data["group_sizes"]
    high = data["group_high"]
    n = len(sizes)
    # try random shuffle of groups into knapsack
    for _ in range(40):
        order = rng.permutation(n)
        mask = np.zeros(n, dtype=int)
        rem = PUBLIC_N
        for i in order:
            if sizes[i] <= rem:
                # random accept with bias
                if rem == sizes[i] or rng.random() < 0.5 or rem - sizes[i] < 3:
                    if rem - sizes[i] == 0 or rem - sizes[i] >= 1:
                        mask[i] = 1
                        rem -= int(sizes[i])
                        if rem == 0:
                            break
        if rem != 0:
            continue
        hp = int(mask @ high)
        if hp in (3, 4):
            return mask
    return None
