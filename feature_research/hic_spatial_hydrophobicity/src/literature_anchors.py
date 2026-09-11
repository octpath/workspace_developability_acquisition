#!/usr/bin/env python3
"""Literature-anchor descriptors (exact formulas; not forced into generic factory)."""
from __future__ import annotations

import numpy as np

from scales import BM_SAP, property_minmax, property_raw
from spatial_engine import aggregate, pairwise_centroid, pairwise_closest_sc, scope_mask


def lit1_static_sap_kd_v1(ab: dict) -> dict[str, float]:
    """Exact H094 STATIC_SAP_KD_v1: centroid R=5, KD minmax, total_rASA_Tien, ALL_FV MAX/MEAN/SUM."""
    D = pairwise_centroid(ab["centroids"])
    hydro = np.asarray([property_minmax("KD")[a] for a in ab["aa"]], float)
    exp = ab["total_rASA_Tien"]
    contrib = hydro * exp
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)
    neigh = D <= 5.0
    scores = neigh.astype(float) @ contrib
    # invalid centroids -> nan scores
    bad = ~np.isfinite(ab["centroids"]).all(axis=1)
    scores[bad] = np.nan
    agg = aggregate(scores, np.ones(len(scores), bool))
    return {
        "LIT1_STATIC_SAP_KD_v1__ALL_FV__MAX": agg["MAX"],
        "LIT1_STATIC_SAP_KD_v1__ALL_FV__MEAN": agg["MEAN"],
        "LIT1_STATIC_SAP_KD_v1__ALL_FV__SUM": agg["SUM"],
    }


def lit2_static_canonical_sap_bm(ab: dict, R: float = 5.0) -> dict[str, float]:
    """Repo STATIC-SAP approx: CA/centroid-equivalent via CA fallback centroids,
    Black–Mould Gly-centered, total_rASA_Tien, R in {5,10}, positive + topk style.

    Note: original Chennamsetty SAP is atom-centered + MD; this is the repository
    static single-structure approximation (CA neighborhood in FEATURE_SPEC; we use
    centroids which equal CA for Gly and approximate for others — also provide
    CA-only by using centroids as stored which include sc centroids).

    For fidelity to FEATURE_SPEC formula (CA distance), recompute with CA≈centroid
    when centroid_prov is gly/ca, else use residue CA from centroid fallback path.
    Here we use the same CA-distance semantics as extract_physical_batch1 by using
    centroids only when they are CA fallback; for sidechain centroids we ALSO
    compute a CA-proxy: the geometry cache stores centroid; for LIT2 we use
    pairwise on a CA-like array = centroid for gly_ca/ca_fallback else we need CA.

    Practical fidelity: use total_rASA * BM_SAP with CENTROID distances at R —
    labeled STATIC_CANONICAL_SAP_APPROX_REPO (documented difference).
    """
    D = pairwise_centroid(ab["centroids"])
    hydro = np.asarray([BM_SAP[a] for a in ab["aa"]], float)
    exp = ab["total_rASA_Tien"]
    contrib = hydro * exp
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)
    neigh = (D <= R) & np.isfinite(D)
    scores = neigh.astype(float) @ contrib
    bad = ~np.isfinite(ab["centroids"]).all(axis=1)
    scores[bad] = np.nan
    pos = scores[np.isfinite(scores) & (scores > 0)]
    tag = f"R{str(R).replace('.', 'p')}"
    out = {
        f"LIT2_STATIC_SAP_BM_{tag}__max_positive": float(pos.max()) if len(pos) else 0.0,
        f"LIT2_STATIC_SAP_BM_{tag}__mean_positive": float(pos.mean()) if len(pos) else 0.0,
        f"LIT2_STATIC_SAP_BM_{tag}__sum_positive": float(pos.sum()) if len(pos) else 0.0,
    }
    order = np.sort(scores[np.isfinite(scores)])[::-1]
    out[f"LIT2_STATIC_SAP_BM_{tag}__top3_mean"] = float(order[:3].mean()) if len(order) else 0.0
    out[f"LIT2_STATIC_SAP_BM_{tag}__top5_mean"] = float(order[:5].mean()) if len(order) else 0.0
    cdr = scope_mask(ab, "ALL_CDR")
    cdr_pos = scores[cdr & np.isfinite(scores) & (scores > 0)]
    out[f"LIT2_STATIC_SAP_BM_{tag}__CDR_sum_positive"] = float(cdr_pos.sum()) if len(cdr_pos) else 0.0
    out[f"LIT2_STATIC_SAP_BM_{tag}__CDR_max_positive"] = float(cdr_pos.max()) if len(cdr_pos) else 0.0
    return out


def lit3_psh_kd(ab: dict, R: float = 7.5, expose_thr: float = 0.075) -> dict[str, float]:
    """TAP/PSH-like (Raybould 2019): exposed sidechain rASA>=7.5%, closest heavy <7.5Å,
    H normalized to [1,2] on KD, sum H_i H_j / r_ij^2 over pairs (i<j).

    Salt-bridge neutralization omitted (requires salt-bridge detection); documented.
    """
    kd = property_raw("KD")
    lo, hi = min(kd.values()), max(kd.values())
    # map to [1,2]
    H = np.asarray([1.0 + (kd[a] - lo) / (hi - lo) for a in ab["aa"]], float)
    exp = ab["sidechain_over_Tien"]  # approx sidechain relative ASA / Tien
    exposed = np.isfinite(exp) & (exp >= expose_thr)
    D = pairwise_closest_sc(ab["sc_heavy"])
    n = len(H)
    s_fv = 0.0
    s_cdr = 0.0
    cdr = scope_mask(ab, "ALL_CDR")
    for i in range(n):
        if not exposed[i]:
            continue
        for j in range(i + 1, n):
            if not exposed[j]:
                continue
            rij = D[i, j]
            if not np.isfinite(rij) or rij >= R or rij <= 0:
                continue
            term = H[i] * H[j] / (rij * rij)
            s_fv += term
            if cdr[i] or cdr[j]:
                s_cdr += term
    return {
        "LIT3_PSH_KD__FV": float(s_fv),
        "LIT3_PSH_KD__CDR_vicinity_proxy": float(s_cdr),
    }


def lit4_positive_sasa(ab: dict, scale_id: str) -> dict[str, float]:
    """Positive-SASA style: sum max(h_raw,0) * total_SASA over residues."""
    h = np.asarray([property_raw(scale_id)[a] for a in ab["aa"]], float)
    sasa = ab["total_SASA"]
    pos = np.maximum(h, 0.0) * sasa
    pos = pos[np.isfinite(pos)]
    return {
        f"LIT4_POS_SASA_{scale_id}__SUM": float(pos.sum()) if len(pos) else np.nan,
        f"LIT4_POS_SASA_{scale_id}__MAX": float(pos.max()) if len(pos) else np.nan,
        f"LIT4_POS_SASA_{scale_id}__MEAN": float(pos.mean()) if len(pos) else np.nan,
    }


def lit5_direct_sasa(ab: dict, scale_id: str) -> dict[str, float]:
    """Direct/total surface hydrophobicity: sum h_raw * total_SASA (signed)."""
    h = np.asarray([property_raw(scale_id)[a] for a in ab["aa"]], float)
    sasa = ab["total_SASA"]
    w = h * sasa
    w = w[np.isfinite(w)]
    return {
        f"LIT5_DIRECT_SASA_{scale_id}__SUM": float(w.sum()) if len(w) else np.nan,
        f"LIT5_DIRECT_SASA_{scale_id}__MEAN": float(w.mean()) if len(w) else np.nan,
        f"LIT5_DIRECT_SASA_{scale_id}__MAX": float(w.max()) if len(w) else np.nan,
    }


def all_literature_anchors(ab: dict) -> dict[str, float]:
    out = {}
    out.update(lit1_static_sap_kd_v1(ab))
    out.update(lit2_static_canonical_sap_bm(ab, 5.0))
    out.update(lit2_static_canonical_sap_bm(ab, 10.0))
    out.update(lit3_psh_kd(ab))
    for sid in ("KD", "BM", "WW", "EIS", "MEEK", "MIY", "FP"):
        out.update(lit4_positive_sasa(ab, sid))
        out.update(lit5_direct_sasa(ab, sid))
    return out
