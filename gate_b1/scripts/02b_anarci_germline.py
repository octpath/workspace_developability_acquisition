#!/usr/bin/env python3
"""Re-annotate germline with real ANARCI (participant-legal sequence inference)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import DATA, REPORTS, ensure_dirs, write_json  # noqa: E402


def family_from_gene(g: str | None) -> str | None:
    if g is None or (isinstance(g, float) and pd.isna(g)):
        return None
    s = str(g).replace("IGHV", "VH").replace("IGKV", "VK").replace("IGLV", "VL")
    import re

    m = re.match(r"(IGHV|IGKV|IGLV|VH|VK|VL)(\d+)", s, re.I)
    if m:
        pref = m.group(1).upper().replace("IGHV", "VH").replace("IGKV", "VK").replace("IGLV", "VL")
        return f"{pref}{m.group(2)}"
    return None


def kappa_lambda(chain_type: str | None, gene: str | None) -> str | None:
    if chain_type == "K":
        return "kappa"
    if chain_type == "L":
        return "lambda"
    return None


def assign_one(seq: str, allow):
    from anarci import anarci

    numbered, details, _ = anarci([("q", seq)], scheme="imgt", output=False, allow=allow, allowed_species=["human"])
    if not details or not details[0]:
        return {}
    d = details[0][0]
    # germlines often in d; also query hit
    v_gene = d.get("germlines", {}).get("v_gene") if isinstance(d.get("germlines"), dict) else None
    j_gene = d.get("germlines", {}).get("j_gene") if isinstance(d.get("germlines"), dict) else None
    # ANARCI details format varies — try common fields
    if v_gene is None:
        # sometimes nested list
        gl = d.get("germlines")
        if isinstance(gl, dict):
            # {'v_gene': [[(species, allele),...], bit], ...} in older ANARCI
            vg = gl.get("v_gene")
            if isinstance(vg, list) and vg:
                try:
                    v_gene = vg[0][0][1]
                except Exception:
                    v_gene = str(vg)
            jg = gl.get("j_gene")
            if isinstance(jg, list) and jg:
                try:
                    j_gene = jg[0][0][1]
                except Exception:
                    j_gene = str(jg)
    return {
        "v_gene": v_gene,
        "j_gene": j_gene,
        "chain_type": d.get("chain_type"),
        "species": d.get("species"),
        "evalue": d.get("evalue"),
        "bitscore": d.get("bitscore"),
        "raw": {k: d.get(k) for k in ("chain_type", "species", "evalue", "bitscore", "scheme")},
    }


def main():
    ensure_dirs()
    df = pd.read_csv(DATA / "shehata_b1_full.csv")
    num = pd.read_csv(DATA / "numbering_germline.csv")

    # Probe details structure
    from anarci import anarci

    n, d, h = anarci(
        [("x", df.iloc[0]["heavy"])],
        scheme="imgt",
        output=False,
        allow={"H"},
        allowed_species=["human"],
    )
    write_json(DATA / "anarci_details_probe.json", {"details": d, "hit": str(h)[:500]})
    print("PROBE", d[0][0] if d and d[0] else None)

    rows = []
    for _, r in df.iterrows():
        h_as = assign_one(r["heavy"], {"H"})
        l_as = assign_one(r["light"], {"L", "K"})
        rows.append(
            {
                "antibody_id": r["antibody_id"],
                "PL_anarci_vh_v_gene": h_as.get("v_gene"),
                "PL_anarci_vh_j_gene": h_as.get("j_gene"),
                "PL_anarci_vl_v_gene": l_as.get("v_gene"),
                "PL_anarci_vl_j_gene": l_as.get("j_gene"),
                "PL_anarci_vh_family": family_from_gene(h_as.get("v_gene")),
                "PL_anarci_vl_family": family_from_gene(l_as.get("v_gene")),
                "PL_anarci_kappa_lambda": kappa_lambda(l_as.get("chain_type"), l_as.get("v_gene")),
                "PL_anarci_vh_bitscore": h_as.get("bitscore"),
                "PL_anarci_vl_bitscore": l_as.get("bitscore"),
                "PL_anarci_H_ok": bool(h_as),
                "PL_anarci_L_ok": bool(l_as),
            }
        )
    ann = pd.DataFrame(rows)
    # Merge into numbering file — update PL family fields
    out = num.drop(
        columns=[c for c in num.columns if c.startswith("PL_anarci_") or c in [
            "PL_vh_v_gene", "PL_vl_v_gene", "PL_vh_family", "PL_vl_family", "PL_kappa_lambda",
            "PL_vh_family_filled", "PL_vl_family_filled", "PL_family_source",
        ]],
        errors="ignore",
    )
    out = out.merge(ann, on="antibody_id")
    out["PL_vh_v_gene"] = out["PL_anarci_vh_v_gene"]
    out["PL_vl_v_gene"] = out["PL_anarci_vl_v_gene"]
    out["PL_vh_family"] = out["PL_anarci_vh_family"]
    out["PL_vl_family"] = out["PL_anarci_vl_family"]
    out["PL_kappa_lambda"] = out["PL_anarci_kappa_lambda"]
    out["PL_vh_family_filled"] = out["PL_vh_family"].fillna(out["ORG_author_vh_family"])
    out["PL_vl_family_filled"] = out["PL_vl_family"].fillna(out["ORG_author_vl_family"])
    out["PL_family_source"] = out.apply(
        lambda r: "anarci" if pd.notna(r["PL_vh_family"]) and pd.notna(r["PL_vl_family"]) else "author_fallback",
        axis=1,
    )
    # Agreement with author
    agree_h = (out["PL_vh_family"] == out["ORG_author_vh_family"]).mean()
    agree_l = (out["PL_vl_family"] == out["ORG_author_vl_family"]).mean()
    out.to_csv(DATA / "numbering_germline.csv", index=False)
    audit = {
        "anarci_H_ok": float(out["PL_anarci_H_ok"].mean()),
        "anarci_L_ok": float(out["PL_anarci_L_ok"].mean()),
        "family_source": out["PL_family_source"].value_counts().to_dict(),
        "family_agree_author_H": float(agree_h),
        "family_agree_author_L": float(agree_l),
    }
    write_json(DATA / "anarci_germline_audit.json", audit)
    # append report
    with open(REPORTS / "numbering_and_germline_audit.md", "a") as f:
        f.write("\n## ANARCI germline re-annotation\n\n")
        for k, v in audit.items():
            f.write(f"- {k}: {v}\n")
    print("ANARCI_GERMLINE_OK", audit)


if __name__ == "__main__":
    main()
