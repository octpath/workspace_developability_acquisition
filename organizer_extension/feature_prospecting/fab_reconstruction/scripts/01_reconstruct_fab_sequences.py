#!/usr/bin/env python3
"""Shehata Fab reconstruction: Gates R1–R3, sequence generation, QC (target-blind)."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
SEQ = OUT / "sequences"
QC = OUT / "qc"
B1 = ROOT / "gate_b1/data"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
RETRIEVAL_DATE = "2026-09-05"

# UniProt / IMGT-aligned canonical constants (secreted IGHG1 CH1 + EPKSC stub)
# CH1 EU ~118–215 + upper hinge through Cys220 (EPKSC) for κ H–L disulfide.
# Papain exact terminus UNKNOWN — stub is a reconstructed chemical-complete choice, not experimental.
CH1_ID = "IGHG1_CH1_EPKSC_UNIPROT_P01857"
CH1_SEQ = (
    "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKV"
    "EPKSC"
)
CK_ID = "IGKC_UNIPROT_P01834"
CK_SEQ = "RTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
CL_ID = "IGLC2_UNIPROT_P0DOY2"
CL_SEQ = "GQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS"

RECON_CLASS = "RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def write_policy() -> None:
    SEQ.mkdir(parents=True, exist_ok=True)
    md = f"""# CONSTANT DOMAIN POLICY (FROZEN)

**CONSTANT_DOMAIN_POLICY_FROZEN_BEFORE_STRUCTURE_PREDICTION = true**  
**Freeze date:** {RETRIEVAL_DATE}  
**Reconstruction class:** `{RECON_CLASS}`

## Decision

Gate R0 establishes recombinant **IgG1** (HIGH) and papain Fab production (HIGH), but **not** exact CH1 / Cκ / Cλ alleles or papain C-terminus.

Therefore:

- **POLICY B** — reconstruct experimental-like Fab using authoritative canonical constant sequences.
- Do **not** claim `EXACT_EXPERIMENTAL_FAB`.
- Do **not** choose constants using TmApp/HIC (target-blind).

## Sequences used

| Role | Sequence ID | Source | Accession | Notes |
|------|-------------|--------|-----------|-------|
| Heavy constant | `{CH1_ID}` | UniProt IGHG1 | P01857 | CH1 domain + `EPKSC` (through Cys220). Exact papain hinge length **unknown**. |
| C-kappa | `{CK_ID}` | UniProt IGKC | P01834 | Full Cκ including C-terminal Cys. |
| C-lambda | `{CL_ID}` | UniProt IGLC2 | P0DOY2 | **Single** predeclared Cλ for all λ Abs; cognate IGLC gene unknown. |

## Light-chain assignment rule

1. Prefer project `ORG_kappa_lambda` / `PL_kappa_lambda` from `numbering_germline.csv`.
2. Else gene prefix: IGKV→κ, IGLV→λ.
3. Unresolved → flag; do not invent.

## Explicit uncertainties

- Exact IGHG1 allele / CH1 SNPs
- Exact IGKC / IGLC alleles
- Cloning junctions / scars
- Papain cleavage heterogeneity / hinge residues beyond/short of `EPKSC`
- Whether yeast vectors used engineered constant interfaces (orthogonal CH1 patents exist at Adimab but not tied to this cohort)

## Exact AA strings

See `CONSTANT_DOMAIN_SEQUENCES.fasta` and `CONSTANT_DOMAIN_PROVENANCE.csv`.
"""
    (SEQ / "CONSTANT_DOMAIN_POLICY.md").write_text(md)

    fasta = (
        f">{CH1_ID} CH1+EPKSC UniProt:P01857 retrieval:{RETRIEVAL_DATE}\n{CH1_SEQ}\n"
        f">{CK_ID} UniProt:P01834 retrieval:{RETRIEVAL_DATE}\n{CK_SEQ}\n"
        f">{CL_ID} UniProt:P0DOY2 retrieval:{RETRIEVAL_DATE}\n{CL_SEQ}\n"
    )
    (SEQ / "CONSTANT_DOMAIN_SEQUENCES.fasta").write_text(fasta)

    prov = pd.DataFrame(
        [
            {
                "sequence_id": CH1_ID,
                "role": "CH1_plus_hinge_stub",
                "amino_acid_sequence": CH1_SEQ,
                "length": len(CH1_SEQ),
                "source_database": "UniProt",
                "accession": "P01857",
                "gene_allele": "IGHG1 (allele not pinned; CH1+EPKSC excerpt)",
                "domain_boundaries": "CH1 (ASTK...VDKKV) + EPKSC through EU Cys220",
                "rationale": "IgG1 HIGH from Shehata; allele/terminus unknown; chemically complete Fab stub",
                "retrieval_date": RETRIEVAL_DATE,
            },
            {
                "sequence_id": CK_ID,
                "role": "C_kappa",
                "amino_acid_sequence": CK_SEQ,
                "length": len(CK_SEQ),
                "source_database": "UniProt",
                "accession": "P01834",
                "gene_allele": "IGKC",
                "domain_boundaries": "full IGKC including C-terminal C",
                "rationale": "canonical human Cκ; exact vector allele unknown",
                "retrieval_date": RETRIEVAL_DATE,
            },
            {
                "sequence_id": CL_ID,
                "role": "C_lambda_primary",
                "amino_acid_sequence": CL_SEQ,
                "length": len(CL_SEQ),
                "source_database": "UniProt",
                "accession": "P0DOY2",
                "gene_allele": "IGLC2 (predeclared primary for all λ)",
                "domain_boundaries": "full IGLC2 including C-terminal C",
                "rationale": "one common Cλ; do not assign IGLC gene per antibody",
                "retrieval_date": RETRIEVAL_DATE,
            },
        ]
    )
    prov.to_csv(SEQ / "CONSTANT_DOMAIN_PROVENANCE.csv", index=False)

    # Disulfide expectations
    # κ: heavy Cys220 (in EPKSC) ↔ light C-term Cys; intradomain CH1 Cys (EU 144–200), Cκ Cys (134–194)
    # λ: often heavy Cys131 (in CH1) ↔ light C-term; Cys220 may be unpaired — record both candidates
    rows = [
        {
            "pair_id": "VH_intradomain",
            "chain": "heavy",
            "cys_a_domain": "VH",
            "cys_b_domain": "VH",
            "topology_note": "conserved Ig fold Cys in VH (IMGT ~23–104); sequence-dependent positions",
            "kappa_lambda": "both",
            "required_for_prediction": False,
        },
        {
            "pair_id": "VL_intradomain",
            "chain": "light",
            "cys_a_domain": "VL",
            "cys_b_domain": "VL",
            "topology_note": "conserved Ig fold Cys in VL",
            "kappa_lambda": "both",
            "required_for_prediction": False,
        },
        {
            "pair_id": "CH1_intradomain",
            "chain": "heavy",
            "cys_a_domain": "CH1",
            "cys_b_domain": "CH1",
            "topology_note": "CH1 Cys pair in ASTKGPS...ALGCLVK...ICNVNHK (two Cys in CH1 block)",
            "kappa_lambda": "both",
            "required_for_prediction": False,
        },
        {
            "pair_id": "CL_intradomain",
            "chain": "light",
            "cys_a_domain": "CL",
            "cys_b_domain": "CL",
            "topology_note": "Cκ/Cλ intradomain Cys pair + terminal Cys for H–L",
            "kappa_lambda": "both",
            "required_for_prediction": False,
        },
        {
            "pair_id": "HL_interchain_kappa",
            "chain": "H-L",
            "cys_a_domain": "hinge_stub_EPKSC",
            "cys_b_domain": "Ckappa_Cterm",
            "topology_note": "Expected IgG1-κ: heavy Cys220 ↔ κ Cys214 (C-term). Papain terminus uncertain.",
            "kappa_lambda": "kappa",
            "required_for_prediction": False,
        },
        {
            "pair_id": "HL_interchain_lambda",
            "chain": "H-L",
            "cys_a_domain": "CH1_Cys131_candidate",
            "cys_b_domain": "Clambda_Cterm",
            "topology_note": "IgG1-λ often uses CH1 Cys131↔λ C-term; Cys220 may be unpaired. Record distances for both heavy Cys candidates in QC.",
            "kappa_lambda": "lambda",
            "required_for_prediction": False,
        },
    ]
    pd.DataFrame(rows).to_csv(SEQ / "FAB_DISULFIDE_EXPECTATIONS.csv", index=False)


def audit_junction(vh: str, vl: str) -> dict:
    """Trim accidental constant overlaps; do not delete valid FR4."""
    notes = []
    vh_used = vh
    vl_used = vl
    vh_removed = ""
    vl_removed = ""

    # If VH already contains CH1 start, trim from first ASTKGPS
    idx = vh.find("ASTKGPS")
    if idx >= 0:
        vh_removed = vh[idx:]
        vh_used = vh[:idx]
        notes.append(f"VH_trimmed_CH1_overlap_len{len(vh_removed)}")

    # If VL already contains Cκ/Cλ starts
    for marker, name in [("RTVAAP", "CK"), ("GQPKAA", "CL"), ("GQPKAN", "CL1")]:
        j = vl.find(marker)
        if j >= 0:
            vl_removed = vl[j:]
            vl_used = vl[:j]
            notes.append(f"VL_trimmed_{name}_overlap_len{len(vl_removed)}")
            break

    # Biological boundary heuristic: FR4-like endings
    vh_ok_end = vh_used.endswith("VTVSS") or vh_used.endswith("VTVSA") or vh_used[-5:] in {
        "VTVSS",
        "ITVSS",
        "LTVSS",
    }
    vl_ok_end = vl_used[-1] in "KL" or vl_used.endswith("TVL") or vl_used.endswith("EIK") or vl_used.endswith("DIK")

    status = "OK"
    if vh_removed or vl_removed:
        status = "TRIMMED_CONSTANT_OVERLAP"
    if not vh_used or not vl_used:
        status = "CRITICAL_EMPTY_AFTER_TRIM"
    if vh_used != vh and not vh_removed:
        status = "UNEXPECTED"

    return {
        "VH_original": vh,
        "VL_original": vl,
        "VH_used": vh_used,
        "VL_used": vl_used,
        "VH_residues_removed": vh_removed,
        "VL_residues_removed": vl_removed,
        "VH_removal_reason": "leading_CH1_overlap" if vh_removed else "",
        "VL_removal_reason": "leading_CL_overlap" if vl_removed else "",
        "VH_CL_junction": (vh_used[-4:] + "|" + CH1_SEQ[:4]) if vh_used else "",
        "VL_CL_junction": (
            vl_used[-4:] + "|" + (CK_SEQ[:4] if True else "")
        ),
        "junction_status": status,
        "vh_fr4_like_end": bool(vh_ok_end),
        "vl_fr4_like_end": bool(vl_ok_end),
        "notes": ";".join(notes),
    }


def main() -> None:
    write_policy()

    full = pd.read_csv(B1 / "shehata_b1_full.csv")
    ng = pd.read_csv(B1 / "numbering_germline.csv")
    # Prefer TmApp∪HIC union size used elsewhere: use all 400 in shehata_b1_full,
    # but competition Fab cohort often 324 triple-complete. Use ids present in both
    # numbering + full. User asked for all 324 — use intersection with tmapp if 324.
    tm = pd.read_csv(B1 / "tmapp_full.csv")
    hic = pd.read_csv(B1 / "hic_full.csv")
    # Target-blind: use ID lists only (no label values)
    ids_324 = sorted(set(tm["antibody_id"]) & set(hic["antibody_id"]))
    # If not 324, fall back to full∩ng
    if len(ids_324) != 324:
        ids_324 = sorted(set(full["antibody_id"]) & set(ng["antibody_id"]))
        # still try length 324 from triple if available
    df = full[full["antibody_id"].isin(ids_324)].copy()
    df = df.merge(
        ng[
            [
                "antibody_id",
                "ORG_author_vh_germline",
                "ORG_author_vl_germline",
                "ORG_kappa_lambda",
                "PL_kappa_lambda",
                "PL_anarci_vh_v_gene",
                "PL_anarci_vl_v_gene",
                "PL_anarci_kappa_lambda",
            ]
        ],
        on="antibody_id",
        how="left",
    )
    assert len(df) == 324, f"expected 324, got {len(df)}"

    audit_rows = []
    for _, r in df.iterrows():
        locus = r.get("ORG_kappa_lambda") or r.get("PL_kappa_lambda") or r.get("PL_anarci_kappa_lambda")
        if pd.isna(locus) or locus not in ("kappa", "lambda"):
            vg = str(r.get("PL_anarci_vl_v_gene") or r.get("ORG_author_vl_germline") or "")
            if vg.startswith("IGKV") or vg.startswith("VK"):
                locus = "kappa"
            elif vg.startswith("IGLV") or vg.startswith("VL"):
                locus = "lambda"
            else:
                locus = "unresolved"
        audit_rows.append(
            {
                "antibody_id": r["antibody_id"],
                "VH_sequence": r["heavy"],
                "VL_sequence": r["light"],
                "VH_length": len(r["heavy"]),
                "VL_length": len(r["light"]),
                "heavy_V_gene": r.get("PL_anarci_vh_v_gene") or r.get("ORG_author_vh_germline") or r.get("vh_germline"),
                "light_V_gene": r.get("PL_anarci_vl_v_gene") or r.get("ORG_author_vl_germline") or r.get("vl_germline"),
                "light_locus": locus,
                "source_b_cell_subset": r.get("b_cell_subset"),
                "sequence_source": r.get("sequence_source"),
                "locus_source": "ORG_kappa_lambda" if r.get("ORG_kappa_lambda") in ("kappa", "lambda") else "fallback",
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(SEQ / "SHEHATA_SEQUENCE_AUDIT.csv", index=False)
    n_k = int((audit["light_locus"] == "kappa").sum())
    n_l = int((audit["light_locus"] == "lambda").sum())
    n_u = int((audit["light_locus"] == "unresolved").sum())
    print(f"R1: n={len(audit)} kappa={n_k} lambda={n_l} unresolved={n_u}")
    if n_u:
        raise SystemExit("STOP: unresolved light locus")

    # Junction audit
    jrows = []
    for _, r in audit.iterrows():
        j = audit_junction(r["VH_sequence"], r["VL_sequence"])
        j["antibody_id"] = r["antibody_id"]
        j["light_locus"] = r["light_locus"]
        # fix VL junction label with correct CL
        cl = CK_SEQ if r["light_locus"] == "kappa" else CL_SEQ
        j["VL_CL_junction"] = j["VL_used"][-4:] + "|" + cl[:4]
        jrows.append(j)
    jdf = pd.DataFrame(jrows)
    jdf.to_csv(SEQ / "FAB_JUNCTION_AUDIT.csv", index=False)
    print("R2 junction_status:", jdf["junction_status"].value_counts().to_dict())
    if (jdf["junction_status"] == "CRITICAL_EMPTY_AFTER_TRIM").any():
        raise SystemExit("STOP: critical junction failure")

    # Manual QC examples (target-blind)
    man = []
    for locus in ("kappa", "lambda"):
        sub = audit[audit["light_locus"] == locus].sample(n=min(5, (audit.light_locus == locus).sum()), random_state=0)
        for _, r in sub.iterrows():
            man.append({"pick": f"random_{locus}", "antibody_id": r["antibody_id"]})
    man.append({"pick": "shortest_VH", "antibody_id": audit.loc[audit["VH_length"].idxmin(), "antibody_id"]})
    man.append({"pick": "longest_VH", "antibody_id": audit.loc[audit["VH_length"].idxmax(), "antibody_id"]})
    man.append({"pick": "shortest_VL", "antibody_id": audit.loc[audit["VL_length"].idxmin(), "antibody_id"]})
    man.append({"pick": "longest_VL", "antibody_id": audit.loc[audit["VL_length"].idxmax(), "antibody_id"]})
    unusual = jdf[jdf["junction_status"] != "OK"]
    for _, r in unusual.iterrows():
        man.append({"pick": "unusual_junction", "antibody_id": r["antibody_id"]})
    pd.DataFrame(man).drop_duplicates("antibody_id").to_csv(SEQ / "JUNCTION_MANUAL_QC_PICKS.csv", index=False)

    # Build Fab sequences
    fab_rows = []
    for _, r in audit.iterrows():
        j = jdf[jdf["antibody_id"] == r["antibody_id"]].iloc[0]
        vh_u, vl_u = j["VH_used"], j["VL_used"]
        if r["light_locus"] == "kappa":
            cl_id, cl = CK_ID, CK_SEQ
        else:
            cl_id, cl = CL_ID, CL_SEQ
        heavy_fab = vh_u + CH1_SEQ
        light_fab = vl_u + cl
        unc = []
        if j["junction_status"] != "OK":
            unc.append(j["junction_status"])
        unc.append("CH1_allele_unknown;papain_terminus_assumed_EPKSC;CL_allele_surrogate")
        fab_rows.append(
            {
                "id": r["antibody_id"],
                "VH_original": r["VH_sequence"],
                "VL_original": r["VL_sequence"],
                "VH_used": vh_u,
                "VL_used": vl_u,
                "light_locus": r["light_locus"],
                "heavy_fab_seq": heavy_fab,
                "light_fab_seq": light_fab,
                "heavy_length": len(heavy_fab),
                "light_length": len(light_fab),
                "reconstruction_class": RECON_CLASS,
                "CH1_sequence_id": CH1_ID,
                "CL_sequence_id": cl_id,
                "junction_status": j["junction_status"],
                "uncertainty_notes": ";".join(unc),
                "VH_len_used": len(vh_u),
                "VL_len_used": len(vl_u),
                "heavy_V_gene": r["heavy_V_gene"],
                "light_V_gene": r["light_V_gene"],
            }
        )
    fab = pd.DataFrame(fab_rows)
    fab.to_csv(SEQ / "SHEHATA_RECONSTRUCTED_FAB.csv", index=False)

    # FASTA
    h_lines, l_lines, m_lines = [], [], []
    for _, r in fab.iterrows():
        h_lines.append(f">{r['id']}_H\n{r['heavy_fab_seq']}")
        l_lines.append(f">{r['id']}_L\n{r['light_fab_seq']}")
        m_lines.append(f">{r['id']}\n{r['heavy_fab_seq']}:{r['light_fab_seq']}")
    (SEQ / "fab_heavy.fasta").write_text("\n".join(h_lines) + "\n")
    (SEQ / "fab_light.fasta").write_text("\n".join(l_lines) + "\n")
    (SEQ / "fab_esmfold_multimer.fasta").write_text("\n".join(m_lines) + "\n")

    # Sequence QC
    qc_rows = []
    crit = False
    for _, r in fab.iterrows():
        issues = []
        for label, s in [("H", r["heavy_fab_seq"]), ("L", r["light_fab_seq"])]:
            if any(c not in AA20 for c in s):
                issues.append(f"{label}_nonstandard_AA")
            if "*" in s or "-" in s or "X" in s:
                issues.append(f"{label}_stop_gap_or_X")
        if not r["heavy_fab_seq"].endswith(CH1_SEQ):
            issues.append("CH1_not_suffix")
        if r["light_locus"] == "kappa" and not r["light_fab_seq"].endswith(CK_SEQ):
            issues.append("CK_not_suffix")
        if r["light_locus"] == "lambda" and not r["light_fab_seq"].endswith(CL_SEQ):
            issues.append("CL_not_suffix")
        if r["heavy_fab_seq"].count(CH1_SEQ) != 1:
            issues.append("CH1_duplicated_or_missing")
        # variable preserved
        if not r["heavy_fab_seq"].startswith(r["VH_used"]):
            issues.append("VH_not_prefix")
        if not r["light_fab_seq"].startswith(r["VL_used"]):
            issues.append("VL_not_prefix")
        # conserved Cys counts
        if r["heavy_fab_seq"].count("C") < 4:
            issues.append("heavy_few_Cys")
        if r["light_fab_seq"].count("C") < 3:
            issues.append("light_few_Cys")
        if issues:
            crit = True
        qc_rows.append(
            {
                "id": r["id"],
                "pass": len(issues) == 0,
                "issues": ";".join(issues),
                "heavy_sha256": sha256_text(r["heavy_fab_seq"]),
                "light_sha256": sha256_text(r["light_fab_seq"]),
                "heavy_length": r["heavy_length"],
                "light_length": r["light_length"],
                "light_locus": r["light_locus"],
                "reconstruction_class": r["reconstruction_class"],
            }
        )
    qcdf = pd.DataFrame(qc_rows)
    QC.mkdir(parents=True, exist_ok=True)
    qcdf.to_csv(QC / "FAB_SEQUENCE_QC.csv", index=False)
    manifest = {
        "n": len(fab),
        "n_kappa": n_k,
        "n_lambda": n_l,
        "n_unresolved": n_u,
        "n_qc_pass": int(qcdf["pass"].sum()),
        "n_qc_fail": int((~qcdf["pass"]).sum()),
        "reconstruction_class": RECON_CLASS,
        "CH1_sequence_id": CH1_ID,
        "CK_sequence_id": CK_ID,
        "CL_sequence_id": CL_ID,
        "CH1_sha256": sha256_text(CH1_SEQ),
        "CK_sha256": sha256_text(CK_SEQ),
        "CL_sha256": sha256_text(CL_SEQ),
        "CONSTANT_DOMAIN_POLICY_FROZEN_BEFORE_STRUCTURE_PREDICTION": True,
        "critical_stop_structure": bool(crit or n_u > 0),
        "length_stats": {
            "heavy_min": int(fab["heavy_length"].min()),
            "heavy_max": int(fab["heavy_length"].max()),
            "light_min": int(fab["light_length"].min()),
            "light_max": int(fab["light_length"].max()),
        },
        "date": str(date.today()),
    }
    (QC / "FAB_SEQUENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("QC pass", manifest["n_qc_pass"], "/", manifest["n"], "stop_structure", manifest["critical_stop_structure"])
    if manifest["critical_stop_structure"]:
        raise SystemExit("STOP STRUCTURE GENERATION: sequence QC critical failures")
    print("SEQUENCE_RECONSTRUCTION_OK")


if __name__ == "__main__":
    main()
