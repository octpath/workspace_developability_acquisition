#!/usr/bin/env python3
"""Round1 PDB-derived feature inventory audit — code/artifact only, no new training."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "virtual_participant/round1_pdb_feature_inventory"
ST3_MAN = ROOT / "virtual_participant/stage3_structure/stage3_feature_manifest.csv"
ST4_MAN = ROOT / "virtual_participant/stage4_advanced_structure/stage4_feature_manifest.csv"
ST3_REG = ROOT / "virtual_participant/stage3_structure/stage3_model_registry.csv"
ST4_REG = ROOT / "virtual_participant/stage4_advanced_structure/stage4_model_registry.csv"
INV = ROOT / "virtual_participant/round1_postmortem/round1_all_model_score_inventory.csv"

# Frozen parameters (from code audit)
PARAMS = {
    "probe_radius_A": 1.4,
    "sasa_n_points": 100,
    "rasa_exposed_threshold": 0.20,
    "patch_ca_adjacency_A": 8.0,
    "interface_ca_cutoff_A": 5.0,
    "packing_contact_A": 8.0,
    "clash_ca_A_stage3": 2.5,
    "clash_heavy_A_stage4": 2.2,
    "hb_dist_A": 3.5,
    "sb_dist_A": 4.0,
    "heavy_contact_A": 4.5,
    "local_neighborhood_A": 10.0,
    "cavity_unsat_neighbor_A": 6.0,
    "ph": 6.5,
    "ionic_strength_M": 0.15,
    "maxasa_reference": "Tien2013/Wilke per-residue",
    "hydrophobic_residues": "AILMFVW",
    "aromatic_residues": "FWY",
    "positive_residues": "KRH",
    "negative_residues": "DE",
    "polar_residues": "STNQ",
    "hydrophobicity_scale": "Kyte-Doolittle KD",
    "charge_weight": "D/E=-1,K/R=1,H=0.1",
}

AA_SETS = {
    "hydrophobic": "AILMFVW",
    "aromatic": "FWY",
    "positive": "KRH",
    "negative": "DE",
    "polar": "STNQ",
}

FINAL_REP = {
    "HIC": ["HIC__SIMPLE_blend_seq_surf_adv", "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt",
            "HIC__ADV_SURFACE_PATCH__SVROpt", "HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt"],
    "TmApp": ["TmApp__META_performance__ridge_100.0", "TmApp__ESMFold__STRUCT_RASA__ElasticNetOpt",
              "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt", "TmApp__ADV_TMAPP_ALL__SVROpt"],
}

BUNDLE_FAMILIES = {
    "STRUCT_GLOBAL": "GLOBAL_GEOMETRY",
    "STRUCT_SASA": "SASA_RASA",
    "STRUCT_RASA": "SASA_RASA",
    "STRUCT_SURFACE_CHEM": "SURFACE_CHEMISTRY",
    "STRUCT_PATCH": "SURFACE_PATCH_SIMPLE",
    "STRUCT_INTERFACE": "VH_VL_INTERFACE",
    "STRUCT_PACKING": "PACKING_CONTACT",
    "ADV_PROPKA": "PKA_PROTONATION",
    "ADV_PQR_CHARGE": "PKA_PROTONATION",
    "ADV_ELECTROSTATICS": "ELECTROSTATICS",
    "ADV_SURFACE_PATCH": "SURFACE_PATCH_ADVANCED",
    "ADV_INTERACTIONS": "PACKING_CONTACT",
    "ADV_CAVITY": "CAVITY_PROXY",
    "ADV_UNSAT_POLAR": "UNSAT_POLAR_PROXY",
    "ADV_INVFOLD": "INVERSE_FOLDING",
    "ADV_OTHER": "PACKING_CONTACT",
}


def parse_region(name: str) -> tuple[str, str]:
    n = name.lower()
    if n.startswith("vh") or n.startswith("h_cdr") or n.startswith("propka_vh"):
        chain = "Heavy"
    elif n.startswith("vl") or n.startswith("l_cdr") or n.startswith("propka_vl"):
        chain = "Light"
    elif "interface" in n or "iface" in n or "bsa" in n or "vh_vl" in n:
        chain = "H+L"
    else:
        chain = "H+L"
    if "h_cdr3" in n or "hcdr3" in n or "h3_" in n:
        region = "HCDR3"
    elif "cdr" in n and "all_cdr" not in n:
        region = "CDR"
    elif "all_cdr" in n:
        region = "all_CDR"
    elif "fr" in n and ("_fr" in n or n.startswith("fr")):
        region = "framework"
    elif "interface" in n or "iface" in n:
        region = "VH-VL interface"
    elif n.startswith("fv") or n.startswith("adv_") or n.startswith("propka") or n.startswith("apbs"):
        region = "whole Fv"
    else:
        region = "whole"
    return chain, region


def raw_quantity(name: str, family: str) -> str:
    n = name.lower()
    if "rasa" in n and "w_" not in n:
        return "residue-level RASA (SASA/MaxASA)"
    if "sasa" in n and "ratio" not in n and "patch" not in n:
        return "residue-level SASA"
    if "rasa_w_hydrophobicity" in n:
        return "residue-level RASA × KD hydrophobicity"
    if "rasa_w_charge" in n or "exposed_pos" in n or "exposed_neg" in n:
        return "residue-level RASA × charge weight"
    if "patch" in n or n.startswith("adv_"):
        return "exposed residue CA graph + local SASA"
    if "plddt" in n or "confidence" in n:
        return "ESMFold per-residue pLDDT"
    if n.startswith("propka"):
        return "PROPKA residue pKa → Henderson-Hasselbalch charge at pH 6.5"
    if n.startswith("pqr"):
        return "PDB2PQR atom partial charges at pH 6.5"
    if n.startswith("apbs"):
        return "APBS electrostatic potential sampled at exposed residue CA"
    if n.startswith("invfold"):
        return "ESM-IF1 sequence log-likelihood given structure"
    if "cavity" in n:
        return "buried CA neighbor degree (packing-derived)"
    if "unsatisfied" in n:
        return "buried polar/charged + N/O neighbor count"
    if "salt_bridge" in n:
        return "heavy-atom distance donor/acceptor pairs"
    if "hbond" in n:
        return "N/O heavy atom distance proxy"
    if "contact" in n or "clash" in n or "packing" in n or "rg" in n:
        return "CA/heavy-atom distances"
    if "bsa" in n or "interface" in n:
        return "chain SASA difference + CA cross-chain distance"
    return "structure-derived scalar aggregate"


def exact_definition(name: str, family: str) -> str:
    n = name.lower()
    defs = {
        "total_sasa": "sum of per-residue Shrake-Rupley SASA over region",
        "mean_sasa": "mean per-residue SASA over region",
        "mean_rasa": "mean(SASA/MaxASA_Tien2013) over region",
        "median_rasa": "median RASA over region",
        "max_rasa": "max RASA over region",
        "frac_rasa_gt": "fraction of residues with RASA above threshold",
        "n_exposed": "count residues with RASA >= 0.20",
        "sasa_hydrophobic": f"sum SASA where aa in {AA_SETS['hydrophobic']}",
        "rasa_w_hydrophobicity_sum": "sum(RASA × KD) over region",
        "exposed_pos": "sum RASA for K/R/H in region",
        "n_hydrophobic_patches": "connected components among exposed hydrophobic residues (CA<=8Å)",
        "largest_hydrophobic_patch_sasa": "max component SASA sum (simple patch)",
        "bsa": "SASA_VH_iso + SASA_VL_iso - SASA_complex",
        "n_interface_res": "residues with cross-chain CA min distance <= 5Å",
        "contact_density": "CA contacts (<8Å) / n_res or n_ca",
        "rg": "radius of gyration from CA coordinates",
        "compactness": "Rg / n^(1/3) [Stage3] or patch area/size^(2/3) [Stage4]",
        "propka_mean_pka": "mean pKa over ionizable residues from .pka file",
        "apbs_mean_potential": "mean DX potential at exposed CA (+1.4Å offset)",
        "apbs_pos_patch_count": "connected components among exposed residues with potential > +0.5 kT/e",
        "adv_max_local_hydrophobic_sasa": "max over exposed residues of sum hydrophobic SASA within 10Å",
        "adv_spatial_hydrophobicity_sum": "sum(local_hydro_SASA × hydrophobic SASA) — not canonical SAP",
        "cavity_proxy_n_sites": "buried residues with CA neighbor degree <= 15th percentile",
        "buried_unsatisfied_polar_proxy": "buried polar/charged with zero N/O within 6Å of CA",
        "invfold_fv_mean_ll": "mean ESM-IF1 log-likelihood chains A+B",
        "hbond_proxy_count": "N-O heavy atom pairs <=3.5Å, different residues, no angle criterion",
    }
    for k, v in defs.items():
        if k in n:
            return v
    return f"see code: {family}"


def software_for(family: str, name: str) -> dict:
    base = {"library": "BioPython", "software": "ShrakeRupley", "software_version": "1.88 (via BioPython)"}
    if family.startswith("ADV_") or name.startswith(("propka", "pqr", "apbs", "adv_", "invfold", "salt", "hbond", "cavity", "buried")):
        if name.startswith("propka") or name.startswith("pqr"):
            return {"software": "PROPKA 3.5.1 + PDB2PQR 3.7.1", "library": "CLI", "software_version": "3.5.1 / 3.7.1"}
        if name.startswith("apbs"):
            return {"software": "APBS", "library": "CLI", "software_version": "3.4.1"}
        if name.startswith("invfold"):
            return {"software": "ESM-IF1", "library": "fair-esm", "software_version": "2.0.0"}
        if name.startswith("adv_") or "cavity" in name or "unsatisfied" in name or "salt" in name or "hbond" in name:
            return {"software": "custom Python (Stage4)", "library": "numpy/BioPython", "software_version": "repo"}
    if family.startswith("STRUCT_PACKING") or "contact" in name or "rg" in name:
        return {"software": "custom Python (Stage3 packing)", "library": "numpy", "software_version": "repo"}
    return base


def source_file(family: str, name: str) -> str:
    if name.startswith(("propka", "pqr", "apbs", "adv_", "invfold", "salt", "hbond", "cavity", "buried", "clash", "heavy", "packing_degree", "low_packed", "interface_contact", "vh_vl", "hydrophobic_interface", "aromatic_interface", "core_hydrophobic", "contact_density")):
        if "adv_" in name or name.startswith(("propka", "pqr", "apbs", "invfold", "cavity", "buried_unsatisfied")):
            return "virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py"
        return "virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py"
    if any(x in name for x in ["contact", "clash", "rg", "compactness", "n_ca", "nn_dist"]) and not name.startswith("adv"):
        if "interface_contact" in name or "VH_VL_center" in name:
            return "virtual_participant/stage3_structure/scripts/run_stage3.py + gate_b2/03_structure_features.py"
        return "virtual_participant/stage3_structure/scripts/run_stage3.py"
    return "gate_b2/scripts/03_structure_features.py"


def load_usage() -> dict:
    usage = {}
    for reg_path, stage in [(ST3_REG, "Stage3"), (ST4_REG, "Stage4")]:
        if not reg_path.exists():
            continue
        reg = pd.read_csv(reg_path)
        for _, r in reg.iterrows():
            eid = r["experiment_id"]
            fam = r.get("feature_family", "")
            tgt = r["target"]
            for c in reg.columns:
                pass
            bundle = fam.replace("STRUCT_", "STRUCT_") if fam else ""
            usage.setdefault(bundle, {"TmApp": set(), "HIC": set()}).setdefault(tgt, set()).add(eid)
    # parse experiment_id patterns from inventory
    if INV.exists():
        inv = pd.read_csv(INV_PATH := INV)
        for _, r in inv.iterrows():
            eid = str(r.get("experiment_id", ""))
            if "STRUCT_" in eid or "ADV_" in eid or "ESMFold" in eid:
                m = re.search(r"(STRUCT_[A-Z_]+|ADV_[A-Z_]+)", eid)
                if m:
                    bundle = m.group(1)
                    tgt = r.get("target", "")
                    if tgt in ("TmApp", "HIC"):
                        usage.setdefault(bundle, {"TmApp": set(), "HIC": set()}).setdefault(tgt, set()).add(eid)
    return usage


def bundle_used_in_final(bundle: str, target: str) -> bool:
    reps = FINAL_REP.get(target, [])
    key = bundle.replace("STRUCT_", "STRUCT_")
    for eid in reps:
        if key in eid or bundle in eid:
            return True
    if target == "HIC" and bundle in ("STRUCT_SURFACE_CHEM", "ADV_SURFACE_PATCH"):
        return True
    if target == "TmApp" and bundle in ("STRUCT_RASA", "ADV_INTERACTIONS", "ADV_TMAPP_ALL", "ADV_CAVITY", "ADV_UNSAT_POLAR", "ADV_INVFOLD"):
        return True
    return False


def build_stage3_rows(manifest: pd.DataFrame, usage: dict) -> list[dict]:
    esm = manifest[manifest.structure_source == "ESMFold"].copy()
    rows = []
    for i, r in esm.iterrows():
        fname = r["feature_name"]
        bundle = r["family"]
        sfam = BUNDLE_FAMILIES.get(bundle, "OTHER")
        chain, region = parse_region(fname)
        sw = software_for(bundle, fname)
        exps_tm = sorted(usage.get(bundle.replace("STRUCT_", "STRUCT_"), {}).get("TmApp", []))
        exps_hi = sorted(usage.get(bundle.replace("STRUCT_", "STRUCT_"), {}).get("HIC", []))
        used = bool(exps_tm or exps_hi)
        rows.append({
            "feature_id": f"ST3_ESM_{fname}",
            "feature_name": fname,
            "feature_family": sfam,
            "subfamily": bundle.replace("STRUCT_", ""),
            "bundle_name": bundle,
            "stage_first_introduced": "Stage3",
            "physical_quantity": raw_quantity(fname, bundle),
            "biological_interpretation": "antibody surface/geometry descriptor for developability",
            "dimensionality": "scalar",
            "unit": "Å² or dimensionless or count (mixed by column)",
            "sign_meaning": "context-dependent; higher SASA/RASA = more exposure",
            "structure_source": "ESMFold",
            "chain_scope": chain,
            "region_scope": region,
            "exact_definition": exact_definition(fname, bundle),
            "aggregation_method": "region sum/mean/median/max/count per gate_b2 aggregate() or surface_patches()",
            "normalization": "RASA uses MaxASA Tien2013; ratios divide by total SASA",
            "thresholds": f"RASA exposed >= {PARAMS['rasa_exposed_threshold']}; patch adjacency {PARAMS['patch_ca_adjacency_A']}Å",
            "cutoff_distance": str(PARAMS.get("interface_ca_cutoff_A", "")) if "interface" in fname.lower() else PARAMS["patch_ca_adjacency_A"],
            "neighborhood_radius": PARAMS["local_neighborhood_A"] if "local" in fname else "",
            "hydrophobicity_scale": PARAMS["hydrophobicity_scale"] if "hydrophobic" in fname.lower() or "rasa_w" in fname else "",
            "residue_classes": json.dumps(AA_SETS) if any(x in fname.lower() for x in ["hydrophobic", "aromatic", "positive", "negative", "polar"]) else "",
            "weighting_method": "RASA weight for chem columns; KD for hydrophobicity",
            "pH": "",
            "ionic_strength": "",
            "other_fixed_conditions": json.dumps({k: PARAMS[k] for k in ["probe_radius_A", "sasa_n_points"]}),
            "software": sw["software"],
            "software_version": sw["software_version"],
            "library": sw["library"],
            "implementation_function": "aggregate|surface_patches|interface_features|packing_from_cas",
            "source_file": source_file(bundle, fname),
            "source_lines_or_symbol": "see code_reference_index.csv",
            "command_or_API": "ShrakeRupley(probe_radius=1.4,n_points=100)",
            "preprocessing_required": "PDB parse; chain H/L map; region tag (IMGT or author segments)",
            "hydrogens_added": "no",
            "protonation_assigned": "no",
            "structure_relaxed": "no",
            "atom_completion": "CA for graph; full atoms for SASA",
            "numbering_or_region_mapping": "ESMFold: author segment lengths; ABB: IMGT resseq",
            "failure_handling": "missing PDB → skip; interface NaN → median impute in modeling",
            "computed": True,
            "saved_to_feature_table": True,
            "used_in_model": used,
            "target_TmApp_used": bool(exps_tm),
            "target_HIC_used": bool(exps_hi),
            "selected_in_best_model": bundle_used_in_final(bundle, "TmApp") or bundle_used_in_final(bundle, "HIC"),
            "representative_experiment_ids": ";".join((exps_tm[:3] + exps_hi[:3])[:5]),
            "artifact_path": "virtual_participant/stage3_structure/cache/features_ESMFold.csv",
            "manifest_path": str(ST3_MAN.relative_to(ROOT)),
            "code_evidence": source_file(bundle, fname),
            "notes": r.get("notes", ""),
            "quantity_tier": "AGGREGATED" if True else "RAW",
            "also_computed_ABBodyBuilder2": True,
        })
    return rows


def build_stage4_rows(manifest: pd.DataFrame, usage: dict) -> list[dict]:
    skip = {"apbs_ok", "apbs_error", "invfold_ok", "error"}
    rows = []
    for _, r in manifest.iterrows():
        fname = r["feature_name"]
        if fname in skip:
            continue
        bundle = r["family"]
        sfam = BUNDLE_FAMILIES.get(bundle, "OTHER")
        chain, region = parse_region(fname)
        sw = software_for(bundle, fname)
        exps_tm = sorted(usage.get(bundle, {}).get("TmApp", []))
        exps_hi = sorted(usage.get(bundle, {}).get("HIC", []))
        used = bool(exps_tm or exps_hi)
        rows.append({
            "feature_id": f"ST4_{fname}",
            "feature_name": fname,
            "feature_family": sfam,
            "subfamily": bundle.replace("ADV_", ""),
            "bundle_name": bundle,
            "stage_first_introduced": "Stage4",
            "physical_quantity": raw_quantity(fname, bundle),
            "biological_interpretation": "advanced structure biophysics for developability",
            "dimensionality": "scalar",
            "unit": "kT/e for APBS; charge units; log-lik for invfold; counts otherwise",
            "sign_meaning": "APBS: positive=repulsive for positive probe; charge: signed",
            "structure_source": "ESMFold",
            "chain_scope": chain,
            "region_scope": region,
            "exact_definition": exact_definition(fname, bundle),
            "aggregation_method": "Stage4 extract_one() per-antibody scalar extraction",
            "normalization": "fractions normalized by n_res or n_atoms where noted",
            "thresholds": "APBS strong ±1.0 kT/e; patch ±0.5 kT/e (reparse); RASA buried <0.2",
            "cutoff_distance": PARAMS.get("sb_dist_A") if "salt" in fname else PARAMS.get("hb_dist_A", ""),
            "neighborhood_radius": PARAMS["local_neighborhood_A"],
            "hydrophobicity_scale": PARAMS["hydrophobicity_scale"] if "hydrophobic" in fname or "adv_" in fname else "",
            "residue_classes": json.dumps(AA_SETS) if "hydrophobic" in fname or "adv_" in fname else "",
            "weighting_method": "local SASA weighting in adv_spatial_hydrophobicity",
            "pH": PARAMS["ph"] if bundle.startswith("ADV_P") or bundle == "ADV_ELECTROSTATICS" else "",
            "ionic_strength": PARAMS["ionic_strength_M"] if bundle == "ADV_ELECTROSTATICS" else "",
            "other_fixed_conditions": json.dumps({"apbs_pdie": 2.0, "apbs_sdie": 78.54, "apbs_dime": "97^3"}) if fname.startswith("apbs") else "",
            "software": sw["software"],
            "software_version": sw["software_version"],
            "library": sw["library"],
            "implementation_function": "extract_stage4_features.py functions",
            "source_file": source_file(bundle, fname),
            "source_lines_or_symbol": "see code_reference_index.csv",
            "command_or_API": "propka3|pdb2pqr|apbs|fair-esm score_sequence",
            "preprocessing_required": "ESMFold PDB; PROPKA→PDB2PQR→APBS pipeline",
            "hydrogens_added": "PDB2PQR adds hydrogens for PQR",
            "protonation_assigned": "PROPKA at pH 6.5",
            "structure_relaxed": "no",
            "atom_completion": "heavy atoms from PDB; PQR for electrostatics",
            "numbering_or_region_mapping": "chain A/B → H/L by sequence length match",
            "failure_handling": "per-tool try/except; apbs_ok/invfold_ok flags",
            "computed": True,
            "saved_to_feature_table": True,
            "used_in_model": used if bundle != "ADV_OTHER" else False,
            "target_TmApp_used": bool(exps_tm) if bundle != "ADV_OTHER" else False,
            "target_HIC_used": bool(exps_hi) if bundle != "ADV_OTHER" else False,
            "selected_in_best_model": bundle_used_in_final(bundle, "TmApp") or bundle_used_in_final(bundle, "HIC"),
            "representative_experiment_ids": ";".join((exps_tm[:3] + exps_hi[:3])[:5]),
            "artifact_path": "virtual_participant/stage4_advanced_structure/cache/features/stage4_all_features.csv",
            "manifest_path": str(ST4_MAN.relative_to(ROOT)),
            "code_evidence": "virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py",
            "notes": r.get("notes", "") + ("; computed but NOT in any Stage4 FAMILIES bundle" if bundle == "ADV_OTHER" else ""),
            "quantity_tier": "AGGREGATED",
            "also_computed_ABBodyBuilder2": False,
        })
    return rows


def build_lineage(inventory: pd.DataFrame) -> pd.DataFrame:
    templates = [
        ("ESMFold", "PDB parse + ShrakeRupley", "residue SASA", "per-residue SASA", "residue", "ShrakeRupley", "region mask", "none", "Fv_total_sasa", "STRUCT_SASA", "TmApp/HIC STRUCT_SASA experiments"),
        ("ESMFold", "PDB parse + ShrakeRupley", "residue SASA", "RASA=SASA/MaxASA", "residue", "divide by Tien2013", "region mask", "none", "Fv_mean_rasa", "STRUCT_RASA", "TmApp final STRUCT_RASA"),
        ("ESMFold", "RASA + AA class", "residue SASA", "class-filtered SASA sum", "residue", "sum", "region + AA set", "none", "Fv_sasa_hydrophobic", "STRUCT_SURFACE_CHEM", "HIC final blend component"),
        ("ESMFold", "RASA + KD scale", "residue RASA", "RASA×KD weighted sum", "residue", "weighted sum", "region", "KD scale", "Fv_rasa_w_hydrophobicity_sum", "STRUCT_SURFACE_CHEM", "used"),
        ("ESMFold", "exposed CA graph", "residue SASA + CA coords", "connected hydrophobic patches", "residue", "CA<=8Å union-find", "exposed RASA>=0.2", "none", "n_hydrophobic_patches", "STRUCT_PATCH", "simple patch"),
        ("ESMFold", "iso-SASA chains", "chain SASA", "BSA", "chain", "ShrakeRupley iso", "VH+VL", "none", "BSA", "STRUCT_INTERFACE", "used"),
        ("ESMFold", "CA coordinates", "CA-CA distances", "contact density/Rg", "residue", "distance count", "Fv/VH/VL", "none", "Fv_contact_density", "STRUCT_PACKING", "Stage3 packing"),
        ("ESMFold", "PROPKA CLI", "residue pKa", "Henderson-Hasselbalch charge", "residue", "charge sum at pH6.5", "whole Fv", "pH=6.5", "propka_net_charge", "ADV_PROPKA", "computed; bundle tested"),
        ("ESMFold", "PDB2PQR+APBS", "3D potential grid", "sample at exposed CA", "residue", "nearest DX grid", "exposed", "lpbe mg-auto", "apbs_mean_potential", "ADV_ELECTROSTATICS", "HIC bundle"),
        ("ESMFold", "APBS potential", "potential at exposed CA", "positive/negative patch CC", "residue", "CA<=8Å CC on thresholded", "exposed", "±0.5 kT/e", "apbs_pos_patch_count", "ADV_ELECTROSTATICS", "partial electrostatic topology"),
        ("ESMFold", "exposed CA + local SASA", "local neighborhood SASA", "advanced patch stats", "residue", "10Å sphere + 8Å CC", "whole Fv", "none", "adv_hydrophobic_patch_max_sasa", "ADV_SURFACE_PATCH", "HIC final component"),
        ("ESMFold", "heavy atom distances", "salt bridge pairs", "count within 4Å", "atom", "distance cutoff", "whole Fv", "4Å", "salt_bridge_count", "ADV_INTERACTIONS", "TmApp final component"),
        ("ESMFold", "CA neighbor degree", "buried low-degree sites", "cavity proxy count", "residue", "percentile threshold", "buried RASA<0.2", "none", "cavity_proxy_n_sites", "ADV_CAVITY", "PROXY not fpocket"),
        ("ESMFold", "buried polar CA", "N/O neighbor search", "unsatisfied polar proxy", "residue", "6Å CA shell", "buried polar/charged", "none", "buried_unsatisfied_polar_proxy", "ADV_UNSAT_POLAR", "PROXY not explicit H-bond net"),
        ("ESMFold", "ESM-IF1", "sequence given structure", "log-likelihood per chain", "chain", "score_sequence", "chains A,B", "none", "invfold_fv_mean_ll", "ADV_INVFOLD", "sequence consistency score not embedding"),
    ]
    cols = ["structure_source", "preprocessing_step", "intermediate_quantity", "raw_descriptor",
            "residue_or_atom_level", "spatial_operation", "region_aggregation", "normalization",
            "final_feature", "feature_bundle", "model_usage"]
    return pd.DataFrame(templates, columns=cols)


def build_families(inventory: pd.DataFrame) -> pd.DataFrame:
    fam_perf = {}
    if INV.exists():
        inv = pd.read_csv(INV)
        for fam in inventory["feature_family"].unique():
            sub = inv[inv["experiment_id"].astype(str).str.contains(fam.split("_")[0], case=False, na=False)]
            if sub.empty:
                sub = inv[inv["experiment_id"].astype(str).str.contains(inventory[inventory.feature_family == fam]["bundle_name"].iloc[0].replace("STRUCT_", ""), na=False)]
            if not sub.empty:
                fam_perf[fam] = {
                    "n_experiments": len(sub),
                    "median_Primary_CV_HIC": sub[sub.target == "HIC"]["Primary_CV_MAE"].median() if (sub.target == "HIC").any() else np.nan,
                    "median_Primary_CV_TmApp": sub[sub.target == "TmApp"]["Primary_CV_MAE"].median() if (sub.target == "TmApp").any() else np.nan,
                }
    rows = []
    for fam, g in inventory.groupby("feature_family"):
        pp = fam_perf.get(fam, {})
        rows.append({
            "feature_family": fam,
            "n_features": len(g),
            "n_bundles": g["bundle_name"].nunique(),
            "bundles": ";".join(sorted(g["bundle_name"].unique())),
            "structure_sources": ";".join(sorted(g["structure_source"].unique())),
            "stages": ";".join(sorted(g["stage_first_introduced"].unique())),
            "raw_quantities": ";".join(sorted(set(g["physical_quantity"].unique()))[:5]),
            "primary_software": ";".join(sorted(set(g["software"].unique()))),
            "computed": True,
            "used_in_any_model": bool(g["used_in_model"].any()),
            "TmApp_used": bool(g["target_TmApp_used"].any()),
            "HIC_used": bool(g["target_HIC_used"].any()),
            "in_final_representative": bool(g["selected_in_best_model"].any()),
            "performance_note": "supplementary only — not used for inventory filtering",
            "median_Primary_CV_HIC": pp.get("median_Primary_CV_HIC", np.nan),
            "median_Primary_CV_TmApp": pp.get("median_Primary_CV_TmApp", np.nan),
        })
    return pd.DataFrame(rows)


def build_overlap_matrix() -> tuple[pd.DataFrame, dict]:
    concepts = [
        "total_SASA", "RASA", "SURFACE_CHEM", "simple_PATCH", "ADV_SURFACE_PATCH",
        "PACKING", "INTERFACE", "ADV_INTERACTIONS", "UNSAT_POLAR", "cavity_proxy",
        "PROPKA", "APBS", "ESM-IF",
    ]
    n = len(concepts)
    mat = np.zeros((n, n), dtype=int)
    reasons = {}
    pairs = {
        ("total_SASA", "RASA"): (3, "same ShrakeRupley SASA; RASA is normalized SASA"),
        ("total_SASA", "SURFACE_CHEM"): (2, "surface chem uses same SASA filtered by AA class"),
        ("total_SASA", "simple_PATCH"): (2, "patches use exposed residue SASA sums"),
        ("RASA", "SURFACE_CHEM"): (2, "SURFACE_CHEM includes RASA-weighted chem"),
        ("SURFACE_CHEM", "simple_PATCH"): (2, "both use exposed hydrophobic/charge SASA"),
        ("simple_PATCH", "ADV_SURFACE_PATCH"): (2, "both CA-graph patches; ADV adds local 10Å SASA + compactness"),
        ("PACKING", "ADV_INTERACTIONS"): (2, "shared contact/clash/packing_degree logic"),
        ("PACKING", "cavity_proxy"): (2, "cavity proxy uses CA neighbor degree like packing"),
        ("INTERFACE", "ADV_INTERACTIONS"): (2, "interface contacts and BSA-related proxies overlap"),
        ("PROPKA", "APBS"): (2, "PDB2PQR uses PROPKA protonation for APBS input"),
        ("APBS", "ADV_SURFACE_PATCH"): (1, "both surface-related but different physics"),
        ("UNSAT_POLAR", "ADV_INTERACTIONS"): (1, "hbond_proxy shares polar atom distances but different aggregation"),
        ("ESM-IF", "PACKING"): (0, "sequence likelihood vs geometric contacts"),
        ("cavity_proxy", "UNSAT_POLAR"): (1, "both buried-site proxies; different criteria"),
    }
    idx = {c: i for i, c in enumerate(concepts)}
    for i in range(n):
        mat[i, i] = 3
    for (a, b), (score, reason) in pairs.items():
        i, j = idx[a], idx[b]
        mat[i, j] = mat[j, i] = score
        reasons[f"{a}|{b}"] = reason
    df = pd.DataFrame(mat, index=concepts, columns=concepts)
    return df, reasons


def not_computed_audit() -> list[dict]:
    candidates = [
        "detailed electrostatic surface topology", "dipole / multipole",
        "explicit pKa-shift descriptors", "pH titration curves",
        "OpenMM minimization", "force-field energy decomposition",
        "per-atom / per-residue forces", "relaxation ΔE", "minimization RMSD",
        "elastic network", "ANM", "GNM", "normal modes", "rigidity analysis",
        "MaSIF", "dMaSIF", "pretrained geometric GNN",
        "structure-aware pretrained embedding", "ML force field",
        "molecular dynamics", "conformer ensemble", "canonical SAP",
        "continuous molecular hydrophobic potential", "VH/VL orientation angle",
        "explicit cavity detection",
    ]
    status_map = {
        "detailed electrostatic surface topology": ("PARTIAL", "apbs_pos/neg_patch_count only; no full surface mesh topology"),
        "dipole / multipole": ("NOT_FOUND", "no repository implementation"),
        "explicit pKa-shift descriptors": ("PARTIAL", "propka_mean_pka aggregate only; no per-residue pKa distribution features"),
        "pH titration curves": ("NOT_FOUND", "fixed pH=6.5 only"),
        "OpenMM minimization": ("NOT_FOUND", "no OpenMM import or usage in repo"),
        "force-field energy decomposition": ("NOT_FOUND", ""),
        "per-atom / per-residue forces": ("NOT_FOUND", ""),
        "relaxation ΔE": ("NOT_FOUND", ""),
        "minimization RMSD": ("NOT_FOUND", ""),
        "elastic network": ("NOT_FOUND", ""),
        "ANM": ("NOT_FOUND", ""),
        "GNM": ("NOT_FOUND", ""),
        "normal modes": ("NOT_FOUND", ""),
        "rigidity analysis": ("NOT_FOUND", ""),
        "MaSIF": ("NOT_FOUND", ""),
        "dMaSIF": ("NOT_FOUND", ""),
        "pretrained geometric GNN": ("NOT_FOUND", ""),
        "structure-aware pretrained embedding": ("PARTIAL", "ESM-IF1 scalar log-likelihood only; no 3D GNN embedding stored"),
        "ML force field": ("NOT_FOUND", ""),
        "molecular dynamics": ("NOT_FOUND", ""),
        "conformer ensemble": ("NOT_FOUND", "single static ESMFold structure per antibody"),
        "canonical SAP": ("NOT_FOUND", "adv_spatial_hydrophobicity explicitly not SAP formula"),
        "continuous molecular hydrophobic potential": ("NOT_FOUND", "discrete KD-weighted SASA only"),
        "VH/VL orientation angle": ("NOT_FOUND", "vh_vl_center_dist only; no angle"),
        "explicit cavity detection": ("NOT_FOUND", "cavity_proxy is packing-derived; no fpocket/pyKVFinder"),
    }
    return [{"candidate": c, "status": status_map.get(c, ("NOT_FOUND", ""))[0],
             "evidence": status_map.get(c, ("NOT_FOUND", ""))[1]} for c in candidates]


def handoff_table(not_comp: list[dict]) -> list[dict]:
    return [
        {"Candidate concept": "OpenMM minimization", "Round1 status": "NOT_FOUND",
         "Closest Round1 feature": "ADV_INTERACTIONS",
         "What Round1 actually computed": "static geometric contacts/clashes only",
         "Remaining difference to investigate": "force-field relaxation/strain not computed"},
        {"Candidate concept": "pKa-shift descriptors", "Round1 status": "PARTIAL",
         "Closest Round1 feature": "ADV_PROPKA",
         "What Round1 actually computed": "propka_mean_pka, net charge at fixed pH; not residue-level pKa vector",
         "Remaining difference to investigate": "abnormal pKa distribution / burial shift not featurized"},
        {"Candidate concept": "electrostatic patch topology", "Round1 status": "PARTIAL",
         "Closest Round1 feature": "ADV_ELECTROSTATICS",
         "What Round1 actually computed": "APBS potential sampled at CA; CC patch count/size on ±0.5 kT/e",
         "Remaining difference to investigate": "full surface mesh topology / multipole not computed"},
        {"Candidate concept": "normal modes / elastic network", "Round1 status": "NOT_FOUND",
         "Closest Round1 feature": "STRUCT_PACKING / ADV_INTERACTIONS",
         "What Round1 actually computed": "static contact density and Rg only",
         "Remaining difference to investigate": "mechanical flexibility absent"},
        {"Candidate concept": "pretrained geometric DL", "Round1 status": "PARTIAL",
         "Closest Round1 feature": "ADV_INVFOLD",
         "What Round1 actually computed": "ESM-IF1 per-chain log-likelihood scalars",
         "Remaining difference to investigate": "no MaSIF/dMaSIF/GNN embedding"},
        {"Candidate concept": "canonical SAP", "Round1 status": "NOT_FOUND",
         "Closest Round1 feature": "ADV_SURFACE_PATCH",
         "What Round1 actually computed": "local hydrophobic SASA + CA patches; code comment: not SAP formula",
         "Remaining difference to investigate": "continuous hydrophobic potential / SAP weighting absent"},
        {"Candidate concept": "explicit cavity detection", "Round1 status": "NOT_FOUND",
         "Closest Round1 feature": "ADV_CAVITY",
         "What Round1 actually computed": "buried low CA degree count × sphere volume proxy",
         "Remaining difference to investigate": "fpocket/pyKVFinder-style void detection absent"},
        {"Candidate concept": "VH/VL orientation angle", "Round1 status": "NOT_FOUND",
         "Closest Round1 feature": "STRUCT_INTERFACE",
         "What Round1 actually computed": "VH_VL_center_dist and interface contacts only",
         "Remaining difference to investigate": "no dihedral/angle between domain axes"},
    ]


def write_report(families: pd.DataFrame, inventory: pd.DataFrame, lineage: pd.DataFrame,
                 overlap: pd.DataFrame, not_comp: list[dict], handoff: list[dict], summary: dict):
    md = ["# Round1 PDB由来特徴量 Inventory\n\n",
          "**状態:** `ROUND1_PDB_FEATURE_INVENTORY_AUDIT_COMPLETE`\n\n",
          "新規 training / Optuna / feature engineering / DeepResearch なし。repository 実装・artifact の監査のみ。\n\n",
          "**重要:** incremental signal による feature 選別は行っていない。computed / used / selected を区別。\n\n",
          "## 1. 目的\n\n",
          "Round1 Stage3/Stage4 で PDB（ESMFold 予測構造）から **実際に計算した物理量** を feature-level で棚卸しし、DeepResearch の baseline とする。\n\n",
          "## 2. 調査対象コード / artifact\n\n",
          "- `gate_b2/scripts/03_structure_features.py` — SASA/RASA/patch/interface\n",
          "- `virtual_participant/stage3_structure/scripts/run_stage3.py` — packing + bundles\n",
          "- `virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py` — PROPKA/APBS/patch/interactions\n",
          "- Manifests: `stage3_feature_manifest.csv`, `stage4_feature_manifest.csv`\n",
          "- Feature tables: `cache/features_ESMFold.csv`, `stage4_all_features.csv`\n\n",
          "## 3. 構造入力\n\n",
          "| Source | Path | Region mapping | Primary? |\n",
          "|--------|------|----------------|----------|\n",
          "| ESMFold native | `esmfold_native/{id}.pdb` | author segment lengths | **Yes** |\n",
          "| ABodyBuilder2 | `gate_b1/cache/structures/abodybuilder2/` | IMGT resseq | ablation only |\n\n",
          "Preprocessing: BioPython PDB parse; no relaxation; no OpenMM.\n\n",
          "### Success / failure coverage\n\n",
          "| Pipeline | Attempted | Successful | Failed | Notes |\n",
          "|----------|----------:|-------------:|-------:|-------|\n",
          "| Stage3 ESMFold features | 162 | 162 | 0 | gate_b2 + packing merged |\n",
          "| Stage3 ABodyBuilder2 features | 162 | 162 | 0 | ablation; same 218 cols |\n",
          "| Stage4 PROPKA | 162 | 162 | 0 | extraction_status.csv |\n",
          "| Stage4 PDB2PQR | 162 | 162 | 0 | extraction_status.csv |\n",
          "| Stage4 APBS | 162 | 162 | 0 | reparse_apbs_features.py |\n",
          "| Stage4 ESM-IF1 | 162 | 162 | 0 | invfold_ok per antibody |\n",
          "| Stage3 iface_frac (ESMFold) | — | 63 | 99 NaN | median impute in modeling |\n\n",
          "## 4. PDB-derived feature family 一覧\n\n"]
    md.append(families[["feature_family", "n_features", "bundles", "primary_software", "TmApp_used", "HIC_used"]].to_markdown(index=False))
    md.append("\n\n**Summary:** ")
    md.append(f"- Semantic families: **{summary['n_families']}**\n")
    md.append(f"- Semantic features (ESMFold primary): **{summary['n_features']}**\n")
    md.append(f"- Largest family: **{summary['largest_family']}** ({summary['largest_n']} features)\n\n")

    md.append("### SURFACE_CHEM exact summary\n\n")
    md.append("Per region (Fv, VH, VL, H_CDR1/2/3, all_CDR, FR) — 10 columns each:\n")
    md.append("1. `sasa_hydrophobic/aromatic/positive/negative/polar` — sum SASA where aa ∈ fixed class\n")
    md.append("2. `rasa_w_hydrophobicity_sum/mean` — Σ(RASA × Kyte-Doolittle KD)\n")
    md.append("3. `rasa_w_charge_sum` — Σ(RASA × charge); D/E=−1, K/R=+1, H=+0.1\n")
    md.append("4. `exposed_pos/neg` — Σ RASA for positive/negative aa\n")
    md.append("Plus global: `cdr_hydrophobic_sasa_exposed`, `h3_hydrophobic_sasa_exposed`.\n\n")
    md.append("### Simple PATCH vs ADV_SURFACE_PATCH\n\n")
    md.append("| Aspect | STRUCT_PATCH (Stage3) | ADV_SURFACE_PATCH (Stage4) |\n")
    md.append("|--------|----------------------|----------------------------|\n")
    md.append("| Graph | Exposed CA, edge if ≤8Å | Exposed CA, edge if ≤8Å |\n")
    md.append("| Patch | Union-find on hydrophobic/charge/aromatic | CC + local 10Å neighborhood SASA |\n")
    md.append("| Extra | CDR/H3 hydrophobic exposed SASA | compactness, extent, mixed arom-hydro patches |\n")
    md.append("| SAP | No | Code: **not canonical SAP formula** |\n\n")

    sections = [
        ("5. Global geometry", "GLOBAL_GEOMETRY", "STRUCT_GLOBAL: n_res, mean_plddt, Fv_buried_frac"),
        ("6. SASA / RASA", "SASA_RASA", "ShrakeRupley probe=1.4Å; RASA=SASA/MaxASA(Tien2013); thresholds 0.20/0.25/0.50/1.0"),
        ("7. Surface chemistry", "SURFACE_CHEMISTRY", "AA-class SASA sums + RASA×KD + RASA×charge; 7 regions × ~10 cols"),
        ("8. Surface patch", "SURFACE_PATCH_SIMPLE", "Stage3: exposed CA graph 8Å CC; 5 global patch cols + CDR/H3 hydrophobic exposed SASA"),
        ("9. Packing / contacts", "PACKING_CONTACT", "Stage3 CA<8Å contacts, Rg, compactness; Stage4 heavy contacts/clash/packing_degree"),
        ("10. VH–VL interface", "VH_VL_INTERFACE", "BSA, n_interface_res (CA<=5Å), iface composition; no orientation angle"),
        ("11. H-bond / salt bridge / polar", "PACKING_CONTACT", "salt_bridge 4Å; hbond_proxy 3.5Å N-O; no angle criterion"),
        ("12. Cavity / buried-unsatisfied-polar", "CAVITY_PROXY", "cavity_proxy: buried + low CA degree; UNSAT: buried polar + no N/O within 6Å — **both PROXY**"),
        ("13. Electrostatics", "ELECTROSTATICS", "APBS lpbe mg-auto; 14 potential stats + patch CC on exposed CA"),
        ("14. pKa / protonation", "PKA_PROTONATION", "PROPKA 9 cols + PQR 6 charge cols; no residue-level pKa vector in ML"),
        ("15. Energy / force / mechanics", "NOT_COMPUTED", "OpenMM/MM energy/normal modes: **NOT_FOUND**"),
        ("16. Inverse folding", "INVERSE_FOLDING", "ESM-IF1 log-likelihood per chain + Fv mean; **scalar score not embedding**"),
        ("17. Structure confidence", "GLOBAL_GEOMETRY", "mean_plddt, mean_confidence from ESMFold B-factor"),
    ]
    for title, fam, desc in sections:
        md.append(f"## {title}\n\n{desc}\n\n")
        sub = families[families.feature_family == fam] if fam != "NOT_COMPUTED" else pd.DataFrame()
        if not sub.empty:
            md.append(sub.to_markdown(index=False))
            md.append("\n\n")

    md.append("## 18. ESMFold vs ABodyBuilder2 differences\n\n")
    md.append("- Same pipeline code; ABB uses IMGT CDR numbering, ESMFold uses author segment lengths.\n")
    md.append("- ESMFold: L_CDR1/2/3_empty flags; ABB has full L-CDR region features.\n")
    md.append("- Stage4 advanced features: **ESMFold only** (162/162).\n\n")

    md.append("## 19. Feature lineage\n\n")
    md.append("See `round1_pdb_feature_lineage.csv` (" + str(len(lineage)) + " traced paths).\n\n")

    md.append("## 20. Round1内のsemantic overlap\n\n")
    md.append(overlap.to_markdown())
    md.append("\n\n")

    md.append("## 21. Round1で計算していない候補群\n\n")
    md.append("| Candidate | Status | Evidence |\n|-----------|--------|----------|\n")
    for r in not_comp:
        md.append(f"| {r['candidate']} | {r['status']} | {r['evidence']} |\n")
    md.append("\n")

    md.append("## 22. DeepResearch handoff\n\n")
    md.append("| Candidate concept | Round1 status | Closest Round1 feature | What Round1 actually computed | Remaining difference to investigate |\n")
    md.append("|-------------------|---------------|--------------------------|-------------------------------|-------------------------------------|\n")
    for r in handoff:
        md.append(f"| {r['Candidate concept']} | {r['Round1 status']} | {r['Closest Round1 feature']} | {r['What Round1 actually computed']} | {r['Remaining difference to investigate']} |\n")

    md.append("\n\n## Q1–Q20 回答要約\n\n")
    qa = summary["qa"]
    for q, a in qa.items():
        md.append(f"- **{q}** {a}\n")

    md.append("\n---\n\n**状態:** `ROUND1_PDB_FEATURE_INVENTORY_AUDIT_COMPLETE`\n")
    (OUT / "ROUND1_PDB_FEATURE_INVENTORY_JA.md").write_text("".join(md))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    usage = load_usage()
    st3 = pd.read_csv(ST3_MAN)
    st4 = pd.read_csv(ST4_MAN)
    rows = build_stage3_rows(st3, usage) + build_stage4_rows(st4, usage)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUT / "round1_pdb_feature_inventory.csv", index=False)

    families = build_families(inventory)
    families.to_csv(OUT / "round1_pdb_feature_families.csv", index=False)

    lineage = build_lineage(inventory)
    lineage.to_csv(OUT / "round1_pdb_feature_lineage.csv", index=False)

    overlap, overlap_reasons = build_overlap_matrix()
    overlap.to_csv(OUT / "round1_pdb_feature_overlap_matrix.csv")
    pd.DataFrame([{"pair": k, "score": v, "overlap_reason": overlap_reasons[k]} for k, v in overlap_reasons.items()]).to_csv(
        OUT / "round1_pdb_feature_overlap_reasons.csv", index=False)

    not_comp = not_computed_audit()
    handoff = handoff_table(not_comp)

    # raw column lists
    st3_cols = st3[st3.structure_source == "ESMFold"]["feature_name"].tolist()
    st4_cols = st4[~st4.feature_name.isin({"apbs_ok", "apbs_error", "invfold_ok", "error"})]["feature_name"].tolist()
    (OUT / "raw_feature_columns/stage3_esmfold_columns.txt").write_text("\n".join(st3_cols))
    (OUT / "raw_feature_columns/stage4_columns.txt").write_text("\n".join(st4_cols))

    code_refs = pd.DataFrame([
        {"symbol": "residue_rows", "file": "gate_b2/scripts/03_structure_features.py", "purpose": "ShrakeRupley SASA/RASA per residue"},
        {"symbol": "aggregate", "file": "gate_b2/scripts/03_structure_features.py", "purpose": "region SASA/RASA/chem aggregation"},
        {"symbol": "surface_patches", "file": "gate_b2/scripts/03_structure_features.py", "purpose": "simple exposed CA patch CC"},
        {"symbol": "interface_features", "file": "gate_b2/scripts/03_structure_features.py", "purpose": "BSA + interface residue composition"},
        {"symbol": "packing_from_cas", "file": "virtual_participant/stage3_structure/scripts/run_stage3.py", "purpose": "CA contact density Rg compactness"},
        {"symbol": "advanced_patch_features", "file": "virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py", "purpose": "local 10Å SASA + advanced patch CC"},
        {"symbol": "run_apbs_features", "file": "virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py", "purpose": "APBS potential sample + patch CC"},
        {"symbol": "parse_propka_features", "file": "virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py", "purpose": "PROPKA charge aggregates"},
        {"symbol": "cavity_and_unsat_features", "file": "virtual_participant/stage4_advanced_structure/scripts/extract_stage4_features.py", "purpose": "cavity/unsat PROXY"},
        {"symbol": "score_invfold_batch", "file": "virtual_participant/stage4_advanced_structure/cache/score_invfold_batch.py", "purpose": "ESM-IF1 log-likelihood"},
    ])
    code_refs.to_csv(OUT / "code_reference_index.csv", index=False)

    largest = families.sort_values("n_features", ascending=False).iloc[0]
    qa = {
        "Q1 SASA/RASA features?": f"{len(st3_cols)} Stage3 cols: SASA(20)+RASA(64)+ratios; see inventory",
        "Q2 SURFACE_CHEM exactly?": "Per-region sum SASA by AA class + RASA×KD + RASA×charge + exposed_pos/neg",
        "Q3 simple vs ADV patch?": "Simple: global exposed CA CC 8Å on hydrophobic/charge. ADV: local 10Å SASA + CC + compactness/extent + mixed patches",
        "Q4 patch uses surface geometry?": "Yes: CA Euclidean graph on exposed residues; not pure sequence aggregation",
        "Q5 PACKING?": "CA contact count/density, Rg, compactness, clash proxy; Stage4 adds heavy-atom contacts",
        "Q6 cavity explicit?": "NO — packing-derived proxy (buried + low CA degree)",
        "Q7 UNSAT_POLAR explicit?": "NO — proxy (buried polar + zero N/O within 6Å)",
        "Q8 PROPKA residue pKa as feature?": "PARTIAL — aggregates only (mean_pka, net_charge); not per-residue pKa vector",
        "Q9 APBS statistics?": "14 cols: mean/median/std/quantiles/abs mean, pos/neg/strong frac, pos/neg patch count/max",
        "Q10 electrostatic patch topology?": "PARTIAL — APBS CC patch count/size on thresholded exposed CA only",
        "Q11 dipole/multipole?": "NOT_FOUND",
        "Q12 OpenMM/MM energy?": "NOT_FOUND",
        "Q13 energy minimization?": "NOT_FOUND",
        "Q14 normal modes/ENM?": "NOT_FOUND",
        "Q15 ESM-IF features?": "Per-chain ll/nll/len + fv mean/var/worst — sequence consistency scalars",
        "Q16 geometric DL embedding?": "NOT_FOUND (ESM-IF scalars only)",
        "Q17 canonical SAP?": "NOT_FOUND — code explicitly disclaims SAP formula",
        "Q18 ESMFold vs ABB?": "Stage3 both; Stage4 ESMFold only",
        "Q19 definition differs by source?": "Region mapping differs (IMGT vs author segments); same formulas",
        "Q20 DeepResearch overlap?": "See overlap matrix + handoff table",
    }

    summary = {
        "n_families": len(families),
        "n_features": len(inventory),
        "largest_family": largest["feature_family"],
        "largest_n": int(largest["n_features"]),
        "qa": qa,
    }

    machine = {
        "state": "ROUND1_PDB_FEATURE_INVENTORY_AUDIT_COMPLETE",
        "n_semantic_features_esmfold_primary": len(inventory),
        "n_feature_families": len(families),
        "stage3_features_esmfold": len(st3_cols),
        "stage4_features": len(st4_cols),
        "parameters": PARAMS,
        "not_computed": not_comp,
        "overlap_reasons": overlap_reasons,
        "handoff": handoff,
        "summary": summary,
    }
    (OUT / "ROUND1_PDB_FEATURE_INVENTORY_MACHINE.json").write_text(json.dumps(machine, indent=2, ensure_ascii=False))

    write_report(families, inventory, lineage, overlap, not_comp, handoff, summary)

    print("ROUND1_PDB_FEATURE_INVENTORY_AUDIT_COMPLETE")
    print("families:", len(families), "features:", len(inventory))
    print("largest:", largest["feature_family"], int(largest["n_features"]))


if __name__ == "__main__":
    main()
