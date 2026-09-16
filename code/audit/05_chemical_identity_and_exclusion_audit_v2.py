#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter
import json
import re
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/BigSolDBv2.0.csv"
SPLIT_DIR = ROOT / "data/processed_splits"
OUT = ROOT / "results/chemical_identity_audit"
OUT.mkdir(parents=True, exist_ok=True)

SOLUTE = "SMILES_Solute"
SOLVENT = "SMILES_Solvent"
SOLVENT_NAME = "Solvent"
LOGS = "LogS(mol/L)"
SOL = "Solubility(mol/L)"
TEMP = "Temperature_K"
COMPOUND = "Compound_Name"
CAS = "CAS"
CID = "PubChem_CID"

BASE_COLS = [SOLUTE, TEMP, SOLVENT_NAME, SOLVENT, SOL, LOGS, COMPOUND, CAS, CID]

SPLITS = {
    "random": (
        "train_random_maccs_map4_padel_sen_features.csv",
        "test_random_maccs_map4_padel_sen_features.csv",
        "overlap_allowed",
    ),
    "unseen_pair": (
        "train_unseen_pair_maccs_map4_padel_sen_features.csv",
        "test_unseen_pair_maccs_map4_padel_sen_features.csv",
        "no_pair_overlap",
    ),
    "strict_unseen_solute": (
        "train_strict_unseen_solute_maccs_map4_padel_sen_features.csv",
        "test_strict_unseen_solute_maccs_map4_padel_sen_features.csv",
        "no_solute_overlap",
    ),
    "strict_unseen_solvent": (
        "train_strict_unseen_solvent_maccs_map4_padel_sen_features.csv",
        "test_strict_unseen_solvent_maccs_map4_padel_sen_features.csv",
        "no_solvent_overlap",
    ),
    "fully_unseen_solute_solvent": (
        "train_strict_unseen_solute_solvent_maccs_map4_padel_sen_features.csv",
        "test_strict_unseen_solute_solvent_maccs_map4_padel_sen_features.csv",
        "no_solute_or_solvent_overlap",
    ),
}

BAD_SMILES = {"", "-", "nan", "NaN", "None", "none", "NULL", "null", "NA", "N/A"}

POLYMER_MIXTURE_REGEX = re.compile(
    r"(PEG|PEG[-\s]*400|polyethylene\s*glycol|macrogol|polymer|polymeric|mixture|mixed|blend)",
    flags=re.IGNORECASE,
)

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def read_base_csv(path):
    header = pd.read_csv(path, nrows=0, low_memory=False)
    header.columns = header.columns.astype(str).str.strip()
    available = [c for c in BASE_COLS if c in header.columns]
    df = pd.read_csv(path, usecols=available, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df

def smiles_info(s):
    s = clean(s)
    if s in BAD_SMILES:
        return {
            "valid": False,
            "canonical": "",
            "canonical_nonisomeric": "",
            "inchikey": "",
            "n_fragments": np.nan,
            "disconnected": False,
        }

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return {
            "valid": False,
            "canonical": "",
            "canonical_nonisomeric": "",
            "inchikey": "",
            "n_fragments": np.nan,
            "disconnected": "." in s,
        }

    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        can = ""

    try:
        can_noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        can_noniso = ""

    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        ik = ""

    try:
        nfrag = len(Chem.GetMolFrags(mol))
    except Exception:
        nfrag = np.nan

    return {
        "valid": True,
        "canonical": can,
        "canonical_nonisomeric": can_noniso,
        "inchikey": ik,
        "n_fragments": nfrag,
        "disconnected": bool(nfrag > 1) if not pd.isna(nfrag) else "." in s,
    }

def add_identity(df):
    df = df.copy()

    for c in [SOLUTE, SOLVENT, SOLVENT_NAME, COMPOUND, CAS, CID]:
        if c in df.columns:
            df[c] = df[c].map(clean)

    for c in [LOGS, SOL, TEMP]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    solute_info = df[SOLUTE].map(smiles_info)
    solvent_info = df[SOLVENT].map(smiles_info)

    for role, info in [("solute", solute_info), ("solvent", solvent_info)]:
        df[f"{role}_valid_smiles"] = info.map(lambda d: d["valid"])
        df[f"{role}_canonical_smiles"] = info.map(lambda d: d["canonical"])
        df[f"{role}_canonical_smiles_nonisomeric"] = info.map(lambda d: d["canonical_nonisomeric"])
        df[f"{role}_inchikey"] = info.map(lambda d: d["inchikey"])
        df[f"{role}_n_fragments"] = info.map(lambda d: d["n_fragments"])
        df[f"{role}_disconnected"] = info.map(lambda d: d["disconnected"])

    df["raw_pair"] = df[SOLUTE].map(clean) + "||" + df[SOLVENT].map(clean)
    df["canonical_pair"] = df["solute_canonical_smiles"] + "||" + df["solvent_canonical_smiles"]
    df["inchikey_pair"] = df["solute_inchikey"] + "||" + df["solvent_inchikey"]

    return df

def multiset_excluded(raw, model_ready):
    key_cols = [SOLUTE, TEMP, SOLVENT_NAME, SOLVENT, SOL, LOGS, COMPOUND, CAS, CID]
    key_cols = [c for c in key_cols if c in raw.columns and c in model_ready.columns]

    def norm_series(df, col):
        if col in [TEMP, SOL, LOGS]:
            return pd.to_numeric(df[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{float(x):.12g}")
        return df[col].map(clean)

    def make_key(df):
        key = norm_series(df, key_cols[0])
        for c in key_cols[1:]:
            key = key + "||" + norm_series(df, c)
        return key

    raw_key = make_key(raw)
    model_key = make_key(model_ready)
    counter = Counter(model_key.tolist())

    excluded_mask = []
    for k in raw_key.tolist():
        if counter[k] > 0:
            counter[k] -= 1
            excluded_mask.append(False)
        else:
            excluded_mask.append(True)

    return raw.loc[excluded_mask].copy()

def exclusion_reason_table(raw, model_ready):
    excluded = multiset_excluded(raw, model_ready)
    excluded = add_identity(excluded)

    reasons = []

    for _, r in excluded.iterrows():
        row_reasons = []

        if pd.isna(r.get(LOGS, np.nan)):
            row_reasons.append("missing_or_invalid_LogS")
        if pd.isna(r.get(SOL, np.nan)):
            row_reasons.append("missing_or_invalid_solubility_mol_L")
        elif r.get(SOL, np.nan) <= 0:
            row_reasons.append("nonpositive_solubility_mol_L")

        if not r.get("solute_valid_smiles", False):
            row_reasons.append("missing_or_invalid_solute_smiles")
        if not r.get("solvent_valid_smiles", False):
            row_reasons.append("missing_or_invalid_solvent_smiles")

        if not row_reasons:
            row_reasons.append("other_model_readiness_or_descriptor_preprocessing_difference")

        reasons.append(";".join(row_reasons))

    excluded["exclusion_reason"] = reasons
    excluded.to_csv(OUT / "excluded_records_with_reasons.csv", index=False)

    counter = Counter()
    for r in reasons:
        for item in r.split(";"):
            counter[item] += 1

    table = pd.DataFrame(
        [{"exclusion_category": k, "n_records": v} for k, v in counter.most_common()]
    )

    table.to_csv(OUT / "exclusion_accounting_table.csv", index=False)

    summary = pd.DataFrame([{
        "initial_raw_records": int(len(raw)),
        "model_ready_records": int(len(model_ready)),
        "excluded_records_by_multiset_difference": int(len(excluded)),
        "raw_records_with_missing_LogS": int(pd.to_numeric(raw[LOGS], errors="coerce").isna().sum()),
        "raw_records_with_missing_solubility_mol_L": int(pd.to_numeric(raw[SOL], errors="coerce").isna().sum()),
        "raw_records_with_dash_solvent_smiles": int((raw[SOLVENT].map(clean) == "-").sum()),
    }])
    summary.to_csv(OUT / "exclusion_accounting_summary.csv", index=False)

    return table, summary, excluded

def component_summary(df):
    df = add_identity(df)

    rows = []
    for role, raw_col in [("solute", SOLUTE), ("solvent", SOLVENT)]:
        valid = df[df[f"{role}_valid_smiles"] == True]
        alias = (
            valid.groupby(f"{role}_canonical_smiles")[raw_col]
            .nunique()
            .reset_index(name="n_raw_smiles")
        )

        rows.append({
            "component_role": role,
            "rows": int(len(df)),
            "unique_raw_smiles": int(df[raw_col].map(clean).nunique()),
            "valid_smiles_rows": int((df[f"{role}_valid_smiles"] == True).sum()),
            "invalid_or_missing_smiles_rows": int((df[f"{role}_valid_smiles"] != True).sum()),
            "unique_canonical_smiles": int(valid[f"{role}_canonical_smiles"].replace("", np.nan).dropna().nunique()),
            "unique_inchikey": int(valid[f"{role}_inchikey"].replace("", np.nan).dropna().nunique()),
            "components_with_multiple_raw_smiles_same_canonical": int((alias["n_raw_smiles"] > 1).sum()) if len(alias) else 0,
            "max_raw_smiles_per_canonical": int(alias["n_raw_smiles"].max()) if len(alias) else 0,
            "disconnected_structure_rows": int(valid[f"{role}_disconnected"].fillna(False).sum()),
            "unique_disconnected_canonical_components": int(valid.loc[valid[f"{role}_disconnected"].fillna(False), f"{role}_canonical_smiles"].nunique()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "chemical_identity_audit_summary.csv", index=False)

    alias_rows = []
    for role, raw_col in [("solute", SOLUTE), ("solvent", SOLVENT)]:
        valid = df[df[f"{role}_valid_smiles"] == True]
        for can, g in valid.groupby(f"{role}_canonical_smiles"):
            raws = sorted(set(g[raw_col].map(clean)))
            if len(raws) > 1:
                alias_rows.append({
                    "component_role": role,
                    "canonical_smiles": can,
                    "n_raw_smiles": len(raws),
                    "raw_smiles_examples": " | ".join(raws[:10]),
                    "n_rows": int(len(g)),
                })

    pd.DataFrame(alias_rows).to_csv(OUT / "smiles_aliases_by_canonical_identity.csv", index=False)
    return out

def split_leakage_table():
    rows = []

    for split, (train_name, test_name, rule) in SPLITS.items():
        train = add_identity(read_base_csv(SPLIT_DIR / train_name))
        test = add_identity(read_base_csv(SPLIT_DIR / test_name))

        def ov(col):
            a = set(train[col].dropna().astype(str)) - {""}
            b = set(test[col].dropna().astype(str)) - {""}
            return a & b

        rows.append({
            "split": split,
            "expected_rule": rule,

            "raw_shared_solutes": len(ov(SOLUTE)),
            "raw_shared_solvents": len(ov(SOLVENT)),
            "raw_shared_pairs": len(ov("raw_pair")),

            "canonical_shared_solutes": len(ov("solute_canonical_smiles")),
            "canonical_shared_solvents": len(ov("solvent_canonical_smiles")),
            "canonical_shared_pairs": len(ov("canonical_pair")),

            "inchikey_shared_solutes": len(ov("solute_inchikey")),
            "inchikey_shared_solvents": len(ov("solvent_inchikey")),
            "inchikey_shared_pairs": len(ov("inchikey_pair")),

            "canonical_pair_leakage_passed": "" if split == "random" else len(ov("canonical_pair")) == 0,
            "canonical_solute_exclusion_passed": "" if split not in ["strict_unseen_solute", "fully_unseen_solute_solvent"] else len(ov("solute_canonical_smiles")) == 0,
            "canonical_solvent_exclusion_passed": "" if split not in ["strict_unseen_solvent", "fully_unseen_solute_solvent"] else len(ov("solvent_canonical_smiles")) == 0,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "canonical_identity_leakage_audit.csv", index=False)
    return out

def solvent_one_to_one(raw, model_ready):
    df = add_identity(model_ready)

    name_to_can = df.groupby(SOLVENT_NAME)["solvent_canonical_smiles"].nunique()
    can_to_name = df.groupby("solvent_canonical_smiles")[SOLVENT_NAME].nunique()

    ambiguities = []
    for name, n in name_to_can.items():
        if n > 1:
            ambiguities.append({
                "ambiguity_type": "one_solvent_name_multiple_structures",
                "identifier": name,
                "n_counterparts": int(n),
            })
    for can, n in can_to_name.items():
        if n > 1:
            ambiguities.append({
                "ambiguity_type": "one_structure_multiple_solvent_names",
                "identifier": can,
                "n_counterparts": int(n),
            })

    pd.DataFrame(ambiguities).to_csv(OUT / "solvent_identity_ambiguities.csv", index=False)

    out = pd.DataFrame([{
        "unique_solvent_names_model_ready": int(df[SOLVENT_NAME].nunique()),
        "unique_raw_solvent_smiles_model_ready": int(df[SOLVENT].nunique()),
        "unique_canonical_solvent_smiles_model_ready": int(df["solvent_canonical_smiles"].replace("", np.nan).dropna().nunique()),
        "names_with_multiple_canonical_structures": int((name_to_can > 1).sum()),
        "canonical_structures_with_multiple_names": int((can_to_name > 1).sum()),
        "one_to_one_name_structure": bool((name_to_can > 1).sum() == 0 and (can_to_name > 1).sum() == 0),
    }])
    out.to_csv(OUT / "solvent_identity_one_to_one_audit.csv", index=False)
    return out

def polymer_mixture_salt_like(raw, model_ready):
    df = add_identity(model_ready)

    rows = []
    for role, smiles_col, name_col in [
        ("solute", SOLUTE, COMPOUND),
        ("solvent", SOLVENT, SOLVENT_NAME),
    ]:
        name_flag = df[name_col].map(clean).map(lambda x: bool(POLYMER_MIXTURE_REGEX.search(x)))
        dot_flag = df[f"{role}_disconnected"].fillna(False)
        mask = name_flag | dot_flag

        sub = df.loc[mask, [smiles_col, name_col, f"{role}_canonical_smiles", f"{role}_n_fragments", f"{role}_disconnected"]].copy()
        sub.insert(0, "component_role", role)
        sub["flag_polymer_or_mixture_like_name"] = name_flag[mask].values
        sub["flag_disconnected_or_salt_like_smiles"] = dot_flag[mask].values
        sub = sub.rename(columns={
            smiles_col: "raw_smiles",
            name_col: "name",
            f"{role}_canonical_smiles": "canonical_smiles",
            f"{role}_n_fragments": "n_fragments",
            f"{role}_disconnected": "disconnected",
        })
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(OUT / "polymer_mixture_salt_like_component_audit.csv", index=False)

    summary = (
        out.groupby(["component_role", "flag_polymer_or_mixture_like_name", "flag_disconnected_or_salt_like_smiles"])
        .size()
        .reset_index(name="n_rows")
        if len(out) else pd.DataFrame()
    )
    summary.to_csv(OUT / "polymer_mixture_salt_like_summary.csv", index=False)

    return out, summary

def main():
    raw = read_base_csv(RAW_PATH)

    train_random = read_base_csv(SPLIT_DIR / "train_random_maccs_map4_padel_sen_features.csv")
    test_random = read_base_csv(SPLIT_DIR / "test_random_maccs_map4_padel_sen_features.csv")
    model_ready = pd.concat([train_random, test_random], ignore_index=True)

    exclusion_table, exclusion_summary, excluded = exclusion_reason_table(raw, model_ready)
    identity_summary = component_summary(model_ready)
    leakage = split_leakage_table()
    solvent_audit = solvent_one_to_one(raw, model_ready)
    poly_table, poly_summary = polymer_mixture_salt_like(raw, model_ready)

    latex = []
    latex.append("% Exclusion accounting summary\n")
    latex.append(exclusion_summary.to_latex(index=False, escape=False))
    latex.append("\n% Exclusion accounting table\n")
    latex.append(exclusion_table.to_latex(index=False, escape=False))
    latex.append("\n% Chemical identity audit summary\n")
    latex.append(identity_summary.to_latex(index=False, escape=False))
    latex.append("\n% Canonical identity leakage audit\n")
    latex.append(leakage.to_latex(index=False, escape=False))
    latex.append("\n% Solvent identity one-to-one audit\n")
    latex.append(solvent_audit.to_latex(index=False, escape=False))
    latex.append("\n% Polymer / mixture / salt-like summary\n")
    if len(poly_summary):
        latex.append(poly_summary.to_latex(index=False, escape=False))
    else:
        latex.append("% No flagged polymer/mixture/salt-like rows.\n")

    (OUT / "chemical_identity_audit_latex_tables.tex").write_text("\n".join(latex), encoding="utf-8")

    report = {
        "raw_path": str(RAW_PATH),
        "model_ready_basis": "train_random + test_random processed split files",
        "outputs": {
            "exclusion_accounting_summary": "results/chemical_identity_audit/exclusion_accounting_summary.csv",
            "exclusion_accounting_table": "results/chemical_identity_audit/exclusion_accounting_table.csv",
            "excluded_records_with_reasons": "results/chemical_identity_audit/excluded_records_with_reasons.csv",
            "chemical_identity_audit_summary": "results/chemical_identity_audit/chemical_identity_audit_summary.csv",
            "canonical_identity_leakage_audit": "results/chemical_identity_audit/canonical_identity_leakage_audit.csv",
            "solvent_identity_one_to_one_audit": "results/chemical_identity_audit/solvent_identity_one_to_one_audit.csv",
            "solvent_identity_ambiguities": "results/chemical_identity_audit/solvent_identity_ambiguities.csv",
            "polymer_mixture_salt_like_component_audit": "results/chemical_identity_audit/polymer_mixture_salt_like_component_audit.csv",
            "polymer_mixture_salt_like_summary": "results/chemical_identity_audit/polymer_mixture_salt_like_summary.csv",
            "smiles_aliases_by_canonical_identity": "results/chemical_identity_audit/smiles_aliases_by_canonical_identity.csv",
            "latex_tables": "results/chemical_identity_audit/chemical_identity_audit_latex_tables.tex",
        },
        "interpretation_note": (
            "The original split identifiers were the SMILES strings present in the BigSolDB-derived modeling file. "
            "This audit adds post hoc RDKit canonical-SMILES and InChIKey checks where possible. "
            "Successful descriptor calculation is treated as computational usability, not proof of chemical identity."
        ),
    }

    (OUT / "chemical_identity_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== EXCLUSION ACCOUNTING SUMMARY ===")
    print(exclusion_summary.to_string(index=False))

    print("\n=== EXCLUSION ACCOUNTING TABLE ===")
    print(exclusion_table.to_string(index=False))

    print("\n=== CHEMICAL IDENTITY SUMMARY ===")
    print(identity_summary.to_string(index=False))

    print("\n=== CANONICAL / INCHIKEY LEAKAGE AUDIT ===")
    print(leakage.to_string(index=False))

    print("\n=== SOLVENT ONE-TO-ONE AUDIT ===")
    print(solvent_audit.to_string(index=False))

    print("\n=== POLYMER / MIXTURE / SALT-LIKE SUMMARY ===")
    print(poly_summary.to_string(index=False) if len(poly_summary) else "No flagged rows.")

    print("\nSaved outputs to:", OUT)

if __name__ == "__main__":
    main()
