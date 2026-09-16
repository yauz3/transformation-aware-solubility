#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict
import json
import re
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except Exception as e:
    Chem = None
    RDKIT_IMPORT_ERROR = str(e)
else:
    RDKIT_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "BigSolDBv2.0.csv"
SPLIT_DIR = ROOT / "data" / "processed_splits"
OUT = ROOT / "results" / "chemical_identity_audit"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

SPLITS = {
    "random": {
        "train": SPLIT_DIR / "train_random_maccs_map4_padel_sen_features.csv",
        "test": SPLIT_DIR / "test_random_maccs_map4_padel_sen_features.csv",
        "expected": "overlap_allowed",
    },
    "unseen_pair": {
        "train": SPLIT_DIR / "train_unseen_pair_maccs_map4_padel_sen_features.csv",
        "test": SPLIT_DIR / "test_unseen_pair_maccs_map4_padel_sen_features.csv",
        "expected": "no_pair_overlap",
    },
    "strict_unseen_solute": {
        "train": SPLIT_DIR / "train_strict_unseen_solute_maccs_map4_padel_sen_features.csv",
        "test": SPLIT_DIR / "test_strict_unseen_solute_maccs_map4_padel_sen_features.csv",
        "expected": "no_solute_overlap",
    },
    "strict_unseen_solvent": {
        "train": SPLIT_DIR / "train_strict_unseen_solvent_maccs_map4_padel_sen_features.csv",
        "test": SPLIT_DIR / "test_strict_unseen_solvent_maccs_map4_padel_sen_features.csv",
        "expected": "no_solvent_overlap",
    },
    "fully_unseen_solute_solvent": {
        "train": SPLIT_DIR / "train_strict_unseen_solute_solvent_maccs_map4_padel_sen_features.csv",
        "test": SPLIT_DIR / "test_strict_unseen_solute_solvent_maccs_map4_padel_sen_features.csv",
        "expected": "no_solute_or_solvent_overlap",
    },
}

CANDIDATES = {
    "solute_smiles": [
        "SMILES_Solute", "Solute_SMILES", "solute_smiles", "solute SMILES",
        "solute_SMILES", "solute_smile", "SoluteSMILES"
    ],
    "solvent_smiles": [
        "SMILES_Solvent", "Solvent_SMILES", "solvent_smiles", "solvent SMILES",
        "solvent_SMILES", "solvent_smile", "SolventSMILES"
    ],
    "logs": [
        "LogS(mol/L)", "LogS", "logS", "logs", "logS(mol/L)"
    ],
    "solubility": [
        "Solubility(mol/L)", "Solubility", "solubility", "S(mol/L)"
    ],
    "temperature": [
        "Temperature_K", "Temperature", "temperature", "T", "Temp_K"
    ],
    "solute_name": [
        "Solute", "Solute_Name", "solute_name", "Name_Solute", "compound_name_solute",
        "Compound_Name_Solute", "Solute name"
    ],
    "solvent_name": [
        "Solvent", "Solvent_Name", "solvent_name", "Name_Solvent", "Solvent name"
    ],
    "solute_cid": [
        "CID_Solute", "PubChemCID_Solute", "PubChem_CID_Solute", "solute_cid",
        "PubChem CID Solute"
    ],
    "solvent_cid": [
        "CID_Solvent", "PubChemCID_Solvent", "PubChem_CID_Solvent", "solvent_cid",
        "PubChem CID Solvent"
    ],
    "solute_cas": [
        "CAS_Solute", "CASRN_Solute", "solute_cas", "CAS Solute"
    ],
    "solvent_cas": [
        "CAS_Solvent", "CASRN_Solvent", "solvent_cas", "CAS Solvent"
    ],
}

POLYMER_MIXTURE_REGEX = re.compile(
    r"(PEG|polyethylene\s*glycol|polymer|polymeric|macrogol|mixture|mixed|blend|PEG[-\s]*400|400)",
    flags=re.IGNORECASE
)

def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize_columns(pd.read_csv(path, low_memory=False))

def pick_col(df, key, required=False):
    for c in CANDIDATES[key]:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not identify required column for {key}. Available columns include: {list(df.columns)[:40]}")
    return None

def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "na", "n/a", ""}:
        return ""
    return s

def normalize_number(x):
    if pd.isna(x):
        return ""
    try:
        v = float(x)
        if not np.isfinite(v):
            return ""
        return f"{v:.12g}"
    except Exception:
        return clean_str(x)

def make_signature(df, cols):
    parts = []
    for c in cols:
        if c not in df.columns:
            continue
        if c.lower().find("temperature") >= 0 or "logs" in c.lower() or "solubility" in c.lower():
            parts.append(df[c].map(normalize_number))
        else:
            parts.append(df[c].map(clean_str))
    if not parts:
        raise RuntimeError("No columns available for row signature.")
    sig = parts[0]
    for p in parts[1:]:
        sig = sig + "||" + p
    return sig

def parse_smiles(s):
    s = clean_str(s)
    if not s or Chem is None:
        return {
            "input": s,
            "valid": False if not s else None,
            "canonical_isomeric": "",
            "canonical_nonisomeric": "",
            "inchi_key": "",
            "n_fragments": np.nan,
            "disconnected": False,
        }

    try:
        mol = Chem.MolFromSmiles(s)
    except Exception:
        mol = None

    if mol is None:
        return {
            "input": s,
            "valid": False,
            "canonical_isomeric": "",
            "canonical_nonisomeric": "",
            "inchi_key": "",
            "n_fragments": np.nan,
            "disconnected": "." in s,
        }

    try:
        can_iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        can_iso = ""

    try:
        can_noniso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        can_noniso = ""

    try:
        inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        inchi_key = ""

    try:
        n_frag = len(Chem.GetMolFrags(mol))
    except Exception:
        n_frag = np.nan

    return {
        "input": s,
        "valid": True,
        "canonical_isomeric": can_iso,
        "canonical_nonisomeric": can_noniso,
        "inchi_key": inchi_key,
        "n_fragments": n_frag,
        "disconnected": bool(n_frag > 1) if not pd.isna(n_frag) else ("." in s),
    }

def add_identity_columns(df, solute_col, solvent_col):
    df = df.copy()

    solute_parsed = df[solute_col].map(parse_smiles)
    solvent_parsed = df[solvent_col].map(parse_smiles)

    for prefix, parsed in [("solute", solute_parsed), ("solvent", solvent_parsed)]:
        df[f"{prefix}_smiles_valid"] = parsed.map(lambda d: d["valid"])
        df[f"{prefix}_canonical_smiles"] = parsed.map(lambda d: d["canonical_isomeric"])
        df[f"{prefix}_canonical_smiles_nonisomeric"] = parsed.map(lambda d: d["canonical_nonisomeric"])
        df[f"{prefix}_inchi_key"] = parsed.map(lambda d: d["inchi_key"])
        df[f"{prefix}_n_fragments"] = parsed.map(lambda d: d["n_fragments"])
        df[f"{prefix}_disconnected"] = parsed.map(lambda d: d["disconnected"])

    df["canonical_pair"] = df["solute_canonical_smiles"] + "||" + df["solvent_canonical_smiles"]
    df["inchikey_pair"] = df["solute_inchi_key"] + "||" + df["solvent_inchi_key"]

    return df

def component_identity_summary(df, role, raw_col):
    valid_col = f"{role}_smiles_valid"
    can_col = f"{role}_canonical_smiles"
    inchi_col = f"{role}_inchi_key"
    frag_col = f"{role}_n_fragments"
    disc_col = f"{role}_disconnected"

    valid_df = df[df[valid_col] == True].copy()

    alias_counts = (
        valid_df.groupby(can_col)[raw_col]
        .nunique()
        .reset_index(name="n_raw_smiles_for_canonical")
    )

    n_alias_components = int((alias_counts["n_raw_smiles_for_canonical"] > 1).sum())
    max_alias = int(alias_counts["n_raw_smiles_for_canonical"].max()) if len(alias_counts) else 0

    return {
        "component_role": role,
        "rows": int(len(df)),
        "unique_raw_smiles": int(df[raw_col].map(clean_str).nunique()),
        "valid_smiles_rows": int((df[valid_col] == True).sum()),
        "invalid_or_missing_smiles_rows": int((df[valid_col] != True).sum()),
        "unique_canonical_smiles": int(valid_df[can_col].nunique()),
        "unique_inchikey": int(valid_df[inchi_col].replace("", np.nan).dropna().nunique()),
        "components_with_multiple_raw_smiles_same_canonical": n_alias_components,
        "max_raw_smiles_per_canonical": max_alias,
        "disconnected_structure_rows": int(valid_df[disc_col].fillna(False).sum()),
        "unique_disconnected_canonical_components": int(valid_df.loc[valid_df[disc_col].fillna(False), can_col].nunique()),
    }

def alias_table(df, role, raw_col):
    can_col = f"{role}_canonical_smiles"
    valid_col = f"{role}_smiles_valid"

    valid_df = df[df[valid_col] == True].copy()
    rows = []
    for can, g in valid_df.groupby(can_col):
        raws = sorted(set(clean_str(x) for x in g[raw_col] if clean_str(x)))
        if len(raws) > 1:
            rows.append({
                "component_role": role,
                "canonical_smiles": can,
                "n_raw_smiles": len(raws),
                "raw_smiles_examples": " | ".join(raws[:10]),
                "n_rows": int(len(g)),
            })
    return pd.DataFrame(rows)

def leakage_for_split(split_name, cfg, solute_col, solvent_col):
    train = add_identity_columns(read_csv(cfg["train"]), solute_col, solvent_col)
    test = add_identity_columns(read_csv(cfg["test"]), solute_col, solvent_col)

    def overlap(col):
        a = set(train[col].dropna().astype(str)) - {""}
        b = set(test[col].dropna().astype(str)) - {""}
        return a & b

    raw_solute_overlap = overlap(solute_col)
    raw_solvent_overlap = overlap(solvent_col)
    raw_pair_train = set((train[solute_col].map(clean_str) + "||" + train[solvent_col].map(clean_str)).dropna())
    raw_pair_test = set((test[solute_col].map(clean_str) + "||" + test[solvent_col].map(clean_str)).dropna())
    raw_pair_overlap = raw_pair_train & raw_pair_test

    can_solute_overlap = overlap("solute_canonical_smiles")
    can_solvent_overlap = overlap("solvent_canonical_smiles")
    can_pair_overlap = overlap("canonical_pair")

    inchi_solute_overlap = overlap("solute_inchi_key")
    inchi_solvent_overlap = overlap("solvent_inchi_key")
    inchi_pair_overlap = overlap("inchikey_pair")

    return {
        "split": split_name,
        "expected_rule": cfg["expected"],
        "raw_shared_solutes": int(len(raw_solute_overlap)),
        "raw_shared_solvents": int(len(raw_solvent_overlap)),
        "raw_shared_pairs": int(len(raw_pair_overlap)),
        "canonical_shared_solutes": int(len(can_solute_overlap)),
        "canonical_shared_solvents": int(len(can_solvent_overlap)),
        "canonical_shared_pairs": int(len(can_pair_overlap)),
        "inchikey_shared_solutes": int(len(inchi_solute_overlap)),
        "inchikey_shared_solvents": int(len(inchi_solvent_overlap)),
        "inchikey_shared_pairs": int(len(inchi_pair_overlap)),
        "canonical_pair_leakage_passed": bool(len(can_pair_overlap) == 0) if split_name != "random" else "",
        "canonical_solute_exclusion_passed": bool(len(can_solute_overlap) == 0) if split_name in {"strict_unseen_solute", "fully_unseen_solute_solvent"} else "",
        "canonical_solvent_exclusion_passed": bool(len(can_solvent_overlap) == 0) if split_name in {"strict_unseen_solvent", "fully_unseen_solute_solvent"} else "",
        "inchi_available": bool(train["solute_inchi_key"].replace("", np.nan).notna().any()),
    }

def solvent_one_to_one_audit(df, solvent_smiles_col, solvent_name_col):
    valid = add_identity_columns(df, pick_col(df, "solute_smiles", required=True), solvent_smiles_col)

    if solvent_name_col is None:
        return pd.DataFrame([{
            "audit": "solvent_name_to_structure",
            "solvent_name_column": "",
            "unique_solvent_names": np.nan,
            "unique_raw_solvent_smiles": int(valid[solvent_smiles_col].map(clean_str).nunique()),
            "unique_canonical_solvent_smiles": int(valid["solvent_canonical_smiles"].replace("", np.nan).dropna().nunique()),
            "names_with_multiple_canonical_structures": np.nan,
            "canonical_structures_with_multiple_names": np.nan,
            "one_to_one_name_structure": "not_assessable_no_solvent_name_column_detected",
        }])

    tmp = valid[[solvent_name_col, solvent_smiles_col, "solvent_canonical_smiles"]].copy()
    tmp[solvent_name_col] = tmp[solvent_name_col].map(clean_str)
    tmp = tmp[tmp[solvent_name_col] != ""]
    tmp = tmp[tmp["solvent_canonical_smiles"] != ""]

    name_to_can = tmp.groupby(solvent_name_col)["solvent_canonical_smiles"].nunique()
    can_to_name = tmp.groupby("solvent_canonical_smiles")[solvent_name_col].nunique()

    one_to_one = bool(
        int((name_to_can > 1).sum()) == 0
        and int((can_to_name > 1).sum()) == 0
        and int(tmp[solvent_name_col].nunique()) == int(tmp["solvent_canonical_smiles"].nunique())
    )

    detailed = []
    for name, n in name_to_can.items():
        if n > 1:
            detailed.append({
                "type": "one_name_multiple_structures",
                "name_or_structure": name,
                "n_counterparts": int(n),
            })
    for can, n in can_to_name.items():
        if n > 1:
            detailed.append({
                "type": "one_structure_multiple_names",
                "name_or_structure": can,
                "n_counterparts": int(n),
            })

    pd.DataFrame(detailed).to_csv(OUT / "solvent_identity_ambiguities.csv", index=False)

    return pd.DataFrame([{
        "audit": "solvent_name_to_structure",
        "solvent_name_column": solvent_name_col,
        "unique_solvent_names": int(tmp[solvent_name_col].nunique()),
        "unique_raw_solvent_smiles": int(valid[solvent_smiles_col].map(clean_str).nunique()),
        "unique_canonical_solvent_smiles": int(valid["solvent_canonical_smiles"].replace("", np.nan).dropna().nunique()),
        "names_with_multiple_canonical_structures": int((name_to_can > 1).sum()),
        "canonical_structures_with_multiple_names": int((can_to_name > 1).sum()),
        "one_to_one_name_structure": one_to_one,
    }])

def polymer_mixture_audit(df, solute_col, solvent_col, solute_name_col, solvent_name_col):
    tmp = add_identity_columns(df, solute_col, solvent_col)

    rows = []

    for role, smiles_col, name_col in [
        ("solute", solute_col, solute_name_col),
        ("solvent", solvent_col, solvent_name_col),
    ]:
        name_series = tmp[name_col].map(clean_str) if name_col and name_col in tmp.columns else pd.Series([""] * len(tmp))
        smiles_series = tmp[smiles_col].map(clean_str)

        mask_name = name_series.map(lambda x: bool(POLYMER_MIXTURE_REGEX.search(x)))
        mask_dot = tmp[f"{role}_disconnected"].fillna(False)
        mask = mask_name | mask_dot

        sub = tmp.loc[mask, [smiles_col, f"{role}_canonical_smiles", f"{role}_n_fragments", f"{role}_disconnected"]].copy()
        sub["component_role"] = role
        sub["name"] = name_series[mask].values
        sub["flag_polymer_or_mixture_like_name"] = mask_name[mask].values
        sub["flag_disconnected_or_salt_like_smiles"] = mask_dot[mask].values
        rows.append(sub.rename(columns={smiles_col: "raw_smiles", f"{role}_canonical_smiles": "canonical_smiles", f"{role}_n_fragments": "n_fragments"}))

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=[
            "raw_smiles", "canonical_smiles", "n_fragments", "component_role", "name",
            "flag_polymer_or_mixture_like_name", "flag_disconnected_or_salt_like_smiles"
        ])

    return out

def exclusion_accounting(raw, model_ready, cols, solute_col=None, solvent_col=None, logs_col=None, sol_col=None):
    raw_sig = make_signature(raw, cols)
    model_sig = make_signature(model_ready, cols)

    model_counter = Counter(model_sig.tolist())

    excluded_mask = []
    for s in raw_sig.tolist():
        if model_counter[s] > 0:
            model_counter[s] -= 1
            excluded_mask.append(False)
        else:
            excluded_mask.append(True)

    excluded = raw.loc[excluded_mask].copy()

    # When raw/model tables are renamed to sig_0, sig_1, ... for multiset matching,
    # the original SMILES column names are no longer present. Therefore, the caller
    # can pass explicit signature-column names for exclusion-reason accounting.
    if solute_col is None:
        solute_col = pick_col(raw, "solute_smiles", required=True)
    if solvent_col is None:
        solvent_col = pick_col(raw, "solvent_smiles", required=True)
    if logs_col is None:
        logs_col = pick_col(raw, "logs", required=False)
    if sol_col is None:
        sol_col = pick_col(raw, "solubility", required=False)

    reasons = []
    for _, row in excluded.iterrows():
        row_reasons = []

        solute = clean_str(row.get(solute_col, ""))
        solvent = clean_str(row.get(solvent_col, ""))

        if not solute:
            row_reasons.append("missing_solute_smiles")
        elif Chem is not None and Chem.MolFromSmiles(solute) is None:
            row_reasons.append("invalid_solute_smiles")

        if not solvent:
            row_reasons.append("missing_solvent_smiles")
        elif Chem is not None and Chem.MolFromSmiles(solvent) is None:
            row_reasons.append("invalid_solvent_smiles")

        if logs_col:
            v = pd.to_numeric(pd.Series([row.get(logs_col)]), errors="coerce").iloc[0]
            if pd.isna(v):
                row_reasons.append("missing_or_invalid_LogS")

        if sol_col:
            v = pd.to_numeric(pd.Series([row.get(sol_col)]), errors="coerce").iloc[0]
            if pd.isna(v):
                row_reasons.append("missing_or_invalid_solubility_mol_L")
            elif v <= 0:
                row_reasons.append("nonpositive_solubility_mol_L")

        if not row_reasons:
            row_reasons.append("not_in_model_ready_after_structure_target_checks_or_descriptor_preprocessing")

        reasons.append(";".join(row_reasons))

    excluded["exclusion_reason"] = reasons

    reason_counter = Counter()
    for r in reasons:
        for item in r.split(";"):
            reason_counter[item] += 1

    reason_rows = []
    for reason, n in reason_counter.most_common():
        reason_rows.append({
            "exclusion_category": reason,
            "n_records": int(n),
        })

    summary = pd.DataFrame(reason_rows)
    excluded.to_csv(OUT / "excluded_records_with_reasons.csv", index=False)
    summary.to_csv(OUT / "exclusion_accounting_table.csv", index=False)

    return excluded, summary

def main():
    if Chem is None:
        print("WARNING: RDKit could not be imported:", RDKIT_IMPORT_ERROR)
    else:
        print("RDKit loaded successfully.")

    raw = read_csv(RAW_PATH)

    random_train = read_csv(SPLITS["random"]["train"])
    random_test = read_csv(SPLITS["random"]["test"])
    model_ready = pd.concat([random_train, random_test], ignore_index=True)

    raw_solute_col = pick_col(raw, "solute_smiles", required=True)
    raw_solvent_col = pick_col(raw, "solvent_smiles", required=True)
    raw_logs_col = pick_col(raw, "logs", required=False)
    raw_sol_col = pick_col(raw, "solubility", required=False)
    raw_temp_col = pick_col(raw, "temperature", required=False)

    model_solute_col = pick_col(model_ready, "solute_smiles", required=True)
    model_solvent_col = pick_col(model_ready, "solvent_smiles", required=True)
    model_logs_col = pick_col(model_ready, "logs", required=False)
    model_sol_col = pick_col(model_ready, "solubility", required=False)
    model_temp_col = pick_col(model_ready, "temperature", required=False)

    raw_name_solute = pick_col(raw, "solute_name", required=False)
    raw_name_solvent = pick_col(raw, "solvent_name", required=False)
    raw_cid_solute = pick_col(raw, "solute_cid", required=False)
    raw_cid_solvent = pick_col(raw, "solvent_cid", required=False)
    raw_cas_solute = pick_col(raw, "solute_cas", required=False)
    raw_cas_solvent = pick_col(raw, "solvent_cas", required=False)

    signature_cols_raw = []
    signature_cols_model = []

    for r_col, m_col in [
        (raw_solute_col, model_solute_col),
        (raw_solvent_col, model_solvent_col),
        (raw_temp_col, model_temp_col),
        (raw_logs_col, model_logs_col),
        (raw_sol_col, model_sol_col),
    ]:
        if r_col and m_col:
            signature_cols_raw.append(r_col)
            signature_cols_model.append(m_col)

    raw_for_sig = raw.rename(columns={r: f"sig_{i}" for i, r in enumerate(signature_cols_raw)})
    model_for_sig = model_ready.rename(columns={m: f"sig_{i}" for i, m in enumerate(signature_cols_model)})
    sig_cols = [f"sig_{i}" for i in range(len(signature_cols_raw))]

    sig_role_map = {f"sig_{i}": original for i, original in enumerate(signature_cols_raw)}

    excluded, exclusion_summary = exclusion_accounting(
        raw_for_sig,
        model_for_sig,
        sig_cols,
        solute_col="sig_0" if len(sig_cols) > 0 else None,
        solvent_col="sig_1" if len(sig_cols) > 1 else None,
        logs_col=next((k for k, v in sig_role_map.items() if v == raw_logs_col), None),
        sol_col=next((k for k, v in sig_role_map.items() if v == raw_sol_col), None),
    )

    model_id = add_identity_columns(model_ready, model_solute_col, model_solvent_col)

    identity_summary = pd.DataFrame([
        component_identity_summary(model_id, "solute", model_solute_col),
        component_identity_summary(model_id, "solvent", model_solvent_col),
    ])
    identity_summary.to_csv(OUT / "chemical_identity_audit_summary.csv", index=False)

    aliases = pd.concat([
        alias_table(model_id, "solute", model_solute_col),
        alias_table(model_id, "solvent", model_solvent_col),
    ], ignore_index=True)
    aliases.to_csv(OUT / "smiles_aliases_by_canonical_identity.csv", index=False)

    leakage_rows = []
    for split_name, cfg in SPLITS.items():
        leakage_rows.append(leakage_for_split(split_name, cfg, model_solute_col, model_solvent_col))
    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(OUT / "canonical_identity_leakage_audit.csv", index=False)

    solvent_audit = solvent_one_to_one_audit(raw, raw_solvent_col, raw_name_solvent)
    solvent_audit.to_csv(OUT / "solvent_identity_one_to_one_audit.csv", index=False)

    polymer_mix = polymer_mixture_audit(raw, raw_solute_col, raw_solvent_col, raw_name_solute, raw_name_solvent)
    polymer_mix.to_csv(OUT / "polymer_mixture_salt_like_component_audit.csv", index=False)

    latex_parts = []

    latex_parts.append("% Exclusion accounting table\n")
    latex_parts.append(exclusion_summary.to_latex(index=False, escape=False))

    latex_parts.append("\n% Chemical identity summary\n")
    latex_parts.append(identity_summary.to_latex(index=False, escape=False))

    latex_parts.append("\n% Canonical identity leakage audit\n")
    latex_parts.append(leakage.to_latex(index=False, escape=False))

    latex_parts.append("\n% Solvent one-to-one audit\n")
    latex_parts.append(solvent_audit.to_latex(index=False, escape=False))

    (OUT / "chemical_identity_audit_latex_tables.tex").write_text("\n".join(latex_parts), encoding="utf-8")

    report = {
        "raw_path": str(RAW_PATH),
        "model_ready_basis": "random train + random test split files",
        "raw_records": int(len(raw)),
        "model_ready_records": int(len(model_ready)),
        "excluded_records_by_multiset_signature": int(len(excluded)),
        "rdkit_available": Chem is not None,
        "rdkit_import_error": RDKIT_IMPORT_ERROR,
        "detected_columns": {
            "raw_solute_smiles": raw_solute_col,
            "raw_solvent_smiles": raw_solvent_col,
            "raw_logs": raw_logs_col,
            "raw_solubility": raw_sol_col,
            "raw_temperature": raw_temp_col,
            "raw_solute_name": raw_name_solute,
            "raw_solvent_name": raw_name_solvent,
            "raw_solute_cid": raw_cid_solute,
            "raw_solvent_cid": raw_cid_solvent,
            "raw_solute_cas": raw_cas_solute,
            "raw_solvent_cas": raw_cas_solvent,
            "model_solute_smiles": model_solute_col,
            "model_solvent_smiles": model_solvent_col,
        },
        "outputs": {
            "exclusion_accounting_table": "results/chemical_identity_audit/exclusion_accounting_table.csv",
            "excluded_records_with_reasons": "results/chemical_identity_audit/excluded_records_with_reasons.csv",
            "chemical_identity_audit_summary": "results/chemical_identity_audit/chemical_identity_audit_summary.csv",
            "canonical_identity_leakage_audit": "results/chemical_identity_audit/canonical_identity_leakage_audit.csv",
            "solvent_identity_one_to_one_audit": "results/chemical_identity_audit/solvent_identity_one_to_one_audit.csv",
            "polymer_mixture_salt_like_component_audit": "results/chemical_identity_audit/polymer_mixture_salt_like_component_audit.csv",
            "smiles_aliases_by_canonical_identity": "results/chemical_identity_audit/smiles_aliases_by_canonical_identity.csv",
            "latex_tables": "results/chemical_identity_audit/chemical_identity_audit_latex_tables.tex",
        },
        "interpretation_notes": {
            "no_record_removal_reconciliation": "Repeated and duplicate-like observations were retained and not collapsed. However, model-ready filtering excluded records that lacked usable structure/target information or were absent after descriptor/preprocessing requirements.",
            "canonicalization": "The original split identifiers were the SMILES strings present in the BigSolDB-derived modeling file. This audit adds post hoc RDKit canonical-SMILES and InChIKey checks where possible.",
            "descriptor_success": "Successful descriptor generation demonstrates computational usability but is not treated as proof of identity integrity; identity integrity is assessed separately by canonical-SMILES/InChIKey audits.",
        }
    }
    (OUT / "chemical_identity_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== EXCLUSION ACCOUNTING ===")
    print(exclusion_summary.to_string(index=False))

    print("\n=== CHEMICAL IDENTITY SUMMARY ===")
    print(identity_summary.to_string(index=False))

    print("\n=== CANONICAL / INCHIKEY LEAKAGE AUDIT ===")
    print(leakage.to_string(index=False))

    print("\n=== SOLVENT ONE-TO-ONE AUDIT ===")
    print(solvent_audit.to_string(index=False))

    print("\n=== POLYMER / MIXTURE / SALT-LIKE AUDIT ===")
    print(polymer_mix.head(40).to_string(index=False))
    print(f"\nTotal flagged polymer/mixture/salt-like rows: {len(polymer_mix)}")

    print("\nSaved outputs to:", OUT)

if __name__ == "__main__":
    main()
