#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute solute + solvent MACCS fingerprints (RDKit MACCS, 167 bits).

Output: original columns + solute_maccs_0..166 + solvent_maccs_0..166
"""

import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.error")
warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
INPUT_CSV = "test_group-based.csv"
OUTPUT_CSV = "test_group-based_maccs.csv"

SOLUTE_COL = "SMILES_Solute"
SOLVENT_COL = "SMILES_Solvent"

MACCS_LEN = 167  # fixed for RDKit MACCS


# =========================
# Helpers
# =========================
def safe_mol(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)

def fp_to_np(fp, n, dtype=np.int8):
    arr = np.zeros((n,), dtype=dtype)
    ConvertToNumpyArray(fp, arr)
    return arr

def compute_maccs_bits(mol):
    if mol is None:
        return np.zeros(MACCS_LEN, dtype=np.int8)
    fp = MACCSkeys.GenMACCSKeys(mol)   # 167-bit MACCS
    return fp_to_np(fp, MACCS_LEN, dtype=np.int8)

def build_maccs_side(df: pd.DataFrame, smiles_col: str, prefix: str) -> pd.DataFrame:
    smiles_list = df[smiles_col].fillna("").astype(str).tolist()
    mols = [safe_mol(s) for s in smiles_list]

    bits = np.vstack([compute_maccs_bits(m) for m in mols])
    cols = [f"{prefix}_maccs_{i}" for i in range(MACCS_LEN)]
    return pd.DataFrame(bits, columns=cols)


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    for c in [SOLUTE_COL, SOLVENT_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    print("✅ Computing SOLUTE MACCS (167 bits)...")
    sol_feat = build_maccs_side(df, SOLUTE_COL, "solute")

    print("✅ Computing SOLVENT MACCS (167 bits)...")
    sv_feat = build_maccs_side(df, SOLVENT_COL, "solvent")

    out = pd.concat(
        [df.reset_index(drop=True), sol_feat.reset_index(drop=True), sv_feat.reset_index(drop=True)],
        axis=1
    )
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Saved: {OUTPUT_CSV}")
    print(f"Rows: {out.shape[0]}  Cols: {out.shape[1]}")
    print("Notes:")
    print("- RDKit MACCS is fixed-length 167 bits.")
    print("- Invalid/empty SMILES -> all-zeros bit vector for that row.")

if __name__ == "__main__":
    main()
