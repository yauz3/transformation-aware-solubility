#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6) MAP4 fingerprint (robust fixed-length)

Problem fixed:
- Some MAP4 packages/versions return variable-length token lists instead of fixed-size minhash signature.
- We convert ANY MAP4 output into a fixed-length vector using feature hashing (count sketch).

Install:
    pip install map4-ojmb pandas numpy
    (RDKit should already exist in your env)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")

try:
    from map4 import MAP4Calculator
except Exception as e:
    raise ImportError("map4 not available. Install with: pip install map4-ojmb") from e


# =========================
# Config
# =========================
INPUT_CSV  = "train_group-based_maccs.csv"
OUTPUT_CSV = "train_group-based_maccs_map4.csv"

SOLUTE_COL  = "SMILES_Solute"
SOLVENT_COL = "SMILES_Solvent"

MAP4_DIM = 1024   # final fixed vector length


# =========================
# Helpers
# =========================
def safe_mol(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)


def to_fixed_map4_vector(map4_out, dim: int) -> np.ndarray:
    """
    Convert MAP4 output (whatever shape/type) into fixed-length vector of size=dim.

    Handles:
    - already fixed length (len == dim) -> returns as int32
    - variable-length token/hash list -> hashed count vector (count sketch)
    - dict/sparse-like -> also hashed into dim
    """
    if map4_out is None:
        return np.zeros(dim, dtype=np.int32)

    # If it's dict-like: keys are tokens/hashes
    if isinstance(map4_out, dict):
        vec = np.zeros(dim, dtype=np.int32)
        for k, v in map4_out.items():
            try:
                kk = int(k)
                vv = int(v) if v is not None else 1
            except Exception:
                kk = hash(str(k))
                vv = 1
            vec[kk % dim] += vv
        return vec

    # Convert to 1D array if possible
    try:
        arr = np.asarray(map4_out).ravel()
    except Exception:
        arr = np.array([map4_out], dtype=object)

    # If it's numeric and already fixed size -> use directly
    if arr.size == dim and np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.int32)

    # Otherwise treat as tokens (variable-length)
    vec = np.zeros(dim, dtype=np.int32)

    # If numeric array but not fixed size: treat entries as tokens
    for t in arr:
        try:
            tok = int(t)
        except Exception:
            tok = hash(str(t))
        vec[tok % dim] += 1

    return vec


def compute_map4_cache(unique_smiles, map4_calc: "MAP4Calculator", dim: int):
    """
    Returns dict: smiles -> fixed-length np.ndarray shape (dim,)
    """
    cache = {}

    mols = []
    smiles_valid = []
    for s in unique_smiles:
        m = safe_mol(s)
        if m is None:
            cache[s] = np.zeros(dim, dtype=np.int32)
        else:
            mols.append(m)
            smiles_valid.append(s)

    if mols:
        outs = map4_calc.calculate_many(mols)
        for s, out in zip(smiles_valid, outs):
            cache[s] = to_fixed_map4_vector(out, dim)

    return cache


def side_df_from_cache(smiles_series: pd.Series, cache: dict, prefix: str, dim: int) -> pd.DataFrame:
    X = np.vstack([cache[s] for s in smiles_series.tolist()]).astype(np.int32)
    cols = [f"{prefix}_map4_{i}" for i in range(dim)]
    return pd.DataFrame(X, columns=cols)


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    for c in [SOLUTE_COL, SOLVENT_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    solute_smiles  = df[SOLUTE_COL].fillna("").astype(str)
    solvent_smiles = df[SOLVENT_COL].fillna("").astype(str)

    uniq = sorted(set(solute_smiles.tolist() + solvent_smiles.tolist()))

    print(f"✅ Unique SMILES: {len(uniq)}")
    print("✅ Building MAP4 calculator...")
    map4_calc = MAP4Calculator(dimensions=MAP4_DIM)

    print("✅ Computing MAP4 (cached, fixed-length via hashing)...")
    cache = compute_map4_cache(uniq, map4_calc, MAP4_DIM)

    print("✅ Expanding to dataset rows...")
    sol_df = side_df_from_cache(solute_smiles, cache, prefix="solute", dim=MAP4_DIM)
    sv_df  = side_df_from_cache(solvent_smiles, cache, prefix="solvent", dim=MAP4_DIM)

    out = pd.concat([df.reset_index(drop=True), sol_df, sv_df], axis=1)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Saved: {OUTPUT_CSV}")
    print(f"Rows: {out.shape[0]}  Cols: {out.shape[1]}")


if __name__ == "__main__":
    main()
