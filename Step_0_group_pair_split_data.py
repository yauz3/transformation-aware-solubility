#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sadettin Y. Uğurlu
# 8/03/2026

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# =========================
# CONFIG
# =========================
DATA_PATH = "BigSolDBv2.0.csv"
TRAIN_OUT = "train_group-based.csv"
TEST_OUT  = "test_group-based.csv"

SOLUTE_COL = "SMILES_Solute"
SOLVENT_COL = "SMILES_Solvent"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# =========================
# LOAD
# =========================
df = pd.read_csv(DATA_PATH, low_memory=False)

# Basic column cleanup
df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

# Required columns check
for col in [SOLUTE_COL, SOLVENT_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================
# GROUP-PAIR DEFINITION
# =========================
# Same solute-solvent pair must stay entirely in train or test
groups = (
    df[SOLUTE_COL].astype(str).str.strip() +
    "||" +
    df[SOLVENT_COL].astype(str).str.strip()
)

# Optional: save group key for inspection
df = df.copy()
df["group_pair"] = groups

# =========================
# GROUP-BASED SPLIT
# =========================
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

indices = df.index.to_numpy()
train_idx, test_idx = next(gss.split(indices, groups=groups))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

# =========================
# LEAKAGE CHECK
# =========================
train_groups = set(train_df["group_pair"].unique())
test_groups = set(test_df["group_pair"].unique())

overlap = train_groups.intersection(test_groups)
if len(overlap) > 0:
    raise RuntimeError(
        f"Data leakage detected: {len(overlap)} group-pairs appear in both train and test!"
    )

# =========================
# SAVE
# =========================
train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

# =========================
# REPORT
# =========================
print("✅ Group-pair split completed successfully.")
print(f"Input file: {DATA_PATH}")
print(f"Train file: {TRAIN_OUT}")
print(f"Test file : {TEST_OUT}")
print()

print(f"Total rows        : {len(df)}")
print(f"Train rows        : {len(train_df)}")
print(f"Test rows         : {len(test_df)}")
print()

print(f"Total group-pairs : {df['group_pair'].nunique()}")
print(f"Train group-pairs : {train_df['group_pair'].nunique()}")
print(f"Test group-pairs  : {test_df['group_pair'].nunique()}")
print()

print(f"Observed test row ratio   : {len(test_df) / len(df):.4f}")
print(f"Requested test row ratio  : {TEST_SIZE:.4f}")
print()
print("Leakage check: PASSED (no shared group-pair between train and test)")
