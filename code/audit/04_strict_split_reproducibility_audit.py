#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed_splits"
OUT = ROOT / "results" / "split_audit"
MANIFEST = ROOT / "manifests"

OUT.mkdir(parents=True, exist_ok=True)
MANIFEST.mkdir(parents=True, exist_ok=True)

SOLUTE_COL = "SMILES_Solute"
SOLVENT_COL = "SMILES_Solvent"
LOGS_COL = "LogS(mol/L)"
SOL_COL = "Solubility(mol/L)"

RANDOM_STATE = 42
INTENDED_TEST_FRACTION = 0.20

SPLITS = {
    "Random split": {
        "train": DATA / "train_random_maccs_map4_padel_sen_features.csv",
        "test": DATA / "test_random_maccs_map4_padel_sen_features.csv",
        "algorithm": "Row-level ShuffleSplit with test_size=0.20 and random_state=42.",
        "outer_grouping": "rows",
    },
    "Unseen-pair split": {
        "train": DATA / "train_unseen_pair_maccs_map4_padel_sen_features.csv",
        "test": DATA / "test_unseen_pair_maccs_map4_padel_sen_features.csv",
        "algorithm": "GroupShuffleSplit with solute--solvent pair identifiers as groups, test_size=0.20 and random_state=42.",
        "outer_grouping": "solute--solvent pair",
    },
    "Strict unseen-solute split": {
        "train": DATA / "train_strict_unseen_solute_maccs_map4_padel_sen_features.csv",
        "test": DATA / "test_strict_unseen_solute_maccs_map4_padel_sen_features.csv",
        "algorithm": "GroupShuffleSplit with solute identifiers as groups, test_size=0.20 and random_state=42.",
        "outer_grouping": "solute",
    },
    "Strict unseen-solvent split": {
        "train": DATA / "train_strict_unseen_solvent_maccs_map4_padel_sen_features.csv",
        "test": DATA / "test_strict_unseen_solvent_maccs_map4_padel_sen_features.csv",
        "algorithm": "GroupShuffleSplit with solvent identifiers as groups, test_size=0.20 and random_state=42.",
        "outer_grouping": "solvent",
    },
    "Fully unseen solute--solvent split": {
        "train": DATA / "train_strict_unseen_solute_solvent_maccs_map4_padel_sen_features.csv",
        "test": DATA / "test_strict_unseen_solute_solvent_maccs_map4_padel_sen_features.csv",
        "removed": DATA / "removed_mixed_status_rows_strict_unseen_solute_solvent.csv",
        "algorithm": "Unique solutes and unique solvents were selected separately using a fixed NumPy random state of 42. Test records contain both held-out solutes and held-out solvents; training records contain neither; mixed-status rows are saved separately.",
        "outer_grouping": "solute and solvent components",
    },
}

def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df

def read_split(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)

    for c in [SOLUTE_COL, SOLVENT_COL]:
        if c not in df.columns:
            raise ValueError(f"{path}: missing required column {c}")

    df[SOLUTE_COL] = df[SOLUTE_COL].astype(str).str.strip()
    df[SOLVENT_COL] = df[SOLVENT_COL].astype(str).str.strip()
    df["group_pair"] = df[SOLUTE_COL] + "||" + df[SOLVENT_COL]

    return df

def n_unique(df, col):
    return int(df[col].nunique())

def numeric(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()

def describe_target(train_df, test_df, split_name, target_col):
    tr = numeric(train_df, target_col)
    te = numeric(test_df, target_col)

    def desc(x):
        return {
            "n": int(len(x)),
            "mean": float(x.mean()) if len(x) else np.nan,
            "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "median": float(x.median()) if len(x) else np.nan,
            "q05": float(x.quantile(0.05)) if len(x) else np.nan,
            "q25": float(x.quantile(0.25)) if len(x) else np.nan,
            "q75": float(x.quantile(0.75)) if len(x) else np.nan,
            "q95": float(x.quantile(0.95)) if len(x) else np.nan,
            "min": float(x.min()) if len(x) else np.nan,
            "max": float(x.max()) if len(x) else np.nan,
        }

    trd = desc(tr)
    ted = desc(te)

    pooled_sd = np.nan
    if np.isfinite(trd["sd"]) and np.isfinite(ted["sd"]):
        pooled_sd = float(np.sqrt((trd["sd"] ** 2 + ted["sd"] ** 2) / 2))

    smd = np.nan
    if np.isfinite(pooled_sd) and pooled_sd > 0:
        smd = float((ted["mean"] - trd["mean"]) / pooled_sd)

    return {
        "split": split_name,
        "target": target_col,
        "train_n": trd["n"],
        "test_n": ted["n"],
        "train_mean": trd["mean"],
        "test_mean": ted["mean"],
        "train_sd": trd["sd"],
        "test_sd": ted["sd"],
        "train_median": trd["median"],
        "test_median": ted["median"],
        "train_q05": trd["q05"],
        "test_q05": ted["q05"],
        "train_q25": trd["q25"],
        "test_q25": ted["q25"],
        "train_q75": trd["q75"],
        "test_q75": ted["q75"],
        "train_q95": trd["q95"],
        "test_q95": ted["q95"],
        "train_min": trd["min"],
        "test_min": ted["min"],
        "train_max": trd["max"],
        "test_max": ted["max"],
        "standardized_mean_difference_test_minus_train": smd,
    }

def write_manifest(values, path, col):
    pd.DataFrame({col: sorted(values)}).to_csv(path, index=False)

def main():
    construction_rows = []
    target_rows = []
    leakage_rows = []
    manifest_rows = []
    fully_rows = []

    for split_name, cfg in SPLITS.items():
        print(f"Processing: {split_name}")

        train_df = read_split(cfg["train"])
        test_df = read_split(cfg["test"])

        train_solutes = set(train_df[SOLUTE_COL])
        test_solutes = set(test_df[SOLUTE_COL])
        train_solvents = set(train_df[SOLVENT_COL])
        test_solvents = set(test_df[SOLVENT_COL])
        train_pairs = set(train_df["group_pair"])
        test_pairs = set(test_df["group_pair"])

        shared_solutes = train_solutes & test_solutes
        shared_solvents = train_solvents & test_solvents
        shared_pairs = train_pairs & test_pairs

        heldout_solutes = test_solutes - train_solutes
        heldout_solvents = test_solvents - train_solvents

        total_rows = len(train_df) + len(test_df)
        achieved_fraction = len(test_df) / total_rows if total_rows else np.nan

        construction_rows.append({
            "split": split_name,
            "random_seed": RANDOM_STATE,
            "intended_test_fraction": INTENDED_TEST_FRACTION,
            "achieved_test_fraction": achieved_fraction,
            "outer_grouping": cfg["outer_grouping"],
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_solutes": n_unique(train_df, SOLUTE_COL),
            "test_solutes": n_unique(test_df, SOLUTE_COL),
            "heldout_test_solutes": int(len(heldout_solutes)),
            "shared_solutes": int(len(shared_solutes)),
            "train_solvents": n_unique(train_df, SOLVENT_COL),
            "test_solvents": n_unique(test_df, SOLVENT_COL),
            "heldout_test_solvents": int(len(heldout_solvents)),
            "shared_solvents": int(len(shared_solvents)),
            "train_pairs": int(len(train_pairs)),
            "test_pairs": int(len(test_pairs)),
            "shared_pairs": int(len(shared_pairs)),
            "algorithm": cfg["algorithm"],
            "train_file": str(cfg["train"].relative_to(ROOT)),
            "test_file": str(cfg["test"].relative_to(ROOT)),
        })

        leakage_rows.append({
            "split": split_name,
            "shared_solutes": int(len(shared_solutes)),
            "shared_solvents": int(len(shared_solvents)),
            "shared_pairs": int(len(shared_pairs)),
            "pair_leakage_passed": bool(len(shared_pairs) == 0) if split_name != "Random split" else "",
            "solute_exclusion_passed": bool(len(shared_solutes) == 0) if "solute" in split_name.lower() else "",
            "solvent_exclusion_passed": bool(len(shared_solvents) == 0) if "solvent" in split_name.lower() else "",
        })

        for target_col in [LOGS_COL, SOL_COL]:
            target_rows.append(describe_target(train_df, test_df, split_name, target_col))

        if heldout_solutes:
            out_path = MANIFEST / f"heldout_solutes__{split_name.replace(' ', '_').replace('--', '_').replace('/', '_')}.csv"
            write_manifest(heldout_solutes, out_path, SOLUTE_COL)
            manifest_rows.append({
                "split": split_name,
                "manifest_type": "heldout_solutes",
                "n": int(len(heldout_solutes)),
                "file": str(out_path.relative_to(ROOT)),
            })

        if heldout_solvents:
            out_path = MANIFEST / f"heldout_solvents__{split_name.replace(' ', '_').replace('--', '_').replace('/', '_')}.csv"
            write_manifest(heldout_solvents, out_path, SOLVENT_COL)
            manifest_rows.append({
                "split": split_name,
                "manifest_type": "heldout_solvents",
                "n": int(len(heldout_solvents)),
                "file": str(out_path.relative_to(ROOT)),
            })

        if split_name == "Fully unseen solute--solvent split":
            removed_path = cfg.get("removed")
            removed_n = 0
            removed_pairs = 0

            parts = [
                train_df.assign(split_assignment="train"),
                test_df.assign(split_assignment="test"),
            ]

            if removed_path is not None and removed_path.exists():
                removed_df = read_split(removed_path)
                removed_n = int(len(removed_df))
                removed_pairs = int(removed_df["group_pair"].nunique())
                parts.append(removed_df.assign(split_assignment="removed_mixed_status"))

            all_df = pd.concat(parts, ignore_index=True)

            solute_is_heldout = all_df[SOLUTE_COL].isin(heldout_solutes)
            solvent_is_heldout = all_df[SOLVENT_COL].isin(heldout_solvents)

            all_df["component_status"] = np.select(
                [
                    (~solute_is_heldout) & (~solvent_is_heldout),
                    (solute_is_heldout) & (solvent_is_heldout),
                    (solute_is_heldout) & (~solvent_is_heldout),
                    (~solute_is_heldout) & (solvent_is_heldout),
                ],
                [
                    "neither_component_heldout",
                    "both_components_heldout",
                    "heldout_solute_only",
                    "heldout_solvent_only",
                ],
                default="unknown",
            )

            cross = (
                all_df.groupby(["split_assignment", "component_status"])
                .agg(rows=("group_pair", "size"), pairs=("group_pair", "nunique"))
                .reset_index()
            )
            cross.to_csv(OUT / "fully_unseen_cross_combination_audit.csv", index=False)

            fully_rows.append({
                "split": split_name,
                "heldout_solutes": int(len(heldout_solutes)),
                "heldout_solvents": int(len(heldout_solvents)),
                "test_rows_both_components_heldout": int(len(test_df)),
                "train_rows_neither_component_heldout": int(len(train_df)),
                "removed_mixed_status_rows": removed_n,
                "removed_mixed_status_pairs": removed_pairs,
                "removed_file": str(removed_path.relative_to(ROOT)) if removed_path is not None and removed_path.exists() else "",
            })

    construction_df = pd.DataFrame(construction_rows)
    target_df = pd.DataFrame(target_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    fully_df = pd.DataFrame(fully_rows)

    construction_df.to_csv(OUT / "strict_split_reproducibility_table.csv", index=False)
    target_df.to_csv(OUT / "strict_split_target_balance.csv", index=False)
    leakage_df.to_csv(OUT / "strict_split_leakage_checks.csv", index=False)
    manifest_df.to_csv(OUT / "split_manifest_index.csv", index=False)
    fully_df.to_csv(OUT / "fully_unseen_treatment_summary.csv", index=False)

    construction_latex = construction_df[[
        "split",
        "random_seed",
        "intended_test_fraction",
        "achieved_test_fraction",
        "outer_grouping",
        "train_rows",
        "test_rows",
        "train_solutes",
        "test_solutes",
        "train_solvents",
        "test_solvents",
        "train_pairs",
        "test_pairs",
        "heldout_test_solutes",
        "heldout_test_solvents",
        "shared_pairs",
    ]].to_latex(index=False, escape=False, float_format="%.3f")

    (OUT / "strict_split_reproducibility_table_latex.tex").write_text(construction_latex, encoding="utf-8")

    target_latex = target_df[[
        "split",
        "target",
        "train_mean",
        "test_mean",
        "train_sd",
        "test_sd",
        "train_median",
        "test_median",
        "standardized_mean_difference_test_minus_train",
    ]].to_latex(index=False, escape=False, float_format="%.3f")

    (OUT / "strict_split_target_balance_latex.tex").write_text(target_latex, encoding="utf-8")

    report = {
        "random_seed": RANDOM_STATE,
        "intended_test_fraction": INTENDED_TEST_FRACTION,
        "notes": {
            "random_split": "Row-level optimistic reference.",
            "unseen_pair": "No solute--solvent pair is shared between train and test.",
            "strict_component_splits": "Single fixed component selections; interpret as illustrative stress tests, not repeated split-level uncertainty.",
            "fully_unseen": "Mixed-status rows are excluded from train/test and retained as audit records.",
            "inner_validation": "Main modelling uses pair-grouped inner validation; strict component-aware inner validation is discussed as a limitation/future work."
        },
        "outputs": {
            "strict_split_reproducibility_table": "results/split_audit/strict_split_reproducibility_table.csv",
            "strict_split_target_balance": "results/split_audit/strict_split_target_balance.csv",
            "strict_split_leakage_checks": "results/split_audit/strict_split_leakage_checks.csv",
            "split_manifest_index": "results/split_audit/split_manifest_index.csv",
            "fully_unseen_treatment_summary": "results/split_audit/fully_unseen_treatment_summary.csv",
            "fully_unseen_cross_combination_audit": "results/split_audit/fully_unseen_cross_combination_audit.csv",
        }
    }

    (OUT / "strict_split_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== STRICT SPLIT REPRODUCIBILITY TABLE ===")
    print(construction_df.to_string(index=False))

    print("\n=== TARGET BALANCE TABLE ===")
    print(target_df[[
        "split",
        "target",
        "train_mean",
        "test_mean",
        "train_sd",
        "test_sd",
        "train_median",
        "test_median",
        "standardized_mean_difference_test_minus_train",
    ]].to_string(index=False))

    print("\n=== LEAKAGE CHECKS ===")
    print(leakage_df.to_string(index=False))

    print("\n=== MANIFEST INDEX ===")
    print(manifest_df.to_string(index=False))

    print("\n=== FULLY UNSEEN TREATMENT SUMMARY ===")
    print(fully_df.to_string(index=False))

    print("\nSaved outputs to:", OUT)

if __name__ == "__main__":
    main()
