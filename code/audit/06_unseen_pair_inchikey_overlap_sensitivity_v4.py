#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed_splits"
PRED_DIR = ROOT / "results" / "validation_rerun" / "full_outputs_no_models" / "fold_predictions"
OUT = ROOT / "results" / "chemical_identity_audit"
OUT.mkdir(parents=True, exist_ok=True)

TEST_PATH = SPLIT_DIR / "test_unseen_pair_maccs_map4_padel_sen_features.csv"
TRAIN_PATH = SPLIT_DIR / "train_unseen_pair_maccs_map4_padel_sen_features.csv"

BASE_COLS = [
    "SMILES_Solute",
    "Temperature_K",
    "Solvent",
    "SMILES_Solvent",
    "Solubility(mol/L)",
    "LogS(mol/L)",
    "Compound_Name",
    "CAS",
    "PubChem_CID",
]

EXPECTED_DIRECT_LOGS = {
    "r2": 0.842680,
    "mae": 0.312926,
    "rmse": 0.483807,
}

BAD = {"", "-", "nan", "NaN", "None", "none", "NULL", "null", "NA", "N/A"}

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def read_base(path):
    header = pd.read_csv(path, nrows=0, low_memory=False)
    cols = [c for c in BASE_COLS if c in header.columns]
    df = pd.read_csv(path, usecols=cols, low_memory=False)

    for c in df.columns:
        if c not in ["Temperature_K", "Solubility(mol/L)", "LogS(mol/L)"]:
            df[c] = df[c].map(clean)

    for c in ["Temperature_K", "Solubility(mol/L)", "LogS(mol/L)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def inchikey_from_smiles(s):
    s = clean(s)
    if s in BAD:
        return ""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""

def add_inchikey_pair(df):
    df = df.copy()

    solute_cache = {
        s: inchikey_from_smiles(s)
        for s in sorted(set(df["SMILES_Solute"].map(clean)))
    }
    solvent_cache = {
        s: inchikey_from_smiles(s)
        for s in sorted(set(df["SMILES_Solvent"].map(clean)))
    }

    df["solute_inchikey"] = df["SMILES_Solute"].map(lambda x: solute_cache[clean(x)])
    df["solvent_inchikey"] = df["SMILES_Solvent"].map(lambda x: solvent_cache[clean(x)])
    df["inchikey_pair"] = df["solute_inchikey"] + "||" + df["solvent_inchikey"]

    return df

def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    resid = y - p
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot)

    return {
        "n": int(len(y)),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }

def collect_fold_groups(n_test):
    pattern = re.compile(r"(.+)_fold_([1-5])_test_predictions\.csv$")
    groups = {}

    for path in sorted(PRED_DIR.glob("*_fold_*_test_predictions.csv")):
        m = pattern.match(path.name)
        if not m:
            continue

        group = m.group(1)
        fold = int(m.group(2))

        try:
            header = pd.read_csv(path, nrows=0, low_memory=False)
        except Exception:
            continue

        if "y_pred_logs" not in header.columns:
            continue

        try:
            n_rows = sum(1 for _ in open(path, "rb")) - 1
        except Exception:
            continue

        if n_rows != n_test:
            continue

        groups.setdefault(group, {})[fold] = path

    complete = {
        g: folds
        for g, folds in groups.items()
        if sorted(folds.keys()) == [1, 2, 3, 4, 5]
    }

    return complete

def main():
    train = add_inchikey_pair(read_base(TRAIN_PATH))
    test = add_inchikey_pair(read_base(TEST_PATH))

    train_pairs = set(train["inchikey_pair"]) - {""}
    test_pairs = set(test["inchikey_pair"]) - {""}
    overlap_pairs = sorted(train_pairs & test_pairs)

    test["inchikey_overlap_pair"] = test["inchikey_pair"].isin(overlap_pairs)
    mask_keep = ~test["inchikey_overlap_pair"].to_numpy(dtype=bool)

    n_affected = int(test["inchikey_overlap_pair"].sum())
    affected_fraction = n_affected / len(test)

    print("\n=== InChIKey-overlap rows ===")
    print("overlap pairs:", len(overlap_pairs))
    print("affected test rows:", n_affected)
    print("total test rows:", len(test))
    print("affected fraction:", affected_fraction)

    test[test["inchikey_overlap_pair"]].to_csv(
        OUT / "unseen_pair_inchikey_overlap_removed_test_rows_v4.csv",
        index=False,
    )

    y_true_log = test["LogS(mol/L)"].to_numpy(dtype=float)
    y_true_sol = test["Solubility(mol/L)"].to_numpy(dtype=float)

    groups = collect_fold_groups(len(test))

    rows = []
    vectors = {}

    for group, folds in groups.items():
        preds = []

        for fold in [1, 2, 3, 4, 5]:
            df = pd.read_csv(folds[fold], low_memory=False)
            preds.append(pd.to_numeric(df["y_pred_logs"], errors="coerce").to_numpy(dtype=float))

        pred_avg = np.mean(np.vstack(preds), axis=0)
        vectors[group] = pred_avg

        m = metrics(y_true_log, pred_avg)

        dist = (
            abs(m["r2"] - EXPECTED_DIRECT_LOGS["r2"])
            + abs(m["mae"] - EXPECTED_DIRECT_LOGS["mae"])
            + abs(m["rmse"] - EXPECTED_DIRECT_LOGS["rmse"])
        )

        rows.append({
            "group": group,
            "n_folds": 5,
            "LogS_R2": m["r2"],
            "LogS_MAE": m["mae"],
            "LogS_RMSE": m["rmse"],
            "distance_to_expected_direct_LogS": dist,
        })

    group_table = pd.DataFrame(rows).sort_values("distance_to_expected_direct_LogS")
    group_table.to_csv(
        OUT / "unseen_pair_group_averaged_prediction_candidates_v4.csv",
        index=False,
    )

    print("\n=== Fold-averaged prediction groups closest to expected Direct LogS ===")
    print(group_table.head(30).to_string(index=False))

    preferred_names = [
        "logs_direct",
        "logS_direct",
        "direct_logs",
        "direct_LogS",
        "primary_logs",
        "primary_LogS",
    ]

    selected_group = None

    for name in preferred_names:
        if name in vectors:
            selected_group = name
            break

    if selected_group is None:
        selected_group = str(group_table.iloc[0]["group"])

    selected_dist = float(group_table[group_table["group"] == selected_group]["distance_to_expected_direct_LogS"].iloc[0])

    print("\n=== Selected group ===")
    print("selected_group:", selected_group)
    print("distance_to_expected_direct_LogS:", selected_dist)

    if selected_dist > 0.015:
        print("\nSTOP: No fold-averaged prediction group sufficiently matches the expected Direct LogS operational metrics.")
        print("Do NOT use this sensitivity result in the manuscript yet.")
        print("Send the file:")
        print(OUT / "unseen_pair_group_averaged_prediction_candidates_v4.csv")
        raise SystemExit(2)

    pred_log = vectors[selected_group]
    pred_sol = np.power(10.0, pred_log)

    result_rows = []

    for label, mask in [
        ("Original unseen-pair test set", np.ones(len(test), dtype=bool)),
        ("After excluding InChIKey-overlap test rows", mask_keep),
    ]:
        log_m = metrics(y_true_log[mask], pred_log[mask])
        sol_m = metrics(y_true_sol[mask], pred_sol[mask])

        result_rows.append({
            "evaluation_set": label,
            "n_test_rows": int(mask.sum()),
            "removed_rows": int(len(test) - mask.sum()),
            "LogS_R2": log_m["r2"],
            "LogS_MAE": log_m["mae"],
            "LogS_RMSE": log_m["rmse"],
            "Solubility_R2": sol_m["r2"],
            "Solubility_MAE": sol_m["mae"],
            "Solubility_RMSE": sol_m["rmse"],
            "prediction_group": selected_group,
        })

    orig = result_rows[0]
    filt = result_rows[1]

    result_rows.append({
        "evaluation_set": "Change after exclusion",
        "n_test_rows": filt["n_test_rows"],
        "removed_rows": filt["removed_rows"],
        "LogS_R2": filt["LogS_R2"] - orig["LogS_R2"],
        "LogS_MAE": filt["LogS_MAE"] - orig["LogS_MAE"],
        "LogS_RMSE": filt["LogS_RMSE"] - orig["LogS_RMSE"],
        "Solubility_R2": filt["Solubility_R2"] - orig["Solubility_R2"],
        "Solubility_MAE": filt["Solubility_MAE"] - orig["Solubility_MAE"],
        "Solubility_RMSE": filt["Solubility_RMSE"] - orig["Solubility_RMSE"],
        "prediction_group": selected_group,
    })

    out_df = pd.DataFrame(result_rows)
    out_df.to_csv(
        OUT / "unseen_pair_inchikey_sensitivity_metrics_v4.csv",
        index=False,
    )

    latex = r"""\begin{table}[htbp]
\centering
\small
\caption{Sensitivity of the primary unseen-pair result to removal of post hoc InChIKey-overlap test records. Metrics were recomputed using the same five-fold averaged operational prediction vector after excluding the 39 test records involved in identity-collapsed InChIKey-pair overlap.}
\label{tab:supp_inchikey_sensitivity}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrrrr}
\toprule
\textbf{Evaluation set}
& \textbf{Test n}
& \textbf{Removed n}
& $\mathbf{R^2_{\mathrm{LogS}}}$
& \textbf{MAE$_{\mathrm{LogS}}$}
& \textbf{RMSE$_{\mathrm{LogS}}$}
& $\mathbf{R^2_{S}}$
& \textbf{MAE$_S$}
& \textbf{RMSE$_S$} \\
\midrule
"""

    for _, r in out_df.iloc[:2].iterrows():
        latex += (
            f"{r['evaluation_set']} & "
            f"{int(r['n_test_rows']):,} & "
            f"{int(r['removed_rows']):,} & "
            f"{r['LogS_R2']:.3f} & "
            f"{r['LogS_MAE']:.3f} & "
            f"{r['LogS_RMSE']:.3f} & "
            f"{r['Solubility_R2']:.3f} & "
            f"{r['Solubility_MAE']:.3f} & "
            f"{r['Solubility_RMSE']:.3f} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    (OUT / "unseen_pair_inchikey_sensitivity_latex_table_v4.tex").write_text(
        latex,
        encoding="utf-8",
    )

    report = {
        "selected_group": selected_group,
        "distance_to_expected_direct_LogS": selected_dist,
        "n_overlapping_inchikey_pairs": len(overlap_pairs),
        "n_affected_test_rows": n_affected,
        "total_unseen_pair_test_rows": len(test),
        "affected_fraction": affected_fraction,
        "outputs": {
            "metrics": "results/chemical_identity_audit/unseen_pair_inchikey_sensitivity_metrics_v4.csv",
            "group_candidates": "results/chemical_identity_audit/unseen_pair_group_averaged_prediction_candidates_v4.csv",
            "removed_rows": "results/chemical_identity_audit/unseen_pair_inchikey_overlap_removed_test_rows_v4.csv",
            "latex_table": "results/chemical_identity_audit/unseen_pair_inchikey_sensitivity_latex_table_v4.tex",
        },
    }

    (OUT / "unseen_pair_inchikey_sensitivity_report_v4.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL VALID SENSITIVITY METRICS ===")
    print(out_df.to_string(index=False))

    print("\nSaved:")
    print(OUT / "unseen_pair_inchikey_sensitivity_metrics_v4.csv")
    print(OUT / "unseen_pair_inchikey_sensitivity_latex_table_v4.tex")
    print(OUT / "unseen_pair_inchikey_sensitivity_report_v4.json")

if __name__ == "__main__":
    main()
