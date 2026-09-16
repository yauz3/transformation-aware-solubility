#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import json
import math
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed_splits"
OUT = ROOT / "results" / "chemical_identity_audit"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN = SPLIT_DIR / "train_unseen_pair_maccs_map4_padel_sen_features.csv"
TEST = SPLIT_DIR / "test_unseen_pair_maccs_map4_padel_sen_features.csv"

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

EXPECTED = {
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
    solute_cache = {s: inchikey_from_smiles(s) for s in sorted(set(df["SMILES_Solute"].map(clean)))}
    solvent_cache = {s: inchikey_from_smiles(s) for s in sorted(set(df["SMILES_Solvent"].map(clean)))}
    df["solute_inchikey"] = df["SMILES_Solute"].map(lambda x: solute_cache[clean(x)])
    df["solvent_inchikey"] = df["SMILES_Solvent"].map(lambda x: solvent_cache[clean(x)])
    df["inchikey_pair"] = df["solute_inchikey"] + "||" + df["solvent_inchikey"]
    return df

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "n": int(len(y_true)),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }

def count_csv_rows(path):
    try:
        with open(path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None

def possible_pred_cols(cols):
    out = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["pred", "prediction", "yhat", "y_hat"]):
            if not any(bad in cl for bad in ["std", "sd", "var", "uncert", "error", "mae", "rmse", "r2"]):
                out.append(c)
    return out

def scan_prediction_candidates(n_test):
    candidates = []

    skip_parts = {
        "data/raw",
        "data/processed_splits",
        "results/chemical_identity_audit",
        "results/split_audit",
        "manifests",
        "logs",
    }

    csv_files = sorted(ROOT.rglob("*.csv"))

    for path in csv_files:
        rel = str(path.relative_to(ROOT))

        if any(part in rel for part in skip_parts):
            continue

        # Avoid huge unrelated matrices.
        try:
            size_mb = path.stat().st_size / 1024 / 1024
        except Exception:
            size_mb = 0

        if size_mb > 800:
            continue

        try:
            header = pd.read_csv(path, nrows=0, low_memory=False)
        except Exception:
            continue

        cols = list(header.columns)
        pred_cols = possible_pred_cols(cols)

        if not pred_cols:
            continue

        n_rows = count_csv_rows(path)
        if n_rows != n_test:
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue

        for pc in pred_cols:
            pred = pd.to_numeric(df[pc], errors="coerce")
            if pred.notna().sum() < n_test * 0.95:
                continue

            candidates.append({
                "path": str(path),
                "relative_path": rel,
                "prediction_column": pc,
                "n_rows": n_rows,
                "size_mb": round(size_mb, 3),
                "pred_values": pred.to_numpy(dtype=float),
            })

    return candidates

def main():
    train = add_inchikey_pair(read_base(TRAIN))
    test = add_inchikey_pair(read_base(TEST))

    train_pairs = set(train["inchikey_pair"]) - {""}
    test_pairs = set(test["inchikey_pair"]) - {""}
    overlap_pairs = sorted(train_pairs & test_pairs)

    test["inchikey_overlap_pair"] = test["inchikey_pair"].isin(overlap_pairs)
    excluded = test[test["inchikey_overlap_pair"]].copy()
    retained = test[~test["inchikey_overlap_pair"]].copy()

    excluded.to_csv(OUT / "unseen_pair_inchikey_overlap_removed_test_rows.csv", index=False)

    print("\n=== InChIKey overlap in unseen-pair split ===")
    print("n overlapping InChIKey pairs:", len(overlap_pairs))
    print("n affected test rows:", int(test["inchikey_overlap_pair"].sum()))
    print("total unseen-pair test rows:", len(test))
    print("affected fraction:", int(test["inchikey_overlap_pair"].sum()) / len(test))

    print("\nOverlapping InChIKey pairs:")
    for p in overlap_pairs:
        print(" -", p)

    print("\n=== Removed test-row examples ===")
    cols_show = [
        "SMILES_Solute", "solute_inchikey", "Solvent", "SMILES_Solvent",
        "solvent_inchikey", "Temperature_K", "LogS(mol/L)",
        "Solubility(mol/L)", "Compound_Name", "CAS", "PubChem_CID"
    ]
    print(excluded[cols_show].head(30).to_string(index=False))

    print("\n=== Scanning prediction files ===")
    candidates = scan_prediction_candidates(len(test))
    print("n candidate prediction columns:", len(candidates))

    if not candidates:
        print("\nERROR: No suitable prediction CSV with 20,133 rows and prediction-like column was found.")
        print("Please inspect files under results/validation_rerun or send the output of:")
        print("find \"$PACK\" -type f \\( -iname '*pred*.csv' -o -iname '*prediction*.csv' -o -iname '*operational*.csv' \\)")
        raise SystemExit(1)

    y_true_log = test["LogS(mol/L)"].to_numpy(dtype=float)
    y_true_sol = test["Solubility(mol/L)"].to_numpy(dtype=float)

    rows = []

    for cand in candidates:
        pred_log = cand["pred_values"]

        m_all = regression_metrics(y_true_log, pred_log)
        score = (
            abs(m_all["r2"] - EXPECTED["r2"])
            + abs(m_all["mae"] - EXPECTED["mae"])
            + abs(m_all["rmse"] - EXPECTED["rmse"])
        )

        rows.append({
            "relative_path": cand["relative_path"],
            "prediction_column": cand["prediction_column"],
            "n_rows": cand["n_rows"],
            "size_mb": cand["size_mb"],
            "all_logS_R2": m_all["r2"],
            "all_logS_MAE": m_all["mae"],
            "all_logS_RMSE": m_all["rmse"],
            "distance_to_expected_direct_LogS": score,
        })

    candidate_table = pd.DataFrame(rows).sort_values("distance_to_expected_direct_LogS")
    candidate_table.to_csv(OUT / "unseen_pair_inchikey_sensitivity_prediction_candidates.csv", index=False)

    print("\n=== Top candidate prediction files ===")
    print(candidate_table.head(20).to_string(index=False))

    best = candidate_table.iloc[0]
    best_path = ROOT / best["relative_path"]
    best_col = best["prediction_column"]

    best_df = pd.read_csv(best_path, low_memory=False)
    pred_log = pd.to_numeric(best_df[best_col], errors="coerce").to_numpy(dtype=float)

    # Basic guard: selected vector should match direct LogS operational metric.
    if best["distance_to_expected_direct_LogS"] > 0.05:
        print("\nWARNING: Best candidate does not closely match expected direct LogS operational metrics.")
        print("Do not use the sensitivity result until the prediction source is verified.")
        print("Best candidate:", best_path)
        print("Best column:", best_col)

    mask_keep = ~test["inchikey_overlap_pair"].to_numpy(dtype=bool)

    y_pred_sol = np.power(10.0, pred_log)

    metrics = []

    for label, mask in [
        ("Original unseen-pair test set", np.ones(len(test), dtype=bool)),
        ("After excluding InChIKey-overlap test rows", mask_keep),
    ]:
        log_m = regression_metrics(y_true_log[mask], pred_log[mask])
        sol_m = regression_metrics(y_true_sol[mask], y_pred_sol[mask])

        metrics.append({
            "evaluation_set": label,
            "n_test_rows": int(mask.sum()),
            "removed_rows": int(len(test) - mask.sum()),
            "LogS_R2": log_m["r2"],
            "LogS_MAE": log_m["mae"],
            "LogS_RMSE": log_m["rmse"],
            "Solubility_R2": sol_m["r2"],
            "Solubility_MAE": sol_m["mae"],
            "Solubility_RMSE": sol_m["rmse"],
            "prediction_source": str(best_path.relative_to(ROOT)),
            "prediction_column": best_col,
        })

    metrics_df = pd.DataFrame(metrics)

    # Add delta row.
    orig = metrics_df.iloc[0]
    filt = metrics_df.iloc[1]
    delta = {
        "evaluation_set": "Change after exclusion",
        "n_test_rows": int(filt["n_test_rows"]),
        "removed_rows": int(filt["removed_rows"]),
        "LogS_R2": float(filt["LogS_R2"] - orig["LogS_R2"]),
        "LogS_MAE": float(filt["LogS_MAE"] - orig["LogS_MAE"]),
        "LogS_RMSE": float(filt["LogS_RMSE"] - orig["LogS_RMSE"]),
        "Solubility_R2": float(filt["Solubility_R2"] - orig["Solubility_R2"]),
        "Solubility_MAE": float(filt["Solubility_MAE"] - orig["Solubility_MAE"]),
        "Solubility_RMSE": float(filt["Solubility_RMSE"] - orig["Solubility_RMSE"]),
        "prediction_source": str(best_path.relative_to(ROOT)),
        "prediction_column": best_col,
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([delta])], ignore_index=True)

    metrics_df.to_csv(OUT / "unseen_pair_inchikey_sensitivity_metrics.csv", index=False)

    print("\n=== FINAL SENSITIVITY METRICS ===")
    print(metrics_df.to_string(index=False))

    # Compact LaTeX table.
    latex_rows = metrics_df.iloc[:2].copy()
    latex = r"""\begin{table}[htbp]
\centering
\small
\caption{Sensitivity of the primary unseen-pair result to removal of post hoc InChIKey-overlap test records. Metrics were recomputed using the same operational Direct LogS prediction vector after excluding the 39 test records involved in identity-collapsed InChIKey-pair overlap.}
\label{tab:supp_inchikey_sensitivity}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Evaluation set} & \textbf{Test n} & \textbf{Removed n} & $\mathbf{R^2_{\mathrm{LogS}}}$ & \textbf{MAE$_{\mathrm{LogS}}$} & \textbf{RMSE$_{\mathrm{LogS}}$} & $\mathbf{R^2_{S}}$ & \textbf{MAE$_S$} \\
\midrule
"""
    for _, r in latex_rows.iterrows():
        latex += (
            f"{r['evaluation_set']} & "
            f"{int(r['n_test_rows']):,} & "
            f"{int(r['removed_rows']):,} & "
            f"{r['LogS_R2']:.3f} & "
            f"{r['LogS_MAE']:.3f} & "
            f"{r['LogS_RMSE']:.3f} & "
            f"{r['Solubility_R2']:.3f} & "
            f"{r['Solubility_MAE']:.3f} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "unseen_pair_inchikey_sensitivity_latex_table.tex").write_text(latex, encoding="utf-8")

    report = {
        "n_overlapping_inchikey_pairs": len(overlap_pairs),
        "n_affected_test_rows": int(test["inchikey_overlap_pair"].sum()),
        "total_unseen_pair_test_rows": len(test),
        "affected_fraction": float(int(test["inchikey_overlap_pair"].sum()) / len(test)),
        "selected_prediction_source": str(best_path.relative_to(ROOT)),
        "selected_prediction_column": best_col,
        "candidate_distance_to_expected_direct_logS": float(best["distance_to_expected_direct_LogS"]),
        "outputs": {
            "metrics": "results/chemical_identity_audit/unseen_pair_inchikey_sensitivity_metrics.csv",
            "removed_rows": "results/chemical_identity_audit/unseen_pair_inchikey_overlap_removed_test_rows.csv",
            "candidate_predictions": "results/chemical_identity_audit/unseen_pair_inchikey_sensitivity_prediction_candidates.csv",
            "latex_table": "results/chemical_identity_audit/unseen_pair_inchikey_sensitivity_latex_table.tex",
        },
    }

    (OUT / "unseen_pair_inchikey_sensitivity_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(OUT / "unseen_pair_inchikey_sensitivity_metrics.csv")
    print(OUT / "unseen_pair_inchikey_overlap_removed_test_rows.csv")
    print(OUT / "unseen_pair_inchikey_sensitivity_prediction_candidates.csv")
    print(OUT / "unseen_pair_inchikey_sensitivity_latex_table.tex")
    print(OUT / "unseen_pair_inchikey_sensitivity_report.json")

if __name__ == "__main__":
    main()
