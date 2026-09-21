#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG — matched to original Direct LogS training script
# =========================================================

BASE = Path("/home/yavuz/yavuz_proje/13_projects/14_solubility_vs_temperature_github")
SPLIT_DIR = BASE / "validation/github_reproducibility_package/data/processed_splits"

TRAIN_RANDOM = SPLIT_DIR / "train_random_maccs_map4_padel_sen_features.csv"
TEST_RANDOM  = SPLIT_DIR / "test_random_maccs_map4_padel_sen_features.csv"

OUT_DIR = BASE / "validation/github_reproducibility_package/results/repeated_strict_component_resampling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = OUT_DIR / "fold_predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# columns
SOLUTE_COL  = "SMILES_Solute"
SOLVENT_COL = "SMILES_Solvent"
Y_SOL       = "Solubility(mol/L)"
Y_LOGS      = "LogS(mol/L)"

EXCLUDE_COLS = {
    SOLUTE_COL,
    SOLVENT_COL,
    Y_SOL,
    Y_LOGS,
    "LogS",
    "logS",
    "logs",
    "Solubility(mole_fraction)",
    "Solvent",
    "Compound_Name",
    "CAS",
    "PubChem_CID",
    "FDA_Approved",
    "Source",
    "group_pair",
}

LOSS_FUNCTION = "mae"
EPS = 1e-6

# repeated outer component-selection seeds
OUTER_SEEDS = [7, 42, 123]
OUTER_TEST_SIZE = 0.20

# inner fold models
N_FOLDS      = 5
RANDOM_STATE = 42

# training — matched to original
EPOCHS       = 100
BATCH_SIZE   = 1024
LR           = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS  = 4
PATIENCE     = 7

# model — matched to original
MODEL_SIZE = "medium"

# mixed precision / numerical safety
USE_AMP  = True
CLIP_ABS = 1e9


# =========================================================
# LOSSES
# =========================================================

class DifferentiableMAELoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class DifferentiableRMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        return torch.sqrt(mse + self.eps)


class DifferentiableR2Loss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        target_mean = torch.mean(target)
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target_mean) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + self.eps))
        return 1.0 - r2


def get_loss_criterion(loss_name: str) -> nn.Module:
    loss_name = str(loss_name).strip().lower()
    if loss_name == "mae":
        return DifferentiableMAELoss()
    if loss_name == "rmse":
        return DifferentiableRMSELoss()
    if loss_name == "r2":
        return DifferentiableR2Loss()
    raise ValueError("LOSS_FUNCTION must be one of: 'r2', 'mae', 'rmse'")


def get_selection_mode(loss_name: str):
    loss_name = str(loss_name).strip().lower()
    if loss_name == "mae":
        return "logs_mae", "min"
    if loss_name == "rmse":
        return "logs_rmse", "min"
    if loss_name == "r2":
        return "logs_r2", "max"
    raise ValueError("LOSS_FUNCTION must be one of: 'r2', 'mae', 'rmse'")


# =========================================================
# HELPERS — matched to original numerical handling
# =========================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def sanitize_numeric_array(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    x = np.clip(x, -CLIP_ABS, CLIP_ABS)
    return x


def safe_pow10(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return np.power(10.0, x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
    }


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return None, None
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def get_hidden_config(model_size: str):
    if model_size == "basic":
        return [1024, 512], 0.10
    if model_size == "deep":
        return [4096, 2048, 1024, 512, 256], 0.20
    return [4096, 2048, 1024, 512, 256, 128, 64, 32], 0.15


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA
# =========================================================

def load_and_prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_columns(df)

    required = [SOLUTE_COL, SOLVENT_COL, Y_SOL, Y_LOGS]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column in {csv_path}: {c}")

    y_sol  = pd.to_numeric(df[Y_SOL],  errors="coerce")
    y_logs = pd.to_numeric(df[Y_LOGS], errors="coerce")

    mask = y_sol.notna() & y_logs.notna() & (y_sol > 0)
    df = df.loc[mask].reset_index(drop=True)

    df[Y_SOL]  = pd.to_numeric(df[Y_SOL],  errors="coerce").astype(np.float32)
    df[Y_LOGS] = pd.to_numeric(df[Y_LOGS], errors="coerce").astype(np.float32)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if len(cols) == 0:
        raise ValueError("No feature columns remain after applying EXCLUDE_COLS.")
    return cols


def build_arrays_from_df(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X = sanitize_numeric_array(X)

    y_sol  = pd.to_numeric(df[Y_SOL], errors="coerce").to_numpy(dtype=np.float32)
    y_logs = pd.to_numeric(df[Y_LOGS], errors="coerce").to_numpy(dtype=np.float32)

    return X, y_sol, y_logs


def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    X_train = sanitize_numeric_array(X_train)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler(copy=True)
    scaler.fit(X_train)
    return scaler


# =========================================================
# DATASET
# =========================================================

class ArrayDataset(Dataset):
    def __init__(self, X, y_sol, y_logs, scaler: StandardScaler):
        self.X      = X
        self.y_sol  = y_sol
        self.y_logs = y_logs
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = np.array(self.X[i], dtype=np.float32, copy=True)
        x = sanitize_numeric_array(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.scaler.transform(x.reshape(1, -1)).reshape(-1).astype(np.float32)

        y_sol  = float(self.y_sol[i])
        y_logs = float(self.y_logs[i])

        targets = {
            "sol_true":  torch.tensor(y_sol,  dtype=torch.float32),
            "logs_true": torch.tensor(y_logs, dtype=torch.float32),
        }
        return torch.from_numpy(x), targets


# =========================================================
# MODEL — matched to original
# =========================================================

class SingleTargetMLP(nn.Module):
    def __init__(self, input_dim: int, model_size: str = "medium"):
        super().__init__()
        hidden, dropout = get_hidden_config(model_size)

        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# =========================================================
# TRAIN / EVALUATE
# =========================================================

def compute_loss(outputs: torch.Tensor, targets: dict, criterion: nn.Module):
    loss = criterion(outputs, targets["logs_true"])
    return loss, {"loss_total": float(loss.detach().cpu().item())}


def evaluate(model, loader, device, criterion, amp_enabled):
    model.eval()
    losses       = []
    y_true_logs  = []
    y_pred_logs  = []
    y_true_sol   = []

    with torch.no_grad():
        for xb, tb in loader:
            xb = xb.to(device, non_blocking=True)
            tb = {k: v.to(device, non_blocking=True) for k, v in tb.items()}

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(xb)
                total_loss, loss_dict = compute_loss(outputs, tb, criterion)

            losses.append(loss_dict["loss_total"])
            y_true_logs.append(tb["logs_true"].detach().cpu().numpy())
            y_pred_logs.append(outputs.detach().cpu().numpy())
            y_true_sol.append(tb["sol_true"].detach().cpu().numpy())

    y_true_logs = np.concatenate(y_true_logs)
    y_pred_logs = np.concatenate(y_pred_logs)
    y_true_sol  = np.concatenate(y_true_sol)

    y_pred_sol_from_logs = safe_pow10(y_pred_logs)

    logs_metrics = regression_metrics(y_true_logs, y_pred_logs)
    sol_metrics  = regression_metrics(y_true_sol,  y_pred_sol_from_logs)

    return {
        "loss":                 float(np.mean(losses)),
        "logs_r2":              logs_metrics["r2"],
        "logs_mae":             logs_metrics["mae"],
        "logs_rmse":            logs_metrics["rmse"],
        "transfer_sol_r2":      sol_metrics["r2"],
        "transfer_sol_mae":     sol_metrics["mae"],
        "transfer_sol_rmse":    sol_metrics["rmse"],
        "y_true_logs":          y_true_logs,
        "y_pred_logs":          y_pred_logs,
        "y_true_sol":           y_true_sol,
        "y_pred_sol_from_logs": y_pred_sol_from_logs,
    }


def train_model(model, train_loader, val_loader, device, fold_id: int):
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion  = get_loss_criterion(LOSS_FUNCTION)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    selection_metric, selection_mode = get_selection_mode(LOSS_FUNCTION)
    best_score = np.inf if selection_mode == "min" else -np.inf

    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, tb in train_loader:
            xb = xb.to(device, non_blocking=True)
            tb = {k: v.to(device, non_blocking=True) for k, v in tb.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                outputs = model(xb)
                total_loss, loss_dict = compute_loss(outputs, tb, criterion)

            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_losses.append(loss_dict["loss_total"])

        train_loss  = float(np.mean(train_losses))
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            USE_AMP and device.type == "cuda",
        )

        current_score = val_metrics[selection_metric]

        print(
            f"[Fold {fold_id}] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_logs_R2={val_metrics['logs_r2']:.4f} "
            f"val_logs_MAE={val_metrics['logs_mae']:.6f} "
            f"val_logs_RMSE={val_metrics['logs_rmse']:.6f} "
            f"val_sol_R2={val_metrics['transfer_sol_r2']:.4f} "
            f"selection_metric={selection_metric}:{current_score:.6f}",
            flush=True,
        )

        improved = (
            current_score < best_score if selection_mode == "min"
            else current_score > best_score
        )

        if improved:
            best_score = current_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[Fold {fold_id}] early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================================================
# OUTER STRICT SPLITS
# =========================================================

def make_outer_split(df: pd.DataFrame, regime: str, seed: int):
    if regime == "strict_unseen_solute":
        group_col = SOLUTE_COL
    elif regime == "strict_unseen_solvent":
        group_col = SOLVENT_COL
    else:
        raise ValueError(f"Unknown regime: {regime}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=OUTER_TEST_SIZE,
        random_state=seed,
    )

    idx = np.arange(len(df))
    groups = df[group_col].astype(str).str.strip().values
    train_idx, test_idx = next(splitter.split(idx, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df, group_col


def evaluate_one_outer_seed(df, feature_cols, regime, outer_seed, device):
    print("\n" + "=" * 120, flush=True)
    print(f"REGIME={regime} | OUTER_SEED={outer_seed}", flush=True)
    print("=" * 120, flush=True)

    train_df, test_df, outer_group_col = make_outer_split(df, regime, outer_seed)

    train_pairs = (
        train_df[SOLUTE_COL].astype(str).str.strip()
        + "||"
        + train_df[SOLVENT_COL].astype(str).str.strip()
    )
    test_pairs = (
        test_df[SOLUTE_COL].astype(str).str.strip()
        + "||"
        + test_df[SOLVENT_COL].astype(str).str.strip()
    )

    shared_solutes = len(set(train_df[SOLUTE_COL].astype(str)) & set(test_df[SOLUTE_COL].astype(str)))
    shared_solvents = len(set(train_df[SOLVENT_COL].astype(str)) & set(test_df[SOLVENT_COL].astype(str)))
    shared_pairs = len(set(train_pairs) & set(test_pairs))

    print(f"train_rows={len(train_df)} test_rows={len(test_df)}", flush=True)
    print(f"achieved_test_fraction={len(test_df)/len(df):.6f}", flush=True)
    print(f"shared_solutes={shared_solutes} shared_solvents={shared_solvents} shared_pairs={shared_pairs}", flush=True)

    X_train_full, y_sol_train_full, y_logs_train_full = build_arrays_from_df(train_df, feature_cols)
    X_test, y_sol_test, y_logs_test = build_arrays_from_df(test_df, feature_cols)

    # Inner CV deliberately matches original random-KFold training behavior.
    # This is reported as a limitation for strict component-level deployment.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    fold_pred_logs = []
    fold_pred_sol = []

    for fold_id, (fit_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        print("\n" + "-" * 100, flush=True)
        print(f"{regime} seed={outer_seed} | FOLD {fold_id}/{N_FOLDS}", flush=True)
        print("-" * 100, flush=True)

        X_fit      = X_train_full[fit_idx]
        y_sol_fit  = y_sol_train_full[fit_idx]
        y_logs_fit = y_logs_train_full[fit_idx]

        X_val      = X_train_full[val_idx]
        y_sol_val  = y_sol_train_full[val_idx]
        y_logs_val = y_logs_train_full[val_idx]

        print(f"fold_fit_n={len(fit_idx)} fold_val_n={len(val_idx)}", flush=True)
        print("Inner CV mode: row-level KFold, matched to original Direct LogS rerun script.", flush=True)

        scaler = fit_scaler_on_train(X_fit)

        fit_ds  = ArrayDataset(X_fit,  y_sol_fit,  y_logs_fit,  scaler)
        val_ds  = ArrayDataset(X_val,  y_sol_val,  y_logs_val,  scaler)
        test_ds = ArrayDataset(X_test, y_sol_test, y_logs_test, scaler)

        pin_memory = (device.type == "cuda")

        fit_loader = DataLoader(
            fit_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=(NUM_WORKERS > 0),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=(NUM_WORKERS > 0),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=(NUM_WORKERS > 0),
        )

        seed_everything(outer_seed * 1000 + fold_id)

        model = SingleTargetMLP(
            input_dim=len(feature_cols),
            model_size=MODEL_SIZE,
        ).to(device)

        model = train_model(model, fit_loader, val_loader, device, fold_id=fold_id)

        eval_criterion = get_loss_criterion(LOSS_FUNCTION)
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            eval_criterion,
            USE_AMP and device.type == "cuda",
        )

        fold_pred_logs.append(test_metrics["y_pred_logs"])
        fold_pred_sol.append(test_metrics["y_pred_sol_from_logs"])

        fold_rows.append({
            "regime": regime,
            "outer_seed": outer_seed,
            "fold": fold_id,
            "test_n": int(len(X_test)),
            "test_logs_R2": float(test_metrics["logs_r2"]),
            "test_logs_MAE": float(test_metrics["logs_mae"]),
            "test_logs_RMSE": float(test_metrics["logs_rmse"]),
            "test_transfer_sol_R2": float(test_metrics["transfer_sol_r2"]),
            "test_transfer_sol_MAE": float(test_metrics["transfer_sol_mae"]),
            "test_transfer_sol_RMSE": float(test_metrics["transfer_sol_rmse"]),
            "test_loss": float(test_metrics["loss"]),
        })

        print(
            f"[Fold {fold_id}] TEST LogS R2={test_metrics['logs_r2']:.6f} "
            f"MAE={test_metrics['logs_mae']:.6f} RMSE={test_metrics['logs_rmse']:.6f}",
            flush=True,
        )

        del model, fit_loader, val_loader, test_loader, fit_ds, val_ds, test_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / f"{regime}_seed{outer_seed}_fold_metrics.csv", index=False)

    fold_pred_logs = np.vstack(fold_pred_logs)
    fold_pred_sol = np.vstack(fold_pred_sol)

    op_pred_logs = fold_pred_logs.mean(axis=0)
    op_pred_sol_from_logs = safe_pow10(op_pred_logs)

    op_logs = regression_metrics(y_logs_test, op_pred_logs)
    op_sol = regression_metrics(y_sol_test, op_pred_sol_from_logs)

    pred_meta = test_df[[SOLUTE_COL, SOLVENT_COL, "Temperature_K", Y_SOL, Y_LOGS]].copy()
    pred_meta["y_pred_logs_operational"] = op_pred_logs
    pred_meta["y_pred_sol_from_logs_operational"] = op_pred_sol_from_logs
    for i in range(N_FOLDS):
        pred_meta[f"y_pred_logs_fold_{i+1}"] = fold_pred_logs[i]

    pred_meta.to_csv(
        PRED_DIR / f"{regime}_seed{outer_seed}_direct_logs_operational_predictions.csv",
        index=False,
    )

    pd.DataFrame({
        "heldout_component": sorted(test_df[outer_group_col].astype(str).str.strip().unique())
    }).to_csv(
        OUT_DIR / f"{regime}_seed{outer_seed}_heldout_components.csv",
        index=False,
    )

    result = {
        "regime": regime,
        "outer_seed": outer_seed,
        "outer_group_col": outer_group_col,
        "outer_test_size_intended": OUTER_TEST_SIZE,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "achieved_test_fraction": float(len(test_df) / len(df)),
        "train_solutes": int(train_df[SOLUTE_COL].nunique()),
        "test_solutes": int(test_df[SOLUTE_COL].nunique()),
        "train_solvents": int(train_df[SOLVENT_COL].nunique()),
        "test_solvents": int(test_df[SOLVENT_COL].nunique()),
        "train_pairs": int(train_pairs.nunique()),
        "test_pairs": int(test_pairs.nunique()),
        "shared_solutes": int(shared_solutes),
        "shared_solvents": int(shared_solvents),
        "shared_pairs": int(shared_pairs),
        "operational_logs_R2": op_logs["r2"],
        "operational_logs_MAE": op_logs["mae"],
        "operational_logs_RMSE": op_logs["rmse"],
        "operational_transfer_sol_R2": op_sol["r2"],
        "operational_transfer_sol_MAE": op_sol["mae"],
        "operational_transfer_sol_RMSE": op_sol["rmse"],
        "fold_logs_R2_mean": mean_sd(fold_df["test_logs_R2"])[0],
        "fold_logs_R2_sd": mean_sd(fold_df["test_logs_R2"])[1],
        "fold_logs_MAE_mean": mean_sd(fold_df["test_logs_MAE"])[0],
        "fold_logs_MAE_sd": mean_sd(fold_df["test_logs_MAE"])[1],
        "fold_logs_RMSE_mean": mean_sd(fold_df["test_logs_RMSE"])[0],
        "fold_logs_RMSE_sd": mean_sd(fold_df["test_logs_RMSE"])[1],
    }

    print(
        f"\nOPERATIONAL {regime} seed={outer_seed}: "
        f"LogS R2={op_logs['r2']:.6f}, MAE={op_logs['mae']:.6f}, RMSE={op_logs['rmse']:.6f}",
        flush=True,
    )

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    device = get_device()
    print("device:", device, flush=True)
    if device.type == "cuda":
        print("cuda_name:", torch.cuda.get_device_name(0), flush=True)

    print(f"\nLOSS_FUNCTION: {LOSS_FUNCTION}", flush=True)
    print("MODEL: SingleTarget Direct LogS repeated strict component resampling", flush=True)

    train_random = load_and_prepare_dataframe(TRAIN_RANDOM)
    test_random  = load_and_prepare_dataframe(TEST_RANDOM)

    df = pd.concat([train_random, test_random], axis=0, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    print(f"\nRows reconstructed from random split: {before}", flush=True)
    print(f"Rows after drop_duplicates: {after}", flush=True)

    feature_cols = get_feature_columns(df)

    print(f"\nn_features={len(feature_cols)}", flush=True)
    print(f"feature_names first: {feature_cols[0]}", flush=True)
    print(f"feature_names last : {feature_cols[-1]}", flush=True)

    all_results = []

    for regime in ["strict_unseen_solute", "strict_unseen_solvent"]:
        for outer_seed in OUTER_SEEDS:
            all_results.append(
                evaluate_one_outer_seed(
                    df=df,
                    feature_cols=feature_cols,
                    regime=regime,
                    outer_seed=outer_seed,
                    device=device,
                )
            )

    summary_df = pd.DataFrame(all_results)
    summary_path = OUT_DIR / "repeated_strict_component_resampling_direct_logs_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    agg_rows = []
    for regime, g in summary_df.groupby("regime"):
        agg_rows.append({
            "regime": regime,
            "n_outer_component_selection_seeds": int(len(g)),
            "logs_R2_mean_across_outer_seeds": float(g["operational_logs_R2"].mean()),
            "logs_R2_sd_across_outer_seeds": float(g["operational_logs_R2"].std(ddof=1)),
            "logs_R2_min": float(g["operational_logs_R2"].min()),
            "logs_R2_max": float(g["operational_logs_R2"].max()),
            "logs_MAE_mean_across_outer_seeds": float(g["operational_logs_MAE"].mean()),
            "logs_MAE_sd_across_outer_seeds": float(g["operational_logs_MAE"].std(ddof=1)),
            "logs_MAE_min": float(g["operational_logs_MAE"].min()),
            "logs_MAE_max": float(g["operational_logs_MAE"].max()),
            "logs_RMSE_mean_across_outer_seeds": float(g["operational_logs_RMSE"].mean()),
            "logs_RMSE_sd_across_outer_seeds": float(g["operational_logs_RMSE"].std(ddof=1)),
            "logs_RMSE_min": float(g["operational_logs_RMSE"].min()),
            "logs_RMSE_max": float(g["operational_logs_RMSE"].max()),
            "transfer_sol_R2_mean_across_outer_seeds": float(g["operational_transfer_sol_R2"].mean()),
            "transfer_sol_R2_sd_across_outer_seeds": float(g["operational_transfer_sol_R2"].std(ddof=1)),
            "transfer_sol_R2_min": float(g["operational_transfer_sol_R2"].min()),
            "transfer_sol_R2_max": float(g["operational_transfer_sol_R2"].max()),
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_path = OUT_DIR / "repeated_strict_component_resampling_direct_logs_aggregate.csv"
    agg_df.to_csv(agg_path, index=False)

    report = {
        "purpose": "Repeated strict unseen-solute and strict unseen-solvent Direct LogS evaluations over multiple outer component-selection seeds.",
        "outer_component_selection_seeds": OUTER_SEEDS,
        "outer_test_size": OUTER_TEST_SIZE,
        "n_inner_folds": N_FOLDS,
        "inner_cv": "row-level KFold matched to original Direct LogS rerun script",
        "loss_function": LOSS_FUNCTION,
        "model_size": MODEL_SIZE,
        "principal_metric_definition": "R2, MAE, and RMSE computed from the final fold-averaged operational prediction vector.",
        "fold_sd_definition": "Fold mean±SD values summarize variability among fold-specific models within each fixed outer split only.",
        "summary_csv": str(summary_path),
        "aggregate_csv": str(agg_path),
    }

    with open(OUT_DIR / "repeated_strict_component_resampling_direct_logs_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "#" * 120, flush=True)
    print("PER-SEED OPERATIONAL SUMMARY", flush=True)
    print("#" * 120, flush=True)
    print(summary_df[[
        "regime",
        "outer_seed",
        "train_rows",
        "test_rows",
        "achieved_test_fraction",
        "shared_solutes",
        "shared_solvents",
        "shared_pairs",
        "operational_logs_R2",
        "operational_logs_MAE",
        "operational_logs_RMSE",
        "operational_transfer_sol_R2",
    ]].to_string(index=False), flush=True)

    print("\n" + "#" * 120, flush=True)
    print("AGGREGATE ACROSS OUTER COMPONENT-SELECTION SEEDS", flush=True)
    print("#" * 120, flush=True)
    print(agg_df.to_string(index=False), flush=True)

    print(f"\nSaved outputs to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
