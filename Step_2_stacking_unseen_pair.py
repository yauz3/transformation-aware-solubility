#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd

from typing import List
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
TRAIN_OUT = "train_unseen_pair_maccs_map4_padel_sen_features.csv"
TEST_OUT  = "test_unseen_pair_maccs_map4_padel_sen_features.csv"

WORK_DIR = "dl_work_three_stage_residual_stacking"
os.makedirs(WORK_DIR, exist_ok=True)

PRED_DIR = os.path.join(WORK_DIR, "fold_predictions")
os.makedirs(PRED_DIR, exist_ok=True)

SOLUTE_COL  = "SMILES_Solute"
SOLVENT_COL = "SMILES_Solvent"
Y_SOL       = "Solubility(mol/L)"
Y_LOGS      = "LogS(mol/L)"
TEMP_COL    = "Temperature_K"

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

N_FOLDS      = 5
RANDOM_STATE = 42
USE_GROUP_PAIR_SPLIT = True

EPOCHS       = 100
BATCH_SIZE   = 1024
LR           = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS  = 4
PATIENCE     = 7

MODEL_SIZE = "medium"
USE_AMP = True
CLIP_ABS = 1e9

SUMMARY_METRICS_JSON = os.path.join(WORK_DIR, "three_stage_cv5_summary_metrics.json")
FEATURE_META_JSON    = os.path.join(WORK_DIR, "three_stage_feature_meta.json")
FOLD_METRICS_CSV     = os.path.join(WORK_DIR, "three_stage_cv5_fold_metrics.csv")


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
    loss_name = str(loss_name).lower().strip()
    if loss_name == "mae":
        return DifferentiableMAELoss()
    if loss_name == "rmse":
        return DifferentiableRMSELoss()
    if loss_name == "r2":
        return DifferentiableR2Loss()
    raise ValueError("LOSS_FUNCTION must be 'mae', 'rmse', or 'r2'")


def get_selection_mode(loss_name: str):
    loss_name = str(loss_name).lower().strip()
    if loss_name == "mae":
        return "mae", "min"
    if loss_name == "rmse":
        return "rmse", "min"
    if loss_name == "r2":
        return "r2", "max"
    raise ValueError("LOSS_FUNCTION must be 'mae', 'rmse', or 'r2'")


# =========================================================
# HELPERS
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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
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
    elif model_size == "deep":
        return [4096, 2048, 1024, 512, 256], 0.20
    else:
        return [4096, 2048, 1024, 512, 256, 128, 64, 32], 0.15


# =========================================================
# DATA
# =========================================================
def load_and_prepare_dataframe(csv_path: str) -> pd.DataFrame:
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
        raise ValueError("No feature columns remain after EXCLUDE_COLS.")
    return cols


def build_group_pairs(df: pd.DataFrame) -> np.ndarray:
    if USE_GROUP_PAIR_SPLIT:
        return (
            df[SOLUTE_COL].astype(str).str.strip()
            + "||"
            + df[SOLVENT_COL].astype(str).str.strip()
        ).values
    return np.array([f"row_{i}" for i in range(len(df))], dtype=object)


def assert_no_group_leakage(train_groups, test_groups, label):
    overlap = set(train_groups).intersection(set(test_groups))
    if len(overlap) > 0:
        raise RuntimeError(
            f"{label}: leakage detected — {len(overlap)} overlapping group-pairs found."
        )


def build_arrays_from_df(df, feature_cols):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X = sanitize_numeric_array(X)

    y_sol = pd.to_numeric(df[Y_SOL], errors="coerce").to_numpy(dtype=np.float32)
    y_logs = pd.to_numeric(df[Y_LOGS], errors="coerce").to_numpy(dtype=np.float32)
    groups = build_group_pairs(df)

    return X, y_sol, y_logs, groups


# =========================================================
# FEATURE SELECTION
# =========================================================
def infer_feature_groups(feature_cols):
    temp_cols = []
    solute_cols = []
    solvent_cols = []
    other_cols = []

    for c in feature_cols:
        cl = c.lower()

        if c == TEMP_COL or "temperature" in cl:
            temp_cols.append(c)
        elif cl.startswith("solute_"):
            solute_cols.append(c)
        elif cl.startswith("solvent_"):
            solvent_cols.append(c)
        else:
            other_cols.append(c)

    return {
        "temperature": temp_cols,
        "solute": solute_cols,
        "solvent": solvent_cols,
        "other": other_cols,
    }


def col_indices(feature_cols, selected_cols):
    idx = []
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    for c in selected_cols:
        if c in col_to_idx:
            idx.append(col_to_idx[c])
    return np.array(idx, dtype=int)


def build_stage_features(
    X_base,
    feature_cols,
    stage,
    previous_prediction=None,
):
    groups = infer_feature_groups(feature_cols)

    if stage == 1:
        selected = groups["solute"] + groups["temperature"]

    elif stage == 2:
        selected = groups["solvent"] + groups["temperature"]

    elif stage == 3:
        selected = groups["solute"] + groups["solvent"] + groups["temperature"]

    else:
        raise ValueError("stage must be 1, 2, or 3")

    idx = col_indices(feature_cols, selected)

    if len(idx) == 0:
        raise RuntimeError(f"No features selected for stage {stage}.")

    X_stage = X_base[:, idx]

    if previous_prediction is not None:
        previous_prediction = np.asarray(previous_prediction, dtype=np.float32).reshape(-1, 1)
        X_stage = np.concatenate([X_stage, previous_prediction], axis=1)

    return X_stage.astype(np.float32), selected


# =========================================================
# SCALER / DATASET
# =========================================================
def fit_scaler_on_train(X_train):
    X_train = sanitize_numeric_array(X_train)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler(copy=True)
    scaler.fit(X_train)
    return scaler


class ArrayDataset(Dataset):
    def __init__(self, X, y, scaler):
        self.X = X
        self.y = y
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = np.array(self.X[i], dtype=np.float32, copy=True)
        x = sanitize_numeric_array(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.scaler.transform(x.reshape(1, -1)).reshape(-1).astype(np.float32)

        y = float(self.y[i])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# =========================================================
# MODEL
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


def make_loaders(X_fit, y_fit, X_val, y_val, scaler, device):
    fit_ds = ArrayDataset(X_fit, y_fit, scaler)
    val_ds = ArrayDataset(X_val, y_val, scaler)

    pin_memory = device.type == "cuda"

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

    return fit_loader, val_loader


def predict_model(model, X, scaler, device):
    ds = ArrayDataset(X, np.zeros(len(X), dtype=np.float32), scaler)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model.eval()
    preds = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                out = model(xb)
            preds.append(out.detach().cpu().numpy())

    return np.concatenate(preds)


def evaluate_prediction(y_true_logs, y_pred_logs, y_true_sol):
    y_pred_sol = safe_pow10(y_pred_logs)

    logs_metrics = regression_metrics(y_true_logs, y_pred_logs)
    sol_metrics = regression_metrics(y_true_sol, y_pred_sol)

    return {
        "logs_r2": logs_metrics["r2"],
        "logs_mae": logs_metrics["mae"],
        "logs_rmse": logs_metrics["rmse"],
        "sol_r2": sol_metrics["r2"],
        "sol_mae": sol_metrics["mae"],
        "sol_rmse": sol_metrics["rmse"],
    }


def train_single_model(X_fit, y_fit, X_val, y_val, device, stage_name):
    scaler = fit_scaler_on_train(X_fit)

    fit_loader, val_loader = make_loaders(
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
        scaler=scaler,
        device=device,
    )

    model = SingleTargetMLP(
        input_dim=X_fit.shape[1],
        model_size=MODEL_SIZE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = get_loss_criterion(LOSS_FUNCTION)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    metric_key, mode = get_selection_mode(LOSS_FUNCTION)
    best_score = np.inf if mode == "min" else -np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in fit_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_losses.append(float(loss.detach().cpu().item()))

        val_pred = predict_model(model, X_val, scaler, device)
        val_metrics = regression_metrics(y_val, val_pred)

        current = val_metrics[metric_key]

        print(
            f"[{stage_name}] epoch={epoch:03d} "
            f"train_loss={np.mean(train_losses):.6f} "
            f"val_R2={val_metrics['r2']:.4f} "
            f"val_MAE={val_metrics['mae']:.6f} "
            f"val_RMSE={val_metrics['rmse']:.6f}"
        )

        improved = current < best_score if mode == "min" else current > best_score

        if improved:
            best_score = current
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[{stage_name}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    del fit_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, scaler


# =========================================================
# MAIN
# =========================================================
def main():
    device = get_device()
    print("device:", device)
    if device.type == "cuda":
        print("cuda_name:", torch.cuda.get_device_name(0))

    print("\nMODEL: Three-stage residual stacking in LogS space")
    print("Stage 1: solute + temperature -> base LogS")
    print("Stage 2: solvent + temperature + Stage1 prediction -> residual correction")
    print("Stage 3: solute + solvent + temperature + Stage2 prediction -> final correction")

    train_df = load_and_prepare_dataframe(TRAIN_OUT)
    test_df = load_and_prepare_dataframe(TEST_OUT)

    feature_cols_train = get_feature_columns(train_df)
    feature_cols_test = get_feature_columns(test_df)

    if feature_cols_train != feature_cols_test:
        raise ValueError("Train and test feature columns do not match exactly.")

    feature_cols = feature_cols_train

    train_groups = build_group_pairs(train_df)
    test_groups = build_group_pairs(test_df)
    assert_no_group_leakage(train_groups, test_groups, label="Train-vs-Test")

    X_train_full, y_sol_train_full, y_logs_train_full, groups_train_full = build_arrays_from_df(
        train_df, feature_cols
    )
    X_test, y_sol_test, y_logs_test, groups_test = build_arrays_from_df(
        test_df, feature_cols
    )

    fg = infer_feature_groups(feature_cols)
    print("\nFeature groups")
    print(f"solute features     : {len(fg['solute'])}")
    print(f"solvent features    : {len(fg['solvent'])}")
    print(f"temperature features: {len(fg['temperature'])}")
    print(f"other features      : {len(fg['other'])}")

    if len(fg["temperature"]) == 0:
        print("WARNING: No explicit temperature feature detected.")

    with open(FEATURE_META_JSON, "w") as f:
        json.dump(
            {
                "train_file": TRAIN_OUT,
                "test_file": TEST_OUT,
                "model_type": "three_stage_residual_stacking",
                "target": Y_LOGS,
                "n_features_total": len(feature_cols),
                "n_solute_features": len(fg["solute"]),
                "n_solvent_features": len(fg["solvent"]),
                "n_temperature_features": len(fg["temperature"]),
                "n_other_features": len(fg["other"]),
                "feature_cols": feature_cols,
            },
            f,
            indent=2,
        )

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rows = []

    for fold_id, (fit_idx, val_idx) in enumerate(
        gkf.split(X_train_full, y_logs_train_full, groups=groups_train_full),
        start=1,
    ):
        print("\n" + "=" * 120)
        print(f"FOLD {fold_id}/{N_FOLDS}")
        print("=" * 120)

        X_fit_base = X_train_full[fit_idx]
        X_val_base = X_train_full[val_idx]

        y_fit_logs = y_logs_train_full[fit_idx]
        y_val_logs = y_logs_train_full[val_idx]

        y_fit_sol = y_sol_train_full[fit_idx]
        y_val_sol = y_sol_train_full[val_idx]

        groups_fit = groups_train_full[fit_idx]
        groups_val = groups_train_full[val_idx]
        assert_no_group_leakage(groups_fit, groups_val, label=f"Fold {fold_id} Fit-vs-Val")

        # -------------------------
        # Stage 1
        # -------------------------
        X1_fit, stage1_features = build_stage_features(
            X_fit_base, feature_cols, stage=1, previous_prediction=None
        )
        X1_val, _ = build_stage_features(
            X_val_base, feature_cols, stage=1, previous_prediction=None
        )
        X1_test, _ = build_stage_features(
            X_test, feature_cols, stage=1, previous_prediction=None
        )

        model1, scaler1 = train_single_model(
            X1_fit, y_fit_logs, X1_val, y_val_logs, device, f"Fold {fold_id} | Stage 1"
        )

        pred1_fit = predict_model(model1, X1_fit, scaler1, device)
        pred1_val = predict_model(model1, X1_val, scaler1, device)
        pred1_test = predict_model(model1, X1_test, scaler1, device)

        # -------------------------
        # Stage 2 residual
        # -------------------------
        residual2_fit = y_fit_logs - pred1_fit
        residual2_val = y_val_logs - pred1_val

        X2_fit, stage2_features = build_stage_features(
            X_fit_base, feature_cols, stage=2, previous_prediction=pred1_fit
        )
        X2_val, _ = build_stage_features(
            X_val_base, feature_cols, stage=2, previous_prediction=pred1_val
        )
        X2_test, _ = build_stage_features(
            X_test, feature_cols, stage=2, previous_prediction=pred1_test
        )

        model2, scaler2 = train_single_model(
            X2_fit, residual2_fit, X2_val, residual2_val, device, f"Fold {fold_id} | Stage 2"
        )

        delta2_fit = predict_model(model2, X2_fit, scaler2, device)
        delta2_val = predict_model(model2, X2_val, scaler2, device)
        delta2_test = predict_model(model2, X2_test, scaler2, device)

        pred2_fit = pred1_fit + delta2_fit
        pred2_val = pred1_val + delta2_val
        pred2_test = pred1_test + delta2_test

        # -------------------------
        # Stage 3 residual
        # -------------------------
        residual3_fit = y_fit_logs - pred2_fit
        residual3_val = y_val_logs - pred2_val

        X3_fit, stage3_features = build_stage_features(
            X_fit_base, feature_cols, stage=3, previous_prediction=pred2_fit
        )
        X3_val, _ = build_stage_features(
            X_val_base, feature_cols, stage=3, previous_prediction=pred2_val
        )
        X3_test, _ = build_stage_features(
            X_test, feature_cols, stage=3, previous_prediction=pred2_test
        )

        model3, scaler3 = train_single_model(
            X3_fit, residual3_fit, X3_val, residual3_val, device, f"Fold {fold_id} | Stage 3"
        )

        delta3_val = predict_model(model3, X3_val, scaler3, device)
        delta3_test = predict_model(model3, X3_test, scaler3, device)

        pred3_val = pred2_val + delta3_val
        pred3_test = pred2_test + delta3_test

        # -------------------------
        # Metrics
        # -------------------------
        val_m1 = evaluate_prediction(y_val_logs, pred1_val, y_val_sol)
        val_m2 = evaluate_prediction(y_val_logs, pred2_val, y_val_sol)
        val_m3 = evaluate_prediction(y_val_logs, pred3_val, y_val_sol)

        test_m1 = evaluate_prediction(y_logs_test, pred1_test, y_sol_test)
        test_m2 = evaluate_prediction(y_logs_test, pred2_test, y_sol_test)
        test_m3 = evaluate_prediction(y_logs_test, pred3_test, y_sol_test)

        print("\nTEST RESULTS")
        print(f"Stage 1 LogS R2={test_m1['logs_r2']:.4f} MAE={test_m1['logs_mae']:.6f} RMSE={test_m1['logs_rmse']:.6f}")
        print(f"Stage 2 LogS R2={test_m2['logs_r2']:.4f} MAE={test_m2['logs_mae']:.6f} RMSE={test_m2['logs_rmse']:.6f}")
        print(f"Stage 3 LogS R2={test_m3['logs_r2']:.4f} MAE={test_m3['logs_mae']:.6f} RMSE={test_m3['logs_rmse']:.6f}")

        # -------------------------
        # Save models
        # -------------------------
        torch.save(
            {
                "model_state_dict": model1.state_dict(),
                "input_dim": X1_fit.shape[1],
                "model_size": MODEL_SIZE,
                "stage": 1,
                "stage_features": stage1_features,
                "scaler_mean": scaler1.mean_.astype(np.float32),
                "scaler_scale": scaler1.scale_.astype(np.float32),
            },
            os.path.join(WORK_DIR, f"fold_{fold_id}_stage1_model.pt"),
        )

        torch.save(
            {
                "model_state_dict": model2.state_dict(),
                "input_dim": X2_fit.shape[1],
                "model_size": MODEL_SIZE,
                "stage": 2,
                "stage_features": stage2_features,
                "uses_previous_prediction": True,
                "scaler_mean": scaler2.mean_.astype(np.float32),
                "scaler_scale": scaler2.scale_.astype(np.float32),
            },
            os.path.join(WORK_DIR, f"fold_{fold_id}_stage2_model.pt"),
        )

        torch.save(
            {
                "model_state_dict": model3.state_dict(),
                "input_dim": X3_fit.shape[1],
                "model_size": MODEL_SIZE,
                "stage": 3,
                "stage_features": stage3_features,
                "uses_previous_prediction": True,
                "scaler_mean": scaler3.mean_.astype(np.float32),
                "scaler_scale": scaler3.scale_.astype(np.float32),
            },
            os.path.join(WORK_DIR, f"fold_{fold_id}_stage3_model.pt"),
        )

        # -------------------------
        # Save predictions
        # -------------------------
        pred_df = test_df[[SOLUTE_COL, SOLVENT_COL, Y_SOL, Y_LOGS]].copy().reset_index(drop=True)
        pred_df["group_pair"] = groups_test

        pred_df["y_true_logs"] = y_logs_test
        pred_df["stage1_pred_logs"] = pred1_test
        pred_df["stage2_pred_logs"] = pred2_test
        pred_df["stage3_pred_logs"] = pred3_test

        pred_df["stage2_delta_logs"] = delta2_test
        pred_df["stage3_delta_logs"] = delta3_test

        pred_df["y_true_sol"] = y_sol_test
        pred_df["stage1_pred_sol"] = safe_pow10(pred1_test)
        pred_df["stage2_pred_sol"] = safe_pow10(pred2_test)
        pred_df["stage3_pred_sol"] = safe_pow10(pred3_test)

        pred_df.to_csv(
            os.path.join(PRED_DIR, f"fold_{fold_id}_three_stage_test_predictions.csv"),
            index=False,
        )

        row = {
            "fold": fold_id,
            "fit_n": int(len(fit_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(X_test)),
        }

        for stage_name, vm, tm in [
            ("stage1", val_m1, test_m1),
            ("stage2", val_m2, test_m2),
            ("stage3", val_m3, test_m3),
        ]:
            row[f"val_{stage_name}_logs_R2"] = vm["logs_r2"]
            row[f"val_{stage_name}_logs_MAE"] = vm["logs_mae"]
            row[f"val_{stage_name}_logs_RMSE"] = vm["logs_rmse"]
            row[f"val_{stage_name}_sol_R2"] = vm["sol_r2"]
            row[f"val_{stage_name}_sol_MAE"] = vm["sol_mae"]
            row[f"val_{stage_name}_sol_RMSE"] = vm["sol_rmse"]

            row[f"test_{stage_name}_logs_R2"] = tm["logs_r2"]
            row[f"test_{stage_name}_logs_MAE"] = tm["logs_mae"]
            row[f"test_{stage_name}_logs_RMSE"] = tm["logs_rmse"]
            row[f"test_{stage_name}_sol_R2"] = tm["sol_r2"]
            row[f"test_{stage_name}_sol_MAE"] = tm["sol_mae"]
            row[f"test_{stage_name}_sol_RMSE"] = tm["sol_rmse"]

        fold_rows.append(row)

        del model1, model2, model3
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(FOLD_METRICS_CSV, index=False)

    metrics_to_summarize = []
    for stage in ["stage1", "stage2", "stage3"]:
        metrics_to_summarize.extend([
            f"test_{stage}_logs_R2",
            f"test_{stage}_logs_MAE",
            f"test_{stage}_logs_RMSE",
            f"test_{stage}_sol_R2",
            f"test_{stage}_sol_MAE",
            f"test_{stage}_sol_RMSE",
        ])

    summary = {}
    for m in metrics_to_summarize:
        mu, sd = mean_sd(fold_df[m].values)
        summary[m] = {"mean": mu, "sd": sd}

    with open(SUMMARY_METRICS_JSON, "w") as f:
        json.dump(
            {
                "train_file": TRAIN_OUT,
                "test_file": TEST_OUT,
                "model_type": "three_stage_residual_stacking",
                "target": Y_LOGS,
                "n_folds": N_FOLDS,
                "fold_metrics": fold_rows,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print("\n" + "#" * 120)
    print("THREE-STAGE STACKING SUMMARY")
    print("#" * 120)

    for m in metrics_to_summarize:
        print(f"{m}: {summary[m]['mean']:.6f} ± {summary[m]['sd']:.6f}")

    print(f"\n✅ Saved fold metrics : {FOLD_METRICS_CSV}")
    print(f"✅ Saved summary JSON : {SUMMARY_METRICS_JSON}")
    print(f"✅ Saved predictions  : {PRED_DIR}")
    print(f"✅ Saved models       : {WORK_DIR}")


if __name__ == "__main__":
    main()
