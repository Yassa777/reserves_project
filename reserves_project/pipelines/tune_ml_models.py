#!/usr/bin/env python3
"""Time-series CV tuning for XGBoost and LSTM models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False

from reserves_project.config.paths import PROJECT_ROOT
from reserves_project.config.varsets import TARGET_VAR, TRAIN_END, VALID_END, VARSET_ORDER, OUTPUT_DIR
from scripts.academic.ml_models import create_lag_features

OUTPUT_BASE = PROJECT_ROOT / "data" / "model_verification" / "ml_tuning"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def load_varset_data(varset: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"varset_{varset}" / "vecm_levels.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    return df


def _tscv_splits(index: pd.DatetimeIndex, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return list(splitter.split(np.arange(len(index))))


def tune_xgboost(df: pd.DataFrame, param_grid: List[Dict], n_splits: int = 3) -> Dict:
    if not HAS_XGB:
        raise RuntimeError("XGBoost not available")

    features_df = create_lag_features(df, TARGET_VAR)
    train_mask = features_df.index <= TRAIN_END
    X = features_df.loc[train_mask].drop(columns=["target"])
    y = features_df.loc[train_mask, "target"]

    splits = _tscv_splits(X.index, n_splits=n_splits)

    best_params = None
    best_rmse = np.inf

    for params in param_grid:
        rmses = []
        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    return {"best_params": best_params, "best_rmse": best_rmse}


def _build_sequences(data: np.ndarray, seq_length: int):
    X, y, idx = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 0])
        idx.append(i + seq_length)
    return np.array(X), np.array(y), np.array(idx)


def tune_lstm(df: pd.DataFrame, param_grid: List[Dict], n_splits: int = 3) -> Dict:
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available")

    data = df[[TARGET_VAR] + [c for c in df.columns if c != TARGET_VAR]].dropna()
    train_mask = data.index <= TRAIN_END
    data_train = data.loc[train_mask]

    best_params = None
    best_rmse = np.inf

    for params in param_grid:
        seq_length = params.get("seq_length", 6)
        rmses = []

        splits = _tscv_splits(data_train.index, n_splits=n_splits)
        for train_idx, val_idx in splits:
            train_slice = data_train.iloc[train_idx]
            val_slice = data_train.iloc[val_idx]

            scaler = MinMaxScaler()
            scaler.fit(train_slice)

            scaled_full = scaler.transform(pd.concat([train_slice, val_slice]))
            X_all, y_all, idx_all = _build_sequences(scaled_full, seq_length)

            # Split sequences based on index mapping
            train_limit = len(train_slice)
            train_mask_seq = idx_all < train_limit
            val_mask_seq = (idx_all >= train_limit) & (idx_all < len(train_slice) + len(val_slice))

            X_train, y_train = X_all[train_mask_seq], y_all[train_mask_seq]
            X_val, y_val = X_all[val_mask_seq], y_all[val_mask_seq]

            if len(X_train) < 5 or len(X_val) < 5:
                continue

            model = Sequential([
                LSTM(params.get("units", 32), activation="tanh", input_shape=(seq_length, X_train.shape[2])),
                Dropout(params.get("dropout", 0.3)),
                Dense(max(8, params.get("units", 32) // 2), activation="relu"),
                Dense(1),
            ])

            optimizer = tf.keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.001))
            model.compile(optimizer=optimizer, loss="mse")
            early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

            model.fit(
                X_train,
                y_train,
                epochs=params.get("epochs", 60),
                batch_size=params.get("batch_size", 16),
                verbose=0,
                callbacks=[early_stop],
            )

            preds = model.predict(X_val, verbose=0).flatten()
            # Inverse transform
            y_pred_full = np.zeros((len(preds), scaled_full.shape[1]))
            y_pred_full[:, 0] = preds
            y_pred = scaler.inverse_transform(y_pred_full)[:, 0]

            y_val_full = np.zeros((len(y_val), scaled_full.shape[1]))
            y_val_full[:, 0] = y_val
            y_val_inv = scaler.inverse_transform(y_val_full)[:, 0]

            rmse = np.sqrt(mean_squared_error(y_val_inv, y_pred))
            rmses.append(rmse)

        if rmses:
            mean_rmse = float(np.mean(rmses))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = params

    return {"best_params": best_params, "best_rmse": best_rmse}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune XGBoost and LSTM with time-series CV")
    parser.add_argument("--varset", default="parsimonious")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()

    df = load_varset_data(args.varset)

    if not args.skip_xgb:
        xgb_grid = [
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
            {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
            {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 0.8, "random_state": 42},
        ]
        result = tune_xgboost(df, xgb_grid)
        out_path = OUTPUT_BASE / f"xgb_best_params_{args.varset}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved XGBoost tuning: {out_path}")

    if not args.skip_lstm:
        lstm_grid = [
            {"seq_length": 6, "units": 32, "dropout": 0.3, "learning_rate": 0.001, "epochs": 60, "batch_size": 16},
            {"seq_length": 12, "units": 32, "dropout": 0.2, "learning_rate": 0.001, "epochs": 60, "batch_size": 16},
            {"seq_length": 6, "units": 16, "dropout": 0.2, "learning_rate": 0.001, "epochs": 60, "batch_size": 16},
        ]
        result = tune_lstm(df, lstm_grid)
        out_path = OUTPUT_BASE / f"lstm_best_params_{args.varset}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved LSTM tuning: {out_path}")


if __name__ == "__main__":
    main()
