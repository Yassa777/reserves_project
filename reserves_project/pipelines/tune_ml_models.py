#!/usr/bin/env python3
"""Time-series CV tuning for XGBoost and LSTM models."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    HAS_TF = True
except Exception:
    HAS_TF = False

from reserves_project.config.paths import PROJECT_ROOT
from reserves_project.config.varsets import OUTPUT_DIR, TARGET_VAR, TRAIN_END
from reserves_project.models.ml_models import create_lag_features
from reserves_project.utils.run_manifest import write_run_manifest

OUTPUT_BASE = PROJECT_ROOT / "data" / "model_verification" / "ml_tuning"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

DEFAULT_N_SPLITS = 3
DEFAULT_SEED = 42
SELECTION_CRITERION = "mean_validation_rmse"
LSTM_EARLY_STOPPING_MONITOR = "val_loss"
LSTM_EARLY_STOPPING_PATIENCE = 10

XGB_SEARCH_SPACE: List[Dict[str, Any]] = [
    {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
]

LSTM_SEARCH_SPACE: List[Dict[str, Any]] = [
    {
        "seq_length": 6,
        "units": 32,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "epochs": 60,
        "batch_size": 16,
    },
    {
        "seq_length": 12,
        "units": 32,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 60,
        "batch_size": 16,
    },
    {
        "seq_length": 6,
        "units": 16,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 60,
        "batch_size": 16,
    },
]


def load_varset_data(varset: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"varset_{varset}" / "vecm_levels.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TF:
        tf.keras.utils.set_random_seed(seed)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    return value


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _serialize_value(val) for key, val in params.items()}


def _latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = str(text)
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _format_number(value: Any, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _tscv_splits(
    index: pd.DatetimeIndex,
    n_splits: int = DEFAULT_N_SPLITS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return list(splitter.split(np.arange(len(index))))


def _fold_metadata(
    index: pd.DatetimeIndex,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        rows.append(
            {
                "fold": fold_id,
                "train_start": index[train_idx][0],
                "train_end": index[train_idx][-1],
                "validation_start": index[val_idx][0],
                "validation_end": index[val_idx][-1],
                "train_obs": int(len(train_idx)),
                "validation_obs": int(len(val_idx)),
            }
        )
    return [_serialize_value(row) for row in rows]


def _build_sequences(data: np.ndarray, seq_length: int):
    X, y, idx = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 0])
        idx.append(i + seq_length)
    return np.array(X), np.array(y), np.array(idx)


def _xgb_search_space_note() -> str:
    return (
        "Three explicit candidate configurations: "
        "C1=(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8), "
        "C2=(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8), "
        "C3=(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=1.0); "
        "colsample_bytree=0.8 and random_state=42 in all cases."
    )


def _lstm_search_space_note() -> str:
    return (
        "Three explicit candidate configurations: "
        "C1=(seq_length=6, units=32, dropout=0.3, learning_rate=0.001, epochs=60, batch_size=16), "
        "C2=(seq_length=12, units=32, dropout=0.2, learning_rate=0.001, epochs=60, batch_size=16), "
        "C3=(seq_length=6, units=16, dropout=0.2, learning_rate=0.001, epochs=60, batch_size=16); "
        f"early stopping monitors {LSTM_EARLY_STOPPING_MONITOR} with patience={LSTM_EARLY_STOPPING_PATIENCE}."
    )


def _fold_design_note(summary: Dict[str, Any]) -> str:
    parts = []
    for fold in summary["fold_design"]:
        parts.append(
            "F{fold}: train {train_start} to {train_end} (n={train_obs}), "
            "validate {validation_start} to {validation_end} (n={validation_obs})".format(**fold)
        )
    note = "; ".join(parts)
    if summary["model"] == "LSTM":
        note += "; usable sequence counts vary by seq_length and are reported in the CV results CSV."
    return note


def _build_design_rows(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "Model": summary["model"],
                "Varset": summary["varset"],
                "Search_Space": summary["search_space_note"],
                "Fold_Design": _fold_design_note(summary),
                "Selection_Rule": summary["selection_criterion"].replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def _build_selected_rows(summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        params = summary.get("best_params") or {}
        fixed = summary.get("fixed_training_config") or {}
        rows.append(
            {
                "Model": summary["model"],
                "Varset": summary["varset"],
                "learning_rate": params.get("learning_rate"),
                "max_depth": params.get("max_depth"),
                "n_estimators": params.get("n_estimators"),
                "subsample": params.get("subsample"),
                "epochs": params.get("epochs"),
                "hidden_units": params.get("units"),
                "dropout": params.get("dropout"),
                "early_stopping": fixed.get("early_stopping"),
                "mean_cv_rmse": summary.get("best_rmse"),
            }
        )
    return pd.DataFrame(rows)


def _design_table_latex(design_df: pd.DataFrame) -> str:
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Machine-learning hyperparameter search design}",
        r"\label{tab:ml_tuning_design}",
        r"\begin{tabular}{llp{5.8cm}p{5.6cm}p{2.7cm}}",
        r"\toprule",
        r"Model & Varset & Search space & Fold design & Selection rule \\",
        r"\midrule",
    ]

    for _, row in design_df.iterrows():
        latex.append(
            " & ".join(
                [
                    _latex_escape(row["Model"]),
                    _latex_escape(row["Varset"]),
                    _latex_escape(row["Search_Space"]),
                    _latex_escape(row["Fold_Design"]),
                    _latex_escape(row["Selection_Rule"]),
                ]
            )
            + r" \\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item Notes: Fold design is reported on the effective estimation sample after feature construction. ",
            r"For LSTM, sequence-effective train counts differ by sequence length and are therefore reported in the exported CV results.",
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)


def _selected_table_latex(selected_df: pd.DataFrame) -> str:
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Selected machine-learning hyperparameters}",
        r"\label{tab:ml_tuning_selected}",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"Model & Varset & LR & Depth & Trees & Subsample & Epochs & Units & Dropout & Early stopping & Mean CV RMSE \\",
        r"\midrule",
    ]

    for _, row in selected_df.iterrows():
        latex.append(
            " & ".join(
                [
                    _latex_escape(row["Model"]),
                    _latex_escape(row["Varset"]),
                    _latex_escape(_format_number(row["learning_rate"], 3)),
                    _latex_escape(_format_number(row["max_depth"], 0)),
                    _latex_escape(_format_number(row["n_estimators"], 0)),
                    _latex_escape(_format_number(row["subsample"], 1)),
                    _latex_escape(_format_number(row["epochs"], 0)),
                    _latex_escape(_format_number(row["hidden_units"], 0)),
                    _latex_escape(_format_number(row["dropout"], 1)),
                    _latex_escape(row["early_stopping"] or "--"),
                    _latex_escape(_format_number(row["mean_cv_rmse"], 1)),
                ]
            )
            + r" \\"
        )

    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item Notes: LR = learning rate. Dashes indicate parameters not applicable to the given model family. ",
            r"Mean CV RMSE is the selection objective reported in Tables~\ref{tab:ml_tuning_design} and~\ref{tab:ml_tuning_selected}.",
            r"\end{tablenotes}",
            r"\end{table}",
        ]
    )
    return "\n".join(latex)


def _load_saved_summaries(output_dir: Path) -> List[Dict[str, Any]]:
    summaries = []
    for path in sorted(output_dir.glob("*_tuning_summary_*.json")):
        with open(path, "r") as f:
            summaries.append(json.load(f))
    model_order = {"XGBoost": 0, "LSTM": 1}
    varset_order = {"parsimonious": 0, "bop": 1, "monetary": 2, "pca": 3, "full": 4}
    summaries.sort(
        key=lambda item: (
            model_order.get(item["model"], 99),
            varset_order.get(item["varset"], 99),
            item["varset"],
        )
    )
    return summaries


def write_appendix_table_artifacts(output_dir: Path) -> Dict[str, str]:
    summaries = _load_saved_summaries(output_dir)
    if not summaries:
        return {}

    design_df = _build_design_rows(summaries)
    selected_df = _build_selected_rows(summaries)

    design_csv = output_dir / "ml_tuning_design_summary.csv"
    selected_csv = output_dir / "ml_tuning_selected_summary.csv"
    design_tex = output_dir / "table_a7_ml_tuning_design.tex"
    selected_tex = output_dir / "table_a8_ml_tuning_selected.tex"

    design_df.to_csv(design_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    design_tex.write_text(_design_table_latex(design_df))
    selected_tex.write_text(_selected_table_latex(selected_df))

    return {
        "design_csv": str(design_csv),
        "selected_csv": str(selected_csv),
        "design_tex": str(design_tex),
        "selected_tex": str(selected_tex),
    }


def tune_xgboost(
    df: pd.DataFrame,
    param_grid: List[Dict[str, Any]],
    varset: str,
    n_splits: int = DEFAULT_N_SPLITS,
) -> Dict[str, Any]:
    if not HAS_XGB:
        raise RuntimeError("XGBoost not available")

    features_df = create_lag_features(df, TARGET_VAR)
    train_mask = features_df.index <= TRAIN_END
    X = features_df.loc[train_mask].drop(columns=["target"])
    y = features_df.loc[train_mask, "target"]

    splits = _tscv_splits(X.index, n_splits=n_splits)
    fold_design = _fold_metadata(X.index, splits)

    best_params = None
    best_rmse = np.inf
    search_rows: List[Dict[str, Any]] = []

    for candidate_id, params in enumerate(param_grid, start=1):
        rmses = []
        row: Dict[str, Any] = {
            "model": "XGBoost",
            "varset": varset,
            "candidate_id": candidate_id,
            **_serialize_params(params),
        }

        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            rmses.append(rmse)
            row[f"fold{fold_id}_rmse"] = rmse

        mean_rmse = float(np.mean(rmses))
        row["mean_validation_rmse"] = mean_rmse
        row["std_validation_rmse"] = float(np.std(rmses))
        search_rows.append(row)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = _serialize_params(params)

    return {
        "model": "XGBoost",
        "varset": varset,
        "n_splits": n_splits,
        "selection_criterion": SELECTION_CRITERION,
        "fold_design": fold_design,
        "search_space": [_serialize_params(params) for params in param_grid],
        "search_space_note": _xgb_search_space_note(),
        "fixed_training_config": {
            "early_stopping": "none",
            "feature_engineering": (
                "target lags [1,2,3,6,12]; exogenous lags [1,3]; "
                "rolling means [3,6]; rolling std [3]; momentum [1,3]"
            ),
        },
        "results": search_rows,
        "best_params": best_params,
        "best_rmse": float(best_rmse) if best_params is not None else None,
    }


def tune_lstm(
    df: pd.DataFrame,
    param_grid: List[Dict[str, Any]],
    varset: str,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available")

    data = df[[TARGET_VAR] + [c for c in df.columns if c != TARGET_VAR]].dropna()
    train_mask = data.index <= TRAIN_END
    data_train = data.loc[train_mask]

    splits = _tscv_splits(data_train.index, n_splits=n_splits)
    fold_design = _fold_metadata(data_train.index, splits)

    best_params = None
    best_rmse = np.inf
    search_rows: List[Dict[str, Any]] = []

    for candidate_id, params in enumerate(param_grid, start=1):
        seq_length = int(params.get("seq_length", 6))
        rmses = []
        epoch_counts = []
        row: Dict[str, Any] = {
            "model": "LSTM",
            "varset": varset,
            "candidate_id": candidate_id,
            **_serialize_params(params),
            "early_stopping_monitor": LSTM_EARLY_STOPPING_MONITOR,
            "early_stopping_patience": LSTM_EARLY_STOPPING_PATIENCE,
            "seed_base": seed,
        }

        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            _set_random_seed(seed + candidate_id * 100 + fold_id)

            train_slice = data_train.iloc[train_idx]
            val_slice = data_train.iloc[val_idx]

            scaler = MinMaxScaler()
            scaler.fit(train_slice)

            scaled_full = scaler.transform(pd.concat([train_slice, val_slice]))
            X_all, y_all, idx_all = _build_sequences(scaled_full, seq_length)

            train_limit = len(train_slice)
            train_mask_seq = idx_all < train_limit
            val_mask_seq = (idx_all >= train_limit) & (idx_all < len(train_slice) + len(val_slice))

            X_train, y_train = X_all[train_mask_seq], y_all[train_mask_seq]
            X_val, y_val = X_all[val_mask_seq], y_all[val_mask_seq]

            row[f"fold{fold_id}_train_sequences"] = int(len(X_train))
            row[f"fold{fold_id}_val_sequences"] = int(len(X_val))

            if len(X_train) < 5 or len(X_val) < 5:
                row[f"fold{fold_id}_rmse"] = None
                row[f"fold{fold_id}_epochs"] = None
                continue

            model = Sequential(
                [
                    LSTM(
                        int(params.get("units", 32)),
                        activation="tanh",
                        input_shape=(seq_length, X_train.shape[2]),
                    ),
                    Dropout(float(params.get("dropout", 0.3))),
                    Dense(max(8, int(params.get("units", 32)) // 2), activation="relu"),
                    Dense(1),
                ]
            )

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=float(params.get("learning_rate", 0.001))
            )
            model.compile(optimizer=optimizer, loss="mse")
            early_stop = EarlyStopping(
                monitor=LSTM_EARLY_STOPPING_MONITOR,
                patience=LSTM_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=int(params.get("epochs", 60)),
                batch_size=int(params.get("batch_size", 16)),
                verbose=0,
                callbacks=[early_stop],
            )

            epochs_run = int(len(history.history["loss"]))
            preds = model.predict(X_val, verbose=0).flatten()

            y_pred_full = np.zeros((len(preds), scaled_full.shape[1]))
            y_pred_full[:, 0] = preds
            y_pred = scaler.inverse_transform(y_pred_full)[:, 0]

            y_val_full = np.zeros((len(y_val), scaled_full.shape[1]))
            y_val_full[:, 0] = y_val
            y_val_inv = scaler.inverse_transform(y_val_full)[:, 0]

            rmse = float(np.sqrt(mean_squared_error(y_val_inv, y_pred)))
            rmses.append(rmse)
            epoch_counts.append(epochs_run)
            row[f"fold{fold_id}_rmse"] = rmse
            row[f"fold{fold_id}_epochs"] = epochs_run

        if rmses:
            mean_rmse = float(np.mean(rmses))
            row["mean_validation_rmse"] = mean_rmse
            row["std_validation_rmse"] = float(np.std(rmses))
            row["mean_trained_epochs"] = float(np.mean(epoch_counts))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = _serialize_params(params)
        else:
            row["mean_validation_rmse"] = None
            row["std_validation_rmse"] = None
            row["mean_trained_epochs"] = None

        search_rows.append(row)

    return {
        "model": "LSTM",
        "varset": varset,
        "n_splits": n_splits,
        "selection_criterion": SELECTION_CRITERION,
        "fold_design": fold_design,
        "search_space": [_serialize_params(params) for params in param_grid],
        "search_space_note": _lstm_search_space_note(),
        "fixed_training_config": {
            "early_stopping": f"{LSTM_EARLY_STOPPING_MONITOR}, patience={LSTM_EARLY_STOPPING_PATIENCE}",
            "seed": seed,
            "scaling": "MinMaxScaler fitted within each training fold",
        },
        "results": search_rows,
        "best_params": best_params,
        "best_rmse": float(best_rmse) if best_params is not None else None,
    }


def _write_model_outputs(model_key: str, varset: str, summary: Dict[str, Any]) -> Dict[str, str]:
    summary_path = OUTPUT_BASE / f"{model_key}_tuning_summary_{varset}.json"
    cv_results_path = OUTPUT_BASE / f"{model_key}_cv_results_{varset}.csv"
    fold_path = OUTPUT_BASE / f"{model_key}_fold_design_{varset}.csv"
    legacy_best_path = OUTPUT_BASE / f"{model_key}_best_params_{varset}.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(summary["results"]).to_csv(cv_results_path, index=False)
    pd.DataFrame(summary["fold_design"]).to_csv(fold_path, index=False)

    with open(legacy_best_path, "w") as f:
        json.dump(
            {
                "best_params": summary.get("best_params"),
                "best_rmse": summary.get("best_rmse"),
                "selection_criterion": summary.get("selection_criterion"),
            },
            f,
            indent=2,
        )

    return {
        "summary_json": str(summary_path),
        "cv_results_csv": str(cv_results_path),
        "fold_design_csv": str(fold_path),
        "best_params_json": str(legacy_best_path),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune XGBoost and LSTM with time-series CV")
    parser.add_argument("--varset", default="parsimonious")
    parser.add_argument("--model", choices=["xgb", "lstm", "all"], default="all")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()

    run_xgb = args.model in {"xgb", "all"} and not args.skip_xgb
    run_lstm = args.model in {"lstm", "all"} and not args.skip_lstm
    if not run_xgb and not run_lstm:
        raise SystemExit("No tuning job selected. Check --model / --skip-* flags.")

    df = load_varset_data(args.varset)
    outputs: Dict[str, Dict[str, str]] = {}

    if run_xgb:
        result = tune_xgboost(df, XGB_SEARCH_SPACE, varset=args.varset, n_splits=args.n_splits)
        outputs["xgb"] = _write_model_outputs("xgb", args.varset, result)
        print(f"Saved XGBoost tuning outputs for {args.varset}")

    if run_lstm:
        result = tune_lstm(
            df,
            LSTM_SEARCH_SPACE,
            varset=args.varset,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        outputs["lstm"] = _write_model_outputs("lstm", args.varset, result)
        print(f"Saved LSTM tuning outputs for {args.varset}")

    appendix_outputs = write_appendix_table_artifacts(OUTPUT_BASE)

    config = {
        "varset": args.varset,
        "model": args.model,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "skip_xgb": args.skip_xgb,
        "skip_lstm": args.skip_lstm,
        "selection_criterion": SELECTION_CRITERION,
        "output_dir": str(OUTPUT_BASE),
        "artifact_paths": {
            **{key: value for key, value in outputs.items()},
            "appendix_tables": appendix_outputs,
        },
    }
    write_run_manifest(OUTPUT_BASE, config)


if __name__ == "__main__":
    main()
