#!/usr/bin/env python3
"""Unified rolling-origin evaluator with a common forecast API."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.statespace.structural import UnobservedComponents

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:  # pragma: no cover - optional
    HAS_TF = False

try:
    from scripts.academic.models.bvar import BayesianVAR
    HAS_BVAR = True
except Exception:  # pragma: no cover - optional
    try:
        from reserves_project.scripts.academic.models.bvar import BayesianVAR
        HAS_BVAR = True
    except Exception:
        BayesianVAR = None
        HAS_BVAR = False

try:
    from .metrics import compute_metrics, naive_mae_scale, asymmetric_loss
except ImportError:  # pragma: no cover - fallback for script execution
    from scripts.forecasting_models.metrics import compute_metrics, naive_mae_scale, asymmetric_loss

try:
    from .ms_switching_var import MarkovSwitchingVAR
except Exception:  # pragma: no cover - fallback for script execution
    from scripts.forecasting_models.ms_switching_var import MarkovSwitchingVAR

Z_80 = 1.281551565545
Z_95 = 1.959963984540


@dataclass
class ForecastOutput:
    mean: np.ndarray
    std: Optional[np.ndarray] = None


class BaseForecaster:
    name: str

    def fit(self, train_df: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        raise NotImplementedError


class BaseExogForecaster:
    def fit(self, train_df: pd.DataFrame, exog_cols: List[str]) -> None:
        raise NotImplementedError

    def forecast(
        self,
        horizon: int,
        forecast_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        raise NotImplementedError


class NaiveExogForecaster(BaseExogForecaster):
    def __init__(self):
        self.last_values: Optional[pd.Series] = None
        self.exog_cols: List[str] = []

    def fit(self, train_df: pd.DataFrame, exog_cols: List[str]) -> None:
        self.exog_cols = exog_cols
        self.last_values = train_df[exog_cols].iloc[-1]

    def forecast(self, horizon: int, forecast_index: pd.DatetimeIndex) -> pd.DataFrame:
        if self.last_values is None:
            raise RuntimeError("Exog forecaster not fitted")
        data = np.tile(self.last_values.values, (horizon, 1))
        return pd.DataFrame(data, index=forecast_index, columns=self.exog_cols)


class ARIMAExogForecaster(BaseExogForecaster):
    def __init__(self, order: tuple[int, int, int] = (1, 1, 0)):
        self.order = order
        self.models: Dict[str, object] = {}
        self.exog_cols: List[str] = []
        self.last_values: Optional[pd.Series] = None

    def fit(self, train_df: pd.DataFrame, exog_cols: List[str]) -> None:
        self.exog_cols = exog_cols
        self.models = {}
        self.last_values = train_df[exog_cols].iloc[-1]
        for col in exog_cols:
            series = train_df[col].dropna()
            if len(series) < 10:
                continue
            try:
                model = SARIMAX(series, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
                self.models[col] = model.fit(disp=False)
            except Exception:
                continue

    def forecast(self, horizon: int, forecast_index: pd.DatetimeIndex) -> pd.DataFrame:
        if not self.exog_cols:
            return pd.DataFrame(index=forecast_index)
        rows = {}
        for col in self.exog_cols:
            if col in self.models:
                try:
                    fc = self.models[col].forecast(steps=horizon)
                    rows[col] = np.asarray(fc)
                    continue
                except Exception:
                    pass
            # Fallback to naive
            if self.last_values is None:
                rows[col] = np.full(horizon, np.nan)
            else:
                rows[col] = np.full(horizon, float(self.last_values[col]))
        return pd.DataFrame(rows, index=forecast_index)


class NaiveForecaster(BaseForecaster):
    def __init__(self, target_col: str):
        self.name = "Naive"
        self.target_col = target_col
        self.last_value: float | None = None
        self.resid_std: float | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        series = train_df[self.target_col].dropna()
        self.last_value = float(series.iloc[-1])
        diffs = series.diff().dropna()
        self.resid_std = float(diffs.std()) if len(diffs) else np.nan

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.last_value is None:
            raise RuntimeError("NaiveForecaster not fitted")
        mean = np.full(horizon, self.last_value)
        std = np.full(horizon, self.resid_std) if self.resid_std is not None else None
        return ForecastOutput(mean=mean, std=std)


class ArimaForecaster(BaseForecaster):
    def __init__(
        self,
        target_col: str,
        exog_cols: Optional[List[str]] = None,
        order: tuple[int, int, int] | None = None,
    ):
        self.name = "ARIMA"
        self.target_col = target_col
        self.exog_cols = exog_cols or []
        self.order = order
        self.model = None

    def _select_order(self, y: pd.Series, exog: pd.DataFrame | None, d: int = 1):
        candidates = [(p, d, q) for p in range(0, 4) for q in range(0, 4)]
        best = None
        for order in candidates:
            try:
                model = SARIMAX(
                    y,
                    order=order,
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                if best is None or res.aic < best[0]:
                    best = (res.aic, order)
            except Exception:
                continue
        return best[1] if best else (1, d, 1)

    def fit(self, train_df: pd.DataFrame) -> None:
        y = train_df[self.target_col]
        exog = train_df[self.exog_cols] if self.exog_cols else None
        order = self.order or self._select_order(y, exog, d=1)
        model = SARIMAX(
            y,
            order=order,
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model = model.fit(disp=False)

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.model is None:
            raise RuntimeError("ARIMA model not fitted")
        if self.exog_cols:
            exog_future = exog_future[self.exog_cols] if exog_future is not None else None
        forecast = self.model.get_forecast(steps=horizon, exog=exog_future)
        mean = np.asarray(forecast.predicted_mean)
        var = forecast.var_pred_mean
        std = np.sqrt(np.asarray(var)) if var is not None else None
        return ForecastOutput(mean=mean, std=std)


class VECMForecaster(BaseForecaster):
    def __init__(
        self,
        target_col: str,
        system_cols: List[str],
        k_ar_diff: int = 2,
        coint_rank: Optional[int] = None,
        deterministic: str = "co",
    ):
        self.name = "VECM"
        self.target_col = target_col
        self.system_cols = system_cols
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.results = None
        self.target_idx = system_cols.index(target_col)
        self.resid_std: float | None = None

    def _estimate_rank(self, train_levels: pd.DataFrame) -> int:
        try:
            joh = coint_johansen(train_levels, det_order=0, k_ar_diff=max(1, self.k_ar_diff))
            trace_stats = joh.lr1
            crit_vals = joh.cvt[:, 1]
            rank = int(sum(stat > cv for stat, cv in zip(trace_stats, crit_vals)))
            max_rank = max(1, len(self.system_cols) - 1)
            return max(1, min(rank, max_rank))
        except Exception:
            return 1

    def _select_k_ar_diff(self, train_levels: pd.DataFrame, max_lags: int = 6) -> int:
        try:
            from statsmodels.tsa.api import VAR
        except Exception:
            return self.k_ar_diff
        if len(train_levels) < max(30, max_lags + 1):
            return self.k_ar_diff
        try:
            sel = VAR(train_levels).select_order(maxlags=max_lags)
            chosen = None
            if hasattr(sel, "selected_orders") and sel.selected_orders:
                chosen = sel.selected_orders.get("aic") or sel.selected_orders.get("bic")
            if chosen is None and hasattr(sel, "aic"):
                chosen = sel.aic
            if chosen is None:
                return self.k_ar_diff
            chosen = int(chosen)
            return max(1, chosen - 1)
        except Exception:
            return self.k_ar_diff

    def fit(self, train_df: pd.DataFrame) -> None:
        levels = train_df[self.system_cols].dropna()
        if len(levels) < max(30, self.k_ar_diff + 5):
            raise RuntimeError("Insufficient observations for VECM")
        if self.coint_rank is None:
            rank = self._estimate_rank(levels)
        else:
            rank = self.coint_rank
        self.k_ar_diff = self._select_k_ar_diff(levels)
        vecm = VECM(
            levels,
            k_ar_diff=max(1, self.k_ar_diff),
            coint_rank=max(1, min(rank, len(self.system_cols) - 1)),
            deterministic=self.deterministic,
        )
        self.results = vecm.fit()
        resid = self.results.resid
        if resid is not None:
            self.resid_std = float(np.std(resid[:, self.target_idx]))

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.results is None:
            raise RuntimeError("VECM model not fitted")
        fc = self.results.predict(steps=horizon)
        mean = np.asarray(fc)[:, self.target_idx]
        std = np.full(horizon, self.resid_std) if self.resid_std is not None else None
        return ForecastOutput(mean=mean, std=std)


class BVARForecaster(BaseForecaster):
    def __init__(
        self,
        target_col: str,
        system_cols: List[str],
        n_lags: int = 2,
        lambda1: float = 0.2,
        lambda3: float = 1.0,
        n_draws: int = 1000,
        n_burn: int = 200,
    ):
        if not HAS_BVAR:
            raise RuntimeError("BVAR dependencies unavailable")
        self.name = "BVAR"
        self.target_col = target_col
        self.system_cols = system_cols
        self.target_idx = system_cols.index(target_col)
        self.n_lags = n_lags
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        self.n_draws = n_draws
        self.n_burn = n_burn
        self.model: Optional[BayesianVAR] = None

    def fit(self, train_df: pd.DataFrame) -> None:
        Y = train_df[self.system_cols].values
        model = BayesianVAR(
            n_lags=self.n_lags,
            lambda1=self.lambda1,
            lambda3=self.lambda3,
            n_draws=self.n_draws,
            n_burn=self.n_burn,
        )
        model.fit(Y, var_names=self.system_cols)
        self.model = model

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.model is None:
            raise RuntimeError("BVAR model not fitted")
        dist = self.model.forecast(h=horizon, return_draws=False, include_shock=True)
        mean = dist["mean"][:, self.target_idx]
        std = dist["std"][:, self.target_idx]
        return ForecastOutput(mean=mean, std=std)


class XGBoostForecaster(BaseForecaster):
    def __init__(
        self,
        target_col: str,
        exog_cols: Optional[List[str]] = None,
        lags: List[int] | None = None,
        exog_lags: List[int] | None = None,
        params: Optional[Dict] = None,
    ):
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost not available")
        self.name = "XGBoost"
        self.target_col = target_col
        self.exog_cols = exog_cols or []
        self.lags = lags or [1, 2, 3, 6, 12]
        self.exog_lags = exog_lags or [1, 3]
        self.params = params or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.model = None
        self.feature_cols: List[str] = []
        self.resid_std: float | None = None

    def _build_supervised(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = pd.DataFrame(index=df.index)
        target = df[self.target_col]

        for lag in self.lags:
            features[f"{self.target_col}_lag{lag}"] = target.shift(lag)

        for col in self.exog_cols:
            for lag in self.exog_lags:
                features[f"{col}_lag{lag}"] = df[col].shift(lag)

        features[f"{self.target_col}_ma3"] = target.rolling(3).mean().shift(1)
        features[f"{self.target_col}_ma6"] = target.rolling(6).mean().shift(1)
        features[f"{self.target_col}_std3"] = target.rolling(3).std().shift(1)
        features[f"{self.target_col}_mom1"] = target.diff(1).shift(1)
        features[f"{self.target_col}_mom3"] = target.diff(3).shift(1)

        supervised = features.copy()
        supervised["target"] = target
        supervised = supervised.dropna()
        X = supervised.drop(columns=["target"])
        y = supervised["target"]
        return X, y

    def fit(self, train_df: pd.DataFrame) -> None:
        X, y = self._build_supervised(train_df)
        if len(X) < 10:
            raise RuntimeError("Insufficient observations for XGBoost")
        model = xgb.XGBRegressor(**self.params)
        model.fit(X, y, verbose=False)
        self.model = model
        self.feature_cols = list(X.columns)
        resid = y - model.predict(X)
        self.resid_std = float(np.std(resid)) if len(resid) else np.nan

    def _get_lag_value(self, series: pd.Series, date: pd.Timestamp, lag: int) -> float:
        lag_date = date - pd.DateOffset(months=lag)
        if lag_date in series.index:
            return float(series.loc[lag_date])
        hist = series.loc[:lag_date]
        if len(hist):
            return float(hist.iloc[-1])
        return float("nan")

    def _build_feature_vector(
        self,
        target_series: pd.Series,
        exog_full: Optional[pd.DataFrame],
        forecast_date: pd.Timestamp,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        prev_date = forecast_date - pd.DateOffset(months=1)
        hist_target = target_series.loc[:prev_date]

        for lag in self.lags:
            features[f"{self.target_col}_lag{lag}"] = self._get_lag_value(target_series, forecast_date, lag)

        if self.exog_cols and exog_full is not None:
            for col in self.exog_cols:
                series = exog_full[col]
                for lag in self.exog_lags:
                    features[f"{col}_lag{lag}"] = self._get_lag_value(series, forecast_date, lag)

        tail3 = hist_target.tail(3)
        tail6 = hist_target.tail(6)
        features[f"{self.target_col}_ma3"] = float(tail3.mean()) if len(tail3) else float("nan")
        features[f"{self.target_col}_ma6"] = float(tail6.mean()) if len(tail6) else float("nan")
        features[f"{self.target_col}_std3"] = float(tail3.std()) if len(tail3) else float("nan")

        if len(hist_target) >= 2:
            features[f"{self.target_col}_mom1"] = float(hist_target.iloc[-1] - hist_target.iloc[-2])
        else:
            features[f"{self.target_col}_mom1"] = float("nan")

        if len(hist_target) >= 4:
            features[f"{self.target_col}_mom3"] = float(hist_target.iloc[-1] - hist_target.iloc[-4])
        else:
            features[f"{self.target_col}_mom3"] = float("nan")

        return features

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.model is None:
            raise RuntimeError("XGBoost model not fitted")
        history = history_df.copy()
        history = history.sort_index()
        target_series = history[self.target_col].copy()

        exog_full = None
        if self.exog_cols:
            exog_hist = history[self.exog_cols].copy()
            exog_full = exog_hist
            if exog_future is not None:
                exog_full = pd.concat([exog_hist, exog_future])

        last_date = history.index.max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        preds = []
        for fdate in forecast_dates:
            features = self._build_feature_vector(target_series, exog_full, fdate)
            row = pd.DataFrame([features])
            row = row.reindex(columns=self.feature_cols)
            pred = float(self.model.predict(row)[0])
            preds.append(pred)
            target_series.loc[fdate] = pred

        std = np.full(horizon, self.resid_std) if self.resid_std is not None else None
        return ForecastOutput(mean=np.asarray(preds), std=std)


class MSVARForecaster(BaseForecaster):
    def __init__(
        self,
        target_col: str,
        system_cols: List[str],
        ar_order: int = 1,
    ):
        self.name = "MS-VAR"
        self.target_col = target_col
        self.system_cols = system_cols
        self.ar_order = ar_order
        self.model: Optional[MarkovSwitchingVAR] = None
        self.target_idx = system_cols.index(target_col)
        self.diff_std: float | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        levels = train_df[self.system_cols].dropna()
        diffs = levels.diff().dropna()
        if len(diffs) < max(20, self.ar_order + 2):
            raise RuntimeError("Insufficient observations for MS-VAR")

        target_diff = diffs[self.target_col]
        rolling_vol = target_diff.rolling(6).std()
        threshold = float(rolling_vol.quantile(0.75)) if len(rolling_vol.dropna()) else float(target_diff.std())
        init_states = (rolling_vol > threshold).astype(int).values

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=self.ar_order)
        model.fit(diffs.values, init_states=init_states)
        self.model = model
        self.diff_std = float(target_diff.std())

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.model is None:
            raise RuntimeError("MS-VAR model not fitted")
        levels = history_df[self.system_cols].dropna()
        diffs = levels.diff().dropna()
        if len(diffs) < self.ar_order:
            raise RuntimeError("Insufficient history for MS-VAR forecast")

        history = diffs.values[-self.ar_order :]
        diff_fc = self.model.forecast(history, steps=horizon)
        target_diff_fc = diff_fc[:, self.target_idx]

        last_level = float(levels[self.target_col].iloc[-1])
        level_fc = last_level + np.cumsum(target_diff_fc)

        std = np.full(horizon, self.diff_std) if self.diff_std is not None else None
        return ForecastOutput(mean=level_fc, std=std)


class MSVECMForecaster(BaseForecaster):
    def __init__(
        self,
        target_col: str,
        system_cols: List[str],
        ar_order: int = 1,
        k_ar_diff: int = 2,
    ):
        self.name = "MS-VECM"
        self.target_col = target_col
        self.system_cols = system_cols
        self.target_idx = system_cols.index(target_col)
        self.ar_order = ar_order
        self.k_ar_diff = k_ar_diff
        self.model: Optional[MarkovSwitchingVAR] = None
        self.diff_std: float | None = None
        self.beta_norm: Optional[np.ndarray] = None

    def _estimate_beta(self, train_levels: pd.DataFrame) -> np.ndarray:
        joh = coint_johansen(train_levels, det_order=0, k_ar_diff=max(1, self.k_ar_diff))
        beta = np.asarray(joh.evec[:, 0], dtype=float)
        target_coeff = beta[0] if not np.isclose(beta[0], 0.0) else 1.0
        return beta / target_coeff

    def fit(self, train_df: pd.DataFrame) -> None:
        levels = train_df[self.system_cols].dropna()
        diffs = levels.diff().dropna()
        if len(levels) < max(30, self.k_ar_diff + 5):
            raise RuntimeError("Insufficient observations for MS-VECM")

        self.beta_norm = self._estimate_beta(levels)
        ect = pd.Series(levels.values @ self.beta_norm, index=levels.index).shift(1)
        ect = ect.reindex(diffs.index)
        ect_values = ect.values.reshape(-1, 1)

        target_diff = diffs[self.target_col]
        rolling_vol = target_diff.rolling(6).std()
        threshold = float(rolling_vol.quantile(0.75)) if len(rolling_vol.dropna()) else float(target_diff.std())
        init_states = (rolling_vol > threshold).astype(int).values

        model = MarkovSwitchingVAR(n_regimes=2, ar_order=self.ar_order)
        model.fit(diffs.values, exog=ect_values, init_states=init_states)
        self.model = model
        self.diff_std = float(target_diff.std())

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.model is None or self.beta_norm is None:
            raise RuntimeError("MS-VECM model not fitted")

        levels = history_df[self.system_cols].dropna()
        diffs = levels.diff().dropna()
        if len(diffs) < self.ar_order:
            raise RuntimeError("Insufficient history for MS-VECM forecast")

        ect_series = pd.Series(levels.values @ self.beta_norm, index=levels.index).shift(1)
        ect_series = ect_series.reindex(diffs.index)
        ect_last = float(ect_series.dropna().iloc[-1]) if ect_series.notna().any() else 0.0
        exog_future_vals = np.full((horizon, 1), ect_last)

        history = diffs.values[-self.ar_order :]
        diff_fc = self.model.forecast(history, steps=horizon, exog_future=exog_future_vals)
        target_diff_fc = diff_fc[:, self.target_idx]

        last_level = float(levels[self.target_col].iloc[-1])
        level_fc = last_level + np.cumsum(target_diff_fc)
        std = np.full(horizon, self.diff_std) if self.diff_std is not None else None
        return ForecastOutput(mean=level_fc, std=std)


class LSTMForecaster(BaseForecaster):
    def __init__(
        self,
        target_col: str,
        exog_cols: Optional[List[str]] = None,
        seq_length: int = 6,
        epochs: int = 60,
        batch_size: int = 16,
        units: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
    ):
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available")
        self.name = "LSTM"
        self.target_col = target_col
        self.exog_cols = exog_cols or []
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.feature_cols: List[str] = []

    def _build_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i : i + self.seq_length])
            y.append(data[i + self.seq_length, 0])
        return np.array(X), np.array(y)

    def fit(self, train_df: pd.DataFrame) -> None:
        from sklearn.preprocessing import MinMaxScaler

        df = train_df[[self.target_col] + self.exog_cols].dropna()
        if len(df) < self.seq_length + 10:
            raise RuntimeError("Insufficient observations for LSTM")

        self.feature_cols = [self.target_col] + self.exog_cols
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(df[self.feature_cols])
        X, y = self._build_sequences(scaled)

        model = Sequential([
            LSTM(self.units, activation="tanh", input_shape=(self.seq_length, X.shape[2])),
            Dropout(self.dropout),
            Dense(max(8, self.units // 2), activation="relu"),
            Dense(1),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[early_stop])
        self.model = model

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.model is None or self.scaler is None:
            raise RuntimeError("LSTM model not fitted")

        history = history_df[[self.target_col] + self.exog_cols].copy()
        history = history.dropna()
        if self.exog_cols and exog_future is not None:
            exog_future = exog_future[self.exog_cols]
            future_df = pd.DataFrame(
                {self.target_col: np.nan},
                index=exog_future.index,
            )
            future_df = future_df.join(exog_future, how="left")
            history = pd.concat([history, future_df])

        scaled = self.scaler.transform(history[self.feature_cols].ffill())
        preds = []

        for step in range(horizon):
            start = len(scaled) - self.seq_length
            if start < 0:
                raise RuntimeError("Insufficient history for LSTM forecast")
            seq = scaled[start : start + self.seq_length]
            seq = np.expand_dims(seq, axis=0)
            pred_scaled = float(self.model.predict(seq, verbose=0)[0, 0])

            # Build inverse transform row
            if len(scaled) > 0:
                last_features = scaled[-1].copy()
            else:
                last_features = np.zeros(len(self.feature_cols))
            last_features[0] = pred_scaled
            inv = self.scaler.inverse_transform(last_features.reshape(1, -1))[0]
            pred_value = float(inv[0])
            preds.append(pred_value)

            # Append predicted target to scaled history
            next_row = scaled[-1].copy()
            next_row[0] = pred_scaled
            scaled = np.vstack([scaled, next_row])

        return ForecastOutput(mean=np.asarray(preds), std=None)


class LocalLevelSVForecaster(BaseForecaster):
    def __init__(self, target_col: str):
        self.name = "LocalLevelSV"
        self.target_col = target_col
        self.model = None
        self.result = None

    def fit(self, train_df: pd.DataFrame) -> None:
        y = train_df[self.target_col].dropna()
        if len(y) < 20:
            raise RuntimeError("Insufficient observations for LocalLevelSV")
        model = UnobservedComponents(
            y,
            level="local level",
            stochastic_level=True,
            irregular=True,
        )
        self.result = model.fit(disp=False)

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.result is None:
            raise RuntimeError("LocalLevelSV model not fitted")
        forecast = self.result.get_forecast(steps=horizon)
        mean = np.asarray(forecast.predicted_mean)
        var = forecast.var_pred_mean
        std = np.sqrt(np.asarray(var)) if var is not None else None
        return ForecastOutput(mean=mean, std=std)


class BoPIdentityForecaster(BaseForecaster):
    def __init__(self, target_col: str, exog_cols: List[str]):
        self.name = "BoPIdentity"
        self.target_col = target_col
        self.exog_cols = exog_cols
        self.last_level: float | None = None

    def _flow_components(self, df: pd.DataFrame) -> pd.Series:
        components = []
        if "trade_balance_usd_m" in df.columns:
            components.append(df["trade_balance_usd_m"])
        else:
            if "exports_usd_m" in df.columns and "imports_usd_m" in df.columns:
                components.append(df["exports_usd_m"] - df["imports_usd_m"])

        for col in ["remittances_usd_m", "tourism_usd_m", "cse_net_usd_m"]:
            if col in df.columns:
                components.append(df[col])

        if not components:
            return pd.Series(np.zeros(len(df)), index=df.index)

        total = components[0].copy()
        for comp in components[1:]:
            total = total.add(comp, fill_value=0.0)
        return total

    def fit(self, train_df: pd.DataFrame) -> None:
        series = train_df[self.target_col].dropna()
        if len(series) == 0:
            raise RuntimeError("No target data for BoP identity")
        self.last_level = float(series.iloc[-1])

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> ForecastOutput:
        if self.last_level is None:
            raise RuntimeError("BoPIdentity model not fitted")
        if exog_future is None:
            raise RuntimeError("BoPIdentity requires exogenous forecasts")

        flows = self._flow_components(exog_future)
        level_forecast = self.last_level + flows.cumsum().values[:horizon]
        return ForecastOutput(mean=level_forecast, std=None)


def crps_gaussian(y: float, mu: float, sigma: float) -> float:
    if sigma is None or sigma <= 0 or np.isnan(sigma):
        return np.nan
    z = (y - mu) / sigma
    # standard normal pdf and cdf
    phi = 1.0 / sqrt(2 * pi) * np.exp(-0.5 * z ** 2)
    Phi = 0.5 * (1 + erf(z / sqrt(2)))
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))


def log_score_gaussian(y: float, mu: float, sigma: float) -> float:
    if sigma is None or sigma <= 0 or np.isnan(sigma):
        return np.nan
    z = (y - mu) / sigma
    return -0.5 * log(2 * pi * sigma ** 2) - 0.5 * z ** 2


def erf(x: float) -> float:
    # Approximate error function (Abramowitz-Stegun)
    # Avoid scipy dependency in core evaluation.
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


class RollingOriginEvaluator:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        exog_cols: Optional[List[str]],
        models: List[BaseForecaster],
        horizons: List[int],
        train_end: pd.Timestamp,
        valid_end: pd.Timestamp,
        refit_interval: int = 12,
        exog_mode: str = "forecast",
        exog_forecaster: Optional[BaseExogForecaster] = None,
        start_origin: Optional[pd.Timestamp] = None,
        end_origin: Optional[pd.Timestamp] = None,
    ):
        self.data = data.copy().sort_index()
        self.target_col = target_col
        self.exog_cols = exog_cols or []
        self.models = models
        self.horizons = sorted(list(set(horizons)))
        self.train_end = train_end
        self.valid_end = valid_end
        self.refit_interval = refit_interval
        self.exog_mode = exog_mode
        self.exog_forecaster = exog_forecaster or NaiveExogForecaster()
        self.start_origin = start_origin or (train_end + pd.DateOffset(months=1))
        self.end_origin = end_origin or self.data.index.max()

    def _get_origin_dates(self) -> pd.DatetimeIndex:
        dates = self.data.index
        return dates[(dates >= self.start_origin) & (dates <= self.end_origin)]

    def run(self) -> pd.DataFrame:
        results = []
        origins = self._get_origin_dates()
        max_h = max(self.horizons)

        last_fit = {model.name: None for model in self.models}

        for origin in origins:
            history = self.data.loc[:origin].copy()
            forecast_index = pd.date_range(
                start=origin + pd.DateOffset(months=1),
                periods=max_h,
                freq="MS",
            )

            exog_future = None
            if self.exog_cols:
                if self.exog_mode == "actual":
                    exog_future = self.data.reindex(forecast_index)[self.exog_cols]
                else:
                    self.exog_forecaster.fit(history, self.exog_cols)
                    exog_future = self.exog_forecaster.forecast(max_h, forecast_index)

            for model in self.models:
                last_fit_date = last_fit.get(model.name)
                need_refit = (
                    last_fit_date is None
                    or (origin - last_fit_date).days >= self.refit_interval * 28
                )
                if need_refit:
                    try:
                        model.fit(history)
                        last_fit[model.name] = origin
                    except Exception:
                        continue

                try:
                    output = model.predict(history, horizon=max_h, exog_future=exog_future)
                except Exception:
                    continue
                mean = output.mean
                std = output.std if output.std is not None else np.full(max_h, np.nan)

                for h in self.horizons:
                    fdate = forecast_index[h - 1]
                    actual = self.data[self.target_col].get(fdate, np.nan)
                    mu = mean[h - 1]
                    sigma = std[h - 1] if std is not None else np.nan

                    lower_80 = mu - Z_80 * sigma if sigma == sigma else np.nan
                    upper_80 = mu + Z_80 * sigma if sigma == sigma else np.nan
                    lower_95 = mu - Z_95 * sigma if sigma == sigma else np.nan
                    upper_95 = mu + Z_95 * sigma if sigma == sigma else np.nan

                    split = "validation" if fdate <= self.valid_end else "test"

                    crps = np.nan
                    log_score = np.nan
                    if actual == actual and sigma == sigma:
                        crps = crps_gaussian(actual, mu, sigma)
                        log_score = log_score_gaussian(actual, mu, sigma)

                    results.append({
                        "model": model.name,
                        "forecast_origin": origin,
                        "forecast_date": fdate,
                        "horizon": h,
                        "split": split,
                        "actual": actual,
                        "forecast": mu,
                        "std": sigma,
                        "lower_80": lower_80,
                        "upper_80": upper_80,
                        "lower_95": lower_95,
                        "upper_95": upper_95,
                        "crps": crps,
                        "log_score": log_score,
                    })

        return pd.DataFrame(results)


def summarize_results(
    results: pd.DataFrame,
    train_series: pd.Series,
) -> pd.DataFrame:
    scale = naive_mae_scale(train_series.values)
    rows = []
    for (model, split, horizon), subset in results.groupby(["model", "split", "horizon"]):
        metrics = compute_metrics(
            subset["actual"].values,
            subset["forecast"].values,
            mase_scale=scale,
        )
        policy_loss = asymmetric_loss(
            subset["actual"].values,
            subset["forecast"].values,
            under_weight=2.0,
            over_weight=1.0,
        )
        crps_vals = subset["crps"].values
        log_vals = subset["log_score"].values
        crps = float(np.nanmean(crps_vals)) if np.isfinite(crps_vals).any() else np.nan
        log_score = float(np.nanmean(log_vals)) if np.isfinite(log_vals).any() else np.nan
        valid_80 = subset["actual"].notna() & subset["lower_80"].notna() & subset["upper_80"].notna()
        valid_95 = subset["actual"].notna() & subset["lower_95"].notna() & subset["upper_95"].notna()
        coverage_80 = float(
            np.nanmean(
                (subset.loc[valid_80, "actual"] >= subset.loc[valid_80, "lower_80"])
                & (subset.loc[valid_80, "actual"] <= subset.loc[valid_80, "upper_80"])
            )
        ) if valid_80.any() else np.nan
        coverage_95 = float(
            np.nanmean(
                (subset.loc[valid_95, "actual"] >= subset.loc[valid_95, "lower_95"])
                & (subset.loc[valid_95, "actual"] <= subset.loc[valid_95, "upper_95"])
            )
        ) if valid_95.any() else np.nan
        rows.append({
            "model": model,
            "split": split,
            "horizon": horizon,
            **metrics,
            "policy_loss": policy_loss,
            "crps": crps,
            "log_score": log_score,
            "coverage_80": coverage_80,
            "coverage_95": coverage_95,
            "n": int(subset["actual"].notna().sum()),
        })
    return pd.DataFrame(rows)


def load_varset_levels(varset: str) -> pd.DataFrame:
    try:
        from scripts.academic.variable_sets.config import OUTPUT_DIR
    except ImportError:  # pragma: no cover - fallback for script execution
        from reserves_project.scripts.academic.variable_sets.config import OUTPUT_DIR

    path = OUTPUT_DIR / f"varset_{varset}" / "vecm_levels.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df


def build_models(
    target_col: str,
    exog_cols: List[str],
    include_bvar: bool = True,
    include_ms: bool = False,
    include_lstm: bool = False,
    include_xgb: bool = True,
    include_llsv: bool = False,
    include_bop: bool = False,
    xgb_params: Optional[Dict] = None,
    lstm_params: Optional[Dict] = None,
) -> List[BaseForecaster]:
    models: List[BaseForecaster] = [
        NaiveForecaster(target_col),
        ArimaForecaster(target_col, exog_cols=exog_cols),
        VECMForecaster(target_col, system_cols=[target_col] + exog_cols),
    ]
    if include_bvar and HAS_BVAR:
        models.append(BVARForecaster(target_col, system_cols=[target_col] + exog_cols))
    if include_xgb and HAS_XGBOOST:
        models.append(XGBoostForecaster(target_col, exog_cols=exog_cols, params=xgb_params))
    if include_ms:
        models.append(MSVARForecaster(target_col, system_cols=[target_col] + exog_cols))
        models.append(MSVECMForecaster(target_col, system_cols=[target_col] + exog_cols))
    if include_lstm and HAS_TF:
        lstm_params = lstm_params or {}
        models.append(LSTMForecaster(target_col, exog_cols=exog_cols, **lstm_params))
    if include_llsv:
        models.append(LocalLevelSVForecaster(target_col))
    if include_bop:
        models.append(BoPIdentityForecaster(target_col, exog_cols=exog_cols))
    return models


def main():
    import argparse
    from pathlib import Path
    try:
        from scripts.academic.variable_sets.config import TARGET_VAR, TRAIN_END, VALID_END
    except ImportError:  # pragma: no cover - fallback for script execution
        from reserves_project.scripts.academic.variable_sets.config import TARGET_VAR, TRAIN_END, VALID_END

    parser = argparse.ArgumentParser(description="Unified rolling-origin evaluator")
    parser.add_argument("--varset", default="parsimonious", help="Variable set to evaluate")
    parser.add_argument("--refit-interval", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="1,3,6,12")
    parser.add_argument("--exog-mode", choices=["forecast", "actual"], default="forecast")
    parser.add_argument("--exog-forecast", choices=["naive", "arima"], default="naive")
    parser.add_argument("--output-dir", default="data/forecast_results_unified")
    parser.add_argument("--include-ms", action="store_true", help="Include MS-VAR and MS-VECM adapters")
    parser.add_argument("--include-lstm", action="store_true", help="Include LSTM adapter (requires TensorFlow)")
    parser.add_argument("--include-llsv", action="store_true", help="Include local-level state-space model")
    parser.add_argument("--include-bop", action="store_true", help="Include structural BoP identity model")
    parser.add_argument("--exclude-bvar", action="store_true", help="Exclude BVAR adapter")
    parser.add_argument("--exclude-xgb", action="store_true", help="Exclude XGBoost adapter")
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    df = load_varset_levels(args.varset)
    df = df.sort_index()

    exog_cols = [c for c in df.columns if c != TARGET_VAR]

    exog_forecaster: BaseExogForecaster
    if args.exog_forecast == "arima":
        exog_forecaster = ARIMAExogForecaster()
    else:
        exog_forecaster = NaiveExogForecaster()

    models = build_models(
        TARGET_VAR,
        exog_cols,
        include_bvar=not args.exclude_bvar,
        include_ms=args.include_ms,
        include_lstm=args.include_lstm,
        include_xgb=not args.exclude_xgb,
        include_llsv=args.include_llsv,
        include_bop=args.include_bop,
    )
    evaluator = RollingOriginEvaluator(
        data=df,
        target_col=TARGET_VAR,
        exog_cols=exog_cols,
        models=models,
        horizons=horizons,
        train_end=TRAIN_END,
        valid_end=VALID_END,
        refit_interval=args.refit_interval,
        exog_mode=args.exog_mode,
        exog_forecaster=exog_forecaster,
    )

    results = evaluator.run()
    summary = summarize_results(results, df.loc[df.index <= TRAIN_END, TARGET_VAR])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"rolling_origin_forecasts_{args.varset}.csv", index=False)
    summary.to_csv(output_dir / f"rolling_origin_summary_{args.varset}.csv", index=False)

    print("Saved:")
    print(f"  - {output_dir / f'rolling_origin_forecasts_{args.varset}.csv'}")
    print(f"  - {output_dir / f'rolling_origin_summary_{args.varset}.csv'}")


if __name__ == "__main__":
    main()
