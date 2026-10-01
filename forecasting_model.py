"""
forecasting_model.py

Financial Trend Forecasting Model
----------------------------------
A regression-based time-series forecasting model that analyzes historical
demand data and projects trends 8 weeks (56 days) into the future.

Approach:
1. Feature engineering — turns raw dates into model-ready signals:
   time index (captures trend), day-of-week (captures weekly seasonality),
   and rolling averages (captures recent momentum).
2. Model — Linear Regression on engineered features. Simple, interpretable,
   and a solid baseline for demand/revenue forecasting before reaching for
   more complex models (ARIMA, Prophet, XGBoost, etc.).
3. Forecast — projects the fitted trend + seasonality forward 8 weeks.
4. Evaluation — reports MAE / RMSE / R^2 on a held-out test split so you
   can quote real accuracy numbers on your resume/interviews.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FORECAST_HORIZON_WEEKS = 8
FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7


def load_real_data(csv_path: str) -> pd.DataFrame:
    """
    Load a real historical dataset from CSV. Expects at least two columns:
    a date column and a numeric target column. Rename/adjust as needed
    for your actual data source.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def engineer_features(df: pd.DataFrame, target_col: str = "demand") -> pd.DataFrame:
    """
    Build model features from a date + target dataframe.
    """
    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for day-of-week so the model understands
    # Sunday (6) is "close to" Monday (0), not far from it.
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Rolling momentum features (recent trend strength)
    df["rolling_7"] = df[target_col].rolling(window=7, min_periods=1).mean()
    df["rolling_14"] = df[target_col].rolling(window=14, min_periods=1).mean()

    df["rolling_7"] = df["rolling_7"].bfill()
    df["rolling_14"] = df["rolling_14"].bfill()

    return df


FEATURE_COLUMNS = ["time_idx", "dow_sin", "dow_cos", "is_weekend", "rolling_7", "rolling_14"]


@dataclass
class ForecastResult:
    model: LinearRegression
    metrics: dict
    history: pd.DataFrame
    forecast: pd.DataFrame


def train_and_evaluate(df: pd.DataFrame, target_col: str = "demand") -> tuple:
    """
    Split data, train the regression model, and compute accuracy metrics.
    """
    features = engineer_features(df, target_col)
    X = features[FEATURE_COLUMNS]
    y = features[target_col]

    # Time-series aware split: train on the earlier ~80%, test on the most
    # recent ~20% (never shuffle time-series data).
    split_idx = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

    return model, metrics, features


def forecast_forward(model: LinearRegression, history: pd.DataFrame,
                      target_col: str = "demand",
                      horizon_days: int = FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """
    Project the model forward `horizon_days` beyond the last known date.
    Rolling features are carried forward using a simple recursive approach:
    each new prediction feeds into the next day's rolling average.
    """
    last_date = history["date"].max()
    last_time_idx = history["time_idx"].max()

    extended = history[["date", target_col]].copy()

    future_rows = []
    for i in range(1, horizon_days + 1):
        future_date = last_date + pd.Timedelta(days=i)
        time_idx = last_time_idx + i
        day_of_week = future_date.dayofweek
        is_weekend = int(day_of_week >= 5)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        rolling_7 = extended[target_col].tail(7).mean()
        rolling_14 = extended[target_col].tail(14).mean()

        row_features = pd.DataFrame([{
            "time_idx": time_idx,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "is_weekend": is_weekend,
            "rolling_7": rolling_7,
            "rolling_14": rolling_14,
        }])

        pred = model.predict(row_features[FEATURE_COLUMNS])[0]
        pred = max(pred, 0)

        future_rows.append({"date": future_date, target_col: pred, "type": "forecast"})
        extended = pd.concat(
            [extended, pd.DataFrame([{"date": future_date, target_col: pred}])],
            ignore_index=True
        )

    return pd.DataFrame(future_rows)


def run_full_pipeline(df: pd.DataFrame, target_col: str = "demand",
                       horizon_days: int = FORECAST_HORIZON_DAYS) -> ForecastResult:
    model, metrics, features = train_and_evaluate(df, target_col)
    forecast = forecast_forward(model, features, target_col, horizon_days=horizon_days)
    history = features.copy()
    history["type"] = "history"
    return ForecastResult(model=model, metrics=metrics, history=history, forecast=forecast)
