"""
trend_model.py

Trend Lifecycle Model
----------------------
Instead of forecasting one number forward (like the financial model), this
tracks each trend's position in its lifecycle — rising, peaking, declining,
or dormant — and flags trends that look "due" for a comeback based on how
their past cycles behaved.

Approach:
1. For each trend, find its historical peaks (scipy.signal.find_peaks) to
   estimate a typical cycle length — how long it usually takes to come
   back around.
2. Classify current status from recent slope + how close the current level
   is to that trend's historical peak.
3. Score "comeback readiness": how long it's been since the last peak,
   relative to that trend's typical cycle length — plus a bonus if it's
   already ticking upward again.
4. Forecast each trend forward with a lightweight regression (time index +
   rolling averages), the same core idea as the financial model, adapted
   for weekly trend data instead of daily.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

FEATURE_COLUMNS = ["time_idx", "rolling_4", "rolling_8"]


def load_trend_data(csv_path: str) -> pd.DataFrame:
    """Load a real trend dataset from CSV. Expects date, trend, value columns."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.sort_values(["trend", "date"]).reset_index(drop=True)


def _engineer_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"value": series.values})
    df["time_idx"] = np.arange(len(df))
    df["rolling_4"] = df["value"].rolling(window=4, min_periods=1).mean().bfill()
    df["rolling_8"] = df["value"].rolling(window=8, min_periods=1).mean().bfill()
    return df


def classify_status_and_comeback(dates: pd.Series, values: pd.Series,
                                  recent_window: int = 8) -> dict:
    """
    Determine a trend's current lifecycle status and how "due" it is for
    a comeback, based on its own historical peak pattern.
    """
    values = values.reset_index(drop=True)
    n = len(values)
    historical_max = values.max()
    current_level = values.iloc[-1]

    # Recent slope: simple linear fit over the last `recent_window` points
    window = min(recent_window, n)
    recent_y = values.iloc[-window:].values
    recent_x = np.arange(window)
    if window >= 2:
        recent_slope = np.polyfit(recent_x, recent_y, 1)[0]
    else:
        recent_slope = 0.0
    slope_threshold = max(historical_max * 0.01, 0.5)

    # Find historical peaks to estimate typical cycle length
    peak_indices, _ = find_peaks(values.values, distance=8, prominence=historical_max * 0.25)
    avg_cycle_weeks = None
    weeks_since_peak = None
    if len(peak_indices) >= 1:
        weeks_since_peak = n - 1 - peak_indices[-1]
    if len(peak_indices) >= 2:
        avg_cycle_weeks = float(np.mean(np.diff(peak_indices)))

    # Status classification
    if current_level >= 0.75 * historical_max:
        status = "Peaking"
    elif recent_slope > slope_threshold:
        status = "Rising"
    elif recent_slope < -slope_threshold:
        status = "Declining"
    else:
        status = "Dormant"

    # Comeback readiness score (only meaningful for dormant trends with a known cycle)
    comeback_score = 0.0
    if status == "Dormant" and avg_cycle_weeks and weeks_since_peak is not None:
        comeback_score = min(weeks_since_peak / avg_cycle_weeks, 2.0)
        if recent_slope > 0:
            comeback_score *= 1.2

    due_for_comeback = status == "Dormant" and comeback_score >= 0.8

    return {
        "status": status,
        "due_for_comeback": due_for_comeback,
        "comeback_score": round(float(comeback_score), 2),
        "weeks_since_peak": weeks_since_peak,
        "avg_cycle_weeks": round(avg_cycle_weeks, 1) if avg_cycle_weeks else None,
        "current_level": round(float(current_level), 1),
        "historical_max": round(float(historical_max), 1),
        "recent_slope": round(float(recent_slope), 2),
    }


def forecast_trend_forward(dates: pd.Series, values: pd.Series,
                            horizon_weeks: int = 12) -> pd.DataFrame:
    """
    Forecast a single trend's popularity forward using a lightweight
    regression on time index + rolling averages, recursively applied.
    """
    features = _engineer_features(values)
    X = features[FEATURE_COLUMNS]
    y = features["value"]

    model = LinearRegression()
    model.fit(X, y)

    last_date = pd.to_datetime(dates.iloc[-1])
    last_time_idx = features["time_idx"].iloc[-1]
    extended_values = list(values.values)

    future_rows = []
    for i in range(1, horizon_weeks + 1):
        future_date = last_date + pd.Timedelta(weeks=i)
        time_idx = last_time_idx + i
        rolling_4 = np.mean(extended_values[-4:])
        rolling_8 = np.mean(extended_values[-8:])

        row = pd.DataFrame([{"time_idx": time_idx, "rolling_4": rolling_4, "rolling_8": rolling_8}])
        pred = model.predict(row[FEATURE_COLUMNS])[0]
        pred = float(np.clip(pred, 0, 100))

        future_rows.append({"date": future_date, "value": pred})
        extended_values.append(pred)

    return pd.DataFrame(future_rows)


@dataclass
class TrendResult:
    summary: pd.DataFrame
    history: dict = field(default_factory=dict)
    forecasts: dict = field(default_factory=dict)


def analyze_all_trends(df: pd.DataFrame, trend_col: str = "trend",
                        value_col: str = "popularity", date_col: str = "date",
                        horizon_weeks: int = 12) -> TrendResult:
    """
    Run status classification, comeback scoring, and forward forecasting
    for every trend in the dataset. Returns a summary table plus per-trend
    history and forecast data for plotting.
    """
    summary_rows = []
    history = {}
    forecasts = {}

    for trend_name, group in df.groupby(trend_col):
        group = group.sort_values(date_col)
        dates = group[date_col]
        values = group[value_col]

        info = classify_status_and_comeback(dates, values)
        forecast = forecast_trend_forward(dates, values, horizon_weeks=horizon_weeks)

        summary_rows.append({"trend": trend_name, **info})
        history[trend_name] = group[[date_col, value_col]].rename(
            columns={date_col: "date", value_col: "value"})
        forecasts[trend_name] = forecast

    summary = pd.DataFrame(summary_rows).sort_values(
        ["due_for_comeback", "comeback_score"], ascending=[False, False]
    ).reset_index(drop=True)

    return TrendResult(summary=summary, history=history, forecasts=forecasts)
