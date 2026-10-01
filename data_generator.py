"""
data_generator.py

Generates a synthetic daily demand/sales dataset that mimics realistic
financial time-series behavior: a long-term trend, weekly seasonality,
occasional promotional spikes, and random noise.

This stands in for a real historical sales/demand export (e.g. from
POS, ERP, or CRM data). Swap load_real_data() in forecasting_model.py
to point at an actual CSV once you have one.
"""

import numpy as np
import pandas as pd


def generate_demand_data(n_points: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Generate n_points days of synthetic demand data.

    Components:
    - Base level
    - Linear upward trend (growth over time)
    - Weekly seasonality (higher demand on weekends)
    - Occasional promotional spikes
    - Gaussian noise
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_points, freq="D")

    t = np.arange(n_points)

    base_level = 500
    trend = 0.8 * t  # gradual upward growth over the period

    # Weekly seasonality: Saturday/Sunday demand boost
    day_of_week = dates.dayofweek
    weekly_effect = np.where(day_of_week >= 5, 60, 0) + np.where(day_of_week == 4, 25, 0)

    # Random promotional spikes (~5% of days)
    promo_spike = np.zeros(n_points)
    promo_days = rng.choice(n_points, size=max(1, n_points // 20), replace=False)
    promo_spike[promo_days] = rng.uniform(80, 200, size=len(promo_days))

    # Noise
    noise = rng.normal(0, 30, size=n_points)

    demand = base_level + trend + weekly_effect + promo_spike + noise
    demand = np.maximum(demand, 0).round(2)

    df = pd.DataFrame({
        "date": dates,
        "demand": demand
    })

    return df


if __name__ == "__main__":
    df = generate_demand_data(n_points=365)
    out_path = "data/demand_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows of synthetic demand data -> {out_path}")
    print(df.head())
