"""
trend_generator.py

Generates synthetic weekly "popularity" data for multiple trends, each
following a realistic lifecycle: a fast rise, a peak plateau, a decline,
a dormant period, and then — for some trends — a resurgence into a second
(often smaller, sometimes bigger) cycle. This stands in for real trend data
you'd normally pull from Google Trends, social listening tools, or internal
sell-through data.

This mirrors the kind of pattern real trends follow: ruffles rise, peak,
fade, sit dormant for a while, then come back around. Swap load_trend_data()
in trend_model.py to point at a real export once you have one.
"""

import numpy as np
import pandas as pd

DEFAULT_TRENDS = [
    "Ruffled Details",
    "Cabbage Edge Hems",
    "Layered Textures",
    "Oversized Silhouettes",
    "Metallic Finishes",
    "Sustainable Materials",
    "Minimalist Design",
    "Bold Prints",
]


def _lifecycle_shape(phase: np.ndarray) -> np.ndarray:
    """
    Maps a position within a cycle (0 to 1) to a popularity multiplier
    (0 to 1) following a realistic trend lifecycle:
    fast rise -> peak plateau -> slower decline -> dormant tail.
    """
    shape = np.zeros_like(phase)

    rising = phase < 0.15
    shape[rising] = phase[rising] / 0.15

    plateau = (phase >= 0.15) & (phase < 0.30)
    shape[plateau] = 1.0

    declining = (phase >= 0.30) & (phase < 0.75)
    shape[declining] = 1.0 - (phase[declining] - 0.30) / (0.75 - 0.30)

    dormant = phase >= 0.75
    shape[dormant] = 0.05

    return np.clip(shape, 0, 1)


def generate_trend_data(n_weeks: int = 260, trends: list = None, seed: int = 11) -> pd.DataFrame:
    """
    Generate n_weeks of synthetic weekly popularity data (0-100 scale) for
    each trend in `trends`. Returns a long-format DataFrame:
    columns = [date, trend, popularity]
    """
    if trends is None:
        trends = DEFAULT_TRENDS

    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end_date, periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    rows = []
    for i, trend in enumerate(trends):
        rng = np.random.default_rng(seed + i * 17)

        cycle_length = rng.integers(40, 95)
        phase_offset = rng.uniform(0, cycle_length)
        base_amplitude = rng.uniform(55, 92)
        revival_multiplier = rng.uniform(0.55, 1.2)
        noise_level = rng.uniform(3, 7)
        floor = rng.uniform(3, 10)

        shifted_t = t + phase_offset
        cycle_index = np.floor(shifted_t / cycle_length)
        phase = (shifted_t % cycle_length) / cycle_length

        amplitude_this_cycle = base_amplitude * np.power(revival_multiplier, cycle_index)
        amplitude_this_cycle = np.clip(amplitude_this_cycle, 8, 100)

        shape = _lifecycle_shape(phase)
        noise = rng.normal(0, noise_level, size=n_weeks)

        popularity = floor + amplitude_this_cycle * shape + noise
        popularity = np.clip(popularity, 0, 100).round(1)

        for d, p in zip(dates, popularity):
            rows.append({"date": d, "trend": trend, "popularity": p})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_trend_data()
    out_path = "data/trend_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows across {df['trend'].nunique()} trends -> {out_path}")
    print(df.head())
