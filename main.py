"""
main.py

Entry point: loads historical data (yours, or a synthetic demo dataset),
trains the regression-based forecasting model, prints accuracy metrics,
and plots history vs. the N-week forward projection.

Run with your own data:
    python main.py --data path/to/your_data.csv --date-col date --target-col revenue

Run with the built-in demo data (no --data given):
    python main.py

Options:
    --data          Path to a CSV with a date column and a numeric target column.
    --date-col      Name of the date column in your CSV (default: date)
    --target-col    Name of the numeric column to forecast (default: demand)
    --horizon-weeks Weeks to forecast forward (default: 8)
    --output-dir    Where to save the forecast CSV + plot (default: output)
    --plot-history-days  How many days of history to show on the chart (default: 60)
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from data_generator import generate_demand_data
from forecasting_model import load_real_data, run_full_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Regression-based time-series forecasting.")
    parser.add_argument("--data", type=str, default=None,
                         help="Path to your CSV file. If omitted, uses built-in synthetic demo data.")
    parser.add_argument("--date-col", type=str, default="date",
                         help="Name of the date column in your CSV (default: date)")
    parser.add_argument("--target-col", type=str, default="demand",
                         help="Name of the numeric column to forecast (default: demand)")
    parser.add_argument("--horizon-weeks", type=int, default=8,
                         help="Number of weeks to forecast forward (default: 8)")
    parser.add_argument("--output-dir", type=str, default="output",
                         help="Directory to save the forecast CSV + plot (default: output)")
    parser.add_argument("--plot-history-days", type=int, default=60,
                         help="How many days of history to show on the chart (default: 60). "
                              "The model still trains on all your data — this only trims the view.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    horizon_days = args.horizon_weeks * 7

    # 1. Load data — either the user's CSV or the built-in demo dataset
    if args.data:
        df = pd.read_csv(args.data)
        if args.date_col not in df.columns:
            raise SystemExit(
                f"Column '{args.date_col}' not found in data. "
                f"Available columns: {list(df.columns)}. "
                f"Use --date-col to specify the right one."
            )
        if args.date_col != "date":
            df = df.rename(columns={args.date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        target_col = args.target_col
        print(f"Loaded {len(df)} data points from {args.data}\n")
    else:
        os.makedirs("data", exist_ok=True)
        df = generate_demand_data(n_points=365)
        df.to_csv("data/demand_data.csv", index=False)
        target_col = "demand"
        print(f"No --data provided — using built-in synthetic demo data "
              f"({len(df)} points). Pass --data your_file.csv to use real data.\n")

    if target_col not in df.columns:
        raise SystemExit(
            f"Column '{target_col}' not found in data. "
            f"Available columns: {list(df.columns)}. "
            f"Use --target-col to specify the right one."
        )

    # 2. Train model + forecast forward
    result = run_full_pipeline(df, target_col=target_col, horizon_days=horizon_days)

    print("Model performance on held-out test data:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.2f}")

    print(f"\n{args.horizon_weeks}-week forward projection (first 10 rows):")
    print(result.forecast.head(10).to_string(index=False))

    # 3. Save forecast to CSV
    forecast_path = os.path.join(args.output_dir, "forecast.csv")
    result.forecast.to_csv(forecast_path, index=False)
    print(f"\nSaved forecast -> {forecast_path}")

    # 4. Plot recent history + forecast (zoomed in, forecast emphasized)
    plot_history = result.history.tail(args.plot_history_days)

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("#fdfcf9")
    ax.set_facecolor("#fdfcf9")

    history_color = "#7cb0d6"
    forecast_color = "#e8734a"

    # Soft area fill under the history line
    ax.fill_between(plot_history["date"], plot_history[target_col],
                     plot_history[target_col].min() * 0.95,
                     color=history_color, alpha=0.12, linewidth=0)

    # History line — thinner, softer, behind the forecast
    ax.plot(plot_history["date"], plot_history[target_col],
            label="Recent history", color=history_color, linewidth=2.2,
            solid_capstyle="round", zorder=2)

    # Forecast area fill — bolder, draws the eye
    ax.fill_between(result.forecast["date"], result.forecast[target_col],
                     plot_history[target_col].min() * 0.95,
                     color=forecast_color, alpha=0.18, linewidth=0, zorder=3)

    # Forecast line — thick, bold, with markers so it visually "pops"
    ax.plot(result.forecast["date"], result.forecast[target_col],
            label=f"{args.horizon_weeks}-week forecast", color=forecast_color,
            linewidth=3.5, marker="o", markersize=4.5, markerfacecolor="white",
            markeredgecolor=forecast_color, markeredgewidth=1.5,
            solid_capstyle="round", zorder=4)

    # "Today" marker where history hands off to forecast
    handoff_date = plot_history["date"].max()
    ax.axvline(handoff_date, color="#b7ae9f", linestyle=(0, (3, 3)), linewidth=1.2, zorder=1)
    ax.annotate("today", xy=(handoff_date, ax.get_ylim()[1]), xytext=(6, -14),
                textcoords="offset points", fontsize=9, color="#9a9284", style="italic")

    # Styling: soft grid, no top/right border, friendly title
    ax.grid(axis="y", color="#e8e3d8", linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c9c2b3")

    ax.set_title(f"{target_col.title()} — What's Coming Next",
                 fontsize=17, fontweight="bold", color="#3c3833", pad=18, loc="left")
    ax.text(0, 1.06, f"Last {args.plot_history_days} days vs. the next {args.horizon_weeks} weeks",
            transform=ax.transAxes, fontsize=10.5, color="#8a8377")

    ax.set_ylabel(target_col.title(), fontsize=10.5, color="#5c5748")
    ax.tick_params(colors="#5c5748", labelsize=9.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    legend = ax.legend(loc="upper left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color("#5c5748")

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "forecast_plot.png")
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
