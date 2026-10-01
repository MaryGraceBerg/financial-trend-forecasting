"""
trend_main.py

Entry point for the trend lifecycle model: loads trend data (yours, or a
synthetic demo set), classifies each trend's current status, scores how
"due" dormant trends are for a comeback, forecasts each trend forward, and
plots the top trends.

Run with the built-in demo data (no --data given):
    python trend_main.py

Run with your own data (long format: one row per date per trend):
    python trend_main.py --data your_trends.csv --date-col date --trend-col trend --value-col popularity

Options:
    --data          Path to a long-format CSV: one row per (date, trend, value).
    --date-col      Name of the date column (default: date)
    --trend-col     Name of the column identifying which trend each row belongs to (default: trend)
    --value-col     Name of the numeric popularity/interest column (default: popularity)
    --horizon-weeks Weeks to forecast forward (default: 12)
    --top-n         How many trends to show on the chart (default: 6)
    --output-dir    Where to save the summary CSV + plot (default: output)
"""

import argparse
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from trend_generator import generate_trend_data
from trend_model import analyze_all_trends

STATUS_COLORS = {
    "Rising": "#5b9c59",
    "Peaking": "#e8734a",
    "Declining": "#a89bd0",
    "Dormant": "#b7ae9f",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Trend lifecycle tracking and comeback forecasting.")
    parser.add_argument("--data", type=str, default=None,
                         help="Path to a long-format CSV (date, trend, value). If omitted, uses demo data.")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--trend-col", type=str, default="trend")
    parser.add_argument("--value-col", type=str, default="popularity")
    parser.add_argument("--horizon-weeks", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=6,
                         help="How many trends to show on the chart (default: 6)")
    parser.add_argument("--output-dir", type=str, default="output")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data:
        df = pd.read_csv(args.data)
        for col in [args.date_col, args.trend_col, args.value_col]:
            if col not in df.columns:
                raise SystemExit(f"Column '{col}' not found. Available columns: {list(df.columns)}")
        df = df.rename(columns={args.date_col: "date", args.trend_col: "trend",
                                 args.value_col: "popularity"})
        df["date"] = pd.to_datetime(df["date"])
        print(f"Loaded {len(df)} rows across {df['trend'].nunique()} trends from {args.data}\n")
    else:
        os.makedirs("data", exist_ok=True)
        df = generate_trend_data(n_weeks=260)
        df.to_csv("data/trend_data.csv", index=False)
        print(f"No --data provided — using built-in synthetic demo data "
              f"({df['trend'].nunique()} trends, {len(df)} rows). "
              f"Pass --data your_file.csv to use real data.\n")

    result = analyze_all_trends(df, horizon_weeks=args.horizon_weeks)

    # 1. Print summary table
    print("Trend status summary (sorted by comeback readiness):\n")
    display_cols = ["trend", "status", "current_level", "historical_max",
                     "weeks_since_peak", "avg_cycle_weeks", "comeback_score"]
    print(result.summary[display_cols].to_string(index=False))

    due = result.summary[result.summary["due_for_comeback"]]
    if len(due) > 0:
        print(f"\nDue for a comeback: {', '.join(due['trend'].tolist())}")
    else:
        print("\nNo trends currently flagged as due for a comeback.")

    # 2. Save summary CSV
    summary_path = os.path.join(args.output_dir, "trend_summary.csv")
    result.summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")

    # 3. Plot the top-N most currently relevant trends (Rising/Peaking first, then others)
    status_priority = {"Peaking": 0, "Rising": 1, "Declining": 2, "Dormant": 3}
    plot_trends = result.summary.copy()
    plot_trends["priority"] = plot_trends["status"].map(status_priority)
    plot_trends = plot_trends.sort_values(["priority", "current_level"], ascending=[True, False])
    top_trends = plot_trends.head(args.top_n)["trend"].tolist()

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#fdfcf9")
    ax.set_facecolor("#fdfcf9")

    palette = ["#7cb0d6", "#e8734a", "#5b9c59", "#c9a227", "#a89bd0", "#d67ba0"]
    handoff_date = None

    for i, trend_name in enumerate(top_trends):
        color = palette[i % len(palette)]
        hist = result.history[trend_name].tail(104)  # last 2 years
        fcst = result.forecasts[trend_name]
        handoff_date = hist["date"].max()

        ax.plot(hist["date"], hist["value"], color=color, linewidth=2.2,
                solid_capstyle="round", label=trend_name, zorder=2)
        ax.plot(fcst["date"], fcst["value"], color=color, linewidth=2.2,
                linestyle=(0, (4, 2)), solid_capstyle="round", zorder=2, alpha=0.85)

    if handoff_date is not None:
        ax.axvline(handoff_date, color="#b7ae9f", linestyle=(0, (3, 3)), linewidth=1.2, zorder=1)
        ax.annotate("today", xy=(handoff_date, ax.get_ylim()[1]), xytext=(6, -14),
                    textcoords="offset points", fontsize=9, color="#9a9284", style="italic")

    ax.grid(axis="y", color="#e8e3d8", linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c9c2b3")

    ax.set_title("Trend Watch — What's Rising, Peaking, and Due to Return",
                  fontsize=16, fontweight="bold", color="#3c3833", pad=18, loc="left")
    ax.text(0, 1.045, f"Solid = history · Dashed = {args.horizon_weeks}-week forecast",
            transform=ax.transAxes, fontsize=10, color="#8a8377")

    ax.set_ylabel("Popularity", fontsize=10.5, color="#5c5748")
    ax.tick_params(colors="#5c5748", labelsize=9.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=1)
    for text in legend.get_texts():
        text.set_color("#5c5748")

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "trend_plot.png")
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
