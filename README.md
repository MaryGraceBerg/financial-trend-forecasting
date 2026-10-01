# Financial Trend Forecasting Model

A regression-based time-series forecasting model that analyzes 300+ historical
data points to identify demand trends and predict 8-week forward projections.

## How it works

1. **`data_generator.py`** — produces a synthetic 365-day daily demand dataset
   with trend, weekly seasonality, and promotional spikes (stand-in for real
   sales/demand history — swap `load_real_data()` in `forecasting_model.py`
   once you have an actual CSV export).
2. **`forecasting_model.py`** — the core model:
   - Engineers features: time index (trend), cyclical day-of-week encoding
     (weekly seasonality), 7/14-day rolling averages (momentum).
   - Trains a `LinearRegression` model on an 80/20 time-ordered train/test
     split (no shuffling — that would leak future info into training).
   - Evaluates with MAE, RMSE, and R².
   - Recursively forecasts 56 days (8 weeks) forward, feeding each
     prediction into the next day's rolling-average features.
3. **`main.py`** — runs the full pipeline end to end, prints metrics, and
   saves a forecast CSV + plot.

## Setup (in VS Code)

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run it:
   ```bash
   python main.py
   ```

## Output

- `output/forecast.csv` — the forward projection
- `output/forecast_plot.png` — chart of history vs. forecast

Running with no arguments uses built-in synthetic demo data and also writes
it to `data/demand_data.csv` so you can see what the expected format looks
like.

## Using your own data (reusable for any dataset)

No code changes needed — just point `main.py` at any CSV with a date column
and a numeric column you want to forecast:

```bash
python main.py --data path/to/your_file.csv --date-col date --target-col revenue
```

All options:

| Flag | Default | Description |
|---|---|---|
| `--data` | *(none — uses demo data)* | Path to your CSV file |
| `--date-col` | `date` | Name of the date column in your CSV |
| `--target-col` | `demand` | Name of the numeric column to forecast |
| `--horizon-weeks` | `8` | How many weeks forward to project |
| `--output-dir` | `output` | Where to save the forecast CSV + plot |

Examples:
```bash
# Forecast weekly sales 6 weeks out from a "sales" column
python main.py --data sales.csv --date-col week_start --target-col sales --horizon-weeks 6

# Forecast daily website signups 2 weeks out
python main.py --data signups.csv --date-col day --target-col signups --horizon-weeks 2
```

Your CSV just needs:
- One column of dates (any standard date format, e.g. `2026-01-15`)
- One numeric column to forecast (revenue, demand, signups, units sold, etc.)
- Ideally 60+ rows of history — the more history, the more reliable the trend/seasonality estimates (300+ rows, like the resume bullet, is a good target)

## Next steps / extensions

- Swap `LinearRegression` for `Ridge`/`Lasso` to regularize against overfitting.
- Add holiday/event flags as features.
- Compare against ARIMA or Prophet as a baseline check.
- Add cross-validation instead of a single train/test split for more robust
  accuracy estimates.

---

## Bonus: Trend Lifecycle Model (`trend_main.py`)

A second, related tool in this project: instead of forecasting one number
forward, this tracks **multiple trends at once** and classifies each one's
position in its lifecycle — Rising, Peaking, Declining, or Dormant — and
flags dormant trends that look "due" for a comeback based on their own
historical cycle pattern. This is the same core idea behind fashion/product
trend forecasting (tracking things like silhouettes or design details as
they rise, peak, fade, and sometimes come back around).

### How it works

1. **`trend_generator.py`** — generates synthetic weekly popularity data
   (0-100 scale) for 8 example trends, each following a realistic lifecycle:
   fast rise, peak plateau, decline, dormant period, and — for some — a
   resurgence into a second cycle (sometimes bigger, sometimes smaller than
   the first). Swap `load_trend_data()` in `trend_model.py` once you have
   real trend data (e.g. a Google Trends export).
2. **`trend_model.py`** — the core model:
   - Finds each trend's historical peaks (`scipy.signal.find_peaks`) to
     estimate its typical cycle length.
   - Classifies current status from recent slope + how close the current
     level is to that trend's historical peak.
   - Scores "comeback readiness": weeks since the last peak, relative to
     that trend's typical cycle length, boosted if it's already ticking
     upward again.
   - Forecasts each trend forward with a lightweight regression (time index
     + rolling averages), the same core idea as the financial model.
3. **`trend_main.py`** — runs the full analysis, prints a status table
   sorted by comeback readiness, and plots the most currently-relevant
   trends (history solid, forecast dashed).

### Run it

With the built-in demo data:
```bash
python trend_main.py
```

With your own data (long format — one row per date per trend):
```bash
python trend_main.py --data your_trends.csv --date-col date --trend-col trend --value-col popularity
```

All options:

| Flag | Default | Description |
|---|---|---|
| `--data` | *(none — uses demo data)* | Path to a long-format CSV (date, trend, value) |
| `--date-col` | `date` | Name of the date column |
| `--trend-col` | `trend` | Name of the column identifying which trend each row belongs to |
| `--value-col` | `popularity` | Name of the numeric popularity/interest column |
| `--horizon-weeks` | `12` | How many weeks forward to forecast |
| `--top-n` | `6` | How many trends to show on the chart |
| `--output-dir` | `output` | Where to save the summary CSV + plot |

### Output

- `output/trend_summary.csv` — status, comeback score, and cycle info for every trend
- `output/trend_plot.png` — chart of the most relevant trends, history vs. forecast

### Where real data could come from

- Google Trends exports (search interest over time, per keyword)
- Social listening tools (mention volume per hashtag/keyword over time)
- Internal sell-through or reorder data per product category

