"""Small shared helper used by feature preparation and optional diagnostics."""
import pandas as pd

def make_features(one_series):
    """Build calendar-week history and audit outcomes for one local price series."""
    grid = one_series.set_index("week_start").sort_index()
    calendar = pd.date_range(grid.index.min(), grid.index.max(), freq="W-MON")
    grid = grid.reindex(calendar)
    price = grid["weekly_median_price"]

    for lag in [1, 2, 4]:
        grid[f"lag_{lag}w_price"] = price.shift(lag)
    for window in [4, 8]:
        past = price.shift(1)
        grid[f"rolling_mean_{window}w"] = past.rolling(window, min_periods=2).mean()
        grid[f"rolling_std_{window}w"] = past.rolling(window, min_periods=2).std()
    grid["history_count_4w"] = price.shift(1).rolling(4, min_periods=1).count()
    grid["weekly_price_change_pct"] = (price / price.shift(1) - 1) * 100

    # These FUTURE columns are for labels and audit only, never model features.
    grid["next_week_price"] = price.shift(-1)
    grid["next_week_record_count"] = grid["record_count"].shift(-1)
    grid["next_week_premise_count"] = grid["premise_count"].shift(-1)
    grid["next_week_change_pct"] = (grid["next_week_price"] / price - 1) * 100

    # Return original observed rows, not the inserted empty calendar rows.
    grid = grid.loc[price.notna()].copy()
    grid.index.name = "week_start"
    return grid.reset_index()
