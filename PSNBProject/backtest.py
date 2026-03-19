from __future__ import annotations

from collections.abc import Callable
import pandas as pd


class BacktestError(Exception):
    """Raised when backtesting cannot be run safely."""


def expanding_window_backtest(
    s: pd.Series,
    h_max: int,
    methods: dict[str, Callable[[pd.Series, int], pd.DataFrame]],
    min_train: int = 24,
) -> pd.DataFrame:
    """
    Run an expanding-window backtest on a monthly time series.

    For each forecast origin t:
      - train on s.iloc[:t]
      - forecast horizons 1..h_max
      - compare forecast dates against actual observed values
      - store predictions and actuals in one combined DataFrame

    Parameters
    ----------
    s : pd.Series
        Monthly series indexed by DatetimeIndex.
    h_max : int
        Maximum forecast horizon in months.
    methods : dict[str, callable]
        Mapping of method name -> forecasting function.
        Each forecasting function must accept (series, h) and return a DataFrame
        with columns: date, method, horizon, point, lo80, hi80, lo95, hi95
    min_train : int, default 24
        Minimum number of observations required before the first forecast origin.

    Returns
    -------
    pd.DataFrame
        Columns:
        origin, date, method, horizon, point, lo80, hi80, lo95, hi95, actual
    """
    if not isinstance(s, pd.Series):
        raise BacktestError("Input must be a pandas Series.")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise BacktestError("Series index must be a pandas DatetimeIndex.")

    if h_max <= 0:
        raise BacktestError("h_max must be a positive integer.")

    if min_train < 1:
        raise BacktestError("min_train must be at least 1.")

    if len(s) < (min_train + 1):
        raise BacktestError("Series is too short for backtesting with the given min_train.")

    if not methods:
        raise BacktestError("At least one forecasting method must be provided.")

    s = s.sort_index()

    rows = []

    # t is the end of the training window, exclusive in iloc slicing
    # so train = s.iloc[:t], and the forecast origin is the last observed date in train
    last_possible_t = len(s) - 1

    for t in range(min_train, last_possible_t + 1):
        train = s.iloc[:t]
        if train.empty:
            continue

        origin = train.index[-1]

        max_h_available = min(h_max, len(s) - t)
        if max_h_available <= 0:
            continue

        for method_name, forecast_fn in methods.items():
            forecast_df = forecast_fn(train, max_h_available).copy()

            required_cols = {
                "date", "method", "horizon", "point", "lo80", "hi80", "lo95", "hi95"
            }
            missing = required_cols - set(forecast_df.columns)
            if missing:
                raise BacktestError(
                    f"Method '{method_name}' returned a DataFrame missing columns: {sorted(missing)}"
                )

            forecast_df["origin"] = origin
            forecast_df["actual"] = forecast_df["date"].map(s)

            cols = [
                "origin",
                "date",
                "method",
                "horizon",
                "point",
                "lo80",
                "hi80",
                "lo95",
                "hi95",
                "actual",
            ]
            rows.append(forecast_df[cols])

    if not rows:
        raise BacktestError("Backtest produced no rows.")

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["method", "origin", "horizon"]).reset_index(drop=True)
    return result