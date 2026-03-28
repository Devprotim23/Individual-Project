from __future__ import annotations
# backtest.py
#
# This module runs expanding-window backtests for monthly time series forecasts.
#
# For each forecast origin, the model is trained on all data available up to
# that point, then asked to forecast the next 1 to h_max months. The resulting
# forecasts are compared with the realised values that actually occurred.
from collections.abc import Callable

import pandas as pd


# Raised when backtesting cannot be run safely.
class BacktestError(Exception):
    pass


# Every forecasting method must return these columns so the backtest can
# compare forecasts consistently across methods.
REQUIRED_FORECAST_COLUMNS = {
    "date",
    "method",
    "horizon",
    "point",
    "lo80",
    "hi80",
    "lo95",
    "hi95",
}


# Validate the main backtest inputs before any forecasting begins.
def _validate_inputs(
    series: pd.Series,
    h_max: int,
    methods: dict[str, Callable[[pd.Series, int], pd.DataFrame]],
    min_train: int,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise BacktestError("Input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise BacktestError("Series index must be a pandas DatetimeIndex.")

    if h_max <= 0:
        raise BacktestError("h_max must be a positive integer.")

    if min_train < 1:
        raise BacktestError("min_train must be at least 1.")

    if len(series) < (min_train + 1):
        raise BacktestError(
            "Series is too short for backtesting with the given minimum training size."
        )

    if not methods:
        raise BacktestError("At least one forecasting method must be provided.")

    series = series.sort_index().copy()

    if series.isna().any():
        raise BacktestError(
            "Series contains missing values; clean or impute before backtesting."
        )

    return series


# Run an expanding-window backtest across all supplied forecasting methods.
def expanding_window_backtest(
    series: pd.Series,
    h_max: int,
    methods: dict[str, Callable[[pd.Series, int], pd.DataFrame]],
    min_train: int = 24,
) -> pd.DataFrame:
    series = _validate_inputs(
        series,
        h_max=h_max,
        methods=methods,
        min_train=min_train,
    )

    all_backtest_rows: list[pd.DataFrame] = []

    # The loop variable marks the exclusive end of the training window.
    # Example: if train_end_position is 24, the training set is series.iloc[:24].
    last_train_end_position = len(series) - 1

    for train_end_position in range(min_train, last_train_end_position + 1):
        training_series = series.iloc[:train_end_position]

        if training_series.empty:
            continue

        # The forecast origin is the final observed month available to the model.
        forecast_origin = training_series.index[-1]

        # Do not request horizons beyond the remaining realised data.
        available_horizon = min(h_max, len(series) - train_end_position)

        if available_horizon <= 0:
            continue

        for method_name, forecast_function in methods.items():
            forecast_output = forecast_function(training_series, available_horizon).copy()

            missing_columns = REQUIRED_FORECAST_COLUMNS - set(forecast_output.columns)
            if missing_columns:
                raise BacktestError(
                    f"Method '{method_name}' returned a DataFrame missing columns: "
                    f"{sorted(missing_columns)}"
                )

            # Ensure dates are in datetime format and rows are ordered correctly.
            forecast_output["date"] = pd.to_datetime(forecast_output["date"])
            forecast_output = forecast_output.sort_values("date").reset_index(drop=True)

            if forecast_output.empty:
                raise BacktestError(
                    f"Method '{method_name}' returned an empty forecast DataFrame."
                )

            # Add the historical forecast origin so each row can be traced back to
            # the exact point in time where the forecast was generated.
            forecast_output["origin"] = forecast_origin

            # Look up the realised values for the same forecast dates.
            forecast_output["actual"] = series.reindex(forecast_output["date"]).values

            # Compute point forecast errors for later metric calculation.
            forecast_output["error"] = forecast_output["actual"] - forecast_output["point"]
            forecast_output["abs_error"] = forecast_output["error"].abs()
            forecast_output["sq_error"] = forecast_output["error"] ** 2

            # Record whether the realised value fell inside each prediction interval.
            forecast_output["hit80"] = (
                (forecast_output["actual"] >= forecast_output["lo80"])
                & (forecast_output["actual"] <= forecast_output["hi80"])
            ).astype(float)

            forecast_output["hit95"] = (
                (forecast_output["actual"] >= forecast_output["lo95"])
                & (forecast_output["actual"] <= forecast_output["hi95"])
            ).astype(float)

            # Record interval widths to evaluate uncertainty size as well as coverage.
            forecast_output["width80"] = forecast_output["hi80"] - forecast_output["lo80"]
            forecast_output["width95"] = forecast_output["hi95"] - forecast_output["lo95"]

            output_columns = [
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
                "error",
                "abs_error",
                "sq_error",
                "hit80",
                "hit95",
                "width80",
                "width95",
            ]
            all_backtest_rows.append(forecast_output[output_columns])

    if not all_backtest_rows:
        raise BacktestError("Backtest produced no rows.")

    backtest_results = pd.concat(all_backtest_rows, ignore_index=True)
    backtest_results = backtest_results.sort_values(
        ["method", "origin", "horizon"]
    ).reset_index(drop=True)

    return backtest_results