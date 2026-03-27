from __future__ import annotations

"""
metrics.py

This module calculates forecast evaluation metrics from backtest prediction data.
It supports both point forecast accuracy metrics and prediction interval metrics.
"""

import numpy as np
import pandas as pd


# Raised when the input data is not suitable for metric calculation.
class MetricsError(Exception):
    pass


# Check that a DataFrame contains every column required by a metric function.
def _require_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    df_name: str = "DataFrame",
) -> None:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise MetricsError(
            f"{df_name} is missing required columns: {sorted(missing_columns)}"
        )


# Prepare a predictions DataFrame for point forecast error calculations.
def _prepare_error_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"method", "horizon", "point", "actual"}
    _require_columns(predictions_df, required_columns, df_name="predictions_df")

    prepared_df = predictions_df.copy()

    # Error metrics can only be computed where both forecast and realised values exist.
    prepared_df = prepared_df.dropna(subset=["point", "actual"])

    if prepared_df.empty:
        raise MetricsError("No valid rows available to compute error metrics.")

    # Error is realised value minus forecast value.
    if "error" not in prepared_df.columns:
        prepared_df["error"] = prepared_df["actual"] - prepared_df["point"]

    # Absolute error is used when calculating MAE.
    if "abs_error" not in prepared_df.columns:
        prepared_df["abs_error"] = prepared_df["error"].abs()

    # Squared error is used when calculating RMSE.
    if "sq_error" not in prepared_df.columns:
        prepared_df["sq_error"] = prepared_df["error"] ** 2

    return prepared_df


# Prepare a predictions DataFrame for interval coverage and width calculations.
def _prepare_coverage_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"method", "horizon", "actual", "lo80", "hi80", "lo95", "hi95"}
    _require_columns(predictions_df, required_columns, df_name="predictions_df")

    prepared_df = predictions_df.copy()

    # Coverage metrics need the realised value and all interval bounds to exist.
    prepared_df = prepared_df.dropna(subset=["actual", "lo80", "hi80", "lo95", "hi95"])

    if prepared_df.empty:
        raise MetricsError("No valid rows available to compute interval metrics.")

    # hit80 is 1.0 when the realised value lies within the 80% interval.
    if "hit80" not in prepared_df.columns:
        prepared_df["hit80"] = (
            (prepared_df["actual"] >= prepared_df["lo80"])
            & (prepared_df["actual"] <= prepared_df["hi80"])
        ).astype(float)

    # hit95 is 1.0 when the realised value lies within the 95% interval.
    if "hit95" not in prepared_df.columns:
        prepared_df["hit95"] = (
            (prepared_df["actual"] >= prepared_df["lo95"])
            & (prepared_df["actual"] <= prepared_df["hi95"])
        ).astype(float)

    # Width measures how wide each prediction interval is.
    if "width80" not in prepared_df.columns:
        prepared_df["width80"] = prepared_df["hi80"] - prepared_df["lo80"]

    if "width95" not in prepared_df.columns:
        prepared_df["width95"] = prepared_df["hi95"] - prepared_df["lo95"]

    return prepared_df


# Calculate point forecast error metrics for each method and forecast horizon.
def compute_errors(predictions_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = _prepare_error_df(predictions_df)

    errors_df = (
        prepared_df.groupby(["method", "horizon"], as_index=False)
        .agg(
            n=("actual", "size"),
            me=("error", "mean"),
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
        )
    )

    # RMSE is the square root of the mean squared error.
    errors_df["rmse"] = np.sqrt(errors_df["mse"])
    errors_df = errors_df.drop(columns=["mse"])

    return errors_df.sort_values(["method", "horizon"]).reset_index(drop=True)


# Calculate interval coverage and average interval width for each method and horizon.
def compute_coverage(predictions_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = _prepare_coverage_df(predictions_df)

    coverage_df = (
        prepared_df.groupby(["method", "horizon"], as_index=False)
        .agg(
            n=("actual", "size"),
            coverage80=("hit80", "mean"),
            avg_width80=("width80", "mean"),
            coverage95=("hit95", "mean"),
            avg_width95=("width95", "mean"),
        )
    )

    return coverage_df.sort_values(["method", "horizon"]).reset_index(drop=True)


# Calculate overall point forecast error metrics for each method across all horizons.
def compute_error_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = _prepare_error_df(predictions_df)

    summary_df = (
        prepared_df.groupby(["method"], as_index=False)
        .agg(
            n=("actual", "size"),
            me=("error", "mean"),
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
        )
    )

    # RMSE is derived from the aggregated mean squared error.
    summary_df["rmse"] = np.sqrt(summary_df["mse"])
    summary_df = summary_df.drop(columns=["mse"])

    return summary_df.sort_values(["method"]).reset_index(drop=True)


# Calculate overall interval metrics for each method across all horizons.
def compute_coverage_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = _prepare_coverage_df(predictions_df)

    summary_df = (
        prepared_df.groupby(["method"], as_index=False)
        .agg(
            n=("actual", "size"),
            coverage80=("hit80", "mean"),
            avg_width80=("width80", "mean"),
            coverage95=("hit95", "mean"),
            avg_width95=("width95", "mean"),
        )
    )

    return summary_df.sort_values(["method"]).reset_index(drop=True)